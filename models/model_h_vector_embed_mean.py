import copy

import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import torch.random as TR

import utils
from models.tools import *
from models.revin import RevIN
from models.mlp import gen_mlp_module
from models.tcn import gen_cnn_module, gen_res_cnn, get_tcn_module
from models.transformer_encoder import get_transformer_encoder_module

class CustomLayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super(CustomLayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        # return (x - mean) / (std + self.eps)
        return x - mean

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def kld_loss(mean, log_var, target_mean=0.0, target_var=1.0):
    mean_var_loss = 0.5 * (torch.mean(torch.sum(torch.square(mean), dim=-1)) + torch.mean(torch.sum(torch.square(log_var), dim=-1)))
    # mean_var_loss = (torch.mean(torch.sum(torch.abs(mean - target_mean), dim=-1)) + torch.mean(torch.sum(torch.abs(var - target_var), dim=-1)))
    return mean_var_loss



class RecurrentTCNModule(nn.Module):
    def __init__(self, input_len: int, out_len:int, kernel_sizes: List, dropout=0.0, norm=True, element_affine=True, activation='ReLU',
                 last_acti=False, last_dropout=False, last_norm=True, norm_type='bn', resnet=False):
        super(RecurrentTCNModule, self).__init__()
        if not resnet:
            self.tcn_m, _ = gen_cnn_module(kernel_sizes, input_len=input_len, dropout=dropout, norm=norm, last_norm=last_norm, element_affine=element_affine,
                                           activation=activation, last_acti=last_acti, last_dropout=last_dropout, norm_type=norm_type)

        else:
            self.tcn_m, _ = gen_res_cnn(kernel_sizes, dropout=dropout, norm=norm, last_norm=last_norm,
                                        activation=activation, last_acti=last_acti, norm_type=norm_type)

        self.input_len = input_len
        self.out_len = out_len

    def forward(self, inputs):
        
        x, last_outs = inputs
        input_x = x[..., -self.input_len:]
        out = self.tcn_m(input_x)
        # out = self.out_norm(out)

        detach_out = out.detach() 
        x = torch.cat((x, detach_out), dim=-1)
        
        if last_outs is not None: 
            last_outs.append(out)
        else:
            last_outs = [out]
            
        return x, last_outs


class OneNeuronModule(nn.Module):
    def __init__(self, input_len, param, rec_iter_num, dropout, activation, norm_type, last_norm=False, last_acti=True):
        super(OneNeuronModule, self).__init__()

        init_ks = [(1,)]
        if norm_type is None:
            norm = False
        else:
            norm = True
        for c in param['channels']:
            init_ks.append((c, param['kernel_size']))

        self.input_len = input_len
        self.rec_iter_num = rec_iter_num
        self.recurrent_cell = RecurrentTCNModule(input_len, init_ks, dropout=dropout,
                                                 norm=norm, last_norm=last_norm,
                                                 activation=activation, last_acti=last_acti,
                                                 norm_type=norm_type)

    def forward(self, x):
        out = x
        for _ in range(self.rec_iter_num):
            rec_out = self.recurrent_cell(out[..., -self.input_len:])
            out = torch.cat((out, rec_out), dim=-1)
        return out


def get_out_size(params, embedding_size, embed_y2x_len):
    last_module_param = params[-1]
    if last_module_param['module_type'] == 'mlp':
        out_size = last_module_param['hidden_size'][-1]
    elif last_module_param['module_type'] == 'transformer_encoder':
        out_size = last_module_param['d_model']
    elif last_module_param['module_type'] == 'Flatten':
        if len(params) <= 1:
            out_size = embedding_size * embed_y2x_len
        else:

            if params[-2]['module_type'] == 'transformer_encoder':
                out_size = params[-2]['d_model'] * params[-2]['seq_len']
            elif params[-2]['module_type'] == 'mlp':
                out_size = params[-2]['hidden_size'][-1] * embed_y2x_len
    elif last_module_param['module_type'] == 'tcn':
        out_size = last_module_param['channels'][-1]
    else:
        raise NotImplementedError()
    return out_size


def compute_rf(kernel_sizes, dilations:List):
    layers_num = len(dilations)
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes]* layers_num

    rf = 1
    for ks, d in zip(kernel_sizes, dilations):
        rf += (ks-1)*d
        # print(rf)
    return rf

class PredTCNRecurrentDEAE_H_Vector_Embed_Mean(nn.Module):
    def __init__(self, input_size, output_size, dropout, activation, train_coupled_len: int, val_coupled_len: int, use_mean_embedds: bool,
                 pred_len: int, encoder_param, one_neuron_param, decoder_param, use_revin=True, inverse_out=False, **kwargs):
        super(PredTCNRecurrentDEAE_H_Vector_Embed_Mean, self).__init__()

        self.encoder_param = encoder_param
        self.one_neuron_param = one_neuron_param
        self.decoder_param = decoder_param
        self.inverse_out = inverse_out
        if self.inverse_out:
            self.normalizer = kwargs.get('data_normalizer', None)
            assert self.normalizer is not None, "normalizer can not be None."
            
        self.extra_forward = kwargs.get('extra_forward')  
        
        self.use_revin = use_revin
        if self.use_revin:
            self.revin_layer = RevIN(input_size, affine=True)
        
        self.use_mean_embedds = use_mean_embedds 


        self.train_coupled_len = train_coupled_len
        self.val_coupled_len = val_coupled_len
        self.coupled_len = self.train_coupled_len 
        self.pred_len = pred_len
        
        self.recurrent_times = self.one_neuron_param['recurrent_times']
        self.tcn_layer_num = len(self.one_neuron_param['channels'])
        self.kernel_size = self.one_neuron_param['kernel_size']
        self.embedding_rec_times = self.one_neuron_param['embedding_rec_times'] + (self.coupled_len if self.extra_forward else 0)
        self.embedding_size = self.recurrent_times // self.embedding_rec_times
        self.embed_y2x_len = self.embedding_rec_times - self.pred_len - (self.coupled_len if self.extra_forward else 0)  
        assert self.recurrent_times % self.embedding_rec_times == 0, 'recurrent_times must be the multiple of embedding_rec_times'

        self.output_size = output_size
        self.encoder_out_size = get_out_size(self.encoder_param, self.embedding_size, self.embed_y2x_len)

        self.each_rec_out_size = self.encoder_out_size - compute_rf(self.kernel_size, self.one_neuron_param['dilations']) + 1 
        assert self.recurrent_times % self.each_rec_out_size == 0, f'recurrent_times({self.recurrent_times}) should be divisible by each_rec_out_size({self.each_rec_out_size})'
        self.rec_iter_num = self.recurrent_times // self.each_rec_out_size

        train_embedding_mask = torch.unsqueeze(get_embedding_masks(self.train_coupled_len, self.embedding_rec_times), dim=-1)
        # [same_y_len, coupled_len, embedding_rec_times, 1]
        self.register_buffer('train_embed_mask', train_embedding_mask)
        
        val_embedding_mask = torch.unsqueeze(get_embedding_masks(self.val_coupled_len, self.embedding_rec_times), dim=-1)
        # [same_y_len, coupled_len, embedding_rec_times, 1]
        self.register_buffer('val_embed_mask', val_embedding_mask)
        
        self.embed_mask = self.train_embed_mask  

        norm_type = kwargs.get('norm_type', None)

        # Encoder
        self.encoder = self.build_module(input_size, self.encoder_param, dropout, activation, norm_type=norm_type,
                                         last_acti=True, last_norm=True)

        # one_neuron
        one_neuron_last_norm = True
        self.one_neuron = self.build_one_neuron_module(self.encoder_out_size, self.each_rec_out_size, self.one_neuron_param, self.rec_iter_num,
                                                       dropout, activation, norm_type,  last_norm=one_neuron_last_norm, last_acti=False)
        self.is_one_neuron_last_norm = self.one_neuron_param.get('last_norm', one_neuron_last_norm)
        # self.embedd_norm = self.build_embed_norm(norm_type, self.embedding_size)
        self.embedd_norm = self.build_embed_norm('custom_ln', self.embedding_size)

        # Decoder
        self.decoder = self.build_module(input_size=self.embedding_size, module_param=self.decoder_param, dropout=dropout,
                                         activation=activation, norm_type=norm_type, last_acti=True, last_norm=True)
        # output head
        head_param = [
            {
            'module_type': 'mlp',
            'hidden_size': [self.output_size],
            }
            ]
        head_input_size = get_out_size(self.decoder_param, self.embedding_size, self.embed_y2x_len)
        print(head_input_size)
        self.out_head = self.build_module(head_input_size, head_param, dropout, activation, norm_type=None,
                                          last_acti=False, last_norm=False, last_dropout=self.use_revin)

        tmp_model = nn.ModuleList([self.encoder, self.one_neuron, self.embedd_norm, self.decoder, self.out_head])
        utils.print_model_param_num(tmp_model)

        init_method = kwargs.get('init_method', None)
        self.init_model(init_method)

    
    def train(self, mode=True):
        """_summary_
        Args:
            mode (bool, optional): _description_. Defaults to True.
        """
        super(PredTCNRecurrentDEAE_H_Vector_Embed_Mean, self).train(mode)
        if mode:
            self.coupled_len = self.train_coupled_len
            self.embed_mask = self.train_embed_mask
        else:
            self.coupled_len = self.val_coupled_len
            self.embed_mask = self.val_embed_mask
    
    def forward(self, x: torch.Tensor):
        """
        X: # [batch, history_len, input_dim]
        """
        if self.use_revin:
            x = self.revin_layer(x, mode='norm')  
        
        h0 = self.encoder(x)
        recur_in = torch.unsqueeze(h0, dim=-2)  # [batch, history_len, 1, hidden_size]

        recur_in = torch.reshape(recur_in, shape=(-1, recur_in.shape[-2], recur_in.shape[-1]))  # [batch * history_len, 1, hidden_size]

        _, recur_units = self.one_neuron([recur_in, None])
        recur_units = torch.squeeze(torch.cat(recur_units, dim=-1))  # [batch * history_len, 1, recurrent_times] -> [batch * history_len, recurrent_times]  
        recur_units = torch.reshape(recur_units, shape=(-1, self.coupled_len, self.embedding_rec_times, self.embedding_size))         # [batch, history_len, embedding_rec_times, embedding_size]
        embedds = recur_units

        
        out_dict = {}
        same_embedds_y_list, mean_embedds_y = embedding_to_vec_y(embedds, self.embed_mask, reduce_mean=True)  # [batch, y_len, embedding_size]
        if self.extra_forward:
            mean_embedds_y = mean_embedds_y[:, :-self.coupled_len]  
            same_embedds_y_list = same_embedds_y_list[:-self.coupled_len]
        # same_embedds_y_list -> [history_len + pred_len, [batch, same_y_len, embedding_size]]
        # mean_embedds_y -> [batch, history_len + pred_len + embed_y2x_len, embedding_size]
        if self.use_mean_embedds:
            if self.is_one_neuron_last_norm:
                n_embedds_y = mean_embedds_y 
            else:
                n_embedds_y = self.embedd_norm(mean_embedds_y)  
 
            all_time_embeds = n_embedds_y.unfold(1, self.embed_y2x_len, step=1)  # [batch, couple_len + pred_len, embedding_size, embedding_len]
            all_time_embeds = all_time_embeds.permute(0, 1, 3, 2)  # [batch, couple_len + pred_len, embedding_len, embedding_size]
            
            
            decoder_out = self.decoder(all_time_embeds)  # [batch, couple_len + pred_len, embedding_len * d_model]

            mean_outs = self.out_head(decoder_out)  # [batch, couple_len + pred_len, input_dim]
            
            if self.use_revin:
                mean_outs = self.revin_layer(mean_outs, mode='denorm')

            if self.inverse_out:
                mean_outs = self.normalizer.inverse_transform(mean_outs)

            out = mean_outs[:, :self.coupled_len]
            pred_outs = mean_outs[:, self.coupled_len:]
            # print(pred_outs.shape)
        else:
            embedds = embedds.reshape(shape=(-1, self.embedding_rec_times, self.embedding_size))  # [batch * history_len, embedding_rec_times, embedding_size]
            embedds = embedds.unfold(1, self.embed_y2x_len, step=1)  # [batch * history_len, pred_len + 1, embedding_size, embed_y2x_len]
            embedds = embedds.permute(0, 1, 3, 2)  # [batch * history_len, pred_len + 1, embed_y2x_len, embedding_size]
            
            decoder_out = self.decoder(embedds)  # [batch * history_len, pred_len + 1, embedding_len * d_model]
            full_out = self.out_head(decoder_out)  # [batch * history_len, pred_len + 1, input_dim]
            full_out = full_out.reshape(-1, self.coupled_len, self.pred_len + 1, self.output_size)  # [batch, history_len, pred_len + 1, input_dim]
            
            if self.use_revin:
                full_out = self.revin_layer(full_out, mode='denorm')

            if self.inverse_out:
                full_out = self.normalizer.inverse_transform(full_out)
            
            out = full_out[..., 0, :].squeeze(dim=2)  # [batch, history_len, input_dim]
            pred_outs = full_out[..., 1:, :]  # [batch, history_len, pred_len, input_dim]
                    
        out_dict.update({'embedds': embedds, 'same_embedds_y_list': same_embedds_y_list, 'embedds_y': mean_embedds_y, 'out': out, 'pred_outs': pred_outs})
        return out_dict

    def init_model(self, init_method):
        if init_method is not None:
            print(f'initializing the model by {init_method}...')
            for m in self.modules():
                # print(type(m))
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                    nn.init.__dict__[init_method](m.weight)
                    if m.bias.data is not None:
                        nn.init.constant_(m.bias, 0)

    def build_embed_norm(self, norm_type, embed_size):
        if norm_type == 'bn':
            layers = []
            layers.append(TransLastTwoC()) 
            layers.append(nn.BatchNorm1d(embed_size))
            layers.append(TransLastTwoC())
            return nn.Sequential(*layers)
        elif norm_type == 'ln':
            return nn.LayerNorm(embed_size)
        elif norm_type == 'custom_ln':
            return CustomLayerNorm()
        else:
            return None

    def build_one_neuron_module(self, input_len, each_out_len, param, rec_iter_num, dropout, activation, norm_type, last_norm=True, last_acti=False):
        init_ks = [(1,)]
        if norm_type is None:
            norm = False
        else:
            norm = True
        for c, d, s in zip(param['channels'], param['dilations'], param['strides']):
            init_ks.append((c, param['kernel_size'], d, s))
        
        last_acti = param.get('last_acti', last_acti)
        last_norm = param.get('last_norm', last_norm)
        activation = param.get('acti', activation)
        last_dropout = param.get('last_dropout', last_acti)
        
        tcn_layer = RecurrentTCNModule(input_len, out_len=each_out_len, kernel_sizes=init_ks, dropout=dropout,
                                       norm=norm, last_norm=last_norm, element_affine=param['ln_ele_affine'], activation=activation, 
                                       last_acti=last_acti, last_dropout=last_dropout, norm_type=norm_type, resnet=param['resnet'])
        return nn.Sequential(*[tcn_layer for _ in range(rec_iter_num)])


    def build_module(self, input_size, module_param, dropout, activation, norm_type, last_acti=True, last_norm=True, last_dropout=None, **kwargs):
        modules = []
        if norm_type is None:
            norm = False
        else:
            norm = True
        last_dropout = last_acti if last_dropout is None else last_dropout
        
        for i, p in enumerate(module_param):
            if p['module_type'] == 'mlp':

                l_n = True
                l_a = True
                if i == len(module_param) - 1:
                    l_a = last_acti
                    l_n = last_norm

                in_size = None
                if i == 0:
                    in_size = input_size
                else:
                    last_param = module_param[i-1]
                    if last_param['module_type'] == 'transformer_encoder':
                        in_size = last_param['d_model']
                    elif last_param['module_type'] == 'mlp' or last_param['module_type'] == 'position_encoding':
                        in_size = last_param['hidden_size'][-1]
                    elif last_param['module_type'] == 'Flatten':
                        in_size = input_size * self.embed_y2x_len
                    elif last_param['module_type'] == 'tcn':
                        in_size = last_param['channels'][-1]
                    else:
                        raise NotImplementedError()

                encoder_nodes = [in_size]
                encoder_nodes.extend(p['hidden_size'])
                encoder_module, _ = gen_mlp_module(encoder_nodes, dropout=dropout,
                                                   norm=norm, activation=activation,
                                                   last_acti=l_a, last_norm=l_n,
                                                   norm_type=norm_type, last_dropout=last_dropout)
                modules.append(encoder_module)

            elif p['module_type'] == 'transformer_encoder':
                if i != 0 and module_param[i-1]['module_type'] == 'patch':
                    seq_len = module_param[i-1]['patch_num']
                    p['seq_len'] = seq_len # 保存seq_len
                    seq_dim = module_param[i-1]['patch_size']
                elif i == 0:
                    seq_len = p['seq_len']
                    seq_dim = input_size
                else:
                    raise NotImplementedError()
                
                encoder_module = get_transformer_encoder_module(seq_len=seq_len, seq_dim=seq_dim, n_layers=p['n_layers'], d_model=p['d_model'], 
                                                                n_heads=p['n_heads'], d_ff=p['d_ff'], dropout=dropout, norm_type=norm_type, activation=activation,
                                                                res_attention=p['res_attention'], norm_pos=p['norm_pos'], pe=p['pe'], learn_pe=p['learn_pe'],
                                                                attention_dropout=p.get('attention_dropout', None))
                modules.append(encoder_module)
            elif p['module_type'] == 'position_encoding':
                modules.append(PositionalEncoding(d_model=p['hidden_size'][0], dropout=dropout))
            elif p['module_type'] == 'identity':
                modules.append(IdentityMap())
            elif p['module_type'] == 'tcn':
                init_ks = [(input_size,)]
                for c, k, d, s in zip(p['channels'], p['kernel_size'], p['dilations'], p['strides']):
                    init_ks.append((c, k, d, s))
                modules.append(TransLastTwoC())
                tcn_module, _ = get_tcn_module(init_ks, norm=True, norm_type=norm_type, activation=activation, last_acti=last_acti, dropout=dropout, 
                                            last_dropout=last_dropout, last_norm=last_norm)
                modules.append(tcn_module)
                modules.append(TransLastTwoC())
            elif p['module_type'] == 'patch':
                if i != 0 and module_param[i-1]['module_type'] == 'Flatten':
                    context_window = (self.embedding_rec_times - self.pred_len) * self.embedding_size  
                elif i == 0:
                    context_window = input_size
                else:
                    raise NotImplementedError()
                
                patch_layer = PatchLayer(context_window, p['patch_size'], p['stride'], p['padding'])
                modules.append(patch_layer)
                p['patch_num'] = patch_layer.patch_num
            else:
                layer = nn.__dict__[p['module_type']](**p['module_params'])
                modules.append(layer)

        return nn.Sequential(*modules)


class TCNRecurrentDEAE(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, channels: List, recurrent_times: int, output_size,
                 dropout, activation, embedding_size=None, embedding_rec_times=None, **kwargs):
        super(TCNRecurrentDEAE, self).__init__()
        hidden_size = [hidden_size] if type(hidden_size) is not list else hidden_size
        assert recurrent_times >= 2, 'recurrent_times must at least 2'
        self.embedding_size = hidden_size[-1] if embedding_size is None else embedding_size 
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.tcn_layer_num = len(channels)
        self.embedding_rec_times = recurrent_times if embedding_rec_times is None else embedding_rec_times  
        self.recurrent_times = recurrent_times
        bn = True

        assert self.recurrent_times % self.embedding_rec_times == 0, 'recurrent_times must be the multiple of embedding_rec_times'

        if self.embedding_size > 1:
            self.embed_select_masks = torch.nn.Parameter(get_embedding_masks(self.embedding_rec_times, history_len=self.embedding_size),
                                                         requires_grad=False)  # [same_y_len, recurrent_times, hidden_size]
        encoder_nodes = [input_size]
        encoder_nodes.extend(hidden_size)
        self.encoder, _ = gen_mlp_module(encoder_nodes, dropout=dropout,
                                         norm=bn, activation=activation, last_acti=True, last_norm=True)

        init_ks = [(1,)]
        for c in channels:
            init_ks.append((c, self.kernel_size))
        self.tcn_m, _ = gen_cnn_module(init_ks, dropout=0,
                                       norm=bn, activation=activation, last_acti=False, last_norm=True)

        self.nonlinear = nn.__dict__[activation]()

        self.embed, _ = gen_mlp_module([(self.recurrent_times // self.embedding_rec_times) * 1, self.embedding_size], dropout=dropout,
                                       norm=bn, activation=activation, last_acti=False)

        self.decoder, _ = gen_mlp_module([self.embedding_size + self.embedding_rec_times - 1, output_size], dropout=dropout,
                                         norm=bn, activation=activation, last_acti=False)

    def forward(self, x: torch.Tensor):
        h0 = self.encoder(x)

        recur_in = torch.unsqueeze(h0, dim=-1)  # [..., batch/times, 1, hidden_size]

        each_rec_out_size = self.hidden_size[-1] - (self.kernel_size - 1) * self.tcn_layer_num  
        assert self.recurrent_times % each_rec_out_size == 0, f'recurrent_times({self.recurrent_times}) should be divisible by each_rec_out_size({each_rec_out_size})'
        rec_iter_num = self.recurrent_times // each_rec_out_size

        recur_units = recur_in
        for i in range(rec_iter_num):
            rec_iter_out = self.tcn_m(recur_in)  # [batch, 1, each_rec_out_size]

            recur_units = torch.cat((recur_units, rec_iter_out), dim=-1)
            recur_in = recur_units[:, :, -self.hidden_size[-1]:]

        rec_out = torch.permute(recur_units[:, :, self.hidden_size[-1]:], (0, 2, 1))  # [batch, 1, recurrent_times] -> [batch, recurrent_times, 1]
        rec_out = torch.reshape(rec_out,
                                shape=(-1, self.embedding_rec_times,
                                       (self.recurrent_times // self.embedding_rec_times) * rec_out.shape[-1]))

        embedds = self.embed(self.nonlinear(rec_out))  
        # embedds = self.embed(torch.sigmoid(rec_out))

        # [batch, hidden_size + recurrent_times - 1]
        out_dict = {}
        if self.embedding_size > 1:
            same_ys, embedd_ys = embedding_to_y(embedding=embedds, embedding_masks=self.embed_select_masks, reduce_mean=True)
            out_dict['same_ys'] = same_ys
        else:
            embedd_ys = torch.squeeze(embedds)

        out = self.decoder(embedd_ys)
        out_dict.update({'embedds': embedds, 'embed_ys': embedd_ys, 'out': out})
        return out_dict


class TaskModel(nn.Module):

    def __init__(self, base_model: str, task_type: str, **kwargs):

        super(TaskModel, self).__init__()
        self.base_m = eval(f'{base_model}(**kwargs)')
        self.task_type = task_type

        if task_type == 'classification':
            class_num = kwargs.get('class_num')
            encoder_size = kwargs.get('encoder_param')[-1].get('hidden_size')
            if type(encoder_size) is list:
                encoder_size = encoder_size[-1]
            embedding_rec_times = kwargs.get('one_neuron_param').get('embedding_rec_times')
            if embedding_rec_times is None:
                embedding_rec_times = kwargs.get('one_neuron_param').get('recurrent_times')
            embedding_size = kwargs.get('one_neuron_param').get('embedding_size', encoder_size)

            self.class_layer = nn.Linear(embedding_size + embedding_rec_times - 1, class_num)
        elif self.task_type == 'prediction':
            self.pred_len = kwargs.get('pred_len')  
            self.use_mean_embedds = kwargs.get('use_mean_embedds', True)
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor, **kwargs):
        out_dict = self.base_m(x)

        if self.task_type == 'classification':
            class_logits = self.class_layer(out_dict['embed_ys'])  # [batch, class_num]
            out_dict['class_softmax'] = F.softmax(class_logits, dim=-1)  


            loss_dict = {}

            loss_dict['CE_loss'] = F.cross_entropy(class_logits, kwargs.get('class_labels'))

            if 'same_ys' in out_dict:
                y_embed_loss = []
                for i, y in enumerate(out_dict['same_ys']):
                    y_embed_loss.append(torch.mean(torch.square(y - torch.unsqueeze(out_dict['embed_ys'][:, i], dim=-1))))
                y_embed_loss = torch.stack(y_embed_loss).mean()
                loss_dict['y_embed_loss'] = y_embed_loss

            loss_dict['mse_loss'] = F.mse_loss(out_dict['out'], x)

            return out_dict, loss_dict
        elif self.task_type == 'prediction':

            loss_dict = {}

            out_embed_loss = []
            out_same_ys, out_embedd_ys = out_dict['same_embedds_y_list'], out_dict['embedds_y']               # out_same_ys: [coupled_time_len + embed_y_len - 1, [batch, same_num, embed_size]]
            # out_embedd_ys: [batch, coupled_time_len + embed_y_len - 1, embed_size]

            for i, y in enumerate(out_same_ys):
                out_embed_loss.append(torch.mean(torch.square(y - torch.unsqueeze(out_embedd_ys[:, i], dim=1)))) 
            out_embed_loss = torch.stack(out_embed_loss).mean()
            loss_dict['coupled_embed_loss'] = out_embed_loss

            # embed variance loss
            var = torch.var(out_embedd_ys, dim=(-1, -2))  
            embed_std_loss = torch.mean(1.0 / (torch.square(var) + 1e-8))
            loss_dict['embed_std_loss'] = embed_std_loss

            pred_label = kwargs.get('pred_label')  # [batch, pred_len, system_dim]
            rec_label = kwargs.get('rec_label')  # [batch, history_len, system_dim]
            if not self.use_mean_embedds:
                cat_label = torch.cat((rec_label, pred_label), dim=1)[:, 1:]  # [batch, history_len + pred_len, system_dim]
                pred_label = cat_label.unfold(1, self.pred_len, step=1).permute(0, 1, 3, 2)  # [batch, history_len + 1, pred_len, system_dim]

            loss_dict['pred_loss'] = F.mse_loss(out_dict['pred_outs'], pred_label)

            loss_dict['rec_loss'] = F.mse_loss(out_dict['out'], rec_label)

            return out_dict, loss_dict
        else:
            raise NotImplementedError()
