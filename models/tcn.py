import torch.nn as nn
import torch
from models.tools import TransLastTwoC, CustomLayerNorm, get_norm_layer, PrintLayer
from typing import List


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CusGAP1d(nn.Module):

    def __init__(self):
        super(CusGAP1d, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=-2, keepdim=True)


def get_tcn_module(kernel_sizes: List, norm=True, norm_type='bn', activation='ReLU', last_acti=False, 
                   dropout=0.0, last_dropout=False, last_norm=True):

    assert type(kernel_sizes) == list, 'invalid input node_dims'
    layers = []
    n_layers = len(kernel_sizes) - 1

    non_linear = None
    acti_is_list = False
    if type(activation) == str:
        non_linear = nn.__dict__[activation]
    elif type(activation) == list:
        acti_is_list = True
        if last_acti:
            assert len(activation) == n_layers, 'improper number of activation layers'
        else:
            assert len(activation) == n_layers - 1, 'improper number of activation layers'

    norm_layer = get_norm_layer(norm_type)

    for i in range(n_layers):
        acti = nn.__dict__[activation[i]] if acti_is_list else non_linear
        if (i != n_layers - 1) or last_acti:  
            l_acti = True
        else:
            l_acti = False
        
        if (i != n_layers - 1) or last_dropout:  
            l_drop = True
        else:
            l_drop = False

        if norm and ((i != n_layers - 1) or last_norm):  
            l_norm = True
        else:
            l_norm = False

        layers.append(TemporalBlock(n_inputs=kernel_sizes[i][0], n_outputs=kernel_sizes[i + 1][0], norm_layer=norm_layer,
                                       kernel_size=kernel_sizes[i + 1][1], non_linear=acti, stride=kernel_sizes[i + 1][3],
                                       dilation=kernel_sizes[i + 1][2], dropout=dropout, last_acti=l_acti, last_norm=l_norm, last_drop=l_drop))
    return nn.Sequential(*layers), None
    


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, non_linear, norm_layer, stride, dilation, dropout=0.2,
                 last_acti=True, last_drop=True, last_norm=True):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.nonlinear1 = non_linear()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = norm_layer(n_outputs)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.nonlinear2 = non_linear() if last_acti else None
        self.dropout2 = nn.Dropout(p=dropout) if last_drop else None
        self.norm2 = norm_layer(n_outputs) if last_norm else None

        self.net = nn.Sequential(self.conv1, self.chomp1, self.nonlinear1, self.dropout1, self.norm1,
                                 self.conv2, self.chomp2, self.nonlinear2, self.dropout2, self.norm2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = non_linear()


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalResBlock(nn.Module):
    """
    Temporal Residual Block
    """
    def __init__(self, n_inputs, n_outputs, norm_layer, kernel_size, non_linear, stride=1, dilation=1,
                 padding=0, dropout=0.2, last_acti=True, last_norm=True):
        super(TemporalResBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.non_linear1 = non_linear()
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = norm_layer(n_outputs)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=1, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.chomp2 = Chomp1d((kernel_size-1)*dilation)
        self.non_linear2 = non_linear() if last_acti else None
        self.dropout2 = nn.Dropout(p=dropout) if last_acti else None
        self.norm2 = norm_layer(n_outputs) if last_norm else None

        layers = [self.conv1, self.non_linear1, self.dropout1, self.norm1,
                  self.conv2, self.chomp2, self.non_linear2, self.dropout2, self.norm2]
        self.net = nn.Sequential(*[l for l in layers if l is not None])


        if stride == 1:
            self.time_select = Chomp1d((kernel_size-1)*dilation) 
        else:
            self.time_select = nn.AvgPool1d(kernel_size=((kernel_size-1)*dilation)+1, stride=stride, padding=0)
        self.down_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.down_norm = norm_layer(n_outputs) if last_norm else None
        down_layers = [self.time_select, self.down_conv, self.down_norm]
        self.downsample = nn.Sequential(*[l for l in down_layers if l is not None])

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res
    

def gen_res_cnn(kernel_sizes: List, dropout=0.0, norm=True, activation='ReLU', last_acti=False, last_norm=True,
                norm_type='bn'):

    assert type(kernel_sizes) == list, 'invalid input node_dims'
    layers = []
    n_layers = len(kernel_sizes) - 1

    non_linear = None
    acti_is_list = False
    if type(activation) == str:
        non_linear = nn.__dict__[activation]
    elif type(activation) == list:
        acti_is_list = True
        if last_acti:
            assert len(activation) == n_layers, 'improper number of activation layers'
        else:
            assert len(activation) == n_layers - 1, 'improper number of activation layers'

    norm_layer = get_norm_layer(norm_type)

    for i in range(n_layers):
        acti = nn.__dict__[activation[i]] if acti_is_list else non_linear
        if (i != n_layers - 1) or last_acti: 
            l_acti = True
        else:
            l_acti = False

        if norm and ((i != n_layers - 1) or last_norm): 
            l_norm = True
        else:
            l_norm = False

        layers.append(TemporalResBlock(n_inputs=kernel_sizes[i][0], n_outputs=kernel_sizes[i + 1][0], norm_layer=norm_layer,
                                       kernel_size=kernel_sizes[i + 1][1], non_linear=acti, stride=kernel_sizes[i + 1][3],
                                       dilation=kernel_sizes[i + 1][2], dropout=dropout, last_acti=l_acti, last_norm=l_norm))

    if kernel_sizes[-1][0] != 1:
        layers.append(CusGAP1d())

    return nn.Sequential(*layers), None



def gen_one_neuron_tcn(kernel_sizes: List, input_len, dropout=0.0, norm=True, activation='ReLU', last_acti=False, last_norm=True, element_affine=False,
                   norm_type='bn'):
    layers = []
    layers.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_sizes[1][1]))
    layers.append(nn.GELU())
    return nn.Sequential(*layers), None
    
    
def gen_cnn_module(kernel_sizes: List, input_len, dropout=0.0, norm=True, activation='ReLU', last_acti=False, last_dropout=False,
                   last_norm=True, element_affine=False, norm_type='bn'):

    assert type(kernel_sizes) == list, 'invalid input node_dims'
    layers = []
    n_layers = len(kernel_sizes) - 1
    named_layers = []
    layer_input_len = input_len
    layer_output_len = None

    non_linear = None
    acti_is_list = False
    if type(activation) == str:
        non_linear = nn.__dict__[activation]()
    elif type(activation) == list:
        acti_is_list = True
        if last_acti:
            assert len(activation) == n_layers, 'improper number of activation layers'
        else:
            assert len(activation) == n_layers - 1, 'improper number of activation layers'
    
    

    for i in range(n_layers):
        layers.append(nn.Conv1d(in_channels=kernel_sizes[i][0], out_channels=kernel_sizes[i+1][0],
                                kernel_size=kernel_sizes[i+1][1], dilation=kernel_sizes[i+1][2],
                                stride=kernel_sizes[i+1][3], padding=0))

        layer_output_len = (layer_input_len - kernel_sizes[i+1][2] * (kernel_sizes[i+1][1] - 1) - 1) // kernel_sizes[i+1][3] + 1
        layer_input_len = layer_output_len  
        named_layers.append('conv1d')

        if (i != n_layers - 1) or last_acti: 
            non_linear = nn.__dict__[activation[i]]() if acti_is_list else non_linear
            layers.append(non_linear)
            named_layers.append('non_linear')
            
        if (i != n_layers - 1) or last_dropout:  
            layers.append(nn.Dropout(p=dropout))
            named_layers.append('dropout')

        if norm and ((i != n_layers - 1) or last_norm):  
            if norm_type == 'bn':
                layers.append(nn.BatchNorm1d(kernel_sizes[i+1][0]))
                named_layers.append('bn')
            elif norm_type == 'ln':
                if kernel_sizes[i+1][0] == 1:
                    layers.append(nn.LayerNorm(normalized_shape=(kernel_sizes[i+1][0], layer_output_len), elementwise_affine=element_affine))
                else:
                    layers.append(CustomLayerNorm(kernel_sizes[i+1][0]))   # 如果输出通道不是1，那么在输出的通道上进行LN
                named_layers.append('cus_ln')

    return nn.Sequential(*layers), named_layers