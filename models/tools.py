import torch.nn as nn
import torch


def get_embedding_masks(embedding_len, history_len) -> torch.Tensor:

    masks = []
    j_max = 0  
    i_max = 1  

    while j_max < history_len:
        mask = torch.zeros(size=(embedding_len, history_len))

        j = j_max
        i = 0
        while j>=0 and i < embedding_len:
            mask[i,j] = 1
            i += 1
            j -= 1

        masks.append(mask)
        j_max += 1

    while i_max < embedding_len:
        mask = torch.zeros(size=(embedding_len, history_len))

        j = history_len - 1
        i = i_max
        while i < embedding_len and j>=0:
            mask[i, j] = 1
            i += 1
            j -= 1
        masks.append(mask)
        i_max += 1

    masks = [m.bool() for m in masks]
    return torch.stack(masks)



def embedding_to_y(embedding: torch.Tensor, embedding_masks: torch.Tensor, reduce_mean=True):
    """

    :param embedding: [batch_size, embedding_len, history_len]
    :param embedding_masks: same_y_num * [embedding_len, history_len]
    :param reduce_mean: 
    :return:
    """
    batch_size = embedding.shape[0]

    ys = []
    # print(embedding_masks.device)
    for m in embedding_masks:
        ys.append(torch.masked_select(embedding, m).reshape(batch_size, -1)) 
    if reduce_mean:
        mean_ys = torch.cat([torch.mean(y, dim=-1, keepdim=True) for y in ys], dim=-1)
        return ys, mean_ys  # [batch_size, same_y_num]
    else:
        return ys  


def embedding_to_vec_y(embedding: torch.Tensor, embedding_masks: torch.Tensor, reduce_mean=True):

    batch_size = embedding.shape[0]
    vec_len = embedding.shape[-1]

    ys = []
    # print(embedding_masks.device)
    for m in embedding_masks:
        ys.append(torch.masked_select(embedding, m).reshape(batch_size, -1, vec_len))  
    if reduce_mean:
        mean_ys = torch.cat([torch.mean(y, dim=-2, keepdim=True) for y in ys], dim=-2)
        return ys, mean_ys  # [batch_size, same_y_num, embed_vec_len]
    else:
        return ys  


def get_final_preds(pred_outs, history_len, pred_len, pred_type='last'):

    if pred_type == 'last':
        final_preds = pred_outs[:, -1, :, :].squeeze(dim=1)  # [n_samples, pred_len, dim]
    elif pred_type == 'mean':
        mean_mask = get_embedding_masks(history_len, pred_len).unsqueeze(dim=-1)  # [same_y_num, history_len, pred_len, 1]
        mean_mask = mean_mask[-pred_len:].to(pred_outs.device)  
        _, final_preds = embedding_to_vec_y(pred_outs, mean_mask)   # [n_samples, pred_len, dim]
    else:
        raise ValueError('invalid pred_type')
    return final_preds


# classes
class PatchLayer(nn.Module):
    def __init__(self, context_window, patch_len, stride_len, padding='end'):
        super(PatchLayer, self).__init__()
        self.context_window = context_window
        self.patch_len = patch_len
        self.stride_len = stride_len
        self.patch_num = (context_window - patch_len) // stride_len + 1
        if padding == 'end':
            padding_len = stride_len - (context_window - patch_len) % stride_len
            if padding_len == stride_len:
                self.padding_layer = None
            else:
                self.padding_layer = nn.ReplicationPad1d((0, padding_len))
                self.patch_num += 1
            
    def forward(self, x):   # x: [batch_size, seq_len]
        if self.padding_layer is not None:
            x = self.padding_layer(x)
        x = x.unfold(-1, self.patch_len, self.stride_len)  # [batch_size, patch_num, patch_len]
        return x


class TransLastTwoC(nn.Module):
    def __init__(self):
        super(TransLastTwoC, self).__init__()

    def forward(self, x):
        if x.dim() > 2:
            return torch.transpose(x, -1, -2)
        else:
            return x

class CustomLayerNorm(nn.Module):

    def __init__(self, norm_shape):
        super(CustomLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(norm_shape)
        self.trans = TransLastTwoC()

    def forward(self, x):
        x = self.trans(x)
        x = self.ln(x)
        x = self.trans(x)
        return x


class PrintLayer(nn.Module):
    def __init__(self, name):

        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(self.name, x.size())
        return x


class IdentityMap(nn.Module):
    def __init__(self):

        super(IdentityMap, self).__init__()

    def forward(self, x):
        return x

    
# functions

def get_norm_layer(norm_type='bn'):
    if norm_type == 'bn':
        return nn.BatchNorm1d
    elif norm_type == 'ln':
        return CustomLayerNorm
    