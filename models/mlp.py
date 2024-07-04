import torch.nn as nn
import torch
from models.tools import TransLastTwoC


def gen_mlp_module(node_dims, dropout=0.0, norm=True, activation='ReLU', last_acti=False, last_norm=True, norm_type='bn',
                   last_dropout: bool = None):

    if last_dropout is None:
        last_dropout = last_acti
        
    assert type(node_dims) == list, 'invalid input node_dims'
    layers = []
    n_layers = len(node_dims) - 1
    named_layers = []

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
        layers.append(nn.Linear(node_dims[i], node_dims[i+1]))

        named_layers.append('linear')

        if (i != n_layers - 1) or last_acti: 
            non_linear = nn.__dict__[activation[i]]() if acti_is_list else non_linear
            layers.append(non_linear)
            named_layers.append('non_linear')
            
        if (i != n_layers - 1) or last_dropout:  
            layers.append(nn.Dropout(p=dropout))
            named_layers.append('dropout')

        if norm and ((i != n_layers - 1) or last_norm): 
            if norm_type == 'bn':
                layers.extend([TransLastTwoC(), nn.BatchNorm1d(node_dims[i + 1]), TransLastTwoC()])
                named_layers.extend(['trans', 'bn', 'trans'])
            elif norm_type == 'ln':
                layers.append(nn.LayerNorm(node_dims[i + 1]))
                named_layers.append('ln')

    return nn.Sequential(*layers), named_layers