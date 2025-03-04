import datetime
from shutil import copyfile
import random
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
from torch.nn import Module
import torch.distributed as dist

def get_time_log_dir(*args, **kwargs):
    time_stamp = datetime.datetime.now()
    arg_str = '_'.join([str(a) for a in args])
    karg_str = '_'.join([f'{k}={v}' for k, v in kwargs.items()])
    log_dir_name = f'({time_stamp.strftime("%Y_%m_%d-%H_%M_%S")}) {arg_str} {karg_str}'
    return log_dir_name


def add_model_postfix(model_name, model_params):
    model_postfix = ''
    ocn_type = None if len(model_params['one_neuron_param']['channels']) == 1 else 'mlocn'
    
    is_sub_time = 'subtime' if model_params['one_neuron_param']['recurrent_times'] != model_params['one_neuron_param']['embedding_rec_times'] else 'nosubtime'
    
    decoder_type = None
    is_patch = False
    for dp in model_params['decoder_param']:
        if dp['module_type'] == 'transformer_encoder':
            is_patch = True
            break
    if is_patch:
        decoder_type = 'patch'
    elif model_params['decoder_param'][-1]['module_type'] == 'mlp':
        decoder_type = 'mlp'
        
    postfixs = [ocn_type, is_sub_time, decoder_type]
    postfixs = [p for p in postfixs if p is not None]
    
    model_postfix = '_'.join(postfixs)
    
    model_postfix = f'({model_postfix})'
    return model_name + ' ' + model_postfix


def get_log_dir_from_file(file_path):
    dir_path = '/'.join(file_path.split('/')[:-1])
    return dir_path


def cp_file2dir(file_path, target_dir):
    target_file_path = os.path.join(target_dir, file_path.split('/')[-1])
    copyfile(file_path, target_file_path)
    return target_file_path


def setup_seed(seed, cuda_deterministic=False):
    random.seed(0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def print_model_param_num(model:Module):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f'model have {num_params :.5e} parameters in total')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_results = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = -1
        self.delta = delta

    def __call__(self, val_loss, model, val_results, path, is_distribute, optimizer=None, scheduler=None, epoch=None):
        score = -val_loss
        is_best = False
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.val_results = val_results
            is_best = True
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            is_best = False

        self.save_checkpoint(val_loss, model, path, is_distribute, is_best, optimizer, scheduler, epoch)

    def save_checkpoint(self, val_loss, model, path, is_distribute, best, optimizer=None, scheduler=None, epoch=None):
        if not is_distribute or dist.get_rank() == 0:
            state = {'epoch': epoch,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict() if optimizer is not None else None,
                     'scheduler': scheduler.state_dict() if scheduler is not None else None,
                     'early_stopping': self}
            if best:
                if self.verbose:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                    torch.save(state, path+'/'+f'checkpoint.pth')
                    self.val_loss_min = val_loss
                self.best_epoch = epoch
            # torch.save(state, path + '/' + f'checkpoint_epoch:{epoch}.pth')
