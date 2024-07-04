import pickle
import sys

import pandas as pd
sys.path.append('..')

import time

from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

import data_utils
import utils
from models import model_h_vector_embed_mean
from models.tools import get_final_preds
import numpy as np
import yaml
import os
import metrics
from enum import Enum

from itertools import product


# is_distributed = False
is_distributed = True
accumulation_steps = 1 
print_freq = 1
os.environ['MASTER_ADDR'] = 'localhost'
MASTER_PORT = '22235'
node_rank = 0  
nnodes = 1
nproc_per_node = -1  
world_size = -1  


def main():
    cfg_file_path = '../cfg/pred_lorenz_tcn_model_mp (h_vector_embed_mean).yml'

    with open(cfg_file_path, 'r') as file:
        cfgs = yaml.safe_load(file)
    print(f'Dataset: {cfgs["data_name"]}, Model type: {cfgs["model_name"]}')

    start_time = time.time()
    if is_distributed:
        if nproc_per_node == -1:
            local_size = torch.cuda.device_count()
        else:
            local_size = nproc_per_node
        cfgs['world_size'] = nnodes * local_size

        mp.spawn(main_worker, nprocs=local_size, args=(cfgs, node_rank, local_size, cfgs['world_size'], cfg_file_path, MASTER_PORT), join=True)
        train_time = time.time() - start_time
    else:
        main_worker(cfgs['gpu_idx'], cfgs, 0, 0, 0, cfg_file_path)
        train_time = time.time() - start_time
    print(f'total training time: {train_time}s')

def main_worker(local_rank, cfgs, node_rank, local_size, world_size, cfg_file_path, master_port, base_log_dir = '../log', ):

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_distributed:
        rank = local_rank + node_rank * local_size  
        print(rank, 'master_port', master_port)
        dist.init_process_group('nccl', init_method=f"tcp://{os.environ['MASTER_ADDR']}:{master_port}",
                                rank=rank, world_size=world_size)


    utils.setup_seed(cfgs['seed'] + rank if is_distributed else 0, cuda_deterministic=True)

    train_dataset, val_dataset, test_dataset = data_utils.get_dataset(cfgs)

    input_size = train_dataset.data_dim
    output_size = train_dataset.output_dim if hasattr(train_dataset, 'output_dim') else input_size
    print('input_size: ', input_size, ', output_size: ', output_size)



    m = model_h_vector_embed_mean.TaskModel(base_model=cfgs['model_name'], task_type='prediction', input_size=input_size, output_size=output_size, 
                                            train_coupled_len=cfgs['train_coupled_len'], val_coupled_len=cfgs['coupled_len'],
                                            pred_len=cfgs['pred_len'], seq_tcn=True, inverse_out=cfgs['inverse_out'], data_normalizer=train_dataset.normalizer,
                                            use_mean_embedds=cfgs['use_mean_embedds'], extra_forward=cfgs['extra_forward'], **cfgs['model_params'])

    m = m.to(device)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)

        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[local_rank])
        
        train_data_loader = DataLoader(train_dataset, batch_size=cfgs['b_size'], num_workers=cfgs['data_workers'],
                                   shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)

        val_data_loader = DataLoader(val_dataset, batch_size=cfgs['b_size'], num_workers=cfgs['data_workers'],
                                  shuffle=False, pin_memory=True, sampler=val_sampler)
        test_data_loader = DataLoader(test_dataset, batch_size=cfgs['b_size'], num_workers=cfgs['data_workers'],
                                  shuffle=False, pin_memory=True, sampler=test_sampler)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        
        train_batch_sampler = data_utils.TimeBatchSampler(train_dataset, batch_size=cfgs['b_size'],
                                                          sample_stride=cfgs['train_sample_stride'], shuffle=True, drop_last=False)
        print("train_batch_sampler length: ", len(train_batch_sampler))
        val_batch_sampler = data_utils.TimeBatchSampler(val_dataset, batch_size=cfgs['b_size'],
                                                        sample_stride=1, shuffle=True, drop_last=False)
        print("val_batch_sampler length: ", len(val_batch_sampler))
        test_batch_sampler = data_utils.TimeBatchSampler(test_dataset, batch_size=cfgs['b_size'], 
                                                         sample_stride=1, shuffle=True, drop_last=False)
        print("test_batch_sampler length: ", len(test_batch_sampler))

        train_data_loader = DataLoader(train_dataset, num_workers=cfgs['data_workers'], pin_memory=True, batch_sampler=train_batch_sampler)
        val_data_loader = DataLoader(val_dataset, num_workers=cfgs['data_workers'], pin_memory=True, batch_sampler=val_batch_sampler)
        test_data_loader = DataLoader(test_dataset, num_workers=cfgs['data_workers'], pin_memory=True, batch_sampler=test_batch_sampler)
    

        
    optimizer = optim.Adam(m.parameters(), lr=cfgs['lr'], weight_decay=cfgs['l2_weight'])
    
    if cfgs['lr_scheduler']['sched_type'] == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfgs['lr'], total_steps=len(train_data_loader) * cfgs['epoches'],
                                            **cfgs['lr_scheduler']['lr_scheduler_params'])
    else:
        scheduler = lr_scheduler.__dict__[cfgs['lr_scheduler']['sched_type']](optimizer, **cfgs['lr_scheduler']['lr_scheduler_params'])
    early_stopping = utils.EarlyStopping(patience=cfgs['early_stop_patience'], verbose=True)

    epoches = cfgs['epoches']


    log_dir_name = utils.get_time_log_dir('prediction',
                                          data=cfgs['data_name'],
                                          model_name=cfgs['model_name'],
                                          pred_len=cfgs['pred_len'],
                                          n_samples=cfgs['n_samples'],
                                          data_dim=train_dataset.data_dim)
    log_dir_name = utils.add_model_postfix(log_dir_name, cfgs['model_params'])

    log_dir = os.path.join(base_log_dir, log_dir_name)

    if cfgs['resume_file'] is not None and os.path.isfile(cfgs['resume_file']):
        print("=> loading checkpoint '{}' to device {}".format(cfgs['resume_file'], device))
        ckpt_state = torch.load(cfgs['resume_file'], device)
        m.load_state_dict(ckpt_state['model'])

        early_stopping = ckpt_state['early_stopping']

        if ckpt_state['epoch'] is not None:
            start_epoch = ckpt_state['epoch']
            print(f'Loaded strat epoch: {start_epoch}')
        if ckpt_state['optimizer'] is not None:
            optimizer.load_state_dict(ckpt_state['optimizer'])
        if ckpt_state['scheduler'] is not None:
            scheduler.load_state_dict(ckpt_state['scheduler'])

        log_dir = utils.get_log_dir_from_file(cfgs['resume_file'])
        print(log_dir)

    else:
        print("=> no checkpoint found at '{}'".format(cfgs['resume_file']))
        start_epoch = 0

    os.makedirs(log_dir, exist_ok=True)
    if cfgs['evaluate']:

        val_losses, val_results = validate(test_data_loader, m, device, cfgs)
        if not is_distributed or dist.get_rank() == 0:
            labels, preds, rec_labels, rec_outs = val_results
            skip_idxs = np.arange(0, labels.shape[0], cfgs['pred_len'])
            label_m = torch.reshape(labels[skip_idxs], shape=[-1, labels.shape[-1]])
            pred_m = torch.reshape(preds[skip_idxs], shape=[-1, preds.shape[-1]])
            with open(os.path.join(log_dir, 'pred_res.pkl'), 'wb') as file:
                pickle.dump([labels.cpu().numpy(), preds.cpu().numpy(), rec_labels.cpu().numpy(), rec_outs.cpu().numpy()], file)
        return

    tb_writer = None
    if not is_distributed or dist.get_rank() == 0:
        utils.cp_file2dir(cfg_file_path, log_dir)
        tb_log_dir = os.path.join(log_dir, utils.add_model_postfix('tb_logs', cfgs['model_params']))
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_log_dir)

    for ei in range(start_epoch, epoches):
        if is_distributed:
            train_data_loader.sampler.set_epoch(ei)


        train_losses = train(train_data_loader, m, optimizer, ei, epoches, device, cfgs['loss_weights'],
                             is_one_cycle_scheduler=isinstance(scheduler, lr_scheduler.OneCycleLR), oc_scheduler=scheduler)
        val_losses, val_results = validate(val_data_loader, m, device, cfgs, prefix='Val')
        test_losses, test_results = validate(test_data_loader, m, device, cfgs)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_losses['pred_loss'])
        elif not isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        early_stopping(val_losses['pred_loss'], m, test_results, path=log_dir,
                       is_distribute=is_distributed, optimizer=optimizer, scheduler=scheduler, epoch=ei+1)

        if tb_writer:
            tb_writer.add_scalars('pred_loss', {'train': train_losses['pred_loss'],
                                                'val': val_losses['pred_loss'],
                                                'test': test_losses['pred_loss']}, global_step=ei)
            
            tb_writer.add_scalars('coupled_embed_loss', {'train': train_losses['coupled_embed_loss'],
                                    'val': val_losses['coupled_embed_loss'],
                                    'test': test_losses['coupled_embed_loss']}, global_step=ei)
            
            tb_writer.add_scalars('embed_std_loss', {'train': train_losses['embed_std_loss'],
                        'val': val_losses['embed_std_loss'],
                        'test': test_losses['embed_std_loss']}, global_step=ei)

            tb_writer.add_scalars('rec_loss', {'train': train_losses['rec_loss'],
                                                'val': val_losses['rec_loss'],
                                                'test': test_losses['rec_loss']}, global_step=ei)

        if ei + 1 == epoches or early_stopping.early_stop:
            if early_stopping.early_stop:
                print(f"Early stopping, best epoch: {early_stopping.best_epoch}")
            break

    labels, preds, _, _ = early_stopping.val_results
    if (not is_distributed) or dist.get_rank() == 0:
        with open(os.path.join(log_dir, 'pred_res.pkl'), 'wb') as file:
            pickle.dump([labels.cpu().numpy(), preds.cpu().numpy()], file)


def train(train_loader, model, optimizer, epoch, epoches, device, loss_weights, is_one_cycle_scheduler=False, oc_scheduler=None):
    model.train()

    data_time = AverageMeter('DataTime', fmt=':6.3f')
    batch_time = AverageMeter('BatchTime', fmt=':6.3f')
    loss_meters = {}
    loss_meters['loss'] = AverageMeter('Loss', fmt=':.3f')
    loss_meters.update({k: AverageMeter(k, fmt=':.3f') for k, _ in loss_weights.items()})
    progress = ProgressMeter(num_batches=len(train_loader),
                             meters=[batch_time, data_time] + list(loss_meters.values()),
                             prefix=f'Epoch: [{epoch+1}/{epoches}]')

    start = time.time()
    optimizer.zero_grad()
    for i, batch_data in enumerate(train_loader):
        batch_data = [data.to(device, non_blocking=True) for data in batch_data]
        input_data, rec_label, pred_label = batch_data
        # measure data loading time
        data_time.update(time.time() - start)

        my_context = model.no_sync if is_distributed and i % accumulation_steps != 0 else nullcontext
        with my_context():
            out_dict, loss_dict = model(input_data, rec_label=rec_label, pred_label=pred_label)

            loss = 0
            for l_name, l in loss_dict.items():
                tmp_l = l * loss_weights.get(l_name, 1.0)
                loss_meters[l_name].update(tmp_l.item(), n=input_data.shape[0])
                if not l_name.endswith('metric'):  
                    loss += tmp_l/accumulation_steps  
            loss_meters['loss'].update((loss * accumulation_steps).item(), n=input_data.shape[0])
            loss.backward() 

        if i % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if i % print_freq == 0:
            if is_distributed:
                for k, meter in loss_meters.items():
                    meter.reduce_val()
                if dist.get_rank() == 0:
                    progress.display(i+1)
            else:
                progress.display(i+1)

        if is_one_cycle_scheduler:
            oc_scheduler.step()
        
        batch_time.update(time.time() - start)
        start = time.time()
        
    if is_distributed:
        for k, meter in loss_meters.items():
            meter.all_reduce()
        if dist.get_rank() == 0:
            progress.display_summary()
    else:
        progress.display_summary()

    return {k: v.avg for k, v in loss_meters.items()}


def validate(val_loader, model, device, cfgs, prefix='Test'):
    def run_validate(val_loader, base_progress=0):
        with torch.no_grad():
            start = time.time()
            labels = []
            preds = []
            rec_labels = []
            rec_outs = []
            all_t_idxes = []
            for i, batch_data in enumerate(val_loader):
                i = base_progress + i
                batch_data = [data.to(device) for data in batch_data]
                input_data, rec_label, pred_label, t_idxes = batch_data

                out_dict, loss_dict = model(input_data, rec_label=rec_label, pred_label=pred_label)
                labels.append(pred_label)
                preds.append(out_dict['pred_outs']) 

                rec_labels.append(rec_label)
                rec_outs.append(out_dict['out'])

                all_t_idxes.append(t_idxes)
                loss = 0
                for l_name, l in loss_dict.items():
                    tmp_l = l * cfgs['loss_weights'].get(l_name, 1.0)
                    loss_meters[l_name].update(tmp_l.item(), n=input_data.shape[0])
                    if not l_name.endswith('metric'): 
                        loss += tmp_l
                loss_meters['loss'].update(loss.item(), n=input_data.shape[0])

                if i % print_freq == 0:
                    if is_distributed:
                        for k, meter in loss_meters.items():
                            meter.reduce_val()
                        if dist.get_rank() == 0:
                            progress.display(i+1)
                    else:
                        progress.display(i+1)

            labels = torch.cat(labels, dim=0)  # [n_samples, pred_len, dim] 
            preds = torch.cat(preds, dim=0)  # [n_samples, pred_len, dim] or [n_samples, history_len, pred_len, dim]
            rec_outs = torch.cat(rec_outs, dim=0)
            rec_labels = torch.cat(rec_labels, dim=0)
            all_t_idxes = torch.cat(all_t_idxes, dim=0)

            return labels, preds, rec_labels, rec_outs, all_t_idxes

    loss_meters = {}
    loss_meters['loss'] = AverageMeter('Loss', fmt=':.3f')
    loss_meters.update({k: AverageMeter(k, fmt=':.3f') for k, _ in cfgs['loss_weights'].items()})
    progress = ProgressMeter(num_batches=len(val_loader) + (is_distributed and (len(val_loader.sampler) * cfgs['world_size'] < len(val_loader.dataset))),
                             meters=loss_meters.values(),
                             prefix=prefix)

    model.eval()
    labels, preds, rec_labels, rec_outs, t_idxs = run_validate(val_loader)

    if is_distributed:
        labels_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        preds_list = [torch.zeros_like(preds) for _ in range(dist.get_world_size())]
        rec_labels_list = [torch.zeros_like(rec_labels) for _ in range(dist.get_world_size())]
        rec_outs_list = [torch.zeros_like(rec_outs) for _ in range(dist.get_world_size())]
        t_idxs_list = [torch.zeros_like(t_idxs) for _ in range(dist.get_world_size())]
        dist.all_gather(labels_list, labels, async_op=False)
        dist.all_gather(preds_list, preds, async_op=False)
        dist.all_gather(rec_labels_list, rec_labels, async_op=False)
        dist.all_gather(rec_outs_list, rec_outs, async_op=False)
        dist.all_gather(t_idxs_list, t_idxs, async_op=False)

        for k, meter in loss_meters.items():
            meter.all_reduce()
        if len(val_loader.sampler) * cfgs['world_size'] < len(val_loader.dataset):
            aux_val_dataset = Subset(val_loader.dataset,
                                     range(len(val_loader.sampler) * cfgs['world_size'], len(val_loader.dataset)))
            aux_val_loader = torch.utils.data.DataLoader(
                aux_val_dataset, batch_size=cfgs['b_size'], shuffle=False,
                num_workers=cfgs['data_workers'], pin_memory=True)
            aux_labels, aux_preds, aux_rec_labels, aux_rec_outs, aux_t_idxs = run_validate(aux_val_loader, len(val_loader))
            labels_list.append(aux_labels)
            preds_list.append(aux_preds)
            t_idxs_list.append(aux_t_idxs)
            rec_labels_list.append(aux_rec_labels)
            rec_outs_list.append(aux_rec_outs)

        labels = torch.cat(labels_list, dim=0)  # [n_samples, pred_len, dim]
        preds = torch.cat(preds_list, dim=0)  # [n_samples, pred_len, dim] or [n_samples, history_len, pred_len, dim]
        rec_labels = torch.cat(rec_labels_list, dim=0)
        rec_outs = torch.cat(rec_outs_list, dim=0)
        t_idxs = torch.cat(t_idxs_list, dim=0)
        
        if dist.get_rank() == 0:
            if not cfgs['use_mean_embedds']:

                last_preds = get_final_preds(preds, cfgs['coupled_len'], cfgs['pred_len'], pred_type='last')
                mean_preds = get_final_preds(preds, cfgs['coupled_len'], cfgs['pred_len'], pred_type='mean')
                progress.display_summary()
                last_mae, last_mse, last_rmse, last_mape, last_mspe = metrics.metric(np.array(last_preds.cpu()), np.array(labels.cpu()))
                mean_mae, mean_mse, mean_rmse, mean_mape, mean_mspe = metrics.metric(np.array(mean_preds.cpu()), np.array(labels.cpu()))
                print('Last_mse:{}, mae:{}; Mean_mse:{}, mae:{}'.format(last_mse, last_mae, mean_mse, mean_mae))
            else:
                progress.display_summary()
                mae, mse, rmse, mape, mspe = metrics.metric(np.array(preds.cpu()), np.array(labels.cpu()))
                print('mse:{}, mae:{}'.format(mse, mae))
    else:
        if not cfgs['use_mean_embedds']:

            last_preds = get_final_preds(preds, cfgs['coupled_len'], cfgs['pred_len'], pred_type='last')
            mean_preds = get_final_preds(preds, cfgs['coupled_len'], cfgs['pred_len'], pred_type='mean')
            progress.display_summary()
            last_mae, last_mse, last_rmse, last_mape, last_mspe = metrics.metric(np.array(last_preds.cpu()), np.array(labels.cpu()))
            mean_mae, mean_mse, mean_rmse, mean_mape, mean_mspe = metrics.metric(np.array(mean_preds.cpu()), np.array(labels.cpu()))
            print('Last_mse:{}, mae:{}; Mean_mse:{}, mae:{}'.format(last_mse, last_mae, mean_mse, mean_mae))
        else:
            progress.display_summary()
            mae, mse, rmse, mape, mspe = metrics.metric(np.array(preds.cpu()), np.array(labels.cpu()))
            print('mse:{}, mae:{}'.format(mse, mae))

    reindex_idxs = torch.argsort(t_idxs)
    preds = preds[reindex_idxs]
    labels = labels[reindex_idxs]
    rec_labels = rec_labels[reindex_idxs]
    rec_outs = rec_outs[reindex_idxs]

    return {k: v.avg if type(v.avg) is torch.Tensor else torch.tensor(v.avg) for k, v in loss_meters.items()},\
        [labels, preds, rec_labels, rec_outs]



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.n = 0

    def update(self, val, n=1):
        self.val = val
        self.n = n
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reduce_val(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.val * self.n, self.n], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        total_val, total_n = total.tolist()
        self.val = total_val / total_n

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.batch = 0

    def display(self, batch):
        self.batch = batch
        entries = [self.prefix + ' ' + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\r'+', '.join(entries), end='', flush=True)

    def display_summary(self):
        entries = [self.prefix + ' ' + self.batch_fmtstr.format(self.batch)]
        entries += [meter.summary() for meter in self.meters]
        print('\r'+', '.join(entries), end='', flush=True)
        print('')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return 'batch: [' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()
    