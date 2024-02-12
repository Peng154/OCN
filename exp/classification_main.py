import copy
import os
import torch
import sys
sys.path.append('..')
from torch.utils.data import DataLoader
import data_utils
import utils
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision
import model_h_vector_embed_mean
import numpy as np
import yaml
from itertools import product
import pandas as pd

def run():
    cfg_file_path = '../cfg/class_cifar10_tcn_model (h_vector_embed) gpu_aug.yml' 

    with open(cfg_file_path, 'r') as file:
        cfgs = yaml.safe_load(file)

    train(cfgs, cfg_file_path)


def build_gpu_transform(gpu_transform_list):
    gpu_before_noise_transform = None
    gpu_noise_transform = None
    gpu_after_noise_transform = None

    if gpu_transform_list is not None:
        noise_idx = -1
        for i, t in enumerate(gpu_transform_list):
            if type(t) == data_utils.GuassionNoiseGPU:
                noise_idx = i
                break

        if noise_idx == -1:
            gpu_before_noise_transform = gpu_transform_list
            if gpu_before_noise_transform is not None and type(gpu_before_noise_transform) is not list:
                gpu_before_noise_transform = [gpu_before_noise_transform]
            gpu_before_noise_transform = torch.nn.Sequential(
                *gpu_before_noise_transform) if gpu_before_noise_transform is not None else None
            return gpu_before_noise_transform, gpu_noise_transform, gpu_after_noise_transform

        gpu_before_noise_transform = gpu_transform_list[:noise_idx] if noise_idx != 0 else None
        gpu_noise_transform = gpu_transform_list[noise_idx]
        gpu_after_noise_transform = gpu_transform_list[noise_idx + 1:] if noise_idx != len(
            gpu_transform_list) - 1 else None

        if gpu_before_noise_transform is not None and type(gpu_before_noise_transform) is not list:
            gpu_before_noise_transform = [gpu_before_noise_transform]
        if gpu_after_noise_transform is not None and type(gpu_after_noise_transform) is not list:
            gpu_after_noise_transform = [gpu_after_noise_transform]
        gpu_before_noise_transform = torch.nn.Sequential(
            *gpu_before_noise_transform) if gpu_before_noise_transform is not None else None
        gpu_after_noise_transform = torch.nn.Sequential(
            *gpu_after_noise_transform) if gpu_after_noise_transform is not None else None

    return gpu_before_noise_transform, gpu_noise_transform, gpu_after_noise_transform


def train(cfgs, cfg_file_path, base_log_dir='../log'):
    utils.setup_seed(cfgs['seed'])

    print(f'Dataset: {cfgs["data_name"]}, Model type: {cfgs["model_name"]}')
    data_dir = f'../data/{cfgs.get("data_dir", cfgs["data_name"])}'
    channel_means = (0,)
    channel_stds = (1,)
    if cfgs['z_score']:
        print('z-score the data')
        train_data = []
        if cfgs['data_name'] not in ['CustomCIFAR100', 'SVHN']:
            train_dataset = torchvision.datasets.__dict__[cfgs['data_name']](root=data_dir, download=True, train=True,
                                                                             transform=transforms.ToTensor())
        elif cfgs['data_name'] == 'CustomCIFAR100':
            train_dataset = data_utils.custom_datasets[cfgs['data_name']](root=data_dir, train=True,
                                                                          transform=transforms.ToTensor())
        elif cfgs['data_name'] == 'SVHN':
            train_dataset = torchvision.datasets.SVHN(root=data_dir, download=True, split='train',
                                                      transform=transforms.ToTensor())
        for d in train_dataset:
            train_data.append(d[0])
        train_data = torch.stack(train_data, dim=1)
        channel_means = torch.mean(train_data, dim=[1, 2, 3])
        channel_stds = torch.std(train_data, dim=[1, 2, 3])

    trans_list = [transforms.ToTensor(),
                  transforms.Normalize(mean=channel_means, std=channel_stds)]

    eval_transform = copy.deepcopy(trans_list)
    trans_list.extend(data_utils.get_aug_list(cfgs)) 
    # print(trans_list)
    train_transform = trans_list

    if cfgs['gpu_aug']:
        gpu_train_transform = train_transform[1:]
        gpu_eval_transform = eval_transform[1:]
        cpu_train_transform = [train_transform[0]] 
        cpu_eval_transform = [eval_transform[0]]
    else:
        gpu_train_transform = None
        gpu_eval_transform = None
        cpu_train_transform = train_transform
        cpu_eval_transform = eval_transform

    data_params = cfgs.get('data_params', {})
    train_dataset = data_utils.Classification_Dataset(data_name=cfgs['data_name'], root=data_dir, download=True,
                                                      train=True,
                                                      transform=cpu_train_transform, **data_params)
    test_dataset = data_utils.Classification_Dataset(data_name=cfgs['data_name'], root=data_dir, download=True,
                                                     train=False,
                                                     transform=cpu_eval_transform, **data_params)

    print(len(train_dataset))

    train_data_loader = DataLoader(train_dataset, batch_size=cfgs['b_size'], shuffle=True, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=cfgs['b_size'], shuffle=True, pin_memory=True)

    input_size = torch.prod(torch.tensor(train_dataset[0][0].shape)).numpy()
    print(input_size)

    device = torch.device(f"cuda:{cfgs['gpu_idx']}" if torch.cuda.is_available() else "cpu")

    m = model_h_vector_embed_mean.TaskModel(base_model=cfgs['model_name'], task_type='classification',
                                       class_num=cfgs['class_num'], input_size=input_size, output_size=input_size,
                                       **cfgs['model_params'])
    m = m.to(device)
    optimizer = optim.Adam(m.parameters(), lr=cfgs['lr'], weight_decay=cfgs['l2_weight'])

    scheduler = lr_scheduler.__dict__[cfgs['lr_scheduler']](optimizer, **cfgs['lr_scheduler_params'])

    epoches = cfgs['epoches']
    loss_weights = cfgs['loss_weights']
    print(loss_weights)

    best_eval_acc = 0.0
    best_eval_epoch = -1

    N, L = data_utils.get_model_N_L(cfgs)
    log_dir_name = utils.get_time_log_dir('classification',
                                          data=cfgs['data_name'],
                                          model_name=cfgs['model_name'],
                                          N=N, L=L)

    log_dir = os.path.join(base_log_dir, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    tmp_cfg_path = utils.cp_file2dir(cfg_file_path, log_dir)

    gpu_before_noise_train_transform, gpu_noise_train_transform, gpu_after_noise_train_transform = build_gpu_transform(gpu_train_transform)
    gpu_before_noise_eval_transform, _, _ = build_gpu_transform(gpu_eval_transform)

    for ei in range(epoches):
        m.train()

        all_loss_dict = {'loss': []}
        train_accs = []
        batch_num = len(train_data_loader)
        for i, batch_data in enumerate(train_data_loader):
            batch_data = [data.to(device) for data in batch_data]
            input_img, true_labels, rec_img = batch_data
            bs = input_img.shape[0]

            if gpu_train_transform is not None:
                if gpu_before_noise_train_transform is not None:
                    input_img = gpu_before_noise_train_transform(input_img)
                    rec_img = gpu_before_noise_train_transform(rec_img)
                input_img = gpu_noise_train_transform(input_img)
                if gpu_after_noise_train_transform is not None:
                    input_img = gpu_after_noise_train_transform(input_img)
                    rec_img = gpu_after_noise_train_transform(rec_img)

            input_img = torch.reshape(input_img, (bs, -1))
            rec_img = torch.reshape(rec_img, (bs, -1))

            out_dict, loss_dict = m(input_img, class_labels=true_labels, rec_x_labels=rec_img) 

            loss = 0
            verbose_str = f'epoch: {ei + 1}/{epoches}, samples: {i + 1} / {batch_num}'
            for l_name, l in loss_dict.items():
                tmp_l = l * loss_weights[f'{l_name}_param']
                loss += tmp_l
                verbose_str += f', {l_name}: {tmp_l.cpu().item():.3}'

                # 保存记录，供输出
                if l_name not in all_loss_dict:
                    all_loss_dict[l_name] = []
                all_loss_dict[l_name].append(tmp_l.cpu().item())

            all_loss_dict['loss'].append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_labels = torch.argmax(out_dict['class_softmax'], dim=-1)
                acc = torch.mean((pred_labels == true_labels).to(torch.float32)).detach().cpu()
                train_accs.append(acc)
            verbose_str += f', loss: {loss:.3f}, train_acc: {np.mean(train_accs):.4}'
            print('\r' + verbose_str, end='', flush=True)

        verbose_str = f'epoch: {ei + 1}/{epoches}, samples: {batch_num} / {batch_num}'
        for l_name, l in all_loss_dict.items():
            if np.mean(l) != 0:
                verbose_str += f', {l_name}: {np.mean(l):.3}'
        verbose_str += f', train_acc: {np.mean(train_accs):.4}'
        print('\r' + verbose_str, end='\n', flush=True)

        if (ei + 1) % cfgs['eval_interval'] == 0 or (ei + 1) == epoches:
            m.eval()
            all_loss_dict = {'eval_loss': []}
            eval_accs = []
            batch_num = len(test_data_loader)
            with torch.no_grad():
                for i, batch_data in enumerate(test_data_loader):
                    batch_data = [data.to(device) for data in batch_data]
                    input_img, true_labels, rec_img = batch_data
                    if gpu_eval_transform is not None:
                        input_img = gpu_before_noise_eval_transform(input_img)
                        rec_img = gpu_before_noise_eval_transform(rec_img)

                    bs = input_img.shape[0]
                    input_img = torch.reshape(input_img, (bs, -1))
                    rec_img = torch.reshape(rec_img, (bs, -1))

                    out_dict, loss_dict = m(input_img, class_labels=true_labels, rec_x_labels=rec_img)
                    loss = 0
                    verbose_str = f'Eval epoch: {ei + 1}/{epoches}, samples: {i + 1} / {batch_num}'
                    for l_name, l in loss_dict.items():
                        tmp_l = l * loss_weights[f'{l_name}_param']
                        loss += tmp_l
                        verbose_str += f', {l_name}: {tmp_l.cpu().item():.3}'

                        if l_name not in all_loss_dict:
                            all_loss_dict[l_name] = []
                        all_loss_dict[l_name].append(tmp_l.cpu().item())
                    all_loss_dict['eval_loss'].append(loss.cpu().item())

                    pred_labels = torch.argmax(out_dict['class_softmax'], dim=-1)
                    acc = torch.mean((pred_labels == true_labels).to(torch.float32)).detach().cpu()
                    eval_accs.append(acc)

                    verbose_str += f', eval_loss: {loss:.3f}, eval_acc: {np.mean(eval_accs):.4}'
                    print('\r' + verbose_str, end='', flush=True)

                verbose_str = f'Eval epoch: {ei + 1}/{epoches}, samples: {batch_num} / {batch_num}'
                for l_name, l in all_loss_dict.items():
                    if np.mean(l) != 0:
                        verbose_str += f', {l_name}: {np.mean(l):.3}'
                verbose_str += f', eval_acc: {np.mean(eval_accs):.4}'
                print('\r' + verbose_str, end='\n', flush=True)
                eval_acc = np.mean(eval_accs)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_eval_epoch = ei + 1

        scheduler.step(np.mean(all_loss_dict['eval_loss']))

    print(f'Best eval ACC: {best_eval_acc}, epoch: {best_eval_epoch}.')
    log_cfg_name = f'({best_eval_acc * 100 :.2f}, e:{best_eval_epoch}) ' + cfg_file_path.split('/')[-1]
    os.rename(tmp_cfg_path, os.path.join(log_dir, log_cfg_name))


if __name__ == '__main__':
    run()


