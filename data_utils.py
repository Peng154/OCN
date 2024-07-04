import os

from PIL import Image
import numpy as np
import pickle

import torchvision
from torchvision import transforms as T

import torch
from torch.utils.data import Dataset, Sampler
from typing import Dict, List
import pandas as pd
import numpy as np


def get_lorenz_matrix(n, time=100, step=0.02, c=0.1, time_invariant=True, init_way='uniform', init_param=None):
    length = int(time / step)  #
    x = np.zeros((n * 3, length), dtype=np.float32)  
    if init_way == 'uniform':
        x[:, 0] = np.random.rand(n * 3) 
    elif init_way == 'norm':
        x[:, 0] = np.random.randn(n * 3) * init_param['std'] + init_param['mean']
    sigma = 10.0

    for i in range(1, length):

        if not time_invariant:
            sigma = 10.0 + 0.1 * i % 10

        x[0, i] = x[0, i - 1] + step * (sigma * (x[1, i - 1] - x[0, i - 1]) + np.sign(n - 1) * c * x[(n - 1) * 3, i - 1])
        x[1, i] = x[1, i - 1] + step * (28 * x[0, i - 1] - x[1, i - 1] - x[0, i - 1] * x[2, i - 1])
        x[2, i] = x[2, i - 1] + step * (-8 / 3 * x[2, i - 1] + x[0, i - 1] * x[1, i - 1])

        for j in range(1, n):
            x[3 * j, i] = x[3 * j, i - 1] + step * (
                        10 * (x[3 * j + 1, i - 1] - x[3 * j, i - 1]) + c * x[3 * (j - 1), i - 1])
            x[3 * j + 1, i] = x[3 * j + 1, i - 1] + step * (
                        28 * x[3 * j, i - 1] - x[3 * j + 1, i - 1] - x[3 * j, i - 1] * x[3 * j + 2, i - 1])
            x[3 * j + 2, i] = x[3 * j + 2, i - 1] + step * (
                        -8 / 3 * x[3 * j + 2, i - 1] + x[3 * j, i - 1] * x[3 * j + 1, i - 1])

    return x.T


def load_lorenz_data(data_dir, n, skip_time_num, time=100, step=0.02, c=0.1, time_invariant=True,
                     init_way='uniform', init_param=None):

    data_file_name = 'lorenz_({})_n={}_time={}_step={}_c={}_init_way={}_init_param={}.pkl'.format(
        'time_invariant' if time_invariant else 'time-variant',
        n, time, step, c, init_way, init_param)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = None
    file_path = os.path.join(data_dir, data_file_name)

    if os.path.exists(file_path):
        print('load data from {}'.format(file_path))
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = get_lorenz_matrix(n, time, step, c, time_invariant, init_way, init_param)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    return data[skip_time_num:]


def get_time_sample_idxs(total_time_len, coupled_len, pred_len, n_samples=None, skip_rate=1):
    all_idxs = np.arange(0, total_time_len - (coupled_len+pred_len), skip_rate)
    if n_samples:
        return all_idxs[:n_samples]
    else:
        return all_idxs


def get_noOverlap_select_idxs(total_time_len, train_len, embedding_len, n_samples=None):

    max_nb_samples = (total_time_len - embedding_len - 1) // train_len
    assert 0 < n_samples < max_nb_samples, 'nb_samples is too large!'
    if n_samples is not None:
        return np.array([(i+1) * train_len for i in range(max_nb_samples)])[:n_samples]
    else:
        return np.array([(i + 1) * train_len for i in range(max_nb_samples)])


class Classification_Dataset(Dataset):
    def __init__(self, data_name, root, download, train, transform: List, **kwargs):
        if data_name in torchvision.datasets.__dict__:
            if data_name == 'SVHN':
                self.dataset = torchvision.datasets.__dict__[data_name](root=root, download=download,
                                                                        split='train' if train else 'test',
                                                                        transform=None)
            else:
                self.dataset = torchvision.datasets.__dict__[data_name](root=root, download=download, train=train, transform=None)
        else:
            self.dataset = custom_datasets[data_name](root=root, train=train, **kwargs)

        print(transform)

        self.before_transform = None
        self.after_transform = None
        self.add_noise = None

        self.add_noise_idx = -1
        for i, t in enumerate(transform):
            if type(t) is GuassionNoise or type(t) is GuassionNoiseGPU:
                add_noise_idx = i
                break
        # assert add_noise_idx == len(transform) - 1 or add_noise_idx == -1, 'Add noise must be the last one!'
        if self.add_noise_idx != -1:
            self.before_transform = torchvision.transforms.Compose(transform[:self.add_noise_idx])
            self.add_noise = transform[self.add_noise_idx]
            if self.add_noise_idx == len(transform) - 1:  # 加噪声在最后
                self.after_transform = lambda x: x
            else:
                self.after_transform = torchvision.transforms.Compose(transform[self.add_noise_idx+1:])
        else:
            self.before_transform = torchvision.transforms.Compose(transform)
            self.add_noise = lambda x: x
            self.after_transform = lambda x: x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        rec_img = self.before_transform(img)
        if self.add_noise_idx == -1:
            t_img = rec_img
        else:
            t_img = self.add_noise(rec_img)
            imgs = torch.stack((rec_img, t_img), dim=0)
            imgs = self.after_transform(imgs)
            rec_img, t_img = imgs[0], imgs[1]

        return t_img, label, rec_img


# Custom Classification datasets
class CustomCIFAR100(Dataset):
    def __init__(self, root, train, coarse_label=False, transform=None, *args, **kwargs):
        with open(os.path.join(root, 'cifar-100-python/train' if train else 'cifar-100-python/test'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.raw_data = data[b'data']
            if coarse_label:
                self.labels = data[b'coarse_labels']
            else:
                self.labels = data[b'fine_labels']
        self.data_len = len(self.raw_data)
        self.img_data = [Image.fromarray(img, 'RGB') for img in self.raw_data.reshape(self.data_len, 3, 32, 32).transpose(0, 2, 3, 1)]
        if transform is None:
            self.transform = lambda x:x
        else:
            self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.transform(self.img_data[idx]), self.labels[idx]

custom_datasets = {'CustomCIFAR100': CustomCIFAR100}

def get_dataset(cfgs):
    data_class_map = {
        'Lorenz': LorenzDataset,
        'LorenzNoise': LorenzDataset,
        'ETTh1': Dataset_Time_Seires,
        'ETTh2': Dataset_Time_Seires,
        'ETTm1': Dataset_Time_Seires,
        'ETTm2': Dataset_Time_Seires,
        'WTH': Dataset_Time_Seires,
        'Weather': Dataset_Time_Seires,
        'ECL': Dataset_Time_Seires
    }
    boarders_map = {
        'ETTh1': [[0, 12 * 30 * 24 - cfgs['coupled_len'], 12 * 30 * 24 + 4 * 30 * 24 - cfgs['coupled_len']],
                [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]],
        'ETTh2': [[0, 12 * 30 * 24 - cfgs['coupled_len'], 12 * 30 * 24 + 4 * 30 * 24 - cfgs['coupled_len']],
                [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]],
        'ETTm1': [[0 * 30 * 24 * 4, 12 * 30 * 24 * 4 - cfgs['coupled_len'], 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - cfgs['coupled_len']],
                  [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]],
        'ETTm2': [[0 * 30 * 24 * 4, 12 * 30 * 24 * 4 - cfgs['coupled_len'], 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - cfgs['coupled_len']],
                  [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]],
        'WTH': [[0, 28 * 30 * 24 - cfgs['coupled_len'], 28 * 30 * 24 + 10 * 30 * 24 - cfgs['coupled_len']],
                [28 * 30 * 24, 28 * 30 * 24 + 10 * 30 * 24, 28 * 30 * 24 + 20 * 30 * 24]],
        'ECL': [[0, 15 * 30 * 24 - cfgs['coupled_len'], 15 * 30 * 24 + 3 * 30 * 24 - cfgs['coupled_len']],
                [15 * 30 * 24, 15 * 30 * 24 + 3 * 30 * 24, 15 * 30 * 24 + 7 * 30 * 24]],
        'Weather': [[0, int(51840 * 0.7) - cfgs['coupled_len'], int(51840 * 0.8) - cfgs['coupled_len']],
                  [int(51840 * 0.7), int(51840 * 0.8), 51840]],
    }
    data_name = cfgs['data_name']
    DataClass = data_class_map[data_name]
    if data_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'WTH', 'Weather', 'ECL']:
        boarders = boarders_map[data_name]
        train_dataset = DataClass(cfgs['data_dir'], data_path=cfgs['data_file'],
                                  size=[cfgs['coupled_len'], cfgs['pred_len']], flag='train',
                                  features='M', scale=cfgs['z_score'], inverse=cfgs['inverse_out'], boarders=boarders)
        val_dataset = DataClass(cfgs['data_dir'], data_path=cfgs['data_file'],
                                 size=[cfgs['coupled_len'], cfgs['pred_len']], flag='val', return_t_idx=True,
                                 features='M', scale=cfgs['z_score'], inverse=cfgs['inverse_out'], boarders=boarders)
        test_dataset = DataClass(cfgs['data_dir'], data_path=cfgs['data_file'],
                                  size=[cfgs['coupled_len'], cfgs['pred_len']], flag='test', return_t_idx=True,
                                  features='M', scale=cfgs['z_score'], inverse=cfgs['inverse_out'], boarders=boarders)
    else:
        train_dataset = DataClass(cfgs['data_dir'], cfgs['coupled_systems_n'], train_coupled_len=cfgs['train_coupled_len'], coupled_len=cfgs['coupled_len'],
                                  n_samples=cfgs['n_samples'], pred_len=cfgs['pred_len'],
                                  mode='train', split_ratios=cfgs['split_ratios'], noise_strength=cfgs['noise_strength'],
                                  lorenz_time=cfgs['time'], lorenz_step=cfgs['step'], skip_time_num=cfgs['skip_time_num'],
                                  z_score=cfgs['z_score'], inverse_out=cfgs['inverse_out'], select_dims=cfgs['select_dims'], target_dim=cfgs['target_dim'],
                                  train_sample_fraction=cfgs['sample_faction'] if 'sample_faction' in cfgs else 1.0)

        val_dataset = DataClass(cfgs['data_dir'], cfgs['coupled_systems_n'], train_coupled_len=cfgs['train_coupled_len'], coupled_len=cfgs['coupled_len'],
                                n_samples=cfgs['n_samples'], pred_len=cfgs['pred_len'], noise_strength=cfgs['noise_strength'],
                                lorenz_time=cfgs['time'], lorenz_step=cfgs['step'], skip_time_num=cfgs['skip_time_num'],
                                mode='val', split_ratios=cfgs['split_ratios'], return_t_idx=True, select_dims=cfgs['select_dims'], target_dim=cfgs['target_dim'],
                                z_score=cfgs['z_score'], inverse_out=cfgs['inverse_out'])
        test_dataset = DataClass(cfgs['data_dir'], cfgs['coupled_systems_n'], train_coupled_len=cfgs['train_coupled_len'], coupled_len=cfgs['coupled_len'],
                                 n_samples=cfgs['n_samples'], pred_len=cfgs['pred_len'], noise_strength=cfgs['noise_strength'],
                                 lorenz_time=cfgs['time'], lorenz_step=cfgs['step'], skip_time_num=cfgs['skip_time_num'],
                                 mode='test', split_ratios=cfgs['split_ratios'], return_t_idx=True, select_dims=cfgs['select_dims'], target_dim=cfgs['target_dim'],
                                 z_score=cfgs['z_score'], inverse_out=cfgs['inverse_out'])
    return train_dataset, val_dataset, test_dataset


# Time series prediction datasets
class TimeBatchSampler(Sampler[List[int]]):
    def __init__(self, data_source, batch_size: int, sample_stride: int,
                 shuffle=True, drop_last=False):

        self.samples_count = len(data_source)  
        self.sample_stride = sample_stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.strided_idxs_count = ((self.samples_count-1) // self.sample_stride + 1)
        self.last_stride_samples_num = (self.samples_count - 1) % sample_stride  
                
        self.batches_num = self.strided_idxs_count // self.batch_size 
        self.last_batch_samples_num = self.strided_idxs_count % self.batch_size  
        
        print('last_stride_samples_num: ', self.last_stride_samples_num)
        print('last_batch_samples_num: ', self.last_batch_samples_num)
        
        assert not (shuffle and drop_last), '如果drop_last=True, shuffle必须为False'

    def __iter__(self):
        if self.drop_last:
            iter_bidxs = self.idxs.unfold(0, self.batch_size, self.batch_size).tolist()
        else:
            if self.shuffle:
                start_idxs = np.random.choice(self.last_stride_samples_num + 1, 1)[0] 
                # print(self.last_samples_num, start_idxs)
                tmp_idxs = torch.arange(self.samples_count)[start_idxs::self.sample_stride]
            else:
                tmp_idxs = torch.arange(self.samples_count)[::self.sample_stride]
            iter_bidxs = tmp_idxs.unfold(0, self.batch_size, self.batch_size).tolist()
            if self.last_batch_samples_num != 0:
                iter_bidxs.extend([tmp_idxs[-self.last_batch_samples_num:].tolist()])
        return iter(iter_bidxs)

    def __len__(self):
        return self.batches_num


class LorenzDataset(Dataset):
    def __init__(self, data_dir, coupled_system_n, n_samples, train_coupled_len: int, coupled_len: int, pred_len, lorenz_time, lorenz_step, noise_strength,
                 skip_time_num, mode='train', split_ratios=[0.8, 0.1, 0.1], select_dims=None, target_dim=None,
                 return_t_idx=False, z_score=False, inverse_out=False, train_sample_fraction=1.0):

        data = self._load_data(data_dir, coupled_system_n, time=lorenz_time, step=lorenz_step,
                               skip_time_num=skip_time_num, noise_strength=noise_strength)   # [t_len, coupled_system_n*3]
        
        if select_dims is not None:
            if type(select_dims) is list:
                select_dims = np.array(select_dims)
                data = data[:, select_dims]
            elif type(select_dims) is int:
                data = data[:, :select_dims]
        
        self.target_dim = target_dim
        self.coupled_len = coupled_len
        self.pred_len = pred_len
        self.data_dim = data.shape[1]
        self.output_dim = data.shape[1] if target_dim is None else 1
        self.mode = mode
        self.return_t_idxs = return_t_idx
        self.z_score = z_score
        self.inverse_out = inverse_out

        sample_idxs = get_time_sample_idxs(data.shape[0], coupled_len=coupled_len,
                                           pred_len=pred_len, n_samples=n_samples, skip_rate=1)

        samples_num = sample_idxs.shape[0]
        assert train_coupled_len >= coupled_len, 'train_coupled_len must be greater than coupled_len'
        cl_er = train_coupled_len - coupled_len  # >= 0
        data_idxs = {'train': sample_idxs[:int(samples_num * split_ratios[0])],
                     'val': sample_idxs[int(samples_num * split_ratios[0]): int(samples_num * (split_ratios[0] + split_ratios[1]))],
                     'test': sample_idxs[int(samples_num * (split_ratios[0] + split_ratios[1])):]}

        if train_sample_fraction != 1.0:
            train_idxs = data_idxs['train']
            np.random.shuffle(train_idxs)
            train_idxs = train_idxs[:int(train_idxs.shape[0] * train_sample_fraction)]
            print(train_idxs[:20])
            data_idxs['train'] = train_idxs

        self.select_dims = data_idxs[self.mode]
        
        # z-score normalization 
        if not z_score:
            self.data_x = data
            self.data_y = data[:, target_dim] if target_dim is not None else data
            if self.data_y.ndim == 1:
                self.data_y = self.data_y[:, np.newaxis]
            self.normalizer = None
        else:
            train_data = data[np.min(data_idxs['train']):np.max(data_idxs['train']) + coupled_len]
            self.normalizer = ZScoreNormalizer().fit(train_data)
            self.data_x = self.normalizer.transform(data)
            if self.inverse_out:
                self.data_y = data[:, target_dim] if target_dim is not None else data
            else:
                self.data_y = self.data_x[:, target_dim] if target_dim is not None else self.data_x
            if self.data_y.ndim == 1:
                self.data_y = self.data_y[:, np.newaxis]
        
        if self.mode == 'train':
            if cl_er != 0:
                self.select_dims = self.select_dims[:-cl_er]
            self.coupled_len = train_coupled_len

    def _load_data(self, data_dir, coupled_system_n, time, step, skip_time_num=2000, noise_strength=0.0):

        lorenz_data = load_lorenz_data(data_dir, coupled_system_n, skip_time_num=skip_time_num,
                                       time_invariant=True, time=time, step=step, init_way='uniform', init_param=None)
        if noise_strength != 0:
            print('add noise to data, noise strength: ', noise_strength)
            np.random.seed(666)  
            lorenz_data += np.random.normal(0, noise_strength, lorenz_data.shape)
        print('lorenz_data shape: ', lorenz_data.shape)
        return lorenz_data

    def __len__(self):
        return len(self.select_dims)

    def __getitem__(self, item):
        idx = self.select_dims[item]
        input_data = self.data_x[idx: idx + self.coupled_len]
        rec_label = self.data_y[idx: idx + self.coupled_len]
        pred_label = self.data_y[idx + self.coupled_len: idx + self.coupled_len + self.pred_len]
        if self.return_t_idxs:
            return input_data, rec_label, pred_label, idx
        else:
            return input_data, rec_label, pred_label

class Dataset_Time_Seires(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features=None, data_path=None,
                 target=None, scale=True, inverse=False, return_t_idx=False, boarders=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.return_t_idx = return_t_idx
        self.boarders = boarders
        self.__read_data__()
        self.data_dim = self.data_x.shape[1]

        self.full_label_select_idxs = np.arange(self.pred_len + 1)[np.newaxis, :] + np.arange(self.seq_len)[:, np.newaxis]

    def __read_data__(self):
        self.normalizer = ZScoreNormalizer()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # print(df_raw.shape)
        # print(df_raw)
        border1s = self.boarders[0]
        border2s = self.boarders[1]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        df_data = df_data.astype(np.float32)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  # only train data
            # train_data = df_data[0:border2s[0]]  # only train data
            self.normalizer.fit(train_data.values)
            data = self.normalizer.transform(df_data.values)
        else:
            data = df_data.values


        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2] 
        else:
            self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        input_x = self.data_x[s_begin:s_end]

        rec_label = self.data_y[s_begin: s_end]

        pred_label = self.data_y[r_begin: r_end]

        if self.return_t_idx:
            return input_x, rec_label, pred_label, index
        else:
            return input_x, rec_label, pred_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.normalizer.inverse_transform(data)


class ZScoreNormalizer(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data:np.ndarray):
        """

        :param data: [seq_len, data_dim]
        :return:
        """
        self.mean = data.mean(0)
        self.std = data.std(0)
        return self


    def _get_mean_std(self, data):
        is_tensor = torch.is_tensor(data)
        if is_tensor:
            device = data.device
            mean = torch.tensor(self.mean, dtype=data.dtype, device=device)
            std = torch.tensor(self.std, dtype=data.dtype, device=device)
        else:
            mean = self.mean
            std = self.std
        return mean, std

    def transform(self, data):
        """

        :param data: ndarray/torch.Tensor [..., data_dim]
        :return:
        """
        mean, std = self._get_mean_std(data)
        return (data - mean)/std

    def inverse_transform(self, data):
        """

        :param data: ndarray/torch.Tensor [...., data_dim]
        :return:
        """
        mean, std = self._get_mean_std(data)
        return data * std + mean


# Custom data augmentation classes
class GuassionNoise(object):
    def __init__(self, mode=None, mean=0, std=1e-2, clip_range=False, p=1.0):

        self.mode = mode
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        # print(self.clip_range)
        self.p = p

    def __call__(self, input_img: torch.Tensor):
        """

        :param input_img: (c, h, w)
        :return:
        """
        tmp_p = np.random.uniform()
        if tmp_p <= self.p:
            noise_m = torch.randn(*input_img.shape[-2:]) * self.std + self.mean
            if self.clip_range:
                out_img = torch.clip(input_img + noise_m, min=self.clip_range[0], max=self.clip_range[1])
            else:
                out_img = input_img + noise_m
        else:
            out_img = input_img
        return out_img


class GuassionNoiseGPU(torch.nn.Module):
    def __init__(self, mode=None, mean=0, std=1e-2, clip_range=False, p=1.0):

        super(GuassionNoiseGPU, self).__init__()
        self.mode = mode
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        # print(self.clip_range)
        self.p = p

    def forward(self, input_img: torch.Tensor):
        """

        :param input_img: (batch, c, h, w)
        :return:
        """
        tmp_p = torch.rand(input_img.shape[0], device=input_img.device)
        tmp_p = torch.where(tmp_p < self.p, torch.ones_like(tmp_p), torch.zeros_like(tmp_p))
        noise_m = torch.randn((input_img.shape[0], input_img.shape[2], input_img.shape[3]), device=input_img.device) * self.std + self.mean
        noise_m = (noise_m * tmp_p[:, None, None]).unsqueeze(1)
        out_img = input_img + noise_m
        if self.clip_range:
            out_img = torch.clip(out_img, min=self.clip_range[0], max=self.clip_range[1])
        else:
            out_img = out_img
        return out_img

custom_aug = {'GuassionNoise': GuassionNoise, 'GuassionNoiseGPU': GuassionNoiseGPU}


def get_model_N_L(cfgs):
    hidden_size = cfgs['model_params']['hidden_size']
    if type(hidden_size) is not list:
        hidden_size = [hidden_size]
    hidden_size = hidden_size[-1]
    N = hidden_size - (cfgs['model_params']['kernel_size'] - 1) * len(cfgs['model_params']['channels'])
    L = cfgs['model_params']['recurrent_times'] // N
    return N, L


def get_aug_list(cfgs):
    if cfgs['data_aug']:
        aug_list = []
        for aug_n, aug_p in cfgs['aug_params'].items():
            if aug_n in custom_aug:
                aug_t = custom_aug[aug_n](**aug_p)
            else:
                aug_t = T.__dict__[aug_n](**aug_p)
            aug_list.append(aug_t)
        return aug_list
    else:
        return []


def process_weather_data(wth_dir):
    wth_file = 'mpi_roof_2022.csv'
    wth_data = pd.read_csv(os.path.join(wth_dir, wth_file))
    print(wth_data.shape)
    print(wth_data.head())