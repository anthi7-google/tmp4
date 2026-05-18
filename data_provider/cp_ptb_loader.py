import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Dataset_PTB(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ptbxl_data.npy',
                 target=None, scale=True, timeenc=0, freq='h',
                 channel=0, meta_path='ptbxl_meta.csv',
                 in_memory=True):
        # basic split flag
        assert flag in ['train', 'val', 'test']
        self.flag = flag

        # keep the same external interface as the other loaders
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.meta_path = meta_path
        self.channel = channel
        self.in_memory = in_memory

        # PTB uses sample-level crop, not sliding windows
        if size is None:
            self.seq_len = None
            self.label_len = None
            self.pred_len = None
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.__read_data__()

    def __read_data__(self):
        data_file = os.path.join(self.root_path, self.data_path)
        meta_file = os.path.join(self.root_path, self.meta_path)

        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"PTB data file not found: {data_file}")
        if not os.path.isfile(meta_file):
            raise FileNotFoundError(f"PTB meta file not found: {meta_file}")

        # load packed waveform tensor: expected shape (N, T, C)
        if self.in_memory:
            data = np.load(data_file)
        else:
            data = np.load(data_file, mmap_mode='r')

        meta = pd.read_csv(meta_file)

        if data.ndim != 3:
            raise ValueError(f"Expected 3D PTB data, got shape {data.shape}")

        n_samples, total_len, n_channels = data.shape

        if len(meta) != n_samples:
            raise ValueError(
                f"Meta/data size mismatch: len(meta)={len(meta)} vs data.shape[0]={n_samples}"
            )

        if not (0 <= self.channel < n_channels):
            raise ValueError(
                f"Invalid channel index {self.channel}. Valid range is [0, {n_channels - 1}]"
            )

        # split by subset column if available, otherwise derive from strat_fold
        if 'subset' in meta.columns:
            split_mask = meta['subset'] == self.flag
        elif 'strat_fold' in meta.columns:
            if self.flag == 'train':
                split_mask = meta['strat_fold'].between(1, 8)
            elif self.flag == 'val':
                split_mask = meta['strat_fold'] == 9
            else:
                split_mask = meta['strat_fold'] == 10
        else:
            raise ValueError("PTB meta must contain either 'subset' or 'strat_fold'")

        split_indices = np.where(split_mask.values)[0]
        if len(split_indices) == 0:
            raise ValueError(f"No samples found for split: {self.flag}")

        # keep only the selected split
        split_meta = meta.iloc[split_indices].reset_index(drop=True)
        split_data = data[split_indices]

        # front crop if seq_len is set and shorter than the full record length
        if self.seq_len is not None:
            use_len = min(self.seq_len, total_len)
        else:
            use_len = total_len

        split_data = split_data[:, :use_len, :]

        # fit scaler on train split only, then apply to current split
        # scaling is done channel-wise over all time points from all train samples
        if self.scale:
            train_mask = None
            if 'subset' in meta.columns:
                train_mask = meta['subset'] == 'train'
            elif 'strat_fold' in meta.columns:
                train_mask = meta['strat_fold'].between(1, 8)

            train_indices = np.where(train_mask.values)[0]
            train_data = data[train_indices]

            if self.seq_len is not None:
                train_data = train_data[:, :use_len, :]

            self.scaler = StandardScaler()
            train_2d = train_data.reshape(-1, n_channels)
            self.scaler.fit(train_2d)

            split_2d = split_data.reshape(-1, n_channels)
            split_data = self.scaler.transform(split_2d).reshape(split_data.shape)
        else:
            self.scaler = None

        self.meta = split_meta
        self.data_x = split_data.astype(np.float32)
        self.data_y = split_data.astype(np.float32)

        # no calendar/time feature for PTB
        self.data_stamp = None

    def __getitem__(self, index):
        seq_x = self.data_x[index]   # (T, C)
        seq_y = self.data_y[index]   # (T, C)

        # keep raw cropped views for compatibility with the current training code
        seq_x_mark = seq_x.copy()
        seq_y_mark = seq_y.copy()

        # remove target channel from input and keep only that channel as output
        seq_x = np.delete(seq_x, self.channel, axis=1)             # (T, C-1)
        seq_y = seq_y[:, self.channel:self.channel + 1]            # (T, 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # PTB is sample-based, so one ECG record is one sample
        return len(self.data_x)

    def inverse_transform(self, data):
        if self.scaler is None:
            return data

        data = np.asarray(data)

        # if full-channel data comes in, inverse directly
        if data.ndim >= 2 and data.shape[-1] == self.data_x.shape[-1]:
            original_shape = data.shape
            data_2d = data.reshape(-1, original_shape[-1])
            out = self.scaler.inverse_transform(data_2d)
            return out.reshape(original_shape)

        # if single target channel comes in, inverse only that channel
        target_mean = self.scaler.mean_[self.channel]
        target_scale = self.scaler.scale_[self.channel]
        return data * target_scale + target_mean