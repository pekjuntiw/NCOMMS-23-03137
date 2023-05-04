import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd


def loader(dataset, device):
    """
        Data generator
    """
    for data, label in dataset:
        data = data.to(device)
        label = label.to(device)
        yield data, label


class DeltaTransformedECG(data.dataset.Dataset):
    def __init__(self, path, n_class=2, preprocess=True, output_cue_length=120):
        """
            Dataset for spike-encoded ECG

            path
            |  down
            |  |  (csv files here)
            |  up
            |  |  (csv files here)
            :param path:
        """
        ecg_cat = ['N', 'F', 'SVEB', 'VEB']

        if n_class == 2:
            self.classes_numeric = np.arange(n_class).astype(int)
            self.classes = ['N', 'not N']
        elif n_class == 4:
            self.classes_numeric = np.arange(n_class).astype(int)
            self.classes = ecg_cat
        else:
            raise RuntimeError(
                'Currently supports 2 types of classifications only: [N, not N] or [N, F, SVEB, VEB]'
            )

        self.data, self.labels = None, None
        for index, cat in enumerate(ecg_cat):
            up = pd.read_csv(f'{path}/up/up_{cat}_guiyi.csv', header=None).to_numpy()
            down = pd.read_csv(f'{path}/down/down_{cat}_guiyi.csv', header=None).to_numpy()

            # up.shape = down.shape = (n_data, times)
            # we need to combine up and down, with final shape (n_data, times, n_input)
            a = np.transpose(np.asarray([up, down]), axes=[1, 2, 0])
            if self.data is None:
                self.data = a
            else:
                self.data = np.concatenate([self.data, a], axis=0)
            if self.labels is None:
                if n_class == 2:
                    self.labels = np.ones(up.shape[0]) if cat == 'N' \
                        else np.zeros(up.shape[0])
                elif n_class == 4:
                    self.labels = np.ones(up.shape[0]) * index
            else:
                if n_class == 2:
                    self.labels = np.append(
                        self.labels,
                        np.ones(up.shape[0]) if cat == 'N' else np.zeros(up.shape[0])
                    )
                elif n_class == 4:
                    self.labels = np.append(self.labels, np.ones(up.shape[0]) * index)

        # add CUE signal to prompt the model to generate a valid output
        if preprocess:
            self.preprocess(output_cue_length)

        self.data = torch.Tensor(self.data)
        self.labels = torch.Tensor(self.labels)
        self.labels = self.labels.to(dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def preprocess(self, output_cue_length):
        n_data, times, n_input = self.data.shape
        pad_shape = [n_data, output_cue_length, n_input]
        pre_cue_shape = [n_data, times, 1]
        cue_shape = [n_data, output_cue_length, 1]
        pad = np.zeros(pad_shape)
        pre_cue = np.zeros(pre_cue_shape)
        cue = np.ones(cue_shape)
        self.data = np.concatenate([self.data, pad], axis=1)
        self.data = np.concatenate([self.data, np.concatenate([pre_cue, cue], axis=1)], axis=-1)

    def get_classes(self, numeric=True):
        return self.classes_numeric if numeric else self.classes


class DeltaTransformedEEG(data.dataset.Dataset):
    def __init__(self, path, n_class=2, preprocess=True, output_cue_length=120):
        """
            Dataset for spike-encoded EEG

            path
            |  (npy files here)
            :param path:
            Training:
            shape of files = (1000, 18, 1000)
            dim 0 = number of samples, 0-499 is normal, 500-999 is epileptic
            dim 1 = number of channels, 18 for up, 18 for down, so require 36 input channels in total
            dim 2 = timesteps, sampling rate = 800 Hz, so dt = 1.25ms
            or
            shape of files = (2530, 18, 1000)
            0-1264 is normal, 1265-2529 is epileptic
            labels are in label.npy, normal = 0, epilepsy = 1
            Testing:
            shape of files = (2878, 18, 1000)
            labels are in label.npy, normal = 0, epilepsy = 1
        """
        if n_class == 2:
            # normal = 0, epilepsy = 1
            self.classes_numeric = np.arange(n_class).astype(int)
            self.classes = ['N', 'E']
        else:
            raise RuntimeError(
                'Currently supports 1 type of classification only: [Normal, Epilepsy]'
            )

        up = np.load(f'{path}/up.npy')
        down = np.load(f'{path}/down.npy')

        # up.shape = down.shape = (n_data, n_channels, times)
        # we need to combine up and down, with final shape (n_data, times, n_input), where n_input = 2 * n_channels
        # unimportant, but we'll interleave up and down such that for each channel, the up-down pair is together
        self.data = np.empty([up.shape[0], up.shape[2], up.shape[1] + down.shape[1]])
        self.data[..., 1::2] = np.transpose(up, axes=[0, 2, 1])
        self.data[..., ::2] = np.transpose(down, axes=[0, 2, 1])

        if os.path.exists(f'{path}/label.npy'):
            self.labels = np.squeeze(np.load(f'{path}/label.npy'))
        else:
            self.labels = np.concatenate([np.zeros(500), np.ones(500)])

        # add CUE signal to prompt the model to generate a valid output
        if preprocess:
            self.preprocess(output_cue_length)

        self.data = torch.Tensor(self.data)
        self.labels = torch.Tensor(self.labels)
        self.labels = self.labels.to(dtype=torch.long)

        # vr.shape = (n_data, n_channels, times), we need (n_data, times, n_channels)
        self.vr = np.transpose(np.load(f'{path}/Vr.npy'), axes=[0, 2, 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def preprocess(self, output_cue_length):
        n_data, times, n_input = self.data.shape
        pad_shape = [n_data, output_cue_length, n_input]
        pre_cue_shape = [n_data, times, 1]
        cue_shape = [n_data, output_cue_length, 1]
        pad = np.zeros(pad_shape)
        pre_cue = np.zeros(pre_cue_shape)
        cue = np.ones(cue_shape)
        self.data = np.concatenate([self.data, pad], axis=1)
        self.data = np.concatenate([self.data, np.concatenate([pre_cue, cue], axis=1)], axis=-1)

    def get_classes(self, numeric=True):
        return self.classes_numeric if numeric else self.classes

    def get_vr(self, index):
        return self.vr[index]
