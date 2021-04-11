import torch
import torch.nn as nn

import os
import librosa
import numpy as np

from scipy.io import wavfile
from sklearn.decomposition import PCA

from const import Const
from model import forward_submodule


def read_wave(filename):
    sr, wave_data = wavfile.read(filename)
    wave_data = wave_data.astype('float')
    return wave_data / np.max(abs(wave_data))


def wave2spec(wave_data, norm=False):
    # numpy: (257, frames)
    D = librosa.stft(wave_data,
                     n_fft=Const.N_FFT,
                     hop_length=Const.HOP_LENGTH,
                     win_length=Const.WIN_LENGTH,
                     window=Const.WINDOW)

    S = 2 * np.log10(abs(D) + Const.EPSILON)
    phase = np.exp(1j * np.angle(D))
    
    if norm:
        mean = np.mean(S, axis=-1, keepdims=True)
        std = np.std(S, axis=-1, keepdims=True) + Const.EPSILON
        S = (S - mean) / std
    else:
        mean = 0
        std = 1
    
    return S, phase, mean, std


def spec2wave(S, phase, mean=0, std=1):
    D = np.multiply(10**((S * std + mean) / 2), phase)
    return librosa.istft(D,
                         hop_length=Const.HOP_LENGTH,
                         win_length=Const.WIN_LENGTH,
                         window=Const.WINDOW)


def to_TMHINT_name(sample_id):
    return f'TMHINT_{(sample_id-1)//10+1:02d}_{(sample_id-1)%10+1:02d}'


def load_wave_data(sample_id, noise_type=None, SNR_type=None, is_training=True, dataset_path='.', norm=False):
    '''
    return (noisy spec, noisy phase, noisy spec mean, noisy spec std) in numpy form.
    '''
    
    if is_training:
        if noise_type is not None:
            wave_file = os.path.join(dataset_path, f'Training/Noisy/{noise_type}/{sample_id-70}.wav')
        else:
            wave_file = os.path.join(dataset_path, f'Training/Clean/{sample_id-70}.wav')
    else:
        if noise_type is not None:
            wave_file = os.path.join(dataset_path, f'Testing/Noisy/{noise_type}/a1/{SNR_type}/{to_TMHINT_name(sample_id)}.wav')
        else:
            wave_file = os.path.join(dataset_path, f'Testing/Clean/a1/{to_TMHINT_name(sample_id)}.wav')

    wave_data = read_wave(wave_file)
    return wave2spec(wave_data, norm)


def load_elec_data(sample_id, cutoff, elec_channel, dataset_path='.'):
    '''
    return elec :math:`(cutoff length, electrodes)` in numpy form.
    '''
    csv_filename = os.path.join(dataset_path, f'elec/E{sample_id:03d}.csv')
    elec = np.genfromtxt(csv_filename, delimiter=',', dtype=np.float32)
    
    # ===== Extract channels and time shift =====
    # numpy shape: (spec signal length, electrodes)
    elec = elec[:cutoff-Const.SHIFT, elec_channel[0]:elec_channel[1]+1]
    elec = np.vstack([np.zeros((Const.SHIFT, elec.shape[1])), elec])
    return elec


def cache_clean_data(elec_preprocessors=None, is_training=True, split_ratio=1, dataset_path='.', force_update=False, device=None):
    '''
    Args:
        elec_preprocessors: a list of int, (int, int), nn.Module, sklearn.decomposition.PCA or None
            Example:
                elec_preprocessors = [45]        # from 1 to 45 
                elec_preprocessors = [(1, 124)]  # from 1 to 124
                elec_preprocessors = [Encoder]   # input 124
                elec_preprocessors = [None]      # do not load elec data
                elec_preprocessors = [(1, 124), PCA, Encoder]
        
        is_training (bool, optional): split datasets into 'Train' and 'Valid' if is training. Otherwise load 'Test'. Default: ``True``
        force_update (bool, optional): force reload data. Default: ``False``
        device (str, optional): load data into cuda or cpu.  Default: ``None``

    Output: dataset
        - **dataset** is a dictionary with 3 keys, namely 'Train', 'Valid' and 'Test'. For each data set, it
          contains a list of 4 tuples:
                                  :math:`(sample_id, elec, audio)`

          **sample_id** represents the ID of the sample.
          
          **elec** of shape `(1, wave signal length, [electrodes/hidden])`: tensor.

          **audio** of shape `(1, wave signal length, 257)`: tensor if dataset is training or validation set
          `(wave signal length)`: numpy if dataset is testing set
    '''
    global dataset
    
    pca = None
    auto_encoder = None
    if elec_preprocessors is None:
        elec_preprocessors = [None]
    for elec_preprocessor in elec_preprocessors:
        if isinstance(elec_preprocessor, nn.Module):
            auto_encoder = elec_preprocessor
            elec_channel = (1, 124)
            hidden_size = list(auto_encoder.state_dict().values())[-1].shape[0]

        elif isinstance(elec_preprocessor, int):
            elec_channel = (1, elec_preprocessor)
            hidden_size = elec_channel[1] - elec_channel[0] + 1

        elif isinstance(elec_preprocessor, tuple) and len(elec_preprocessor) == 2:
            elec_channel = elec_preprocessor
            hidden_size = elec_channel[1] - elec_channel[0] + 1

        elif isinstance(elec_preprocessor, PCA):
            pca = elec_preprocessor
            elec_channel = (1, 124)
            hidden_size = elec_preprocessor.n_components

        elif elec_preprocessor is None:
            elec_channel = (1, 124)
            hidden_size = elec_channel[1] - elec_channel[0] + 1

    training_size = int(250 * split_ratio)
    # If the data set is in the format we expect, reuse it
    if not force_update and 'dataset' in globals() and ( \
        (is_training and 'Train' in dataset and \
         len(dataset['Train']) == training_size - 2 and \
         hidden_size == dataset['Train'][0][1].size(2)) or \
        (not is_training and 'Test' in dataset and \
         len(dataset['Test']) == 70 and \
         hidden_size == dataset['Test'][0][1].size(2))):
        return dataset
    
    # ===== Initialize dataset =====
    dataset = { 'Train': [], 'Valid': [], 'Test': [] }

    if is_training:
        data_range = range(71, 321)
    else:
        data_range = range(1, 71)

    # ===== For all samples =====
    for sample_id in data_range:
        try:
            # ===== Load wave data =====
            spec, phase, _, _ = load_wave_data(sample_id, is_training=is_training, dataset_path=dataset_path, norm=False)

            # ===== Load electrical data =====
            elec = load_elec_data(sample_id, spec.shape[1], elec_channel, dataset_path)
            
            if pca is not None:
                elec = pca.transform(elec)

            # tensor shape: (1, spec signal length, electrodes)
            elec = torch.Tensor([elec]).to(device)
            if auto_encoder is not None:
                with torch.no_grad():
                    # tensor shape: (1, spec signal length, hidden)
                    elec = forward_submodule(auto_encoder, elec)

            # Train/Valid
            if is_training:
                set_type = 'Train' if (sample_id - 70 < training_size) else 'Valid'
                # tensor shape: (1, spec signal length, 257)
                audio = torch.Tensor([spec.T]).to(device)

            # Test
            else:
                set_type = 'Test'
                # numpy shape: (spec signal length)
                audio = spec2wave(spec, phase)

            dataset[set_type].append((sample_id, elec, audio))
        
        except Exception as e:
            # E173.csv is broken
            print(e)
            pass
        
    return dataset