import torch
import torch.nn as nn

import os
import random
import platform
import subprocess
import numpy as np
import librosa
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from tqdm.notebook import tqdm
from pypesq import pesq
from pystoi import stoi
from multiprocessing import Pool
from IPython.display import Audio, display

from const import Const
from preprocess import spec2wave, to_TMHINT_name, load_wave_data, load_elec_data
from model import get_device, Reshape, Unsqueeze, Squeeze


_platform = platform.system()

TRAIN_NOISE_TYPE = [
#     'cafeteria_babble',
#     'crowd-party-adult-med',
    *[f'n{i}' for i in range(1, 101)],
#     'street noise',
#     'street noise_downtown',
]

TEST_NOISE_TYPE = [
    'car_noise_idle_noise_60_mph',
    'engine',
    'pinknoise_16k',
    'street',
    'street noise',
    'taiwan_3talker',
    'white',
]

TEST_SNR_TYPE = [
    'n10dB', 'n7dB', 'n6dB', 'n5dB', 'n3dB', 'n1dB', '0dB',
    '1dB', '3dB', '4dB', '5dB', '6dB', '9dB', '10dB', '15dB',
]


if _platform == 'Windows':
    def pesq_windows(clean, enhanced, test_sample, sr=16000, dataset_path='.', enhanced_path='./Enhanced/'):
        amplitude = np.iinfo(np.int16).max

        clean = (amplitude * clean).astype(np.int16)
        enhanced = (amplitude * enhanced).astype(np.int16)

        try:
            os.makedirs(enhanced_path)
        except:
            pass
        from scipy.io import wavfile

        clean_wav_filename = os.path.join(dataset_path, f'Testing/Clean/a1/{to_TMHINT_name(test_sample)}.wav')
        test_wav_filename = os.path.join(enhanced_path, f'{to_TMHINT_name(test_sample)}.wav')
        
        wavfile.write(test_wav_filename, sr, enhanced)
        
        pesq_results = subprocess.check_output(
            f'PESQ +{sr} {clean_wav_filename} {test_wav_filename}',
            shell=True
        ).decode("utf-8")

        return float(pesq_results.split()[-1])


def train(model, dataset, from_epoch, batch_size, valid_loss_threshold, loss_coef, loss_fn, optimizer, save_filename, dataset_path='.', use_zero_pad=False):
    try:
        os.makedirs(os.path.dirname(save_filename))
    except:
        pass
    
    device = get_device(model)
    
    epoch = from_epoch + 1
    saved_epoch = from_epoch
    early_stop = 0
    min_valid_loss = valid_loss_threshold
    loss_hist = { 'train loss': [], 'valid loss': [] }
    
    with tqdm(total=len(TRAIN_NOISE_TYPE)) as pbar1, \
         tqdm(total=len(TRAIN_NOISE_TYPE)) as pbar2:
        
        while True:
            # ===== Training =====
            loss_hist['train loss'] = []

            pbar1.reset()
            for noise_type in TRAIN_NOISE_TYPE:
                bs = 0
                loss = 0

                pbar1.set_description_str(f'(Epoch {epoch}) noise type: {noise_type}')
                random.shuffle(dataset['Train'])
                for sample_id, elec, clean_spec in dataset['Train']:

                    noisy_spec, _, _, _ = load_wave_data(
                        sample_id=sample_id, noise_type=noise_type,
                        dataset_path=dataset_path, norm=model.use_norm
                    )
                    noisy_spec = torch.Tensor([noisy_spec.T]).to(device)

                    if use_zero_pad:
                        r = random.random() * 3
                        if r <= 1:
                            elec = torch.zeros(elec.shape).to(device)
                        elif r <= 2:
                            noisy_spec = torch.zeros(noisy_spec.shape).to(device)

                    pred = model(noisy_spec, elec)
                    loss += model.get_loss(loss_fn, pred, clean_spec, None, loss_coef)
                    bs += 1
                    if bs >= batch_size:
                        loss /= bs
                        train_loss = loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        loss_hist['train loss'].append(train_loss)
                        loss = 0
                        bs = 0

                    pbar1.refresh()

                if bs != 0:
                    loss /= bs
                    train_loss = loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_hist['train loss'].append(train_loss)
                    loss = 0
                    bs = 0

                pbar1.set_postfix(loss=train_loss)
                pbar1.update()
                
                model.save_model(f'{save_filename} test.pt', 0, valid_loss_threshold)

            # ===== Validation =====
            pbar2.reset()
            valid_loss = 0
            valid_sample = 0
            for noise_type in TRAIN_NOISE_TYPE:
                pbar2.set_description_str(f'(Saved Epoch {saved_epoch}), min valid loss: {min_valid_loss:.3f}, noise type: {noise_type}')
                for sample_id, elec, clean_spec in dataset['Valid']:
                    noisy_spec, _, _, _ = load_wave_data(
                        sample_id=sample_id, noise_type=noise_type,
                        dataset_path=dataset_path, norm=model.use_norm
                    )
                    noisy_spec = torch.Tensor([noisy_spec.T]).to(device)

                    with torch.no_grad():
                        pred = model(noisy_spec, elec)
                        valid_loss += model.get_loss(loss_fn, pred, clean_spec).item()
                    valid_sample += 1

                pbar2.set_postfix(valid_loss=valid_loss/valid_sample)
                pbar1.refresh()
                pbar2.update()
            
            valid_loss /= valid_sample
            loss_hist['valid loss'].append(valid_loss)

            # ===== Save model =====
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                saved_epoch = epoch
                model.save_model(f'{save_filename}.pt', saved_epoch, valid_loss)
                early_stop = 0
            else:
                early_stop += 1

            pbar2.set_description_str(f'(Saved Epoch {saved_epoch}), min valid loss: {min_valid_loss:.3f}')
            epoch += 1

            # ===== Plot loss =====
            plt.plot(loss_hist['train loss'])
            plt.plot(np.linspace(0, len(loss_hist['train loss']), len(loss_hist['valid loss'])), loss_hist['valid loss'])
            plt.legend(['Train', 'Valid'])
            plt.tight_layout(pad=0.2)
            plt.show()
            
            # ===== Early stop =====
            if early_stop >= 5:
                pbar1.close()
                pbar2.close()
                break


def test(model, noise_type, SNR_type, test_sample, pca=None, dataset_path='.', use_S=True, use_E=True, display_audio=False, show_graph=True):
    print(f'{noise_type}, {SNR_type}, {test_sample}')
    
    device = get_device(model)
    if use_S:
        Sx, phasex, meanx, stdx = load_wave_data(
            sample_id=test_sample, noise_type=noise_type, SNR_type=SNR_type,
            is_training=False, dataset_path=dataset_path, norm=model.use_norm
        )
        noisy = torch.Tensor([Sx.T]).to(device)
    else:
        Sx = None
        noisy = None
    
    Sy, phasey, _, _ = load_wave_data(
        sample_id=test_sample,
        is_training=False, dataset_path=dataset_path, norm=False
    )
    
    if use_E and model.is_use_E():
        elec_data = load_elec_data(test_sample, Sy.shape[1], (1, 124), dataset_path)
    else:
        elec_data = np.zeros((Sy.shape[1], 124))
    if pca:
        elec_data = pca.transform(elec_data)
    elec = torch.Tensor([elec_data]).to(device)
    elec_data = elec_data.T
    
    with torch.no_grad():
        for _ in range(2):
            Ss, Se, Sf, Sy_, e_ = model(noisy, elec)
    
    if Ss is not None:
        Ss = Ss[0].cpu().detach().numpy().T
    if Se is not None:
        Se = Se[0].cpu().detach().numpy().T
    if Sf is not None:
        Sf = Sf[0].cpu().detach().numpy().T
    if e_ is not None:
        e_ = e_[0].cpu().detach().numpy().T
    if Sy_ is not None:
        Sy_ = Sy_[0].cpu().detach().numpy().T
    else:
        return
    
    if noisy is not None:
        enhanced = spec2wave(Sy_, phasex)
    else:
        enhanced = librosa.core.griffinlim(10**(Sy_ / 2),
                                           n_iter=5,
                                           hop_length=Const.HOP_LENGTH,
                                           win_length=Const.WIN_LENGTH,
                                           window=Const.WINDOW)
    clean = spec2wave(Sy, phasey)
    
    if use_S:
        noisy = spec2wave(Sx, phasex, meanx, stdx)
    
    sr = 16000
    if _platform == 'Windows':
        print('PESQ: ', pesq_windows(clean, enhanced, test_sample, sr, dataset_path))
#     else:
    print('PESQ: ', pesq(clean, enhanced, sr))
    print('STOI: ', stoi(clean, enhanced, sr, False))
    print('ESTOI:', stoi(clean, enhanced, sr, True))
    
    if display_audio:
#         display(Audio(clean, rate=16000, autoplay=False))
        if use_S:
            display(Audio(noisy, rate=sr, autoplay=False))
        display(Audio(enhanced, rate=sr, autoplay=False))

    if show_graph:
        show_data = [
            (Sx, 'lower', 'jet'),
            (elec_data, 'lower', 'jet'),#None, cm.Blues),
            (Ss, 'lower', 'jet'),
            (Se, 'lower', 'jet'),
            (Sf, 'lower', 'jet'),
            (Sy_, 'lower', 'jet'),
            (Sy, 'lower', 'jet'),
#             (e_, None, cm.Blues),
        ]

        f, axes = plt.subplots(len(show_data), 1, sharex=True, figsize=(18, 12))
        axes[0].set_xlim(0, Sy.shape[1])

        for i, (data, origin, cmap) in enumerate(show_data):
            if data is not None:
                axes[i].imshow(data, origin=origin, aspect='auto', cmap=cmap)

        plt.tight_layout(pad=0.2)
        plt.show()
        
#         plt.figure(figsize=(20, 5))
#         plt.imshow(Sx, origin='lower', aspect='auto', cmap='jet')
#         plt.savefig(os.path.join('./checkpoint/AutoEncoder/', f'noisyS39.png'))
        
#         plt.figure(figsize=(20, 5))
#         plt.imshow(Se, origin='lower', aspect='auto', cmap='binary')
#         plt.savefig(os.path.join('./checkpoint/AutoEncoder/', f'AE39.png'))
        
#         plt.figure(figsize=(20, 5))
#         plt.imshow(Sy_, origin='lower', aspect='auto', cmap='jet')
#         plt.savefig(os.path.join('./checkpoint/AutoEncoder/', f'Enhanced39.png'))
        
#         plt.figure(figsize=(20, 5))
#         plt.imshow(Sy, origin='lower', aspect='auto', cmap='jet')
#         plt.savefig(os.path.join('./checkpoint/AutoEncoder/', f'cleanS39.png'))


def analyze(model, dataset, model_name, processes=None, use_S=True, use_griffin=False, evaluation_path='Evaluation', dataset_path='.'):
    evaluation_dir = os.path.join(evaluation_path, model_name)
    try:
        os.makedirs(evaluation_dir)
    except:
        pass
    
    device = get_device(model)

    sr = 16000
    
    with Pool(processes) as p, \
            tqdm(TEST_NOISE_TYPE) as noise_bar, \
            tqdm(total=len(TEST_SNR_TYPE)) as SNR_bar, \
            tqdm(total=len(dataset['Test'])) as test_bar:
        for noise_type in noise_bar:
            noise_bar.set_description(noise_type)
            result_file = os.path.join(evaluation_dir, f'{noise_type}.txt')
            open(result_file, 'w')

            SNR_bar.reset()
            for SNR_type in TEST_SNR_TYPE:
                SNR_bar.set_description(SNR_type)
                total_sample = 0
                folder_result = []

                test_bar.reset()
                for sample_id, elec, clean in dataset['Test']:
                    test_bar.set_description(f'{to_TMHINT_name(sample_id)}.wav')

                    noisy, phasex, _, _ = load_wave_data(
                        sample_id=sample_id, noise_type=noise_type, SNR_type=SNR_type,
                        is_training=False, dataset_path=dataset_path, norm=model.use_norm
                    )
                    if use_S:
                        noisy = torch.Tensor([noisy.T]).to(device)
                    else:
                        noisy = None

                    with torch.no_grad():
                        _, _, _, pred_y, _ = model(noisy, elec)
                    pred_y = pred_y[0].cpu().detach().numpy().T

                    if not use_griffin:
                        enhanced = spec2wave(pred_y, phasex)
                    else:
                        enhanced = librosa.core.griffinlim(10**(pred_y / 2),
                                                           n_iter=5,
                                                           hop_length=Const.HOP_LENGTH,
                                                           win_length=Const.WIN_LENGTH,
                                                           window=Const.WINDOW)

                    folder_result.append([
                        p.apply_async(pesq, (clean, enhanced, sr)),
                        p.apply_async(stoi, (clean, enhanced, sr, False)),
                        p.apply_async(stoi, (clean, enhanced, sr, True)),
                    ])

                    total_sample += 1
                    test_bar.refresh()

                if total_sample:
                    results = [0] * 3
                    for single_result in folder_result:
                        for i, result in enumerate(single_result):
                            results[i] += result.get()
                        noise_bar.refresh()
                        SNR_bar.refresh()
                        test_bar.update()

                    results = [_ / total_sample for _ in results]

                    with open(result_file, 'a') as writer:
                        writer.write(f'SNR: {SNR_type}\n')
                        writer.write(f'PESQ:  {results[0]}\n')
                        writer.write(f'STOI:  {results[1]}\n')
                        writer.write(f'ESTOI: {results[2]}\n')
                        writer.write('\n')

                SNR_bar.update()
            noise_bar.update()


def autolabel(ax, rects, length):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_y() + rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8*(20-length)*rect.get_width())


def show_analyze(compare_dict,
                 evaluation_path='Evaluation',
                 metrics=['PESQ', 'STOI', 'ESTOI'],
                 bottom={ 'PESQ': 1, 'STOI': 0.4, 'ESTOI': 0.1 },
                 test_noise_type=TEST_NOISE_TYPE,
                 test_SNR_type=TEST_SNR_TYPE,
                 figsize=(20, 6),
                 show='text',
                 use_label=False,
                 save_dir=None):
    
    if save_dir:
        save_dir = os.path.join(evaluation_path, save_dir)
        try:
            os.makedirs(save_dir)
        except:
            pass

    label_name = list(compare_dict.keys())

    compare_size = len(compare_dict)
    total_test_SNR_type = len(test_SNR_type)
    
    for noise_type in test_noise_type:
        compare_models = [
            os.path.join(evaluation_path, model_name, f'{noise_type}.txt')
            for model_name in compare_dict.values()
        ]

        results = {
            metric: np.zeros((compare_size, total_test_SNR_type))
            for metric in metrics
        }

        for i, file in enumerate(compare_models):
            update = False
            try:
                for line in open(file, 'r'):
                    if 'SNR' in line:
                        try:
                            index = test_SNR_type.index(line.split()[-1])
                            update = True
                        except:
                            update = False

                    if update:
                        for metric in results.keys():
                            if line.startswith(metric):
                                results[metric][i][index] = float(line.split(' ')[-1])
                                break
            except:
                for metric in results.keys():
                    results[metric][i] = [0] * total_test_SNR_type
        
        if 'text' in show:
            print(test_SNR_type)
            for metric, values in results.items():
                print(f'====\n{noise_type.title()} {metric}:')
                for model_name, result in zip(compare_dict.keys(), values):
                    print(f'\t{model_name} :\t{result}')
        
        if 'graph' in show:
            x = np.arange(total_test_SNR_type)  # the label locations
            width = 1.0 / (compare_size + 1)  # the width of the bars

            for metric, result in results.items():
                fig, ax = plt.subplots(figsize=figsize)

                for i, compared in enumerate(result):
                    offset = (1 - compare_size) / 2 + i
                    offset *= width
                    rects = ax.bar(
                        x + offset, compared - bottom[metric], width,
                        label=label_name[i], bottom=bottom[metric]
                    )
                    if use_label:
                        autolabel(ax, rects, total_test_SNR_type)

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel(f'{metric} Scores')
                ax.set_title(f'{noise_type.title()} {metric}')
                ax.set_xticks(x)
                ax.set_xticklabels(test_SNR_type)
                ax.legend()
                plt.tight_layout(pad=0.5)

                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{ax.get_title()}.png'))

                plt.show()


def avg_analyze(compare_dict,
                evaluation_path='Evaluation',
                metrics=['PESQ', 'STOI', 'ESTOI'],
                bottom={ 'PESQ': 1, 'STOI': 0.4, 'ESTOI': 0.1 },
                patterns='x.-*/\o|+O',
                test_noise_type=TEST_NOISE_TYPE,
                test_SNR_type=TEST_SNR_TYPE,
                figsize=(20, 6),
                show='text',
                use_label=False,
                save_dir=None):
    
    if save_dir:
        save_dir = os.path.join(evaluation_path, save_dir)
        try:
            os.makedirs(save_dir)
        except:
            pass

    label_name = list(compare_dict.keys())
    
    compare_size = len(compare_dict)
    total_test_SNR_type = len(test_SNR_type)
    avg_results = {
        metric: np.zeros((compare_size, total_test_SNR_type))
        for metric in metrics
    }
    
    for noise_type in test_noise_type:
        compare_models = [
            os.path.join(evaluation_path, model_name, f'{noise_type}.txt')
            for model_name in compare_dict.values()
        ]

        results = {
            metric: np.zeros((compare_size, total_test_SNR_type))
            for metric in metrics
        }

        for i, file in enumerate(compare_models):
            update = False
            try:
                for line in open(file, 'r'):
                    if 'SNR' in line:
                        try:
                            index = test_SNR_type.index(line.split()[-1])
                            update = True
                        except:
                            update = False

                    if update:
                        for metric in results.keys():
                            if line.startswith(metric):
                                results[metric][i][index] = float(line.split(' ')[-1])
                                break
            except:
                for metric in results.keys():
                    results[metric][i] = [0] * total_test_SNR_type
        
        for k, v in results.items():
            avg_results[k] += v
    
    for metric in results.keys():
        avg_results[metric] /= len(test_noise_type)
    
    if 'text' in show:
        print(test_SNR_type)
        for metric in results.keys():
            print(f'====\n{metric}:')
            for model_name, result in zip(compare_dict.keys(), avg_results[metric]):
                print(f'\t{model_name} :\t{result}\t{result.mean()}')
    
    if 'graph' in show:
        x = np.arange(compare_size)  # the label locations
        width = 1 # the width of the bars
#         patterns = 'x.-*/\o|+O'
        pic_index = 'abc'
        
        fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize, gridspec_kw=dict(hspace=0.5, wspace=0.5))
        fig.set_facecolor('white')
        for i, (metric, results) in enumerate(avg_results.items()):
            for j, compared in enumerate(results):
                rects = ax[i].bar(
                    j, compared.mean() - bottom[metric],
                    label=label_name[j], bottom=bottom[metric],
                    color='white', edgecolor='black', hatch=patterns[j]
                )
                if use_label:
                    autolabel(ax[i], rects, total_test_SNR_type)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            # removing the default axis on all sides:
            for side in ['right', 'top']:
                ax[i].spines[side].set_visible(False)
            ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#             arrowed_spines(fig, ax[i])
            ax[i].xaxis.set_ticklabels([])
#             ax[i].set_xticks(x)
#             ax[i].set_xticklabels([f'({i})' for i in range(1, compare_size + 1)])#compare_dict.keys())
            ax[i].set(xlabel=f'({pic_index[i]}) {metric}')

#         plt.subplots_adjust(wspace=0.5)
#         plt.tight_layout(pad=0, w_pad=0.5, h_pad=0.5)
        fig.legend(loc='upper center', labels=compare_dict.keys(), bbox_to_anchor=(0.5, 0.025), ncol=compare_size)
#         fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f'Avg Performance.eps'), bbox_inches='tight')
        plt.show()


def arrowed_spines(fig, ax):
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()
#     print(xmin, xmax)
#     print(ymin, ymax)

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels 
#     plt.yticks([])
#     ax.xaxis.set_ticks_position('none') # tick markers
#     ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
#     print(width, height)

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)