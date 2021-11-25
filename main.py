import torch
import torch.nn as nn

import os
import argparse
import pickle

from collections import OrderedDict

from const import Const
from preprocess import cache_clean_data
from model import AutoEncoder, MultiModal_SE, Reshape, Unsqueeze, Squeeze, SeqUnfold, SeqUnfold_Reshape, SeqFlatten, ResCNN, TransformerModel
from utils import TRAIN_NOISE_TYPE, TEST_NOISE_TYPE, TEST_SNR_TYPE, train, test, analyze, show_analyze, avg_analyze


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=__doc__
)

# ===== training =====
parser.add_argument('--dataset_path', type=str, default='../Dataset/', help='the dataset folder.')
parser.add_argument('--AE_checkpoint_path', type=str, default='./checkpoint/AutoEncoder/', help='the checkpoint folder for the autoencoder.')
parser.add_argument('--MM_checkpoint_path', type=str, default='./checkpoint/MultiModal/', help='the checkpoint folder for the MultiModal.')
parser.add_argument('--AE_name', type=str, default='AE124_BiLSTM12NR S', help='the name of the autoencoder model.')
parser.add_argument('--model_name', type=str, help='the name of the model.')
parser.add_argument('--train', action='store_true', help='to train the model.')
parser.add_argument('--split_ratio', type=float, default=0.896, help='the ratio of splitting the training set into training/validation sets.')
parser.add_argument('--frame_seq', type=int, default=5, help='the frames amount of model input.')
parser.add_argument('--batch_size', type=int, default=1, help='the batch size wanted to be trained.')
parser.add_argument('--loss_coef', type=float, default=0.1, help='loss = noisy_loss + loss_coefficient * elec_loss')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of the optimizer.')
parser.add_argument('--loss', type=str, default='MSE', help='option: MSE')
parser.add_argument('--opt', type=str, default='Adam', help='option: Adam')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='data on which device.')

# ===== testing =====
parser.add_argument('--evaluation_path', type=str, default='./Evaluation/', help='the folder to save evaluation results.')
parser.add_argument('--test', action='store_true', default=True, help='to test an existing model.')
parser.add_argument('--test_noise_type', type=int, choices=range(len(TEST_NOISE_TYPE)), default=4, help='the index of the test noise type.')
parser.add_argument('--test_SNR_type', type=int, choices=range(len(TEST_SNR_TYPE)), default=1, help='the index of the test SNR.')
parser.add_argument('--test_sample', type=int, choices=range(70), default=1, help='the index of the test sample.')
parser.add_argument('--analyze', action='store_true', default=True, help='test and generate reports in evaluation_path.')
parser.add_argument('--processes', type=int, default=1, help='number of threads used for analysis')
parser.add_argument('--griffin', type=bool, help='use griffin algorithm for analyze.')

args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    # ===== Load AutoEncoder =====
    AE, _, _ = AutoEncoder().load_model(
        os.path.join(args.AE_checkpoint_path, f'{args.AE_name}.pt'),
        device=args.device
    )
    
    model, _, _ = MultiModal_SE().load_model(
        os.path.join(args.MM_checkpoint_path, f'{args.model_name}.pt'),
        args.device
    )

    if args.test:
        if isinstance(model.E_Encoder, nn.Identity) and not isinstance(model.S_Encoder, nn.Identity):
            model.E_Encoder = AE.Encoder
        
        test(
            model,
            TEST_NOISE_TYPE[args.test_noise_type],
            TEST_SNR_TYPE[args.test_SNR_type],
            args.test_sample,
            dataset_path=args.dataset_path
        )