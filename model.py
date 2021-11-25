import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import itertools

from collections import OrderedDict
from torch.nn.utils import weight_norm


def get_device(model):
    try:
        return next(model.parameters()).device
    except:
        return 'cpu'

    
def forward_submodule(module, data):
    '''
    Assist RNN-based modules to pass forward in nn.Sequential recursively.
    '''
    if module is None or data is None:
        return None

    if isinstance(module, nn.Sequential):
        for submodule in module:
            data = forward_submodule(submodule, data)
        return data

    elif isinstance(module, nn.RNNBase):
        module.flatten_parameters()
        return module(data)[0]

    return module(data)


class MultiModal_SE(nn.Module):
    '''
    Args:
        S_Encoder (nn.Sequential): a model encodes noisy spectrum into latent state.
        E_Encoder (nn.Sequential): a model encodes the EPG signal into latent state.
        S_Decoder (nn.Sequential): a model decodes fusion state into clean spectrum.
        E_Decoder (nn.Sequential): a model decodes fusion state into the EPG signal.
        Fusion_layer (nn.Sequential): a model fuses the information from two encoders into a fusion state.
        
        is_late_fusion (bool, optional): Choose whether the model is early fusion or late fusion. Default: ``False``
        fusion_type (str, optional): Choose the fusion type. It can be 'concatenate', 'mean' or 'mask'. Default: 'concat'
        fusion_channel (int, optional): Choose fusion channel. :math:`(N[, C], Seq, hidden_size)`. Default: ``-1``
        use_norm (bool, optional): Whether the input spectrum is normalized. Default: ``False``
    
    Outputs: :math:`h_s, h_e, h_f, y_, e_`
        - **h_s** is the hidden state of spectrum
        - **h_e** is the hidden state of electric signal
        - **h_f** is the hidden state of fusion
        - **y_** is the prediction of spectrum
        - **e_** is the prediction of electric

    Shape:
        - Input:
            s  :math:`(N, Seq, 257)`
            e  :math:`(N, Seq, 124)`

        - Output: :math:`(N, Seq, hidden_size)` for each element.
    '''
    def __init__(self,
                 S_Encoder: nn.Sequential = None,
                 E_Encoder: nn.Sequential = None,
                 S_Decoder: nn.Sequential = None,
                 E_Decoder: nn.Sequential = None,
                 Fusion_layer: nn.Sequential = None,
                 is_late_fusion: bool = False,
                 fusion_type: str = 'concat',
                 fusion_channel: int = -1,
                 use_norm: bool = False):
        
        super(MultiModal_SE, self).__init__()
        
        self.S_Encoder = S_Encoder
        self.E_Encoder = E_Encoder
        self.S_Decoder = S_Decoder
        self.E_Decoder = E_Decoder
        self.Fusion_layer = Fusion_layer
        
        self.is_late_fusion = is_late_fusion
        self.fusion_type = fusion_type
        self.fusion_channel = fusion_channel
        
        self.use_norm = use_norm
    
    def forward(self, s, e, elec_only=False):
        # Make sure all data exists.
        if s is None and e is not None and self.S_Encoder is not None:
            device = get_device(self)
            s = torch.zeros((1, e.shape[1], 257)).to(device)
        
        if e is None and s is not None and self.E_Encoder is not None:
            device = get_device(self)
            e = torch.zeros((1, s.shape[1], 124)).to(device)

        h_e = forward_submodule(self.E_Encoder, e)

        if self.is_late_fusion:
            # In late fusion, we mix the information of h_s and h_e
            h_s = forward_submodule(self.S_Encoder, s)
        else:
            # In early fusion, we mix the information of s and h_e
            h_s = s

        
        if h_s is None or elec_only:
            # EPG to Speech (EPG2S)
            h = h_e
        
        elif h_e is None:
            # Speech Enhancement (baseline)
            h = h_s

        elif self.fusion_type in 'concatenate':
            # fution type (Default: 'concat')
            # concat position: 0 for channel, -1 for frequency domain (Default: -1)
            h = torch.cat((h_s, h_e), dim=self.fusion_channel)

        elif self.fusion_type in 'mean':
            h = (h_s + h_e) / 2

        elif self.fusion_type in 'mask':
            h = torch.mul(h_s, h_e)
            h_e = h

        h_f = forward_submodule(self.Fusion_layer, h)
        
        if h_f is None:
            h_f = forward_submodule(self.S_Encoder, h_s)
            y_ = forward_submodule(self.S_Decoder, h_f)
        else:
            y_ = forward_submodule(self.S_Decoder, h_f)
            
        e_ = forward_submodule(self.E_Decoder, h_f)

        if self.is_late_fusion:
            return h_s, h_e, h_f, y_, e_
        return s, h_e, h_f, y_, e_

#         else:
#             h_s = forward_submodule(self.S_Encoder, s)
#             y_ = forward_submodule(self.S_Decoder, h_s)
#             e_ = forward_submodule(self.E_Decoder, h_s)
#             return h_s, None, None, y_, e_
    
    def get_loss(self, loss_fn, feat_loss_fn, pred_y, true_y, true_e=None, lamb=0.1):
        loss = loss_fn(pred_y[-2], true_y)
        
        if pred_y[-1] is not None and true_e is not None:
            loss += lamb * loss_fn(pred_y[-1], true_e)
            
        if feat_loss_fn is not None and pred_y[1] is not None and pred_y[0] is not None:
            loss += lamb * feat_loss_fn(pred_y[1], pred_y[0])
        
        return loss
    
    def is_use_E(self):
        return self.E_Encoder is not None

    def load_model(self, filename, device=None):
        epoch, valid_loss = 0, 1e9
        try:
            state_dict = torch.load(filename, map_location=device)
            
            self.S_Encoder = state_dict.get('S_Encoder', self.S_Encoder)
            self.E_Encoder = state_dict.get('E_Encoder', self.E_Encoder)
            self.S_Decoder = state_dict.get('S_Decoder', self.S_Decoder)
            self.E_Decoder = state_dict.get('E_Decoder', self.E_Decoder)
            self.Fusion_layer = state_dict.get('Fusion_layer', self.Fusion_layer)
            
            self.is_late_fusion = state_dict.get('is_late_fusion', self.is_late_fusion)
            self.fusion_type = state_dict.get('fusion_type', self.fusion_type)
            self.fusion_channel = state_dict.get('fusion_channel', self.fusion_channel)
            
            self.use_norm = state_dict.get('use_norm', self.use_norm)
            
            epoch = state_dict.get('epoch', epoch)
            valid_loss = state_dict.get('valid_loss', valid_loss)
            
            print(f"Model '{filename}' loaded.")
        except Exception as e:
            print(e)
        
        self.to(device)
        return self, epoch, valid_loss
    
    def save_model(self, filename, epoch=0, valid_loss=1e9):
        state_dict = {
            'S_Encoder': self.S_Encoder,
            'E_Encoder': self.E_Encoder,
            'S_Decoder': self.S_Decoder,
            'E_Decoder': self.E_Decoder,
            'Fusion_layer': self.Fusion_layer,
            'is_late_fusion': self.is_late_fusion,
            'fusion_type': self.fusion_type,
            'fusion_channel': self.fusion_channel,
            'use_norm': self.use_norm,
            
            'epoch': epoch,
            'valid_loss': valid_loss,
        }
    
        torch.save(state_dict, filename)


class AutoEncoder(nn.Module):
    def __init__(self,
                 Encoder: nn.Sequential = None,
                 Decoder: nn.Sequential = None):
        super(AutoEncoder, self).__init__()
        
        self.Encoder = Encoder
        self.Decoder = Decoder
    
    def forward(self, e):
        hidden = forward_submodule(self.Encoder, e)
        return forward_submodule(self.Decoder, hidden)

    def load_model(self, filename, device=None):
        epoch, valid_loss = 0, 1e9
        try:
            state_dict = torch.load(filename, map_location=device)
            
            self.Encoder = state_dict.get('Encoder', self.Encoder)
            self.Decoder = state_dict.get('Decoder', self.Decoder)
            
            epoch = state_dict.get('epoch', epoch)
            valid_loss = state_dict.get('valid_loss', valid_loss)
            
            print(f"Model '{filename}' loaded.")
        except Exception as e:
            print(e)
        
        self.to(device)
        return self, epoch, valid_loss
    
    def save_model(self, filename, epoch=0, valid_loss=1e9):
        state_dict = {
            'Encoder': self.Encoder,
            'Decoder': self.Decoder,
            
            'epoch': epoch,
            'valid_loss': valid_loss,
        }
    
        torch.save(state_dict, filename)


class Reshape(nn.Module):
    def __init__(self, channel, height):
        super(Reshape, self).__init__()
        self.channel = channel
        self.height = height
        
    def forward(self, x):
        return x.permute(0, 2, 3, 1).reshape(1, -1, self.channel * self.height)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.squeeze(self.dim)


class SeqUnfold(nn.Module):
    def __init__(self, padding=(2, 0)):
        super(SeqUnfold, self).__init__()
        self.padding = padding
        
    def forward(self, x):
        '''
        Shape:
        - Input:
            x  :math:`(N, 1, Seq, hidden_size)`

        - Output: :math:`(N, Seq, frame_seq * hidden_size)`
        '''
    
        seq_len = x.shape[-2]
        
        return F.unfold(
            x,
            kernel_size=(seq_len, 1),
            padding=self.padding
        )


class SeqUnfold_Reshape(nn.Module):
    def __init__(self, frame_seq):
        super(SeqUnfold_Reshape, self).__init__()
        self.frame_seq = frame_seq
        self.padding = (frame_seq // 2, 0)
        
    def forward(self, x):
        '''
        Shape:
        - Input:
            x  :math:`(1, Seq, hidden_size)`

        - Output: :math:`(Seq, frame_seq, hidden_size)`
        '''
    
        seq_len = x.shape[-2]
        
        return F.unfold(
            x.unsqueeze_(0),
            kernel_size=(seq_len, 1),
            padding=self.padding
        ).reshape((seq_len, self.frame_seq, -1))


class SeqFlatten(nn.Module):
    def __init__(self):
        super(SeqFlatten, self).__init__()
        
    def forward(self, x):
        '''
        Shape:
        - Input:
            x  :math:`(Seq, frame_seq, hidden_size)`

        - Output: :math:`(1, Seq, frame_seq * hidden_size)`
        '''
        
        return x.reshape((1, x.shape[0], -1))


class ResCNN(nn.Module):
    def __init__(self, CNN_param: list = None, input_channel: int = 1, input_size: int = 257, use_residual: bool = True):
        super(ResCNN, self).__init__()
        
        if CNN_param is None:
            self.CNN_param = [[[2**(3+i), 2**(4+i), 1],
                               [2**(4+i), 2**(4+i), 1],
                               [2**(4+i), 2**(4+i), 3]]
                              for i in range(4)]
        else:
            self.CNN_param = CNN_param

        self.CNN_param[0][0][0] = input_channel
        
        self.input_size = input_size
        self.output_size = input_size
        for submodule in self.CNN_param:
            for _, _, y_stride in submodule:
                self.output_size = math.ceil(self.output_size / y_stride)

        self.CNN_model = nn.Sequential(OrderedDict([
            ('unsqueeze', Unsqueeze(0)),
            *list(itertools.chain(*[
                [
                    (f'conv_{i}_{j}', nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, y_stride), padding=(1, 1))),
                    (f'norm_{i}_{j}', nn.InstanceNorm2d(out_channel, affine=True)),
                    (f'relu_{i}_{j}', nn.LeakyReLU(negative_slope=0.3, inplace=True))
                ]
                for i, submodule in enumerate(self.CNN_param)
                for j, (in_channel, out_channel, y_stride) in enumerate(submodule)
            ])),
            ('reshape', Reshape(self.output_size, self.CNN_param[-1][-1][1]))
        ]))
        
        self.use_residual = use_residual

    def forward(self, s):
        if self.use_residual:
            h = s
            for name, module in self.CNN_model.named_children():
                if 'conv' in name and module.in_channels == module.out_channels and module.stride[1] == 1:
                    x = h
                    h = module(h)
                elif 'relu' in name and name[-1] == '1':
                    h = module(h) + x
                else:
                    h = module(h)

            return h
        
        return self.CNN_model(s)

    
class WienerCNN(nn.Module):
    def __init__(self, input_channel: int = 1, hidden_channels: int = 32, layers: int = 6):
        super(WienerCNN, self).__init__()
        
        self.layers = layers
        self.hidden_channels = hidden_channels
        self.CNN_param = [[self.hidden_channels, self.hidden_channels]
                          for i in range(self.layers)]
        
        self.CNN_param[0][0] = input_channel
        
        self.CNN_model = nn.Sequential(OrderedDict([
            ('unsqueeze', Unsqueeze(0)),
            *list(itertools.chain(*[
                [
                    (f'conv_{i}', nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                    (f'norm_{i}', nn.InstanceNorm2d(out_channel, affine=True)),
                    (f'relu_{i}', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ]
                for i, (in_channel, out_channel) in enumerate(self.CNN_param)
            ])),
        ]))
       
    def forward(self, s):
        h = s
        x = None
        for name, module in self.CNN_model.named_children():
            if 'conv' in name and module.in_channels == module.out_channels:
                x = h
                h = module(h)
            elif 'relu' in name and x is not None:
                h = torch.mul(module(h), x)
            else:
                h = module(h)
        
        return h


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1, bidirectional=False):
        super(TransformerModel, self).__init__()
        
        # Models
        self.encoder_embedding = nn.Linear(input_size, hidden_size)
        self.decoder_embedding = nn.Linear(output_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead, dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead, dropout=dropout)
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.L1 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        
        # init model weights
        self.init_weights()
    
        # parameter
        self.hidden_size = hidden_size
        self.scalar = math.sqrt(hidden_size)
        
        self.bidirectional = bidirectional
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
#         initrange = 0.1
#         self.L1.weight.data.uniform_(-initrange, initrange)
#         self.L2.bias.data.zero_()
#         self.L2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt): #, src_mask):
        '''
        Shape:
        - Input:
            src  :math:`(Batch, Seq, hidden_size)`

        - Output: :math:`(Batch, Seq, hidden_size)`
        '''
        
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        src = self.encoder_embedding(src) * self.scalar
        src = self.pos_encoder(src)
        
        tgt = torch.roll(tgt, 1, 0)
        tgt[0,:] = 0
        tgt = self.decoder_embedding(tgt) * self.scalar
        tgt = self.pos_encoder(tgt)
        
        device = get_device(self)
        mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
        
        if self.bidirectional:
            memory = self.encoder(src, mask=None)
        else:
            memory = self.encoder(src, mask=mask)
        
        output = self.decoder(tgt, memory, tgt_mask=mask, memory_mask=mask)
        
        output = self.relu(self.L1(output))
        output = output.transpose(0, 1)
        
        return output
    
    def evaluate(self, src):
        src = src.transpose(0, 1)

        src = self.encoder_embedding(src) * self.scalar
        src = self.pos_encoder(src)

        device = get_device(self)
        mask = self.generate_square_subsequent_mask(src.size(0)).to(device)

        if self.bidirectional:
            memory = self.encoder(src, mask=None)
        else:
            memory = self.encoder(src, mask=mask)

        output = torch.zeros((1, 1, 512)).to(device)

        for _ in range(src.size(0)):
            output[_,:] = output[_,:] + self.pos_encoder.pe[_, :]
            mask = (torch.triu(torch.ones(src.size(0), output.size(0))) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.to(device)
            pred = self.decoder(output, memory, memory_mask=mask)
            output = torch.cat((output, pred[-1].unsqueeze(0)), 0)
        
#         output = torch.zeros(src.shape).to(device)

#         for _ in range(src.size(0)):
#             output[_,:] = output[_,:] + self.pos_encoder.pe[_, :]
#             pred = self.decoder(output, memory, tgt_mask=mask, memory_mask=mask)
#             output[_,:] = pred[_,:]

        output = self.relu(self.L1(output[1:]))
        output = output.transpose(0, 1)
        
        return output
        
    
    def get_loss(self, loss_fn, feat_loss_fn, pred_y, true_y, true_e=None, lamb=0.1):
        loss = loss_fn(pred_y[-2], true_y)
        
        if pred_y[-1] is not None and true_e is not None:
            loss += lamb * loss_fn(pred_y[-1], true_e)
            
        if feat_loss_fn is not None and pred_y[1] is not None and pred_y[0] is not None:
            loss += lamb * feat_loss_fn(pred_y[1], pred_y[0])
        
        return loss
    
    def load_model(self, filename, device=None):
        epoch, valid_loss = 0, 1e9
        try:
            state_dict = torch.load(filename, map_location=device)
            
            self.encoder_embedding = state_dict.get('encoder_embedding', self.encoder_embedding)
            self.decoder_embedding = state_dict.get('decoder_embedding', self.decoder_embedding)
            self.encoder = state_dict.get('encoder', self.encoder)
            self.decoder = state_dict.get('decoder', self.decoder)
            self.L1 = state_dict.get('L1', self.L1)
            
            self.hidden_size = state_dict.get('hidden_size', self.hidden_size)
            self.scalar = math.sqrt(self.hidden_size)
            self.bidirectional = state_dict.get('bidirectional', self.bidirectional)
            
            epoch = state_dict.get('epoch', epoch)
            valid_loss = state_dict.get('valid_loss', valid_loss)
            print(f"Model '{filename}' loaded.")
            
        except Exception as e:
            print(e)
        
        self.to(device)
        return self, epoch, valid_loss
    
    def save_model(self, filename, epoch=0, valid_loss=1e9):
        state_dict = {
            'encoder_embedding': self.encoder_embedding,
            'decoder_embedding': self.decoder_embedding,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'L1': self.L1,
            
            'hidden_size': self.hidden_size,
            'bidirectional': self.bidirectional,
            
            'epoch': epoch,
            'valid_loss': valid_loss,
        }
    
        torch.save(state_dict, filename)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
#         pe = torch.randn((max_len, 1, d_model), requires_grad=True)
#         pe.data.uniform_(-1, 1)
        self.register_buffer('pe', pe)
#         nn.init.xavier_uniform_(pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)