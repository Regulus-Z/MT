import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchlibrosa.augmentation import SpecAugmentation
import torchlibrosa as tl
import numpy as np
import matplotlib.pyplot as plt
import config
from conformer.encoder import ConformerEncoder
from conformer.modules import Linear

def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    elif 'bool' in str(x.dtype):
        x = torch.BoolTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg+max', activation='relu'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if activation == 'relu':
            x = F.relu_(self.bn2(self.conv2(x)))
        elif activation == 'sigmoid':
            x = torch.sigmoid(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x
#####
class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
        
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)
        
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        tmp = self.att(x)
        tmp = torch.clamp(tmp, -10, 10)
        att = torch.exp(tmp / self.temperature) + 1e-6
        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
####################################
# The following Transformer modules are modified from Yu-Hsiang Huang's code:
# https://github.com/jadore801120/attention-is-all-you-need-pytorch
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHead(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.w_qs.bias.data.fill_(0)
        self.w_ks.bias.data.fill_(0)
        self.w_vs.bias.data.fill_(0)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.fill_(0)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()   # (batch_size, 80, 512)
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # (batch_size, T, 8, 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk, (batch_size*8, T, 64)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)   # (n_head * batch_size, T, 64), (n_head * batch_size, T, T)
        
        output = output.view(n_head, sz_b, len_q, d_v)  # (n_head, batch_size, T, 64)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv), (batch_size, T, 512)
        output = F.relu_(self.dropout(self.fc(output)))
        return output
        
#####
class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num，audio_num,model_type):
        super(DecisionLevelMaxPooling, self).__init__()
        sample_rate=config.sample_rate
        window_size = config.win_length
        hop_size = config.hop_length
        mel_bins = config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = tl.Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)
#################################fmax(2000)
        self.logmel_extractor = tl.LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=20, fmax=8000, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)
        ###SpecAugument
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #    freq_drop_width=8, freq_stripes_num=2)
        if model_type=='baseline':
            for i in range(audio_num):
                exec(f'self.cnn_encoder{i}=CNN_encoder()')
        elif model_type=='Transformer':
            n_head = 8
            n_hid = 512
            d_k = 64
            d_v = 64
            dropout = 0.2
            for i in range(audio_num):
                exec(f'self.cnn_encoder{i}=CNN_encoder()')
                exec(f'self.multihead{i}= MultiHead(n_head, n_hid, d_k, d_v, dropout)')            
        elif model_type=='Conformer':
            for i in range(audio_num):
                exec(f'self.cnn_encoder{i}=CNN_encoder2()')
                exec(f'self.conformer_encoder{i}1 = ConformerEncoder(input_dim=64,\
encoder_dim=128,num_layers=1,num_attention_heads=8,\
feed_forward_expansion_factor=4,conv_expansion_factor=2,\
input_dropout_p=0.1,feed_forward_dropout_p=0.1,attention_dropout_p=0.1,\
conv_dropout_p=0.1,conv_kernel_size=15,half_step_residual=True)')
                exec(f'self.conformer_encoder{i}2 = ConformerEncoder(input_dim=128,\
encoder_dim=512,num_layers=2,num_attention_heads=8,\
feed_forward_expansion_factor=4,conv_expansion_factor=2,\
input_dropout_p=0.1,feed_forward_dropout_p=0.1,attention_dropout_p=0.1,\
conv_dropout_p=0.1,conv_kernel_size=15,half_step_residual=True)')
        
        Ndim=512
        for i in range(audio_num):
            exec(f'self.fc_{i} = nn.Linear(Ndim, Ndim)')
        self.fc_final = nn.Linear(audio_num*Ndim, classes_num)
        self.init_weights()

    def init_weights(self):
        for i in range(audio_num):
            exec(f'init_layer(self.fc_{i})')
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, date_length,audio_num)
        """
        for i in range(audio_num):
            input1=input[i]
            x1 = self.spectrogram_extractor(input1)   # (batch_size, 1, time_steps, freq_bins)
            x1 = self.logmel_extractor(x1)    # (batch_size, 1, time_steps, mel_bins)
            batch_size, channel_num, _, mel_bins = x1.shape
                
            x1_diff1 = torch.diff(x1, n=1, dim=2, append=x1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
            x1_diff2 = torch.diff(x1_diff1, n=1, dim=2, append=x1_diff1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
            x1 = torch.cat((x1, x1_diff1, x1_diff2), dim=1)
            #ii=i+1
            exec(f'x1 = self.cnn_encoder{i}(x1)')

            if model_type=='baseline':
                exec(f'output{i} = F.max_pool2d(x1, kernel_size=x1.shape[2:])')
                exec(f'output{i} = output{i}.view(output{i}.shape[0:2])')
            elif model_type=='Transformer':
                x1 = torch.mean(x1, dim=3)
                x1 = x1.transpose(1, 2)   # (batch_size, time_steps, channels)
                exec(f'output{i} = self.multihead{i}(x1, x1, x1)')
            elif model_type=='Conformer':
                x1 = torch.mean(x1, dim=3)
                x1 = x1.transpose(1, 2)   # (batch_size, time_steps, channels)
                exec(f'outputs,_=self.conformer_encoder{i}(x1, x1.shape[-2])')
                exec(f'output{i},_=self.conformer_encoder{i}2(outputs, outputs.shape[-2])')
                exec(f'output{i}=torch.mean(output{i},dim=1)')
                exec(f'xx{i}=F.relu(self.fc_{i}(output{i}))')
                if i==0:
                    combined=xx0
                else:
                    exec(f'combined=torch.cat((combined,xx{i}),1)')

        '''
        x1 = torch.mean(x1, dim=3)
        x1 = x1.transpose(1, 2)   # (batch_size, time_steps, channels)
        x1 = self.multihead(x1, x1, x1)
        x2 = torch.mean(x2, dim=3)
        x2 = x2.transpose(1, 2)   # (batch_size, time_steps, channels)
        x2 = self.multihead(x2, x2, x2)
        '''
        '''
        output1 = F.max_pool2d(x1, kernel_size=x1.shape[2:])
        output1 = output1.view(output1.shape[0:2])
        output2 = F.max_pool2d(x2, kernel_size=x2.shape[2:])
        output2 = output2.view(output2.shape[0:2])
        output3 = F.max_pool2d(x3, kernel_size=x3.shape[2:])
        output3 = output2.view(output2.shape[0:2])
        '''
        #(batch_size,channels,
        #x1 = torch.mean(x1,dim=3)
        '''
        x1=F.max_pool2d(x1,kernel_size=(10,10))
        x1=torch.flatten(x1,start_dim=2,end_dim=-1)

        x1 = x1.transpose(1, 2)
        output11, encoder_output_length_1 = self.conformer_encoder1(x1, x1.shape[-2])# (batch_size, time_steps, channels)
        output1,lenth1=self.conformer_encoder11(output11,output11.shape[-2])
        output1 = torch.mean(output1,dim=1)
        #x2 = torch.mean(x2,dim=3)
        x2=F.max_pool2d(x2,kernel_size=(10,10))
        x2=torch.flatten(x2,start_dim=2,end_dim=-1)
        
        x2 = x2.transpose(1, 2)
        output22, encoder_output_length_2 = self.conformer_encoder2(x2, x2.shape[-2])# (batch_size, time_steps, channels)
        output2,lenth2=self.conformer_encoder22(output22,output22.shape[-2])
        output2 = torch.mean(output2,dim=1)
        #x3 = torch.mean(x3,dim=3)
        x3=F.max_pool2d(x3,kernel_size=(10,10))
        x3=torch.flatten(x3,start_dim=2,end_dim=-1)
        #x3=F.max_pool2d(x3,kernel_size=(4,4))
        x3 = x3.transpose(1, 2)
        output33, encoder_output_length_3 = self.conformer_encoder3(x3, x3.shape[-2])# (batch_size, time_steps, channels)
        output3,lenth3=self.conformer_encoder33(output33,output33.shape[-2])
        output3 = torch.mean(output3,dim=1)
        
                       
        xx1=F.relu(self.fc_1(output1))
        xx2=F.relu(self.fc_2(output2))
        xx3=F.relu(self.fc_3(output3))
        '''
        #combined=torch.cat((xx1,xx2,xx3),1)
        output = F.softmax(self.fc_final(combined), dim=-1)

        return output
        
class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        mel_bins=config.mel_bins
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input):
        # (batch_size, 3, time_steps, mel_bins)
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # (samples_num, channel, time_steps, freq_bins)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)

        return x

class CNN_encoder2(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        mel_bins=config.mel_bins
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        #self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        #self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        #self.conv4 = ConvBlock(in_channels=256, out_channels=512)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input):
        # (batch_size, 3, time_steps, mel_bins)
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # (samples_num, channel, time_steps, freq_bins)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        #x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        #x = F.dropout(x, p=0.2, training=self.training)

        return x
