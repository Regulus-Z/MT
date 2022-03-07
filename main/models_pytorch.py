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

    def forward(self, input, pool_size=(2, 2), pool_type='max', activation='relu'):
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

class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num):
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
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
        self.cnn_encoder = CNN_encoder()
        ###Encoder for cough-heavy
        self.cnn_encoder_ch = CNN_encoder()
        
        self.fc_final = nn.Linear(1024, classes_num)#original:512 double?

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input1,input2,input3):
        """input1,2: (samples_num, date_length) input3:(samples_num,class)
        """
        x1 = self.spectrogram_extractor(input1)   # (batch_size, 1, time_steps, freq_bins)
        x1 = self.logmel_extractor(x1)    # (batch_size, 1, time_steps, mel_bins)
        batch_size, channel_num, _, mel_bins = x1.shape
        #find pos samples for SpecAugument
        if self.training:
            x_neg_ind=torch.nonzero(input3)
            x_pos_ind=torch.nonzero(input3==0)
            if len(x_neg_ind) is 0:
                x1=self.spec_augmenter(x1)
            elif len(x_pos_ind) is 0:
                pass
            else:
                """
                neg=torch.eq(input3,1)
                neg=torch.reshape(-1,1)
                pos=torch.eq(input3,0)
                pos=torch.reshape(-1,1)
                x_neg=x1*neg
                x_pos=x1*pos
                x_pos=self.spec_augmenter(x_pos)
                x1=x_neg+x_pos
                """
                x_neg=torch.index_select(x1, 0, x_neg_ind)
                x_pos=torch.index_select(x1, 0, x_pos_ind)
                x_pos=self.spec_augmenter(x_pos)
                j=0
                k=0
                for i in range(0,batch_size):
                    if x_pos_ind[i]==0:
                        x1[i,:,:,:]=x_pos[j,:,:,:]
                        j=j+1
                    else:
                        x1[i,:,:,:]=x_neg[k,:,:,:]
                        k=k+1
            """
            x_neg_ind=torch.reshape(x_neg_ind, (-1,))
            x_pos_ind=torch.reshape(x_pos_ind, (-1,))
            x_neg=torch.index_select(x1, 0, x_neg_ind)
            x_pos=torch.index_select(x1, 0, x_pos_ind)
            """
            
        x1_diff1 = torch.diff(x1, n=1, dim=2, append=x1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x1_diff2 = torch.diff(x1_diff1, n=1, dim=2, append=x1_diff1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x1 = torch.cat((x1, x1_diff1, x1_diff2), dim=1)

        if False:
            x1_array = x1.data.cpu().numpy()[0]
            x1_array = np.squeeze(x1_array)
            plt.matshow(x1_array.T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig('test.png')

        x1 = self.cnn_encoder(x1)
        ######add x2
        x2 = self.spectrogram_extractor(input2)   # (batch_size, 1, time_steps, freq_bins)
        x2 = self.logmel_extractor(x2)    # (batch_size, 1, time_steps, mel_bins)
        batch_size, channel_num, _, mel_bins = x2.shape
        
        if self.training:
            if len(x_neg_ind) is 0:
                x2=self.spec_augmenter(x2)
            elif len(x_pos_ind) is 0:
                pass
            else:
                """
                neg=torch.eq(input3,1)
                neg=torch.reshape(-1,1)
                pos=torch.eq(input3,0)
                pos=torch.reshape(-1,1)
                x_neg=x1*neg
                x_pos=x1*pos
                x_pos=self.spec_augmenter(x_pos)
                x1=x_neg+x_pos
                """
                x_neg=torch.index_select(x2, 0, x_neg_ind)
                x_pos=torch.index_select(x2, 0, x_pos_ind)
                x_pos=self.spec_augmenter(x_pos)
                j=0
                k=0
                for i in range(0,batch_size):
                    if x_pos_ind[i]==0:
                        x2[i,:,:,:]=x_pos[j,:,:,:]
                        j=j+1
                    else:
                        x2[i,:,:,:]=x_neg[k,:,:,:]
                        k=k+1
        x2_diff1 = torch.diff(x2, n=1, dim=2, append=x2[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x2_diff2 = torch.diff(x2_diff1, n=1, dim=2, append=x2_diff1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x2 = torch.cat((x2, x2_diff1, x2_diff2), dim=1)

        if False:
            x2_array = x2.data.cpu().numpy()[0]
            x2_array = np.squeeze(x2_array)
            plt.matshow(x2_array.T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig('test.png')

        x2 = self.cnn_encoder(x2)
        # (samples_num, 512, hidden_units)
        ##################### where concatenate?
        output1 = F.max_pool2d(x1, kernel_size=x1.shape[2:])
        output1 = output1.view(output1.shape[0:2])
        output2 = F.max_pool2d(x2, kernel_size=x2.shape[2:])
        output2 = output2.view(output2.shape[0:2])
        combined=torch.cat((output1,output2),1)
        output = F.log_softmax(self.fc_final(combined), dim=-1)

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
        x = self.conv1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)

        return x

class PrototypeLayer(nn.Module):
    def __init__(self, n_proto, distance='euclidean', proto_form=None):
        super(PrototypeLayer, self).__init__()
        self.n_proto = n_proto
        self.distance = distance
        self.proto_form = proto_form

        # (n_proto, channel, time_steps, freq_bins)
        if self.proto_form == 'vector1d':
            self.prototype = nn.Parameter(torch.empty((self.n_proto, 512), dtype=torch.float), requires_grad=True)
        elif 'att' in self.proto_form:
            self.prototype = nn.Parameter(torch.empty((self.n_proto, 512, 56), dtype=torch.float), requires_grad=True)
        else:
            self.prototype = nn.Parameter(torch.empty((self.n_proto, 512, 7, 8), dtype=torch.float), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype.shape), requires_grad=False)

        if 'att' in self.proto_form:
            self.fc = nn.Linear(512, 1)

        self.bn = nn.BatchNorm2d(self.n_proto)
        self.bn1d = nn.BatchNorm1d(self.n_proto)
        self.ln = nn.LayerNorm(self.n_proto)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.prototype)
        init_bn(self.bn)
        init_bn(self.bn1d)
        if 'att' in self.proto_form:
            init_layer(self.fc)

    def forward(self, input):
        # (samples_num, channel, time_steps, freq_bins)
        if self.distance == 'cosine':
            if self.proto_form == 'vector1d':
                x = F.max_pool2d(input, kernel_size=input.shape[2:])
                similarity = np.zeros((x.shape[0], self.n_proto), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones(1, dtype=torch.float), True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        a = torch.flatten(x[i], start_dim=0, end_dim=-1)
                        b = torch.flatten(self.prototype[j], start_dim=0, end_dim=-1)
                        #similarity[i][j] = F.cosine_similarity(a, b, dim=0)
                        similarity[i][j] = torch.sum(a * b, dim=0) / torch.maximum(torch.norm(a, p=2, dim=0)*torch.norm(b, p=2, dim=0), sim_lowbound)
                similarity = self.ln(similarity)

            elif self.proto_form == 'vector2d':
                x = input
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[3]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[3]), dtype=torch.float), True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        #similarity[i][j] = F.cosine_similarity(x[i], self.prototype[j], dim=0)
                        similarity[i][j] = torch.sum(x[i] * self.prototype[j], dim=0) / torch.maximum(torch.norm(x[i], p=2, dim=0) * torch.norm(self.prototype[j], p=2, dim=0), sim_lowbound)
                similarity = self.bn(similarity)
                similarity = F.avg_pool2d(similarity, kernel_size=similarity.shape[2:])
                similarity = similarity.view(similarity.shape[0:2])
                #similarity = self.ln(similarity)

            elif self.proto_form == 'vector2d_att':
                x = torch.flatten(input, start_dim=2, end_dim=-1) # 16, 512, 56
                similarity_att = np.zeros((x.shape[0], self.n_proto, x.shape[2]), dtype=np.float)
                similarity_att = move_data_to_gpu(similarity_att, True)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2]), dtype=torch.float), True)
                for i in range(0, similarity_att.shape[0]):
                    for j in range(0, similarity_att.shape[1]):
                        #similarity_att[i][j] = F.cosine_similarity(x[i], self.prototype[j], dim=0)
                        similarity_att[i][j] = torch.sum(x[i] * self.prototype[j], dim=0) / torch.maximum(torch.norm(x[i], p=2, dim=0) * torch.norm(self.prototype[j], p=2, dim=0), sim_lowbound)

                similarity_att = F.softmax(similarity_att, dim=-1)  # (16, 4, 56)
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[1]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        similarity[i][j] = torch.sum(torch.unsqueeze(similarity_att[i][j], dim=0) * x[i] * self.prototype[j], dim=1)
                #similarity_att = self.bn1d(similarity_att)
                #similarity = F.avg_pool1d(similarity, kernel_size=similarity.shape[2])
                similarity = F.relu_(self.fc(similarity))
                similarity = similarity.view(similarity.shape[0:2])

            elif self.proto_form == 'vector2d_avgp':
                x = input
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=np.float)
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(torch.unsqueeze(x, dim=4), dim=5)
                b = torch.unsqueeze(torch.unsqueeze(self.prototype, dim=2), dim=3)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_all = torch.mean(similarity_all, 5)
                similarity = torch.mean(similarity_all, 4)
                similarity = self.bn(similarity)
                similarity = F.avg_pool2d(similarity, kernel_size=similarity.shape[2:])
                similarity = similarity.view(similarity.shape[0:2])


            elif self.proto_form == 'vector2d_maxp':
                x = input
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=np.float)
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(torch.unsqueeze(x, dim=4), dim=5)
                b = torch.unsqueeze(torch.unsqueeze(self.prototype, dim=2), dim=3)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_all, _ = torch.max(similarity_all, 5)
                similarity, _ = torch.max(similarity_all, 4)
                '''similarity = self.bn(similarity)'''
                similarity = F.avg_pool2d(similarity, kernel_size=similarity.shape[2:])
                similarity = similarity.view(similarity.shape[0:2])
                #similarity = self.ln(similarity)

            elif self.proto_form == 'vector2d_maxp_att':
                x = torch.flatten(input, start_dim=2, end_dim=-1) # 16, 512, 56
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[2]), dtype=np.float) # 16, 4, 56, 56
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(x, dim=3)  # 16, 512, 56, 1
                b = torch.unsqueeze(self.prototype, dim=2) # 4, 512, 1, 56
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[2]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_att, ind = torch.max(similarity_all, 3) # 16, 4, 56

                #prototype_ind = np.zeros((x.shape[0], self.n_proto, self.prototype.shape[1], self.prototype.shape[2]), dtype=np.float)  # (16, 4, 512, 56)
                #prototype_ind = move_data_to_gpu(prototype_ind, True)
                #for i in range(0, ind.shape[0]):
                #    for j in range(0, ind.shape[1]):
                #            prototype_ind[i, j, :, :] = self.prototype[j, :, ind[i, j, :]]
                similarity_att = F.softmax(similarity_att, dim=-1)  # (16, 4, 56)
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[1]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        similarity[i][j] = torch.sum(torch.unsqueeze(similarity_att[i][j], dim=0) * x[i] * self.prototype[j, :, ind[i, j, :]], dim=-1)
                similarity = F.relu_(self.fc(similarity))
                similarity = similarity.view(similarity.shape[0:2])


        else:
            print('Wrong distance type!')


        if self.distance == 'cosine':
            return similarity, self.prototype
        else:
            print('Wrong distance type!')


