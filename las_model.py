import torch
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from las_utils import TimeDistributed,CreateOnehotVariable
import numpy as np
import copy


# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True,
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        if timestep%2!=0:
            input_x = input_x[:,:-1, :]
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, listener_layer, rnn_unit, use_gpu, dropout_rate=0.0, **kwargs):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        assert self.listener_layer>=1,'Listener should have at least 1 layer'
        
        self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

        for i in range(1,self.listener_layer):
            setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        output,_  = self.pLSTM_layer0(input_x)
        for i in range(1,self.listener_layer):
            output, _ = getattr(self,'pLSTM_layer'+str(i))(output)
        
        return output
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
        self.num_classes = num_classes

    def forward(self, x):
        # x [10, 900, 2048]
        x = x.permute(0,2,1)  # 10, 2048, 900
        out = self.stage1(x)  # 10, 256, 450
        # print(out.size())
        # outputs = out.unsqueeze(0)
        for s in self.stages:
            # out = s(F.softmax(out, dim=1)) # 10, 256, 450
            out = s(out) # 10, 256, 450
            # outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        # return outputs
        # print(out.size())
        out = out.permute(0,2,1)
        return out

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        # self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=3, stride=2, padding=1)  # to reduce the sequence length

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        # out = self.conv_out(out)
        out = F.relu(self.conv_out(out))
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

    
class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, time)
        x = x.transpose(2, 1).contiguous() # (batch, time, channel)
        x = self.layer_norm(x)
        return x.transpose(2, 1).contiguous() # (batch, channel, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv1d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(out_channels)
        self.layer_norm2 = CNNLayerNorm(out_channels)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class ConvListener(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, stride=2, dropout=0.1, in_features = 77):
        super(ConvListener, self).__init__()
        self.cnn = nn.Conv1d(in_features, 64, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(64, 64, kernel=3, stride=1, dropout=dropout) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(64, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
#         self.classifier = nn.Sequential(
#             nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(rnn_dim, n_class)
#         )

    def forward(self, x):
        x = x.transpose(1, 2) # (batch, feature, time)
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] , sizes[2])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)         # b,t,rnn_dim
        x = self.birnn_layers(x)
#         x = self.classifier(x)
        return x

# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, output_class_dim,  speller_hidden_dim, rnn_unit, speller_rnn_layer, use_gpu, max_label_len,
                 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention, listener_hidden_dim,
                 multi_head, decode_mode,use_attention, **kwargs):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = output_class_dim
        self.rnn_layer = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = Attention( mlp_preprocess_input=use_mlp_in_attention, preprocess_mlp_dim=mlp_dim_in_attention,
                                        activate=mlp_activate_in_attention, input_feature_dim=2*listener_hidden_dim,
                                        multi_head=multi_head)
        self.character_distribution = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        if self.use_gpu:
            self = self.cuda()

    # Stepwise operation of each sequence
    def forward_step(self,input_word, last_hidden_state,listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word,last_hidden_state)
        if self.use_attention:
            attention_score, context = self.attention(rnn_output,listener_feature)
            # attention_score [batch size, T (attention score of each time step)]
            # context  [batch size,  listener feature dimension]
            concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
            # context  [batch size,  listener feature dimension+]
            raw_pred = self.softmax(self.character_distribution(concat_feature))

            return raw_pred, hidden_state, context, attention_score
        else:
            context = torch.mean(listener_feature, dim = 1)    # listener_feature b,t,listener feature dimension
            concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
            raw_pred = self.softmax(self.character_distribution(concat_feature))
            return raw_pred, hidden_state,  context, None

    def forward(self, listener_feature, ground_truth=None, teacher_force_rate = 0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

        batch_size = listener_feature.size()[0]

        output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size,1))),self.label_dim)  # b,1,num_class
        
        if self.use_gpu:
            output_word = output_word.cuda()
        rnn_input = torch.cat([output_word,listener_feature[:,0:1,:]],dim=-1)

        hidden_state = None
        raw_pred_seq = []
        output_seq = []
        attention_record = []

        if (ground_truth is None) or (not teacher_force):
            max_step = self.max_label_len
        else:
            max_step = ground_truth.size()[1]

        for step in range(max_step):
            raw_pred, hidden_state, context, attention_score = self.forward_step(rnn_input, hidden_state, listener_feature)
            # print(raw_pred)
            raw_pred_seq.append(raw_pred)
            attention_record.append(attention_score)
            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:,step:step+1,:].type(self.float_type)
            else:
                # Case 0. raw output as input
                if self.decode_mode == 0:
                    output_word = raw_pred.unsqueeze(1)
                # Case 1. Pick character with max probability
                elif self.decode_mode == 1:
                    output_word = torch.zeros_like(raw_pred)
                    for idx,i in enumerate(raw_pred.topk(1)[1]):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)             
                # Case 2. Sample categotical label from raw prediction
                else:
                    sampled_word = Categorical(raw_pred).sample()
                    output_word = torch.zeros_like(raw_pred)
                    for idx,i in enumerate(sampled_word):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)
                
            rnn_input = torch.cat([output_word,context.unsqueeze(1)],dim=-1)

        return raw_pred_seq,attention_record


# Attention mechanism
# Currently only 'dot' is implemented
# please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
# Input : Decoder state                      with shape [batch size, 1, decoder hidden dimension]
#         Compressed feature from Listner    with shape [batch size, T, listener feature dimension]
# Output: Attention score                    with shape [batch size, T (attention score of each time step)]
#         Context vector                     with shape [batch size,  listener feature dimension]
#         (i.e. weighted (by attention score) sum of all timesteps T's feature)
class Attention(nn.Module):  

    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, mode='dot', input_feature_dim=512,
                multi_head=1):
        super(Attention,self).__init__()
        self.mode = mode.lower()
        self.mlp_preprocess_input = mlp_preprocess_input
        self.multi_head = multi_head
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim  = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim,preprocess_mlp_dim*multi_head)
            self.psi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            if self.multi_head > 1:
                self.dim_reduce = nn.Linear(input_feature_dim*multi_head,input_feature_dim)
            if activate != 'None':
                self.activate = getattr(F,activate)
            else:
                self.activate = None

    def forward(self, decoder_state, listener_feature):
        if self.mlp_preprocess_input:
            if self.activate:
                comp_decoder_state = self.activate(self.phi(decoder_state))
                comp_listener_feature = self.activate(TimeDistributed(self.psi,listener_feature))
            else:
                comp_decoder_state = self.phi(decoder_state)
                comp_listener_feature = TimeDistributed(self.psi,listener_feature)
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature

        if self.mode == 'dot':
            if self.multi_head == 1:
                energy = torch.bmm(comp_decoder_state,comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
                attention_score = [self.softmax(energy)]
                context = torch.sum(listener_feature*attention_score[0].unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1)
            else:
                attention_score =  [ self.softmax(torch.bmm(att_querry,comp_listener_feature.transpose(1, 2)).squeeze(dim=1))\
                                    for att_querry in torch.split(comp_decoder_state, self.preprocess_mlp_dim, dim=-1)]
                # print(attention_score[0].size())
                projected_src = [torch.sum(listener_feature*att_s.unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1) \
                                for att_s in attention_score]
                context = self.dim_reduce(torch.cat(projected_src,dim=-1))
        else:
            # TODO: other attention implementations
            pass

        return attention_score,context
