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
import math
from tcn import *



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
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, listener_layer, rnn_unit, use_gpu, dropout_rate=0.0,only_encoder=False, **kwargs):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        assert self.listener_layer>=1,'Listener should have at least 1 layer'
        
        self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

        for i in range(1,self.listener_layer):
            setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

        self.use_gpu = use_gpu
        self.only_encoder = only_encoder
        if self.only_encoder:
            self.fc = nn.Linear(listener_hidden_dim*2,5)
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        output,_  = self.pLSTM_layer0(input_x)
        for i in range(1,self.listener_layer):
            output, _ = getattr(self,'pLSTM_layer'+str(i))(output)
        
        if self.only_encoder:
            output1 = self.fc(output)
            
        return [output,output1]
    
class Listener_tcn(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, listener_layer, rnn_unit, use_gpu,freeze_tcn,wt_path, dropout_rate=0.0,only_encoder=False, **kwargs):
        super(Listener_tcn, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        self.tcn_model = model = ActionSegmentRefinementFramework(
                                                    in_channel=77,
                                                    n_features=64,
                                                    n_classes=5,
                                                    n_stages=4,
                                                    n_layers=10,
                                                    n_stages_asb=3,
                                                    n_stages_brb=3,)
        model_wts = torch.load(wt_path, map_location=torch.device('cpu'))
        self.tcn_model.load_state_dict(model_wts)
        self.freeze_tcn = freeze_tcn
        if self.freeze_tcn:
            for p in self.tcn_model.parameters():
                p.requires_grad = False

        assert self.listener_layer>=1,'Listener should have at least 1 layer'
        
        self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

        for i in range(1,self.listener_layer):
            setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

        self.use_gpu = use_gpu
        self.only_encoder = only_encoder
        if self.only_encoder:
            self.fc = nn.Linear(listener_hidden_dim*2,5)
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        if self.tcn_model.training:
            input_x = self.tcn_model(input_x.transpose(1,2))[0][0]
        else:
            input_x = self.tcn_model(input_x.transpose(1,2))[0]
        input_x = input_x.transpose(1,2)
        output,_  = self.pLSTM_layer0(input_x)
        for i in range(1,self.listener_layer):
            output, _ = getattr(self,'pLSTM_layer'+str(i))(output)
        
        if self.only_encoder:
            output1 = self.fc(output)
            
        return [output,output1]

    
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


class Conv_GRU_Listener(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, stride=2, dropout=0.1, in_features = 77,only_encoder=False):
        super(Conv_GRU_Listener, self).__init__()
        self.cnn = nn.Conv1d(in_features, 1024, 3, stride=stride, padding=0)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(1024, 1024, kernel=3, stride=1, dropout=dropout) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(1024, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.only_encoder=only_encoder
        if self.only_encoder:
            self.fc = nn.Linear(rnn_dim*2,5)
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
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        if self.only_encoder:
            output1 = self.fc(x)
#         x = self.classifier(x)
        return [x,output1]
    
class Transformer_Listener(nn.Module):
    def __init__(self, n_transformer_layers, transformer_hid_dim, conv_feature_dim,n_attention_head, stride=4, dropout=0.1, in_features = 77, **kwargs):
        super(Transformer_Listener, self).__init__()
        self.cnn = nn.Conv1d(in_features, conv_feature_dim, 5, stride=stride, padding=5//2)  # cnn for extracting heirachal features
        self.transformer_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = conv_feature_dim, nhead =n_attention_head, dim_feedforward=transformer_hid_dim,dropout=dropout),n_transformer_layers)
        self.pos_encoder = PositionalEncoding(conv_feature_dim, dropout)

    def forward(self, x):
        x = x.transpose(1, 2) # (batch, feature, time)
        x = self.cnn(x)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.pos_encoder(x)
        x = self.transformer_model(x)
#         x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
#         self.pe.transpose_(0,1)
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)

# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, output_class_dim,  speller_hidden_dim, rnn_unit, speller_rnn_layer, use_gpu, max_label_len,
                 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention,
                 multi_head, decode_mode,use_attention,listener_type, **kwargs):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.listener_type = listener_type
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = output_class_dim
        self.rnn_layer = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = Attention( mlp_preprocess_input=use_mlp_in_attention, preprocess_mlp_dim=mlp_dim_in_attention,
                                        activate=mlp_activate_in_attention, input_feature_dim=speller_hidden_dim,
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
            concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
            raw_pred = self.softmax(self.character_distribution(concat_feature))

            return raw_pred, hidden_state, context, attention_score
        else:
            context = torch.mean(listener_feature, dim = 1)
            concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
            raw_pred = self.softmax(self.character_distribution(concat_feature))
            return raw_pred, hidden_state,  context, None

    def forward(self, listener_feature, ground_truth=None, teacher_force_rate = 0.9, previous_pred = None):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

        batch_size = listener_feature.size()[0]
        
        if (ground_truth is not None) and (teacher_force):
            output_word = self.float_type(previous_pred)
        elif previous_pred is not None:
#             print("prev pred:",previous_pred.shape)
            output_word = self.float_type(previous_pred)
        else:
            output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size,1))),self.label_dim)
        
        if self.use_gpu:
            output_word = output_word.cuda()
        
        if self.listener_type == 'transformer':
#             rnn_input = torch.cat([output_word,torch.mean(listener_feature,dim = 1).unsqueeze(1)],dim=-1)
            rnn_input = torch.cat([output_word,listener_feature[:,-2:-1,:]],dim=-1)
        else:
#             print("output_word", output_word.shape)
#             print("Listener_featue",listener_feature.shape)
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
                projected_src = [torch.sum(listener_feature*att_s.unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1) \
                                for att_s in attention_score]
                context = self.dim_reduce(torch.cat(projected_src,dim=-1))
        else:
            # TODO: other attention implementations
            pass
        
        

        return attention_score,context