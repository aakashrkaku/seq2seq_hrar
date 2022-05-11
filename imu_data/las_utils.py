import torch
import torch.nn as nn
from torch.autograd import Variable  
import numpy as np
import editdistance as ed

# CreateOnehotVariable function
# *** DEV NOTE : This is a workaround to achieve one, I'm not sure how this function affects the training speed ***
# This is a function to generate an one-hot encoded tensor with given batch size and index
# Input : input_x which is a Tensor or Variable with shape [batch size, timesteps]
#         encoding_dim, the number of classes of input
# Output: onehot_x, a Variable containing onehot vector with shape [batch size, timesteps, encoding_dim]
def CreateOnehotVariable( input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data 
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
    
    return onehot_x

# TimeDistributed function
# This is a pytorch version of TimeDistributed layer in Keras I wrote
# The goal is to apply same module on each timestep of every instance
# Input : module to be applied timestep-wise (e.g. nn.Linear)
#         3D input (sequencial) with shape [batch size, timestep, feature]
# output: Processed output      with shape [batch size, timestep, output feature dim of input module]
def TimeDistributed(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size,time_steps,-1)

# LetterErrorRate function
# Merge the repeated prediction and calculate editdistance of prediction and ground truth
def LetterErrorRate(pred_y,true_y,data):
    ed_accumalate = []
    for p,t in zip(pred_y,true_y):
        compressed_t = [w for w in t if (w!=1 and w!=0)]
        
        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)
        if data == 'timit':
            compressed_t = collapse_phn(compressed_t)
            compressed_p = collapse_phn(compressed_p)
        ed_accumalate.append(ed.eval(compressed_p,compressed_t)/len(compressed_t))
    return ed_accumalate

def label_smoothing_loss(pred_y,true_y,label_smoothing=0.1):
    # Self defined loss for label smoothing
    # pred_y is log-scaled and true_y is one-hot format padded with all zero vector
    assert pred_y.size() == true_y.size()
    seq_len = torch.sum(torch.sum(true_y,dim=-1),dim=-1,keepdim=True)
    
    # calculate smoothen label, last term ensures padding vector remains all zero
    class_dim = true_y.size()[-1]
    smooth_y = ((1.0-label_smoothing)*true_y+(label_smoothing/class_dim))*torch.sum(true_y,dim=-1,keepdim=True)

    loss = - torch.mean(torch.sum((torch.sum(smooth_y * pred_y,dim=-1)/seq_len),dim=-1))

    return loss


def batch_iterator(batch_data, batch_label, listener, speller, optimizer, tf_rate, is_training, data='timit',**kwargs):
#     bucketing = kwargs['bucketing']
    use_gpu = kwargs['use_gpu']
    output_class_dim = kwargs['output_class_dim']
    label_smoothing = kwargs['label_smoothing']

    # Load data
#     if bucketing:
#         batch_data = batch_data.squeeze(dim=0)
#         batch_label = batch_label.squeeze(dim=0)
    current_batch_size = len(batch_data)
    max_label_len = min([batch_label.size()[1],kwargs['max_label_len']])

    batch_data = Variable(batch_data).type(torch.FloatTensor)
    batch_label = Variable(batch_label, requires_grad=False)
    criterion = nn.NLLLoss(ignore_index=0)
    if use_gpu:
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        criterion = criterion.cuda()
    # Forwarding
    optimizer.zero_grad()
    listner_feature = listener(batch_data)
    if is_training:
        raw_pred_seq, _ = speller(listner_feature,ground_truth=batch_label,teacher_force_rate=tf_rate)
    else:
        raw_pred_seq, _ = speller(listner_feature,ground_truth=None,teacher_force_rate=0)

    pred_y = (torch.cat([torch.unsqueeze(each_y,1) for each_y in raw_pred_seq],1)[:,:max_label_len,:]).contiguous()

    if label_smoothing == 0.0 or not(is_training):
        pred_y = pred_y.permute(0,2,1)#pred_y.contiguous().view(-1,output_class_dim)
        true_y = torch.max(batch_label,dim=2)[1][:,:max_label_len].contiguous()#.view(-1)

        loss = criterion(pred_y,true_y)
        # variable -> numpy before sending into LER calculator
#         batch_ler = LetterErrorRate(torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
#                                     true_y.cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)

    else:
        true_y = batch_label[:,:max_label_len,:].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        loss = label_smoothing_loss(pred_y,true_y,label_smoothing=label_smoothing)
        true_y = torch.max(true_y,dim=2)[1].cpu().numpy()
#         batch_ler = LetterErrorRate(torch.max(pred_y,dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
#                                     torch.max(true_y,dim=2)[1].cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)

    if is_training:
        loss.backward()
        optimizer.step()

    batch_loss = loss.cpu().data.numpy()
    

#     pred_y = torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy()
    
    
    
    return batch_loss, pred_y, true_y

def batch_iterator_2preds(batch_data, batch_label,batch_f_label, listener, speller, optimizer, tf_rate, is_training, data='timit',previous_pred = None,down_sample = True,**kwargs):
#     bucketing = kwargs['bucketing']
    use_gpu = kwargs['use_gpu']
    output_class_dim = kwargs['output_class_dim']
    label_smoothing = kwargs['label_smoothing']
    m_preds = kwargs['m_preds']
    # Load data
#     if bucketing:
#         batch_data = batch_data.squeeze(dim=0)
#         batch_label = batch_label.squeeze(dim=0)
    current_batch_size = len(batch_data)
    max_label_len = min([batch_label.size()[1],kwargs['max_label_len']])
    

    batch_data = Variable(batch_data).type(torch.FloatTensor)
    batch_label = Variable(batch_label, requires_grad=False)
    batch_f_label = Variable(batch_f_label, requires_grad=False)
    down_sample_factor = 2**(kwargs["listener_layer"])
    if down_sample:
        batch_f_label = batch_f_label[:,::down_sample_factor]
    
    criterion = nn.NLLLoss(ignore_index=0)
    criterion_f_label = nn.CrossEntropyLoss()
    if use_gpu:
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        batch_f_label = batch_f_label.cuda()
        criterion = criterion.cuda()
        criterion_f_label = criterion_f_label.cuda()
    # Forwarding
    optimizer.zero_grad()
    listner_feature = listener(batch_data)
    if m_preds:
        listner_feature[1] = listner_feature[1].transpose(1, 2)
        loss_f_label = criterion_f_label(listner_feature[1], batch_f_label)
        f_pred_y = torch.max(listner_feature[1],dim = 1)[1].cpu()
        f_true_y = batch_f_label.cpu()
        m = nn.Softmax(dim=1)
        f_pred_prob = m(listner_feature[1]).cpu()
        
    if is_training:
        raw_pred_seq, _ = speller(listner_feature[0],ground_truth=batch_label,teacher_force_rate=tf_rate,previous_pred=previous_pred)
    else:
        raw_pred_seq, attn = speller(listner_feature[0],ground_truth=None,teacher_force_rate=0, previous_pred=previous_pred)

    pred_y = (torch.cat([torch.unsqueeze(each_y,1) for each_y in raw_pred_seq],1)[:,:max_label_len,:]).contiguous()

    if label_smoothing == 0.0 or not(is_training):
        pred_y = pred_y.permute(0,2,1)#pred_y.contiguous().view(-1,output_class_dim)
        true_y = torch.max(batch_label,dim=2)[1][:,:max_label_len].contiguous()#.view(-1)

        loss = criterion(pred_y,true_y)
        # variable -> numpy before sending into LER calculator
#         batch_ler = LetterErrorRate(torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
#                                     true_y.cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)

    else:
        true_y = batch_label[:,:max_label_len,:].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        loss = label_smoothing_loss(pred_y,true_y,label_smoothing=label_smoothing)
        true_y = torch.max(true_y,dim=2)[1].cpu().numpy()
#         batch_ler = LetterErrorRate(torch.max(pred_y,dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
#                                     torch.max(true_y,dim=2)[1].cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)
    if m_preds:
        loss_ratio = loss/loss_f_label
    
    if is_training:
#         if m_preds:
#             loss += loss_f_label
        loss.backward()
        optimizer.step()

    batch_loss = loss.cpu().data.numpy()
    

#     pred_y = torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy()
    
    if not m_preds:
        f_pred_y = None
        f_true_y = None
        loss_ratio = None
    if is_training:
        return batch_loss, pred_y, true_y, f_pred_y, f_true_y, loss_ratio
    else:
        return batch_loss, pred_y, true_y, f_pred_y, f_true_y, f_pred_prob, loss_ratio,attn
    
def batch_iterator_2preds_ensemble(batch_data, batch_label,batch_f_label, listener1,listener2,listener3,listener4, speller, optimizer, tf_rate, is_training, data='timit',previous_pred = None,down_sample = True,**kwargs):
#     bucketing = kwargs['bucketing']
    use_gpu = kwargs['use_gpu']
    output_class_dim = kwargs['output_class_dim']
    label_smoothing = kwargs['label_smoothing']
    m_preds = kwargs['m_preds']
    # Load data
#     if bucketing:
#         batch_data = batch_data.squeeze(dim=0)
#         batch_label = batch_label.squeeze(dim=0)
    current_batch_size = len(batch_data)
    max_label_len = min([batch_label.size()[1],kwargs['max_label_len']])
    

    batch_data = Variable(batch_data).type(torch.FloatTensor)
    batch_label = Variable(batch_label, requires_grad=False)
    batch_f_label = Variable(batch_f_label, requires_grad=False)
    down_sample_factor = 2**(kwargs["listener_layer"])
    if down_sample:
        batch_f_label = batch_f_label[:,::down_sample_factor]
    
    criterion = nn.NLLLoss(ignore_index=0)
    criterion_f_label = nn.CrossEntropyLoss()
    if use_gpu:
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        batch_f_label = batch_f_label.cuda()
        criterion = criterion.cuda()
        criterion_f_label = criterion_f_label.cuda()
    # Forwarding
#     optimizer.zero_grad()
    listner_feature1 = listener1(batch_data)
    listner_feature2 = listener1(batch_data)
    listner_feature3 = listener1(batch_data)
    listner_feature4 = listener1(batch_data)
    if m_preds:
        listner_feature1[1] = listner_feature1[1].transpose(1, 2)
        listner_feature2[1] = listner_feature2[1].transpose(1, 2)
        listner_feature3[1] = listner_feature3[1].transpose(1, 2)
        listner_feature4[1] = listner_feature4[1].transpose(1, 2)
        ave_listerner_feature = torch.mean(torch.cat([listner_feature1[1][:,:,:,None],listner_feature2[1][:,:,:,None],listner_feature3[1][:,:,:,None],listner_feature4[1][:,:,:,None]],dim=3),dim = (3))
        loss_f_label = criterion_f_label(ave_listerner_feature, batch_f_label)
        f_pred_y = torch.max(ave_listerner_feature,dim = 1)[1].cpu()
        f_true_y = batch_f_label.cpu()
        m = nn.Softmax(dim=1)
        f_pred_prob = m(ave_listerner_feature).cpu()
        
    if is_training:
        raw_pred_seq, _ = speller(listner_feature1[0],listner_feature2[0],listner_feature3[0],listner_feature4[0],ground_truth=batch_label,teacher_force_rate=tf_rate,previous_pred=previous_pred)
    else:
        raw_pred_seq, attn1, attn2, attn3, attn4 = speller(listner_feature1[0],listner_feature2[0],listner_feature3[0],listner_feature4[0],ground_truth=None,teacher_force_rate=0, previous_pred=previous_pred)

    pred_y = (torch.cat([torch.unsqueeze(each_y,1) for each_y in raw_pred_seq],1)[:,:max_label_len,:]).contiguous()

    if label_smoothing == 0.0 or not(is_training):
        pred_y = pred_y.permute(0,2,1)#pred_y.contiguous().view(-1,output_class_dim)
        true_y = torch.max(batch_label,dim=2)[1][:,:max_label_len].contiguous()#.view(-1)

        loss = criterion(pred_y,true_y)
        # variable -> numpy before sending into LER calculator
#         batch_ler = LetterErrorRate(torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
#                                     true_y.cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)

    else:
        true_y = batch_label[:,:max_label_len,:].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        loss = label_smoothing_loss(pred_y,true_y,label_smoothing=label_smoothing)
        true_y = torch.max(true_y,dim=2)[1].cpu().numpy()
#         batch_ler = LetterErrorRate(torch.max(pred_y,dim=2)[1].cpu().numpy(),#.reshape(current_batch_size,max_label_len),
#                                     torch.max(true_y,dim=2)[1].cpu().data.numpy(),data) #.reshape(current_batch_size,max_label_len), data)
    if m_preds:
        loss_ratio = loss/loss_f_label
    
    if is_training:
        if m_preds:
            loss += loss_f_label
        loss.backward()
        optimizer.step()

    batch_loss = loss.cpu().data.numpy()
    

#     pred_y = torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy()
    
    if not m_preds:
        f_pred_y = None
        f_true_y = None
        loss_ratio = None
    if is_training:
        return batch_loss, pred_y, true_y, f_pred_y, f_true_y, loss_ratio
    else:
        return batch_loss, pred_y, true_y, f_pred_y, f_true_y, f_pred_prob, loss_ratio,attn1,attn2,attn3,attn4