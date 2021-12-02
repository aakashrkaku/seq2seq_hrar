import comet_ml
import os
from comet_ml import Experiment
import torch.optim as optim
import numpy as np
import pickle

import random
from las_model import Speller
from las_model import MultiStageModel as Listener
from las_utils import CreateOnehotVariable, batch_iterator

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import itertools
import pandas as pd


# evaluation metrics
from metrics import *


import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def pickling(file, path):
    pickle.dump(file, open(path, 'wb'))


def unpickling(path):
    file_return = pickle.load(open(path, 'rb'))
    return file_return


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self, datasetname):
        if datasetname == '50salads':
            char_map_str = """
            0  <SPACE>
            19 cut_tomato
            1 place_tomato_into_bowl
            2 cut_cheese
            3 place_cheese_into_bowl
            4 cut_lettuce
            5 place_lettuce_into_bowl
            6 add_salt
            7 add_vinegar
            8 add_oil
            9 add_pepper
            10 mix_dressing
            11 peel_cucumber
            12 cut_cucumber
            13 place_cucumber_into_bowl
            14 add_dressing
            15 mix_ingredients
            16 serve_salad_onto_plate
            17 action_start
            18 action_end
            20 <eoa>
            """
        elif datasetname == 'breakfast':
            char_map_str = """
            0 <SPACE>
            48 SIL
            1 pour_cereals
            2 pour_milk
            3 stir_cereals
            4 take_bowl
            5 pour_coffee
            6 take_cup
            7 spoon_sugar
            8 stir_coffee
            9 pour_sugar
            10 pour_oil
            11 crack_egg
            12 add_saltnpepper
            13 fry_egg
            14 take_plate
            15 put_egg2plate
            16 take_eggs
            17 butter_pan
            18 take_knife
            19 cut_orange
            20 squeeze_orange
            21 pour_juice
            22 take_glass
            23 take_squeezer
            24 spoon_powder
            25 stir_milk
            26 spoon_flour
            27 stir_dough
            28 pour_dough2pan
            29 fry_pancake
            30 put_pancake2plate
            31 pour_flour
            32 cut_fruit
            33 put_fruit2bowl
            34 peel_fruit
            35 stir_fruit
            36 cut_bun
            37 smear_butter
            38 take_topping
            39 put_toppingOnTop
            40 put_bunTogether
            41 take_butter
            42 stir_egg
            43 pour_egg2pan
            44 stirfry_egg
            45 add_teabag
            46 pour_water
            47 stir_tea
            49 <eoa> 
            """
        elif datasetname == 'stroke':
            char_map_str = """
            0 <SPACE>
            1 reach
            2 transport
            3 reposition
            4 stabilize
            5 rest
            6 <eoa> 
            """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            index, ch = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = index


    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ' '.join(string).replace('<SPACE>', ' ')


def feature_normalize(dataset, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(dataset, axis=0)
    if sigma is None:
        sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def windows(data, size, stride=20):
    start = 0
    data_size = data.shape[0]
    while start < data_size:
        yield int(start), int(start + size)
        start += stride


def segment_signal_eval(data, label_p, window_size=60, stride=20):
    # data: features shape: T,feature_len    label: shape: T
    segments = []
    labels_p = []
    for (start, end) in windows(data, window_size, stride):
        x = np.array(data[start:end])  # window_size, feature_len
        label = np.array(label_p[start:end])  # window_size
        if (x.shape[0] == window_size):
            segments.append(x)
            labels_p.append(label[window_size // 2 - stride // 2:window_size // 2 + stride // 2])
    labels_p = np.array(labels_p)  # b, stride
    segments = np.transpose(np.dstack(segments), (2, 1, 0))  # b, d, window_size
    return (segments, labels_p)

def segment_signal(data, label_p, text_transform, window_size=60, stride=20):
    segments = []
    labels_p = []
    inp_lens = []
    label_lens = []
    for (start, end) in windows(data, window_size, stride):
        x = np.array(data[start:end])
        label = [k for k, g in itertools.groupby(label_p[start:end])]
        label.append('<eoa>')
        label = torch.Tensor(text_transform.text_to_int(label))
        if (x.shape[0] == window_size):
            segments.append(torch.tensor(x))
            inp_lens.append(x.shape[0])
            labels_p.append(label)
            label_lens.append(len(label))
    if len(labels_p) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    labels_p = nn.utils.rnn.pad_sequence(labels_p, batch_first=True)
    segments = nn.utils.rnn.pad_sequence(segments, batch_first=True)
    return (segments.transpose(1,2), labels_p, inp_lens, label_lens)

def segment_signal_pad(data,label_p,text_transform, window_size = 60, stride = 20):
    segments = []
    labels_p = []
    inp_lens = []
    label_lens = []
    for (start, end) in windows(data, window_size, stride):
        x = np.array(data[start:end])
        label = [k for k, g in itertools.groupby(label_p[start:end])]
        label.append('<eoa>')
        label = torch.Tensor(text_transform.text_to_int(label))

        segments.append(torch.tensor(x))
        inp_lens.append(x.shape[0])
        labels_p.append(label)
        label_lens.append(len(label))
    if len(labels_p) == 0:
        return (np.array([]), np.array([]), np.array([]),np.array([]))
    labels_p = nn.utils.rnn.pad_sequence(labels_p, batch_first=True)
    segments = nn.utils.rnn.pad_sequence(segments, batch_first=True)
    return (segments.transpose(1,2), labels_p, inp_lens, label_lens)


class VideoDataset(Dataset):
    '''Video Dataset'''

    def __init__(self, data_list, file_dir, label_dir, text_transform, sample_rate, ws=90, stride=20, \
                 resamp_rate=False, eval_mode=False, train_mode=False,pad_windows=True):
        self.features_path = file_dir
        self.gt_path = label_dir
        self.ws = ws
        self.stride = stride
        self.filtered_list = []
        self.resamp_rate = resamp_rate
        self.eval_mode = eval_mode
        self.train_mode = train_mode
        self.text_transform = text_transform
        self.sample_rate = sample_rate
        self.pad_windows = pad_windows

        # print(data_list)
        if ".npy" in data_list:
            self.read_data_np(data_list)
        else:
            self.read_data(data_list)

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.filtered_list = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.filtered_list)

    def read_data_np(self, vid_list_file):
        self.filtered_list = list(np.load(vid_list_file))
        random.shuffle(self.filtered_list)

    def __len__(self):
        return len(self.filtered_list)

    def __getitem__(self, idx):
        #         print(idx)
        if self.train_mode:
            features = np.transpose(np.load(self.features_path + self.filtered_list[idx].split('.')[0] + '.npy'),
                                    (1, 0))  # 2048, 8453
            file_ptr = open(self.gt_path + self.filtered_list[idx].split('.')[0] + '.txt', 'r')  # 8453
            content = file_ptr.read().split('\n')[:-1]
            classes = content
        else:
            features = np.transpose(np.load(self.features_path + self.filtered_list[idx].split('.')[0] + '.npy'),
                                    (1, 0))  # 2048, 8453
            file_ptr = open(self.gt_path + self.filtered_list[idx].split('.')[0] + '.txt', 'r')  # 8453
            content = file_ptr.read().split('\n')[:-1]
            classes = content

        if self.eval_mode:
            segments, labels_p = segment_signal_eval(features, classes, self.ws, self.stride)
        else:
            if self.pad_windows:
                segments, labels_p, inp_lens, label_lens = segment_signal_pad(features, classes, self.text_transform, \
                                                                          self.ws, self.stride, )
            else:
                segments, labels_p, inp_lens, label_lens = segment_signal(features, classes, self.text_transform, \
                                                                          self.ws, self.stride, )

            if len(segments) == 0:
                # print(idx)
                idx_next = np.random.randint(0, self.__len__() - 1)
                return self.__getitem__(idx_next)
        return (segments[:, :, :], labels_p, inp_lens, label_lens)


def my_collate(batch):
    data_list = []
    labelp_list = []
    inp_lens_list = []
    label_lens_list = []
    for item in batch:
        data_list += item[0].transpose(1,2)
        labelp_list+=item[1]
        inp_lens_list += item[2]
        label_lens_list+= item[3]
    # data = torch.cat(data_list,dim=0)
    data = nn.utils.rnn.pad_sequence(data_list, batch_first=True).transpose(1, 2)
    labels_p = nn.utils.rnn.pad_sequence(labelp_list, batch_first=True)
    return data, labels_p, inp_lens_list, label_lens_list




def GreedyDecoder(output, labels, label_lengths, tt, blank_label=20, collapse_repeated=True, eoa_index=19):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(tt.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if (index != blank_label):
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                if (index == eoa_index):
                    break
                decode.append(index.item())
        decodes.append(tt.int_to_text(decode))
    return decodes, targets


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(listener, speller, optimizer, tf_rate, model_parameters, train_loader, scheduler, epoch, iter_meter,
          experiment, bs=30, encoding_dim=20):
    listener.train()
    speller.train()
    batch_idx = 0

    with experiment.train():
        for _, _data in enumerate(train_loader):
            bsz = _data[0].size()[0]
            idx = np.arange(bsz)
            np.random.shuffle(idx)
            inputs = _data[0][idx]
            all_labels = _data[1][idx]
            input_lens = [_data[2][k] for k in idx]
            label_lens = [_data[3][k] for k in idx]
            sub_batches = bsz // bs
            batch_idx += 1
            sub_idx = 0
            # print(bsz)
            for j in range(sub_batches + 1):
                sub_idx += 1
                sub_data = (
                inputs[j * bs:(j + 1) * bs], all_labels[j * bs:(j + 1) * bs], input_lens[j * bs:(j + 1) * bs],
                label_lens[j * bs:(j + 1) * bs])
                optimizer.zero_grad()
                x, labels, input_lengths, label_lengths = sub_data
                #  x batch, t, dim     label batch,t,7
                batch_size = x.size()[0]
                if batch_size == 0:
                    continue

                labels = CreateOnehotVariable(labels, encoding_dim=encoding_dim)
                loss, output, true_y = batch_iterator(x.transpose(1, 2), labels, listener, speller, optimizer, \
                                                      tf_rate=tf_rate, is_training=True, **model_parameters)
                experiment.log_metric('loss', loss.item(), step=iter_meter.get())
                experiment.log_metric('learning_rate', scheduler.get_last_lr()[0], step=iter_meter.get())

                optimizer.step()
                scheduler.step()
                iter_meter.step()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}({})/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, sub_idx, len(train_loader),
                        100. * batch_idx / len(train_loader), loss.item()))




def test(listener, speller, optimizer, model_parameters, test_loader, epoch, iter_meter, experiment, tt, bs=30,
         encoding_dim=20, dataset_name='50salads'):
    print('\nevaluating...')
    listener.eval()
    speller.eval()
    test_loss = 0

    test_cer = []
    batch_idx = 0
    running_total = 0
    preds = []
    gts = []

    with experiment.test():
        with torch.no_grad():
            for _, _data in enumerate(test_loader):
                bsz = _data[0].size()[0]
                inputs = _data[0]
                all_labels = _data[1]
                input_lens = _data[2]
                label_lens = _data[3]
                sub_batches = bsz // bs
                one_sess_preds = []
                one_sess_tars = []
                for j in range(sub_batches + 1):
                    batch_idx += 1
                    sub_data = (
                    inputs[j * bs:(j + 1) * bs], all_labels[j * bs:(j + 1) * bs], input_lens[j * bs:(j + 1) * bs],
                    label_lens[j * bs:(j + 1) * bs])
                    x, labels_input, input_lengths, label_lengths = sub_data
                    batch_size = x.size()[0]
                    if batch_size == 0:
                        continue
                    labels = CreateOnehotVariable(labels_input, encoding_dim=encoding_dim)
                    loss, output, labels = batch_iterator(x.transpose(1, 2), labels, listener, speller, optimizer, \
                                                          tf_rate=0, is_training=False, **model_parameters)
                    running_total += x.shape[0]
                    test_loss += loss.item() * (x.shape[0])

                    if dataset_name == '50salads':
                        blank_label_idx = 21
                        eoa_index_idx = 20
                    elif dataset_name == 'breakfast':
                        blank_label_idx = 50
                        eoa_index_idx = 49
                    elif dataset_name == 'stroke':
                        blank_label_idx = 7
                        eoa_index_idx = 6

                    decoded_preds, decoded_targets = GreedyDecoder(output.transpose(2, 1), labels_input, label_lengths,
                                                                   tt, blank_label=blank_label_idx,
                                                                   eoa_index=eoa_index_idx)
                    one_sess_preds += decoded_preds
                    one_sess_tars += decoded_targets
                one_sess_preds = ' '.join(one_sess_preds)
                one_sess_tars = ' '.join(one_sess_tars)

                one_sess_preds = one_sess_preds.split(' ')
                one_sess_tars = one_sess_tars.split(' ')

                preds.append(one_sess_preds)
                gts.append(one_sess_tars)

    df = get_df((preds, gts), names=test_loader.dataset.filtered_list, eos_idx=str(eoa_index_idx))

    test_loss = test_loss / running_total
    avg_cer = np.mean(df['wers'])
    avg_edit_score = np.mean(df['edit_score'])
    fdr, tpr, df = get_fdr_tpr(df)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f}, Average Edit Score: {:4f} \n'.format(test_loss, avg_cer,
                                                                                                    avg_edit_score))
    print('Test set: Average fdr: {:.4f}, Average tpr: {:4f} \n'.format(fdr, tpr))

    return test_loss, avg_cer, df



def removeDuplicates(S):
    n = len(S)
    if (n < 2):
        return
    j = 0
    for i in range(n):
        if (S[j] != S[i]):
            j += 1
            S[j] = S[i]
    j += 1
    S = S[:j]
    return S


# Dynamic Programming Python3
# implementation to find minimum
# number of deletions and insertions

# Returns length of length
# common subsequence for
# str1[0..m-1], str2[0..n-1]
def lcs(str1, str2, m, n):
    L = [[0 for i in range(n + 1)]
         for i in range(m + 1)]

    # Following steps build L[m+1][n+1]
    # in bottom up fashion. Note that
    # L[i][j] contains length of LCS
    # of str1[0..i-1] and str2[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                L[i][j] = 0
            elif (str1[i - 1] == str2[j - 1]):
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j],
                              L[i][j - 1])

                # L[m][n] contains length of LCS
    # for X[0..n-1] and Y[0..m-1]
    return L[m][n]


# function to find minimum number
# of deletions and insertions
def printMinDelAndInsert(str1, str2):
    m = len(str1)
    n = len(str2)
    leng = lcs(str1, str2, m, n)
    #     print("Minimum number of deletions = ",
    #                        m - leng, sep = ' ')
    #     print("Minimum number of insertions = ",
    #                         n - leng, sep = ' ')
    return (m - leng + n - leng) / n, (m - leng) / n, (n - leng) / n


def smoothen_string(string, len_same=2):
    string = list(string.rstrip())
    for i in range(len(string)):
        count = 0
        if i > len_same and i < len(string) - len_same:
            for j in range(len_same):
                if (string[i] == string[i - j]) and (string[i] == string[i + j]):
                    count += 1
                else:
                    break
            if count == len_same + 1:
                string[i] = string[i - len_same + 1]
    return ''.join(string)


def get_df(best_outdata, names, eos_idx='19'):
    new_preds = []
    for i in best_outdata[0]:
        # ['1', '2', ...,'100']
        if eos_idx in i:
            # get rid of the eos sign
            # i.remove(eos_idx)
            i = list(filter(lambda a: a != eos_idx, i))
        if len(i) >= 2:
            s = removeDuplicates(i)
        else:
            s = i
        new_preds.append(s)

    new_gts = []
    for i in best_outdata[1]:
        if eos_idx in i:
            # get rid of the eos sign
            # i.remove(eos_idx)  this seems to be wrong, only remove the first one
            i = list(filter(lambda a: a != eos_idx, i))
        if len(i) >= 2:
            s = removeDuplicates(i)
        else:
            s = i
        new_gts.append(s)

    data_dict = {'Gt': new_gts, 'Pred': new_preds}
    df = pd.DataFrame(data_dict)

    wers = []
    dels_rate = []
    ins_rate = []
    del_ins_rate = []
    lens_gt = []
    edit_score_list = []
    for i, j in zip(new_gts, new_preds):
        len_gt = len(i)
        wers.append(wer(i, j))
        del_in, dels, ins = printMinDelAndInsert(j, i)
        del_ins_rate.append(del_in)
        dels_rate.append(dels)
        ins_rate.append(ins)
        lens_gt.append(len_gt)
        edit_score_list.append(edit_score(j, i))

    df['wers'] = wers
    df['edit_score'] = edit_score_list
    df['del_in_error'] = del_ins_rate
    df['del_error'] = dels_rate
    df['in_error'] = ins_rate
    df['len_gt'] = lens_gt
    df['names'] = names
    df_new = df
    return df_new


def main(model_name, dataset_name, fold, learning_rate=5e-4, epochs=100,
         experiment=Experiment(api_key='dummy_key', disabled=True), listener_hidden_dim=256, listener_layer=4,
         multi_head=4, mlp_dim_in_attention=256, speller_rnn_layer=1, speller_hidden_dim=512, output_class_dim=20,
         label_smoothing=0.1, tf_rate_lowerbound=0.05, batch_size=256, random_seed=7, use_attention=True,
         use_mlp_in_attention=True, rnn_unit='LSTM', windowsize=90, stridesize=20, test_only='no', max_label_len=20,
         model_path='',
         path_root='',
         bs_for_loader=10,
         LR_scheduler=0,
         pad_windows= True):
    use_cuda = torch.cuda.is_available()



    # the following dimension are the dimension of the raw feature
    # if you want to use the frame-wise prediction of segmentation model, please change the dimension accordingly
    if dataset_name == 'stroke':
        input_feature_dim = 432
    elif dataset_name=='breakfast' or dataset_name=='50salads':
        input_feature_dim = 2048



    hparams = {"max_label_len": max_label_len,  #
               "input_feature_dim": input_feature_dim,  #
               "listener_hidden_dim": listener_hidden_dim,  #  listener LSTM output dimension
               "listener_layer": listener_layer,  # Number of layers in listener, the paper is using 3
               "multi_head": multi_head,  # Number of heads for multi-head attention
               "decode_mode": 1,
               "use_mlp_in_attention": use_mlp_in_attention,  # Set to False to exclude phi and psi in attention formula
               "mlp_dim_in_attention": mlp_dim_in_attention,  #
               "mlp_activate_in_attention": 'relu',  #
               "speller_rnn_layer": speller_rnn_layer,  # Default RNN layer number
               "speller_hidden_dim": 2 * listener_hidden_dim,  # speller LSTM output dimension
               "output_class_dim": output_class_dim,  #
               "rnn_unit": rnn_unit,  # Default recurrent unit in the original paper
               "use_gpu": use_cuda,
               "learning_rate": learning_rate,  # Learning rate
               "epochs": epochs,  # Number of epochs
               "tf_rate_upperbound": 1,  # teacher force rate: Upperbound
               "tf_rate_lowerbound": tf_rate_lowerbound,  # teacher force rate: Lowerbound
               "label_smoothing": label_smoothing,  # Epsilon for label smoothing (set 0 to disable LS)}
               "use_attention": use_attention,
               'windowsize': windowsize,
               'stridesize': stridesize
               }
    experiment.log_parameters(hparams)

    if not os.path.exists(model_path + model_name + '/' + fold):
        os.makedirs(model_path + model_name + '/' + fold)
    model_path = model_path + model_name + '/' + fold + '/'
    pickling(hparams, model_path + model_name + '_hparams_dict.p')

    torch.manual_seed(random_seed)

    if dataset_name == 'stroke':
        vid_list_file = path_root + "/data/" + dataset_name + "/splits/train.split" + fold + ".bundle.npy"
        vid_list_file_tst = path_root + "/data/" + dataset_name + "/splits/test.split" + fold + ".bundle.npy"
        vid_list_file_tst_all = path_root + "/data/" + dataset_name + "/splits/test_all.npy"

    elif dataset_name == '50salads' or dataset_name == "breakfast":
        # the following train/val/test list for each split are constructed according to the official code from ASRF https://github.com/yiskw713/asrf
        vid_list_file = path_root + "/data/" + dataset_name + "_asrf/train.split" + fold + ".bundle.npy"
        vid_list_file_tst = path_root + "/data/" + dataset_name + "_asrf/val.split" + fold + ".bundle.npy"
        vid_list_file_tst_all = path_root + "/data/" + dataset_name + "_asrf/test.split" + fold + ".bundle.npy"



    # if you want to use the frame-wise prediction by segmentation model as input, then set the features_path to be the one that store the frame-wise prediction
    # otherwise, this should be the path for raw features
    features_path = path_root + "/data/" + dataset_name + "/features/"
    gt_path = path_root + "/data/" + dataset_name + "/groundTruth/"


    mapping_file = path_root + "/data/" + dataset_name + "/mapping.txt"

    text_transform = TextTransform(dataset_name)
    transformed_dataset = {
        'train': VideoDataset(vid_list_file, features_path, gt_path, text_transform, sample_rate=1, ws=windowsize,
                              stride=stridesize, \
                              resamp_rate=False, eval_mode=False, train_mode=True,pad_windows=pad_windows),
        'validate': VideoDataset(vid_list_file_tst, features_path, gt_path, text_transform, sample_rate=1,
                                 ws=windowsize, stride=stridesize, \
                                 resamp_rate=False, eval_mode=False, train_mode=False,pad_windows=pad_windows),
        'test': VideoDataset(vid_list_file_tst_all, features_path, gt_path, text_transform, sample_rate=1,
                                 ws=windowsize, stride=stridesize, \
                                 resamp_rate=False, eval_mode=False, train_mode=False,pad_windows=pad_windows)
        }

    bs = bs_for_loader
    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=1 if x != 'train' else bs,\
                                shuffle=False if x != 'train' else True,\
                                num_workers=1,collate_fn=my_collate, pin_memory=True) for x in ['train', 'validate','test']}

    num_layers = 10
    num_f_maps = 64
    num_stages = listener_layer

    # listener = Listener(3, 5,256, 4, dropout= 0.,in_features=2048).cuda()
    listener = Listener(num_stages, num_layers, num_f_maps, input_feature_dim, 2 * listener_hidden_dim).cuda()
    speller = Speller(**hparams)
    p1 = sum([param.nelement() for param in listener.parameters()])
    p2 = sum([param.nelement() for param in speller.parameters()])
    print('Num Model Parameters', p1 + p2)

    optimizer = optim.AdamW([{'params': listener.parameters()}, {'params': speller.parameters()}], \
                            hparams['learning_rate'], weight_decay=0.0001)

    if LR_scheduler == 0:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                                  steps_per_epoch=int(len(dataloader['train'])) * int(
                                                      1500 / batch_size),
                                                  epochs=hparams['epochs'], final_div_factor=1,
                                                  anneal_strategy='linear')
    elif LR_scheduler == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 60, gamma=0.1, last_epoch=-1)
    elif LR_scheduler == 2:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 180], gamma=0.1, last_epoch=-1)
    elif LR_scheduler == 3:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999, last_epoch=-1)
    else:
        scheduler = None

    iter_meter = IterMeter()
    best_cer = 100
    best_edit = 0


# the training code
    if test_only == 'no':
        for epoch in range(1, epochs + 1):
            tf_rate = hparams["tf_rate_upperbound"] - (
                        hparams["tf_rate_upperbound"] - hparams["tf_rate_lowerbound"]) * (epoch / hparams['epochs'])
            train(listener, speller, optimizer, tf_rate, hparams, dataloader['train'], scheduler, epoch, iter_meter, \
                  experiment, bs=batch_size, encoding_dim=output_class_dim)
            val_loss, avg_cer, df_val = test(listener, speller, optimizer, hparams, dataloader['validate'], epoch,
                                             iter_meter, experiment, tt=text_transform, bs=batch_size,
                                             encoding_dim=output_class_dim, dataset_name=dataset_name)
            if avg_cer < best_cer:
                best_cer = avg_cer
                torch.save(listener.state_dict(), model_path + model_name + '_listener.pth')
                torch.save(speller.state_dict(), model_path + model_name + '_speller.pth')

            if np.mean(df_val['edit_score'])>best_edit:
                best_edit = np.mean(df_val['edit_score'])
                torch.save(listener.state_dict(), model_path + model_name + '_listener_editscore.pth')
                torch.save(speller.state_dict(), model_path + model_name + '_speller_editscore.pth')

            experiment.log_metric('test_loss', val_loss, step=iter_meter.get())
            experiment.log_metric('cer', avg_cer, step=iter_meter.get())
            experiment.log_metric('edit_score', np.mean(df_val['edit_score']), step=iter_meter.get())

# best model selected by edit score on validation set for testing
    elif test_only=='editscore':
        listener_path = model_path + model_name + '_listener_editscore.pth'
        speller_path = model_path + model_name + '_speller_editscore.pth'

        listener.load_state_dict(torch.load(listener_path))
        speller.load_state_dict(torch.load(speller_path))

        val_loss, avg_cer, df_val = test(listener, speller, optimizer, hparams, dataloader['test'], 0,
                                         iter_meter, experiment, tt=text_transform, bs=batch_size, encoding_dim=output_class_dim,dataset_name=dataset_name)
        experiment.log_metric('test_loss', val_loss, step=iter_meter.get())
        experiment.log_metric('cer', avg_cer, step=iter_meter.get())
        experiment.log_metric('edit_score', np.mean(df_val['edit_score']), step=iter_meter.get())
        df_val.to_pickle(model_path + "test_result_es.pkl")
        df_val.to_csv(model_path + "test_result_es.csv")

# best model selected by WER (AER) on validation set for testing
    else:
        listener_path = model_path + model_name + '_listener.pth'
        speller_path = model_path + model_name + '_speller.pth'

        listener.load_state_dict(torch.load(listener_path))
        speller.load_state_dict(torch.load(speller_path))

        val_loss, avg_cer, df_val = test(listener, speller, optimizer, hparams, dataloader['test'], 0,
                                         iter_meter, experiment, tt=text_transform, bs=batch_size,
                                         encoding_dim=output_class_dim, dataset_name=dataset_name)
        experiment.log_metric('test_loss', val_loss, step=iter_meter.get())
        experiment.log_metric('cer', avg_cer, step=iter_meter.get())
        experiment.log_metric('edit_score', np.mean(df_val['edit_score']), step=iter_meter.get())
        df_val.to_pickle(model_path + "val_result.pkl")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_fold', type=str, required=True,
                        help="'1','2','3','4'")
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--rnn_unit', type=str, default='GRU', help="'LSTM' or 'GRU'")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--listener_hidden_dim', type=int, default=256)
    parser.add_argument('--listener_layer', type=int, default=4)
    parser.add_argument('--multi_head', type=int, default=4)
    parser.add_argument('--mlp_dim_in_attention', type=int, default=256)
    parser.add_argument('--speller_rnn_layer', type=int, default=1)
    parser.add_argument('--speller_hidden_dim', type=int, default=512)
    parser.add_argument('--output_class_dim', type=int, default=21)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--tf_rate_lowerbound', type=float, default=0.05)
    parser.add_argument('--use_attention', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--use_mlp_in_attention', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='stroke')
    parser.add_argument('--windowsize', type=int, default=90)
    parser.add_argument('--stridesize', type=int, default=20)
    parser.add_argument('--test_only', type=str, default='no')
    parser.add_argument('--max_label_len', type=int, default=20, help='max label length')
    parser.add_argument('--model_path', type=str,
                        default='',
                        help='the model root')
    parser.add_argument('--path_root', type=str,
                        default='',
                        help='the data root')
    parser.add_argument('--bs_for_loader', type=int, default=10, help='the loader file nun')
    parser.add_argument('--LR_scheduler', type=int, default=0, help='which LR_scheduler to use')
    parser.add_argument('--pad_windows',  type=str2bool, nargs='?',
                        const=True, default=True)

# comet related
    parser.add_argument('--comet_api_key',  type=str,  default='',
                        help='the api key of comet')
    parser.add_argument('--project_name',  type=str,  default='',
                        help='the project name of comet')
    parser.add_argument('--workspace',  type=str,  default='',
                        help='the workspace name of comet')




    args = parser.parse_args()
    model_name = args.model_name
    fold = args.val_fold
    epochs = args.num_epochs
    batch_size = args.batch_size
    random_seed = args.seed
    learning_rate = args.learning_rate
    listener_hidden_dim = args.listener_hidden_dim
    listener_layer = args.listener_layer
    multi_head = args.multi_head
    mlp_dim_in_attention = args.mlp_dim_in_attention
    use_mlp_in_attention = args.use_mlp_in_attention
    use_attention = args.use_attention
    speller_rnn_layer = args.speller_rnn_layer
    speller_hidden_dim = args.speller_hidden_dim
    output_class_dim = args.output_class_dim
    rnn_unit = args.rnn_unit
    label_smoothing = args.label_smoothing
    tf_rate_lowerbound = args.tf_rate_lowerbound


    experiment = Experiment(api_key=args.comet_api_key,project_name=args.project_name, workspace=args.workspace)

    if not model_name:
        model_name = rnn_unit + '_lhd_' + str(listener_hidden_dim) + '_ll_' + str(listener_layer) + '_ua_' + str(
            use_attention) + '_um_' + str(use_mlp_in_attention) + \
                     '_mda_' + str(mlp_dim_in_attention) + '_mh_' + str(multi_head) + '_lr_' + str(learning_rate)
    experiment.set_name(model_name + '_' + fold)

    main(model_name, args.dataset_name, fold, learning_rate=learning_rate, epochs=epochs, experiment=experiment, \
         listener_hidden_dim=listener_hidden_dim, listener_layer=listener_layer, multi_head=multi_head, \
         mlp_dim_in_attention=mlp_dim_in_attention, speller_rnn_layer=speller_rnn_layer, \
         speller_hidden_dim=speller_hidden_dim, output_class_dim=output_class_dim, label_smoothing=label_smoothing, \
         tf_rate_lowerbound=tf_rate_lowerbound, batch_size=batch_size, random_seed=random_seed,
         use_attention=use_attention, \
         use_mlp_in_attention=use_mlp_in_attention, rnn_unit=rnn_unit, windowsize=args.windowsize,
         stridesize=args.stridesize, test_only=args.test_only, max_label_len=args.max_label_len,
         model_path=args.model_path, path_root=args.path_root, bs_for_loader=args.bs_for_loader,
         LR_scheduler=args.LR_scheduler,pad_windows=args.pad_windows)

