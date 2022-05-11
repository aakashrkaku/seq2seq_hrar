import pickle
def pickling(file,path):
    pickle.dump(file,open(path,'wb'))
def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return
import comet_ml

from las_model import Speller, Transformer_Listener, Conv_GRU_Listener, Listener
from las_utils import CreateOnehotVariable, batch_iterator_2preds_ensemble

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import itertools
import pandas as pd
from scipy.interpolate import CubicSpline
import torch
import collections
from torch.autograd import Variable
import torch.nn.functional as F
import time

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        idle 1
        rest 1
        reach 2
        reposition 3
        retract 3
        stabilize 4
        transport 5
        <eoa> 6
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
#             self.index_map[int(index)] = ch
            self.index_map[int(index)] = index
#         self.index_map[1] = ' '

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
        return ''.join(string).replace('<SPACE>', ' ')

def feature_normalize(dataset, mu = None, sigma = None):
    if mu is None:
        mu = np.mean(dataset,axis = 0)
    if sigma is None:
        sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def windows(data, size, stride = 20):
    start = 0
    data_size = data.shape[0]
    while start < data_size:
        yield int(start), int(start + size)
        start += stride
        
def segment_signal_eval(data,label_p, window_size = 60, stride = 20):
    segments = []
    labels_p = []
    for (start, end) in windows(data, window_size, stride):
        x = np.array(data[start:end])
        label = np.array(label_p[start:end])
        if(x.shape[0] == window_size):
            segments.append(x)
            labels_p.append(label[window_size//2-stride//2:window_size//2+stride//2])
    labels_p = np.array(labels_p)
    segments = np.transpose(np.dstack(segments),(2,1,0))
    return (segments, labels_p)

def segment_signal(data,label_p,text_transform, window_size = 60, stride = 20):
    segments = []
    labels_p = []
    inp_lens = []
    label_lens = []
    f_labels_p = []
    for (start, end) in windows(data, window_size, stride):
        x = np.array(data[start:end])
        fine_label = np.array(label_p[start:end])
        label = [k for k, g in itertools.groupby(label_p[start:end])]
        label.append('<eoa>')
        label = torch.Tensor(text_transform.text_to_int(label))
#         label = np.array(label_p[start:end])
        if(x.shape[0] == window_size):
            segments.append(x)
            inp_lens.append(x.shape[0])
            labels_p.append(label)
            label_lens.append(len(label))
            f_labels_p.append(fine_label)
#     labels_p = np.array(labels_p)
    labels_p = nn.utils.rnn.pad_sequence(labels_p, batch_first=True)
    segments = np.transpose(np.dstack(segments),(2,1,0))
    f_labels_p = np.array(f_labels_p)
    return (segments, labels_p, f_labels_p, inp_lens, label_lens)

def get_frac(end_points,e,ws):
    fraction = 1
    for i in range(end_points.shape[0]):
        if end_points[i] > e and i > 0:
            len_primi = end_points[i] - end_points[i-1]
            inside_window = e - end_points[i-1]
            fraction = inside_window/len_primi
            break
        elif end_points[i] > e  and i == 0:
            len_primi= end_points[i]
            inside_window = e
            fraction = inside_window/len_primi
            break
        else:
            fraction = 1
    
    return fraction

def segment_signal_new(data,label_p,text_transform, window_size = 60, stride = 20):
    segments = []
    labels_p = []
    inp_lens = []
    label_lens = []
    f_labels_p = []
    prv_window_lab = []
    
    label_p = np.array(label_p)
    dict_names = {'rest':0,'reach':1,'retract':2,'reposition':2,'stabilize':3,'transport':4,'idle':0}
    for k,v in dict_names.items():
        label_p[label_p == k] = v
    end_points = np.where(np.diff(label_p)!=0)[0]
    
    prv_label = None
    for (start, end) in windows(data, window_size, stride):
        x = np.array(data[start:end])
#         print('Before padding:',x.shape)
        fine_label = np.array(label_p[start:end])
        fine_label = torch.tensor(fine_label.astype(int)).long()
        
        num_steps = x.shape[0]
        if num_steps%8 != 0:
            num_steps-= num_steps%8
            x = x[:num_steps]
            fine_label = fine_label[:num_steps]
            end -= num_steps%8
        if fine_label.shape[0]==0 or fine_label.shape[0]<100:
            continue
        fine_label_4_lab = fine_label + 1
        label_int = [k for k, g in itertools.groupby(fine_label_4_lab[100:500])]
        frac_last_prim = get_frac(end_points,end,window_size)
        if frac_last_prim < 0.5 and len(label_int) > 1:
            label_int = label_int[:-1]
#         label.append('<eoa>')
        label_int.append(6)
#         label = torch.Tensor(text_transform.text_to_int(label))
        label = torch.Tensor(label_int)
#         print(label)
        last_lab = torch.Tensor([label_int[-2]])
#         label = np.array(label_p[start:end])
#         if(x.shape[0] == window_size):
        segments.append(torch.tensor(x))
        inp_lens.append(x.shape[0])
        labels_p.append(label)
        label_lens.append(len(label)-1)
        f_labels_p.append(fine_label)
        if prv_label is None:
            prv_window_lab.append(torch.tensor([1]))
        else:
            prv_window_lab.append(last_lab)
#     print(labels_p)
#     labels_p = np.array(labels_p)
    segments = nn.utils.rnn.pad_sequence(segments,batch_first=True)
#     print('After padding:',segments.shape)
    labels_p = nn.utils.rnn.pad_sequence(labels_p, batch_first=True)
    prv_window_lab = nn.utils.rnn.pad_sequence(prv_window_lab, batch_first = True)
#     segments = np.transpose(np.dstack(segments),(2,1,0))
    f_labels_p = nn.utils.rnn.pad_sequence(f_labels_p, batch_first=True, padding_value=-100)
#     f_labels_p = np.array(f_labels_p,dtype=object)
    return (segments.transpose(1,2), labels_p, f_labels_p, inp_lens, label_lens, prv_window_lab)
#     return (segments, labels_p, f_labels_p, inp_lens, label_lens)

from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import freqz

def butter_bandpass(lowcut=7.5, highcut=20, fs =100, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut = 7.5, highcut = 20, fs = 100.0, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def magnitude_warp(x, sigma=0.6, knot=5):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def warp_data(df):
    cols_to_magW = ['lumbarflexiondeg',
 'lumbarlateralrtdeg',
 'lumbaraxialrtdeg',
 'thoracicflexiondeg',
 'thoraciclateralrtdeg',
 'thoracicaxialrtdeg',
 'elbowflexionltdeg',
 'elbowflexionrtdeg',
 'shouldertotalflexionltdeg',
 'shouldertotalflexionrtdeg',
 'shoulderflexionltdeg',
 'shoulderflexionrtdeg',
 'shoulderabductionltdeg',
 'shoulderabductionrtdeg',
 'shoulderrotationoutltdeg',
 'shoulderrotationoutrtdeg',
 'wristextensionltdeg',
 'wristextensionrtdeg',
 'wristradialltdeg',
 'wristradialrtdeg',
 'wristsupinationltdeg',
 'wristsupinationrtdeg']
    
    x = np.array(df[cols_to_magW])
    x = magnitude_warp(x[None,:,:])
    df[cols_to_magW] = x[0,:,:]
    return df


def GenerateRandomCurves(X, sigma=0.1, knot=2):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs=[]
    for i in range(X.shape[1]):
        cs.append(CubicSpline(xx[:,i], yy[:,i])(x_range))
    
    return np.array(cs).transpose()

def DistortTimesteps(X, sigma=0.1):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = []
    for i in range(X.shape[1]):
        t_scale.append((X.shape[0]-1)/tt_cum[-1,i])
    #t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    for i in range(X.shape[1]):
        tt_cum[:,i] = tt_cum[:,i]*t_scale[i]
    
    return tt_cum
def DA_MagWarp(X, sigma=0.2):
    for i in range(X.shape[0]):
        X[i]=X[i] * GenerateRandomCurves(X[i].T, sigma).T
    return X

def DA_TimeWarp(X, sigma=0.1):
    for i in range(X.shape[0]):
        v = X[i].T
        tt_new = DistortTimesteps(v, sigma)
        X_new = np.zeros(v.shape)
        x_range = np.arange(v.shape[0])
        for j in range(v.shape[1]):
            X_new[:,j] = np.interp(x_range, tt_new[:,j], v[:,j])
        X[i] = X_new.T 
    return X

def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape) #B,C,T
    return X+myNoise

def DA_Scaling(X, sigma=0.8):
    for i in range(X.shape[0]):
        #scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
        #myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(X[i].shape[0],1)) # shape=(1,3)
        myNoise = np.matmul(scalingFactor,np.ones((1,X[i].shape[1])) )
        X[i] = X[i]*myNoise
    return X

def data_aug(segments,augs):
    
    for aug in augs:
        if aug=='jitter':
            segments = DA_Jitter(segments)
            
            
        elif aug=='magW':
            segments = DA_MagWarp(segments)
            
            
        elif aug == 'timeW':
            segments = DA_TimeWarp(segments)
            
        elif aug == 'scaling':
            segments = DA_Scaling(segments)
    return segments
      
def printChanges(s1, s2, dp, print_true = False): 
      
    i = len(s1) 
    j = len(s2) 
    deletions = 0
    insertions = 0
    substitutions = 0
    
    subs_for_margin = 0
    subs_for_middle = 0
    
    ins_for_margin = 0
    ins_for_middle = 0
    
    dels_for_margin = 0
    dels_for_middle = 0
    
    rt_margin = 0
    lt_margin = 0
      
   # Check till the end  
    while(i > 0 and j > 0): 
#         print('i=',i)
#         print('j=',j)
#         print('==')
          
        # If characters are same  
        if s1[i - 1] == s2[j - 1]: 
            i -= 1
            j -= 1
            if (s2[j-1]==s2[0]):
                lt_margin = 1
            if (s2[j-1]==s2[-1]):
                rt_margin = 1
                
            
              
        # Replace 
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            if print_true:
                print("change", s1[i - 1], 
                          "to", s2[j - 1])
            if (s2[j-1] == s2[-1]) or (s2[j-1]==s2[0]):
                subs_for_margin += 1
                if s2[j-1] == s2[-1]:
                    rt_margin = 1
                else:
                    lt_margin = 1
            else:
                subs_for_middle += 1
            j -= 1
            i -= 1
            substitutions += 1
            
        # Add 
        elif dp[i][j] == dp[i][j - 1] + 1:
            if print_true:
                print("Add", s2[j - 1])
            if (s2[j-1] == s2[-1]) or (s2[j-1]==s2[0]):
                ins_for_margin += 1
                if s2[j-1] == s2[-1]:
                    rt_margin = 1
                else:
                    lt_margin = 1
            else:
                ins_for_middle += 1
            j -= 1
            insertions += 1
        
        # Delete 
        elif dp[i][j] == dp[i - 1][j] + 1: 
            if print_true:
                print("Delete", s1[i - 1]) 
            i -= 1
            deletions += 1
            if (lt_margin == 0) and (rt_margin == 0):
                dels_for_margin += 1
            elif (lt_margin == 1) and (rt_margin == 1):
                dels_for_margin += 1
            else:
                dels_for_middle += 1
                
        
        elif i==0 and j==0:
            break
    
    if i == 0:
        for k in range(j):
            if print_true:
                print("Add", s2[k])
            insertions +=1
            if (s2[k] == s2[-1]) or (s2[k]==s2[0]):
                ins_for_margin += 1
                if s2[k] == s2[-1]:
                    rt_margin = 1
                else:
                    lt_margin = 1
            else:
                ins_for_middle += 1
    if j == 0:
        for k in range(i):
            if print_true:
                print("Delete", s1[k])
            deletions += 1
            if (lt_margin == 0) and (rt_margin == 0):
                dels_for_margin += 1
            elif (lt_margin == 1) and (rt_margin == 1):
                dels_for_margin += 1
            else:
                dels_for_middle += 1
    
    return insertions, deletions, substitutions, ins_for_margin, ins_for_middle, dels_for_margin, dels_for_middle, subs_for_margin, subs_for_middle
    
#     print(i)
#     print(j)
              
# Function to compute the DP matrix  
def editDP(s1, s2, print_true = False): 
      
    len1 = len(s1) 
    len2 = len(s2) 
    dp = [[0 for i in range(len2 + 1)] 
             for j in range(len1 + 1)] 
      
    # Initilize by the maximum edits possible  
    for i in range(len1 + 1): 
        dp[i][0] = i 
    for j in range(len2 + 1): 
        dp[0][j] = j 
      
    # Compute the DP Matrix 
    for i in range(1, len1 + 1): 
        for j in range(1, len2 + 1): 
              
            # If the characters are same  
            # no changes required  
            if s2[j - 1] == s1[i - 1]: 
                dp[i][j] = dp[i - 1][j - 1] 
                  
            # Minimum of three operations possible  
            else: 
                dp[i][j] = 1 + min(dp[i][j - 1], 
                                   dp[i - 1][j - 1], 
                                   dp[i - 1][j]) 
                                     
    # Print the steps  
#     print(dp[len1][len2])
#     print(np.array(dp))
    i,d,s, i_mr, i_mi, d_mr, d_mi, s_mr, s_mi = printChanges(s1, s2, dp, print_true)
    return i,d,s, i_mr, i_mi,d_mr, d_mi, s_mr, s_mi

def get_fdr_tpr(df):
    df['Subject'] = df['Subject'].apply(lambda x:x.split('/')[-1])
    ins = []
    dels = []
    subs = []
    pred_len = []
    for i in range(df.shape[0]):
        out = editDP(df['Pred'].iloc[i],df['Gt'].iloc[i])
        ins.append(out[0])
        dels.append(out[1])
        subs.append(out[2])
        pred_len.append(len(df['Pred'].iloc[i]))
    df['ins'] = ins
    df['dels'] = dels
    df['subs'] = subs
    df['len_pred'] = pred_len
    out = pd.pivot_table(df,values = ['ins','dels','subs','len_gt','len_pred'],index = 'Subject',aggfunc=sum)
    out['FDR'] = (out['dels'] + out['subs'])/out['len_pred']
    out['TPR'] = 1-(out['ins']+out['subs'])/out['len_gt']
    mean_fdr = np.mean(out['FDR'])
    mean_tpr = np.mean(out['TPR'])
    
    return mean_fdr, mean_tpr, df

class IMUDataset(Dataset):
    '''IMU Dataset'''
    def __init__(self, root_dir, subjects,activities,sample_num,text_transform,  normalize = True, ws = 60, stride = 20,\
                 resamp_rate = False, one_arm_cols = False,remove_rot = True, eval_mode = False, train_mode = False,\
                 use_synthetic_data = False, synthetic_data_root_dir = None, percentage_s_data_use=0.5, use_band_pass_filter = False,\
                healthy_control_dir = None, healthy_subjects = None, warp_rotations = False,augments=None,sev_mode=False):
        self.root_dir = root_dir
        self.subjects = subjects
        self.normalize = normalize
        self.activities = activities
        self.sample_num = sample_num
        self.ws = ws
        self.stride = stride
        file_list = os.listdir(root_dir)
        self.filtered_list = []
        self.resamp_rate = resamp_rate
        self.one_arm_cols = one_arm_cols
        self.remove_rot = remove_rot
        self.eval_mode = eval_mode
        self.train_mode = train_mode
        self.text_transform=text_transform
        self.use_synthetic_data = use_synthetic_data
        self.use_band_pass_filter = use_band_pass_filter
        self.warp_rotations = warp_rotations
        self.augments = augments
        self.sev_mode = sev_mode
        
        self.impaired_subs = []
        self.healthy_subs = []
        self.synthetic_subs = []
        
        if healthy_control_dir:
            healthy_file_list = os.listdir(healthy_control_dir)
            if healthy_subjects:
                for hf in healthy_file_list:
                    for hs in healthy_subjects:
                        if '_'+hs+'_' in hf:
                            self.healthy_subs.append(healthy_control_dir+hf)
            else:
                self.healthy_subs += [healthy_control_dir+i for i in healthy_file_list]
            
        for f in file_list:
            for s in self.subjects:
                if s+'_' in f:
                    for a in self.activities:
                        if a in f:
                            self.impaired_subs.append(self.root_dir+f)
        if self.use_synthetic_data:
            s_data_file_list = os.listdir(synthetic_data_root_dir)
            np.random.shuffle(s_data_file_list)
            
            num_s_data = len(s_data_file_list)
            max_num_s_data = int(percentage_s_data_use*len(self.impaired_subs))
            
            self.synthetic_subs = [synthetic_data_root_dir+i for i in s_data_file_list[:max_num_s_data]]
            
        self.filtered_list = self.impaired_subs + self.healthy_subs + self.synthetic_subs
        self.filtered_list = list(np.unique(self.filtered_list))
        
        
    
    def __len__(self):
        return len(self.filtered_list)
    
    def __getitem__(self,idx):
#         print(idx)
        rand_value = np.random.random()
        used_s_data = False
        if self.train_mode:
            random_shift = np.random.randint(0,self.stride)
            data = pd.read_csv(self.filtered_list[idx]).iloc[random_shift:,:]
            if self.warp_rotations and ('synthetic_samples' not in self.filtered_list[idx]):
                p = np.random.random()
                if p < 0.5:
                    data = warp_data(data)
        else:
            data = pd.read_csv(self.filtered_list[idx])
        data['motion_apd'] = data.iloc[:,-6:-1].idxmax(1)
        data.drop(labels = ['markers','markernames'],axis = 1, inplace = True)
        split_path = self.filtered_list[idx].split('_')
        sub_name = split_path[0]
        act_name = split_path[1]
        rep_num = self.filtered_list[idx][-5]
        
        if self.remove_rot:
            data.drop(labels = ['upperspinecoursedeg','upperspinepitchdeg','upperspinerolldeg',\
                                'upperarmcourseltdeg','upperarmpitchltdeg','upperarmrollltdeg',\
                                'forearmcourseltdeg','forearmpitchltdeg','forearmrollltdeg',\
                                'handcourseltdeg','handpitchltdeg','handrollltdeg','upperarmcoursertdeg',\
                                'upperarmpitchrtdeg','upperarmrollrtdeg','forearmcoursertdeg',\
                                'forearmpitchrtdeg','forearmrollrtdeg','handcoursertdeg','handpitchrtdeg',\
                                'handrollrtdeg','lowerspinecoursedeg','lowerspinepitchdeg','lowerspinerolldeg',\
                                'pelviscoursedeg','pelvispitchdeg','pelvisrolldeg'], axis = 1, inplace = True)
        if self.resamp_rate:
            rate = np.random.uniform(0.25,2)
#             print(rate)
            data = resample_data(data, rate)
        
        if self.normalize and ('synthetic_samples' not in self.filtered_list[idx]):
            data.iloc[:,0:-17] = feature_normalize(data.iloc[:,0:-17])
            
        readings = np.array(data.iloc[:,:-16])
        
        if self.eval_mode:
            segments, labels_p = segment_signal_eval(readings, data['class'],self.ws, self.stride)
        else:
            segments, labels_p, f_labels_p, inp_lens, label_lens, prv_lab = segment_signal_new(readings, data['class'],\
                                                                                               self.text_transform,self.ws, self.stride,)
        if self.augments:
            p_aug = np.random.random()
            if p_aug < 0.5:
                segments = data_aug(segments,self.augments)
            
        if self.use_band_pass_filter:
            segments = butter_bandpass_filter(segments)
            
        n = segments.shape[0]
        sub_name = [sub_name]*segments.shape[0]
        act_name = [act_name]*segments.shape[0]
        rep_num = [rep_num]*segments.shape[0]
            
        return (segments[:,1:,:], labels_p, f_labels_p, inp_lens, label_lens, sub_name, act_name, rep_num, prv_lab)   

def my_collate(batch):
    data_list = []
    labelp_list = []
    f_labelp_list = []
    inp_lens_list = []
    label_lens_list = []
    sub_labels = []
    act_labels = []
    rep_num = []
    prv_labels = []
    for item in batch:
        data_list+=item[0].transpose(1,2)
        labelp_list+=item[1]
        f_labelp_list+=item[2]
        inp_lens_list += item[3]
        label_lens_list+= item[4]
        sub_labels += item[5]
        act_labels += item[6]
        rep_num += item[7]
        prv_labels += item[8]
#     data = torch.cat(data_list,dim =0)
    data = nn.utils.rnn.pad_sequence(data_list, batch_first=True).transpose(1,2)
    labels_p = nn.utils.rnn.pad_sequence(labelp_list, batch_first=True)
    f_labels_p = nn.utils.rnn.pad_sequence(f_labelp_list, batch_first=True, padding_value=-100)
    prv_labels = nn.utils.rnn.pad_sequence(prv_labels,batch_first=True)
    return data, labels_p, f_labels_p, inp_lens_list, label_lens_list, sub_labels, act_labels, rep_num, prv_labels
#     return data, labels_p, inp_lens_list, label_lens_list, sub_labels, act_labels, rep_num

import os
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)
#                 distance[cur_row_idx][j] = min(i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

def GreedyDecoder(output, labels, label_lengths,tt, blank_label=7, collapse_repeated=True, eoa_index=6):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(tt.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if (index != blank_label):
                if collapse_repeated and j != 0 and index == args[j -1]:
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

def test(listener1,listener2,listener3,listener4,speller,optimizer,model_parameters,test_loader,tt, phase, bs = 30,\
         window_preds = True):
    print('\nevaluating...')
    listener1.eval()
    listener2.eval()
    listener3.eval()
    listener4.eval()
    speller.eval()
    test_loss = 0
#     test_cer, test_wer = [], []
    test_cer = []
    batch_idx = 0
    running_total = 0
    preds = []
    gts = []
    subs = []
    acts = []
    rep_label = []
    running_correct = 0
    running_total_acc = 0
    loss_ratios = []
    if window_preds:
        w_preds = []
        w_gts = []
    all_f_probs = []
    all_output_probs = []
    
    attention_scrs1 = []
    attention_scrs2 = []
    attention_scrs3 = []
    attention_scrs4 = []
    all_f_labels = []
    all_f_preds = []
#     all_xs = []
    all_label_lens = []
    all_input_lens = []
    
    with torch.no_grad():
        for _, _data in enumerate(test_loader[phase]):
            bsz = _data[0].size()[0]
#                 idx = np.arange(bsz)
#                 np.random.shuffle(idx)
            inputs = _data[0]
            all_labels = _data[1]
            fine_labels = _data[2]
            input_lens = _data[3]
            label_lens = _data[4]
            sub_labels = _data[5]
            act_labels = _data[6]
            rep_num = _data[7]
            sub_batches = bsz//bs
            one_sess_preds = []
            one_sess_tars = []
            previous_pred = torch.Tensor([[1]])
            for j in range(sub_batches + 1):
                batch_idx += 1
                sub_data = (inputs[j*bs:(j+1)*bs],all_labels[j*bs:(j+1)*bs],fine_labels[j*bs:(j+1)*bs],\
                            input_lens[j*bs:(j+1)*bs],label_lens[j*bs:(j+1)*bs],sub_labels[j*bs:(j+1)*bs],act_labels[j*bs:(j+1)*bs],rep_num[j*bs:(j+1)*bs])
                x,labels_input,f_labels,input_lengths, label_lengths,sub_minibatch,act_minibatch,rep_minibatch = sub_data
                batch_size = x.size()[0]
                if batch_size == 0:
                    continue
                labels = CreateOnehotVariable(labels_input,encoding_dim=7)
                previous_pred = CreateOnehotVariable(previous_pred,encoding_dim=7)
                previous_pred = previous_pred.cuda() if model_parameters['use_gpu'] else previoud_pred
#                     print(previous_pred)
                loss, output, labels, f_output, f_labels, f_output_prob, loss_r,attn_score1,attn_score2,attn_score3,attn_score4, = batch_iterator_2preds_ensemble(x.transpose(1,2),\
                                                                                                                   labels,f_labels,\
                                                                                                                   listener1,listener2,listener3,listener4,speller,\
                                                                                                                   optimizer,\
                                                                                                                   tf_rate = 0,\
                                                                                                                   is_training=False,\
                                                                                                         previous_pred=previous_pred,\
                                                                                                                   **model_parameters)
                previous_pred = output
                running_total += x.shape[0]
                test_loss += loss.item()*(x.shape[0])

                running_correct += torch.sum((f_output == f_labels)).item()
                running_total_acc += f_labels.shape[0] * f_labels.shape[1]

                loss_ratios.append(loss_r.item())

                decoded_preds, decoded_targets = GreedyDecoder(output.transpose(2, 1), labels_input, label_lengths, tt)
#                     print(decoded_preds)
#                     print(int(decoded_preds[0][-1]))
                if len(decoded_preds[0])>0:
                    previous_pred = torch.Tensor([[int(decoded_preds[0][-1])]])
                else:
                    previous_pred = torch.Tensor([[1]])

                one_sess_preds += decoded_preds
                one_sess_tars += decoded_targets
                if window_preds:
                    w_preds.append(decoded_preds)
                    w_gts.append(decoded_targets)
#                     break
                all_f_probs.append(f_output_prob.numpy())
                all_output_probs.append(output.cpu().numpy())
                all_label_lens.append(label_lengths)
                all_input_lens.append(input_lengths)

                all_f_labels.append(f_labels.cpu().numpy())
                all_f_preds.append(f_output.cpu().numpy())
#                     break
                subs.append(sub_minibatch)
                acts.append(act_minibatch)
                rep_label.append(rep_minibatch)

                attn_score_np1 = []
                for a in attn_score1:
                    try:
                        attn_score_np1.append(a[0].cpu().numpy())
                    except:
                        attn_score_np1.append(0)
                attention_scrs1.append(attn_score_np1)
                
                attn_score_np2 = []
                for a in attn_score2:
                    try:
                        attn_score_np2.append(a[0].cpu().numpy())
                    except:
                        attn_score_np2.append(0)
                attention_scrs2.append(attn_score_np2)
                
                attn_score_np3 = []
                for a in attn_score3:
                    try:
                        attn_score_np3.append(a[0].cpu().numpy())
                    except:
                        attn_score_np3.append(0)
                attention_scrs3.append(attn_score_np3)
                
                attn_score_np4 = []
                for a in attn_score4:
                    try:
                        attn_score_np4.append(a[0].cpu().numpy())
                    except:
                        attn_score_np4.append(0)
                attention_scrs4.append(attn_score_np4)
                    
    return all_label_lens,all_input_lens, w_preds, all_output_probs, w_gts, all_f_preds, all_f_probs, all_f_labels, subs, acts, rep_label, (attention_scrs1,attention_scrs2,attention_scrs3,attention_scrs4)

def removeDuplicates(S):      
    n = len(S)   
    if (n < 2) : 
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
            elif(str1[i - 1] == str2[j - 1]): 
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
    return (m-leng + n - leng)/n, (m-leng)/n, (n-leng)/n  

def smoothen_string(string, len_same=2):
    string = list(string.rstrip())
    for i in range(len(string)):
        count = 0
        if i>len_same and i<len(string)-len_same:
            for j in range(len_same):
                if (string[i] == string[i-j]) and (string[i]==string[i+j]):
                    count += 1
                else:
                    break
            if count == len_same+1:
                string[i] = string[i-len_same+1]
    return ''.join(string)

def get_df(best_outdata):
    new_preds = []
    for i in best_outdata[0]:
        i = i.replace('6','')
        if len(i) >=2:
            s = removeDuplicates(list(i.rstrip()))
            s = ''.join(s)
        else:
            s = i
        new_preds.append(s)

    new_gts = []
    for i in best_outdata[1]:
        i = i.replace('6','')
        if len(i)>=2:
            s = removeDuplicates(list(i.rstrip()))
            s = ''.join(s)
        else:
            s = i
        new_gts.append(s)

    data_dict = {'Subject':best_outdata[2],'Activity':best_outdata[3],'Rep_num':best_outdata[4],'Gt':new_gts,'Pred':new_preds}
    df = pd.DataFrame(data_dict)

    cers = []
    dels_rate = []
    ins_rate = []
    del_ins_rate= []
    lens_gt = []
    for i,j in zip(list(df['Gt']),list(df['Pred'])):
#         i = i.replace('6','')
#         j = j.replace('6','') 
        if len(i) >=2:
            i = ''.join(removeDuplicates(list(i.rstrip())))
        if len(j)>=2:
            j = ''.join(removeDuplicates(list(j.rstrip())))
        len_gt= len(i)
        cers.append(cer(i,j,ignore_case=True,remove_space=True))
        del_in, dels, ins = printMinDelAndInsert(j,i)
        del_ins_rate.append(del_in)
        dels_rate.append(dels)
        ins_rate.append(ins)
        lens_gt.append(len_gt)
    df['cers'] = cers

    df['del_in_error'] = del_ins_rate
    df['del_error'] = dels_rate
    df['in_error'] = ins_rate
    df['len_gt'] = lens_gt

    df_new = df.drop_duplicates()
    return df_new

def dot_attention(comp_decoder_state, comp_listener_feature):
    energy = torch.bmm(comp_decoder_state,comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
    attention_score = [F.softmax(energy,dim=-1)]
    context = torch.sum(comp_listener_feature*attention_score[0].unsqueeze(2).repeat(1,1,comp_listener_feature.size(2)),dim=1)
    return attention_score, context

def replace_name_keys(sp_sd, num):
    sp_sd['rnn_layer'+num+'.weight_ih_l0'] = sp_sd.pop('rnn_layer.weight_ih_l0')
    sp_sd['rnn_layer'+num+'.weight_hh_l0'] = sp_sd.pop('rnn_layer.weight_hh_l0')
    sp_sd['rnn_layer'+num+'.bias_ih_l0'] = sp_sd.pop('rnn_layer.bias_ih_l0')
    sp_sd['rnn_layer'+num+'.bias_hh_l0'] = sp_sd.pop('rnn_layer.bias_hh_l0')
    sp_sd['character_distribution'+num+'.weight'] = sp_sd.pop('character_distribution.weight')
    sp_sd['character_distribution'+num+'.bias'] = sp_sd.pop('character_distribution.bias')
    return sp_sd

class Ensemble_Speller(nn.Module):
    def __init__(self, output_class_dim,  speller_hidden_dim, rnn_unit, speller_rnn_layer, use_gpu, max_label_len,
                 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention,
                 multi_head, decode_mode,use_attention,listener_type, **kwargs):
        super(Ensemble_Speller, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.listener_type = listener_type
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = output_class_dim
        self.rnn_layer1 = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.rnn_layer2 = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.rnn_layer3 = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.rnn_layer4 = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = dot_attention
        self.character_distribution1 = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.character_distribution2 = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.character_distribution3 = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.character_distribution4 = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        if self.use_gpu:
            self = self.cuda()

    # Stepwise operation of each sequence
    def forward_step(self,input_word, last_hidden_state,listener_feature,rl,cd):
        rnn_output, hidden_state = rl(input_word,last_hidden_state)
        if self.use_attention:
            attention_score, context = self.attention(rnn_output,listener_feature)
            concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
            raw_pred = self.softmax(cd(concat_feature))

            return raw_pred, hidden_state, context, attention_score
        else:
            context = torch.mean(listener_feature, dim = 1)
            concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
            raw_pred = self.softmax(cd(concat_feature))
            return raw_pred, hidden_state,  context, None

    def forward(self, listener_feature1,listener_feature2,listener_feature3,listener_feature4, ground_truth=None,\
                teacher_force_rate = 0.9, previous_pred = None):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

        batch_size = listener_feature1.size()[0]
        
        if previous_pred is not None:
#             print("prev pred:",previous_pred.shape)
            output_word = self.float_type(previous_pred)
        else:
            output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size,1))),self.label_dim)
        
        if self.use_gpu:
            output_word = output_word.cuda()
        
        
        rnn_input1 = torch.cat([output_word,listener_feature1[:,0:1,:]],dim=-1)
        rnn_input2 = torch.cat([output_word,listener_feature2[:,0:1,:]],dim=-1)
        rnn_input3 = torch.cat([output_word,listener_feature3[:,0:1,:]],dim=-1)
        rnn_input4 = torch.cat([output_word,listener_feature4[:,0:1,:]],dim=-1)

        hidden_state1 = None
        hidden_state2 = None
        hidden_state3 = None
        hidden_state4 = None
        
        raw_pred_seq = []
        output_seq = []
        attention_record1 = []
        attention_record2 = []
        attention_record3 = []
        attention_record4 = []

        if (ground_truth is None) or (not teacher_force):
            max_step = self.max_label_len
        else:
            max_step = ground_truth.size()[1]

        for step in range(max_step):
            raw_pred1, hidden_state1, context1, attention_score1 = self.forward_step(rnn_input1, hidden_state1,\
                                                                                     listener_feature1,self.rnn_layer1,\
                                                                                     self.character_distribution1)
            raw_pred2, hidden_state2, context2, attention_score2 = self.forward_step(rnn_input2, hidden_state2,\
                                                                                     listener_feature2,self.rnn_layer2,\
                                                                                     self.character_distribution2)
            raw_pred3, hidden_state3, context3, attention_score3 = self.forward_step(rnn_input3, hidden_state3,\
                                                                                     listener_feature3,self.rnn_layer3,\
                                                                                     self.character_distribution3)
            raw_pred4, hidden_state4, context4, attention_score4 = self.forward_step(rnn_input4, hidden_state4,\
                                                                                     listener_feature4,self.rnn_layer4,\
                                                                                     self.character_distribution4)
            all_preds = torch.cat([raw_pred1,raw_pred2,raw_pred3,raw_pred4],dim=0)
            prob_preds = torch.exp(all_preds)
            average_prob = torch.mean(prob_preds,axis = 0).unsqueeze(0)
            average_raw_pred = torch.log(average_prob)
            raw_pred_seq.append(average_raw_pred)
            attention_record1.append(attention_score1)
            attention_record2.append(attention_score2)
            attention_record3.append(attention_score3)
            attention_record4.append(attention_score4)
            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:,step:step+1,:].type(self.float_type)
            else:
                # Case 0. raw output as input
                if self.decode_mode == 0:
                    output_word = average_raw_pred.unsqueeze(1)
                # Case 1. Pick character with max probability
                elif self.decode_mode == 1:
                    output_word = torch.zeros_like(average_raw_pred)
                    for idx,i in enumerate(average_raw_pred.topk(1)[1]):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)             
                # Case 2. Sample categotical label from raw prediction
                else:
                    sampled_word = Categorical(average_raw_pred).sample()
                    output_word = torch.zeros_like(average_raw_pred)
                    for idx,i in enumerate(sampled_word):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)
                
            rnn_input1 = torch.cat([output_word,context1.unsqueeze(1)],dim=-1)
            rnn_input2 = torch.cat([output_word,context2.unsqueeze(1)],dim=-1)
            rnn_input3 = torch.cat([output_word,context3.unsqueeze(1)],dim=-1)
            rnn_input4 = torch.cat([output_word,context4.unsqueeze(1)],dim=-1)

        return raw_pred_seq,attention_record1,attention_record2,attention_record3,attention_record4

def main(model_name):
    start_time = time.time()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model_path = '/scratch/ark576/seq2seq_model_har/'+model_name+'/'
    hparams_dict_path = model_path+ 'fold_1/'+model_name+'_hparams_dict.p'
    
    hparams = unpickling(hparams_dict_path)
    
    fold = 'fold_1'

    path = '/scratch/ark576/HAR-oat-data-july-2020/HAR-data-processed-2/'
    resamp = False
    rr = True
    train_activity = ['deodorant','combing','glasses','drinking','feeding','brushing','RTT','shelf left side',\
                      'shelf right side','face wash','deodrant','RTT right side','RTT left side']
    val_activity = ['deodorant','combing','glasses','drinking','feeding','brushing','RTT','shelf left side',\
                      'shelf right side','face wash','deodrant','RTT right side','RTT left side']
    train_sample_num = ['1','2','3','4']
    val_train_sample_num = ['5']
    val_sample_num = ['1','2','3','4','5']
    test_activity = ['deodorant','combing','glasses','drinking','feeding','brushing','RTT','shelf left side',\
                      'shelf right side','face wash','deodrant','RTT right side','RTT left side']
    test_sample_num = ['1','2','3','4','5']
    # test_subjects =  ['S0004','S0044','S0042','S0026','S0039','S0017','S0037','S0047']
    test_subjects =  ['s4','s44','s42','s26','s39','s17','s37','s47']
    severe_subjects = ['s21','s29','s31','s34','s49','s50','s51']
    
    fold_dict = unpickling('transformed_data_four_fold_dict.p')
    all_fold = ['fold_1','fold_2','fold_3','fold_4']
    
    idx_val_fold = all_fold.index(fold)
    all_fold.pop(idx_val_fold)
    text_transform = TextTransform()

    train_subjects = fold_dict[all_fold[0]]+fold_dict[all_fold[1]]+fold_dict[all_fold[2]]
    val_subjects = fold_dict[fold]

    transformed_dataset = {'test':IMUDataset(path,test_subjects, test_activity,test_sample_num,text_transform, normalize = True,\
                                             ws = 600, stride = 400, resamp_rate = False, one_arm_cols = None,\
                                             use_band_pass_filter = False ),\
                            'severe':IMUDataset(path,severe_subjects, test_activity,test_sample_num,text_transform, normalize = True,\
                                             ws = 600, stride = 400, resamp_rate = False, one_arm_cols = None,\
                                             use_band_pass_filter = False,sev_mode=True),\
                                          }
    bs = 10
    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=1 if x == 'test' or x=='severe' else bs,\
                                shuffle=False if x == 'test' or x == 'severe' else True,\
                                num_workers=1,collate_fn=my_collate, pin_memory=True) for x in ['test','severe']}
    data_sizes ={x: len(transformed_dataset[x]) for x in ['test','severe']}
#     print('Train subjects = ', train_subjects)
    print('Val subjects = ',val_subjects)
    print('Test subjects = ', test_subjects)
    
    listener_path = model_path+model_name+'_listener.pth'
    speller_path = model_path+model_name+'_speller.pth'
    
    lt1_sd = torch.load( model_path+ 'fold_1/'+model_name+'_listener.pth')
    lt2_sd = torch.load( model_path+ 'fold_2/'+model_name+'_listener.pth')
    lt3_sd = torch.load( model_path+ 'fold_3/'+model_name+'_listener.pth')
    lt4_sd = torch.load( model_path+ 'fold_4/'+model_name+'_listener.pth')
    
    sp1_sd = torch.load( model_path+ 'fold_1/'+model_name+'_speller.pth')
    sp2_sd = torch.load( model_path+ 'fold_2/'+model_name+'_speller.pth')
    sp3_sd = torch.load( model_path+ 'fold_3/'+model_name+'_speller.pth')
    sp4_sd = torch.load( model_path+ 'fold_4/'+model_name+'_speller.pth')
    
    list1 = Listener(**hparams)
    list1.load_state_dict(lt1_sd)
    list2 = Listener(**hparams)
    list2.load_state_dict(lt2_sd)
    list3 = Listener(**hparams)
    list3.load_state_dict(lt3_sd)
    list4 = Listener(**hparams)
    list4.load_state_dict(lt4_sd)
    
    sp1_sd = replace_name_keys(sp1_sd,'1')
    sp2_sd = replace_name_keys(sp2_sd,'2')
    sp3_sd = replace_name_keys(sp3_sd,'3')
    sp4_sd = replace_name_keys(sp4_sd,'4')
    
    sp = Ensemble_Speller(**hparams)
    combined_sp_dicts = collections.OrderedDict(list(sp1_sd.items())+list(sp2_sd.items())+list(sp3_sd.items())+list(sp4_sd.items()))
    sp.load_state_dict(combined_sp_dicts)
    
    optimizer = None
    iter_meter = IterMeter()
    best_cer = 100
        
    val_label_lens, val_input_lens, val_wp,val_wp_prob, val_wgt,val_f_pred, val_f_prob, val_f_gt, val_sub, val_act, val_rep, val_attn = test(list1,list2,list3,list4,sp,optimizer,hparams,dataloader,phase = 'test',tt=text_transform, bs=1, window_preds = True)
    pickling({'window_pred':val_wp,'window_prob':val_wp_prob,'window_gt':val_wgt,'f_pred':val_f_pred,'f_prob':val_f_prob,\
              'f_gt':val_f_gt,'window_sub':val_sub,'window_act':val_act,'window_rep':val_rep,'attention_scores':val_attn,\
              'input_lens':val_input_lens,'label_lens':val_label_lens},\
              model_path+model_name+'_test_ensemble_window_predictions_with_prob.p')
    
    end_time = time.time()
    print('Total time taken to perform prediction on the entire dataset is {} mins.'.format((end_time-start_time)/60))
    
    
    val_label_lens, val_input_lens, val_wp,val_wp_prob, val_wgt,val_f_pred, val_f_prob, val_f_gt, val_sub, val_act, val_rep, val_attn = test(list1,list2,list3,list4,sp,optimizer,hparams,dataloader,phase = 'severe',tt=text_transform, bs=1, window_preds = True)
    pickling({'window_pred':val_wp,'window_prob':val_wp_prob,'window_gt':val_wgt,'f_pred':val_f_pred,'f_prob':val_f_prob,\
              'f_gt':val_f_gt,'window_sub':val_sub,'window_act':val_act,'window_rep':val_rep,'attention_scores':val_attn,\
              'input_lens':val_input_lens,'label_lens':val_label_lens},\
              model_path+model_name+'_severe_ensemble_window_predictions_with_prob.p')
            
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


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default=None)
    
    args = parser.parse_args()

    model_name=args.model_name
    
    main(model_name)
    