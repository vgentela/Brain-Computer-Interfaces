"""
Name:        test_project3.py
Description: Invokes the necessary routines to perform the requirements of BCI Project 3
"""
#%%  Import Modules, Load Data, and extract Epochs 
# Import necessary modules

import import_ssvep_data
import loadmat
import numpy as np
import h5py
import matplotlib.pyplot as plt
import eeglib.eeg as eeg
import seaborn as sns
import pandas as pd
#%%
def load_data(subject,directory = f'DASPS_Database/Raw data.mat/'):
    file_path = f'{directory}S{str(subject).rjust(2,"0") if subject < 10 else subject}.mat'
    container = h5py.File(file_path)
    data,ham_scores,labels,situations = [container.get(i) for i in container.keys()]
    return data,ham_scores,labels,situations
#%%
data,ham_scores,labels,situations = load_data(1)

#%%
def trials(data):
    
    trial_dic = {}
    
    for _ in range(len(data)):
        trial_dic[f'trial_{_}'] = np.vstack(data[_])
    
    return trial_dic
    
#%% hjorth activity:
def hjorth_params(data_dic):
    params = {}
    param_list = [eeg.hjorthActivity,eeg.hjorthComplexity,eeg.hjorthMobility]
    for _ in range(len(data_dic)):
        params[f'params_{_}'] = np.concatenate([np.apply_along_axis(i,0,data_dic.get(f'trial_{_}')) for i in param_list]).T
    return params
#%%
trial_dic = trials(data)
params = hjorth_params(trial_dic)
#%%
df = pd.DataFrame(np.array(list(params.values())),index = params.keys())
#%%
df['valence'] = labels[0]
df['arousal'] = labels[1]
#%%
corr = df.corr()

sns.heatmap(corr)
plt.savefig('correlation_heatmap')
