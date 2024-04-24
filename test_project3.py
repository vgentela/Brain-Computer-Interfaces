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
#%%
def load_data(subject,directory = f'DASPS_Database/Raw data.mat/'):
    file_path = f'{directory}S{str(subject).rjust(2,"0") if subject < 10 else subject}.mat'
    container = h5py.File(file_path)
    data,ham_scores,labels,situations = [container.get(i) for i in container.keys()]
    return data,ham_scores,labels,situations
#%%
data,ham_scores,labels,situations = load_data(1)

#%% hjorth activity:
def hjorth_params(data):
    data = np.vstack([data[i][j:j+1] for j in range(len(data[0]))for i in range(len(data))])
    #print(data)
    param_list = [eeg.hjorthActivity,eeg.hjorthComplexity,eeg.hjorthMobility]
    params = np.array([[i(data.T[j]) for j in range(len(data.T))]for i in param_list])
    return params.flatten()
#%%
params = hjorth_params(data)
#%%
len(params )






