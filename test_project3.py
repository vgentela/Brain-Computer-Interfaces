"""
Name:        test_project3.py
Revision:   1-May-2024
Description: Invokes the necessary routines to perform the requirements of BCI Project 3


#TODO list functions called, and provide an explaination as to why 

Authors:
    Luke
    Varsney
    Jim.
    
#TODO add references
"""
#%%  Import Modules, Load Data, and extract Epochs 
# Import necessary modules
import import_ssvep_data
import loadmat
import numpy as np
#c:\users\18023\anaconda3\lib\site-packages
import h5py
#import seaborn as sb
import matplotlib.pyplot as plt
import mne as mne
#C:\Users\18023\Documents\GitHub\BCI-Project-3
import plot_topo as pt
from project_3 import *
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
#%% Load edf file
#C:\Users\18023\Documents\GitHub\BCI-Project-3\DASPS_Database\Raw data .edf
subjects = ['06']
directory = 'DASPS_Database/Raw data .edf/'

raw_data_edf,channels_edf,info = p3.loadedf(directory,subjects)
#%%
len(raw_data_edf[0])
#%% Visualize Data Scalp Map 
#This has only been tested with data from edf format files.
#TODO for each paradigm there are 6 situations and 2 phases, consider using a for loop 
# situation = 6   # at least for subject 6 the scalp plots were asymetrical and simular for each situation

method = 'mean'
domain = 'time'
data_type = '.edf'
run = 1  
subject = '06'
# Visualize topographical maps for various electrode selections and mapping to the data 
#TODO case statement or for loop ?
#TODO subplots
# Case 1 baseline 
title=f'Case 1: Baseline; Subject {subject}, Run{run}'
fig_num = 20
data = raw_data_edf[2:16,:]     #  Electrode channels start at index 2, end at index 16 for a total of 14 electrodes
electrodes =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']  # based upon the edf file and ref [Asma Baghdadi]
p3.plot_scalp_map ( subjects, electrodes, data, title, fig_num, data_type = data_type, run = run ,method = method, domain = domain)
plt.savefig(f'Topo_plot_sub_{subject}_run{run}_case_1.png')  
#Case 2 ref based upon [Farah Muhammand]
title=f'Case 2:  [Farah Muhammand]; Subject {subject}, Run{run}'
fig_num = 21
data = raw_data_edf [[2,3,6,7,12,13],:]
electrodes = ['AF3','AF4', 'FC5','FC6','P7','P8'] # based upon [Farah Muhammand] 
p3.plot_scalp_map ( subjects, electrodes, data, title, fig_num, data_type = data_type, run = run ,method = method, domain = domain)
plt.savefig(f'Topo_plot_sub_{subject}_run{run}_case_2.png') 
# Case 3 Expected Brain Region Respons
title=f'Case 3: Expected Brain Region Response; Subject {subject}, Run{run}'
fig_num = 22
data = raw_data_edf[2:16,:] 
electrodes =['AF3', 'O1','F7','P7', 'F3','T7', 'FC5','O2','AF4', 'T8', 'F8','P8','F4','FC6'  ]  # Re-order of channels to eliminate asymetrical topo plot
p3.plot_scalp_map ( subjects, electrodes, data, title, fig_num, data_type = data_type, run = run ,method = method, domain = domain)
plt.savefig(f'Topo_plot_sub_{subject}_run{run}_case_3.png') 
# Case 4 
title=f'Case 4: [Asma Baghdadi] and Alpha PSD; Subject {subject}, Run{run}'
fig_num = 23
data = raw_data_edf [[2,3,5,8,9],:]  # [Asma Baghdadi] mapping
electrodes =['AF3','AF4','T8', 'O2' , 'P8' ]  #  PSD 
p3.plot_scalp_map ( subjects, electrodes, data, title, fig_num, data_type = data_type, run = run ,method = method, domain = domain)
plt.savefig(f'Topo_plot_sub_{subject}_run{run}_case_4.png') 
# Case 5 
title=f'Case 5: [Asma Baghdadi] and Expected ; Subject {subject}, Run{run}'
fig_num = 24
data = raw_data_edf [[2,10,11,9,13],:]  # Expected mapping
electrodes =['AF3','AF4','T8', 'O2' , 'P8' ]  # PSD 
p3.plot_scalp_map ( subjects, electrodes, data, title, fig_num, data_type = data_type, run = run ,method = method, domain = domain)
plt.savefig(f'Topo_plot_sub_{subject}_run{run}_case_5.png') 


#%% Visualize the edf dataset in time and frequency domain

# raw_data_edf from loadedf

p3.plot_edf_data (raw_data_edf,electrode_index = (2,16), subjects=6, run=1,fs_edf=128)

#%% Load .mat file dataset associated with the processed raw data, bandpass filter and artifact removal using ICA
#subjects = ['01','02','04','05','09','13','15','18','20','22',]# vallence = 1 Arousal = 8
#subjects = ['06','09','01','04','05','07','10','11','12','13','14','17','18','19','20','21','02','03','08','15','16','22','23']
#subjects = ['06']
directory = Path(f'{Path.cwd()}//DASPS_Database/Raw data.mat/')
assert directory.exists()
#electrodes =['AF3', 'O1','F7','P7', 'F3','T7','FC5','O2','AF4','T8', 'F8','P8','F4','FC6'  ]
#electrodes =['AF4', 'T8','FC6'  ]
#TODO ds_arr is associated with the last subject in subjects, update for .mat fils
#ds_arr, count, label_array = p3.load_data_epoch_anxiety_levels(directory,subjects,electrodes) # changed what is returned need to resolve ds_array

#%% Visualize the mat data in time domain
#TODO defer task, not much info.

#%%  Label the anxiety levels of the trials

#subjects = [06','09','01','04','05','07','10','11','12','13','14','17','18','19','20','21','02','03','08','15','16','22','23']
#lectrodes =['AF3','AF4', 'FC5','FC6','P7','P8']
#%%
subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
df= load_data_epoch_anxiety_levels(directory ,subjects)
#%%
train_data,test_data= transformations(df,'autoencoder',[8/10,2/10])
#%%
x_train,y_train,x_test,y_test = transformations(df, 'randomforest',test_size=0.2)
#%%

y_train
#%%
#optimizer = optim.Adam
vae = train([64,44],30,'cpu',train_data,0.0001,epochs=1200)
#%%
encoder,tl,el = vae.training()
#%%
clf =  latent_training(encoder,30, [25,10],4, train_data,'cpu')
#%%
classifier,accs,epoch_loss = clf.train()
#print(len(se['eeg'][0][0][0][~np.isnan(se['eeg'][0][0][0])]))
# TODO numbers do not match 
# expected 
#Severe Anxiety 90.0, Moderate Anxiety 10.0, Light Anxiety 20.0, Normal Anciety 2.0, No Anxiety no report Reported Normal as 156
# actual 
#Severe Anxiety 90.0, Moderate Anxiety 30.0, Light Anxiety 0.0, Normal Anxiety 58, No Anxiety 98.0  Total = 276 
#TODO looks like they combined Light Anxiety + Normal anxiety + Normal Low Arousal High Valence = 156?
#TODO Why the descrepency with moderate?
#%%
n_estimators =[300,500,700]
criterion = "log_loss"
max_depth = [5,10,15]
class_weight = ['balanced']
scoring = 'accuracy'
rf = randomforest(n_estimators, criterion, max_depth,scoring)
#%%
model,best_params,accuracy = rf.r_forest(x_train, y_train, x_test, y_test)
#%%
best_rf_model,test_accuracy = rf.latent_forest(encoder, model, x_train, y_train, x_test, y_test)
#%%
svc= SVClassifier(encoder)

#%% Create a confusion matrix based upon the above
svc.train(x_train, y_train,x_test,y_test)
#%%
svc.latent_train(x_train,y_train)

#%%
accuracy, confusion_mat, precision, sensitivity= svc.latent_test(x_test, y_test)

#%% Time permitting investigate bootstrapping to identify p-values...
svc.plot_metrics(confusion_mat, precision, sensitivity) 
