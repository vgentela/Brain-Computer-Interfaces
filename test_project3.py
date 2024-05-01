"""
Name:        test_project3.py
Revision:   1-May-2024
Description: Invokes the necessary routines to perform the requirements of BCI Project 3


#TODO list functions called, and provide an explaination as to why 

Authors:
    Luke
    Varsney
    Jim.
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
import plot_topo as pt
import project_3 as p3
#%% Load edf file
#C:\Users\18023\Documents\GitHub\BCI-Project-3\DASPS_Database\Raw data .edf
subjects = ['06']
directory = 'DASPS_Database/Raw data .edf/'

raw_data_edf,channels_edf,info = p3.loadedf(directory,subjects)

#%% Visualize Data Scalp Map 
#This has only been tested with data from edf format files.

#electrodes = [ 'F3']  # index = 2
#electrodes = ['AF3', 'AF4', 'F3', 'F4']
##electrodes = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']  
#electrodes =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']  # based upon the edf file
#electrodes =['AF3', 'O1','F7','P7', 'F3', 'FC5','T7', 'T8', 'FC6', 'F4','P8', 'F8', 'O2','AF4']  # jim's guess
electrodes =['AF3', 'O1','F7','P7', 'F3','T7', 'FC5','O2','AF4', 'T8', 'F8','P8','F4','FC6'  ]  # jim's guess #2 this matches expectations the best
#electrodes =['AF3','F7', 'P7', 'F3','T7', 'FC5','AF4', 'T8', 'F8','P8','F4','FC6' ,'O1','O2' ]  # jim's guess #3

method = 'mean'
domain = 'time'
data_type = '.edf'

#TODO only tested for .edf files
# data depends on the type of file
# for edf 
data = raw_data_edf

#TODO this has not yet been tested 
# # for mat
# data = ds_arr

#TODO for each paradigm there are 6 situations and 2 phases 
run = 1

p3.plot_scalp_map ( subjects, electrodes, data, data_type = data_type, run =run ,method = method, domain = domain)

#situation = 6   # at least for subject 6 the scalp plots were asymetrical and simular for each situation
#pt.plot_topo(channel_names=eletrodes, channel_data=ds_arr[situation,128,:],title=f'Subject {subject}, Situation {situation}',cbar_label='Voltage (uV)',montage_name='biosemi64')

#%% Visualize the edf dataset in time and frequency domain

fs_edf = 128  #TODO README file, I can't think of a way to infer this onfo from the data set

# raw_data_edf from loadedf

p3.plot_edf_data (raw_data_edf,electrode_index = (2,16),fs_edf=128)


#%% Load .mat file dataset associated with the processed raw data, bandpass filter and artifact removal using ICA

subjects = ['06','09','13','01','04','05','07','10','11','12','13','14','17','18','19','20','21','02','03','08','15','16','22','23']
directory = 'DASPS_Database/Raw data.mat/'
electrodes =['AF3', 'O1','F7','P7', 'F3','T7', 'FC5','O2','AF4', 'T8', 'F8','P8','F4','FC6'  ]
#TODO ds_arr is associated with the last subject in subjects
ds_arr, count, label_array = p3.load_data_epoch_anxiety_levels(directory,subjects,electrodes)


#%% Visualize the mat data in time domain
#TODO defer task, not much info.

#%%  Visualize the anxiety levels

#TODO plot the anxiety levels on the x-axis and the mean PSD on the y-axis

#TODO this is not working yetplot_PSD_by_anxiety (subject, electrodes, anxiet_level, PSD_band ,run = 1)
    
#%% Feature Extraction 
# Looks like some electrodes show alpha band and beta band power need to complete the visualization of anxiety to determine wht 
# the best statistic would be

# Based upon ref [] power between 0-20 Hz
# Or it could be the diff in alpha band and beta band

#%% Perform classification based on statistic in Feature Extraction
# Need to identify a threshold

#%% Create a confusion matrix based upon the above

#%% Calculate statisticsw, accuracy, sensitivity and specificity.

#%% Time permitting investigate bootstrapping to identify p-values...
 
