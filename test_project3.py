"""
Name:        test_project3.py
Description: Invokes the necessary routines to perform the requirements of BCI Project 3
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


subject = 0 
channel = 1

#%% Load the data Processed Data
filename = "S01preprocessed.mat"


with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array 

#Processed data has only one key data ???

    # labels = list(h5py.File(filename, "r")['labels'])
    # hamilton = list(h5py.File(filename, "r")['hamilton'])
    # situation = list(h5py.File(filename, "r")['situation'])

#%% Visualize the data (Tiome and Frequency Domain)
subject = 0 
channel = 1
fs = 128
#Time Domain
sample_size = len(ds_arr[subject,:,channel])
t = np.arange(0,sample_size/fs,1/fs)
plt.figure(num = 1, clear=all)
plt.title(f"Processed Time Domain Subject {subject}")
plt.ylabel("Amplitude (uV ?")
plt.xlabel("Time (sec)")
for channel in range(14):
    print(f"channel {channel}")
    plt.plot(t,ds_arr[subject,:,channel],label = (f"Chan {channel}"))
plt.legend(loc='upper left')
#Freq Domain
freq = np.arange(0,((fs/2)+1/fs),1/15)
PSD = 10*np.log10(np.fft.rfft(ds_arr[subject,:,channel])**2)

plt.figure(num = 2, clear=all)
plt.title(f"Processed Frequency Domain Subject {subject}")
plt.ylabel("PSD (dB)")
plt.xlabel("Freq (Hz)")
for channel in range(14):
    print(f"channel {channel}")
    plt.plot(freq,PSD,label = (f"Chan {channel}"))
plt.legend(loc='upper right')





#%% Load the RAW data 
filename = "S01.mat"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array 

    labels = list(h5py.File(filename, "r")['labels'])
    hamilton = list(h5py.File(filename, "r")['hamilton'])
    situation = list(h5py.File(filename, "r")['situation'])


#%% Visualize the data (Time and Frequency Domain)


subject = 0 
channel = 1
fs = 128
#Time Domain
sample_size = len(ds_arr[subject,:,channel])
t = np.arange(0,sample_size/fs,1/fs)
plt.figure(num = 3, clear=all)
plt.title(f"Raw Time Domain Subject {subject}")
plt.ylabel("Amplitude (uV ?")
plt.xlabel("Time (sec)")
for channel in range(14):
    print(f"channel {channel}")
    plt.plot(t,ds_arr[subject,:,channel],label = (f"Chan {channel}"))
plt.legend(loc='upper left')
#Freq Domain
freq = np.arange(0,((fs/2)+1/fs),1/15)
PSD = 10*np.log10(np.fft.rfft(ds_arr[subject,:,channel])**2)

plt.figure(num = 4, clear=all)
plt.title(f"Raw Frequency Domain Subject {subject}")
plt.ylabel("PSD (dB)")
plt.xlabel("Freq (Hz)")
for channel in range(14):
    print(f"channel {channel}")
    plt.plot(freq,PSD,label = (f"Chan {channel}"))
plt.legend(loc='upper right')








