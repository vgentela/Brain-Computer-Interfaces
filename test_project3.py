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
import mne as mne
import plot_topo as pt
eletrodes = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']

# #channel = 1

# #%% Load the data Processed Data
# # No need to use processed data, all the information is included in raw
 
# filename = f'DASPS_Database/Preprocessed data .mat/S0{subject}preprocessed.mat'  


#%% Load the RAW data 
# subject = 6  # Hamilton sores = severe,  Arousal = 1
# Hamilton = 'Severe'
# subject = 9  # Hamilton sores = light ,  Arousal = 8
# Hamilton = 'Light'
Hamilton = 'n/a'
subjects = ['06','09','13']
for plot,subject in enumerate(subjects ):

#filename = "DASPS_Database/Raw data.mat/S09.mat"   #DASPS_Database/Raw data.mat/S09.mat
    filename = f'DASPS_Database/Raw data.mat/S{subject}.mat'  #??? problem with leading zero 
    with h5py.File(filename, "r") as f:  #DASPS_Database/Raw data.mat/S09.mat
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

    print(f'Loaded Subject {subject}')
    # Visualize the data (Time and Frequency Domain)
    # loop through all 12 situations 6 runs x (recitation = recall)
    fs = 128
    #Time Domain
    sample_size = len(ds_arr[0,:,0]) # ??? assumes same size
    t = np.arange(0,sample_size/fs,1/fs)
    num_of_situations = 2
    for situation in range (num_of_situations):
        print(f"Situation {situation}")
        #situation = 0 # the first step in the protocol
        # fs = 128
        # #Time Domain
        # sample_size = len(ds_arr[subject,:,0])
        # t = np.arange(0,sample_size/fs,1/fs)
        #plt.figure(num = situation +4, clear=all)
        plt.figure()
        plt.suptitle(f"Raw Time Domain Subject {subject} Hamilton {Hamilton}")  #??? not loading this field correctly
        plt.title(f"Situation {situation}, Valance = {labels[0][situation]}, Arousal ={labels[1][situation]} ")
        plt.ylabel("Amplitude (uV ?)")
        plt.ylim([-60,60])
        plt.xlabel("Time (sec)")
        for channel in range(14):
            
            plt.plot(t,ds_arr[situation,:,channel],label = (f"Chan {channel}"))
        plt.legend(loc='upper left')
        #Freq Domain
        T = 15
        freq = np.arange(0,((fs/2)+1/fs),1/T)
        plt.figure()
        plt.suptitle(f"Raw Frequency Domain Subject {subject}  Hamilton {Hamilton}")
        plt.title(f"Situation {situation}, Valance = {labels[0][situation]}, Arousal ={labels[1][situation]} ")
        plt.ylabel("PSD (dB)")
        plt.ylim([0,65])
        plt.xlabel("Freq (Hz)")
        alpha_band=np.zeros(14)
        high_beta_band=np.zeros(14)        
        
        for channel in range(14):

            PSD = 10*np.log10((np.fft.rfft(ds_arr[situation,:,channel]))**2)
            plt.plot(freq,PSD,label = (f"Chan {channel}"))
            # Integrated and normalized
            alpha_band[channel] = np.sum(PSD[8*T:12*T])/(12*T-8*T)  # 8 to 12 Hz
            high_beta_band[channel] = np.sum(PSD[18*T:40*T])/(40*T-18*T) # 18 to 40 Hz
            print (f'{situation},{channel}')
            print(f'{PSD[0]}')
        plt.legend(loc='upper right')
        plt.figure()
        plt.plot(alpha_band,label = ("Alpha Band"))
        plt.xlabel("Channel")
        plt.plot(high_beta_band,label = ("Beta Band"))
        plt.ylim([30,65])
        plt.title (f'Power in the Bands Situation {situation} Subject {subject}')
        plt.legend(loc='upper right')
       
#%% Topo Maps
# Plot time average for a giving subject, situation

situation_mean = np.mean (ds_arr, axis = 0)  # This is the average across the 12 situations
sample_mean = np.mean (ds_arr, axis = 1)  # This is the average across samples [12 situations x 14 channels]
plt.figure(num = 21, clear =all)

#

situation = 6   # at least for subject 6 the scalp plots were asymetrical and simular for each situation
#pt.plot_topo(channel_names=eletrodes, channel_data=ds_arr[situation,128,:],title=f'Subject {subject}, Situation {situation}',cbar_label='Voltage (uV)',montage_name='biosemi64')


pt.plot_topo(channel_names=eletrodes, channel_data=sample_mean[situation,:],title=f'Subject {subject}, Situation {situation}',cbar_label='Voltage (uV)',montage_name='biosemi64')



