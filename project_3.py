"""
Name:       project3.py
Revision:   1-May-2024
Description: Functions necessary perform the requirements of BCI Project 3
1) Load Data.  Data set is in two formats .mat and .edf
    a) Load data set from .mat file.  The mat file format requires the use of h5py
    b) Load data set from .mat file. 
2) Epoch the data.  For each paradigm assessments were made as to the level of anxiety
3) Visualize the data
    a) Time domain
    b) Frequency domain
    c) Scalp topographical maps
Authors:
    Luke
    Varsney
    Jim
"""
#%%  Import Modules

import import_ssvep_data
import loadmat
import numpy as np
#TODO #c:\users\18023\anaconda3\lib\site-packages
import h5py
#import seaborn as sb
import matplotlib.pyplot as plt
import mne as mne
import plot_topo as pt
import pandas as pd
from torcheeg import transforms as tfs
import torch
from torch.utils.data import TensorDataset,DataLoader,random_split
from sklearn.model_selection import train_test_split
from torch import nn
#%% Load .edf file
#TODO ensure that the dataset is in the correct directory. C:\Users\18023\Documents\GitHub\BCI-Project-3\DASPS_Database\Raw data .edf
#file = "S06.edf"

def loadedf(directory, subjects):
    '''
   
    Parameters
    ----------
    file : TYPE, optional
        DESCRIPTION. The default is file.

    Returns
    -------
    raw_data_edf : TYPE
        DESCRIPTION.
    channels_edf : TYPE
        DESCRIPTION.
    info : TYPE
        DESCRIPTION.

    '''
   
    subject = subjects
    for plot,subject in enumerate(subjects):
    
    
        filename = f'{directory}S{subject}.edf'
        data_edf = mne.io.read_raw_edf(filename)
        raw_data_edf = data_edf.get_data()
        channels_edf = data_edf.ch_names
    #TODO format 
            # ['COUNTER', 'INTERPOLATED',
            # 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4',
            # 'RAW_CQ', 
            # 'CQ_AF3', 'CQ_F7', 'CQ_F3', 'CQ_FC5', 'CQ_T7', 'CQ_P7', 'CQ_O1', 'CQ_O2', 'CQ_P8', 'CQ_T8', 'CQ_FC6', 'CQ_F4', 'CQ_F8', 'CQ_AF4',
            # 'CQ_CMS', 'CQ_DRL', 'GYROX', 'GYROY', 'MARKER']
    #TODO README indicates that the electrode channels are:
    #electrodes = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2'] 
    # you can get the metadata included in the file and a list of all channels:
        info = data_edf.info
    
    return  raw_data_edf,channels_edf,info


#%% Visualize the data

def plot_edf_data(raw_data_edf,electrode_index = (2,16),subjects=1 ,run=1,fs_edf=128):
    '''

    Parameters
    ----------
    raw_data_edf : TYPE
        DESCRIPTION.
    electrode_index : TYPE, optional
        DESCRIPTION. The default is (2,16).
    fs_edf : TYPE, optional
        DESCRIPTION. The default is 128.

    Returns
    -------
    None.

    '''

    plt.figure(num = 200,figsize=(8,6),clear = all)
    for index in range(2,16):  # raw data valid eed electrode channels index 2 - 15
        print(index)
        time_edf = np.arange(0,len(raw_data_edf[0,:])/fs_edf,1/fs_edf)
        plt.plot(time_edf,(raw_data_edf[index,:] - np.mean(raw_data_edf[index,:])),label = (f"Chan {index -1}"))

        T_edf = (len(raw_data_edf[0,:])-3)/fs_edf          #TODO sample interval
        freq_edf = np.arange(0,((fs_edf/2)+1/fs_edf),1/T_edf)
        #raw_data_edfm [36 channels x samples ]
    plt.suptitle(f'Subject {subjects} Run {run}')
    plt.title('Time Domain (.edf)')
    plt.ylabel("Amplitude (uV)")
    #plt.ylim([])
    plt.xlabel("Time (sec)")
    plt.legend()
    plt.tight_layout()
     #Save figure 
    plt.savefig('Time_Domain_edf.png')          
      # ....then show  
    plt.show()
    plt.figure(num=300,figsize=(8,6),clear = all)
    for index in range(2,16):  # raw data valid eed electrode channels index 2 - 15
        PSD_edf = np.real(10*np.log10((np.fft.rfft(raw_data_edf[index,:]-np.mean(raw_data_edf[index,:])))**2)) # imaginary part = 0, extract real to avoid warnings
        plt.plot(freq_edf,PSD_edf, label = (f"Chan {index-1}"))
    plt.suptitle(f'Subject {subjects} Run {run}')
    plt.title('Frequency Domain (.edf)')
    plt.ylabel("PSD (dB)")
    #plt.ylim([-150,0])
    plt.xlabel("Freq (Hz)")
    plt.legend()
    plt.tight_layout()
    #Save figure 
    plt.savefig('Freq_Domain_edf.png')          #TODO Light Severe
     # ....then show  
    plt.show()
    
    
    
    return

#%% Visualize the data Scalp topo maps

def plot_scalp_map ( subject, electrodes, data, title, fig_num, data_type = '.edf', run = 1,method = 'mean', domain = 'time'):
    '''
    

    Parameters
    ----------
    subject : TYPE
        DESCRIPTION.
    electrodes : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    data_type : TYPE, optional
        DESCRIPTION. The default is '.edf'.
    run : TYPE, optional
        DESCRIPTION. #TODO  paradigm per subject consists of 6 situations composed of a recital phase followed
        by a recall phase. In this data set run is a number from 1 to (6 x 2), i.e 1 -12.  The default is 1.
        This paramater is used for labeling of the plot.
    method : TYPE, optional
        DESCRIPTION. The default is 'mean'.
    domain : TYPE, optional
        DESCRIPTION. The default is 'time'.

    Returns
    -------
    None.

    '''

# Plot time average for a giving subject, situation
    plt.figure(num = fig_num, clear =all)
    if method == 'mean':
        if data_type == '.mat':
            #situation_mean = np.mean (ds_arr, axis = 0)  # This is the average across the 12 situations
            sample_mean = np.mean (data, axis = 1)  # This is the average across samples [12 situations x 14 channels]
            #TODO # channel_data must match selected electrodes.  In this case they all channels 1-14
            pt.plot_topo(channel_names=electrodes, channel_data=sample_mean[run,0:14],title=title,cbar_label='Voltage (uV)',montage_name='biosemi64')
        elif data_type == '.edf':
             #raw_data_edf[index for the first electrode:index for the last electrode, EEG data]
             #TODO this only works for all electrodes
             #TODO (raw_data_edf[Index of electrodes start at 2 end at 16,data]
             #TODO raw_data_mean = np.mean(raw_data_edf[2:16,:],axis = 1) 
             
             
             #TODO correct for the data passed , don't make assumptions here corret by passing the right data from the test module
             #raw_data_mean = np.mean(data[2:16,:],axis = 1)  # The .edf has a different mapping Electrodes start on index 2,  Note ther is also gyroscope data present 
             raw_data_mean = np.mean(data[:,:],axis = 1)*1000
             #TODO voltage scaling , for now assume need to multiply by 1000
             pt.plot_topo(channel_names=electrodes, channel_data=raw_data_mean,title=title,cbar_label='Voltage (uV)',montage_name='biosemi64')
    plt.tight_layout()
    #Save figure 
    plt.savefig('Scalp_Map.png')          #TODO Light Severe
     # ....then show  
    plt.show()
    return 


#%% House Keeping Functions

#%% Clear figures
def clear_all_figures():
    '''
    Get a list of figure numbers currently active in the matplotlib session, then close these figures

    Returns
    -------
    None.

    '''
    fig = plt.get_fignums()
    for index,fig_num in enumerate (fig):
        plt.close(fig_num)
#%%
def labelling(data,labels):
    
    data = np.vstack(data[:])
    labels = labels.T
    
    df = pd.DataFrame(data)
    label_indices = np.where(df.index % 1920 == 0)[0]
    
    severe_count = 0
    moderate_count = 0
    light_count = 0
    normal_count = 0
    
    for idx,index in enumerate(label_indices):
        #print(index)
        df.at[index,'valence']  = labels[idx][0]
        df.at[index,'arousal'] = labels[idx][1]
        df.at[index,'trial'] = f'trial_{idx}'
        val,aro= (labels[idx][0],labels[idx][1])
        
        if val<5 and aro>5:
            if 0<val<=2 and 7<=aro<=9:
                df.at[index,'Anxiety_level'] = 'severe'
                severe_count+= 1
            elif 2<=val<=4 and 6<=aro<=7:
                df.at[index,'Anxiety_level'] = 'moderate'
                moderate_count+=1
            elif 4<=val<5 and 5<aro<=6:
                df.at[index,'Anxiety_level'] = 'light'
                light_count+= 1
        else:
            df.at[index,'Anxiety_level'] = 'normal'
            normal_count += 1
       
        
            
       
    #df = df.replace(np.nan,'')
    df.set_index('trial',inplace= True)
    
    
    return df,(severe_count,moderate_count,light_count,normal_count)
#%%
def transformations(df,model):
    
    key= next(key for key in df.keys())
    bpower = tfs.BandPowerSpectralDensity(128,band_dict={'alpha': [8, 14], 'beta': [14, 31], 'gamma': [31, 49]})
    de = tfs.BandDifferentialEntropy(128,band_dict={'alpha': [8, 14], 'beta': [14, 31], 'gamma': [31, 49]})

      
    eeg_data = df[f'{key}'].drop(['valence','arousal','Anxiety_level'],axis=1)
    
    normalized_eeg = eeg_data.apply(lambda x: x-np.mean(x),axis =1).to_numpy()
    
    normalized_eeg = normalized_eeg.reshape(12,1920,14)
    
    anxiety_degree = df[f'{key}']['Anxiety_level'][~pd.isna(df[f'{key}']['Anxiety_level'])]

    labels = pd.get_dummies(anxiety_degree).to_numpy()

    labels = labels.reshape(12,labels.shape[1])
    
    trial_band_powers=[]
    trial_entropys =[]
    
    for trial in normalized_eeg:
        
        powers =bpower(eeg=trial.T)
        band_powers = powers['eeg']
        
        
        differential_entropy = de(eeg=trial.T)    
        band_entropys = differential_entropy['eeg']
        
        trial_band_powers.append(band_powers)
        trial_entropys.append(band_entropys)
        
        
    if model == 'autoencoder':
        
        band_tensors = torch.tensor(trial_band_powers)
        entropy_tensors = torch.tensor(trial_entropys)
        
        
        features = torch.transpose(torch.cat((band_tensors,entropy_tensors),dim=2),1,2)
        labels = torch.tensor(labels)
        #print(features.shape,labels.shape)

        features= torch.stack([feature for feature in features])
        labels = torch.stack([label for label in labels])
        #print(features.shape,labels.shape)

        
        dataset= TensorDataset(features,labels)
        
        train_data, test_data = random_split(dataset,[7/10,3/10])
        
        return train_data,test_data
        
    if model =='randomforest':
        band_arrays = np.asarray(trial_band_powers)
        entropy_arrays = np.asarray(trial_entropys)
        
        features = np.transpose(np.concatenate((band_arrays,entropy_arrays),axis=2),(0,2,1))
        print(len(features),len(labels))
        
        train_data, test_data, train_labels, test_labels = train_test_split(features,labels,test_size=0.3)
        
        return train_data, train_labels, test_data, test_labels
#%%
class Classifier(nn.Module):
    
    def __init__(self,feature_nodes,hidden_nodes):
        super(Classifier,self).__init__()
        
        self.encoder
        
#%% Load preprocessed data.  This is the raw data contained in the .edf files after bandpass filtering and application of ICA

def load_data_epoch_anxiety_levels(directory ,subjects ,electrodes):
   '''
   Assumption: data from the .mat files
   Description:  Draft
      
   Two methods of getting at the eeg data contained in the dataset: list, and nimpy array.
   The numpy array is perfrered method
   #TODO do I have the correct interpretation from our readme file , and the excel file?

   ds_arr; size [trials x samples x electrodes]; This is the processed 15 second eeg data from the 12 trials (6 situation* 2 runs) and 23 subjects
   Processed by ICA artifact removal and bandpass filtered. 

   labels: Two columns for the subject Self-Assessment Manikin (SAM). One column is an event's positive or negative score for valence, 
   and the other is the arousal spectrum, from calmness to excitement. A combination of these two scores establishes anxiety levels. 
   #After transpose of the loaded row infromation 
    
    Parameters
    ----------
    directory : TYPE, 
        DESCRIPTION. 
    subjects : TYPE, 
        DESCRIPTION. 

    Returns
    -------
    ds_arr : TYPE
        DESCRIPTION.
    count : TYPE
        DESCRIPTION.
    label_array : TYPE
        DESCRIPTION.

    '''
    # count the anxiety levels 
   #anxiety_levels = ['Severe','Moderate','Light','Normal']
   
    # The intention of this code is to replicate the labeling flow chart of Fig 5 ref [Asma Baghdadi]
    #TODO clean up references
    
   subjects_df = {}
   for index,subject in enumerate(subjects):
        
    
    #filename = "DASPS_Database/Raw data.mat/S09.mat"   #DASPS_Database/Raw data.mat/S09.mat
        #filename = f'DASPS_Database/Raw data.mat/S{subject}.mat'   
        filename = f'{directory}S{subject}.mat'
        with h5py.File(filename, "r") as f:  #DASPS_Database/Raw data.mat/S09.mat
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
            #print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
            a_group_key = list(f.keys())[0]
    
        # get the object type for a_group_key: usually group or dataset
            #print(type(f[a_group_key])) 
    
        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
            #data = list(f[a_group_key])
            data = f['data'][:]
            
  
        # preferred methods to get dataset values:
            ds_obj = f[a_group_key]      # returns as a h5py dataset object
            ds_arr = f[a_group_key][()]  # returns as a numpy array 
    
            #labels = list(h5py.File(filename, "r")['labels'])
            
            labels= f['labels'][:]
            
            df,count_tuple = labelling(data,labels)
            
            if f'{subject}' in subjects_df.keys():
                subjects_df[f'subject'].append(df)
            else:
                subjects_df[f'{subject}'] = df
            #TODO get into array format
            """label_array = np.zeros((4,12)) # [Valence + Arousal + severe_count, + moderate_count + light_count + normal_ count x trials]; [6x12]
            # Load the valance and arousal data
            #label_array[0:2,:] = np.array(labels)[0:2,:]   # row[0] = Valence, row[1]=Arousal
        
            # sort anxiety levels
            # Total count per subject = 6 situations x 2 phases X number of subjects 
            
            # Start with Anxiety
            
            # Severe
            label_array[2] = np.where((label_array[0,:] <= 2) & (label_array[1,:] >= 7) & (label_array[1,:] <= 9), 1, 0)        # severe
            severe_count = severe_count + np.sum(label_array[2])         
            
            # Moderate
            label_array[3]=np.where((label_array[0,:] >= 2) & (label_array[0,:] <= 4) & (label_array[1,:] >= 6) & (label_array[1,:] <= 7), 1, 0) #moderate
            moderate_count = moderate_count + np.sum(label_array[3])  
            
            # Light
            label_array[4]=np.where((label_array[0,:] >= 4) & (label_array[0,:] <= 5) & (label_array[1,:] >= 5) & (label_array[1,:] <= 6), 1, 0) #light
            light_count = light_count + np.sum(label_array[4])  
            
            # Normal anxiety
            label_array[5]=np.where((label_array[0,:] >= 5) & (label_array[0,:] <= 8) & (label_array[1,:] > 4) & (label_array[1,:] < 7), 1, 0) #normal
            #label_array[5]=np.where((label_array[0,:] >= 5) & (label_array[1,:] <= 5) & (label_array[1,:] <= 5), 1, 0) #normal
            #label_array[5]=np.where((label_array[0,:] > 5) & (label_array[1,:] > 5)) 
            normal_count = normal_count + np.sum(label_array[5])  
            
            
            hamilton = list(h5py.File(filename, "r")['hamilton'])  # The pre and post Hamilton socres as evaluated by therapist 
            situation = list(h5py.File(filename, "r")['situation']) #This is the number of situations per paridigm
            
            # make into an array to facilitate the display of information
            count = [severe_count,moderate_count,light_count,normal_count]
            
            # Whats left is Normal or no anxiety observed in the subject.
            no_anxiety_count = len(subjects)* 12 - np.sum(count)
            
            print(f'Loaded Subject {subject}')      # Provide user feedback
            
            #TODO temporary plotting at this stage
            #plot_PSD (index,electrodes,ds_arr,level = 1,freq_band = [4,20],run = 1, sample_interval=15,fs =128)

            if np.sum(label_array[2]) >0:  #TODO for now plot the PSD if any values are servere
                plot_PSD (index,electrodes,ds_arr, level = 1,freq_band = [4,20],run = next((index for index,value in enumerate(label_array[2]) if value != 0), None), sample_interval=15,fs =128)"""
            severe_count,moderate_count,light_count,normal_count = count_tuple   
            print('severe count:',severe_count)
            print('moderate_count:',moderate_count)
            print('light_count:',light_count)
            print('normal_count:',normal_count)
    
   #TODO not properly dealing with subject 
  # label the data base
   
   # add an axis to ds_arr that contains the anxiety level
    
   #ds_arr_anxiety = np.expand_dims(ds_arr,axis = 0)  TODO not the right way to preceed
   return subjects_df
   
 #%% Visualize the data Time Domain #TODO Not much value in viewing the time data 

#     # loop through all 12 situations 6 runs x (recitation = recall)
#     fs = 128
#     #Time Domain
#     sample_size = len(ds_arr[0,:,0]) # TODO assumes same size for all trials
    
    
#     t = np.arange(0,sample_size/fs,1/fs)
#     num_of_trials = 12                              # trials = 2 x situations
#     # plt.figure()
#     for trial in range (num_of_trials):
#         # print(f"Situation {1+(trial//2)}")              # Convert trials to situations to match excel
#         # #situation = 0 # the first step in the protocol
#         # # fs = 128
#         # # #Time Domain
#         # # sample_size = len(ds_arr[subject,:,0])
#         # # t = np.arange(0,sample_size/fs,1/fs)
#         # #plt.figure(num = situation +4, clear=all)
#         # plt.figure()
#         # plt.suptitle(f"Raw Time Domain Subject {subject} Hamilton {Hamilton}")  #??? not loading this field correctly
#         # plt.title(f"Situation {1+(trial//2)}, Valance = {labels[0][trial]}, Arousal ={labels[1][trial]} ")
#         # plt.ylabel("Amplitude (uV)")    #TODO is the data in uV?
#         # plt.ylim([-60,60])
#         # plt.xlabel("Time (sec)")
#         # for channel in range(14):
            
#         #     plt.plot(t,ds_arr[trial,:,channel],label = (f"Chan {channel}"))
#         # plt.legend(loc='upper left')
#         #Freq Domain
#         T = 2          #TODO sample interval
#         freq = np.arange(0,((fs/2)+1/fs),1/T)
#         #plt.figure()
#         # plt.suptitle(f"Raw Frequency Domain Subject {subject}  Hamilton {Hamilton}")
#         # plt.title(f"Situation {1+(trial//2)}, Valance = {labels[0][trial//2]}, Arousal ={labels[1][trial//2]} ")
#         # plt.ylabel("PSD (dB)")
#         # plt.ylim([0,65])
#         # plt.xlabel("Freq (Hz)")
#         # Pre allocated the PSD arrays
 
def plot_PSD (subject,electrodes,data, level,freq_band = [4,20], run = 1, sample_interval=15, fs =128):
    '''
    Visualize the data Frequency Domain.  First subtract the mean then calculate the PSD, in (dB), for the defined interval.
    
    Parameters
    ----------
    subject #TODO
    electrodes #TODO
    data : TYPE numpy array size [run,sampled data,electrode channel]
        DESCRIPTION.
    freq_band : TYPE, optional
        DESCRIPTION. This specoicifies the start and end freq, in Hz, to be evaluated for PSD. The default is [1,20].
    run : Type int, paradigm run of interest.  6 situations and 2 phases, recital, reall #TODO make this a an enumerate
    channels : TYPE, optional #TODO need to make channels enumerate so that they can be selected
        DESCRIPTION. The default is 14.
    sample_interval : TYPE int time in seconds, optional
        DESCRIPTION. This is the end time for the sample interval starting from 0 seconds. The default is 15 seconds.
    fs#TODO
    Returns 
    -------
    None.

    '''
#         alpha_band=np.zeros(14)
#         high_beta_band=np.zeros(14)        
#         less_20_hz=np.zeros(14)
#         
    plt.figure(num=subject+51, figsize=(8,6),clear=all)    # 51 is arbritrary This should result in fig 51 being associated with subject 01, Sub 23 Fig 73
    #TODO channels=14 to electrode enumerate
    
    # initialize the power in frequecy band to zereo
    PSD_band = np.zeros(len(electrodes))
    # PSD for trial sample interval 
    sample_index =sample_interval * fs  # sample interval from start to sample time in seconds mult by fs 
    # Convert the frequncy band from herts to index
    freq_low = freq_band[0]*sample_interval
    freq_high = freq_band[1]*sample_interval
    
    # For the purpose of plotting get the PSD frequency components
    freq = np.arange(0,((fs/2)+1/fs),1/sample_interval)
    
    for electrode_index,electrode in enumerate(electrodes):  # TODO change from all "14" to channels
        print(electrode_index,electrode)
        # Calculate and plot the entire PSD 
        #PSD = np.real(10*np.log10((np.fft.rfft(data[trial,0:sample_index,channel]))**2)) # imaginary part = 0, extract real to avoid warnings #TODO delete this reference
        PSD = np.real(10*np.log10((np.fft.rfft(data[run,:,electrode_index]-np.mean(data[run,:,electrode_index])))**2)) # imaginary part = 0, extract real to avoid warnings
        plt.plot(freq,PSD,label = (f"Chan {electrode}"))
        plt.ylabel ('PSD (dB)')
        plt.ylim([-5,80])  #TODO problem plots zero hz value
        plt.xlabel ('Frequency (Hz)')
        plt.suptitle (f'Power Spectral Density Run {run} Subject {subject+1}') # Indicate Trial number index starting from 1
        #     # plt.suptitle (f'PSD 4-20 Hz Band for Light Anxiety Qty {light_count}')  #normal, light, severe
        plt.title (f'Level {level},Time Interval {sample_interval} sec')
        plt.grid('on')
        plt.legend(loc='upper right') 
        # Integrated the spectrum and normalized based on the length of the freq band
        PSD_band [electrode_index] =  np.sum(PSD[freq_low:freq_high])/(freq_high-freq_low) # 4 to 20 Hz  # Remeber that the raw data is BP filtered 4 to 45 hz.
    plt.tight_layout()
    #Save figure 
    plt.savefig(f'PSD_subject{subject}.png')          #TODO Light Severe
     # ....then show  
    plt.show()
    return   PSD_band 



#%% This does not work ...showed this to Dr J.  He suggested that the info be plotted on the x-axis or use other methods such as violin plots.

def plot_PSD_by_anxiety (subject, electrodes, anxiet_level, PSD_band ,run = 1):
#   count = [0,0,0,0]    # make into an array to facilitate the display of information
    plt.figure(num = 1000, clear=all) 
# #            # print (f'{situation+1},{channel}')  # TODO test
# #            # print(f'{PSD[0]}')                  #TODO test
         
# #             #plt.plot(alpha_band,label = (f'Channel = {channel} Alpha Band'))
# #             #plt.plot(high_beta_band,label = ("Beta Band"))
            
# #             # Plot all four anxiety levels in subplots
                
#     for plot,anxiety in enumerate (anxiety_levels):
#         print(f'plot {plot}, anxiety {anxiety}')
#         #TODO not sure if I need this if statement.
#         if label_array[plot+2,trial]:  # plot trials associated with sever anxiety= 2, moderate = 3 ,light = 4  and normal = 5    index into label array
#             plt.subplot(2, 2, plot+1)
#             #TODO only want slected channels
#             plt.plot(PSD_band[0:len(electrodes)],'.')#,label = ("4-20 Hz"))  ... add a dot to the associatded subplot
        
#               # Take the mean across all trials  #TODO
#              #        psd_sum[plot] = psd_sum[plot] +  less_20_hz[0:len(electrodes)]
        
#             plt.xlabel("Channel")
#             plt.xticks(np.arange(len(electrodes)),electrodes) #[0,1,2,3,4,5,6,7,8,9,10,11,12,13], electrodes)
                  
#             plt.ylabel ('PSD (dB)')
#             plt.ylim([10,70])  #TODO problem plots zero for each channel then correct value
#                 #plt.suptitle (f'Power in the Bands Trial {trial + 1} Subject {subject}') # Indicate Trial number index starting from 1
# #                     plt.suptitle (f'PSD 4-20 Hz Band;All Subjects; All Trials; Time Interval {T} sec')  #normal, light, severe
                            
# #                     plt.title (f'Anxiety Level {anxiety}; Qty {count[plot]}')    
             

        
#             # if label_array[2,trial]:  # plot trials associated with sever anxiety= 2, moderate = 3 ,light = 4  and normal = 5    index into label array
#             #     plt.plot(less_20_hz,'.')#,label = ("4-20 Hz"))
#     # plt.xlabel("Channel")
#     # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13], electrodes)
#     # plt.ylabel ('PSD (dB)')
#     # plt.ylim([10,70])  #TODO problem plots zero for each channel then correct value
#     # #plt.suptitle (f'Power in the Bands Trial {trial + 1} Subject {subject}') # Indicate Trial number index starting from 1
#     # plt.suptitle (f'PSD 4-20 Hz Band for Light Anxiety Qty {light_count}')  #normal, light, severe
#     # plt.title (f'All Subjects; All Trials Time Interval {T} sec')
#     # #plt.title ('AF3, AF4, F3, F4, FC5, FC6, F7, F8, T7, T8, P7, P8, O1, O2')
#     # plt.label =  (f'{subject}')
#     # #plt.legend(loc='upper right') 
    
#     #Save figure 
#     fig.tight_layout()
#     plt.savefig(f'20Hz_PSD_All_Subjects_All_Channels_T_{T}.png')          #TODO Light Severe
#      # ....then show  
#     plt.show()
    
    
#     #TODO Plotting of the mean does not work correctly yet  
    
#     # To aid visual inspoection, plot anxiety levels on the x axis and 
#     plt.figure(num = 1000)
    
#         #for plot,anxiety in enumerate (anxiety_levels): 
#     psd_mean = np.divide(psd_sum[plot,0:8],count[plot])/12      # To get the mean divide the sum for each electrode by number of (trials * count )  
          
#     plt.plot(psd_mean[plot:len(electrodes)],'rp', markersize=14)  




