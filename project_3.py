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

def plot_edf_data(raw_data_edf,electrode_index = (2,16),fs_edf=128):
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

    plt.figure(num = 200)
    for index in range(2,16):  # raw data valid eed electrode channels index 2 - 15
        print(index)
        time_edf = np.arange(0,len(raw_data_edf[0,:])/fs_edf,1/fs_edf)
        plt.plot(time_edf,(raw_data_edf[index,:] - np.mean(raw_data_edf[index,:])))

        T_edf = (len(raw_data_edf[0,:])-3)/fs_edf          #TODO sample interval
        freq_edf = np.arange(0,((fs_edf/2)+1/fs_edf),1/T_edf)
        #raw_data_edfm [36 channels x samples ]
    for index in range(2,16):  # raw data valid eed electrode channels index 2 - 15
        PSD_edf = np.real(10*np.log10((np.fft.rfft(raw_data_edf[index,:]-np.mean(raw_data_edf[index,:])))**2)) # imaginary part = 0, extract real to avoid warnings
        
        plt.figure(num=300)
        plt.plot(freq_edf,PSD_edf, label = (f"Chan {index}"))
    plt.legend()
    plt.tight_layout()
    #Save figure 
    plt.savefig('Raw_Data_edf.png')          #TODO Light Severe
     # ....then show  
    plt.show()
    
    
    
    
    return

#%% Visualize the data Scalp topo maps

def plot_scalp_map ( subject, electrodes, data, data_type = '.edf', run = 1,method = 'mean', domain = 'time'):

#paradigm per subject consists of 6 situations composed of a recital phase followed by a recall phase 
#TODO In this data set run + is a number from 1 to 6 x 2

# Plot time average for a giving subject, situation
    plt.figure(num = 21, clear =all)
    if method == 'mean':
        if data_type == '.mat':
            #situation_mean = np.mean (ds_arr, axis = 0)  # This is the average across the 12 situations
            sample_mean = np.mean (data, axis = 1)  # This is the average across samples [12 situations x 14 channels]
            #TODO # channel_data must match selected electrodes.  In this case they all electrodes 0:14
            pt.plot_topo(channel_names=electrodes, channel_data=sample_mean[run,0:14],title=f'Subject {subject}, Run{run}',cbar_label='Voltage (uV)',montage_name='biosemi64')
        elif data_type == '.edf':
             #raw_data_edf[index for the first electrode:index for the last electrode, EEG data]
             #TODO this only works for all electrodes
             #TODO (raw_data_edf[Index of electrodes start at 2 end at 16,data]
             #TODO raw_data_mean = np.mean(raw_data_edf[2:16,:],axis = 1) 
             raw_data_mean = np.mean(data[2:16,:],axis = 1)   
             pt.plot_topo(channel_names=electrodes, channel_data=raw_data_mean,title=f'Subject {subject}, Run{run}',cbar_label='Voltage (uV)',montage_name='biosemi64')
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
   anxiety_levels = ['Severe','Moderate','Light','Normal']
   severe_count = 0
   moderate_count = 0
   light_count = 0
   normal_count = 0
 
   for index,subject in enumerate(subjects):
        
    
    #filename = "DASPS_Database/Raw data.mat/S09.mat"   #DASPS_Database/Raw data.mat/S09.mat
        #filename = f'DASPS_Database/Raw data.mat/S{subject}.mat'   
        filename = f'{directory}S{subject}.mat'
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
            #TODO get into array format
            label_array = np.zeros([6,12])
            label_array[0:2,:] = np.array(labels)[0:2,:]   # row[0] = Valence, row[1]=Arousal
        
            # sort anxiety levels
            label_array[2] = np.where((label_array[0,:] <= 2) & (label_array[1,:] > 6) & (label_array[1,:] < 10), 1, 0)        # severe
            severe_count = severe_count + np.sum(label_array[2])         
            label_array[3]=np.where((label_array[0,:] > 2) & (label_array[0,:] <= 4) & (label_array[1,:] > 5) & (label_array[1,:] < 8), 1, 0) #moderate
            moderate_count = moderate_count + np.sum(label_array[3])  
            label_array[4]=np.where((label_array[0,:] > 4) & (label_array[0,:] <= 5) & (label_array[1,:] > 4) & (label_array[1,:] < 7), 1, 0) #light
            light_count = light_count + np.sum(label_array[4])  
            label_array[5]=np.where((label_array[0,:] > 5) & (label_array[0,:] <= 8) & (label_array[1,:] > 4) & (label_array[1,:] < 7), 1, 0) #normal
            normal_count = normal_count + np.sum(label_array[5])  
            hamilton = list(h5py.File(filename, "r")['hamilton'])  # The pre and post Hamilton socres as evaluated by therapist 
            situation = list(h5py.File(filename, "r")['situation']) #This is the number of situations per paridigm
            
            # make into an array to facilitate the display of information
            count = [severe_count,moderate_count,light_count,normal_count]
            print(f'Loaded Subject {subject}')      # Provide user feedback
            
            
            plot_PSD (index,electrodes,ds_arr,freq_band = [4,20],run = 1, sample_interval=15,fs =128)
    
   #TODO not properly dealing with subject 
    
   return ds_arr, count, label_array  
       
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
 
def plot_PSD (subject,electrodes,data,freq_band = [4,20],run = 1, sample_interval=15,fs =128):
    '''
    Visualize the data Frequency Domain.  First subtract the mean then calculate the PSD, in (dB), for the defined interval.
    
    Parameters
    ----------
    data : TYPE numpy array size [trial,sampled data,electrode channel]
        DESCRIPTION.
    freq_band : TYPE, optional
        DESCRIPTION. The default is [1,20].
    run : Type int, paradigm run of interest.  6 situations and 2 phases, recital, reall #TODO make this a an enumerate
    channels : TYPE, optional #TODO need to make channels enumerate so that they can be selected
        DESCRIPTION. The default is 14.
    sample_interval : TYPE int time in seconds, optional
        DESCRIPTION. This is the end time for the sample interval starting from 0 seconds. The default is 15 seconds.

    Returns 
    -------
    None.

    '''
#         alpha_band=np.zeros(14)
#         high_beta_band=np.zeros(14)        
#         less_20_hz=np.zeros(14)
#         
    plt.figure(num=subject+50)    # 50 is arbritrary 
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
        
        #PSD = np.real(10*np.log10((np.fft.rfft(data[trial,0:sample_index,channel]))**2)) # imaginary part = 0, extract real to avoid warnings #TODO delete this reference
        PSD = np.real(10*np.log10((np.fft.rfft(data[run,:,electrode_index]-np.mean(data[run,:,electrode_index])))**2)) # imaginary part = 0, extract real to avoid warnings
        plt.plot(freq,PSD,label = (f"Chan {electrode}"))
        plt.ylabel ('PSD (dB)')
        plt.ylim([-5,80])  #TODO problem plots zero hz value
        plt.xlabel ('Frequency (Hz)')
        plt.suptitle (f'Power Spectral Density Run {run} Subject {subject}') # Indicate Trial number index starting from 1
        #     # plt.suptitle (f'PSD 4-20 Hz Band for Light Anxiety Qty {light_count}')  #normal, light, severe
        plt.title (f'All Electrodes, Time Interval {sample_interval} sec')
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
    plt.figure(num = 0, clear=all) 
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




