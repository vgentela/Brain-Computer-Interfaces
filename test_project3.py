"""
Name:        test_project3.py
Description: Invokes the necessary routines to perform the requirements of BCI Project 3

Working on a method to extract spectral features.
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


#%% Load the RAW data 
# Two methods of getting at the eeg data contained in the dataset: list, and nimpy array.
# The numpy array is perfrered method
#TODO do I have the correct interpretation from our readme file , and the excel file?

# ds_arr; size [trials x samples x electrodes]; This is the processed 15 second eeg data from the 12 trials (6 situation* 2 runs) and 23 subjects
# Processed by ICA artifact removal and bandpass filtered. 

#labels: Two columns for the subject Self-Assessment Manikin (SAM). One column is an event's positive or negative score for valence, 
#and the other is the arousal spectrum, from calmness to excitement. A combination of these two scores establishes anxiety levels. 
# After transpose of the loaded row infromation 
#Explore if there is a diffrence in topo maps between subject anxiety levels

# subject = 6  # Hamilton sores = severe,  Arousal = 1
# Hamilton = 'Severe'
# subject = 9  # Hamilton sores = light ,  Arousal = 8
# Hamilton = 'Light'
Hamilton = 'n/a'
# Setup the plot 
#plt.figure()           # Initally a single plot per figure

#fig, axs = plt.subplots(2, 2)

fig = plt.figure()

# Select electodes of associated with anxiety.  Frontal electodes
# TODO Cite reference []
# All avaialbe electrodes
#electrodes = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']  
    
 # Frontal Electrodes
electrodes = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8']
    

# Choa Chen ref Electrodes
    

#electrodes = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8']  #TODO please check to ensure that I have selected the correct electrodes
#electrodes = [ 'F3', 'F4']

#subjects = ['06','09','13']
#subjects = ['01','04','05','07','10','11','12','13','14','17','18','19','20','21']
#subjects = ['02','03','08','15','16','22','23']  # light
#subjects = ['05']  

# count the anxiety levels 
anxiety_levels = ['Severe','Moderate','Light','Normal']
severe_count = 0
moderate_count = 0
light_count = 0
normal_count = 0
count = [0,0,0,0]    # make into an array to facilitate the display of information

#Calculate the mean value of each electrode across trials catigorized based on severity of anxiety
# Pre assign the array  [ levels of axiety x number of electrodes ]
psd_sum = np.zeros([len(anxiety_levels),len(electrodes)])


subjects = ['06','09','13','01','04','05','07','10','11','12','13','14','17','18','19','20','21','02','03','08','15','16','22','23']

for plot,subject in enumerate(subjects):

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
        situation = list(h5py.File(filename, "r")['situation']) #TODO Not sure how this is encoded for the 6 situations?
        
        # make into an array to facilitate the display of information
        count = [severe_count,moderate_count,light_count,normal_count]
           
          



    # #%% Clear figures
    # fig = plt.get_fignums()
    # for index,fig_num in enumerate (fig):
    #     plt.close(fig_num)

    print(f'Loaded Subject {subject}')
    # Visualize the data (Time and Frequency Domain)
    # loop through all 12 situations 6 runs x (recitation = recall)
    fs = 128
    #Time Domain
    sample_size = len(ds_arr[0,:,0]) # TODO assumes same size for all trials
    
    
    t = np.arange(0,sample_size/fs,1/fs)
    num_of_trials = 12                              # trials = 2 x situations
    # plt.figure()
    for trial in range (num_of_trials):
        # print(f"Situation {1+(trial//2)}")              # Convert trials to situations to match excel
        # #situation = 0 # the first step in the protocol
        # # fs = 128
        # # #Time Domain
        # # sample_size = len(ds_arr[subject,:,0])
        # # t = np.arange(0,sample_size/fs,1/fs)
        # #plt.figure(num = situation +4, clear=all)
        # plt.figure()
        # plt.suptitle(f"Raw Time Domain Subject {subject} Hamilton {Hamilton}")  #??? not loading this field correctly
        # plt.title(f"Situation {1+(trial//2)}, Valance = {labels[0][trial]}, Arousal ={labels[1][trial]} ")
        # plt.ylabel("Amplitude (uV)")    #TODO is the data in uV?
        # plt.ylim([-60,60])
        # plt.xlabel("Time (sec)")
        # for channel in range(14):
            
        #     plt.plot(t,ds_arr[trial,:,channel],label = (f"Chan {channel}"))
        # plt.legend(loc='upper left')
        #Freq Domain
        T = 2          #TODO sample interval
        freq = np.arange(0,((fs/2)+1/fs),1/T)
        #plt.figure()
        # plt.suptitle(f"Raw Frequency Domain Subject {subject}  Hamilton {Hamilton}")
        # plt.title(f"Situation {1+(trial//2)}, Valance = {labels[0][trial//2]}, Arousal ={labels[1][trial//2]} ")
        # plt.ylabel("PSD (dB)")
        # plt.ylim([0,65])
        # plt.xlabel("Freq (Hz)")
        # Pre allocated the PSD arrays
        
        alpha_band=np.zeros(14)
        high_beta_band=np.zeros(14)        
        less_20_hz=np.zeros(14)
        for channel in range(14):  # frontal only
            # PSD for trial sample interval 
            sample_interval =T *128  # sample interval from start to sample time in seconds mult by fs 
            PSD = np.real(10*np.log10((np.fft.rfft(ds_arr[trial,0:sample_interval,channel]))**2)) # imaginary part = 0, extract real to avoid warnings
            #plt.plot(freq,PSD,label = (f"Chan {channel}"))
            # Integrated and normalized
            alpha_band[channel] = np.sum(PSD[8*T:12*T])/(12*T-8*T)  # 8 to 12 Hz   sum each freq component then normalized by dividing by bandwidth
            high_beta_band[channel] = np.sum(PSD[18*T:40*T])/(40*T-18*T) # 18 to 40 Hz
            less_20_hz [channel] =  np.sum(PSD[4*T:20*T])/(20*T-4*T) # 4 to 20 Hz  # Remeber that the raw data is BP filtered 4 to 45 hz.
           # print (f'{situation+1},{channel}')  # TODO test
           # print(f'{PSD[0]}')                  #TODO test
         
            #plt.plot(alpha_band,label = (f'Channel = {channel} Alpha Band'))
            #plt.plot(high_beta_band,label = ("Beta Band"))
            
            # Plot all four anxiety levels in subplots
            
            for plot,anxiety in enumerate (anxiety_levels):
                print(f'plot {plot}, anxiety {anxiety}')
                #TODO not sure if I need this if statement.
                if label_array[plot+2,trial]:  # plot trials associated with sever anxiety= 2, moderate = 3 ,light = 4  and normal = 5    index into label array
                    plt.subplot(2, 2, plot+1)
                    
                    #TODO only want slected channels
                    plt.plot(less_20_hz[0:len(electrodes)],'.')#,label = ("4-20 Hz"))  ... add a dot to the associatded subplot
                    
                    # Take the mean across all trials  #TODO
                    psd_sum[plot] = psd_sum[plot] +  less_20_hz[0:len(electrodes)]
                    
                  
                    plt.xlabel("Channel")
                    plt.xticks(np.arange(len(electrodes)),electrodes) #[0,1,2,3,4,5,6,7,8,9,10,11,12,13], electrodes)
                    #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13], electrodes)   #TODO delete this
                    
                    plt.ylabel ('PSD (dB)')
                    plt.ylim([10,70])  #TODO problem plots zero for each channel then correct value
                    #plt.suptitle (f'Power in the Bands Trial {trial + 1} Subject {subject}') # Indicate Trial number index starting from 1
                    plt.suptitle (f'PSD 4-20 Hz Band;All Subjects; All Trials; Time Interval {T} sec')  #normal, light, severe
                            
                    plt.title (f'Anxiety Level {anxiety}; Qty {count[plot]}')    
   
 #TODO Plotting of the mean does not work correctly yet  
   
    # for plot,anxiety in enumerate (anxiety_levels): 
    #     psd_mean = np.divide(psd_sum[plot,0:8],count[plot])/12      # To get the mean divide the sum for each electrode by number of (trials * count )  
    #     plt.subplot(2,2,plot+1)
    #     plt.plot(psd_mean[plot:len(electrodes)],'rp', markersize=14)                
            
        
            # if label_array[2,trial]:  # plot trials associated with sever anxiety= 2, moderate = 3 ,light = 4  and normal = 5    index into label array
            #     plt.plot(less_20_hz,'.')#,label = ("4-20 Hz"))
    # plt.xlabel("Channel")
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13], electrodes)
    # plt.ylabel ('PSD (dB)')
    # plt.ylim([10,70])  #TODO problem plots zero for each channel then correct value
    # #plt.suptitle (f'Power in the Bands Trial {trial + 1} Subject {subject}') # Indicate Trial number index starting from 1
    # plt.suptitle (f'PSD 4-20 Hz Band for Light Anxiety Qty {light_count}')  #normal, light, severe
    # plt.title (f'All Subjects; All Trials Time Interval {T} sec')
    # #plt.title ('AF3, AF4, F3, F4, FC5, FC6, F7, F8, T7, T8, P7, P8, O1, O2')
    # plt.label =  (f'{subject}')
    # #plt.legend(loc='upper right') 
    
    #Save figure 
    fig.tight_layout()
    plt.savefig(f'20Hz_PSD_All_Subjects_All_Channels_T_{T}.png')          #TODO Light Severe
     # ....then show  
    plt.show()

#%% Topo Maps
# Plot time average for a giving subject, situation

situation_mean = np.mean (ds_arr, axis = 0)  # This is the average across the 12 situations
sample_mean = np.mean (ds_arr, axis = 1)  # This is the average across samples [12 situations x 14 channels]
plt.figure(num = 21, clear =all)


situation = 6   # at least for subject 6 the scalp plots were asymetrical and simular for each situation
#pt.plot_topo(channel_names=eletrodes, channel_data=ds_arr[situation,128,:],title=f'Subject {subject}, Situation {situation}',cbar_label='Voltage (uV)',montage_name='biosemi64')

# channel_data must match selected electrodes.  In this case they are the first 8 so 0:7

pt.plot_topo(channel_names=electrodes, channel_data=sample_mean[situation,2:4],title=f'Subject {subject}, Situation {situation}',cbar_label='Voltage (uV)',montage_name='biosemi64')



#%% Clear figures
fig = plt.get_fignums()
for index,fig_num in enumerate (fig):
    plt.close(fig_num)