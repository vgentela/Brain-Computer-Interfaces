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
import torch.nn.functional as  F
from torch.distributions import Normal
import torch.optim as optim
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
    
    keys= list(key for key in df.keys())
  
    bpower = tfs.BandPowerSpectralDensity(128,band_dict={'alpha': [8, 14], 'beta': [14, 31], 'gamma': [31, 49]})
    de = tfs.BandDifferentialEntropy(128,band_dict={'alpha': [8, 14], 'beta': [14, 31], 'gamma': [31, 49]})
    
    trial_band_powers=[]
    trial_entropys =[]
    combined_labels= pd.DataFrame()
    
    for key in keys:
        eeg_data = df[f'{key}'].drop(['valence','arousal','Anxiety_level'],axis=1)
        
        normalized_eeg = eeg_data.apply(lambda x: x-np.mean(x),axis =1).to_numpy()
        #print(len(normalized_eeg))
        normalized_eeg = normalized_eeg.reshape(12,1920,14)
        
        anxiety_degree = df[f'{key}']['Anxiety_level'][~pd.isna(df[f'{key}']['Anxiety_level'])]
        combined_labels = pd.concat([combined_labels,anxiety_degree])
    
       
        for trial in normalized_eeg:
            
            powers =bpower(eeg=trial.T)
            band_powers = powers['eeg']
            
            
            differential_entropy = de(eeg=trial.T)    
            band_entropys = differential_entropy['eeg']
            
            trial_band_powers.append(band_powers)
            trial_entropys.append(band_entropys)
    #print(combined_labels) 
    labels = pd.get_dummies(combined_labels).to_numpy()
    #print(labels)
    #labels = labels.reshape(12,labels.shape[1])  
    
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
        
    elif model =='randomforest':
        band_arrays = np.asarray(trial_band_powers)
        entropy_arrays = np.asarray(trial_entropys)
        
        features = np.transpose(np.concatenate((band_arrays,entropy_arrays),axis=2),(0,2,1))
        #labels = np.stack(labels)
        print(len(features),len(labels))
        
        train_data, test_data, train_labels, test_labels = train_test_split(features,labels,test_size=0.3)
        
        return train_data, train_labels, test_data, test_labels
    else:
        raise Exception('please choose either autoencoder or randomforest')
     
            


#%%
class Encoder(nn.Module):
    
    def __init__(self,input_size,latent_embedding,device):
        super().__init__()
        
        self.linear1 = nn.Linear(input_size[0]*input_size[1],50)
        self.linear2 = nn.Linear(50,20)
        self.flatten = nn.Flatten()
        self.z_mean = nn.Linear(20,latent_embedding)
        self.log_var =nn.Linear(20,latent_embedding)
        #print(latent_embedding[0])
        self.device = device
        
    def forward(self,inputs):
        inputs = F.relu(self.linear1(inputs.float().flatten(1)))
        print(inputs.shape)
        inputs = F.relu(self.linear2(inputs))
        print(inputs.shape)
        inputs = self.flatten(inputs)
       
        z_mean = self.z_mean(inputs).to(self.device)
        log_var = self.log_var(inputs).to(self.device)
        
        batch, dim = z_mean.shape
        epsilon = Normal(0, 1).sample((batch,dim)).to(self.device)
        
        z = z_mean + torch.exp(0.5 * log_var) * epsilon
        #print(z)
        return z_mean, log_var, z
        
#%%
class Decoder(nn.Module):
    def __init__(self,embedding_dim,orig_dim):
        super().__init__()
   
        self.linear1 = nn.Linear(embedding_dim,orig_dim[0]*orig_dim[1])
        self.reshape = lambda x: x.view(-1,orig_dim[0],orig_dim[1])
        self.linear2 = nn.Linear(20, 50)
        self.linear3 = nn.Linear(50,84)
        
    
    def forward(self,latent_vector):
        latent_vector = self.linear1(latent_vector.float())
        latent_vector = self.reshape(latent_vector)
        latent_vector= F.relu(self.linear2(latent_vector))
        latent_vector = torch.sigmoid(self.linear3(latent_vector))
        
        return latent_vector.view(-1,6,14)
#%%
class Classifier(nn.Module):
    
    def __init__(self,embedding_dim,target_number):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim,7)
        self.linear2 = nn.Linear(7,target_number)
    def forward(self,z):
        z = self.linear1(z)
        z = F.relu(self.linear2(z))
        return torch.sigmoid(z)
#%%

class VAE(nn.Module):
    def __init__(self,encoder,classifier,decoder,device):
        super().__init__()
        if device =='cuda' and torch.cuda.is_available():
            self.encoder = encoder.to('cuda')
            self.decoder = decoder.to('cuda')
            self.classifier = classifier.to('cuda')
        elif device == 'cpu':
            self.encoder = encoder.to('cpu')
            self.decoder = decoder.to('cpu')
            self.classifier = classifier.to('cpu')
        else:
            raise Exception('Please choose cuda or cpu')
    
    def forward(self,x):
        z_mean,log_var,z = self.encoder.forward(x)
        
        output = self.classifier.forward(z)
        
        reconstruction = self.decoder.forward(z)
        
        return z_mean,log_var,output,reconstruction
#%%
class Loss():
    
    def kl_divergence(self,z_mean,log_var):
        
        kld =-0.5 * torch.sum(1 +log_var - z_mean.pow(2) - log_var.exp(),dim=1)
        
        return kld.mean()
    
    def reconstruction_loss(self,x_reconstructed,x):
        mse_loss = nn.MSELoss() 
        #print(x_reconstructed)
        return mse_loss(x_reconstructed,x)
    
    def classification_loss (self,y_pred,y_true):
        loss = nn.BCELoss()
        return loss(y_pred,y_true)
    
    def vae_loss(self,pred,true,label):
        
        z_mean,log_var,output,reconstruction_x =pred
        #print(output,label)
        recon_loss =  self.reconstruction_loss(reconstruction_x, true)
        
        kld_loss = self.kl_divergence(z_mean, log_var)
        
        class_loss = self.classification_loss(output,label)
        
        return 100*recon_loss + kld_loss + 300*class_loss,class_loss
#%%

class train():
    def __init__(self,optimizer,latent_embedding,device,train_loader,lr=1e-3,epochs=40):
        self.optimizer = optimizer
        self.lr =  lr
        self.epochs =epochs
        self.device = device
        self.latent_embedding =  latent_embedding
        self.train_loader = train_loader
    def training(self):
        encoder = Encoder((6,14), self.latent_embedding,self.device)
        decoder = Decoder(self.latent_embedding, (1,20))
        classifier = Classifier(self.latent_embedding,3)
        vae = VAE(encoder,classifier,decoder, self.device)
        los = Loss()
        opt = self.optimizer
        optimizer = opt(list(encoder.parameters())+list(decoder.parameters()),
                                         lr =self.lr)
        
        train_loss = []
        epoch_loss = []
        accs = []
        classification_loss =[]
        for epoch in range(self.epochs):
            vae.to(self.device)
            correct = 0
            for batch_id, (data,target) in enumerate(self.train_loader):
                data = data.to(self.device)
                print(data.shape)
                optimizer.zero_grad()
                
                pred= vae.forward(data)
                
                loss,class_loss = los.vae_loss(pred,data.float(),target.float())
                print(class_loss)
                loss.backward()
                
                optimizer.step()
                
                train_loss.append(loss.item())
                classification_loss.append(class_loss.item())
                
                b_correct= torch.sum(torch.abs(pred[2]-target) <0.5)
                correct += b_correct
                
            acc = float(correct)/len(self.train_loader)
            accs.append(acc)   
            epoch_loss.append(loss/len(self.train_loader))
            print('Average Train Loss:',np.mean(classification_loss[epoch]),sep=':')
            print('Accuracy:',acc,sep =':')
            print('----------------------------------------------------------------')
    
            
        return classifier,accs,train_loss,epoch_loss
        
        
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
    
   subjects_df = {}
   counts = []
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
                
            if len(counts) ==0:
                counts.append(count_tuple)
            else:
                count= counts.pop()
                updated_count = tuple(map(sum,zip(count,count_tuple)))
                counts.append(updated_count)
            
   print(counts)
   severe_count,moderate_count,light_count,normal_count = counts[0] 
   print('severe count:',severe_count)
   print('moderate_count:',moderate_count)
   print('light_count:',light_count)
   print('normal_count:',normal_count)
    
   return subjects_df
   

 
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








