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
import plot_topo as pt
import numpy as np
#TODO #c:\users\18023\anaconda3\lib\site-packages
import h5py
#import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
import mne as mne
import pandas as pd
from torcheeg import transforms as tfs
import torch
from torch.utils.data import TensorDataset,DataLoader,random_split
from torch import nn
import torch.nn.functional as  F
from torch.distributions import Normal
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,f1_score,recall_score,accuracy_score,confusion_matrix,roc_curve,auc
from sklearn.model_selection import GridSearchCV as gsv
from sklearn.svm import SVC
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
    """
   Label the EEG data based on the anxiety levels extracted from labels.
   
   Args:
       data (list): List of EEG data samples.
       labels (ndarray): Array of anxiety levels corresponding to each data sample.
       
   Returns:
       DataFrame: DataFrame containing the labeled EEG data.
       tuple: A tuple containing the count of samples for each anxiety level category (severe, moderate, light, normal).
   """
   
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
       
        if val<=5 and aro>=5:
            if 0<=val<=2 and 7<=aro<=9:
                df.at[index,'Anxiety_level'] = 'severe'
                severe_count+= 1
            elif 2<val<=4 and 6<=aro<7:
                df.at[index,'Anxiety_level'] = 'moderate'
                moderate_count+=1
            elif 4<val<=5 and 5<=aro<6:
                df.at[index,'Anxiety_level'] = 'light'
                light_count+= 1
            else:
                df.at[index,'Anxiety_level']= 'normal'
                normal_count +=1
                
        else:
            df.at[index,'Anxiety_level'] = 'normal'
            normal_count += 1   
        
            
       
    #df = df.replace(np.nan,'')
    df.set_index('trial',inplace= True)
    
    
    return df,(severe_count,moderate_count,light_count,normal_count)
#%%
def transformations(df,model,split_ratio= None, test_size = None):
    """
    Transform the raw EEG data into features suitable for either autoencoder or random forest models.
    
    Args:
        df (DataFrame): DataFrame containing the raw EEG data.
        model (str): Model type to transform the data for ('autoencoder' or 'randomforest').
        split_ratio (float, optional if using randomforest): Ratio to split the data into training and testing sets (only for autoencoder).
        test_size (float, optional if using autoencoder): Proportion of the dataset to include in the test split (only for randomforest).
        
    Returns:
        tuple: Tuple containing the transformed training and testing data (for autoencoder) or 
               training data, training labels, testing data, and testing labels (for randomforest).
    """
    keys= list(key for key in df.keys())
  
    bpower = tfs.BandPowerSpectralDensity(128,band_dict={'alpha': [8, 14], 'beta': [14, 31], 'gamma': [31, 49]})
    de = tfs.BandDifferentialEntropy(128,band_dict={'alpha': [8, 14], 'beta': [14, 31], 'gamma': [31, 49]})
    
    trial_band_powers=[]
    trial_entropys =[]
    combined_labels= pd.DataFrame()
    
    for key in keys:
        eeg_data = df[f'{key}'].drop(['valence','arousal','Anxiety_level'],axis=1)
        
        normalized_eeg = eeg_data.apply(lambda x: (x-np.mean(x))/np.std(x),axis =1).to_numpy()
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
        
        train_data, test_data = random_split(dataset,split_ratio)
        
        
        return train_data,test_data
        
    elif model =='randomforest':
        band_arrays = np.asarray(trial_band_powers)
        entropy_arrays = np.asarray(trial_entropys)
        
        features = np.transpose(np.concatenate((band_arrays,entropy_arrays),axis=2),(0,2,1))
        #labels = np.stack(labels)
        #print(len(features),len(labels))
        
        train_data, test_data, train_labels, test_labels = train_test_split(features,labels,test_size=test_size)
        
        return train_data, train_labels, test_data, test_labels
    else:
        raise Exception('please choose either autoencoder or randomforest')
     
            


#%%
class Encoder(nn.Module):
    """
    Neural network encoder module for extracting latent representations from EEG data.
    
    Args:
        input_dim (int): Input dimensionality of the EEG data.
        hidden_dims (list): List of hidden layer dimensions for the encoder network.
        latent_dim (int): Dimensionality of the latent space.
        device (str): Device to use for computations ('cuda' or 'cpu').
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim, device):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.z_mean = nn.Linear(hidden_dims[1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[1], latent_dim)

        
    def forward(self,inputs):
        """
        Forward pass of the encoder module.
        
        Args:
            inputs (Tensor): Input EEG data.
            
        Returns:
            tuple: Tuple containing the mean and log variance of the latent space and the sampled latent vector.
        """
        
        inputs = F.relu(self.linear1(inputs.float().view(inputs.size(0), -1)))
        #print(inputs.shape)
        inputs = F.relu(self.linear2(inputs))
        #print(inputs.shape)
        #inputs = self.flatten(inputs)
       
        z_mean = self.z_mean(inputs).to(self.device)
        log_var = self.log_var(inputs).to(self.device)
        
       
        epsilon = Normal(0, 1).sample(z_mean.shape).to(self.device)
        
        z = z_mean + torch.exp(0.5 * log_var) * epsilon
        #print(z)
        return z_mean, log_var, z
        
#%%
class Decoder(nn.Module):
    """
    Neural network decoder module for reconstructing EEG data from latent representations.
    
    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_dim (int): Dimensionality of the output EEG data.
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, output_dim)

       

    def forward(self,latent_vector):
        """
        Forward pass of the decoder module.
        
        Args:
            latent_vector (Tensor): Latent representation vector.
            
        Returns:
            Tensor: Reconstructed EEG data.
        """
        latent_vector = self.linear1(latent_vector)
        
        return torch.sigmoid(latent_vector).view(-1, 6, 14) 
   
#%%
class Classifier(nn.Module):
    """
   Neural network classifier module for predicting anxiety levels from latent representations.
   
   Args:
       embedding_dim (int): Dimensionality of the latent space.
       hidden_dim (list): List of hidden layer dimensions for the classifier network.
       target_number (int): Number of target classes.
   """
    
    def __init__(self,embedding_dim,hidden_dim,target_number):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim,hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0],hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1],10)
        self.linear4 = nn.Linear(10,target_number)
        
    def forward(self,z):
        """
       Forward pass of the classifier module.
       
       Args:
           z (Tensor): Latent representation vector.
           
       Returns:
           Tensor: Predicted probabilities for each target class.
       """

        z = self.linear1(z)
        z = F.relu(self.linear2(z))
        z  =F.relu(self.linear3(z))
        return torch.sigmoid(self.linear4(z))

#%%

class VAE(nn.Module):
    """
    Variational autoencoder (VAE) module for jointly learning latent representations and reconstructing EEG data.
    
    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        device (str): Device to use for computations ('cuda' or 'cpu').
    """
    
    def __init__(self,encoder,decoder,device):
        super().__init__()
        if device =='cuda' and torch.cuda.is_available():
            self.encoder = encoder.to('cuda')
            self.decoder = decoder.to('cuda')
        elif device == 'cpu':
            self.encoder = encoder.to('cpu')
            self.decoder = decoder.to('cpu')
            
        else:
            raise Exception('Please choose cuda or cpu')
    
    def forward(self,x):
        """
       Forward pass of the VAE module.
       
       Args:
           x (Tensor): Input EEG data.
           
       Returns:
           tuple: Tuple containing the mean and log variance of the latent space and the reconstructed EEG data.
       """

        z_mean,log_var,z = self.encoder.forward(x)
        
        #output = self.classifier.forward(z)
        
        reconstruction = self.decoder.forward(z)
        
        return z_mean,log_var,reconstruction
#%%
class Loss():
    """
   Loss function class for calculating VAE loss.
   """
    
    @staticmethod
    def kl_divergence(z_mean, log_var):
        return -0.5 * torch.sum(1 + log_var - z_mean.pow(2) - torch.exp(log_var), dim=1).mean()

    @staticmethod
    def reconstruction_loss(x_reconstructed, x):
        return nn.MSELoss()(x_reconstructed, x)

    @staticmethod
    def classification_loss(y_pred, y_true):
        return nn.BCELoss()(y_pred, y_true)


    
    def vae_loss(self,pred,true):
        """
        Calculate VAE loss.
        
        Args:
            pred (tuple): Tuple containing the mean and log variance of the latent space and the reconstructed data.
            true (Tensor): True EEG data.
            
        Returns:
            Tensor: VAE loss.
        """

        z_mean,log_var,reconstruction_x =pred
        #print(output,label)
        recon_loss =  self.reconstruction_loss(reconstruction_x, true)
        
        kld_loss = self.kl_divergence(z_mean, log_var)
        
        #class_loss = self.classification_loss(output,label)
        
        return recon_loss + kld_loss
#%%

class train():
    """
   Trainer class for training the VAE model.
   
   Args:
       hidden_dimensions (list): List of hidden layer dimensions for the encoder and decoder networks.
       latent_embedding (int): Dimensionality of the latent space.
       device (str): Device to use for computations ('cuda' or 'cpu').
       train_data (TensorDataset): Training dataset.
       max_grad_norm (float): Maximum gradient norm for gradient clipping.
       lr (float, optional): Learning rate for optimization.
       epochs (int, optional): Number of training epochs.
       patience (int, optional): Patience for early stopping.
   """
   
    def __init__(self,hidden_dimensions,latent_embedding,device,train_data,max_grad_norm,lr=1e-4,epochs=1000,patience=10):
   
        self.lr =  lr
        self.epochs =epochs
        self.device = device
        self.latent_embedding =  latent_embedding
        self.hidden_dimensions =hidden_dimensions
        self.train_data = train_data
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.max_grad_norm= max_grad_norm
        self.train_loader =  self.dataloader()
        
    def dataloader(self):
        """
        Generates a DataLoader from the train dataset.
        
        Returns
        -------
        train_loader(DataLoader) : Uses torch's DataLoader and converts the train dataset
        into a DataLoader.

        """
        train_loader = DataLoader(self.train_data,batch_size=15,shuffle=True,drop_last=True)
        return train_loader
    
    def training(self):
        """
        Training method for the VAE model.
        
        Returns:
            tuple: Tuple containing the trained encoder, training losses, and epoch losses.
        """
        encoder = Encoder(84,self.hidden_dimensions, self.latent_embedding,self.device)
        decoder = Decoder(self.latent_embedding, 84)
        vae = VAE(encoder,decoder, self.device)
        los = Loss()
       
        optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()),
                                         lr =self.lr)
        
        train_loss = []
        epoch_loss = []
        accs = []
        classification_loss =[]
        for epoch in range(self.epochs):
            vae.to(self.device)
            vae.train()
            correct = 0
            total_samples=0
            for batch_id, (data,target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                #print(data.shape)
                optimizer.zero_grad()
                
                pred= vae.forward(data)
                #_,_,output,_  =  pred
           
                loss = los.vae_loss(pred,data.float())
                
                #class_loss.backward()
                #print(class_loss)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()), self.max_grad_norm)
                optimizer.step()
                
                train_loss.append(loss.item())
                #classification_loss.append(class_loss.item())
              
                #b_correct= torch.sum(torch.abs(output-target) <0.5).item()
                #correct += b_correct
                #total_samples += data.size(0)
                
            #acc = float(correct)/len(self.train_data)
            #accs.append(acc)   
            epoch_loss.append(np.mean(train_loss))
            
    
        

            print('Beginning Epoch',epoch)
            #print('Classification Accuracy:',acc,sep =':')
            print('Average Combined Loss:',np.mean(epoch_loss[-1]))
            #print('Average Classification Loss:',np.mean(classification_loss),sep=':')
            print('----------------------------------------------------------------')
    
       
            """if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f'Early stopping at epoch {epoch}')
                    break"""
        
        plt.figure(figsize=(10,6))
        #plt.plot(list(range(self.epochs)),accs,label = 'Accuracy')
        plt.plot(list(range(self.epochs)),epoch_loss,label = 'Loss')
        plt.legend()
        plt.title('VAE Training')
        plt.tight_layout()
        plt.show()
        plt.savefig('encoder_acc_loss.png')
        return encoder,train_loss,epoch_loss
#%%
class latent_training():
    """
    Trainer class for training the classifier using latent representations.
    
    Args:
        encoder (Encoder): Encoder module for extracting latent representations.
        embedding_dimension (int): Dimensionality of the latent space.
        hidden_dim (list): List of hidden layer dimensions for the classifier network.
        target_dim (int): Dimensionality of the target space.
        train_data (TensorDataset): Training dataset.
        device (str): Device to use for computations ('cuda' or 'cpu').
        loss (nn.Module, optional): Loss function for training.
        lr (float, optional): Learning rate for optimization.
        epochs (int, optional): Number of training epochs.
    """
    
    def __init__(self,encoder,embedding_dimension,hidden_dim,target_dim,train_data,device,loss=nn.BCELoss(),lr =1e-2, epochs = 30):
        self.classifier = Classifier(embedding_dimension, hidden_dim, target_dim)
        self.train_data = train_data
        self.train_loader = self.data_loader()
        self.epochs = epochs
        self.encoder = encoder
        self.lr = lr
        self.loss = loss
        self.device = device
        
    def data_loader(self):
        """
        Generates a DataLoader from the train dataset.
        
        Returns
        -------
        train_loader(DataLoader) : Uses torch's DataLoader and converts the train dataset
        into a DataLoader.

        """
        return DataLoader(self.train_data,batch_size=15,shuffle=True,drop_last=True)
    
    def train(self):
        """
        Training method for the classifier using latent representations.
        
        Returns:
            tuple: Tuple containing the trained classifier, accuracies, and epoch losses.
        """
        
        train_losses=[]
        accs = []
        epoch_loss= []
        
        preds = []
        targets = []
        optimizer = optim.Adam(self.classifier.parameters(),lr=self.lr)
        
        self.classifier.to(self.device)
        self.encoder.eval()
        self.classifier.train()
        
        for epoch in range(self.epochs):    
            
            correct = 0
            total_samples =0    
            for batch_id, (data,target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                #print(data.shape)
                optimizer.zero_grad()
                
                z = self.encoder(data)
                
                #_,_,output,_  =  pred
                output = self.classifier(z[2])
                preds.extend(output)
                targets.extend(target)
                
                train_loss = self.loss(output,target.float())
                
                #class_loss.backward()
                #print(class_loss)
                train_loss.backward()
                
                #torch.nn.utils.clip_grad_norm_(, self.max_grad_norm)
                optimizer.step()
                
                train_losses.append(train_loss.item())
               
                
                predicted_labels = output.round()  # Assuming binary classification

        # Calculate number of correct predictions
                correct += (predicted_labels == target).sum().item()
                total_samples += target.size(0)
                
            # Calculate accuracy for the epoch
            acc = correct / total_samples
            accs.append(acc)
            epoch_loss.append(np.mean(train_losses[-len(self.train_loader):]))  # Average loss for the epoch
        
            print(f'Beginning Epoch {epoch}')
            #print(f'Classifier Accuracy: {acc:.4f}')
            print(f'Classifier Average Loss: {np.mean(train_losses[-len(self.train_loader):]):.4f}')
            print('----------------------------------------------------------------')

            
        plt.figure(figsize=(10,6))
        #plt.plot(list(range(self.epochs)),accs,label = 'Accuracy')
        plt.plot(list(range(self.epochs)),epoch_loss,label = 'Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('classifier_acc_loss.png')
        return self.classifier,accs,epoch_loss
#%%
class randomforest():
    """
    Random forest classifier for anxiety level prediction.
    
    Args:
        n_estimators (int): Number of trees in the forest.
        criterion (str): Criterion for measuring the quality of a split.
        max_depth (int): Maximum depth of the trees.
        scoring (str): Scoring metric for model evaluation.
    """
    
    def __init__(self,n_estimators,criterion,max_depth,scoring):
        self.n_estimators = n_estimators
        self.criterion =  criterion
        self.max_depth = max_depth
        #self.class_weight = class_weight
        self.scoring = scoring
        self.forest = rfc(criterion=self.criterion)
        self.grid  = gsv(self.forest,[{'n_estimators':n_estimators},{'max_depth':self.max_depth}],scoring=self.scoring)
    
    def r_forest(self,x_train,y_train,x_test,y_test):
        """
        Train and evaluate the random forest classifier.
        
        Args:
            x_train (array_like): Training data features.
            y_train (array_like): Training data labels.
            x_test (array_like): Test data features.
            y_test (array_like): Test data labels.
        
        Returns:
            model: Trained random forest model.
            best_params: Best parameters found during grid search.
            test_accuracy: Accuracy of the model on the test data.
        """
        
        x_train = x_train.reshape(x_train.shape[0],-1)
        x_test = x_test.reshape(x_test.shape[0],-1)
        print(x_train.shape,x_test.shape)
        
        self.grid.fit(x_train,y_train)
        
        model = self.grid.best_estimator_
        
        best_params = self.grid.best_params_
        
        train_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train,train_pred)
        
        test_pred = model.predict(x_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        train_labels = np.argmax(y_train,axis =1)
        train_pred_labels = np.argmax(train_pred,axis =1 )
        
        test_labels = np.argmax(y_test,axis=1)
        pred_labels = np.argmax(test_pred,axis=1)
        #print(pred_labels)
        
        test_conf = confusion_matrix(test_labels, pred_labels)
        train_conf = confusion_matrix(train_labels, train_pred_labels)
        
        fpr, tpr, thresholds = roc_curve(y_test.ravel(), test_pred.ravel())
        roc_auc = auc(fpr, tpr)
        
        train_precision = precision_score(train_labels, train_pred_labels, average=None)
        train_sensitivity = recall_score(train_labels, train_pred_labels, average=None)
        
        test_precision = precision_score(test_labels, pred_labels,average =None)
        test_sensitivity = recall_score(test_labels, pred_labels, average=None)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)

        sns.heatmap(test_conf, annot=True, cmap='magma', xticklabels=['light', 'moderate', 'normal', 'severe'], yticklabels=['light', 'moderate', 'normal', 'severe'], ax=ax1)
        ax1.set_title('Test Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        sns.heatmap(train_conf, annot=True, cmap='vlag', xticklabels=['light', 'moderate', 'normal', 'severe'], yticklabels=['light', 'moderate', 'normal', 'severe'], ax=ax2)
        ax2.set_title('Train Confusion Matrix')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('Rforest_confusion_matrices.png')
        plt.show()
        
        
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Rforest Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig('Rforest_roc_curve.png')
        plt.show()
        
        print('Rforest Test Accuracy:', test_accuracy)
        print('Rforest Train Accuracy:',train_acc)
        print('Rforest Train Precision:', train_precision)
        print('Rforest Test Precision:',test_precision)
        print('Rforest Train Sensitivity:',train_sensitivity)
        print('Rforest Test Sensitivity:',test_sensitivity)
        
        return model,best_params,test_accuracy
    
    def latent_forest(self,encoder,best_rf_model,x_train,y_train,x_test,y_test):
        """
       Train and evaluate the random forest classifier using latent representations.
       
       Args:
           encoder: Encoder model for obtaining latent representations.
           best_rf_model: Best random forest model obtained from previous training.
           x_train (array_like): Training data features.
           y_train (array_like): Training data labels.
           x_test (array_like): Test data features.
           y_test (array_like): Test data labels.
       
       Returns:
           best_rf_model: Trained random forest model.
           test_accuracy: Accuracy of the model on the test data.
       """
        
        x_train = x_train.reshape(x_train.shape[0],-1)
        x_test = x_test.reshape(x_test.shape[0],-1)
        x_train_tens,x_test_tens = torch.tensor(x_train),torch.tensor(x_test)
        print(x_train_tens.shape)
        z_mean,_, z_train = encoder(x_train_tens)
        
        z_train = np.asarray(z_train.detach())
        
        best_rf_model.fit(z_train,y_train)
        
        train_pred = best_rf_model.predict(z_train)
        train_acc = accuracy_score(y_train,train_pred)
        
        z_m,_,z_test = encoder(x_test_tens)
        
        z_test= np.asarray(z_test.detach())
        
        test_pred = best_rf_model.predict(z_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        train_labels = np.argmax(y_train,axis =1)
        train_pred_labels = np.argmax(train_pred,axis =1 )
        
        test_labels = np.argmax(y_test,axis=1)
        pred_labels = np.argmax(test_pred,axis=1)
        
        test_conf = confusion_matrix(test_labels, pred_labels)
        train_conf = confusion_matrix(train_labels, train_pred_labels)
        
        train_precision = precision_score(train_labels, train_pred_labels, average=None)
        train_sensitivity = recall_score(train_labels, train_pred_labels, average=None)
        
        test_precision = precision_score(test_labels, pred_labels,average =None)
        test_sensitivity = recall_score(test_labels, pred_labels, average=None)
        
        fpr, tpr, thresholds = roc_curve(y_test.ravel(), test_pred.ravel())
        roc_auc = auc(fpr, tpr)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)

        sns.heatmap(test_conf, annot=True, cmap='magma', xticklabels=['light', 'moderate', 'normal', 'severe'], yticklabels=['light', 'moderate', 'normal', 'severe'], ax=ax1)
        ax1.set_title('Test Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        sns.heatmap(train_conf, annot=True, cmap='vlag', xticklabels=['light', 'moderate', 'normal', 'severe'], yticklabels=['light', 'moderate', 'normal', 'severe'], ax=ax2)
        ax2.set_title('Train Confusion Matrix')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('latentforest_confusion_matrices.png')
        plt.show()
        
        
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('LatentForest Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig('latentforest_roc_curve.png')
        plt.show()
        
        print('LatentForest Test Accuracy:', test_accuracy)
        print('LatentForest Train Accuracy:',train_acc)
        print('LatentForest Train Precision:', train_precision)
        print('LatentForest Test Precision:',test_precision)
        print('LatentForest Train Sensitivity:',train_sensitivity)
        print('LatentForest Test Sensitivity:',test_sensitivity)
        
        return best_rf_model,test_accuracy
        
#%%
class SVClassifier:
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.classifier = SVC()
    
    def train(self,x_train,y_train,x_test,y_test):
        
        x_train,x_test = x_train.reshape(x_train.shape[0],-1),x_test.reshape(x_test.shape[0],-1)
        train_labels_1d,test_labels_1d= np.argmax(y_train,axis=1), np.argmax(y_test,axis=1)
        
        self.classifier.fit(x_train,train_labels_1d)
        
        train_pred = self.classifier.predict(x_train)
        test_pred = self.classifier.predict(x_test)
        
        train_accuracy = accuracy_score(train_labels_1d, train_pred)
        test_accuracy = accuracy_score(test_labels_1d, test_pred)
        
        #train_pred_1d = np.argmax(train_pred,axis=1)
        #test_pred_1d = np.argmax(test_pred,axis=1)
        
        test_confusion_mat = confusion_matrix(test_labels_1d, test_pred)
        train_confusion_mat = confusion_matrix(train_labels_1d, train_pred)
        
        train_precision = precision_score(train_labels_1d, train_pred, average=None)
        train_sensitivity = recall_score(train_labels_1d, train_pred, average=None)
        
        test_precision = precision_score(test_labels_1d, test_pred, average=None)
        test_sensitivity = recall_score(test_labels_1d, test_pred, average=None)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)

        sns.heatmap(test_confusion_mat, annot=True, cmap='magma', xticklabels=['light', 'moderate', 'normal', 'severe'], yticklabels=['light', 'moderate', 'normal', 'severe'], ax=ax1)
        ax1.set_title('Test Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        sns.heatmap(train_confusion_mat, annot=True, cmap='vlag', xticklabels=['light', 'moderate', 'normal', 'severe'], yticklabels=['light', 'moderate', 'normal', 'severe'], ax=ax2)
        ax2.set_title('Train Confusion Matrix')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('svc_confusion_matrices.png')
        plt.show()
        
        print('SVC Test Accuracy:', test_accuracy)
        print('SVC Train Accuracy:',train_accuracy)
        print('SVC Train Precision:', train_precision)
        print('SVC Test Precision:',test_precision)
        print('SVC Train Sensitivity:',train_sensitivity)
        print('SVC Test Sensitivity:',test_sensitivity)
        
        
        
    def latent_train(self, train_data, train_labels):
        
        latent_train = self.encode(train_data)
        
        train_labels_1d = np.argmax(train_labels, axis=1)

        self.classifier.fit(latent_train, train_labels_1d)

    def latent_test(self, test_data, test_labels):
        
        latent_test = self.encode(test_data)
        predictions = self.classifier.predict(latent_test)
        
        test_labels_1d = np.argmax(test_labels,axis =1 )
        
        accuracy = accuracy_score(test_labels_1d, predictions)
        
        confusion_mat = confusion_matrix(test_labels_1d, predictions)
        precision = precision_score(test_labels_1d, predictions, average=None)
        sensitivity = recall_score(test_labels_1d, predictions, average=None)
        
        print('LatentSVC Test Accuracy of SVM:',accuracy)
        print('LatentSVC Precision Scores of SVM :',precision)
        print('LatentSVC Sensitivity of SVM:',sensitivity)
        
        return accuracy, confusion_mat, precision, sensitivity
        

    def encode(self, data):
        data = data.reshape(data.shape[0],-1)
        data= torch.tensor(data)
        with torch.no_grad():
            z_mean, _, z = self.encoder(data)
        return z.cpu().numpy()
    
    def plot_metrics(self, confusion_mat, precision, sensitivity):
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g',xticklabels=['light', 'moderate', 'normal', 'severe'], yticklabels=['light', 'moderate', 'normal', 'severe'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        plt.savefig('LatentSVC_confusion_matrix.png')
        
        # Plot precision
        plt.figure(figsize=(8, 6))
        plt.bar(np.arange(len(precision)), precision, color='skyblue')
        plt.title('Precision')
        plt.xlabel('Class')
        plt.ylabel('Precision Score')
        plt.xticks(np.arange(len(precision)))
        plt.show()
        plt.savefig('LatentSVC_Precision.png')

        # Plot sensitivity
        plt.figure(figsize=(8, 6))
        plt.bar(np.arange(len(sensitivity)), sensitivity, color='salmon')
        plt.title('Sensitivity')
        plt.xlabel('Class')
        plt.ylabel('Sensitivity Score')
        plt.xticks(np.arange(len(sensitivity)))
        plt.show()
        plt.savefig('LatentSVC_Sensitivity.png')
        
        
#%% Load preprocessed data.  This is the raw data contained in the .edf files after bandpass filtering and application of ICA

def load_data_epoch_anxiety_levels(directory, subjects):
    '''
    Assumption: data from the .mat files
    Description:  Draft
      
    Two methods of getting at the eeg data contained in the dataset: list, and numpy array.
    The numpy array is the preferred method.
    #TODO do I have the correct interpretation from our readme file, and the excel file?

    ds_arr; size [trials x samples x electrodes]; This is the processed 15-second EEG data from the 12 trials (6 situations * 2 runs) and 23 subjects.
    Processed by ICA artifact removal and bandpass filtered. 

    labels: Two columns for the subject Self-Assessment Manikin (SAM). One column is an event's positive or negative score for valence, 
    and the other is the arousal spectrum, from calmness to excitement. A combination of these two scores establishes anxiety levels. 
    #After transpose of the loaded row information.
    
    Parameters
    ----------
    directory : str
        Path to the directory containing the data files.
    subjects : list
        List of subject identifiers.

    Returns
    -------
    subjects_df : dict
        Dictionary containing DataFrame for each subject.
    '''
    subjects_df = {}
    counts = []
    for index, subject in enumerate(subjects):
        filename = f'{directory}//S{str(subject).rjust(2, "0") if subject < 10 else subject}.mat'
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]
            data = f['data'][:]
            labels = f['labels'][:]
            df, count_tuple = labelling(data, labels)
            
            if f'subject' in subjects_df.keys():
                subjects_df[f'subject'].append(df)
            else:
                subjects_df[f'{subject}'] = df
                
            if len(counts) == 0:
                counts.append(count_tuple)
            else:
                count = counts.pop()
                updated_count = tuple(map(sum, zip(count, count_tuple)))
                counts.append(updated_count)
    
    #print(counts)
    severe_count, moderate_count, light_count, normal_count = counts[0] 
    print('severe count:', severe_count)
    print('moderate_count:', moderate_count)
    print('light_count:', light_count)
    print('normal_count:', normal_count)
    
    return subjects_df

   

#%%
def plot_PSD (subject,electrodes,data, level,freq_band = [4,20], run = 1, sample_interval=15, fs =128):
    '''
    Visualizes the data Frequency Domain. First subtracts the mean then calculate the PSD, in (dB), for the defined interval.

    Parameters
    ----------
    subject : int
        Subject identifier.
    electrodes : list
        List of electrode channels.
    data : numpy array, shape [run, sampled data, electrode channel]
        EEG data.
    level : str
        Level of anxiety.
    freq_band : list, optional
        Specifies the start and end frequency, in Hz, to be evaluated for PSD. The default is [4, 20].
    run : int, optional
        Paradigm run of interest. The default is 1.
    sample_interval : int, optional
        End time for the sample interval starting from 0 seconds. The default is 15 seconds.
    fs : int, optional
        Sampling frequency.

    Returns
    -------
    PSD_band : numpy array
        Power Spectral Density in the specified frequency band.
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








