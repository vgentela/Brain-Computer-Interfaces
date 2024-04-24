# -*- coding: utf-8 -*-
"""
import_ssvep_data.py

Load data and plot frequency specturm of steady-state visual evoked potentials (SSVEPs).
BME6710 BCI Spring 2024 Lab #3

The SSVEP dataset is derived from a tutorial in the MNE-Python package.The dataset
includes electropencephalograpy (EEG) data from 32 channels collected during
a visual checkerboard experiment, where the checkerboard flickered at 12 Hz or
15 Hz. 

The functions in this module can be used to load the dataset into variables, 
plot the raw data, epoch the data, calculate the Fourier Transform (FT), 
and plot the power spectra for each subject. 

Created on Feb 24 2024

@author: 
    Ardyn Olszko
    Yoonki Hong
"""

#%% Part 1: Load the Data

# import packages

import numpy as np
from matplotlib import pyplot as plt
import scipy.fft 

# function to load data
def load_ssvep_data(subject, data_directory):
    '''
    Load the SSVEP EEG data for a given subject.
    The format of the data is described in the README.md file with the data.

    Parameters
    ----------
    subject : int
        Subject number, 1 or 2.
    data_directory : str
        Path to the folder where the data files exist.

    Returns
    -------
    data_dict : dict
        Dictionary of data for a subject.

    '''
    
    # Load dictionary
    data_dict = np.load(data_directory + f'/SSVEP_S{subject}.npz',allow_pickle=True)    

    return data_dict

#%% Part 2: Plot the Data

# function to plot the raw data
def plot_raw_data(data,subject,channels_to_plot):
    '''
    Plot events and raw data from specified electrodes.
    Creates a figure with two subplots, where the first is the events and the
    second is the EEG voltage for specified channels. The figure is saved to
    the current directory.

    Parameters
    ----------
    data : dict
        Dictionary of data for a subject.
    subject : int
        Subject number, 1 or 2 (used to annotate plot)
    channels_to_plot : list or array of size n where n is the number of channels
        Channel names of data to plot. Channel name must be in "data['channels']".

    Returns
    -------
    None.

    '''
    
    # extract variables from dictionary
    eeg = data['eeg'] # eeg data in Volts. Each row is a channel and each column is a sample.
    channels = data['channels'] # name of each channel, in the same order as the eeg matrix.
    fs = data['fs'] # sampling frequency in Hz.
    event_samples = data['event_samples'] # sample when each event occurred.
    event_durations = data['event_durations'] # durations of each event in samples.
    event_types = data['event_types'] # frequency of flickering checkerboard for each event.
    
    # calculate time array
    time = np.arange(0,1/fs*eeg.shape[1],1/fs)
    
    # set up figure
    plt.figure(f'raw subject{subject}',clear=True)
    plt.suptitle(f'SSVEP subject {subject} raw data')
    
    # plot the event start and end times and types
    ax1 = plt.subplot(2,1,1)
    start_times = time[event_samples]
    end_times = time[event_samples+event_durations.astype(int)]
    for event_type in np.unique(event_types):
        is_event = event_types == event_type
        plt.plot()
        event_data = np.array([start_times[is_event],end_times[is_event]])
        plt.plot(event_data,
                 np.full_like(event_data,float(event_type[:-2])),
                 marker='o',linestyle='-',color='b',
                 label=event_type)
    plt.xlabel('time (s)')
    plt.ylabel('flash frequency (Hz)')
    plt.grid()
    
    # plot the raw data from the channels spcified
    plt.subplot(2,1,2, sharex=ax1)
    for channel in channels_to_plot:
        is_channel = channels == channel
        plt.plot(time, 10e5*eeg[is_channel,:].transpose(),label=channel) # multiply by 10e5 to convert to uV (confirmed that this matches the original dataset from mne)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (uV)')
    plt.grid()
    plt.legend()
    
    # save the figure
    plt.tight_layout()
    plt.savefig(f'SSVEP_S{subject}_rawdata.png')
    
    return


#%% Part 3: Extract the Epochs

# function to epoch the data
def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20):
    '''
    Epoch the EEG data.

    Parameters
    ----------
    data_dict : dict
        Dictionary of data for a subject.
    epoch_start_time : int or float, optional
        Start time of the epoch relative the the event time. Units in seconds. The default is 0.
    epoch_end_time : int or float, optional
        End time of the epoch relative the the event time. Units in seconds. The default is 20.

    Returns
    -------
    eeg_epochs : array of float, size M x N x T where M is the number of trials,
    N is the number of channels, and T is the number of samples in the epoch
        EEG data after each epoch. Units in uV.
    epoch_times : array of float, size T where T is the number of samples
        Time (relative to the event) of each sample. Units in seconds.
    is_trial_15Hz : array of bool, size M where M is the number of trials
        Event label, where True is 15 Hz event and False is 12 Hz event.

    '''
    
    # extract relevant variables from dictionary
    eeg_raw = data_dict['eeg'] # eeg data in Volts. Each row is a channel and each column is a sample. (not sure data are actually in Volts)
    fs = data_dict['fs'] # sampling frequency in Hz.
    event_samples = data_dict['event_samples'] # sample when each event occurred.
    event_types = data_dict['event_types'] # frequency of flickering checkerboard for each event. 

    # convert eeg data to uV
    eeg = 10e5*eeg_raw # multiply by 10e5 to convert to uV (confirmed that this matches the original dataset from mne)
    
    # define boolean for event types
    is_trial_15Hz = event_types == '15hz'
    
    # create time array for each epoch
    epoch_times = np.arange(epoch_start_time,epoch_end_time,1/fs)
    
    # define size of 3d array of epochs
    epoch_count = len(event_samples) 
    #sample_count = len(epoch_times)  # commented out in project 2
    channel_count = eeg.shape[0]
       
    # calculate end samples for all epochs
    start_samples = event_samples + int(epoch_start_time*fs)
    end_samples = event_samples + int(epoch_end_time*fs)
    
    # adjust any start and end samples that occur outside the times of the available data
    start_samples[start_samples<0] = 0
    end_samples[end_samples>eeg.shape[1]]=eeg.shape[1]
    
    sample_count = (end_samples-start_samples)[0] # added in project2
    
    # define 3d array for epoch data
    eeg_epochs = np.full((epoch_count,channel_count,sample_count),np.nan,dtype='float32')
    
    # fill in the epoch data
    for epoch_index in np.arange(epoch_count):
        eeg_epochs[epoch_index] = eeg[:, start_samples[epoch_index]:end_samples[epoch_index]]
    
    return eeg_epochs, epoch_times, is_trial_15Hz

#%% Part 4: Take the Fourier Transform

# function to calculate frequency spectrum
def get_frequency_spectrum(eeg_epochs, fs):
    '''
    Calcluate the FT each channel in each epoch.

    Parameters
    ----------
    eeg_epochs : array of float, size M x N x T where M is the number of trials,
    N is the number of channels, and T is the number of samples in the epoch
        EEG data after each epoch. Units in uV.
    fs : int or float
        Sampling frequency of the EEG data.

    Returns
    -------
    eeg_epochs_fft : array of float, size M x N X F where M is the number of trials,
    N is the number of channels, and F is number of frequencies measured in the epoch
        FT frequency content of each channel in each epoch .
    fft_frequencies : array of float, size F
        Frequencies measured, where the maximum frequency measured is 1/2*fs.

    '''
    
    # calculate FT on each channel
    eeg_epochs_fft = scipy.fft.rfft(eeg_epochs)
    
    # calculate frequencies
    sample_count = eeg_epochs.shape[2]
    total_duration = sample_count/fs
    fft_frequencies = np.arange(0,eeg_epochs_fft.shape[2])/total_duration
    
    return eeg_epochs_fft, fft_frequencies

#%% Part 5: Plot the Power Spectra

# function to plot the mean power spectra for specified channesl
def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject):
    '''
    Calculate and plot the mean power spectra for specified channels.
    Each channel is plotted on a separate subplot. Event types, 12 Hz and 15 Hz,
    are plotted separately for each channel.

    Parameters
    ----------
    eeg_epochs_fft : array of float, size M x N X F where M is the number of trials,
    N is the number of channels, and F is number of frequencies measured in the epoch
        FT frequency content of each channel in each epoch.
    fft_frequencies : array of float, size F
        Frequencies measured, where the maximum frequency measured is 1/2*fs.
    is_trial_15Hz : array of bool, size M where M is the number of trials
        Event label, where True is 15 Hz event and False is 12 Hz event.
    channels : list of size N where N is the number of channels
        Channel names available in  the original dataset.
    channels_to_plot : list or array of size n where n is the number of channels
        Channel names of data to plot. Channel name must be in "data['channels']".
    subject : int
        Subject number, 1 or 2 (used to annotate plot)
        
    Returns
    -------
    spectrum_db_12Hz : array of float, size n x F where n is the number of channels
    and F is the number of frequencies
        Mean power spectrum of 12 Hz trials. Units in dB.
    spectrum_db_15Hz : array of float, size n x F where n is the number of channels
    and F is the number of frequencies
        Mean power spectrum of 15 Hz trials. Units in dB.

    '''
    
    # calculate mean power spectra for each channel
    signal_power = abs(eeg_epochs_fft)**2 # calculate power by squaring absolute value
    # calculate mean across trials
    power_12Hz = np.mean(signal_power[~is_trial_15Hz],axis=0)
    power_15Hz = np.mean(signal_power[is_trial_15Hz],axis=0)
    
    # normalize (divide by max value)
    norm_power_12Hz = power_12Hz/np.reshape(np.max(power_12Hz, axis=1), (power_12Hz.shape[0],1))
    norm_power_15Hz = power_15Hz/np.reshape(np.max(power_15Hz, axis=1), (power_12Hz.shape[0],1))
    
    # convert to decibel units
    power_db_12Hz = 10*np.log10(norm_power_12Hz)
    power_db_15Hz = 10*np.log10(norm_power_15Hz)
    
    # set up figure and arrays for mean power spectra
    channel_count = len(channels_to_plot)
    freq_count = len(fft_frequencies)
    spectrum_db_12Hz = np.full([channel_count,freq_count],np.nan,dtype=float) # set up arrays to store power spectrum
    spectrum_db_15Hz = np.full_like(spectrum_db_12Hz,np.nan)
    row_count = int(np.ceil(np.sqrt(channel_count))) # calculate number of rows of subplots
    if (row_count**2 - channel_count) >= row_count: # calculate number of columns of subplots
        col_count = row_count-1 
    else:
        col_count = row_count

    fig = plt.figure(f'spectrum subject{subject}',clear=True,figsize=(6+0.5*channel_count,6+0.5*channel_count))
    plt.suptitle(f'Frequency content for SSVEP subject {subject}')
    axs=[] # set up empty list for subplot axes
    
    # plot and extract data for specified channels
    for channel_index, channel in enumerate(channels_to_plot):
        is_channel = channels == channel
        spectrum_db_12Hz[channel_index,:] = power_db_12Hz[is_channel,:]
        spectrum_db_15Hz[channel_index,:] = power_db_15Hz[is_channel,:]
        
        if channel_index == 0: 
            axs.append(fig.add_subplot(row_count,col_count,channel_index+1))
        else:
            axs.append(fig.add_subplot(row_count,col_count,channel_index+1,
                                       sharex=axs[0],
                                       sharey=axs[0]))
        # plot the mean power spectra
        axs[channel_index].plot(fft_frequencies,spectrum_db_12Hz[channel_index,:],label='12Hz',color='r')
        axs[channel_index].plot(fft_frequencies,spectrum_db_15Hz[channel_index,:],label='15Hz',color='g')
        # plot corresponding frequencies
        axs[channel_index].axvline(12,color='r',linestyle=':')
        axs[channel_index].axvline(15,color='g',linestyle=':')
        # annotate
        axs[channel_index].set_title(channel)
        axs[channel_index].set(xlabel='frequency (Hz)',ylabel='power (db)')
        axs[channel_index].grid()
        axs[channel_index].legend()
           
    plt.tight_layout()
    
    return spectrum_db_12Hz, spectrum_db_15Hz
