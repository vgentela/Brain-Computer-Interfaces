#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_topo.py

Created on Mon Sep  6 13:06:10 2021

@author: djangraw
"""

# Import packages
import numpy as np
from matplotlib import pyplot as plt
import mne


# Declare main function
def plot_topo(channel_names=[], channel_data=[],title='',cbar_label='Voltage (uV)',montage_name='biosemi64'):
    """
    Plots a topomap (colored brain) of the specified channels and values.

    Parameters
    ----------
    channel_names : list/arr of strings, optional
        Channels to plot (must be . The default is [].
    channel_data : Arr of shape [len(channel_names),1], optional
        Voltages to plot on each channel. The default is [].
    title : str, optional
        Title to place above the plot. The default is ''.
    cbar_label : str, optional
        Label to go on the colorbar. The default is 'Voltage (uV)'.
    montage_name : str, optional
        Name of the channel montage to use (must be valid input to 
        mne.channels.make_standard_montage). The default is 'biosemi64'.
    
    Returns
    -------
    im : image
        Topoplot image object.
    cbar : colorbar
        Colorbar object associated with the image.

    """

    # create montage according to montage_name specs
    montage = mne.channels.make_standard_montage(montage_name)
    if len(channel_names)==0: # if no channel names were given
        channel_names = montage.ch_names # plot all by default
    n_channels = len(channel_names)
    # Create MNE info struct
    fake_info = mne.create_info(ch_names=channel_names, sfreq=250.,
                                ch_types='eeg')
    
    # Prepare data
    if len(channel_data)==0: # if no input was given
        channel_data = np.random.normal(size=(n_channels, 1)) # plot random data by default
    if channel_data.ndim==1: # if it's a 1D array
        channel_data = channel_data.reshape([-1,1]) # make it 2D
    
    # Create MNE evoked array object with our data & channel info
    fake_evoked = mne.EvokedArray(channel_data, fake_info)
    fake_evoked.set_montage(montage) # set montage (channel locations)

    # Clear current axes
    plt.cla()    
    # Plot topomap on current axes    
    im,_ = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info,show=True)
    # Annotate plot
    plt.title(title)
    cbar = plt.colorbar(im,label=cbar_label)
    
    # return image and colorbar objects
    return im,cbar


# Helper and QA functions
def get_channel_names(montage_name='biosemi64'):
    """
    Returns all the channels contained in a given montage. Useful for checking 
    capitalization conventions and subsets of channels found in a given montage.

    Parameters
    ----------
    montage_name : str, optional
        Name of the channel montage to use (must be valid input to 
        mne.channels.make_standard_montage). The default is 'biosemi64'.

    Returns
    -------
    arr of strings
        names of channels in the given montage.

    """
    # create montage
    montage = mne.channels.make_standard_montage(montage_name)
    # return channel names in that montage
    return montage.ch_names