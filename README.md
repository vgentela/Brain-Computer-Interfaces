# Brain-Computer-Interfaces

README for DASPS (Database for Anxious States based on a Psychological Stimulation)

This dataset can be used to explore the effects of stimuli on the anxiety levels of patients. The DASPS database contains recorded Electroencephalogram (EEG) signals of 23 participants during anxiety elicitation by means of face-to-face psychological stimuli.

The protocol for collecting this data consists of two stages: a psychotherapist reciting a stressful stimulus to the subject for 15 seconds and then having them imagine the situation for 15 seconds. Self-assessment establishes the stress levels the subject feels during this situation. This sequence is repeated for five more situations for a total of 6 runs. Baseline anxiety levels are established before and then again after testing.

We've extracted the data from two different formats : h5py dataset object and .mat format. The following functions from the module [project_3.py](project_3.py) were used to perform data extraction.
``` 
`load_data_epoch_anxiety_levels` is a function that takes directory(the path to data),subjects and electrodes(channels from which EEG data is to be extracted) as parameters and returns all the data pertaining to a subject(an array), the counts of subjects experiencing different degrees of anxiety and labels for classification. It also plots the Power Spectral Density of the EEG data in the 4-20Hz frequency band.

`loadedf` is a function that takes directory and subjects as parameters and returns data across all the channels, names of channels and a dictionary of key: value pairs where keys represent different attributes of the retieved data that can be accessed.
```

`plot_edf_data` is a fuction that plots the Power Spectral Density of the EEG data from the edf file. This function takes raw_data_edf, electrode_index(the channels to plot),fs_edf(sampling frequency).

`plot_scalp_map` is a function that plots the spatial maps of the EEG activity in the brain. It takes subject, electrodes, data as parameters along with data_type = '.edf', run (corresponds to the situations with values from 1 to 12), method = 'mean'(describes the statistical method to be applied on the data), domain(domain of the features to be plotted).

`clear_all_figures` is a function that closes all the open plotted figures.

`plot_PSD` is a funtion that visualizes the data in frequency domain. It subtracts the mean from the data and plots the PSD in the defined interval. It takes subject,electrodes,data as parameters along with freq_band(the frequency band to filter the data),run and sample_interval(the upper bound of the time intreval for plotting),fs as optional parameters.


**Notes:**
- The Trials were performed with (12) 6 stimuli, in  2 stages (recitation, recall)
- electrodes: 14 (+2 references) sensors placed at this locations: AF3, AF4, F3, F4, FC5, FC6, F7, F8, T7, T8, P7, P8, O1, O2 with a sampling rate of 128 SPS (2048 Hz internal) and a resolution of 14 bits 1 LSB = 0.51μV. Emotiv Epoc is a wireless device with 2.4GHz band.
- A 4-45 hz Finite impulse response (FIR) pass-band filter was applied to the raw data
-The Automatic Artifact removal for EEGLAB toolbox (AAR) was used to remove EOG and EMG artifacts.
