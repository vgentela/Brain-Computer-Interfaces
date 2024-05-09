# Brain-Computer-Interfaces

README for DASPS (Database for Anxious States based on a Psychological Stimulation)

This dataset can be used to explore the effects of stimuli on the anxiety levels of patients. The DASPS database contains recorded Electroencephalogram (EEG) signals of 23 participants during anxiety elicitation by means of face-to-face psychological stimuli.

The protocol for collecting this data consists of two stages: a psychotherapist reciting a stressful stimulus to the subject for 15 seconds and then having them imagine the situation for 15 seconds. Self-assessment establishes the stress levels the subject feels during this situation. This sequence is repeated for five more situations for a total of 6 runs. Baseline anxiety levels are established before and then again after testing.

We've extracted the data from two different formats : h5py dataset object and .mat format. The following two functions from the module [project_3.py](project_3.py) were used to perform data extraction.

**load_data_epoch_anxiety_levels**
``` 
This function extracts EEG data and labels from .mat files in the specified directory. The EEG data is stored in a dictionary with DataFrames for each subject.
```
**loadedf**
```
For each subject, it constructs the filename based on the subject ID and reads the .edf file using `mne.io.read_raw_edf`. It extracts the raw EEG data and channel names from the loaded data. It retrieves metadata about the EEG data using `data_edf.info`.
```
- The [project_3.py](project_3.py) contains various classes and functions necessary for performing data analysis and classification tasks.
- The [test_project3.py](test_project_3.py) modules uses the classes and function from the project_3.py module to perform the data analysis, data preprocessing, data labelling and classification tasks.

**Notes:**
- The Trials were performed with (12) 6 stimuli, in  2 stages (recitation, recall)
- electrodes: 14 (+2 references) sensors placed at this locations: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 with a sampling rate of 128 SPS (2048 Hz internal) and a resolution of 14 bits 1 LSB = 0.51μV. Emotiv Epoc is a wireless device with 2.4GHz band.
- A 4-45 hz Finite impulse response (FIR) pass-band filter was applied to the raw data
-The Automatic Artifact removal for EEGLAB toolbox (AAR) was used to remove EOG and EMG artifacts.

**Fixes**
- Issue with the colorbar of scalp maps: the scalp maps generate an empty figure and a figure with content.(This should be fixed).
- There are a few hard coded hyperparameters in neural networks and dataloaders. It should be made flexible.
