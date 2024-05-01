# Brain-Computer-Interfaces

README for DASPS (Database for Anxious States based on a Psychological Stimulation)

This dataset can be used to explore the effects of stimuli on the anxiety levels of patients. The DASPS database contains recorded Electroencephalogram (EEG) signals of 23 participants during anxiety elicitation by means of face-to-face psychological stimuli.

The protocol for collecting this data consists of two stages: a psychotherapist reciting a stressful stimulus to the subject for 15 seconds and then having them imagine the situation for 15 seconds. Self-assessment establishes the stress levels the subject feels during this situation. This sequence is repeated for five more situations for a total of 6 runs. Baseline anxiety levels are established before and then again after testing.

Weve extracted the data into an h5py dataset object using the following code:

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


Keys:
- situation: a string describing the stimuli using the subject's words.  
- hamilton: psychotherapist rating using the Hamilton Anxiety Rating Scale (HAM-A) . Total score 0-56 for 14 evaluation categories. 
- labels: Two columns for the subject Self-Assessment Manikin (SAM). One column is an event's positive or negative score for valence, and the other is the arousal spectrum, from calmness to excitement. A combination of these two scores establishes anxiety levels.   
- data: array size [ trial x samples x channels].  
	Electrode: (14): AF3, AF4, F3, F4, FC5, FC6, F7, F8, T7, T8, P7, P8, O1, O2
	Samples: (1920) 15 second * fs = 1920. fs = 128 Hz. Electrode voltage in uV 

**Notes:**
- The Trials were performed with (12) 6 stimuli, in  2 stages (recitation, recall)
- electrodes: 14 (+2 references) sensors placed at this locations: AF3, AF4, F3, F4, FC5, FC6, F7, F8, T7, T8, P7, P8, O1, O2 with a sampling rate of 128 SPS (2048 Hz internal) and a resolution of 14 bits 1 LSB = 0.51μV. Emotiv Epoc is a wireless device with 2.4GHz band.
- A 4-45 hz Finite impulse response (FIR) pass-band filter was applied to the raw data
-The Automatic Artifact removal for EEGLAB toolbox (AAR) was used to remove EOG and EMG artifacts.
