# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:16:03 2024

@author: Luke Camarao2
"""

"""
Name:        test_project3.py
Description: Invokes the necessary routines to perform the requirements of BCI Project 3
"""
# %%  Import Modules, Load Data, and extract Epochs
# Import necessary modules

# %%




import numpy as np
import h5py
import matplotlib.pyplot as plt
import eeglib.eeg as eeg
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(subject, directory = f'Raw data.mat/'):
    file_path = f'{directory}S{str(subject).rjust(2,"0") if subject < 10 else subject}.mat'
    container = h5py.File(file_path)
    data, ham_scores, labels, situations = [
        container.get(i) for i in container.keys()]
    return data, ham_scores, labels, situations


# %%
data, ham_scores, labels, situations = load_data(1)

# %%


def trials(data):

    trial_dic = {}

    for _ in range(len(data)):
        trial_dic[f'trial_{_}'] = np.vstack(data[_])

    return trial_dic


# %%
def label(data):

    label_dic = {}

    for _ in range(len(data)):
        label_dic[f'label_{_}'] = np.vstack(data[_])

    return label_dic


label_dic = label(labels)
#%%labls into arrays
labels_ar= np.empty(shape=(12,2))
for val in range(2):
    key = f'label_{val}'
    for labels_ind in range(12):
        labels_ar[labels_ind,val]= label_dic[key][labels_ind]

# label_str= np.empty(shape=12)
    
# for index,val in enumerate(labels_ar):
    
#     for index,ar in enumerate(labels_ar):
#         if val <=5 & ar>=5:
#             if val >
            
            
        

# %% hjorth activity:


def hjorth_params(data_dic):
    params = {}
    param_list = [eeg.hjorthActivity, eeg.hjorthComplexity, eeg.hjorthMobility]
    print(param_list)
    for _ in range(len(data_dic)):
        params[f'params_{_}'] = np.concatenate([np.apply_along_axis(
            i, 0, data_dic.get(f'trial_{_}')) for i in param_list]).T
    return params


# %%
trial_dic = trials(data)
params = hjorth_params(trial_dic)
# %%
df = pd.DataFrame(np.array(list(params.values())), index=params.keys())

# %%
df['valence'] = labels[0]
df['arousal'] = labels[1]
# %%
corr = df.corr()

sns.heatmap(corr)
plt.savefig('correlation_heatmap')

# %%




labels_sub0 = ['severe anxiety','severe anxiety','moderate anxiety','moderate anxiety'
               , 'severe anxiety', 'severe anxiety', 'light anxiety', 'light anxiety',
               'light anxiety', 'light anxiety', 'severe anxiety', 'severe anxiety']




final_pca = {}
power_spec = {}
epochs = np.empty(shape=(12, 1920, 14))

for trial in range(12):
    key = f'epoch_{trial}'

    epoch = trial_dic[f'trial_{trial}']
    epochs[trial] = epoch

    freq = 10*np.log10((np.fft.rfft(epoch, axis=0))**2)
    power_spec[key] = freq

    standardized_data = (freq - np.mean(freq, axis=0)) / np.std(freq, axis=0)

    # cov_matrix = np.cov(standardized_data, rowvar=False)  # Setting rowvar=False treats each column as a variable

    # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # percentage_variance = eigenvalues/np.sum(eigenvalues)

    # feature_vector = eigenvectors[0,:]

    # final_data = feature_vector.T * standardized_data[2,:].T

    pca = PCA(.95)

    pca.fit(np.abs(standardized_data))
    print(pca.n_components_)
    print(pca.explained_variance_ratio_)

    trial_img = pca.transform(np.abs(standardized_data))

    final_pca[key] = trial_img
    
    
train_img, test_img, train_lbl, test_lbl = train_test_split( epochs, labels_sub0, test_size=1/7.0, random_state=0)





#%%

scaler = StandardScaler()



pca = PCA(.95)
for epoch_ind_train in range(len(train_img)):
    # Fit on training set only.
    scaler.fit(train_img[epoch_ind_train,:,:])
    
    # Apply transform to both the training set and the test set.
    train_img_scalar = scaler.transform(train_img[epoch_ind_train,:,:])
    for epoch_ind_test in range(len(test_img)):
        test_img_scalar = scaler.transform(test_img[epoch_ind_test,:,:])
    
    

#%%
# scaler = StandardScaler()

# # Fit on training set only.
# scaler.fit(train_img)

# # Apply transform to both the training set and the test set.
# train_img = scaler.transform(train_img)
# test_img = scaler.transform(test_img)




pca = PCA(.95)

pca.fit(train_img_scalar)
pca.get_feature_names_out(input_features=labels_sub0)
train_img = pca.transform(train_img_scalar)
test_img =pca.transform(test_img_scalar)
#%%
from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl[:len(train_img)])

































