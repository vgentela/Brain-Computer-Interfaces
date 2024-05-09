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

# %%  Import Modules
import plot_topo as pt
import numpy as np

# TODO #c:\users\18023\anaconda3\lib\site-packages
import h5py

# import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
import mne as mne
import pandas as pd
from torcheeg import transforms as tfs
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    f1_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import GridSearchCV as gsv
from sklearn.svm import SVC

# %% Load .edf file


def loadedf(directory, subjects):
    """

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

    """

    subject = subjects
    for plot, subject in enumerate(subjects):

        # load the files using the directory and subject information
        filename = f"{directory}S{subject}.edf"
        data_edf = mne.io.read_raw_edf(filename)
        raw_data_edf = data_edf.get_data()
        channels_edf = data_edf.ch_names
        # summary of dataframe
        info = data_edf.info

    return raw_data_edf, channels_edf, info


# %% Visualize the data


def plot_edf_data(raw_data_edf, electrode_index=(2, 16), subjects=1, run=1, fs_edf=128):
    """

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

    """

    plt.figure(num=200, figsize=(8, 6), clear=all)
    for index in range(2, 16):  # raw data valid eed electrode channels index 2 - 15
        print(index)
        time_edf = np.arange(0, len(raw_data_edf[0, :]) / fs_edf, 1 / fs_edf)
        plt.plot(
            time_edf,
            (raw_data_edf[index, :] - np.mean(raw_data_edf[index, :])),
            label=(f"Chan {index -1}"),
        )

        T_edf = (len(raw_data_edf[0, :]) - 3) / fs_edf  
        freq_edf = np.arange(0, ((fs_edf / 2) + 1 / fs_edf), 1 / T_edf)
        # raw_data_edfm [36 channels x samples ]
    plt.suptitle(f"Subject {subjects} Run {run}")
    plt.title("Time Domain (.edf)")
    plt.ylabel("Amplitude (uV)")
    # plt.ylim([])
    plt.xlabel("Time (sec)")
    plt.legend()
    plt.tight_layout()
    # Save figure
    plt.savefig("Time_Domain_edf.png")
    # ....then show
    plt.show()
    plt.figure(num=300, figsize=(8, 6), clear=all)
    for index in range(2, 16):  # raw data valid eed electrode channels index 2 - 15
        PSD_edf = np.real(
            10
            * np.log10(
                (np.fft.rfft(raw_data_edf[index, :] - np.mean(raw_data_edf[index, :])))
                ** 2
            )
        )  # imaginary part = 0, extract real to avoid warnings
        plt.plot(freq_edf, PSD_edf, label=(f"Chan {index-1}"))
    plt.suptitle(f"Subject {subjects} Run {run}")
    plt.title("Frequency Domain (.edf)")
    plt.ylabel("PSD (dB)")
    # plt.ylim([-150,0])
    plt.xlabel("Freq (Hz)")
    plt.legend()
    plt.tight_layout()
    # Save figure
    plt.savefig("Freq_Domain_edf.png")  
    # ....then show
    plt.show()

    return


# %% Visualize the data Scalp topo maps


def plot_scalp_map(
    subject,
    electrodes,
    data,
    title,
    fig_num,
    data_type=".edf",
    run=1,
    method="mean",
    domain="time",
):
    """


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

    """

    # Plot time average for a giving subject, situation
    plt.figure(num=fig_num, clear=all)
    if method == "mean":
        if data_type == ".mat":

            sample_mean = np.mean(
                data, axis=1
            )  # This is the average across samples [12 situations x 14 channels]

            pt.plot_topo(
                channel_names=electrodes,
                channel_data=sample_mean[run, 0:14],
                title=title,
                cbar_label="Voltage (uV)",
                montage_name="biosemi64",
            )
        elif data_type == ".edf":
            raw_data_mean = np.mean(data[:, :], axis=1) * 1000

            pt.plot_topo(
                channel_names=electrodes,
                channel_data=raw_data_mean,
                title=title,
                cbar_label="Voltage (uV)",
                montage_name="biosemi64",
            )
    plt.tight_layout()
    # Save figure
    plt.savefig("Scalp_Map.png")
    # ....then show
    plt.show()
    return


# %% Clear figures
def clear_all_figures():
    """
    Get a list of figure numbers currently active in the matplotlib session, then close these figures

    Returns
    -------
    None.

    """
    # find amount of figures in session
    fig = plt.get_fignums()
    # close em
    for index, fig_num in enumerate(fig):
        plt.close(fig_num)


# %%
def labelling(data, labels):
    """
    Label the EEG data based on the anxiety levels extracted from labels.

    Args:
        data (list): List of EEG data samples.
        labels (ndarray): Array of anxiety levels corresponding to each data sample.

    Returns:
        DataFrame: DataFrame containing the labeled EEG data.
        tuple: A tuple containing the count of samples for each anxiety level category (severe, moderate, light, normal).
    """
    # organize the data and transpose the labels
    data = np.vstack(data[:])
    labels = labels.T

    # create dataframe
    df = pd.DataFrame(data)
    # pull label indicies
    label_indices = np.where(df.index % 1920 == 0)[0]

    # generate anxiety counts
    severe_count = 0
    moderate_count = 0
    light_count = 0
    normal_count = 0

    # assign labels to dataframe using the conditons
    for idx, index in enumerate(label_indices):
        # print(index)
        df.at[index, "valence"] = labels[idx][0]
        df.at[index, "arousal"] = labels[idx][1]
        df.at[index, "trial"] = f"trial_{idx}"
        val, aro = (labels[idx][0], labels[idx][1])

        if val <= 5 and aro >= 5:
            if 0 <= val <= 2 and 7 <= aro <= 9:
                df.at[index, "Anxiety_level"] = "severe"
                severe_count += 1
            elif 2 < val <= 4 and 6 <= aro < 7:
                df.at[index, "Anxiety_level"] = "moderate"
                moderate_count += 1
            elif 4 < val <= 5 and 5 <= aro < 6:
                df.at[index, "Anxiety_level"] = "light"
                light_count += 1
            else:
                df.at[index, "Anxiety_level"] = "normal"
                normal_count += 1

        else:
            df.at[index, "Anxiety_level"] = "normal"
            normal_count += 1

    df.set_index("trial", inplace=True)

    return df, (severe_count, moderate_count, light_count, normal_count)


# %%
def transformations(df, model, split_ratio=None, test_size=None):
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
    # pull dictionary keys
    keys = list(key for key in df.keys())

    # find the bandpower and entropy for alpha beta and gamma waves
    bpower = tfs.BandPowerSpectralDensity(
        128, band_dict={"alpha": [8, 14], "beta": [14, 31], "gamma": [31, 49]}
    )
    de = tfs.BandDifferentialEntropy(
        128, band_dict={"alpha": [8, 14], "beta": [14, 31], "gamma": [31, 49]}
    )

    # initialize variables
    trial_band_powers = []
    trial_entropys = []
    combined_labels = pd.DataFrame()

    # generate combined labels and put into labels variable using for loop
    for key in keys:
        eeg_data = df[f"{key}"].drop(["valence", "arousal", "Anxiety_level"], axis=1)
        # normalize eeg
        normalized_eeg = eeg_data.apply(
            lambda x: (x - np.mean(x)) / np.std(x), axis=1
        ).to_numpy()
        normalized_eeg = normalized_eeg.reshape(12, 1920, 14)

        anxiety_degree = df[f"{key}"]["Anxiety_level"][
            ~pd.isna(df[f"{key}"]["Anxiety_level"])
        ]
        # Concatenate the extracted anxiety levels with the existing combined_labels
        combined_labels = pd.concat([combined_labels, anxiety_degree])

        # Iterate through each trial in the normalized EEG data
        for trial in normalized_eeg:

            # Calculate band powers using the bpower function
            powers = bpower(eeg=trial.T)
            band_powers = powers["eeg"]

            # Calculate band entropies using the de function
            differential_entropy = de(eeg=trial.T)
            band_entropys = differential_entropy["eeg"]

            # Append the calculated band powers and band entropies to the respective lists
            trial_band_powers.append(band_powers)
            trial_entropys.append(band_entropys)
            labels = pd.get_dummies(combined_labels).to_numpy()

    # based on model input, load the data, pass in labels and features, and then split the data
    if model == "autoencoder":
        # Convert trial_band_powers and trial_entropys to PyTorch tensors
        band_tensors = torch.tensor(trial_band_powers)
        entropy_tensors = torch.tensor(trial_entropys)

        # Concatenate band_tensors and entropy_tensors along the last dimension, then transpose the result
        features = torch.transpose(
            torch.cat((band_tensors, entropy_tensors), dim=2), 1, 2
        )
        # Convert labels to a PyTorch tensor
        labels = torch.tensor(labels)

        # Stack features and labels into a TensorDataset
        features = torch.stack([feature for feature in features])
        labels = torch.stack([label for label in labels])
        dataset = TensorDataset(features, labels)

        # Split dataset into train_data and test_data
        train_data, test_data = random_split(dataset, split_ratio)

        # Return train_data and test_data
        return train_data, test_data

    elif model == "randomforest":
        # Convert trial_band_powers and trial_entropys to NumPy arrays
        band_arrays = np.asarray(trial_band_powers)
        entropy_arrays = np.asarray(trial_entropys)

        # Concatenate band_arrays and entropy_arrays along the last dimension, then transpose the result
        features = np.transpose(
            np.concatenate((band_arrays, entropy_arrays), axis=2), (0, 2, 1)
        )

        # Split features and labels into train_data, test_data, train_labels, and test_labels
        train_data, test_data, train_labels, test_labels = train_test_split(
            features, labels, test_size=test_size
        )

        # Return train_data, train_labels, test_data, and test_labels
        return train_data, train_labels, test_data, test_labels

    else:
        # Raise an exception if model is neither "autoencoder" nor "randomforest"
        raise Exception("please choose either autoencoder or randomforest")


# %%
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
        # Initialize the neural network module
        super().__init__()
        # Store the device for computations (CPU or GPU)
        self.device = device

        # Define the first linear layer mapping input to the first hidden layer
        self.linear1 = nn.Linear(input_dim, hidden_dims[0])

        # Define the second linear layer mapping the first hidden layer to the second hidden layer
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        # Define the linear layer mapping the second hidden layer to the mean of the latent space
        self.z_mean = nn.Linear(hidden_dims[1], latent_dim)

        # Define the linear layer mapping the second hidden layer to the log variance of the latent space
        self.log_var = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, inputs):
        """
        Forward pass of the encoder module.

        Args:
            inputs (Tensor): Input EEG data.

        Returns:
            tuple: Tuple containing the mean and log variance of the latent space and the sampled latent vector.
        """

        # Apply ReLU activation to the output of the first linear layer
        inputs = F.relu(self.linear1(inputs.float().view(inputs.size(0), -1)))

        # Apply ReLU activation to the output of the second linear layer
        inputs = F.relu(self.linear2(inputs))

        # Calculate the mean of the latent space
        z_mean = self.z_mean(inputs).to(self.device)

        # Calculate the log variance of the latent space
        log_var = self.log_var(inputs).to(self.device)

        # Sample epsilon from a normal distribution
        epsilon = Normal(0, 1).sample(z_mean.shape).to(self.device)

        # Reparameterization trick to sample from the latent space
        z = z_mean + torch.exp(0.5 * log_var) * epsilon

        # Return the mean, log variance, and sampled latent vector
        return z_mean, log_var, z


# %%
class Decoder(nn.Module):
    """
    Neural network decoder module for reconstructing EEG data from latent representations.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_dim (int): Dimensionality of the output EEG data.
    """

    def __init__(self, latent_dim, output_dim):
        # Initialize the decoder module
        super().__init__()

        # Define a linear layer mapping latent space to output EEG data
        self.linear1 = nn.Linear(latent_dim, output_dim)

    def forward(self, latent_vector):
        """
        Forward pass of the decoder module.

        Args:
            latent_vector (Tensor): Latent representation vector.

        Returns:
            Tensor: Reconstructed EEG data.
        """

        # Map the latent vector to the output EEG data using the linear layer
        latent_vector = self.linear1(latent_vector)

        # Apply sigmoid activation function to the output and reshape it to match the input shape
        return torch.sigmoid(latent_vector).view(-1, 6, 14)


# %%
class Classifier(nn.Module):
    """
    Neural network classifier module for predicting anxiety levels from latent representations.

    Args:
        embedding_dim (int): Dimensionality of the latent space.
        hidden_dim (list): List of hidden layer dimensions for the classifier network.
        target_number (int): Number of target classes.
    """

    def __init__(self, embedding_dim, hidden_dim, target_number):
        # Initialize the decoder module
        super().__init__()

        # Define linear layers mapping latent space to hidden layers and output layer
        self.linear1 = nn.Linear(embedding_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(
            hidden_dim[1], 10
        )  # Assuming a fixed size of 10 for the hidden layer
        self.linear4 = nn.Linear(10, target_number)

    def forward(self, z):
        """
        Forward pass of the classifier module.

        Args:
            z (Tensor): Latent representation vector.

        Returns:
            Tensor: Predicted probabilities for each target class.
        """
        # Map the latent vector to hidden layers using linear layers and apply ReLU activation
        z = self.linear1(z)
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))

        # Map the output of the hidden layers to the output layer and apply sigmoid activation
        return torch.sigmoid(self.linear4(z))


# %%


class VAE(nn.Module):
    """
    Variational autoencoder (VAE) module for jointly learning latent representations and reconstructing EEG data.

    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        device (str): Device to use for computations ('cuda' or 'cpu').
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        # move encoderr and decoder to specified device
        if device == "cuda" and torch.cuda.is_available():
            self.encoder = encoder.to("cuda")
            self.decoder = decoder.to("cuda")
        elif device == "cpu":
            self.encoder = encoder.to("cpu")
            self.decoder = decoder.to("cpu")

        else:
            raise Exception("Please choose cuda or cpu")

    def forward(self, x):
        """
        Forward pass of the VAE module.

        Args:
            x (Tensor): Input EEG data.

        Returns:
            tuple: Tuple containing the mean and log variance of the latent space and the reconstructed EEG data.
        """
        # perform forward pass tthroough the encoder to obtain laternet representation
        z_mean, log_var, z = self.encoder.forward(x)
        # perform forward pass through the decoder to obtain reconstruction
        reconstruction = self.decoder.forward(z)

        return z_mean, log_var, reconstruction


# %%
class Loss:
    """
    Loss function class for calculating VAE loss.
    """

    @staticmethod
    def kl_divergence(z_mean, log_var):
        return (
            -0.5
            * torch.sum(1 + log_var - z_mean.pow(2) - torch.exp(log_var), dim=1).mean()
        )

    @staticmethod
    def reconstruction_loss(x_reconstructed, x):
        return nn.MSELoss()(x_reconstructed, x)

    @staticmethod
    def classification_loss(y_pred, y_true):
        return nn.BCELoss()(y_pred, y_true)

    def vae_loss(self, pred, true):
        """
        Calculate VAE loss.

        Args:
            pred (tuple): Tuple containing the mean and log variance of the latent space and the reconstructed data.
            true (Tensor): True EEG data.

        Returns:
            Tensor: VAE loss.
        """

        z_mean, log_var, reconstruction_x = pred

        # calc reconstruction loss
        recon_loss = self.reconstruction_loss(reconstruction_x, true)

        # calc KL divergence test
        kld_loss = self.kl_divergence(z_mean, log_var)

        return recon_loss + kld_loss


# %%


class train:
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

    def __init__(
        self,
        hidden_dimensions,
        latent_embedding,
        device,
        train_data,
        max_grad_norm,
        lr=1e-4,
        epochs=1000,
        patience=10,
    ):

        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.latent_embedding = latent_embedding
        self.hidden_dimensions = hidden_dimensions
        self.train_data = train_data
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.max_grad_norm = (
            max_grad_norm  # Maximum gradient norm for gradient clipping
        )
        self.train_loader = self.dataloader()  # DataLoader for the training dataset

    def dataloader(self):
        """
        Generates a DataLoader from the train dataset.

        Returns
        -------
        train_loader(DataLoader) : Uses torch's DataLoader and converts the train dataset
        into a DataLoader.

        """
        # data loader forr the training data
        train_loader = DataLoader(
            self.train_data, batch_size=15, shuffle=True, drop_last=True
        )
        return train_loader

    def training(self):
        """
        Training method for the VAE model.

        Returns:
            tuple: Tuple containing the trained encoder, training losses, and epoch losses.
        """
        # init encoder, decoder, vae, and the loss func
        encoder = Encoder(
            84, self.hidden_dimensions, self.latent_embedding, self.device
        )
        decoder = Decoder(self.latent_embedding, 84)
        vae = VAE(encoder, decoder, self.device)
        los = Loss()

        # init optimizer
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr
        )

        # init lists for storing vars
        train_loss = []
        epoch_loss = []
        accs = []
        classification_loss = []

        # iterate over dataset epochs and batches
        for epoch in range(self.epochs):
            vae.to(self.device)
            vae.train()
            correct = 0
            total_samples = 0
            for batch_id, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                # zero gradients
                optimizer.zero_grad()

                # forward pass
                pred = vae.forward(data)
                loss = los.vae_loss(pred, data.float())

                # back propagation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    self.max_grad_norm,
                )
                optimizer.step()

                # SAVE TRAIN LOSSS
                train_loss.append(loss.item())

            # save epoch loss
            epoch_loss.append(np.mean(train_loss))

            print("Beginning Epoch", epoch)
            print("Average Combined Loss:", np.mean(epoch_loss[-1]))
            print("----------------------------------------------------------------")

            """if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f'Early stopping at epoch {epoch}')
                    break"""
        # plot and save train loss
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(self.epochs)), epoch_loss, label="Loss")
        plt.legend()
        plt.title("VAE Training")
        plt.tight_layout()
        plt.show()
        plt.savefig("encoder_acc_loss.png")
        return encoder, train_loss, epoch_loss


# %%
class latent_training:
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

    # initilaize all the stuff
    def __init__(
        self,
        encoder,
        embedding_dimension,
        hidden_dim,
        target_dim,
        train_data,
        device,
        loss=nn.BCELoss(),
        lr=1e-2,
        epochs=30,
    ):
        self.classifier = Classifier(embedding_dimension, hidden_dim, target_dim)
        self.train_data = train_data
        self.train_loader = self.data_loader()  # DataLoader for the train dataset
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
        # Create DataLoader for the training dataset
        return DataLoader(self.train_data, batch_size=15, shuffle=True, drop_last=True)

    def train(self):
        """
        Training method for the classifier using latent representations.

        Returns:
            tuple: Tuple containing the trained classifier, accuracies, and epoch losses.
        """
        # initialize lists for train stats
        train_losses = []
        accs = []
        epoch_loss = []

        preds = []
        targets = []
        # init opt
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)

        # move classifier to correct
        self.classifier.to(self.device)
        self.encoder.eval()
        self.classifier.train()

        # iterate over epochs and then batches
        for epoch in range(self.epochs):

            correct = 0
            total_samples = 0
            for batch_id, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                # zero graidents
                optimizer.zero_grad()

                # get latent representations from encoder
                z = self.encoder(data)

                # forward pass through classifier
                output = self.classifier(z[2])
                preds.extend(output)
                targets.extend(target)
                # loss
                train_loss = self.loss(output, target.float())

                # back propogation
                train_loss.backward()
                optimizer.step()

                # save train loss
                train_losses.append(train_loss.item())

                # calculate number of predicted labels
                predicted_labels = output.round()  # Assuming binary classification

                # Calculate number of correct predictions
                correct += (predicted_labels == target).sum().item()
                total_samples += target.size(0)

            # Calculate accuracy for the epoch
            acc = correct / total_samples
            accs.append(acc)
            epoch_loss.append(
                np.mean(train_losses[-len(self.train_loader) :])
            )  # Average loss for the epoch

            print(f"Beginning Epoch {epoch}")
            print(
                f"Classifier Average Loss: {np.mean(train_losses[-len(self.train_loader):]):.4f}"
            )
            print("----------------------------------------------------------------")
        # plot train loss and save
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(self.epochs)), epoch_loss, label="Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig("classifier_acc_loss.png")
        return self.classifier, accs, epoch_loss


# %%
class randomforest:
    """
    Random forest classifier for anxiety level prediction.

    Args:
        n_estimators (int): Number of trees in the forest.
        criterion (str): Criterion for measuring the quality of a split.
        max_depth (int): Maximum depth of the trees.
        scoring (str): Scoring metric for model evaluation.
    """

    def __init__(self, n_estimators, criterion, max_depth, scoring):
        # Initialize instance variables
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        # Initialize Random Forest classifier with specified parameters
        self.scoring = scoring
        self.forest = rfc(criterion=self.criterion)

        # Initialize Random Forest classifier with specified parameters
        self.grid = gsv(
            self.forest,
            [{"n_estimators": n_estimators}, {"max_depth": self.max_depth}],
            scoring=self.scoring,
        )

    def r_forest(self, x_train, y_train, x_test, y_test):
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
        # Reshape input data
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Perform grid search for hyperparameter tuning
        self.grid.fit(x_train, y_train)
        # Get best model and its parameters
        model = self.grid.best_estimator_
        best_params = self.grid.best_params_

        # Predict on training and test data
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        # Calculate accuracy scores
        train_acc = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Compute confusion matrices
        train_labels = np.argmax(y_train, axis=1)
        train_pred_labels = np.argmax(train_pred, axis=1)
        test_labels = np.argmax(y_test, axis=1)
        pred_labels = np.argmax(test_pred, axis=1)
        test_conf = confusion_matrix(test_labels, pred_labels)
        train_conf = confusion_matrix(train_labels, train_pred_labels)

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y_test.ravel(), test_pred.ravel())
        roc_auc = auc(fpr, tpr)

        # Compute precision and sensitivity scores
        train_precision = precision_score(train_labels, train_pred_labels, average=None)
        train_sensitivity = recall_score(train_labels, train_pred_labels, average=None)
        test_precision = precision_score(test_labels, pred_labels, average=None)
        test_sensitivity = recall_score(test_labels, pred_labels, average=None)

        # Plot confusion matrices and ROC curve
        fig, (ax1, ax2) = plt.subplots(1, 2)

        sns.heatmap(
            test_conf,
            annot=True,
            cmap="magma",
            xticklabels=["light", "moderate", "normal", "severe"],
            yticklabels=["light", "moderate", "normal", "severe"],
            ax=ax1,
        )
        ax1.set_title("Test Confusion Matrix")
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        sns.heatmap(
            train_conf,
            annot=True,
            cmap="vlag",
            xticklabels=["light", "moderate", "normal", "severe"],
            yticklabels=["light", "moderate", "normal", "severe"],
            ax=ax2,
        )
        ax2.set_title("Train Confusion Matrix")
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")

        plt.tight_layout()
        plt.savefig("Rforest_confusion_matrices.png")
        plt.show()

        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Rforest Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.savefig("Rforest_roc_curve.png")
        plt.show()

        # Print evaluation metrics
        print("Rforest Test Accuracy:", test_accuracy)
        print("Rforest Train Accuracy:", train_acc)
        print("Rforest Train Precision:", train_precision)
        print("Rforest Test Precision:", test_precision)
        print("Rforest Train Sensitivity:", train_sensitivity)
        print("Rforest Test Sensitivity:", test_sensitivity)

        return model, best_params, test_accuracy

    def latent_forest(self, encoder, best_rf_model, x_train, y_train, x_test, y_test):
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
        # Reshape input data if needed
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        x_train_tens, x_test_tens = torch.tensor(x_train), torch.tensor(x_test)

        # Obtain latent representations for training data
        z_mean, _, z_train = encoder(x_train_tens)
        z_train = np.asarray(z_train.detach())

        # Fit the random forest model with latent representations
        best_rf_model.fit(z_train, y_train)

        # Predict on training data and calculate accuracy
        train_pred = best_rf_model.predict(z_train)
        train_acc = accuracy_score(y_train, train_pred)

        # Obtain latent representations for test data
        z_m, _, z_test = encoder(x_test_tens)
        z_test = np.asarray(z_test.detach())

        # Predict on test data and calculate accuracy
        test_pred = best_rf_model.predict(z_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Compute confusion matrices
        train_labels = np.argmax(y_train, axis=1)
        train_pred_labels = np.argmax(train_pred, axis=1)
        test_labels = np.argmax(y_test, axis=1)
        pred_labels = np.argmax(test_pred, axis=1)
        test_conf = confusion_matrix(test_labels, pred_labels)
        train_conf = confusion_matrix(train_labels, train_pred_labels)

        # Compute precision and sensitivity scores
        train_precision = precision_score(train_labels, train_pred_labels, average=None)
        train_sensitivity = recall_score(train_labels, train_pred_labels, average=None)
        test_precision = precision_score(test_labels, pred_labels, average=None)
        test_sensitivity = recall_score(test_labels, pred_labels, average=None)

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y_test.ravel(), test_pred.ravel())
        roc_auc = auc(fpr, tpr)

        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2)

        sns.heatmap(
            test_conf,
            annot=True,
            cmap="magma",
            xticklabels=["light", "moderate", "normal", "severe"],
            yticklabels=["light", "moderate", "normal", "severe"],
            ax=ax1,
        )
        ax1.set_title("Test Confusion Matrix")
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        sns.heatmap(
            train_conf,
            annot=True,
            cmap="vlag",
            xticklabels=["light", "moderate", "normal", "severe"],
            yticklabels=["light", "moderate", "normal", "severe"],
            ax=ax2,
        )
        ax2.set_title("Train Confusion Matrix")
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")

        plt.tight_layout()
        plt.savefig("latentforest_confusion_matrices.png")
        plt.show()

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("LatentForest Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.savefig("latentforest_roc_curve.png")
        plt.show()

        # Print evaluation metrics
        print("LatentForest Test Accuracy:", test_accuracy)
        print("LatentForest Train Accuracy:", train_acc)
        print("LatentForest Train Precision:", train_precision)
        print("LatentForest Test Precision:", test_precision)
        print("LatentForest Train Sensitivity:", train_sensitivity)
        print("LatentForest Test Sensitivity:", test_sensitivity)

        return best_rf_model, test_accuracy


# %%
class SVClassifier:
    """
    Support Vector Classifier for anxiety level prediction.

    Attributes:
        encoder: Encoder model for obtaining latent representations.
        classifier: Support Vector Classifier.
    """

    def __init__(self, encoder):
        """
        Initialize the SVClassifier object.

        Args:
            encoder: Encoder model for obtaining latent representations.
        """
        self.encoder = encoder
        self.classifier = SVC()

    def train(self, x_train, y_train, x_test, y_test):
        """
        Train and evaluate the support vector classifier.

        Args:
            x_train (array_like): Training data features.
            y_train (array_like): Training data labels.
            x_test (array_like): Test data features.
            y_test (array_like): Test data labels.
        """
        # Reshape input data if needed
        x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(
            x_test.shape[0], -1
        )
        train_labels_1d, test_labels_1d = np.argmax(y_train, axis=1), np.argmax(
            y_test, axis=1
        )

        # Train the support vector classifier
        self.classifier.fit(x_train, train_labels_1d)

        # Predict on training and test data
        train_pred = self.classifier.predict(x_train)
        test_pred = self.classifier.predict(x_test)

        # Calculate accuracy scores
        train_accuracy = accuracy_score(train_labels_1d, train_pred)
        test_accuracy = accuracy_score(test_labels_1d, test_pred)

        # Compute confusion matrices
        test_confusion_mat = confusion_matrix(test_labels_1d, test_pred)
        train_confusion_mat = confusion_matrix(train_labels_1d, train_pred)

        # Compute precision and sensitivity scores
        train_precision = precision_score(train_labels_1d, train_pred, average=None)
        train_sensitivity = recall_score(train_labels_1d, train_pred, average=None)
        test_precision = precision_score(test_labels_1d, test_pred, average=None)
        test_sensitivity = recall_score(test_labels_1d, test_pred, average=None)

        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(
            test_confusion_mat,
            annot=True,
            cmap="magma",
            xticklabels=["light", "moderate", "normal", "severe"],
            yticklabels=["light", "moderate", "normal", "severe"],
            ax=ax1,
        )
        ax1.set_title("Test Confusion Matrix")
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        sns.heatmap(
            train_confusion_mat,
            annot=True,
            cmap="vlag",
            xticklabels=["light", "moderate", "normal", "severe"],
            yticklabels=["light", "moderate", "normal", "severe"],
            ax=ax2,
        )
        ax2.set_title("Train Confusion Matrix")
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")

        plt.tight_layout()
        plt.savefig("svc_confusion_matrices.png")
        plt.show()

        # Print evaluation metrics
        print("SVC Test Accuracy:", test_accuracy)
        print("SVC Train Accuracy:", train_accuracy)
        print("SVC Train Precision:", train_precision)
        print("SVC Test Precision:", test_precision)
        print("SVC Train Sensitivity:", train_sensitivity)
        print("SVC Test Sensitivity:", test_sensitivity)

    def latent_train(self, train_data, train_labels):
        """
        Train the support vector classifier using latent representations.

        Args:
            train_data (array_like): Training data features.
            train_labels (array_like): Training data labels.
        """

        latent_train = self.encode(train_data)

        train_labels_1d = np.argmax(train_labels, axis=1)

        self.classifier.fit(latent_train, train_labels_1d)

    def latent_test(self, test_data, test_labels):
        """
        Test the support vector classifier using latent representations.

        Args:
            test_data (array_like): Test data features.
            test_labels (array_like): Test data labels.

        Returns:
            accuracy (float): Accuracy of the model on the test data.
            confusion_mat (array_like): Confusion matrix of the model on the test data.
            precision (array_like): Precision scores of the model on the test data.
            sensitivity (array_like): Sensitivity scores of the model on the test data.
        """
        # Encode the test data and make predictions using classifier
        latent_test = self.encode(test_data)
        predictions = self.classifier.predict(latent_test)

        test_labels_1d = np.argmax(test_labels, axis=1)

        # calculate accuracy and eval metrics
        accuracy = accuracy_score(test_labels_1d, predictions)

        confusion_mat = confusion_matrix(test_labels_1d, predictions)
        precision = precision_score(test_labels_1d, predictions, average=None)
        sensitivity = recall_score(test_labels_1d, predictions, average=None)

        print("LatentSVC Test Accuracy of SVM:", accuracy)
        print("LatentSVC Precision Scores of SVM :", precision)
        print("LatentSVC Sensitivity of SVM:", sensitivity)

        return accuracy, confusion_mat, precision, sensitivity

    def encode(self, data):
        """
        Encode the data using the encoder.

        Args:
            data (array_like): Input data to be encoded.

        Returns:
            array_like: Encoded data.
        """

        # Reshape data for compatibility with PyTorch
        data = data.reshape(data.shape[0], -1)
        data = torch.tensor(data)
        # encode it
        with torch.no_grad():
            z_mean, _, z = self.encoder(data)
        return z.cpu().numpy()

    def plot_metrics(self, confusion_mat, precision, sensitivity):
        """
        Plot evaluation metrics including confusion matrix, precision, and sensitivity.

        Args:
            confusion_mat (array_like): Confusion matrix of the model on the test data.
            precision (array_like): Precision scores of the model on the test data.
            sensitivity (array_like): Sensitivity scores of the model on the test data.
        """

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_mat,
            annot=True,
            cmap="Blues",
            fmt="g",
            xticklabels=["light", "moderate", "normal", "severe"],
            yticklabels=["light", "moderate", "normal", "severe"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        plt.savefig("LatentSVC_confusion_matrix.png")

        # Plot precision
        plt.figure(figsize=(8, 6))
        plt.bar(np.arange(len(precision)), precision, color="skyblue")
        plt.title("Precision")
        plt.xlabel("Class")
        plt.ylabel("Precision Score")
        plt.xticks(np.arange(len(precision)))
        plt.show()
        plt.savefig("LatentSVC_Precision.png")

        # Plot sensitivity
        plt.figure(figsize=(8, 6))
        plt.bar(np.arange(len(sensitivity)), sensitivity, color="salmon")
        plt.title("Sensitivity")
        plt.xlabel("Class")
        plt.ylabel("Sensitivity Score")
        plt.xticks(np.arange(len(sensitivity)))
        plt.show()
        plt.savefig("LatentSVC_Sensitivity.png")


# %% Load preprocessed data.  This is the raw data contained in the .edf files after bandpass filtering and application of ICA


def load_data_epoch_anxiety_levels(directory, subjects):
    """
    Assumption: data from the .mat files

    Two methods of getting at the eeg data contained in the dataset: list, and numpy array.
    The numpy array is the preferred method.


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
    """

    subjects_df = {}
    counts = []
    for index, subject in enumerate(subjects):
        # Construct the filename for the current subject
        filename = f'{directory}//S{str(subject).rjust(2, "0") if subject < 10 else subject}.mat'

        # Load the .mat file
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]
            data = f["data"][:]
            labels = f["labels"][:]
            df, count_tuple = labelling(data, labels)

            # Append DataFrame to the dictionary
            if f"subject" in subjects_df.keys():
                subjects_df[f"subject"].append(df)
            else:
                subjects_df[f"{subject}"] = df
            # Update counts for each anxiety level
            if len(counts) == 0:
                counts.append(count_tuple)
            else:
                count = counts.pop()
                updated_count = tuple(map(sum, zip(count, count_tuple)))
                counts.append(updated_count)

    # Display counts for each anxiety level
    severe_count, moderate_count, light_count, normal_count = counts[0]
    print("severe count:", severe_count)
    print("moderate_count:", moderate_count)
    print("light_count:", light_count)
    print("normal_count:", normal_count)

    return subjects_df


# %%
def plot_PSD(
    subject,
    electrodes,
    data,
    level,
    freq_band=[4, 20],
    run=1,
    sample_interval=15,
    fs=128,
):
    """
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
    """
    #         alpha_band=np.zeros(14)
    #         high_beta_band=np.zeros(14)
    #         less_20_hz=np.zeros(14)
    #
    plt.figure(
        num=subject + 51, figsize=(8, 6), clear=all
    )  # 51 is arbritrary This should result in fig 51 being associated with subject 01, Sub 23 Fig 73
    # TODO channels=14 to electrode enumerate

    # initialize the power in frequecy band to zereo
    PSD_band = np.zeros(len(electrodes))
    # PSD for trial sample interval
    sample_index = (
        sample_interval * fs
    )  # sample interval from start to sample time in seconds mult by fs
    # Convert the frequncy band from herts to index
    freq_low = freq_band[0] * sample_interval
    freq_high = freq_band[1] * sample_interval

    # For the purpose of plotting get the PSD frequency components
    freq = np.arange(0, ((fs / 2) + 1 / fs), 1 / sample_interval)

    for electrode_index, electrode in enumerate(
        electrodes
    ):  # TODO change from all "14" to channels
        print(electrode_index, electrode)
        # Calculate and plot the entire PSD
        # PSD = np.real(10*np.log10((np.fft.rfft(data[trial,0:sample_index,channel]))**2)) # imaginary part = 0, extract real to avoid warnings #TODO delete this reference
        PSD = np.real(
            10
            * np.log10(
                (
                    np.fft.rfft(
                        data[run, :, electrode_index]
                        - np.mean(data[run, :, electrode_index])
                    )
                )
                ** 2
            )
        )  # imaginary part = 0, extract real to avoid warnings
        plt.plot(freq, PSD, label=(f"Chan {electrode}"))
        plt.ylabel("PSD (dB)")
        plt.ylim([-5, 80])  # TODO problem plots zero hz value
        plt.xlabel("Frequency (Hz)")
        plt.suptitle(
            f"Power Spectral Density Run {run} Subject {subject+1}"
        )  # Indicate Trial number index starting from 1
        #     # plt.suptitle (f'PSD 4-20 Hz Band for Light Anxiety Qty {light_count}')  #normal, light, severe
        plt.title(f"Level {level},Time Interval {sample_interval} sec")
        plt.grid("on")
        plt.legend(loc="upper right")
        # Integrated the spectrum and normalized based on the length of the freq band
        PSD_band[electrode_index] = np.sum(PSD[freq_low:freq_high]) / (
            freq_high - freq_low
        )  # 4 to 20 Hz  # Remeber that the raw data is BP filtered 4 to 45 hz.
    plt.tight_layout()
    # Save figure
    plt.savefig(f"PSD_subject{subject}.png")  # TODO Light Severe
    # ....then show
    plt.show()
    return PSD_band
