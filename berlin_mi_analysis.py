# %% Import packages
# !%load_ext autoreload
# !%autoreload 2
'''
Analysis of the Berlin BCI motor imagery dataset using MNE-Python.

Author:
    Karahan Yilmazer

Email:
    yilmazerkarahan@gmail.com
'''

#!%matplotlib qt

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from utils import load_data


# %%
def calc_erds(epochs):
    # TODO: Write a function to calculate the ERD/ERS curves.
    # You can extend the function however you want (e.g., with other helper
    # functions or with more input parameters)
    pass

    return erds


# %% [markdown]
## Raw: Continuous data
# First, load in the continuous EEG recording to MNE Raw objects.

# %% Load the data
part = 'e'  # Select the participant
data_path = r'C:\Files\Coding\Python\Neuro\data\Motor Imagery\BCICIV'
file = os.path.join(
    data_path,
    'mat_data',
    'BrainAmp MR plus',
    'BCICIV_1calib_mat',
    'BCICIV_calib_ds1' + part + '.mat',
)  # Adapt the path to your system

# Load the data
raw, event_arr, event_id = load_data(file)

# Print the classes and their encodings
print('Class 1:', list(event_id.keys())[0], '-->', list(event_id.values())[0])
print('Class 2:', list(event_id.keys())[1], '-->', list(event_id.values())[1])

# Check the Raw object information
raw

# %%
# Plot the unfiltered signal
raw.plot(scalings='auto')

# %%
# Plot the PSD of the unfiltered signal
psd = raw.compute_psd()
psd.plot()

# %% [markdown]
# In the interactive PSD plot, you can click on individual lines to see which electrodes they correspond to. In this case, we can see that there is peak around 12 Hz coming from the two channels on the sensorimotor region: C3 and C4.
#
# Even without any preprocessing from our side, the significance of the mu band activity is very pronounced. That is why we can say that this participant is particularly good at producing seperable motor imagery signals.

# %% [markdown]
## ICA
# Even though the signal is relatively clean, it would be better if we could get rid of the eye blinks. Fortunately, ICA comes to the rescue.
#
# ICA (independent component analysis) decomposes the signal to its estimated sources. If we can catch an ICA component that captures the eye signals, we can exclude it. This way, we can reconstruct a cleaner signal without the eye blinks.
#
# ICA is sensitive to low frequency drifts. So, we have to apply a high-pass filter before fitting ICA to our data.

# %%
# Set the cutoff frequencies for a high-pass filter
flow, fhigh = 1, None
# Without the copy() method, filtering would be done in-place on the raw
# variable
raw_filt = raw.copy().filter(flow, fhigh)

# %%
# Initialize an ICA object
ica = mne.preprocessing.ICA(random_state=42)

# Uncomment the code below if your ICA is taking too long
# You can play around with the n_components parameter to speed up the process
# ica = mne.preprocessing.ICA(random_state=42, n_components=0.9)

# Fit ICA on the high-pass filtered data
# This may take more than 2 minutes if you are using all channels
ica.fit(raw_filt)

# %%
# Plot the ICA components
ica.plot_sources(raw_filt)

# Project the ICA components on an interpolated sensor topography
ica.plot_components()

# %% [markdown]
# Clearly, ICA001 and ICA032 captured eye blinks. Which is why we can exclude them from our signal.
ica.exclude = [1, 32]

# Visualize the effect of excluding the noisy ICA component
ica.plot_overlay(raw)

# %%
# Reconstruct the signal without the noisy ICA component
raw_ica = raw.copy()
ica.apply(raw_ica)

# Delete the unfiltered version that is not needed any more
del raw

# %% [markdown]
## Band-pass filtering
# Now, we can apply band-pass filtering to extract the frequency band of interest. For that, we can use a broad band, assuming that it would capture most frequency bands of interest.
raw_filt = raw_ica.copy().filter(7, 30)

# Plot the PSD of the filtered signal
raw_filt.plot_psd()

# %% [markdown]
# The lower frequency band (0-7 Hz) is distorted by our filter. So, let's check out how to our filter actually looks like.

# %%
# Create the default filter
h = mne.filter.create_filter(
    data=raw_filt.get_data(), sfreq=raw_filt.info['sfreq'], l_freq=7, h_freq=30
)

# Visualize the default filter
mne.viz.plot_filter(h, sfreq=raw_filt.info['sfreq'])

# %%
# Create a custom filter
h = mne.filter.create_filter(
    data=raw_filt.get_data(),
    sfreq=raw_filt.info['sfreq'],
    l_freq=7,
    h_freq=30,
    method='fir',
    fir_window='blackman',
    fir_design='firwin2',
)

# Visualize the custom filter
mne.viz.plot_filter(h, sfreq=raw_filt.info['sfreq'])

# %%
# Apply the custom filter
raw_filt = raw_ica.copy().filter(7, 30, method='fir', fir_window='blackman')

# Plot the PSD after applying the new filter
raw_filt.plot_psd()

# %% [markdown]
## Epoched data: Epochs
# At this point, we can cut our data into the so-called epochs. Each epoch will correspond to either left or right hand motor imagery.

# %%
# Define the epoching window
tmin, tmax = -1, 5

# Cut the continuous signal into epochs
epochs = mne.Epochs(
    raw_filt,
    events=event_arr,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
)

# Check the Epochs object information
epochs
# %%
# Plot the first five left MI epochs
# epochs['left'][0:5].plot(scalings='auto')
epochs.plot(scalings='auto', events=True)
# Plot all epochs (will probably crash if you haven't installed mne-qt-browser)
epochs.plot(scalings='auto', events=True)

# %%
# Plot the PSD
epochs['left'].compute_psd().plot()

# %% [markdown]
## ERD/ERS
# We will be plotting the ERD/ERS curves before and after spatial filtering. The spatial filtering will be in the form of rereferencing with a commmon average reference (CAR).

# %%
# Set the sigma for the Gaussian smoothing
sigma = 7
rest_period = (-1, 0)

# Calculate and plot the ERD/ERS curves
erds = calc_erds(epochs)

# %%
# Apply common average referencing (CAR)
epochs_car = epochs.copy().set_eeg_reference(ref_channels='average')

# Plot the ERD/ERS curves of the spatially filtered signal
erds_car = calc_erds(epochs_car)

# %% [markdown]
## Classification Using CSP
# Now, we will be using MNE's CSP implementation to classify between both classes.

# %%
# Get the epoched data and class labels
X = epochs.get_data()
y = epochs.events[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50
)

# Initialize LDA
lda = LinearDiscriminantAnalysis()

# Try fitting the CSP with the default settings
try:
    csp = mne.decoding.CSP()
    X_csp_train = csp.fit_transform(X_train, y_train)
# Use a more robust estimation method if the default settings run into
# numerical problems
except:
    csp = mne.decoding.CSP(reg='pca', rank='full')
    X_csp_train = csp.fit_transform(X_train, y_train)
    print('Full rank PCA was used to estimate the covariance matrix.')

# %%
# Fit the CSP to training data
X_csp_train = csp.fit_transform(X_train, y_train)

# Plot the calculated patterns
csp.plot_patterns(epochs.info, units='Patterns (AU)', size=1.5)

# %%
# Fit the LDA to training data
lda.fit(X_csp_train, y_train)

# Calculate the cross validation score of each fold
cv_score = cross_val_score(lda, X_csp_train, y_train, cv=10)
cv_mean = np.mean(cv_score)
cv_std = np.std(cv_score)

# Transform the test data using CSP
X_csp_test = csp.transform(X_test)

# Make predictions usingLDA
y_pred_train = lda.predict(X_csp_train)
y_pred_test = lda.predict(X_csp_test)

# Get the confusion matrices for the training and test data
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

# Calculate the sensitivity and specificity
sens_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1])
spec_test = cm_test[1, 1] / (cm_test[1, 0] + cm_test[1, 1])
acc_test = lda.score(X_csp_test, y_test)

# Print the classifier performance scores
print(f'CV Score:\t{cv_mean:.2f} ± {cv_std:.2f}')
print(f'Sensitivity:\t{sens_test:.2f}')
print(f'Specificity:\t{spec_test:.2f}')
print(f'Accuracy:\t{acc_test:.2f}')

# %%
cmap = 'magma'
labels = list(epochs.event_id.keys())

fig, axs = plt.subplots(1, 2)

# Plot the confusion matrices
sns.heatmap(cm_train, annot=True, ax=axs[0], cbar=False, cmap=cmap)
sns.heatmap(cm_test, annot=True, ax=axs[1], cbar=False, yticklabels=False, cmap=cmap)

axs[0].set_title('Training Set')
axs[0].set_xlabel('Predicted label')
axs[0].set_ylabel('True label')
axs[0].xaxis.set_ticklabels(labels)
axs[0].yaxis.set_ticklabels(labels)
axs[0].tick_params(
    axis='both', which='both', bottom=False, left=False, right=False, top=False
)

axs[1].set_title('Test Set')
axs[1].set_xlabel('Predicted label')
axs[1].xaxis.set_ticklabels(labels)
axs[1].tick_params(
    axis='both', which='both', bottom=False, left=False, right=False, top=False
)

plt.suptitle(f'Participant {part.capitalize()} Confusion Matrices')
plt.show()

# %%
