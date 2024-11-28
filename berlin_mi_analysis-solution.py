# %% Import packages
# !%load_ext autoreload
# !%autoreload 2
'''
Analysis of the Berlin BCI motor imagery dataset using MNE-Python (solution).

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
def calc_erds(epochs, channels=None, sigma=2, rest_period=(-1, 0)):

    # Get the sampling frequency
    sfreq = epochs.info['sfreq']

    # Get the channel indices
    if channels is not None:
        ch_idx = [epochs.ch_names.index(ch) for ch in channels]
    else:
        ch_idx = np.arange(len(epochs.ch_names))

    # Get the event names
    events = list(epochs.event_id.keys())

    # Initialize the dictionary for storing the ERD/ERS curves
    erds_dict = {}

    # Get the reference period limits
    rmin = rest_period[0] * sfreq
    rmax = rest_period[1] * sfreq

    # If the epoching window start from before the cue was shown
    if epochs.tmin < 0:
        # Shift both reference period limits accordingly
        rmin += -epochs.tmin * sfreq
        rmax += -epochs.tmin * sfreq

    # Convert the limits to integer for slicing
    rmin = int(rmin)
    rmax = int(rmax)

    # Iterate over the events
    for event in events:

        # Get the trials data for the relevant channels
        epochs_arr = epochs[event].copy().get_data()[:, ch_idx, :]

        # Initialize an empty array for the band-powers
        epochs_bp = np.zeros(epochs_arr.shape)

        # Iterate over the trials
        for i, trial in enumerate(epochs_arr):
            # Iterate over the channels
            for ch in range(len(ch_idx)):
                # Square the signal to get an estimate of the band-powers
                epochs_bp[i, ch, :] = trial[ch] ** 2

        # Average the band-powers over trials
        A = np.mean(epochs_bp, axis=0)

        # Get the reference period
        R = np.mean(A[:, rmin:rmax], axis=1).reshape(-1, 1)

        # Compute the ERD/ERS
        erds = (A - R) / R * 100

        # Smoothen the ERD/ERS curve
        erds = gaussian_filter1d(erds, sigma=sigma)

        # Append the curves to the corresponding events
        erds_dict[event] = erds

    return erds_dict


def plot_erds(
    erds_dict,
    epochs,
    channels=None,
    events=None,
    view='channel',
    title_suffix='',
):

    if view not in ['task', 'channel']:
        raise ValueError(
            "Please provide a valid view parameter. Valid view parameters are: 'task' and 'channel'."
        )

    if view == 'channel' and channels is None:
        raise ValueError("Please provide the channels for the 'channel' view.")

    # The values for plotting
    tmin = epochs.tmin
    tmax = epochs.tmax
    flow = float(epochs.info['highpass'])
    fhigh = float(epochs.info['lowpass'])
    events = list(epochs.event_id.keys())

    # Get the number of samples in the ERD/ERS curves
    n_chs = list(erds_dict.values())[0].shape[0]
    n_samples = list(erds_dict.values())[0].shape[1]

    x = np.linspace(tmin, tmax, n_samples)

    # Initialize the plot
    n_rows = n_chs if view == 'channel' else len(events)
    fig, axs = plt.subplots(n_rows, 1)
    axs = axs.ravel()

    if view == 'task':
        for ax, (event_name, erds_arr) in zip(axs, erds_dict.items()):

            if 'left' in event_name:
                event_name = 'Left\nMI'
            if 'right' in event_name:
                event_name = 'Right\nMI'

            if channels is not None:
                ax.plot(x, erds_arr.T, lw=2)
                ax.legend(channels)
                title = f'ERD/ERS Curves{title_suffix}\n({flow}-{fhigh} Hz BP)'
            else:
                ax.plot(x, np.mean(erds_arr, axis=0), lw=2, color='navy')
                title = f'ERD/ERS Curves Averaged Over Available Channels{title_suffix}\n({flow}-{fhigh} Hz BP)'

            if tmin <= 0:
                ax.axvline(0, color='gray', lw=2)
            ax.axhline(0, color='gray', ls='--')
            ax.set_xticks(np.arange(tmin, tmax + 0.1, 0.5))
            if ax != axs[-1]:
                ax.set_xticklabels([])
            ax.grid()
            ax_twin = ax.twinx()
            ax_twin.set_ylabel(event_name, rotation=0, labelpad=17)
            ax_twin.set_yticklabels([])

    elif view == 'channel':
        for i in range(len(events)):
            if 'left' in events[i]:
                events[i] = 'Left MI'
            if 'right' in events[i]:
                events[i] = 'Right MI'

        for i, ax in enumerate(axs):

            for erds_arr in erds_dict.values():

                if channels is not None:
                    ax.plot(x, erds_arr[i], lw=2)
                    title = f'ERD/ERS Curves{title_suffix}\n({flow}-{fhigh} Hz BP)'
                else:
                    ax.plot(x, np.mean(erds_arr, axis=0), lw=2, color='navy')
                    title = f'ERD/ERS Curves Averaged Over Available Channels{title_suffix}\n({flow}-{fhigh} Hz BP)'

            ax.legend(events)

            if tmin <= 0:
                ax.axvline(0, color='gray', lw=2)
            ax.axhline(0, color='gray', ls='--')
            ax.set_xticks(np.arange(tmin, tmax + 0.1, 0.5))
            if ax != axs[-1]:
                ax.set_xticklabels([])
            ax.grid()
            ax_twin = ax.twinx()
            ax_twin.set_ylabel(channels[i], rotation=0, labelpad=10)
            ax_twin.set_yticklabels([])

    ax = fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axes
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.tick_params(
        axis='x', which='both', top=False, bottom=False, left=False, right=False
    )
    ax.tick_params(
        axis='y', which='both', top=False, bottom=False, left=False, right=False
    )
    ax.grid(False)
    ax.set_xlabel('Time Relative to the Cue (in s)')
    ax.set_ylabel('Relative Band Power (in %)')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# %% [markdown]
## Raw: Continuous data
# First, load in the continuous EEG recording to MNE Raw objects.

# %% Load the data
part = 'e'  # Select the participant
data_path = os.path.join('data', 'BCICIV_calib_ds1' + part + '.mat')

# Load the data
raw, event_arr, event_id = load_data(data_path)

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

# %%
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
epochs['left'][0:5].plot(scalings='auto')

# Plot all epochs (will probably crash if you haven't installed mne-qt-browser)
# epochs.plot(scalings='auto', events=True)

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
channels = ['C3', 'Cz', 'C4']

# Calculate and plot the ERD/ERS curves
erds_base = calc_erds(
    epochs,
    channels=channels,
    sigma=sigma,
    rest_period=rest_period,
)
plot_erds(erds_base, epochs, channels=channels, view='task')

# %%
# Apply common average referencing (CAR)
epochs_car = epochs.copy().set_eeg_reference(ref_channels='average')

# Plot the ERD/ERS curves of the spatially filtered signal
erds_car = calc_erds(
    epochs_car,
    channels=channels,
    sigma=sigma,
    rest_period=rest_period,
)
plot_erds(
    erds_car,
    epochs_car,
    channels=channels,
    view='task',
    title_suffix=' (CAR)',
)

# %% [markdown]
## Effect of Spatial Filtering
# We can see that spatial filtering can reveal more pronounced ERD/ERS curves. This is because the common average reference (CAR) can remove the common noise sources from the signal.


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
