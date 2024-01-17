import mne
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def load_data(file):
    # Extract the MATLAB struct from the file
    m = loadmat(file)

    # Get the sampling rate (downsampled to 100 Hz)
    sample_rate = m['nfo']['fs'][0][0][0][0]

    # Get the raw EEG data
    # data: (n_channels x n_samples)
    data = m['cnt'].T

    # Get the number of channels (59) and samples (190473)
    # _, n_samples = data.shape

    # Get the names of the electrodes (channels)
    ch_names = [ch[0] for ch in m['nfo']['clab'][0][0][0]]

    # Create the info structure needed by MNE
    info = mne.create_info(ch_names, sample_rate, 'eeg')

    # Add the system name to the info
    info['description'] = 'BrainAmp MR plus'

    # Create the Raw object
    raw = mne.io.RawArray(data, info)

    # Create a montage out of the 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')

    # Exclude channels which are missing in the 10-20 montage
    raw.pick(
        'eeg',
        exclude=[
            'CFC7',
            'CFC5',
            'CFC3',
            'CFC1',
            'CFC2',
            'CFC4',
            'CFC6',
            'CFC8',
            'CCP7',
            'CCP5',
            'CCP3',
            'CCP1',
            'CCP2',
            'CCP4',
            'CCP6',
            'CCP8',
        ],
    )

    # Apply the montage
    raw.set_montage(montage, on_missing='ignore')

    # Get the class labels (left and right)
    classes = [cl[0] for cl in m['nfo']['classes'][0][0][0]]

    # Get the onsets (indices) of the events (markers)
    event_onsets = m['mrk'][0][0][0][0]
    # Get the encoded labels of the events
    event_codes = m['mrk'][0][0][1][0]
    # Convert event codes to a list of class names
    event_names = [classes[0] if x == -1 else classes[1] for x in event_codes]

    # Initialize the label encoder
    le = LabelEncoder()
    # Encode the cues using the label encoder
    events_encoded = le.fit_transform(event_names)

    # Create the event information
    event_arr = np.zeros((len(event_onsets), 3), dtype=int)
    event_arr[:, 0] = event_onsets
    event_arr[:, 2] = events_encoded

    # Create a class-encoding correspondence dictionary for the Epochs
    # object
    event_id = dict(zip(classes, range(len(le.classes_))))

    return raw, event_arr, event_id
