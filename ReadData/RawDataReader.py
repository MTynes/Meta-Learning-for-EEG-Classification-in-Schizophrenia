import pandas as pd
import pyedflib

import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils as np_utils

import numpy as np
import random
import math
from IPython.display import display

from sklearn.utils import shuffle
import sklearn.decomposition as decomposition
from sklearn.decomposition import FastICA
import pywt

from math import sqrt
import itertools
import mne
from mne import io


mne.set_log_level("WARNING")


all_channels = ['Fp2', 'F8', 'T4', 'T6', 'O2', 'Fp1', 'F7', 'T3', 'T5', 'O1', 'F4',
                'C4', 'P4', 'F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz']

default_resolution = 250
def get_edf_file_duration(file_name):
    f = pyedflib.EdfReader(file_name)
    duration = f.file_duration
    f.close()
    return duration


# get the minimum length of the files
def get_minimum_duration(group_directory_name, patient_group_file_prefix, raw_data_dir):
    file_durations = []
    for i in range (1, 15): # reading 14 files
        patient_id = "{}{:02d}".format(patient_group_file_prefix, i)
        file_name = raw_data_dir + '{}/{}.edf'.format(group_directory_name, patient_id)
        file_durations.append(get_edf_file_duration(file_name))

    return min(file_durations)

def get_file_durations(group_directory_name, patient_group_file_prefix, raw_data_dir):
    file_durations = []
    for i in range (1, 15): # reading 14 files
        patient_id = "{}{:02d}".format(patient_group_file_prefix, i)
        file_name = raw_data_dir + '{}/{}.edf'.format(group_directory_name, patient_id)
        file_durations.append(get_edf_file_duration(file_name))
    return file_durations


# Get a list of randomly selected sets of numbers based on a range
# The proportion of values selected for each set is determined by the ratio_array
def get_mixed_indexes_for_ml_train_test(length, ratios_array):
    input_indexes = range(0, length)
    output_indexes = []
    for ratio in ratios_array:
        input_indexes = [i for i in input_indexes if i not in list(itertools.chain(*output_indexes))]
        selection = random.sample(input_indexes, k=math.floor(ratio * length))
        output_indexes.append(selection)
    return output_indexes


# chunk and chunk_list are being renamed but need to be updated in the notebooks
# Based on https://stackoverflow.com/a/48704557/2466781
def chunk(seq, size):
    # exclude values that will be out of range
    sl = len(seq) - (len(seq) % size) if len(seq) % size != 0 else len(seq)
    r = [np.asarray(seq[pos:pos + size]) for pos in range(0, sl, size)]
    r = np.asarray(r)
    return r


def chunk_list(nested_list, size):
    v = []
    for d in nested_list:
        df = pd.DataFrame(np.asarray(d))
        c = chunk(df, size)
        for e in c:
            v.append(e)
    return v


# Based on https://stackoverflow.com/a/48704557/2466781
def windowize(seq, size):
    # exclude values that will be out of range
    sl = len(seq) - (len(seq) % size) if len(seq) % size != 0 else len(seq)
    r = [np.asarray(seq[pos:pos + size]) for pos in range(0, sl, size)]
    r = np.asarray(r)
    return r


# create time windows with an overlap
# seq: the dataframe of values to segment
# size: the desired length of the individual windows in the output list
# overlap_rate: the ratio of the size to indicate the amount of overlap
#               a rate of .9 will set 90% overlap and will begin the second 
#               segment at 10% of the length of the first segment. Ratios will
#               be rounded up
def windowize_with_overlap(seq, size, overlap_rate):
    start_pos, end_pos = 0, size
    slices = []
    while end_pos <= len(seq):
        window = np.asarray(seq[start_pos: end_pos])
        start_pos = start_pos + size - math.ceil(overlap_rate * size)
        end_pos = start_pos + size
        slices.append(window)
    slices = np.asarray(slices)
    return slices


def windowize_list(nested_list, size, overlap_rate=None):
    wl = []
    for d in nested_list:
        df = pd.DataFrame(np.asarray(d))
        w = windowize(df, size) if overlap_rate is None else windowize_with_overlap(df, size, overlap_rate)
        for e in w:
            wl.append(e)
    return wl


# add random noise using the mean and standard deviation of the original dataset
# noise is added to original data and returned as a list
# seq: data in list format
def generate_channel_noise(seq):
    noise = np.random.normal(np.mean(seq), np.std(seq), len(seq))
    return seq + noise
  

def get_raw_eeg_mne(file_name, resolution_hz, tmin=None, tmax=None, exclude=[]):
    raw = mne.io.read_raw_edf(file_name, preload=True, exclude=exclude).load_data()
    raw.set_montage("standard_1020")  # set montage to 10-20
    # print('tmin: ', tmin)
    # print('tmax: ', tmax)
    tmin = tmin if tmin else 1
    tmax = tmax if tmax else (get_edf_file_duration(file_name) - 1)  # get_edf_file_duration rounds values up
    raw.crop(tmin=tmin, tmax=tmax)
    if resolution_hz != default_resolution:
        raw.resample(resolution_hz, npad="auto")  # set sampling frequency (default is 250Hz)
    return raw


# Uses the raw EDF files and converts to dataframe, dropping the first 150 and last 30 seconds of the shortest  file
# All other files are trimmed similarly to produce the same size
# Adapted from page 1 of https://buildmedia.readthedocs.org/media/pdf/pyedflib/latest/pyedflib.pdf
def process_patient_group(group_directory_name, patient_group_file_prefix,
                          minimum_duration,
                          resolution_hz,
                          raw_data_dir,
                          ignore_list,
                          excluded_channels=[]):
    meta = []
    patient_id_list = []

    for i in range(1, 15):  # reading 14 files
        patient_id = "{}{:02d}".format(patient_group_file_prefix, i)
        patient_id_list.append(patient_id)
        tmin, tmax = 120, minimum_duration - 120
        file_name = raw_data_dir + '{}/{}.edf'.format(group_directory_name, patient_id)
        data = get_raw_eeg_mne(file_name, resolution_hz, exclude=excluded_channels, tmin=tmin,
                               tmax=tmax)
        df = data.to_data_frame()[:resolution_hz * (tmax-tmin)]  # bug fix -- removing additional 1/250 s of data
        if patient_id not in ignore_list:
            meta.append(np.asarray(df))

    return np.asarray(meta)


# independent component analysis
def get_ica(data):
  transformer = FastICA(n_components=None,  # use all components
        random_state=0)
  data_transformed = transformer.fit_transform(data)
  return data_transformed


# source : https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
# parameters: X array wavelet, 
#             str built-in wavelet name (use wavelist() to get list of all), 
#             str mode (optional) use pywt.Modes.modes
#             or see https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes
# returns a tuple representing [estimate, coefficients]
def discrete_wt(X, wavelet_name='db1', mode=None):
    if mode is None:
        r = pywt.dwt(X, wavelet_name)
    else:
        r = pywt.dwt(X, wavelet_name, mode=mode)
    return r


def inverse_discrete_wt(coeffs, wavelet, wavelet_name='db1', mode=None):
    if mode is None:
        r = pywt.idwt(coeffs, wavelet, wavelet_name)
    else:
        r = pywt.idwt(coeffs, wavelet, wavelet_name, mode=mode)
    return r


def get_denoised_dwt_cA(data):
    dwt_cA, dwt_cD = discrete_wt(data)
    denoised_dwt_cA = data_reader.select_denoised_data(dwt_cA)
    return denoised_dwt_cA


# flatten the feature vectors
def flatten_features(data):
    flattened_data = []
    for entry in data:
        # shift axes so that data shape is time * channels * features. Then flatten data
        flattened_data.append(np.moveaxis(entry, 0, -1).flatten())
    return np.asarray(flattened_data, dtype=np.float32)


# Calculate column means
def column_means(dataset):
    col_means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        col_means[i] = sum(col_values) / float(len(dataset))
    return col_means


# calculate column standard deviations
def column_stdevs(dataset, means):
    col_stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
        col_stdevs[i] = sum(variance)
    col_stdevs = [sqrt(x / (float(len(dataset) - 1))) for x in col_stdevs]
    return col_stdevs


# standardize dataset
def standardize_dataset(full_dataset, col_means, col_stdevs):
    for row in full_dataset:
        for i in range(len(row)):
            row[i] = (row[i] - col_means[i]) / col_stdevs[i]
    return full_dataset


# Remove outliers
# Adapted from https://stackoverflow.com/a/45399188/2466781
# Function is expecting data in format samples, time-steps, channels
# It will replace outliers with median value, where outliers are defined
# as values greater than or less than 3 x median
def replace_outliers(data, m=1.5):
    cleaned_data = []
    for sample in data:
        sample = sample.transpose()
        cleaned_sample = []
        for channel in sample:
            d = np.abs(channel - np.median(channel))
            mdev = np.median(d)
            s = d / (mdev if mdev else 1.)
            channel[s >= m] = np.median(channel)
            cleaned_sample.append(channel)

        cleaned_data.append(np.asarray(cleaned_sample).transpose())
    return cleaned_data


# Returns a regular list of integers from a one-hot encoded list.
# Conversion is based on the index of the value 1.
# For example, if one_hot_labels[0] == [0., 1.], label_list[0] == 1
def convert_one_hot_labels_to_class_int_list(one_hot_labels):
    label_list = [label.index(1.) for label in one_hot_labels]
    return label_list


# flatten the feature vectors so that input can be used in scikit learn
def flatten_features(data):
    flattened_data = []
    for entry in data: 
        # shift axes so that data shape is time * channels * features. Then flatten data
        flattened_data.append(np.moveaxis(entry, 0, -1).flatten())
    return np.asarray(flattened_data, dtype=np.float32)


# reshape a 3-d array so that the final two indexes (channels and time-steps) have been switched
def reshape_indexes(data):
    data_sh = np.asarray(data).shape
    data = np.asarray(data).reshape(data_sh[0], data_sh[2], data_sh[1])
    return data


def smooth_data(x_train, x_validate, x_test):
    # Estimate mean and standard deviation

    # use the training data to fit the scaler, as suggested here https://stackoverflow.com/a/50567308
    # flatten the features then reintroduce the 3rd dimension
    dataset_flattened = flatten_features(x_train)
    means = column_means(dataset_flattened)
    stdevs = column_stdevs(dataset_flattened, means)
    
    x_train_shape = np.asarray(x_train).shape
    x_validate_shape = np.asarray(x_validate).shape
    x_test_shape = np.asarray(x_test).shape
    # set outliers to median values for each channel
    print('Removing outliers by channel')
    x_train = standardize_dataset(flatten_features(x_train),means, stdevs)
    x_train = x_train.reshape(x_train_shape)
    x_validate = standardize_dataset(flatten_features(x_validate), means, stdevs)  # use means and stdevs from training
    x_validate = x_validate.reshape(x_validate_shape)
    x_test = standardize_dataset(flatten_features(x_test), means, stdevs)
    x_test = x_test.reshape(x_test_shape)
    return x_train, x_validate, x_test


# Apply principle component analysis to denoise participant data
# patient data: list containing raw patient EEG data, with channels last
# returns a list of the features which is equivalent in size to the input list
def select_denoised_data(patient_data):
    all_features = []
    
    for entry in patient_data:
        pca_denoise = decomposition.PCA(n_components=np.asarray(patient_data).shape[-1])
        pca_denoise.fit(entry.transpose())
        denoised_data = pca_denoise.components_ 
        all_features.append(np.asarray(denoised_data)) 

    return all_features


########################
# with default settings
def get_raw_data(raw_data_dir, resolution_hz, ignore_list, excluded_channels, time_window,
                 shuffle_train_test_sets=False, use_common_train_test_idxs=True,
                 remove_outliers=True, overlap_rate=None):

    sz_file_durations = get_file_durations('SZ Patients', 's', raw_data_dir)
    hc_file_durations = get_file_durations("Healthy Controls", "h", raw_data_dir)
    minimum_duration = min(min(sz_file_durations), min(hc_file_durations))
    maximum_duration = max(max(sz_file_durations), max(hc_file_durations))
    median_duration =  np.median([sz_file_durations, hc_file_durations])
    print('Minimum EEG reading duration: ', minimum_duration, ' seconds')
    print('Maximum EEG reading duration: ', maximum_duration, ' seconds')
    print('Median EEG reading duration: ', median_duration, 'seconds')
    mne.set_log_level("WARNING")
    hc_data = process_patient_group('Healthy Controls', 'h', minimum_duration, resolution_hz, raw_data_dir, ignore_list,
                                    excluded_channels=excluded_channels)
    display('Shape of raw data for healthy controls: ', np.asarray(hc_data).shape)

    sz_data = process_patient_group('SZ Patients', 's', minimum_duration, resolution_hz, raw_data_dir, ignore_list,
                                    excluded_channels=excluded_channels)
    display('Shape of raw data for schizophrenic patients: ', np.asarray(sz_data).shape)

    if use_common_train_test_idxs:
        ################################
        # Manually assign patients, using previous randomly selected groups.
        # This is intended to help with stabilizing results while in the stage of finding the best model and methodology
        hc_train_idxs = [4, 7, 13, 2, 9, 6, 3, 1, 0, 5]
        hc_test_idxs = [11, 10]
        hc_validate_idxs = [8, 12]

        sz_train_idxs = [9, 5, 1, 7, 10, 0, 3, 4]
        sz_test_idxs = [2, 8]
        sz_validate_idxs = [11, 6]
    else:
        print('Selecting training  / testing / validation sets randomly from patient data')
        # Select patients for each set
        hc_train_idxs, hc_validate_idxs, hc_test_idxs = get_mixed_indexes_for_ml_train_test(len(hc_data), [.60, 0.2, 0.2])
        sz_train_idxs, sz_validate_idxs, sz_test_idxs = get_mixed_indexes_for_ml_train_test(len(sz_data), [.60, 0.2, 0.2])

    ############################################
    # append excluded indexes (due to rounding) to the train sets
    hc_train_idxs = hc_train_idxs \
                    + [i for i in range(0, len(hc_data)) if i not in
                       hc_train_idxs and i not in hc_validate_idxs and i not in hc_test_idxs]
    sz_train_idxs = sz_train_idxs \
                    + [i for i in range(0, len(sz_data)) if i not in
                       sz_train_idxs and i not in sz_validate_idxs and i not in sz_test_idxs]

    # select train/test/validate sets for each patient group
    print('Splitting data into time windows to improve stability of results')
    hc_data = np.asarray(hc_data)
    hc_train = windowize_list(hc_data[hc_train_idxs][0:, ], time_window, overlap_rate=overlap_rate)
    hc_validate = windowize_list(hc_data[hc_validate_idxs][0:, ], time_window, overlap_rate=overlap_rate)
    hc_test = windowize_list(hc_data[hc_test_idxs][0:, ], time_window, overlap_rate=overlap_rate)

    sz_data = np.asarray(sz_data)
    sz_train = windowize_list(sz_data[sz_train_idxs][0:, ], time_window, overlap_rate=overlap_rate)
    sz_validate = windowize_list(sz_data[sz_validate_idxs][0:, ], time_window, overlap_rate=overlap_rate)
    sz_test = windowize_list(sz_data[sz_test_idxs][0:, ], time_window, overlap_rate=overlap_rate)

    # Merge the sz and hc groups to form the full train, test and validation sets
    X_train = np.concatenate((hc_train, sz_train), axis=0)
    Y_train = ([0] * len(hc_train)) + ([1] * len(sz_train))

    X_validate = np.concatenate((hc_validate, sz_validate), axis=0)
    Y_validate = ([0] * len(hc_validate)) + ([1] * len(sz_validate))

    X_test = np.concatenate((hc_test, sz_test), axis=0)
    Y_test = ([0] * len(hc_test)) + ([1] * len(sz_test))

    x_train_shape = np.asarray(X_train).shape
    x_validate_shape = np.asarray(X_validate).shape
    x_test_shape = np.asarray(X_test).shape

    # Shuffle values and labels within their respective groups
    if shuffle_train_test_sets:
        X_train, Y_train = shuffle(X_train, Y_train)
        X_validate, Y_validate = shuffle(X_validate, Y_validate)
        X_test, Y_test = shuffle(X_test, Y_test)

    print(hc_data[hc_train_idxs][0:, ].shape)
    print(np.asarray(hc_train).shape)
    print('Shape of X_train: ', X_train.shape)
    print('Shape of X_validate: ', X_validate.shape)
    print('Shape of X_test: ', X_test.shape)

    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train, num_classes=2)
    Y_validate = np_utils.to_categorical(Y_validate, num_classes=2)
    Y_test = np_utils.to_categorical(Y_test, num_classes=2)

    # Reshape the data
    chans = len(all_channels) - len(excluded_channels)
    X_train = X_train.reshape(X_train.shape[0], time_window, chans)
    X_validate = X_validate.reshape(X_validate.shape[0], time_window, chans)
    X_test = X_test.reshape(X_test.shape[0], time_window, chans)

    # fit scaler on entire dataset; then transform each set separately
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=[0, 1])

    X_train = replace_outliers(X_train)
    X_validate = replace_outliers(X_validate)
    X_test = replace_outliers(X_test)

    if remove_outliers: # set outliers to median values for each channel
        ###########################
		# Standardize dataset
        dataset = X_train
        dataset_shape0 = np.asarray(dataset).shape

        # Estimate mean and standard deviation

        # use the training data to fit the scaler, as suggested here https://stackoverflow.com/a/50567308
        # flatten the features then reintroduce the 3rd dimension
        dataset_flattened = flatten_features(dataset)
        means = column_means(dataset_flattened)
        stdevs = column_stdevs(dataset_flattened, means)

        print('Removing outliers by channel')
        X_train = standardize_dataset(flatten_features(X_train),means, stdevs)
        X_train = X_train.reshape(x_train_shape)
        X_validate = standardize_dataset(flatten_features(X_validate), means, stdevs)  # use means and stdevs from training
        X_validate = X_validate.reshape(x_validate_shape)
        X_test = standardize_dataset(flatten_features(X_test), means, stdevs)
        X_test = X_test.reshape(x_test_shape)

    print('Selected indexes for control group: ')
    print('Training: ', hc_train_idxs)
    print('Testing: ', hc_test_idxs)
    print('Validation: ', hc_validate_idxs)

    print('\nSelected indexes for patient group: ')
    print('Training: ', sz_train_idxs)
    print('Testing: ', sz_test_idxs)
    print('Validation: ', sz_validate_idxs)
    return {'hc_data': hc_data, 'sz_data': sz_data,
            'X_train': X_train, 'Y_train': Y_train,
            'X_validate': X_validate, 'Y_validate': Y_validate,
            'X_test': X_test, 'Y_test': Y_test,
            'minimum_duration': minimum_duration,
            'maximum_duration': maximum_duration
            }


def clean_example_participant_list(participant_list, ignore_list, group_symbol):
    for participant_id in ignore_list:
        if participant_id.startswith(group_symbol):
            group_id = int(participant_id[1:])
            participant_list.remove(group_id)
    return participant_list


# build a list of ID's but drop excluded participants
def get_random_participants(raw_data_dir, data, ignore_list, resolution_hz):
    hc_participant_list = clean_example_participant_list(list(range(1, 15, 1)), ignore_list, 'h')
    rand_control_id = random.choice(hc_participant_list)

    sz_patient_list = clean_example_participant_list(list(range(1, 15, 1)), ignore_list, 's')
    rand_patient_id = random.choice(sz_patient_list)
    rand_control_file = raw_data_dir + 'Healthy Controls/{}{:02d}.edf'.format('h', rand_control_id)
    rand_patient_file = raw_data_dir + 'SZ Patients/{}.edf'.format("{}{:02d}").format('s', rand_patient_id)

    mne_raw_sz = get_raw_eeg_mne(rand_patient_file, resolution_hz, tmin=120,
                                             tmax=data['minimum_duration'] - 120, exclude=excluded_channels)
    mne_raw_hc = get_raw_eeg_mne(rand_control_file, resolution_hz, tmin=120,
                                             tmax=data['minimum_duration'] - 120, exclude=excluded_channels)
    return {'hc_id': rand_control_id, 'hc_raw_eeg': mne_raw_hc, 'sz_id': rand_patient_id, 'sz_raw_eeg': mne_raw_sz}

