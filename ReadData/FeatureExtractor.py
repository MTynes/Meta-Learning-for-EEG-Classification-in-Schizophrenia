import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct, idct


# read in raw files
hc_data_raw = np.load("HC_data_raw.npy")
sz_data_raw = np.load("Sz_data_raw.npy")

# create train / validate / test sets
X = np.concatenate((hc_data_raw, sz_data_raw), axis=0)  # combine groups
y = ([0] * len(hc_data_raw)) +([1] * len(sz_data_raw))  # determine labels

# from https://datascience.stackexchange.com/a/15136
# 1. determine test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1)
# 2. from remaining values, determine training and validation sets
X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# extract FFT features
# adapted from: https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html#type-iii-dct
def fast_fourier_transform(x, dct_type=2):
    # norm = 'None' | 'ortho'
    # get discreet cosine transforms
    fft_dct = dct(dct(x, type=dct_type, norm='ortho'), type=3, norm='ortho')
    # get inverse discreet cosine transforms
    fft_idct = idct(dct(x, type=dct_type), type=2)
    return fft_dct, fft_idct


train_fft, train_inverse_fft = fast_fourier_transform(X_train)
test_fft, test_inverse_fft = fast_fourier_transform(X_test)
validation_fft, validation_inverse_fft = fast_fourier_transform(X_val)
np.save("Train_FFT", train_fft)
np.save("Validation_FFT", validation_fft)
np.save("Test_FFT", test_fft)
print(np.asarray(train_fft).shape)