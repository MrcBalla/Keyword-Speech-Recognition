import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from python_speech_features import logfbank, mfcc, delta
import seaborn as sns
import random
import tarfile
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import contextlib
import lzma
import tarfile
import matplotlib.pyplot as plt
import scipy

np.random.seed(1234)
random.seed(1234)
tf.random.set_seed(1234)

global label_list
label_list = [
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'down',
    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'go',
    'happy',
    'house',
    'learn',
    'left',
    'marvin',
    'nine',
    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',
    'sheila',
    'six',
    'stop',
    'three',
    'tree',
    'two',
    'up',
    'visual',
    'wow',
    'yes',
    'zero'
]

def load_label_wav_file(file_name, label):
    try:
      file_name=file_name.decode('utf-8')
      label=label.decode('utf-8')
    except AttributeError:
      pass

    _,signal=scipy.io.wavfile.read("content/speech_recognition/{}/{}".format(label, file_name))

    signal=padding_cropping(signal)

    norm_factor=1/(np.max(np.abs(signal)))
    signal=signal*norm_factor

    return signal.astype(np.float32)


def get_spectrogram(signal,samplerate = 16000,winlen     = 25,winstep    = 10,nfft       = 512, winfunc    = tf.signal.hamming_window):

    spectrogram = tf.signal.stft(signal.astype(float),int(samplerate*winlen/1000),int(samplerate*winstep/1000),nfft,winfunc)
    spectrogram = tf.abs(spectrogram)

    spectrogram = np.array(spectrogram)

    spectrogram = np.log(spectrogram.T + np.finfo(float).eps)

    return spectrogram.astype(np.float32)


def padding_cropping(data, output_sequence_length=16000):

    data_shape = data.shape[0]

    if data_shape > output_sequence_length:
        data = data[:output_sequence_length]

    elif data_shape < output_sequence_length:
        tot_pad    = output_sequence_length - data_shape
        pad_before = int(np.ceil(tot_pad/2))
        pad_after  = int(np.floor(tot_pad/2))
        data       = np.pad(data, pad_width=(pad_before, pad_after), mode='mean')

    return data

def get_logfbank(signal, samplerate = 16000, winlen     = 25,    winstep    = 10,
                nfilt      = 40, nfft       = 512, lowfreq    = 300,   highfreq   = 16000 / 2):
    '''
    Function to compute log Filterbank Energies

    signal: input signal
    samplerate: sample rate of the signal
    winlen: the length of the analysis window in seconds
    winstep: the step between successive windows
    nfilt: the number of filters in the filterbank, default would be 26, in this implementation
            in order to be coherent with the paper, the default is 40
    nfft: FFT size
    lowfreq: lowest band edge of mel filters
    highfreq:  highest band edge of mel filters
    '''


    logfbank_feat = logfbank(signal, samplerate, winlen/1000, winstep/1000, nfilt,
                            nfft, lowfreq,highfreq)
    logfbank_feat = logfbank_feat.T

    return logfbank_feat.astype(np.float32)


def get_mfcc(signal, delta_order= 2, delta_window = 1, samplerate  = 16000,
             winlen = 25,winstep  = 10, numcep = 13,nfilt  = 40, nfft  = 512,   lowfreq= 300,
             highfreq= None,  appendEnergy = True,  winfunc = np.hamming , new_channel=False):

    '''
    Compute MFCC features from an audio signal
    signal: input signal
    samplerate: sample rate of the signal
    winlen: the length of the analysis window in seconds
    winstep: the step between successive windows
    numcep: the number of cepstrum to return
    nfilt: the number of filters in the filterbank, default would be 26, in this implementation
            in order to be coherent with the paper, the default is 40
    nfft: FFT size
    lowfreq: lowest band edge of mel filters
    highfreq:  highest band edge of mel filters
    appendenergy: if this is true, the zeroth cepstral coefficient is
                  replaced with the log of the total frame energy.
    winfunc:  the analysis window to apply to each frame.
    '''
    
    if unfold_mask==True:
        delta_order=0
    
    features = []

    # Extract MFCC features
    mfcc_feat = mfcc(signal,samplerate,  winlen/1000, winstep/1000, numcep,nfilt, nfft,
                    lowfreq,highfreq,appendEnergy=appendEnergy, winfunc=winfunc)
    mfcc_feat = mfcc_feat.T
    features.append(mfcc_feat)
    
    for i in range(delta_order):
        features.append(delta(features[-1], delta_window))

    full_feat = np.vstack(features)

    return full_feat.astype(np.float32)

def browse_directory():
    label_file_dict={}
    for elem in label_list:
        files=os.listdir("content/speech_recognition/{}".format(elem))
        label_file_dict[elem]=files
    return label_file_dict

def one_hot_encoding(l, elem):
    return np.where(np.array(l)==elem, 1, 0).astype(np.float32)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def from_string_to_number(l,elem):
    a = np.where(np.array(l)==elem, 1, 0).astype(np.float32)
    return np.argwhere(a==1)[0][0].astype(np.float32)


def split(df, train_size, validation_size):

    n = df.shape[0]
    train_mask = random.sample(range(0,n), int(n*train_size))
    valid_set = set(list(range(0,n)))-set(train_mask)
    validation_mask = random.sample(list(valid_set), int(n*validation_size))
    return train_mask, validation_mask

def _set_shape_auto(data_1, data_2):
    
    if typ == 'mfcc':
        data_1=tf.reshape(data_1, shape=[39,99,1])
        data_2=tf.reshape(data_2, shape=[39,99,1])
    if typ == 'logfbank':
        data_1=tf.reshape(data_1, shape=[40,99,1])
        data_2=tf.reshape(data_2, shape=[40,99,1])
    if typ == 'spectrogram':
        data_1=tf.reshape(data_1, shape=[257,98,1])
        data_2=tf.reshape(data_2, shape=[257,98,1])
    
    return data_1, data_2

def _set_shape(data_1, label):
    
    if typ == 'mfcc' and unfold_mask==True:
        data_1=tf.reshape(data_1, shape=[13,99,1])
    if typ == 'mfcc' and unfold_mask==False:
        data_1=tf.reshape(data_1, shape=[39,99,1])
    if typ == 'logfbank':
        data_1=tf.reshape(data_1, shape=[40,99,1])
    if typ == 'spectrogram':
        data_1=tf.reshape(data_1, shape=[257,98,1])
        
    return data_1, label


def apply_noise(data_signal, signal_parameter=0.5):
    noise_signal = load_label_wav_file('white_noise.wav', '_background_noise_')
    audio_noised = data_signal + noise_signal*signal_parameter
    
    norm_factor=1/(np.max(np.abs(audio_noised)))
    audio_noised=audio_noised*norm_factor
    
    return audio_noised

def create_dataset_autoencoder(file_names, label, batch_size ,cache_file_value=False, cache_file_label=False, shuffle=False, batching=True, repeat=False, type='logfbank', input_noise=False, simple_mfcc=False):
    
    global typ
    typ=type
    
    global unfold_mask
    unfold_mask=simple_mfcc
    
    index_seed = np.random.randint(0, 99999999, 1)[0]

    assert type in ["logfbank", "mfcc", "spectrogram"]

    file_names=file_names.tolist()
    file_label=label.tolist()

    dataset = tf.data.Dataset.from_tensor_slices((file_names, file_label))

    dataset = dataset.map(lambda file_name, file_label: (tf.numpy_function(load_label_wav_file, [file_name, file_label], tf.float32), file_label), num_parallel_calls=os.cpu_count())

    dataset = dataset.map(lambda data, file_label: (data, tf.numpy_function(from_string_to_number,  [label_list, file_label], tf.float32)), num_parallel_calls=os.cpu_count())

    dataset_value = dataset.map(lambda data_1, label: (data_1, data_1))
    dataset_label = dataset.map(lambda data_1, label: (label))

    if input_noise:
        dataset_value = dataset_value.map(lambda data_1, data_2: (tf.numpy_function(apply_noise,  [data_1], tf.float32), data_2), num_parallel_calls=os.cpu_count())

    if type=='logfbank':
        dataset_value = dataset_value.map(lambda data_1,data_2: (tf.numpy_function(get_logfbank, [data_1], tf.float32), data_2))
        dataset_value = dataset_value.map(lambda data_1,data_2: (data_1,tf.numpy_function(get_logfbank, [data_2], tf.float32)))

    elif type=='mfcc':
        dataset_value = dataset_value.map(lambda data_1,data_2: (tf.numpy_function(get_mfcc, [data_1], tf.float32), data_2))
        dataset_value = dataset_value.map(lambda data_1,data_2: (data_1, tf.numpy_function(get_mfcc, [data_2], tf.float32)))  

    elif type=='spectrogram':
        dataset_value = dataset_value.map(lambda data_1,data_2: (tf.numpy_function(get_spectrogram, [data_1], tf.float32), data_2))
        dataset_value = dataset_value.map(lambda data_1,data_2: (data_1, tf.numpy_function(get_spectrogram, [data_2], tf.float32))) 

    dataset_value = dataset_value.map(_set_shape_auto)       

    if cache_file_value:
        dataset_value = dataset_value.cache(cache_file_value)
        dataset_label = dataset_label.cache(cache_file_label)

    if shuffle:
        dataset_value = dataset_value.shuffle(len(file_names), seed=index_seed)
        dataset_label = dataset_label.shuffle(len(file_names), seed=index_seed)
        print("Shuffled dataset")

    if repeat:
      dataset_value = dataset_value.repeat()

    dataset_value = dataset_value.batch(batch_size=batch_size)

    dataset_value = dataset_value.prefetch(buffer_size=1)

    return dataset_value, dataset_label


def create_dataset(file_names, label, batch_size, cache_file_value=False, shuffle=False, repeat=True, type='logfbank', testing=False, simple_mfcc=False, input_noise=False):

    global typ
    typ=type

    global unfold_mask
    unfold_mask=simple_mfcc

    label_df=[]

    assert type in ["logfbank", "mfcc", "spectrogram"]
    
    file_names = file_names.tolist()
    file_label = label.tolist()

    ht=LabelBinarizer(sparse_output=False)

    dataset=tf.data.Dataset.from_tensor_slices((file_names,file_label))

    dataset = dataset.map(lambda file_name, file_label: (tf.numpy_function(load_label_wav_file, [file_name, file_label], tf.float32), file_label), num_parallel_calls=os.cpu_count())

    dataset = dataset.map(lambda data, file_label: (data, tf.numpy_function(from_string_to_number,  [label_list, file_label], tf.float32)), num_parallel_calls=os.cpu_count())

    if input_noise:
        dataset = dataset.map(lambda data_1, label: (tf.numpy_function(apply_noise,  [data_1], tf.float32), label), num_parallel_calls=os.cpu_count())

    if type=='logfbank':
        dataset = dataset.map(lambda data,label: (tf.numpy_function(get_logfbank, [data], tf.float32), label))
    elif type=='mfcc':
        dataset = dataset.map(lambda data,label: (tf.numpy_function(get_mfcc, [data], tf.float32), label))
    elif type=='spectrogram':
        dataset = dataset.map(lambda data,label: (tf.numpy_function(get_spectrogram, [data], tf.float32), label))

    dataset = dataset.map(_set_shape)

    if testing:
        label_df=ht.fit_transform(file_label)
    else:
        label_df = None

    if cache_file_value:
        dataset = dataset.cache(cache_file_value)
        
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_names))
        print("Shuffled dataset")

    if repeat:
      dataset = dataset.repeat()

    dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.prefetch(buffer_size=1)

    return dataset, label_df

