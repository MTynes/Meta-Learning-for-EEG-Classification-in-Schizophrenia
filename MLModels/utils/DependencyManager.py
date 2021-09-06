import os
import sys
raw_data_dir = ''
acc_key = 'acc'
plot_examples = False # notebook file size will increase by between 30 and 60MB if set to True; <1MB otherwise

if  'COLAB_GPU' in os.environ :
    acc_key = 'accuracy'
    print('Using Google Colab. Setting up environment')
    raw_data_dir = '/content/drive/My Drive/Colab Notebooks/' 
    #raw_data_dir = 'Raw/'

    !pip install mne==0.19.2
    !pip install pyedflib==0.1.15
    !pip install chart_studio==1.0.0

    print('\n \n To load files from Google Drive, account validation is required.')
    #mount to drive -- files should be located in the /Colab Notebooks directory
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    
    if not os.path.exists('/content/tmp/eeg_sz/ReadData'):
        os.makedirs('/content/tmp/eeg_sz/ReadData')
        os.makedirs('/content/tmp/eeg_sz/utils')
    # download project utilities and data reader 
    !curl -u MTynes:08ec0e86b785be94d2cefc791029008c7155da8b https://raw.githubusercontent.com/WinAIML/schizophrenia/master/ReadData/RawDataReader.py > /content/tmp/eeg_sz/ReadData/RawDataReader.py
    !curl -u MTynes:08ec0e86b785be94d2cefc791029008c7155da8b https://raw.githubusercontent.com/WinAIML/schizophrenia/master/MLModels/utils/ModelBuilder.py > /content/tmp/eeg_sz/utils/ModelBuilder.py
    !curl -u MTynes:08ec0e86b785be94d2cefc791029008c7155da8b https://raw.githubusercontent.com/WinAIML/schizophrenia/master/MLModels/utils/ChartBuilder.py > /content/tmp/eeg_sz/utils/ChartBuilder.py
    sys.path.append('/content/tmp/eeg_sz/')
    
elif 'KAGGLE_URL_BASE' in os.environ:
    acc_key = 'accuracy'
    print('Using Kaggle kernel. Setting up environment')
    !pip install update mne==0.19.2
    !pip install pyedflib==0.1.15
    !pip install chart_studio
    !svn checkout https://github.com/WinAIML/schizophrenia/trunk/Data/Raw
    raw_data_dir = 'Raw/'
    
    if not os.path.exists('/kaggle/working/eeg_sz/ReadData'):
        os.makedirs('/kaggle/working/eeg_sz/ReadData')
        os.makedirs('/kaggle/working/eeg_sz/utils')
        # download project utilities and data reader 
    !curl -u MTynes:08ec0e86b785be94d2cefc791029008c7155da8b https://raw.githubusercontent.com/WinAIML/schizophrenia/master/ReadData/RawDataReader.py > /kaggle/working/eeg_sz/ReadData/RawDataReader.py
    !curl -u MTynes:08ec0e86b785be94d2cefc791029008c7155da8b https://raw.githubusercontent.com/WinAIML/schizophrenia/master/MLModels/utils/ModelBuilder.py > /kaggle/working/eeg_sz/utils/ModelBuilder.py
    !curl -u MTynes:08ec0e86b785be94d2cefc791029008c7155da8b https://raw.githubusercontent.com/WinAIML/schizophrenia/master/MLModels/utils/ChartBuilder.py > /kaggle/working/eeg_sz/utils/ChartBuilder.py

    sys.path.append('/kaggle/working/eeg_sz/')

    # Dataset needs to be manually added to the current session. Find the directory name using
    # print(os.listdir("../input"))

    # Then set the data directory
    raw_data_dir = '../input/eeg-in-schizophrenia/Raw/'

        
else: 
    # assuming that a local run will be launched only from a github project; 
    # add the utils and ReadData directories to the temporary path
    if 'HOMEPATH' in os.environ:
        print('Using homepath ' + os.environ['HOMEPATH'])
    raw_data_dir = '../../Data/Raw/'
    
    from pathlib import Path
    import sys
    sys.path.append(os.path.realpath('..'))
    path = Path(os.getcwd())
    sys.path.append(str(path.parent.parent))


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, AveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

import pandas as pd
import numpy as np
import random 

from importlib import reload  #reload(chart_builder)


#################
# import project utilities and the raw data reader
# Kaggle environment does not accept 'utils' as a file, so it must be accessed seperately

import ReadData.RawDataReader as data_reader
import utils.ModelBuilder as model_builder
import utils.ChartBuilder as chart_builder





import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, AveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

import pandas as pd
import numpy as np
import random 

from importlib import reload  #reload(chart_builder)


#################
# import project utilities and the raw data reader
# Kaggle environment does not accept 'utils' as a file, so it must be accessed seperately

import ReadData.RawDataReader as data_reader
import utils.ModelBuilder as model_builder
import utils.ChartBuilder as chart_builder
