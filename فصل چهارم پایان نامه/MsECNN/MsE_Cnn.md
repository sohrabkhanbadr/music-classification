use thensorflow for this code


```python
import tensorflow as tf
```


```python
tf.__version__
```




    '1.15.3'




```python

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

```

    tf.__version__ is 1.15.3
    tf.keras.__version__ is: 2.2.4-tf


    Using TensorFlow backend.


from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file


```python
from tensorflow.keras.models import Sequential
```


```python
'''
MsE-CNN for Music Tagging in Keras
Nima Hamidi - April 2019
'''

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, ELU
from keras.layers.merge import Concatenate

'''-------------------
   Required functions
---------------------'''
def concat (L1, L2):
    L = Concatenate()([L1, L2])
    return L

'''-------------------------------------------
Traditional CNN for Music tagging
This model has been developped by Keunwoo Choi
This model is baseline for MsE-CNN model
--------------------------------------------'''
def MusicTaggerCNN(weights='msd', input_tensor=None, include_top=True):

    input_shape = (96, 1366, 1)
    melgram_input = Input(shape=input_shape)
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    # Input block
    x = BatchNormalization(name='bn_0_freq', axis=1)(melgram_input)

    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn1', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)

    # Conv block 5
    x = Conv2D(64, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    x = Flatten()(x)
    if include_top:
        x = Dense(10, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else:
        model.load_weights('data/cnn_weights.h5',by_name=True)
        return model

'''-------------------------------------------
Multi-scale CNN for Music tagging
This model has been developped by Nima Hamidi
This is a keras base model for music Tagging based on a the paper
"Multi-scale CNN for Music Tagging, accepted at ML4MD at ICML"
--------------------------------------------'''
def MS_CNN_MusicTagger(weights='msd', input_tensor=None, include_top=True):

    input_shape = (96, 1366, 1)
    melgram_input = Input(shape=input_shape)
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    # Input block
    x = BatchNormalization(name='bn_0_freq', axis=1)(melgram_input)
    x_g = MaxPooling2D(pool_size=(2, 4), name='pool1_g')(x)

    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn1', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(2, 4), name='pool1_comb')(x_comb)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(2, 4), name='pool2_comb')(x_comb)

    # Conv block 3
    x = Conv2D(256, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(3, 5), name='pool3_comb')(x_comb)
    # Conv block 4
    x = Conv2D(512, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(4, 4), name='pool4_comb')(x_comb)

    # Conv block 5
    x = Conv2D(1024, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)
    x = concat(x_g, x)

    # Output
    x = Flatten()(x)
    if include_top:
        x = Dense(10, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)

    if weights is None:
        return model
    else:
        model.load_weights('data/MScnn_weights.h5',by_name=True)
        return model

if __name__ == "__main__":
    model = MS_CNN_MusicTagger(weights=None, input_tensor=None, include_top=True)
```

    WARNING:tensorflow:From /home/user/Music/gtzan4work/genres/mseCnn/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From /home/user/Music/gtzan4work/genres/mseCnn/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    


### -*- coding: utf-8 -*-
MusicTaggerCRNN model for <font color='red'>Keras</font>.
### Reference:
- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)



```python
TH_WEIGHTS_PATH = 'https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_crnn_weights_theano.h5'
TF_WEIGHTS_PATH = 'https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/music_tagger_crnn_weights_tensorflow.h5'
```

# MusicTaggerCRNN


   Instantiate the MusicTaggerCRNN architecture, optionally loading weights pre-trained on Million Song Dataset. Note that when using TensorFlow, for best performance you should set image_dim_ordering="tf" in your Keras config at ~/.keras/keras.json.
    
   The model and the weights are compatible with both TensorFlow and Theano. The dimension ordering convention used by the model is the one specified in your Keras config file.
    
    
For preparing mel-spectrogram input, see [audio_conv_utils.py] in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it. 


## Arguments

   weights: one of `None` (random initialization)
       or "msd" (pre-training on ImageNet).
   input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
       to use as image input for the model.
   include_top: whether to include the 1 fully-connected
       layer (output layer) at the top of the network.
       If False, the network outputs 32-dim features.
            
## Returns

   A Keras model instance.



```python

```


```python
def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.
    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if keras.backend.image_data_format() == 'channels_first':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if keras.backend.image_data_format() == 'channels_first':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)
    if include_top:
        x = Dense(10, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else: 
        # Load input
        if keras.backend.image_data_format() == 'channels_last':
            raise RuntimeError("Please set keras.backend.image_data_format() == 'channels_first'."
                               "You can set it at ~/.keras/keras.json")
    
        model.load_weights('data/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)
        return model
```


```python
from IPython.display import Image
Image(filename='tf_th_keras_v2.png')
```




![png](output_13_0.png)




```python
import keras

if keras.backend.image_data_format() == 'channels_last':
    print("here backend is tensorflow" , keras.backend.image_data_format())
elif keras.backend.image_data_format() == 'channels_first':
    print("here backend is theano" , keras.backend.image_data_format())

#print(K.image_dim_ordering())
```

    here backend is tensorflow channels_last


TRY TO MAKE A MODEL <font color='green'>USER-FRIENDLY</font> IN NEAR FUTURE.

model = MusicTaggerCRNN(weights=None)


```python
model = MS_CNN_MusicTagger(weights=None, input_tensor=None, include_top=True)
```


```python
model.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            (None, 96, 1366, 1)  0                                            
    __________________________________________________________________________________________________
    bn_0_freq (BatchNormalization)  (None, 96, 1366, 1)  384         input_2[0][0]                    
    __________________________________________________________________________________________________
    conv1 (Conv2D)                  (None, 96, 1366, 64) 640         bn_0_freq[0][0]                  
    __________________________________________________________________________________________________
    bn1 (BatchNormalization)        (None, 96, 1366, 64) 256         conv1[0][0]                      
    __________________________________________________________________________________________________
    elu_6 (ELU)                     (None, 96, 1366, 64) 0           bn1[0][0]                        
    __________________________________________________________________________________________________
    pool1 (MaxPooling2D)            (None, 48, 341, 64)  0           elu_6[0][0]                      
    __________________________________________________________________________________________________
    conv2 (Conv2D)                  (None, 48, 341, 128) 73856       pool1[0][0]                      
    __________________________________________________________________________________________________
    bn2 (BatchNormalization)        (None, 48, 341, 128) 512         conv2[0][0]                      
    __________________________________________________________________________________________________
    elu_7 (ELU)                     (None, 48, 341, 128) 0           bn2[0][0]                        
    __________________________________________________________________________________________________
    pool2 (MaxPooling2D)            (None, 24, 85, 128)  0           elu_7[0][0]                      
    __________________________________________________________________________________________________
    conv3 (Conv2D)                  (None, 24, 85, 256)  295168      pool2[0][0]                      
    __________________________________________________________________________________________________
    bn3 (BatchNormalization)        (None, 24, 85, 256)  1024        conv3[0][0]                      
    __________________________________________________________________________________________________
    elu_8 (ELU)                     (None, 24, 85, 256)  0           bn3[0][0]                        
    __________________________________________________________________________________________________
    pool1_g (MaxPooling2D)          (None, 48, 341, 1)   0           bn_0_freq[0][0]                  
    __________________________________________________________________________________________________
    pool3 (MaxPooling2D)            (None, 12, 21, 256)  0           elu_8[0][0]                      
    __________________________________________________________________________________________________
    concatenate_6 (Concatenate)     (None, 48, 341, 65)  0           pool1_g[0][0]                    
                                                                     pool1[0][0]                      
    __________________________________________________________________________________________________
    conv4 (Conv2D)                  (None, 12, 21, 512)  1180160     pool3[0][0]                      
    __________________________________________________________________________________________________
    pool1_comb (MaxPooling2D)       (None, 24, 85, 65)   0           concatenate_6[0][0]              
    __________________________________________________________________________________________________
    bn4 (BatchNormalization)        (None, 12, 21, 512)  2048        conv4[0][0]                      
    __________________________________________________________________________________________________
    concatenate_7 (Concatenate)     (None, 24, 85, 193)  0           pool1_comb[0][0]                 
                                                                     pool2[0][0]                      
    __________________________________________________________________________________________________
    elu_9 (ELU)                     (None, 12, 21, 512)  0           bn4[0][0]                        
    __________________________________________________________________________________________________
    pool2_comb (MaxPooling2D)       (None, 12, 21, 193)  0           concatenate_7[0][0]              
    __________________________________________________________________________________________________
    pool4 (MaxPooling2D)            (None, 4, 4, 512)    0           elu_9[0][0]                      
    __________________________________________________________________________________________________
    concatenate_8 (Concatenate)     (None, 12, 21, 449)  0           pool2_comb[0][0]                 
                                                                     pool3[0][0]                      
    __________________________________________________________________________________________________
    conv5 (Conv2D)                  (None, 4, 4, 1024)   4719616     pool4[0][0]                      
    __________________________________________________________________________________________________
    pool3_comb (MaxPooling2D)       (None, 4, 4, 449)    0           concatenate_8[0][0]              
    __________________________________________________________________________________________________
    bn5 (BatchNormalization)        (None, 4, 4, 1024)   4096        conv5[0][0]                      
    __________________________________________________________________________________________________
    concatenate_9 (Concatenate)     (None, 4, 4, 961)    0           pool3_comb[0][0]                 
                                                                     pool4[0][0]                      
    __________________________________________________________________________________________________
    elu_10 (ELU)                    (None, 4, 4, 1024)   0           bn5[0][0]                        
    __________________________________________________________________________________________________
    pool4_comb (MaxPooling2D)       (None, 1, 1, 961)    0           concatenate_9[0][0]              
    __________________________________________________________________________________________________
    pool5 (MaxPooling2D)            (None, 1, 1, 1024)   0           elu_10[0][0]                     
    __________________________________________________________________________________________________
    concatenate_10 (Concatenate)    (None, 1, 1, 1985)   0           pool4_comb[0][0]                 
                                                                     pool5[0][0]                      
    __________________________________________________________________________________________________
    flatten_2 (Flatten)             (None, 1985)         0           concatenate_10[0][0]             
    __________________________________________________________________________________________________
    output (Dense)                  (None, 10)           19860       flatten_2[0][0]                  
    ==================================================================================================
    Total params: 6,297,620
    Trainable params: 6,293,460
    Non-trainable params: 4,160
    __________________________________________________________________________________________________


### Compiling the model


```python
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

### Load data


```python
import numpy as np
```


```python
concat_x = np.load('concat_x.npy')
concat_y = np.load('concat_y.npy')
```


```python
print(len(concat_x),'   ',(len(concat_y)) )
```

    1000     1000



```python
train_x = concat_x[0:750]
train_y = concat_y[0:750]
```


```python
valid_x = concat_x[750:900]
valid_y = concat_y[750:900]
```


```python
test_x = concat_x[900:1000]
test_y = concat_y[900:1000]
```




```python
np.zeros((3,3)).reshape((3,3,1))
```




    array([[[0.],
            [0.],
            [0.]],
    
           [[0.],
            [0.],
            [0.]],
    
           [[0.],
            [0.],
            [0.]]])




```python
test_x[5][0].reshape((96,1366,1)).shape
```




    (96, 1366, 1)






```python
train_x_reshape = [] #np.zeros((800,96,1366,1))
```


```python
for item in train_x:
    # print(item[0].reshape((96,1366,1)).shape)
    #test_x_reshape.append = item[0].reshape((96,1366,1))
    train_x_reshape.append(item[0].reshape((96,1366,1)))
    print(np.array(train_x_reshape).shape)
```

    (1, 96, 1366, 1)
    (2, 96, 1366, 1)
    (3, 96, 1366, 1)
    (4, 96, 1366, 1)
    (5, 96, 1366, 1)
    (6, 96, 1366, 1)
    (7, 96, 1366, 1)
    (8, 96, 1366, 1)
    (9, 96, 1366, 1)
    (10, 96, 1366, 1)
    (11, 96, 1366, 1)
    (12, 96, 1366, 1)
    (13, 96, 1366, 1)
    (14, 96, 1366, 1)
    (15, 96, 1366, 1)
    (16, 96, 1366, 1)
    (17, 96, 1366, 1)
    (18, 96, 1366, 1)
    (19, 96, 1366, 1)
    (20, 96, 1366, 1)
    (21, 96, 1366, 1)
    (22, 96, 1366, 1)
    (23, 96, 1366, 1)
    (24, 96, 1366, 1)
    (25, 96, 1366, 1)
    (26, 96, 1366, 1)
    (27, 96, 1366, 1)
    (28, 96, 1366, 1)
    (29, 96, 1366, 1)
    (30, 96, 1366, 1)
    (31, 96, 1366, 1)
    (32, 96, 1366, 1)
    (33, 96, 1366, 1)
    (34, 96, 1366, 1)
    (35, 96, 1366, 1)
    (36, 96, 1366, 1)
    (37, 96, 1366, 1)
    (38, 96, 1366, 1)
    (39, 96, 1366, 1)
    (40, 96, 1366, 1)
    (41, 96, 1366, 1)
    (42, 96, 1366, 1)
    (43, 96, 1366, 1)
    (44, 96, 1366, 1)
    (45, 96, 1366, 1)
    (46, 96, 1366, 1)
    (47, 96, 1366, 1)
    (48, 96, 1366, 1)
    (49, 96, 1366, 1)
    (50, 96, 1366, 1)
    (51, 96, 1366, 1)
    (52, 96, 1366, 1)
    (53, 96, 1366, 1)
    (54, 96, 1366, 1)
    (55, 96, 1366, 1)
    (56, 96, 1366, 1)
    (57, 96, 1366, 1)
    (58, 96, 1366, 1)
    (59, 96, 1366, 1)
    (60, 96, 1366, 1)
    (61, 96, 1366, 1)
    (62, 96, 1366, 1)
    (63, 96, 1366, 1)
    (64, 96, 1366, 1)
    (65, 96, 1366, 1)
    (66, 96, 1366, 1)
    (67, 96, 1366, 1)
    (68, 96, 1366, 1)
    (69, 96, 1366, 1)
    (70, 96, 1366, 1)
    (71, 96, 1366, 1)
    (72, 96, 1366, 1)
    (73, 96, 1366, 1)
    (74, 96, 1366, 1)
    (75, 96, 1366, 1)
    (76, 96, 1366, 1)
    (77, 96, 1366, 1)
    (78, 96, 1366, 1)
    (79, 96, 1366, 1)
    (80, 96, 1366, 1)
    (81, 96, 1366, 1)
    (82, 96, 1366, 1)
    (83, 96, 1366, 1)
    (84, 96, 1366, 1)
    (85, 96, 1366, 1)
    (86, 96, 1366, 1)
    (87, 96, 1366, 1)
    (88, 96, 1366, 1)
    (89, 96, 1366, 1)
    (90, 96, 1366, 1)
    (91, 96, 1366, 1)
    (92, 96, 1366, 1)
    (93, 96, 1366, 1)
    (94, 96, 1366, 1)
    (95, 96, 1366, 1)
    (96, 96, 1366, 1)
    (97, 96, 1366, 1)
    (98, 96, 1366, 1)
    (99, 96, 1366, 1)
    (100, 96, 1366, 1)
    (101, 96, 1366, 1)
    (102, 96, 1366, 1)
    (103, 96, 1366, 1)
    (104, 96, 1366, 1)
    (105, 96, 1366, 1)
    (106, 96, 1366, 1)
    (107, 96, 1366, 1)
    (108, 96, 1366, 1)
    (109, 96, 1366, 1)
    (110, 96, 1366, 1)
    (111, 96, 1366, 1)
    (112, 96, 1366, 1)
    (113, 96, 1366, 1)
    (114, 96, 1366, 1)
    (115, 96, 1366, 1)
    (116, 96, 1366, 1)
    (117, 96, 1366, 1)
    (118, 96, 1366, 1)
    (119, 96, 1366, 1)
    (120, 96, 1366, 1)
    (121, 96, 1366, 1)
    (122, 96, 1366, 1)
    (123, 96, 1366, 1)
    (124, 96, 1366, 1)
    (125, 96, 1366, 1)
    (126, 96, 1366, 1)
    (127, 96, 1366, 1)
    (128, 96, 1366, 1)
    (129, 96, 1366, 1)
    (130, 96, 1366, 1)
    (131, 96, 1366, 1)
    (132, 96, 1366, 1)
    (133, 96, 1366, 1)
    (134, 96, 1366, 1)
    (135, 96, 1366, 1)
    (136, 96, 1366, 1)
    (137, 96, 1366, 1)
    (138, 96, 1366, 1)
    (139, 96, 1366, 1)
    (140, 96, 1366, 1)
    (141, 96, 1366, 1)
    (142, 96, 1366, 1)
    (143, 96, 1366, 1)
    (144, 96, 1366, 1)
    (145, 96, 1366, 1)
    (146, 96, 1366, 1)
    (147, 96, 1366, 1)
    (148, 96, 1366, 1)
    (149, 96, 1366, 1)
    (150, 96, 1366, 1)
    (151, 96, 1366, 1)
    (152, 96, 1366, 1)
    (153, 96, 1366, 1)
    (154, 96, 1366, 1)
    (155, 96, 1366, 1)
    (156, 96, 1366, 1)
    (157, 96, 1366, 1)
    (158, 96, 1366, 1)
    (159, 96, 1366, 1)
    (160, 96, 1366, 1)
    (161, 96, 1366, 1)
    (162, 96, 1366, 1)
    (163, 96, 1366, 1)
    (164, 96, 1366, 1)
    (165, 96, 1366, 1)
    (166, 96, 1366, 1)
    (167, 96, 1366, 1)
    (168, 96, 1366, 1)
    (169, 96, 1366, 1)
    (170, 96, 1366, 1)
    (171, 96, 1366, 1)
    (172, 96, 1366, 1)
    (173, 96, 1366, 1)
    (174, 96, 1366, 1)
    (175, 96, 1366, 1)
    (176, 96, 1366, 1)
    (177, 96, 1366, 1)
    (178, 96, 1366, 1)
    (179, 96, 1366, 1)
    (180, 96, 1366, 1)
    (181, 96, 1366, 1)
    (182, 96, 1366, 1)
    (183, 96, 1366, 1)
    (184, 96, 1366, 1)
    (185, 96, 1366, 1)
    (186, 96, 1366, 1)
    (187, 96, 1366, 1)
    (188, 96, 1366, 1)
    (189, 96, 1366, 1)
    (190, 96, 1366, 1)
    (191, 96, 1366, 1)
    (192, 96, 1366, 1)
    (193, 96, 1366, 1)
    (194, 96, 1366, 1)
    (195, 96, 1366, 1)
    (196, 96, 1366, 1)
    (197, 96, 1366, 1)
    (198, 96, 1366, 1)
    (199, 96, 1366, 1)
    (200, 96, 1366, 1)
    (201, 96, 1366, 1)
    (202, 96, 1366, 1)
    (203, 96, 1366, 1)
    (204, 96, 1366, 1)
    (205, 96, 1366, 1)
    (206, 96, 1366, 1)
    (207, 96, 1366, 1)
    (208, 96, 1366, 1)
    (209, 96, 1366, 1)
    (210, 96, 1366, 1)
    (211, 96, 1366, 1)
    (212, 96, 1366, 1)
    (213, 96, 1366, 1)
    (214, 96, 1366, 1)
    (215, 96, 1366, 1)
    (216, 96, 1366, 1)
    (217, 96, 1366, 1)
    (218, 96, 1366, 1)
    (219, 96, 1366, 1)
    (220, 96, 1366, 1)
    (221, 96, 1366, 1)
    (222, 96, 1366, 1)
    (223, 96, 1366, 1)
    (224, 96, 1366, 1)
    (225, 96, 1366, 1)
    (226, 96, 1366, 1)
    (227, 96, 1366, 1)
    (228, 96, 1366, 1)
    (229, 96, 1366, 1)
    (230, 96, 1366, 1)
    (231, 96, 1366, 1)
    (232, 96, 1366, 1)
    (233, 96, 1366, 1)
    (234, 96, 1366, 1)
    (235, 96, 1366, 1)
    (236, 96, 1366, 1)
    (237, 96, 1366, 1)
    (238, 96, 1366, 1)
    (239, 96, 1366, 1)
    (240, 96, 1366, 1)
    (241, 96, 1366, 1)
    (242, 96, 1366, 1)
    (243, 96, 1366, 1)
    (244, 96, 1366, 1)
    (245, 96, 1366, 1)
    (246, 96, 1366, 1)
    (247, 96, 1366, 1)
    (248, 96, 1366, 1)
    (249, 96, 1366, 1)
    (250, 96, 1366, 1)
    (251, 96, 1366, 1)
    (252, 96, 1366, 1)
    (253, 96, 1366, 1)
    (254, 96, 1366, 1)
    (255, 96, 1366, 1)
    (256, 96, 1366, 1)
    (257, 96, 1366, 1)
    (258, 96, 1366, 1)
    (259, 96, 1366, 1)
    (260, 96, 1366, 1)
    (261, 96, 1366, 1)
    (262, 96, 1366, 1)
    (263, 96, 1366, 1)
    (264, 96, 1366, 1)
    (265, 96, 1366, 1)
    (266, 96, 1366, 1)
    (267, 96, 1366, 1)
    (268, 96, 1366, 1)
    (269, 96, 1366, 1)
    (270, 96, 1366, 1)
    (271, 96, 1366, 1)
    (272, 96, 1366, 1)
    (273, 96, 1366, 1)
    (274, 96, 1366, 1)
    (275, 96, 1366, 1)
    (276, 96, 1366, 1)
    (277, 96, 1366, 1)
    (278, 96, 1366, 1)
    (279, 96, 1366, 1)
    (280, 96, 1366, 1)
    (281, 96, 1366, 1)
    (282, 96, 1366, 1)
    (283, 96, 1366, 1)
    (284, 96, 1366, 1)
    (285, 96, 1366, 1)
    (286, 96, 1366, 1)
    (287, 96, 1366, 1)
    (288, 96, 1366, 1)
    (289, 96, 1366, 1)
    (290, 96, 1366, 1)
    (291, 96, 1366, 1)
    (292, 96, 1366, 1)
    (293, 96, 1366, 1)
    (294, 96, 1366, 1)
    (295, 96, 1366, 1)
    (296, 96, 1366, 1)
    (297, 96, 1366, 1)
    (298, 96, 1366, 1)
    (299, 96, 1366, 1)
    (300, 96, 1366, 1)
    (301, 96, 1366, 1)
    (302, 96, 1366, 1)
    (303, 96, 1366, 1)
    (304, 96, 1366, 1)
    (305, 96, 1366, 1)
    (306, 96, 1366, 1)
    (307, 96, 1366, 1)
    (308, 96, 1366, 1)
    (309, 96, 1366, 1)
    (310, 96, 1366, 1)
    (311, 96, 1366, 1)
    (312, 96, 1366, 1)
    (313, 96, 1366, 1)
    (314, 96, 1366, 1)
    (315, 96, 1366, 1)
    (316, 96, 1366, 1)
    (317, 96, 1366, 1)
    (318, 96, 1366, 1)
    (319, 96, 1366, 1)
    (320, 96, 1366, 1)
    (321, 96, 1366, 1)
    (322, 96, 1366, 1)
    (323, 96, 1366, 1)
    (324, 96, 1366, 1)
    (325, 96, 1366, 1)
    (326, 96, 1366, 1)
    (327, 96, 1366, 1)
    (328, 96, 1366, 1)
    (329, 96, 1366, 1)
    (330, 96, 1366, 1)
    (331, 96, 1366, 1)
    (332, 96, 1366, 1)
    (333, 96, 1366, 1)
    (334, 96, 1366, 1)
    (335, 96, 1366, 1)
    (336, 96, 1366, 1)
    (337, 96, 1366, 1)
    (338, 96, 1366, 1)
    (339, 96, 1366, 1)
    (340, 96, 1366, 1)
    (341, 96, 1366, 1)
    (342, 96, 1366, 1)
    (343, 96, 1366, 1)
    (344, 96, 1366, 1)
    (345, 96, 1366, 1)
    (346, 96, 1366, 1)
    (347, 96, 1366, 1)
    (348, 96, 1366, 1)
    (349, 96, 1366, 1)
    (350, 96, 1366, 1)
    (351, 96, 1366, 1)
    (352, 96, 1366, 1)
    (353, 96, 1366, 1)
    (354, 96, 1366, 1)
    (355, 96, 1366, 1)
    (356, 96, 1366, 1)
    (357, 96, 1366, 1)
    (358, 96, 1366, 1)
    (359, 96, 1366, 1)
    (360, 96, 1366, 1)
    (361, 96, 1366, 1)
    (362, 96, 1366, 1)
    (363, 96, 1366, 1)
    (364, 96, 1366, 1)
    (365, 96, 1366, 1)
    (366, 96, 1366, 1)
    (367, 96, 1366, 1)
    (368, 96, 1366, 1)
    (369, 96, 1366, 1)
    (370, 96, 1366, 1)
    (371, 96, 1366, 1)
    (372, 96, 1366, 1)
    (373, 96, 1366, 1)
    (374, 96, 1366, 1)
    (375, 96, 1366, 1)
    (376, 96, 1366, 1)
    (377, 96, 1366, 1)
    (378, 96, 1366, 1)
    (379, 96, 1366, 1)
    (380, 96, 1366, 1)
    (381, 96, 1366, 1)
    (382, 96, 1366, 1)
    (383, 96, 1366, 1)
    (384, 96, 1366, 1)
    (385, 96, 1366, 1)
    (386, 96, 1366, 1)
    (387, 96, 1366, 1)
    (388, 96, 1366, 1)
    (389, 96, 1366, 1)
    (390, 96, 1366, 1)
    (391, 96, 1366, 1)
    (392, 96, 1366, 1)
    (393, 96, 1366, 1)
    (394, 96, 1366, 1)
    (395, 96, 1366, 1)
    (396, 96, 1366, 1)
    (397, 96, 1366, 1)
    (398, 96, 1366, 1)
    (399, 96, 1366, 1)
    (400, 96, 1366, 1)
    (401, 96, 1366, 1)
    (402, 96, 1366, 1)
    (403, 96, 1366, 1)
    (404, 96, 1366, 1)
    (405, 96, 1366, 1)
    (406, 96, 1366, 1)
    (407, 96, 1366, 1)
    (408, 96, 1366, 1)
    (409, 96, 1366, 1)
    (410, 96, 1366, 1)
    (411, 96, 1366, 1)
    (412, 96, 1366, 1)
    (413, 96, 1366, 1)
    (414, 96, 1366, 1)
    (415, 96, 1366, 1)
    (416, 96, 1366, 1)
    (417, 96, 1366, 1)
    (418, 96, 1366, 1)
    (419, 96, 1366, 1)
    (420, 96, 1366, 1)
    (421, 96, 1366, 1)
    (422, 96, 1366, 1)
    (423, 96, 1366, 1)
    (424, 96, 1366, 1)
    (425, 96, 1366, 1)
    (426, 96, 1366, 1)
    (427, 96, 1366, 1)
    (428, 96, 1366, 1)
    (429, 96, 1366, 1)
    (430, 96, 1366, 1)
    (431, 96, 1366, 1)
    (432, 96, 1366, 1)
    (433, 96, 1366, 1)
    (434, 96, 1366, 1)
    (435, 96, 1366, 1)
    (436, 96, 1366, 1)
    (437, 96, 1366, 1)
    (438, 96, 1366, 1)
    (439, 96, 1366, 1)
    (440, 96, 1366, 1)
    (441, 96, 1366, 1)
    (442, 96, 1366, 1)
    (443, 96, 1366, 1)
    (444, 96, 1366, 1)
    (445, 96, 1366, 1)
    (446, 96, 1366, 1)
    (447, 96, 1366, 1)
    (448, 96, 1366, 1)
    (449, 96, 1366, 1)
    (450, 96, 1366, 1)
    (451, 96, 1366, 1)
    (452, 96, 1366, 1)
    (453, 96, 1366, 1)
    (454, 96, 1366, 1)
    (455, 96, 1366, 1)
    (456, 96, 1366, 1)
    (457, 96, 1366, 1)
    (458, 96, 1366, 1)
    (459, 96, 1366, 1)
    (460, 96, 1366, 1)
    (461, 96, 1366, 1)
    (462, 96, 1366, 1)
    (463, 96, 1366, 1)
    (464, 96, 1366, 1)
    (465, 96, 1366, 1)
    (466, 96, 1366, 1)
    (467, 96, 1366, 1)
    (468, 96, 1366, 1)
    (469, 96, 1366, 1)
    (470, 96, 1366, 1)
    (471, 96, 1366, 1)
    (472, 96, 1366, 1)
    (473, 96, 1366, 1)
    (474, 96, 1366, 1)
    (475, 96, 1366, 1)
    (476, 96, 1366, 1)
    (477, 96, 1366, 1)
    (478, 96, 1366, 1)
    (479, 96, 1366, 1)
    (480, 96, 1366, 1)
    (481, 96, 1366, 1)
    (482, 96, 1366, 1)
    (483, 96, 1366, 1)
    (484, 96, 1366, 1)
    (485, 96, 1366, 1)
    (486, 96, 1366, 1)
    (487, 96, 1366, 1)
    (488, 96, 1366, 1)
    (489, 96, 1366, 1)
    (490, 96, 1366, 1)
    (491, 96, 1366, 1)
    (492, 96, 1366, 1)
    (493, 96, 1366, 1)
    (494, 96, 1366, 1)
    (495, 96, 1366, 1)
    (496, 96, 1366, 1)
    (497, 96, 1366, 1)
    (498, 96, 1366, 1)
    (499, 96, 1366, 1)
    (500, 96, 1366, 1)
    (501, 96, 1366, 1)
    (502, 96, 1366, 1)
    (503, 96, 1366, 1)
    (504, 96, 1366, 1)
    (505, 96, 1366, 1)
    (506, 96, 1366, 1)
    (507, 96, 1366, 1)
    (508, 96, 1366, 1)
    (509, 96, 1366, 1)
    (510, 96, 1366, 1)
    (511, 96, 1366, 1)
    (512, 96, 1366, 1)
    (513, 96, 1366, 1)
    (514, 96, 1366, 1)
    (515, 96, 1366, 1)
    (516, 96, 1366, 1)
    (517, 96, 1366, 1)
    (518, 96, 1366, 1)
    (519, 96, 1366, 1)
    (520, 96, 1366, 1)
    (521, 96, 1366, 1)
    (522, 96, 1366, 1)
    (523, 96, 1366, 1)
    (524, 96, 1366, 1)
    (525, 96, 1366, 1)
    (526, 96, 1366, 1)
    (527, 96, 1366, 1)
    (528, 96, 1366, 1)
    (529, 96, 1366, 1)
    (530, 96, 1366, 1)
    (531, 96, 1366, 1)
    (532, 96, 1366, 1)
    (533, 96, 1366, 1)
    (534, 96, 1366, 1)
    (535, 96, 1366, 1)
    (536, 96, 1366, 1)
    (537, 96, 1366, 1)
    (538, 96, 1366, 1)
    (539, 96, 1366, 1)
    (540, 96, 1366, 1)
    (541, 96, 1366, 1)
    (542, 96, 1366, 1)
    (543, 96, 1366, 1)
    (544, 96, 1366, 1)
    (545, 96, 1366, 1)
    (546, 96, 1366, 1)
    (547, 96, 1366, 1)
    (548, 96, 1366, 1)
    (549, 96, 1366, 1)
    (550, 96, 1366, 1)
    (551, 96, 1366, 1)
    (552, 96, 1366, 1)
    (553, 96, 1366, 1)
    (554, 96, 1366, 1)
    (555, 96, 1366, 1)
    (556, 96, 1366, 1)
    (557, 96, 1366, 1)
    (558, 96, 1366, 1)
    (559, 96, 1366, 1)
    (560, 96, 1366, 1)
    (561, 96, 1366, 1)
    (562, 96, 1366, 1)
    (563, 96, 1366, 1)
    (564, 96, 1366, 1)
    (565, 96, 1366, 1)
    (566, 96, 1366, 1)
    (567, 96, 1366, 1)
    (568, 96, 1366, 1)
    (569, 96, 1366, 1)
    (570, 96, 1366, 1)
    (571, 96, 1366, 1)
    (572, 96, 1366, 1)
    (573, 96, 1366, 1)
    (574, 96, 1366, 1)
    (575, 96, 1366, 1)
    (576, 96, 1366, 1)
    (577, 96, 1366, 1)
    (578, 96, 1366, 1)
    (579, 96, 1366, 1)
    (580, 96, 1366, 1)
    (581, 96, 1366, 1)
    (582, 96, 1366, 1)
    (583, 96, 1366, 1)
    (584, 96, 1366, 1)
    (585, 96, 1366, 1)
    (586, 96, 1366, 1)
    (587, 96, 1366, 1)
    (588, 96, 1366, 1)
    (589, 96, 1366, 1)
    (590, 96, 1366, 1)
    (591, 96, 1366, 1)
    (592, 96, 1366, 1)
    (593, 96, 1366, 1)
    (594, 96, 1366, 1)
    (595, 96, 1366, 1)
    (596, 96, 1366, 1)
    (597, 96, 1366, 1)
    (598, 96, 1366, 1)
    (599, 96, 1366, 1)
    (600, 96, 1366, 1)
    (601, 96, 1366, 1)
    (602, 96, 1366, 1)
    (603, 96, 1366, 1)
    (604, 96, 1366, 1)
    (605, 96, 1366, 1)
    (606, 96, 1366, 1)
    (607, 96, 1366, 1)
    (608, 96, 1366, 1)
    (609, 96, 1366, 1)
    (610, 96, 1366, 1)
    (611, 96, 1366, 1)
    (612, 96, 1366, 1)
    (613, 96, 1366, 1)
    (614, 96, 1366, 1)
    (615, 96, 1366, 1)
    (616, 96, 1366, 1)
    (617, 96, 1366, 1)
    (618, 96, 1366, 1)
    (619, 96, 1366, 1)
    (620, 96, 1366, 1)
    (621, 96, 1366, 1)
    (622, 96, 1366, 1)
    (623, 96, 1366, 1)
    (624, 96, 1366, 1)
    (625, 96, 1366, 1)
    (626, 96, 1366, 1)
    (627, 96, 1366, 1)
    (628, 96, 1366, 1)
    (629, 96, 1366, 1)
    (630, 96, 1366, 1)
    (631, 96, 1366, 1)
    (632, 96, 1366, 1)
    (633, 96, 1366, 1)
    (634, 96, 1366, 1)
    (635, 96, 1366, 1)
    (636, 96, 1366, 1)
    (637, 96, 1366, 1)
    (638, 96, 1366, 1)
    (639, 96, 1366, 1)
    (640, 96, 1366, 1)
    (641, 96, 1366, 1)
    (642, 96, 1366, 1)
    (643, 96, 1366, 1)
    (644, 96, 1366, 1)
    (645, 96, 1366, 1)
    (646, 96, 1366, 1)
    (647, 96, 1366, 1)
    (648, 96, 1366, 1)
    (649, 96, 1366, 1)
    (650, 96, 1366, 1)
    (651, 96, 1366, 1)
    (652, 96, 1366, 1)
    (653, 96, 1366, 1)
    (654, 96, 1366, 1)
    (655, 96, 1366, 1)
    (656, 96, 1366, 1)
    (657, 96, 1366, 1)
    (658, 96, 1366, 1)
    (659, 96, 1366, 1)
    (660, 96, 1366, 1)
    (661, 96, 1366, 1)
    (662, 96, 1366, 1)
    (663, 96, 1366, 1)
    (664, 96, 1366, 1)
    (665, 96, 1366, 1)
    (666, 96, 1366, 1)
    (667, 96, 1366, 1)
    (668, 96, 1366, 1)
    (669, 96, 1366, 1)
    (670, 96, 1366, 1)
    (671, 96, 1366, 1)
    (672, 96, 1366, 1)
    (673, 96, 1366, 1)
    (674, 96, 1366, 1)
    (675, 96, 1366, 1)
    (676, 96, 1366, 1)
    (677, 96, 1366, 1)
    (678, 96, 1366, 1)
    (679, 96, 1366, 1)
    (680, 96, 1366, 1)
    (681, 96, 1366, 1)
    (682, 96, 1366, 1)
    (683, 96, 1366, 1)
    (684, 96, 1366, 1)
    (685, 96, 1366, 1)
    (686, 96, 1366, 1)
    (687, 96, 1366, 1)
    (688, 96, 1366, 1)
    (689, 96, 1366, 1)
    (690, 96, 1366, 1)
    (691, 96, 1366, 1)
    (692, 96, 1366, 1)
    (693, 96, 1366, 1)
    (694, 96, 1366, 1)
    (695, 96, 1366, 1)
    (696, 96, 1366, 1)
    (697, 96, 1366, 1)
    (698, 96, 1366, 1)
    (699, 96, 1366, 1)
    (700, 96, 1366, 1)
    (701, 96, 1366, 1)
    (702, 96, 1366, 1)
    (703, 96, 1366, 1)
    (704, 96, 1366, 1)
    (705, 96, 1366, 1)
    (706, 96, 1366, 1)
    (707, 96, 1366, 1)
    (708, 96, 1366, 1)
    (709, 96, 1366, 1)
    (710, 96, 1366, 1)
    (711, 96, 1366, 1)
    (712, 96, 1366, 1)
    (713, 96, 1366, 1)
    (714, 96, 1366, 1)
    (715, 96, 1366, 1)
    (716, 96, 1366, 1)
    (717, 96, 1366, 1)
    (718, 96, 1366, 1)
    (719, 96, 1366, 1)
    (720, 96, 1366, 1)
    (721, 96, 1366, 1)
    (722, 96, 1366, 1)
    (723, 96, 1366, 1)
    (724, 96, 1366, 1)
    (725, 96, 1366, 1)
    (726, 96, 1366, 1)
    (727, 96, 1366, 1)
    (728, 96, 1366, 1)
    (729, 96, 1366, 1)
    (730, 96, 1366, 1)
    (731, 96, 1366, 1)
    (732, 96, 1366, 1)
    (733, 96, 1366, 1)
    (734, 96, 1366, 1)
    (735, 96, 1366, 1)
    (736, 96, 1366, 1)
    (737, 96, 1366, 1)
    (738, 96, 1366, 1)
    (739, 96, 1366, 1)
    (740, 96, 1366, 1)
    (741, 96, 1366, 1)
    (742, 96, 1366, 1)
    (743, 96, 1366, 1)
    (744, 96, 1366, 1)
    (745, 96, 1366, 1)
    (746, 96, 1366, 1)
    (747, 96, 1366, 1)
    (748, 96, 1366, 1)
    (749, 96, 1366, 1)
    (750, 96, 1366, 1)





```python
test_x_reshape = []#np.zeros((96,1366,1))
```


```python
for item in test_x:
    # print(item[0].reshape((96,1366,1)).shape)
    #test_x_reshape.append = item[0].reshape((96,1366,1))
    test_x_reshape.append(item[0].reshape((96,1366,1)))
    print(np.array(test_x_reshape).shape)
```

    (1, 96, 1366, 1)
    (2, 96, 1366, 1)
    (3, 96, 1366, 1)
    (4, 96, 1366, 1)
    (5, 96, 1366, 1)
    (6, 96, 1366, 1)
    (7, 96, 1366, 1)
    (8, 96, 1366, 1)
    (9, 96, 1366, 1)
    (10, 96, 1366, 1)
    (11, 96, 1366, 1)
    (12, 96, 1366, 1)
    (13, 96, 1366, 1)
    (14, 96, 1366, 1)
    (15, 96, 1366, 1)
    (16, 96, 1366, 1)
    (17, 96, 1366, 1)
    (18, 96, 1366, 1)
    (19, 96, 1366, 1)
    (20, 96, 1366, 1)
    (21, 96, 1366, 1)
    (22, 96, 1366, 1)
    (23, 96, 1366, 1)
    (24, 96, 1366, 1)
    (25, 96, 1366, 1)
    (26, 96, 1366, 1)
    (27, 96, 1366, 1)
    (28, 96, 1366, 1)
    (29, 96, 1366, 1)
    (30, 96, 1366, 1)
    (31, 96, 1366, 1)
    (32, 96, 1366, 1)
    (33, 96, 1366, 1)
    (34, 96, 1366, 1)
    (35, 96, 1366, 1)
    (36, 96, 1366, 1)
    (37, 96, 1366, 1)
    (38, 96, 1366, 1)
    (39, 96, 1366, 1)
    (40, 96, 1366, 1)
    (41, 96, 1366, 1)
    (42, 96, 1366, 1)
    (43, 96, 1366, 1)
    (44, 96, 1366, 1)
    (45, 96, 1366, 1)
    (46, 96, 1366, 1)
    (47, 96, 1366, 1)
    (48, 96, 1366, 1)
    (49, 96, 1366, 1)
    (50, 96, 1366, 1)
    (51, 96, 1366, 1)
    (52, 96, 1366, 1)
    (53, 96, 1366, 1)
    (54, 96, 1366, 1)
    (55, 96, 1366, 1)
    (56, 96, 1366, 1)
    (57, 96, 1366, 1)
    (58, 96, 1366, 1)
    (59, 96, 1366, 1)
    (60, 96, 1366, 1)
    (61, 96, 1366, 1)
    (62, 96, 1366, 1)
    (63, 96, 1366, 1)
    (64, 96, 1366, 1)
    (65, 96, 1366, 1)
    (66, 96, 1366, 1)
    (67, 96, 1366, 1)
    (68, 96, 1366, 1)
    (69, 96, 1366, 1)
    (70, 96, 1366, 1)
    (71, 96, 1366, 1)
    (72, 96, 1366, 1)
    (73, 96, 1366, 1)
    (74, 96, 1366, 1)
    (75, 96, 1366, 1)
    (76, 96, 1366, 1)
    (77, 96, 1366, 1)
    (78, 96, 1366, 1)
    (79, 96, 1366, 1)
    (80, 96, 1366, 1)
    (81, 96, 1366, 1)
    (82, 96, 1366, 1)
    (83, 96, 1366, 1)
    (84, 96, 1366, 1)
    (85, 96, 1366, 1)
    (86, 96, 1366, 1)
    (87, 96, 1366, 1)
    (88, 96, 1366, 1)
    (89, 96, 1366, 1)
    (90, 96, 1366, 1)
    (91, 96, 1366, 1)
    (92, 96, 1366, 1)
    (93, 96, 1366, 1)
    (94, 96, 1366, 1)
    (95, 96, 1366, 1)
    (96, 96, 1366, 1)
    (97, 96, 1366, 1)
    (98, 96, 1366, 1)
    (99, 96, 1366, 1)
    (100, 96, 1366, 1)



```python
valid_x_reshape = []#np.zeros((96,1366,1))
```


```python
for item in valid_x:
    # print(item[0].reshape((96,1366,1)).shape)
    #test_x_reshape.append = item[0].reshape((96,1366,1))
    valid_x_reshape.append(item[0].reshape((96,1366,1)))
    print(np.array(valid_x_reshape).shape)
```

    (1, 96, 1366, 1)
    (2, 96, 1366, 1)
    (3, 96, 1366, 1)
    (4, 96, 1366, 1)
    (5, 96, 1366, 1)
    (6, 96, 1366, 1)
    (7, 96, 1366, 1)
    (8, 96, 1366, 1)
    (9, 96, 1366, 1)
    (10, 96, 1366, 1)
    (11, 96, 1366, 1)
    (12, 96, 1366, 1)
    (13, 96, 1366, 1)
    (14, 96, 1366, 1)
    (15, 96, 1366, 1)
    (16, 96, 1366, 1)
    (17, 96, 1366, 1)
    (18, 96, 1366, 1)
    (19, 96, 1366, 1)
    (20, 96, 1366, 1)
    (21, 96, 1366, 1)
    (22, 96, 1366, 1)
    (23, 96, 1366, 1)
    (24, 96, 1366, 1)
    (25, 96, 1366, 1)
    (26, 96, 1366, 1)
    (27, 96, 1366, 1)
    (28, 96, 1366, 1)
    (29, 96, 1366, 1)
    (30, 96, 1366, 1)
    (31, 96, 1366, 1)
    (32, 96, 1366, 1)
    (33, 96, 1366, 1)
    (34, 96, 1366, 1)
    (35, 96, 1366, 1)
    (36, 96, 1366, 1)
    (37, 96, 1366, 1)
    (38, 96, 1366, 1)
    (39, 96, 1366, 1)
    (40, 96, 1366, 1)
    (41, 96, 1366, 1)
    (42, 96, 1366, 1)
    (43, 96, 1366, 1)
    (44, 96, 1366, 1)
    (45, 96, 1366, 1)
    (46, 96, 1366, 1)
    (47, 96, 1366, 1)
    (48, 96, 1366, 1)
    (49, 96, 1366, 1)
    (50, 96, 1366, 1)
    (51, 96, 1366, 1)
    (52, 96, 1366, 1)
    (53, 96, 1366, 1)
    (54, 96, 1366, 1)
    (55, 96, 1366, 1)
    (56, 96, 1366, 1)
    (57, 96, 1366, 1)
    (58, 96, 1366, 1)
    (59, 96, 1366, 1)
    (60, 96, 1366, 1)
    (61, 96, 1366, 1)
    (62, 96, 1366, 1)
    (63, 96, 1366, 1)
    (64, 96, 1366, 1)
    (65, 96, 1366, 1)
    (66, 96, 1366, 1)
    (67, 96, 1366, 1)
    (68, 96, 1366, 1)
    (69, 96, 1366, 1)
    (70, 96, 1366, 1)
    (71, 96, 1366, 1)
    (72, 96, 1366, 1)
    (73, 96, 1366, 1)
    (74, 96, 1366, 1)
    (75, 96, 1366, 1)
    (76, 96, 1366, 1)
    (77, 96, 1366, 1)
    (78, 96, 1366, 1)
    (79, 96, 1366, 1)
    (80, 96, 1366, 1)
    (81, 96, 1366, 1)
    (82, 96, 1366, 1)
    (83, 96, 1366, 1)
    (84, 96, 1366, 1)
    (85, 96, 1366, 1)
    (86, 96, 1366, 1)
    (87, 96, 1366, 1)
    (88, 96, 1366, 1)
    (89, 96, 1366, 1)
    (90, 96, 1366, 1)
    (91, 96, 1366, 1)
    (92, 96, 1366, 1)
    (93, 96, 1366, 1)
    (94, 96, 1366, 1)
    (95, 96, 1366, 1)
    (96, 96, 1366, 1)
    (97, 96, 1366, 1)
    (98, 96, 1366, 1)
    (99, 96, 1366, 1)
    (100, 96, 1366, 1)
    (101, 96, 1366, 1)
    (102, 96, 1366, 1)
    (103, 96, 1366, 1)
    (104, 96, 1366, 1)
    (105, 96, 1366, 1)
    (106, 96, 1366, 1)
    (107, 96, 1366, 1)
    (108, 96, 1366, 1)
    (109, 96, 1366, 1)
    (110, 96, 1366, 1)
    (111, 96, 1366, 1)
    (112, 96, 1366, 1)
    (113, 96, 1366, 1)
    (114, 96, 1366, 1)
    (115, 96, 1366, 1)
    (116, 96, 1366, 1)
    (117, 96, 1366, 1)
    (118, 96, 1366, 1)
    (119, 96, 1366, 1)
    (120, 96, 1366, 1)
    (121, 96, 1366, 1)
    (122, 96, 1366, 1)
    (123, 96, 1366, 1)
    (124, 96, 1366, 1)
    (125, 96, 1366, 1)
    (126, 96, 1366, 1)
    (127, 96, 1366, 1)
    (128, 96, 1366, 1)
    (129, 96, 1366, 1)
    (130, 96, 1366, 1)
    (131, 96, 1366, 1)
    (132, 96, 1366, 1)
    (133, 96, 1366, 1)
    (134, 96, 1366, 1)
    (135, 96, 1366, 1)
    (136, 96, 1366, 1)
    (137, 96, 1366, 1)
    (138, 96, 1366, 1)
    (139, 96, 1366, 1)
    (140, 96, 1366, 1)
    (141, 96, 1366, 1)
    (142, 96, 1366, 1)
    (143, 96, 1366, 1)
    (144, 96, 1366, 1)
    (145, 96, 1366, 1)
    (146, 96, 1366, 1)
    (147, 96, 1366, 1)
    (148, 96, 1366, 1)
    (149, 96, 1366, 1)
    (150, 96, 1366, 1)


for item in test_x:
    # print(item[0].reshape((96,1366,1)).shape)
    # test_x_reshape.append = item[0].reshape((96,1366,1))
    test_x_reshape.append(item)
    print(np.array(test_x_reshape).shape)


```python

```


```python
test_x[5][0]
```




    array([[-1.63497219e+01, -1.64027100e+01, -1.66768284e+01, ...,
            -1.58324146e+01, -1.55317421e+01, -1.42583351e+01],
           [ 3.05063874e-02, -1.83508635e+00, -2.44407034e+00, ...,
            -3.48070574e+00, -1.90951288e+00, -7.99118996e+00],
           [ 9.15635765e-01, -4.04687881e+00, -5.39213085e+00, ...,
            -7.02798319e+00, -6.20999384e+00, -7.13931274e+00],
           ...,
           [-5.70205765e+01, -5.70205765e+01, -5.70205765e+01, ...,
            -5.70205765e+01, -5.70205765e+01, -5.70205765e+01],
           [-5.70205765e+01, -5.70205765e+01, -5.70205765e+01, ...,
            -5.70205765e+01, -5.70205765e+01, -5.70205765e+01],
           [-5.70205765e+01, -5.70205765e+01, -5.70205765e+01, ...,
            -5.70205765e+01, -5.70205765e+01, -5.70205765e+01]])




```python
np.array([[[1,5,6]]]).shape
```




    (1, 1, 3)




```python
test_x[5]
```




    array([[[-1.63497219e+01, -1.64027100e+01, -1.66768284e+01, ...,
             -1.58324146e+01, -1.55317421e+01, -1.42583351e+01],
            [ 3.05063874e-02, -1.83508635e+00, -2.44407034e+00, ...,
             -3.48070574e+00, -1.90951288e+00, -7.99118996e+00],
            [ 9.15635765e-01, -4.04687881e+00, -5.39213085e+00, ...,
             -7.02798319e+00, -6.20999384e+00, -7.13931274e+00],
            ...,
            [-5.70205765e+01, -5.70205765e+01, -5.70205765e+01, ...,
             -5.70205765e+01, -5.70205765e+01, -5.70205765e+01],
            [-5.70205765e+01, -5.70205765e+01, -5.70205765e+01, ...,
             -5.70205765e+01, -5.70205765e+01, -5.70205765e+01],
            [-5.70205765e+01, -5.70205765e+01, -5.70205765e+01, ...,
             -5.70205765e+01, -5.70205765e+01, -5.70205765e+01]]])



### Training the model

model = load_model('Crnn.h5')


```python
#train the model
model.fit(np.array(train_x_reshape), train_y, validation_data=(np.array(valid_x_reshape), valid_y), epochs=100 )
```

    WARNING:tensorflow:From /home/user/Music/gtzan4work/genres/mseCnn/lib/python3.7/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /home/user/Music/gtzan4work/genres/mseCnn/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    Train on 750 samples, validate on 150 samples
    Epoch 1/100
    750/750 [==============================] - 282s 375ms/step - loss: 2.3115 - accuracy: 0.1960 - val_loss: 9.2082 - val_accuracy: 0.0800
    Epoch 2/100
    750/750 [==============================] - 298s 397ms/step - loss: 2.1299 - accuracy: 0.2147 - val_loss: 3.6627 - val_accuracy: 0.0800
    Epoch 3/100
    750/750 [==============================] - 308s 411ms/step - loss: 2.0677 - accuracy: 0.2187 - val_loss: 3.5932 - val_accuracy: 0.1000
    Epoch 4/100
    750/750 [==============================] - 294s 392ms/step - loss: 2.0245 - accuracy: 0.2120 - val_loss: 3.5569 - val_accuracy: 0.1733
    Epoch 5/100
    750/750 [==============================] - 294s 393ms/step - loss: 1.9964 - accuracy: 0.1707 - val_loss: 2.8048 - val_accuracy: 0.1800
    Epoch 6/100
    750/750 [==============================] - 301s 401ms/step - loss: 2.0097 - accuracy: 0.1773 - val_loss: 4.1016 - val_accuracy: 0.1867
    Epoch 7/100
    750/750 [==============================] - 305s 406ms/step - loss: 1.9431 - accuracy: 0.1587 - val_loss: 3.3720 - val_accuracy: 0.1200
    Epoch 8/100
    750/750 [==============================] - 312s 416ms/step - loss: 1.8636 - accuracy: 0.1627 - val_loss: 2.4660 - val_accuracy: 0.1133
    Epoch 9/100
    750/750 [==============================] - 319s 425ms/step - loss: 1.8828 - accuracy: 0.1533 - val_loss: 2.5962 - val_accuracy: 0.1333
    Epoch 10/100
    750/750 [==============================] - 315s 420ms/step - loss: 1.8313 - accuracy: 0.1733 - val_loss: 2.6453 - val_accuracy: 0.1667
    Epoch 11/100
    750/750 [==============================] - 372s 496ms/step - loss: 1.7935 - accuracy: 0.1333 - val_loss: 2.9876 - val_accuracy: 0.1333
    Epoch 12/100
    750/750 [==============================] - 320s 427ms/step - loss: 1.7181 - accuracy: 0.1173 - val_loss: 3.1150 - val_accuracy: 0.1267
    Epoch 13/100
    750/750 [==============================] - 308s 411ms/step - loss: 1.7507 - accuracy: 0.1587 - val_loss: 2.4253 - val_accuracy: 0.1800
    Epoch 14/100
    750/750 [==============================] - 305s 407ms/step - loss: 1.7082 - accuracy: 0.1227 - val_loss: 3.1389 - val_accuracy: 0.1800
    Epoch 15/100
    750/750 [==============================] - 322s 430ms/step - loss: 1.7624 - accuracy: 0.1200 - val_loss: 2.8237 - val_accuracy: 0.2400
    Epoch 16/100
    750/750 [==============================] - 311s 415ms/step - loss: 1.7906 - accuracy: 0.1973 - val_loss: 3.8231 - val_accuracy: 0.1933
    Epoch 17/100
    750/750 [==============================] - 313s 417ms/step - loss: 1.6462 - accuracy: 0.1560 - val_loss: 2.5201 - val_accuracy: 0.1667
    Epoch 18/100
    750/750 [==============================] - 325s 433ms/step - loss: 1.5713 - accuracy: 0.1333 - val_loss: 3.7378 - val_accuracy: 0.2067
    Epoch 19/100
    750/750 [==============================] - 325s 433ms/step - loss: 1.5482 - accuracy: 0.1213 - val_loss: 1.9658 - val_accuracy: 0.1733
    Epoch 20/100
    750/750 [==============================] - 348s 464ms/step - loss: 1.6360 - accuracy: 0.1173 - val_loss: 2.4038 - val_accuracy: 0.1733
    Epoch 21/100
    750/750 [==============================] - 359s 479ms/step - loss: 1.6367 - accuracy: 0.1947 - val_loss: 2.8690 - val_accuracy: 0.2000
    Epoch 22/100
    750/750 [==============================] - 337s 450ms/step - loss: 1.6031 - accuracy: 0.1813 - val_loss: 3.0632 - val_accuracy: 0.2067
    Epoch 23/100
    750/750 [==============================] - 318s 423ms/step - loss: 1.5840 - accuracy: 0.1373 - val_loss: 2.5802 - val_accuracy: 0.2333
    Epoch 24/100
    750/750 [==============================] - 332s 443ms/step - loss: 1.5988 - accuracy: 0.2027 - val_loss: 3.2365 - val_accuracy: 0.2200
    Epoch 25/100
    750/750 [==============================] - 327s 437ms/step - loss: 1.4398 - accuracy: 0.1307 - val_loss: 1.9234 - val_accuracy: 0.1333
    Epoch 26/100
    750/750 [==============================] - 337s 449ms/step - loss: 1.4805 - accuracy: 0.1347 - val_loss: 2.5292 - val_accuracy: 0.2200
    Epoch 27/100
    750/750 [==============================] - 340s 454ms/step - loss: 1.5225 - accuracy: 0.1800 - val_loss: 1.7922 - val_accuracy: 0.1800
    Epoch 28/100
    750/750 [==============================] - 344s 459ms/step - loss: 1.4176 - accuracy: 0.1280 - val_loss: 1.9908 - val_accuracy: 0.1600
    Epoch 29/100
    750/750 [==============================] - 319s 425ms/step - loss: 1.3678 - accuracy: 0.1253 - val_loss: 1.8823 - val_accuracy: 0.1933
    Epoch 30/100
    750/750 [==============================] - 320s 427ms/step - loss: 1.5428 - accuracy: 0.1667 - val_loss: 2.8622 - val_accuracy: 0.1600
    Epoch 31/100
    750/750 [==============================] - 328s 437ms/step - loss: 1.4581 - accuracy: 0.1387 - val_loss: 2.1270 - val_accuracy: 0.2000
    Epoch 32/100
    750/750 [==============================] - 325s 433ms/step - loss: 1.3234 - accuracy: 0.1293 - val_loss: 1.7256 - val_accuracy: 0.1333
    Epoch 33/100
    750/750 [==============================] - 312s 417ms/step - loss: 1.2832 - accuracy: 0.1147 - val_loss: 2.4242 - val_accuracy: 0.1800
    Epoch 34/100
    750/750 [==============================] - 304s 405ms/step - loss: 1.2390 - accuracy: 0.1107 - val_loss: 1.5906 - val_accuracy: 0.1533
    Epoch 35/100
    750/750 [==============================] - 299s 399ms/step - loss: 1.2590 - accuracy: 0.1347 - val_loss: 2.3794 - val_accuracy: 0.1467
    Epoch 36/100
    750/750 [==============================] - 309s 411ms/step - loss: 1.2170 - accuracy: 0.1173 - val_loss: 1.7157 - val_accuracy: 0.1533
    Epoch 37/100
    750/750 [==============================] - 318s 424ms/step - loss: 1.2373 - accuracy: 0.1107 - val_loss: 1.9799 - val_accuracy: 0.1400
    Epoch 38/100
    750/750 [==============================] - 319s 426ms/step - loss: 1.1974 - accuracy: 0.1133 - val_loss: 1.7027 - val_accuracy: 0.1200
    Epoch 39/100
    750/750 [==============================] - 316s 421ms/step - loss: 1.1960 - accuracy: 0.1187 - val_loss: 1.7483 - val_accuracy: 0.1200
    Epoch 40/100
    750/750 [==============================] - 327s 436ms/step - loss: 1.3219 - accuracy: 0.1293 - val_loss: 1.7696 - val_accuracy: 0.2133
    Epoch 41/100
    750/750 [==============================] - 329s 438ms/step - loss: 1.4472 - accuracy: 0.2000 - val_loss: 2.8583 - val_accuracy: 0.1800
    Epoch 42/100
    750/750 [==============================] - 316s 422ms/step - loss: 1.3296 - accuracy: 0.1773 - val_loss: 2.3237 - val_accuracy: 0.1600
    Epoch 43/100
    750/750 [==============================] - 334s 445ms/step - loss: 1.2470 - accuracy: 0.1227 - val_loss: 2.9756 - val_accuracy: 0.1133
    Epoch 44/100
    750/750 [==============================] - 326s 434ms/step - loss: 1.2520 - accuracy: 0.1120 - val_loss: 2.0796 - val_accuracy: 0.1467
    Epoch 45/100
    750/750 [==============================] - 333s 444ms/step - loss: 1.2357 - accuracy: 0.1147 - val_loss: 2.0235 - val_accuracy: 0.1133
    Epoch 46/100
    750/750 [==============================] - 325s 433ms/step - loss: 1.2163 - accuracy: 0.1307 - val_loss: 1.8961 - val_accuracy: 0.1467
    Epoch 47/100
    750/750 [==============================] - 327s 437ms/step - loss: 1.1926 - accuracy: 0.1933 - val_loss: 2.3773 - val_accuracy: 0.2000
    Epoch 48/100
    750/750 [==============================] - 317s 422ms/step - loss: 1.2735 - accuracy: 0.1613 - val_loss: 1.7586 - val_accuracy: 0.1667
    Epoch 49/100
    750/750 [==============================] - 323s 431ms/step - loss: 1.1337 - accuracy: 0.1560 - val_loss: 3.5485 - val_accuracy: 0.1933
    Epoch 50/100
    750/750 [==============================] - 320s 426ms/step - loss: 1.0992 - accuracy: 0.1293 - val_loss: 1.9553 - val_accuracy: 0.1333
    Epoch 51/100
    750/750 [==============================] - 314s 419ms/step - loss: 1.1272 - accuracy: 0.1293 - val_loss: 1.6633 - val_accuracy: 0.1533
    Epoch 52/100
    750/750 [==============================] - 328s 438ms/step - loss: 1.1113 - accuracy: 0.1187 - val_loss: 1.8246 - val_accuracy: 0.1333
    Epoch 53/100
    750/750 [==============================] - 349s 466ms/step - loss: 1.1102 - accuracy: 0.1160 - val_loss: 2.3149 - val_accuracy: 0.1600
    Epoch 54/100
    750/750 [==============================] - 336s 448ms/step - loss: 1.1112 - accuracy: 0.1267 - val_loss: 2.0792 - val_accuracy: 0.1133
    Epoch 55/100
    750/750 [==============================] - 316s 422ms/step - loss: 1.1105 - accuracy: 0.1373 - val_loss: 2.2804 - val_accuracy: 0.1400
    Epoch 56/100
    750/750 [==============================] - 327s 436ms/step - loss: 1.0994 - accuracy: 0.1400 - val_loss: 1.7342 - val_accuracy: 0.1267
    Epoch 57/100
    750/750 [==============================] - 323s 431ms/step - loss: 1.1058 - accuracy: 0.1267 - val_loss: 1.8470 - val_accuracy: 0.1267
    Epoch 58/100
    750/750 [==============================] - 320s 427ms/step - loss: 1.1344 - accuracy: 0.1707 - val_loss: 1.9257 - val_accuracy: 0.1800
    Epoch 59/100
    750/750 [==============================] - 311s 415ms/step - loss: 1.0839 - accuracy: 0.1733 - val_loss: 1.7702 - val_accuracy: 0.1800
    Epoch 60/100
    750/750 [==============================] - 315s 420ms/step - loss: 1.0813 - accuracy: 0.1373 - val_loss: 1.8241 - val_accuracy: 0.1867
    Epoch 61/100
    750/750 [==============================] - 321s 428ms/step - loss: 1.0936 - accuracy: 0.1240 - val_loss: 2.6501 - val_accuracy: 0.1200
    Epoch 62/100
    750/750 [==============================] - 316s 422ms/step - loss: 1.0663 - accuracy: 0.1627 - val_loss: 2.3667 - val_accuracy: 0.1200
    Epoch 63/100
    750/750 [==============================] - 305s 407ms/step - loss: 1.0631 - accuracy: 0.1507 - val_loss: 1.8791 - val_accuracy: 0.1933
    Epoch 64/100
    750/750 [==============================] - 331s 441ms/step - loss: 1.0573 - accuracy: 0.1533 - val_loss: 2.3575 - val_accuracy: 0.1533
    Epoch 65/100
    750/750 [==============================] - 316s 421ms/step - loss: 1.0432 - accuracy: 0.1293 - val_loss: 1.9633 - val_accuracy: 0.1200
    Epoch 66/100
    750/750 [==============================] - 333s 444ms/step - loss: 1.0746 - accuracy: 0.1320 - val_loss: 1.7470 - val_accuracy: 0.1200
    Epoch 67/100
    750/750 [==============================] - 328s 438ms/step - loss: 1.0520 - accuracy: 0.1373 - val_loss: 2.3776 - val_accuracy: 0.1467
    Epoch 68/100
    750/750 [==============================] - 356s 474ms/step - loss: 1.1264 - accuracy: 0.1333 - val_loss: 2.1358 - val_accuracy: 0.1600
    Epoch 69/100
    750/750 [==============================] - 354s 472ms/step - loss: 1.1693 - accuracy: 0.1213 - val_loss: 3.3582 - val_accuracy: 0.1800
    Epoch 70/100
    750/750 [==============================] - 318s 425ms/step - loss: 1.1029 - accuracy: 0.1480 - val_loss: 1.6751 - val_accuracy: 0.1800
    Epoch 71/100
    750/750 [==============================] - 316s 421ms/step - loss: 1.0863 - accuracy: 0.1200 - val_loss: 2.3954 - val_accuracy: 0.1267
    Epoch 72/100
    750/750 [==============================] - 316s 422ms/step - loss: 1.0690 - accuracy: 0.1093 - val_loss: 2.5777 - val_accuracy: 0.1200
    Epoch 73/100
    750/750 [==============================] - 309s 412ms/step - loss: 1.1049 - accuracy: 0.1147 - val_loss: 2.1593 - val_accuracy: 0.1133
    Epoch 74/100
    750/750 [==============================] - 328s 437ms/step - loss: 1.0889 - accuracy: 0.1200 - val_loss: 2.9118 - val_accuracy: 0.1200
    Epoch 75/100
    750/750 [==============================] - 323s 431ms/step - loss: 1.0814 - accuracy: 0.1160 - val_loss: 3.1064 - val_accuracy: 0.1333
    Epoch 76/100
    750/750 [==============================] - 322s 429ms/step - loss: 1.0883 - accuracy: 0.1427 - val_loss: 2.2116 - val_accuracy: 0.1733
    Epoch 77/100
    750/750 [==============================] - 309s 412ms/step - loss: 1.0941 - accuracy: 0.1587 - val_loss: 2.3745 - val_accuracy: 0.1733
    Epoch 78/100
    750/750 [==============================] - 321s 428ms/step - loss: 1.0546 - accuracy: 0.1347 - val_loss: 2.3044 - val_accuracy: 0.1200
    Epoch 79/100
    750/750 [==============================] - 316s 422ms/step - loss: 1.0402 - accuracy: 0.1413 - val_loss: 1.9338 - val_accuracy: 0.1467
    Epoch 80/100
    750/750 [==============================] - 316s 421ms/step - loss: 1.0770 - accuracy: 0.1347 - val_loss: 2.1178 - val_accuracy: 0.1267
    Epoch 81/100
    750/750 [==============================] - 343s 457ms/step - loss: 1.0583 - accuracy: 0.1053 - val_loss: 2.0272 - val_accuracy: 0.1400
    Epoch 82/100
    750/750 [==============================] - 314s 418ms/step - loss: 1.0451 - accuracy: 0.1040 - val_loss: 2.0915 - val_accuracy: 0.1133
    Epoch 83/100
    750/750 [==============================] - 315s 419ms/step - loss: 1.0896 - accuracy: 0.1520 - val_loss: 2.1141 - val_accuracy: 0.1800
    Epoch 84/100
    750/750 [==============================] - 338s 451ms/step - loss: 1.0797 - accuracy: 0.1680 - val_loss: 2.0943 - val_accuracy: 0.1600
    Epoch 85/100
    750/750 [==============================] - 335s 447ms/step - loss: 1.0463 - accuracy: 0.1467 - val_loss: 2.1054 - val_accuracy: 0.1533
    Epoch 86/100
    750/750 [==============================] - 369s 491ms/step - loss: 1.0375 - accuracy: 0.1307 - val_loss: 2.1642 - val_accuracy: 0.1267
    Epoch 87/100
    750/750 [==============================] - 361s 481ms/step - loss: 1.0274 - accuracy: 0.1187 - val_loss: 1.8601 - val_accuracy: 0.1333
    Epoch 88/100
    750/750 [==============================] - 347s 462ms/step - loss: 1.0305 - accuracy: 0.1280 - val_loss: 2.1767 - val_accuracy: 0.1133
    Epoch 89/100
    750/750 [==============================] - 392s 523ms/step - loss: 1.0258 - accuracy: 0.1293 - val_loss: 1.9682 - val_accuracy: 0.1400
    Epoch 90/100
    750/750 [==============================] - 433s 577ms/step - loss: 1.0254 - accuracy: 0.1347 - val_loss: 2.2558 - val_accuracy: 0.1133
    Epoch 91/100
    750/750 [==============================] - 412s 549ms/step - loss: 1.0289 - accuracy: 0.1533 - val_loss: 2.0911 - val_accuracy: 0.1800
    Epoch 92/100
    750/750 [==============================] - 411s 549ms/step - loss: 1.0299 - accuracy: 0.1547 - val_loss: 2.0881 - val_accuracy: 0.1533
    Epoch 93/100
    750/750 [==============================] - 392s 523ms/step - loss: 1.0242 - accuracy: 0.1520 - val_loss: 2.1704 - val_accuracy: 0.1200
    Epoch 94/100
    750/750 [==============================] - 373s 498ms/step - loss: 1.0238 - accuracy: 0.1253 - val_loss: 2.1075 - val_accuracy: 0.1200
    Epoch 95/100
    750/750 [==============================] - 355s 474ms/step - loss: 1.0232 - accuracy: 0.1440 - val_loss: 2.1664 - val_accuracy: 0.1400
    Epoch 96/100
    750/750 [==============================] - 380s 507ms/step - loss: 1.0229 - accuracy: 0.1427 - val_loss: 2.1584 - val_accuracy: 0.1267
    Epoch 97/100
    750/750 [==============================] - 368s 491ms/step - loss: 1.0225 - accuracy: 0.1280 - val_loss: 2.1043 - val_accuracy: 0.1133
    Epoch 98/100
    750/750 [==============================] - 344s 458ms/step - loss: 1.0225 - accuracy: 0.1293 - val_loss: 2.1125 - val_accuracy: 0.1333
    Epoch 99/100
    750/750 [==============================] - 315s 420ms/step - loss: 1.0224 - accuracy: 0.1360 - val_loss: 2.0872 - val_accuracy: 0.1333
    Epoch 100/100
    750/750 [==============================] - 317s 423ms/step - loss: 1.0235 - accuracy: 0.1533 - val_loss: 2.1335 - val_accuracy: 0.1267





    <keras.callbacks.callbacks.History at 0x7f4fb041ee10>




```python
from keras.models import load_model

model.save('MsE_CNN_epoch100.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('k2c2.h5')
```

# Predicting


```python
from keras.models import load_model
model = load_model('MsE_CNN_epoch100.h5')
```

    WARNING:tensorflow:From /home/user/Music/gtzan4work/genres/mseCnn/lib/python3.7/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /home/user/Music/gtzan4work/genres/mseCnn/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    



```python
output = model.predict(np.array(test_x_reshape))
```


```python
output_class = output.argmax(axis=-1)
```


```python
print(output[0])
```

    [9.9805987e-01 0.0000000e+00 1.0000000e+00 2.9206276e-05 2.3841858e-07
     9.2387199e-07 5.7667494e-05 0.0000000e+00 1.0000000e+00 1.1920929e-07]



```python
print(test_y[0])
```

    [0 0 0 0 0 0 0 0 0 1]



```python
print(10000*output[0])
sum = 0
for i in output[0]:
    sum += i
print(sum)
```

    [9.9805986e+03 0.0000000e+00 1.0000000e+04 2.9206276e-01 2.3841858e-03
     9.2387199e-03 5.7667494e-01 0.0000000e+00 1.0000000e+04 1.1920929e-03]
    2.998148024082184



```python
test_y_class = test_y.argmax(axis=-1)
```


```python
print(output_class)
```

    [2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]



```python
print(test_y_class)
```

    [9 0 4 3 7 9 9 1 2 4 7 9 1 9 4 8 0 3 3 7 1 9 6 1 8 6 6 8 2 8 1 7 4 7 4 4 7
     5 4 1 2 6 8 5 1 5 3 3 5 1 5 9 7 5 8 9 7 4 8 6 9 1 1 4 8 7 2 8 9 6 6 0 3 3
     8 8 4 3 2 5 4 6 7 3 1 5 7 8 7 0 2 8 7 2 9 9 9 0 7 2]



```python
print(test_y[-1])
```

    [0 0 1 0 0 0 0 0 0 0]



```python
!pip install scikit-learn
```

    Collecting scikit-learn
      Using cached scikit_learn-0.23.1-cp37-cp37m-manylinux1_x86_64.whl (6.8 MB)
    Requirement already satisfied: numpy>=1.13.3 in ./mseCnn/lib/python3.7/site-packages (from scikit-learn) (1.18.5)
    Requirement already satisfied: scipy>=0.19.1 in ./mseCnn/lib/python3.7/site-packages (from scikit-learn) (1.4.1)
    Collecting joblib>=0.11
      Downloading joblib-0.16.0-py3-none-any.whl (300 kB)
    [K     |████████████████████████████████| 300 kB 35 kB/s eta 0:00:01
    [?25hCollecting threadpoolctl>=2.0.0
      Downloading threadpoolctl-2.1.0-py3-none-any.whl (12 kB)
    Installing collected packages: joblib, threadpoolctl, scikit-learn
    Successfully installed joblib-0.16.0 scikit-learn-0.23.1 threadpoolctl-2.1.0
    [33mWARNING: You are using pip version 20.1.1; however, version 20.2 is available.
    You should consider upgrading via the '/home/user/Music/gtzan4work/genres/mseCnn/bin/python -m pip install --upgrade pip' command.[0m



```python
from sklearn import preprocessing
output_class_bin = preprocessing.label_binarize(output_class, classes=[0, 1, 2, 3,4,5,6,7,8,9])
```


```python
print(output_class_bin)
```

    [[0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0]]


# auc-roc


```python
model = load_model('MsE_CNN_epoch100.h5')
```

# Computing AUROC and ROC curve values


```python
from sklearn.metrics import roc_curve, roc_auc_score
```

### **Calculate AUROC**
**ROC** is the receiver operating characteristic
**AUROC** is the area under the ROC curve


```python
msecnn_auc = roc_auc_score(test_y, output)
```


```python
msecnn_auc_bin = roc_auc_score(test_y, output_class_bin)
```

### Print AUROC scores


```python
print('MSECNN: AUROC = %.3f' % (msecnn_auc))
```

    MSECNN: AUROC = 0.866



```python
print('MSECNN: AUROC = %.3f' % (msecnn_auc_bin))
```

    MSECNN: AUROC = 0.515


### Calculate ROC curve


```python
print(test_y.shape)
```

    (100, 10)



```python
print(output.shape)
```

    (100, 10)



```python
print(output_class.shape)
```

    (100,)



```python
print( len(np.squeeze(output)))
```

    100



```python
msecnn_fpr, msecnn_tpr, _ = roc_curve(test_y[0], output[0])
```


```python
print(msecnn_fpr,msecnn_tpr)
```

    [0.         0.22222222 0.77777778 0.77777778 1.        ] [0. 0. 0. 1. 1.]



```python
msecnn_fpr = []
msecnn_tpr = []
for i in range(10):
    i_fpr, i_tpr, _ = roc_curve(test_y[:, i], output[:, i])
    msecnn_fpr.append(i_fpr)
    msecnn_tpr.append(i_tpr)
```


```python
from sklearn.metrics import roc_curve, auc
```


```python
n_classes = 10
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], output[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
```

### Plot the ROC curve


```python
!python -m pip install -U matplotlib
```

    Collecting matplotlib
      Downloading matplotlib-3.3.0-1-cp37-cp37m-manylinux1_x86_64.whl (11.5 MB)
    [K     |████████████████████████████████| 11.5 MB 71 kB/s eta 0:00:011
    [?25hCollecting cycler>=0.10
      Using cached cycler-0.10.0-py2.py3-none-any.whl (6.5 kB)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.1 in ./mseCnn/lib/python3.7/site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied, skipping upgrade: numpy>=1.15 in ./mseCnn/lib/python3.7/site-packages (from matplotlib) (1.18.5)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in ./mseCnn/lib/python3.7/site-packages (from matplotlib) (2.4.7)
    Collecting kiwisolver>=1.0.1
      Downloading kiwisolver-1.2.0-cp37-cp37m-manylinux1_x86_64.whl (88 kB)
    [K     |████████████████████████████████| 88 kB 89 kB/s eta 0:00:01
    [?25hCollecting pillow>=6.2.0
      Downloading Pillow-7.2.0-cp37-cp37m-manylinux1_x86_64.whl (2.2 MB)
    [K     |████████████████████████████████| 2.2 MB 193 kB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: six in ./mseCnn/lib/python3.7/site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    Installing collected packages: cycler, kiwisolver, pillow, matplotlib
    Successfully installed cycler-0.10.0 kiwisolver-1.2.0 matplotlib-3.3.0 pillow-7.2.0
    [33mWARNING: You are using pip version 20.1.1; however, version 20.2 is available.
    You should consider upgrading via the '/home/user/Music/gtzan4work/genres/mseCnn/bin/python -m pip install --upgrade pip' command.[0m



```python
import matplotlib.pyplot as plt
```

    Matplotlib is building the font cache; this may take a moment.

plt.plot(np.array(k2c2_fpr), np.array(k2c2_tpr), marker='.', label='K2C2 (AUROC = %0.3f)' % k2c2_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()

```python
# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
```


![png](output_86_0.png)



![png](output_86_1.png)



![png](output_86_2.png)



![png](output_86_3.png)



![png](output_86_4.png)



![png](output_86_5.png)



![png](output_86_6.png)



![png](output_86_7.png)



![png](output_86_8.png)



![png](output_86_9.png)



```python

```
