---
title: CNN learning to map pictures into mass of galaxy clusters
layout: post
post-image: "https://jizhaoqin.github.io/assets/images/post/CNN-loss-function.png"
description: major code to perform a regression CNN learning 
tags:
- AI
- Python
- CNN
- regression
---

# CNN learning to map pictures into mass of galaxy clusters

## 1. Load the files and preprocess data for ML

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
import time
```

```python
# define the function for data loading
def get_data(image_p, catalog_f):
    '''
    Given the image and catalogue path, this function returns the
    images and labels needed to start directly with model training.
    train_labels: Can be used to filter a particular mass range
    Returns:
    images: [index, energy-band-image]
    labels_z: [log10(mass), redshift, index]
    '''
    is_efedssim=True
    is_efedsobs=False
    images = pd.read_pickle(image_p)
    key_mass = 'm500_wl_final' if is_efedsobs else 'M500c' if is_efedssim else 'HALO_M500c'
    key_redshift = 'z_final' if is_efedsobs else 'z' if is_efedssim else 'redshift_R'
    catalog_df = pd.read_feather(catalog_f)
    labels = np.log10(catalog_df[key_mass])
    redshifts = catalog_df[key_redshift].values
    indices = catalog_df.index.values
    labels_z = np.transpose([labels, redshifts, indices])
    return images, labels_z

# define the method for generating training, validation and test set
def split_data(images, labels, rand_seed=40):
    '''
    Generate training, validation and test data set for the following learning process. Each set contains the energy-band-image, the redshift of the cluster, and its mass. The splitting factor is 0.8, 0.1, 0.1.

    Args:
        images (dict):
            keys()=['clusters', 'gsm_3dImgs'], which are index and energy-band-image.
        labels (np.ndarray):
            [log10(mass), redshift, index] corresponding to each energy-band-image.
        rand_seed (int, optional):
            Shuffle parameter for train_test_split() method. Defaults to 40.

    Returns:
        splitted_data (np.ndarray):
            *_images.shape()=(..., 50, 50, 10); *_labels.shape()=(..., 2), [log10(mass), redshift]
    '''
    train_images, test_images, train_labels, test_labels = train_test_split(
        np.array(images['gsm_3dImgs']), 
        labels[:, :-1], 
        test_size=0.1, 
        random_state=rand_seed
    )
    train_images, valid_images, train_labels, valid_labels = train_test_split(
        train_images, 
        train_labels, 
        test_size=0.1, 
        random_state=rand_seed
    )
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels
```

```python
data_directory = '../data/'
image_file = 'eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle'
catalog_file = 'eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f'

# load data
images, labels = get_data(image_p=data_directory+image_file, catalog_f=data_directory+catalog_file)
# split data
train_images, train_labels, valid_images, valid_labels, test_images, test_labels = split_data(images, labels)
```

```python
# check data type and shape
print(type(images), type(labels))
print(images.keys(), labels.shape)
print(type(train_images), type(train_labels))
print(train_images.shape, train_labels.shape)
```

    <class 'dict'> <class 'numpy.ndarray'>
    dict_keys(['cluster', 'gsm_3dImgs']) (7946, 3)
    <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    (6435, 50, 50, 10) (6435, 2)

In this section,

- I first import the necessary modules for reading data, machine learning and plot.

- I then define two methods to read data and to split data for learning.

- Next, I read data, split them and check data type and shape.

## 2. Neural network - regression

A simple neural network model setup and training.

```python
# Define loss functions and gradient descent optimizer for future use
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
huber = tf.keras.losses.Huber(delta=.5)

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
```

```python
class TimeHistory(tf.keras.callbacks.Callback):
    '''
    To store the time consumed for each epoch and the whole fit. Also for future use.
    '''
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()
        
    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def train_model(model_type, train_images, train_labels, valid_images, valid_labels, loss_func=mse, n_epochs=40, useRedshift=False):
    '''
    Build and train a specific type of model with given parameters.

    Args:
        model_type (keras.Model):
            Sequential or general model.
        loss_func (tf.function):
            Build-in or customized loss function.
        train_images (np.ndarray):
            Energy images for training inputs.
        train_labels (np.ndarray):
            Including target mass and corresponding redshift for training.
        valid_images (np.ndarray):
            Energy images for validation.
        valid_labels (np.ndarray):
            Target mass and corresponding redshift for validation.
        n_epochs (int, optional):
            Epochs for training process. Defaults to 40.
        useRedshift (bool, optional):
            If use redshift as input or not. Defaults to False.

    Returns:
        model (keras.Model):
            Structured and trained model.
        history (keras.callbacks):
            Stor the training information of every epochs for future analysis.
        time_callback (keras.callbacks):
            Store the time consuming information for future analysis.
    '''
    # create a model
    model = model_type()
    model.compile(optimizer=adam, loss=loss_func)
    
    # create an instance to store time consuming info
    time_callback = TimeHistory()

    # whether includeing redshift as input or not
    if useRedshift:
        inputs = [train_images, train_labels[:, 1]]
        valid_inputs = [valid_images, valid_labels[:, 1]]
    else:
        inputs = train_images
        valid_inputs = valid_images
    
    # target mass and validation mass
    target = train_labels[:, 0]
    valid_target = valid_labels[:, 0]
    
    # train the model
    history = model.fit(
        inputs,
        target,
        epochs=n_epochs,
        callbacks=[time_callback],
        validation_data=(valid_inputs, valid_target)
    )

    return model, history, time_callback
```

```python
def simple_sequential():
    '''
    Build a simple neural network with keras.

    Returns:
        Model (keras.Sequential):
            An instance of sequential model. Detail structure of the model is printed with model.summary() method.
    '''
    # creat an keras.Sequential instance 
    model = models.Sequential()
    
    # add convolutional and pooling layer
    model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(50, 50, 10)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # add flatten and dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    
    model.summary()
    return model
```

```python
# create, train and record a simple sequential model
simple_model, history, time_callback = train_model(
    model_type=simple_sequential,
    train_images=train_images,
    train_labels=train_labels,
    valid_images=valid_images,
    valid_labels=valid_labels
)
```

In this section,

- I first bulid a very simple sequential model, with only very few layers.

- I then compile it and train it with training data set. I specify the optimizer as 'adam' and the loss function as 'mean square error'.

## 3. Model evaluation

Define some evaluation utilities for future use. And estimate the simple sequential model.

```python
# model evaluating cell, methods in this cell are for evaluating the trained model
def loss_epoch_plot(history, loss_name):
    '''
    Plot loss wrt. epochs, to check if the model converges.

    Args:
        history (Keras.callbacks.History):
            Training history.
        loss_name (str):
            Name of used loss function during training.
    '''
    plt.figure(figsize=(10,8))
    plt.plot(history.epoch, history.history.get('loss'), 'o-',label = 'loss')
    plt.plot(history.epoch, history.history.get('val_loss'), 'o-',label = 'validation loss')
    # plt.ylim([0.5, 1])
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss evolvement')
    plt.legend()
    plt.show()


def mass_pred_plot(test_pred, test_target, redshift, loss_name):
    '''
    Plot predicted mass from the model wrt. the real mass, to see how well the model works.

    Args:
        test_pred (np.ndarray):
            Predicted mass.
        test_target (np.ndarray):
            Desired mass.
        loss_name (str):
            Name of loss function.
    '''
    plt.figure(figsize=(10,8))
    plt.scatter(test_target, test_pred, label='predict', s=15, c=redshift)
    x = np.linspace(13,15,20)
    plt.plot(x, x, 'orange', label='desired')
    plt.xlabel('real mass')
    plt.ylabel('predicted mass')
    plt.title('Mass prediction')
    plt.colorbar(label="Redshift")
    plt.legend()
    plt.show()

def error_distribution(test_pred, test_target, loss_name):
    '''
    Plot a histogram showing the distribution of the differences between predicted mass and real mass.

    Args:
        test_pred (np.ndarray):
            Predicted mass.
        test_target (np.ndarray):
            Real mass.
        loss_name (str):
            name of loss function.
    '''
    plt.figure(figsize=(10, 8))
    plt.hist(test_pred-test_target, bins=15, rwidth=.95, label=loss_name)
    plt.xlabel('$M_{pred}-M_{true}$')
    plt.ylabel('frequency')
    plt.title('Mass prediction error distribution')
    plt.legend()
    plt.show()

def error_evaluation(test_pred, test_target):
    '''
    Evaluate the error between predicted mass and real mass. Print out the mean error and standard deviation.

    Args:
        test_pred (np.ndarray): 
            Predicted mass.
        test_target (np.ndarray): 
            Real mass.
    '''
    mean = np.mean(test_pred - test_target)
    std = np.std(test_pred - test_target)
    
    print(f'Mean error: {mean:.5f}')
    print(f'Standard deviation of the error: {std:.5f}')

def model_evaluations(loss_name, model, test_images, test_labels, time_callback, header='', n_epochs=40, useRedshift=False):
    '''
    Evaluate a specific model.

    Args:
        loss_name (str):
            Name of loss function used.
        model (keras.Model):
            Trained model need to be estimated.
        test_images (np.ndarray):
            Energy band image.
        test_labels (np.ndarray):
            Including redshift and target mass.
        time_callback (keras.callbacks):
            Containing time consuming info of training.
        header (str, optional):
            Info of the model. Defaults to ''.
        n_epochs (int):
            Epochs when training the model. Defaults to 40.
        useRedshift (bool, optional):
            Whether the redshift data is used during training. Defaults to False.
    '''
    print('-'*60)
    print(header)
    print('-'*60)
    print('Loss function: ' + loss_name)
    print('Optimizer: adam')
    print('-'*60)
    
    if useRedshift:
        test_pred = model.predict([test_images, test_labels[:, 1]])
    else:
        test_pred = model.predict(test_images)
    
    # Evaluate the model with test data set.
    error_evaluation(test_pred[:, 0], test_labels[:, 0])
    print('Time consumed for {:d} epochs in total: {:.5f}s'.format(n_epochs, time_callback.totaltime))
    print('-'*60)
    
    # Draw the plots.
    loss_epoch_plot(history, loss_name)
    mass_pred_plot(test_pred[:, 0], test_labels[:, 0], test_labels[:, 1], loss_name)
    error_distribution(test_pred[:, 0], test_labels[:, 0], loss_name)    
```

```python
# Evaluate the simple sequential model.
model_evaluations(
    loss_name=mse.name,
    model=simple_model,
    test_images=test_images,
    test_labels=test_labels,
    time_callback=time_callback,
    header='Sequential model'
)
```

    ------------------------------------------------------------
    Sequential model
    ------------------------------------------------------------
    Loss function: mean_squared_error
    Optimizer: adam
    ------------------------------------------------------------
    Mean error: 0.25227
    Standard deviation of the error: 0.40285
    Time consumed for 40 epochs in total: 20.16562s
    ------------------------------------------------------------

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_15_1.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_15_2.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_15_3.png)

In this section,

- I first define three method for visualizing the estimation of the model. And they are also useful for the future estimation for other models.

- I then use the model to predict the mass with input test image data set. And the results are evaluated and visualized as plots. It turns out that the model is working, though we expect better performance.

## 4. Neural network - regression improvement

Next I am going to try some approach to improve the model in order to get a better performance in mass prediction.

### 4.1 Deeper network

Note that I have only one convolution layer in the simple sequential model. And a deeper CNN network is generally more efficient in learning. I will try to add more convolution layers to the sequential model to see if the training efficiency increases.

```python
def deeper_sequential():
    '''
    Build a sequential neural network with more convolution layers.

    Returns:
        Model (keras.Sequential):
            An instance of sequential model. Detail structure of the model is printed with model.summary() method.
    '''
    # creat an keras.Sequential instance 
    model = models.Sequential()
    
    # add convolutional and pooling layer
    model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(50, 50, 10)))
    # model.add(layers.Conv2D(30, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(30, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(30, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(30, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(30, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # add flatten and dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    
    model.summary()
    return model
```

```python
# create, train and record a deeper sequential model
deeper_model, history, time_callback = train_model(
    model_type=deeper_sequential,
    train_images=train_images,
    train_labels=train_labels,
    valid_images=valid_images,
    valid_labels=valid_labels
)
```

```python
# Evaluate the deeper sequential model.
model_evaluations(
    loss_name=mse.name,
    model=deeper_model,
    test_images=test_images,
    test_labels=test_labels,
    time_callback=time_callback,
    header='Deeper sequential model'
)
```

    ------------------------------------------------------------
    Deeper sequential model
    ------------------------------------------------------------
    Loss function: mean_squared_error
    Optimizer: adam
    ------------------------------------------------------------
    Mean error: 0.17715
    Standard deviation of the error: 0.36279
    Time consumed for 40 epochs in total: 20.85197s
    ------------------------------------------------------------

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_20_1.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_20_2.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_20_3.png)

In 4.1 section,

- I creat a model same as the one in section 2 but with two additional convolutional and maxpooling layers.

- Compare these two models' performance, I find that the deeper model works better except for time consuming. For mass prediction, the mean error between predicted mass and real mass, decreases by 92%, the standard deviation decreases by 13%. Number of parameters decreases by 95%, and at the same time the time consuming increases by 19% from 17.5s to 20.9s, for same 40 epochs' training. I am not quiet sure about the reason for the increase. A reasonable conjecture is that more layers increase the time for forward and backward propagations, although we have much more parameters to train in deeper sequential model.

### 4.2 Nonsequential network

Note that I only use the energy image data set as the single input for training the sequential model in the previous sections. A sequential model is actually grouping a linear stack of layers with only one input and output tensor, into a general nonsequential model in Keras. Thus, I will next try to include the redshift data as the second input tensor, making use of as much existing data as possible.

```python
def nonsequential():
    '''
    Build a nonsequential neural network with energy images and redshift as inputs.

    Returns:
        Model (keras.Sequential):
            An instance of sequential model. Detail structure of the model is printed with model.summary() method.
    '''    
    image_input = layers.Input(shape=(50, 50, 10))
    redshift_input = layers.Input(shape=1)

    conv = layers.Conv2D(30, (3, 3), activation='relu', input_shape=(50, 50, 10))(image_input)
    conv = layers.Conv2D(30, (3, 3), activation='relu')(conv)
    pooling = layers.MaxPooling2D((2, 2))(conv)
    conv = layers.Conv2D(30, (3, 3), activation='relu')(pooling)
    conv = layers.Conv2D(30, (3, 3), activation='relu')(conv)
    pooling = layers.MaxPooling2D((2, 2))(conv)
    conv = layers.Conv2D(64, (3, 3), activation='relu')(pooling)
    conv = layers.Conv2D(30, (3, 3), activation='relu')(conv)
    pooling = layers.MaxPooling2D((2, 2))(conv)
    image_flatten = layers.Flatten()(pooling)
    image_dense = layers.Dense(200, activation='relu')(image_flatten)
    image_dense = layers.Dense(100, activation='relu')(image_dense)
    image_dense = layers.Dense(2, activation='relu')(image_dense)
    
    redshift_dense = layers.Dense(1)(redshift_input)
    
    dense = tf.concat([image_dense, redshift_dense], axis=1)
    output = layers.Dense(1, activation='relu')(dense)

    model = tf.keras.Model(inputs=[image_input, redshift_input], outputs=output)

    model.summary()
    return model
```

```python
# create, train and record a nonsequential model
nonsequential_model, history, time_callback = train_model(
    model_type=nonsequential,
    train_images=train_images,
    train_labels=train_labels,
    valid_images=valid_images,
    valid_labels=valid_labels,
    useRedshift=True
)
```

```python
# Evaluate the nonsequential model.
model_evaluations(
    loss_name=mse.name,
    model=nonsequential_model,
    test_images=test_images,
    test_labels=test_labels,
    time_callback=time_callback,
    header='Nonsequential model',
    useRedshift=True
)
```

    ------------------------------------------------------------
    Nonsequential model
    ------------------------------------------------------------
    Loss function: mean_squared_error
    Optimizer: adam
    ------------------------------------------------------------
    Mean error: 0.25971
    Standard deviation of the error: 0.24940
    Time consumed for 40 epochs in total: 30.71719s
    ------------------------------------------------------------

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_25_1.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_25_2.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_25_3.png)

In section 4.2,

- I build a model with two inputs tensor, the energy image and the corresponding redshift. The structure is shown with summary() method.

- Compare with the previous models, this one is not performing well in time consuming and in mean error of mass prediction. But it gives a better standard deviation of the defference between $M_{true}$ and $M_{pred}$ and a better mass prediction plot than the previous models.

### 4.3 Loss functions

I am using the most popular mean square error all the time before. Now it's time to try some other loss functions to see if the model could be better.

Note I will use the nonsequential model in section 4.2 to test loss functions.

```python
loss = mae
# Create, train and record a nonsequential model with mae loss
mae_model, history, time_callback = train_model(
    model_type=nonsequential,
    train_images=train_images,
    train_labels=train_labels,
    valid_images=valid_images,
    valid_labels=valid_labels,
    loss_func=loss,
    useRedshift=True
)

# Evaluate the model.
model_evaluations(
    loss_name=loss.name,
    model=mae_model,
    test_images=test_images,
    test_labels=test_labels,
    time_callback=time_callback,
    header='Model: Nonsequential',
    useRedshift=True
)
```

    Model: "model_6"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_13 (InputLayer)          [(None, 50, 50, 10)  0           []                               
                                    ]                                                                 
                                                                                                      
     conv2d_40 (Conv2D)             (None, 48, 48, 30)   2730        ['input_13[0][0]']               
                                                                                                      
     conv2d_41 (Conv2D)             (None, 46, 46, 30)   8130        ['conv2d_40[0][0]']              
                                                                                                      
     max_pooling2d_22 (MaxPooling2D  (None, 23, 23, 30)  0           ['conv2d_41[0][0]']              
     )                                                                                                
                                                                                                      
     conv2d_42 (Conv2D)             (None, 21, 21, 30)   8130        ['max_pooling2d_22[0][0]']       
                                                                                                      
     conv2d_43 (Conv2D)             (None, 19, 19, 30)   8130        ['conv2d_42[0][0]']              
                                                                                                      
     max_pooling2d_23 (MaxPooling2D  (None, 9, 9, 30)    0           ['conv2d_43[0][0]']              
     )                                                                                                
                                                                                                      
     conv2d_44 (Conv2D)             (None, 7, 7, 64)     17344       ['max_pooling2d_23[0][0]']       
                                                                                                      
     conv2d_45 (Conv2D)             (None, 5, 5, 30)     17310       ['conv2d_44[0][0]']              
                                                                                                      
     max_pooling2d_24 (MaxPooling2D  (None, 2, 2, 30)    0           ['conv2d_45[0][0]']              
     )                                                                                                
                                                                                                      
     flatten_8 (Flatten)            (None, 120)          0           ['max_pooling2d_24[0][0]']       
                                                                                                      
     dense_34 (Dense)               (None, 200)          24200       ['flatten_8[0][0]']              
                                                                                                      
     dense_35 (Dense)               (None, 100)          20100       ['dense_34[0][0]']               
                                                                                                      
     input_14 (InputLayer)          [(None, 1)]          0           []                               
                                                                                                      
     dense_36 (Dense)               (None, 2)            202         ['dense_35[0][0]']               
                                                                                                      
     dense_37 (Dense)               (None, 1)            2           ['input_14[0][0]']               
                                                                                                      
     tf.concat_6 (TFOpLambda)       (None, 3)            0           ['dense_36[0][0]',               
                                                                      'dense_37[0][0]']               
                                                                                                      
     dense_38 (Dense)               (None, 1)            4           ['tf.concat_6[0][0]']            
                                                                                                      
    ==================================================================================================
    Total params: 106,282
    Trainable params: 106,282
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Epoch 1/40
    202/202 [==============================] - 1s 4ms/step - loss: 12.9148 - val_loss: 11.7307
    Epoch 2/40
    202/202 [==============================] - 1s 4ms/step - loss: 10.5093 - val_loss: 9.2425
    Epoch 3/40
    202/202 [==============================] - 1s 4ms/step - loss: 7.9663 - val_loss: 6.6448
    Epoch 4/40
    202/202 [==============================] - 1s 4ms/step - loss: 5.3268 - val_loss: 3.9958
    Epoch 5/40
    202/202 [==============================] - 1s 4ms/step - loss: 2.9266 - val_loss: 2.0605
    Epoch 6/40
    202/202 [==============================] - 1s 4ms/step - loss: 1.6974 - val_loss: 1.4465
    Epoch 7/40
    202/202 [==============================] - 1s 4ms/step - loss: 1.3855 - val_loss: 1.2965
    Epoch 8/40
    202/202 [==============================] - 1s 4ms/step - loss: 1.2687 - val_loss: 1.1942
    Epoch 9/40
    202/202 [==============================] - 1s 4ms/step - loss: 1.1656 - val_loss: 1.0941
    Epoch 10/40
    202/202 [==============================] - 1s 4ms/step - loss: 1.0640 - val_loss: 0.9939
    Epoch 11/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.9639 - val_loss: 0.8953
    Epoch 12/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.8656 - val_loss: 0.7974
    Epoch 13/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.7698 - val_loss: 0.7041
    Epoch 14/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.6777 - val_loss: 0.6150
    Epoch 15/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.5914 - val_loss: 0.5339
    Epoch 16/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.5127 - val_loss: 0.4630
    Epoch 17/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.4453 - val_loss: 0.4039
    Epoch 18/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.3906 - val_loss: 0.3570
    Epoch 19/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.3490 - val_loss: 0.3228
    Epoch 20/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.3197 - val_loss: 0.2994
    Epoch 21/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.3004 - val_loss: 0.2859
    Epoch 22/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2885 - val_loss: 0.2765
    Epoch 23/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2831 - val_loss: 0.2722
    Epoch 24/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2808 - val_loss: 0.2703
    Epoch 25/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2799 - val_loss: 0.2692
    Epoch 26/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2797 - val_loss: 0.2693
    Epoch 27/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2796 - val_loss: 0.2692
    Epoch 28/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2796 - val_loss: 0.2707
    Epoch 29/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2796 - val_loss: 0.2685
    Epoch 30/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2797 - val_loss: 0.2696
    Epoch 31/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2796 - val_loss: 0.2686
    Epoch 32/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2797 - val_loss: 0.2684
    Epoch 33/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2795 - val_loss: 0.2692
    Epoch 34/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2795 - val_loss: 0.2688
    Epoch 35/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2796 - val_loss: 0.2688
    Epoch 36/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2796 - val_loss: 0.2690
    Epoch 37/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2798 - val_loss: 0.2688
    Epoch 38/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2796 - val_loss: 0.2684
    Epoch 39/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2794 - val_loss: 0.2683
    Epoch 40/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.2795 - val_loss: 0.2688
    ------------------------------------------------------------
    Model: Nonsequential
    ------------------------------------------------------------
    Loss function: mean_absolute_error
    Optimizer: adam
    ------------------------------------------------------------
    Mean error: 0.00819
    Standard deviation of the error: 0.35067
    Time consumed for 40 epochs in total: 30.15076s
    ------------------------------------------------------------

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_28_1.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_28_2.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_28_3.png)

```python
loss = huber
# Create, train and record a nonsequential model with huber loss
hubber_model, history, time_callback = train_model(
    model_type=nonsequential,
    train_images=train_images,
    train_labels=train_labels,
    valid_images=valid_images,
    valid_labels=valid_labels,
    loss_func=loss,
    useRedshift=True
)

# Evaluate the model.
model_evaluations(
    loss_name=loss.name,
    model=hubber_model,
    test_images=test_images,
    test_labels=test_labels,
    time_callback=time_callback,
    header='Model: Nonsequential',
    useRedshift=True
)
```

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_5 (InputLayer)           [(None, 50, 50, 10)  0           []                               
                                    ]                                                                 
                                                                                                      
     conv2d_16 (Conv2D)             (None, 48, 48, 30)   2730        ['input_5[0][0]']                
                                                                                                      
     conv2d_17 (Conv2D)             (None, 46, 46, 30)   8130        ['conv2d_16[0][0]']              
                                                                                                      
     max_pooling2d_10 (MaxPooling2D  (None, 23, 23, 30)  0           ['conv2d_17[0][0]']              
     )                                                                                                
                                                                                                      
     conv2d_18 (Conv2D)             (None, 21, 21, 30)   8130        ['max_pooling2d_10[0][0]']       
                                                                                                      
     conv2d_19 (Conv2D)             (None, 19, 19, 30)   8130        ['conv2d_18[0][0]']              
                                                                                                      
     max_pooling2d_11 (MaxPooling2D  (None, 9, 9, 30)    0           ['conv2d_19[0][0]']              
     )                                                                                                
                                                                                                      
     conv2d_20 (Conv2D)             (None, 7, 7, 64)     17344       ['max_pooling2d_11[0][0]']       
                                                                                                      
     conv2d_21 (Conv2D)             (None, 5, 5, 30)     17310       ['conv2d_20[0][0]']              
                                                                                                      
     max_pooling2d_12 (MaxPooling2D  (None, 2, 2, 30)    0           ['conv2d_21[0][0]']              
     )                                                                                                
                                                                                                      
     flatten_4 (Flatten)            (None, 120)          0           ['max_pooling2d_12[0][0]']       
                                                                                                      
     dense_14 (Dense)               (None, 200)          24200       ['flatten_4[0][0]']              
                                                                                                      
     dense_15 (Dense)               (None, 100)          20100       ['dense_14[0][0]']               
                                                                                                      
     input_6 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                      
     dense_16 (Dense)               (None, 2)            202         ['dense_15[0][0]']               
                                                                                                      
     dense_17 (Dense)               (None, 1)            2           ['input_6[0][0]']                
                                                                                                      
     tf.concat_2 (TFOpLambda)       (None, 3)            0           ['dense_16[0][0]',               
                                                                      'dense_17[0][0]']               
                                                                                                      
     dense_18 (Dense)               (None, 1)            4           ['tf.concat_2[0][0]']            
                                                                                                      
    ==================================================================================================
    Total params: 106,282
    Trainable params: 106,282
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Epoch 1/40
    202/202 [==============================] - 1s 5ms/step - loss: 0.7897 - val_loss: 0.1099
    Epoch 2/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.1043 - val_loss: 0.0704
    Epoch 3/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0812 - val_loss: 0.0768
    Epoch 4/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0710 - val_loss: 0.0615
    Epoch 5/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0574 - val_loss: 0.0828
    Epoch 6/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0504 - val_loss: 0.0360
    Epoch 7/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0415 - val_loss: 0.0998
    Epoch 8/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0394 - val_loss: 0.0485
    Epoch 9/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0343 - val_loss: 0.0420
    Epoch 10/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0327 - val_loss: 0.0460
    Epoch 11/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0489 - val_loss: 0.0547
    Epoch 12/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0355 - val_loss: 0.0347
    Epoch 13/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0334 - val_loss: 0.0558
    Epoch 14/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0320 - val_loss: 0.0396
    Epoch 15/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0360 - val_loss: 0.0416
    Epoch 16/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0331 - val_loss: 0.0301
    Epoch 17/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0310 - val_loss: 0.0384
    Epoch 18/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0355 - val_loss: 0.0358
    Epoch 19/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0349 - val_loss: 0.0490
    Epoch 20/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0323 - val_loss: 0.0335
    Epoch 21/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0305 - val_loss: 0.0316
    Epoch 22/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0288 - val_loss: 0.0899
    Epoch 23/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0325 - val_loss: 0.0354
    Epoch 24/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0296 - val_loss: 0.0304
    Epoch 25/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0289 - val_loss: 0.0397
    Epoch 26/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0284 - val_loss: 0.0533
    Epoch 27/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0305 - val_loss: 0.0417
    Epoch 28/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0308 - val_loss: 0.0537
    Epoch 29/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0288 - val_loss: 0.0494
    Epoch 30/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0288 - val_loss: 0.0286
    Epoch 31/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0288 - val_loss: 0.0414
    Epoch 32/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0292 - val_loss: 0.0265
    Epoch 33/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0282 - val_loss: 0.0271
    Epoch 34/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0300 - val_loss: 0.0265
    Epoch 35/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0292 - val_loss: 0.0338
    Epoch 36/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0264 - val_loss: 0.0402
    Epoch 37/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0277 - val_loss: 0.0300
    Epoch 38/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0259 - val_loss: 0.0267
    Epoch 39/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0298 - val_loss: 0.0285
    Epoch 40/40
    202/202 [==============================] - 1s 4ms/step - loss: 0.0291 - val_loss: 0.0429
    ------------------------------------------------------------
    Model: Nonsequential
    ------------------------------------------------------------
    Loss function: huber_loss
    Optimizer: adam
    ------------------------------------------------------------
    Mean error: 0.17451
    Standard deviation of the error: 0.22869
    Time consumed for 40 epochs in total: 30.66767s
    ------------------------------------------------------------

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_29_1.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_29_2.png)

![png](https://jizhaoqin.github.io/assets/images/post/CNN/Jizhao_code_29_3.png)

In 4.3 section,

- I try mae loss and the huber loss (the combination of mae and mse) for the task. It looks like they don't do any better than the mse loss function.

- Maybe adjusting the delta parameter for huber loss function would improve the model's performance, but it is not adaptive then and would consume an unworthy bunch of time of humans for this simple task.

## 5. Log-likelihood loss

```python
@tf.function
def mle(y_true, y_pred):
    '''
    Define a log-likelihood loss function used for training.

    Args:
        y_true (tf.tensor):
            Training target.
        y_pred (tf.tensor):
            Prediction of the training model.

    Returns:
        loss (tf.tensor):
            The loss for the epoch.
    '''
    n = len(y_true)
    
    # Calculate the deviation between y_true and y_pred
    sigma = tf.math.reduce_std(y_true-y_pred, axis=-1)
    
    # Calculate loss
    loss = tf.math.reduce_sum((y_true-y_pred)**2 / 2. / sigma**2, axis=-1)
    loss += tf.math.reduce_sum(tf.math.log(sigma), axis=-1)
    loss += n/2. * (tf.math.log(2*np.pi))
    
    return -loss
```

```python
loss = mle
# Create, train and record a nonsequential model with log-likelihood loss
log_likelihood_model, history, time_callback = train_model(
    model_type=nonsequential,
    train_images=train_images,
    train_labels=train_labels,
    valid_images=valid_images,
    valid_labels=valid_labels,
    loss_func=loss,
    n_epochs=10,
    useRedshift=True
)

# Evaluate the model.
model_evaluations(
    loss_name=loss.name,
    model=log_likelihood_model,
    test_images=test_images,
    test_labels=test_labels,
    time_callback=time_callback,
    header='Model: Nonsequential',
    useRedshift=True
)
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 50, 50, 10)  0           []                               
                                    ]                                                                 
                                                                                                      
     conv2d (Conv2D)                (None, 48, 48, 30)   2730        ['input_1[0][0]']                
                                                                                                      
     conv2d_1 (Conv2D)              (None, 46, 46, 30)   8130        ['conv2d[0][0]']                 
                                                                                                      
     max_pooling2d (MaxPooling2D)   (None, 23, 23, 30)   0           ['conv2d_1[0][0]']               
                                                                                                      
     conv2d_2 (Conv2D)              (None, 21, 21, 30)   8130        ['max_pooling2d[0][0]']          
                                                                                                      
     conv2d_3 (Conv2D)              (None, 19, 19, 30)   8130        ['conv2d_2[0][0]']               
                                                                                                      
     max_pooling2d_1 (MaxPooling2D)  (None, 9, 9, 30)    0           ['conv2d_3[0][0]']               
                                                                                                      
     conv2d_4 (Conv2D)              (None, 7, 7, 64)     17344       ['max_pooling2d_1[0][0]']        
                                                                                                      
     conv2d_5 (Conv2D)              (None, 5, 5, 30)     17310       ['conv2d_4[0][0]']               
                                                                                                      
     max_pooling2d_2 (MaxPooling2D)  (None, 2, 2, 30)    0           ['conv2d_5[0][0]']               
                                                                                                      
     flatten (Flatten)              (None, 120)          0           ['max_pooling2d_2[0][0]']        
                                                                                                      
     dense (Dense)                  (None, 200)          24200       ['flatten[0][0]']                
                                                                                                      
     dense_1 (Dense)                (None, 100)          20100       ['dense[0][0]']                  
                                                                                                      
     input_2 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                      
     dense_2 (Dense)                (None, 2)            202         ['dense_1[0][0]']                
                                                                                                      
     dense_3 (Dense)                (None, 1)            2           ['input_2[0][0]']                
                                                                                                      
     tf.concat (TFOpLambda)         (None, 3)            0           ['dense_2[0][0]',                
                                                                      'dense_3[0][0]']                
                                                                                                      
     dense_4 (Dense)                (None, 1)            4           ['tf.concat[0][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 106,282
    Trainable params: 106,282
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Epoch 1/10



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /scratch-local/slurm-job-tmp-3240682/ipykernel_69/2415240838.py in <cell line: 3>()
          1 loss = mle
          2 # Create, train and record a nonsequential model with log-likelihood loss
    ----> 3 log_likelihood_model, history, time_callback = train_model(
          4     model_type=nonsequential,
          5     train_images=train_images,


    /scratch-local/slurm-job-tmp-3240682/ipykernel_69/3247384336.py in train_model(model_type, train_images, train_labels, valid_images, valid_labels, loss_func, n_epochs, useRedshift)
         66 
         67     # train the model
    ---> 68     history = model.fit(
         69         inputs,
         70         target,


    /software/opt/focal/x86_64/python/3.10-2022.08/lib/python3.10/site-packages/keras/utils/traceback_utils.py in error_handler(*args, **kwargs)
         65     except Exception as e:  # pylint: disable=broad-except
         66       filtered_tb = _process_traceback_frames(e.__traceback__)
    ---> 67       raise e.with_traceback(filtered_tb) from None
         68     finally:
         69       del filtered_tb


    /software/opt/focal/x86_64/python/3.10-2022.08/lib/python3.10/site-packages/tensorflow/python/framework/func_graph.py in autograph_handler(*args, **kwargs)
       1145           except Exception as e:  # pylint:disable=broad-except
       1146             if hasattr(e, "ag_error_metadata"):
    -> 1147               raise e.ag_error_metadata.to_exception(e)
       1148             else:
       1149               raise


    TypeError: in user code:
    
        File "/software/opt/focal/x86_64/python/3.10-2022.08/lib/python3.10/site-packages/keras/engine/training.py", line 1021, in train_function  *
            return step_function(self, iterator)
        File "/scratch-local/slurm-job-tmp-3240682/ipykernel_69/589662153.py", line 24, in mle  *
            loss += n/2. * (tf.math.log(2*np.pi))
    
        TypeError: `x` and `y` must have the same dtype, got tf.int32 != tf.float32.

In section 5,

- I define the log-likelihood loss function called mle.

- Then build and run a nonsequential model with this mle loss function.

- Next I evaluate the performance of the model. The model does work, however not as well as is expected. Time consuming is always the same because I use the same nonsequential model. But it doesn't make any other improvements over the same model with mse loss function, for this mass prediction task.

- <font color='red'>**For the first time I ran the code, it just worked well and I got the above comments. However, the customized loss function totally broke down for the second time running, to which I have no clue about what happened at all. I spend hours trying some fixes but just didn't work anymore.**</font>
