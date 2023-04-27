from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
from os import environ

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import DenseNet169
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.layers import GlobalAveragePooling2D
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Reshape

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut up tensorflow!
print("tf : {}".format(tf.__version__))
print("keras : {}".format(keras.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--b', '--batch-size', default=16, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('--lr-wait', default=10, type=int, help='how long to wait on plateu')
parser.add_argument('--decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--fullretrain', dest='fullretrain', action='store_true', help='retrain all layers of the model')
parser.add_argument('--seed', default=1337, type=int, help='random seed')
parser.add_argument('--img_channels', default=3, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--early_stop', default=20, type=int)


def new_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', strides=2, input_shape=(224, 224, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Activation('relu'))

    for _ in range(5):
        model.add(Conv2D(32, (3, 3), padding='same', strides=2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

    return model


def train():
    global args
    args = parser.parse_args()
    img_shape = (args.img_size, args.img_size, args.img_channels)
    now_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')

    # We then scale the variable-sized images to 224x224
    # We augment ... by applying random lateral inversions and rotations.
    traindf=pd.read_csv('/home/ilab/Downloads/ISIC2018_Task3_Training_class/test_train57_meta.csv',dtype=str)
    valdf=pd.read_csv('/home/ilab/Downloads/ISIC2018_Task3_Training_class/train_test_meta.csv',dtype=str)
    testdf=pd.read_csv('/home/ilab/Downloads/ISIC2018_Task3_Training_class/test_test57_meta.csv',dtype=str)
    
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
        # rotation_range=45,
        # # width_shift_range=0.2,
        # # height_shift_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
        )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe = traindf,
        directory='/home/ilab/Downloads/ISIC2018_Task3_Training_class/',
        x_col='id',
        y_col='label',
        color_mode='grayscale',
        batch_size=5,
        shuffle=True,
        class_mode='categorical',
        target_size=(args.img_size, args.img_size))

    
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_dataframe(
        dataframe = valdf,
        directory='/home/ilab/Downloads/ISIC2018_Task3_Training_class/',
        x_col="id",
        y_col="label",
        color_mode='grayscale',
        shuffle=True,
        target_size=(args.img_size, args.img_size),
        class_mode='categorical',
        batch_size=5)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe = testdf,
        directory='/home/ilab/Downloads/ISIC2018_Task3_Training_class/',
        x_col="id",
        y_col='label',
        color_mode='grayscale',
        shuffle=False,
        target_size=(args.img_size, args.img_size),
        class_mode='categorical',
        batch_size=5)

    classes = len(train_generator.class_indices)
    assert classes > 0
    assert classes is len(val_generator.class_indices)
    n_of_train_samples = train_generator.samples
    n_of_val_samples = val_generator.samples

    # Architectures
    '''
    base_model = DenseNet169(input_shape=img_shape, weights='imagenet', include_top=False)
    x = base_model.output  # Recast classification layer
    # x = Flatten()(x)  # Uncomment for Resnet based models
    x = GlobalAveragePooling2D(name='predictions_avg_pool')(x)  # comment for RESNET models
    
    x = Dense(args.classes, activation='softmax', name='predictions_ft_metatesttrain57_from_metatrainclass_scratch')(x)
    model = Model(inputs=base_model.input, outputs=x)
    '''

    model = new_model()
    
    # checkpoint
    checkpoint = ModelCheckpoint(filepath='./model_checkpoint/keras_finetune_models/Model1_ft_metatesttrain57_from_metatrainclass_scratch.hdf5', verbose=1)
    early_stop = EarlyStopping(patience=args.early_stop)
    tensorboard = TensorBoard(log_dir='./model_checkpoint/keras_finetune_models/logs/Model1_ft_metatesttrain57_from_metatrainclass_scratch/{}/'.format(now_iso))
    # reduce_lr = ReduceLROnPlateau(factor=0.03, cooldown=0, patience=args.lr_wait, min_lr=0.1e-6)
    callbacks = [checkpoint]

    # Calculate class weights
    weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
    
    model.load_weights('model_checkpoint/keras_finetune_models/Model1_metatrainclass_scratch.hdf5', by_name = True)
    #print('model.layers: ', model.layers)
    model.layers.pop()
    model.add(Dense(args.classes, activation='softmax'))

    # The network is trained end-to-end using Adam with default parameters
    model.compile(
        optimizer=Adam(lr=args.lr, decay=args.decay),
        # optimizer=SGD(lr=args.lr, decay=args.decay,momentum=args.momentum, nesterov=True),
        loss=categorical_crossentropy,
        metrics=[categorical_accuracy], )


    model.fit_generator(
        train_generator,
        steps_per_epoch=n_of_train_samples // 5,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=n_of_val_samples // 10,
        class_weight=weights,
        callbacks=callbacks)

    
    model.load_weights('model_checkpoint/keras_finetune_models/Model1_ft_metatesttrain57_from_metatrainclass_scratch.hdf5')


    predicted_test_outputs = model.predict_generator(test_generator)
    print('predicted_test_outputs: ', predicted_test_outputs)
    predicted_class_indices=np.argmax(predicted_test_outputs,axis=1)
    print('predicted_class_indices: ', predicted_class_indices)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    print('labels: ', labels)

    predictions = [labels[k] for k in predicted_class_indices]
    print('predictions: ', predictions)

    print('testdf[label]: ', testdf['label'][0])

    correct_prediction = 0

    for i in range(len(predictions)):
        if(int(predictions[i]) == int(testdf['label'][i])):
            correct_prediction+=1

    print('Meta-Test Accuracy: ', float(correct_prediction)/len(predictions))


if __name__ == '__main__':
    train()
