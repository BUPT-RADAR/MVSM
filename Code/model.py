# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:11:04 2022

@author: zhoujl
"""

import numpy as np
import math,random,string
import sklearn
import matplotlib.pyplot as plt
import datetime as dt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,Conv2D,Input,LSTM,Dropout,MaxPool2D,Flatten,Reshape
from tensorflow.keras import optimizers,regularizers
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,classification_report
import os
from collections import Counter

class SELayer(tf.keras.Model):
    def __init__(self, filters, reduction=16):
        super(SELayer, self).__init__()
        self.filters = filters
        self.reduction = reduction
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.FC = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=self.filters // self.reduction, input_shape=(self.filters, )),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(units=filters),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid')
        ])
        self.Multiply = tf.keras.layers.Multiply()

    def call(self, inputs, training=None, mask=None):
        x = self.GAP(inputs)
        x = self.FC(x)
        x = self.Multiply([x, inputs])
        return x
    
    def build_graph(self, input_shape):
        input_shape_without_batch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_without_batch)
        _ = self.call(inputs)

BATCH_SIZE=50

class_weight = {0: 0.3, 1: 0.7}
def lr(epoch):
    initial_lrate = 0.01
    drop = 0.6
    epochs_drop = 20
    learning_rate = round(initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop)),6)
    print("Learning rate is {}.".format(learning_rate))
    return learning_rate

def savefig(hist,lb,modelname):
    if lb=='loss':
        plt.figure(figsize=(18,12))
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./lossfig/'+modelname+'.jpg')
        
    elif lb=='accuracy':
        plt.figure(figsize=(18,12))
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
#         print(hist.history['val_acc'])
#         np.savetxt(modelname+'.txt', hist.history['val_accuracy'], fmt="%.2f", delimiter=',') #保存为2位小数的浮点数，用逗号分隔
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./accfig/'+modelname+'.jpg')
        
    else:
        print('error param.')

#"valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input
#. When padding="same" and strides=1, the output has the same size as the input.

def CreateModel1():#CNN+LSTM
    inputs=Input(batch_shape=(BATCH_SIZE,50,50,1))
    conv2d_1_1=Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(inputs)
    conv2d_1_2=Conv2D(filters=8,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')(inputs)
    conv2d_1_3=Conv2D(filters=8,kernel_size=(8,8),strides=(1,1),padding='same',activation='relu')(inputs)
    concat_1=keras.layers.Concatenate(axis=3)([conv2d_1_1,conv2d_1_2,conv2d_1_3])
    conv2d_2_1=SELayer(24)(concat_1)
    maxpooling2d=MaxPool2D(pool_size=(2, 2),strides=None, padding='valid')(conv2d_2_1)
    conv2d_3_1=Conv2D(filters=64,kernel_size=(1,25),strides=(1,1),padding='valid',activation='relu')(maxpooling2d)  
    reshape=Reshape(target_shape=(25,64))(conv2d_3_1)
    lstm=LSTM(128)(reshape)
    outputs=Dense(2,activation='softmax')(lstm)
    model=Model(inputs=inputs,outputs=outputs)
    adam = optimizers.Adam(lr=0.003,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    
    return model


def trainmodel(model,sample_train,label_train,sample_test,label_test):
    # model=CreateModel1()
    model.summary()
    model_filename=str(dt.datetime.now())
    model_filename=model_filename[:10]
    model_filename=model_filename+''.join(random.sample(string.ascii_letters, 5))
    history=model.fit(sample_train, 
                      label_train,
                      epochs=100,
                      verbose=1, 
                      batch_size=BATCH_SIZE,
                       class_weight=class_weight,
                      validation_data=(sample_test,label_test)
                      ,callbacks=[
                          # keras.callbacks.LearningRateScheduler(lr),
                          keras.callbacks.ModelCheckpoint(model_filename +'.h5', monitor='val_accuracy',
                                                          verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')]
                     )
    model.save('./model/'+model_filename+'.h5')
    
    return model,history,model_filename

NUM_CLASSES=2

data, label=loadtrainData()
data=data.reshape((-1,50,50,1))

Index=np.arange(data.shape[0])
np.random.shuffle(Index)
data=data[Index]
label=label[Index]
print(Counter(label))
label=keras.utils.to_categorical(label,NUM_CLASSES)

data_train=data[:6550]
label_train=label[:6550]

data_test=data[6550:]
label_test=label[6550:]

model=CreateModel1()

model,history,model_filename=trainmodel(model,data_train, label_train, data_test, label_test)














