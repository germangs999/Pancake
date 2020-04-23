# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 07:38:37 2019

@author: German
"""

#import needed classes
import keras
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation,GlobalMaxPooling2D,GlobalAveragePooling2D,LeakyReLU
from keras.models import Model,Input
from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical
from math import ceil
import os
import numpy as np
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from keras_applications.resnet import ResNet101

#fijar la semilla aleatoria para la repetibilidad del experimento
np.random.seed(32)

# Para indicar que GPU vas a usar
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Funcio para leer las imagenes de los arreglos
def LeerEstructuras(nombre, no_im):
    Arreglo = hdf5storage.loadmat(nombre)
    Imagenes = Arreglo['Arreglo'].astype('float16')
    etiquetas = Arreglo['etiqueta']
    no_img = no_im*np.ones(shape=(Imagenes.shape[0]))
    Imagenes = Imagenes.reshape((Imagenes.shape[0], Imagenes.shape[1], Imagenes.shape[2],1))
    return Imagenes, etiquetas, no_img

###############Importacion del modelo 
base_model=ResNet101(weights='imagenet',include_top=False, 
                     backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024)(x) 
x = LeakyReLU(alpha=0.3)(x)
x=Dense(1024)(x) 
x = LeakyReLU(alpha=0.3)(x)
x=Dense(512)(x) 
x = LeakyReLU(alpha=0.3)(x)
x=Dropout(0.4)(x)
preds=Dense(4,activation='softmax')(x) 

###Leer las imagenes y sus etiqeutas
Imagenes_9E49, etiquetas_9E49, noimg_9E49 = LeerEstructuras('Filtrado1/VentanasHH_full_traslaTodo_w224_9E49_TN_calibDB_FIX.mat',0)
Imagenes_0309, etiquetas_0309, noimg_0309 = LeerEstructuras('Filtrado1/VentanasHH_full_traslaTodo_w224_0309_TN_calibDB_FIX.mat',1)
Imagenes_D419, etiquetas_D419, noimg_D419 = LeerEstructuras('Filtrado1/VentanasHH_full_traslaTodo_w224_D419_TN_calibDB_FIX.mat',2)
Imagenes_FFAF, etiquetas_FFAF, noimg_FFAF = LeerEstructuras('Filtrado1/VentanasHH_full_traslaTodo_w224_FFAF_TN_calibDB_FIX.mat',3)

#Concatenar las matrices con las imagenes de 234x234 y generar las 3 dimensiones RGB
X_train = np.concatenate((Imagenes_9E49, Imagenes_0309, Imagenes_D419, Imagenes_FFAF), axis=0)
X_train = np.repeat(X_train, 3, -1)
print(X_train.shape) 

no_img = np.concatenate((noimg_9E49, noimg_0309, noimg_D419, noimg_FFAF), axis=0)
etiquetas = np.concatenate((etiquetas_9E49, etiquetas_0309, etiquetas_D419, etiquetas_FFAF), axis=0)
etiquetas = to_categorical(etiquetas)


#Eliminar las variables pesadas
del Imagenes_9E49, Imagenes_0309, Imagenes_D419, Imagenes_FFAF

y_train = etiquetas
    
input_shape = (X_train.shape[1:])
model=Model(inputs=base_model.input,outputs=preds)

#Mantener fijos los pesos de la primera capa
for layer in model.layers[:1]:
    layer.trainable=False
for layer in model.layers[1:]:
    layer.trainable=True
    
model.summary()
model.compile(optimizer=Nadam(),loss="categorical_crossentropy",metrics=["accuracy"])
epochs = 500

##Callbacks
# checkpoint
filepath="Transfer_ResNET50_HH4.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='acc',  min_delta=0.001, patience=20, mode='auto', verbose=1
                          ,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='acc', min_delta=0.001, factor=0.5, patience=5, 
                              verbose=1, 
                              min_lr=0.00001, mode='max')
callbacks_list = [checkpoint, earlystop, reduce_lr]
model.fit(X_train, y_train, epochs=epochs, verbose=1,callbacks=callbacks_list,shuffle=True,)