# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:54:20 2019

@author: German

Segmentacion de las imagenes solo HH y las diferentes opciuones de RESNET
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
import math
import glob

#import needed classes
import keras
from keras.datasets import cifar10
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation
from keras.models import Model,Input,load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from math import ceil, floor
from skimage.util import view_as_blocks
import numpy as np
import hdf5storage
from keras.preprocessing.image import ImageDataGenerator
#from skimage import io, feature
from scipy.io import savemat
import matplotlib.pyplot as plt
from scipy import ndimage

np.random.seed(32)


# Funcion para Dividir una imagen dependiendo del tama;o de ventana
def dividir_conjunto_imagen(I, tam):
    tamx, tamy, tamz = I.shape
    ventx = floor(tamx/tam)
    venty = floor(tamy/tam)
    ##Arreglofinal de las imagenes divididas
    arreglo_imagen = np.empty((ventx*venty, tam, tam,3), dtype=I.dtype)
    I_rec = I[:(ventx*tam),:(venty*tam)]
    #for k in range(I_rec.shape[0]):
        #print(k)
    patches2 = view_as_blocks(I_rec, block_shape=(tam,tam,3))
    arreglo_imagen=patches2.reshape(patches2.shape[0]*patches2.shape[1],patches2.shape[3], patches2.shape[4],3)
    return I_rec,arreglo_imagen,ventx,venty

def recuperar_tam_original(X_train, tam, ventx1, venty1):
    im = np.empty(shape = I_rec.shape)
    k = 0
    for i in range(ventx1):
        for j in range(venty1):
            im[(i*tam):((i+1)*tam),(j*tam):((j+1)*tam),:] = X_train[k,:,:,:]
            k = k+1    
    return im

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
    
####Inicio del programa
        


name_imgs = ("8F1A_HH_TN_dB_MAT.mat", "52FC_HH_TN_dB_MAT.mat", "4424_HH_TN_dB_MAT.mat", "D07D_HH_TN_dB_MAT.mat", 
             "E94E_HH_TN_dB_MAT.mat", "EA9D_HH_TN_dB_MAT.mat")    
modelo = 'Transfer_ResNET50_HH'
#modelos = ('01.hdf5','02.hdf5','03.hdf5','04.hdf5','05.hdf5','06.hdf5','07.hdf5','08.hdf5','11.hdf5','12.hdf5'
#,'13.hdf5','15.hdf5','16.hdf5')
modelos = ('17.hdf5','18.hdf5','19.hdf5','20.hdf5')
carpeta = "ImgsValidacionHH/"
carp_etiqueta = "etiquetas_Transfer_ResNET50_HH/"


tam = 224
num_clases = 4

# primero cargar el modelo adecuado
for mod in range(int(len(modelos))):
    model = load_model(modelo + modelos[mod])
    print(modelos[mod])
# despues... cargar las imagenes que le corresponden
    for img in range(int(len(name_imgs))):
        nombre = carpeta + name_imgs[img]
        print(nombre)
        Imagen = hdf5storage.loadmat(nombre)['I'].astype('float16')
        Imagen = Imagen.reshape(Imagen.shape[0],Imagen.shape[1],1)
        ImagenRGB = np.repeat(Imagen, 3, axis = 2)
        #I_rec es la imagen original recortada, X_train son los bloques y vent es el numero de ventans en cda dimension
        I_rec,X_train, ventx1, venty1 = dividir_conjunto_imagen(ImagenRGB, tam)
        print(X_train.shape)
        y_pred = model.predict(X_train)
        
#            y_pred2 = y_pred.max(axis=1)
#            y_pred2  = np.reshape(y_pred2, (y_pred2.shape[0], 1))
#            y_pred2  = np.repeat(y_pred2, num_clases, axis=1)
#            y_pred3 = np.where(y_pred == y_pred2)[1]
#            
#            #####Crear un vector de etiquetas de solo pancake
#            y_pred4 = (y_pred3 == 0).astype(int)
#            
#            #Ahora sí, a generar matrices que se multiplicaran por las etiquetas
#            Mat = np.ones(shape = X_train.shape[:3])
#            Mat2 = np.empty(shape = Mat.shape)
#            #Mat2 = np.multiply( y_pred3 , Mat)
#            for idx in range(Mat.shape[0]):
#                Mat2[idx,:,:] = y_pred4[idx]*Mat[idx,:,:]
#            
#            Mat2 = np.reshape(Mat2, (Mat2.shape[0], Mat2.shape[1],Mat2.shape[2],1))
#            
#            #Recuperar la imagen original
#            Mat_rec2 = recuperar_tam_original(Mat2, tam, ventx1, venty1)
        
        savemat(carp_etiqueta + modelo[-2:] + modelos[mod][-7:-5] +'_y_pred_'+ name_imgs[img][:4] +'.mat', {'y_pred': y_pred})

