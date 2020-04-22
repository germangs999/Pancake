# Pancake
Descripción del algoritmo de clasificación/ detección de hielo pancake

## Algortimo

Las paqueterías necesarias para ejecutar el algoritmo son :

```python
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
```
Se observa que la red utilizada en este caso es la ResNET101. A este modelo importado se le agregan unas capas totalmente conectadas, por lo que el modelo final se obtiene de la siguiente manera:

```python
base_model=ResNet101(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
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
```
También se observa el uso de la función de activación **LeakyReLU** y técnicas de optimización como **Dropout**. La capa final tiene 4 nodos o neuronas finales porque esta descripción se utilizará para describir las siguientes 4 clases:

1. Pancake
2. Tierra
3. Hielo claro
4. Hielo oscuro

Por cada imagen satelital se obtienen 2000 parches de cada clase, es decir, al final se tienen 8000 parches por clase para entrenamiento. Cada parche mide 224 x 224 pixeles porque es el tamaño de entrada de la ResNet. Estos parches están almacenados en archivos de Matlab que se pueden leer con la siguiente función:

```python
def LeerEstructuras(nombre, no_im):
    Arreglo = hdf5storage.loadmat(nombre)
    Imagenes = Arreglo['Arreglo'].astype('float16')
    etiquetas = Arreglo['etiqueta']
    no_img = no_im*np.ones(shape=(Imagenes.shape[0]))
    Imagenes = Imagenes.reshape((Imagenes.shape[0], Imagenes.shape[1], Imagenes.shape[2],1))
    return Imagenes, etiquetas, no_img
```

