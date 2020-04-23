# Pancake
Descripción del algoritmo de clasificación/ detección de hielo pancake

## Algoritmo

Las paqueterías necesarias para ejecutar el algoritmo son:

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
```
También se observa el uso de la función de activación **LeakyReLU** y técnicas de optimización como **Dropout**. La capa final tiene 4 nodos o neuronas finales porque esta descripción se utilizará para describir las siguientes 4 clases:

1. Pancake
2. Mar Blanco
3. Mar Negro
4. Tierra y trozos de hielo

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

Antes de compilar el modelo, mantenemos sin reentrenar los pesos importados de la primera capa convolucional:

```python
model=Model(inputs=base_model.input,outputs=preds)
for layer in model.layers[:1]:
    layer.trainable=False
for layer in model.layers[1:]:
    layer.trainable=True
model.summary()
model.compile(optimizer=Nadam(),loss="categorical_crossentropy",metrics=["accuracy"])
```

Además, se utilizaron 3 callbacks:
1. Almacenar el modelo que tenga el modelo con la mejor exactitud.
2. Paro anticipado para que detenga el ajuste cuando no se ha presentado una mejora en la exactitud de clasificación en cierto número de épocas.
3. Reducir la tasa de aprendizaje (learning rate) cuando no exista una mejora en la exactitud cada cierto número de épocas.

Finalmente se entrena el modelo bajo las condiciones antes mencionadas:

```python
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='acc',  min_delta=0.001, patience=20, mode='auto', verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='acc', min_delta=0.001, factor=0.5, patience=5, verbose=1, min_lr=0.00001, mode='max')
callbacks_list = [checkpoint, earlystop, reduce_lr]

model.fit(X_train, y_train, epochs=epochs, verbose=1,callbacks=callbacks_list,shuffle=True,)
```

Una vez que se calculan y almacenan los pesos del mejor modelo. Las imágenes del conjunto de validación son procesadas por el programa llamado **SegmentacionImgsNuevas224HH.py**. Cada una de esas imágenes debe ser dividida en parches de 224 x 224 pixeles y para ello se tiene la siguiente función:

```python
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
```

Se carga el modelo calculado anteriormente y cada parche es clasificado. Se almacenan las probabilidades de clase para cada parche en un archivo **.mat** para su posterior procesamiento:

```python
y_pred = model.predict(X_train)
savemat(carp_etiqueta + modelo[-2:] + modelos[mod][-7:-5] +'_y_pred_'+ name_imgs[img][:4] +'.mat', {'y_pred': y_pred})
```

Ahora, el despliegue de resultados se realiza a través del programa **Indice_DICE.m**. De los datos almacenados el archivo **.mat** se puede calcular la etiqueta de clase para cada parche de 224 x 224 pixeles, entonces hay que reconstruir una máscara de segmentación del mismo tamaño de la imagen por cada parche que la compone. La construcción de la máscara se obtiene a partir de determinar la clase del parche y reacomodar dichos parches en la posición que ocupan dentro de la imagen original de validación:

```matlab
%Leer las etiquetas
y_pred = load([dir_etiquetas model{mod} '_' fnum{fn} '_y_pred_' im{ix} '.mat']).y_pred;
%Clasificación final
[~,y_pred2] = max(y_pred,[],2);

%Imagen recortada a un tamaño adecuado
I = Imagen(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N);
%Obtener las ventanas
block_0 = mat2cell(I, repmat(N, [1, floor(size(I,1)/N)]), repmat(N, [1, floor(size(I,2)/N)]));
%Imagen que tendrá las etiquetas de clasificación
Iseg = zeros(size(I));

l = 1;
for k = 1:size(block_0,1)
    for m = 1:size(block_0,2)
        %Creación de las ventanas
        cuadro = y_pred2(l,1).*ones(N);
        Iseg(((k-1)*N)+1:k*N, ((m-1)*N)+1:m*N) = cuadro;
        l = l+1;
    end
end

%Usar solo los que tiene la etiqueta de pancake
Iseg_1 = (Iseg==1);
```

También podemos aprovechar que se tiene las máscaras de tierra para cada imagen de validación y la quitamos de la máscara obtenida anteriormente para formar la máscara final de pancake:

```matlab
%Quitar la tierra
Tierra = load([dir_tierras im{ix} '_LAND.mat']);
Tierra = Tierra.h;
Tierra = logical(Tierra(1:(floor(size(Imagen,1)/N))*N, 1:(floor(size(Imagen,2)/N))*N));

%MÁSCARA FINAL DE PANCAKE
Pancake_final = Iseg_1 & ~Tierra;
```

Finalmente, esa máscara de pancake final se compara con la máscara generada por el experto (ground truth) a través del índice Dice, una métrica muy conocida para la validación de segmentaciones:

```matlab
%Leer las mascaras objetivo
original = load([dir_original im{ix} '_MASK.mat']).BW;
original = original(1:(floor(size(original,1)/N))*N, 1:(floor(size(original,2)/N))*N);
Res_dice(mod,ix,fn) = dice(original,Pancake_final);
```


