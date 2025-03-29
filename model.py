#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

path = 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
plt.figure(figsize=(70, 70))
count = 0
plant_names = []
total_images = 0
for i in os.listdir(path):
    count += 1
    plant_names.append(i)
    plt.subplot(7, 7, count)

    images_path = os.listdir(path + "/" + i)
    print("Number of images of " + i + ":", len(images_path), "||", end=" ")
    total_images += len(images_path)

    image_show = plt.imread(path + "/" + i + "/" + images_path[0])

    plt.imshow(image_show)
    plt.xlabel(i)

    plt.xticks([])
    plt.yticks([])

print("Total number of images we have", total_images)

import tensorflow
from tensorflow import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,AveragePooling2D,Dense,Flatten,ZeroPadding2D,BatchNormalization,Activation,Add,Input,Dropout,GlobalAveragePooling2D
from keras.metrics import Recall, Precision
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras import layers
from keras.layers import Input

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
base_model_tf=ResNet50(include_top=False,weights='imagenet',input_shape=(256,256,3),classes=38)

# base_model_tf=VGG19(include_top=False,weights='imagenet',input_shape=(256,256,3),classes=38)

# #Model building
base_model_tf.trainable=False

pt=Input(shape=(256,256,3))
func=tensorflow.cast(pt,tensorflow.float32)
x=preprocess_input(func) #This function used to zero-center each color channel wrt Imagenet dataset
model_resnet=base_model_tf(x,training=True)
model_resnet=GlobalAveragePooling2D()(model_resnet)
model_resnet=Dense(256,activation='relu')(model_resnet)
model_resnet=Dense(128,activation='relu')(model_resnet)
model_resnet=Dense(64,activation='relu')(model_resnet)
model_resnet=Dense(38,activation='softmax')(model_resnet)


model_main=Model(inputs=pt,outputs=model_resnet)
model_main.summary()

# Image augmentation
train_datagen= ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip=False,vertical_flip=False
                                  ,fill_mode='nearest',width_shift_range=0.2,height_shift_range=0.2)

val_datagen=ImageDataGenerator()

path_train='New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'

path_valid='New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

train= train_datagen.flow_from_directory(directory=path_train,batch_size=32,target_size=(256,256),
                                         color_mode='rgb',class_mode='categorical',seed=42)

valid=val_datagen.flow_from_directory(directory=path_valid,batch_size=32,target_size=(256,256),color_mode='rgb',class_mode='categorical')
import datetime

path = "./"
dt = datetime.datetime.now()
dt = "{}{}{}_{}{}{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
save_path= path + "model" + dt + '.h5'

#CallBacks
# checkpoint = ModelCheckpoint(save_path, monitor='val_f1_m', verbose=1, save_best_only=True, mode='max')
es=EarlyStopping(monitor='val_loss',verbose=1,patience=7,mode='auto')
mc=ModelCheckpoint(save_path,monitor='val_loss',verbose=1,save_best_only=True)
lr=ReduceLROnPlateau(monitor='val_loss',verbose=1,patience=5,min_lr=0.001)
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
model_main.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy', f1_m, precision_m, recall_m]) #, F1Score(num_classes=38, threshold=0.8)])
#Training
model_main.fit(train,validation_data=valid,epochs=3,steps_per_epoch=500,verbose=1,callbacks=[mc,es,lr])

model_main.save("RESNET50_PLANT_DISEASE.h5")
import matplotlib.pyplot as plt

#import cv2
#from PIL import Image
plt.figure(figsize=(10,5))
plt.plot(model_main.history.history['loss'],color='b',label='Training loss')
plt.plot(model_main.history.history['val_loss'],color='r',label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel("epochs")
plt.ylabel("loss_value")
plt.title("loss")
plt.show()
plt.figure(figsize=(10,5))
plt.plot(model_main.history.history['accuracy'],color='b',label='Training accuracy')
plt.plot(model_main.history.history['val_accuracy'],color='r',label='Validation accuracy')
plt.legend(loc='lower right')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy graph")
plt.show()

