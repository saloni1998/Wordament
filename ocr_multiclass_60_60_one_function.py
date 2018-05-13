
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[11]:

import os
from PIL import Image


# In[62]:

from keras.models import Model
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,GlobalAveragePooling2D,Activation,Dropout
from keras.losses import binary_crossentropy
from keras.optimizers import Adam,SGD


# In[63]:

def model():
    inputs=l=Input((60,60,1))
    l=Conv2D(32,(3,3),padding="same",name="conv_1")(l)
    l=Activation("relu",name="conv_act_1")(l)
    l=MaxPooling2D((3,3),strides=(2,2),padding="valid",name="conv_pool1")(l)
    
    
    l=Conv2D(64,(3,3),padding="same",name="conv_2")(l)
    l=Activation("relu",name="conv_act_2")(l)
    l=MaxPooling2D((3,3),strides=(2,2),padding="valid",name="conv_pool2")(l)
    
    l=Conv2D(128,(3,3),padding="same",name="conv_3")(l)
    l=Activation("relu",name="conv_act_3")(l)
    l=GlobalAveragePooling2D()(l)
    l=Dropout(0.5)(l)
    
    l=Dense(64,activation="relu",name="d1")(l)
    l=Dense(26,activation="softmax",name="d2")(l)
    outputs=l
    return inputs,outputs
    


# In[68]:

from keras.preprocessing.image import ImageDataGenerator



# In[ ]:

def get_ocr(tiles):
    inputs,outputs=model()
    mod=Model(inputs=inputs,outputs=outputs)
    optim=Adam(1e-4)
    mod.compile(optimizer=optim,loss="categorical_crossentropy",metrics=["accuracy"])
    mod.load_weights("ocr_multiclass_dropout_weights_60*60_TransferLearning.h5",by_name=True)
    test=np.asarray(tiles).reshape(-1,60,60,1)
    idg=ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
    idg.fit(test)
    testing_gen = idg.flow(test,[0]*len(tiles),shuffle=False)
    
    

