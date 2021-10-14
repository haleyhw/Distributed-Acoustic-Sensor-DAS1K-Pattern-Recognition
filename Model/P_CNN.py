# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:25:21 2021

@author: Haley_Wu
"""

import os
import tensorflow
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpu_devices[0], True)

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

P_Mel_X=np.load('PP_feature_X.npy')
P_Mel_Y=np.load('PP_Y.npy')


le=LabelEncoder()
p_y=to_categorical(le.fit_transform(P_Mel_Y))


num_rows =40
num_columns = 98
num_channels = 1
p_x = P_Mel_X.reshape(P_Mel_X.shape[0], num_rows, num_columns, num_channels)
############################################shuffle#################################
RS=20
p_xx,p_yy = shuffle(p_x,p_y,random_state=RS)

###########################################split#####################################

kf = KFold(n_splits=5,shuffle=True,random_state=RS)
p_x_train=[]
p_x_test=[]
p_y_train=[]
p_y_test=[]
for train, test in kf.split(p_xx, p_yy):
    p_x_train.append(p_xx[train])
    p_x_test.append(p_xx[test])
    p_y_train.append(p_yy[train])
    p_y_test.append(p_yy[test])
    
##########################################P_CNN#######################################

filter_size=3
def model(num_rows, num_columns, num_channels):
    # construct model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=filter_size, input_shape=(num_rows, num_columns, num_channels), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=filter_size, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=filter_size, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=filter_size, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))
  
    model.add(GlobalAveragePooling2D())

    model.add(Dense(10,activation='softmax'))

    return model
##########################################Training#######################################

num_epochs =3000
num_batch_size =128

acc_per_fold=[]
loss_per_fold=[]
p_history_accuracy=[]
p_history_val_accuracy=[]
p_history_loss=[]
p_history_val_loss=[]
p_conf_matrix=[]
start_time=time.time()
for i in np.arange(5):
    P_model=model(num_rows, num_columns, num_channels)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    P_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=opt)
   
    p_history=P_model.fit(p_x_train[i], p_y_train[i], 
                        batch_size=num_batch_size, 
                        epochs=num_epochs, 
                        validation_data=(p_x_test[i], p_y_test[i]), 
                        verbose=1)
    p_history_accuracy.append(p_history.history['accuracy'])
    p_history_val_accuracy.append(p_history.history['val_accuracy'])
    p_history_loss.append(p_history.history['loss'])
    p_history_val_loss.append(p_history.history['val_loss'])
    predictions=P_model.predict_classes(p_x_test[i])
    p_conf_matrix.append(confusion_matrix(np.argmax(p_y_test[i],axis=1), predictions))
    #P_model.save('./saved_model/P_model_'+str(i)+'.h5')
print(time.time()-start_time) 

p_history_accuracy_mean=np.mean(p_history_accuracy,axis=0)
p_history_val_accuracy_mean=np.mean(p_history_val_accuracy,axis=0)
p_history_loss_mean=np.mean(p_history_loss,axis=0)
p_history_val_loss_mean=np.mean(p_history_val_loss,axis=0)

################ plotting##########
import matplotlib as mpl

plt.figure(figsize=(18,7))
label_size=26
legend_size=24
mpl.rcParams["font.family"] = "Arial"
plt.subplot(121)
plt.plot(p_history_accuracy_mean)
plt.plot(p_history_val_accuracy_mean)
plt.xlim([0,3000])
plt.ylim([0,1])
plt.xticks(fontsize=legend_size-2)
plt.yticks(fontsize=legend_size-2)
plt.legend(['train', 'validation'], loc='upper left',fontsize=legend_size,frameon=False)
plt.ylabel('accuracy',fontsize=label_size)
plt.xlabel('epoch',fontsize=label_size)

plt.subplot(122)
plt.plot(p_history_loss_mean)
plt.plot(p_history_val_loss_mean)
plt.xlim([0,3000])
plt.ylim([0,2.5])
plt.legend(['train', 'validation'], loc='upper right',fontsize=legend_size,frameon=False)
plt.xticks(fontsize=legend_size-2)
plt.yticks(fontsize=legend_size-2)

plt.ylabel('loss',fontsize=label_size)
plt.xlabel('epoch',fontsize=label_size)
plt.show()

