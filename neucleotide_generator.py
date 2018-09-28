
import os

import cv2
import os
import skvideo.io

# Import all modules
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras.utils import np_utils
#from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp

# Loading the CIFAR-10 datasets
from keras.datasets import cifar10

# Declare variables

batch_size = 32
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch

num_classes = 10 #
epochs = 20 # repeat 100 times

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train - training data(images), y_train - labels(digits)

# Convert and pre-processing

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255
sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)


# load json and create model
json_file = open("/home/aswin/proj/assets/models/model5.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/aswin/proj/assets/model5.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def calc_params(res_output):
    ne=1-max(res_output)
    sh=res_output.count(1)
    kl=[]
    for j in res_output:
        if j>0.5:
            kl.append(j)
    mkl=list(kl)
    if len(kl)>=2:
        maxi=max(kl)
        kl.remove(maxi)
        max2=max(kl)
        kl.remove(max2)
        se=maxi+max2
        for j in kl:
            se-=j
    elif len(kl)==1:
        se=kl[0]
    else:
        se=0

    if len(mkl)>0:
        sv=sum(mkl)

    else:
        sv=0

    # print(ne,sh,se,sv)
    return ne,sh,se,sv


def calc_CS(ne,tne,sh,tsh,se,tse,sv,tsv):
    lo=['A','C','G','T']
    if tne==-1 and tsh==-1 and tse==-1 and tsv==-1:
        if ne==max(ne,sh,se,sv):
            return lo[0]
        elif sh==max(ne,sh,se,sv):
            return lo[1]
        elif se==max(ne,sh,se,sv):
            return lo[2]
        elif sv==max(ne,sh,se,sv):
            return lo[3]

    else:
        w1=0.6
        w2=0.4
        diffne=w1*(1-ne+tne)+w2*ne
        diffsh=w1*(1-sh+tsh)+w2*sh
        diffse=w1*(1-se+tse)+w2*se
        diffsv=w1*(1-sv+tsv)+w2*sv
        if diffsh==max(diffsh,diffse,diffsv):
            return lo[1]
        elif diffse==max(diffsh,diffse,diffsv):
            return lo[2]
        elif diffsv==max(diffsh,diffse,diffsv):
            return lo[3]





model=loaded_model
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck','None']
cnt=1

#take every n Frames
n=10

sequence=[]
tne,tsh,tse,tsv=-1,-1,-1,-1

for fn in os.listdir('/home/aswin/proj/frames'):
    cnt+=1
    if cnt%n==0:
        test_image = image.load_img('/home/aswin/proj/frames/'+str(fn), target_size = (32, 32))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        res_output=list(result[0])
        # print(res_output)
        ne,sh,se,sv=calc_params(res_output)
        frame_lab=calc_CS(ne,tne,sh,tsh,se,tse,sv,tsv)
        print(frame_lab)
        sequence.append(frame_lab)
        tne,tsh,tse,tsv=ne,sh,se,sv
