# Import all modules
import time
import matplotlib.pyplot as plt
import numpy as np
import random
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
epochs = 1 # repeat 100 times

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train - training data(images), y_train - labels(digits)

# Convert and pre-processing

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255
# sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.5, nesterov=True)

def custom_optimizer(init_flag,par1,par2,par3):
    if init_flag==0:
        param1=random.randint(1,10)/100
        param2=random.randint(1,10)/10000
        param3=random.randint(1,10)/10

    else:
        param1=par1+random.randint(-10,10)/1000
        param2=par2+random.randint(-10,10)/100000
        param3=par3+random.randint(-10,10)/100

    return param1,param2,param3

def generate_custompop(n):
    l=[]
    par=[]
    for i in range(n):
        lr,dec,mom = custom_optimizer(0,0,0,0)
        par.append((lr,dec,mom))
        sgd = SGD(lr = lr, decay=dec, momentum=mom, nesterov=True)
        l.append(sgd)

    return l,par

def get_next_generation(pop,n):
    l=[]
    par=[]
    for i in range(n):
        lr,dec,mom = custom_optimizer(1,pop[0],pop[1],pop[2])
        par.append((lr,dec,mom))
        sgd = SGD(lr = lr, decay=dec, momentum=mom, nesterov=True)
        l.append(sgd)

    return l,par






def base_model(sgd):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))



# Train model

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def train(sgd):
    cnn_n = base_model(sgd)
    cnn_n.summary()

    # Fit model

    cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)

    #pyplot
    # Plots for training and testing process: loss and accuracy

    plt.figure(0)
    plt.plot(cnn.history['acc'],'r')
    plt.plot(cnn.history['val_acc'],'g')
    plt.xticks(np.arange(0, 101, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])


    plt.figure(1)
    plt.plot(cnn.history['loss'],'r')
    plt.plot(cnn.history['val_loss'],'g')
    plt.xticks(np.arange(0, 101, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])
    return cnn_n


    plt.show()




def run_optimizer(n,iters):
    current_pop,current_pars=generate_custompop(n)
    score_list=[]
    print("Iteration 1")
    for i in range(n):
        print("Population "+str(i+1))
        #evaluate
        cnn_n=train(current_pop[i])
        scores = cnn_n.evaluate(x_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        score_list.append(scores[1]*100)

    m=max(score_list)
    ind=score_list.index(m)
    for ite in range(iters-1):
        print("Iteration "+str(ite+1))
        current_pop,current_pars=get_next_generation(current_pars[ind],n)
        score_list=[]
        for i in range(n):
            #evaluate
            print("Population "+str(i+1))
            cnn_n=train(current_pop[i])
            scores = cnn_n.evaluate(x_test, y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1]*100))
            score_list.append(scores[1]*100)

        m=max(score_list)
        ind=score_list.index(m)

    print('\n.\n.\n.\n.\nGetting best model')
    cnn_n=train(current_pop[ind])
    sgd = SGD(lr = current_pars[ind][0], decay=current_pars[ind][1], momentum=current_pars[ind][0], nesterov=True)

    #save and retrieve
    with open("/home/aswin/proj/assets/model_num.txt",'r') as f:
        model_num=int(f.read())
    model_json = cnn_n.to_json()
    with open("/home/aswin/proj/assets/models/model"+str(model_num)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn_n.save_weights("/home/aswin/proj/assets/model"+str(model_num)+".h5")
    print("Saved model to disk")

    # load json and create model
    json_file = open("/home/aswin/proj/assets/models/model"+str(model_num)+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/aswin/proj/assets/model"+str(model_num)+".h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    with open("/home/aswin/proj/assets/model_num.txt",'r+') as f:
        f.truncate(0)
        f.write(str(model_num+1))

run_optimizer(3,2)

##
# sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.5, nesterov=True)
# #save and retrieve
# cnn_n=train(sgd)
# scores = cnn_n.evaluate(x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# with open("/home/aswin/proj/assets/model_num.txt",'r') as f:
#     model_num=int(f.read())
# model_json = cnn_n.to_json()
# with open("/home/aswin/proj/assets/models/model"+str(model_num)+".json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# cnn_n.save_weights("/home/aswin/proj/assets/model"+str(model_num)+".h5")
# print("Saved model to disk")
#
# # load json and create model
# json_file = open("/home/aswin/proj/assets/models/model"+str(model_num)+".json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("/home/aswin/proj/assets/model"+str(model_num)+".h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# score = loaded_model.evaluate(x_test, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#
# with open("/home/aswin/proj/assets/model_num.txt",'r+') as f:
#     f.truncate(0)
#     f.write(str(model_num+1))
