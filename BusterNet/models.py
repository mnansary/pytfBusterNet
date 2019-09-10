# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input,Lambda,BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K 
import tensorflow as tf 
#--------------------------------------------------------------------------------------
def inception_bn(X, nb_filters=16, kernel_sizes=[(1,1), (3,3), (5,5)]) :
    CXs = []
    for kernel_size in kernel_sizes :
        CX = Conv2D( nb_filters,kernel_size, activation='linear', padding='same')(X)
        CXs.append(CX)
    if (len(CXs)>1):
        X = Concatenate( axis=-1)(CXs)
    else :
        X = CXs[0]
    X= BatchNormalization()(X)
    X= Activation('relu')(X)
    return X

def lambda_fcn(X):
    _,nb_rows,nb_cols,_=K.int_shape(X)
    return tf.image.resize_images(X,tf.constant([nb_rows*2,nb_cols*2],dtype = tf.int32),align_corners=True)
def lambda_out(in_shape):
    return tuple([in_shape[0],in_shape[1]*2,in_shape[2]*2,in_shape[3]])
#-------------------------------------------------------------------------------------------
def std_norm_Channel(X) :
    mean = K.mean(X, axis=-1, keepdims=True)
    std = K.maximum(1e-4, K.std(X, axis=-1, keepdims=True))
    norm=(X - mean) / std
    return norm

def corrPercPool_fcn(X):
    nb_pool=100
    nb_data,nb_row,nb_col,nb_channel=K.int_shape(X)
    rc=nb_col*nb_row
    X_3d = K.reshape(X,tf.stack([-1,rc,nb_channel]))
    X_corr=K.reshape(tf.matmul(X_3d,X_3d,transpose_b=True)/nb_channel,tf.stack([-1,nb_row,nb_col,rc]))
    
    ranks=K.cast(K.round(tf.linspace( start=1.0, stop=rc - 1,num=nb_pool)),'int32') # !!!!!!!!!!!!
    X_s,_=tf.nn.top_k(X_corr,rc,sorted = True)
    X_pool=K.permute_dimensions(tf.gather(K.permute_dimensions(X_s,(3,0,1,2)),ranks),(1,2,3,0)) # !!!!!!!!! 
    return X_pool
def cpp_out(in_shape):
    nb_data,nb_row,nb_col,_=in_shape
    nb_pool=100
    return tuple([nb_data,nb_row,nb_col,nb_pool])
#-----------------------------------------------------------------------------------------------
def man_net(img_dim=256,nb_channels=3):
    nb_filters=[64,64,128,128,256,256,256,512,512,512]
    pool_idx=[1,3,6,9]
    nb_f=[i for i in range(8,0,-2)]
    
    img_shape=(img_dim,img_dim,nb_channels)
    IN=Input(shape=img_shape)
    for i in range(len(nb_filters)):
        if i==0:
            X_prev=IN
        X = Conv2D(nb_filters[i], (3, 3), activation='relu', padding='same')(X_prev)
        if i in pool_idx:
            X= MaxPooling2D((2, 2), strides=(2, 2))(X)
        X_prev=X
    
    for i in range(len(nb_f)):
        X=inception_bn(X,nb_f[i])
        X=Lambda(lambda_fcn,output_shape=lambda_out)(X)
    
    X= inception_bn(X,nb_filters=2,kernel_sizes=[(5,5),(7,7),(11,11)])
    X = Conv2D(1, (3,3), activation='sigmoid', padding='same')(X)
    model = Model(inputs=IN, outputs=X)
    return model
#-------------------------------------------------------------------------------------------------------
def sim_net(img_dim=256,nb_channels=3):
    nb_filters=[64,64,128,128,256,256,256,512,512,512]
    pool_idx=[1,3,6,9]
    nb_f=[6,4,2]
    img_shape=(img_dim,img_dim,nb_channels)
    IN=Input(shape=img_shape)
    for i in range(len(nb_filters)):
        if i==0:
            X_prev=IN
        X = Conv2D(nb_filters[i], (3, 3), activation='relu', padding='same')(X_prev)
        if i in pool_idx:
            X= MaxPooling2D((2, 2), strides=(2, 2))(X)
        X_prev=X
    
    X = Activation(std_norm_Channel)(X)
    X = Lambda(corrPercPool_fcn,output_shape=cpp_out)(X)
    X = BatchNormalization()(X)
    Xb  = inception_bn(X, 8)
    Xa=Lambda(lambda_fcn,output_shape=lambda_out)(X)
    for i in range(len(nb_f)):    
        Xb  = inception_bn(Xa, nb_f[i])
        Xa=Lambda(lambda_fcn,output_shape=lambda_out)(Xa)
        Xb=Lambda(lambda_fcn,output_shape=lambda_out)(Xb)
        X=Concatenate()([Xa,Xb])
    Xb =inception_bn(X,2)
    X=Concatenate()([Xa,Xb])
    X= inception_bn(X,nb_filters=2,kernel_sizes=[(5,5),(7,7),(11,11)])
    X = Conv2D(1, (3,3), activation='sigmoid', padding='same')(X)
    model = Model(inputs=IN, outputs=X)
    return model

if __name__=='__main__':
    model=sim_net()
    model.summary()