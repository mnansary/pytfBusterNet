# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


from models import man_net,sim_net,fusion_net
import numpy as np 
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import img_to_array, load_img

def plot_buster_data(img,Xs,Xm,Yp) :
    plt.figure('BusterNet')
    
    plt.subplot(221)
    plt.imshow(img)
    plt.title(' Forged image ')
    
    plt.subplot(222)
    plt.title('Similiarity Prediction')
    plt.imshow(Xs)

    plt.subplot(223)
    plt.title('Manipulation Prediction')
    plt.imshow(Xm)
    
    plt.subplot(224)
    plt.title('Fusion Net Prediction')
    plt.imshow(Yp)
    
    plt.show()
    plt.clf()
    plt.close()



def see_prediction(img_path):
    # models
    man=man_net()
    sim=sim_net()
    fusion=fusion_net()
    # load weights
    man.load_weights(os.path.join(os.getcwd(),'model_weights','man_net.h5'))
    sim.load_weights(os.path.join(os.getcwd(),'model_weights','sim_net.h5'))
    fusion.load_weights(os.path.join(os.getcwd(),'model_weights','fusion_net.h5'))
    
    img=load_img(img_path,color_mode="rgb",target_size=(256,256)) 

    arr=img_to_array(img) # plot
    print('IMG:{}'.format(arr.shape))

    tensor_m = np.expand_dims(arr,axis=0)
    print('MAN_IN:{}'.format(tensor_m.shape))

    man_p=man.predict(tensor_m)[0] # plot
    print('MAN_out:{}'.format(man_p.shape))

    tensor_s = np.expand_dims(arr,axis=0)
    print('SIM_IN:{}'.format(tensor_s.shape))

    sim_p=sim.predict(tensor_s)[0] # plot
    print('SIM_out:{}'.format(sim_p.shape))

    fusion_p=fusion.predict([np.expand_dims(sim_p,axis=0),np.expand_dims(man_p,axis=0)])[0] # plot
    print('FUSION_out: {}'.format(fusion_p.shape))
    
    plot_buster_data(img,np.squeeze(sim_p),np.squeeze(man_p),fusion_p)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Image Predictions')
    parser.add_argument("img_path", help="/path/to/imagefile/ext")
    args = parser.parse_args()
    img_path=args.img_path
    see_prediction(img_path)