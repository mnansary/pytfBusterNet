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
import h5py

## section ----
def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',data=data)
    hf.close()

def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

def get_prediction(model,path):
    img=load_img(path)
    arr=img_to_array(img)
    tensor = np.expand_dims(arr,axis=0)
    pred = model.predict(tensor)[0]
    return pred

def plot_data(img,gt,pred) :
    plt.figure('BusterNet')
    plt.subplot(131)
    plt.imshow(img)
    plt.title('input image')
    plt.subplot(132)
    plt.title('ground truth')
    plt.imshow(gt)
    plt.subplot(133)
    plt.imshow(pred)
    plt.title('busterNet pred')
    plt.show()
    plt.clf()
    plt.close()

def plot_fusion_data(Xm,Xs,gt,pred) :
    plt.figure('BusterNet')
    
    plt.subplot(221)
    plt.imshow(Xm)
    plt.title('input image from man-net')
    
    plt.subplot(222)
    plt.title('input image from sim-net')
    plt.imshow(Xs)

    plt.subplot(223)
    plt.imshow(gt)
    plt.title('Ground Truth')

    plt.subplot(224)
    plt.imshow(pred)
    plt.title('Fusion Net Prediction')
    
    plt.show()
    plt.clf()
    plt.close()

## ----- End
#----------------------------------------------------------------------------------------
def gen_man_predictions(dset_dir):
    
    # load model
    model=man_net()
    model.load_weights(os.path.join(os.getcwd(),'model_weights','man_net.h5'))

    # load data
    imgs=readh5(os.path.join(dset_dir,'X.h5'))
    # generate predictions
    preds =[model.predict(np.expand_dims(tensor,axis=0)) for tensor in imgs]
    # save predictions
    saveh5(os.path.join(dset_dir,'Xm.h5'),np.vstack(preds))

def visualize_man_predictions(dset_dir,num_viz=10):
    imgs=readh5(os.path.join(dset_dir,'X.h5'))
    gts=readh5(os.path.join(dset_dir,'Ym.h5'))
    preds=readh5(os.path.join(dset_dir,'Xm.h5'))
    
    for i in range(num_viz):
        img=imgs[i]
        
        gt=gts[i]
        gt=np.squeeze(gt)
        
        pred=preds[i]
        pd=np.squeeze(pred)

        plot_data(img,gt,pd)    

#----------------------------------------------------------------------------------------------------------------        
#----------------------------------------------------------------------------------------
def gen_sim_predictions(dset_dir):
    
    # load model
    model=sim_net()
    model.load_weights(os.path.join(os.getcwd(),'model_weights','sim_net.h5'))

    # load data
    imgs=readh5(os.path.join(dset_dir,'X.h5'))
    # generate predictions
    preds =[model.predict(np.expand_dims(tensor,axis=0)) for tensor in imgs]
    # save predictions
    saveh5(os.path.join(dset_dir,'Xs.h5'),np.vstack(preds))

def visualize_sim_predictions(dset_dir,num_viz=10):
    imgs=readh5(os.path.join(dset_dir,'X.h5'))
    gts=readh5(os.path.join(dset_dir,'Ys.h5'))
    preds=readh5(os.path.join(dset_dir,'Xs.h5'))
    
    for i in range(num_viz):
        img=imgs[i]
        
        gt=gts[i]
        gt=np.squeeze(gt)
        
        pred=preds[i]
        pd=np.squeeze(pred)

        plot_data(img,gt,pd)    

#----------------------------------------------------------------------------------------------------------------        
#----------------------------------------------------------------------------------------
def gen_fusion_predictions(dset_dir):
    
    # load model
    model=fusion_net()
    model.load_weights(os.path.join(os.getcwd(),'model_weights','fusion_net.h5'))

    # load data
    Xss=readh5(os.path.join(dset_dir,'Xs.h5'))
    Xms=readh5(os.path.join(dset_dir,'Xm.h5'))
    
    preds=[]
    # generate predictions
    for i in range(Xss.shape[0]):
        pred =model.predict([np.expand_dims(Xss[i],axis=0),np.expand_dims(Xms[i],axis=0)])
        print('Num of preds: {}'.format(i))
        pred=np.expand_dims(pred,axis=0)
        preds.append(pred)
        print(pred.shape)
    preds=np.vstack(preds)
    # save predictions
    saveh5(os.path.join(dset_dir,'YF.h5'),np.vstack(preds))

def visualize_fusion_predictions(dset_dir,num_viz=10):
    Xss=readh5(os.path.join(dset_dir,'Xs.h5'))
    Xms=readh5(os.path.join(dset_dir,'Xm.h5'))
    gts=readh5(os.path.join(dset_dir,'Y.h5'))
    preds=readh5(os.path.join(dset_dir,'YF.h5'))
    print(preds.shape)
    for i in range(num_viz):
        Xs=Xss[i]
        Xs=np.squeeze(Xs)
        
        Xm=Xms[i]
        Xm=np.squeeze(Xm)
        
        gt=gts[i]
        
        pred=preds[i]
        
        plot_fusion_data(Xm,Xs,gt,pred)   

#----------------------------------------------------------------------------------------------------------------        

    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dataset Predictions')
    parser.add_argument("dset_dir", help="/path/to/MICC-F2000/preprocessed/DataSet/Folder/")
    args = parser.parse_args()
    dset_dir=args.dset_dir
    #gen_man_predictions(dset_dir)
    #visualize_man_predictions(dset_dir)
    #gen_sim_predictions(dset_dir)
    #visualize_sim_predictions(dset_dir)
    #gen_fusion_predictions(dset_dir)
    visualize_fusion_predictions(dset_dir)