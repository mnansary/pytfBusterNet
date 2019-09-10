#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from keras.preprocessing.image import load_img,img_to_array

import argparse
import os
import numpy as np 
from glob import glob
import random
import cv2
import h5py
import matplotlib.pyplot as plt
import imageio as imgio


class Preprocessor(object):
    def __init__(self,data_dir,save_dir,image_dim=256):
        self.data_dir=data_dir

        self.save_dir=os.path.join(save_dir,'MICC-F2000 Preprocessed')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.img_dir=os.path.join(self.save_dir,'Images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.gt_dir=os.path.join(self.save_dir,'GroundTruths')
        if not os.path.exists(self.gt_dir):
            os.mkdir(self.gt_dir)

        self.man_dir=os.path.join(self.save_dir,'Manipulation')
        if not os.path.exists(self.man_dir):
            os.mkdir(self.man_dir)

        self.sim_dir=os.path.join(self.save_dir,'Similiarity')
        if not os.path.exists(self.sim_dir):
            os.mkdir(self.sim_dir)

        self.tmpl_dir=os.path.join(self.save_dir,'Templates')
        if not os.path.exists(self.tmpl_dir):
            os.mkdir(self.tmpl_dir)
        
        self.image_dim=image_dim
        # Dataset Specific
        self.tamper_iden='tamp'
        self.orig_iden='_scale'
        self.base_tamp_iden='tamp1.jpg'
        self.prb_idens=['P1000231','DSCN47']

    def __renameProblematicFile(self):
        file_to_rename='nikon7_scale.jpg'
        proper_name='nikon_7_scale.jpg'
        try:
            os.rename(os.path.join(self.data_dir,file_to_rename),os.path.join(self.data_dir,proper_name))
        except Exception as e:
            print(colored('!!! An exception occurred while renaming {}'.format(file_to_rename),'red'))
            print(colored(e,'green'))
        

    def listFiles(self):
        self.tampered_files=[]
        self.tamper_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.*'.format(self.tamper_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.tamper_iden)]
            if base_name not in self.prb_idens:
                self.tamper_idens.append(base_name)
                self.tampered_files.append(file_name)
            
        self.original_files=[]
        self.orig_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.orig_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.orig_iden)]
            if base_name not in self.prb_idens:
                self.orig_idens.append(base_name)
                self.original_files.append(file_name)
        


    def __createGroundTruth(self):
        for iden in self.orig_idens:
            #<-Saves the templates
            if iden in self.tamper_idens:
                base_image=self.original_files[int(self.orig_idens.index(iden))]
                img_data=load_img(base_image,color_mode = "grayscale",target_size=(self.image_dim,self.image_dim))    
                idx=[i for i, e in enumerate(self.tamper_idens) if e == iden] 
                for id_s in idx:
                    tmp_file=self.tampered_files[id_s]
                    if self.base_tamp_iden in tmp_file:
                        tamp_data=load_img(tmp_file,color_mode = "grayscale",target_size=(self.image_dim,self.image_dim))
                        back=np.array(tamp_data)-np.array(img_data)
                        back[back!=0]=np.array(tamp_data)[back!=0]
                        idx_box = np.where(back!=0)
                        y,h,x,w = np.min(idx_box[0]), np.max(idx_box[0]), np.min(idx_box[1]), np.max(idx_box[1])
                        template = np.array(tamp_data)[y:h,x:w]
                        self.__saveData(template,'{}_template'.format(iden))
            # Saves the template->
                tmplt_file=os.path.join(self.tmpl_dir,'{}_template.png'.format(iden))
                template_arr=np.array(load_img(tmplt_file,color_mode = "grayscale"))
                w, h = template_arr.shape[::-1]
                for id_s in idx:
                    rand_iden=random.randint(1,100)
                    #save img 
                    tmp_file=self.tampered_files[id_s]
                    tamp_data=load_img(tmp_file,color_mode = "rgb",target_size=(self.image_dim,self.image_dim))
                    tamp_data=np.array(tamp_data)
                    print(colored('\t {}'.format(tmp_file),'red'))
                    self.__saveData(tamp_data,'{}{}_{}'.format(rand_iden,id_s,iden))
                    
                    # create ground truth
                    tamp_data=load_img(tmp_file,color_mode = "grayscale",target_size=(self.image_dim,self.image_dim))               
                    back=np.array(tamp_data)-np.array(img_data)
                    # manipulation data
                    y_m=np.zeros(back.shape)
                    y_m[back!=0]=255
                    self.__saveData(y_m,'{}{}_{}_t_man'.format(rand_iden,id_s,iden))

                    back[back!=0]=50
                    img_arr=np.array(img_data)
                    res = cv2.matchTemplate(img_arr,template_arr,5)
                    _,_,_,top_left = cv2.minMaxLoc(res)
                    back[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]=100
                    
                    # similiarity data
                    y_s=np.zeros(back.shape)
                    y_s[back!=0]=255
                    self.__saveData(y_s,'{}{}_{}_t_sim'.format(rand_iden,id_s,iden))
                    
                    
                    # grayscale to RGB Ground Truth
                    gt_data= np.zeros((self.image_dim,self.image_dim,3), np.uint8)
                    gt_data[:,:]=(0,0,255) # blue back_ground
                    gt_data[back==100]=(0,255,0) # original location
                    gt_data[back==50]=(255,0,0) # tampered location
                    
                    
                    self.__saveData(gt_data,'{}{}_{}_gt'.format(rand_iden,id_s,iden))
                    
        
        
    def __saveData(self,data,identifier):
        if '_gt' in identifier:
            save_dir=self.gt_dir
            p_color='green'
        
        elif '_t_sim' in identifier:
            save_dir=self.sim_dir
            p_color='cyan'    
        
        elif '_t_man' in identifier:
            save_dir=self.man_dir
            p_color='white'    
        
        elif '_template' in identifier:
            save_dir=self.tmpl_dir
            p_color='yellow'    
        else:
            save_dir=self.img_dir
            p_color='blue'
        
        file_name=os.path.join(save_dir,identifier+'.png')
        print(colored('\t # Saving {} at {}'.format(identifier+'.png',save_dir),p_color))
        imgio.imsave(file_name,data)

    def preprocess(self,rename_flag=False):
        if rename_flag:
            self.__renameProblematicFile()

        self.listFiles()
        self.__createGroundTruth()
        
        

def create_image(data_dir,save_dir):
    PreprocessorOBJ=Preprocessor(data_dir,save_dir)
    PreprocessorOBJ.preprocess()

def toTensor(file_path,mode):
    img=load_img(file_path,color_mode = mode,target_size=(256,256))
    arr=img_to_array(img)
    arr=arr.astype('float32')/255
    tensor=np.expand_dims(arr,axis=0)
    return tensor

def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',data=data)
    hf.close()

def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data


def save_dataset(data_dir,save_dir):
    Obj=Preprocessor(data_dir,save_dir)
    X_p=Obj.img_dir
    Y_p=Obj.gt_dir
    Ys_p=Obj.sim_dir
    Ym_p=Obj.man_dir
    #np.vstack(tensor)
    X=[]
    Y=[]
    YS=[]
    YM=[]
    
    for file_name in glob(os.path.join(X_p,'*.png')):
        base_path,_=os.path.splitext(file_name)
        base_name=os.path.basename(base_path)
            
        xp=file_name
        yp=os.path.join(Y_p,'{}_gt.png'.format(base_name))
        ymp=os.path.join(Ym_p,'{}_t_man.png'.format(base_name))
        ysp=os.path.join(Ys_p,'{}_t_sim.png'.format(base_name))
        
        print(colored('IMG:{}'.format(xp),'green'))
        print(colored('GT:{}'.format(yp),'blue'))
        print(colored('SIM:{}'.format(ysp),'red'))
        print(colored('MAN:{}'.format(ymp),'yellow'))
        
        X.append(toTensor(xp,"rgb"))
        Y.append(toTensor(yp,"rgb"))
        YS.append(toTensor(ysp,"grayscale"))
        YM.append(toTensor(ymp,"grayscale"))

    X=np.vstack(X)
    Y=np.vstack(Y)
    YS=np.vstack(YS)
    YM=np.vstack(YM)
    
    dset_dir=os.path.join(Obj.save_dir,'DataSet')
    if not os.path.exists(dset_dir):
        os.mkdir(dset_dir)

    XP_S=os.path.join(dset_dir,'X.h5')
    YP_S=os.path.join(dset_dir,'Y.h5')
    YSP_S=os.path.join(dset_dir,'Ys.h5')
    YMP_S=os.path.join(dset_dir,'Ym.h5')
    
    saveh5(XP_S,X)
    saveh5(YP_S,Y)
    saveh5(YSP_S,YS)
    saveh5(YMP_S,YM)

    return XP_S,YP_S,YSP_S,YMP_S


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MICC_F2000 Dataset preprocessing')
    parser.add_argument("data_dir", help="/path/to/MICC-F2000 Folder")
    parser.add_argument("save_dir", help="/path/to/save/preprocessed/data")
    args = parser.parse_args()
    data_dir=args.data_dir
    save_dir=args.save_dir
    create_image(data_dir,save_dir)
    xp,yp,ysp,ymp=save_dataset(data_dir,save_dir)
    
    x=readh5(xp)
    print('IMG:{}'.format(x.shape))
    
    y=readh5(yp)
    print('GT:{}'.format(y.shape))
    
    ys=readh5(ysp)
    print('SIM:{}'.format(ys.shape))
    
    ym=readh5(ymp)
    print('MAN:{}'.format(ym.shape))