# BusterNet implementation for TPU Training In Colab

    Version: 0.0.2  
    Python : 3.6.8
    Author : Shakir Hossain
             Md. Nazmuddoha Ansary
             Mohammad Bin Monjil
             Habibur Rahman
             Shahriar Prince
             Md Aminul Islam
    
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements
    Keras==2.2.5  
    numpy==1.16.4  
    opencv-python==4.1.1.26  
    tensorflow==1.13.1 
    termcolor==1.1.0  
    Pillow==6.1.0
    imageio==2.5.0
      
* Python == 3.6.8
* pip3 install -r requirements.txt

#  Preprocessing The Data
1. Download [MICC-F2000](http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/) dataset    
2. Unzip MICC-F2000.zip    
*NOTE:The dataset contains a file named: nikon7_scale.jpg. It has to be renamed as nikon_7_scale.jpg.*       
3. Run **preprocess.py** in **scripts** folder 
    
        usage: ./preprocess.py [-h] data_dir save_dir    
        MICC_F2000 Dataset preprocessing    
        positional arguments:    
            data_dir    /path/to/MICC-F2000 Folder    
            save_dir    /path/to/save/preprocessed/data    
        optional arguments:    
            -h, --help  show this help message and exit         

* The total number of tampered images in the dataset is 700. BUT for processing convenience images with identifiers 'P1000231' and 'DSCN47' are avoided for generalization of template matching procedure.
The dataset is preprocessed such that there are 3 types of ground truths.For example:
![](/info/img.jpg?raw=true)
This to train by **the three stage strategy** BusterNet.Sample Ground Truths for **manipulation**, **similiarity** and **fusion** branch respectively:  
![](/info/man.png?raw=true)
![](/info/sim.png?raw=true)
![](/info/gt.png?raw=true)

# BusterNet
The model implementation is based on [BusterNet: Detecting Copy-Move Image Forgery with Source/Target Localization](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_11)
> Authors and Researchers: Yue Wu,Wael Abd-Almageed,Prem Natarajan 
![](/info/net.png?raw=true)

# Colab and TPU(Tensor Processing Unit)
*TPU’s have been recently added to the Google Colab portfolio making it even more attractive for quick-and-dirty machine learning projects when your own local processing units are just not fast enough. While the **Tesla K80** available in Google Colab delivers respectable **1.87 TFlops** and has **12GB RAM**, the **TPUv2** available from within Google Colab comes with a whopping **180 TFlops**, give or take. It also comes with **64 GB** High Bandwidth Memory **(HBM)**.*
[Visit This For More Info](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da)  
**For this model the approx time/epoch=11s**

#### Manipulation Region Detection
![](/info/man_net.png?raw=true)
A sample result from the **manipulation** detection branch:
![](/info/manp.png?raw=true)
#### Similiar Region Detection
![](/info/sim_net.png?raw=true)
A sample result from the **similiarity** detection branch:
![](/info/simp.png?raw=true)
