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

The dataset is preprocessed such that there are 3 types of ground truths to train **the three stage strategy** BusterNet.As an example:  
![](/info/img.png?raw=true)
![](/info/man.png?raw=true)
![](/info/sim.png?raw=true)
![](/info/gt.png?raw=true)

