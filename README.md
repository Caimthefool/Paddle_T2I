# Paddle_T2I
this is a paddle repo of Generative Adversarial Text to Image Synthesis

English | [简体中文](./README_cn.md)
   
   * [Paddle_T2I]
      * [1 Introduction](#1-introduction)
      * [2 Accuracy](#2-accuracy)
      * [3 Dataset](#3-dataset)
         * [Data Organization Format](#data-organization-format)
         * [Dataset size](#dataset-size)
      * [4 Environment](#4-environment)
      * [5 Quick start](#5-quick-start)
         * [step1:clone](#step1clone)
         * [step2:Training](#step2Training)
         * [step3:Test](#step3Test)
         * [Prediction using pre training model](#prediction-using-pre-training-model)
      * [6 Code structure](#6-code-structure)
         * [6.1 structure](#61-structure)
         * [6.2 Parameter description](#62-parameter-description)
         * [6.3 Training](#63-training)
            * [Training output](#training-output)
         * [6.4 Evaluation and Test](#64-evaluation-and-test)
      * [7 Model information](#7-model-information)
## 1 Introduction
This project replicates T2I_GAN, the first conditional GAN for text-to-image synthesis tasks, based on the paddlepaddle framework. given a text description, the model is able to understand the meaning of the text and synthesize a semantic image 

**Paper:**
- [1] Reed S, Akata Z, Yan X, et al. Generative adversarial text to image synthesis[C]//International Conference on Machine Learning. PMLR, 2016: 1060-1069. 

**Reference project：**
- [https://github.com/aelnouby/Text-to-Image-Synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis)
## 2 Accuracy
The acceptance criteria for this project is to evaluate the images generated on the Oxford-102 dataset with the human eye, so there are no specific quantitative metrics and only synthetic samples are shown
Dataset | Paddle_T2I | Text_to_Image_Synthesis
:------:|:----------:|:------------------------:|
[Oxford-102]|<img src="examples/paddle_T2I_64images.png" height = "300" width="300"/><br/>|<img src="examples/Text_to_Image_Synthesis_64_images.png" height = "300" width="300"/><br/>|
## 3 Dataset
[Oxford-102](https://drive.google.com/open?id=1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8)
This dataset was provided by [text-to-image-synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis).The dataset has been converted to hd5 format for faster reading.The datasets are downloaded and saved in: ```Data\```   
If you want to convert the data format yourself, you can follow the steps below (actually the format of the data storage is changed, the information of the data itself remains unchanged and no feature extraction is performed by the neural network). 
- Download dataset:[flowers](https://drive.google.com/open?id=0B0ywwgffWnLLcms2WWJQRFNSWXM)（谷歌云盘）
- Add the path to the dataset to ```config.yaml```
- Run ```convert_flowers_to_hd5_script.py``` to convert the dataset storage format
### Data Organization Format
There are three subsets under the whole dataset, namely "train", "valid" and "test". Each subset contains 5 types of data (Note: the text embedding vector is provided by the author of the paper, which has been converted from string form to vector form, and this part of data is included in the above downloaded dataset)
- File Name```name```
- Image```img```
- Text embeddings```embeddings```
- class of image```class```
- text description of image```txt```

### Dataset size：
  - Train+Validation：8192
  - Test：800
  - Number of text per image:5
  - Data format: flower images and the corresponding text data set of the images
## 4 Environment
- Hardware：GPU、CPU

- Framwork：
  - PaddlePaddle >= 2.0.0
## 5 Quick start
### step1:clone
```bash
git clone https://github.com/Caimthefool/Paddle_T2I.git
cd Paddle_T2I
```
### step2:Training
```
python main.py --split=0
```
### step3:Test
Save the parameters of the model in ```model\```, then change the value of pretrain_model, and then run the following command to save the output image in the ```image\``` directory
```
python main.py --validation --split=2 --pretrain_model=model/netG.pdparams
```
### Prediction using pre training model

Place the files to be tested in the directory determined by the parameter pretrain_model, run the following command, and save the output images in the ```image\``` directory
```
python main.py --validation --split=2 --pretrain_model=model/netG.pdparams
```
## 6 Code structure

### 6.1 structure
Because the acceptance of this project is through the human eye to observe the image, i.e. user_study, the evaluation is the same way as the prediction

```
├─config                                                
├─dataset                                               
├─models                                                
├─results                                               
├─utils                                                 
│  convert_flowers_to_hd5_script.py                     
│  README.md                                            
│  README_cn.md                                         
│  requirement.txt                                      
│  trainer.py                                           
|  main.py                                              
```

### 6.2 Parameter description

Training and evaluation-related parameters can be set in `main.py`, as follows.

|  参数   | 默认值  | 说明 | 其他 |
|  -------  |  ----  |  ----  |  ----  |
| config| None, Mandatory| Configuration file path ||
| --split| 0, Mandatory | Data set splitused |0 represents the training set, 1 represents the validation set, 2 represents the test set|
| --validation| false, Optional | Prediction and evaluation ||
| --pretrain_model| None, Optional | Pre-trained model path ||
### 6.3 Training
```bash
python main.py --split=0
```
#### Training output
When training is executed, the output will look like the following. Each round of `batch` training will print the current epoch, step, and loss values.
```text
Epoch: [1 | 600]
(1/78) Loss_D: 1.247 | Loss_G: 20.456 | D(X): 0.673 | D(G(X)): 0.415
```
### 6.4 Evaluation and Test
Our pre-trained model is already included in this repo, in the model directory
```bash
python main.py --validation --split=2 --pretrain_model=model/netG.pdparams
```
## 7 Model information

For other information about the model, please refer to the following table:

| information | description |
| --- | --- |
| Author | weiyuan zeng|
| Date | 2021.09 |
| Framework version | Paddle 2.0.2 |
| Application scenarios | Text-to-Image Synthesis |
| Support hardware | GPU、CPU |
# Log
```
visualdl --logdir Log --port 8080
```
# Results
Dataset | Paddle_T2I | Text_to_Image_Synthesis
:------:|:----------:|:------------------------:|
[Oxford-102]|<img src="examples/paddle_T2I_64images.png" height = "300" width="300"/><br/>|<img src="examples/Text_to_Image_Synthesis_64_images.png" height = "300" width="300"/><br/>|
