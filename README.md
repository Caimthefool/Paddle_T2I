# Paddle_T2I
Generative Adversarial Text to Image Synthesis 论文复现
# Dataset
we use the [Oxford-102](https://drive.google.com/open?id=1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8) datatset that provided in [text-to-image-synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis).
Download the data to ```Data\```.The dataset is convered to hd5 format for faster reading.  
if you want to convert the dataset yourself  
# Training
```
python main.py
```
# Test
```
python main.py --validation
```
# Log
```
visualdl --logdir Log --port 8080
```
# Results
Dataset | Paddle_T2I | Text_to_Image_Synthesis
:------:|:----------:|:------------------------:|
[Oxford-102]|<img src="examples/paddle_T2I_64images.png" height = "300" width="300"/><br/>|<img src="examples/Text_to_Image_Synthesis_64_images.png" height = "300" width="300"/><br/>|
