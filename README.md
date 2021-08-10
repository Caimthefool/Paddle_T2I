# Paddle_T2I
Generative Adversarial Text to Image Synthesis 论文复现
# Dataset
我们使用的是 [Oxford-102](https://drive.google.com/open?id=1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8) (谷歌云盘)数据集，这个数据集是由 [text-to-image-synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis)项目提供的。为了更快地进行读取，数据集被转换成了hd5格式。数据集下载下来后保存在： ```Data\```   
如果想要自行转换数据格式，可按照如下步骤操作（实际上就是把数据存储的格式改变了而已，数据本身的信息没有变动，没有经过神经网络进行特征提取）：  
- 下载数据集：[flowers](https://drive.google.com/open?id=0B0ywwgffWnLLcms2WWJQRFNSWXM)（谷歌云盘）
- 将数据集的路径添加到```config.yaml```文件中
- 运行```convert_flowers_to_hd5_script.py```从而转换数据集存储格式
## 数据组织格式
整个数据集下有三个子集，分别是"train"、"valid"、"test".  
每个子集中包含5类数据(注：文本嵌入向量是由论文作者本人在[icml2016](https://github.com/reedscot/icml2016)提供的,已经由字符串形式转换成了向量形式，这部分数据包含在上面下载的数据集中)
- 文件名```name```
- 图像数据```img```
- 文本嵌入向量```embeddings```
- 图像所属的花的类别```class```
- 图像对应的字符串文本```txt```
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
