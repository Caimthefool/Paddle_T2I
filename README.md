# Paddle_T2I
Generative Adversarial Text to Image Synthesis 论文复现
## 一、简介
本项目基于paddlepaddle框架复现T2I_GAN，T2I_GAN是第一个用于文本到图像合成任务的条件式GAN。给定一句文本描述，该模型能够理解文本的含义，合成出符合语义的图像
**论文:**
- [1] Reed S, Akata Z, Yan X, et al. Generative adversarial text to image synthesis[C]//International Conference on Machine Learning. PMLR, 2016: 1060-1069.
**参考项目：**
- [https://github.com/aelnouby/Text-to-Image-Synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis)
## 二、复现精度
本项目验收标准为Oxford-102数据集上人眼评估生成的图像，因此无具体定量指标，只展示合成的样例
Dataset | Paddle_T2I | Text_to_Image_Synthesis
:------:|:----------:|:------------------------:|
[Oxford-102]|<img src="examples/paddle_T2I_64images.png" height = "300" width="300"/><br/>|<img src="examples/Text_to_Image_Synthesis_64_images.png" height = "300" width="300"/><br/>|
## 三、数据集
[Oxford-102花文本图像数据集](https://drive.google.com/open?id=1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8)
这个数据集是由 [text-to-image-synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis)项目提供的。为了更快地进行读取，数据集被转换成了hd5格式。数据集下载下来后保存在： ```Data\```   
如果想要自行转换数据格式，可按照如下步骤操作（实际上就是把数据存储的格式改变了而已，数据本身的信息没有变动，没有经过神经网络进行特征提取）：  
- 下载数据集：[flowers](https://drive.google.com/open?id=0B0ywwgffWnLLcms2WWJQRFNSWXM)（谷歌云盘）
- 将数据集的路径添加到```config.yaml```文件中
- 运行```convert_flowers_to_hd5_script.py```从而转换数据集存储格式
### 数据组织格式
整个数据集下有三个子集，分别是"train"、"valid"、"test".  
每个子集中包含5类数据(注：文本嵌入向量是由论文作者本人在[icml2016](https://github.com/reedscot/icml2016)提供的,已经由字符串形式转换成了向量形式，这部分数据包含在上面下载的数据集中)
- 文件名```name```
- 图像数据```img```
- 文本嵌入向量```embeddings```
- 图像所属的花的类别```class```
- 图像对应的字符串文本```txt```

### 数据集大小：
  - 训练集+验证集：8192张
  - 测试集：800张
  - 每张图像对应的文本数:5句
  - 数据格式：花卉图像以及图像对应的文本数据集
## 四、环境依赖
- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.0.0
## 五、快速开始
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
将模型的参数保存在```model\```中，然后改变pretrain_model的值，再运行以下命令，输出图片保存在```image\```目录中
```
python main.py --validation --split=2 --pretrain_model=model/netG.pdparams
```
### 使用预训练模型预测

将需要测试的文件放在参数pretrain_model确定的目录下，运行下面指令，输出图片保存在```image\```目录中
```
python main.py --validation --split=2 --pretrain_model=model/netG.pdparams
```
## 六、代码结构与详细说明

### 6.1 代码结构
因为本项目的验收是通过人眼观察图像，即user_study，因此评估脚本跟预测是同一个方式

```
├─config                                                # 配置
├─dataset                                               # 数据集加载
├─models                                                # 模型
├─results                                               # 可视化结果
├─utils                                                 # 工具代码
│  convert_flowers_to_hd5_script.py                     # 将数据集转换成hd5格式
│  README.md                                            # 英文readme
│  README_cn.md                                         # 中文readme
│  requirement.txt                                      # 依赖
│  trainer.py                                           # 训练器
|  main.py                                              # 主程序入口
```

### 6.2 参数说明

可以在 `main.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  -------  |  ----  |  ----  |  ----  |
| config| None, 必选| 配置文件路径 ||
| --split| 0, 必选 | 使用的数据集分割 |0代表训练集，1代表验证集，2代表测试集|
| --validation| false, 可选 | 进行预测和评估 ||
| --pretrain_model| None, 可选 | 预训练模型路径 ||
### 6.3 训练
```bash
python main.py --split=0
```
#### 训练输出
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。
```text
Epoch: [1 | 600]
(1/78) Loss_D: 1.247 | Loss_G: 20.456 | D(X): 0.673 | D(G(X)): 0.415
```
### 6.4 评估和预测流程
我们的预训练模型已经包含在了这个repo中，就在model目录下
```bash
python main.py --validation --split=2 --pretrain_model=model/netG.pdparams
```
## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 曾威远|
| 时间 | 2021.07 |
| 框架版本 | Paddle 2.0.2 |
| 应用场景 | 文本到图像的合成 |
| 支持硬件 | GPU、CPU |
# Log
```
visualdl --logdir Log --port 8080
```
# Results
Dataset | Paddle_T2I | Text_to_Image_Synthesis
:------:|:----------:|:------------------------:|
[Oxford-102]|<img src="examples/paddle_T2I_64images.png" height = "300" width="300"/><br/>|<img src="examples/Text_to_Image_Synthesis_64_images.png" height = "300" width="300"/><br/>|
