# Visual Saliency Transformer (VST)

source code for our ICCV 2021 paper “Visual Saliency Transformer” by Nian Liu, Ni Zhang, Kaiyuan Wan, Junwei Han, and Ling Shao.

created by Ni Zhang, email: nnizhang.1995@gmail.com

![avatar](https://github.com/nnizhang/VST/blob/main/Network.png)

## Requirement
1. Pytorch 1.6.0
2. Torchvison 0.7.0

## RGB VST for RGB salient object detection
### Data Preparation
We use the training set of [DUTS](http://saliencydetection.net/duts/) to train our VST. Besides, we follow [Egnet](https://github.com/JXingZhao/EGNet) to generate contour maps of DUTS trainset for training. You can directly download the generated contour maps (DUTS-TR-Contour) from [[baidu pan](https://pan.baidu.com/s/17OnUi09YuOOq23xNrdYCLQ) fetch code: ow76 | [Google drive]()] and put it into Data directory.

Your `Data` directory should look like this:

````
-- Data
   |-- DUTS
   |   |-- DUTS-TR
   |   |-- | DUTS-TR-Image
   |   |-- | DUTS-TR-Mask
   |   |-- | DUTS-TR-Contour
   |   |-- DUTS-TE
   |   |-- | DUTS-TE-Image
   |   |-- | DUTS-TE-Mask
````



## RGB-D VST for RGB-D salient object detection
Coming soon...
