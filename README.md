# Visual Saliency Transformer (VST)

source code for our ICCV 2021 paper “Visual Saliency Transformer” by Nian Liu, Ni Zhang, Kaiyuan Wan, Junwei Han, and Ling Shao.

created by Ni Zhang, email: nnizhang.1995@gmail.com

![avatar](https://github.com/nnizhang/VST/blob/main/Network.png)

## Requirement
1. Pytorch 1.6.0
2. Torchvison 0.7.0

## RGB VST for RGB Salient Object Detection
### Data Preparation
#### Train Set
We use the training set of [DUTS](http://saliencydetection.net/duts/) to train our VST for RGB SOD. Besides, we follow [Egnet](https://github.com/JXingZhao/EGNet) to generate contour maps of DUTS trainset for training. You can directly download the generated contour maps (DUTS-TR-Contour) from [[baidu pan](https://pan.baidu.com/s/17OnUi09YuOOq23xNrdYCLQ) fetch code: ow76 | [Google drive](https://drive.google.com/file/d/1NizY8WZSz-5i5KV7bATODi76fovrLuVf/view?usp=sharing)] and put it into `Data` folder.

Your `Data` folder should look like this:

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
#### Test Set
We use the testing set of [DUTS](http://saliencydetection.net/duts/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://cbi.gatech.edu/salobj/), [DUT-O](http://saliencydetection.net/dut-omron/), and [SOD](http://elderlab.yorku.ca/SOD.) to test our VST.

### Training, Testing, and Evaluation
1. Download the pretrained T2T-ViT_t-14 model [[baidu pan]() fetch code:  | [Google drive]()].


### Testing on Our Pretrained RGB VST Model
1. Download our pretrained `RGB_VST.pth` and then put it in `Models/` folder.

Our saliency maps can be download from [[baidu pan](https://pan.baidu.com/s/1CDkCjq9fRvOHLou9S9oGiA) fetch code: 92t0 | [Google drive](https://drive.google.com/file/d/1T4zDvBobQdT7L7i0HijOZSMfTS5hK-Ec/view?usp=sharing)].

Coming soon...



## RGB-D VST for RGB-D Salient Object Detection
### Data Preparation
#### Train Set
We use 1,485 images from NJUD, 700 images from NLPR, and 800 images from DUTLF-Depth to train our VST for RGB-D SOD. Besides, we follow [Egnet](https://github.com/JXingZhao/EGNet) to generate corresponding contour maps for training. You can directly download the whole training set from here [[baidu pan]() fetch code:  | [Google drive]()] and put it into `Data` folder.


#### Test Set
We use the testing set of NJUD, NLPR, and DUTLF-Depth and [STERE](http://dpfan.net/d3netbenchmark/), [LFSD](http://dpfan.net/d3netbenchmark/), [RGBD135](http://dpfan.net/d3netbenchmark/), [SSD](http://dpfan.net/d3netbenchmark/), [SIP](http://dpfan.net/d3netbenchmark/), and [ReDWeb-S](https://github.com/nnizhang/SMAC) to test our VST. 

### Training, Testing, and Evaluation
1. Download the pretrained T2T-ViT_t-14 model [[baidu pan]() fetch code:  | [Google drive]()].

### Testing on Our Pretrained RGB-D VST Model

Our saliency maps can be download from [[baidu pan](https://pan.baidu.com/s/1yPo9C-WrBXiN8WXNEOP4Hg) fetch code: jovk | [Google drive](https://drive.google.com/file/d/1ccpQv6dnZbC-hx9pZjNTTI-_5qm8QLm9/view?usp=sharing)].

Coming soon...

## Acknowledgement
We thank the authors of [Egnet](https://github.com/JXingZhao/EGNet) for providing codes of generating contour maps. We also thank [Zhao Zhang](https://github.com/zzhanghub/eval-co-sod) for providing the efficient evaluation tool.

## Citation
If you think our work is helpful, please cite 
```
@inproceedings{liu2021VST, 
  title={Visual Saliency Transformer}, 
  author={Liu, Nian and Zhang, Ni and Han, Junwei and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```



