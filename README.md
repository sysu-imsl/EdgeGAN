# EdgeGAN
### [Project Page](https://sysu-imsl.com/EdgeGAN/) | [Paper](https://arxiv.org/abs/2003.02683)
SketchyCOCO: Image Generation from Freehand Scene Sketches  
Chengying Gao, Qi Liu, Qi Xu, Limin Wang, Jianzhuang Liu, Changqing Zou

# Installation
Clone this repo.  
```
git@github.com:sysu-imsl/EdgeGAN.git
cd EdgeGAN
```
This repo requires TensorFlow 1.14.0 and python 3+.  
`conda create/activate` is suggested to manage multiple versions of tensorflow.  
After switching to proper conda environment, run `conda install --file requirements.txt`

# Dataset
Our dataset can be found in [SketchyCOCO](https://github.com/sysu-imsl/SketchyCOCO). Follow the guide and prepare the dataset.

## Directory Structure
For singleclass dataset
```
EdegGAN
└───data
    └───train
    |   |    <file00>.png
    |   |    <file01>.png
    |   |    ...
    |   
    └───test
        |    <file00>.png
        |    <file01>.png
        |    ...
```
For multiclass dataset

```
EdegGAN
└───data
    └───train
    |   └───<0>
    |   |    |    <file01>.png
    |   |    |    ...
    |   └───<1>
    |   |   ...
    |   
    └───test
    |   └───<0>
    |   |    |    <file01>.png
    |   |    |    ...
    |   └───<1>
    |   |   ...
```
For our pretrained model, the class label 0 to 13 correspond to "airplane, cat, giraffe, zebra, dog, elephant, fire hydrant, horse, bicycle, car, traffic light, cow, motorcycle, sheep". Please prepare the input as similar as the examples of training images and test images in [images/dataset_example](https://github.com/sysu-imsl/EdgeGAN/tree/master/images/dataset_example).

## Example
### Train
![60975.png](images/dataset_example/train/60975.png?raw=true)
![60981.png](images/dataset_example/train/60981.png?raw=true)
![60987.png](images/dataset_example/train/60987.png?raw=true)
![60991.png](images/dataset_example/train/60991.png?raw=true)
![60994.png](images/dataset_example/train/60994.png?raw=true)
### Test
![14809.png](images/dataset_example/test/14809.png?raw=true)
![14810.png](images/dataset_example/test/14810.png?raw=true)
![14811.png](images/dataset_example/test/14811.png?raw=true)
![14812.png](images/dataset_example/test/14812.png?raw=true)

# Testing
1. Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1ilxx_mLKaiMRhwzzcrXjIaNlsmfqR6MT/view?usp=sharing) trained with 14 classes, and run:
``` bash
mkdir -p outputs/edgegan
cd outputs/edgegan
cp <checkpoints download path> .
unzip checkpoints.zip
cd ../..
```
2. Generate images with models:
``` bash
python -m edgegan.test --name=edgegan --dataroot=<root of dataset> --dataset=<dataset> --gpu=<gpuid> #(model trained with multi-classes)
python -m edgegan.test --name=[model_name] --dataroot=<root of dataset> --dataset=<dataset> --nomulticlasses --gpu=<gpuid> #(model trained with single class)
```
3. the outputs will be located at `outputs/edgegan/test_output/` by default

# Training
It will cost about fifteen hours to run on a single Nvidia RTX 2080 Ti card.

``` bash
python -m edgegan.train --name=<new_name> --dataroot=<root of dataset> --dataset=<datsaet_name> --gpu=<gpuid> #(with multi-classes)
python -m edgegan.train --name=<new_name> --dataroot=<root of dataset> --dataset=<datsaet_name> --nomulticlasses --gpu=<gpuid> #(with single class)

```

# Citation
If you use this code for your research, please cite our papers.  
```
@inproceedings{gao2020sketchycoco,
  title={SketchyCOCO: Image Generation From Freehand Scene Sketches},
  author={Gao, Chengying and Liu, Qi and Xu, Qi and Wang, Limin and Liu, Jianzhuang and Zou, Changqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5174--5183},
  year={2020}
}
```
