# UNet
Reimplementation of UNet for Image Segmentation Course

## Details on Project
A review on UNet and how well it performs without any data augmentation (i.e. only how well the architecture will perform)

## Directory Structure
```
chkpt/
    fss1000-epoch100.pt
    voc-epoch100.pt
data/
    VOCdevkit/
        ...
fss1000plots/
vocplots/

.gitignore
fss1000.py
fssaccuracy.py
README.md
train.py
unet.py
visualisefss.py
visualisevoc.py
voc.py
vocaccuracy.py
```

## Models
Pretrained models are available at https://drive.google.com/drive/folders/165oVabNzCnjuY0htpUxQK7JFU4H9nkIV?usp=sharing

## Training
To train new model
```
$ python train.py
```
Parameters can be set in the file to toggle between different datasets

## Evaluation and Visualisation
To evalute models
```
$ python vocaccuracy.py
$ python fssaccuracy.py
```

To visualise segmentation
```
$ python visualisevoc.py
$ python visualisefss.py
```
Visualisation can be toggled to see the best segmentations and the worst