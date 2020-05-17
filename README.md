<img src=".github/Detectron2-Logo-Horz.svg" width="300" >
This is an test implementation for PASCAL VOC instance segmentation

## usage

1. be sure the datasets are in datasets/VOC20{07,12}.
2. You can train as original detectron2.

for example
```
python tools/train_net.py --config-file configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2
```
※I'm sorry but results are not good for now (AP: 33%)!!

## result example

![c](https://user-images.githubusercontent.com/39827302/82139181-0c125880-9861-11ea-9e41-99c2c0473248.png)
![b](https://user-images.githubusercontent.com/39827302/82139185-22b8af80-9861-11ea-9dc7-5f3c48f74c9b.png)
