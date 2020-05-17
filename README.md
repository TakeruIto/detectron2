<img src=".github/Detectron2-Logo-Horz.svg" width="300" >
This is an test implementation for PASCAL VOC instance segmentation

## usage

1. be sure the datasets are in datasets/VOC20{07,12}.
2. You can train as original detectron2.

for example
```
python tools/train_net.py --config-file configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml  
```
※I'm sorry but results are not good for now (AP: 33%)!!

## result example
