# RetinaFace Face Detector - 

## Introduction
This Project is based on [Insightface-Retinaface](https://github.com/deepinsight/insightface). 
I have imporved some usability. And I focused on the optimization and testing of `ONNX` and `TensorRT`.  

## Start
use master branch
```shell
$ git clone https://github.com/dongle94/retinaface.git
```

## Data
### Dataset
Use [Widerface](http://shuoyang1213.me/WIDERFACE/index.html) dataset
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/index.html) dataset. Only Images.

2. Download Retinaface annotations (face bounding boxes & five facial landmarks).
   - [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA)
   - [gdrive](https://drive.google.com/file/d/1BbXxIiY-F74SumCNG6iwmJJ5K3heoemT/view?usp=sharing)

3. Move datasets directory under `retinface/data/` as follows:
```Shell
data
└── retinaface
    ├── test
    │   ├── images
    │   └── label.txt
    ├── train
    │   ├── images
    │   └── label.txt
    └── val
        ├── images
        └── label.txt
```

### Pretrained Weight for Transfer learning
This Project use `resnet` weight for training. `resnet50`, `resnet152`. Put them into `retinaface/model/`
- ImageNet ResNet50 ([baidu cloud](https://pan.baidu.com/s/1WAkU9ZA_j-OmzO-sdk9whA) and [googledrive](https://drive.google.com/file/d/1ibQOCG4eJyTrlKAJdnioQ3tyGlnbSHjy/view?usp=sharing)).
- ImageNet ResNet152 ([baidu cloud](https://pan.baidu.com/s/1nzQ6CzmdKFzg8bM8ChZFQg) and [googledrive](https://drive.google.com/file/d/1FEjeiIB4u-XBYdASgkyx78pFybrlKUA4/view?usp=sharing)). 

**! These models are not for detection testing/inferencing but training and parameters initialization. **

### Pretrained Weight for TEST
Pretrained Model: `RetinaFace-R50` is a medium size model with ResNet50 backbone.
- [baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ)
- [google drive](https://drive.google.com/file/d/1_DKgGxQWqlTqe78pw0KavId9BIMNUWfu/view?usp=sharing)

WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4.

## Install
### GPU
- It needs NVIDIA Drvier & CUDA & cuDNN & NCCL Library.
- Test Python Environment is `Python 3.8.15` version with `conda`.
- Some modules are replaced by `cython` implements.
```shell
$ conda create --name retinaface python==3.8.15
$ conda activate retinaface
$ pip install -r requirements.txt

# replace some modules with cython
$ make
```


## Training

Please check ``train.py`` for training. 

1. Copy ``rcnn/sample_config.py`` to ``rcnn/config.py``

2. Start training with below sample script.
```shell
$ CUDA_VISIBLE_DEVICES='0,1' python -u train.py \
  --network resnet \
  --image_set train \               # train set name
  --root_path data \                # dataset directory path 
  --dataset_path data/retinaface \  # detail dataset path
  --pretrained model/resnet-50 \    # pretrained model path(symbol)
  --pretrained_epoch 0 \            # pretrained model epoch(parmas)
  --prefix model/retinaface \       # train model name prefix
  --end_epoch 100 \                 # train epochs 
```
  - Before training, you should check the ``resnet`` network configuration about hyperparameters in ``rcnn/config.py``.

## Converting & Optimization

### mxnet - op_translater 
- `SoftmaxActivation` operation in mxnet needs to add in onnx op_translater.
- Add python script `_op_translations_opset_custom.py` in mxnet install path (e.g. `site-packages/mxnet/onnx/mx2onnx/_op_translations`)
- add `import` sciprt in `mxnet/onnx/mx2onnx/__init__.py`  
```shell
$ cp _op_translations_opset_custom.py {CONDA_PATH}/site-packages/mxnet/onnx/mx2onnx/_op_translations/
$ echo 'from ._op_translations import _op_translations_opset_custom' >> {CONDA_PATH}/site-packages/mxnet/onnx/mx2onnx/__init__.py
```

### ONNX Converting
Example weights are `R50-symbol.json`, `R50-0000.params`
```shell
$ python export.py \
  --symbol ./R50-symbol.json \
  --params ./R50-0000.parmas \
  --type onnx
```
It makes `./R50.onnx` file for optimizing tensorrt.

### TensorRT Optimization
```shell
$ python export.py \
  --symbol ./R50-symbol.json \
  --params ./R50-0000.parmas \
  --type tensorrt \
  -b 1 \
  -w 30 \
  --fp16   
```
- If you wirte `--fp16` option, it will make `fp32` tensorrt engine 

### Int8 Qunatization
Int8 Quntization need calibration process. 
calibration data structure is different with train dataset. Train Set image directories have many sub dirs.
Calibration dataset have one directory and images.
```shell
data/retinaface
└── calib
    └── images
        ├── image001.jpg
        ├── ...
        └── imageXXX.jpg
```

```shell
$ python export.py \
  --symbol ./R50-symbol.json \
  --params ./R50-0000.parmas \
  --type tensorrt \
  -b 1 \
  -w 30 \
  --int8 \
  --calib_input ./data/calib/images \
  --calib_cache ./calib_data.cache \
  --calib_num_images 8000 \
  --calib_batch_size 16
```
- `--calib_input` is calibration dataset path
- Calibaration steps are controlled by `--calib_num_images` and `--calib_batch_size` options.

## Testing
There are three test scripts(`test.py`, `test_onnx.py`, `test_tensort.py`) available. 
Each script infer their network about input and create output image.

### test.py - mxnet
```shell
$ python test.py \
  -s ./model/R50
  -e 0
  -f ./test_image.jpg
```
It makes result output image named by input name example `./model/R50-mxnet.jpg`.

### test_onnx.py - onnxruntime
```shell
$ python test_onnx.py \
  -i ./model/R50.onnx \
  -f ./test_image.jpg
  
```
It makes result output image named by input name example `./model/R50-onnx.jpg`.

### test_tensorrt.py - tensorrt
```shell
$ python test_tensort.py \
  -i ./model/R50-fp16.engine \
  -f ./test_image.jpg
```
It makes result output image named by input name example `./model/R50-fp16.jpg`.