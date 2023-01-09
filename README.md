# PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition

This repository implements PatchNet from paper [PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition](https://arxiv.org/abs/2203.14325)

## Reference
[1] PatchNet: [PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition](https://arxiv.org/abs/2203.14325) \
[2] CDCN repository: [CDCN-Face-Anti-Spoofing.pytorch](https://github.com/voqtuyen/CDCN-Face-Anti-Spoofing.pytorch)

## Project Structure
```
Implementation-patchnet
      |
      |---config
      |     |--config.yaml
      |
      |---dataset
      |     |--FAS_dataset.py
      |     |--transform.py
      |
      |---engine
      |     |--__init__.py
      |     |--base_trainer.py
      |     |--Patchnet_trainer.py
      |
      |---metrics
      |     |--losses.py
      |     |--meter.py
      |
      |---models
      |     |--CDCNs.py
      |     |--convnext_tiny.py
      |     |--DC_CDN.py
      |     |--resnet18.py
      |     |--swin_base.py
      |
      |---tool
      |     |--test.py
      |     |--train.py
      |
      |---utils
      |     |--utils.py
      |
      |---README.md
      |---requirements.txt
```

## Installation
```
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```
## Usage
### Data preparation
```
datasets
    |---images
    |     |--img1
    |     |--img2
    |     |...
    |---train.csv
    |---val.csv
    |---test.csv
```
with [set_name.csv] have format (label only has 2 class: 0-Spoofing, 1-Liveness): \
```
image_name  |  label
img_name1   |    0
img_name2   |    1
...
```
### training
```
python3 train.py
```
### Testing
Go to tool/test.py and fix saved_name to your path to checkpoint \
Run
```
python3 test.py
```
## Contributer
Tien Thong Doan \
Minh Chau Nguyen \
Minh Hung Nguyen
