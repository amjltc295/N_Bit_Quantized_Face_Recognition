# N-bit Quantized Face Recognition

Implement the quantization of ["Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR, 2018](https://arxiv.org/abs/1712.05877) for face recognition.

## Installation

1. Install miniconda/anaconda, a package for  package/environment management
```
wget repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

2. Build conda environment from file
```
conda env create -f environment.yaml
```

3. Activate the environment
```
source activate n_bit_quantized_face_recognition
```

## Dataset
Please download the MS-Celeb-1M(Align_112x112) dataset from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch and put it under `datasets/`

## Training
### Train from scratch
1. Setup the configs about training, dataset, model and loss
2. Run
```
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/basic/config.json configs/dataset/ms1m.json configs/model/mobilenetv2.json configs/loss/config.json
```
### Load pretrained backbones
1. Download the pretrained backbone from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
2. Run
```
CUDA_VISIBLE_DEVICES=1 python train.py -c configs/basic/config.json configs/dataset/ms1m.json configs/model/face_recognition.json configs/loss/config.json -p ../pretrained_weights/backbone_ir50_ms1m_epoch120.pth
```

## Repository Structure
```
├── .flake8                 Syntax and style settings for Flake8
├── .gitignore              Filenames in this file would be ignored by Git
├── .travis.yml             For Travis CI configuration
├── environment.yaml        For Conda environment
├── README.md
├── LICENSE                 LICENSE file (MIT license here)
├── .github/                For the PR template
├── tests/                  For tests
├── lib/                    For third-party libraries
├── pretrained_weights/     For pretrained backbones
├── src/
│   ├── base/
│   ├── configs/            For training, each group of config should be loaded
│   │   ├── basic/
│   │   ├── dataset/
│   │   ├── loss/
│   │   └── model/
│   ├── data_loader
│   │   └── data_loaders.py
│   ├── logger/
│   │   ├── logger.py
│   │   ├── logger_config.json
│   │   └── visualization.py
│   ├── model/
│   │   ├── backbone/       Backbones to extract face features
│   │   ├── head/           Loss heads for metric learning
│   │   ├── loss.py
│   │   ├── metric.py
│   │   └── model.py
│   ├── parse_config.py
│   ├── test.py
│   ├── train.py
│   ├── trainer/
│   │   └── trainer.py
│   └── utils/
│       └── util.py
└── tests/
    └── test_quantization.py

```
## License
MIT 

## Author
Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295)

## Disclaimers
This project is based on the following sources:
* https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
* https://github.com/eladhoffer/quantized.pytorch
* https://github.com/victoresque/pytorch-template
* https://github.com/amjltc295/PythonRepoTemplate
* https://github.com/eladhoffer/convNet.pytorch
