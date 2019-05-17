# N-bit Quantized Face Recognition

Implement the quantization of ["Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR, 2018](https://arxiv.org/abs/1712.05877) for face recognition using MobileNetV2.

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
└── src/                    For source code

```
## License
MIT 

## Author
Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295)
