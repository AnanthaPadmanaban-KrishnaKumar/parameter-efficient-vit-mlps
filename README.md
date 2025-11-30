# Parameter Reduction for PyTorch Image Models

A parameter reduction module designed to integrate with [timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) for efficient model training.

## Overview

This repository provides a parameter reduction implementation that can be seamlessly integrated into the timm training pipeline. It includes a modified `train.py` script and a `parameter_reduction` module.

## Installation

### 1. Clone timm

```bash
git clone https://github.com/huggingface/pytorch-image-models.git
```

### 2. Install timm in editable mode

```bash
cd pytorch-image-models
pip install -e .
cd ..
```

### 3. Clone this repository

```bash
git clone https://github.com/AnanthaPadmanaban-KrishnaKumar/parameter-efficient-vit-mlps

```

### 4. Copy the parameter reduction module and training script into timm

```bash
cp -r parameter_reduction_githubrepo/parameter_reduction pytorch-image-models/
cp parameter_reduction_githubrepo/train.py pytorch-image-models/train.py
```

## Usage

Navigate to the timm directory and run training:

```bash
cd pytorch-image-models
python train.py [your-arguments-here]
```

Replace `[your-arguments-here]` with your desired timm training arguments (e.g., dataset path, model name, batch size, etc.).

### Example

```bash
python train.py \
    --model resnet50 \
    --dataset torch/imagenet \
    --data-dir /path/to/imagenet \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.1
```

## Project Structure

```
parameter_reduction_githubrepo/
├── parameter_reduction/    # Parameter reduction module
│   └── ...
├── train.py                # Modified training script
└── README.md
```

## License

[Add your license here]

## Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman and Hugging Face
