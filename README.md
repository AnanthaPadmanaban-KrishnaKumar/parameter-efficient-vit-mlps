# Parameter Reduction for PyTorch Image Models

A parameter reduction module designed to integrate with [timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) for efficient model training.

## Overview

This repository provides parameter reduction strategies for Vision Transformers (ViT) that can be seamlessly integrated into the timm training pipeline. It includes a modified `train.py` script and a `parameter_reduction` module with three different strategies for reducing MLP parameters.

### Reduction Strategies

| Strategy | Description | When Applied | Parameter Reduction |
|----------|-------------|--------------|---------------------|
| **ThinMLP** | Reduces MLP hidden dimension via `mlp_ratio` | Before model creation | ~33% fewer params |
| **ShallowMLP** | Prunes MLP weights after full initialization | After model creation | ~33% fewer params |
| **GroupedMLP** | Shares MLPs between adjacent block pairs | After model creation | ~33% fewer params |

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
cp -r parameter-efficient-vit-mlps/parameter_reduction pytorch-image-models/
cp parameter-efficient-vit-mlps/train.py pytorch-image-models/train.py
```

## Usage

Navigate to the timm directory and run training:

```bash
cd pytorch-image-models
```

### Training Configurations

#### Baseline (No Reduction)

Standard ViT-B/16 training without any parameter reduction:

```bash
./distributed_train.sh 2 /path/to/imagenet \
    --model vit_base_patch16_224 \
    --batch-size 256 \
    --grad-accum-steps 2 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --sched cosine \
    --epochs 300 \
    --warmup-epochs 5 \
    --drop-path 0.1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --aa rand-m9-mstd0.5-inc1 \
    --aug-repeats 3 \
    --amp \
    --channels-last \
    --pin-mem \
    --workers 20 \
    --model-ema \
    --model-ema-decay 0.9998 \
    --seed 42 \
    --experiment baseline_vit_b16
```

#### ThinMLP Strategy

Creates model with reduced MLP ratio from scratch (default: 2.0 instead of 4.0):

```bash
./distributed_train.sh 2 /path/to/imagenet \
    --model vit_base_patch16_224 \
    --reduction-strategy thin \
    --reduction-mlp-ratio 2.0 \
    --batch-size 256 \
    --grad-accum-steps 2 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --sched cosine \
    --epochs 300 \
    --warmup-epochs 5 \
    --drop-path 0.1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --aa rand-m9-mstd0.5-inc1 \
    --aug-repeats 3 \
    --amp \
    --channels-last \
    --pin-mem \
    --workers 20 \
    --model-ema \
    --model-ema-decay 0.9998 \
    --seed 42 \
    --experiment thin_mlp_vit_b16 \
    --output ./output/thin_mlp
```

#### ShallowMLP Strategy

Creates full model, then prunes MLP hidden dimensions (default: keep 50%):

```bash
./distributed_train.sh 2 /path/to/imagenet \
    --model vit_base_patch16_224 \
    --reduction-strategy shallow \
    --reduction-mlp-ratio 0.5 \
    --batch-size 256 \
    --grad-accum-steps 2 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --sched cosine \
    --epochs 300 \
    --warmup-epochs 5 \
    --drop-path 0.1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --aa rand-m9-mstd0.5-inc1 \
    --aug-repeats 3 \
    --amp \
    --channels-last \
    --pin-mem \
    --workers 20 \
    --model-ema \
    --model-ema-decay 0.9998 \
    --seed 42 \
    --experiment shallow_mlp_vit_b16 \
    --output ./output/shallow_mlp
```

#### GroupedMLP Strategy

Shares MLP modules between adjacent block pairs (6 unique MLPs instead of 12):

```bash
./distributed_train.sh 2 /path/to/imagenet \
    --model vit_base_patch16_224 \
    --reduction-strategy grouped \
    --reduction-mlp-ratio 0.707 \
    --batch-size 256 \
    --grad-accum-steps 2 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --sched cosine \
    --epochs 300 \
    --warmup-epochs 5 \
    --drop-path 0.1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --aa rand-m9-mstd0.5-inc1 \
    --aug-repeats 3 \
    --amp \
    --channels-last \
    --pin-mem \
    --workers 20 \
    --model-ema \
    --model-ema-decay 0.9998 \
    --seed 42 \
    --experiment grouped_mlp_vit_b16 \
    --output ./output/grouped_mlp
```

**GroupedMLP Scale Factor Options:**
- `0.707` (1/√2): Compensates for doubled gradient accumulation from shared weights
- `1.0`: No scaling (raw gradient accumulation)

## Strategy Details

### ThinMLP

Modifies the `mlp_ratio` parameter during model creation. Standard ViT uses `mlp_ratio=4.0` (768 → 3072 → 768). With `--reduction-mlp-ratio 2.0`, the MLP becomes (768 → 1536 → 768).

**Key characteristic:** Model is created smaller from the start — never instantiates full 86M parameters.

### ShallowMLP

Creates the full ViT-B model (86M params), then prunes the MLP hidden dimensions. With `--reduction-mlp-ratio 0.5`, the hidden dimension is reduced from 3072 to 1536.

**Key characteristic:** Preserves initialization benefits from the larger model before pruning.

### GroupedMLP

Creates the full ViT-B model, then shares MLP module references between adjacent transformer blocks:
- Blocks (0,1), (2,3), (4,5), (6,7), (8,9), (10,11) share MLPs
- Results in 6 unique MLPs instead of 12

**Key characteristic:** Reduces parameters through weight sharing, not dimension reduction.

## Parameter Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--reduction-strategy` | Strategy to use: `thin`, `shallow`, or `grouped` | `None` (baseline) |
| `--reduction-mlp-ratio` | Strategy-specific parameter (see below) | `2.0` |

**`--reduction-mlp-ratio` meaning per strategy:**

| Strategy | Parameter Meaning | Recommended Value |
|----------|-------------------|-------------------|
| ThinMLP | MLP ratio (hidden_dim / embed_dim) | `2.0` |
| ShallowMLP | Fraction of hidden dim to keep | `0.5` |
| GroupedMLP | Weight scale factor | `0.707` or `1.0` |

## Project Structure

```
parameter-efficient-vit-mlps/
├── parameter_reduction/
│   ├── __init__.py
│   ├── strategies/
│   │   ├── thin_mlp.py      # ThinMLPStrategy
│   │   ├── shallow_mlp.py   # ShallowMLPStrategy
│   │   └── grouped_mlp.py   # GroupedMLPStrategy
├── train.py                  # Modified timm training script
└── README.md
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman and Hugging Face
