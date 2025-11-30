# Parameter Reduction for Vision Transformers

This repository contains the official implementation of **"Parameter Reduction Improves Vision Transformers: A Comparative Study of Sharing and Pruning"**.

[[arXiv]](TBD)

## Key Findings

- **Parameter reduction can improve accuracy**: Both GroupedMLP and ShallowMLP outperform the 86.6M baseline while using only 67.3% of parameters
- **Dramatically improved training stability**: Peak-to-final accuracy degradation drops from 0.47% (baseline) to 0.03–0.06% (reduced models)
- **Complementary trade-offs**: GroupedMLP preserves compute cost but reduces memory; ShallowMLP reduces both parameters and FLOPs for 38% faster inference
- **ViT-B/16 is overparameterized**: Standard training operates in a regime where MLP capacity can be reduced without harming—and can even improve—performance

## Results

### ImageNet-1K Validation Results

| Model | Params | Top-1 Acc (%) | Top-5 Acc (%) | Peak Epoch | Δ(P→F) (%) | Throughput (img/s) |
|-------|--------|---------------|---------------|------------|------------|-------------------|
| Baseline | 86.6M | 81.05±0.11 | 95.36±0.08 | 219±13 | 0.47±0.04 | 1,020 |
| **GroupedMLP** | 58.2M | **81.47±0.11** | **95.66±0.08** | 272±1 | **0.06±0.06** | 1,017 |
| **ShallowMLP** | 58.2M | 81.25±0.02 | 95.52±0.02 | 273±0 | **0.03±0.01** | **1,411** |

*Both parameter-reduced models significantly outperform baseline (p < 0.05, paired t-test). Δ(P→F) = Peak-to-final accuracy gap (lower is better). Mean ± std over two seeds.*

### Architecture Comparison

| Model | Params | MLP Params | Unique MLPs | GFLOPs | Expansion Ratio |
|-------|--------|------------|-------------|--------|-----------------|
| Baseline | 86.6M | 56.7M | 12 | 16.9 | 4× |
| GroupedMLP | 58.2M | 28.3M | 6 | 16.9 | 4× |
| ShallowMLP | 58.2M | 28.3M | 12 | 11.3 | 2× |

## Reduction Strategies

| Strategy | Description | When Applied | Key Benefit |
|----------|-------------|--------------|-------------|
| **GroupedMLP** | Shares MLP weights between adjacent block pairs | After model creation | Same compute, reduced memory |
| **ShallowMLP** | Prunes MLP hidden dimension (3072 → 1536) | After model creation | 38% faster inference |
| **ThinMLP*** | Reduces MLP ratio during model creation | Before model creation | Smaller model from scratch |

*\*ThinMLP is an additional experimental strategy not included in the paper.*

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision ≥ 0.13
- timm == 0.9.x or later
- CUDA ≥ 11.3 (for GPU training)

```bash
pip install torch torchvision timm
```

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

#### ShallowMLP Strategy

Creates full model, then prunes MLP hidden dimensions (3072 → 1536):

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

#### ThinMLP Strategy (Experimental)

Creates model with reduced MLP ratio from scratch:

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

## Strategy Details

### GroupedMLP

Shares MLP parameters between adjacent transformer blocks: blocks (2i, 2i+1) for i ∈ {0,...,5} reference identical parameters, reducing 12 unique MLPs to 6. Weights are scaled at initialization to maintain proper gradient flow:

```
θ_shared ← (1/√2) · θ_init  for W_fc1, W_fc2, b_fc1
```

**Key characteristics:**
- Maintains baseline computational cost (16.9 GFLOPs)
- Reduces memory footprint through weight sharing
- Same MLP serves multiple depths, acting as implicit regularizer

### ShallowMLP

Reduces the MLP hidden dimension from 3072 to 1536 across all blocks while preserving initialization statistics from the full model:

```
W_fc1 ← W_fc1_full[:d/2, :]
W_fc2 ← W_fc2_full[:, :d/2]
```

**Key characteristics:**
- Reduces both parameters and compute (11.3 GFLOPs)
- 38% higher inference throughput
- Inherits initialization from larger model before pruning

### ThinMLP (Experimental)

Modifies the `mlp_ratio` parameter during model creation. Standard ViT uses `mlp_ratio=4.0` (768 → 3072 → 768). With `--reduction-mlp-ratio 2.0`, the MLP becomes (768 → 1536 → 768).

**Key characteristics:**
- Model is created smaller from scratch
- Does not inherit initialization from larger model
- Useful as a baseline comparison

## Parameter Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--reduction-strategy` | Strategy: `thin`, `shallow`, or `grouped` | `None` (baseline) |
| `--reduction-mlp-ratio` | Strategy-specific parameter (see below) | `2.0` |

**`--reduction-mlp-ratio` meaning per strategy:**

| Strategy | Parameter Meaning | Recommended Value |
|----------|-------------------|-------------------|
| GroupedMLP | Weight scale factor (1/√2 for gradient balance) | `0.707` |
| ShallowMLP | Fraction of hidden dim to keep | `0.5` |
| ThinMLP | MLP ratio (hidden_dim / embed_dim) | `2.0` |

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
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{krishnakumar2025parameter,
  title={Parameter Reduction Improves Vision Transformers: A Comparative Study of Sharing and Pruning},
  author={Krishna Kumar, Anantha Padmanaban},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman and Hugging Face
- This work was conducted at Boston University
