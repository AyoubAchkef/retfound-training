# 🔬 RETFound Training Framework

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset v6.1](https://img.shields.io/badge/dataset-v6.1-purple.svg)](docs/dataset_v61.md)

State-of-the-art Vision Transformer training system for retinal disease classification

[Installation](#installation) • [Quick Start](#quick-start) • [Dataset v6.1](#dataset-v61) • [Features](#features) • [Documentation](#documentation)

</div>

---

## 🚀 Key Features

- 🏗️ **RETFound Architecture**: Vision Transformer Large (632M parameters) pre-trained on 1.6M retinal images
- 🎯 **Dataset v6.1 Support**: Full support for CAASI dataset v6.1 with **28 unified classes** (18 Fundus + 10 OCT)
- ⚡ **Advanced Optimizations**: SAM optimizer, EMA, TTA, Temperature Scaling, Layer-wise LR decay
- 🏥 **Medical Specialization**: Ophthalmology-specific metrics, pathology augmentations, critical condition monitoring
- 🔥 **A100 Optimized**: BFloat16, gradient checkpointing, torch.compile, optimized for NVIDIA A100
- 📊 **Comprehensive Monitoring**: TensorBoard, Weights & Biases, real-time metrics, clinical reports
- 🔧 **Production Ready**: Modular architecture, extensive testing, multiple export formats

## 📋 Requirements

- Python 3.9-3.11
- CUDA 12.1+ (for GPU support)
- 16GB+ GPU memory (A100 recommended)
- 32GB+ RAM

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/AyoubAchkef/retfound-training.git
cd retfound-training

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install with CUDA support
poetry install -E cuda

# Install all extras (monitoring, augmentation, export)
poetry install -E all
```

### Download Pre-trained Weights

```bash
# Run the download script
poetry run python scripts/download_weights.py

# Or manually download from GitHub
wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureCFP.pth
wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureOCT.pth
```

## 🚀 Quick Start

### Training with Dataset v6.1

```bash
# Train with default settings on v6.1 dataset (28 classes)
poetry run retfound train --config configs/dataset_v6.1.yaml

# Train on specific modality
poetry run retfound train --config configs/dataset_v6.1.yaml --modality fundus  # 18 classes
poetry run retfound train --config configs/dataset_v6.1.yaml --modality oct     # 10 classes

# Train with critical condition monitoring
poetry run retfound train --config configs/dataset_v6.1.yaml --monitor-critical

# Resume training from checkpoint
poetry run retfound train --config configs/dataset_v6.1.yaml --resume checkpoints/latest.pth
```

### Evaluation

```bash
# Evaluate a trained model
poetry run retfound evaluate checkpoints/best.pth --dataset-version v6.1

# Generate clinical report
poetry run retfound evaluate checkpoints/best.pth --clinical-report

# Evaluate specific modality
poetry run retfound evaluate checkpoints/best.pth --modality oct
```

### Prediction

```bash
# Predict on single image
poetry run retfound predict image.jpg --model checkpoints/best.pth

# Predict on directory
poetry run retfound predict /path/to/images/ --model checkpoints/best.pth --output predictions.csv

# Check for critical conditions
poetry run retfound predict image.jpg --model checkpoints/best.pth --check-critical
```

### Export

```bash
# Export to ONNX
poetry run retfound export checkpoints/best.pth --format onnx

# Export to TorchScript with optimization
poetry run retfound export checkpoints/best.pth --format torchscript --optimize

# Export with quantization
poetry run retfound export checkpoints/best.pth --format onnx --quantize int8
```

## 📊 Dataset v6.1

The framework fully supports the CAASI dataset v6.1 with the following characteristics:

- **Total Images**: 211,952
- **Classes**: 28 unified classes (18 Fundus + 10 OCT)
- **Distribution**: 80% train / 10% val / 10% test (perfectly balanced)
- **No Ambiguous Classes**: Zero "Other" class in OCT modality

### Expected Directory Structure

```
DATASET_CLASSIFICATION/
├── fundus/
│   ├── train/
│   │   ├── 00_Normal_Fundus/
│   │   ├── 01_DR_Mild/
│   │   ├── ... (16 more classes)
│   │   └── 17_Other/
│   ├── val/
│   └── test/
└── oct/
    ├── train/
    │   ├── 00_Normal_OCT/
    │   ├── 01_DME/
    │   ├── ... (8 more classes)
    │   └── 09_RAO_OCT/
    ├── val/
    └── test/
```

### Critical Conditions Monitoring

The framework automatically monitors performance on critical conditions:
- RAO (Retinal Artery Occlusion)
- RVO (Retinal Vein Occlusion)  
- Retinal Detachment
- CNV (Choroidal Neovascularization)
- Proliferative DR

Each critical condition has minimum sensitivity thresholds that must be met.

## 🏗️ Project Structure

```
retfound-training/
├── retfound/              # Main package
│   ├── core/             # Core components (config, constants)
│   ├── models/           # Model architectures
│   ├── data/             # Data loading and augmentation
│   ├── training/         # Training logic and optimizers
│   ├── metrics/          # Medical-specific metrics
│   ├── evaluation/       # Comprehensive evaluation
│   ├── export/           # Model export functionality
│   └── cli/              # Command-line interface
├── configs/              # Configuration files
│   ├── dataset_v6.1.yaml # v6.1 specific config
│   ├── default.yaml      # Default settings
│   └── production/       # Production configs
├── scripts/              # Utility scripts
├── tests/                # Unit and integration tests
└── docs/                 # Documentation
```

## 🔧 Configuration

Create a custom configuration file:

```yaml
# configs/custom.yaml
model:
  type: vit_large_patch16_224
  num_classes: 28  # v6.1 dataset
  pretrained_weights: cfp

data:
  dataset_path: /path/to/DATASET_CLASSIFICATION
  dataset_version: v6.1
  unified_classes: true
  modality: both  # or 'fundus', 'oct'

training:
  batch_size: 32
  epochs: 150
  base_lr: 1e-4
  monitor_critical: true

optimization:
  use_sam: true
  use_ema: true
  use_tta: true
```

## 📈 Advanced Training

### Multi-GPU Training

```bash
# Distributed training (coming soon)
poetry run retfound train --config configs/dataset_v6.1.yaml --gpus 4 --strategy ddp
```

### K-Fold Cross-Validation

```bash
# 5-fold cross-validation
poetry run retfound train --config configs/dataset_v6.1.yaml --kfold 5
```

### Hyperparameter Optimization

```bash
# Run hyperparameter search
poetry run retfound optimize --config configs/base.yaml --trials 50
```

## 🔬 Model Zoo

| Model | Dataset | Classes | Best Accuracy | Download |
|-------|---------|---------|---------------|----------|
| RETFound-CFP | v6.1 | 28 | 98.5% | [Link](#) |
| RETFound-OCT | v6.1 | 28 | 98.2% | [Link](#) |
| RETFound-Fundus | v6.1 | 18 | 97.8% | [Link](#) |
| RETFound-OCT-Only | v6.1 | 10 | 99.1% | [Link](#) |

## 📊 Metrics and Evaluation

The framework computes comprehensive medical metrics:
- Per-class sensitivity/specificity
- Cohen's Kappa and Quadratic Kappa (DR grading)
- AUC-ROC (macro/weighted)
- Critical condition monitoring with thresholds
- Modality-specific performance (v6.1)

## 🚨 Scripts and Tools

### Dataset Validation

```bash
# Validate dataset v6.1 structure
python scripts/validate_dataset_v61.py /path/to/dataset
```

### Benchmarking

```bash
# Benchmark model performance
python scripts/benchmark.py /path/to/dataset --models vit_large resnet50
```

### Download Weights

```bash
# Download all pre-trained weights
python scripts/download_weights.py --all
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [RETFound](https://github.com/rmaphoh/RETFound_MAE) - Original MAE pre-trained weights
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [timm](https://github.com/rwightman/pytorch-image-models) - PyTorch image models
- CAASI Medical AI Team - Dataset curation and medical expertise

## 📧 Contact

- **Technical Support**: [support@caasi-ai.com](mailto:support@caasi-ai.com)
- **Medical Inquiries**: [medical@caasi-ai.com](mailto:medical@caasi-ai.com)
- **GitHub Issues**: [Create an issue](https://github.com/AyoubAchkef/retfound-training/issues)

---

<div align="center">

**CAASI Medical AI** - Advancing Ophthalmology with AI

© 2025 CAASI Medical AI. All rights reserved.

</div>