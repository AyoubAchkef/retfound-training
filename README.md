# 🏥 RETFound Training Framework

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3.1-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.1-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-Proprietary-orange.svg" alt="License">
</div>

<div align="center">
  <h3>Professional RETFound Training Framework for Ophthalmology</h3>
  <p>State-of-the-art Vision Transformer training system for retinal disease classification</p>
</div>

---

## 🚀 Features

- **🏗️ RETFound Architecture**: Vision Transformer Large (632M parameters) pre-trained on 1.6M retinal images
- **⚡ Advanced Optimizations**: SAM optimizer, EMA, TTA, Temperature Scaling, Layer-wise LR decay
- **🏥 Medical Specialization**: Ophthalmology-specific metrics, pathology augmentations, critical condition monitoring
- **🔥 A100 Optimized**: BFloat16, gradient checkpointing, torch.compile, optimized for NVIDIA A100
- **📊 Comprehensive Monitoring**: TensorBoard, Weights & Biases, real-time metrics, clinical reports
- **🔧 Production Ready**: Modular architecture, extensive testing, multiple export formats

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Export & Deployment](#export--deployment)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.9-3.11
- CUDA 12.1+ (for GPU support)
- 16GB+ GPU memory (A100 recommended)
- 32GB+ RAM

### Install with Poetry

```bash
# Clone the repository
git clone https://github.com/caasi/retfound-training.git
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

### Download RETFound Weights

```bash
# Run the download script
poetry run python scripts/download_weights.py

# Or manually download from GitHub
wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureCFP.pth
wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureOCT.pth
wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_meh.pth
```

## 🚀 Quick Start

### Basic Training

```bash
# Train with default settings
poetry run retfound train --weights cfp --epochs 100

# Train with custom configuration
poetry run retfound train --config configs/production/a100_optimized.yaml

# Resume training from checkpoint
poetry run retfound train --weights cfp --resume checkpoints/retfound/latest.pth
```

### Evaluation

```bash
# Evaluate a trained model
poetry run retfound evaluate --checkpoint checkpoints/retfound/best.pth

# Generate clinical report
poetry run retfound evaluate --checkpoint checkpoints/retfound/best.pth --clinical-report
```

### Export

```bash
# Export to multiple formats
poetry run retfound export --checkpoint checkpoints/retfound/best.pth --formats onnx,torchscript

# Export with optimization
poetry run retfound export --checkpoint checkpoints/retfound/best.pth --optimize --quantize
```

## 📁 Project Structure

```
retfound-training/
├── retfound/               # Main package
│   ├── core/              # Core components (config, registry, exceptions)
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and augmentation
│   ├── training/          # Training logic and optimizers
│   ├── metrics/           # Evaluation metrics
│   ├── utils/             # Utilities
│   ├── export/            # Model export functionality
│   └── cli/               # Command-line interface
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## ⚙️ Configuration

### YAML Configuration

Create a custom configuration file:

```yaml
# configs/custom.yaml
model:
  type: vit_large_patch16_224
  num_classes: 22
  pretrained_weights: cfp

training:
  batch_size: 32
  epochs: 150
  base_lr: 1e-4
  
optimization:
  use_sam: true
  use_ema: true
  use_tta: true
  
data:
  dataset_path: /path/to/dataset
  augmentation_level: strong
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your settings
```

## 🏃 Training

### Standard Training

```bash
# Single GPU training
poetry run retfound train --weights cfp --batch-size 32 --epochs 100

# Multi-GPU training (coming soon)
poetry run retfound train --weights cfp --gpus 4 --strategy ddp
```

### Advanced Training

```bash
# K-fold cross-validation
poetry run retfound train --weights cfp --kfold 5

# Hyperparameter optimization
poetry run retfound optimize --config configs/base.yaml --trials 50

# Custom learning rate schedule
poetry run retfound train --weights cfp --scheduler cosine --warmup-epochs 10
```

## 📊 Evaluation

### Model Evaluation

```bash
# Basic evaluation
poetry run retfound evaluate --checkpoint path/to/checkpoint.pth

# Detailed evaluation with plots
poetry run retfound evaluate --checkpoint path/to/checkpoint.pth --save-plots --save-predictions

# Test-time augmentation
poetry run retfound evaluate --checkpoint path/to/checkpoint.pth --tta
```

### Clinical Metrics

The framework automatically computes:
- Sensitivity/Specificity per class
- Cohen's Kappa
- AUC-ROC (macro/weighted)
- Quadratic Kappa for DR
- Critical condition alerts

## 📦 Export & Deployment

### Export Models

```bash
# Export to ONNX
poetry run retfound export --checkpoint best.pth --format onnx

# Export to TorchScript
poetry run retfound export --checkpoint best.pth --format torchscript

# Export with quantization
poetry run retfound export --checkpoint best.pth --format onnx --quantize int8
```

### Inference

```python
from retfound.export import load_model

# Load exported model
model = load_model("path/to/exported_model.onnx")

# Run inference
result = model.predict("path/to/retinal_image.jpg")
print(f"Prediction: {result['class']}, Confidence: {result['confidence']:.2%}")
```

## 📚 API Reference

### Python API

```python
from retfound import RETFoundTrainer, RETFoundConfig
from retfound.data import create_datamodule

# Configure training
config = RETFoundConfig(
    model_type="vit_large_patch16_224",
    batch_size=32,
    epochs=100,
    use_sam=True
)

# Create data module
datamodule = create_datamodule(config)

# Initialize trainer
trainer = RETFoundTrainer(config)

# Train model
trainer.fit(datamodule)

# Evaluate
results = trainer.test(datamodule)
```

### CLI Commands

```bash
# Training
retfound train [OPTIONS]

# Evaluation
retfound evaluate [OPTIONS]

# Export
retfound export [OPTIONS]

# Prediction
retfound predict IMAGE_PATH [OPTIONS]

# Hyperparameter optimization
retfound optimize [OPTIONS]
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=retfound --cov-report=html

# Run specific test categories
poetry run pytest -m unit
poetry run pytest -m integration
poetry run pytest -m "not slow"

# Run in parallel
poetry run pytest -n auto
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is proprietary software. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [RETFound](https://github.com/rmaphoh/RETFound_MAE) - Original MAE pre-trained weights
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [timm](https://github.com/rwightman/pytorch-image-models) - PyTorch image models

## 📞 Contact

- **Technical Support**: support@caasi-ai.com
- **Medical Inquiries**: medical@caasi-ai.com
- **Business**: business@caasi-ai.com

---

<div align="center">
  <p><strong>© 2025 CAASI Medical AI - Advancing Ophthalmology with AI</strong></p>
</div>
