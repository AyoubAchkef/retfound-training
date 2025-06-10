"""RETFound Training Framework - Professional implementation for ophthalmology AI.

This package provides a modular, extensible framework for training RETFound models
on ophthalmological data with state-of-the-art optimizations.
"""

from retfound.__version__ import __version__
from retfound.core.config import RETFoundConfig
from retfound.core.exceptions import (
    RETFoundError,
    ConfigurationError,
    DatasetError,
    ModelError,
    TrainingError,
)
from retfound.models import RETFoundModel, create_model
from retfound.training import RETFoundTrainer
from retfound.data import RETFoundDataModule, create_datamodule

# Public API
__all__ = [
    # Version
    "__version__",
    # Configuration
    "RETFoundConfig",
    # Models
    "RETFoundModel",
    "create_model",
    # Training
    "RETFoundTrainer",
    # Data
    "RETFoundDataModule",
    "create_datamodule",
    # Exceptions
    "RETFoundError",
    "ConfigurationError",
    "DatasetError",
    "ModelError",
    "TrainingError",
]

# Module information
__author__ = "CAASI Medical AI Team"
__email__ = "support@caasi-ai.com"
__license__ = "Proprietary"
__copyright__ = "Copyright (c) 2025 CAASI Medical AI"