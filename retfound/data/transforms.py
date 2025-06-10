"""
Transform Functions for RETFound
================================

Implements advanced image transformations including pathology-specific
augmentations, MixUp/CutMix, and test-time augmentation.
"""

import logging
import random
from typing import Optional, List, Tuple, Dict, Any, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

from ..core.config import RETFoundConfig

logger = logging.getLogger(__name__)


class PathologyAugmentation:
    """Augmentations specific to retinal pathologies"""
    
    def __init__(self, input_size: int = 224):
        """
        Initialize pathology-specific augmentations
        
        Args:
            input_size: Input image size
        """
        self.input_size = input_size
        
        if not ALBUMENTATIONS_AVAILABLE:
            logger.warning(
                "Albumentations not available, pathology augmentations disabled"
            )
            self.augmentations = {}
            return
        
        # Define pathology-specific augmentations
        self.augmentations = {
            # Diabetic Retinopathy - enhance vascular features
            'dr': A.Compose([
                A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.7),
                A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.5, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.3, p=0.6
                ),
            ]),
            
            # CNV/AMD - enhance structural changes
            'cnv_amd': A.Compose([
                A.ElasticTransform(
                    alpha=120, sigma=6, alpha_affine=6, p=0.5
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(
                    distort_limit=0.5, shift_limit=0.5, p=0.3
                ),
            ]),
            
            # Glaucoma - focus on optic disc
            'glaucoma': A.Compose([
                A.RandomCrop(
                    height=int(input_size*0.8), 
                    width=int(input_size*0.8), 
                    p=0.3
                ),
                A.Resize(input_size, input_size),
                A.Rotate(limit=15, p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
            ]),
            
            # OCT specific - reduce speckle noise
            'oct': A.Compose([
                A.MedianBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ]),
            
            # Hemorrhages/Occlusions - enhance red channel
            'hemorrhage': A.Compose([
                A.ChannelShuffle(p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=10, 
                    p=0.5
                ),
                A.RGBShift(
                    r_shift_limit=20, 
                    g_shift_limit=10, 
                    b_shift_limit=10, 
                    p=0.5
                ),
            ])
        }
    
    def get_augmentation(self, label_name: str) -> Optional[Callable]:
        """
        Get augmentation based on pathology type
        
        Args:
            label_name: Name of the class/label
            
        Returns:
            Augmentation transform or None
        """
        if not self.augmentations:
            return None
        
        label_lower = label_name.lower()
        
        # Match pathology type
        if any(dr in label_lower for dr in ['dr_', 'diabetic', 'proliferative']):
            return self.augmentations.get('dr', None)
        elif any(cnv in label_lower for cnv in ['cnv', 'amd', 'choroidal']):
            return self.augmentations.get('cnv_amd', None)
        elif 'glaucoma' in label_lower:
            return self.augmentations.get('glaucoma', None)
        elif 'oct' in label_lower:
            return self.augmentations.get('oct', None)
        elif any(hem in label_lower for hem in ['rao', 'rvo', 'hemorrhage']):
            return self.augmentations.get('hemorrhage', None)
        
        return None


def create_train_transforms(
    config: RETFoundConfig,
    pathology_aug: Optional[PathologyAugmentation] = None
) -> Callable:
    """
    Create training transforms with advanced augmentations
    
    Args:
        config: Configuration object
        pathology_aug: Pathology augmentation instance
        
    Returns:
        Transform function
    """
    if ALBUMENTATIONS_AVAILABLE:
        # Base augmentations
        transforms_list = [
            # Spatial augmentations
            A.RandomResizedCrop(
                height=config.input_size,
                width=config.input_size,
                scale=(0.5, 1.0),
                ratio=(0.8, 1.2),
                interpolation=cv2.INTER_LANCZOS4 if CV2_AVAILABLE else 1,
                p=1.0
            ),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Color augmentations
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.8
            ),
            
            # Quality augmentations
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                A.Equalize(p=1),
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MotionBlur(blur_limit=(3, 7), p=1),
                A.MedianBlur(blur_limit=(3, 5), p=1),
            ], p=0.3),
            
            # Advanced augmentations
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
                A.ElasticTransform(
                    alpha=120, sigma=6, alpha_affine=6, p=1
                ),
            ], p=0.3),
            
            # Cutout/Dropout
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
        ]
        
        # Final normalization
        transforms_list.extend([
            A.Normalize(
                mean=config.pixel_mean,
                std=config.pixel_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        transform = A.Compose(transforms_list)
        
    else:
        # Fallback to torchvision
        logger.warning("Using basic torchvision transforms")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                config.input_size,
                scale=(0.5, 1.0),
                ratio=(0.8, 1.2),
                interpolation=InterpolationMode.LANCZOS
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.pixel_mean, std=config.pixel_std)
        ])
    
    return transform


def create_val_transforms(config: RETFoundConfig) -> Callable:
    """
    Create validation transforms (minimal augmentation)
    
    Args:
        config: Configuration object
        
    Returns:
        Transform function
    """
    if ALBUMENTATIONS_AVAILABLE:
        transform = A.Compose([
            A.Resize(
                height=config.input_size,
                width=config.input_size,
                interpolation=cv2.INTER_LANCZOS4 if CV2_AVAILABLE else 1
            ),
            A.Normalize(
                mean=config.pixel_mean,
                std=config.pixel_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(
                (config.input_size, config.input_size),
                interpolation=InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.pixel_mean, std=config.pixel_std)
        ])
    
    return transform


def create_test_transforms(config: RETFoundConfig) -> Callable:
    """
    Create test transforms (same as validation)
    
    Args:
        config: Configuration object
        
    Returns:
        Transform function
    """
    return create_val_transforms(config)


def create_advanced_transforms(
    config: RETFoundConfig,
    mode: str = 'train',
    pathology_aug: Optional[PathologyAugmentation] = None
) -> Callable:
    """
    Create advanced augmentation pipeline
    
    Args:
        config: Configuration object
        mode: Mode ('train', 'val', 'test')
        pathology_aug: Pathology augmentation instance
        
    Returns:
        Transform function
    """
    if mode == 'train':
        return create_train_transforms(config, pathology_aug)
    elif mode in ['val', 'validation']:
        return create_val_transforms(config)
    elif mode == 'test':
        return create_test_transforms(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


class MixupCutmixTransform:
    """Apply MixUp or CutMix augmentation"""
    
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5
    ):
        """
        Initialize MixUp/CutMix transform
        
        Args:
            mixup_alpha: MixUp beta distribution parameter
            cutmix_alpha: CutMix beta distribution parameter
            mixup_prob: Probability of applying MixUp
            cutmix_prob: Probability of applying CutMix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
    
    def mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation"""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def _rand_bbox(
        self,
        size: torch.Size,
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
        """
        Apply MixUp or CutMix
        
        Returns:
            Tuple of (mixed_x, y_a, y_b, lam, method)
        """
        r = np.random.rand()
        
        if r < self.mixup_prob:
            mixed_x, y_a, y_b, lam = self.mixup(x, y)
            return mixed_x, y_a, y_b, lam, 'mixup'
        elif r < self.mixup_prob + self.cutmix_prob:
            mixed_x, y_a, y_b, lam = self.cutmix(x, y)
            return mixed_x, y_a, y_b, lam, 'cutmix'
        else:
            return x, y, y, 1.0, 'none'


class TestTimeAugmentation:
    """Test-Time Augmentation (TTA) wrapper"""
    
    def __init__(
        self,
        config: RETFoundConfig,
        num_augmentations: int = 5
    ):
        """
        Initialize TTA
        
        Args:
            config: Configuration object
            num_augmentations: Number of augmentations to apply
        """
        self.config = config
        self.num_augmentations = num_augmentations
        
        # Create TTA transforms
        self.transforms = self._create_tta_transforms()
    
    def _create_tta_transforms(self) -> List[Callable]:
        """Create TTA transformations"""
        if not ALBUMENTATIONS_AVAILABLE:
            logger.warning("Albumentations not available for TTA")
            return [create_val_transforms(self.config)]
        
        transforms_list = [
            # Original
            A.Compose([
                A.Resize(self.config.input_size, self.config.input_size),
                A.Normalize(
                    mean=self.config.pixel_mean,
                    std=self.config.pixel_std
                ),
                ToTensorV2()
            ]),
            
            # Horizontal flip
            A.Compose([
                A.Resize(self.config.input_size, self.config.input_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(
                    mean=self.config.pixel_mean,
                    std=self.config.pixel_std
                ),
                ToTensorV2()
            ]),
            
            # Vertical flip
            A.Compose([
                A.Resize(self.config.input_size, self.config.input_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(
                    mean=self.config.pixel_mean,
                    std=self.config.pixel_std
                ),
                ToTensorV2()
            ]),
            
            # 90 degree rotation
            A.Compose([
                A.Resize(self.config.input_size, self.config.input_size),
                A.RandomRotate90(p=1.0),
                A.Normalize(
                    mean=self.config.pixel_mean,
                    std=self.config.pixel_std
                ),
                ToTensorV2()
            ]),
            
            # Slight zoom
            A.Compose([
                A.RandomResizedCrop(
                    height=self.config.input_size,
                    width=self.config.input_size,
                    scale=(0.9, 1.0),
                    ratio=(0.95, 1.05),
                    p=1.0
                ),
                A.Normalize(
                    mean=self.config.pixel_mean,
                    std=self.config.pixel_std
                ),
                ToTensorV2()
            ])
        ]
        
        return transforms_list[:self.num_augmentations]
    
    def apply_transforms(self, image: np.ndarray) -> List[torch.Tensor]:
        """
        Apply all TTA transforms to an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of transformed tensors
        """
        results = []
        
        for transform in self.transforms:
            if hasattr(transform, 'transform'):  # Albumentations
                augmented = transform(image=image)['image']
            else:  # torchvision
                augmented = transform(image)
            results.append(augmented)
        
        return results
