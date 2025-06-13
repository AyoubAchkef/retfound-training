"""
Transform Functions for RETFound
================================

Implements advanced image transformations including pathology-specific
augmentations, MixUp/CutMix, and test-time augmentation.
Updated for dataset v6.1 with specific handling for minority classes.
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
    """Augmentations specific to retinal pathologies - Dataset v6.1"""
    
    def __init__(self, input_size: int = 224):
        """
        Initialize pathology-specific augmentations
        Updated for dataset v6.1 minority classes
        
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
            
            # Hemorrhages/Occlusions for Fundus - enhance red channel
            'hemorrhage_fundus': A.Compose([
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
            ]),
            
            # ERM (Epiretinal Membrane) - v6.1 minority class (0.4%)
            'erm': A.Compose([
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.2, 
                    p=0.6
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.1, 
                    rotate_limit=10, 
                    p=0.4
                ),
            ]),
            
            # Myopia Degenerative - v6.1 minority class (1.3%)
            'myopia_degenerative': A.Compose([
                A.RandomCrop(
                    height=int(input_size*0.9), 
                    width=int(input_size*0.9), 
                    p=0.4
                ),
                A.Resize(input_size, input_size),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
                A.ElasticTransform(
                    alpha=50, sigma=5, alpha_affine=5, p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.2, p=0.5
                ),
            ]),
            
            # OCT vascular occlusions (RVO/RAO) - v6.1 minority classes
            'vascular_oct': A.Compose([
                A.RandomGamma(gamma_limit=(90, 110), p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.4),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.RandomContrast(limit=0.2, p=0.5),
            ]),
            
            # CSR (Central Serous Retinopathy) - fluid accumulation
            'csr': A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.3, p=0.6
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.ElasticTransform(
                    alpha=60, sigma=5, alpha_affine=5, p=0.3
                ),
            ]),
            
            # Retinal Detachment - severe geometric distortion
            'retinal_detachment': A.Compose([
                A.ElasticTransform(
                    alpha=150, sigma=8, alpha_affine=8, p=0.6
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1), 
                    num_shadows_lower=1, 
                    num_shadows_upper=2, 
                    p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.3, p=0.5
                ),
            ]),
        }
    
    def get_augmentation(self, label_name: str) -> Optional[Callable]:
        """
        Get augmentation based on pathology type
        Updated for dataset v6.1 with all 28 classes
        
        Args:
            label_name: Name of the class/label
            
        Returns:
            Augmentation transform or None
        """
        if not self.augmentations:
            return None
        
        label_lower = label_name.lower()
        
        # ===== Dataset v6.1 Minority Classes =====
        
        # ERM (Epiretinal Membrane) - 0.4% in OCT
        if 'erm' in label_lower or 'epiretinal' in label_lower:
            return self.augmentations.get('erm', None)
        
        # Myopia Degenerative - 1.3% in Fundus
        elif 'myopia' in label_lower and 'degenerative' in label_lower:
            return self.augmentations.get('myopia_degenerative', None)
        
        # ===== Vascular Pathologies with OCT/Fundus distinction =====
        
        # RVO/RAO - Different handling for OCT vs Fundus
        elif any(vascular in label_lower for vascular in ['rao', 'rvo']):
            if 'oct' in label_lower:
                # OCT vascular occlusions (0.4-0.5% minority)
                return self.augmentations.get('vascular_oct', None)
            else:
                # Fundus vascular occlusions
                return self.augmentations.get('hemorrhage_fundus', None)
        
        # ===== Diabetic Retinopathy Grades =====
        
        elif any(dr in label_lower for dr in ['dr_', 'diabetic', 'proliferative']):
            return self.augmentations.get('dr', None)
        
        # ===== Macular Pathologies =====
        
        # CNV/AMD
        elif any(cnv in label_lower for cnv in ['cnv', 'amd', 'wet_amd', 'choroidal']):
            return self.augmentations.get('cnv_amd', None)
        
        # Dry AMD (OCT only)
        elif 'dry' in label_lower and 'amd' in label_lower:
            return self.augmentations.get('oct', None)
        
        # CSR
        elif 'csr' in label_lower or 'serous' in label_lower:
            return self.augmentations.get('csr', None)
        
        # DME (Diabetic Macular Edema)
        elif 'dme' in label_lower or 'macular_edema' in label_lower:
            return self.augmentations.get('dr', None)
        
        # ===== Glaucoma =====
        
        elif 'glaucoma' in label_lower:
            return self.augmentations.get('glaucoma', None)
        
        # ===== Other Fundus Pathologies =====
        
        # Retinal Detachment
        elif 'retinal_detachment' in label_lower or 'detachment' in label_lower:
            return self.augmentations.get('retinal_detachment', None)
        
        # Hypertensive Retinopathy
        elif 'hypertensive' in label_lower:
            return self.augmentations.get('hemorrhage_fundus', None)
        
        # Drusen
        elif 'drusen' in label_lower:
            return self.augmentations.get('cnv_amd', None)
        
        # Macular Scar
        elif 'macular_scar' in label_lower or 'scar' in label_lower:
            return self.augmentations.get('cnv_amd', None)
        
        # Cataract
        elif 'cataract' in label_lower:
            # Mild blur augmentation for cataract
            return A.Compose([
                A.GaussianBlur(blur_limit=(5, 9), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.2, p=0.6
                ),
            ])
        
        # Optic Disc Anomaly
        elif 'optic_disc' in label_lower or 'disc_anomaly' in label_lower:
            return self.augmentations.get('glaucoma', None)
        
        # Vitreomacular Interface Disease
        elif 'vitreomacular' in label_lower or 'interface' in label_lower:
            return self.augmentations.get('erm', None)
        
        # ===== General OCT augmentation for unmatched OCT classes =====
        
        elif 'oct' in label_lower and 'normal' not in label_lower:
            return self.augmentations.get('oct', None)
        
        # No specific augmentation for Normal classes or Other
        return None


def create_train_transforms(
    config: RETFoundConfig,
    pathology_aug: Optional[PathologyAugmentation] = None
) -> Callable:
    """
    Create training transforms with advanced augmentations
    Updated for dataset v6.1
    
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


# Dataset v6.1 specific augmentation weights
DATASET_V61_CLASS_WEIGHTS = {
    # OCT minority classes
    "04_ERM": 2.0,          # 0.4%
    "07_RVO_OCT": 2.0,      # 0.4%
    "09_RAO_OCT": 1.5,      # 0.5%
    
    # Fundus minority class
    "12_Myopia_Degenerative": 1.5,  # 1.3%
}


def get_class_augmentation_weight(class_name: str) -> float:
    """
    Get augmentation weight for a specific class in dataset v6.1
    
    Args:
        class_name: Name of the class
        
    Returns:
        Weight multiplier for augmentation
    """
    for key, weight in DATASET_V61_CLASS_WEIGHTS.items():
        if key in class_name:
            return weight
    return 1.0


# Alias for backward compatibility
def get_eval_transforms(config: RETFoundConfig) -> Callable:
    """
    Get evaluation transforms (alias for create_test_transforms)
    
    Args:
        config: Configuration object
        
    Returns:
        Transform function
    """
    return create_test_transforms(config)
