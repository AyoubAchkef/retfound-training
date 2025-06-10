#!/usr/bin/env python3
"""
Validate CAASI dataset v6.1 structure and integrity.
Ensures the dataset follows the expected format with 28 classes.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from retfound.core.constants import DATASET_V61_CLASSES


class DatasetValidator:
    """Validate CAASI dataset v6.1 structure and content."""
    
    def __init__(self, dataset_path: Path, verbose: bool = False):
        """
        Initialize validator.
        
        Args:
            dataset_path: Root path to dataset
            verbose: Whether to print detailed information
        """
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def validate(self) -> bool:
        """
        Run complete validation.
        
        Returns:
            True if validation passed, False otherwise
        """
        print(f"Validating dataset at: {self.dataset_path}")
        print("=" * 80)
        
        # Check basic structure
        if not self._validate_structure():
            return False
        
        # Validate each modality
        self._validate_modality('fundus', 18)
        self._validate_modality('oct', 10)
        
        # Check class distribution
        self._validate_class_distribution()
        
        # Check image integrity
        self._validate_images()
        
        # Generate report
        self._generate_report()
        
        return len(self.errors) == 0
    
    def _validate_structure(self) -> bool:
        """Validate basic directory structure."""
        print("Checking directory structure...")
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            self.errors.append(f"Dataset path does not exist: {self.dataset_path}")
            return False
        
        # Check for fundus and oct directories
        fundus_path = self.dataset_path / 'fundus'
        oct_path = self.dataset_path / 'oct'
        
        if not fundus_path.exists():
            self.errors.append("Missing 'fundus' directory")
            return False
        
        if not oct_path.exists():
            self.errors.append("Missing 'oct' directory")
            return False
        
        # Check for train/val/test splits
        for modality in ['fundus', 'oct']:
            for split in ['train', 'val', 'test']:
                split_path = self.dataset_path / modality / split
                if not split_path.exists():
                    self.errors.append(f"Missing {modality}/{split} directory")
                    return False
        
        print("✓ Directory structure is correct")
        return True
    
    def _validate_modality(self, modality: str, expected_classes: int):
        """Validate a specific modality (fundus or oct)."""
        print(f"\nValidating {modality} modality...")
        
        # Get expected class names
        if modality == 'fundus':
            expected_class_names = DATASET_V61_CLASSES[:18]
            class_indices = list(range(18))
        else:  # oct
            expected_class_names = DATASET_V61_CLASSES[18:28]
            class_indices = list(range(18, 28))
        
        # Check each split
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / modality / split
            
            # Get all class directories
            class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
            
            # Check number of classes
            if len(class_dirs) != expected_classes:
                self.errors.append(
                    f"{modality}/{split}: Expected {expected_classes} classes, "
                    f"found {len(class_dirs)}"
                )
                continue
            
            # Check class names
            for i, (class_dir, expected_name) in enumerate(zip(class_dirs, expected_class_names)):
                # Expected format: "00_ClassName", "01_ClassName", etc.
                expected_dir_name = f"{class_indices[i]:02d}_{expected_name}"
                
                if class_dir.name != expected_dir_name:
                    self.errors.append(
                        f"{modality}/{split}: Expected class directory '{expected_dir_name}', "
                        f"found '{class_dir.name}'"
                    )
                
                # Count images
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                              list(class_dir.glob("*.png"))
                
                self.stats[modality][split] += len(image_files)
                self.stats[f"{modality}_{split}_classes"][class_dir.name] = len(image_files)
                
                # Warn if no images
                if len(image_files) == 0:
                    self.warnings.append(f"{modality}/{split}/{class_dir.name}: No images found")
                
                # Check for very few images
                elif len(image_files) < 10:
                    self.warnings.append(
                        f"{modality}/{split}/{class_dir.name}: "
                        f"Only {len(image_files)} images (might be too few)"
                    )
        
        print(f"✓ {modality} modality validation complete")
    
    def _validate_class_distribution(self):
        """Validate class distribution and balance."""
        print("\nValidating class distribution...")
        
        # Expected 80/10/10 split
        expected_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
        
        for modality in ['fundus', 'oct']:
            total = sum(self.stats[modality].values())
            if total == 0:
                continue
            
            print(f"\n{modality.upper()} distribution:")
            for split, expected_ratio in expected_ratios.items():
                count = self.stats[modality][split]
                actual_ratio = count / total if total > 0 else 0
                
                print(f"  {split}: {count:,} images ({actual_ratio:.1%})")
                
                # Check if ratio is within acceptable range (±2%)
                if abs(actual_ratio - expected_ratio) > 0.02:
                    self.warnings.append(
                        f"{modality} {split}: Expected {expected_ratio:.0%} "
                        f"but got {actual_ratio:.1%}"
                    )
        
        # Check for class imbalance
        self._check_class_balance()
    
    def _check_class_balance(self):
        """Check for severe class imbalances."""
        print("\nChecking class balance...")
        
        for modality in ['fundus', 'oct']:
            for split in ['train', 'val', 'test']:
                key = f"{modality}_{split}_classes"
                if key not in self.stats:
                    continue
                
                class_counts = self.stats[key]
                if not class_counts:
                    continue
                
                counts = list(class_counts.values())
                if not counts:
                    continue
                
                mean_count = np.mean(counts)
                std_count = np.std(counts)
                min_count = min(counts)
                max_count = max(counts)
                
                # Warn about severe imbalance
                if max_count > 10 * min_count:
                    self.warnings.append(
                        f"{modality}/{split}: Severe class imbalance detected "
                        f"(max: {max_count}, min: {min_count})"
                    )
                
                # Find underrepresented classes
                for class_name, count in class_counts.items():
                    if count < mean_count - 2 * std_count:
                        self.warnings.append(
                            f"{modality}/{split}/{class_name}: "
                            f"Underrepresented ({count} images)"
                        )
    
    def _validate_images(self):
        """Validate image files (sample check)."""
        print("\nValidating image integrity (sampling)...")
        
        # Sample some images from each class
        sample_size = 5
        corrupted_images = []
        
        for modality in ['fundus', 'oct']:
            for split in ['train', 'val', 'test']:
                split_path = self.dataset_path / modality / split
                
                for class_dir in split_path.iterdir():
                    if not class_dir.is_dir():
                        continue
                    
                    # Get sample of images
                    image_files = list(class_dir.glob("*.jpg")) + \
                                  list(class_dir.glob("*.jpeg")) + \
                                  list(class_dir.glob("*.png"))
                    
                    sample_files = image_files[:sample_size]
                    
                    for img_path in sample_files:
                        try:
                            # Try to open and verify image
                            img = Image.open(img_path)
                            img.verify()
                            
                            # Re-open for additional checks
                            img = Image.open(img_path)
                            
                            # Check image properties
                            if img.mode not in ['RGB', 'L']:
                                self.warnings.append(
                                    f"{img_path}: Unusual image mode {img.mode}"
                                )
                            
                            # Check size
                            if min(img.size) < 100:
                                self.warnings.append(
                                    f"{img_path}: Very small image {img.size}"
                                )
                            
                        except Exception as e:
                            corrupted_images.append(str(img_path))
                            self.errors.append(f"{img_path}: Corrupted image - {str(e)}")
        
        if corrupted_images:
            print(f"✗ Found {len(corrupted_images)} corrupted images")
        else:
            print("✓ Image integrity check passed")
    
    def _generate_report(self):
        """Generate validation report."""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        
        # Summary statistics
        print("\nDataset Statistics:")
        print("-" * 40)
        
        # Overall statistics
        total_fundus = sum(self.stats['fundus'].values())
        total_oct = sum(self.stats['oct'].values())
        total_images = total_fundus + total_oct
        
        print(f"Total images: {total_images:,}")
        print(f"  Fundus: {total_fundus:,} ({total_fundus/total_images*100:.1f}%)")
        print(f"  OCT: {total_oct:,} ({total_oct/total_images*100:.1f}%)")
        
        # Distribution table
        print("\nDistribution by split:")
        table_data = []
        for modality in ['fundus', 'oct']:
            for split in ['train', 'val', 'test']:
                count = self.stats[modality][split]
                percentage = count / total_images * 100 if total_images > 0 else 0
                table_data.append([modality.upper(), split, f"{count:,}", f"{percentage:.1f}%"])
        
        print(tabulate(table_data, headers=['Modality', 'Split', 'Images', 'Percentage']))
        
        # Class distribution (top/bottom 5)
        print("\nClass distribution highlights:")
        for modality in ['fundus', 'oct']:
            print(f"\n{modality.upper()}:")
            
            # Aggregate across splits
            class_totals = defaultdict(int)
            for split in ['train', 'val', 'test']:
                key = f"{modality}_{split}_classes"
                if key in self.stats:
                    for class_name, count in self.stats[key].items():
                        class_totals[class_name] += count
            
            if class_totals:
                # Sort by count
                sorted_classes = sorted(class_totals.items(), key=lambda x: x[1], reverse=True)
                
                # Show top 5
                print("  Top 5 classes:")
                for class_name, count in sorted_classes[:5]:
                    print(f"    {class_name}: {count:,}")
                
                # Show bottom 5
                print("  Bottom 5 classes:")
                for class_name, count in sorted_classes[-5:]:
                    print(f"    {class_name}: {count:,}")
        
        # Errors and warnings
        print("\nValidation Results:")
        print("-" * 40)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:10], 1):
                print(f"  {i}. {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        else:
            print("\n✅ No errors found!")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        else:
            print("\n✅ No warnings!")
        
        # Save detailed report
        self._save_detailed_report()
    
    def _save_detailed_report(self):
        """Save detailed validation report to file."""
        report_path = self.dataset_path / "validation_report_v61.json"
        
        report = {
            'dataset_path': str(self.dataset_path),
            'validation_timestamp': str(np.datetime64('now')),
            'dataset_version': 'v6.1',
            'expected_classes': 28,
            'statistics': {
                'total_images': sum(sum(self.stats[m].values()) for m in ['fundus', 'oct']),
                'modalities': {
                    'fundus': dict(self.stats['fundus']),
                    'oct': dict(self.stats['oct'])
                },
                'class_distribution': {
                    'fundus': {
                        split: dict(self.stats[f'fundus_{split}_classes'])
                        for split in ['train', 'val', 'test']
                        if f'fundus_{split}_classes' in self.stats
                    },
                    'oct': {
                        split: dict(self.stats[f'oct_{split}_classes'])
                        for split in ['train', 'val', 'test']
                        if f'oct_{split}_classes' in self.stats
                    }
                }
            },
            'validation_results': {
                'passed': len(self.errors) == 0,
                'errors': self.errors,
                'warnings': self.warnings,
                'error_count': len(self.errors),
                'warning_count': len(self.warnings)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate CAASI dataset v6.1 structure and integrity"
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = DatasetValidator(
        dataset_path=Path(args.dataset_path),
        verbose=args.verbose
    )
    
    success = validator.validate()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()