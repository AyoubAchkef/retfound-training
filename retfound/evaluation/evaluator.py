"""
Model evaluation utilities for RETFound.
Supports comprehensive evaluation for v6.1 dataset with 28 classes.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.constants import DATASET_V61_CLASSES, CRITICAL_CONDITIONS
from ..core.exceptions import EvaluationError
from ..data import create_datamodule
from ..metrics.medical import OphthalmologyMetrics
from ..models import load_model
from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RETFoundEvaluator:
    """Comprehensive evaluator for RETFound models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        dataset_version: str = "v6.1",
        modality: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config file
            device: Device to use for evaluation
            save_dir: Directory to save evaluation results
            dataset_version: Dataset version ("v4.0" or "v6.1")
            modality: Optional modality filter ("fundus", "oct", or None)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = device or get_device()
        self.save_dir = Path(save_dir) if save_dir else self.checkpoint_path.parent / "evaluation"
        self.dataset_version = dataset_version
        self.modality = modality
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and config
        self.model, self.config = self._load_model_and_config()
        
        # Auto-detect dataset version if not specified
        if hasattr(self.model, 'num_classes'):
            if self.model.num_classes == 28 and dataset_version == "v4.0":
                logger.warning("Model has 28 classes but dataset_version is v4.0. Switching to v6.1")
                self.dataset_version = "v6.1"
            elif self.model.num_classes == 22 and dataset_version == "v6.1":
                logger.warning("Model has 22 classes but dataset_version is v6.1. Switching to v4.0")
                self.dataset_version = "v4.0"
        
        # Initialize metrics
        num_classes = 28 if self.dataset_version == "v6.1" else 22
        self.metrics = OphthalmologyMetrics(
            num_classes=num_classes,
            dataset_version=self.dataset_version,
            modality=self.modality,
            monitor_critical=True
        )
        
        logger.info(f"Initialized evaluator for {self.dataset_version} with {num_classes} classes")
        if self.modality:
            logger.info(f"Filtering for {self.modality} modality")
    
    def _load_model_and_config(self) -> Tuple[torch.nn.Module, Dict]:
        """Load model and configuration."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Extract config
            if 'config' in checkpoint:
                config = checkpoint['config']
            elif self.config_path:
                with open(self.config_path) as f:
                    config = json.load(f)
            else:
                raise EvaluationError("No config found in checkpoint and no config_path provided")
            
            # Load model
            model = load_model(
                config=config,
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            
            return model, config
            
        except Exception as e:
            raise EvaluationError(f"Failed to load model: {str(e)}")
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        split: str = "test",
        save_predictions: bool = True,
        save_plots: bool = True,
        save_report: bool = True,
        tta: bool = False,
        temperature_scaling: bool = False
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation.
        
        Args:
            dataloader: Optional dataloader. If None, creates from config
            split: Data split to evaluate on
            save_predictions: Whether to save predictions
            save_plots: Whether to save visualization plots
            save_report: Whether to save detailed report
            tta: Whether to use test-time augmentation
            temperature_scaling: Whether to apply temperature scaling
            
        Returns:
            Dictionary of computed metrics
        """
        # Create dataloader if not provided
        if dataloader is None:
            dataloader = self._create_dataloader(split)
        
        # Run inference
        logger.info(f"Running inference on {split} set...")
        predictions, labels, probabilities, metadata = self._run_inference(
            dataloader, tta=tta
        )
        
        # Apply temperature scaling if requested
        if temperature_scaling and hasattr(self.model, 'temperature'):
            logger.info(f"Applying temperature scaling (T={self.model.temperature:.3f})")
            probabilities = self._apply_temperature_scaling(probabilities)
            predictions = probabilities.argmax(dim=1)
        
        # Update metrics
        self.metrics.reset()
        self.metrics.update(predictions, labels, probabilities, metadata)
        
        # Compute all metrics
        results = self.metrics.compute()
        
        # Save results
        if save_predictions:
            self._save_predictions(predictions, labels, probabilities, metadata, split)
        
        if save_plots:
            self._save_plots(results, split)
        
        if save_report:
            self._save_report(results, split)
        
        # Log summary
        self._log_summary(results)
        
        return results
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create dataloader for evaluation."""
        # Update config for evaluation
        eval_config = self.config.copy()
        eval_config['data']['batch_size'] = eval_config.get('eval_batch_size', 32)
        eval_config['data']['num_workers'] = eval_config.get('eval_num_workers', 4)
        
        # Set dataset version and modality
        eval_config['data']['dataset_version'] = self.dataset_version
        if self.modality:
            eval_config['data']['modality'] = self.modality
        
        # Create datamodule
        datamodule = create_datamodule(eval_config)
        datamodule.setup('test')
        
        # Get appropriate dataloader
        if split == 'train':
            return datamodule.train_dataloader()
        elif split == 'val':
            return datamodule.val_dataloader()
        else:
            return datamodule.test_dataloader()
    
    @torch.no_grad()
    def _run_inference(
        self, 
        dataloader: DataLoader,
        tta: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """Run inference on dataloader."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_metadata = []
        
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            metadata = batch.get('metadata', [{}] * len(images))
            
            if tta:
                # Test-time augmentation
                logits = self._run_tta(images)
            else:
                # Standard inference
                outputs = self.model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_predictions.append(preds)
            all_labels.append(labels)
            all_probabilities.append(probs)
            all_metadata.extend(metadata)
        
        # Concatenate all batches
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        probabilities = torch.cat(all_probabilities)
        
        return predictions, labels, probabilities, all_metadata
    
    def _run_tta(self, images: torch.Tensor) -> torch.Tensor:
        """Run test-time augmentation."""
        # Original
        outputs = self.model(images)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Horizontal flip
        outputs_hflip = self.model(torch.flip(images, dims=[3]))
        logits_hflip = outputs_hflip['logits'] if isinstance(outputs_hflip, dict) else outputs_hflip
        
        # Average logits
        return (logits + logits_hflip) / 2
    
    def _apply_temperature_scaling(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to probabilities."""
        temperature = getattr(self.model, 'temperature', 1.0)
        return F.softmax(torch.log(probabilities + 1e-10) / temperature, dim=1)
    
    def _save_predictions(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor,
        metadata: List[Dict],
        split: str
    ):
        """Save predictions to file."""
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np = probabilities.cpu().numpy()
        
        # Create results dataframe
        results_data = {
            'true_label': labels_np,
            'predicted_label': preds_np,
            'true_class': [self.metrics.class_names[l] for l in labels_np],
            'predicted_class': [self.metrics.class_names[p] for p in preds_np],
            'confidence': probs_np.max(axis=1),
            'correct': preds_np == labels_np
        }
        
        # Add top-k predictions
        top_k = min(5, self.metrics.num_classes)
        top_k_indices = np.argsort(probs_np, axis=1)[:, -top_k:][:, ::-1]
        
        for k in range(top_k):
            results_data[f'top{k+1}_class'] = [
                self.metrics.class_names[idx] for idx in top_k_indices[:, k]
            ]
            results_data[f'top{k+1}_prob'] = [
                probs_np[i, idx] for i, idx in enumerate(top_k_indices[:, k])
            ]
        
        # Add metadata if available
        if metadata and any(m for m in metadata):
            for key in metadata[0].keys():
                results_data[f'meta_{key}'] = [m.get(key, '') for m in metadata]
        
        # Create dataframe and save
        df = pd.DataFrame(results_data)
        
        # Save as CSV
        csv_path = self.save_dir / f"{split}_predictions.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")
        
        # Save raw probabilities
        np_path = self.save_dir / f"{split}_probabilities.npy"
        np.save(np_path, probs_np)
        logger.info(f"Saved probabilities to {np_path}")
        
        # Save critical conditions analysis
        if self.dataset_version == "v6.1" and len(self.metrics.critical_indices) > 0:
            self._save_critical_analysis(df, split)
    
    def _save_critical_analysis(self, df: pd.DataFrame, split: str):
        """Save analysis of critical conditions."""
        critical_data = []
        
        for condition, info in CRITICAL_CONDITIONS.items():
            if condition in self.metrics.class_names:
                class_idx = self.metrics.class_names.index(condition)
                
                # Filter for this condition
                condition_mask = df['true_label'] == class_idx
                if condition_mask.sum() > 0:
                    condition_df = df[condition_mask]
                    
                    critical_data.append({
                        'condition': condition,
                        'severity': info['severity'],
                        'urgency': info['urgency'],
                        'total_cases': len(condition_df),
                        'correctly_identified': condition_df['correct'].sum(),
                        'sensitivity': condition_df['correct'].mean(),
                        'min_sensitivity_threshold': info['min_sensitivity'],
                        'meets_threshold': condition_df['correct'].mean() >= info['min_sensitivity'],
                        'missed_cases': len(condition_df[~condition_df['correct']]),
                        'avg_confidence_when_correct': condition_df[condition_df['correct']]['confidence'].mean(),
                        'avg_confidence_when_wrong': condition_df[~condition_df['correct']]['confidence'].mean() if (~condition_df['correct']).sum() > 0 else 0
                    })
        
        if critical_data:
            critical_df = pd.DataFrame(critical_data)
            critical_path = self.save_dir / f"{split}_critical_conditions_analysis.csv"
            critical_df.to_csv(critical_path, index=False)
            logger.info(f"Saved critical conditions analysis to {critical_path}")
    
    def _save_plots(self, results: Dict[str, float], split: str):
        """Save visualization plots."""
        # Confusion matrix
        self._plot_confusion_matrix(split)
        
        # Per-class metrics
        self._plot_per_class_metrics(results, split)
        
        # Critical conditions performance
        if self.dataset_version == "v6.1":
            self._plot_critical_performance(results, split)
        
        # ROC curves (if AUC metrics available)
        if any('auc_' in key for key in results.keys()):
            self._plot_roc_curves(split)
    
    def _plot_confusion_matrix(self, split: str):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(20, 16))
        
        cm = self.metrics.get_confusion_matrix(normalize='true')
        class_names = [self.metrics.class_names[i] for i in self.metrics.active_classes]
        
        # Use appropriate colormap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title(f'Confusion Matrix - {split} set\nDataset {self.dataset_version}', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = self.save_dir / f"{split}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {cm_path}")
    
    def _plot_per_class_metrics(self, results: Dict[str, float], split: str):
        """Plot per-class precision, recall, and F1."""
        # Extract per-class metrics
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for i, class_idx in enumerate(self.metrics.active_classes):
            class_name = self.metrics.class_names[class_idx]
            if f'precision_{class_name}' in results:
                classes.append(class_name)
                precisions.append(results[f'precision_{class_name}'])
                recalls.append(results[f'recall_{class_name}'])
                f1_scores.append(results[f'f1_{class_name}'])
        
        if not classes:
            return
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))
        
        # Precision
        bars1 = ax1.bar(range(len(classes)), precisions, color='skyblue')
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title(f'Per-Class Precision - {split} set', fontsize=14)
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=np.mean(precisions), color='red', linestyle='--', label=f'Mean: {np.mean(precisions):.3f}')
        ax1.legend()
        
        # Add value labels
        for bar, value in zip(bars1, precisions):
            height = bar.get_height()
            ax1.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        # Recall
        bars2 = ax2.bar(range(len(classes)), recalls, color='lightgreen')
        ax2.set_ylabel('Recall (Sensitivity)', fontsize=12)
        ax2.set_title(f'Per-Class Recall - {split} set', fontsize=14)
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=np.mean(recalls), color='red', linestyle='--', label=f'Mean: {np.mean(recalls):.3f}')
        ax2.legend()
        
        # Add value labels
        for bar, value in zip(bars2, recalls):
            height = bar.get_height()
            ax2.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        # F1-Score
        bars3 = ax3.bar(range(len(classes)), f1_scores, color='salmon')
        ax3.set_ylabel('F1-Score', fontsize=12)
        ax3.set_title(f'Per-Class F1-Score - {split} set', fontsize=14)
        ax3.set_ylim(0, 1.1)
        ax3.axhline(y=np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.legend()
        
        # Add value labels
        for bar, value in zip(bars3, f1_scores):
            height = bar.get_height()
            ax3.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        plt.tight_layout()
        
        metrics_path = self.save_dir / f"{split}_per_class_metrics.png"
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved per-class metrics to {metrics_path}")
    
    def _plot_critical_performance(self, results: Dict[str, float], split: str):
        """Plot critical conditions performance."""
        critical_conditions = []
        sensitivities = []
        thresholds = []
        meets_threshold = []
        
        for condition, info in CRITICAL_CONDITIONS.items():
            key = f'critical_{condition}_sensitivity'
            if key in results:
                critical_conditions.append(condition)
                sensitivities.append(results[key])
                thresholds.append(info['min_sensitivity'])
                meets_threshold.append(results[f'critical_{condition}_meets_threshold'])
        
        if not critical_conditions:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(critical_conditions))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, sensitivities, width, label='Actual Sensitivity',
                       color=['green' if m else 'red' for m in meets_threshold])
        bars2 = ax.bar(x + width/2, thresholds, width, label='Required Threshold',
                       color='gray', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars1, sensitivities):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_xlabel('Critical Conditions', fontsize=12)
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.set_title(f'Critical Conditions Performance - {split} set\nGreen: Meets threshold, Red: Below threshold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(critical_conditions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        critical_path = self.save_dir / f"{split}_critical_performance.png"
        plt.savefig(critical_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved critical performance plot to {critical_path}")
    
    def _plot_roc_curves(self, split: str):
        """Plot ROC curves (placeholder - implement if needed)."""
        # This would require storing the raw probabilities and labels
        # Implementation depends on specific requirements
        pass
    
    def _save_report(self, results: Dict[str, float], split: str):
        """Save detailed evaluation report."""
        report_path = self.save_dir / f"{split}_evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"RETFound Evaluation Report - {split} set\n")
            f.write("=" * 80 + "\n\n")
            
            # Model information
            f.write("Model Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Dataset Version: {self.dataset_version}\n")
            f.write(f"Number of Classes: {self.metrics.num_classes}\n")
            if self.modality:
                f.write(f"Modality Filter: {self.modality}\n")
            f.write("\n")
            
            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results.get('accuracy', 0):.4f}\n")
            f.write(f"Balanced Accuracy: {results.get('balanced_accuracy', 0):.4f}\n")
            f.write(f"Cohen's Kappa: {results.get('cohen_kappa', 0):.4f}\n")
            if 'quadratic_kappa' in results:
                f.write(f"Quadratic Kappa (DR): {results['quadratic_kappa']:.4f}\n")
            f.write(f"Macro F1: {results.get('f1_macro', 0):.4f}\n")
            f.write(f"Weighted F1: {results.get('f1_weighted', 0):.4f}\n")
            if 'auc_macro' in results:
                f.write(f"Macro AUC: {results['auc_macro']:.4f}\n")
            if 'auc_weighted' in results:
                f.write(f"Weighted AUC: {results['auc_weighted']:.4f}\n")
            f.write("\n")
            
            # Modality-specific metrics (v6.1)
            if self.dataset_version == "v6.1" and self.modality is None:
                f.write("Modality-Specific Metrics:\n")
                f.write("-" * 40 + "\n")
                if 'fundus_accuracy' in results:
                    f.write(f"Fundus Accuracy: {results['fundus_accuracy']:.4f}\n")
                    f.write(f"Fundus Correct Modality: {results.get('fundus_correct_modality', 0):.4f}\n")
                if 'oct_accuracy' in results:
                    f.write(f"OCT Accuracy: {results['oct_accuracy']:.4f}\n")
                    f.write(f"OCT Correct Modality: {results.get('oct_correct_modality', 0):.4f}\n")
                f.write("\n")
            
            # Critical conditions (v6.1)
            if self.dataset_version == "v6.1" and any('critical_' in k for k in results.keys()):
                f.write("Critical Conditions Performance:\n")
                f.write("-" * 40 + "\n")
                
                for condition in CRITICAL_CONDITIONS:
                    sens_key = f'critical_{condition}_sensitivity'
                    if sens_key in results:
                        sensitivity = results[sens_key]
                        threshold = CRITICAL_CONDITIONS[condition]['min_sensitivity']
                        meets = results[f'critical_{condition}_meets_threshold']
                        status = "✓ PASS" if meets else "✗ FAIL"
                        
                        f.write(f"{condition}: {sensitivity:.3f} (threshold: {threshold:.2f}) {status}\n")
                
                if 'critical_avg_sensitivity' in results:
                    f.write(f"\nAverage Critical Sensitivity: {results['critical_avg_sensitivity']:.4f}\n")
                if 'critical_min_sensitivity' in results:
                    f.write(f"Minimum Critical Sensitivity: {results['critical_min_sensitivity']:.4f}\n")
                f.write("\n")
            
            # Classification report
            f.write("Detailed Classification Report:\n")
            f.write("-" * 40 + "\n")
            f.write(self.metrics.get_classification_report())
            f.write("\n")
            
            # Save JSON version
            json_path = self.save_dir / f"{split}_evaluation_results.json"
            with open(json_path, 'w') as json_file:
                json.dump(results, json_file, indent=2)
            
        logger.info(f"Saved evaluation report to {report_path}")
        logger.info(f"Saved evaluation results to {json_path}")
    
    def _log_summary(self, results: Dict[str, float]):
        """Log evaluation summary."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {results.get('accuracy', 0):.4f}")
        logger.info(f"Balanced Accuracy: {results.get('balanced_accuracy', 0):.4f}")
        logger.info(f"Macro F1: {results.get('f1_macro', 0):.4f}")
        logger.info(f"Cohen's Kappa: {results.get('cohen_kappa', 0):.4f}")
        
        if 'auc_macro' in results:
            logger.info(f"Macro AUC: {results['auc_macro']:.4f}")
        
        # Critical conditions summary
        if self.dataset_version == "v6.1" and 'critical_avg_sensitivity' in results:
            logger.info("-" * 60)
            logger.info("CRITICAL CONDITIONS")
            logger.info("-" * 60)
            
            all_pass = True
            for condition in CRITICAL_CONDITIONS:
                sens_key = f'critical_{condition}_sensitivity'
                if sens_key in results:
                    meets = results[f'critical_{condition}_meets_threshold']
                    status = "✓" if meets else "✗"
                    logger.info(f"{status} {condition}: {results[sens_key]:.3f}")
                    if not meets:
                        all_pass = False
            
            if all_pass:
                logger.info("✓ All critical conditions meet sensitivity thresholds!")
            else:
                logger.warning("✗ Some critical conditions below threshold!")
        
        logger.info("=" * 60)