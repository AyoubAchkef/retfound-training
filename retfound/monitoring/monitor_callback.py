"""
PyTorch Lightning Callback for Real-time Training Monitoring
===========================================================

Integrates with the RETFound trainer to send real-time metrics to the monitoring dashboard.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import threading

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from ..core.constants import CRITICAL_CONDITIONS
from .server import get_server

logger = logging.getLogger(__name__)


class MonitoringCallback(Callback):
    """PyTorch Lightning callback for real-time monitoring integration"""
    
    def __init__(
        self,
        update_frequency: int = 10,  # Update every N batches
        enable_system_monitoring: bool = True,
        critical_conditions: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize monitoring callback
        
        Args:
            update_frequency: How often to send updates (every N batches)
            enable_system_monitoring: Whether to collect system metrics
            critical_conditions: Custom critical conditions configuration
        """
        super().__init__()
        
        self.update_frequency = update_frequency
        self.enable_system_monitoring = enable_system_monitoring
        self.critical_conditions = critical_conditions or CRITICAL_CONDITIONS
        
        # State tracking
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # Metrics accumulation
        self.batch_metrics = {}
        self.epoch_metrics = {}
        
        # System monitoring
        self.last_system_update = 0
        self.system_update_interval = 5.0  # seconds
        
        # Async event loop for WebSocket updates
        self.loop = None
        self.loop_thread = None
        self._setup_async_loop()
    
    def _setup_async_loop(self):
        """Setup async event loop for WebSocket communication"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
    
    def _run_async(self, coro):
        """Run async coroutine in the background loop"""
        if self.loop and not self.loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            try:
                return future.result(timeout=1.0)
            except Exception as e:
                logger.error(f"Async operation failed: {e}")
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when training starts"""
        logger.info("Training monitoring started")
        
        # Update server status
        server = get_server()
        if server:
            self._run_async(server.update_training_status(
                'training',
                total_epochs=trainer.max_epochs,
                start_time=datetime.now()
            ))
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when training ends"""
        logger.info("Training monitoring ended")
        
        # Update server status
        server = get_server()
        if server:
            self._run_async(server.update_training_status('completed'))
        
        # Cleanup async loop
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the start of each training epoch"""
        self.current_epoch = trainer.current_epoch
        self.epoch_start_time = time.time()
        self.epoch_metrics = {}
        
        logger.debug(f"Epoch {self.current_epoch} started")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each training epoch"""
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Collect epoch-level metrics
        if hasattr(trainer, 'logged_metrics'):
            self.epoch_metrics.update(trainer.logged_metrics)
        
        # Send epoch summary
        self._send_epoch_summary(trainer, pl_module, epoch_duration)
        
        logger.debug(f"Epoch {self.current_epoch} completed in {epoch_duration:.2f}s")
    
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        """Called at the start of each training batch"""
        self.current_batch = batch_idx
        self.total_batches = trainer.num_training_batches
        self.batch_start_time = time.time()
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Called at the end of each training batch"""
        # Update batch metrics
        if hasattr(trainer, 'logged_metrics'):
            self.batch_metrics.update(trainer.logged_metrics)
        
        # Send updates at specified frequency
        if batch_idx % self.update_frequency == 0:
            self._send_batch_update(trainer, pl_module, outputs)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of validation epoch"""
        # Collect validation metrics
        val_metrics = {}
        if hasattr(trainer, 'logged_metrics'):
            val_metrics = {k: v for k, v in trainer.logged_metrics.items() if 'val' in k.lower()}
        
        # Send validation update
        self._send_validation_update(trainer, pl_module, val_metrics)
    
    def _send_batch_update(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs):
        """Send batch-level metrics update"""
        try:
            # Collect metrics
            metrics = self._collect_metrics(trainer, pl_module, outputs)
            
            # Collect system metrics
            system_metrics = self._collect_system_metrics()
            
            # Prepare update data
            update_data = {
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'total_batches': self.total_batches,
                'metrics': metrics,
                'system': system_metrics
            }
            
            # Send to monitoring server
            server = get_server()
            if server:
                self._run_async(server.update_metrics(update_data))
            
        except Exception as e:
            logger.error(f"Failed to send batch update: {e}")
    
    def _send_epoch_summary(self, trainer: pl.Trainer, pl_module: pl.LightningModule, duration: float):
        """Send epoch summary"""
        try:
            # Collect comprehensive epoch metrics
            metrics = self._collect_metrics(trainer, pl_module, None, is_epoch_end=True)
            
            # Add epoch-specific info
            metrics['epoch_duration'] = duration
            metrics['batches_per_second'] = self.total_batches / duration if duration > 0 else 0
            
            # System metrics
            system_metrics = self._collect_system_metrics()
            
            # Prepare update data
            update_data = {
                'epoch': self.current_epoch,
                'batch': self.total_batches,
                'total_batches': self.total_batches,
                'metrics': metrics,
                'system': system_metrics
            }
            
            # Send to monitoring server
            server = get_server()
            if server:
                self._run_async(server.update_metrics(update_data))
            
        except Exception as e:
            logger.error(f"Failed to send epoch summary: {e}")
    
    def _send_validation_update(self, trainer: pl.Trainer, pl_module: pl.LightningModule, val_metrics: Dict):
        """Send validation metrics update"""
        try:
            # Collect all metrics including validation
            metrics = self._collect_metrics(trainer, pl_module, None)
            metrics.update(val_metrics)
            
            # System metrics
            system_metrics = self._collect_system_metrics()
            
            # Prepare update data
            update_data = {
                'epoch': self.current_epoch,
                'batch': self.total_batches,  # End of epoch
                'total_batches': self.total_batches,
                'metrics': metrics,
                'system': system_metrics
            }
            
            # Send to monitoring server
            server = get_server()
            if server:
                self._run_async(server.update_metrics(update_data))
            
        except Exception as e:
            logger.error(f"Failed to send validation update: {e}")
    
    def _collect_metrics(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any,
        is_epoch_end: bool = False
    ) -> Dict[str, Any]:
        """Collect training metrics"""
        metrics = {}
        
        # Get logged metrics from trainer
        if hasattr(trainer, 'logged_metrics'):
            logged_metrics = trainer.logged_metrics
            
            # Organize metrics by type
            train_metrics = {}
            val_metrics = {}
            
            for key, value in logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                key_lower = key.lower()
                if 'val' in key_lower or 'validation' in key_lower:
                    clean_key = key.replace('val_', '').replace('validation_', '')
                    val_metrics[clean_key] = value
                else:
                    train_metrics[key] = value
            
            # Structure metrics
            if train_metrics:
                metrics['train'] = train_metrics
            if val_metrics:
                metrics['val'] = val_metrics
            
            # Extract key metrics for dashboard
            self._extract_key_metrics(metrics, logged_metrics)
        
        # Add batch-specific metrics
        if self.batch_metrics:
            metrics.update(self.batch_metrics)
        
        # Add learning rate
        if trainer.optimizers:
            optimizer = trainer.optimizers[0] if isinstance(trainer.optimizers, list) else trainer.optimizers
            if hasattr(optimizer, 'param_groups'):
                metrics['learning_rate'] = optimizer.param_groups[0]['lr']
        
        # Add critical conditions monitoring
        critical_metrics = self._monitor_critical_conditions(metrics)
        if critical_metrics:
            metrics['critical_conditions'] = critical_metrics
        
        return metrics
    
    def _extract_key_metrics(self, metrics: Dict, logged_metrics: Dict):
        """Extract key metrics for dashboard display"""
        # Loss
        loss_data = {}
        for key, value in logged_metrics.items():
            if 'loss' in key.lower():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                if 'val' in key.lower():
                    loss_data['val'] = value
                else:
                    loss_data['train'] = value
        
        if loss_data:
            metrics['loss'] = loss_data
        
        # Accuracy
        acc_data = {}
        for key, value in logged_metrics.items():
            if 'acc' in key.lower() or 'accuracy' in key.lower():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                if 'val' in key.lower():
                    acc_data['val'] = value
                else:
                    acc_data['train'] = value
        
        if acc_data:
            metrics['accuracy'] = acc_data
        
        # AUC-ROC
        auc_data = {}
        for key, value in logged_metrics.items():
            if 'auc' in key.lower():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                if 'macro' in key.lower():
                    auc_data['macro'] = value
                elif 'weighted' in key.lower():
                    auc_data['weighted'] = value
        
        if auc_data:
            metrics['auc_roc'] = auc_data
        
        # F1 Score
        for key, value in logged_metrics.items():
            if 'f1' in key.lower():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metrics['f1_score'] = value
                break
    
    def _monitor_critical_conditions(self, metrics: Dict) -> Dict[str, Dict]:
        """Monitor critical pathology conditions"""
        critical_metrics = {}
        
        # This would need to be integrated with the actual metrics from the model
        # For now, we'll extract from available metrics if they exist
        
        for condition, config in self.critical_conditions.items():
            threshold = config.get('min_sensitivity', 0.97)
            
            # Look for condition-specific metrics in the logged data
            condition_value = None
            
            # Check various possible metric names
            possible_keys = [
                f'{condition}_sensitivity',
                f'{condition}_recall',
                f'val_{condition}_sensitivity',
                f'val_{condition}_recall',
                f'{condition.lower()}_acc',
                f'val_{condition.lower()}_acc'
            ]
            
            for key in possible_keys:
                if key in metrics.get('val', {}):
                    condition_value = metrics['val'][key]
                    break
                elif key in metrics.get('train', {}):
                    condition_value = metrics['train'][key]
                    break
            
            if condition_value is not None:
                status = 'ok' if condition_value >= threshold else 'warning'
                if condition_value < threshold * 0.95:
                    status = 'critical'
                
                critical_metrics[condition] = {
                    'sensitivity': condition_value,
                    'threshold': threshold,
                    'status': status
                }
        
        return critical_metrics
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        if not self.enable_system_monitoring:
            return {}
        
        current_time = time.time()
        if current_time - self.last_system_update < self.system_update_interval:
            return {}
        
        self.last_system_update = current_time
        
        try:
            import psutil
            import GPUtil
            
            # GPU metrics
            gpu_stats = {
                'usage': 0,
                'memory_used': 0,
                'memory_total': 0,
                'temperature': 0
            }
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_stats = {
                        'usage': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    }
            except:
                pass
            
            # PyTorch GPU memory if available
            if torch.cuda.is_available():
                gpu_stats['torch_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
                gpu_stats['torch_memory_reserved'] = torch.cuda.memory_reserved() / 1e9
                gpu_stats['torch_max_memory_allocated'] = torch.cuda.max_memory_allocated() / 1e9
            
            # RAM usage
            memory = psutil.virtual_memory()
            
            # Calculate ETA if we have progress info
            eta_seconds = None
            if self.epoch_start_time and self.total_batches > 0 and self.current_batch > 0:
                elapsed = current_time - self.epoch_start_time
                progress = self.current_batch / self.total_batches
                if progress > 0:
                    total_estimated = elapsed / progress
                    eta_seconds = total_estimated - elapsed
            
            return {
                'gpu_usage': gpu_stats['usage'],
                'gpu_memory': gpu_stats['memory_used'],
                'gpu_memory_total': gpu_stats['memory_total'],
                'gpu_temp': gpu_stats['temperature'],
                'ram_usage': memory.percent,
                'ram_used': memory.used / 1e9,  # GB
                'ram_total': memory.total / 1e9,  # GB
                'eta_seconds': eta_seconds,
                'torch_gpu_memory': gpu_stats.get('torch_memory_allocated', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}


class RETFoundMonitoringCallback(MonitoringCallback):
    """Specialized monitoring callback for RETFound training"""
    
    def __init__(self, **kwargs):
        """Initialize RETFound-specific monitoring"""
        # Use RETFound-specific critical conditions
        super().__init__(
            critical_conditions=CRITICAL_CONDITIONS,
            **kwargs
        )
        
        # RETFound-specific tracking
        self.class_performance = {}
        self.confusion_matrix_data = None
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Enhanced validation monitoring for RETFound"""
        super().on_validation_epoch_end(trainer, pl_module)
        
        # Collect per-class performance if available
        self._collect_class_performance(trainer, pl_module)
    
    def _collect_class_performance(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Collect per-class performance metrics"""
        try:
            # This would integrate with the actual RETFound metrics
            # For now, we'll check if the module has per-class metrics
            
            if hasattr(pl_module, 'val_metrics') and hasattr(pl_module.val_metrics, 'per_class_accuracy'):
                per_class_acc = pl_module.val_metrics.per_class_accuracy
                if per_class_acc is not None:
                    self.class_performance = {
                        f'class_{i}': acc.item() if torch.is_tensor(acc) else acc
                        for i, acc in enumerate(per_class_acc)
                    }
            
            # Add to metrics update
            if self.class_performance:
                server = get_server()
                if server:
                    update_data = {
                        'epoch': self.current_epoch,
                        'batch': self.total_batches,
                        'total_batches': self.total_batches,
                        'metrics': {
                            'per_class': self.class_performance
                        },
                        'system': {}
                    }
                    self._run_async(server.update_metrics(update_data))
        
        except Exception as e:
            logger.error(f"Failed to collect class performance: {e}")


# Factory function for easy integration
def create_monitoring_callback(
    callback_type: str = 'retfound',
    **kwargs
) -> MonitoringCallback:
    """
    Create monitoring callback
    
    Args:
        callback_type: Type of callback ('basic' or 'retfound')
        **kwargs: Additional arguments for callback
    
    Returns:
        MonitoringCallback instance
    """
    if callback_type.lower() == 'retfound':
        return RETFoundMonitoringCallback(**kwargs)
    else:
        return MonitoringCallback(**kwargs)
