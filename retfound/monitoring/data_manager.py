"""
Data Manager for Training Monitoring
===================================

Manages training metrics data, caching, and history for the monitoring dashboard.
Optimized for real-time updates with rolling windows and efficient storage.
"""

import asyncio
import json
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Deque
import sqlite3
import aiosqlite
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Single metrics snapshot"""
    timestamp: datetime
    epoch: int
    batch: int
    total_batches: int
    metrics: Dict[str, Any]
    system: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'epoch': self.epoch,
            'batch': self.batch,
            'total_batches': self.total_batches,
            'metrics': self.metrics,
            'system': self.system
        }


class RollingBuffer:
    """Efficient rolling buffer for metrics with automatic downsampling"""
    
    def __init__(self, max_size: int = 1000, downsample_factor: int = 10):
        self.max_size = max_size
        self.downsample_factor = downsample_factor
        self.buffer: Deque[MetricsSnapshot] = deque(maxlen=max_size)
        self.full_buffer: Deque[MetricsSnapshot] = deque(maxlen=max_size * downsample_factor)
        self.sample_counter = 0
    
    def add(self, snapshot: MetricsSnapshot):
        """Add new snapshot with automatic downsampling"""
        # Always add to full buffer
        self.full_buffer.append(snapshot)
        
        # Add to main buffer with downsampling
        self.sample_counter += 1
        if self.sample_counter >= self.downsample_factor:
            self.buffer.append(snapshot)
            self.sample_counter = 0
        elif len(self.buffer) == 0:  # Always keep first sample
            self.buffer.append(snapshot)
    
    def get_recent(self, limit: int = 100) -> List[MetricsSnapshot]:
        """Get recent snapshots"""
        return list(self.buffer)[-limit:]
    
    def get_all(self) -> List[MetricsSnapshot]:
        """Get all snapshots in buffer"""
        return list(self.buffer)
    
    def get_latest(self) -> Optional[MetricsSnapshot]:
        """Get latest snapshot"""
        return self.buffer[-1] if self.buffer else None


class DataManager:
    """Manages training metrics data with efficient storage and retrieval"""
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_memory_samples: int = 1000,
        enable_persistence: bool = True
    ):
        self.db_path = db_path or Path("monitoring_data.db")
        self.max_memory_samples = max_memory_samples
        self.enable_persistence = enable_persistence
        
        # In-memory storage for real-time access
        self.metrics_buffer = RollingBuffer(max_memory_samples)
        self.epoch_data: Dict[int, List[MetricsSnapshot]] = defaultdict(list)
        
        # Aggregated metrics for dashboard
        self.latest_metrics: Optional[Dict[str, Any]] = None
        self.epoch_summaries: Dict[int, Dict[str, Any]] = {}
        
        # Critical conditions tracking
        self.critical_alerts: List[Dict[str, Any]] = []
        
        # Database connection
        self.db_connection: Optional[aiosqlite.Connection] = None
        
        # Performance tracking
        self.performance_history = {
            'loss': {'train': [], 'val': []},
            'accuracy': {'train': [], 'val': []},
            'auc_roc': {'macro': [], 'weighted': []},
            'f1_score': [],
            'learning_rate': [],
            'critical_conditions': defaultdict(list)
        }
    
    async def initialize(self):
        """Initialize data manager and database"""
        if self.enable_persistence:
            await self._init_database()
        
        logger.info("Data manager initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db_connection:
            await self.db_connection.close()
        
        logger.info("Data manager cleaned up")
    
    async def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            self.db_connection = await aiosqlite.connect(str(self.db_path))
            
            # Create tables
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    batch INTEGER NOT NULL,
                    total_batches INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    system_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS epoch_summaries (
                    epoch INTEGER PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS critical_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    condition_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    alert_level TEXT NOT NULL,
                    message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_epoch_batch 
                ON metrics_snapshots(epoch, batch)
            """)
            
            await self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics_snapshots(timestamp)
            """)
            
            await self.db_connection.commit()
            
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.enable_persistence = False
    
    async def add_metrics(self, metrics_data: Dict[str, Any]):
        """Add new metrics snapshot"""
        try:
            # Create snapshot
            snapshot = MetricsSnapshot(
                timestamp=datetime.now(),
                epoch=metrics_data.get('epoch', 0),
                batch=metrics_data.get('batch', 0),
                total_batches=metrics_data.get('total_batches', 0),
                metrics=metrics_data.get('metrics', {}),
                system=metrics_data.get('system', {})
            )
            
            # Add to memory buffer
            self.metrics_buffer.add(snapshot)
            
            # Add to epoch data
            self.epoch_data[snapshot.epoch].append(snapshot)
            
            # Update latest metrics
            self.latest_metrics = snapshot.metrics
            
            # Update performance history
            await self._update_performance_history(snapshot)
            
            # Check for critical alerts
            await self._check_critical_conditions(snapshot)
            
            # Persist to database
            if self.enable_persistence and self.db_connection:
                await self._persist_snapshot(snapshot)
            
            # Update epoch summary if batch is complete
            if snapshot.batch == snapshot.total_batches:
                await self._update_epoch_summary(snapshot.epoch)
            
        except Exception as e:
            logger.error(f"Error adding metrics: {e}")
    
    async def _update_performance_history(self, snapshot: MetricsSnapshot):
        """Update performance history for charts"""
        metrics = snapshot.metrics
        
        # Loss curves
        if 'loss' in metrics:
            if isinstance(metrics['loss'], dict):
                if 'train' in metrics['loss']:
                    self.performance_history['loss']['train'].append({
                        'epoch': snapshot.epoch,
                        'batch': snapshot.batch,
                        'value': metrics['loss']['train'],
                        'timestamp': snapshot.timestamp.isoformat()
                    })
                if 'val' in metrics['loss']:
                    self.performance_history['loss']['val'].append({
                        'epoch': snapshot.epoch,
                        'batch': snapshot.batch,
                        'value': metrics['loss']['val'],
                        'timestamp': snapshot.timestamp.isoformat()
                    })
            else:
                # Single loss value (assume training)
                self.performance_history['loss']['train'].append({
                    'epoch': snapshot.epoch,
                    'batch': snapshot.batch,
                    'value': metrics['loss'],
                    'timestamp': snapshot.timestamp.isoformat()
                })
        
        # Accuracy
        if 'accuracy' in metrics:
            if isinstance(metrics['accuracy'], dict):
                for split in ['train', 'val']:
                    if split in metrics['accuracy']:
                        self.performance_history['accuracy'][split].append({
                            'epoch': snapshot.epoch,
                            'batch': snapshot.batch,
                            'value': metrics['accuracy'][split],
                            'timestamp': snapshot.timestamp.isoformat()
                        })
        
        # AUC-ROC
        if 'auc_roc' in metrics:
            if isinstance(metrics['auc_roc'], dict):
                for metric_type in ['macro', 'weighted']:
                    if metric_type in metrics['auc_roc']:
                        self.performance_history['auc_roc'][metric_type].append({
                            'epoch': snapshot.epoch,
                            'batch': snapshot.batch,
                            'value': metrics['auc_roc'][metric_type],
                            'timestamp': snapshot.timestamp.isoformat()
                        })
        
        # F1 Score
        if 'f1_score' in metrics:
            self.performance_history['f1_score'].append({
                'epoch': snapshot.epoch,
                'batch': snapshot.batch,
                'value': metrics['f1_score'],
                'timestamp': snapshot.timestamp.isoformat()
            })
        
        # Learning Rate
        if 'learning_rate' in metrics:
            self.performance_history['learning_rate'].append({
                'epoch': snapshot.epoch,
                'batch': snapshot.batch,
                'value': metrics['learning_rate'],
                'timestamp': snapshot.timestamp.isoformat()
            })
        
        # Critical conditions
        if 'critical_conditions' in metrics:
            for condition, data in metrics['critical_conditions'].items():
                if isinstance(data, dict) and 'sensitivity' in data:
                    self.performance_history['critical_conditions'][condition].append({
                        'epoch': snapshot.epoch,
                        'batch': snapshot.batch,
                        'sensitivity': data['sensitivity'],
                        'threshold': data.get('threshold', 0.97),
                        'status': data.get('status', 'unknown'),
                        'timestamp': snapshot.timestamp.isoformat()
                    })
        
        # Limit history size to prevent memory issues
        max_history = 10000
        for category in self.performance_history:
            if isinstance(self.performance_history[category], list):
                if len(self.performance_history[category]) > max_history:
                    self.performance_history[category] = self.performance_history[category][-max_history:]
            elif isinstance(self.performance_history[category], dict):
                for subcategory in self.performance_history[category]:
                    if isinstance(self.performance_history[category][subcategory], list):
                        if len(self.performance_history[category][subcategory]) > max_history:
                            self.performance_history[category][subcategory] = \
                                self.performance_history[category][subcategory][-max_history:]
    
    async def _check_critical_conditions(self, snapshot: MetricsSnapshot):
        """Check for critical condition alerts"""
        if 'critical_conditions' not in snapshot.metrics:
            return
        
        critical_conditions = snapshot.metrics['critical_conditions']
        
        for condition, data in critical_conditions.items():
            if not isinstance(data, dict):
                continue
            
            sensitivity = data.get('sensitivity', 0)
            threshold = data.get('threshold', 0.97)
            status = data.get('status', 'unknown')
            
            # Create alert if below threshold
            if sensitivity < threshold:
                alert = {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'condition': condition,
                    'current_value': sensitivity,
                    'threshold': threshold,
                    'level': 'critical' if sensitivity < threshold * 0.95 else 'warning',
                    'message': f"{condition} sensitivity ({sensitivity:.3f}) below threshold ({threshold:.3f})",
                    'epoch': snapshot.epoch,
                    'batch': snapshot.batch
                }
                
                self.critical_alerts.append(alert)
                
                # Persist alert
                if self.enable_persistence and self.db_connection:
                    await self._persist_alert(alert)
                
                logger.warning(f"Critical alert: {alert['message']}")
        
        # Limit alerts history
        if len(self.critical_alerts) > 1000:
            self.critical_alerts = self.critical_alerts[-1000:]
    
    async def _persist_snapshot(self, snapshot: MetricsSnapshot):
        """Persist snapshot to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO metrics_snapshots 
                (timestamp, epoch, batch, total_batches, metrics_json, system_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                snapshot.epoch,
                snapshot.batch,
                snapshot.total_batches,
                json.dumps(snapshot.metrics),
                json.dumps(snapshot.system)
            ))
            
            await self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error persisting snapshot: {e}")
    
    async def _persist_alert(self, alert: Dict[str, Any]):
        """Persist critical alert to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO critical_alerts 
                (timestamp, condition_name, current_value, threshold_value, alert_level, message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert['timestamp'],
                alert['condition'],
                alert['current_value'],
                alert['threshold'],
                alert['level'],
                alert['message']
            ))
            
            await self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error persisting alert: {e}")
    
    async def _update_epoch_summary(self, epoch: int):
        """Update epoch summary statistics"""
        if epoch not in self.epoch_data:
            return
        
        epoch_snapshots = self.epoch_data[epoch]
        if not epoch_snapshots:
            return
        
        # Calculate summary statistics
        summary = {
            'epoch': epoch,
            'total_batches': epoch_snapshots[-1].total_batches,
            'start_time': epoch_snapshots[0].timestamp.isoformat(),
            'end_time': epoch_snapshots[-1].timestamp.isoformat(),
            'duration': (epoch_snapshots[-1].timestamp - epoch_snapshots[0].timestamp).total_seconds(),
            'metrics': {}
        }
        
        # Aggregate metrics
        final_metrics = epoch_snapshots[-1].metrics
        summary['metrics'] = final_metrics.copy()
        
        # Calculate averages for some metrics
        if len(epoch_snapshots) > 1:
            # Average loss during epoch
            train_losses = [s.metrics.get('loss', {}).get('train', 0) for s in epoch_snapshots 
                          if 'loss' in s.metrics and isinstance(s.metrics['loss'], dict)]
            if train_losses:
                summary['metrics']['avg_train_loss'] = np.mean(train_losses)
        
        self.epoch_summaries[epoch] = summary
        
        # Persist summary
        if self.enable_persistence and self.db_connection:
            try:
                await self.db_connection.execute("""
                    INSERT OR REPLACE INTO epoch_summaries (epoch, summary_json)
                    VALUES (?, ?)
                """, (epoch, json.dumps(summary, default=str)))
                
                await self.db_connection.commit()
                
            except Exception as e:
                logger.error(f"Error persisting epoch summary: {e}")
    
    # Public API methods
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest metrics snapshot"""
        latest = self.metrics_buffer.get_latest()
        return latest.to_dict() if latest else None
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics history"""
        snapshots = self.metrics_buffer.get_recent(limit)
        return [s.to_dict() for s in snapshots]
    
    def get_epoch_details(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for specific epoch"""
        if epoch in self.epoch_summaries:
            return self.epoch_summaries[epoch]
        
        if epoch in self.epoch_data:
            snapshots = self.epoch_data[epoch]
            return {
                'epoch': epoch,
                'snapshots': [s.to_dict() for s in snapshots],
                'total_batches': snapshots[-1].total_batches if snapshots else 0
            }
        
        return None
    
    def get_performance_history(self, metric: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get performance history for specific metric"""
        if metric in self.performance_history:
            data = self.performance_history[metric]
            if isinstance(data, list):
                return data[-limit:]
            elif isinstance(data, dict):
                return {k: v[-limit:] if isinstance(v, list) else v for k, v in data.items()}
        
        return []
    
    def get_critical_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent critical alerts"""
        return self.critical_alerts[-limit:]
    
    def get_epoch_list(self) -> List[int]:
        """Get list of available epochs"""
        return sorted(list(self.epoch_data.keys()))
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        latest = self.metrics_buffer.get_latest()
        
        summary = {
            'latest_metrics': latest.to_dict() if latest else None,
            'total_epochs': len(self.epoch_data),
            'total_snapshots': len(self.metrics_buffer.get_all()),
            'critical_alerts_count': len(self.critical_alerts),
            'recent_alerts': self.get_critical_alerts(10),
            'performance_trends': {}
        }
        
        # Add performance trends
        for metric in ['loss', 'accuracy', 'auc_roc']:
            if metric in self.performance_history:
                data = self.performance_history[metric]
                if isinstance(data, dict):
                    summary['performance_trends'][metric] = {}
                    for submetric, values in data.items():
                        if isinstance(values, list) and values:
                            recent_values = values[-10:]  # Last 10 points
                            summary['performance_trends'][metric][submetric] = {
                                'current': recent_values[-1]['value'] if recent_values else 0,
                                'trend': 'improving' if len(recent_values) > 1 and 
                                        recent_values[-1]['value'] > recent_values[0]['value'] else 'stable'
                            }
        
        return summary
    
    async def export_data(self, filepath: Path, format: str = 'json'):
        """Export all data to file"""
        try:
            data = {
                'metrics_history': self.get_metrics_history(limit=10000),
                'epoch_summaries': self.epoch_summaries,
                'performance_history': self.performance_history,
                'critical_alerts': self.critical_alerts,
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
    
    async def clear_data(self, keep_recent: int = 100):
        """Clear old data, keeping recent entries"""
        try:
            # Clear memory buffers
            recent_snapshots = self.metrics_buffer.get_recent(keep_recent)
            self.metrics_buffer = RollingBuffer(self.max_memory_samples)
            for snapshot in recent_snapshots:
                self.metrics_buffer.add(snapshot)
            
            # Clear epoch data (keep recent epochs)
            if self.epoch_data:
                recent_epochs = sorted(self.epoch_data.keys())[-10:]  # Keep last 10 epochs
                new_epoch_data = defaultdict(list)
                for epoch in recent_epochs:
                    new_epoch_data[epoch] = self.epoch_data[epoch]
                self.epoch_data = new_epoch_data
            
            # Clear alerts (keep recent)
            self.critical_alerts = self.critical_alerts[-keep_recent:]
            
            # Trim performance history
            for category in self.performance_history:
                if isinstance(self.performance_history[category], list):
                    self.performance_history[category] = self.performance_history[category][-keep_recent:]
                elif isinstance(self.performance_history[category], dict):
                    for subcategory in self.performance_history[category]:
                        if isinstance(self.performance_history[category][subcategory], list):
                            self.performance_history[category][subcategory] = \
                                self.performance_history[category][subcategory][-keep_recent:]
            
            logger.info(f"Data cleared, kept {keep_recent} recent entries")
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
