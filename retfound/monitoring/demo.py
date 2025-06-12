#!/usr/bin/env python3
"""
RETFound Training Monitoring Demo
================================

Demonstrates the complete monitoring system with simulated training data.
Run this script to see the dashboard in action with realistic metrics.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import monitoring components
try:
    from .server import create_server
    from .data_manager import DataManager
    from .monitor_callback import create_monitoring_callback
except ImportError:
    # Handle relative imports when running as script
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from monitoring.server import create_server
    from monitoring.data_manager import DataManager
    from monitoring.monitor_callback import create_monitoring_callback


class TrainingSimulator:
    """Simulates RETFound training with realistic metrics"""
    
    def __init__(self, server):
        self.server = server
        self.epoch = 0
        self.total_epochs = 50
        self.batch = 0
        self.total_batches = 1000
        self.start_time = time.time()
        
        # Initial metrics
        self.train_loss = 2.5
        self.val_loss = 2.8
        self.train_acc = 0.15
        self.val_acc = 0.12
        self.learning_rate = 1e-4
        
        # Critical conditions (RAO, RVO, Retinal_Detachment)
        self.critical_conditions = {
            'RAO': {'sensitivity': 0.95, 'threshold': 0.99},
            'RVO': {'sensitivity': 0.96, 'threshold': 0.97},
            'Retinal_Detachment': {'sensitivity': 0.98, 'threshold': 0.99}
        }
        
        # Per-class performance (28 classes)
        self.class_names = [
            'Normal', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM',
            'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE',
            'ST', 'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP'
        ]
        self.class_performance = {name: random.uniform(0.7, 0.95) for name in self.class_names}
    
    async def simulate_training(self):
        """Simulate complete training process"""
        logger.info("Starting training simulation...")
        
        # Update server status
        await self.server.update_training_status(
            'training',
            total_epochs=self.total_epochs,
            start_time=datetime.now()
        )
        
        for epoch in range(self.total_epochs):
            self.epoch = epoch
            await self.simulate_epoch()
            
            # Small delay between epochs
            await asyncio.sleep(0.5)
        
        # Training completed
        await self.server.update_training_status('completed')
        logger.info("Training simulation completed!")
    
    async def simulate_epoch(self):
        """Simulate one training epoch"""
        logger.info(f"Simulating epoch {self.epoch + 1}/{self.total_epochs}")
        
        for batch in range(0, self.total_batches, 50):  # Update every 50 batches
            self.batch = batch
            await self.simulate_batch()
            
            # Small delay between batch updates
            await asyncio.sleep(0.1)
        
        # End of epoch validation
        await self.simulate_validation()
    
    async def simulate_batch(self):
        """Simulate batch training with realistic metric evolution"""
        # Evolve metrics realistically
        self._evolve_metrics()
        
        # Create metrics update
        metrics_data = {
            'epoch': self.epoch,
            'batch': self.batch,
            'total_batches': self.total_batches,
            'metrics': {
                'loss': {
                    'train': self.train_loss,
                    'val': self.val_loss
                },
                'accuracy': {
                    'train': self.train_acc,
                    'val': self.val_acc
                },
                'auc_roc': {
                    'macro': random.uniform(0.85, 0.95),
                    'weighted': random.uniform(0.87, 0.97)
                },
                'f1_score': random.uniform(0.80, 0.92),
                'learning_rate': self.learning_rate,
                'critical_conditions': self._get_critical_conditions(),
                'per_class': self.class_performance
            },
            'system': self._get_system_metrics()
        }
        
        # Send to monitoring server
        await self.server.update_metrics(metrics_data)
    
    async def simulate_validation(self):
        """Simulate end-of-epoch validation"""
        # Update validation metrics
        self.val_loss = self.train_loss + random.uniform(0.1, 0.3)
        self.val_acc = self.train_acc - random.uniform(0.02, 0.08)
        
        # Update critical conditions
        for condition in self.critical_conditions:
            # Gradually improve critical conditions
            current = self.critical_conditions[condition]['sensitivity']
            target = self.critical_conditions[condition]['threshold']
            improvement = (target - current) * 0.1 + random.uniform(-0.01, 0.02)
            self.critical_conditions[condition]['sensitivity'] = min(target + 0.01, current + improvement)
        
        # Send validation update
        await self.simulate_batch()
    
    def _evolve_metrics(self):
        """Evolve training metrics realistically over time"""
        # Loss decreases with some noise
        loss_decay = 0.995
        self.train_loss *= loss_decay
        self.train_loss += random.uniform(-0.01, 0.005)
        self.train_loss = max(0.1, self.train_loss)  # Don't go below 0.1
        
        # Accuracy increases with some noise
        acc_growth = 1.002
        self.train_acc *= acc_growth
        self.train_acc += random.uniform(-0.005, 0.01)
        self.train_acc = min(0.98, self.train_acc)  # Cap at 98%
        
        # Learning rate decay
        if self.epoch > 0 and self.epoch % 10 == 0 and self.batch == 0:
            self.learning_rate *= 0.5
        
        # Evolve per-class performance
        for class_name in self.class_performance:
            change = random.uniform(-0.01, 0.02)
            self.class_performance[class_name] = max(0.5, min(0.99, 
                self.class_performance[class_name] + change))
    
    def _get_critical_conditions(self):
        """Get current critical conditions status"""
        conditions = {}
        
        for condition, data in self.critical_conditions.items():
            sensitivity = data['sensitivity']
            threshold = data['threshold']
            
            if sensitivity >= threshold:
                status = 'ok'
            elif sensitivity >= threshold * 0.95:
                status = 'warning'
            else:
                status = 'critical'
            
            conditions[condition] = {
                'sensitivity': sensitivity,
                'threshold': threshold,
                'status': status
            }
        
        return conditions
    
    def _get_system_metrics(self):
        """Generate realistic system metrics"""
        # Simulate GPU usage based on training phase
        base_gpu_usage = 85 + random.uniform(-10, 10)
        gpu_memory_used = 20 + random.uniform(-2, 3)  # GB
        gpu_memory_total = 24  # GB
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.epoch > 0:
            time_per_epoch = elapsed / (self.epoch + 1)
            remaining_epochs = self.total_epochs - self.epoch - 1
            eta_seconds = remaining_epochs * time_per_epoch
        else:
            eta_seconds = None
        
        return {
            'gpu_usage': max(0, min(100, base_gpu_usage)),
            'gpu_memory': gpu_memory_used,
            'gpu_memory_total': gpu_memory_total,
            'gpu_temp': 65 + random.uniform(-5, 10),
            'ram_usage': 45 + random.uniform(-5, 10),
            'ram_used': 28 + random.uniform(-2, 4),
            'ram_total': 64,
            'eta_seconds': eta_seconds,
            'torch_gpu_memory': gpu_memory_used * 0.9
        }


async def run_demo():
    """Run the complete monitoring demo"""
    print("🚀 Starting RETFound Training Monitoring Demo")
    print("=" * 50)
    
    # Create monitoring server
    frontend_dir = Path(__file__).parent / "frontend" / "dist"
    server = create_server(
        host="localhost",
        port=8000,
        frontend_dir=frontend_dir if frontend_dir.exists() else None
    )
    
    # Create training simulator
    simulator = TrainingSimulator(server)
    
    # Start server in background
    server_task = asyncio.create_task(server.run_async())
    
    # Wait a moment for server to start
    await asyncio.sleep(2)
    
    print(f"📊 Dashboard available at: http://localhost:8000")
    print(f"🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print(f"📡 API endpoints: http://localhost:8000/api/")
    print()
    print("🎯 Demo Features:")
    print("  • Real-time metrics updates")
    print("  • Critical pathology monitoring (RAO, RVO, Retinal Detachment)")
    print("  • 28-class performance tracking")
    print("  • GPU/System monitoring")
    print("  • Interactive charts and visualizations")
    print()
    print("⏳ Starting training simulation...")
    
    try:
        # Run training simulation
        await simulator.simulate_training()
        
        print("\n✅ Training simulation completed!")
        print("🔄 Server will continue running for dashboard exploration...")
        print("📝 Press Ctrl+C to stop the server")
        
        # Keep server running
        await server_task
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo failed")
    finally:
        # Cleanup
        if not server_task.done():
            server_task.cancel()


def main():
    """Main entry point"""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n👋 Demo terminated")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        logger.exception("Fatal error in demo")


if __name__ == "__main__":
    main()
