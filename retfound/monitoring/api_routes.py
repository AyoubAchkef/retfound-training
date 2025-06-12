"""
REST API Routes for Training Monitoring
======================================

Provides REST endpoints for the monitoring dashboard.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from .data_manager import DataManager

logger = logging.getLogger(__name__)


class ExportRequest(BaseModel):
    """Export request model"""
    format: str = 'json'
    include_history: bool = True
    include_alerts: bool = True
    limit: Optional[int] = None


class MetricsQuery(BaseModel):
    """Metrics query model"""
    metric: str
    start_epoch: Optional[int] = None
    end_epoch: Optional[int] = None
    limit: int = 1000


def create_api_routes(data_manager: DataManager) -> APIRouter:
    """Create API router with all monitoring endpoints"""
    
    router = APIRouter()
    
    @router.get("/metrics/latest")
    async def get_latest_metrics():
        """Get latest metrics snapshot"""
        metrics = data_manager.get_latest_metrics()
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics available")
        
        return {
            "status": "success",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/metrics/history")
    async def get_metrics_history(
        limit: int = Query(100, ge=1, le=10000, description="Number of recent snapshots to return")
    ):
        """Get metrics history"""
        history = data_manager.get_metrics_history(limit=limit)
        
        return {
            "status": "success",
            "data": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/metrics/performance/{metric}")
    async def get_performance_history(
        metric: str,
        limit: int = Query(1000, ge=1, le=10000, description="Number of data points to return")
    ):
        """Get performance history for specific metric"""
        valid_metrics = ['loss', 'accuracy', 'auc_roc', 'f1_score', 'learning_rate', 'critical_conditions']
        
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid metric. Valid options: {valid_metrics}"
            )
        
        history = data_manager.get_performance_history(metric, limit=limit)
        
        return {
            "status": "success",
            "metric": metric,
            "data": history,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/epochs")
    async def get_epochs():
        """Get list of available epochs"""
        epochs = data_manager.get_epoch_list()
        
        return {
            "status": "success",
            "epochs": epochs,
            "count": len(epochs),
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/epochs/{epoch}")
    async def get_epoch_details(epoch: int):
        """Get detailed information for specific epoch"""
        details = data_manager.get_epoch_details(epoch)
        
        if not details:
            raise HTTPException(status_code=404, detail=f"Epoch {epoch} not found")
        
        return {
            "status": "success",
            "epoch": epoch,
            "data": details,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/alerts")
    async def get_critical_alerts(
        limit: int = Query(100, ge=1, le=1000, description="Number of recent alerts to return")
    ):
        """Get critical alerts"""
        alerts = data_manager.get_critical_alerts(limit=limit)
        
        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/dashboard/summary")
    async def get_dashboard_summary():
        """Get comprehensive dashboard summary"""
        summary = data_manager.get_dashboard_summary()
        
        return {
            "status": "success",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.post("/export")
    async def export_data(
        request: ExportRequest,
        background_tasks: BackgroundTasks
    ):
        """Export training data"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retfound_training_data_{timestamp}.{request.format}"
            filepath = Path("exports") / filename
            
            # Create exports directory
            filepath.parent.mkdir(exist_ok=True)
            
            # Add export task to background
            background_tasks.add_task(
                data_manager.export_data,
                filepath,
                request.format
            )
            
            return {
                "status": "success",
                "message": "Export started",
                "filename": filename,
                "format": request.format,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    
    @router.delete("/data")
    async def clear_data(
        keep_recent: int = Query(100, ge=10, le=1000, description="Number of recent entries to keep")
    ):
        """Clear old training data"""
        try:
            await data_manager.clear_data(keep_recent=keep_recent)
            
            return {
                "status": "success",
                "message": f"Data cleared, kept {keep_recent} recent entries",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data clearing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Data clearing failed: {str(e)}")
    
    @router.get("/stats")
    async def get_system_stats():
        """Get system statistics"""
        try:
            import psutil
            import GPUtil
            
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU stats
            gpu_stats = []
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_stats.append({
                        "id": i,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature": gpu.temperature
                    })
            except:
                gpu_stats = []
            
            return {
                "status": "success",
                "stats": {
                    "cpu": {
                        "usage_percent": cpu_percent,
                        "count": psutil.cpu_count()
                    },
                    "memory": {
                        "total": memory.total,
                        "used": memory.used,
                        "available": memory.available,
                        "percent": memory.percent
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": (disk.used / disk.total) * 100
                    },
                    "gpu": gpu_stats
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stats collection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Stats collection failed: {str(e)}")
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "RETFound Training Monitor API",
            "timestamp": datetime.now().isoformat(),
            "data_manager": {
                "total_snapshots": len(data_manager.metrics_buffer.get_all()),
                "total_epochs": len(data_manager.epoch_data),
                "alerts_count": len(data_manager.critical_alerts)
            }
        }
    
    # Advanced query endpoints
    
    @router.post("/query/metrics")
    async def query_metrics(query: MetricsQuery):
        """Advanced metrics query"""
        try:
            # Get performance history for the metric
            history = data_manager.get_performance_history(query.metric, limit=query.limit)
            
            # Filter by epoch range if specified
            if query.start_epoch is not None or query.end_epoch is not None:
                if isinstance(history, dict):
                    # Handle nested metric structure
                    filtered_history = {}
                    for submetric, values in history.items():
                        if isinstance(values, list):
                            filtered_values = []
                            for point in values:
                                epoch = point.get('epoch', 0)
                                if query.start_epoch is not None and epoch < query.start_epoch:
                                    continue
                                if query.end_epoch is not None and epoch > query.end_epoch:
                                    continue
                                filtered_values.append(point)
                            filtered_history[submetric] = filtered_values
                    history = filtered_history
                elif isinstance(history, list):
                    # Handle flat metric structure
                    filtered_history = []
                    for point in history:
                        epoch = point.get('epoch', 0)
                        if query.start_epoch is not None and epoch < query.start_epoch:
                            continue
                        if query.end_epoch is not None and epoch > query.end_epoch:
                            continue
                        filtered_history.append(point)
                    history = filtered_history
            
            return {
                "status": "success",
                "query": query.dict(),
                "data": history,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Metrics query failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
    @router.get("/compare/epochs")
    async def compare_epochs(
        epochs: str = Query(..., description="Comma-separated list of epochs to compare"),
        metrics: str = Query("accuracy,loss,f1_score", description="Comma-separated list of metrics to compare")
    ):
        """Compare metrics across multiple epochs"""
        try:
            # Parse parameters
            epoch_list = [int(e.strip()) for e in epochs.split(',')]
            metric_list = [m.strip() for m in metrics.split(',')]
            
            comparison = {}
            
            for epoch in epoch_list:
                epoch_details = data_manager.get_epoch_details(epoch)
                if epoch_details:
                    comparison[epoch] = {}
                    epoch_metrics = epoch_details.get('metrics', {})
                    
                    for metric in metric_list:
                        if metric in epoch_metrics:
                            comparison[epoch][metric] = epoch_metrics[metric]
                        else:
                            comparison[epoch][metric] = None
            
            return {
                "status": "success",
                "comparison": comparison,
                "epochs": epoch_list,
                "metrics": metric_list,
                "timestamp": datetime.now().isoformat()
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid parameter format: {str(e)}")
        except Exception as e:
            logger.error(f"Epoch comparison failed: {e}")
            raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    
    return router
