"""
FastAPI Server with WebSocket for Real-time Training Monitoring
==============================================================

Provides REST API and WebSocket endpoints for the RETFound training dashboard.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .data_manager import DataManager
    from .api_routes import create_api_routes
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from retfound.monitoring.data_manager import DataManager
    from retfound.monitoring.api_routes import create_api_routes

logger = logging.getLogger(__name__)


class TrainingStatus(BaseModel):
    """Training status model"""
    status: str  # 'idle', 'training', 'paused', 'completed', 'error'
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    start_time: Optional[datetime] = None
    elapsed_time: float = 0.0
    eta_seconds: Optional[float] = None


class MetricsUpdate(BaseModel):
    """Metrics update model matching WebSocket structure"""
    type: str = 'metrics_update'
    epoch: int
    batch: int
    total_batches: int
    metrics: Dict[str, Any]
    system: Dict[str, Any]
    timestamp: str


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_count += 1
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connections"""
        message = json.dumps(data, default=str)
        await self.broadcast(message)


class MonitoringServer:
    """Main monitoring server class"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        frontend_dir: Optional[Path] = None,
        enable_cors: bool = True
    ):
        self.host = host
        self.port = port
        self.frontend_dir = frontend_dir
        self.enable_cors = enable_cors
        
        # Core components
        self.data_manager = DataManager()
        self.connection_manager = ConnectionManager()
        
        # Training state
        self.training_status = TrainingStatus(
            status='idle',
            epoch=0,
            total_epochs=0,
            batch=0,
            total_batches=0
        )
        
        # Create FastAPI app
        self.app = self._create_app()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan manager"""
            # Startup
            logger.info("Starting RETFound Monitoring Server...")
            await self._startup()
            
            yield
            
            # Shutdown
            logger.info("Shutting down RETFound Monitoring Server...")
            await self._shutdown()
        
        app = FastAPI(
            title="RETFound Training Monitor",
            description="Real-time monitoring dashboard for RETFound model training",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # CORS middleware
        if self.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure appropriately for production
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add all routes to the FastAPI app"""
        
        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connection_manager.connect(websocket)
            
            try:
                # Send initial state
                await self._send_initial_state(websocket)
                
                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        # Wait for messages with timeout
                        message = await asyncio.wait_for(
                            websocket.receive_text(), 
                            timeout=30.0
                        )
                        await self._handle_websocket_message(websocket, message)
                    
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send_json({"type": "ping"})
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket)
        
        # Health check
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "connections": len(self.connection_manager.active_connections),
                "training_status": self.training_status.status
            }
        
        # Training control endpoints
        @app.post("/api/training/start")
        async def start_training():
            """Start training (placeholder - actual implementation depends on training setup)"""
            if self.training_status.status == 'training':
                raise HTTPException(status_code=400, detail="Training already in progress")
            
            self.training_status.status = 'training'
            self.training_status.start_time = datetime.now()
            
            await self.connection_manager.broadcast_json({
                "type": "status_update",
                "status": self.training_status.dict()
            })
            
            return {"message": "Training started", "status": self.training_status.status}
        
        @app.post("/api/training/pause")
        async def pause_training():
            """Pause training"""
            if self.training_status.status != 'training':
                raise HTTPException(status_code=400, detail="No training in progress")
            
            self.training_status.status = 'paused'
            
            await self.connection_manager.broadcast_json({
                "type": "status_update", 
                "status": self.training_status.dict()
            })
            
            return {"message": "Training paused", "status": self.training_status.status}
        
        @app.post("/api/training/stop")
        async def stop_training():
            """Stop training"""
            if self.training_status.status not in ['training', 'paused']:
                raise HTTPException(status_code=400, detail="No training to stop")
            
            self.training_status.status = 'idle'
            self.training_status.start_time = None
            self.training_status.elapsed_time = 0.0
            
            await self.connection_manager.broadcast_json({
                "type": "status_update",
                "status": self.training_status.dict()
            })
            
            return {"message": "Training stopped", "status": self.training_status.status}
        
        # Add API routes from separate module
        api_router = create_api_routes(self.data_manager)
        app.include_router(api_router, prefix="/api")
        
        # Serve frontend static files if directory provided
        if self.frontend_dir and self.frontend_dir.exists():
            app.mount("/", StaticFiles(directory=str(self.frontend_dir), html=True), name="frontend")
    
    async def _send_initial_state(self, websocket: WebSocket):
        """Send initial state to newly connected client"""
        initial_data = {
            "type": "initial_state",
            "status": self.training_status.dict(),
            "metrics": self.data_manager.get_latest_metrics(),
            "history": self.data_manager.get_metrics_history(limit=100)
        }
        
        await self.connection_manager.send_personal_message(
            json.dumps(initial_data, default=str),
            websocket
        )
    
    async def _handle_websocket_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif message_type == "get_history":
                limit = data.get("limit", 100)
                history = self.data_manager.get_metrics_history(limit=limit)
                await websocket.send_json({
                    "type": "history_response",
                    "data": history
                })
            
            elif message_type == "get_epoch_details":
                epoch = data.get("epoch")
                if epoch is not None:
                    details = self.data_manager.get_epoch_details(epoch)
                    await websocket.send_json({
                        "type": "epoch_details_response",
                        "epoch": epoch,
                        "data": details
                    })
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _startup(self):
        """Server startup tasks"""
        # Initialize data manager
        await self.data_manager.initialize()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Monitoring server ready at http://{self.host}:{self.port}")
    
    async def _shutdown(self):
        """Server shutdown tasks"""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Cleanup data manager
        await self.data_manager.cleanup()
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # System monitoring task
        task = asyncio.create_task(self._system_monitor_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        # Training status update task
        task = asyncio.create_task(self._training_status_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _system_monitor_task(self):
        """Background task to monitor system resources"""
        import psutil
        import GPUtil
        
        while True:
            try:
                # Get system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Get GPU stats if available
                gpu_stats = {}
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_stats = {
                            "usage": gpu.load * 100,
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "temperature": gpu.temperature
                        }
                except:
                    gpu_stats = {
                        "usage": 0,
                        "memory_used": 0,
                        "memory_total": 0,
                        "temperature": 0
                    }
                
                system_data = {
                    "type": "system_update",
                    "cpu_usage": cpu_percent,
                    "ram_usage": memory.percent,
                    "ram_used": memory.used,
                    "ram_total": memory.total,
                    "gpu": gpu_stats,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.connection_manager.broadcast_json(system_data)
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _training_status_task(self):
        """Background task to update training status"""
        while True:
            try:
                if self.training_status.status == 'training' and self.training_status.start_time:
                    # Update elapsed time
                    elapsed = (datetime.now() - self.training_status.start_time).total_seconds()
                    self.training_status.elapsed_time = elapsed
                    
                    # Calculate ETA if we have progress info
                    if self.training_status.total_epochs > 0 and self.training_status.epoch > 0:
                        progress = self.training_status.epoch / self.training_status.total_epochs
                        if progress > 0:
                            total_estimated = elapsed / progress
                            self.training_status.eta_seconds = total_estimated - elapsed
                
                await asyncio.sleep(1)  # Update every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training status task error: {e}")
                await asyncio.sleep(5)
    
    # Public methods for external integration
    
    async def update_metrics(self, metrics_data: Dict[str, Any]):
        """Update metrics from external source (e.g., training callback)"""
        # Store in data manager
        await self.data_manager.add_metrics(metrics_data)
        
        # Update training status
        if 'epoch' in metrics_data:
            self.training_status.epoch = metrics_data['epoch']
        if 'batch' in metrics_data:
            self.training_status.batch = metrics_data['batch']
        if 'total_batches' in metrics_data:
            self.training_status.total_batches = metrics_data['total_batches']
        
        # Broadcast to clients
        update_data = MetricsUpdate(
            epoch=metrics_data.get('epoch', 0),
            batch=metrics_data.get('batch', 0),
            total_batches=metrics_data.get('total_batches', 0),
            metrics=metrics_data.get('metrics', {}),
            system=metrics_data.get('system', {}),
            timestamp=datetime.now().isoformat()
        )
        
        await self.connection_manager.broadcast_json(update_data.dict())
    
    async def update_training_status(self, status: str, **kwargs):
        """Update training status from external source"""
        self.training_status.status = status
        
        for key, value in kwargs.items():
            if hasattr(self.training_status, key):
                setattr(self.training_status, key, value)
        
        await self.connection_manager.broadcast_json({
            "type": "status_update",
            "status": self.training_status.dict()
        })
    
    def run(self, **kwargs):
        """Run the monitoring server"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )
        server = uvicorn.Server(config)
        server.run()
    
    async def run_async(self, **kwargs):
        """Run the monitoring server asynchronously"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )
        server = uvicorn.Server(config)
        await server.serve()


# Global server instance for easy access
_server_instance: Optional[MonitoringServer] = None


def get_server() -> Optional[MonitoringServer]:
    """Get the global server instance"""
    return _server_instance


def create_server(
    host: str = "localhost",
    port: int = 8000,
    frontend_dir: Optional[Path] = None,
    **kwargs
) -> MonitoringServer:
    """Create and configure monitoring server"""
    global _server_instance
    
    _server_instance = MonitoringServer(
        host=host,
        port=port,
        frontend_dir=frontend_dir,
        **kwargs
    )
    
    return _server_instance


if __name__ == "__main__":
    # Run server directly
    server = create_server()
    server.run()
