# Guide d'Intégration Frontend-Backend RETFound

## 📋 Architecture Complète

Ce document détaille l'intégration complète entre le frontend React/TypeScript et le backend FastAPI pour le monitoring d'entraînement RETFound.

## 🏗️ Architecture du Système

```
┌─────────────────────────────────────────────────────────────┐
│                    RETFound Training Stack                  │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React/TypeScript)                               │
│  ├── Dashboard Components                                  │
│  ├── WebSocket Hooks                                       │
│  ├── State Management (Zustand)                           │
│  └── Real-time Charts (Recharts)                          │
├─────────────────────────────────────────────────────────────┤
│  Backend API (FastAPI)                                     │
│  ├── REST Endpoints                                        │
│  ├── WebSocket Server                                      │
│  ├── Data Manager                                          │
│  └── Training Integration                                  │
├─────────────────────────────────────────────────────────────┤
│  Training Engine                                            │
│  ├── RETFound Models                                       │
│  ├── Monitoring Callbacks                                  │
│  ├── Metrics Collection                                    │
│  └── Critical Alerts                                       │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 Endpoints API Complets

### 1. Contrôle d'Entraînement

```typescript
// Types TypeScript
interface TrainingControlResponse {
  status: 'success' | 'error'
  message: string
  training_status: 'idle' | 'training' | 'paused' | 'completed' | 'error'
}

// Endpoints
POST /api/training/start
POST /api/training/pause
POST /api/training/stop
POST /api/training/resume
```

**Exemple d'utilisation Frontend** :
```typescript
const startTraining = async () => {
  try {
    const response = await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    const data: TrainingControlResponse = await response.json()
    
    if (data.status === 'success') {
      toast.success('Training started successfully')
    }
  } catch (error) {
    toast.error('Failed to start training')
  }
}
```

### 2. Métriques et Données

```typescript
// Types pour les métriques
interface MetricsData {
  loss: {
    train: number
    val: number
  }
  accuracy: {
    train: number
    val: number
  }
  auc_roc: {
    macro: number
    weighted: number
  }
  f1_score: number
  learning_rate: number
  critical_conditions: {
    [condition: string]: {
      sensitivity: number
      threshold: number
      status: 'ok' | 'warning' | 'critical'
    }
  }
  per_class: {
    [className: string]: number
  }
}

// Endpoints
GET /api/metrics/latest
GET /api/metrics/history?limit=100
GET /api/metrics/performance/{metric}?limit=1000
```

### 3. Gestion des Époques

```typescript
interface EpochDetails {
  epoch: number
  total_batches: number
  start_time: string
  end_time?: string
  duration?: number
  metrics: MetricsData
  confusion_matrix?: number[][]
  class_report?: ClassificationReport
}

// Endpoints
GET /api/epochs
GET /api/epochs/{epoch}
GET /api/compare/epochs?epochs=1,2,3&metrics=accuracy,loss
```

### 4. Alertes Critiques

```typescript
interface CriticalAlert {
  timestamp: string
  condition: string
  current_value: number
  threshold: number
  level: 'critical' | 'warning'
  message: string
  epoch: number
  batch: number
  recommendation?: string
}

// Endpoints
GET /api/alerts?limit=100
GET /api/alerts/critical
POST /api/alerts/acknowledge/{alert_id}
```

### 5. Statistiques Système

```typescript
interface SystemStats {
  cpu: {
    usage_percent: number
    count: number
  }
  memory: {
    total: number
    used: number
    available: number
    percent: number
  }
  gpu: Array<{
    id: number
    name: string
    load: number
    memory_used: number
    memory_total: number
    memory_percent: number
    temperature: number
  }>
  disk: {
    total: number
    used: number
    free: number
    percent: number
  }
}

// Endpoints
GET /api/stats
GET /api/health
```

## 🔄 WebSocket Integration

### Messages WebSocket

#### 1. Connexion et État Initial
```typescript
// Message envoyé lors de la connexion
{
  "type": "initial_state",
  "status": TrainingStatus,
  "metrics": MetricsData,
  "history": MetricsSnapshot[]
}
```

#### 2. Mise à Jour des Métriques
```typescript
{
  "type": "metrics_update",
  "epoch": 15,
  "batch": 100,
  "total_batches": 500,
  "metrics": MetricsData,
  "system": SystemMetrics,
  "timestamp": "2025-06-13T10:30:00Z"
}
```

#### 3. Changement de Statut
```typescript
{
  "type": "status_update",
  "status": {
    "status": "training",
    "epoch": 15,
    "total_epochs": 100,
    "batch": 100,
    "total_batches": 500,
    "elapsed_time": 3600,
    "eta_seconds": 7200
  }
}
```

#### 4. Alertes Critiques
```typescript
{
  "type": "critical_alert",
  "alert": CriticalAlert
}
```

#### 5. Statistiques Système
```typescript
{
  "type": "system_update",
  "cpu_usage": 45.2,
  "ram_usage": 67.8,
  "gpu": {
    "usage": 95.2,
    "memory_used": 38.4,
    "memory_total": 40.0,
    "temperature": 78
  },
  "timestamp": "2025-06-13T10:30:00Z"
}
```

### Hook WebSocket Frontend

```typescript
// retfound/monitoring/frontend/src/hooks/useWebSocket.ts
export const useWebSocket = (): UseWebSocketReturn => {
  const wsRef = useRef<WebSocket | null>(null)
  const {
    setConnectionStatus,
    handleMetricsUpdate,
    handleStatusUpdate,
    handleSystemUpdate,
    handleInitialState
  } = useMonitoringStore()
  
  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/ws`
    
    wsRef.current = new WebSocket(wsUrl)
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'initial_state':
          handleInitialState(data)
          break
        case 'metrics_update':
          handleMetricsUpdate(data)
          break
        case 'status_update':
          handleStatusUpdate(data)
          break
        case 'system_update':
          handleSystemUpdate(data)
          break
        case 'critical_alert':
          handleCriticalAlert(data.alert)
          break
      }
    }
  }, [])
  
  return { connect, disconnect, sendMessage }
}
```

## 🎛️ Composants Frontend Principaux

### 1. Dashboard Principal

```typescript
// retfound/monitoring/frontend/src/components/Dashboard/MainDashboard.tsx
export const MainDashboard: React.FC = () => {
  const { connect } = useWebSocket()
  const trainingStatus = useTrainingStatus()
  const currentMetrics = useCurrentMetrics()
  const systemMetrics = useSystemMetrics()
  
  useEffect(() => {
    connect()
  }, [connect])
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Status Cards */}
      <MetricsGrid metrics={currentMetrics} />
      
      {/* Progress */}
      <ProgressBar status={trainingStatus} />
      
      {/* System Stats */}
      <GPUStats metrics={systemMetrics} />
      
      {/* Charts */}
      <div className="col-span-full">
        <MetricsChart />
      </div>
      
      {/* Class Performance */}
      <div className="col-span-full">
        <ClassPerformance />
      </div>
      
      {/* Critical Alerts */}
      <CriticalAlerts />
    </div>
  )
}
```

### 2. Monitoring des Classes Critiques

```typescript
// retfound/monitoring/frontend/src/components/Monitoring/CriticalAlerts.tsx
export const CriticalAlerts: React.FC = () => {
  const alerts = useCriticalAlerts()
  const currentMetrics = useCurrentMetrics()
  
  const criticalConditions = currentMetrics.critical_conditions || {}
  
  return (
    <Card>
      <CardHeader>
        <h3>Conditions Critiques</h3>
      </CardHeader>
      <CardContent>
        {Object.entries(criticalConditions).map(([condition, data]) => (
          <div key={condition} className="flex items-center justify-between p-3">
            <span className="font-medium">{condition}</span>
            <div className="flex items-center gap-2">
              <span className="text-sm">
                {(data.sensitivity * 100).toFixed(1)}%
              </span>
              <Badge variant={data.status === 'ok' ? 'success' : 'destructive'}>
                {data.status}
              </Badge>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
```

### 3. Graphiques de Performance

```typescript
// retfound/monitoring/frontend/src/components/Charts/MetricsChart.tsx
export const MetricsChart: React.FC = () => {
  const performanceHistory = usePerformanceHistory()
  
  const chartData = useMemo(() => {
    return performanceHistory.loss.train.map((point, index) => ({
      epoch: point.epoch,
      batch: point.batch,
      train_loss: point.value,
      val_loss: performanceHistory.loss.val[index]?.value || null,
      train_acc: performanceHistory.accuracy.train[index]?.value || null,
      val_acc: performanceHistory.accuracy.val[index]?.value || null,
    }))
  }, [performanceHistory])
  
  return (
    <Card>
      <CardHeader>
        <h3>Métriques d'Entraînement</h3>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="train_loss" 
              stroke="#ef4444" 
              name="Train Loss"
            />
            <Line 
              type="monotone" 
              dataKey="val_loss" 
              stroke="#f97316" 
              name="Val Loss"
            />
            <Line 
              type="monotone" 
              dataKey="train_acc" 
              stroke="#22c55e" 
              name="Train Acc"
            />
            <Line 
              type="monotone" 
              dataKey="val_acc" 
              stroke="#3b82f6" 
              name="Val Acc"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
```

## 🔧 Backend Implementation

### 1. Serveur FastAPI Principal

```python
# retfound/monitoring/server.py
class MonitoringServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.app = FastAPI(title="RETFound Training Monitor")
        self.connection_manager = ConnectionManager()
        self.data_manager = DataManager()
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._add_routes()
    
    def _add_routes(self):
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connection_manager.connect(websocket)
            try:
                await self._send_initial_state(websocket)
                while True:
                    message = await websocket.receive_text()
                    await self._handle_websocket_message(websocket, message)
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
        
        # Include API routes
        api_router = create_api_routes(self.data_manager)
        self.app.include_router(api_router, prefix="/api")
```

### 2. Gestionnaire de Données

```python
# retfound/monitoring/data_manager.py
class DataManager:
    def __init__(self):
        self.metrics_buffer = CircularBuffer(maxsize=1000)
        self.epoch_data = {}
        self.critical_alerts = []
        
    async def add_metrics(self, metrics_data: Dict[str, Any]):
        """Add new metrics snapshot"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'epoch': metrics_data.get('epoch', 0),
            'batch': metrics_data.get('batch', 0),
            'metrics': metrics_data.get('metrics', {}),
            'system': metrics_data.get('system', {})
        }
        
        self.metrics_buffer.append(snapshot)
        
        # Check for critical conditions
        await self._check_critical_conditions(snapshot)
    
    async def _check_critical_conditions(self, snapshot):
        """Check for critical conditions and generate alerts"""
        metrics = snapshot.get('metrics', {})
        critical_conditions = metrics.get('critical_conditions', {})
        
        for condition, data in critical_conditions.items():
            if data['status'] in ['warning', 'critical']:
                alert = {
                    'timestamp': snapshot['timestamp'],
                    'condition': condition,
                    'current_value': data['sensitivity'],
                    'threshold': data['threshold'],
                    'level': data['status'],
                    'message': f"{condition} sensitivity below threshold",
                    'epoch': snapshot['epoch'],
                    'batch': snapshot['batch']
                }
                self.critical_alerts.append(alert)
```

### 3. Callback d'Entraînement

```python
# retfound/monitoring/monitor_callback.py
class MonitoringCallback:
    def __init__(self, server: MonitoringServer):
        self.server = server
        
    async def on_batch_end(self, epoch, batch, logs):
        """Called at the end of each batch"""
        metrics_data = {
            'epoch': epoch,
            'batch': batch,
            'total_batches': logs.get('total_batches', 0),
            'metrics': {
                'loss': {
                    'train': logs.get('loss', 0),
                    'val': logs.get('val_loss', 0)
                },
                'accuracy': {
                    'train': logs.get('accuracy', 0),
                    'val': logs.get('val_accuracy', 0)
                },
                'learning_rate': logs.get('lr', 0),
                'critical_conditions': self._check_critical_conditions(logs)
            },
            'system': self._get_system_metrics()
        }
        
        await self.server.update_metrics(metrics_data)
    
    def _check_critical_conditions(self, logs):
        """Check critical conditions sensitivity"""
        conditions = {}
        
        # Check each critical condition
        for condition, info in CRITICAL_CONDITIONS.items():
            sensitivity = logs.get(f'{condition}_sensitivity', 0)
            threshold = info['min_sensitivity']
            
            status = 'ok'
            if sensitivity < threshold:
                if sensitivity < threshold * 0.9:
                    status = 'critical'
                else:
                    status = 'warning'
            
            conditions[condition] = {
                'sensitivity': sensitivity,
                'threshold': threshold,
                'status': status
            }
        
        return conditions
```

## 🚀 Déploiement et Configuration

### 1. Variables d'Environnement

```bash
# .env.runpod
DATASET_PATH=/workspace/datasets/DATASET_CLASSIFICATION
MONITORING_HOST=0.0.0.0
MONITORING_PORT=8000
FRONTEND_PORT=3000

# Frontend (.env.runpod)
VITE_API_URL=http://0.0.0.0:8000
VITE_WS_URL=ws://0.0.0.0:8000
VITE_RUNPOD_MODE=true
```

### 2. Scripts de Démarrage

```bash
# start_monitoring.sh
#!/bin/bash
source venv_retfound/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -m retfound.monitoring.server \
    --host 0.0.0.0 \
    --port 8000 \
    --frontend-dir retfound/monitoring/frontend/dist

# start_training.sh
#!/bin/bash
source venv_retfound/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -m retfound.cli.main train \
    --config configs/runpod.yaml \
    --enable-monitoring \
    --monitor-critical
```

### 3. Build Frontend pour Production

```bash
# Dans retfound/monitoring/frontend/
npm install
npm run build

# Le build sera dans dist/ et servi par FastAPI
```

## 🔍 Tests et Validation

### 1. Test de Connexion WebSocket

```typescript
// Test de connexion
const testWebSocket = () => {
  const ws = new WebSocket('ws://localhost:8000/ws')
  
  ws.onopen = () => console.log('✓ WebSocket connected')
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    console.log('✓ Received:', data.type)
  }
  ws.onerror = (error) => console.error('✗ WebSocket error:', error)
}
```

### 2. Test des Endpoints API

```bash
# Test de santé
curl http://localhost:8000/health

# Test des métriques
curl http://localhost:8000/api/metrics/latest

# Test de contrôle d'entraînement
curl -X POST http://localhost:8000/api/training/start
```

## 📊 Monitoring des 28 Classes

### Configuration des Classes

```typescript
// Classes CAASI v6.1
const UNIFIED_CLASSES = [
  // Fundus (0-17)
  'Fundus_Normal', 'Fundus_DR_Mild', 'Fundus_DR_Moderate', 
  'Fundus_DR_Severe', 'Fundus_DR_Proliferative', 'Fundus_Glaucoma_Suspect',
  'Fundus_Glaucoma_Positive', 'Fundus_RVO', 'Fundus_RAO',
  'Fundus_Hypertensive_Retinopathy', 'Fundus_Drusen', 'Fundus_CNV_Wet_AMD',
  'Fundus_Myopia_Degenerative', 'Fundus_Retinal_Detachment', 'Fundus_Macular_Scar',
  'Fundus_Cataract_Suspected', 'Fundus_Optic_Disc_Anomaly', 'Fundus_Other',
  
  // OCT (18-27)
  'OCT_Normal', 'OCT_DME', 'OCT_CNV', 'OCT_Dry_AMD', 'OCT_ERM',
  'OCT_Vitreomacular_Interface_Disease', 'OCT_CSR', 'OCT_RVO',
  'OCT_Glaucoma', 'OCT_RAO'
]

// Classes critiques avec seuils
const CRITICAL_CONDITIONS = {
  'RAO': { indices: [8, 27], min_sensitivity: 0.99 },
  'RVO': { indices: [7, 25], min_sensitivity: 0.97 },
  'Retinal_Detachment': { indices: [13], min_sensitivity: 0.99 },
  'CNV': { indices: [11, 20], min_sensitivity: 0.98 }
}
```

---

**Version** : 2.0.0  
**Dernière mise à jour** : 13 Juin 2025  
**Compatibilité** : React 18, FastAPI 0.104, WebSocket, RunPod A100
