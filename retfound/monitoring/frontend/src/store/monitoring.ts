import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

// Types
export interface TrainingStatus {
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error'
  epoch: number
  totalEpochs: number
  batch: number
  totalBatches: number
  startTime?: string
  elapsedTime: number
  etaSeconds?: number
}

export interface MetricsData {
  loss?: {
    train?: number
    val?: number
  }
  accuracy?: {
    train?: number
    val?: number
  }
  auc_roc?: {
    macro?: number
    weighted?: number
  }
  f1_score?: number
  learning_rate?: number
  critical_conditions?: {
    [condition: string]: {
      sensitivity: number
      threshold: number
      status: 'ok' | 'warning' | 'critical'
    }
  }
  per_class?: {
    [className: string]: number
  }
}

export interface SystemMetrics {
  gpu_usage: number
  gpu_memory: number
  gpu_memory_total: number
  gpu_temp: number
  ram_usage: number
  ram_used: number
  ram_total: number
  eta_seconds?: number
  torch_gpu_memory?: number
}

export interface MetricsSnapshot {
  timestamp: string
  epoch: number
  batch: number
  totalBatches: number
  metrics: MetricsData
  system: SystemMetrics
}

export interface CriticalAlert {
  timestamp: string
  condition: string
  current_value: number
  threshold: number
  level: 'critical' | 'warning'
  message: string
  epoch: number
  batch: number
}

export interface PerformanceHistory {
  loss: {
    train: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
    val: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
  }
  accuracy: {
    train: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
    val: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
  }
  auc_roc: {
    macro: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
    weighted: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
  }
  f1_score: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
  learning_rate: Array<{ epoch: number; batch: number; value: number; timestamp: string }>
  critical_conditions: {
    [condition: string]: Array<{
      epoch: number
      batch: number
      sensitivity: number
      threshold: number
      status: string
      timestamp: string
    }>
  }
}

export interface EpochDetails {
  epoch: number
  totalBatches: number
  startTime: string
  endTime?: string
  duration?: number
  metrics: MetricsData
}

// Store interface
interface MonitoringStore {
  // Connection state
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error'
  
  // Training state
  trainingStatus: TrainingStatus
  
  // Current metrics
  currentMetrics: MetricsData
  systemMetrics: SystemMetrics
  
  // Historical data
  metricsHistory: MetricsSnapshot[]
  performanceHistory: PerformanceHistory
  
  // Alerts
  criticalAlerts: CriticalAlert[]
  
  // Epochs
  epochDetails: { [epoch: number]: EpochDetails }
  
  // UI state
  selectedEpoch: number | null
  isFullscreen: boolean
  darkMode: boolean
  
  // Computed getters
  isTraining: boolean
  currentProgress: number
  
  // Actions
  setConnectionStatus: (status: MonitoringStore['connectionStatus']) => void
  updateTrainingStatus: (status: Partial<TrainingStatus>) => void
  updateMetrics: (snapshot: MetricsSnapshot) => void
  addCriticalAlert: (alert: CriticalAlert) => void
  setSelectedEpoch: (epoch: number | null) => void
  toggleFullscreen: () => void
  toggleDarkMode: () => void
  clearHistory: () => void
  
  // WebSocket message handlers
  handleMetricsUpdate: (data: any) => void
  handleStatusUpdate: (data: any) => void
  handleSystemUpdate: (data: any) => void
  handleInitialState: (data: any) => void
}

// Initial state
const initialTrainingStatus: TrainingStatus = {
  status: 'idle',
  epoch: 0,
  totalEpochs: 0,
  batch: 0,
  totalBatches: 0,
  elapsedTime: 0,
}

const initialSystemMetrics: SystemMetrics = {
  gpu_usage: 0,
  gpu_memory: 0,
  gpu_memory_total: 0,
  gpu_temp: 0,
  ram_usage: 0,
  ram_used: 0,
  ram_total: 0,
}

const initialPerformanceHistory: PerformanceHistory = {
  loss: { train: [], val: [] },
  accuracy: { train: [], val: [] },
  auc_roc: { macro: [], weighted: [] },
  f1_score: [],
  learning_rate: [],
  critical_conditions: {},
}

// Create store
export const useMonitoringStore = create<MonitoringStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    connectionStatus: 'disconnected',
    trainingStatus: initialTrainingStatus,
    currentMetrics: {},
    systemMetrics: initialSystemMetrics,
    metricsHistory: [],
    performanceHistory: initialPerformanceHistory,
    criticalAlerts: [],
    epochDetails: {},
    selectedEpoch: null,
    isFullscreen: false,
    darkMode: true,
    
    // Computed getters
    get isTraining() {
      return get().trainingStatus.status === 'training'
    },
    
    get currentProgress() {
      const { epoch, totalEpochs, batch, totalBatches } = get().trainingStatus
      if (totalEpochs === 0) return 0
      
      const epochProgress = epoch / totalEpochs
      const batchProgress = totalBatches > 0 ? batch / totalBatches : 0
      const currentEpochContribution = batchProgress / totalEpochs
      
      return Math.min(100, (epochProgress + currentEpochContribution) * 100)
    },
    
    // Actions
    setConnectionStatus: (status) => {
      set({ connectionStatus: status })
    },
    
    updateTrainingStatus: (status) => {
      set((state) => ({
        trainingStatus: { ...state.trainingStatus, ...status }
      }))
    },
    
    updateMetrics: (snapshot) => {
      set((state) => {
        // Update current metrics
        const newState = {
          currentMetrics: snapshot.metrics,
          systemMetrics: snapshot.system,
          trainingStatus: {
            ...state.trainingStatus,
            epoch: snapshot.epoch,
            batch: snapshot.batch,
            totalBatches: snapshot.totalBatches,
          }
        }
        
        // Add to history (keep last 1000 entries)
        const newHistory = [...state.metricsHistory, snapshot].slice(-1000)
        
        // Update performance history
        const newPerformanceHistory = { ...state.performanceHistory }
        
        // Update loss history
        if (snapshot.metrics.loss?.train !== undefined) {
          newPerformanceHistory.loss.train = [
            ...newPerformanceHistory.loss.train,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.loss.train,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        if (snapshot.metrics.loss?.val !== undefined) {
          newPerformanceHistory.loss.val = [
            ...newPerformanceHistory.loss.val,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.loss.val,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        // Update accuracy history
        if (snapshot.metrics.accuracy?.train !== undefined) {
          newPerformanceHistory.accuracy.train = [
            ...newPerformanceHistory.accuracy.train,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.accuracy.train,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        if (snapshot.metrics.accuracy?.val !== undefined) {
          newPerformanceHistory.accuracy.val = [
            ...newPerformanceHistory.accuracy.val,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.accuracy.val,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        // Update AUC-ROC history
        if (snapshot.metrics.auc_roc?.macro !== undefined) {
          newPerformanceHistory.auc_roc.macro = [
            ...newPerformanceHistory.auc_roc.macro,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.auc_roc.macro,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        if (snapshot.metrics.auc_roc?.weighted !== undefined) {
          newPerformanceHistory.auc_roc.weighted = [
            ...newPerformanceHistory.auc_roc.weighted,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.auc_roc.weighted,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        // Update F1 score history
        if (snapshot.metrics.f1_score !== undefined) {
          newPerformanceHistory.f1_score = [
            ...newPerformanceHistory.f1_score,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.f1_score,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        // Update learning rate history
        if (snapshot.metrics.learning_rate !== undefined) {
          newPerformanceHistory.learning_rate = [
            ...newPerformanceHistory.learning_rate,
            {
              epoch: snapshot.epoch,
              batch: snapshot.batch,
              value: snapshot.metrics.learning_rate,
              timestamp: snapshot.timestamp
            }
          ].slice(-1000)
        }
        
        // Update critical conditions history
        if (snapshot.metrics.critical_conditions) {
          Object.entries(snapshot.metrics.critical_conditions).forEach(([condition, data]) => {
            if (!newPerformanceHistory.critical_conditions[condition]) {
              newPerformanceHistory.critical_conditions[condition] = []
            }
            
            newPerformanceHistory.critical_conditions[condition] = [
              ...newPerformanceHistory.critical_conditions[condition],
              {
                epoch: snapshot.epoch,
                batch: snapshot.batch,
                sensitivity: data.sensitivity,
                threshold: data.threshold,
                status: data.status,
                timestamp: snapshot.timestamp
              }
            ].slice(-1000)
          })
        }
        
        return {
          ...newState,
          metricsHistory: newHistory,
          performanceHistory: newPerformanceHistory
        }
      })
    },
    
    addCriticalAlert: (alert) => {
      set((state) => ({
        criticalAlerts: [...state.criticalAlerts, alert].slice(-100) // Keep last 100 alerts
      }))
    },
    
    setSelectedEpoch: (epoch) => {
      set({ selectedEpoch: epoch })
    },
    
    toggleFullscreen: () => {
      set((state) => ({ isFullscreen: !state.isFullscreen }))
    },
    
    toggleDarkMode: () => {
      set((state) => ({ darkMode: !state.darkMode }))
    },
    
    clearHistory: () => {
      set({
        metricsHistory: [],
        performanceHistory: initialPerformanceHistory,
        criticalAlerts: [],
        epochDetails: {}
      })
    },
    
    // WebSocket message handlers
    handleMetricsUpdate: (data) => {
      const snapshot: MetricsSnapshot = {
        timestamp: data.timestamp,
        epoch: data.epoch,
        batch: data.batch,
        totalBatches: data.total_batches,
        metrics: data.metrics,
        system: data.system
      }
      
      get().updateMetrics(snapshot)
      
      // Check for critical alerts
      if (data.metrics.critical_conditions) {
        Object.entries(data.metrics.critical_conditions).forEach(([condition, conditionData]: [string, any]) => {
          if (conditionData.status === 'critical' || conditionData.status === 'warning') {
            const alert: CriticalAlert = {
              timestamp: data.timestamp,
              condition,
              current_value: conditionData.sensitivity,
              threshold: conditionData.threshold,
              level: conditionData.status,
              message: `${condition} sensitivity (${conditionData.sensitivity.toFixed(3)}) below threshold (${conditionData.threshold.toFixed(3)})`,
              epoch: data.epoch,
              batch: data.batch
            }
            
            get().addCriticalAlert(alert)
          }
        })
      }
    },
    
    handleStatusUpdate: (data) => {
      get().updateTrainingStatus(data.status)
    },
    
    handleSystemUpdate: (data) => {
      set((state) => ({
        systemMetrics: { ...state.systemMetrics, ...data }
      }))
    },
    
    handleInitialState: (data) => {
      if (data.status) {
        get().updateTrainingStatus(data.status)
      }
      
      if (data.metrics) {
        get().updateMetrics({
          timestamp: new Date().toISOString(),
          epoch: data.metrics.epoch || 0,
          batch: data.metrics.batch || 0,
          totalBatches: data.metrics.total_batches || 0,
          metrics: data.metrics.metrics || {},
          system: data.metrics.system || initialSystemMetrics
        })
      }
      
      if (data.history && Array.isArray(data.history)) {
        set({ metricsHistory: data.history })
      }
    }
  }))
)

// Selectors for performance
export const useTrainingStatus = () => useMonitoringStore((state) => state.trainingStatus)
export const useCurrentMetrics = () => useMonitoringStore((state) => state.currentMetrics)
export const useSystemMetrics = () => useMonitoringStore((state) => state.systemMetrics)
export const usePerformanceHistory = () => useMonitoringStore((state) => state.performanceHistory)
export const useCriticalAlerts = () => useMonitoringStore((state) => state.criticalAlerts)
export const useConnectionStatus = () => useMonitoringStore((state) => state.connectionStatus)
