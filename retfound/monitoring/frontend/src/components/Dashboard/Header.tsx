import React from 'react'
import { Play, Pause, Square, Clock, Activity } from 'lucide-react'
import { useMonitoringStore } from '../../store/monitoring'
import { useWebSocketCommands } from '../../hooks/useWebSocket'
import { formatDuration, formatETA } from '../../utils/formatters'

const Header: React.FC = () => {
  const { trainingStatus, connectionStatus } = useMonitoringStore()
  const { startTraining, pauseTraining, stopTraining } = useWebSocketCommands()

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training':
        return 'status-training'
      case 'paused':
        return 'status-paused'
      case 'completed':
        return 'status-completed'
      case 'error':
        return 'status-error'
      default:
        return 'status-idle'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'training':
        return <Activity className="w-4 h-4 animate-pulse" />
      case 'paused':
        return <Pause className="w-4 h-4" />
      case 'completed':
        return <Square className="w-4 h-4" />
      default:
        return <Clock className="w-4 h-4" />
    }
  }

  return (
    <header className="bg-dark-900/50 backdrop-blur-sm border-b border-dark-700/50 px-6 py-4">
      <div className="container mx-auto">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gradient">RETFound Monitor</h1>
              <p className="text-sm text-dark-400">632M Parameters • 28 Classes • Dataset v6.1</p>
            </div>
          </div>

          {/* Status and Controls */}
          <div className="flex items-center space-x-6">
            {/* Training Status */}
            <div className="flex items-center space-x-3">
              <div className={`${getStatusColor(trainingStatus.status)} flex items-center space-x-2`}>
                {getStatusIcon(trainingStatus.status)}
                <span className="capitalize font-medium">{trainingStatus.status}</span>
              </div>
              
              {trainingStatus.status === 'training' && (
                <div className="text-sm text-dark-400">
                  Epoch {trainingStatus.epoch}/{trainingStatus.totalEpochs}
                </div>
              )}
            </div>

            {/* Timer */}
            {trainingStatus.status === 'training' && (
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-dark-400" />
                  <span className="text-dark-300">
                    {formatDuration(trainingStatus.elapsedTime)}
                  </span>
                </div>
                {trainingStatus.etaSeconds && (
                  <div className="text-dark-400">
                    ETA: {formatETA(trainingStatus.etaSeconds)}
                  </div>
                )}
              </div>
            )}

            {/* Control Buttons */}
            <div className="flex items-center space-x-2">
              {trainingStatus.status === 'idle' && (
                <button
                  onClick={startTraining}
                  className="btn-primary flex items-center space-x-2"
                  disabled={connectionStatus !== 'connected'}
                >
                  <Play className="w-4 h-4" />
                  <span>Start</span>
                </button>
              )}

              {trainingStatus.status === 'training' && (
                <>
                  <button
                    onClick={pauseTraining}
                    className="btn-warning flex items-center space-x-2"
                  >
                    <Pause className="w-4 h-4" />
                    <span>Pause</span>
                  </button>
                  <button
                    onClick={stopTraining}
                    className="btn-danger flex items-center space-x-2"
                  >
                    <Square className="w-4 h-4" />
                    <span>Stop</span>
                  </button>
                </>
              )}

              {trainingStatus.status === 'paused' && (
                <>
                  <button
                    onClick={startTraining}
                    className="btn-primary flex items-center space-x-2"
                  >
                    <Play className="w-4 h-4" />
                    <span>Resume</span>
                  </button>
                  <button
                    onClick={stopTraining}
                    className="btn-danger flex items-center space-x-2"
                  >
                    <Square className="w-4 h-4" />
                    <span>Stop</span>
                  </button>
                </>
              )}
            </div>

            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-success-500' : 
                connectionStatus === 'connecting' ? 'bg-warning-500 animate-pulse' : 
                'bg-danger-500'
              }`} />
              <span className="text-sm text-dark-400 capitalize">
                {connectionStatus}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
