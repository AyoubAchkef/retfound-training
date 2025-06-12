import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useMonitoringStore } from './store/monitoring'
import { useWebSocket } from './hooks/useWebSocket'
import Header from './components/Dashboard/Header'
import MetricsGrid from './components/Dashboard/MetricsGrid'
import ProgressBar from './components/Dashboard/ProgressBar'
import LossChart from './components/Charts/LossChart'
import MetricsChart from './components/Charts/MetricsChart'
import ConfusionMatrix from './components/Charts/ConfusionMatrix'
import ClassPerformance from './components/Charts/ClassPerformance'
import CriticalAlerts from './components/Monitoring/CriticalAlerts'
import GPUStats from './components/Monitoring/GPUStats'
import EpochDetails from './components/Monitoring/EpochDetails'

function App() {
  const { connectionStatus, isTraining } = useMonitoringStore()
  const { connect, disconnect } = useWebSocket()

  useEffect(() => {
    // Connect to WebSocket on mount
    connect()
    
    // Cleanup on unmount
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return (
    <div className="min-h-screen bg-dark-950">
      {/* Header */}
      <Header />
      
      {/* Main Dashboard */}
      <main className="container mx-auto px-6 py-8">
        {/* Connection Status */}
        {connectionStatus !== 'connected' && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-warning-900/20 border border-warning-700/50 rounded-lg"
          >
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-warning-500 rounded-full animate-pulse" />
              <span className="text-warning-300 text-sm font-medium">
                {connectionStatus === 'connecting' ? 'Connecting to server...' : 'Disconnected from server'}
              </span>
            </div>
          </motion.div>
        )}

        {/* Progress Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8"
        >
          <ProgressBar />
        </motion.div>

        {/* Metrics Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <MetricsGrid />
        </motion.div>

        {/* Critical Alerts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-8"
        >
          <CriticalAlerts />
        </motion.div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Loss Chart */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <LossChart />
          </motion.div>

          {/* Metrics Chart */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
          >
            <MetricsChart />
          </motion.div>

          {/* Class Performance */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
          >
            <ClassPerformance />
          </motion.div>

          {/* Confusion Matrix */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 }}
          >
            <ConfusionMatrix />
          </motion.div>
        </div>

        {/* Bottom Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* GPU Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <GPUStats />
          </motion.div>

          {/* Epoch Details */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            className="lg:col-span-2"
          >
            <EpochDetails />
          </motion.div>
        </div>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0 }}
          className="mt-12 py-8 border-t border-dark-700/50"
        >
          <div className="flex flex-col sm:flex-row justify-between items-center text-sm text-dark-400">
            <div className="flex items-center space-x-4">
              <span>RETFound Training Monitor v1.0.0</span>
              <span>•</span>
              <span>632M Parameters</span>
              <span>•</span>
              <span>28 Classes</span>
            </div>
            <div className="flex items-center space-x-2 mt-4 sm:mt-0">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-success-500' : 'bg-danger-500'
              }`} />
              <span className="capitalize">{connectionStatus}</span>
            </div>
          </div>
        </motion.footer>
      </main>
    </div>
  )
}

export default App
