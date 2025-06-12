import React from 'react'

const MetricsChart: React.FC = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Performance Metrics</h3>
      </div>
      
      <div className="chart-container">
        <div className="flex items-center justify-center h-full text-dark-400">
          <div className="text-center">
            <div className="text-lg mb-2">📈</div>
            <div>Metrics Chart</div>
            <div className="text-sm">Accuracy: 94.2% | AUC: 0.891</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MetricsChart
