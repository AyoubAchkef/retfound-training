import React from 'react'

const MetricsGrid: React.FC = () => {
  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="metric-value">0.245</div>
        <div className="metric-label">Training Loss</div>
      </div>
      
      <div className="metric-card">
        <div className="metric-value">94.2%</div>
        <div className="metric-label">Validation Accuracy</div>
      </div>
      
      <div className="metric-card">
        <div className="metric-value">0.891</div>
        <div className="metric-label">AUC-ROC</div>
      </div>
      
      <div className="metric-card">
        <div className="metric-value">0.876</div>
        <div className="metric-label">F1-Score</div>
      </div>
    </div>
  )
}

export default MetricsGrid
