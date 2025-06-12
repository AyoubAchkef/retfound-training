import React from 'react'

const LossChart = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Loss Curves</h3>
      </div>
      
      <div className="chart-container">
        <div className="flex items-center justify-center h-full text-dark-400">
          <div className="text-center">
            <div className="text-lg mb-2">📊</div>
            <div>Loss Chart</div>
            <div className="text-sm">Train: 0.245 | Val: 0.312</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LossChart
