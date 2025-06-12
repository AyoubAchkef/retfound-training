import React from 'react'

const ConfusionMatrix = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Confusion Matrix</h3>
      </div>
      
      <div className="chart-container">
        <div className="flex items-center justify-center h-full text-dark-400">
          <div className="text-center">
            <div className="text-lg mb-2">🔢</div>
            <div>28x28 Matrix</div>
            <div className="text-sm">Interactive Confusion Matrix</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ConfusionMatrix
