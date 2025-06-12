import React from 'react'

const ProgressBar: React.FC = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Training Progress</h3>
        <span className="text-sm text-dark-400">Epoch 15/50</span>
      </div>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span>Overall Progress</span>
            <span>30%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '30%' }}></div>
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span>Current Epoch</span>
            <span>750/1000 batches</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '75%' }}></div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProgressBar
