import React from 'react'

const GPUStats = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">GPU & System</h3>
      </div>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span>GPU Usage</span>
            <span>87%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '87%' }}></div>
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span>GPU Memory</span>
            <span>20.1 / 24 GB</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '84%' }}></div>
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span>RAM Usage</span>
            <span>28.4 / 64 GB</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '44%' }}></div>
          </div>
        </div>
        
        <div className="flex justify-between text-sm">
          <span>GPU Temp</span>
          <span>72°C</span>
        </div>
      </div>
    </div>
  )
}

export default GPUStats
