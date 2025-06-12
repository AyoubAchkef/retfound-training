import React from 'react'

const EpochDetails = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Epoch Timeline</h3>
      </div>
      
      <div className="space-y-3 max-h-64 overflow-y-auto scrollbar-thin">
        <div className="flex items-center justify-between p-3 bg-primary-900/20 border border-primary-700/50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center text-sm font-medium">15</div>
            <div>
              <div className="font-medium">Current Epoch</div>
              <div className="text-sm text-dark-400">750/1000 batches • 2.5min elapsed</div>
            </div>
          </div>
          <div className="text-primary-400">In Progress</div>
        </div>
        
        <div className="flex items-center justify-between p-3 bg-dark-800/50 border border-dark-600/50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-success-600 rounded-full flex items-center justify-center text-sm font-medium">14</div>
            <div>
              <div className="font-medium">Epoch 14</div>
              <div className="text-sm text-dark-400">Val Acc: 94.2% • Loss: 0.245</div>
            </div>
          </div>
          <div className="text-success-400">Best</div>
        </div>
        
        <div className="flex items-center justify-between p-3 bg-dark-800/50 border border-dark-600/50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-dark-600 rounded-full flex items-center justify-center text-sm font-medium">13</div>
            <div>
              <div className="font-medium">Epoch 13</div>
              <div className="text-sm text-dark-400">Val Acc: 93.8% • Loss: 0.267</div>
            </div>
          </div>
          <div className="text-dark-400">Completed</div>
        </div>
      </div>
    </div>
  )
}

export default EpochDetails
