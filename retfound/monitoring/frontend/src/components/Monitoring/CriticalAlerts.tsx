import React from 'react'

const CriticalAlerts = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Critical Conditions</h3>
      </div>
      
      <div className="space-y-3">
        <div className="flex items-center justify-between p-3 bg-success-900/20 border border-success-700/50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="text-success-400">✅</div>
            <div>
              <div className="font-medium">RAO</div>
              <div className="text-sm text-dark-400">99.2% / 99%</div>
            </div>
          </div>
          <div className="text-success-400 font-medium">OK</div>
        </div>
        
        <div className="flex items-center justify-between p-3 bg-warning-900/20 border border-warning-700/50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="text-warning-400">⚠️</div>
            <div>
              <div className="font-medium">RVO</div>
              <div className="text-sm text-dark-400">96.8% / 97%</div>
            </div>
          </div>
          <div className="text-warning-400 font-medium">WARNING</div>
        </div>
        
        <div className="flex items-center justify-between p-3 bg-success-900/20 border border-success-700/50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="text-success-400">✅</div>
            <div>
              <div className="font-medium">Retinal Detachment</div>
              <div className="text-sm text-dark-400">99.8% / 99%</div>
            </div>
          </div>
          <div className="text-success-400 font-medium">OK</div>
        </div>
      </div>
    </div>
  )
}

export default CriticalAlerts
