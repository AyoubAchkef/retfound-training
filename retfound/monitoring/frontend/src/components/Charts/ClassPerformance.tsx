import React from 'react'

const ClassPerformance: React.FC = () => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Class Performance</h3>
      </div>
      
      <div className="chart-container">
        <div className="flex items-center justify-center h-full text-dark-400">
          <div className="text-center">
            <div className="text-lg mb-2">🎯</div>
            <div>28 Classes Heatmap</div>
            <div className="text-sm">Fundus + OCT Performance</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ClassPerformance
