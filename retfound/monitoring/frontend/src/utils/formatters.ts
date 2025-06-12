import { format, formatDistanceToNow, parseISO } from 'date-fns'

/**
 * Format a number to a specific number of decimal places
 */
export const formatNumber = (value: number, decimals: number = 2): string => {
  if (isNaN(value) || !isFinite(value)) {
    return 'N/A'
  }
  return value.toFixed(decimals)
}

/**
 * Format a percentage value
 */
export const formatPercentage = (value: number, decimals: number = 1): string => {
  if (isNaN(value) || !isFinite(value)) {
    return 'N/A'
  }
  return `${(value * 100).toFixed(decimals)}%`
}

/**
 * Format bytes to human readable format
 */
export const formatBytes = (bytes: number, decimals: number = 2): string => {
  if (bytes === 0) return '0 B'
  
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

/**
 * Format duration in seconds to human readable format
 */
export const formatDuration = (seconds: number): string => {
  if (isNaN(seconds) || !isFinite(seconds) || seconds < 0) {
    return 'N/A'
  }
  
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}

/**
 * Format ETA (estimated time of arrival)
 */
export const formatETA = (seconds: number): string => {
  if (isNaN(seconds) || !isFinite(seconds) || seconds <= 0) {
    return 'N/A'
  }
  
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  
  if (hours > 24) {
    const days = Math.floor(hours / 24)
    const remainingHours = hours % 24
    return `${days}d ${remainingHours}h`
  } else if (hours > 0) {
    return `${hours}h ${minutes}m`
  } else if (minutes > 0) {
    return `${minutes}m`
  } else {
    return '< 1m'
  }
}

/**
 * Format timestamp to relative time
 */
export const formatRelativeTime = (timestamp: string): string => {
  try {
    const date = parseISO(timestamp)
    return formatDistanceToNow(date, { addSuffix: true })
  } catch (error) {
    return 'Invalid date'
  }
}

/**
 * Format timestamp to absolute time
 */
export const formatAbsoluteTime = (timestamp: string, formatString: string = 'PPpp'): string => {
  try {
    const date = parseISO(timestamp)
    return format(date, formatString)
  } catch (error) {
    return 'Invalid date'
  }
}

/**
 * Format learning rate in scientific notation
 */
export const formatLearningRate = (lr: number): string => {
  if (isNaN(lr) || !isFinite(lr)) {
    return 'N/A'
  }
  
  if (lr === 0) {
    return '0'
  }
  
  return lr.toExponential(2)
}

/**
 * Format large numbers with K, M, B suffixes
 */
export const formatLargeNumber = (num: number, decimals: number = 1): string => {
  if (isNaN(num) || !isFinite(num)) {
    return 'N/A'
  }
  
  if (num === 0) return '0'
  
  const k = 1000
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['', 'K', 'M', 'B', 'T']
  
  const i = Math.floor(Math.log(Math.abs(num)) / Math.log(k))
  
  if (i === 0) {
    return num.toString()
  }
  
  return parseFloat((num / Math.pow(k, i)).toFixed(dm)) + sizes[i]
}

/**
 * Format GPU temperature
 */
export const formatTemperature = (temp: number): string => {
  if (isNaN(temp) || !isFinite(temp)) {
    return 'N/A'
  }
  
  return `${Math.round(temp)}°C`
}

/**
 * Format metric value with appropriate precision
 */
export const formatMetric = (value: number, metricType: string): string => {
  if (isNaN(value) || !isFinite(value)) {
    return 'N/A'
  }
  
  switch (metricType.toLowerCase()) {
    case 'loss':
      return formatNumber(value, 4)
    case 'accuracy':
    case 'precision':
    case 'recall':
    case 'f1':
    case 'auc':
    case 'sensitivity':
    case 'specificity':
      return formatPercentage(value, 2)
    case 'learning_rate':
    case 'lr':
      return formatLearningRate(value)
    default:
      return formatNumber(value, 3)
  }
}

/**
 * Get status color class based on value and thresholds
 */
export const getStatusColor = (
  value: number,
  goodThreshold: number,
  warningThreshold: number,
  isHigherBetter: boolean = true
): string => {
  if (isNaN(value) || !isFinite(value)) {
    return 'text-dark-400'
  }
  
  if (isHigherBetter) {
    if (value >= goodThreshold) {
      return 'text-success-400'
    } else if (value >= warningThreshold) {
      return 'text-warning-400'
    } else {
      return 'text-danger-400'
    }
  } else {
    if (value <= goodThreshold) {
      return 'text-success-400'
    } else if (value <= warningThreshold) {
      return 'text-warning-400'
    } else {
      return 'text-danger-400'
    }
  }
}

/**
 * Get progress bar color class based on value and thresholds
 */
export const getProgressColor = (
  value: number,
  goodThreshold: number,
  warningThreshold: number,
  isHigherBetter: boolean = true
): string => {
  if (isNaN(value) || !isFinite(value)) {
    return 'bg-dark-600'
  }
  
  if (isHigherBetter) {
    if (value >= goodThreshold) {
      return 'bg-success-500'
    } else if (value >= warningThreshold) {
      return 'bg-warning-500'
    } else {
      return 'bg-danger-500'
    }
  } else {
    if (value <= goodThreshold) {
      return 'bg-success-500'
    } else if (value <= warningThreshold) {
      return 'bg-warning-500'
    } else {
      return 'bg-danger-500'
    }
  }
}

/**
 * Calculate trend from array of values
 */
export const calculateTrend = (values: number[], windowSize: number = 5): 'up' | 'down' | 'stable' => {
  if (values.length < 2) {
    return 'stable'
  }
  
  const recentValues = values.slice(-windowSize)
  if (recentValues.length < 2) {
    return 'stable'
  }
  
  const first = recentValues[0]
  const last = recentValues[recentValues.length - 1]
  const change = (last - first) / Math.abs(first)
  
  const threshold = 0.01 // 1% change threshold
  
  if (change > threshold) {
    return 'up'
  } else if (change < -threshold) {
    return 'down'
  } else {
    return 'stable'
  }
}

/**
 * Get trend icon based on trend direction
 */
export const getTrendIcon = (trend: 'up' | 'down' | 'stable'): string => {
  switch (trend) {
    case 'up':
      return '↗'
    case 'down':
      return '↘'
    case 'stable':
    default:
      return '→'
  }
}

/**
 * Get trend color class based on trend direction and whether higher is better
 */
export const getTrendColor = (
  trend: 'up' | 'down' | 'stable',
  isHigherBetter: boolean = true
): string => {
  if (trend === 'stable') {
    return 'text-dark-400'
  }
  
  if (isHigherBetter) {
    return trend === 'up' ? 'text-success-400' : 'text-danger-400'
  } else {
    return trend === 'up' ? 'text-danger-400' : 'text-success-400'
  }
}

/**
 * Format class name for display
 */
export const formatClassName = (className: string): string => {
  return className
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .replace(/\b\w/g, l => l.toUpperCase())
    .trim()
}

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) {
    return text
  }
  
  return text.substring(0, maxLength - 3) + '...'
}

/**
 * Format epoch display
 */
export const formatEpoch = (epoch: number, totalEpochs: number): string => {
  return `${epoch}/${totalEpochs}`
}

/**
 * Format batch display
 */
export const formatBatch = (batch: number, totalBatches: number): string => {
  return `${batch}/${totalBatches}`
}

/**
 * Get alert level color
 */
export const getAlertColor = (level: 'critical' | 'warning' | 'info' | 'success'): string => {
  switch (level) {
    case 'critical':
      return 'text-danger-400 bg-danger-900/20 border-danger-700/50'
    case 'warning':
      return 'text-warning-400 bg-warning-900/20 border-warning-700/50'
    case 'info':
      return 'text-primary-400 bg-primary-900/20 border-primary-700/50'
    case 'success':
      return 'text-success-400 bg-success-900/20 border-success-700/50'
    default:
      return 'text-dark-400 bg-dark-900/20 border-dark-700/50'
  }
}

/**
 * Format confidence score
 */
export const formatConfidence = (confidence: number): string => {
  if (isNaN(confidence) || !isFinite(confidence)) {
    return 'N/A'
  }
  
  return `${(confidence * 100).toFixed(1)}%`
}

/**
 * Clamp value between min and max
 */
export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max)
}

/**
 * Linear interpolation between two values
 */
export const lerp = (start: number, end: number, factor: number): number => {
  return start + (end - start) * clamp(factor, 0, 1)
}

/**
 * Convert hex color to RGB
 */
export const hexToRgb = (hex: string): { r: number; g: number; b: number } | null => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null
}

/**
 * Generate color for chart series
 */
export const generateChartColor = (index: number, opacity: number = 1): string => {
  const colors = [
    '#8b5cf6', // primary
    '#10b981', // success
    '#f59e0b', // warning
    '#ef4444', // danger
    '#06b6d4', // cyan
    '#8b5a2b', // brown
    '#ec4899', // pink
    '#84cc16', // lime
  ]
  
  const color = colors[index % colors.length]
  const rgb = hexToRgb(color)
  
  if (rgb && opacity < 1) {
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`
  }
  
  return color
}
