import { useCallback, useEffect, useRef } from 'react'
import { useMonitoringStore } from '../store/monitoring'
import toast from 'react-hot-toast'

interface UseWebSocketReturn {
  connect: () => void
  disconnect: () => void
  sendMessage: (message: any) => void
}

export const useWebSocket = (): UseWebSocketReturn => {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5
  const reconnectDelay = 3000
  
  const {
    setConnectionStatus,
    handleMetricsUpdate,
    handleStatusUpdate,
    handleSystemUpdate,
    handleInitialState
  } = useMonitoringStore()
  
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }
    
    setConnectionStatus('connecting')
    
    try {
      // Determine WebSocket URL
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.host
      const wsUrl = `${protocol}//${host}/ws`
      
      wsRef.current = new WebSocket(wsUrl)
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setConnectionStatus('connected')
        reconnectAttemptsRef.current = 0
        
        // Clear any pending reconnect timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = null
        }
        
        toast.success('Connected to monitoring server')
      }
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleMessage(data)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setConnectionStatus('disconnected')
        
        // Attempt to reconnect if not a clean close
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          scheduleReconnect()
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          toast.error('Failed to connect to monitoring server after multiple attempts')
          setConnectionStatus('error')
        }
      }
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('error')
        toast.error('WebSocket connection error')
      }
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionStatus('error')
      toast.error('Failed to connect to monitoring server')
    }
  }, [setConnectionStatus])
  
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }
    
    setConnectionStatus('disconnected')
  }, [setConnectionStatus])
  
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      return
    }
    
    reconnectAttemptsRef.current += 1
    const delay = reconnectDelay * Math.pow(1.5, reconnectAttemptsRef.current - 1)
    
    console.log(`Scheduling reconnect attempt ${reconnectAttemptsRef.current} in ${delay}ms`)
    
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectTimeoutRef.current = null
      connect()
    }, delay) as unknown as number
  }, [connect])
  
  const handleMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'initial_state':
        handleInitialState(data)
        break
        
      case 'metrics_update':
        handleMetricsUpdate(data)
        break
        
      case 'status_update':
        handleStatusUpdate(data)
        break
        
      case 'system_update':
        handleSystemUpdate(data)
        break
        
      case 'ping':
        // Respond to ping with pong
        sendMessage({ type: 'pong' })
        break
        
      case 'pong':
        // Handle pong response (keep-alive)
        break
        
      case 'error':
        console.error('Server error:', data.message)
        toast.error(`Server error: ${data.message}`)
        break
        
      default:
        console.warn('Unknown message type:', data.type)
    }
  }, [handleInitialState, handleMetricsUpdate, handleStatusUpdate, handleSystemUpdate])
  
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message))
      } catch (error) {
        console.error('Failed to send WebSocket message:', error)
      }
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }, [])
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])
  
  // Heartbeat to keep connection alive
  useEffect(() => {
    const heartbeatInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        sendMessage({ type: 'ping' })
      }
    }, 30000) // Send ping every 30 seconds
    
    return () => {
      clearInterval(heartbeatInterval)
    }
  }, [sendMessage])
  
  return {
    connect,
    disconnect,
    sendMessage
  }
}

// Hook for sending specific WebSocket commands
export const useWebSocketCommands = () => {
  const { sendMessage } = useWebSocket()
  
  const requestHistory = useCallback((limit: number = 100) => {
    sendMessage({
      type: 'get_history',
      limit
    })
  }, [sendMessage])
  
  const requestEpochDetails = useCallback((epoch: number) => {
    sendMessage({
      type: 'get_epoch_details',
      epoch
    })
  }, [sendMessage])
  
  const startTraining = useCallback(() => {
    // This would typically be an API call, not WebSocket
    fetch('/api/training/start', { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          toast.success('Training started')
        } else {
          toast.error('Failed to start training')
        }
      })
      .catch(error => {
        console.error('Failed to start training:', error)
        toast.error('Failed to start training')
      })
  }, [])
  
  const pauseTraining = useCallback(() => {
    fetch('/api/training/pause', { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          toast.success('Training paused')
        } else {
          toast.error('Failed to pause training')
        }
      })
      .catch(error => {
        console.error('Failed to pause training:', error)
        toast.error('Failed to pause training')
      })
  }, [])
  
  const stopTraining = useCallback(() => {
    fetch('/api/training/stop', { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          toast.success('Training stopped')
        } else {
          toast.error('Failed to stop training')
        }
      })
      .catch(error => {
        console.error('Failed to stop training:', error)
        toast.error('Failed to stop training')
      })
  }, [])
  
  return {
    requestHistory,
    requestEpochDetails,
    startTraining,
    pauseTraining,
    stopTraining
  }
}
