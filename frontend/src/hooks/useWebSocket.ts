/**
 * WebSocket hook for real-time pipeline updates
 */

import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: 'status' | 'progress' | 'log';
  data?: any;
  message?: string;
}

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
}

export const useWebSocket = (options: UseWebSocketOptions) => {
  const {
    url,
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnect = true,
    reconnectInterval = 3000,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef(reconnect);
  const isConnectingRef = useRef(false);
  
  // Store callbacks in refs to avoid recreating connect function
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);
  
  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
    onErrorRef.current = onError;
  }, [onMessage, onOpen, onClose, onError]);

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connections
    if (isConnectingRef.current || (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    
    // Close existing connection if any
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (e) {
        // Ignore errors
      }
    }
    
    isConnectingRef.current = true;
    
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('WebSocket connected');
        isConnectingRef.current = false;
        setIsConnected(true);
        if (onOpenRef.current) onOpenRef.current();
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          if (onMessageRef.current) onMessageRef.current(message);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected', event.code, event.reason);
        isConnectingRef.current = false;
        setIsConnected(false);
        if (onCloseRef.current) onCloseRef.current();

        // Attempt reconnection only if enabled and not a manual close
        if (shouldReconnectRef.current && reconnectTimeoutRef.current === null && event.code !== 1000) {
          console.log(`Reconnecting in ${reconnectInterval}ms...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        isConnectingRef.current = false;
        if (onErrorRef.current) onErrorRef.current(error);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      isConnectingRef.current = false;
    }
  }, [url, reconnectInterval]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    isConnectingRef.current = false;
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.close(1000, 'Manual disconnect'); // Normal closure
      } catch (e) {
        // Ignore errors
      }
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  // Only reconnect when URL changes, not when callbacks change
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]); // Only depend on URL, not on connect/disconnect functions

  return {
    isConnected,
    lastMessage,
    sendMessage,
    disconnect,
  };
};


