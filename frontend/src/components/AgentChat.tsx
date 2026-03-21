/**
 * Agent Chat Component - OpenAI-powered imaging preprocessing assistant
 */

import React, { useState, useEffect, useRef } from 'react';
import '../styles/AgentChat.css';
import { getAgentBase } from '../lib/apiBase';

interface UploadedImage {
  id: string;
  name: string;
  file: File;
  previewUrl: string;
  base64Data: string;  // Base64 data for API
  isTiff?: boolean;  // Whether original is TIFF
  converted?: boolean;  // Whether preview was converted
  metadata?: {
    original_filename: string;
    is_tiff: boolean;
    converted: boolean;
    bit_depth: number | string | null;
    pages: number;
    width: number;
    height: number;
  };
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  parameters?: any;
  manifestInfo?: {
    saved: boolean;
    run_id?: string;
    image_id?: string;
  };
  attachedImages?: UploadedImage[];  // Images attached to this message
}

interface ApiResponse {
  success: boolean;
  message?: string;
  parameters?: any;
  error?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  manifest?: {
    saved: boolean;
    run_id?: string;
    image_id?: string;
  };
}

const AgentChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiConfigured, setApiConfigured] = useState<boolean | null>(null);
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);  // Array of uploaded images
  const [runId] = useState<string>('agent_session');  // Default run ID
  const [editingParams, setEditingParams] = useState<{messageIndex: number; params: any} | null>(null);
  const [previewModal, setPreviewModal] = useState<{image: UploadedImage | null}>({ image: null });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Derived boolean: hasImage
  const hasImage = uploadedImages.length > 0;
  
  // Debug: Log hasImage state changes (remove in production)
  useEffect(() => {
    console.log('🔍 Debug - hasImage:', hasImage, '| uploadedImages count:', uploadedImages.length, '| apiConfigured:', apiConfigured);
    if (uploadedImages.length > 0) {
      console.log('📸 Uploaded images:', uploadedImages.map(img => img.name));
    }
  }, [hasImage, uploadedImages.length, apiConfigured]);

  // Check if OpenAI API is configured on mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const checkApiHealth = async () => {
    try {
      const response = await fetch('${getAgentBase()}/api/agent/health');
      const data = await response.json();
      setApiConfigured(data.openai_configured);
    } catch (err) {
      console.error('Failed to check API health:', err);
      setApiConfigured(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    // Process each file
    for (const file of Array.from(files)) {
      // Check if it's an image (including TIFF)
      const validTypes = ['image/', 'tif', 'tiff'];
      const isValid = validTypes.some(type => 
        file.type.includes(type) || file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')
      );
      
      if (!isValid) {
        alert(`File "${file.name}" is not a valid image file. Please select JPG, PNG, TIFF, etc.`);
        continue;
      }

      // Check file size (max 50MB)
      const maxSize = 50 * 1024 * 1024;
      if (file.size > maxSize) {
        alert(`Image "${file.name}" must be smaller than ${maxSize / (1024 * 1024)}MB`);
        continue;
      }

      // Check if it's a TIFF file
      const isTiff = file.name.toLowerCase().endsWith('.tif') || 
                     file.name.toLowerCase().endsWith('.tiff') ||
                     file.type === 'image/tiff';

      // Read file as base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        try {
          const base64String = reader.result as string;
          if (!base64String) {
            console.error('Failed to read file as base64');
            alert(`Failed to read file "${file.name}". Please try again.`);
            return;
          }
          
          const base64Data = base64String.split(',')[1]; // Remove data:image/...;base64, prefix
          
          let previewUrl = base64String;
          let metadata: UploadedImage['metadata'] | undefined = undefined;
          let converted = false;

          // If TIFF, call preview API to convert to PNG
          if (isTiff) {
            try {
              const response = await fetch('${getAgentBase()}/api/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  image: base64Data,
                  filename: file.name
                })
              });

              if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
              }

              const data = await response.json();
              
              if (data.success && data.previewUrl) {
                previewUrl = data.previewUrl;
                converted = data.metadata?.converted || false;
                metadata = data.metadata;
              } else {
                // Fallback: use original (may not display in browser)
                console.warn(`Preview conversion failed for ${file.name}:`, data.error);
                // Don't show alert for fallback - just use original
              }
            } catch (err) {
              console.error(`Failed to get preview for ${file.name}:`, err);
              // Don't show alert - just use original base64String
              // The image will still be added to state, just without conversion
            }
          }
          
          const imageObj: UploadedImage = {
            id: `${Date.now()}-${Math.random()}`,
            name: file.name,
            file: file,
            previewUrl: previewUrl,
            base64Data: base64Data,
            isTiff: isTiff,
            converted: converted,
            metadata: metadata
          };
          
          setUploadedImages(prev => {
            const updated = [...prev, imageObj];
            console.log(`Image added. Total images: ${updated.length}`, imageObj.name);
            return updated;
          });
        } catch (err) {
          console.error(`Error processing file ${file.name}:`, err);
          alert(`Error processing file "${file.name}": ${err}`);
        }
      };
      
      reader.onerror = () => {
        console.error('FileReader error');
        alert(`Failed to read file "${file.name}". Please try again.`);
      };
      
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = (imageId?: string) => {
    if (imageId) {
      // Remove specific image
      setUploadedImages(prev => prev.filter(img => img.id !== imageId));
    } else {
      // Remove all images
      setUploadedImages([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSend = async () => {
    if ((!inputValue.trim() && uploadedImages.length === 0) || isLoading) return;

    // Attach images to message if present
    const attachedImages = uploadedImages.length > 0 ? [...uploadedImages] : undefined;

    const userMessage: Message = {
      role: 'user',
      content: inputValue.trim() || (uploadedImages.length > 0 ? '📷 [Image uploaded]' : ''),
      timestamp: new Date(),
      attachedImages: attachedImages
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    const messageText = inputValue.trim();
    
    // Use first image for API (backend currently supports single image)
    const imageData = uploadedImages.length > 0 ? uploadedImages[0].base64Data : null;
    const imagePath = uploadedImages.length > 0 ? uploadedImages[0].name : null;
    
    setInputValue('');
    // Clear images after sending
    setUploadedImages([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setIsLoading(true);

    try {
      // Build history for API
      const history = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // Call agent API with optional image
      const requestBody: any = {
        message: messageText,
        history: history,
        run_id: runId
      };

      if (imageData) {
        requestBody.image = imageData;
        if (imagePath) {
          requestBody.image_path = imagePath;
        }
      }

      const response = await fetch('${getAgentBase()}/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      const data: ApiResponse = await response.json();

      if (data.success && data.message) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
          parameters: data.parameters,
          manifestInfo: data.manifest
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        // Error from API
        const errorMessage: Message = {
          role: 'assistant',
          content: `❌ Error: ${data.error || 'Unknown error occurred'}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (err) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `❌ Failed to reach agent API: ${err}. Make sure the agent server is running on port 5001.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Clear all chat history?')) {
      setMessages([]);
    }
  };

  const handleQuickRecommend = async () => {
    // CRITICAL: Check if image is uploaded - fail-safe behavior
    if (!hasImage) {
      // Show fail-safe message in chat
      const failSafeMessage: Message = {
        role: 'assistant',
        content: "No image detected. Please upload a microscopy image before requesting preprocessing recommendations.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, failSafeMessage]);
      return;
    }

    if (isLoading) return;

    // Fixed prompt as specified (ONLY sent when image exists)
    const predefinedPrompt = "Analyze the uploaded microscopy image and recommend image-specific preprocessing parameters to enhance cell shape visibility and separability for accurate cell counting. Diagnose the image quality first, then provide exact numeric parameters. Output in structured TXT format only.";

    // Create user message with the predefined prompt and attached images
    const userMessage: Message = {
      role: 'user',
      content: predefinedPrompt,
      timestamp: new Date(),
      attachedImages: [...uploadedImages]  // Attach all uploaded images
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Build history for API (excluding the message we just added)
      const history = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // Call agent API with image and predefined prompt
      // Use first image for API (backend currently supports single image)
      const imageData = uploadedImages[0].base64Data;
      const imagePath = uploadedImages[0].name;
      
      const requestBody: any = {
        message: predefinedPrompt,
        history: history,
        image: imageData,
        image_path: imagePath,
        run_id: runId
      };

      const response = await fetch('${getAgentBase()}/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      const data: ApiResponse = await response.json();

      if (data.success && data.message) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
          parameters: data.parameters,
          manifestInfo: data.manifest
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        // Error from API
        const errorMessage: Message = {
          role: 'assistant',
          content: `❌ Error: ${data.error || 'Failed to get recommendations'}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (err) {
      console.error('Quick recommend failed:', err);
      const errorMessage: Message = {
        role: 'assistant',
        content: `❌ Failed to reach agent API: ${err}. Make sure the agent server is running on port 5001.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      // Note: We do NOT clear the image here, so user can use Quick Recommend again or send another message
    }
  };

  const handleApproveParams = async (messageIndex: number, message: Message) => {
    if (!message.manifestInfo?.run_id || !message.manifestInfo?.image_id) {
      alert('Cannot approve: parameters not saved to manifest');
      return;
    }

    try {
      const response = await fetch(
        `${getAgentBase()}/api/agent/manifest/${message.manifestInfo.run_id}/image/${message.manifestInfo.image_id}/approve`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        }
      );

      const data = await response.json();
      if (data.success) {
        const txtFile = data.txt_file || 'Unknown location';
        alert(`✅ Parameters approved!\n\n📄 TXT file created:\n${txtFile}\n\nCheck data/input/ folder for your readable parameters file.`);
        // Update message to show approved status
        setMessages(prev => {
          const updated = [...prev];
          if (updated[messageIndex].manifestInfo) {
            updated[messageIndex] = {
              ...updated[messageIndex],
              manifestInfo: {
                ...updated[messageIndex].manifestInfo!,
                saved: true
              }
            };
          }
          return updated;
        });
      } else {
        alert(`❌ Failed to approve: ${data.error}`);
      }
    } catch (err) {
      alert(`❌ Failed to approve: ${err}`);
    }
  };

  const handleEditParams = (messageIndex: number, params: any) => {
    setEditingParams({ messageIndex, params: {...params.recommended_parameters || params} });
  };

  const handleSaveEditedParams = async () => {
    if (!editingParams) return;

    const message = messages[editingParams.messageIndex];
    if (!message.manifestInfo?.run_id || !message.manifestInfo?.image_id) {
      alert('Cannot save: parameters not saved to manifest');
      return;
    }

    try {
      // Extract flat params
      const flatParams: any = {};
      for (const [key, value] of Object.entries(editingParams.params)) {
        if (typeof value === 'object' && value && 'value' in value) {
          flatParams[key] = (value as any).value;
        } else {
          flatParams[key] = value;
        }
      }

      const response = await fetch(
        `${getAgentBase()}/api/agent/manifest/${message.manifestInfo.run_id}/image/${message.manifestInfo.image_id}/edit`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ final_params: flatParams })
        }
      );

      const data = await response.json();
      if (data.success) {
        const txtFile = data.txt_file || 'Unknown location';
        alert(`✅ Parameters saved!\n\n📄 TXT file created:\n${txtFile}\n\nCheck data/input/ folder for your readable parameters file.`);
        setEditingParams(null);
      } else {
        alert(`❌ Failed to save: ${data.error}`);
      }
    } catch (err) {
      alert(`❌ Failed to save: ${err}`);
    }
  };

  const renderParameters = (params: any, messageIndex: number, message: Message) => {
    if (!params) return null;

    const isEditing = editingParams && editingParams.messageIndex === messageIndex;
    const manifestSaved = message.manifestInfo?.saved;

    return (
      <div className="parameters-block">
        <div className="parameters-header">
          📊 Recommended Parameters
          {manifestSaved && (
            <span className="manifest-badge">✓ Saved to Manifest</span>
          )}
        </div>
        {!isEditing ? (
          <>
            <pre className="parameters-json">
              {JSON.stringify(params, null, 2)}
            </pre>
            <div className="parameters-actions">
              <button 
                className="btn-export-params"
                onClick={() => {
                  navigator.clipboard.writeText(JSON.stringify(params, null, 2));
                  alert('Parameters copied to clipboard!');
                }}
              >
                📋 Copy Parameters
              </button>
              {manifestSaved && (
                <>
                  <button 
                    className="btn-approve-params"
                    onClick={() => handleApproveParams(messageIndex, message)}
                  >
                    ✅ Approve
                  </button>
                  <button 
                    className="btn-edit-params"
                    onClick={() => handleEditParams(messageIndex, params)}
                  >
                    ✏️ Edit
                  </button>
                </>
              )}
            </div>
          </>
        ) : (
          <div className="edit-params-mode">
            <textarea
              className="edit-params-textarea"
              value={JSON.stringify(editingParams.params, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  setEditingParams({ ...editingParams, params: parsed });
                } catch (err) {
                  // Invalid JSON, don't update
                }
              }}
              rows={15}
            />
            <div className="edit-params-actions">
              <button 
                className="btn-save-params"
                onClick={handleSaveEditedParams}
              >
                💾 Save Changes
              </button>
              <button 
                className="btn-cancel-edit"
                onClick={() => setEditingParams(null)}
              >
                ❌ Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{
        role: 'assistant',
        content: `👋 **Hello! I'm your imaging preprocessing assistant.**

I can help you with:
- 🔬 **Analyzing image quality** and preprocessing needs
- 🖼️ **Visual image analysis** - Upload your images (including TIFF!) for direct analysis
- 🎛️ **Recommending parameters** for denoising, background subtraction, contrast enhancement, etc.
- 📐 **Explaining** why certain preprocessing steps are beneficial
- 🎯 **Providing structured outputs** ready for your pipeline

**How to use:**
- 📝 **Ask questions** about preprocessing strategies
- 🖼️ **Upload images** (TIFF, PNG, JPEG, etc.) - Click "Upload Image" button
- 💬 **Describe your problem** and I'll provide specific parameter recommendations

**Supported formats:**
- ✅ TIFF (16-bit, multi-page) - Automatically converted
- ✅ PNG, JPEG, GIF, WebP
- ✅ Grayscale, RGB, multi-channel

**Example questions:**
- "My fluorescence images have high background noise. What preprocessing do you recommend?"
- "I need to enhance low-contrast exosome images. What CLAHE parameters should I use?"
- Upload a TIFF image and ask: "What preprocessing does this image need?"

Just type your question or upload an image below! 🚀`,
        timestamp: new Date()
      }]);
    }
  }, []);

  return (
    <div className="agent-chat">
      <div className="chat-header">
        <div className="header-content">
          <h2>🤖 Agent Chat - Preprocessing Assistant</h2>
          <div className="header-status">
            {apiConfigured === null && (
              <span className="status-checking">Checking API...</span>
            )}
            {apiConfigured === true && (
              <span className="status-ready">✅ OpenAI API Ready</span>
            )}
            {apiConfigured === false && (
              <span className="status-error">⚠️ API Not Configured</span>
            )}
          </div>
        </div>
        <div className="header-actions">
          <button 
            className="btn-quick-recommend"
            onClick={handleQuickRecommend}
            disabled={isLoading || !apiConfigured || !hasImage}
            title={
              !apiConfigured 
                ? "OpenAI API not configured. Check server status." 
                : !hasImage 
                  ? "Upload an image first to use Quick Recommend" 
                  : isLoading
                    ? "Processing..."
                    : "Get automatic preprocessing recommendations"
            }
            style={{
              cursor: (isLoading || !apiConfigured || !hasImage) ? 'not-allowed' : 'pointer',
              opacity: (isLoading || !apiConfigured || !hasImage) ? 0.6 : 1
            }}
          >
            ⚡ Quick Recommend
            {!hasImage && <span style={{fontSize: '0.7em', display: 'block', marginTop: '2px'}}>(Upload image first)</span>}
          </button>
          <button 
            className="btn-clear-chat"
            onClick={handleClearChat}
            disabled={messages.length === 0}
          >
            🗑️ Clear Chat
          </button>
        </div>
      </div>

      {!apiConfigured && apiConfigured !== null && (
        <div className="api-warning">
          <h3>⚠️ OpenAI API Not Configured</h3>
          <p>To use the Agent Chat, you need to:</p>
          <ol>
            <li>Set your OpenAI API key as an environment variable:
              <code>export OPENAI_API_KEY='your-key-here'</code>
            </li>
            <li>Start the agent API server:
              <code>python api_agent_chat.py</code>
            </li>
            <li>Refresh this page</li>
          </ol>
        </div>
      )}

      <div className="messages-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message message-${msg.role}`}>
            <div className="message-header">
              <span className="message-role">
                {msg.role === 'user' ? '👤 You' : '🤖 Agent'}
              </span>
              <span className="message-time">
                {msg.timestamp.toLocaleTimeString()}
              </span>
            </div>
            
            {/* Image attachment badge and filenames */}
            {msg.attachedImages && msg.attachedImages.length > 0 && (
              <div className="message-images-attached">
                <div className="image-attachment-badge">
                  🖼️ Image attached ({msg.attachedImages.length})
                </div>
                <div className="image-filenames">
                  Files: {msg.attachedImages.slice(0, 2).map(img => img.name).join(', ')}
                  {msg.attachedImages.length > 2 && ` (+${msg.attachedImages.length - 2} more)`}
                </div>
                
                {/* Show format info for TIFF images */}
                {msg.attachedImages.some(img => img.isTiff) && (
                  <div className="image-format-info">
                    {msg.attachedImages.filter(img => img.isTiff).map((img) => (
                      <div key={img.id} className="format-info-item">
                        <strong>{img.name}:</strong> Original: TIFF (stored)
                        {img.converted && <span> • Preview: PNG (converted)</span>}
                      </div>
                    ))}
                  </div>
                )}
                
                {/* Thumbnail previews (max 3) */}
                {msg.attachedImages.length > 0 && (
                  <div className="image-thumbnails">
                    {msg.attachedImages.slice(0, 3).map((img) => (
                      <div 
                        key={img.id} 
                        className="image-thumbnail"
                        onClick={() => setPreviewModal({ image: img })}
                        title={`Click to preview: ${img.name}`}
                      >
                        <img src={img.previewUrl} alt={img.name} />
                        <div className="thumbnail-overlay">
                          <span className="thumbnail-name">{img.name}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            <div className="message-content">
              {msg.content.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
            {msg.parameters && renderParameters(msg.parameters, index, msg)}
          </div>
        ))}
        
        {isLoading && (
          <div className="message message-assistant message-loading">
            <div className="message-header">
              <span className="message-role">🤖 Agent</span>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <p>Thinking...</p>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="input-wrapper">
          {/* Show uploaded images preview */}
          {uploadedImages.length > 0 && (
            <div className="uploaded-images-preview">
              {uploadedImages.map((img) => (
                <div key={img.id} className="uploaded-image-item">
                  <img src={img.previewUrl} alt={img.name} className="uploaded-image-thumb" />
                  <span className="uploaded-image-name" title={img.name}>
                    {img.name.length > 20 ? `${img.name.substring(0, 20)}...` : img.name}
                  </span>
                  <button 
                    className="btn-remove-single-image" 
                    onClick={() => handleRemoveImage(img.id)}
                    title="Remove this image"
                  >
                    ✕
                  </button>
                </div>
              ))}
              <button 
                className="btn-clear-all-images" 
                onClick={() => handleRemoveImage()}
                title="Remove all images"
              >
                Clear All
              </button>
            </div>
          )}
          <div className="input-row">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*,.tif,.tiff"
              multiple
              style={{ display: 'none' }}
            />
            <button
              className="btn-upload-image"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || !apiConfigured}
              title="Upload image(s) for analysis"
            >
              🖼️ Upload Image{uploadedImages.length > 0 ? ` (${uploadedImages.length})` : ''}
            </button>
            <textarea
              className="message-input"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={hasImage ? "Describe what you want to know about this image..." : "Ask about preprocessing parameters, image quality, or upload an image for analysis..."}
              rows={3}
              disabled={isLoading || !apiConfigured}
            />
            <button
              className="btn-send"
              onClick={handleSend}
              disabled={(!inputValue.trim() && !hasImage) || isLoading || !apiConfigured}
            >
              {isLoading ? '⏳ Sending...' : '📤 Send'}
            </button>
          </div>
        </div>
      </div>

      <div className="chat-footer">
        <p className="footer-info">
          💡 <strong>Tip:</strong> Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
        </p>
        <p className="footer-model">
          Powered by OpenAI GPT-4o-mini
        </p>
      </div>

      {/* Image Preview Modal */}
      {previewModal.image && (
        <div className="image-preview-modal" onClick={() => setPreviewModal({ image: null })}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button 
              className="modal-close" 
              onClick={() => setPreviewModal({ image: null })}
            >
              ✕
            </button>
            <h3>{previewModal.image.name}</h3>
            <img src={previewModal.image.previewUrl} alt={previewModal.image.name} />
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentChat;

