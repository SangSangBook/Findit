import React, { useState, useEffect, useRef } from 'react';
import './App.css';

interface CharBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface DetectedObject {
  text: string;
  bbox: CharBox;
}

interface MediaItem {
  id: string;
  type: 'video' | 'image';
  url: string;
  file: File;
}

interface TimelineItem {
  timestamp: number;
  texts: DetectedObject[];
}

interface ApiResponse {
  type: 'image' | 'video';
  objects?: DetectedObject[];
  ocr_text?: string;
  file_url?: string;
  timeline?: TimelineItem[];
}

const App: React.FC = () => {
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
  const [selectedMedia, setSelectedMedia] = useState<MediaItem | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mediaType, setMediaType] = useState<'image' | 'video'>('image');
  const [mediaUrl, setMediaUrl] = useState<string>('');
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setIsProcessing(true);
      setError(null);
      const url = URL.createObjectURL(file);
      const type = file.type.startsWith('video/') ? 'video' : 'image';

      const newMediaItem: MediaItem = {
        id: Date.now().toString(),
        type,
        url,
        file
      };

      setMediaItems(prev => [...prev, newMediaItem]);
      setSelectedMedia(newMediaItem);
      setMediaType(type);
      setMediaUrl(url);
      setDetectedObjects([]);
      setTimeline([]);
      setIsProcessing(false);
    }
  };

  const handleSearch = async (mode: 'normal' | 'smart') => {
    if (!searchTerm || !selectedMedia) {
      setDetectedObjects([]);
      setTimeline([]);
      return;
    }

    try {
      const formData = new FormData();
      formData.append('image', selectedMedia.file);
      formData.append('query', searchTerm);
      formData.append('mode', mode);

      const response = await fetch('http://localhost:5001/analyze-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '검색 중 오류가 발생했습니다');
      }

      const data: ApiResponse = await response.json();
      
      if (data.type === 'image') {
        setDetectedObjects(data.objects || []);
        setTimeline([]);
        setMediaType('image');
        setMediaUrl(selectedMedia.url);
      } else {
        setDetectedObjects([]);
        setTimeline(data.timeline || []);
        setMediaType('video');
        // 서버에서 받은 비디오 URL 사용
        setMediaUrl(`http://localhost:5001${data.file_url}`);
      }
    } catch (error) {
      console.error('검색 오류:', error);
      setError(error instanceof Error ? error.message : '검색 중 오류가 발생했습니다');
    }
  };

  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSearchTerm = e.target.value;
    setSearchTerm(newSearchTerm);
    
    // 검색어를 완전히 지웠을 때만 하이라이트 제거
    if (newSearchTerm === '') {
      setDetectedObjects([]);
    }
  };

  const seekToTimestamp = (timestamp: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      videoRef.current.play();  // 해당 시점부터 자동 재생
    }
  };

  // MediaItem 클릭 핸들러
  const handleMediaItemClick = (media: MediaItem) => {
    setSelectedMedia(media);
    setMediaType(media.type);
    setMediaUrl(media.url);
    setDetectedObjects([]); // 이전 검색 결과 초기화
    setTimeline([]); // 타임라인 초기화
  };

  return (
    <div className="App">
      <h1>찾기</h1>
      
      <div className="search-section">
        <div className="search-container">
          <input
            type="text"
            placeholder="검색어를 입력하세요..."
            value={searchTerm}
            onChange={handleSearchInputChange}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleSearch('normal');
              }
            }}
            className="search-input"
          />
          <div className="search-buttons">
            <button 
              onClick={() => handleSearch('normal')}
              className="search-button"
            >
              검색
            </button>
            <button 
              onClick={() => handleSearch('smart')}
              className="search-button smart"
            >
              스마트 검색
            </button>
          </div>
        </div>
      </div>
      
      <div className="upload-section">
        <div className="upload-options">
          <div className="upload-option">
            <h3>이미지 업로드</h3>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="file-input"
            />
          </div>
          <div className="upload-option">
            <h3>동영상 업로드</h3>
            <input
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              className="file-input"
            />
          </div>
        </div>
        {isProcessing && <p>Processing...</p>}
        {error && <p className="error">{error}</p>}
      </div>

      {selectedMedia && (
        <div className="selected-media">
          <div style={{ 
            position: 'relative',
            display: 'inline-block',
            width: 'fit-content'
          }}>
            <img 
              src={mediaUrl} 
              alt="Selected" 
              style={{ 
                maxWidth: '100%',
                maxHeight: '400px',
                display: 'block'
              }}
              ref={imageRef}
            />
            {detectedObjects.map((obj, index) => {
              const img = imageRef.current;
              if (!img) return null;
              
              const rect = img.getBoundingClientRect();
              console.log("\n=== 클라이언트 좌표 디버깅 ===");
              console.log("텍스트:", obj.text);
              console.log("원본 좌표:", obj.bbox);
              console.log("이미지 실제 크기:", {
                naturalWidth: img.naturalWidth,
                naturalHeight: img.naturalHeight
              });
              console.log("화면 표시 크기:", {
                width: rect.width,
                height: rect.height
              });
              
              const scaleX = rect.width / img.naturalWidth;
              const scaleY = rect.height / img.naturalHeight;
              
              // 이미지 내에서의 상대 위치 계산
              const relativeX = obj.bbox.x1 * scaleX;
              const relativeY = obj.bbox.y1 * scaleY;
              
              console.log("스케일:", { scaleX, scaleY });
              console.log("계산된 상대 위치:", { relativeX, relativeY });
              
              const circleSize = 20;
              
              return (
                <div
                  key={index}
                  style={{
                    position: 'absolute',
                    left: `${relativeX - circleSize/2}px`,
                    top: `${relativeY - circleSize/2}px`,
                    width: `${circleSize}px`,
                    height: `${circleSize}px`,
                    border: '2px solid red',
                    borderRadius: '50%',
                    pointerEvents: 'none',
                    zIndex: 1,
                    backgroundColor: 'transparent'
                  }}
                />
              );
            })}
          </div>

          {timeline.length > 0 && (
            <div className="timeline-container">
              <h3>타임라인</h3>
              <div className="timeline">
                {timeline.map((item, index) => (
                  <div 
                    key={index} 
                    className="timeline-item"
                    onClick={() => seekToTimestamp(item.timestamp)}
                  >
                    <span className="timestamp">
                      {Math.floor(item.timestamp / 60)}:{Math.floor(item.timestamp % 60).toString().padStart(2, '0')}
                    </span>
                    <div className="texts">
                      {item.texts.map((text, i) => (
                        <div key={i} className="detected-text">
                          {text.text}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {mediaItems.length > 0 && (
        <div className="media-grid">
          {mediaItems.map(media => (
            <div
              key={media.id}
              className={`media-item ${selectedMedia?.id === media.id ? 'selected' : ''}`}
              onClick={() => handleMediaItemClick(media)}
            >
              <img src={media.url} alt="Uploaded" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default App; 