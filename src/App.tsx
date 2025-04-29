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
  color?: string;
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
  const [searchMode, setSearchMode] = useState<'normal' | 'smart'>('normal');

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

      try {
        const formData = new FormData();
        formData.append(type === 'video' ? 'video' : 'image', file);

        console.log('Uploading file:', file.name, 'Type:', type);
        
        // 서버 URL 확인
        const serverUrl = 'http://localhost:5001';
        const endpoint = type === 'video' ? 'upload' : 'upload-image';
        const fullUrl = `${serverUrl}/${endpoint}`;
        
        console.log('Uploading to:', fullUrl);
        
        const response = await fetch(fullUrl, {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
          },
        });

        console.log('Server response status:', response.status);
        
        if (!response.ok) {
          const errorData = await response.json();
          console.error('Upload error:', errorData);
          throw new Error(errorData.error || '파일 업로드에 실패했습니다');
        }

        const data = await response.json();
        console.log('Upload successful:', data);
        
        // 서버 응답이 성공적일 때만 미디어 아이템 추가 및 선택
        setMediaItems(prev => [...prev, newMediaItem]);
        setSelectedMedia(newMediaItem);
        setMediaType(type);
        setMediaUrl(url);
        setDetectedObjects([]);
        setTimeline([]);

        // 비디오인 경우 자막 추출 시작
        if (type === 'video' && data.file_url) {
          const filename = data.file_url.split('/').pop();
          if (filename) {
            try {
              const subtitleResponse = await fetch(`${serverUrl}/extract-subtitles/${filename}`, {
                method: 'POST',
                headers: {
                  'Accept': 'application/json',
                },
              });

              if (subtitleResponse.ok) {
                const subtitleData = await subtitleResponse.json();
                console.log('Subtitles extracted:', subtitleData);
                // 자막 데이터 처리 (필요한 경우)
              }
            } catch (error) {
              console.error('Subtitle extraction error:', error);
              // 자막 추출 실패는 무시 (비디오 재생에는 영향 없음)
            }
          }
        }
      } catch (error) {
        console.error('Upload error:', error);
        setError(error instanceof Error ? error.message : '업로드 중 오류가 발생했습니다');
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const drawBoundingBoxes = (canvas: HTMLCanvasElement, objects: any[], mode: string) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    objects.forEach(obj => {
      const { bbox, color } = obj;
      const { x1, y1, x2, y2 } = bbox;
      
      // 색상 설정
      ctx.strokeStyle = color || (mode === 'smart' ? 'yellow' : 'red');
      ctx.lineWidth = 2;
      
      // 동그라미 그리기
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      const radius = Math.max((x2 - x1), (y2 - y1)) / 2;
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.stroke();
    });
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
        const canvas = document.getElementById('canvas') as HTMLCanvasElement;
        if (canvas) {
          drawBoundingBoxes(canvas, data.objects || [], mode);
        }
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
          {selectedMedia.type === 'video' ? (
            <video 
              src={mediaUrl} 
              controls 
              className="selected-video"
              ref={videoRef}
            />
          ) : (
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <img 
                src={mediaUrl} 
                alt="Selected" 
                ref={imageRef}
                style={{ maxWidth: '100%', maxHeight: '400px' }}
              />
              {detectedObjects.map((obj, index) => {
                const img = imageRef.current;
                if (!img) return null;
                
                const rect = img.getBoundingClientRect();
                const scaleX = rect.width / img.naturalWidth;
                const scaleY = rect.height / img.naturalHeight;
                
                const bbox = obj.bbox;
                // 첫 글자의 좌표만 사용
                const relativeX = bbox.x1 * scaleX;
                const relativeY = bbox.y1 * scaleY;
                
                return (
                  <div
                    key={index}
                    style={{
                      position: 'absolute',
                      left: `${relativeX - 10}px`,
                      top: `${relativeY - 10}px`,
                      width: '20px',
                      height: '20px',
                      border: `2px solid ${obj.color || (searchMode === 'smart' ? 'yellow' : 'red')}`,
                      borderRadius: '50%',
                      pointerEvents: 'none',
                      zIndex: 1
                    }}
                  />
                );
              })}
            </div>
          )}
          
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
              {media.type === 'video' ? (
                <video src={media.url} className="media-preview" />
              ) : (
                <img src={media.url} alt="Uploaded" className="media-preview" />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default App; 