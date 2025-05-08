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
  summary?: string;
  original_text?: string;
}

const App: React.FC = () => {
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
  const [selectedMedia, setSelectedMedia] = useState<MediaItem | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ocrResults, setOcrResults] = useState<DetectedObject[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [noResults, setNoResults] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mediaType, setMediaType] = useState<'image' | 'video'>('image');
  const [mediaUrl, setMediaUrl] = useState<string>('');
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [searchMode, setSearchMode] = useState<'normal' | 'smart'>('normal');
  const [summary, setSummary] = useState<string | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [isSearchExpanded, setIsSearchExpanded] = useState(true);
  const [searchPosition, setSearchPosition] = useState({ x: window.innerWidth - 450, y: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const searchRef = useRef<HTMLDivElement>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [isProcessingYoutube, setIsProcessingYoutube] = useState(false);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      console.log('=== 파일 업로드 시작 ===');
      console.log('파일 정보:', {
        name: file.name,
        type: file.type,
        size: file.size
      });

      setIsProcessing(true);
      setIsAnalyzing(true);
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
        // 서버 URL 확인
        const serverUrl = 'http://localhost:5001';
        const endpoint = type === 'video' ? 'upload' : 'upload-image';
        const fullUrl = `${serverUrl}/${endpoint}`;

        const formData = new FormData();
        formData.append(type === 'video' ? 'video' : 'image', file);
        formData.append('analyze', 'true');

        console.log('=== OCR 분석 요청 시작 ===');
        console.log('요청 URL:', fullUrl);
        console.log('요청 데이터:', {
          type: type,
          analyze: true
        });
        
        const response = await fetch(fullUrl, {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
          },
        });

        console.log('서버 응답 상태:', response.status);
        
        if (!response.ok) {
          const errorData = await response.json();
          console.error('업로드 오류:', errorData);
          throw new Error(errorData.error || '파일 업로드에 실패했습니다');
        }

        const data = await response.json();
        console.log('=== 서버 응답 데이터 ===');
        console.log('업로드 성공:', data);

        // OCR 분석 시작
        const ocrFormData = new FormData();
        ocrFormData.append('image', file);
        ocrFormData.append('query', '');
        ocrFormData.append('mode', 'normal');

        console.log('=== OCR 분석 요청 ===');
        const ocrResponse = await fetch('http://localhost:5001/analyze-image', {
          method: 'POST',
          body: ocrFormData,
          headers: {
            'Accept': 'application/json',
          },
        });

        if (!ocrResponse.ok) {
          const errorData = await ocrResponse.json();
          console.error('OCR 분석 오류:', errorData);
          throw new Error(errorData.error || 'OCR 분석에 실패했습니다');
        }

        const ocrData = await ocrResponse.json();
        console.log('OCR 결과:', ocrData.objects ? `${ocrData.objects.length}개의 텍스트 감지됨` : '텍스트 없음');

        if (ocrData.objects) {
          setOcrResults(ocrData.objects);
          console.log('OCR 결과 저장 완료');
        }
        
        // 서버 응답이 성공적일 때만 미디어 아이템 추가 및 선택
        setMediaItems(prev => [...prev, newMediaItem]);
        setSelectedMedia(newMediaItem);
        setMediaType(type);
        setMediaUrl(url);
        setDetectedObjects([]);
        setTimeline([]);

        // 비디오인 경우 자막 추출 시작
        if (type === 'video' && data.file_url) {
          console.log('=== 비디오 자막 추출 시작 ===');
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
                console.log('자막 추출 성공:', subtitleData);
              }
            } catch (error) {
              console.error('자막 추출 오류:', error);
            }
          }
        }
      } catch (error) {
        console.error('=== 오류 발생 ===');
        console.error('오류 상세:', error);
        setError(error instanceof Error ? error.message : '업로드 중 오류가 발생했습니다');
      } finally {
        console.log('=== 처리 완료 ===');
        setIsProcessing(false);
        setIsAnalyzing(false);
      }
    }
  };

  const handleRefreshOcr = async () => {
    if (!selectedMedia) return;

    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append(selectedMedia.type === 'video' ? 'video' : 'image', selectedMedia.file);

      const response = await fetch('http://localhost:5001/analyze-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('OCR 분석에 실패했습니다');
      }

      const data = await response.json();
      if (data.objects) {
        setOcrResults(data.objects);
        setDetectedObjects([]); // 검색 결과 초기화
      }
    } catch (error) {
      console.error('OCR refresh error:', error);
      setError(error instanceof Error ? error.message : 'OCR 분석 중 오류가 발생했습니다');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSearch = async (mode: 'normal' | 'smart') => {
    if (!searchTerm || !selectedMedia) {
      setDetectedObjects([]);
      setTimeline([]);
      setNoResults(false);
      return;
    }

    console.log('=== 검색 시작 ===');
    console.log('검색어:', searchTerm);
    console.log('검색 모드:', mode);
    console.log('저장된 OCR 결과:', ocrResults);

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
        const results = data.objects || [];
        setDetectedObjects(results);
        setNoResults(results.length === 0);
        setTimeline([]);
        setMediaType('image');
        setMediaUrl(selectedMedia.url);
      } else {
        const results = data.timeline || [];
        setDetectedObjects([]);
        setTimeline(results);
        setNoResults(results.length === 0);
        setMediaType('video');
        setMediaUrl(`http://localhost:5001${data.file_url}`);
      }
    } catch (error) {
      console.error('검색 중 오류 발생:', error);
      setError(error instanceof Error ? error.message : '검색 중 오류가 발생했습니다');
      setNoResults(false);
    }
  };

  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSearchTerm = e.target.value;
    setSearchTerm(newSearchTerm);
    
    // 검색어를 완전히 지웠을 때만 하이라이트 제거
    if (newSearchTerm === '') {
      setDetectedObjects([]);
      setTimeline([]);
      setNoResults(false);
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

  const handleSummarize = async () => {
    if (!selectedMedia || selectedMedia.type !== 'image') {
      setError('이미지 파일만 요약할 수 있습니다');
      return;
    }

    setIsSummarizing(true);
    setError(null);
    setSummary(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedMedia.file);

      const response = await fetch('http://localhost:5001/summarize', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '요약 중 오류가 발생했습니다');
      }

      const data = await response.json();
      setSummary(data.summary);
    } catch (error) {
      console.error('요약 오류:', error);
      setError(error instanceof Error ? error.message : '요약 중 오류가 발생했습니다');
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.target instanceof HTMLElement && e.target.closest('.drag-handle')) {
      setIsDragging(true);
      const rect = searchRef.current?.getBoundingClientRect();
      if (rect) {
        setDragOffset({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        });
      }
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging && searchRef.current) {
      const x = e.clientX - dragOffset.x;
      const y = e.clientY - dragOffset.y;
      setSearchPosition({ x, y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  const handleYoutubeProcess = async () => {
    if (!youtubeUrl.trim()) {
      alert('YouTube URL을 입력해주세요.');
      return;
    }

    setIsProcessingYoutube(true);
    try {
      const response = await fetch('http://localhost:5001/process-youtube', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          url: youtubeUrl,
          query: '',
          mode: 'normal'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      // 비디오 아이템 생성
      const newMediaItem: MediaItem = {
        id: Date.now().toString(),
        type: 'video',
        url: data.file_url,
        file: new File([], 'youtube-video.mp4')
      };

      setMediaItems(prev => [...prev, newMediaItem]);
      setSelectedMedia(newMediaItem);
      setMediaType('video');
      setMediaUrl(data.file_url);
      setYoutubeUrl('');

      // 타임라인 처리
      if (data.timeline && data.timeline.length > 0) {
        setTimeline(data.timeline);
      }

      alert('영상 처리가 완료되었습니다.');
    } catch (error) {
      console.error('Error:', error);
      alert(`처리 중 오류가 발생했습니다: ${error instanceof Error ? error.message : '알 수 없는 오류'}`);
    } finally {
      setIsProcessingYoutube(false);
    }
  };

  return (
    <div className="App">
      <div className="app-header">
      </div>

      <div className="upload-section">
        <h3>파일 업로드</h3>
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

      <div className="youtube-input-container">
        <h3>YouTube URL</h3>
        <input
          type="text"
          value={youtubeUrl}
          onChange={(e) => setYoutubeUrl(e.target.value)}
          placeholder="YouTube URL을 입력하세요"
          className="youtube-input"
        />
        <button 
          className="youtube-button"
          onClick={handleYoutubeProcess}
          disabled={isProcessingYoutube}
        >
          <i className="fab fa-youtube"></i>
          {isProcessingYoutube ? '처리 중...' : '처리하기'}
        </button>
      </div>

      <div 
        ref={searchRef}
        className={`search-section ${isSearchExpanded ? '' : 'collapsed'}`}
        style={{ 
          left: `${searchPosition.x}px`,
          top: `${searchPosition.y}px`
        }}
        onMouseDown={handleMouseDown}
      >
        <div className="drag-handle">검색 패널</div>
        <button 
          className="toggle-button" 
          onClick={() => setIsSearchExpanded(!isSearchExpanded)}
          aria-label={isSearchExpanded ? '검색 패널 접기' : '검색 패널 펼치기'}
        >
          {isSearchExpanded ? '◀' : '▶'}
        </button>
        <div className="search-container">
          <input
            type="text"
            value={searchTerm}
            onChange={handleSearchInputChange}
            placeholder="검색어를 입력하세요"
            className="search-input"
          />
          <div className="search-buttons">
            <button
              className="search-button"
              onClick={() => handleSearch('normal')}
            >
              <div className="left-side">
                <div className="magnifying-glass"></div>
              </div>
              <div className="right-side">
                <div className="title">일반 검색</div>
                <div className="description">이미지에서 텍스트를 찾습니다</div>
              </div>
            </button>
            <button
              className="search-button smart-search"
              onClick={() => handleSearch('smart')}
            >
              <div className="left-side">
                <div className="magnifying-glass"></div>
              </div>
              <div className="right-side">
                <div className="title">스마트 검색</div>
                <div className="description">AI가 의미를 이해하고 검색합니다</div>
              </div>
            </button>
            <button
              className="search-button summarize-button"
              onClick={handleSummarize}
              disabled={!selectedMedia || selectedMedia.type !== 'image' || isSummarizing}
            >
              <div className="left-side">
                <div className="magnifying-glass"></div>
              </div>
              <div className="right-side">
                <div className="title">{isSummarizing ? '요약 중...' : '요약'}</div>
                <div className="description">이미지의 내용을 요약합니다</div>
              </div>
            </button>
          </div>
        </div>
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
              {isAnalyzing && (
                <div className="analyzing-overlay">
                  <div className="analyzing-content">
                    <i className="fas fa-spinner fa-spin"></i>
                    <span>이미지 분석 중...</span>
                  </div>
                </div>
              )}
              {detectedObjects.map((obj, index) => {
                const img = imageRef.current;
                if (!img) return null;
                
                const rect = img.getBoundingClientRect();
                const scaleX = rect.width / img.naturalWidth;
                const scaleY = rect.height / img.naturalHeight;
                
                const bbox = obj.bbox;
                const centerX = ((bbox.x1 + bbox.x2) / 2) * scaleX;
                const centerY = ((bbox.y1 + bbox.y2) / 2) * scaleY;
                const radius = Math.max((bbox.x2 - bbox.x1), (bbox.y2 - bbox.y1)) * scaleX / 2;
                
                return (
                  <div
                    key={index}
                    style={{
                      position: 'absolute',
                      left: `${centerX - radius}px`,
                      top: `${centerY - radius}px`,
                      width: `${radius * 2}px`,
                      height: `${radius * 2}px`,
                      border: `2px solid ${obj.color || (searchMode === 'smart' ? 'yellow' : 'red')}`,
                      borderRadius: '50%',
                      pointerEvents: 'none',
                      zIndex: 1
                    }}
                  />
                );
              })}
              {!isAnalyzing && (
                <button 
                  className="refresh-button"
                  onClick={handleRefreshOcr}
                  title="OCR 새로고침"
                >
                  <i className="fas fa-sync-alt"></i>
                </button>
              )}
            </div>
          )}
          
          {noResults && (
            <div className="no-results-message">
              <i className="fas fa-search"></i>
              <p>검색 결과가 없습니다.</p>
              <p className="sub-text">다른 검색어를 입력하거나 OCR을 새로고침해보세요.</p>
            </div>
          )}

          {summary && (
            <div className="summary-container">
              <h3>계약서 요약</h3>
              <div className="summary-content">
                {summary.split('\n').map((line, index) => (
                  <p key={index}>{line}</p>
                ))}
              </div>
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