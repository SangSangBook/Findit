import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import NetflixLoader from './components/NetflixLoader';
import MediaUploader from './components/MediaUploader';
import { ImageType, IMAGE_TYPE_ICONS, IMAGE_TYPE_LABELS } from './types';
import ImageTypeSelector from './components/ImageTypeSelector';

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
  fileName?: string;
  pageIndex?: number;
  match_type?: string;
}

interface MediaItem {
  id: string;
  type: 'image' | 'video';
  url: string;
  file: File;
  sessionId?: string;
  imageType?: ImageType;
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
  const [isLoading, setIsLoading] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [searchResultPages, setSearchResultPages] = useState<number[]>([]);
  const [pageNotification, setPageNotification] = useState<{show: boolean, direction: 'prev' | 'next' | null}>({
    show: false,
    direction: null
  });
  const [chatMessage, setChatMessage] = useState('');
  const [chatResponse, setChatResponse] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isModalImageLoaded, setIsModalImageLoaded] = useState(false);
  const modalImageRef = useRef<HTMLImageElement>(null);
  const [selectedImageType, setSelectedImageType] = useState<ImageType>('OTHER');
  const [ocrText, setOcrText] = useState('');

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    console.log('=== 파일 업로드 시작 ===');
    console.log('파일 개수:', files.length);

    setIsProcessing(true);
    setIsAnalyzing(true);
    setError(null);

    try {
      // 파일 타입에 따라 다른 엔드포인트 사용
      const file = files[0];  // 비디오는 한 번에 하나만 업로드
      const isVideo = file.type.startsWith('video/');
      const formData = new FormData();
      
      if (isVideo) {
        formData.append('video', file);
        formData.append('query', searchTerm);
        formData.append('mode', searchMode);
        
        const response = await fetch('http://localhost:5001/upload-video', {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
          },
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || '비디오 업로드에 실패했습니다');
        }

        const data = await response.json();
        console.log('=== 서버 응답 데이터 ===');
        console.log('업로드 성공:', data);

        // 세션 ID 저장
        if (!sessionId) {
          setSessionId(data.session_id);
        }

        // OCR 텍스트 저장
        if (data.text) {
          console.log('OCR 텍스트:', data.text);
          setOcrText(data.text);
        }

        // 비디오 URL 생성
        const videoUrl = URL.createObjectURL(file);

        // 비디오 아이템 생성
        const newMediaItem: MediaItem = {
          id: Date.now().toString(),
          type: 'video',
          url: videoUrl,
          file,
          sessionId: sessionId || data.session_id,
        };

        setMediaItems(prev => [...prev, newMediaItem]);
        setSelectedMedia(newMediaItem);
        setMediaType('video');
        setMediaUrl(videoUrl);
        
        // 타임라인 설정
        if (data.file.timeline) {
          setTimeline(data.file.timeline);
        }
      } else {
        // 이미지 파일 업로드
        for (let i = 0; i < files.length; i++) {
          formData.append('images[]', files[i]);
        }

        const response = await fetch('http://localhost:5001/upload-image', {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json',
          },
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || '이미지 업로드에 실패했습니다');
        }

        const data = await response.json();
        console.log('=== 서버 응답 데이터 ===');
        console.log('업로드 성공:', data);

        // 세션 ID 저장
        if (!sessionId) {
          setSessionId(data.session_id);
        }

        // OCR 텍스트 저장
        if (data.text) {
          console.log('OCR 텍스트:', data.text);
          setOcrText(data.text);
        }

        // 이미지 타입 설정
        if (data.image_type) {
          console.log('감지된 이미지 타입:', data.image_type);
          setSelectedImageType(data.image_type as ImageType);
        }

        // 각 파일에 대한 미디어 아이템 생성
        const newMediaItems: MediaItem[] = Array.from(files).map((file, index) => {
          const url = URL.createObjectURL(file);
          return {
            id: `${Date.now()}_${index}`,
            type: 'image',
            url,
            file,
            sessionId: sessionId || data.session_id,
            imageType: data.image_type as ImageType
          };
        });

        // 새로운 미디어 아이템을 기존 아이템에 추가하고 현재 페이지를 마지막 페이지로 설정
        setMediaItems(prev => {
          const updatedItems = [...prev, ...newMediaItems];
          setCurrentPage(updatedItems.length - 1);
          return updatedItems;
        });

        // 마지막 미디어 아이템을 선택
        const lastMediaItem = newMediaItems[newMediaItems.length - 1];
        setSelectedMedia(lastMediaItem);
        setMediaType('image');
        setMediaUrl(lastMediaItem.url);
      }

      setDetectedObjects([]);

    } catch (error) {
      console.error('=== 오류 발생 ===');
      console.error('오류 상세:', error);
      setError(error instanceof Error ? error.message : '업로드 중 오류가 발생했습니다');
    } finally {
      console.log('=== 처리 완료 ===');
      setIsProcessing(false);
      setIsAnalyzing(false);
    }
  };

  const handleSearch = async (mode: 'normal' | 'smart') => {
    if (!searchTerm || !selectedMedia || !sessionId) {
      setDetectedObjects([]);
      setTimeline([]);
      setNoResults(false);
      setSearchResultPages([]);
      setPageNotification({ show: false, direction: null });
      return;
    }

    console.log('=== 검색 시작 ===');
    console.log('검색어:', searchTerm);
    console.log('검색 모드:', mode);
    console.log('세션 ID:', sessionId);
    console.log('현재 페이지:', currentPage);

    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);
      formData.append('query', searchTerm);
      formData.append('mode', mode);
      formData.append('images[]', selectedMedia.file);

      const response = await fetch('http://localhost:5001/analyze-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '검색 중 오류가 발생했습니다');
      }

      const data = await response.json();
      console.log('=== 서버 응답 데이터 ===');
      console.log('전체 데이터:', JSON.stringify(data, null, 2));
      
      // OCR 텍스트 저장
      if (data.text) {
        console.log('OCR 텍스트:', data.text);
        setOcrText(data.text);
      }
      
      if (data.matches && data.matches.length > 0) {
        const searchResults: DetectedObject[] = data.matches.map((obj: any) => ({
          text: obj.text,
          bbox: obj.bbox,
          confidence: obj.confidence,
          pageIndex: currentPage,
          match_type: obj.match_type || 'exact'
        }));
        
        console.log('검색 결과:', searchResults);
        setDetectedObjects(searchResults);
        setNoResults(false);
        setTimeline([]);

        // 검색 결과가 있는 페이지 번호 저장
        const pages = [currentPage];
        console.log('검색 결과가 있는 페이지들:', pages);
        setSearchResultPages(pages);
        setPageNotification({ show: false, direction: null });
      } else {
        setDetectedObjects([]);
        setNoResults(true);
        setTimeline([]);
        setSearchResultPages([]);
        setPageNotification({ show: false, direction: null });
      }
    } catch (error) {
      console.error('검색 오류:', error);
      setError(error instanceof Error ? error.message : '검색 중 오류가 발생했습니다');
    }
  };

  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSearchTerm = e.target.value;
    setSearchTerm(newSearchTerm);
    
    // 검색어가 비어있을 때만 검색 결과를 초기화
    if (newSearchTerm === '') {
      setDetectedObjects([]);
      setNoResults(false);
      setTimeline([]);
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
    if (!sessionId) {
      setError('이미지가 업로드되지 않았습니다');
      return;
    }

    setIsSummarizing(true);
    setError(null);
    setSummary(null);

    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);

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

  const handleChat = async () => {
    if (!chatMessage.trim() || !sessionId) return;
    
    try {
      console.log('=== 채팅 요청 시작 ===');
      console.log('세션 ID:', sessionId);
      console.log('질문:', chatMessage);
      
      const formData = new FormData();
      formData.append('session_id', sessionId);
      formData.append('message', chatMessage);
      
      // 모든 이미지 파일 추가 (현재 선택된 이미지와 관계없이)
      const imageFiles = mediaItems
        .filter(item => item.type === 'image')
        .map(item => item.file);
      
      console.log('전송할 이미지 개수:', imageFiles.length);
      
      // 모든 이미지 파일을 순서대로 추가하고 각 이미지의 타입 정보도 함께 전송
      imageFiles.forEach((file, index) => {
        const mediaItem = mediaItems.find(item => item.file === file);
        console.log(`이미지 ${index + 1} 추가:`, file.name, '타입:', mediaItem?.imageType);
        formData.append('images', file);
        if (mediaItem?.imageType) {
          formData.append(`image_types[${index}]`, mediaItem.imageType);
        }
      });
      
      console.log('FormData 내용:');
      Array.from(formData.entries()).forEach(([key, value]) => {
        console.log(key, value);
      });
      
      const response = await fetch('http://localhost:5001/summarize', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '채팅 처리 중 오류가 발생했습니다.');
      }
      
      const data = await response.json();
      console.log('서버 응답:', data);
      
      // 응답이 배열인 경우 각 이미지에 대한 설명을 순서대로 표시
      if (Array.isArray(data.summary)) {
        const formattedResponse = data.summary.map((summary: string, index: number) => {
          const mediaItem = mediaItems[index];
          const imageType = mediaItem?.imageType || '알 수 없음';
          return `${index + 1}번째 이미지 (${imageType}):\n${summary}\n`;
        }).join('\n');
        setChatResponse(formattedResponse);
      } else {
        setChatResponse(data.summary);
      }
    } catch (error) {
      console.error('채팅 처리 중 오류:', error);
      alert('채팅 처리 중 오류가 발생했습니다.');
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

    // YouTube URL 형식 검증 및 변환
    let videoId = '';
    try {
      const url = new URL(youtubeUrl);
      if (url.hostname === 'youtube.com' || url.hostname === 'www.youtube.com') {
        videoId = url.searchParams.get('v') || '';
      } else if (url.hostname === 'youtu.be') {
        videoId = url.pathname.slice(1);
      }
      
      if (!videoId) {
        throw new Error('유효한 YouTube URL이 아닙니다.');
      }
    } catch (error) {
      alert('올바른 YouTube URL을 입력해주세요.');
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
          video_id: videoId,
          query: '',
          mode: 'normal'
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      // YouTube 비디오는 미디어 그리드에 추가하지 않고 youtube-preview에만 표시
      const youtubeVideoUrl = `youtube.com/${data.file_url}`;  // YouTube URL 형식으로 변경
      setSelectedMedia({
        id: Date.now().toString(),
        type: 'video',
        url: youtubeVideoUrl,  // YouTube URL 형식으로 설정
        file: new File([], 'youtube-video.mp4'),
        sessionId: data.session_id
      });
      setMediaType('video');
      setMediaUrl(youtubeVideoUrl);  // YouTube URL 형식으로 설정
      setYoutubeUrl('');
      
      // 타임라인 설정
      if (data.timeline) {
        setTimeline(data.timeline);
      }
      
      // OCR 텍스트 설정
      if (data.ocr_text) {
        setOcrText(data.ocr_text);
      }
      
      // 세션 ID 설정
      if (data.session_id) {
        setSessionId(data.session_id);
      }

    } catch (error) {
      console.error('YouTube 처리 중 오류:', error);
      setError(error instanceof Error ? error.message : 'YouTube 처리 중 오류가 발생했습니다');
    } finally {
      setIsProcessingYoutube(false);
    }
  };

  useEffect(() => {
    // 3초 후에 로딩 화면을 숨깁니다
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  // 페이지 변경 시 알림 상태 업데이트
  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
    if (searchResultPages.length > 0) {
      const nextPage = searchResultPages.find(p => p > newPage);
      const prevPage = searchResultPages.find(p => p < newPage);
      
      if (nextPage) {
        setPageNotification({ show: true, direction: 'next' });
      } else if (prevPage) {
        setPageNotification({ show: true, direction: 'prev' });
      } else {
        setPageNotification({ show: false, direction: null });
      }
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 0) {
      handlePageChange(currentPage - 1);
      const prevMedia = mediaItems[currentPage - 1];
      setSelectedMedia(prevMedia);
      setMediaType(prevMedia.type);
      setMediaUrl(prevMedia.url);
      // 검색 결과는 유지
      if (prevMedia.imageType) {
        setSelectedImageType(prevMedia.imageType);
      }
    }
  };

  const handleNextPage = () => {
    if (currentPage < mediaItems.length - 1) {
      handlePageChange(currentPage + 1);
      const nextMedia = mediaItems[currentPage + 1];
      setSelectedMedia(nextMedia);
      setMediaType(nextMedia.type);
      setMediaUrl(nextMedia.url);
      // 검색 결과는 유지
      if (nextMedia.imageType) {
        setSelectedImageType(nextMedia.imageType);
      }
    }
  };

  const handleModalImageLoad = () => {
    setIsModalImageLoaded(true);
  };

  const handleImageTypeSelect = (type: ImageType) => {
    setSelectedImageType(type);
    if (selectedMedia) {
      const updatedMediaItems = mediaItems.map(item => 
        item.id === selectedMedia.id ? { ...item, imageType: type } : item
      );
      setMediaItems(updatedMediaItems);
    }
  };

  if (isLoading) {
    return <NetflixLoader />;
  }

  return (
    <div className="App">
      <div className="left-section">
        <div className="app-logo">Findit!</div>
        <div className="app-subtitle">미디어에서{'\n'}정보를{'\n'}찾아주세요</div>
        <div className="upload-section">
          <div className="upload-options">
            <button
              onClick={() => document.getElementById('image-upload')?.click()}
              className="upload-button"
            >
              <i className="fas fa-camera"></i>
              사진 업로드하기
            </button>
            <input
              id="image-upload"
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileUpload}
              className="file-input"
              style={{ display: 'none' }}
            />
            <button
              onClick={() => document.getElementById('video-upload')?.click()}
              className="upload-button"
            >
              <i className="fas fa-video"></i>
              영상 업로드하기
            </button>
            <input
              id="video-upload"
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              className="file-input"
              style={{ display: 'none' }}
            />
          </div>
          {isProcessing && <p>처리 중...</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </div>

      <div className={`search-section ${isSearchExpanded ? '' : 'collapsed'}`}>
        <div className="search-container">
          <h2 className="search-title">검색패널</h2>
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
        <button 
          className="toggle-button" 
          onClick={() => setIsSearchExpanded(!isSearchExpanded)}
        >
          {isSearchExpanded ? '◀' : '▶'}
        </button>
      </div>

      <div className="right-section">
        <div className="media-container">
          <div className={`selected-media ${selectedMedia ? 'has-media' : ''}`}>
            {selectedMedia ? (
              selectedMedia.type === 'video' && !selectedMedia.url.includes('youtube.com') ? (
                <video 
                  src={mediaUrl} 
                  controls 
                  className="selected-video"
                  ref={videoRef}
                />
              ) : selectedMedia.type === 'image' ? (
                <div className="image-viewer">
                  <div className="image-wrapper">
                    <div className="image-navigation">
                      <div className="nav-button-container">
                        <button 
                          className="nav-button prev"
                          onClick={handlePrevPage}
                          disabled={currentPage === 0}
                        >
                          <i className="fas fa-chevron-left"></i>
                        </button>
                        {pageNotification.show && pageNotification.direction === 'prev' && (
                          <div className="page-notification left">
                            이전 페이지에 있는 결과에요!
                          </div>
                        )}
                      </div>
                      <div className="image-container">
                        <img 
                          src={mediaUrl} 
                          alt="Selected" 
                          ref={imageRef}
                          className={detectedObjects.length > 0 ? 'has-results' : ''}
                        />
                        {detectedObjects.length > 0 && (
                          <div className="preview-overlay">
                            <button 
                              className="view-results-button"
                              onClick={() => setIsModalOpen(true)}
                            >
                              <i className="fas fa-search"></i>
                              결과 보기
                            </button>
                          </div>
                        )}
                        {isAnalyzing && (
                          <div className="analyzing-overlay">
                            <div className="analyzing-content">
                              <i className="fas fa-spinner fa-spin"></i>
                              <span>이미지 분석 중...</span>
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="nav-button-container">
                        <button 
                          className="nav-button next"
                          onClick={handleNextPage}
                          disabled={currentPage === mediaItems.length - 1}
                        >
                          <i className="fas fa-chevron-right"></i>
                        </button>
                        {pageNotification.show && pageNotification.direction === 'next' && (
                          <div className="page-notification right">
                            다음 페이지에 있는 결과에요!
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="page-indicator">
                      {currentPage + 1} / {mediaItems.length}
                    </div>
                  </div>
                </div>
              ) : null
            ) : (
              <div className="media-placeholder">
                <i className="fas fa-image"></i>
                <p>이미지나 영상을 업로드하세요</p>
              </div>
            )}
          </div>

          <div className="chat-section">
            <h3>{selectedMedia ? `${selectedMedia.type === 'image' ? '이미지' : '영상'}에 대해 질문해보세요` : '미디어를 업로드하고 질문해보세요'}</h3>
            <div className="chat-input-container">
              <input
                type="text"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                placeholder="질문을 입력하세요..."
                className="chat-input"
              />
              <button 
                onClick={handleChat} 
                className="chat-button"
                disabled={!selectedMedia}
              >
                분석하기
              </button>
            </div>
            {chatResponse && (
              <div className="chat-response">
                <p style={{ whiteSpace: 'pre-line' }}>{chatResponse}</p>
              </div>
            )}
          </div>
        </div>

        <div className="youtube-input-container">
          <div className="youtube-preview">
            {selectedMedia && selectedMedia.type === 'video' && selectedMedia.url.includes('youtube.com') ? (
              <video 
                src={mediaUrl} 
                controls 
                className="selected-video"
                ref={videoRef}
              />
            ) : (
              <div className="youtube-placeholder">
                <i className="fab fa-youtube"></i>
                <p>YouTube 링크를 업로드하세요</p>
              </div>
            )}
          </div>
          <div className="youtube-controls">
            <h3>YouTube URL</h3>
            <div className="youtube-input-container">
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
                {isProcessingYoutube ? '처리 중...' : '분석하기'}
              </button>
            </div>
          </div>
        </div>

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

        {noResults && (
          <div className="no-results-message">
            <i className="fas fa-search"></i>
            <p>검색 결과가 없습니다.</p>
            <p className="sub-text">다른 검색어를 입력하거나 OCR을 새로고침해보세요.</p>
          </div>
        )}

        {summary && (
          <div className="summary-container">
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
                    {item.texts.map((text, i) => {
                      const isMatch = searchTerm && text.text.toLowerCase().includes(searchTerm.toLowerCase());
                      return (
                        <div 
                          key={i} 
                          className="detected-text"
                          style={{ 
                            backgroundColor: isMatch ? 'rgba(0, 123, 255, 0.3)' : 'transparent'
                          }}
                        >
                          {text.text}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {isModalOpen && selectedMedia && selectedMedia.type === 'image' && (
          <div className="image-modal" onClick={() => {
            setIsModalOpen(false);
            setIsModalImageLoaded(false); // 모달 닫을 때 로딩 상태 초기화
            // 모달이 닫힐 때 이미지 크기 복원
            if (imageRef.current) {
              imageRef.current.style.width = '100%';
              imageRef.current.style.height = 'auto';
              imageRef.current.style.maxHeight = '400px';
            }
          }}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
              <button className="modal-close" onClick={() => {
                setIsModalOpen(false);
                setIsModalImageLoaded(false); // 모달 닫을 때 로딩 상태 초기화
                // 모달이 닫힐 때 이미지 크기 복원
                if (imageRef.current) {
                  imageRef.current.style.width = '100%';
                  imageRef.current.style.height = 'auto';
                  imageRef.current.style.maxHeight = '400px';
                }
              }}>×</button>
              <div className="modal-image-container">
                <img 
                  ref={modalImageRef}
                  src={mediaUrl} 
                  alt="Full size" 
                  style={{ width: '100%', height: 'auto' }}
                  onLoad={handleModalImageLoad}
                  loading="eager"
                  decoding="async"
                />
                {!isModalImageLoaded && (
                  <div className="modal-loading-overlay">
                    <div className="modal-loading-content">
                      <div className="loading-spinner"></div>
                      <span>검색 결과 로딩 중...</span>
                    </div>
                  </div>
                )}
                {isModalImageLoaded && detectedObjects
                  .filter(obj => obj.pageIndex === currentPage)
                  .map((obj, index) => {
                    if (!modalImageRef.current) return null;
                    
                    const imgElement = modalImageRef.current;
                    const rect = imgElement.getBoundingClientRect();
                    const bbox = obj.bbox;
                    
                    const isNormalized = bbox.x1 <= 1 && bbox.y1 <= 1 && bbox.x2 <= 1 && bbox.y2 <= 1;
                    const scaleX = rect.width / imgElement.naturalWidth;
                    const scaleY = rect.height / imgElement.naturalHeight;
                    
                    const x1 = isNormalized ? bbox.x1 * imgElement.naturalWidth : bbox.x1;
                    const y1 = isNormalized ? bbox.y1 * imgElement.naturalHeight : bbox.y1;
                    const x2 = isNormalized ? bbox.x2 * imgElement.naturalWidth : bbox.x2;
                    const y2 = isNormalized ? bbox.y2 * imgElement.naturalHeight : bbox.y2;
                    
                    const centerX = (x1 + x2) / 2;
                    const centerY = (y1 + y2) / 2;
                    
                    // 텍스트 크기에 비례하여 동그라미 크기 계산 (2배 크기)
                    const textWidth = (x2 - x1) * scaleX;
                    const textHeight = (y2 - y1) * scaleY;
                    const radius = Math.max(textWidth, textHeight) * 0.6; // 텍스트 크기의 60%로 설정
                    
                    const displayCenterX = centerX * scaleX;
                    const displayCenterY = centerY * scaleY;
                    
                    return (
                      <div
                        key={index}
                        style={{
                          position: 'absolute',
                          left: `${displayCenterX - radius}px`,
                          top: `${displayCenterY - radius}px`,
                          width: `${radius * 2}px`,
                          height: `${radius * 2}px`,
                          border: `2px solid ${obj.color || 'red'}`,
                          borderRadius: "50%",
                          pointerEvents: "none",
                          zIndex: 1
                        }}
                      />
                    );
                  })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App; 