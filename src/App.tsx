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

interface TaskSuggestion {
  task: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
}

interface SmartSearchResult {
  predicted_keywords: string[];
  action_recommendations: {
    message: string;
    action?: string;
  }[];
  document_type?: 'CONTRACT' | 'PAPER' | 'OTHER';
  legal_updates?: {
    title: string;
    description: string;
    source: string;
    date: string;
  }[];
  task_suggestions?: TaskSuggestion[];
}

const App: React.FC = () => {
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([]);
  const [videoItems, setVideoItems] = useState<MediaItem[]>([]);
  const [selectedMedia, setSelectedMedia] = useState<MediaItem | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<MediaItem | null>(null);
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
  const youtubePlayerRef = useRef<HTMLIFrameElement>(null);
  const [smartSearchResult, setSmartSearchResult] = useState<SmartSearchResult | null>(null);
  const [isSmartSearching, setIsSmartSearching] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [modalImageSize, setModalImageSize] = useState({ width: 0, height: 0 });
  const [taskSuggestions, setTaskSuggestions] = useState<TaskSuggestion[]>([]);
  const [isTaskSuggesting, setIsTaskSuggesting] = useState(false);

  const getTaskSuggestions = async (text: string) => {
    console.log('=== 태스크 제안 시작 ===');
    console.log('OCR 텍스트:', text);
    
    setIsTaskSuggesting(true);
    try {
      const requestData = {
        text: text,
        type: 'task_suggestion'
      };
      console.log('요청 데이터:', requestData);

      const response = await fetch('http://localhost:5001/analyze-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      console.log('태스크 제안 응답 상태:', response.status);
      console.log('태스크 제안 응답 헤더:', response.headers);

      const responseText = await response.text();
      console.log('태스크 제안 응답 텍스트:', responseText);

      if (response.ok) {
        try {
          const data = JSON.parse(responseText);
          console.log('태스크 제안 응답 데이터:', data);
          
          if (data.suggestions) {
            console.log('태스크 제안 목록:', data.suggestions);
            setTaskSuggestions(data.suggestions);
          } else if (data.task_suggestions) {
            console.log('태스크 제안 목록 (task_suggestions):', data.task_suggestions);
            setTaskSuggestions(data.task_suggestions);
          } else {
            console.log('태스크 제안 데이터가 없습니다.');
            console.log('전체 응답 데이터:', data);
            setTaskSuggestions([]);
          }
        } catch (parseError) {
          console.error('JSON 파싱 오류:', parseError);
          console.log('파싱 실패한 응답 텍스트:', responseText);
          setTaskSuggestions([]);
        }
      } else {
        console.error('태스크 제안 실패:', response.status);
        console.error('에러 응답:', responseText);
        setTaskSuggestions([]);
      }
    } catch (error) {
      console.error('태스크 제안 중 오류:', error);
      if (error instanceof Error) {
        console.error('에러 메시지:', error.message);
        console.error('에러 스택:', error.stack);
      }
      setTaskSuggestions([]);
    } finally {
      setIsTaskSuggesting(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    console.log('=== 파일 업로드 시작 ===');
    console.log('파일 개수:', files.length);

    setIsProcessing(true);
    setIsAnalyzing(true);
    setError(null);
    setTaskSuggestions([]);

    try {
      const file = files[0];
      const isVideo = file.type.startsWith('video/');
      const formData = new FormData();
      
      if (isVideo) {
        formData.append('video', file);
        formData.append('query', '');
        formData.append('mode', 'normal');
        
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

        if (!sessionId) {
          setSessionId(data.session_id);
        }

        if (data.text) {
          console.log('=== OCR 텍스트 추출 완료 ===');
          console.log('OCR 텍스트:', data.text);
          setOcrText(data.text);
          
          console.log('=== 태스크 제안 시작 ===');
          await getTaskSuggestions(data.text);
        }

        const videoUrl = URL.createObjectURL(file);

        const newVideoItem: MediaItem = {
          id: Date.now().toString(),
          type: 'video',
          url: videoUrl,
          file,
          sessionId: sessionId || data.session_id,
        };

        setVideoItems(prev => [...prev, newVideoItem]);
        setSelectedVideo(newVideoItem);
        setMediaType('video');
        setMediaUrl(videoUrl);
        
        if (data.file.timeline) {
          setTimeline(data.file.timeline);
        }
      } else {
        for (let i = 0; i < files.length; i++) {
          formData.append('images[]', files[i]);
        }
        formData.append('mode', 'normal');

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

        if (!sessionId) {
          setSessionId(data.session_id);
        }

        if (data.text) {
          console.log('=== OCR 텍스트 추출 완료 ===');
          console.log('OCR 텍스트:', data.text);
          setOcrText(data.text);
          
          console.log('=== 태스크 제안 시작 ===');
          await getTaskSuggestions(data.text);
        }

        if (data.image_type) {
          console.log('감지된 이미지 타입:', data.image_type);
          setSelectedImageType(data.image_type as ImageType);
        }

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

        setMediaItems(prev => {
          const updatedItems = [...prev, ...newMediaItems];
          setCurrentPage(updatedItems.length - 1);
          return updatedItems;
        });

        const lastMediaItem = newMediaItems[newMediaItems.length - 1];
        setSelectedMedia(lastMediaItem);
        setMediaType('image');
        setMediaUrl(lastMediaItem.url);
      }

      setDetectedObjects([]);

    } catch (error) {
      console.error('=== 오류 발생 ===');
      console.error('오류 상세:', error);
      if (error instanceof Error) {
        console.error('에러 메시지:', error.message);
        console.error('에러 스택:', error.stack);
      }
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
    
    // 검색어가 비어있을 때 검색 결과만 초기화
    if (newSearchTerm === '') {
      setDetectedObjects([]);
      setNoResults(false);
    }
  };

  const seekToTimestamp = (timestamp: number) => {
    if (youtubePlayerRef.current) {
      youtubePlayerRef.current.contentWindow?.postMessage(
        JSON.stringify({
          event: 'command',
          func: 'seekTo',
          args: [timestamp, true]
        }),
        '*'
      );
    }
  };

  const handleMediaItemClick = (media: MediaItem) => {
    if (media.type === 'image') {
      setSelectedMedia(media);
      setMediaType('image');
      setMediaUrl(media.url);
    } else {
      setSelectedVideo(media);
      setMediaType('video');
      setMediaUrl(media.url);
    }
    setDetectedObjects([]);
    setTimeline([]);
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
      
      const imageFiles = mediaItems
        .filter(item => item.type === 'image')
        .map(item => item.file);
      
      console.log('전송할 이미지 개수:', imageFiles.length);
      
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
          mode: 'smart'
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

      // YouTube 비디오 아이템 생성
      const newVideoItem: MediaItem = {
        id: Date.now().toString(),
        type: 'video',
        url: videoId,
        file: new File([], 'youtube-video.mp4'),
        sessionId: data.session_id
      };

      // 비디오 그리드에 추가
      setVideoItems(prev => [...prev, newVideoItem]);
      setSelectedVideo(newVideoItem);
      setMediaType('video');
      setMediaUrl(videoId);
      setYoutubeUrl('');

      // OCR 텍스트가 있으면 태스크 제안 실행
      if (data.ocr_text) {
        console.log('=== OCR 텍스트 추출 완료 ===');
        console.log('OCR 텍스트:', data.ocr_text);
        setOcrText(data.ocr_text);
        
        console.log('=== 태스크 제안 시작 ===');
        await getTaskSuggestions(data.ocr_text);
      }
      
      // 타임라인 설정
      if (data.timeline) {
        setTimeline(data.timeline);
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
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

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

  const handleSmartSearch = async () => {
    if (!searchTerm.trim()) {
        alert('검색어를 입력해주세요.');
        return;
    }

    console.log('=== 스마트 검색 시작 ===');
    console.log('검색어:', searchTerm);
    console.log('세션 ID:', sessionId);

    setIsSmartSearching(true);
    setSmartSearchResult(null);

    try {
        const formData = new FormData();
        formData.append('query', searchTerm);
        formData.append('mode', 'smart');
        if (selectedMedia) {
            formData.append('images[]', selectedMedia.file);
        }
        if (sessionId) {
            formData.append('session_id', sessionId);
        }

        const response = await fetch('http://localhost:5001/analyze-image', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('스마트 검색에 실패했습니다.');
        }

        const data = await response.json();
        console.log('=== 스마트 검색 응답 ===');
        console.log('전체 응답:', data);

        if (data.smart_search) {
            console.log('smart_search 데이터:', data.smart_search);
            setSmartSearchResult({
                predicted_keywords: data.smart_search.predicted_keywords || [],
                action_recommendations: data.smart_search.action_recommendations || [],
                document_type: data.smart_search.document_type,
                legal_updates: data.smart_search.legal_updates || [],
                task_suggestions: data.smart_search.task_suggestions || []
            });
        }
    } catch (error) {
        console.error('스마트 검색 중 오류:', error);
        alert('스마트 검색 중 오류가 발생했습니다.');
    } finally {
        setIsSmartSearching(false);
    }
  };

  const handleImageResize = () => {
    if (imageRef.current) {
      const rect = imageRef.current.getBoundingClientRect();
      setImageSize({
        width: rect.width,
        height: rect.height
      });
    }
  };

  useEffect(() => {
    const resizeObserver = new ResizeObserver(handleImageResize);
    if (imageRef.current) {
      resizeObserver.observe(imageRef.current);
    }
    return () => {
      resizeObserver.disconnect();
    };
  }, [imageRef.current]);

  useEffect(() => {
    console.log('smartSearchResult 변경됨:', smartSearchResult);
  }, [smartSearchResult]);

  const handleActionClick = async (action: { message: string; action?: string }) => {
    console.log('=== 액션 클릭 ===');
    console.log('선택된 액션:', action);
    
    if (action.action) {
      const searchQuery = action.action.replace(/^Search for ['"]?/, '').replace(/['"]?$/, '');
      const googleSearchUrl = `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`;
      window.open(googleSearchUrl, '_blank');
      return;
    }
    
    alert(action.message);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileUpload({ target: { files } } as React.ChangeEvent<HTMLInputElement>);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleModalImageResize = () => {
    if (modalImageRef.current) {
      const rect = modalImageRef.current.getBoundingClientRect();
      setModalImageSize({
        width: rect.width,
        height: rect.height
      });
    }
  };

  useEffect(() => {
    const resizeObserver = new ResizeObserver(handleModalImageResize);
    if (modalImageRef.current) {
      resizeObserver.observe(modalImageRef.current);
    }
    return () => {
      resizeObserver.disconnect();
    };
  }, [modalImageRef.current]);

  if (isLoading) {
    return <NetflixLoader />;
  }

  return (
    <div 
      className="App"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      style={{
        position: 'relative',
        minHeight: '100vh'
      }}
    >
      <div 
        className="drag-overlay"
        style={{
          display: 'none',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 123, 255, 0.1)',
          border: '3px dashed #007bff',
          zIndex: 1000,
          pointerEvents: 'none'
        }}
      />

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

      <div className={`search-section ${isSearchExpanded ? '' : 'collapsed'}`} style={{ height: 'auto', minHeight: '200px' }}>
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
          <i className={`fas fa-chevron-${isSearchExpanded ? 'left' : 'right'}`} style={{ fontSize: '30px' }}></i>
        </button>
      </div>

      <div className="right-section">
        <div className="media-container">
          <div className={`selected-media ${selectedMedia && selectedMedia.type === 'image' ? 'has-media' : ''}`} style={{ 
            display: 'flex', 
            flexDirection: 'column',
            minHeight: '0',
            height: '100%'
          }}>
            {selectedMedia && selectedMedia.type === 'image' ? (
              <div className="image-viewer" style={{ 
                flex: '1 1 auto',
                minHeight: '0',
                overflow: 'auto'
              }}>
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
                  <div className="image-container" style={{ position: 'relative' }}>
                    <img 
                      src={mediaUrl} 
                      alt="Selected" 
                      ref={imageRef}
                      className={detectedObjects.length > 0 ? 'has-results' : ''}
                      onLoad={handleImageResize}
                      style={{ maxHeight: 'calc(100vh - 300px)', objectFit: 'contain' }}
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
                    {detectedObjects
                      .filter(obj => obj.pageIndex === currentPage)
                      .map((obj, index) => {
                        if (!imageRef.current) return null;
                        const imgElement = imageRef.current;
                        const rect = imgElement.getBoundingClientRect();
                        const bbox = obj.bbox;
                        const isNormalized = bbox.x1 <= 1 && bbox.y1 <= 1 && bbox.x2 <= 1 && bbox.y2 <= 1;
                        const scaleX = rect.width / imgElement.naturalWidth;
                        const scaleY = rect.height / imgElement.naturalHeight;
                        const x1 = isNormalized ? bbox.x1 * imgElement.naturalWidth : bbox.x1;
                        const y1 = isNormalized ? bbox.y1 * imgElement.naturalHeight : bbox.y1;
                        const x2 = isNormalized ? bbox.x2 * imgElement.naturalWidth : bbox.x2;
                        const y2 = isNormalized ? bbox.y2 * imgElement.naturalHeight : bbox.y2;
                        const lowerText = obj.text.toLowerCase();
                        const lowerSearch = searchTerm.toLowerCase();
                        const startIdx = lowerText.indexOf(lowerSearch);
                        if (startIdx === -1) return null;
                        const totalLen = obj.text.length;
                        const searchLen = searchTerm.length;
                        const charWidth = (x2 - x1) / totalLen;
                        const wordX1 = x1 + charWidth * startIdx;
                        const wordX2 = wordX1 + charWidth * searchLen;
                        const centerX = (wordX1 + wordX2) / 2;
                        const centerY = (y1 + y2) / 2;
                        const textWidth = (wordX2 - wordX1) * scaleX;
                        const textHeight = (y2 - y1) * scaleY;
                        const radius = Math.max(textWidth, textHeight) * 0.5;
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
                              border: `2px solid red`,
                              borderRadius: '50%',
                              pointerEvents: 'none',
                              zIndex: 1,
                            }}
                          />
                        );
                      })}
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
            ) : (
              <div className="media-placeholder">
                <i className="fas fa-image"></i>
                <p>이미지를 업로드하세요</p>
              </div>
            )}

            {mediaItems.length > 0 && (
              <div className="media-grid" style={{ 
                marginTop: '20px',
                display: 'flex',
                justifyContent: 'center',
                gap: '10px',
                flexWrap: 'wrap',
                width: '100%',
                flex: '0 0 auto'
              }}>
                {mediaItems.map(media => (
                  <div
                    key={media.id}
                    className={`media-item ${selectedMedia?.id === media.id ? 'selected' : ''}`}
                    onClick={() => handleMediaItemClick(media)}
                    style={{
                      width: '100px',
                      height: '100px',
                      cursor: 'pointer',
                      border: selectedMedia?.id === media.id ? '2px solid #007bff' : '1px solid #ddd',
                      borderRadius: '4px',
                      overflow: 'hidden'
                    }}
                  >
                    <img src={media.url} alt="Uploaded" className="media-preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  </div>
                ))}
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

          <div className="task-suggestions-section" style={{ 
            padding: '20px', 
            backgroundColor: '#f8f9fa', 
            borderRadius: '8px', 
            marginTop: '20px' 
          }}>
            <h3 style={{ marginBottom: '15px', color: '#333' }}>
              {isTaskSuggesting ? '태스크 제안 생성 중...' : '태스크 제안'}
            </h3>
            
            {isTaskSuggesting ? (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <i className="fas fa-spinner fa-spin" style={{ fontSize: '24px', color: '#007bff' }}></i>
                <p style={{ marginTop: '10px', color: '#666' }}>태스크를 생성하고 있습니다...</p>
              </div>
            ) : taskSuggestions.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {taskSuggestions.map((task, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '15px',
                      backgroundColor: '#fff',
                      borderRadius: '8px',
                      border: '1px solid #e0e0e0',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
                    }}
                  >
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      marginBottom: '8px'
                    }}>
                      <h4 style={{ margin: 0, color: '#333' }}>{task.task}</h4>
                      <span style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '0.8em',
                        backgroundColor: 
                          task.priority === 'high' ? '#ffebee' :
                          task.priority === 'medium' ? '#fff3e0' :
                          '#e8f5e9',
                        color: 
                          task.priority === 'high' ? '#c62828' :
                          task.priority === 'medium' ? '#ef6c00' :
                          '#2e7d32'
                      }}>
                        {task.priority === 'high' ? '높음' :
                         task.priority === 'medium' ? '중간' :
                         '낮음'}
                      </span>
                    </div>
                    <p style={{ margin: 0, color: '#666' }}>{task.description}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: '#666' }}>
                이미지를 업로드하면 태스크 제안이 여기에 표시됩니다
              </div>
            )}
          </div>
        </div>

        <div className="youtube-input-container">
          <div className="youtube-preview">
            {selectedVideo && selectedVideo.type === 'video' && selectedVideo.url.startsWith('blob:') ? (
              <video
                src={selectedVideo.url}
                controls
                style={{ width: '100%', height: '100%' }}
              />
            ) : selectedVideo && selectedVideo.type === 'video' ? (
              <iframe
                ref={youtubePlayerRef}
                width="100%"
                height="100%"
                src={`https://www.youtube.com/embed/${selectedVideo.url}?enablejsapi=1`}
                title="YouTube video player"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            ) : (
              <div className="youtube-placeholder">
                <i className="fab fa-youtube"></i>
                <p>영상을 업로드하세요</p>
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

        {/* 비디오 그리드 */}
        {videoItems.length > 0 && (
          <div className="video-grid" style={{
            marginTop: '20px',
            display: 'flex',
            justifyContent: 'center',
            gap: '10px',
            flexWrap: 'wrap',
            width: '100%'
          }}>
            {videoItems.map(media => (
              <div
                key={media.id}
                className={`media-item ${selectedVideo?.id === media.id ? 'selected' : ''}`}
                onClick={() => handleMediaItemClick(media)}
                style={{
                  width: '100px',
                  height: '100px',
                  cursor: 'pointer',
                  border: selectedVideo?.id === media.id ? '2px solid #007bff' : '1px solid #ddd',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}
              >
                {media.url.startsWith('blob:') ? (
                  <video src={media.url} className="media-preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <img 
                    src={`https://img.youtube.com/vi/${media.url}/mqdefault.jpg`} 
                    alt="YouTube thumbnail" 
                    className="media-preview" 
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
                  />
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
            <h3 style={{ textAlign: 'left', marginTop: '10px', marginBottom: '30px' ,marginLeft: '10px' }}>타임라인</h3>
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
      </div>

      {isModalOpen && (
        <div 
          className="modal-overlay"
          onClick={() => setIsModalOpen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000
          }}
        >
          <div 
            className="modal-content"
            onClick={e => e.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '90vw',
              maxHeight: '90vh'
            }}
          >
            <button
              onClick={() => setIsModalOpen(false)}
              style={{
                position: 'absolute',
                top: '-40px',
                right: '0',
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                color: 'white'
              }}
            >
              ×
            </button>
            <img 
              ref={modalImageRef}
              src={mediaUrl} 
              alt="Enlarged view" 
              style={{
                maxWidth: '100%',
                maxHeight: '90vh',
                objectFit: 'contain'
              }}
              onLoad={handleModalImageResize}
            />
            {detectedObjects
              .filter(obj => obj.pageIndex === currentPage)
              .map((obj, index) => {
                if (!modalImageRef.current) return null;
                const imgElement = modalImageRef.current;
                const bbox = obj.bbox;
                const isNormalized = bbox.x1 <= 1 && bbox.y1 <= 1 && bbox.x2 <= 1 && bbox.y2 <= 1;
                const scaleX = modalImageSize.width / imgElement.naturalWidth;
                const scaleY = modalImageSize.height / imgElement.naturalHeight;
                const x1 = isNormalized ? bbox.x1 * imgElement.naturalWidth : bbox.x1;
                const y1 = isNormalized ? bbox.y1 * imgElement.naturalHeight : bbox.y1;
                const x2 = isNormalized ? bbox.x2 * imgElement.naturalWidth : bbox.x2;
                const y2 = isNormalized ? bbox.y2 * imgElement.naturalHeight : bbox.y2;
                const lowerText = obj.text.toLowerCase();
                const lowerSearch = searchTerm.toLowerCase();
                const startIdx = lowerText.indexOf(lowerSearch);
                if (startIdx === -1) return null;
                const totalLen = obj.text.length;
                const searchLen = searchTerm.length;
                const charWidth = (x2 - x1) / totalLen;
                const wordX1 = x1 + charWidth * startIdx;
                const wordX2 = wordX1 + charWidth * searchLen;
                const centerX = (wordX1 + wordX2) / 2;
                const centerY = (y1 + y2) / 2;
                const textWidth = (wordX2 - wordX1) * scaleX;
                const textHeight = (y2 - y1) * scaleY;
                const radius = Math.max(textWidth, textHeight) * 0.5;
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
                      border: `2px solid red`,
                      borderRadius: '50%',
                      pointerEvents: 'none',
                      zIndex: 1,
                    }}
                  />
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
