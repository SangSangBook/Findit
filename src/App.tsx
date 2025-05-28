import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import NetflixLoader from './components/NetflixLoader';
import MediaUploader from './components/MediaUploader';
import { ImageType, IMAGE_TYPE_ICONS, IMAGE_TYPE_LABELS, IMAGE_TYPE_COLORS } from './types';
import ImageTypeSelector from './components/ImageTypeSelector';
import { koreanToEnglish, englishToKorean, findMatches } from './utils/languageMapping';

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
  isObject?: boolean;
  confidence: number;
}

interface MediaItem {
  id: string;
  type: 'image' | 'video';
  url: string;
  file: File;
  sessionId?: string;
  imageType?: ImageType;
  ocrText?: string;
  detectedObjects?: DetectedObject[];
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
  const [ocrCache, setOcrCache] = useState<Map<string, {text: string, objects: DetectedObject[]}>>(new Map());
  const [taskSuggestionPosition, setTaskSuggestionPosition] = useState({ x: 117, y: 494 });
  const [isDraggingTaskSuggestion, setIsDraggingTaskSuggestion] = useState(false);
  const [taskSuggestionDragOffset, setTaskSuggestionDragOffset] = useState({ x: 0, y: 0 });
  const taskSuggestionRef = useRef<HTMLDivElement>(null);
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<DetectedObject[]>([]);

  const getTaskSuggestions = async (text: string) => {
    console.log('=== 태스크 제안 시작 ===');
    console.log('OCR 텍스트:', text);
    
    setIsTaskSuggesting(true);
    try {
      // 모든 이미지의 OCR 텍스트를 결합
      const allOcrText = mediaItems
        .map(item => item.ocrText)
        .filter(text => text)
        .join('\n\n');

      // 텍스트가 너무 길 경우 앞부분만 사용
      const maxLength = 8000;
      const truncatedText = (allOcrText || text).slice(0, maxLength);

      const requestData = {
        text: truncatedText,
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
        formData.append('fast_ocr', 'true'); // 빠른 OCR 처리 옵션 추가
        
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
        console.log('타임라인 데이터:', data.file?.timeline);

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
        
        if (data.file?.timeline) {
          console.log('타임라인 설정:', data.file.timeline);
          setTimeline(data.file.timeline);
        } else {
          console.log('타임라인 데이터가 없습니다');
        }
      } else {
        for (let i = 0; i < files.length; i++) {
          formData.append('images[]', files[i]);
        }
        formData.append('mode', 'normal');
        
        if (sessionId) {
          formData.append('session_id', sessionId);
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

        if (data.session_id) {
          setSessionId(data.session_id);
        }

        if (data.image_type) {
          console.log('감지된 이미지 타입:', data.image_type);
          setSelectedImageType(data.image_type as ImageType);
        }

        const newMediaItems: MediaItem[] = await Promise.all(Array.from(files).map(async (file, index) => {
          const url = URL.createObjectURL(file);
          const mediaItem: MediaItem = {
            id: `${Date.now()}_${index}`,
            type: 'image',
            url,
            file,
            sessionId: data.session_id || sessionId,
            imageType: data.image_type as ImageType
          };

          const imageFormData = new FormData();
          imageFormData.append('images[]', file);
          imageFormData.append('session_id', mediaItem.sessionId!);
          imageFormData.append('mode', 'normal');
          imageFormData.append('detect_objects', 'true');
          imageFormData.append('perform_ocr', 'true');
          imageFormData.append('detect_text', 'true');

          const imageResponse = await fetch('http://localhost:5001/analyze-image', {
            method: 'POST',
            body: imageFormData,
          });

          if (imageResponse.ok) {
            const imageData = await imageResponse.json();
            mediaItem.ocrText = imageData.ocr_text;
            mediaItem.detectedObjects = imageData.detected_objects;
            
            if (imageData.ocr_text && imageData.detected_objects) {
              setOcrCache(prev => new Map(prev).set(mediaItem.id, {
                text: imageData.ocr_text,
                objects: imageData.detected_objects
              }));
            }
          }

          return mediaItem;
        }));

        newMediaItems.reverse();

        setMediaItems(prev => {
          const updatedItems = [...newMediaItems, ...prev];
          setCurrentPage(0);
          return updatedItems;
        });

        const firstMediaItem = newMediaItems[0];
        setSelectedMedia(firstMediaItem);
        setMediaType('image');
        setMediaUrl(firstMediaItem.url);
        
        const allOcrText = newMediaItems
          .map(item => item.ocrText)
          .filter(text => text)
          .join('\n\n');
        
        if (allOcrText) {
          await getTaskSuggestions(allOcrText);
        }
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
    if (!searchTerm || !sessionId) {
      setDetectedObjects([]);
      setNoResults(false);
      setSearchResultPages([]);
      setPageNotification({ show: false, direction: null });
      return;
    }

    console.log('=== 검색 시작 ===');
    console.log('검색어:', searchTerm);
    console.log('검색 모드:', mode);
    console.log('세션 ID:', sessionId);

    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);
      formData.append('query', searchTerm);
      formData.append('mode', mode);
      formData.append('detect_objects', 'true');
      formData.append('perform_ocr', 'true');
      formData.append('detect_text', 'true');

      // 현재 선택된 이미지 파일 전송
      if (selectedMedia) {
        formData.append('images[]', selectedMedia.file);
      }

      const response = await fetch('http://localhost:5001/analyze-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '검색 중 오류가 발생했습니다');
      }

      const data = await response.json();
      console.log('검색 결과:', data);

      if (data.matches && data.matches.length > 0) {
        console.log('검색 결과 발견:', data.matches);
        setDetectedObjects(data.matches);
        setNoResults(false);
        setShowSearchModal(true);
      } else {
        console.log('검색 결과 없음');
        setDetectedObjects([]);
        setNoResults(true);
        setShowSearchModal(false);
      }
    } catch (error) {
      console.error('검색 오류:', error);
      setError(error instanceof Error ? error.message : '검색 중 오류가 발생했습니다');
    }
  };

  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSearchTerm = e.target.value;
    setSearchTerm(newSearchTerm);
    
    // 타임라인이 있을 때만 하이라이트 처리
    if (timeline.length > 0) {
      const updatedTimeline = timeline.map(item => ({
        ...item,
        texts: item.texts.map(text => {
          const textContent = text.text;
          if (!newSearchTerm) {
            return { ...text, color: '#000000' };
          }
          
          // 검색어를 포함하는 경우, 해당 부분만 하이라이트
          const parts = textContent.split(new RegExp(`(${newSearchTerm})`, 'gi'));
          return {
            ...text,
            text: parts.map(part => 
              part.toLowerCase() === newSearchTerm.toLowerCase() 
                ? `<span style="color: #007bff">${part}</span>` 
                : part
            ).join('')
          };
        })
      }));
      setTimeline(updatedTimeline);
    }

    // 검색어가 비어있을 때 검색 결과 초기화
    if (newSearchTerm === '') {
      setDetectedObjects([]);
      setNoResults(false);
    }
  };

  const seekToTimestamp = (timestamp: number) => {
    console.log('Seeking to timestamp:', timestamp);
    
    if (youtubePlayerRef.current) {
      console.log('Using YouTube player');
      youtubePlayerRef.current.contentWindow?.postMessage(
        JSON.stringify({
          event: 'command',
          func: 'seekTo',
          args: [timestamp, true]
        }),
        '*'
      );
    } else if (videoRef.current) {
      console.log('Using video element');
      videoRef.current.currentTime = timestamp;
      videoRef.current.play();
    }
  };

  const handleMediaItemClick = async (media: MediaItem) => {
    if (!media) return;
    
    if (media.type === 'image') {
      setSelectedMedia(media);
      setMediaType('image');
      setMediaUrl(media.url);
      
      // 모든 이미지의 데이터를 가져오기
      const allDetectedObjects: DetectedObject[] = [];
      const allOcrText: string[] = [];
      
      for (const currentMedia of mediaItems) {
        if (currentMedia.type === 'image') {
          const formData = new FormData();
          if (sessionId) {
            formData.append('session_id', sessionId);
          }
          
          try {
            const response = await fetch('http://localhost:5001/analyze-image', {
              method: 'POST',
              body: formData
            });
            
            if (response.ok) {
              const data = await response.json();
              if (data.detected_objects) {
                allDetectedObjects.push(...data.detected_objects);
              }
              if (data.ocr_text) {
                allOcrText.push(data.ocr_text);
              }
            }
          } catch (error) {
            console.error('Error analyzing image:', error);
          }
        }
      }
      
      setDetectedObjects(allDetectedObjects);
      
      // 모든 OCR 텍스트를 결합
      const combinedOcrText = allOcrText.join('\n\n');
      
      // 태스크 제안 가져오기
      if (combinedOcrText) {
        const formData = new FormData();
        formData.append('text', combinedOcrText);
        formData.append('type', 'task_suggestion');
        if (sessionId) {
          formData.append('session_id', sessionId);
        }
        
        try {
          const response = await fetch('http://localhost:5001/analyze-image', {
            method: 'POST',
            body: formData
          });
          
          if (response.ok) {
            const data = await response.json();
            if (data.suggestions) {
              setTaskSuggestions(data.suggestions);
            }
          }
        } catch (error) {
          console.error('Error getting task suggestions:', error);
        }
      }
    } else {
      setSelectedVideo(media);
      setMediaType('video');
      setMediaUrl(media.url);
      setTimeline([]);
    }
  };

  const handleSummarize = async () => {
    if (!selectedMedia) {
      setError('이미지가 선택되지 않았습니다');
      return;
    }

    setIsSummarizing(true);
    setError(null);
    setSummary(null);

    try {
      const formData = new FormData();
      formData.append('session_id', selectedMedia.sessionId!);
      formData.append('text', selectedMedia.ocrText || '');
      formData.append('type', 'summary');

      const response = await fetch('http://localhost:5001/analyze-image', {
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
    if (!chatMessage.trim() || !selectedMedia) return;
    
    try {
      console.log('=== 채팅 요청 시작 ===');
      console.log('선택된 미디어:', selectedMedia);
      console.log('질문:', chatMessage);
      
      const formData = new FormData();
      formData.append('session_id', selectedMedia.sessionId!);
      formData.append('message', chatMessage);
      
      // 모든 이미지의 OCR 텍스트와 객체 인식 결과를 결합
      const allOcrText = mediaItems
        .map(item => item.ocrText)
        .filter(text => text)
        .join('\n\n');

      const allDetectedObjects = mediaItems
        .flatMap(item => item.detectedObjects || [])
        .map(obj => ({
          ...obj,
          pageIndex: mediaItems.findIndex(item => 
            item.detectedObjects?.some(dObj => dObj === obj)
          )
        }));

      if (allOcrText) {
        console.log('모든 이미지의 OCR 텍스트 사용:', allOcrText);
        formData.append('ocr_text', allOcrText);
        formData.append('use_ocr', 'true');
      }
      
      if (allDetectedObjects.length > 0) {
        console.log('모든 이미지의 객체 인식 결과 사용:', allDetectedObjects);
        formData.append('detected_objects', JSON.stringify(allDetectedObjects));
      }
      
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
      
      if (data.error) {
        throw new Error('분석 중 오류가 발생했습니다.');
      }
      
      // 시스템 메시지 제거하고 핵심 내용만 표시
      const cleanResponse = data.summary
        .replace(/^.*?프로젝트 참여 팀원은 다음과 같습니다:\s*/g, '')
        .replace(/^.*?OCR 텍스트를 통해.*?정보를 확인할 수 있습니다\.\s*/g, '')
        .replace(/^.*?죄송합니다.*?감지되지 않았습니다\.\s*/g, '')
        .replace(/^.*?이미지에서 객체가 감지되지 않았습니다\.\s*/g, '')
        .replace(/^.*?CSS 코딩 연습에 대한 정보가.*?발견되지 않았습니다\.\s*/g, '')
        .replace(/^.*?OCR 텍스트에는.*?포함되어 있지만.*?분석은 이루어지지 않았습니다\.\s*/g, '')
        .replace(/^.*?따라서 해당 질문에 대한 답변을 제공할 수 없습니다\.\s*/g, '')
        .replace(/^.*?부득이하게 OCR 텍스트에 포함된 내용을.*?필요합니다\.\s*/g, '')
        .trim();
      setChatResponse(cleanResponse);
    } catch (error) {
      console.error('채팅 처리 중 오류:', error);
      alert('분석 중 오류가 발생했습니다.');
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

      const newVideoItem: MediaItem = {
        id: Date.now().toString(),
        type: 'video',
        url: videoId,
        file: new File([], 'youtube-video.mp4'),
        sessionId: data.session_id
      };

      setVideoItems(prev => [...prev, newVideoItem]);
      setSelectedVideo(newVideoItem);
      setMediaType('video');
      setMediaUrl(videoId);
      setYoutubeUrl('');

      if (data.ocr_text) {
        console.log('=== OCR 텍스트 추출 완료 ===');
        console.log('OCR 텍스트:', data.ocr_text);
        setOcrText(data.ocr_text);
        
        console.log('=== 태스크 제안 시작 ===');
        await getTaskSuggestions(data.ocr_text);
      }
      
      if (data.timeline) {
        setTimeline(data.timeline);
      }
      
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
    
    // 검색 결과가 있는 페이지 목록에서 현재 페이지의 위치 확인
    const currentIndex = searchResultPages.indexOf(newPage);
    
    // 이전/다음 페이지에 결과가 있는지 확인
    const hasPrevResults = searchResultPages.some(page => page < newPage);
    const hasNextResults = searchResultPages.some(page => page > newPage);
    
    // 알림 상태 업데이트
    if (hasPrevResults) {
      setPageNotification({ show: true, direction: 'prev' });
    } else if (hasNextResults) {
      setPageNotification({ show: true, direction: 'next' });
    } else {
      setPageNotification({ show: false, direction: null });
    }
    
    // 3초 후 알림 자동 숨김
    setTimeout(() => {
      setPageNotification({ show: false, direction: null });
    }, 3000);
  };

  const handlePrevPage = () => {
    if (currentPage > 0) {
      const newPage = currentPage - 1;
      handlePageChange(newPage);
      const prevMedia = mediaItems[newPage];
      if (prevMedia) {
        setSelectedMedia(prevMedia);
        setMediaType(prevMedia.type);
        setMediaUrl(prevMedia.url);
        if (prevMedia.imageType) {
          setSelectedImageType(prevMedia.imageType);
        }
      }
    }
  };

  const handleNextPage = () => {
    if (currentPage < mediaItems.length - 1) {
      const newPage = currentPage + 1;
      handlePageChange(newPage);
      const nextMedia = mediaItems[newPage];
      if (nextMedia) {
        setSelectedMedia(nextMedia);
        setMediaType(nextMedia.type);
        setMediaUrl(nextMedia.url);
        if (nextMedia.imageType) {
          setSelectedImageType(nextMedia.imageType);
        }
      }
    }
  };

  const handleModalImageResize = () => {
    if (modalImageRef.current) {
      const imgElement = modalImageRef.current;
      const rect = imgElement.getBoundingClientRect();
      setModalImageSize({
        width: rect.width,
        height: rect.height
      });
      setIsModalImageLoaded(true);
    }
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

  const handleTaskClick = async (taskTitleOrEvent: string | React.MouseEvent<HTMLButtonElement>) => {
    // MouseEvent인 경우 chatMessage를 사용
    const taskTitle = typeof taskTitleOrEvent === 'string' ? taskTitleOrEvent : chatMessage;
    
    // chatMessage 상태를 task 제목으로 설정
    setChatMessage(taskTitle);
    
    // 비디오 타입인 경우 selectedVideo를 사용
    if (selectedVideo) {
      const formData = new FormData();
      formData.append('session_id', selectedVideo.sessionId!);
      formData.append('message', taskTitle);
      
      // 비디오의 타임라인 데이터 사용
      if (timeline.length > 0) {
        const videoText = timeline
          .map(item => item.texts.map(t => t.text).join(' '))
          .join('\n\n');
        formData.append('ocr_text', videoText);
        formData.append('use_ocr', 'true');
      }
      
      const response = await fetch('http://localhost:5001/summarize', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '채팅 처리 중 오류가 발생했습니다.');
      }
      
      const data = await response.json();
      if (data.error) {
        throw new Error('분석 중 오류가 발생했습니다.');
      }
      
      // 시스템 메시지 제거하고 핵심 내용만 표시
      const cleanResponse = data.summary
        .replace(/^.*?프로젝트 참여 팀원은 다음과 같습니다:\s*/g, '')
        .replace(/^.*?OCR 텍스트를 통해.*?정보를 확인할 수 있습니다\.\s*/g, '')
        .replace(/^.*?죄송합니다.*?감지되지 않았습니다\.\s*/g, '')
        .replace(/^.*?이미지에서 객체가 감지되지 않았습니다\.\s*/g, '')
        .replace(/^.*?CSS 코딩 연습에 대한 정보가.*?발견되지 않았습니다\.\s*/g, '')
        .replace(/^.*?OCR 텍스트에는.*?포함되어 있지만.*?분석은 이루어지지 않았습니다\.\s*/g, '')
        .replace(/^.*?따라서 해당 질문에 대한 답변을 제공할 수 없습니다\.\s*/g, '')
        .replace(/^.*?부득이하게 OCR 텍스트에 포함된 내용을.*?필요합니다\.\s*/g, '')
        .trim();
      setChatResponse(cleanResponse);
    } else if (selectedMedia) {
      // 이미지 타입인 경우 기존 handleChat 함수 호출
      await handleChat();
    }
  };

  const handleTaskSuggestionMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDraggingTaskSuggestion(true);
    const rect = e.currentTarget.getBoundingClientRect();
    setTaskSuggestionDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleTaskSuggestionMouseMove = (e: MouseEvent) => {
    if (isDraggingTaskSuggestion) {
      e.preventDefault();
      const x = e.clientX - taskSuggestionDragOffset.x;
      const y = e.clientY - taskSuggestionDragOffset.y;
      setTaskSuggestionPosition({ x, y });
    }
  };

  const handleTaskSuggestionMouseUp = () => {
    setIsDraggingTaskSuggestion(false);
  };

  useEffect(() => {
    if (isDraggingTaskSuggestion) {
      window.addEventListener('mousemove', handleTaskSuggestionMouseMove);
      window.addEventListener('mouseup', handleTaskSuggestionMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleTaskSuggestionMouseMove);
      window.removeEventListener('mouseup', handleTaskSuggestionMouseUp);
    };
  }, [isDraggingTaskSuggestion, taskSuggestionDragOffset]);

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
        <div className="app-subtitle" style={{ color: '#000000' }}>
          미디어에서{'\n'}
          정보를{'\n'}
          찾아보세요
          <span style={{
            display: 'inline-block',
            width: '10px',
            height: '10px',
            backgroundColor: '#46B876',
            borderRadius: '70%',
            marginLeft: '10px',
          }}></span>
        </div>
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
          <div className={`selected-media ${selectedMedia?.type === 'image' ? 'has-media' : ''}`} style={{ 
            display: 'flex', 
            flexDirection: 'column',
            minHeight: '0',
            height: '100%'
          }}>
            {selectedMedia?.type === 'image' ? (
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
                      <div className="page-notification" style={{
                        position: 'fixed',
                        left: '50%',
                        top: '20px',
                        transform: 'translateX(-50%)',
                        backgroundColor: 'rgba(0, 123, 255, 0.9)',
                        color: 'white',
                        padding: '10px 20px',
                        borderRadius: '5px',
                        fontSize: '14px',
                        whiteSpace: 'nowrap',
                        zIndex: 9999,
                        boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
                        animation: 'fadeIn 0.3s ease-in-out'
                      }}>
                        <i className="fas fa-arrow-left" style={{ marginRight: '8px' }}></i>
                        이전 페이지에 있는 단어에요!
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
                          onClick={() => setShowSearchModal(true)}
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
                    {selectedImageType && (
                      <div
                        style={{
                          position: 'absolute',
                          right: 10,
                          bottom: 10,
                          zIndex: 10
                        }}
                      >
                        <ImageTypeSelector
                          selectedType={selectedImageType}
                          onTypeSelect={handleImageTypeSelect}
                        />
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
                      <div className="page-notification" style={{
                        position: 'fixed',
                        left: '50%',
                        top: '20px',
                        transform: 'translateX(-50%)',
                        backgroundColor: 'rgba(0, 123, 255, 0.9)',
                        color: 'white',
                        padding: '10px 20px',
                        borderRadius: '5px',
                        fontSize: '14px',
                        whiteSpace: 'nowrap',
                        zIndex: 9999,
                        boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
                        animation: 'fadeIn 0.3s ease-in-out'
                      }}>
                        다음 페이지에 있는 단어에요!
                        <i className="fas fa-arrow-right" style={{ marginLeft: '8px' }}></i>
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
                placeholder="검색어를 입력해주세요"
                className="chat-input"
              />
              <button 
                onClick={handleTaskClick} 
                className="chat-button"
                disabled={!selectedMedia && !selectedVideo}
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

          <div style={{ 
            position: 'fixed',
            left: `${taskSuggestionPosition.x}px`,
            top: `${taskSuggestionPosition.y}px`,
            zIndex: 1000
          }}>
            <img 
              src="/images/mp3.png" 
              alt="background" 
              style={{
                width: '500px',
                height: 'auto'
              }}
            />
          </div>
          <div className="task-suggestions-section" 
            ref={taskSuggestionRef}
            onMouseDown={handleTaskSuggestionMouseDown}
            style={{ 
              position: 'fixed',
              left: `${taskSuggestionPosition.x}px`,
              top: `${taskSuggestionPosition.y}px`,
              padding: '0 20px 20px 20px', 
              borderRadius: '8px',
              zIndex: 1000,
              textAlign: 'left',
              width: '455px',
              maxHeight: '300px',
              marginTop: '80px',
              marginLeft: '22px',
              overflow: 'visible',
              cursor: 'move',
              userSelect: 'none'
            }}>
            <div style={{ position: 'relative' }}>
              <h3 style={{ 
                textAlign: 'center',
                fontSize: '25px', 
                color: '#333',
                position: 'absolute',
                top: '-60px',
                left: '0',
                right: '0',
                padding: '10px 0',
                margin: 0,
                zIndex: 1001
              }}>
                {isTaskSuggesting ? '태스크 제안 생성 중...' : '태스크 제안'}
              </h3>
              <div style={{ overflowY: 'auto', height: '300px' }}>
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
                        onClick={() => handleTaskClick(task.task)}
                        style={{
                          padding: '15px',
                          backgroundColor: '#fff',
                          borderRadius: '8px',
                          border: '1px solid #e0e0e0',
                          boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#f8f9fa'}
                        onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#fff'}
                      >
                        <div style={{ 
                          display: 'flex', 
                          justifyContent: 'flex-start', 
                          alignItems: 'center',
                          marginBottom: '8px',
                          gap: '10px'
                        }}>
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
                          <h4 style={{ margin: 0, color: '#333' }}>{task.task}</h4>
                        </div>
                        <p style={{ marginLeft: '15px', color: '#666' }}>{task.description}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', marginTop: '30px', fontSize: '20px', color: '#666' }}>
                    한 번의 업로드, 수많은 가능성의 제안.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {noResults && (
          <div className="no-results-message">
            <i className="fas fa-search"></i>
            <p>검색 결과가 없습니다.</p>
            <p className="sub-text">다른 검색어를 입력하거나 OCR을 새로고침해보세요.</p>
          </div>
        )}

        {summary && (
          <div className="summary-container" style={{
            margin: '20px 0',
            padding: '20px',
            backgroundColor: '#f8f9fa',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <div className="summary-content">
              {summary.split('\n').map((line, index) => (
                <p key={index}>{line}</p>
              ))}
            </div>
          </div>
        )}

        <div className="youtube-input-container">
          <div className="youtube-preview">
            {selectedVideo && selectedVideo.type === 'video' && selectedVideo.url.startsWith('blob:') ? (
              <video
                ref={videoRef}
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

        {mediaType === 'video' && timeline.length > 0 && (
          <div className="timeline-container">
            <h3>타임라인</h3>
            <div className="timeline">
              {timeline.map((item, index) => {
                const minutes = Math.floor(item.timestamp / 60);
                const seconds = Math.floor(item.timestamp % 60);
                const formattedTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                
                return (
                  <div
                    key={index}
                    className="timeline-item"
                    onClick={() => {
                      console.log('Timeline item clicked:', item.timestamp);
                      console.log('Video ref:', videoRef.current);
                      console.log('YouTube ref:', youtubePlayerRef.current);
                      seekToTimestamp(item.timestamp);
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="timestamp">
                      {formattedTime}
                    </div>
                    <div className="texts">
                      <div 
                        className="text-item"
                        style={{ color: item.texts[0]?.color || 'inherit' }}
                        dangerouslySetInnerHTML={{ 
                          __html: item.texts.map(text => text.text).join(' ') 
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {showSearchModal && (
        <div 
          className="modal-overlay"
          onClick={() => setShowSearchModal(false)}
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
            zIndex: 9999
          }}
        >
          <div 
            className="modal-content"
            onClick={e => e.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '90vw',
              maxHeight: '90vh',
              zIndex: 10000
            }}
          >
            <button
              onClick={() => setShowSearchModal(false)}
              style={{
                position: 'absolute',
                top: '-40px',
                right: '0',
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                color: 'white',
                zIndex: 10001
              }}
            >
              ×
            </button>
            <div style={{ position: 'relative', width: '100%', height: '100%' }}>
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <img
                  ref={modalImageRef}
                  src={mediaUrl}
                  alt="검색 결과"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '90vh',
                    objectFit: 'contain',
                    display: 'block'
                  }}
                  onLoad={handleModalImageResize}
                />
                {isModalImageLoaded && modalImageRef.current && (
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: modalImageRef.current.width,
                      height: modalImageRef.current.height,
                      pointerEvents: 'none',
                      zIndex: 10004
                    }}
                  >
                    {detectedObjects.map((obj, index) => {
                      const bbox = obj.bbox;
                      const isNormalized = bbox.x1 <= 1 && bbox.x2 <= 1 && bbox.y1 <= 1 && bbox.y2 <= 1;
                      
                      // 이미지의 실제 표시 크기와 원본 크기 계산
                      const displayWidth = modalImageRef.current!.width;
                      const displayHeight = modalImageRef.current!.height;
                      const originalWidth = imageRef.current?.naturalWidth || 1;
                      const originalHeight = imageRef.current?.naturalHeight || 1;
                      
                      // 이미지의 실제 표시 영역 계산 (object-fit: contain 고려)
                      const scale = Math.min(displayWidth / originalWidth, displayHeight / originalHeight);
                      const scaledWidth = originalWidth * scale;
                      const scaledHeight = originalHeight * scale;
                      const offsetX = (displayWidth - scaledWidth) / 2;
                      const offsetY = (displayHeight - scaledHeight) / 2;
                      
                      let x: number, y: number, width: number, height: number;
                      
                      if (isNormalized) {
                        // 정규화된 좌표를 실제 표시 크기로 변환
                        x = bbox.x1 * scaledWidth + offsetX;
                        y = bbox.y1 * scaledHeight + offsetY;
                        width = (bbox.x2 - bbox.x1) * scaledWidth;
                        height = (bbox.y2 - bbox.y1) * scaledHeight;
                      } else {
                        // 비정규화된 좌표를 실제 표시 크기로 변환
                        x = (bbox.x1 * scale) + offsetX;
                        y = (bbox.y1 * scale) + offsetY;
                        width = (bbox.x2 - bbox.x1) * scale;
                        height = (bbox.y2 - bbox.y1) * scale;
                      }

                      return (
                        <div key={index}>
                          <div
                            style={{
                              position: 'absolute',
                              left: x,
                              top: y,
                              width: width,
                              height: height,
                              border: obj.match_type === 'object' ? '2px solid #00ff00' : '2px solid #ff0000',
                              backgroundColor: obj.match_type === 'object' ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 0, 0, 0.1)',
                              borderRadius: obj.match_type === 'object' ? '0' : '50%',
                              zIndex: 999999999999999,
                              pointerEvents: 'none'
                            }}
                          />
                          <div
                            style={{
                              position: 'absolute',
                              left: x,
                              top: y - 20,
                              backgroundColor: obj.match_type === 'object' ? 'rgba(0, 255, 0, 0.8)' : 'rgba(255, 0, 0, 0.8)',
                              color: 'white',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              fontSize: '12px',
                              fontWeight: 'bold',
                              zIndex: 999999999999999,
                              pointerEvents: 'none',
                              whiteSpace: 'nowrap'
                            }}
                          >
                            {obj.text} ({(obj.confidence * 100).toFixed(1)}%)
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
