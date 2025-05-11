import React, { useState } from 'react';

interface MediaUploaderProps {
  selectedMedia: {
    type: 'video' | 'image';
    file: File;
  } | null;
  onRefresh: () => void;
}

const MediaUploader: React.FC<MediaUploaderProps> = ({ selectedMedia, onRefresh }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRefresh = async () => {
    if (!selectedMedia) return;
    
    setIsProcessing(true);
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append(selectedMedia.type === 'video' ? 'video' : 'image', selectedMedia.file);
      formData.append('analyze', 'true');
      
      const serverUrl = 'http://localhost:5001';
      const endpoint = selectedMedia.type === 'video' ? 'upload' : 'upload-image';
      const fullUrl = `${serverUrl}/${endpoint}`;
      
      const response = await fetch(fullUrl, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to refresh media');
      }
      
      const data = await response.json();
      
      if (selectedMedia.type === 'video' && data.file_url) {
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
            
            if (!subtitleResponse.ok) {
              throw new Error('Failed to extract subtitles');
            }
            
            const subtitleData = await subtitleResponse.json();
            console.log('=== 자막 추출 완료 ===', subtitleData);
            
            onRefresh(); // 부모 컴포넌트에 새로고침 완료 알림
          } catch (error) {
            console.error('Error extracting subtitles:', error);
            setError('자막 추출 중 오류가 발생했습니다.');
          }
        }
      }
      
      setIsProcessing(false);
      setIsAnalyzing(false);
    } catch (error) {
      console.error('Error refreshing media:', error);
      setError('미디어 새로고침 중 오류가 발생했습니다.');
      setIsProcessing(false);
      setIsAnalyzing(false);
    }
  };

  return (
    <div>
      {error && <p className="error">{error}</p>}
      {isProcessing && <p>Processing...</p>}
      {isAnalyzing && <p>Analyzing...</p>}
      <button 
        className="refresh-button"
        onClick={handleRefresh}
        disabled={isProcessing || isAnalyzing}
      >
        <i className="fas fa-sync-alt"></i>
      </button>
    </div>
  );
};

export default MediaUploader; 