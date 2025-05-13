import React, { useState } from 'react';
import { ImageType, IMAGE_TYPE_ICONS, IMAGE_TYPE_LABELS, IMAGE_TYPE_COLORS } from '../types';
import './ImageTypeSelector.css';

interface ImageTypeSelectorProps {
  selectedType: ImageType;
  onTypeSelect: (type: ImageType) => void;
  ocrText?: string;
}

const ImageTypeSelector: React.FC<ImageTypeSelectorProps> = ({
  selectedType,
  onTypeSelect,
  ocrText
}) => {
  const [showSelector, setShowSelector] = useState(false);

  const handleTypeSelect = (type: ImageType) => {
    onTypeSelect(type);
    setShowSelector(false);
  };

  return (
    <div className="image-type-picker">
      <button
        className="picker-button"
        onClick={() => setShowSelector(!showSelector)}
        title={IMAGE_TYPE_LABELS[selectedType]}
        style={{ backgroundColor: IMAGE_TYPE_COLORS[selectedType] }}
      >
        <i className={IMAGE_TYPE_ICONS[selectedType]}></i>
      </button>
      
      {showSelector && (
        <div className="image-type-selector show">
          {Object.entries(IMAGE_TYPE_ICONS).map(([type, icon]) => (
            <button
              key={type}
              className={`type-button ${type === selectedType ? 'selected' : ''}`}
              onClick={() => handleTypeSelect(type as ImageType)}
              title={IMAGE_TYPE_LABELS[type as ImageType]}
              style={{ 
                backgroundColor: IMAGE_TYPE_COLORS[type as ImageType],
                color: 'white'
              }}
            >
              <i className={icon as string}></i>
              <span>{IMAGE_TYPE_LABELS[type as ImageType]}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageTypeSelector; 