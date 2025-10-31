import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';
import { Button } from '../components/ui/button';

const templateSlots = [
  { x: '8%', y: '12%', width: '38%', height: '35%' },
  { x: '54%', y: '12%', width: '38%', height: '35%' },
  { x: '8%', y: '53%', width: '38%', height: '35%' },
  { x: '54%', y: '53%', width: '38%', height: '35%' },
];

export const PlacementPage: React.FC = () => {
  const navigate = useNavigate();
  const { selectedPhotos } = useFileContext();
  const [isPlacing, setIsPlacing] = useState(false);
  const [showButton, setShowButton] = useState(false);
  const [currentPhotoIndex, setCurrentPhotoIndex] = useState(-1);

  useEffect(() => {
    if (selectedPhotos.length === 0) {
      navigate('/upload');
      return;
    }

    const timer = setTimeout(() => {
      setIsPlacing(true);
    }, 1000);

    return () => clearTimeout(timer);
  }, [selectedPhotos.length, navigate]);

  useEffect(() => {
    if (isPlacing && currentPhotoIndex < selectedPhotos.length - 1) {
      const timer = setTimeout(() => {
        setCurrentPhotoIndex(prev => prev + 1);
      }, 1200); // 1.2 seconds per photo

      return () => clearTimeout(timer);
    } else if (isPlacing && currentPhotoIndex >= selectedPhotos.length - 1) {
      const timer = setTimeout(() => {
        setShowButton(true);
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [isPlacing, currentPhotoIndex, selectedPhotos.length]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <h2
        className="mb-12 text-center"
        style={{
          fontFamily: 'Source Serif 4, serif',
          fontSize: '2.5rem',
          color: 'oklch(0.22 0.02 80)',
        }}
      >
        Creating Your Layout...
      </h2>

      <div 
        className="relative w-full max-w-4xl mb-12"
        style={{
          aspectRatio: '16/9',
          backgroundColor: 'oklch(0.99 0.01 80)',
          border: '2px solid oklch(0.85 0.02 80)',
          borderRadius: '8px',
        }}
      >
        {/* Template slots */}
        {templateSlots.map((slot, index) => (
          <div
            key={index}
            className="absolute"
            style={{
              left: slot.x,
              top: slot.y,
              width: slot.width,
              height: slot.height,
              border: '2px dashed oklch(0.7 0.02 80)',
              borderRadius: '4px',
            }}
          />
        ))}

        {/* Photos flying into slots - loops through all photos */}
        {selectedPhotos.map((photo, photoIndex) => {
          const slotIndex = photoIndex % templateSlots.length;
          const slot = templateSlots[slotIndex];
          const shouldShow = currentPhotoIndex >= photoIndex;
          
          return (
            <motion.div
              key={photo.id}
              initial={{
                left: '50%',
                top: '-20%',
                scale: 0.3,
                opacity: 0,
                translateX: '-50%',
              }}
              animate={shouldShow ? {
                left: slot.x,
                top: slot.y,
                scale: 1,
                opacity: 1,
                translateX: '0%',
              } : {}}
              transition={{
                duration: 1,
                ease: "easeOut",
              }}
              className="absolute"
              style={{
                width: slot.width,
                height: slot.height,
                zIndex: shouldShow ? 10 + photoIndex : 0,
              }}
            >
              <img
                src={photo.url}
                alt={photo.name}
                className="w-full h-full object-cover rounded"
                style={{
                  boxShadow: shouldShow ? '0 4px 12px rgba(0,0,0,0.15)' : 'none',
                }}
              />
            </motion.div>
          );
        })}
      </div>

      {showButton && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Button
            onClick={() => navigate('/magazine')}
            size="lg"
            style={{
              backgroundColor: 'oklch(0.22 0.02 80)',
              color: 'oklch(0.97 0.035 80)',
            }}
          >
            View Final Magazine
          </Button>
        </motion.div>
      )}
    </div>
  );
};
