import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';

export const PreprocessPage: React.FC = () => {
  const navigate = useNavigate();
  const { photos } = useFileContext();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (photos.length === 0) {
      navigate('/upload');
      return;
    }

    if (currentIndex < photos.length) {
      const timer = setTimeout(() => {
        setCurrentIndex(prev => prev + 1);
      }, 600);
      return () => clearTimeout(timer);
    } else if (!isComplete) {
      setIsComplete(true);
    }
  }, [currentIndex, photos.length, isComplete, navigate]);

  useEffect(() => {
    if (isComplete) {
      const timer = setTimeout(() => {
        navigate('/cluster');
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [isComplete, navigate]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="w-full max-w-4xl"
      >
        <h2
          className="mb-12 text-center"
          style={{
            fontFamily: 'Source Serif 4, serif',
            fontSize: '2.5rem',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          Enhancing Quality...
        </h2>

        <div className="relative h-96 flex items-center justify-center">
          <AnimatePresence mode="wait">
            {currentIndex < photos.length && (
              <motion.div
                key={photos[currentIndex].id}
                initial={{ opacity: 0, scale: 0.8, rotateY: -90 }}
                animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                exit={{ opacity: 0, scale: 0.8, rotateY: 90 }}
                transition={{ duration: 0.5 }}
                className="relative"
              >
                <img
                  src={photos[currentIndex].url}
                  alt={photos[currentIndex].name}
                  className="max-h-80 max-w-full rounded-lg shadow-lg object-contain"
                />
                
                {/* Scan line effect */}
                <motion.div
                  initial={{ top: 0 }}
                  animate={{ top: '100%' }}
                  transition={{ duration: 0.5 }}
                  className="absolute left-0 right-0 h-1"
                  style={{
                    background: 'linear-gradient(to bottom, transparent, oklch(0.22 0.02 80), transparent)',
                    opacity: 0.5,
                  }}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="mt-8 text-center">
          <p
            className="opacity-70"
            style={{ color: 'oklch(0.22 0.02 80)' }}
          >
            Processing {Math.min(currentIndex + 1, photos.length)} of {photos.length}
          </p>
        </div>
      </motion.div>
    </div>
  );
};
