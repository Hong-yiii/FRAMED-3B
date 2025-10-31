import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';
import { Button } from '../components/ui/button';
import { FolderCheck } from 'lucide-react';

export const FinalizePage: React.FC = () => {
  const navigate = useNavigate();
  const { selectedPhotos } = useFileContext();
  const [isCollecting, setIsCollecting] = useState(false);
  const [showButton, setShowButton] = useState(false);

  useEffect(() => {
    if (selectedPhotos.length === 0) {
      navigate('/upload');
      return;
    }

    const timer = setTimeout(() => {
      setIsCollecting(true);
    }, 1000);

    return () => clearTimeout(timer);
  }, [selectedPhotos.length, navigate]);

  useEffect(() => {
    if (isCollecting) {
      const timer = setTimeout(() => {
        setShowButton(true);
      }, 1000 + selectedPhotos.length * 150 + 1000); // Wait for all photos to animate + extra time

      return () => clearTimeout(timer);
    }
  }, [isCollecting, selectedPhotos.length]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4">
      <h2
        className="mb-12 text-center"
        style={{
          fontFamily: 'Source Serif 4, serif',
          fontSize: '2.5rem',
          color: 'oklch(0.22 0.02 80)',
        }}
      >
        Finalizing Your Selection...
      </h2>

      <div className="relative w-full max-w-5xl h-[500px] flex items-center justify-center">
        {/* Folder icon appears first */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="absolute z-10"
        >
          <FolderCheck
            size={200}
            style={{ color: 'oklch(0.22 0.02 80)', opacity: 0.3 }}
          />
        </motion.div>

        {/* Photos fly into the folder and disappear */}
        {selectedPhotos.map((photo, index) => (
          <motion.div
            key={photo.id}
            initial={{
              x: (index - selectedPhotos.length / 2) * 200,
              y: 0,
              scale: 1,
              opacity: 1,
            }}
            animate={isCollecting ? {
              x: 0,
              y: 0,
              scale: 0,
              opacity: 0,
            } : {}}
            transition={{
              duration: 0.8,
              delay: 1 + index * 0.15,
              ease: "easeInOut",
            }}
            className="absolute"
          >
            <img
              src={photo.url}
              alt={photo.name}
              className="w-48 h-32 object-cover rounded-lg shadow-lg"
            />
          </motion.div>
        ))}
      </div>

      {showButton && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mt-12"
        >
          <Button
            onClick={() => navigate('/placement')}
            size="lg"
            style={{
              backgroundColor: 'oklch(0.22 0.02 80)',
              color: 'oklch(0.97 0.035 80)',
            }}
          >
            Move to Placement
          </Button>
        </motion.div>
      )}
    </div>
  );
};
