import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';
import { Star } from 'lucide-react';

export const RankingPage: React.FC = () => {
  const navigate = useNavigate();
  const { clusters, setSelectedPhotos } = useFileContext();
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (clusters.length === 0) {
      navigate('/upload');
      return;
    }

    // Start selecting heroes
    if (selectedIndices.length === 0) {
      const timer = setTimeout(() => {
        setSelectedIndices([0]);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [clusters.length, selectedIndices.length, navigate]);

  // Select heroes one by one
  useEffect(() => {
    if (selectedIndices.length === 0 || selectedIndices.length >= clusters.length) {
      return;
    }

    const timer = setTimeout(() => {
      setSelectedIndices(prev => [...prev, prev.length]);
    }, 1000);

    return () => clearTimeout(timer);
  }, [selectedIndices.length, clusters.length]);

  // When all heroes are selected, prepare to navigate
  useEffect(() => {
    if (selectedIndices.length === clusters.length && clusters.length > 0 && !isComplete) {
      setIsComplete(true);
    }
  }, [selectedIndices.length, clusters.length, isComplete]);

  useEffect(() => {
    if (isComplete) {
      const timer = setTimeout(() => {
        const heroes = clusters.map(cluster => cluster.photos[0]);
        setSelectedPhotos(heroes);
        navigate('/finalize');
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [isComplete, clusters, navigate, setSelectedPhotos]);

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
        Selecting Your Best Shots...
      </h2>

      <div className="grid grid-cols-2 gap-12 max-w-5xl">
        {clusters.map((cluster, clusterIndex) => (
          <motion.div
            key={cluster.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: clusterIndex * 0.2 }}
            className="space-y-4"
          >
            <h3
              className="text-center mb-4"
              style={{
                fontFamily: 'Source Serif 4, serif',
                color: 'oklch(0.22 0.02 80)',
              }}
            >
              {cluster.name}
            </h3>

            <div className="grid grid-cols-2 gap-4">
              {cluster.photos.map((photo, photoIndex) => {
                const isHero = photoIndex === 0;
                const isSelected = selectedIndices.includes(clusterIndex);

                return (
                  <motion.div
                    key={photo.id}
                    initial={{ opacity: 1, scale: 1 }}
                    animate={{
                      opacity: isSelected && !isHero ? 0.3 : 1,
                      scale: isSelected && isHero ? 1.1 : 1,
                    }}
                    transition={{ duration: 0.5 }}
                    className="relative"
                  >
                    <img
                      src={photo.url}
                      alt={photo.name}
                      className="w-full h-32 object-cover rounded-lg shadow-md"
                    />

                    {isSelected && isHero && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="absolute inset-0 flex items-center justify-center"
                        style={{
                          border: '3px solid oklch(0.22 0.02 80)',
                          borderRadius: '0.5rem',
                        }}
                      >
                        <div
                          className="p-2 rounded-full"
                          style={{
                            backgroundColor: 'oklch(0.22 0.02 80)',
                          }}
                        >
                          <Star
                            size={24}
                            fill="oklch(0.97 0.035 80)"
                            style={{ color: 'oklch(0.97 0.035 80)' }}
                          />
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};
