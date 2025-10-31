import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';
import { demoPhotos } from '../data/demoPhotos';
import { Badge } from '../components/ui/badge';

interface PhotoWithCluster {
  id: string;
  url: string;
  name: string;
  cluster?: string;
  quality?: string;
  position?: { x: number; y: number };
}

export const ClusterPage: React.FC = () => {
  const navigate = useNavigate();
  const { photos, setClusters } = useFileContext();
  const [analyzedPhotos, setAnalyzedPhotos] = useState<PhotoWithCluster[]>([]);
  const [currentAnalyzing, setCurrentAnalyzing] = useState(0);
  const [isGrouping, setIsGrouping] = useState(false);

  useEffect(() => {
    if (photos.length === 0) {
      navigate('/upload');
      return;
    }

    // Phase 1: Analyze photos one by one
    if (currentAnalyzing < photos.length && !isGrouping) {
      const timer = setTimeout(() => {
        const demoPhoto = demoPhotos.find(dp => dp.id === photos[currentAnalyzing].id);
        setAnalyzedPhotos(prev => [...prev, {
          ...photos[currentAnalyzing],
          cluster: demoPhoto?.cluster,
          quality: demoPhoto?.quality,
        }]);
        setCurrentAnalyzing(prev => prev + 1);
      }, 800);
      return () => clearTimeout(timer);
    }

    // Phase 2: Start grouping phase
    if (currentAnalyzing === photos.length && !isGrouping && analyzedPhotos.length === photos.length) {
      setIsGrouping(true);
    }
  }, [currentAnalyzing, photos.length, isGrouping, analyzedPhotos.length, navigate]);

  // Separate effect for grouping and navigation
  useEffect(() => {
    if (isGrouping && analyzedPhotos.length > 0) {
      const timer = setTimeout(() => {
        // Create clusters from analyzed photos
        const clusterMap = new Map<string, PhotoWithCluster[]>();
        analyzedPhotos.forEach(photo => {
          const clusterName = photo.cluster || 'other';
          if (!clusterMap.has(clusterName)) {
            clusterMap.set(clusterName, []);
          }
          clusterMap.get(clusterName)!.push(photo);
        });

        const clustersArray = Array.from(clusterMap.entries()).map(([name, photos]) => ({
          id: name,
          name: name.charAt(0).toUpperCase() + name.slice(1),
          photos: photos.map(p => ({ id: p.id, url: p.url, name: p.name })),
        }));

        setClusters(clustersArray);

        setTimeout(() => {
          navigate('/ranking');
        }, 2000);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [isGrouping, analyzedPhotos, navigate, setClusters]);

  const clusterPositions = {
    portraits: { x: -300, y: -150 },
    landscapes: { x: 300, y: -150 },
    architecture: { x: -300, y: 150 },
    lifestyle: { x: 300, y: 150 },
  };

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
        {isGrouping ? 'Grouping Similar Moments...' : 'Analyzing Photos...'}
      </h2>

      <div className="relative w-full max-w-6xl h-[600px]">
        <AnimatePresence>
          {analyzedPhotos.map((photo, index) => {
            const position = isGrouping && photo.cluster 
              ? clusterPositions[photo.cluster as keyof typeof clusterPositions] 
              : { x: 0, y: 0 };

            return (
              <motion.div
                key={photo.id}
                initial={{ opacity: 0, scale: 0, x: 0, y: 0 }}
                animate={{
                  opacity: 1,
                  scale: isGrouping ? 0.7 : 1,
                  x: position?.x || 0,
                  y: position?.y || 0,
                }}
                transition={{
                  duration: 0.6,
                  delay: isGrouping ? index * 0.1 : 0,
                }}
                className="absolute"
                style={{
                  left: '50%',
                  top: '50%',
                  transform: 'translate(-50%, -50%)',
                }}
              >
                <div className="relative">
                  <img
                    src={photo.url}
                    alt={photo.name}
                    className="w-48 h-32 object-cover rounded-lg shadow-lg"
                  />
                  
                  {index === currentAnalyzing - 1 && !isGrouping && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="absolute -top-10 left-1/2 -translate-x-1/2"
                    >
                      <Badge
                        style={{
                          backgroundColor: 'oklch(0.22 0.02 80)',
                          color: 'oklch(0.97 0.035 80)',
                        }}
                      >
                        {photo.quality === 'high' ? 'Great Quality' : 'Good Quality'}
                      </Badge>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {isGrouping && (
          <>
            {Object.entries(clusterPositions).map(([name, pos]) => (
              <motion.div
                key={name}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute"
                style={{
                  left: `calc(50% + ${pos.x}px)`,
                  top: `calc(50% + ${pos.y}px)`,
                  transform: 'translate(-50%, calc(-100% - 100px))',
                }}
              >
                <p
                  style={{
                    fontFamily: 'Source Serif 4, serif',
                    color: 'oklch(0.22 0.02 80)',
                  }}
                >
                  {name.charAt(0).toUpperCase() + name.slice(1)}
                </p>
              </motion.div>
            ))}
          </>
        )}
      </div>
    </div>
  );
};
