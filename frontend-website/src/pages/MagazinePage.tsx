import React, { useState } from 'react';
import { motion } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';
import { Button } from '../components/ui/button';
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const MagazinePage: React.FC = () => {
  const navigate = useNavigate();
  const { selectedPhotos } = useFileContext();
  const [currentPage, setCurrentPage] = useState(0);

  // Create magazine pages with the selected photos
  const magazinePages = [
    // Cover page
    { type: 'cover', content: 'FRAMED' },
    // Photo pages (2 photos per spread)
    ...selectedPhotos.map((photo, index) => ({
      type: 'photo',
      photo,
      index,
    })),
    // Back cover
    { type: 'back', content: 'Your Story' },
  ];

  const totalPages = magazinePages.length;

  const nextPage = () => {
    if (currentPage < totalPages - 1) {
      setCurrentPage(currentPage + 1);
    }
  };

  const prevPage = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };

  const handleStartOver = () => {
    navigate('/');
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <motion.h2
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 text-center"
        style={{
          fontFamily: 'Source Serif 4, serif',
          fontSize: '2.5rem',
          color: 'oklch(0.22 0.02 80)',
        }}
      >
        Your Curated Magazine
      </motion.h2>

      {/* Magazine viewer */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
        className="relative w-full max-w-4xl mb-8"
        style={{
          aspectRatio: '8.5/11',
          backgroundColor: 'oklch(0.99 0.01 80)',
          border: '1px solid oklch(0.85 0.02 80)',
          borderRadius: '8px',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.15)',
        }}
      >
        <motion.div
          key={currentPage}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.4 }}
          className="absolute inset-0 flex items-center justify-center p-12"
        >
          {magazinePages[currentPage].type === 'cover' && (
            <div className="text-center">
              <h1
                style={{
                  fontFamily: 'Source Serif 4, serif',
                  fontSize: '6rem',
                  letterSpacing: '0.3em',
                  color: 'oklch(0.22 0.02 80)',
                }}
              >
                {magazinePages[currentPage].content}
              </h1>
              <p
                className="mt-4 opacity-60"
                style={{
                  color: 'oklch(0.22 0.02 80)',
                }}
              >
                A curated collection
              </p>
            </div>
          )}

          {magazinePages[currentPage].type === 'photo' && (
            <div className="w-full h-full flex items-center justify-center">
              <img
                src={magazinePages[currentPage].photo.url}
                alt={magazinePages[currentPage].photo.name}
                className="max-w-full max-h-full object-contain rounded"
              />
            </div>
          )}

          {magazinePages[currentPage].type === 'back' && (
            <div className="text-center">
              <h2
                className="mb-4"
                style={{
                  fontFamily: 'Source Serif 4, serif',
                  fontSize: '3rem',
                  color: 'oklch(0.22 0.02 80)',
                }}
              >
                {magazinePages[currentPage].content}
              </h2>
              <p
                className="opacity-60"
                style={{
                  color: 'oklch(0.22 0.02 80)',
                }}
              >
                Curated by FRAMED
              </p>
            </div>
          )}
        </motion.div>

        {/* Page indicator */}
        <div
          className="absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-full"
          style={{
            backgroundColor: 'oklch(0.22 0.02 80)',
            color: 'oklch(0.97 0.035 80)',
          }}
        >
          {currentPage + 1} / {totalPages}
        </div>
      </motion.div>

      {/* Navigation controls */}
      <div className="flex items-center gap-4 mb-8">
        <Button
          onClick={prevPage}
          disabled={currentPage === 0}
          variant="outline"
          size="lg"
          style={{
            borderColor: 'oklch(0.22 0.02 80)',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          <ChevronLeft className="mr-2" />
          Previous
        </Button>

        <Button
          onClick={nextPage}
          disabled={currentPage === totalPages - 1}
          variant="outline"
          size="lg"
          style={{
            borderColor: 'oklch(0.22 0.02 80)',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          Next
          <ChevronRight className="ml-2" />
        </Button>
      </div>

      {/* Start over button */}
      <Button
        onClick={handleStartOver}
        variant="ghost"
        style={{
          color: 'oklch(0.22 0.02 80)',
        }}
      >
        <RotateCcw className="mr-2" size={16} />
        Start Over
      </Button>
    </div>
  );
};
