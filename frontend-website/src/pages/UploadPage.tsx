import React from 'react';
import { motion } from 'motion/react';
import { UploadComponent } from '../components/UploadComponent';

export const UploadPage: React.FC = () => {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-2xl"
      >
        <h2
          className="mb-8 text-center"
          style={{
            fontFamily: 'Source Serif 4, serif',
            fontSize: '3rem',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          Share Your Photos
        </h2>
        
        <p
          className="mb-8 text-center opacity-70"
          style={{ color: 'oklch(0.22 0.02 80)' }}
        >
          Upload your photo library and let us transform it into something beautiful.
        </p>

        <UploadComponent />
      </motion.div>
    </div>
  );
};
