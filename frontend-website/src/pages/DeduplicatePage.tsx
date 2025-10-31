import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { Progress } from '../components/ui/progress';

export const DeduplicatePage: React.FC = () => {
  const navigate = useNavigate();
  const [progress, setProgress] = React.useState(0);

  useEffect(() => {
    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 10;
      });
    }, 200);

    // Navigate after completion
    const timer = setTimeout(() => {
      navigate('/preprocess');
    }, 2500);

    return () => {
      clearInterval(interval);
      clearTimeout(timer);
    };
  }, [navigate]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-xl text-center"
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="mb-8 mx-auto w-16 h-16 rounded-full"
          style={{
            border: '4px solid oklch(0.85 0.02 80)',
            borderTopColor: 'oklch(0.22 0.02 80)',
          }}
        />

        <h2
          className="mb-4"
          style={{
            fontFamily: 'Source Serif 4, serif',
            fontSize: '2.5rem',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          Finding Moments...
        </h2>

        <p
          className="mb-8 opacity-70"
          style={{ color: 'oklch(0.22 0.02 80)' }}
        >
          Analyzing your photos and removing duplicates
        </p>

        <Progress value={progress} className="w-full" />
      </motion.div>
    </div>
  );
};
