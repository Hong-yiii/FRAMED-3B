import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { ArrowRight } from 'lucide-react';
import logo from '../assets/framed-logo.png';

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen items-center justify-center px-8 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="flex flex-col items-center text-center max-w-4xl w-full"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          style={{
            marginBottom: '3.5rem',
            maxWidth: '700px',
            width: '100%',
          }}
        >
          <img 
            src={logo} 
            alt="Framed" 
            style={{
              width: '100%',
              height: 'auto',
            }}
          />
        </motion.div>
        
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          style={{
            fontFamily: 'Inter, sans-serif',
            fontSize: '1.75rem',
            color: '#000',
            marginBottom: '4rem',
            maxWidth: '800px',
          }}
        >
          Turn your digital clutter into a story you can hold.
        </motion.p>

        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => navigate('/start')}
          style={{
            fontFamily: 'Inter, sans-serif',
            fontSize: '1.25rem',
            color: '#fff',
            backgroundColor: '#000',
            border: 'none',
            padding: '1.5rem 3.5rem',
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '1rem',
            transition: 'all 0.2s ease',
          }}
        >
          Start Your Curation
          <ArrowRight size={24} />
        </motion.button>
      </motion.div>
    </div>
  );
};
