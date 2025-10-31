import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';
import { Card } from '../components/ui/card';
import { Calendar, Plane, Gift } from 'lucide-react';

const purposes = [
  { id: 'event', label: 'An Event', icon: Calendar, description: 'Capture special moments' },
  { id: 'trip', label: 'A Trip', icon: Plane, description: 'Document your adventures' },
  { id: 'gift', label: 'A Gift', icon: Gift, description: 'Create something meaningful' },
];

export const StartPage: React.FC = () => {
  const navigate = useNavigate();
  const { setPurpose } = useFileContext();

  const handlePurposeSelect = (purposeId: string) => {
    setPurpose(purposeId);
    navigate('/theme');
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-4xl"
      >
        <h2
          className="mb-12 text-center"
          style={{
            fontFamily: 'Source Serif 4, serif',
            fontSize: '3rem',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          What are you making this zine for?
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {purposes.map((purpose, index) => {
            const Icon = purpose.icon;
            return (
              <motion.div
                key={purpose.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card
                  className="p-8 cursor-pointer transition-all hover:scale-105 hover:shadow-lg h-full"
                  style={{
                    backgroundColor: 'oklch(0.99 0.01 80)',
                    border: '1px solid oklch(0.85 0.02 80)',
                    minHeight: '240px',
                  }}
                  onClick={() => handlePurposeSelect(purpose.id)}
                >
                  <div className="flex flex-col items-center justify-center text-center space-y-4 h-full">
                    <Icon 
                      size={48} 
                      style={{ color: 'oklch(0.22 0.02 80)' }}
                    />
                    <h3
                      style={{
                        fontFamily: 'Source Serif 4, serif',
                        color: 'oklch(0.22 0.02 80)',
                      }}
                    >
                      {purpose.label}
                    </h3>
                    <p
                      className="opacity-70"
                      style={{ color: 'oklch(0.22 0.02 80)' }}
                    >
                      {purpose.description}
                    </p>
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
};
