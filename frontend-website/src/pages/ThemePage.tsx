import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { useFileContext } from '../contexts/FileContext';
import { Card } from '../components/ui/card';
import { Sparkles, Sun, Zap, Palette, Clock } from 'lucide-react';

const themes = [
  { 
    id: 'minimal', 
    label: 'Minimal', 
    icon: Sparkles, 
    description: 'Clean and timeless',
    colors: ['#FFFFFF', '#F5F5F5', '#E0E0E0']
  },
  { 
    id: 'warm', 
    label: 'Warm', 
    icon: Sun, 
    description: 'Cozy and inviting',
    colors: ['#F4E4D7', '#E8C9A3', '#D4A574']
  },
  { 
    id: 'bold', 
    label: 'Bold', 
    icon: Zap, 
    description: 'Vibrant and striking',
    colors: ['#FF6B6B', '#4ECDC4', '#FFE66D']
  },
  { 
    id: 'artistic', 
    label: 'Artistic', 
    icon: Palette, 
    description: 'Creative and expressive',
    colors: ['#9B59B6', '#3498DB', '#E74C3C']
  },
  { 
    id: 'vintage', 
    label: 'Vintage', 
    icon: Clock, 
    description: 'Classic and nostalgic',
    colors: ['#D4A373', '#8B7355', '#A67C52']
  },
];

export const ThemePage: React.FC = () => {
  const navigate = useNavigate();
  const { setTheme } = useFileContext();

  const handleThemeSelect = (themeId: string) => {
    setTheme(themeId);
    navigate('/upload');
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-5xl"
      >
        <h2
          className="mb-4 text-center"
          style={{
            fontFamily: 'Source Serif 4, serif',
            fontSize: '3rem',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          Choose Your Aesthetic
        </h2>
        
        <p
          className="mb-12 text-center opacity-70"
          style={{ 
            color: 'oklch(0.22 0.02 80)',
            fontSize: '1.125rem',
          }}
        >
          Select a theme that matches your vision
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
          {themes.map((theme, index) => {
            const Icon = theme.icon;
            return (
              <motion.div
                key={theme.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card
                  className="p-6 cursor-pointer transition-all hover:scale-105 hover:shadow-lg h-full"
                  style={{
                    backgroundColor: 'oklch(0.99 0.01 80)',
                    border: '1px solid oklch(0.85 0.02 80)',
                    minHeight: '240px',
                  }}
                  onClick={() => handleThemeSelect(theme.id)}
                >
                  <div className="flex flex-col items-center justify-center text-center space-y-4 h-full">
                    <Icon 
                      size={40} 
                      style={{ color: 'oklch(0.22 0.02 80)' }}
                    />
                    <div className="space-y-2">
                      <h3
                        style={{
                          fontFamily: 'Source Serif 4, serif',
                          color: 'oklch(0.22 0.02 80)',
                        }}
                      >
                        {theme.label}
                      </h3>
                      <p
                        className="opacity-70"
                        style={{ 
                          color: 'oklch(0.22 0.02 80)',
                          fontSize: '0.875rem',
                        }}
                      >
                        {theme.description}
                      </p>
                    </div>
                    
                    {/* Color palette preview */}
                    <div className="flex gap-2 mt-2">
                      {theme.colors.map((color, i) => (
                        <motion.div
                          key={i}
                          className="w-8 h-8 rounded-full border"
                          style={{
                            backgroundColor: color,
                            borderColor: 'oklch(0.85 0.02 80)',
                          }}
                          whileHover={{ scale: 1.1 }}
                        />
                      ))}
                    </div>
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
