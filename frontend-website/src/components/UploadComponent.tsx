import React, { useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { Upload, Plus, ArrowRight } from 'lucide-react';
import { useFileContext } from '../contexts/FileContext';
import { demoPhotos } from '../data/demoPhotos';

export const UploadComponent: React.FC = () => {
  const navigate = useNavigate();
  const { setPhotos } = useFileContext();
  const [isDragOver, setIsDragOver] = React.useState(false);
  const [uploadedPhotos, setUploadedPhotos] = React.useState<typeof demoPhotos | null>(null);
  const [isUploading, setIsUploading] = React.useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((files?: FileList | null) => {
    // Show uploading state
    setIsUploading(true);
    
    // Simulate upload delay
    setTimeout(() => {
      // "Wizard of Oz" magic: Ignore the actual files and load our demo photos
      const photos = demoPhotos.map(photo => ({
        id: photo.id,
        url: photo.url,
        name: photo.name,
        cluster: photo.cluster,
        quality: photo.quality,
      }));
      
      setUploadedPhotos(photos);
      setIsUploading(false);
    }, 1500); // 1.5 second delay to simulate upload
  }, []);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
  }, [handleFiles]);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleClick = useCallback(() => {
    // Open file picker
    fileInputRef.current?.click();
  }, []);

  const handleUploadMore = useCallback(() => {
    // Open file picker again
    fileInputRef.current?.click();
  }, []);

  const handleContinue = useCallback(() => {
    if (!uploadedPhotos) return;
    
    const photos = uploadedPhotos.map(photo => ({
      id: photo.id,
      url: photo.url,
      name: photo.name,
    }));
    
    setPhotos(photos);
    
    // Navigate to the first pipeline stage
    setTimeout(() => {
      navigate('/deduplicate');
    }, 300);
  }, [uploadedPhotos, navigate, setPhotos]);

  // Show uploading state
  if (isUploading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center py-16"
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          className="inline-block mb-4"
        >
          <Upload size={48} style={{ color: 'oklch(0.22 0.02 80)' }} />
        </motion.div>
        <p
          style={{
            fontFamily: 'Inter, sans-serif',
            fontSize: '1.125rem',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          Uploading your photos...
        </p>
      </motion.div>
    );
  }

  // Show uploaded photos view
  if (uploadedPhotos) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="mb-6 grid grid-cols-4 gap-3">
          {uploadedPhotos.map((photo) => (
            <motion.div
              key={photo.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="aspect-square overflow-hidden rounded-lg"
              style={{
                backgroundColor: 'oklch(0.95 0.02 80)',
              }}
            >
              <img
                src={photo.url}
                alt={photo.name}
                className="h-full w-full object-cover"
              />
            </motion.div>
          ))}
        </div>

        <div className="flex items-center justify-center gap-4">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleUploadMore}
            className="flex items-center gap-2 transition-all"
            style={{
              fontFamily: 'Inter, sans-serif',
              color: 'oklch(0.22 0.02 80)',
              backgroundColor: 'oklch(0.99 0.01 80)',
              border: '1px solid oklch(0.6 0.02 80)',
              borderRadius: '8px',
              padding: '0.875rem 2rem',
              cursor: 'pointer',
            }}
          >
            <Plus size={20} />
            Upload More Photos
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleContinue}
            className="flex items-center gap-2 transition-all"
            style={{
              fontFamily: 'Inter, sans-serif',
              color: 'oklch(0.99 0.01 80)',
              backgroundColor: 'oklch(0.22 0.02 80)',
              border: 'none',
              borderRadius: '8px',
              padding: '0.875rem 2rem',
              cursor: 'pointer',
            }}
          >
            Continue
            <ArrowRight size={20} />
          </motion.button>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
      </motion.div>
    );
  }

  // Show upload dropzone
  return (
    <>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
        className="cursor-pointer transition-all"
        style={{
          border: `2px dashed ${isDragOver ? 'oklch(0.22 0.02 80)' : 'oklch(0.6 0.02 80)'}`,
          backgroundColor: isDragOver ? 'oklch(0.95 0.02 80)' : 'oklch(0.99 0.01 80)',
          borderRadius: '12px',
          padding: '4rem',
          textAlign: 'center',
        }}
      >
        <Upload 
          size={64} 
          style={{ 
            color: 'oklch(0.22 0.02 80)',
            margin: '0 auto 1.5rem',
          }} 
        />
        <h3
          className="mb-2"
          style={{
            fontFamily: 'Source Serif 4, serif',
            color: 'oklch(0.22 0.02 80)',
          }}
        >
          Drop your photos here
        </h3>
        <p
          className="opacity-70"
          style={{ color: 'oklch(0.22 0.02 80)' }}
        >
          or click to browse
        </p>
      </motion.div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />
    </>
  );
};
