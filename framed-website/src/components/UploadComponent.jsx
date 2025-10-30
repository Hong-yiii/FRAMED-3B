import React, { useState, useRef } from 'react';
import { UploadCloud } from 'lucide-react';

// Component styles (Updated `dragging` style for light theme)
const styles = {
  dropzone: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '32px',
    border: '2px dashed var(--border)',
    borderRadius: '8px',
    backgroundColor: 'var(--card)',
    color: 'var(--muted-foreground)',
    cursor: 'pointer',
    transition: 'border-color 0.3s ease, background-color 0.3s ease',
  },
  dragging: {
    borderColor: 'var(--primary)',
    backgroundColor: 'var(--secondary)', // <-- FIXED: Was a dark color
  },
  icon: {
    width: '50px',
    height: '50px',
    marginBottom: '16px',
  },
};

/**
 * A component that provides a drag-and-drop or click-to-upload area.
 * @param {object} props
 * @param {function(File[]): void} props.onFilesSelected - Callback function when files are selected.
 */
function UploadComponent({ onFilesSelected }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // --- Event Handlers ---

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files && files.length > 0) {
      console.log('Files dropped:', files);
      onFilesSelected(files);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files && files.length > 0) {
      console.log('Files selected via click:', files);
      onFilesSelected(files);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  // --- Render ---

  const dropzoneStyle = {
    ...styles.dropzone,
    ...(isDragging ? styles.dragging : {}),
  };

  return (
    <div
      style={dropzoneStyle}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleUploadClick}
    >
      <input
        type="file"
        multiple
        ref={fileInputRef}
        onChange={handleFileSelect}
        style={{ display: 'none' }}
        accept="image/*"
      />

      <UploadCloud style={styles.icon} />
      <span>Drag & drop your photos here, or click to select photos</span>
      <span style={{ fontSize: '0.8rem', marginTop: '8px' }}>
      </span>
    </div>
  );
}

export default UploadComponent;