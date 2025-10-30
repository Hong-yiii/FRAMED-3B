import React, { useState, useRef } from 'react';
import { UploadCloud } from 'lucide-react';

// Component styles (we'll keep them in the file for simplicity)
const styles = {
  dropzone: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '32px',
    border: '2px dashed #444',
    borderRadius: '8px',
    backgroundColor: '#1e1e1e',
    color: '#888',
    cursor: 'pointer',
    transition: 'border-color 0.3s ease, background-color 0.3s ease',
  },
  dragging: {
    borderColor: '#007bff',
    backgroundColor: '#2a2a3a',
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

  // Handle file drag over
  const handleDragOver = (e) => {
    e.preventDefault(); // Prevent default browser behavior
    e.stopPropagation();
    setIsDragging(true);
  };

  // Handle file drag leave
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  // Handle file drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files && files.length > 0) {
      console.log('Files dropped:', files);
      onFilesSelected(files); // Pass files to parent
    }
  };

  // Handle file selection from the hidden input (for clicking)
  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files && files.length > 0) {
      console.log('Files selected via click:', files);
      onFilesSelected(files); // Pass files to parent
    }
  };

  // Trigger the hidden file input when the dropzone is clicked
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  // --- Render ---

  // Combine base style with dragging style if active
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
      {/* Hidden file input */}
      <input
        type="file"
        multiple // Allow multiple files
        ref={fileInputRef}
        onChange={handleFileSelect}
        style={{ display: 'none' }}
        accept="image/*" // Accept only images
      />

      {/* Visible content */}
      <UploadCloud style={styles.icon} />
      <span>Drag & drop your photos here, or click to select files</span>
      <span style={{ fontSize: '0.8rem', marginTop: '8px' }}>
        (This is a simulation, no files will be uploaded to a server)
      </span>
    </div>
  );
}

export default UploadComponent;