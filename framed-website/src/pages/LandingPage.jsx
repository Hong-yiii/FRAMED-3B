import React, { useState } from 'react';
import UploadComponent from '../components/UploadComponent';

// Styles for this page
const styles = {
  pageContainer: {
    width: '100%',
    maxWidth: '800px',
    margin: '0 auto',
  },
  header: {
    marginBottom: '2rem',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    marginBottom: '0.5rem',
  },
  subtitle: {
    fontSize: '1.2rem',
    color: '#aaa',
  },
  fileList: {
    marginTop: '2rem',
    textAlign: 'left',
  },
  fileListItem: {
    backgroundColor: '#2a2a2a',
    padding: '8px 12px',
    borderRadius: '4px',
    marginBottom: '8px',
    fontSize: '0.9rem',
  },
};

function LandingPage() {
  const [selectedFiles, setSelectedFiles] = useState([]);

  /**
   * This function is called by the UploadComponent when files are selected.
   * It updates the state, which triggers a re-render to show the file list.
   * @param {File[]} files - An array of File objects.
   */
  const handleFilesSelected = (files) => {
    // For this simulation, we'll just set the new files.
    // In a real app, you might *add* them to an existing list.
    setSelectedFiles(files);
  };

  return (
    <div style={styles.pageContainer}>
      <header style={styles.header}>
        <h1 style={styles.title}>FRAMED Curation Pipeline</h1>
        <p style={styles.subtitle}>
          Upload your photos to make your very own zine!
        </p>
      </header>

      {/* --- Upload Component --- */}
      <UploadComponent onFilesSelected={handleFilesSelected} />

      {/* --- File List Section --- */}
      {selectedFiles.length > 0 && (
        <div style={styles.fileList}>
          <h3>Ingesting {selectedFiles.length} Photos:</h3>
          {selectedFiles.map((file, index) => (
            <div key={index} style={styles.fileListItem}>
              {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default LandingPage;