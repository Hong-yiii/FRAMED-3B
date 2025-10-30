import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import UploadComponent from '../components/UploadComponent';
import { useFiles } from '../context/FileContext';
import framedLogo from '../assets/framed_logo.png'; // <-- NEW: Import your logo

// Styles updated for light theme and logo
const styles = {
  pageContainer: {
    width: '100%',
    maxWidth: '800px',
    margin: '0 auto',
    minHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
  },
  header: {
    marginBottom: '1rem',
    textAlign: 'centre',
  },
  logo: { // <-- NEW: Logo specific styles
    width: 'auto',
    height: '260px', /* Adjust height as needed */
    marginBottom: '0.1rem',
    objectFit: 'contain',
  },
  title: { // No longer used for text, but keeping for reference or if you want to add text back
    fontSize: '2.5rem',
    fontWeight: 'bold',
    marginBottom: '0.1rem',
    fontFamily: '"Source Serif 4", serif',
    color: 'var(--foreground)',
  },
  subtitle: {
    fontSize: '1.2rem',
    color: 'var(--muted-foreground)',
  },
  fileList: {
    marginTop: '0.5rem',
    textAlign: 'left',
  },
  fileListItem: {
    backgroundColor: 'var(--secondary)',
    color: 'var(--muted-foreground)',
    padding: '8px 12px',
    borderRadius: '4px',
    marginBottom: '8px',
    fontSize: '0.9rem',
  },
  button: {
    marginTop: '1rem',
    padding: '12px 24px',
    fontSize: '1rem',
    fontWeight: '600',
    color: 'var(--primary-foreground)',
    backgroundColor: 'var(--primary)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'transform 0.2s ease',
  },
};

// Framer Motion variants
const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: 'easeOut' },
};

function LandingPage() {
  const navigate = useNavigate();
  const { files, setFiles } = useFiles();

  const handleFilesSelected = (selectedFiles) => {
    setFiles(selectedFiles);
  };

  const handleStartCuration = () => {
    navigate('/pipeline');
  };

  return (
    <motion.div
      style={styles.pageContainer}
      initial="initial"
      animate="animate"
      variants={{
        animate: { transition: { staggerChildren: 0.1 } },
      }}
    >
      <motion.header style={styles.header} variants={fadeInUp}>
        {/* --- NEW: Replace H1 with Image Logo --- */}
        <img src={framedLogo} alt="FRAMED Logo" style={styles.logo} />
        {/* --- END NEW --- */}
        <p style={styles.subtitle}>
          Upload your photos to begin making your own zine!
        </p>
      </motion.header>

      {/* --- Upload Component --- */}
      <motion.div variants={fadeInUp}>
        <UploadComponent onFilesSelected={handleFilesSelected} />
      </motion.div>

      {/* --- File List & Start Button --- */}
      {files.length > 0 && (
        <motion.div style={styles.fileList} variants={fadeInUp}>
          <h3>Ingesting {files.length} Photos...</h3>
          <p style={{
            fontSize: '0.85rem',
            color: 'var(--muted-foreground)',
            marginBottom: '1.5rem',
            maxHeight: '100px',
            overflowY: 'auto',
            backgroundColor: 'var(--card)',
            padding: '10px',
            borderRadius: '8px'
          }}>
            {files.map((f) => f.name).join(', ')}
          </p>

          <motion.button
            style={styles.button}
            onClick={handleStartCuration}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Start Curation Process →
          </motion.button>
        </motion.div>
      )}
    </motion.div>
  );
}

export default LandingPage;