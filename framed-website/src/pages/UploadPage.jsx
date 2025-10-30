import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import UploadComponent from '../components/UploadComponent';
import { useFiles } from '../context/FileContext';

// This finds your demo files
const demoImageModules = import.meta.glob('../assets/demo-photos/*.{jpg,jpeg,png,gif}');
const presetFiles = Object.entries(demoImageModules).map(([path, _importer]) => {
  const name = path.split('/').pop();
  return {
      name: name,
      path: new URL(path, import.meta.url).href
  };
}).sort((a, b) => {
  const numA = parseInt(a.name.split('.')[0], 10);
  const numB = parseInt(b.name.split('.')[0], 10);
  return numA - numB;
});

// Styles
const styles = {
  pageContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    padding: '2rem',
  },
  header: {
    textAlign: 'center',
    marginBottom: '3rem',
    maxWidth: '600px',
  },
  title: {
    fontFamily: '"Source Serif 4", serif',
    fontSize: '2.5rem',
    fontWeight: 'bold',
    color: 'var(--foreground)',
    marginBottom: '0.5rem',
  },
  subtitle: {
    fontSize: '1.2rem',
    color: 'var(--muted-foreground)',
  },
  uploadContainer: {
    width: '100%',
    maxWidth: '800px',
  }
};

// Framer Motion variants
const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: 'easeOut' },
};

function UploadPage() {
  const navigate = useNavigate();
  const { setFiles } = useFiles();

  const handleFilesSelected = (uploadedFiles) => {
    // This is our "Wizard of Oz" swap
    console.log('Investor uploaded:', uploadedFiles.map(f => f.name));
    console.log(`Swapping with ${presetFiles.length} demo files...`);
    
    // 1. Set the global state with our preset files
    setFiles(presetFiles);
    
    // 2. Immediately navigate to the pipeline!
    navigate('/pipeline');
  };

  return (
    <motion.div 
      style={styles.pageContainer}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div style={styles.header} variants={fadeInUp}>
        <h1 style={styles.title}>Upload Your Photos</h1>
        <p style={styles.subtitle}>
          Drag and drop your photos to begin. We'll automatically start
          the curation as soon as they're added.
        </p>
      </motion.div>

      <motion.div style={styles.uploadContainer} variants={fadeInUp}>
        <UploadComponent onFilesSelected={handleFilesSelected} />
      </motion.div>
    </motion.div>
  );
}

export default UploadPage;