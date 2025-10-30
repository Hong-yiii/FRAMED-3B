
import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useFiles } from '../context/FileContext'; // <-- Get our files

// Styles for this page updated to use CSS Variables
const styles = {
  pageContainer: {
    width: '100%',
    maxWidth: '1000px',
    margin: '0 auto',
    padding: '2rem 0',
  },
  stage: {
    display: 'flex',
    marginBottom: '4rem',
    minHeight: '200px',
  },
  stageInfo: {
    width: '30%',
    paddingRight: '2rem',
    textAlign: 'right',
  },
  stageNumber: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: 'var(--muted-foreground)', // Use variable
    fontFamily: '"Source Serif 4", serif',
  },
  stageTitle: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: 'var(--foreground)', // Use variable
    marginTop: '0.5rem',
    fontFamily: '"Source Serif 4", serif',
  },
  stageDesc: {
    color: 'var(--muted-foreground)', // Use variable
    marginTop: '0.5rem',
  },
  stageContent: {
    width: '70%',
    paddingLeft: '2rem',
    borderLeft: '2px solid var(--border)', // Use variable
    display: 'flex',
    flexWrap: 'wrap',
    gap: '10px',
    alignContent: 'flex-start',
  },
  thumbnail: {
    width: '100px',
    height: '100px',
    borderRadius: '8px',
    objectFit: 'cover',
    backgroundColor: 'var(--card)', // Use variable
  },
};

// Framer Motion variants
const fadeInUp = {
  initial: { opacity: 0, y: 40 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: 'easeOut' },
};

// This component will show a single photo thumbnail
function FilePreview({ file }) {
  const [imgSrc, setImgSrc] = useState(null);

  useEffect(() => {
    // Create a temporary URL to display the image
    const objectUrl = URL.createObjectURL(file);
    setImgSrc(objectUrl);

    // Clean up the object URL when the component unmounts
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  if (!imgSrc) return null;

  return (
    <motion.img
      src={imgSrc}
      alt={file.name}
      style={styles.thumbnail}
      variants={fadeInUp} // Each thumbnail fades in
      layout // This will animate it when it moves!
    />
  );
}

// Main Page Component
function PipelinePage() {
  const { files } = useFiles();

  // We'll add more state here later to simulate progression
  // const [currentStage, setCurrentStage] = useState('ingest');

  return (
    <motion.div
      style={styles.pageContainer}
      initial="initial"
      animate="animate"
      variants={{
        animate: { transition: { staggerChildren: 0.2 } },
      }}
    >
      {/* Stage 1: Ingest */}
      <motion.section style={styles.stage} variants={fadeInUp}>
        <div style={styles.stageInfo}>
          <div style={styles.stageNumber}>01</div>
          <h2 style={styles.stageTitle}>Ingest</h2>
          <p style={styles.stageDesc}>Receiving {files.length} photos...</p>
        </div>
        <div style={styles.stageContent}>
          {/* We'll show the images in the *next* step */}
          <p style={{ color: '#666' }}>Ingest complete. Starting preprocess...</p>
        </div>
      </motion.section>

      {/* Stage 2: Preprocess */}
      <motion.section style={styles.stage} variants={fadeInUp}>
        <div style={styles.stageInfo}>
          <div style={styles.stageNumber}>02</div>
          <h2 style={styles.stageTitle}>Preprocess</h2>
          <p style={styles.stageDesc}>Standardizing and enhancing images.</p>
        </div>
        <motion.div
          style={styles.stageContent}
          variants={{
            animate: { transition: { staggerChildren: 0.05 } }, // Stagger thumbnails
          }}
        >
          {files.map((file, index) => (
            <FilePreview key={file.name + index} file={file} />
          ))}
        </motion.div>
      </motion.section>

      {/* --- SIMULATED STAGES (Wizard of Oz) --- */}
      {/* We'll animate these in the next step */}

      {/* Stage 3-5: Features, Scoring, Clustering */}
      <motion.section style={styles.stage} variants={fadeInUp}>
        <div style={styles.stageInfo}>
          <div style={styles.stageNumber}>03-05</div>
          <h2 style={styles.stageTitle}>Analyze & Cluster</h2>
          <p style={styles.stageDesc}>
            AI analysis of features, quality, and moments.
          </p>
        </div>
        <div style={styles.stageContent}>
          <p style={{ color: '#666' }}>
            Simulating grouping {files.length} photos into moments...
          </p>
          {/* TODO: Animate photos grouping together here */}
        </div>
      </motion.section>

      {/* Stage 6: Ranking */}
      <motion.section style={styles.stage} variants={fadeInUp}>
        <div style={styles.stageInfo}>
          <div style={styles.stageNumber}>06</div>
          <h2 style={styles.stageTitle}>Ranking</h2>
          <p style={styles.stageDesc}>Selecting the "hero" shots from each moment.</p>
        </div>
        <div style={styles.stageContent}>
          <p style={{ color: '#666' }}>Simulating hero selection...</p>
          {/* TODO: Animate "pulling out" the top 5-10 photos */}
        </div>
      </motion.section>

      {/* Stage 7: Optimizer & Exporter */}
      <motion.section style={styles.stage} variants={fadeInUp}>
        <div style={styles.stageInfo}>
          <div style={styles.stageNumber}>07</div>
          <h2 style={styles.stageTitle}>Finalize</h2>
          <p style={styles.stageDesc}>
            Optimizing for diversity and exporting the final set.
          </p>
        </div>
        <div style={styles.stageContent}>
          {/* TODO: Add a button to navigate to the /placement page */}
          <p style={{ color: '#666' }}>Final set selected. Preparing for placement...</p>
        </div>
      </motion.section>
    </motion.div>
  );
}

export default PipelinePage;