import React from 'react'; // Removed useEffect, useState
import { motion } from 'framer-motion';
import { useFiles } from '../context/FileContext'; // <-- Import our global hook

// Styles (No change)
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
    color: 'var(--muted-foreground)',
    fontFamily: '"Source Serif 4", serif',
  },
  stageTitle: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: 'var(--foreground)',
    marginTop: '0.5rem',
    fontFamily: '"Source Serif 4", serif',
  },
  stageDesc: {
    color: 'var(--muted-foreground)',
    marginTop: '0.5rem',
  },
  stageContent: {
    width: '70%',
    paddingLeft: '2rem',
    borderLeft: '2px solid var(--border)',
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
    backgroundColor: 'var(--secondary)',
  },
};

// Framer Motion variants
const fadeInUp = {
  initial: { opacity: 0, y: 40 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: 'easeOut' },
};

// --- *** THIS IS THE CORRECTED COMPONENT *** ---
// It's simpler and works with our new `presetFiles` array
function FilePreview({ file }) {
  return (
    <motion.img
      src={file.path} // <-- It now reads file.path
      alt={file.name}
      style={styles.thumbnail}
      variants={fadeInUp} // Each thumbnail fades in
      layout // This will animate it when it moves!
    />
  );
}
// --- END CORRECTION ---

// Main Page Component
function PipelinePage() {
  const { files } = useFiles();

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
          <p style={{ color: 'var(--muted-foreground)' }}>Ingest complete. Starting preprocess...</p>
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
          {/* This will now render your demo photos */}
          {files.map((file, index) => (
            <FilePreview key={file.name + index} file={file} />
          ))}
        </motion.div>
      </motion.section>

      {/* --- SIMULATED STAGES (Wizard of Oz) --- */}

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
          <p style={{ color: 'var(--muted-foreground)' }}>
            Simulating grouping {files.length} photos into moments...
          </p>
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
          <p style={{ color: 'var(--muted-foreground)' }}>Simulating hero selection...</p>
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
          <p style={{ color: 'var(--muted-foreground)' }}>Final set selected. Preparing for placement...</p>
        </div>
      </motion.section>
    </motion.div>
  );
}

export default PipelinePage;