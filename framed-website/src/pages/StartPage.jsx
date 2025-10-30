import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

// Styles for this page
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
    maxWidth: '500px',
    marginLeft: 'auto',   // <-- ADD THIS
    marginRight: 'auto',
  },
  optionsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1.5rem',
    width: '100%',
    maxWidth: '900px',
  },
  optionCard: {
    backgroundColor: 'var(--card)',
    border: '1px solid var(--border)',
    borderRadius: '12px',
    padding: '2rem',
    textAlign: 'centre',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  optionTitle: {
    fontFamily: '"Source Serif 4", serif',
    fontSize: '1.5rem',
    fontWeight: '600',
    color: 'var(--foreground)',
  },
  optionDesc: {
    color: 'var(--muted-foreground)',
    marginTop: '0.5rem',
  }
};

// Framer Motion variants
const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: 'easeOut' },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const purposes = [
  { id: "event", name: "An Event", subtitle: "Weddings, birthdays, milestones" },
  { id: "travel", name: "A Trip", subtitle: "Travel stories, adventures, guides" },
  { id: "gifting", name: "A Gift", subtitle: "Something thoughtful for someone" },
];

function StartPage() {
  const navigate = useNavigate();

  const handleSelectPurpose = (purposeId) => {
    console.log('Purpose selected:', purposeId);
    // After selecting, we go to the new /upload page
    navigate('/upload');
  };

  return (
    <motion.div 
      style={styles.pageContainer}
      initial="initial" 
      animate="animate" 
      variants={staggerContainer}
    >
      <motion.div style={styles.header} variants={fadeInUp}>
        <h1 style={styles.title}>What are you making this zine for?</h1>
        <p style={styles.subtitle}>
          This helps us recommend the best layouts for your story.
        </p>
      </motion.div>

      <motion.div 
        style={styles.optionsGrid} 
        variants={staggerContainer}
      >
        {purposes.map((purpose) => (
          <motion.div
            key={purpose.id}
            style={styles.optionCard}
            variants={fadeInUp}
            onClick={() => handleSelectPurpose(purpose.id)}
            whileHover={{ 
              transform: 'translateY(-5px)',
              boxShadow: '0 10px 20px rgba(0,0,0,0.05)'
            }}
          >
            <h3 style={styles.optionTitle}>{purpose.name}</h3>
            <p style={styles.optionDesc}>{purpose.subtitle}</p>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
}

export default StartPage;