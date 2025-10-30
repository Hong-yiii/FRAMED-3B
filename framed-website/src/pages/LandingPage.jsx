import React from 'react';
// We no longer need to import the background image!

// Styles for the shared layout
const styles = {
  layoutContainer: {
    width: '100%',
    minHeight: '100vh',
    position: 'relative',
    overflowX: 'hidden', // Prevent horizontal scroll
  },
  backgroundContainer: {
    position: 'fixed', // Use 'fixed' to keep it static on all pages
    inset: 0, // 'inset: 0' is shortcut for top/left/right/bottom: 0
    zIndex: 0, // Behind everything
  },
  backgroundImage: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  backgroundOverlay: {
    position: 'absolute',
    inset: 0,
    zIndex: 1,
    backgroundColor: 'var(--background)',
    opacity: 0.5,
  },
  content: {
    // This will hold the actual page content (e.g., LandingPage, StartPage)
    position: 'relative', // 'relative' so it sits on top of the 'fixed' background
    zIndex: 10,
    width: '100%',
  }
};

function Layout({ children }) {
  return (
    <div style={styles.layoutContainer}>
      
      <div style={styles.backgroundContainer}>
        <img
          // Since the image is in `public/`, we just use the root path
          src="/hero-background.jpg" 
          alt="A minimal zine spread on a beige background"
          style={styles.backgroundImage}
        />
        <div style={styles.backgroundOverlay}></div> 
      </div>

      {/* The active page (e.g., LandingPage, StartPage) renders here */}
      <main style={styles.content}>
        {children}
      </main>
      
    </div>
  );
}

export default Layout;