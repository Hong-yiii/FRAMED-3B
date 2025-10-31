import React, { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
}

const styles = {
  layoutContainer: {
    width: '100%',
    minHeight: '100vh',
    position: 'relative' as const,
    overflowX: 'hidden' as const,
  },
  backgroundContainer: {
    position: 'fixed' as const,
    inset: 0,
    zIndex: 0,
  },
  backgroundImage: {
    width: '100%',
    height: '100%',
    objectFit: 'cover' as const,
  },
  backgroundOverlay: {
    position: 'absolute' as const,
    inset: 0,
    zIndex: 1,
    backgroundColor: 'var(--background)',
    opacity: 0.5,
  },
  content: {
    position: 'relative' as const,
    zIndex: 10,
    width: '100%',
  },
};

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div style={styles.layoutContainer}>
      <div style={styles.backgroundContainer}>
        <img
          src="https://images.unsplash.com/photo-1741308478108-9cfbf220e192?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsJTIwYmVpZ2UlMjBwYXBlciUyMHRleHR1cmV8ZW58MXx8fHwxNzYxODQyNDgwfDA&ixlib=rb-4.1.0&q=80&w=1080"
          alt="A minimal zine spread on a beige background"
          style={styles.backgroundImage}
        />
        <div style={styles.backgroundOverlay}></div>
      </div>

      <main style={styles.content}>
        {children}
      </main>
    </div>
  );
};
