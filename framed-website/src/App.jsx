import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import './App.css';
import Layout from './components/Layout.jsx'; // <-- 1. Import your new Layout

// Lazy load all our pages
const LandingPage = lazy(() => import('./pages/LandingPage.jsx'));
const StartPage = lazy(() => import('./pages/StartPage.jsx'));
const UploadPage = lazy(() => import('./pages/UploadPage.jsx'));
const PipelinePage = lazy(() => import('./pages/PipelinePage.jsx'));

function App() {
  return (
    <div className="App">
      {/* 2. Wrap EVERYTHING in the Layout component */}
      <Layout>
        <Suspense fallback={<div style={{ color: 'var(--foreground)' }}>Loading...</div>}>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/start" element={<StartPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/pipeline" element={<PipelinePage />} />
            {/* We'll add /placement and /magazine here next */}
          </Routes>
        </Suspense>
      </Layout>
    </div>
  );
}

export default App;