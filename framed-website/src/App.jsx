import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import './App.css';

// Lazy load the pages for better performance
const LandingPage = lazy(() => import('./pages/LandingPage.jsx'));
const PipelinePage = lazy(() => import('./pages/PipelinePage.jsx'));

function App() {
  return (
    <div className="App">
      {/* Suspense is required for lazy loading */}
      <Suspense fallback={<div style={{ color: '#fff' }}>Loading...</div>}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/pipeline" element={<PipelinePage />} />
          {/* Add more routes later for /placement and /magazine */}
        </Routes>
      </Suspense>
    </div>
  );
}

export default App;