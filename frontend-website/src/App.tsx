import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { FileProvider } from './contexts/FileContext';
import { Layout } from './components/Layout';
import { LandingPage } from './pages/LandingPage';
import { StartPage } from './pages/StartPage';
import { ThemePage } from './pages/ThemePage';
import { UploadPage } from './pages/UploadPage';
import { DeduplicatePage } from './pages/DeduplicatePage';
import { PreprocessPage } from './pages/PreprocessPage';
import { ClusterPage } from './pages/ClusterPage';
import { RankingPage } from './pages/RankingPage';
import { FinalizePage } from './pages/FinalizePage';
import { PlacementPage } from './pages/PlacementPage';
import { MagazinePage } from './pages/MagazinePage';

export default function App() {
  return (
    <FileProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/start" element={<StartPage />} />
            <Route path="/theme" element={<ThemePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/deduplicate" element={<DeduplicatePage />} />
            <Route path="/preprocess" element={<PreprocessPage />} />
            <Route path="/cluster" element={<ClusterPage />} />
            <Route path="/ranking" element={<RankingPage />} />
            <Route path="/finalize" element={<FinalizePage />} />
            <Route path="/placement" element={<PlacementPage />} />
            <Route path="/magazine" element={<MagazinePage />} />
            {/* Catch-all route - redirect any unmatched paths to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </Router>
    </FileProvider>
  );
}
