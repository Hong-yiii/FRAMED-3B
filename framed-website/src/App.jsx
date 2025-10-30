import React from 'react';
import './App.css'; // Keep this for any future app-wide styles
import LandingPage from './pages/LandingPage';

function App() {
  // For now, the App component just renders the LandingPage.
  // Later, we will add routing here to move between pipeline stages.
  return (
    <div className="App">
      <LandingPage />
    </div>
  );
}

export default App;