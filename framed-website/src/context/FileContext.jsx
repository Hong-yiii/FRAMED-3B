import React, { createContext, useState, useContext } from 'react';

// 1. Create the context
const FileContext = createContext(null);

// 2. Create the provider component
export function FileProvider({ children }) {
  const [files, setFiles] = useState([]);

  return (
    <FileContext.Provider value={{ files, setFiles }}>
      {children}
    </FileContext.Provider>
  );
}

// 3. Create a custom hook to easily use the context
export function useFiles() {
  const context = useContext(FileContext);
  if (!context) {
    throw new Error('useFiles must be used within a FileProvider');
  }
  return context;
}