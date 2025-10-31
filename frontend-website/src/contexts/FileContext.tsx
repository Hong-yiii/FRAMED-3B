import React, { createContext, useContext, useState, ReactNode } from 'react';

interface Photo {
  id: string;
  url: string;
  name: string;
}

interface Cluster {
  id: string;
  name: string;
  photos: Photo[];
  hero?: Photo;
}

interface FileContextType {
  photos: Photo[];
  setPhotos: (photos: Photo[]) => void;
  clusters: Cluster[];
  setClusters: (clusters: Cluster[]) => void;
  selectedPhotos: Photo[];
  setSelectedPhotos: (photos: Photo[]) => void;
  purpose: string;
  setPurpose: (purpose: string) => void;
  theme: string;
  setTheme: (theme: string) => void;
}

const FileContext = createContext<FileContextType | undefined>(undefined);

export const FileProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [selectedPhotos, setSelectedPhotos] = useState<Photo[]>([]);
  const [purpose, setPurpose] = useState<string>('');
  const [theme, setTheme] = useState<string>('');

  return (
    <FileContext.Provider
      value={{
        photos,
        setPhotos,
        clusters,
        setClusters,
        selectedPhotos,
        setSelectedPhotos,
        purpose,
        setPurpose,
        theme,
        setTheme,
      }}
    >
      {children}
    </FileContext.Provider>
  );
};

export const useFileContext = () => {
  const context = useContext(FileContext);
  if (context === undefined) {
    throw new Error('useFileContext must be used within a FileProvider');
  }
  return context;
};
