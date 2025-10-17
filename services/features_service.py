"""
Real Features Service for Photo Curation System

Extracts features from actual photo files using OpenCLIP and technical quality metrics.
"""

import os
import json
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import cv2
from typing import Dict, Any, List, Optional, Union
import numpy.typing as npt
from PIL import Image
from skimage import filters, measure, exposure
from skimage.restoration import estimate_sigma
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPClassifier:
    """OpenCLIP-based image classifier optimized for MacBook Pro."""
    
    def __init__(self, 
                 model_id: str = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
                 config_dir: str = "./config",
                 cache_dir: str = "./data/cache/features"):
        
        # Setup logging for CLIPClassifier
        self.logger = logging.getLogger('CLIPClassifier')
        self.logger.setLevel(logging.DEBUG)
        
        self.model_id = model_id
        self.config_dir = config_dir
        self.cache_dir = cache_dir
        
        # Device selection: prefer MPS on Mac, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info("Using CUDA for acceleration")
        else:
            self.device = "cpu"
            self.logger.info("Using CPU (no GPU acceleration available)")
        
        # Initialize model components (will be set in _load_model and _load_config)
        self.model: Optional[torch.nn.Module] = None
        self.preprocess: Optional[Any] = None  # open_clip transform
        self.tokenizer: Optional[Any] = None   # open_clip tokenizer
        self.label_embeddings: Optional[torch.Tensor] = None
        self.labels: List[str] = []
        self.templates: List[str] = []
        
        # Load model and configurations
        self._load_model()
        self._load_config()
        self._build_label_embeddings()
    
    def _load_model(self):
        """Load OpenCLIP model and preprocessing."""
        try:
            self.logger.info(f"Loading OpenCLIP model: {self.model_id}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_id, 
                device=self.device
            )
            self.model.eval()
            
            # Use half precision on GPU for memory efficiency
            if self.device in ["cuda", "mps"]:
                self.model = self.model.half()
                self.logger.debug("Using half precision (FP16) for GPU acceleration")
            
            self.tokenizer = open_clip.get_tokenizer(self.model_id)
            self.logger.info("OpenCLIP model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load OpenCLIP model: {e}")
            raise
    
    def _load_config(self):
        """Load labels and templates from config files."""
        try:
            # Load labels
            labels_path = os.path.join(self.config_dir, "labels.json")
            with open(labels_path, 'r') as f:
                config = json.load(f)
                self.labels = config["labels"]
            
            # Load templates
            templates_path = os.path.join(self.config_dir, "templates.json")
            with open(templates_path, 'r') as f:
                config = json.load(f)
                self.templates = config["templates"]
            
            self.logger.info(f"Loaded {len(self.labels)} labels and {len(self.templates)} templates")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    @torch.inference_mode()
    def _build_label_embeddings(self):
        """Build and cache label embeddings."""
        cache_file = os.path.join(self.cache_dir, "label_embeddings.pt")
        
        # Check if cached embeddings exist and are valid
        if os.path.exists(cache_file):
            try:
                cached_data = torch.load(cache_file, map_location=self.device)
                if (cached_data["labels"] == self.labels and 
                    cached_data["templates"] == self.templates and
                    cached_data["model_id"] == self.model_id):
                    
                    self.label_embeddings = cached_data["embeddings"]
                    self.logger.info("Loaded cached label embeddings")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Build new embeddings
        self.logger.info("Building label embeddings...")
        embeddings = []
        
        for label in self.labels:
            # Create prompts for this label using all templates
            prompts = [template.format(label) for template in self.templates]
            
            # Tokenize prompts (assert tokenizer is not None after _load_model)
            assert self.tokenizer is not None, "Tokenizer not initialized"
            tokens = self.tokenizer(prompts).to(self.device)
            
            # Get text embeddings (assert model is not None after _load_model)
            assert self.model is not None, "Model not initialized"
            text_embeddings = self.model.encode_text(tokens)  # type: ignore
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            
            # Average across templates
            label_embedding = text_embeddings.mean(dim=0, keepdim=True)
            embeddings.append(label_embedding)
        
        # Stack and normalize
        self.label_embeddings = F.normalize(torch.cat(embeddings, dim=0), dim=-1)
        
        # Cache the embeddings
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save({
            "embeddings": self.label_embeddings,
            "labels": self.labels,
            "templates": self.templates,
            "model_id": self.model_id
        }, cache_file)
        
        self.logger.info(f"Built and cached embeddings for {len(self.labels)} labels")
    
    @torch.inference_mode()
    def classify_image(self, image_path: str, top_k: int = 5, temperature: float = 0.01) -> List[Dict[str, Any]]:
        """Classify an image and return top-k labels with scores."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Assert preprocess is not None after _load_model
            assert self.preprocess is not None, "Preprocess transform not initialized"
            # Type cast to torch.Tensor to satisfy linter
            preprocessed = self.preprocess(image)
            image_tensor = torch.as_tensor(preprocessed).unsqueeze(0).to(self.device)
            
            if self.device in ["cuda", "mps"]:
                image_tensor = image_tensor.half()
            
            # Get image embedding (assert model is not None after _load_model)
            assert self.model is not None, "Model not initialized"
            image_embedding = self.model.encode_image(image_tensor)  # type: ignore
            image_embedding = F.normalize(image_embedding, dim=-1)
            
            # Compute similarities (assert label_embeddings is not None after _build_label_embeddings)
            assert self.label_embeddings is not None, "Label embeddings not initialized"
            similarities = (image_embedding @ self.label_embeddings.T).squeeze(0)
            
            # Apply temperature and get probabilities
            probabilities = F.softmax(similarities / temperature, dim=0)
            
            # Get top-k results
            k = min(top_k, len(self.labels))
            top_probs, top_indices = torch.topk(probabilities, k)
            
            results = []
            for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
                results.append({
                    "label": self.labels[idx],
                    "probability": float(prob),
                    "score": float(similarities[idx].cpu())
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to classify image {image_path}: {e}")
            # Return default labels on error
            return [
                {"label": "photography", "probability": 0.2, "score": 0.1},
                {"label": "landscape", "probability": 0.2, "score": 0.1},
                {"label": "nature", "probability": 0.2, "score": 0.1},
                {"label": "outdoor", "probability": 0.2, "score": 0.1},
                {"label": "scenic", "probability": 0.2, "score": 0.1}
            ]


class TechnicalQualityAnalyzer:
    """Analyzes technical quality metrics of images using optimized algorithms."""

    # Calibration constants for quality metrics
    TENENGRAD_K = 2000.0  # Tenengrad sharpness calibration
    NOISE_SIGMA_MAX = 0.08  # Maximum sigma for noise estimation

    def __init__(self):
        # Enable OpenCV optimizations and set thread count to avoid oversubscription
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)

    @staticmethod
    def _get_gray_preview(image_path: str, max_side: int = 512) -> Optional[np.ndarray]:
        """Get downscaled grayscale preview for efficient analysis."""
        try:
            # Read grayscale image once
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None

            # Resize if needed (keep aspect ratio)
            h, w = gray.shape
            if h > max_side or w > max_side:
                if h > w:
                    new_h, new_w = max_side, int(max_side * w / h)
                else:
                    new_h, new_w = int(max_side * h / w), max_side
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return gray

        except Exception as e:
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to get gray preview for {image_path}: {e}")
            return None

    @staticmethod
    def analyze_sharpness(image_path: str) -> float:
        """Calculate sharpness using Tenengrad (Sobel energy) method."""
        try:
            analyzer = TechnicalQualityAnalyzer()
            gray = analyzer._get_gray_preview(image_path)
            if gray is None:
                return 0.5

            # Compute Sobel gradients
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Tenengrad energy (mean of squared gradients)
            g2 = np.mean(gx*gx + gy*gy)

            # Smooth squashing with tanh
            score = np.tanh(g2 / TechnicalQualityAnalyzer.TENENGRAD_K)
            return float(score)

        except Exception as e:
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to analyze sharpness for {image_path}: {e}")
            return 0.5

    @staticmethod
    def analyze_exposure(image_path: str) -> float:
        """Analyze exposure quality using percentiles and clipping."""
        try:
            analyzer = TechnicalQualityAnalyzer()
            gray = analyzer._get_gray_preview(image_path)
            if gray is None:
                return 0.5

            # Convert to float [0,1]
            vals = gray.astype(np.float32) / 255.0

            # Get percentiles
            p1, p50, p99 = np.percentile(vals, [1, 50, 99])

            # Midtone proximity (closer to 0.5 is better)
            mid = 1.0 - min(abs(p50 - 0.5) / 0.5, 1.0)

            # Dynamic range (wider range is better, normalized to [0,1])
            dr = np.clip((p99 - p1) / 0.9, 0.0, 1.0)

            # Clipping penalty (fraction of pixels in extreme ranges)
            clip_frac = np.mean(vals < 0.02) + np.mean(vals > 0.98)

            # Combine scores
            score = np.clip(0.6 * mid + 0.3 * dr - 0.6 * clip_frac, 0.0, 1.0)
            return float(score)

        except Exception as e:
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to analyze exposure for {image_path}: {e}")
            return 0.5

    @staticmethod
    def analyze_noise(image_path: str) -> float:
        """Analyze noise level using wavelet-based sigma estimation."""
        try:
            analyzer = TechnicalQualityAnalyzer()
            gray = analyzer._get_gray_preview(image_path)
            if gray is None:
                return 0.5

            # Convert to float [0,1]
            gray_float = gray.astype(np.float32) / 255.0

            # Estimate sigma using wavelet method
            sigma_result = estimate_sigma(gray_float, channel_axis=None, average_sigmas=True)
            sigma_array = np.asarray(sigma_result)
            sigma = float(np.mean(sigma_array))

            # Map to [0,1] score (1 = clean, 0 = very noisy)
            score = 1.0 - np.clip(sigma / TechnicalQualityAnalyzer.NOISE_SIGMA_MAX, 0.0, 1.0)
            return float(score)

        except Exception as e:
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to analyze noise for {image_path}: {e}")
            return 0.5



class FeaturesService:
    """Real features service that extracts features from actual photos."""

    def __init__(self):
        # Setup logging similar to ingest service
        self.logger = logging.getLogger('FeaturesService')
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = "intermediateJsons/features"
        os.makedirs(log_dir, exist_ok=True)

        # File handler - logs everything
        file_handler = logging.FileHandler(os.path.join(log_dir, 'features_service.log'))
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler - only errors and important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.cache_dir = "./data/cache/features"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize CLIP classifier
        self.logger.info("Initializing OpenCLIP classifier...")
        self.clip_classifier = CLIPClassifier(cache_dir=self.cache_dir)
        
        # Initialize technical quality analyzer
        self.quality_analyzer = TechnicalQualityAnalyzer()

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process preprocessed photos and extract features."""
        import time
        start_time = time.time()
        
        batch_id = input_data["batch_id"]
        self.logger.info(f"Starting features extraction for batch: {batch_id}")
        print("🧠 Extracting features...")

        artifacts = []
        total_photos = 0
        processed_count = 0
        cache_hits = 0
        error_count = 0

        # Handle both preprocess artifacts and photo_index formats
        if "artifacts" in input_data:
            # From preprocess service (preferred - has standardized images)
            source_artifacts = input_data["artifacts"]
            total_photos = len(source_artifacts)
            self.logger.info(f"Processing {total_photos} artifacts from preprocess service")
        elif "photo_index" in input_data:
            # Direct processing from ingest output (fallback)
            total_photos = len(input_data['photo_index'])
            self.logger.info(f"Processing {total_photos} photos directly from ingest output")
            source_artifacts = []
            for photo in input_data["photo_index"]:
                # Preserve all metadata from ingest output
                artifact = {
                    "photo_id": photo["photo_id"],
                    "std_uri": photo.get("ranking_uri", photo.get("original_uri", "")),
                    "original_uri": photo.get("original_uri", ""),
                    "ranking_uri": photo.get("ranking_uri", ""),
                    "exif": photo.get("exif", {}),
                    "format": photo.get("format", "")
                }
                source_artifacts.append(artifact)
        else:
            self.logger.error("No artifacts or photo_index found in input")
            return {"batch_id": batch_id, "artifacts": []}

        print(f"📊 Processing {total_photos} photos...")

        for i, artifact in enumerate(source_artifacts):
            photo_id = artifact["photo_id"]
            # Use ranking_uri if available (from ingest), otherwise std_uri (from preprocess)
            image_uri = artifact.get("ranking_uri", artifact.get("std_uri", artifact.get("uri", "")))

            # Update progress bar
            progress = f"[{processed_count + error_count + 1}/{total_photos}]"
            filename = os.path.basename(image_uri) if image_uri else photo_id[:8]
            print(f"\r🔄 {progress} Extracting features... {filename}", end="", flush=True)

            try:
                # Check cache first
                cache_key = hashlib.md5(f"{photo_id}_features".encode()).hexdigest()
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

                if os.path.exists(cache_file):
                    try:
                        # Check if cache file is not empty
                        if os.path.getsize(cache_file) > 0:
                            cache_hits += 1
                            self.logger.debug(f"Cache hit for {photo_id[:8]}")
                            with open(cache_file, 'r') as f:
                                cached_features = json.load(f)
                                # Merge cached features with current artifact (preserving all metadata)
                                complete_artifact = artifact.copy()
                                complete_artifact["features"] = cached_features.get("features", {})
                                artifacts.append(complete_artifact)
                            processed_count += 1
                            continue
                        else:
                            # Remove empty cache file
                            os.remove(cache_file)
                            self.logger.warning(f"Removed empty cache file: {cache_file}")
                    except (json.JSONDecodeError, OSError) as e:
                        # Remove corrupted cache file
                        try:
                            os.remove(cache_file)
                            self.logger.warning(f"Removed corrupted cache file {cache_file}: {e}")
                        except OSError:
                            pass

                # Extract features from image (with EXIF context)
                exif_data = artifact.get("exif", {})
                features = self._extract_features(image_uri, photo_id, exif_data)

                # Create complete artifact preserving ALL original metadata and adding features
                complete_artifact = artifact.copy()  # Preserves photo_id, original_uri, ranking_uri, exif, format, etc.
                complete_artifact["features"] = features

                artifacts.append(complete_artifact)

                # Cache only the features (not the full artifact to avoid duplication)
                cache_data = {"features": features}
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                
                processed_count += 1
                self.logger.debug(f"Successfully processed features for {photo_id[:8]}")

            except Exception as e:
                print(f"\r❌ [{processed_count + error_count + 1}/{total_photos}] Error: {filename}")
                self.logger.error(f"Error processing features for {photo_id}: {e}")
                error_count += 1
                continue

        # Clear progress line and show final status
        print(f"\r✅ Processed {processed_count}/{total_photos} photos successfully")

        result = {
            "batch_id": batch_id,
            "artifacts": artifacts
        }

        # Save to intermediate JSONs
        os.makedirs("intermediateJsons/features", exist_ok=True)
        with open(f"intermediateJsons/features/{batch_id}_features_output.json", 'w') as f:
            json.dump(result, f, indent=2)

        # Calculate and display timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        timing_msg = f"Features extraction complete: {processed_count}/{total_photos} photos processed in {elapsed_time:.2f}s"
        
        # Log comprehensive statistics
        self.logger.info(f"Batch {batch_id} processing complete:")
        self.logger.info(f"  - Total photos: {total_photos}")
        self.logger.info(f"  - Successfully processed: {processed_count}")
        self.logger.info(f"  - Cache hits: {cache_hits}")
        self.logger.info(f"  - Errors: {error_count}")
        self.logger.info(f"  - Processing time: {elapsed_time:.2f}s")
        self.logger.info(f"  - Average time per photo: {elapsed_time/max(processed_count, 1):.2f}s")
        
        print(f"🧠 {timing_msg}")
        if cache_hits > 0:
            print(f"📋 Cache hits: {cache_hits}/{total_photos} ({cache_hits/total_photos*100:.1f}%)")
        if error_count > 0:
            print(f"⚠️  Errors: {error_count}/{total_photos}")
            
        return result

    def _extract_features(self, photo_uri: str, photo_id: str, exif_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract features from a single photo using OpenCLIP and technical analysis."""
        try:
            self.logger.debug(f"Extracting features for {photo_id[:8]}...")
            
            # Get CLIP labels with confidence scores
            self.logger.debug(f"Running CLIP classification for {photo_id[:8]}")
            clip_results = self.clip_classifier.classify_image(photo_uri, top_k=5)
            
            # Format labels with confidence: keep label and probability, rename score to confidence for clarity
            clip_labels = [
                {
                    "label": result["label"],
                    "confidence": result["probability"],  # Temperature-scaled probability
                    "cosine_score": result["score"]      # Raw cosine similarity score
                }
                for result in clip_results
            ]
            
            # Get technical quality metrics
            self.logger.debug(f"Analyzing technical quality for {photo_id[:8]}")
            tech_features = {
                "sharpness": self.quality_analyzer.analyze_sharpness(photo_uri),
                "exposure": self.quality_analyzer.analyze_exposure(photo_uri),
                "noise": self.quality_analyzer.analyze_noise(photo_uri)
            }
            
            features = {
                "tech": tech_features,
                "clip_labels": clip_labels
            }
            
            # Log camera info if available
            camera_info = ""
            if exif_data and exif_data.get("camera"):
                camera_info = f" ({exif_data['camera']})"
            
            # Extract just the label names for logging
            label_names = [label["label"] for label in clip_labels]
            self.logger.debug(f"Features extracted successfully for {photo_id[:8]}{camera_info}: {label_names[:3]}...")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting features from {photo_uri}: {e}")
            # Return minimal features on error with confidence 0.2 (fallback)
            return {
                "tech": {"sharpness": 0.5, "exposure": 0.5, "noise": 0.5},
                "clip_labels": [
                    {"label": "photography", "confidence": 0.2, "cosine_score": 0.1},
                    {"label": "landscape", "confidence": 0.2, "cosine_score": 0.1},
                    {"label": "nature", "confidence": 0.2, "cosine_score": 0.1},
                    {"label": "outdoor", "confidence": 0.2, "cosine_score": 0.1},
                    {"label": "scenic", "confidence": 0.2, "cosine_score": 0.1}
                ]
            }
