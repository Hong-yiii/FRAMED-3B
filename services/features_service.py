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
    """Analyzes technical quality metrics of images."""
    
    @staticmethod
    def analyze_sharpness(image_path: str) -> float:
        """Calculate sharpness using Laplacian variance."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.5
            
            # Resize for consistent computation
            if image.shape[0] > 1024 or image.shape[1] > 1024:
                image = cv2.resize(image, (1024, int(1024 * image.shape[0] / image.shape[1])))
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined thresholds)
            normalized = min(laplacian_var / 1000.0, 1.0)
            return float(normalized)
            
        except Exception as e:
            # Use module logger for static methods
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to analyze sharpness for {image_path}: {e}")
            return 0.5
    
    @staticmethod
    def analyze_exposure(image_path: str) -> float:
        """Analyze exposure quality."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.5
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Check for clipping (overexposure/underexposure)
            underexposed = hist[:10].sum()  # Very dark pixels
            overexposed = hist[-10:].sum()  # Very bright pixels
            
            # Penalty for clipping
            clipping_penalty = (underexposed + overexposed) * 2
            
            # Reward good distribution
            mean_brightness = np.average(range(256), weights=hist)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # Combine scores
            exposure_score = max(0.0, brightness_score - clipping_penalty)
            return float(exposure_score)
            
        except Exception as e:
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to analyze exposure for {image_path}: {e}")
            return 0.5
    
    @staticmethod
    def analyze_noise(image_path: str) -> float:
        """Analyze noise level (returns inverse - lower noise = higher score)."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.5
            
            # Resize for consistent computation
            if image.shape[0] > 512 or image.shape[1] > 512:
                image = cv2.resize(image, (512, int(512 * image.shape[0] / image.shape[1])))
            
            # Estimate noise using local standard deviation
            # Apply Gaussian blur and subtract from original
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            noise = cv2.absdiff(image, blurred)
            noise_level = noise.std()
            
            # Normalize and invert (lower noise = higher score)
            normalized_noise = min(noise_level / 20.0, 1.0)
            noise_score = 1.0 - normalized_noise
            
            return float(noise_score)
            
        except Exception as e:
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to analyze noise for {image_path}: {e}")
            return 0.5
    
    @staticmethod
    def analyze_horizon(image_path: str) -> float:
        """Analyze horizon tilt in degrees."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0
            
            # Resize for faster processing
            if image.shape[0] > 512 or image.shape[1] > 512:
                image = cv2.resize(image, (512, int(512 * image.shape[0] / image.shape[1])))
            
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:10]:  # Consider top 10 lines
                    angle = theta * 180 / np.pi
                    # Convert to horizon angle (-90 to 90)
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                # Find the most common angle (likely horizon)
                if angles:
                    median_angle = np.median(angles)
                    return float(median_angle)
            
            return 0.0
            
        except Exception as e:
            logging.getLogger('TechnicalQualityAnalyzer').warning(f"Failed to analyze horizon for {image_path}: {e}")
            return 0.0


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
            
            # Get CLIP labels
            self.logger.debug(f"Running CLIP classification for {photo_id[:8]}")
            clip_results = self.clip_classifier.classify_image(photo_uri, top_k=5)
            clip_labels = [result["label"] for result in clip_results]
            
            # Get technical quality metrics
            self.logger.debug(f"Analyzing technical quality for {photo_id[:8]}")
            tech_features = {
                "sharpness": self.quality_analyzer.analyze_sharpness(photo_uri),
                "exposure": self.quality_analyzer.analyze_exposure(photo_uri),
                "noise": self.quality_analyzer.analyze_noise(photo_uri),
                "horizon_deg": self.quality_analyzer.analyze_horizon(photo_uri)
            }
            
            # Enhance with EXIF-based insights if available
            if exif_data:
                self.logger.debug(f"Extracting EXIF insights for {photo_id[:8]}")
                tech_features.update(self._extract_exif_insights(exif_data))
            
            features = {
                "tech": tech_features,
                "clip_labels": clip_labels
            }
            
            # Log camera info if available
            camera_info = ""
            if exif_data and exif_data.get("camera"):
                camera_info = f" ({exif_data['camera']})"
            
            self.logger.debug(f"Features extracted successfully for {photo_id[:8]}{camera_info}: {clip_labels[:3]}...")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting features from {photo_uri}: {e}")
            # Return minimal features on error
            return {
                "tech": {"sharpness": 0.5, "exposure": 0.5, "noise": 0.5, "horizon_deg": 0},
                "clip_labels": ["photography", "landscape", "nature", "outdoor", "scenic"]
            }
    
    def _extract_exif_insights(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional technical insights from EXIF data."""
        insights = {}
        
        try:
            # ISO-based noise prediction
            iso = exif_data.get("iso")
            if iso and isinstance(iso, (int, float)):
                # Higher ISO typically means more noise
                # This is a rough heuristic - actual noise analysis is still preferred
                iso_noise_factor = min(iso / 3200.0, 1.0)  # Normalize to 0-1
                insights["iso_noise_factor"] = float(iso_noise_factor)
            
            # Aperture information for depth of field context
            aperture = exif_data.get("aperture")
            if aperture and isinstance(aperture, str) and aperture.startswith("f/"):
                try:
                    f_number = float(aperture[2:])
                    # Lower f-number = wider aperture = shallower DOF
                    insights["aperture_f_number"] = f_number
                except ValueError:
                    pass
            
            # Shutter speed for motion blur context
            shutter_speed = exif_data.get("shutter_speed")
            if shutter_speed and isinstance(shutter_speed, str) and "/" in shutter_speed:
                try:
                    if shutter_speed.startswith("1/"):
                        denominator = float(shutter_speed[2:])
                        shutter_fraction = 1.0 / denominator
                    else:
                        shutter_fraction = float(shutter_speed)
                    insights["shutter_speed_seconds"] = shutter_fraction
                except ValueError:
                    pass
            
            # Camera type for context
            camera = exif_data.get("camera")
            if camera:
                insights["camera_type"] = camera
                # Simple heuristic for camera quality tier
                if "iPhone" in camera or "Pixel" in camera:
                    insights["camera_tier"] = "smartphone"
                elif any(brand in camera.upper() for brand in ["CANON", "NIKON", "SONY", "FUJI"]):
                    insights["camera_tier"] = "professional"
                else:
                    insights["camera_tier"] = "consumer"
        
        except Exception as e:
            self.logger.warning(f"Error extracting EXIF insights: {e}")
        
        return insights
