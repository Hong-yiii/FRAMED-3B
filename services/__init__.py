"""
Real Services Package for Photo Curation System

Contains the actual implementations for processing real photo data.
"""

import os
import sys
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ingest_service import IngestService
from .preprocess_service import PreprocessService
from .features_service import FeaturesService
from .scoring_service import ScoringService
from .clustering_service import ClusteringService
from .ranking_service import RankingService
from .optimizer_service import OptimizerService
from .exporter_service import ExporterService

class RealServices:
    """Real services that process actual photo data."""

    def __init__(self):
        self.ingest = IngestService()
        self.preprocess = PreprocessService()
        self.features = FeaturesService()
        self.scoring = ScoringService()
        self.clustering = ClusteringService()
        self.ranking = RankingService()
        self.optimizer = OptimizerService()
        self.exporter = ExporterService()

    def ingest_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.ingest.process(input_data)

    def preprocess_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.preprocess.process(input_data)

    def process_features_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.features.process(input_data)

    def scoring_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.scoring.process(input_data, theme_spec)

    def clustering_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.clustering.process(input_data)

    def cluster_ranking_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.ranking.process(input_data, theme_spec)

    def optimizer_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.optimizer.process(input_data, theme_spec)

    def exporter_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.exporter.process(input_data, theme_spec)
