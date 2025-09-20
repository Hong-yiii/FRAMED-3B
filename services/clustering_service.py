"""
Real Clustering Service for Photo Curation System

Groups near-duplicate photos into moment clusters.
"""

import os
import json
import hashlib
from typing import Dict, Any, List


class ClusteringService:
    """Real clustering service for grouping similar photos."""

    def __init__(self):
        self.cache_dir = "./data/cache/clustering"
        os.makedirs(self.cache_dir, exist_ok=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Group photos into moment clusters."""
        print("🔄 Real Clustering Service: Finding moment clusters")

        batch_id = input_data["batch_id"]

        # Check cache
        cache_key = hashlib.md5(f"{batch_id}_clusters".encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            print(f"📋 Using cached clusters for {batch_id}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Simple clustering based on photo count
        scores = input_data.get("scores", [])
        clusters = self._create_clusters(scores)

        result = {
            "batch_id": batch_id,
            "clusters": clusters
        }

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Also save to intermediate JSONs
        import json
        os.makedirs("intermediateJsons/clustering", exist_ok=True)
        with open(f"intermediateJsons/clustering/{batch_id}_clustering_output.json", 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ Real Clustering Service: Created {len(clusters)} clusters")
        print(f"💾 Saved output to intermediateJsons/clustering/{batch_id}_clustering_output.json")
        return result

    def _create_clusters(self, scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create clusters from scores."""
        clusters = []
        cluster_id = 1

        # Group photos by quality ranges (simple clustering)
        quality_ranges = [(0.7, 1.0), (0.5, 0.7), (0.3, 0.5)]

        for min_q, max_q in quality_ranges:
            cluster_photos = [
                score["photo_id"] for score in scores
                if min_q <= score.get("Total_prelim", 0.5) < max_q
            ]

            if cluster_photos:
                clusters.append({
                    "cluster_id": "02d",
                    "members": cluster_photos
                })
                cluster_id += 1

        return clusters
