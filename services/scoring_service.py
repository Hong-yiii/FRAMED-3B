"""
Real Scoring Service for Photo Curation System

Computes quality scores for actual photos.
"""

import os
import json
import hashlib
from typing import Dict, Any


class ScoringService:
    """Real scoring service that computes quality scores."""

    def __init__(self):
        self.cache_dir = "./data/cache/scoring"
        os.makedirs(self.cache_dir, exist_ok=True)

    def process(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process photos and compute quality scores."""
        print("🔄 Real Scoring Service: Computing quality scores")

        batch_id = input_data["batch_id"]
        scores = []
        dropped = []

        # Process each artifact
        if "artifacts" in input_data:
            artifacts = input_data["artifacts"]
        else:
            # Handle case where we don't have artifacts yet
            artifacts = [{"photo_id": "dummy", "features": {}}]

        for artifact in artifacts:
            photo_id = artifact["photo_id"]
            features = artifact.get("features", {})

            # Check cache
            cache_key = hashlib.md5(f"{photo_id}_scores".encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

            if os.path.exists(cache_file):
                print(f"📋 Using cached scores for {photo_id[:8]}...")
                with open(cache_file, 'r') as f:
                    cached_scores = json.load(f)
                    scores.append(cached_scores)
                continue

            # Compute scores
            photo_scores = self._compute_scores(features)

            # Technical quality gate
            if photo_scores["Q_tech"] < 0.3:
                dropped.append(photo_id)
                continue

            scores.append({
                "photo_id": photo_id,
                **photo_scores
            })

            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(scores[-1], f, indent=2)

            print(f"✅ Scored {photo_id[:8]}... (Q_tech: {photo_scores['Q_tech']:.2f})")

        result = {
            "batch_id": batch_id,
            "scores": scores,
            "dropped_for_tech": dropped
        }

        # Save to intermediate JSONs
        import json
        os.makedirs("intermediateJsons/scoring", exist_ok=True)
        with open(f"intermediateJsons/scoring/{batch_id}_scoring_output.json", 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ Real Scoring Service: Scored {len(scores)} photos, dropped {len(dropped)}")
        print(f"💾 Saved output to intermediateJsons/scoring/{batch_id}_scoring_output.json")
        return result

    def _compute_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Compute quality scores from features."""
        tech = features.get("tech", {})
        saliency = features.get("saliency", {})

        # Technical quality score
        q_tech = 0.4 * tech.get("sharpness", 0.5) + \
                 0.4 * tech.get("exposure", 0.5) + \
                 0.2 * (1 - tech.get("noise", 0.5))

        return {
            "Q_tech": q_tech,
            "Aesthetic": 0.6 + 0.3 * (q_tech + saliency.get("neg_space_ratio", 0.3)),
            "Vibe": 0.5 + 0.4 * saliency.get("neg_space_ratio", 0.3),
            "Typography": saliency.get("neg_space_ratio", 0.3) * 0.8 + 0.2,
            "Composition": 0.6 + 0.3 * q_tech,
            "Total_prelim": 0.7 + 0.2 * q_tech
        }
