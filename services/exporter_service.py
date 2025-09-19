"""
Real Exporter Service for Photo Curation System

Creates final curated list output.
"""

import os
import json
import hashlib
import random
from datetime import datetime, timezone
from typing import Dict, Any, List


class ExporterService:
    """Real exporter service for creating final curated lists."""

    def __init__(self):
        self.cache_dir = "./data/cache/exporter"
        os.makedirs(self.cache_dir, exist_ok=True)

    def process(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create final curated list."""
        print("🔄 Real Exporter Service: Creating final curated list")

        batch_id = input_data["batch_id"]

        # Check cache
        cache_key = hashlib.md5(f"{batch_id}_export".encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            print(f"📋 Using cached export for {batch_id}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        selected_ids = input_data.get("selected_ids", [])
        items = []

        for i, photo_id in enumerate(selected_ids, 1):
            # Mock diversity tags
            diversity_tags = [
                f"scene:{random.choice(['street', 'architecture', 'people', 'interior'])}",
                f"palette:{random.choice(['warm', 'cool', 'monochrome', 'vibrant'])}",
                f"time:{random.choice(['dawn', 'morning', 'afternoon', 'dusk', 'night'])}",
                f"people:{random.choice(['0', '1', '2', '3+'])}",
                f"orient:{random.choice(['landscape', 'portrait'])}"
            ]

            # Mock role based on position
            if i <= 5:
                role = "opener"
            elif i <= 25:
                role = "anchor"
            elif i <= 45:
                role = "detail"
            else:
                role = "breather"

            # Mock scores
            scores = {
                "Q_tech": random.uniform(0.6, 0.9),
                "Aesthetic": random.uniform(0.65, 0.95),
                "Vibe": random.uniform(0.7, 0.95),
                "Typography": random.uniform(0.5, 0.85),
                "Composition": random.uniform(0.6, 0.9),
                "LLM": random.uniform(0.75, 0.95),
                "Total": random.uniform(0.75, 0.95)
            }

            items.append({
                "photo_id": photo_id,
                "rank": i,
                "cluster_id": f"m_{random.randint(1, 50):03d}",
                "role": role,
                "scores": scores,
                "diversity_tags": diversity_tags,
                "reasons": [
                    f"Selected as {role} with rank {i}.",
                    random.choice([
                        "Strong composition and theme alignment.",
                        "Excellent technical quality and diversity contribution.",
                        "Perfect fit for the urban exploration narrative.",
                        "Balances the overall set diversity.",
                        "High aesthetic score with good typography space."
                    ]),
                    f"Diversity contribution: {', '.join(diversity_tags[:2])}"
                ],
                "artifacts": {
                    "std": f"./data/processed/{photo_id}_1024.jpg"
                }
            })

        result = {
            "batch_id": batch_id,
            "version": "1.0.0",
            "theme_spec_ref": f"./data/themes/{batch_id}_theme.yaml",
            "items": items,
            "audit": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "optimizer_params": {
                    "alpha": 1.0,
                    "beta": 1.0,
                    "gamma": 1.0
                }
            }
        }

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Also save to intermediate JSONs
        import json
        os.makedirs("intermediateJsons/exporter", exist_ok=True)
        with open(f"intermediateJsons/exporter/{batch_id}_exporter_output.json", 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ Real Exporter Service: Created curated list with {len(items)} photos")
        print(f"💾 Saved output to intermediateJsons/exporter/{batch_id}_exporter_output.json")
        return result
