"""
Mock Data Generator for Photo Curation System

Generates realistic fake data for testing the end-to-end curation pipeline.
All data conforms to the JSON schemas defined in /schemas/.
"""

import json
import random
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any


class MockDataGenerator:
    """Generates mock data for the photo curation system."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.photo_count = 0
        self.cluster_count = 0

    def generate_photo_id(self, filename: str) -> str:
        """Generate a content-addressable photo ID."""
        return hashlib.sha256(f"mock_{filename}".encode()).hexdigest()

    def generate_theme_spec(self, theme_id: str = "travel_zine_001") -> Dict[str, Any]:
        """Generate a mock theme specification."""
        return {
            "theme_id": theme_id,
            "name": "Urban Exploration",
            "summary": "A zine about discovering hidden gems in city landscapes, focusing on architecture, street life, and the interplay of light and shadow in urban environments.",
            "weights": {
                "comp": 0.22,
                "subject": 0.18,
                "light": 0.20,
                "type": 0.12,
                "story": 0.18,
                "vibe": 0.10
            },
            "roles": {
                "opener": {"min": 3, "max": 6},
                "anchor": {"min": 10, "max": 20},
                "detail": {"min": 8, "max": 20},
                "breathers": {"min": 6, "max": 12}
            },
            "diversity_axes": ["scene_type", "people_count", "time_of_day", "palette_cluster", "location_cluster", "orientation"],
            "keywords": ["urban", "architecture", "street", "light", "shadow", "city", "exploration", "concrete", "neon", "vintage"],
            "palette_reference": {
                "primary_colors": [
                    [45, 15, -20],  # Warm concrete tones
                    [65, -5, 15],   # Street light yellow
                    [25, 5, -25]    # Shadow blues
                ]
            }
        }

    def generate_ingest_input(self, batch_id: str, num_photos: int = 50) -> Dict[str, Any]:
        """Generate mock ingest service input."""
        photos = []
        for i in range(num_photos):
            filename = "02d"
            photos.append({
                "uri": f"./data/input/{filename}.jpg"
            })

        return {
            "batch_id": batch_id,
            "photos": photos,
            "theme_spec_ref": f"./data/themes/{batch_id}_theme.yaml",
            "user_overrides": {
                "lock_in": [f"P{random.randint(1, num_photos):03d}.jpg" for _ in range(2)],
                "exclude": [f"P{random.randint(1, num_photos):03d}.jpg" for _ in range(1)]
            }
        }

    def generate_ingest_output(self, ingest_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock ingest service output."""
        batch_id = ingest_input["batch_id"]
        photo_index = []

        for i, photo in enumerate(ingest_input["photos"]):
            filename = photo["uri"].split("/")[-1]
            # Add index to make photo_id unique
            photo_id = self.generate_photo_id(f"{filename}_{i}")

            photo_index.append({
                "photo_id": photo_id,
                "uri": photo["uri"],
                "exif": {
                    "camera": random.choice(["Canon EOS R5", "Sony A7R IV", "Nikon Z6 II", "iPhone 15 Pro"]),
                    "lens": random.choice(["24-70mm f/2.8", "50mm f/1.4", "16-35mm f/2.8", "85mm f/1.4"]),
                    "iso": random.choice([100, 200, 400, 800, 1600]),
                    "aperture": f"f/{random.choice([1.4, 2.0, 2.8, 4.0, 5.6, 8.0])}",
                    "shutter_speed": f"1/{random.choice([30, 60, 125, 250, 500, 1000])}",
                    "focal_length": f"{random.randint(24, 85)}mm",
                    "datetime": datetime.now(timezone.utc).isoformat(),
                    "gps": {
                        "lat": 40.7128 + random.uniform(-0.1, 0.1),
                        "lon": -74.0060 + random.uniform(-0.1, 0.1)
                    } if random.random() > 0.3 else None
                }
            })

        return {
            "batch_id": batch_id,
            "photo_index": photo_index
        }

    def generate_preprocess_output(self, ingest_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock preprocess service output."""
        batch_id = ingest_output["batch_id"]
        artifacts = []

        for photo in ingest_output["photo_index"]:
            artifacts.append({
                "photo_id": photo["photo_id"],
                "std_uri": f"./data/processed/{photo['photo_id']}_1024.jpg"
            })

        return {
            "batch_id": batch_id,
            "artifacts": artifacts
        }

    def generate_features_output(self, preprocess_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock features extraction output."""
        batch_id = preprocess_output["batch_id"]
        features = []

        scene_types = ["street", "architecture", "people", "nature", "interior", "transport"]
        time_of_day = ["dawn", "morning", "afternoon", "dusk", "night"]
        orientations = ["landscape", "portrait"]
        palette_clusters = ["warm", "cool", "monochrome", "vibrant"]

        for artifact in preprocess_output["artifacts"]:
            photo_id = artifact["photo_id"]

            # Generate realistic feature values
            sharpness = random.uniform(0.3, 0.95)
            exposure = random.uniform(0.4, 0.9)
            noise = random.uniform(0.05, 0.4)
            horizon_tilt = random.uniform(-3, 3)

            features.append({
                "photo_id": photo_id,
                "embeddings": {
                    "clip_L14": f"./data/emb/{photo_id}_clip.npy"
                },
                "hashes": {
                    "phash": hashlib.md5(f"phash_{photo_id}".encode()).hexdigest()[:16]
                },
                "tech": {
                    "sharpness": sharpness,
                    "exposure": exposure,
                    "noise": noise,
                    "horizon_deg": horizon_tilt
                },
                "saliency": {
                    "heatmap_uri": f"./data/sal/{photo_id}.png",
                    "neg_space_ratio": random.uniform(0.1, 0.8)
                },
                "faces": {
                    "count": random.choices([0, 1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.08, 0.02])[0],
                    "landmarks_ok": random.random() > 0.1
                },
                "palette": {
                    "lab_centroids": [
                        [random.uniform(20, 80), random.uniform(-20, 20), random.uniform(-30, 30)]
                        for _ in range(random.randint(2, 5))
                    ],
                    "cluster_id": f"pal_{random.randint(1, 10):02d}"
                }
            })

        return {
            "batch_id": batch_id,
            "features": features
        }

    def generate_score_output(self, features_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock scoring output."""
        batch_id = features_output["batch_id"]
        scores = []
        dropped = []

        for feature in features_output["features"]:
            photo_id = feature["photo_id"]
            tech = feature["tech"]

            # Technical quality gate
            q_tech = (tech["sharpness"] * 0.4 + tech["exposure"] * 0.4 + (1 - tech["noise"]) * 0.2)

            if q_tech < 0.3:
                dropped.append(photo_id)
                continue

            scores.append({
                "photo_id": photo_id,
                "Q_tech": q_tech,
                "Aesthetic": random.uniform(0.5, 0.95),
                "Vibe": random.uniform(0.4, 0.9),
                "Typography": feature["saliency"]["neg_space_ratio"] * 0.8 + random.uniform(0.1, 0.3),
                "Composition": random.uniform(0.5, 0.9),
                "Total_prelim": random.uniform(0.6, 0.9)
            })

        return {
            "batch_id": batch_id,
            "scores": scores,
            "dropped_for_tech": dropped
        }

    def generate_cluster_output(self, score_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock clustering output."""
        batch_id = score_output["batch_id"]
        valid_photos = [s["photo_id"] for s in score_output["scores"]]

        # Create realistic clusters (some singles, some groups)
        clusters = []
        used_photos = set()

        # Create some multi-photo clusters
        remaining = valid_photos.copy()
        random.shuffle(remaining)

        while remaining:
            cluster_size = min(len(remaining), random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0])
            cluster_members = remaining[:cluster_size]
            remaining = remaining[cluster_size:]

            clusters.append({
                "cluster_id": "02d",
                "members": cluster_members
            })
            self.cluster_count += 1

        return {
            "batch_id": batch_id,
            "clusters": clusters
        }

    def generate_cluster_rank_output(self, cluster_output: Dict[str, Any], score_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock cluster ranking output."""
        batch_id = cluster_output["batch_id"]

        cluster_winners = []
        rationales = {}
        used_heroes = set()  # Track used heroes to avoid duplicates

        # Create mapping of photo_id to score for faster lookup
        score_map = {s["photo_id"]: s for s in score_output["scores"]}

        for cluster in cluster_output["clusters"]:
            cluster_id = cluster["cluster_id"]
            members = cluster["members"]

            if len(members) == 1:
                hero = members[0]
                alternates = []
            else:
                # Pick hero based on scores, preferring unused photos
                available_members = [m for m in members if m in score_map]

                if available_members:
                    # Sort by score, then prefer unused photos
                    sorted_members = sorted(
                        available_members,
                        key=lambda m: (score_map[m]["Total_prelim"], m not in used_heroes),
                        reverse=True
                    )
                    hero = sorted_members[0]

                    # Mark as used
                    used_heroes.add(hero)

                    # Get alternates from remaining members
                    remaining = sorted_members[1:]
                    alternates = [m for m in remaining[:min(2, len(remaining))]]
                else:
                    # Fallback if no scores available
                    hero = members[0]
                    alternates = members[1:min(3, len(members))]

            cluster_winners.append({
                "cluster_id": cluster_id,
                "hero": hero,
                "alternates": alternates
            })

            # Add LLM rationales for hero
            if hero in score_map:
                score = score_map[hero]["Total_prelim"]
                rank_text = "1" if len(members) == 1 else f"{random.randint(1, len(members))}"
                rationales[hero] = [
                    f"Cluster hero (ranked {rank_text} in {len(members)}-way comparison).",
                    random.choice([
                        "Balanced subject placement with strong leading lines.",
                        "Excellent color harmony matching theme palette.",
                        "Clear focal point with good negative space for text.",
                        "Captures the urban exploration vibe perfectly.",
                        "Strong composition with interesting light play."
                    ]),
                    f"Technical score: {score:.2f}"
                ]
            else:
                rationales[hero] = [
                    "Cluster hero (selected by default).",
                    "Good technical quality and composition.",
                    "Technical score: 0.75"
                ]

        return {
            "batch_id": batch_id,
            "cluster_winners": cluster_winners,
            "llm_rationales": rationales,
            "judge_costs": {
                "pairs_scored": sum(len(c["members"]) - 1 for c in cluster_output["clusters"] if len(c["members"]) > 1),
                "tokens_est": random.randint(5000, 15000)
            }
        }

    def generate_roles_output(self, cluster_rank_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock role classification output."""
        batch_id = cluster_rank_output["batch_id"]

        # Get all candidate photos (heroes + alternates)
        candidates = set()
        for winner in cluster_rank_output["cluster_winners"]:
            candidates.add(winner["hero"])
            candidates.update(winner["alternates"])

        roles = []
        for photo_id in candidates:
            # Generate role probabilities that sum to 1
            probs = {
                "opener": random.uniform(0, 0.4),
                "anchor": random.uniform(0, 0.5),
                "detail": random.uniform(0, 0.6),
                "breather": random.uniform(0, 0.3)
            }

            # Normalize
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}

            roles.append({
                "photo_id": photo_id,
                "probs": probs
            })

        return {
            "batch_id": batch_id,
            "roles": roles
        }

    def generate_selection_output(self, roles_output: Dict[str, Any], num_select: int = 80) -> Dict[str, Any]:
        """Generate mock diversity optimizer output."""
        batch_id = roles_output["batch_id"]
        candidates = [r["photo_id"] for r in roles_output["roles"]]

        # Select subset
        selected = random.sample(candidates, min(num_select, len(candidates)))

        # Mock marginal gains and coverage
        marginal_gains = {pid: random.uniform(0.01, 0.1) for pid in selected}

        coverage = {
            "scene_type": random.uniform(0.7, 0.95),
            "palette_cluster": random.uniform(0.6, 0.9),
            "time_of_day": random.uniform(0.75, 0.95),
            "location_cluster": random.uniform(0.65, 0.9),
            "people_count": random.uniform(0.7, 0.95),
            "orientation": random.uniform(0.8, 0.98)
        }

        return {
            "batch_id": batch_id,
            "selected_ids": selected,
            "marginal_gains": marginal_gains,
            "coverage": coverage
        }

    def generate_curated_list(self, selection_output: Dict[str, Any], theme_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final curated list output."""
        batch_id = selection_output["batch_id"]
        selected_ids = selection_output["selected_ids"]

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

            items.append({
                "photo_id": photo_id,
                "rank": i,
                "cluster_id": f"m_{random.randint(1, 50):03d}",
                "role": role,
                "scores": {
                    "Q_tech": random.uniform(0.6, 0.9),
                    "Aesthetic": random.uniform(0.65, 0.95),
                    "Vibe": random.uniform(0.7, 0.95),
                    "Typography": random.uniform(0.5, 0.85),
                    "Composition": random.uniform(0.6, 0.9),
                    "LLM": random.uniform(0.75, 0.95),
                    "Total": random.uniform(0.75, 0.95)
                },
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
                    "std": f"./data/std/{photo_id}_1024.jpg"
                }
            })

        return {
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


def main():
    """Generate and save sample data for testing."""
    generator = MockDataGenerator()

    # Generate a complete pipeline
    batch_id = "batch_2025-01-15_test"
    theme_spec = generator.generate_theme_spec()

    # Generate inputs
    ingest_input = generator.generate_ingest_input(batch_id, num_photos=50)

    # Generate all outputs
    ingest_output = generator.generate_ingest_output(ingest_input)
    preprocess_output = generator.generate_preprocess_output(ingest_output)
    features_output = generator.generate_features_output(preprocess_output)
    score_output = generator.generate_score_output(features_output)
    cluster_output = generator.generate_cluster_output(score_output)
    cluster_rank_output = generator.generate_cluster_rank_output(cluster_output, score_output)
    roles_output = generator.generate_roles_output(cluster_rank_output)
    selection_output = generator.generate_selection_output(roles_output, num_select=80)
    curated_list = generator.generate_curated_list(selection_output, theme_spec)

    # Save to files
    import os
    os.makedirs("/Users/hongyilin/projects/framed_3b/mock_data", exist_ok=True)

    outputs = {
        "theme_spec.json": theme_spec,
        "ingest_input.json": ingest_input,
        "ingest_output.json": ingest_output,
        "preprocess_output.json": preprocess_output,
        "features_output.json": features_output,
        "score_output.json": score_output,
        "cluster_output.json": cluster_output,
        "cluster_rank_output.json": cluster_rank_output,
        "roles_output.json": roles_output,
        "selection_output.json": selection_output,
        "curated_list.json": curated_list
    }

    for filename, data in outputs.items():
        with open(f"/Users/hongyilin/projects/framed_3b/mock_data/{filename}", 'w') as f:
            json.dump(data, f, indent=2)

    print("Mock data generated successfully!")
    print(f"Generated {len(curated_list['items'])} curated photos from {len(ingest_input['photos'])} input photos")
    print(f"Created {len(cluster_output['clusters'])} clusters")


if __name__ == "__main__":
    main()
