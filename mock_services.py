"""
Mock Services for Photo Curation System

Simulates each service in the curation pipeline for testing and development.
Each service takes input JSON and returns the expected output JSON format.
"""

import json
import time
from typing import Dict, Any
from mock_data_generator import MockDataGenerator


class MockServices:
    """Collection of mock services that simulate the curation pipeline."""

    def __init__(self):
        self.generator = MockDataGenerator()
        self.data_store = {}  # In-memory storage for pipeline state

    def ingest_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock ingest service: register batch, verify media, extract EXIF."""
        print("🔄 Ingest Service: Processing batch registration...")

        batch_id = input_data["batch_id"]
        self.data_store[batch_id] = {"ingest_input": input_data}

        # Simulate processing time
        time.sleep(0.1)

        output = self.generator.generate_ingest_output(input_data)
        self.data_store[batch_id]["ingest_output"] = output

        print(f"✅ Ingest Service: Registered {len(output['photo_index'])} photos")
        return output

    def process_features_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock combined process & features service: standardize + extract features."""
        print("🔄 Process & Features Service: Standardizing images and extracting features...")

        batch_id = input_data["batch_id"]

        # Simulate processing time (combined operation)
        time.sleep(0.7)

        # Generate preprocess output first
        preprocess_output = self.generator.generate_preprocess_output(input_data)
        self.data_store[batch_id]["preprocess_output"] = preprocess_output

        # Generate features output
        features_output = self.generator.generate_features_output(preprocess_output)

        # Combine into single output for MVP
        combined_output = {
            "batch_id": batch_id,
            "artifacts": []
        }

        for i, artifact in enumerate(preprocess_output["artifacts"]):
            photo_id = artifact["photo_id"]
            combined_artifact = {
                "photo_id": photo_id,
                "thumb_uri": artifact["thumb_uri"],
                "std_uri": artifact["std_uri"],
                "features": features_output["features"][i] if i < len(features_output["features"]) else {}
            }
            combined_output["artifacts"].append(combined_artifact)

        self.data_store[batch_id]["features_output"] = combined_output

        print(f"✅ Process & Features Service: Processed {len(combined_output['artifacts'])} photos with features")
        return combined_output

    def scoring_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock scoring service: compute quality scores without LLM."""
        print("🔄 Scoring Service: Computing quality scores...")

        batch_id = input_data["batch_id"]

        # Simulate processing time
        time.sleep(0.3)

        # Handle both old and new formats for backward compatibility
        if "artifacts" in input_data:
            # New combined format - extract features for scoring
            features_for_scoring = {"batch_id": batch_id, "features": []}
            for artifact in input_data["artifacts"]:
                if "features" in artifact:
                    features_for_scoring["features"].append(artifact["features"])
            output = self.generator.generate_score_output(features_for_scoring)
        else:
            # Old format
            output = self.generator.generate_score_output(input_data)

        self.data_store[batch_id]["score_output"] = output

        print(f"✅ Scoring Service: Scored {len(output['scores'])} photos, dropped {len(output['dropped_for_tech'])} for quality")
        return output

    def clustering_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock clustering service: group near-duplicates."""
        print("🔄 Clustering Service: Finding moment clusters...")

        batch_id = input_data["batch_id"]

        # Get the score output for the same batch
        score_output = self.data_store[batch_id]["score_output"]

        # Simulate processing time
        time.sleep(0.2)

        output = self.generator.generate_cluster_output(score_output)
        self.data_store[batch_id]["cluster_output"] = output

        print(f"✅ Clustering Service: Created {len(output['clusters'])} clusters")
        return output

    def cluster_ranking_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock cluster ranking service: rank within clusters using LLM simulation."""
        print("🔄 Cluster Ranking Service: Ranking photos within clusters...")

        batch_id = input_data["batch_id"]

        # Get required data
        cluster_output = self.data_store[batch_id]["cluster_output"]
        score_output = self.data_store[batch_id]["score_output"]

        # Simulate LLM processing time
        time.sleep(0.8)

        output = self.generator.generate_cluster_rank_output(cluster_output, score_output)
        self.data_store[batch_id]["cluster_rank_output"] = output

        print(f"✅ Cluster Ranking Service: Ranked {len(output['cluster_winners'])} clusters, {output['judge_costs']['pairs_scored']} pairwise judgments")
        return output

    def roles_service(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock roles service: classify layout roles."""
        print("🔄 Roles Service: Classifying layout roles...")

        batch_id = input_data["batch_id"]

        # Get required data
        cluster_rank_output = self.data_store[batch_id]["cluster_rank_output"]

        # Simulate processing time
        time.sleep(0.1)

        output = self.generator.generate_roles_output(cluster_rank_output)
        self.data_store[batch_id]["roles_output"] = output

        print(f"✅ Roles Service: Classified {len(output['roles'])} photos into roles")
        return output

    def optimizer_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock combined optimizer service: roles + diversity selection."""
        print("🔄 Optimizer Service: Classifying roles and selecting diverse photo subset...")

        batch_id = input_data["batch_id"]

        # Simulate optimization processing time
        time.sleep(0.4)

        # Generate roles for cluster winners
        roles_output = self.generator.generate_roles_output(input_data)

        # For now, select a reasonable number
        num_candidates = len(roles_output["roles"])
        num_select = min(80, max(1, num_candidates))  # Ensure at least 1, at most 80

        output = self.generator.generate_selection_output(roles_output, num_select)
        self.data_store[batch_id]["selection_output"] = output

        coverage_avg = sum(output["coverage"].values()) / len(output["coverage"])
        print(".1f")
        return output

    def exporter_service(self, input_data: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock exporter service: create final curated list."""
        print("🔄 Exporter Service: Creating final curated list...")

        batch_id = input_data["batch_id"]

        # Get required data
        selection_output = self.data_store[batch_id]["selection_output"]

        # Simulate processing time
        time.sleep(0.1)

        output = self.generator.generate_curated_list(selection_output, theme_spec or self.generator.generate_theme_spec())
        self.data_store[batch_id]["curated_list"] = output

        print(f"✅ Exporter Service: Created curated list with {len(output['items'])} photos")
        return output

    def get_pipeline_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the current status of a batch in the pipeline."""
        if batch_id not in self.data_store:
            return {"error": "Batch not found"}

        batch_data = self.data_store[batch_id]
        stages_completed = list(batch_data.keys())

        return {
            "batch_id": batch_id,
            "stages_completed": stages_completed,
            "last_stage": stages_completed[-1] if stages_completed else None,
            "data": batch_data
        }

    def run_full_pipeline(self, ingest_input: Dict[str, Any], theme_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the complete curation pipeline end-to-end."""
        print("🚀 Starting Full Curation Pipeline...")

        batch_id = ingest_input["batch_id"]

        # Stage 1: Ingest
        ingest_output = self.ingest_service(ingest_input)

        # Stage 2: Combined Process & Features
        features_output = self.process_features_service(ingest_output)

        # Stage 3: Scoring
        score_output = self.scoring_service(features_output, theme_spec)

        # Stage 4: Clustering
        cluster_output = self.clustering_service(score_output)

        # Stage 5: Ranking (LLM)
        cluster_rank_output = self.cluster_ranking_service(cluster_output, theme_spec)

        # Stage 6: Optimizer (Combined roles + selection)
        selection_output = self.optimizer_service(cluster_rank_output, theme_spec)

        # Stage 7: Export
        curated_list = self.exporter_service(selection_output, theme_spec)

        print("🎉 MVP Pipeline Complete!")
        return curated_list


def main():
    """Run a complete mock pipeline for testing."""
    services = MockServices()
    generator = MockDataGenerator()

    # Generate input
    batch_id = "batch_2025-01-15_demo"
    ingest_input = generator.generate_ingest_input(batch_id, num_photos=100)
    theme_spec = generator.generate_theme_spec()

    # Run full pipeline
    result = services.run_full_pipeline(ingest_input, theme_spec)

    # Save final result
    with open("/Users/hongyilin/projects/framed_3b/mock_data/final_curated_list.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n📄 Final curated list saved with {len(result['items'])} photos")
    print(".1f")

    # Print summary
    print("\n📊 Pipeline Summary:")
    status = services.get_pipeline_status(batch_id)
    for stage in status["stages_completed"]:
        if "output" in stage:
            data = status["data"][stage]
            if "items" in data:
                print(f"  • {stage}: {len(data['items'])} items")
            elif isinstance(data, list):
                print(f"  • {stage}: {len(data)} items")
            else:
                print(f"  • {stage}: {len(data)} keys")


if __name__ == "__main__":
    main()
