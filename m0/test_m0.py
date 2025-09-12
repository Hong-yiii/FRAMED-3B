"""
M0 Implementation Test Suite

Comprehensive tests for the photo curation system skeleton and contracts.
Validates schemas, data generation, services, and orchestration.
"""

import json
import os
import sys
import jsonschema
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mock_data_generator import MockDataGenerator
from mock_services import MockServices
from orchestrator import Orchestrator


class M0TestSuite:
    """Test suite for M0 implementation."""

    def __init__(self):
        self.generator = MockDataGenerator()
        self.services = MockServices()
        self.orchestrator = Orchestrator()
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")

        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })

    def test_schema_validation(self):
        """Test JSON schema validation."""
        print("\n🧪 Testing Schema Validation")

        # Load schemas
        schema_files = [
            "theme_schema.json",
            "ingest_schemas.json",
            "features_schemas.json",
            "scoring_schemas.json",
            "optimizer_schemas.json"
        ]

        for schema_file in schema_files:
            try:
                schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas", schema_file)
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                self.log_test(f"Schema Load: {schema_file}", True, f"Loaded schema with {len(schema)} keys")
            except Exception as e:
                self.log_test(f"Schema Load: {schema_file}", False, str(e))

        # Test theme spec validation
        theme_spec = self.generator.generate_theme_spec()
        try:
            schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas", "theme_schema.json")
            with open(schema_path, 'r') as f:
                theme_schema = json.load(f)
            jsonschema.validate(theme_spec, theme_schema)
            self.log_test("Theme Spec Validation", True, "Theme spec conforms to schema")
        except Exception as e:
            self.log_test("Theme Spec Validation", False, str(e))

    def test_data_generation(self):
        """Test mock data generation."""
        print("\n🧪 Testing Data Generation")

        # Test theme generation
        theme = self.generator.generate_theme_spec()
        required_keys = ["theme_id", "name", "weights", "roles", "diversity_axes"]
        has_required = all(key in theme for key in required_keys)
        self.log_test("Theme Generation", has_required,
                     f"Generated theme with {len(theme)} keys")

        # Test ingest input generation
        batch_id = "test_batch_001"
        ingest_input = self.generator.generate_ingest_input(batch_id, num_photos=20)
        has_photos = len(ingest_input.get("photos", [])) == 20
        self.log_test("Ingest Input Generation", has_photos,
                     f"Generated input with {len(ingest_input['photos'])} photos")

        # Test unique photo IDs
        ingest_output = self.generator.generate_ingest_output(ingest_input)
        photo_ids = [p["photo_id"] for p in ingest_output["photo_index"]]
        unique_ids = len(set(photo_ids)) == len(photo_ids)
        self.log_test("Unique Photo IDs", unique_ids,
                     f"Generated {len(photo_ids)} unique photo IDs")

    def test_service_chain(self):
        """Test service chaining and data flow."""
        print("\n🧪 Testing Service Chain")

        # Create test data
        batch_id = "test_chain_001"
        ingest_input = self.generator.generate_ingest_input(batch_id, num_photos=30)
        theme_spec = self.generator.generate_theme_spec()

        # Test service chain
        try:
            ingest_output = self.services.ingest_service(ingest_input)
            features_output = self.services.process_features_service(ingest_output)
            score_output = self.services.scoring_service(features_output, theme_spec)
            cluster_output = self.services.clustering_service(score_output)
            cluster_rank_output = self.services.cluster_ranking_service(cluster_output, theme_spec)
            selection_output = self.services.optimizer_service(cluster_rank_output, theme_spec)
            curated_list = self.services.exporter_service(selection_output, theme_spec)

            # Validate chain integrity
            chain_valid = (
                curated_list["batch_id"] == batch_id and
                len(curated_list["items"]) > 0 and
                "audit" in curated_list
            )
            self.log_test("Service Chain", chain_valid,
                         f"Processed {len(ingest_input['photos'])} → {len(curated_list['items'])} photos")

        except Exception as e:
            self.log_test("Service Chain", False, f"Chain failed: {str(e)}")

    def test_orchestrator_flow(self):
        """Test event-driven orchestrator."""
        print("\n🧪 Testing Event-Driven Orchestrator")

        # Create test batch
        batch_id = "test_orch_001"
        ingest_input = self.generator.generate_ingest_input(batch_id, num_photos=25)

        try:
            # Start orchestration
            self.orchestrator.start_batch(ingest_input)

            # Wait for completion
            import time
            time.sleep(0.5)

            # Check completion
            status = self.orchestrator.get_batch_status(batch_id)
            completed = status.get("stage") == "curation.completed"
            self.log_test("Orchestrator Completion", completed,
                         f"Batch reached stage: {status.get('stage', 'unknown')}")

            # Check event flow (MVP has 7 events: ingest, features, score, cluster, cluster.rank, selection, curation)
            metrics = self.orchestrator.get_pipeline_metrics()
            events_flow = metrics["total_events"] >= 7  # MVP stages completed
            self.log_test("Event Flow", events_flow,
                         f"Generated {metrics['total_events']} events (MVP: 7 expected)")

        except Exception as e:
            self.log_test("Orchestrator Flow", False, f"Orchestrator failed: {str(e)}")

    def test_data_quality(self):
        """Test data quality and consistency."""
        print("\n🧪 Testing Data Quality")

        # Generate comprehensive test data
        batch_id = "test_quality_001"
        ingest_input = self.generator.generate_ingest_input(batch_id, num_photos=40)
        ingest_output = self.generator.generate_ingest_output(ingest_input)
        preprocess_output = self.generator.generate_preprocess_output(ingest_output)
        features_output = self.generator.generate_features_output(preprocess_output)
        score_output = self.generator.generate_score_output(features_output)

        # Test photo ID consistency
        input_count = len(ingest_input["photos"])
        output_count = len(ingest_output["photo_index"])
        scored_count = len(score_output["scores"])

        consistent_counts = input_count == output_count == scored_count
        self.log_test("Photo Count Consistency", consistent_counts,
                     f"Consistent counts: input={input_count}, output={output_count}, scored={scored_count}")

        # Test that all scored photos exist in the original input
        scored_ids = set(s["photo_id"] for s in score_output["scores"])
        output_ids = set(p["photo_id"] for p in ingest_output["photo_index"])
        all_scored_exist = scored_ids.issubset(output_ids)
        self.log_test("Photo ID Validity", all_scored_exist,
                     f"All {len(scored_ids)} scored photos exist in input")

        # Test score ranges
        scores_valid = all(
            0 <= s["Q_tech"] <= 1 and
            0 <= s["Aesthetic"] <= 1 and
            0 <= s["Total_prelim"] <= 1
            for s in score_output["scores"]
        )
        self.log_test("Score Ranges", scores_valid,
                     "All scores in [0,1] range")

        # Test feature completeness
        features_complete = all(
            len(f) >= 7  # All required feature fields
            for f in features_output["features"]
        )
        self.log_test("Feature Completeness", features_complete,
                     f"All {len(features_output['features'])} photos have complete features")

    def test_performance(self):
        """Test performance characteristics."""
        print("\n🧪 Testing Performance")

        import time

        # Test data generation speed
        start_time = time.time()
        for i in range(10):
            self.generator.generate_ingest_input(f"perf_test_{i}", 50)
        gen_time = time.time() - start_time
        gen_rate = 10 / gen_time if gen_time > 0 else float('inf')
        self.log_test("Data Generation Speed", gen_rate > 1,
                     ".2f batches/sec")

        # Test service chain speed
        start_time = time.time()
        batch_id = "perf_service_001"
        ingest_input = self.generator.generate_ingest_input(batch_id, 30)
        theme_spec = self.generator.generate_theme_spec()

        result = self.services.run_full_pipeline(ingest_input, theme_spec)
        service_time = time.time() - start_time

        service_rate = 30 / service_time if service_time > 0 else float('inf')
        self.log_test("Service Chain Speed", service_rate > 10,
                     ".1f photos/sec")

    def run_all_tests(self):
        """Run all tests and generate report."""
        print("🚀 Running M0 Test Suite")
        print("=" * 50)

        self.test_schema_validation()
        self.test_data_generation()
        self.test_service_chain()
        self.test_orchestrator_flow()
        self.test_data_quality()
        self.test_performance()

        # Generate summary
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        success_rate = passed / total if total > 0 else 0

        print("\n📊 Test Summary")
        print(f"Passed: {passed}/{total} ({success_rate:.1%})")

        if success_rate >= 0.9:
            print("🎉 M0 Implementation: EXCELLENT")
        elif success_rate >= 0.75:
            print("✅ M0 Implementation: GOOD")
        else:
            print("⚠️  M0 Implementation: NEEDS IMPROVEMENT")

        # Save detailed results
        results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                "summary": {
                    "passed": passed,
                    "total": total,
                    "success_rate": success_rate
                },
                "results": self.test_results
            }, f, indent=2)

        return success_rate >= 0.8  # Consider 80% passing as success


def main():
    """Run the complete M0 test suite."""
    test_suite = M0TestSuite()
    success = test_suite.run_all_tests()

    if success:
        print("\n🎯 M0 Milestone: ACHIEVED!")
        print("Ready to proceed to M1: Local Signals")
    else:
        print("\n⚠️  M0 Milestone: ISSUES DETECTED")
        print("Review test results and fix issues before proceeding")


if __name__ == "__main__":
    main()
