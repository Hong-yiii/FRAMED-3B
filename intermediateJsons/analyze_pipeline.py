"""
Pipeline Analysis Script for Intermediate JSONs

Analyzes the performance and output of each pipeline step.
"""

import os
import json
import glob
from typing import Dict, Any, List, Optional
from datetime import datetime


class PipelineAnalyzer:
    """Analyzes intermediate JSON outputs from the curation pipeline."""

    def __init__(self, intermediate_dir: Optional[str] = None):
        if intermediate_dir is None:
            # Auto-detect the correct intermediateJsons directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # Go up one level from intermediateJsons/
            self.intermediate_dir = script_dir  # The script is already in intermediateJsons/
        else:
            self.intermediate_dir = intermediate_dir

        self.steps = ["ingest", "preprocess", "features", "scoring", "clustering", "ranking", "optimizer", "exporter"]

    def analyze_all_batches(self) -> Dict[str, Any]:
        """Analyze all batches found in intermediate JSONs."""
        print("🔍 Analyzing pipeline performance...")

        # Find all batch IDs
        batch_ids = set()
        for step in self.steps:
            step_dir = os.path.join(self.intermediate_dir, step)
            if os.path.exists(step_dir):
                for file in glob.glob(f"{step_dir}/*.json"):
                    filename = os.path.basename(file)
                    # Extract batch_id from filename (format: batch_id_step_output.json)
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        batch_id = '_'.join(parts[:-2])  # Remove step and output parts
                        batch_ids.add(batch_id)

        results = {}
        for batch_id in sorted(batch_ids):
            print(f"\n📊 Analyzing batch: {batch_id}")
            results[batch_id] = self.analyze_batch(batch_id)

        return results

    def analyze_batch(self, batch_id: str) -> Dict[str, Any]:
        """Analyze a specific batch across all pipeline steps."""
        analysis = {
            "batch_id": batch_id,
            "steps_completed": [],
            "performance_metrics": {},
            "data_flow": {},
            "issues": []
        }

        for step in self.steps:
            step_file = os.path.join(self.intermediate_dir, step, f"{batch_id}_{step}_output.json")
            if os.path.exists(step_file):
                try:
                    with open(step_file, 'r') as f:
                        data = json.load(f)

                    analysis["steps_completed"].append(step)
                    analysis["data_flow"][step] = self._analyze_step_data(step, data)
                    analysis["performance_metrics"][step] = self._extract_metrics(step, data)

                except Exception as e:
                    analysis["issues"].append(f"Error reading {step}: {e}")
            else:
                analysis["issues"].append(f"Missing {step} output")

        return analysis

    def _analyze_step_data(self, step: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the data content of a specific step."""
        analysis = {"record_count": 0, "data_quality": "unknown"}

        if step == "ingest":
            if "photo_index" in data:
                analysis["record_count"] = len(data["photo_index"])
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"
                analysis["photos_with_exif"] = sum(1 for p in data["photo_index"] if p.get("exif", {}).get("camera") != "Unknown")

        elif step == "features":
            if "artifacts" in data:
                analysis["record_count"] = len(data["artifacts"])
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"

        elif step == "scoring":
            if "scores" in data:
                analysis["record_count"] = len(data["scores"])
                analysis["dropped_count"] = len(data.get("dropped_for_tech", []))
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"
                if analysis["record_count"] > 0:
                    scores = [s.get("Total_prelim", 0) for s in data["scores"]]
                    analysis["avg_score"] = sum(scores) / len(scores)
                    analysis["score_range"] = f"{min(scores):.2f} - {max(scores):.2f}"

        elif step == "clustering":
            if "clusters" in data:
                analysis["record_count"] = len(data["clusters"])
                analysis["total_photos"] = sum(len(c.get("members", [])) for c in data["clusters"])
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"

        elif step == "ranking":
            if "cluster_winners" in data:
                analysis["record_count"] = len(data["cluster_winners"])
                analysis["judge_costs"] = data.get("judge_costs", {})
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"

        elif step == "optimizer":
            if "selected_ids" in data:
                analysis["record_count"] = len(data["selected_ids"])
                analysis["coverage"] = data.get("coverage", {})
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"

        elif step == "preprocess":
            if "artifacts" in data:
                analysis["record_count"] = len(data["artifacts"])
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"

        elif step == "exporter":
            if "items" in data:
                analysis["record_count"] = len(data["items"])
                analysis["data_quality"] = "good" if analysis["record_count"] > 0 else "empty"

        return analysis

    def _extract_metrics(self, step: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from step data."""
        metrics = {}

        if step == "scoring":
            scores = data.get("scores", [])
            if scores:
                tech_scores = [s.get("Q_tech", 0) for s in scores]
                aesthetic_scores = [s.get("Aesthetic", 0) for s in scores]
                total_scores = [s.get("Total_prelim", 0) for s in scores]

                metrics.update({
                    "avg_tech_score": sum(tech_scores) / len(tech_scores),
                    "avg_aesthetic_score": sum(aesthetic_scores) / len(aesthetic_scores),
                    "avg_total_score": sum(total_scores) / len(total_scores),
                    "tech_score_range": f"{min(tech_scores):.2f} - {max(tech_scores):.2f}",
                    "aesthetic_score_range": f"{min(aesthetic_scores):.2f} - {max(aesthetic_scores):.2f}",
                    "total_score_range": f"{min(total_scores):.2f} - {max(total_scores):.2f}"
                })

        elif step == "ranking":
            judge_costs = data.get("judge_costs", {})
            metrics.update(judge_costs)

        elif step == "optimizer":
            coverage = data.get("coverage", {})
            if coverage:
                avg_coverage = sum(coverage.values()) / len(coverage)
                metrics["avg_coverage"] = avg_coverage
                metrics["coverage_details"] = coverage

        return metrics

    def print_summary_report(self, results: Dict[str, Any]):
        """Print a comprehensive summary report."""
        print("\n" + "="*80)
        print("📊 PIPELINE ANALYSIS SUMMARY REPORT")
        print("="*80)

        for batch_id, analysis in results.items():
            print(f"\n🎯 Batch: {batch_id}")
            print(f"   Steps Completed: {len(analysis['steps_completed'])}/{len(self.steps)}")
            print(f"   Pipeline: {' → '.join(analysis['steps_completed'])}")

            if analysis["issues"]:
                print("   ⚠️  Issues:")
                for issue in analysis["issues"]:
                    print(f"      • {issue}")

            # Show key metrics for each step
            for step in analysis["steps_completed"]:
                step_data = analysis["data_flow"].get(step, {})
                metrics = analysis["performance_metrics"].get(step, {})

                print(f"\n   🔄 {step.upper()}:")
                print(f"      Records: {step_data.get('record_count', 'N/A')}")

                if step == "scoring":
                    print(".2f")
                elif step == "ranking":
                    pairs = metrics.get("pairs_scored", 0)
                    tokens = metrics.get("tokens_est", 0)
                    print(f"      Pairs Judged: {pairs}, Est. Tokens: {tokens}")
                elif step == "optimizer":
                    coverage = metrics.get("avg_coverage", 0)
                    print(".1%")

        print(f"\n{'='*80}")


def main():
    """Main analysis function."""
    analyzer = PipelineAnalyzer()

    # Check if we're in the right directory structure
    if not os.path.exists(analyzer.intermediate_dir):
        print("❌ intermediateJsons directory not found!")
        print(f"   Expected: {analyzer.intermediate_dir}")
        print("   Run the pipeline first to generate intermediate outputs.")
        return

    results = analyzer.analyze_all_batches()
    analyzer.print_summary_report(results)

    # Save detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = os.path.join(analyzer.intermediate_dir, f"analysis_{timestamp}.json")
    with open(analysis_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed analysis saved to {os.path.basename(analysis_file)}")


if __name__ == "__main__":
    main()
