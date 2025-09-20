"""
Event-Driven Orchestrator for Photo Curation Pipeline

Simulates the microservices architecture with event-driven communication.
Services publish events to a message bus, and the orchestrator coordinates
the flow based on event subscriptions.
"""

import json
import time
import threading
from typing import Dict, Any, List, Callable
from datetime import datetime, timezone
from mock_services import MockServices
from mock_data_generator import MockDataGenerator

# Import real services
try:
    from services import RealServices
    REAL_SERVICES_AVAILABLE = True
except ImportError:
    REAL_SERVICES_AVAILABLE = False
    print("⚠️  Real services not available, using mock services")

# Import real services

class MessageBus:
    """Simple in-memory message bus for event-driven communication."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.events_log: List[Dict[str, Any]] = []

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        print(f"📡 Subscribed to event: {event_type}")

    def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to all subscribers."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        }

        self.events_log.append(event)
        print(f"📢 Published event: {event_type}")

        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"❌ Error in event handler for {event_type}: {e}")

    def get_event_log(self, event_type: str = None) -> List[Dict[str, Any]]:
        """Get event log, optionally filtered by event type."""
        if event_type:
            return [e for e in self.events_log if e["event_type"] == event_type]
        return self.events_log.copy()


class Orchestrator:
    """Coordinates the curation pipeline through event-driven architecture."""

    def __init__(self):
        self.message_bus = MessageBus()
        self.mock_services = MockServices()
        self.generator = MockDataGenerator()
        self.real_services = None
        self.use_real_services = False
        self.batch_states: Dict[str, Dict[str, Any]] = {}

        # Initialize real services if available
        if REAL_SERVICES_AVAILABLE:
            try:
                self.real_services = RealServices()
                print("✅ Real services initialized")
            except Exception as e:
                print(f"❌ Error initializing real services: {e}")

        self.setup_event_subscriptions()

    def _should_use_real_services(self, input_data: Dict[str, Any]) -> bool:
        """Determine if we should use real services based on input data."""
        if not self.real_services:
            return False

        # Check if any photos have local file paths (not mock data)
        photos = input_data.get("photos", [])
        if not photos:
            return False

        # If any photo URI starts with "./data/input/", it's real data
        # Also check for other local path indicators
        for photo in photos:
            uri = photo.get("uri", "")
            if uri.startswith("./data/input/"):
                return True

        # If all URIs are S3 or other remote URIs, use mock services
        # unless explicitly overridden
        all_remote = all(
            photo.get("uri", "").startswith(("s3://", "http://", "https://"))
            for photo in photos
        )

        return not all_remote

    def _get_services(self):
        """Get the appropriate services based on current mode."""
        return self.real_services if self.use_real_services else self.mock_services

    def setup_event_subscriptions(self):
        """Set up event subscriptions for the MVP pipeline flow."""

        # Ingest → Preprocess
        self.message_bus.subscribe("ingest.completed", self.handle_ingest_completed)

        # Preprocess → Features
        self.message_bus.subscribe("preprocess.completed", self.handle_preprocess_completed)

        # Features → Scoring
        self.message_bus.subscribe("features.completed", self.handle_features_completed)

        # Scoring → Clustering
        self.message_bus.subscribe("score.completed", self.handle_score_completed)

        # Clustering → Ranking
        self.message_bus.subscribe("cluster.completed", self.handle_cluster_completed)

        # Ranking → Optimizer
        self.message_bus.subscribe("cluster.rank.completed", self.handle_cluster_rank_completed)

        # Optimizer → Exporter
        self.message_bus.subscribe("selection.completed", self.handle_selection_completed)

        # Human review loop (optional)
        self.message_bus.subscribe("review.update", self.handle_review_update)

    def handle_ingest_completed(self, event: Dict[str, Any]):
        """Handle ingest completion and trigger preprocessing."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"✓ Ingest → Preprocess ({batch_id})")
        self.batch_states[batch_id]["stage"] = "ingest.completed"

        # Use the same service type as determined at batch start
        services = self._get_services()

        # Trigger preprocess service
        preprocess_output = services.preprocess_service(data)
        self.message_bus.publish("preprocess.completed", preprocess_output)

    def handle_preprocess_completed(self, event: Dict[str, Any]):
        """Handle preprocess completion and trigger features extraction."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"✓ Preprocess → Features ({batch_id})")
        self.batch_states[batch_id]["stage"] = "preprocess.completed"

        # Use the same service type as determined at batch start
        services = self._get_services()

        # Trigger features service
        features_output = services.process_features_service(data)
        self.message_bus.publish("features.completed", features_output)

    def handle_features_completed(self, event: Dict[str, Any]):
        """Handle features completion and trigger scoring."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"✓ Features → Scoring ({batch_id})")
        self.batch_states[batch_id]["stage"] = "features.completed"

        # Get theme spec (in real system, this would come from a service)
        theme_spec = self.generator.generate_theme_spec()

        # Use the same service type as determined at batch start
        services = self._get_services()

        # Trigger scoring service
        score_output = services.scoring_service(data, theme_spec)
        self.message_bus.publish("score.completed", score_output)

    def handle_score_completed(self, event: Dict[str, Any]):
        """Handle scoring completion and trigger clustering."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"✓ Scoring → Clustering ({batch_id})")
        self.batch_states[batch_id]["stage"] = "score.completed"

        # Use the same service type as determined at batch start
        services = self._get_services()

        # Trigger clustering service
        cluster_output = services.clustering_service(data)
        self.message_bus.publish("cluster.completed", cluster_output)

    def handle_cluster_completed(self, event: Dict[str, Any]):
        """Handle clustering completion and trigger cluster ranking."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"✓ Clustering → Ranking ({batch_id})")
        self.batch_states[batch_id]["stage"] = "cluster.completed"

        # Get required data
        services = self._get_services()
        score_output = services.data_store[batch_id]["score_output"]
        theme_spec = self.generator.generate_theme_spec()

        # Trigger cluster ranking service
        cluster_rank_output = services.cluster_ranking_service(data, theme_spec)
        self.message_bus.publish("cluster.rank.completed", cluster_rank_output)

    def handle_cluster_rank_completed(self, event: Dict[str, Any]):
        """Handle cluster ranking completion and trigger optimization."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"✓ Ranking → Optimization ({batch_id})")
        self.batch_states[batch_id]["stage"] = "cluster.rank.completed"

        # Get theme spec
        theme_spec = self.generator.generate_theme_spec()

        # Use the same service type as determined at batch start
        services = self._get_services()

        # Trigger combined optimizer service
        selection_output = services.optimizer_service(data, theme_spec)
        self.message_bus.publish("selection.completed", selection_output)

    def handle_selection_completed(self, event: Dict[str, Any]):
        """Handle selection completion and trigger export."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"✓ Optimization → Export ({batch_id})")
        self.batch_states[batch_id]["stage"] = "selection.completed"

        # Get theme spec
        theme_spec = self.generator.generate_theme_spec()

        # Use the same service type as determined at batch start
        services = self._get_services()

        # Trigger exporter service
        curated_list = services.exporter_service(data, theme_spec)
        self.message_bus.publish("curation.completed", curated_list)

        # Mark batch as completed
        self.batch_states[batch_id]["stage"] = "curation.completed"
        self.batch_states[batch_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

        print(f"🎉 Pipeline complete: {batch_id}")

    def handle_review_update(self, event: Dict[str, Any]):
        """Handle human review updates and re-run optimization."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Review update received for {batch_id}, re-running optimization...")

        # Use the same service type as determined at batch start
        services = self._get_services()

        # Get current selection
        current_selection = services.data_store[batch_id]["selection_output"]

        # Apply review constraints (in real system, this would modify the optimizer input)
        # For now, just re-run with same data
        theme_spec = self.generator.generate_theme_spec()
        new_selection = services.optimizer_service(current_selection, theme_spec)

        self.message_bus.publish("selection.completed", new_selection)

    def generate_ingest_input_from_directory(self, input_dir: str = "./data/input",
                                            batch_id: str = None) -> Dict[str, Any]:
        """Generate ingest input from photos in a directory using the dedicated service."""

        # Generate batch ID if not provided
        if batch_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            batch_id = f"batch_{timestamp}"

        # Use the dedicated service
        if self.real_services:
            ingest_input = self.real_services.generate_ingest_input_from_directory(batch_id, input_dir)
        else:
            print("⚠️ Real services not available, exiting")
            return 0

        if ingest_input is None:
            raise ValueError(f"No image files found in {input_dir}")

        return ingest_input

    def start_batch(self, ingest_input: Dict[str, Any] = None, batch_id: str = None) -> str:
        """Start a new curation batch."""

        # If no ingest_input provided, try to generate from data/input directory
        if ingest_input is None:
            try:
                ingest_input = self.generate_ingest_input_from_directory(batch_id=batch_id)
                print(f"📁 Auto-generated ingest input from data/input/ directory")
            except ValueError as e:
                print(f"❌ {e}")
                return None

        # Use the batch_id from the ingest_input, or use the provided batch_id
        batch_id = ingest_input["batch_id"]

        # Determine whether to use real services
        self.use_real_services = self._should_use_real_services(ingest_input)
        services = self._get_services()

        service_type = "real" if self.use_real_services else "mock"
        print(f"🚀 Starting batch {batch_id} ({service_type})")

        self.batch_states[batch_id] = {
            "stage": "starting",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "service_type": service_type
        }

        # Trigger initial ingest service
        ingest_output = services.ingest_service(ingest_input)
        self.message_bus.publish("ingest.completed", ingest_output)

        return batch_id

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the current status of a batch."""
        if batch_id not in self.batch_states:
            return {"error": "Batch not found"}

        status = self.batch_states[batch_id].copy()
        status["batch_id"] = batch_id

        # Get event log for this batch
        events = [e for e in self.message_bus.get_event_log()
                 if e["data"].get("batch_id") == batch_id]

        status["events"] = len(events)
        status["last_event"] = events[-1]["event_type"] if events else None

        return status

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get overall pipeline metrics."""
        events = self.message_bus.get_event_log()
        event_counts = {}

        for event in events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        completed_batches = [bid for bid, state in self.batch_states.items()
                           if state.get("stage") == "curation.completed"]

        return {
            "total_events": len(events),
            "event_counts": event_counts,
            "active_batches": len(self.batch_states),
            "completed_batches": len(completed_batches),
            "completion_rate": len(completed_batches) / len(self.batch_states) if self.batch_states else 0
        }



def main():
    """Run the orchestrator with real data."""
    print("🎬 Photo Curation Pipeline Orchestrator")
    print("=" * 50)

    # Create necessary directories
    import os
    os.makedirs("./data/input", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./data/features", exist_ok=True)
    os.makedirs("./data/themes", exist_ok=True)
    os.makedirs("./data/output", exist_ok=True)
    os.makedirs("./data/cache", exist_ok=True)

    # Create intermediate JSONs directories
    os.makedirs("./intermediateJsons", exist_ok=True)
    for step in ["ingest", "preprocess", "features", "scoring", "clustering", "ranking", "optimizer", "exporter"]:
        os.makedirs(f"./intermediateJsons/{step}", exist_ok=True)

    # Create rankingInput directory
    os.makedirs("./data/rankingInput", exist_ok=True)

    orchestrator = Orchestrator()

    # Start the batch - it will auto-generate ingest input and batch_id if needed
    print("🚀 Starting batch (auto-generating batch_id and ingest input)")
    batch_id = orchestrator.start_batch()

    if not batch_id:
        print("❌ Failed to start batch")
        return

    # Wait for completion
    print("⏳ Processing...")
    time.sleep(5)  # Give it more time to complete

    # Check final status
    status = orchestrator.get_batch_status(batch_id)
    print("\n📊 Final Batch Status:")
    print(f"  Batch ID: {status['batch_id']}")
    print(f"  Stage: {status['stage']}")
    print(f"  Events: {status['events']}")
    print(f"  Last Event: {status['last_event']}")

    # Save final result
    services = orchestrator._get_services()
    if hasattr(services, 'data_store') and batch_id in services.data_store:
        final_result = services.data_store[batch_id].get("curated_list")
        if final_result:
            with open("./data/output/curated_list.json", 'w') as f:
                json.dump(final_result, f, indent=2)
            print("\n💾 Final curated list saved to ./data/output/curated_list.json")
            print(f"Selected {len(final_result['items'])} photos")
        else:
            print("⚠️ No curated list found in results")

    print(f"\n🎉 Pipeline complete: {batch_id}")


if __name__ == "__main__":
    main()
