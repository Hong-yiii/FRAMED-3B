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
        self.services = MockServices()
        self.generator = MockDataGenerator()
        self.batch_states: Dict[str, Dict[str, Any]] = {}
        self.setup_event_subscriptions()

    def setup_event_subscriptions(self):
        """Set up event subscriptions for the MVP pipeline flow."""

        # Ingest → Combined Process & Features
        self.message_bus.subscribe("ingest.completed", self.handle_ingest_completed)

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
        """Handle ingest completion and trigger combined process & features."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Ingest completed for {batch_id}, triggering process & features...")
        self.batch_states[batch_id] = {"stage": "ingest.completed"}

        # Trigger combined process & features service
        features_output = self.services.process_features_service(data)
        self.message_bus.publish("features.completed", features_output)

    def handle_features_completed(self, event: Dict[str, Any]):
        """Handle features completion and trigger scoring."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Features completed for {batch_id}, triggering scoring...")
        self.batch_states[batch_id]["stage"] = "features.completed"

        # Get theme spec (in real system, this would come from a service)
        theme_spec = self.generator.generate_theme_spec()

        # Trigger scoring service
        score_output = self.services.scoring_service(data, theme_spec)
        self.message_bus.publish("score.completed", score_output)

    def handle_score_completed(self, event: Dict[str, Any]):
        """Handle scoring completion and trigger clustering."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Scoring completed for {batch_id}, triggering clustering...")
        self.batch_states[batch_id]["stage"] = "score.completed"

        # Trigger clustering service
        cluster_output = self.services.clustering_service(data)
        self.message_bus.publish("cluster.completed", cluster_output)

    def handle_cluster_completed(self, event: Dict[str, Any]):
        """Handle clustering completion and trigger cluster ranking."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Clustering completed for {batch_id}, triggering cluster ranking...")
        self.batch_states[batch_id]["stage"] = "cluster.completed"

        # Get required data
        score_output = self.services.data_store[batch_id]["score_output"]
        theme_spec = self.generator.generate_theme_spec()

        # Trigger cluster ranking service
        cluster_rank_output = self.services.cluster_ranking_service(data, theme_spec)
        self.message_bus.publish("cluster.rank.completed", cluster_rank_output)

    def handle_cluster_rank_completed(self, event: Dict[str, Any]):
        """Handle cluster ranking completion and trigger optimization."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Ranking completed for {batch_id}, triggering optimization...")
        self.batch_states[batch_id]["stage"] = "cluster.rank.completed"

        # Get theme spec
        theme_spec = self.generator.generate_theme_spec()

        # Trigger combined optimizer service
        selection_output = self.services.optimizer_service(data, theme_spec)
        self.message_bus.publish("selection.completed", selection_output)

    def handle_selection_completed(self, event: Dict[str, Any]):
        """Handle selection completion and trigger export."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Selection completed for {batch_id}, triggering export...")
        self.batch_states[batch_id]["stage"] = "selection.completed"

        # Get theme spec
        theme_spec = self.generator.generate_theme_spec()

        # Trigger exporter service
        curated_list = self.services.exporter_service(data, theme_spec)
        self.message_bus.publish("curation.completed", curated_list)

        # Mark batch as completed
        self.batch_states[batch_id]["stage"] = "curation.completed"
        self.batch_states[batch_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

        print(f"🎉 Orchestrator: Curation pipeline completed for {batch_id}!")

    def handle_review_update(self, event: Dict[str, Any]):
        """Handle human review updates and re-run optimization."""
        data = event["data"]
        batch_id = data["batch_id"]

        print(f"🎯 Orchestrator: Review update received for {batch_id}, re-running optimization...")

        # Get current selection
        current_selection = self.services.data_store[batch_id]["selection_output"]

        # Apply review constraints (in real system, this would modify the optimizer input)
        # For now, just re-run with same data
        theme_spec = self.generator.generate_theme_spec()
        new_selection = self.services.optimizer_service(current_selection, theme_spec)

        self.message_bus.publish("selection.completed", new_selection)

    def start_batch(self, ingest_input: Dict[str, Any]) -> str:
        """Start a new curation batch."""
        batch_id = ingest_input["batch_id"]

        print(f"🚀 Orchestrator: Starting new batch {batch_id}")
        self.batch_states[batch_id] = {"stage": "starting", "started_at": datetime.now(timezone.utc).isoformat()}

        # Trigger initial ingest service
        ingest_output = self.services.ingest_service(ingest_input)
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


def demo_orchestrator():
    """Demonstrate the MVP event-driven orchestrator."""
    print("🎬 MVP Event-Driven Orchestrator Demo")
    print("=" * 50)

    # Create local data directories
    import os
    os.makedirs("./data/input", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./data/thumbs", exist_ok=True)
    os.makedirs("./data/features", exist_ok=True)
    os.makedirs("./data/themes", exist_ok=True)
    os.makedirs("./data/output", exist_ok=True)
    os.makedirs("./data/cache", exist_ok=True)

    orchestrator = Orchestrator()
    generator = MockDataGenerator()

    # Create test batch
    batch_id = "batch_mvp_demo_001"
    ingest_input = generator.generate_ingest_input(batch_id, num_photos=30)

    # Start orchestration
    orchestrator.start_batch(ingest_input)

    # Wait for completion (in real system, this would be asynchronous)
    time.sleep(1)

    # Check final status
    status = orchestrator.get_batch_status(batch_id)
    print("\n📊 Final Batch Status:")
    print(f"  Batch ID: {status['batch_id']}")
    print(f"  Stage: {status['stage']}")
    print(f"  Events: {status['events']}")
    print(f"  Last Event: {status['last_event']}")

    # Show pipeline metrics
    metrics = orchestrator.get_pipeline_metrics()
    print("\n📈 Pipeline Metrics:")
    for key, value in metrics.items():
        if key == "event_counts":
            print(f"  {key}:")
            for event_type, count in value.items():
                print(f"    {event_type}: {count}")
        else:
            print(f"  {key}: {value}")

    # Save final result
    if batch_id in orchestrator.services.data_store:
        final_result = orchestrator.services.data_store[batch_id].get("curated_list")
        if final_result:
            with open("./data/output/curated_list.json", 'w') as f:
                json.dump(final_result, f, indent=2)
            print("\n💾 Final curated list saved to ./data/output/curated_list.json")
            print(f"Selected {len(final_result['items'])} photos")


if __name__ == "__main__":
    demo_orchestrator()
