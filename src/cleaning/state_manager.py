import json
from typing import Dict, Any
from pathlib import Path
from src.config import CleaningConfig

class StateManager:
    def __init__(self):
        self.state_file = CleaningConfig.STATE_FILE
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"batches": {}, "completed_files": []}

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def register_batch(self, input_filename: str, batch_id: str):
        """Link a file to a batch ID."""
        self.state["batches"][batch_id] = {
            "input_file": input_filename,
            "status": "in_progress",
            "output_file_id": None,
            "local_output_path": None
        }
        self.save_state()

    def update_batch_status(self, batch_id: str, status: str, output_file_id: str = None):
        if batch_id in self.state["batches"]:
            self.state["batches"][batch_id]["status"] = status
            if output_file_id:
                self.state["batches"][batch_id]["output_file_id"] = output_file_id
            self.save_state()

    def get_pending_batches(self):
        return [bid for bid, info in self.state["batches"].items() 
                if info["status"] not in ["completed", "failed", "cancelled", "expired"]]

    def is_file_processed(self, filename: str) -> bool:
        # Check if this filename is associated with a completed batch
        for info in self.state["batches"].values():
            if info["input_file"] == filename and info["status"] == "completed":
                return True
        return False