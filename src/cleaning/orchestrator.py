import time
import json
import pandas as pd
from tqdm import tqdm
from src.config import CleaningConfig
from src.cleaning.state_manager import StateManager
from src.cleaning.api_client import BatchAPIClient
from src.cleaning.processor import DataProcessor
from src.cleaning.schema import CleanedProduct

class Orchestrator:
    def __init__(self):
        self.state = StateManager()
        self.api = BatchAPIClient()
        self.processor = DataProcessor()

    def run_preparation(self):
        print(">>> STEP 1: Preparing Batch Files")
        self.processor.prepare_jsonl(overwrite=True)

    def run_dispatch(self):
        """
        Uploads files but respects the MAX_CONCURRENT_BATCHES limit.
        This solves the 'Max Token Queued' issue by not sending everything at once.
        """
        print(">>> STEP 2: Dispatching Batches (Queue Management)")
        
        input_files = sorted(list(CleaningConfig.BATCH_INPUT_DIR.glob("batch_input_*.jsonl")))
        
        for file_path in input_files:
            # Check if this file was already successfully processed
            if self.state.is_file_processed(file_path.name):
                print(f"Skipping {file_path.name} (Already processed)")
                continue

            # Check if currently pending/running in state
            pending_in_state = False
            for bid, info in self.state.state["batches"].items():
                if info["input_file"] == file_path.name and info["status"] not in ['completed', 'failed', 'expired']:
                    print(f"Skipping {file_path.name} (Already active in Batch {bid})")
                    pending_in_state = True
                    break
            if pending_in_state: continue

            # --- FLOW CONTROL: Wait for slot ---
            while True:
                active_batches = self.state.get_pending_batches()
                print(active_batches)
                
                # Update status of active batches to see if slots opened up
                real_active_count = 0
                for bid in active_batches:
                    status_obj = self.api.get_batch_status(bid)
                    self.state.update_batch_status(bid, status_obj.status, status_obj.output_file_id)
                    if status_obj.status in ["validating", "in_progress", "finalizing"]:
                        real_active_count += 1
                        print('real_active_count : ',real_active_count)
                
                if real_active_count < CleaningConfig.MAX_CONCURRENT_BATCHES:
                    print(CleaningConfig.MAX_CONCURRENT_BATCHES)
                    break # We have a slot!
                
                print(f"Queue full ({real_active_count} active). Waiting {CleaningConfig.POLL_INTERVAL}s...")
                time.sleep(CleaningConfig.POLL_INTERVAL)

            # --- UPLOAD & EXECUTE ---
            print(f"Uploading {file_path.name}...")
            file_id = self.api.upload_file(file_path)
            batch_id = self.api.create_batch(file_id)
            
            self.state.register_batch(file_path.name, batch_id)
            print(f"Started Batch {batch_id} for {file_path.name}")

    def run_monitoring(self):
        """Polls for completion."""
        print(">>> STEP 3: Monitoring Active Batches")
        while True:
            active_batches = self.state.get_pending_batches()
            if not active_batches:
                print("No active batches remaining.")
                break
            
            print(f"Checking {len(active_batches)} batches...")
            for bid in active_batches:
                status_obj = self.api.get_batch_status(bid)
                self.state.update_batch_status(bid, status_obj.status, status_obj.output_file_id)
                print(f"Batch {bid}: {status_obj.status}")
            
            time.sleep(CleaningConfig.POLL_INTERVAL)

    def run_finalization(self):
        """Downloads results and merges into CSV."""
        print(">>> STEP 4: Finalizing & Merging")
        
        # 1. Download all completed outputs
        for bid, info in self.state.state["batches"].items():
            print(info)
            if info["status"] == "completed" and info.get("output_file_id"):
                local_out = CleaningConfig.BATCH_OUTPUT_DIR / f"output_{bid}.jsonl"
                print(local_out)
                
                if not local_out.exists():
                    print(f"Downloading results for {bid}...")
                    content = self.api.download_results(info["output_file_id"])
                    with open(local_out, "wb") as f:
                        f.write(content)
        
        # 2. Merge logic
        self._merge_results_to_csv()

    def _merge_results_to_csv(self):
        df = pd.read_csv(CleaningConfig.RAW_CSV_PATH)
        
        # Initialize columns
        cols = ["cleaned_title", "style", "material", "color", "object_type"]
        for c in cols:
            if c not in df.columns: df[c] = None

        # Iterate over downloaded JSONL files
        for res_file in CleaningConfig.BATCH_OUTPUT_DIR.glob("*.jsonl"):
            with open(res_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        custom_id = int(data["custom_id"])
                        
                        # Extract content inside the nested OpenAI response structure
                        resp_body = data["response"]["body"]["choices"][0]["message"]["content"]
                        clean_data = json.loads(resp_body)
                        
                        # Use Schema for validation (Automatic Check)
                        validated = CleanedProduct(**clean_data)
                        
                        if custom_id < len(df):
                            df.at[custom_id, "cleaned_title"] = validated.clean_title
                            df.at[custom_id, "style"] = validated.style
                            df.at[custom_id, "material"] = validated.material
                            df.at[custom_id, "color"] = validated.color
                            df.at[custom_id, "object_type"] = validated.object_type
                    except Exception as e:
                        continue # Skip failed lines

        df.to_csv(CleaningConfig.CLEANED_CSV_PATH, index=False)
        print(f"Final CSV saved to {CleaningConfig.CLEANED_CSV_PATH}")

    def run_all(self):
        self.run_preparation()
        self.run_dispatch()
        self.run_monitoring()
        self.run_finalization()