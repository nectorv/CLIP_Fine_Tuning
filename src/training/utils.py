import os
import shutil
import tarfile
import boto3
import torch
from tqdm import tqdm

from src.config import TrainingRunConfig

class S3Manager:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def download_dataset(self, s3_prefix, local_dir):
        """Syncs S3 folder to local directory efficiently"""
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        
        # Note: For 50k files, 'aws s3 sync' via subprocess is faster than boto3 loop
        print(f"🔄 Syncing data from s3://{self.bucket_name}/{s3_prefix} to {local_dir}...")
        aws_cli = shutil.which("aws")
        if not aws_cli:
            print("⚠️ AWS CLI not found in PATH. Install/configure it, or provide a local dataset path.")
            return

        exit_code = os.system(f"{aws_cli} s3 sync s3://{self.bucket_name}/{s3_prefix} {local_dir}")
        print(f"✅ Sync complete. Exit code: {exit_code}")

        image_count = self._count_images(local_dir)
        print(f"📦 Local image count after sync: {image_count}")

        if image_count == 0:
            extracted = self._extract_shards(local_dir)
            if extracted > 0:
                image_count = self._count_images(local_dir)
                print(f"📦 Local image count after extraction: {image_count}")

    def upload_checkpoint(self, local_path, s3_path):
        """Uploads a checkpoint to S3"""
        print(f"⬆️ Uploading checkpoint to s3://{self.bucket_name}/{s3_path}")
        self.s3.upload_file(local_path, self.bucket_name, s3_path)

    def download_checkpoint(self, s3_path, local_path):
        """Downloads a checkpoint from S3 if it exists"""
        try:
            self.s3.download_file(self.bucket_name, s3_path, local_path)
            print(f"⬇️ Checkpoint downloaded from s3://{self.bucket_name}/{s3_path}")
            return True
        except Exception:
            print("⚠️ No checkpoint found in S3 to resume from.")
            return False

    def _count_images(self, local_dir):
        image_count = 0
        for root, _, files in os.walk(local_dir):
            image_count += sum(1 for file in files if file.endswith(("jpg", "jpeg", "png")))
        return image_count

    def _extract_shards(self, local_dir):
        tar_paths = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                if file.endswith(".tar"):
                    tar_paths.append(os.path.join(root, file))

        if not tar_paths:
            print("⚠️ No .tar shards found to extract.")
            return 0

        print(f"📦 Found {len(tar_paths)} tar shards. Extracting...")
        extracted = 0
        for tar_path in tar_paths:
            try:
                # Extract each shard into its own split directory (train/validation/test)
                target_dir = os.path.dirname(tar_path)
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(path=target_dir)
                extracted += 1
            except Exception as e:
                print(f"⚠️ Failed to extract {tar_path}: {e}")

        print(f"✅ Extracted {extracted}/{len(tar_paths)} shards")
        return extracted

class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def find_max_batch_size(model, dataset, device="cuda"):
    """Heuristic to find max physical batch size"""
    if len(dataset) == 0:
        print("⚠️ Empty dataset detected. Skipping batch size search.")
        return 32
    print("🔍 Searching for max batch size...", flush=True)
    batch_size = TrainingRunConfig.BATCH_SEARCH_START
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Dummy optimizer
    
    model.train()
    try:
        while True:
            # Try a forward/backward pass
            try:
                print(f"🧪 Trying batch size {batch_size}...", flush=True)
                dummy_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                batch = next(iter(dummy_loader))
                
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"✅ Batch size {batch_size} OK.", flush=True)
                batch_size *= 2
                
                if batch_size > TrainingRunConfig.BATCH_SEARCH_MAX: # Safety cap
                    break
            except torch.cuda.OutOfMemoryError:
                print(f"💥 OOM at batch size {batch_size}.", flush=True)
                torch.cuda.empty_cache()
                batch_size //= 2
                break
    except Exception as e:
        print(f"⚠️ Error during batch search: {e}", flush=True)
        batch_size = TrainingRunConfig.BATCH_SEARCH_FALLBACK # Fallback
    
    final_batch = max(TrainingRunConfig.BATCH_SEARCH_MIN, batch_size) # Ensure reasonable min
    print(f"🎯 Max physical batch size set to: {final_batch}", flush=True)
    return final_batch