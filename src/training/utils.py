import os
import boto3
import torch
import glob
from tqdm import tqdm

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
        os.system(f"aws s3 sync s3://{self.bucket_name}/{s3_prefix} {local_dir} --quiet")
        print("✅ Sync complete.")

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
        except Exception as e:
            print("⚠️ No checkpoint found in S3 to resume from.")
            return False

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
    print("🔍 Searching for max batch size...")
    batch_size = 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Dummy optimizer
    
    model.train()
    try:
        while True:
            try:
                # Créer un loader temporaire
                dummy_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
                # WDS modif: utiliser next(iter()) au lieu de dataset[...]
                batch = next(iter(dummy_loader))
                
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"✅ Batch size {batch_size} OK.")
                batch_size *= 2
                
                if batch_size > 512: # Safety cap
                    break
            except torch.cuda.OutOfMemoryError:
                print(f"💥 OOM at batch size {batch_size}.")
                torch.cuda.empty_cache()
                batch_size //= 2
                break
    except Exception as e:
         print(f"⚠️ Error during batch search: {e}")
         batch_size = 32 # Fallback
    
    final_batch = max(32, batch_size) # Ensure reasonable min
    print(f"🎯 Max physical batch size set to: {final_batch}")
    return final_batch