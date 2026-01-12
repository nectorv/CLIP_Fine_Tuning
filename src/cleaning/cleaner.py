"""
Batch cleaning script using GPT-5 nano VLM via OpenAI Batch API.

This module processes furniture images and titles using vision-language model
to extract structured product information. Uses OpenAI Batch API for cost efficiency.
"""

import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from src.config import CleaningConfig, DataPaths


# ============================================================================
# Pydantic Schema for Response Validation
# ============================================================================
class CleanedProduct(BaseModel):
    """Schema for cleaned product data from GPT-5 nano."""
    
    clean_title: str = Field(..., description="Cleaned product title")
    style: str = Field(..., description="Furniture style")
    material: str = Field(..., description="Primary material")
    color: str = Field(..., description="Primary color")
    object_type: str = Field(..., description="Type of furniture")


# ============================================================================
# Image Processing Utilities
# ============================================================================
def resize_image_with_padding(
    image_path: Path,
    target_size: int = 512
) -> Optional[Image.Image]:
    """
    Resize image to target size while maintaining aspect ratio with padding.
    
    Args:
        image_path: Path to image file
        target_size: Target size (square)
        
    Returns:
        Resized PIL Image, or None if error
    """
    try:
        img = Image.open(image_path).convert("RGB")
        
        # Calculate scaling to fit within target_size while maintaining aspect ratio
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste centered
        new_img = Image.new("RGB", (target_size, target_size), color="white")
        paste_x = (target_size - img.width) // 2
        paste_y = (target_size - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string (data URL format)
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


# ============================================================================
# Batch Preparation
# ============================================================================
def prepare_batch_jsonl(
    csv_path: Path,
    image_dir: Path,
    output_dir: Path,  # Changed from output_path to output_dir
    title_column: str = "name",
    image_column: str = "local_path",
    incremental: bool = True 
) -> int:
    """
    Prepare SPLIT JSONL files (max 95MB each) for OpenAI Batch API.
    """
    print("Reading CSV file...")
    print("CSV loaded")
    df = pd.read_csv(csv_path)
    
    # 1. Setup Constraints
    MAX_BYTES = 90 * 1024 * 1024  # 90 MB safety limit (OpenAI max is 100MB)
    MAX_REQUESTS = 49000          # 49k safety limit (OpenAI max is 50k)
    
    # 2. Check Existing IDs (across ALL jsonl files in directory)
    existing_ids = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if incremental:
        print("Scanning existing batch files for processed IDs...")
        for existing_file in output_dir.glob("batch_input_*.jsonl"):
            with open(existing_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        existing_ids.add(data.get("custom_id"))
                    except: continue
        print(f"Found {len(existing_ids)} existing requests. Skipping these.")

    # 3. Initialization
    requests_created = 0
    failed_count = 0
    
    file_index = 1
    current_file_path = output_dir / f"batch_input_{file_index:03d}.jsonl"
    current_file_size = 0
    current_file_count = 0
    
    # If file exists and we are appending, check its size first
    if current_file_path.exists():
        current_file_size = current_file_path.stat().st_size
        # Count lines roughly or just start new file if it's close to limit
        if current_file_size > MAX_BYTES:
            file_index += 1
            current_file_path = output_dir / f"batch_input_{file_index:03d}.jsonl"
            current_file_size = 0

    from src.cleaning.prompts import get_cleaning_prompt

    print(f"\nProcessing images. Current batch file: {current_file_path.name}")
    
    # Open the first file handle
    f = open(current_file_path, "a" if incremental else "w", encoding="utf-8")

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing batch"):
            # SKIP if already done
            if str(idx) in existing_ids:
                continue
            
            #to delete
            if requests_created >= 10:
                print("🚀 Smoke test limit reached. Stopping prepare...")
                break

            # --- Image Processing ---
            raw_title = str(row.get(title_column, "")).strip() or "Unknown Item"
            img_name = Path(str(row.get(image_column, f"{idx}.jpg"))).name
            img_path = Path(image_dir) / img_name
            
            if not img_path.exists():
                failed_count += 1
                continue
            
            resized_img = resize_image_with_padding(img_path, CleaningConfig.IMAGE_RESIZE_DIM)
            if resized_img is None:
                failed_count += 1
                continue
            
            img_base64 = image_to_base64(resized_img)
            user_prompt = get_cleaning_prompt(raw_title)
            
            # --- Create Request ---
            request = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CleaningConfig.MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "You are a helpful furniture cataloger with expertise in visual product analysis that outputs valid JSON."},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": img_base64, "detail": "low"}}
                        ]}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": 2000
                }
            }
            
            json_line = json.dumps(request) + "\n"
            line_size = len(json_line.encode('utf-8'))
            
            # --- SPLITTING LOGIC ---
            # If adding this line exceeds size OR count limit, rotate file
            if (current_file_size + line_size > MAX_BYTES) or (current_file_count >= MAX_REQUESTS):
                f.close()
                print(f"  -> File full ({current_file_size/1024/1024:.2f} MB). Rotating...")
                
                file_index += 1
                current_file_path = output_dir / f"batch_input_{file_index:03d}.jsonl"
                f = open(current_file_path, "w", encoding="utf-8")
                current_file_size = 0
                current_file_count = 0
            
            # Write data
            f.write(json_line)
            current_file_size += line_size
            current_file_count += 1
            requests_created += 1

    finally:
        f.close()

    print(f"\n✓ Preparation complete.")
    print(f"  Total requests created: {requests_created}")
    print(f"  Total files generated: {file_index}")
    print(f"  Failed images: {failed_count}")
    
    return requests_created


# ============================================================================
# Batch API Management
# ============================================================================
def upload_batch(input_path: Path) -> Optional[str]:
    """
    Upload JSONL file to OpenAI Batch API.
    
    Args:
        input_path: Path to JSONL file
        
    Returns:
        Batch ID if successful, None otherwise
    """
    if not CleaningConfig.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    
    client = OpenAI(api_key=CleaningConfig.OPENAI_API_KEY)
    
    print(f"Uploading batch file: {input_path}")
    
    try:
        # Upload file
        with open(input_path, "rb") as f:
            uploaded_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        print(f"✓ File uploaded: {uploaded_file.id}")
        
        # Create batch
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        print(f"✓ Batch created: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Request counts: {batch.request_counts}")
        
        return batch.id
    
    except Exception as e:
        print(f"Error uploading batch: {e}")
        return None


def check_status(batch_id: str) -> dict:
    """
    Check the status of a batch job.
    
    Args:
        batch_id: Batch ID from OpenAI
        
    Returns:
        Dictionary with batch status information
    """
    if not CleaningConfig.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found")
    
    client = OpenAI(api_key=CleaningConfig.OPENAI_API_KEY)
    
    try:
        batch = client.batches.retrieve(batch_id)
        if batch.errors:
            print(f"Validation Errors: {batch.errors}")
        return {
            "id": batch.id,
            "status": batch.status,
            "request_counts": batch.request_counts,
            "created_at": batch.created_at,
            "in_progress_at": batch.in_progress_at,
            "expires_at": batch.expires_at,
            "finalizing_at": batch.finalizing_at,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id
        }
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return {}


def wait_for_completion(batch_id: str, max_wait_hours: int = 24) -> bool:
    """
    Poll batch status until completion or timeout.
    
    Args:
        batch_id: Batch ID
        max_wait_hours: Maximum hours to wait
        
    Returns:
        True if completed successfully, False otherwise
    """
    max_wait_seconds = max_wait_hours * 3600
    start_time = time.time()
    poll_interval = CleaningConfig.BATCH_POLL_INTERVAL
    
    print(f"Waiting for batch completion (checking every {poll_interval // 60} minutes)...")
    
    while True:
        status = check_status(batch_id)
        
        if not status:
            print("Failed to get batch status")
            return False
        
        current_status = status.get("status", "unknown")
        print(f"Status: {current_status}")
        
        if current_status == "completed":
            print("✓ Batch completed successfully!")
            return True
        elif current_status == "failed":
            print("✗ Batch failed")
            return False
        elif current_status in ["cancelled", "expired"]:
            print(f"✗ Batch {current_status}")
            return False
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print(f"Timeout: Batch not completed within {max_wait_hours} hours")
            return False
        
        # Wait before next check
        print(f"Waiting {poll_interval // 60} minutes before next check...")
        time.sleep(poll_interval)
    
    return False


# ============================================================================
# Result Processing
# ============================================================================
def download_batch_results(batch_id: str, output_path: Path) -> bool:
    """
    Download batch results from OpenAI.
    
    Args:
        batch_id: Batch ID
        output_path: Path to save results JSONL
        
    Returns:
        True if successful, False otherwise
    """
    if not CleaningConfig.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found")
    
    client = OpenAI(api_key=CleaningConfig.OPENAI_API_KEY)
    
    try:
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            print(f"Batch status is '{batch.status}', not 'completed'")
            return False
        
        if not batch.output_file_id:
            print("No output file ID found")
            return False
        
        print(f"Downloading results from file: {batch.output_file_id}")
        
        # Download file
        file_response = client.files.content(batch.output_file_id)
        
        # Save to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(file_response.read())
        
        print(f"✓ Results downloaded to: {output_path}")
        return True
    
    except Exception as e:
        print(f"Error downloading results: {e}")
        return False


def finalize_results(
    batch_output_path: Path,
    original_csv_path: Path,
    output_csv_path: Path,
    title_column: str = "title",
    image_column: str = "image_path"
) -> pd.DataFrame:
    """
    Parse batch results and merge with original CSV.
    
    Args:
        batch_output_path: Path to batch output JSONL
        original_csv_path: Path to original CSV
        output_csv_path: Path to save final cleaned CSV
        title_column: Column name for titles
        image_column: Column name for images
        
    Returns:
        DataFrame with cleaned data
    """
    print("Reading original CSV...")
    df = pd.read_csv(original_csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Initialize cleaned columns
    # Use 'cleaned_title' for compatibility with embedding/training pipeline
    for col in ["cleaned_title", "clean_title", "style", "material", "color", "object_type"]:
        df[col] = None
    
    print("\nParsing batch results...")
    parsed_count = 0
    error_count = 0
    
    with open(batch_output_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing results"):
            if not line.strip():
                continue
            
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                
                if custom_id is None:
                    continue
                
                idx = int(custom_id)
                
                # Check if response is successful
                response = result.get("response", {})
                if response.get("status_code") != 200:
                    error_count += 1
                    continue
                
                # Parse response body
                body = response.get("body", {})
                choices = body.get("choices", [])
                
                if not choices:
                    error_count += 1
                    continue
                
                content = choices[0].get("message", {}).get("content", "")
                
                if not content:
                    error_count += 1
                    continue
                
                # Parse JSON response
                try:
                    cleaned_data = json.loads(content)
                except json.JSONDecodeError:
                    error_count += 1
                    continue
                
                # Validate with Pydantic
                try:
                    validated = CleanedProduct(**cleaned_data)
                    
                    # Update DataFrame
                    if idx < len(df):
                        # Use 'cleaned_title' for compatibility with embedding/training pipeline
                        df.at[idx, "cleaned_title"] = validated.clean_title
                        # Also keep 'clean_title' for reference
                        df.at[idx, "clean_title"] = validated.clean_title
                        df.at[idx, "style"] = validated.style
                        df.at[idx, "material"] = validated.material
                        df.at[idx, "color"] = validated.color
                        df.at[idx, "object_type"] = validated.object_type
                        parsed_count += 1
                
                except ValidationError as e:
                    print(f"Validation error for row {idx}: {e}")
                    error_count += 1
                    continue
            
            except Exception as e:
                print(f"Error parsing line: {e}")
                error_count += 1
                continue
    
    print(f"\n✓ Parsed {parsed_count} successful results")
    print(f"  Errors: {error_count}")
    print(f"  Missing: {len(df) - parsed_count - error_count}")
    
    # Save cleaned CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Cleaned CSV saved to: {output_csv_path}")
    
    return df


# ============================================================================
# Main Entry Points
# ============================================================================
def main():
    """Main entry point with CLI arguments."""
    
    parser = argparse.ArgumentParser(
        description="Clean furniture data using GPT-5 nano VLM via Batch API"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare batch JSONL file from CSV and images"
    )
    parser.add_argument(
        "--incremental", action="store_true", default=True, 
        help="Skip items already present in the JSONL file"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload batch file to OpenAI and start processing"
    )
    parser.add_argument(
        "--check",
        type=str,
        metavar="BATCH_ID",
        help="Check status of a batch job"
    )
    parser.add_argument(
        "--wait",
        type=str,
        metavar="BATCH_ID",
        help="Wait for batch completion"
    )
    parser.add_argument(
        "--finalize",
        type=str,
        metavar="BATCH_ID",
        help="Download and process batch results"
    )
    parser.add_argument(
        "--finalize-file",
        type=str,
        metavar="PATH",
        help="Process results from a downloaded JSONL file"
    )
    
    args = parser.parse_args()
    
    args = parser.parse_args()

    if args.prepare:
        print("="*60)
        print(f"Preparing Batch JSONL {'(Incremental)' if args.incremental else '(Overwrite)'}")
        print("="*60)
        
        count = prepare_batch_jsonl(
        csv_path=DataPaths.RAW_CSV_PATH,
        image_dir=DataPaths.IMAGE_DIR,
        output_dir=CleaningConfig.BATCH_INPUT_PATH.parent, 
        incremental=args.incremental 
    )
        
        print(f"\n✓ Preparation complete. Created {count} requests.")
        print(f"  Next step: python -m src.cleaning.cleaner --upload")
    
    elif args.upload:
        print("="*60)
        print("Uploading Batches to OpenAI")
        print("="*60)
        
        input_dir = CleaningConfig.BATCH_INPUT_PATH.parent
        
        if not input_dir.exists():
            print(f"Directory not found: {input_dir}")
            return

        batch_files = sorted(input_dir.glob("batch_input_*.jsonl"))
        
        if not batch_files:
            print(f"No batch files found in {input_dir}")
            return

        # Define your log file path
        history_file = input_dir / "batch_history.jsonl"

        for file_path in batch_files:
            print(f"\nProcessing: {file_path.name}...")
            
            # Optional: Check history to avoid double-uploading
            # (You can implement a check here reading history_file if needed)
            
            batch_id = upload_batch(file_path)
            
            if batch_id:
                print(f"Started Batch ID: {batch_id}")
                
                # --- SAVE LOGIC STARTS HERE ---
                log_entry = {
                    "batch_id": batch_id,
                    "file_name": file_path.name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "in_progress"
                }
                
                with open(history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                print(f"  -> Saved to log: {history_file.name}")
    
    elif args.check:
        print("="*60)
        print("Checking Batch Status")
        print("="*60)
        
        status = check_status(args.check)
        if status:
            print(f"\nBatch ID: {status.get('id')}")
            print(f"Status: {status.get('status')}")
            print(f"Request counts: {status.get('request_counts')}")
            if status.get('output_file_id'):
                print(f"Output file ID: {status.get('output_file_id')}")
    
    elif args.wait:
        print("="*60)
        print("Waiting for Batch Completion")
        print("="*60)
        
        completed = wait_for_completion(args.wait)
        if completed:
            print(f"\n✓ Batch completed! Finalize with:")
            print(f"  python -m src.cleaning.cleaner --finalize {args.wait}")
    
    elif args.finalize:
        print("="*60)
        print("Finalizing Batch Results")
        print("="*60)
        
        # Download results
        downloaded = download_batch_results(
            args.finalize,
            CleaningConfig.BATCH_OUTPUT_PATH
        )
        
        if downloaded:
            # Process results
            finalize_results(
                batch_output_path=CleaningConfig.BATCH_OUTPUT_PATH,
                original_csv_path=DataPaths.RAW_CSV_PATH,
                output_csv_path=CleaningConfig.CLEANED_CSV_PATH
            )
    
    elif args.finalize_file:
        print("="*60)
        print("Processing Results from File")
        print("="*60)
        
        finalize_results(
            batch_output_path=Path(args.finalize_file),
            original_csv_path=DataPaths.RAW_CSV_PATH,
            output_csv_path=CleaningConfig.CLEANED_CSV_PATH
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
