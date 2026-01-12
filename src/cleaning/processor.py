import base64
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

from src.config import CleaningConfig
from src.cleaning.prompts import get_cleaning_prompt

class DataProcessor:
    @staticmethod
    def resize_image(image_path: Path) -> Image.Image:
        try:
            img = Image.open(image_path).convert("RGB")
            target = CleaningConfig.IMAGE_RESIZE_DIM
            
            img.thumbnail((target, target), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", (target, target), color="white")
            
            paste_x = (target - img.width) // 2
            paste_y = (target - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            return new_img
        except Exception as e:
            print(f"Error resizing {image_path}: {e}")
            return None

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    def prepare_jsonl(self, overwrite: bool = True, max_rows: int = None):
        """Reads CSV, processes images, splits into chunks."""
        CleaningConfig.ensure_dirs()
        
        # Overwrite logic: Clean old input files if requested
        if overwrite:
            print("Cleaning old batch input files...")
            for f in CleaningConfig.BATCH_INPUT_DIR.glob("*.jsonl"):
                f.unlink()

        df = pd.read_csv(CleaningConfig.RAW_CSV_PATH)
        print(f"Loaded {len(df)} rows from CSV.")

        if max_rows:
            df = df.head(max_rows)
            print(f"🔥 SMOKE TEST MODE: Limited to {max_rows} rows.")

        current_file_idx = 1
        current_size = 0
        current_count = 0
        
        # Initialize first file
        current_path = CleaningConfig.BATCH_INPUT_DIR / f"batch_input_{current_file_idx:03d}.jsonl"
        f = open(current_path, "w", encoding="utf-8")
        
        requests_created = 0

        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Batches"):
                # Image processing
                img_name = Path(str(row.get("local_path", f"{idx}.jpg"))).name
                img_path = CleaningConfig.IMAGE_DIR / img_name
                
                if not img_path.exists(): 
                    print(img_path, 'doesnt exists')
                    continue

                resized = self.resize_image(img_path)
                if not resized: continue
                
                user_prompt = get_cleaning_prompt(str(row.get("name", "")))
                
                # Payload construction (Preserved your format)
                request = {
                    "custom_id": str(idx),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": CleaningConfig.MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": "You are a helpful furniture cataloger with expertise in design that outputs valid JSON. No reasoning."},
                            {"role": "user", "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": self.image_to_base64(resized), "detail": "low"}}
                            ]}
                        ],
                        "response_format": {"type": "json_object"},
                        "max_completion_tokens": 3000
                    }
                }
                
                line = json.dumps(request) + "\n"
                line_len = len(line.encode('utf-8'))

                # Rotation Logic
                if (current_size + line_len > CleaningConfig.MAX_BATCH_FILE_SIZE) or \
                   (current_count >= CleaningConfig.MAX_REQUESTS_PER_FILE):
                    f.close()
                    current_file_idx += 1
                    current_path = CleaningConfig.BATCH_INPUT_DIR / f"batch_input_{current_file_idx:03d}.jsonl"
                    f = open(current_path, "w", encoding="utf-8")
                    current_size = 0
                    current_count = 0

                f.write(line)
                current_size += line_len
                current_count += 1
                requests_created += 1

        finally:
            f.close()
            print(f"Total requests generated: {requests_created}")