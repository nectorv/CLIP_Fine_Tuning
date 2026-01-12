from openai import OpenAI
from pathlib import Path
from src.config import CleaningConfig

class BatchAPIClient:
    def __init__(self):
        self.client = OpenAI(api_key=CleaningConfig.OPENAI_API_KEY)

    def upload_file(self, file_path: Path) -> str:
        """Uploads a file and returns the file ID."""
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose="batch")
        return response.id

    def create_batch(self, input_file_id: str) -> str:
        """Creates the batch job and returns Batch ID."""
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch.id

    def get_batch_status(self, batch_id: str):
        return self.client.batches.retrieve(batch_id)

    def download_results(self, output_file_id: str) -> bytes:
        return self.client.files.content(output_file_id).read()