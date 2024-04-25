from pathlib import Path
from datasets import load_dataset
from ..entity import DataIngestionConfig
from ..logging import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self):
        if not self.config.dataset_name:
            logger.error("Dataset name not provided.")
            return None
        
        try:
            dataset_dir = Path(self.config.local_data_file)
            if dataset_dir.exists() and list(dataset_dir.glob("*")):
                logger.info(f"Dataset files already exist in {self.config.local_data_file}. Skipping downloading.")
                return load_dataset(self.config.dataset_name, data_dir=self.config.local_data_file)
            else:
                logger.info(f"Downloading dataset '{self.config.dataset_name}' from Hugging Face...")
                dataset = load_dataset(self.config.dataset_name)
                dataset.save_to_disk(self.config.local_data_file)
                logger.info(f"Dataset '{self.config.dataset_name}' downloaded and saved to {self.config.local_data_file}")
                return dataset
        except Exception as e:
            logger.error(f"Failed to download dataset '{self.config.dataset_name}'. Error: {str(e)}")
            return None
