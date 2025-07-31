import os
import zipfile
from pathlib import Path
from urllib import request
from src.BloodCellClassifier.entity.config_entity import DataIngestionConfig
from BloodCellClassifier import logger
from src.BloodCellClassifier.utils.common import *

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> bool:
        """Downloads the file from source_URL to local_data_file if not present.
        Returns True if downloaded, False if already exists."""
        os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
        
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading data from {self.config.source_URL}...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"Download completed to: {filename}")
            logger.debug(f"Download headers: {headers}")
            return True
        else:
            logger.info(f"File already exists at {self.config.local_data_file}, size: {get_size(Path(self.config.local_data_file))}")
            return False

    def extract_zip_file(self):
        """Extracts the downloaded zip file to unzip_dir"""
        logger.info(f"Extracting zip file from {self.config.local_data_file} to {self.config.unzip_dir}")
        os.makedirs(self.config.unzip_dir, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)

        logger.info(f"Successfully extracted to {self.config.unzip_dir}")

    def run(self):
        """Skip entire ingestion if zip file already exists"""
        if not os.path.exists(self.config.local_data_file):
            downloaded = self.download_file()
            if downloaded:
                self.extract_zip_file()
        else:
            logger.info(f"File already exists at {self.config.local_data_file}, skipping download and extraction.")
