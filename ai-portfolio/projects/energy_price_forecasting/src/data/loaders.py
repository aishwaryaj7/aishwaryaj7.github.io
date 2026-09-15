"""
Data loading module for Energy Price Forecasting project.
Handles Google Cloud Storage operations for data persistence.
"""
import pandas as pd
import numpy as np
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import io
from ..utils.config import config
from ..utils.helpers import setup_logging

logger = setup_logging()

class GCSDataLoader:
    """
    Google Cloud Storage data loader for energy forecasting data.
    """
    
    def __init__(self, bucket_name: str = None, project_id: str = None):
        self.bucket_name = bucket_name or config.gcp.bucket_name
        self.project_id = project_id or config.gcp.project_id
        self.client = None
        self.bucket = None
        
        try:
            self._initialize_client()
        except Exception as e:
            logger.warning(f"Could not initialize GCS client: {e}")
            logger.info("GCS operations will be simulated locally")
    
    def _initialize_client(self):
        """Initialize Google Cloud Storage client."""
        try:
            if config.gcp.credentials_path and Path(config.gcp.credentials_path).exists():
                self.client = storage.Client.from_service_account_json(
                    config.gcp.credentials_path,
                    project=self.project_id
                )
            else:
                # Try to use default credentials
                self.client = storage.Client(project=self.project_id)
            
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"Successfully initialized GCS client for bucket: {self.bucket_name}")
            
        except DefaultCredentialsError:
            logger.warning("No valid GCS credentials found. Using local storage simulation.")
            self.client = None
            self.bucket = None
        except Exception as e:
            logger.error(f"Error initializing GCS client: {e}")
            self.client = None
            self.bucket = None
    
    def save_raw_data(self, data: pd.DataFrame, data_type: str, country: str, 
                     timestamp: datetime = None) -> str:
        """
        Save raw data to cloud storage.
        
        Args:
            data: DataFrame to save
            data_type: Type of data (prices, load, renewable, weather)
            country: Country code
            timestamp: Timestamp for the data (defaults to now)
            
        Returns:
            Path where data was saved
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create file path
        date_str = timestamp.strftime("%Y/%m/%d")
        filename = f"raw/{data_type}/{country}/{date_str}/{data_type}_{country}_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
        
        try:
            if self.client and self.bucket:
                # Save to GCS
                return self._save_to_gcs(data, filename)
            else:
                # Save locally as fallback
                return self._save_locally(data, filename)
                
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
            # Fallback to local storage
            return self._save_locally(data, filename)
    
    def save_processed_data(self, data: pd.DataFrame, processing_stage: str, 
                          timestamp: datetime = None) -> str:
        """
        Save processed data to cloud storage.
        
        Args:
            data: Processed DataFrame
            processing_stage: Stage of processing (cleaned, featured, final)
            timestamp: Timestamp for the data
            
        Returns:
            Path where data was saved
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        date_str = timestamp.strftime("%Y/%m/%d")
        filename = f"processed/{processing_stage}/{date_str}/processed_{processing_stage}_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
        
        try:
            if self.client and self.bucket:
                return self._save_to_gcs(data, filename)
            else:
                return self._save_locally(data, filename)
                
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return self._save_locally(data, filename)
    
    def load_raw_data(self, data_type: str, country: str, start_date: datetime, 
                     end_date: datetime) -> pd.DataFrame:
        """
        Load raw data from cloud storage for a date range.
        
        Args:
            data_type: Type of data to load
            country: Country code
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Combined DataFrame with all data in the date range
        """
        try:
            if self.client and self.bucket:
                return self._load_from_gcs_range(data_type, country, start_date, end_date)
            else:
                return self._load_locally_range(data_type, country, start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return pd.DataFrame()
    
    def load_latest_processed_data(self, processing_stage: str = "final") -> pd.DataFrame:
        """
        Load the latest processed data.
        
        Args:
            processing_stage: Stage of processing to load
            
        Returns:
            Latest processed DataFrame
        """
        try:
            if self.client and self.bucket:
                return self._load_latest_from_gcs(f"processed/{processing_stage}")
            else:
                return self._load_latest_locally(f"processed/{processing_stage}")
                
        except Exception as e:
            logger.error(f"Error loading latest processed data: {e}")
            return pd.DataFrame()
    
    def save_model_artifacts(self, artifacts: Dict, model_name: str, 
                           version: str = None) -> str:
        """
        Save model artifacts (model, scalers, etc.) to cloud storage.
        
        Args:
            artifacts: Dictionary containing model artifacts
            model_name: Name of the model
            version: Model version (defaults to timestamp)
            
        Returns:
            Path where artifacts were saved
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"models/{model_name}/v{version}/artifacts.pkl"
        
        try:
            if self.client and self.bucket:
                return self._save_artifacts_to_gcs(artifacts, filename)
            else:
                return self._save_artifacts_locally(artifacts, filename)
                
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            return self._save_artifacts_locally(artifacts, filename)
    
    def load_model_artifacts(self, model_name: str, version: str = "latest") -> Dict:
        """
        Load model artifacts from cloud storage.
        
        Args:
            model_name: Name of the model
            version: Model version to load
            
        Returns:
            Dictionary containing model artifacts
        """
        try:
            if version == "latest":
                # Find latest version
                if self.client and self.bucket:
                    version = self._get_latest_model_version_gcs(model_name)
                else:
                    version = self._get_latest_model_version_local(model_name)
            
            filename = f"models/{model_name}/v{version}/artifacts.pkl"
            
            if self.client and self.bucket:
                return self._load_artifacts_from_gcs(filename)
            else:
                return self._load_artifacts_locally(filename)
                
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            return {}
    
    def _save_to_gcs(self, data: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to Google Cloud Storage."""
        blob = self.bucket.blob(filename)
        
        # Convert DataFrame to parquet bytes
        buffer = io.BytesIO()
        data.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        gcs_path = f"gs://{self.bucket_name}/{filename}"
        logger.info(f"Saved data to GCS: {gcs_path}")
        return gcs_path
    
    def _save_locally(self, data: pd.DataFrame, filename: str) -> str:
        """Save DataFrame locally as fallback."""
        local_path = Path("data") / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(local_path, index=False)
        
        logger.info(f"Saved data locally: {local_path}")
        return str(local_path)
    
    def _load_from_gcs_range(self, data_type: str, country: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load data from GCS for a date range."""
        prefix = f"raw/{data_type}/{country}/"
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        
        dataframes = []
        for blob in blobs:
            # Check if blob is within date range (simplified check)
            blob_date_str = blob.name.split('/')[-2]  # Extract date from path
            try:
                blob_date = datetime.strptime(blob_date_str, "%Y/%m/%d")
                if start_date <= blob_date <= end_date:
                    buffer = io.BytesIO()
                    blob.download_to_file(buffer)
                    buffer.seek(0)
                    df = pd.read_parquet(buffer)
                    dataframes.append(df)
            except ValueError:
                continue
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} records from GCS")
            return combined_df
        else:
            return pd.DataFrame()
    
    def _load_locally_range(self, data_type: str, country: str, 
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load data locally for a date range."""
        data_path = Path("data") / "raw" / data_type / country
        
        if not data_path.exists():
            return pd.DataFrame()
        
        dataframes = []
        for file_path in data_path.rglob("*.parquet"):
            try:
                df = pd.read_parquet(file_path)
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} records locally")
            return combined_df
        else:
            return pd.DataFrame()
    
    def _save_artifacts_to_gcs(self, artifacts: Dict, filename: str) -> str:
        """Save model artifacts to GCS."""
        blob = self.bucket.blob(filename)
        
        buffer = io.BytesIO()
        pickle.dump(artifacts, buffer)
        buffer.seek(0)
        
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        gcs_path = f"gs://{self.bucket_name}/{filename}"
        logger.info(f"Saved model artifacts to GCS: {gcs_path}")
        return gcs_path
    
    def _save_artifacts_locally(self, artifacts: Dict, filename: str) -> str:
        """Save model artifacts locally."""
        local_path = Path("data") / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Saved model artifacts locally: {local_path}")
        return str(local_path)
    
    def _load_latest_from_gcs(self, prefix: str) -> pd.DataFrame:
        """Load latest file from GCS with given prefix."""
        blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))
        
        if not blobs:
            return pd.DataFrame()
        
        # Sort by creation time and get latest
        latest_blob = max(blobs, key=lambda x: x.time_created)
        
        buffer = io.BytesIO()
        latest_blob.download_to_file(buffer)
        buffer.seek(0)
        
        df = pd.read_parquet(buffer)
        logger.info(f"Loaded latest data from GCS: {latest_blob.name}")
        return df
    
    def _load_latest_locally(self, prefix: str) -> pd.DataFrame:
        """Load latest file locally with given prefix."""
        data_path = Path("data") / prefix
        
        if not data_path.exists():
            return pd.DataFrame()
        
        parquet_files = list(data_path.rglob("*.parquet"))
        
        if not parquet_files:
            return pd.DataFrame()
        
        # Sort by modification time and get latest
        latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        
        df = pd.read_parquet(latest_file)
        logger.info(f"Loaded latest data locally: {latest_file}")
        return df
