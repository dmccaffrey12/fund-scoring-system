"""
CSV Upload Manager
Handles dynamic CSV uploads and validation
"""

import os
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

class CSVUploadManager:
    """Manages CSV uploads and data validation"""
    
    # Required columns that must be in any uploaded CSV
    REQUIRED_COLUMNS = [
        'Symbol', 'Name', 'Index Fund', 'Category Name', 
        'Net Expense Ratio', 'Share Class Assets Under Management'
    ]
    
    # Optional metric columns
    METRIC_COLUMNS = [
        'Tracking Error (vs Category) (3Y)',
        'Tracking Error (vs Category) (5Y)',
        'Tracking Error (vs Category) (10Y)',
        'R-Squared (vs Category) (5Y)',
        'Downside (vs Category) (5Y)',
        'Downside (vs Category) (10Y)',
        'Max Drawdown (5Y)',
        'Max Drawdown (10Y)',
        'Information Ratio (vs Category) (3Y)',
        'Information Ratio (vs Category) (5Y)',
        'Information Ratio (vs Category) (10Y)',
        'Historical Sortino (3Y)',
        'Historical Sortino (5Y)',
        'Historical Sortino (10Y)',
        'Upside (vs Category) (3Y)',
        'Upside (vs Category) (5Y)',
        'Upside (vs Category) (10Y)',
        '3 Year Total Returns (Daily)',
        '5 Year Total Returns (Daily)',
        '10 Year Total Returns (Daily)',
    ]
    
    # Archive directory for historical CSVs
    ARCHIVE_DIR = 'data_archives'
    
    def __init__(self):
        """Initialize upload manager"""
        self._create_archive_dir()
    
    def _create_archive_dir(self):
        """Create archive directory if it doesn't exist"""
        if not os.path.exists(self.ARCHIVE_DIR):
            os.makedirs(self.ARCHIVE_DIR)
    
    def validate_csv(self, df):
        """
        Validate that CSV has required columns
        
        Args:
            df: pandas DataFrame
            
        Returns:
            (is_valid, error_message)
        """
        # Check for required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check for Symbol column (unique identifier)
        if df['Symbol'].duplicated().any():
            return False, "Duplicate fund symbols found"
        
        # Check for empty dataframe
        if len(df) == 0:
            return False, "CSV file is empty"
        
        # Warn if too few funds
        if len(df) < 100:
            return True, f"Warning: Only {len(df)} funds found (expected 100+)"
        
        return True, "CSV validated successfully"
    
    def upload_csv(self, uploaded_file):
        """
        Process uploaded CSV file
        
        Args:
            uploaded_file: Streamlit uploaded file
            
        Returns:
            (df, success, message)
        """
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate
            is_valid, message = self.validate_csv(df)
            
            if not is_valid and "Warning" not in message:
                return None, False, message
            
            # Archive the uploaded file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(
                self.ARCHIVE_DIR, 
                f"fund_data_{timestamp}.csv"
            )
            df.to_csv(archive_path, index=False)
            
            return df, True, f"CSV uploaded successfully ({len(df)} funds)"
        
        except Exception as e:
            return None, False, f"Error reading CSV: {str(e)}"
    
    def get_archived_files(self):
        """
        Get list of all archived CSV files
        
        Returns:
            List of (filename, path, date) tuples
        """
        if not os.path.exists(self.ARCHIVE_DIR):
            return []
        
        files = []
        for filename in sorted(os.listdir(self.ARCHIVE_DIR), reverse=True):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.ARCHIVE_DIR, filename)
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                files.append((filename, filepath, mod_time))
        
        return files
    
    def load_default_csv(self, default_path='fund_screener_data-41.csv'):
        """
        Load default CSV if it exists
        
        Args:
            default_path: Path to default CSV
            
        Returns:
            (df, success, message)
        """
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                is_valid, message = self.validate_csv(df)
                return df, True, f"Loaded default data ({len(df)} funds)"
            except Exception as e:
                return None, False, f"Error loading default CSV: {str(e)}"
        
        return None, False, "No default CSV found"