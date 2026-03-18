from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import os

class DataSource(ABC):
    """
    Abstract base class for data sources in the RL + ontology project.
    """

    @abstractmethod
    def get_snapshot(self, t_index: int) -> Dict[str, Any]:
        """
        Retrieve a snapshot of data at a specific time index.

        Args:
            t_index (int): Integer timestep index (0, 1, 2, ...).

        Returns:
            dict: A dictionary where keys are column names and values are the data points.
        """
        pass

class CSVDataSource(DataSource):
    """
    Concrete implementation of DataSource that reads from a CSV file.
    """

    def __init__(self, file_path: str):
        """
        Initialize the CSVDataSource.

        Args:
            file_path (str): Path to the CSV file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at: {file_path}")
            
        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

    def get_snapshot(self, t_index: int) -> Dict[str, Any]:
        """
        Retrieve a row from the CSV dataframe as a dictionary.

        Args:
            t_index (int): Integer timestep index.

        Returns:
            dict: The row data as a dictionary.

        Raises:
            IndexError: If t_index is out of bounds.
        """
        if t_index < 0 or t_index >= len(self.df):
            raise IndexError(f"Time index {t_index} is out of bounds. Data length: {len(self.df)}")
        
        return self.df.iloc[t_index].to_dict()

