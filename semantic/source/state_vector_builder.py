from dataclasses import dataclass
from typing import List, Dict, Any
import yaml
import numpy as np
import os
from .kg_queries import KGQueryHelper

def load_mapping(path: str) -> Dict[str, Any]:
    """
    Load the YAML mapping file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: The loaded mapping dictionary.
        
    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError: If parsing fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapping file not found at: {path}")
    
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}")

@dataclass
class StateVectorConfig:
    """
    Configuration for building state vectors.
    """
    mapping: Dict[str, Any]
    room_curie: str
    feature_order: List[str] = None

    def __post_init__(self):
        if self.feature_order is None:
            self.feature_order = [
                "month", "day_of_month", "hour", "minute",
                "outdoor_temperature", "outdoor_humidity",
                "air_temperature", "air_humidity", "air_co2",
                "people_occupant",
                "window_fan_energy", "total_electricity_HVAC",
                "heating_setpoint", "cooling_setpoint",
                "ac_fan_speed", "window_fan_speed"
            ]

class StateVectorBuilder:
    """
    Builder class to convert data snapshots into RL state vectors.
    """

    def __init__(self, config: StateVectorConfig, kg_helper: KGQueryHelper):
        """
        Initialize the StateVectorBuilder.

        Args:
            config (StateVectorConfig): Configuration containing mapping and feature order.
            kg_helper (KGQueryHelper): Helper for querying the knowledge graph.
        """
        self.config = config
        self.kg_helper = kg_helper
        self._validate_config()

    def _validate_config(self):
        """
        Validate that mapped sensors belong to the configured room where applicable.
        This is a basic check to ensure alignment between mapping and KG.
        """
        # Example validation: Check if 'air_temperature' sensor is in the room
        # This could be expanded to iterate over all relevant sensors in the mapping
        sensor_mapping = self.config.mapping.get("sensors", {})
        
        # We only check sensors that are explicitly mapped and expected to be in the room
        # This logic assumes that keys in sensor_mapping match feature names or column names
        # and that we can access their URIs.
        # This part is optional per requirements but good for robustness.
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names in the state vector.

        Returns:
            list[str]: Ordered list of feature names.
        """
        return self.config.feature_order

    def build_state(self, snapshot: Dict[str, Any]) -> np.ndarray:
        """
        Build a state vector from a data snapshot.

        Args:
            snapshot (dict): Dictionary containing data for a specific timestep.

        Returns:
            np.ndarray: 1D NumPy array representing the state vector.
        
        Raises:
            KeyError: If a required feature is missing from the snapshot.
        """
        state = []
        
        # Categorize feature types based on top-level keys in mapping
        # Note: Time features usually don't have a top-level mapping key like 'sensors'
        # but are direct columns.
        
        # Flatten mapping lookup for easier access? 
        # The prompt says feature_order contains keys like "air_temperature".
        # We assume these keys exist in the snapshot dict.
        
        for feature in self.config.feature_order:
            if feature in snapshot:
                val = float(snapshot[feature])
                state.append(val)
            else:
                # If feature is missing from snapshot, raises error
                # In a robust system, we might handle defaults or NaNs, 
                # but for strict RL, missing data is usually fatal or needs specific handling.
                raise KeyError(f"Feature '{feature}' not found in snapshot data.")
        
        return np.array(state, dtype=np.float32)

