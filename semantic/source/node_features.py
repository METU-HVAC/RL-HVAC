import numpy as np
from typing import Dict, Any, List, Optional
from rdflib import URIRef, Literal
from .graph_builder import GraphStructure
from .kg_queries import KGQueryHelper
from .ontology_loader import OntologyContext

# Fixed property types for one-hot encoding
PROPERTY_TYPES = ["Temperature", "Humidity", "CO2", "Occupancy", "Other"]
PROPERTY_TYPE_TO_IDX = {p: i for i, p in enumerate(PROPERTY_TYPES)}

def build_node_type_embedding(node_types: List[str]) -> np.ndarray:
    """
    Create a one-hot embedding for node types.

    Args:
        node_types (list[str]): List of node types for each node.

    Returns:
        np.ndarray: Matrix of shape (num_nodes, num_distinct_types).
    """
    unique_types = sorted(list(set(node_types)))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    
    num_nodes = len(node_types)
    num_types = len(unique_types)
    
    # Create one-hot encoding
    embeddings = np.zeros((num_nodes, num_types), dtype=np.float32)
    for i, t in enumerate(node_types):
        type_idx = type_to_idx[t]
        embeddings[i, type_idx] = 1.0
        
    return embeddings, num_types

def get_property_type_from_class(saref_class: str) -> str:
    """
    Map a SAREF property class to a simplified property type.
    
    Args:
        saref_class (str): The SAREF property class string (e.g., "saref:Temperature").
        
    Returns:
        str: The simplified property type.
    """
    if not saref_class:
        return "Other"
    
    class_lower = saref_class.lower()
    if "temperature" in class_lower:
        return "Temperature"
    elif "humidity" in class_lower:
        return "Humidity"
    elif "co2" in class_lower:
        return "CO2"
    elif "occupancy" in class_lower:
        return "Occupancy"
    else:
        return "Other"

def build_property_type_embedding(
    graph: GraphStructure, 
    mapping: Dict[str, Any]
) -> np.ndarray:
    """
    Create a one-hot embedding for sensor property types.
    
    Args:
        graph (GraphStructure): The graph structure.
        mapping (dict): The mapping configuration.
        
    Returns:
        np.ndarray: Matrix of shape (num_nodes, num_property_types).
    """
    num_nodes = len(graph.node_uris)
    num_prop_types = len(PROPERTY_TYPES)
    
    prop_embeds = np.zeros((num_nodes, num_prop_types), dtype=np.float32)
    
    # Reverse mapping for sensor indices
    idx_to_sensor_col = {v: k for k, v in graph.sensor_indices.items()}
    sensors_map = mapping.get("sensors", {})
    
    for i, node_type in enumerate(graph.node_types):
        if node_type == "sensor":
            col_name = idx_to_sensor_col.get(i)
            if col_name and col_name in sensors_map:
                saref_class = sensors_map[col_name].get("saref_property_class", "")
                prop_type = get_property_type_from_class(saref_class)
                prop_idx = PROPERTY_TYPE_TO_IDX.get(prop_type, PROPERTY_TYPE_TO_IDX["Other"])
                prop_embeds[i, prop_idx] = 1.0
    
    return prop_embeds

def get_room_semantic_features(
    ctx: OntologyContext,
    room_curie: str
) -> np.ndarray:
    """
    Extract static semantic features for a room from the ontology.
    
    Args:
        ctx (OntologyContext): The ontology context.
        room_curie (str): The room CURIE string.
        
    Returns:
        np.ndarray: Array of normalized semantic features [minComfortTemp, maxComfortTemp, maxCO2, occupancyCapacity].
    """
    # Resolve room URI
    if ":" in room_curie:
        prefix, local = room_curie.split(":", 1)
        if prefix == "bld":
            room_uri = ctx.bld[local]
        else:
            room_uri = URIRef(room_curie)
    else:
        room_uri = URIRef(room_curie)
    
    # Define property URIs (these would be defined in the ontology)
    # Using bld namespace for custom properties
    min_comfort_temp_prop = ctx.bld["minComfortTemp"]
    max_comfort_temp_prop = ctx.bld["maxComfortTemp"]
    max_co2_prop = ctx.bld["maxCO2"]
    occupancy_capacity_prop = ctx.bld["occupancyCapacity"]
    
    # Query each property with defaults if not found
    def get_literal_value(prop_uri, default=0.0):
        for s, p, o in ctx.graph.triples((room_uri, prop_uri, None)):
            if isinstance(o, Literal):
                try:
                    return float(o)
                except (ValueError, TypeError):
                    return default
        return default
    
    # Get values with sensible defaults
    min_comfort_temp = get_literal_value(min_comfort_temp_prop, 18.0)
    max_comfort_temp = get_literal_value(max_comfort_temp_prop, 26.0)
    max_co2 = get_literal_value(max_co2_prop, 1000.0)
    occupancy_capacity = get_literal_value(occupancy_capacity_prop, 30.0)
    
    # Normalize
    features = np.array([
        min_comfort_temp / 50.0,  # Temperature normalization
        max_comfort_temp / 50.0,
        max_co2 / 2000.0,         # CO2 normalization
        occupancy_capacity / 100.0  # Occupancy normalization
    ], dtype=np.float32)
    
    return features

def build_node_features(
    graph: GraphStructure, 
    snapshot: Dict[str, Any],
    mapping: Optional[Dict[str, Any]] = None,
    ctx: Optional[OntologyContext] = None
) -> np.ndarray:
    """
    Construct node feature matrix from graph structure, data snapshot, and ontology.

    Args:
        graph (GraphStructure): The graph topology and metadata.
        snapshot (dict): The current data snapshot containing sensor readings.
        mapping (dict, optional): The mapping configuration for property types.
        ctx (OntologyContext, optional): The ontology context for semantic features.

    Returns:
        np.ndarray: Feature matrix of shape (num_nodes, d_in).
    """
    num_nodes = len(graph.node_uris)
    
    # 1. Base Type Embeddings
    # Shape: (num_nodes, num_types)
    type_embeds, num_types = build_node_type_embedding(graph.node_types)
    
    # 2. Property Type Embeddings (for sensors)
    # Shape: (num_nodes, num_property_types)
    num_prop_types = len(PROPERTY_TYPES)
    if mapping is not None:
        prop_embeds = build_property_type_embedding(graph, mapping)
    else:
        prop_embeds = np.zeros((num_nodes, num_prop_types), dtype=np.float32)
    
    # 3. Room Semantic Features
    # Shape: (num_nodes, 4) - [minComfortTemp, maxComfortTemp, maxCO2, occupancyCapacity]
    num_semantic = 4
    semantic_feats = np.zeros((num_nodes, num_semantic), dtype=np.float32)
    
    if ctx is not None and mapping is not None:
        room_curie = mapping.get("room_uri", "")
        if room_curie:
            room_semantic = get_room_semantic_features(ctx, room_curie)
            # Apply to room node only
            semantic_feats[graph.room_index] = room_semantic
    
    # 4. Dynamic Features
    # Shape: (num_nodes, 3) - [value, time_sin, time_cos]
    
    # Time features (global context)
    hour = snapshot.get("hour", 0)
    minute = snapshot.get("minute", 0)
    total_minutes = hour * 60 + minute
    day_minutes = 24 * 60
    
    time_sin = np.sin(2 * np.pi * total_minutes / day_minutes)
    time_cos = np.cos(2 * np.pi * total_minutes / day_minutes)
    
    dynamic_feats = np.zeros((num_nodes, 3), dtype=np.float32)
    
    # Reverse mapping for sensor indices: index -> column name
    idx_to_sensor_col = {v: k for k, v in graph.sensor_indices.items()}
    
    # Build actuator index mapping from mapping config
    actuator_col_to_idx = {}
    if mapping is not None:
        actuators_map = mapping.get("actuators", {})
        for act_name, act_config in actuators_map.items():
            if isinstance(act_config, dict):
                act_curie = act_config.get("actuator_uri")
                if act_curie and ":" in act_curie:
                    prefix, local = act_curie.split(":", 1)
                    if prefix == "bld":
                        # Find index in node_uris
                        full_uri = f"http://example.org/building#{local}"
                        if full_uri in graph.node_uris:
                            idx = graph.node_uris.index(full_uri)
                            actuator_col_to_idx[act_name] = idx
    
    idx_to_actuator_col = {v: k for k, v in actuator_col_to_idx.items()}
    
    for i, node_type in enumerate(graph.node_types):
        # Default time context for all nodes (shared context)
        dynamic_feats[i, 1] = time_sin
        dynamic_feats[i, 2] = time_cos
        
        val = 0.0
        
        if node_type == "sensor":
            # Find which column this sensor corresponds to
            col_name = idx_to_sensor_col.get(i)
            if col_name and col_name in snapshot:
                raw_val = snapshot[col_name]
                
                # Simple normalization heuristics
                if "co2" in col_name.lower():
                    val = raw_val / 1000.0
                elif "temperature" in col_name.lower():
                    val = raw_val / 50.0
                elif "humidity" in col_name.lower():
                    val = raw_val / 100.0
                elif "occupant" in col_name.lower():
                    val = float(raw_val)
                else:
                    val = raw_val
        
        elif node_type == "actuator":
            # Get actuator command value from snapshot
            col_name = idx_to_actuator_col.get(i)
            if col_name and col_name in snapshot:
                raw_val = snapshot[col_name]
                # Normalize actuator values (assuming 0-100 range for fan speeds, etc.)
                if "speed" in col_name.lower() or "fan" in col_name.lower():
                    val = raw_val / 100.0
                elif "setpoint" in col_name.lower():
                    val = raw_val / 50.0
                else:
                    val = raw_val / 100.0

        elif node_type == "room":
            # Room node carries time context primarily (already set above)
            pass
            
        elif node_type == "device":
            # Energy meters - try to find matching column
            pass

        dynamic_feats[i, 0] = val

    # 5. Concatenate all features
    # Final shape: (num_nodes, num_types + num_prop_types + num_semantic + 3)
    final_features = np.concatenate([
        type_embeds,      # Node type one-hot
        prop_embeds,      # Property type one-hot (sensors)
        semantic_feats,   # Room semantic features
        dynamic_feats     # Dynamic values + time
    ], axis=1)
    
    return final_features

def get_feature_dim(graph: GraphStructure) -> int:
    """
    Calculate the expected feature dimension for a given graph.
    
    Args:
        graph (GraphStructure): The graph structure.
        
    Returns:
        int: The feature dimension.
    """
    num_types = len(set(graph.node_types))
    num_prop_types = len(PROPERTY_TYPES)
    num_semantic = 4
    num_dynamic = 3
    
    return num_types + num_prop_types + num_semantic + num_dynamic
