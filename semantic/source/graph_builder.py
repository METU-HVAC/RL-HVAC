from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
from rdflib import URIRef
from .ontology_loader import OntologyContext

@dataclass
class GraphStructure:
    """
    Data structure representing the building graph for GNN processing.
    """
    node_uris: List[str]
    node_types: List[str]
    edge_index: np.ndarray
    room_index: int
    sensor_indices: Dict[str, int]
    topology: str = "star"

def build_graph_structure(
    ctx: OntologyContext,
    mapping: Dict[str, Any],
    topology: str = "star"
) -> GraphStructure:
    """
    Build a graph structure from the ontology and mapping configuration.

    Args:
        ctx (OntologyContext): Loaded ontology context.
        mapping (dict): Configuration mapping dictionary.

    Returns:
        GraphStructure: The constructed graph structure suitable for GNNs.
    """
    node_uris: List[str] = []
    node_types: List[str] = []
    sensor_indices: Dict[str, int] = {}
    
    # Helper to resolve CURIE to full URI string
    def resolve_uri(curie: str) -> str:
        if ":" in curie:
            prefix, local = curie.split(":", 1)
            if prefix == "bld":
                return str(ctx.bld[local])
            elif prefix == "saref":
                return str(ctx.saref[local])
        return curie

    # 1. Add Room Node
    room_curie = mapping.get("room_uri")
    if not room_curie:
        raise ValueError("Mapping must contain 'room_uri'")
    
    room_uri = resolve_uri(room_curie)
    node_uris.append(room_uri)
    node_types.append("room")
    room_index = 0  # Room is always the first node

    # 2. Add Sensor Nodes
    # Iterate through sensors defined in the mapping
    sensors_map = mapping.get("sensors", {})
    for col_name, sensor_config in sensors_map.items():
        sensor_curie = sensor_config.get("sensor_uri")
        if sensor_curie:
            full_uri = resolve_uri(sensor_curie)
            
            # Check if node already exists (avoid duplicates if multiple cols map to same sensor)
            if full_uri in node_uris:
                idx = node_uris.index(full_uri)
            else:
                idx = len(node_uris)
                node_uris.append(full_uri)
                # Differentiate based on config or default to 'sensor'
                # Outdoor sensors usually don't have location in room, so this type might need
                # refinement, but prompt asks for "sensor". 
                # Could check if it's "outdoor" based on name or property, 
                # but "sensor" covers general requirement.
                node_types.append("sensor")
            
            sensor_indices[col_name] = idx

    # 3. Add Actuator Nodes
    actuators_map = mapping.get("actuators", {})
    for act_name, act_config in actuators_map.items():
        act_curie = act_config.get("actuator_uri") if isinstance(act_config, dict) else act_config
        # Handle case where config might be just a URI string or a dict
        
        if act_curie:
            full_uri = resolve_uri(act_curie)
            if full_uri not in node_uris:
                node_uris.append(full_uri)
                node_types.append("actuator")

    # 4. Add Energy Meter Nodes
    meters_map = mapping.get("energy_meters", {})
    for meter_name, meter_config in meters_map.items():
        meter_curie = meter_config.get("device_uri")
        if meter_curie:
            full_uri = resolve_uri(meter_curie)
            if full_uri not in node_uris:
                node_uris.append(full_uri)
                node_types.append("device")

    # 5. Build Edges
    edges = []

    # Pre-compute indices for fast lookup
    uri_to_idx = {uri: i for i, uri in enumerate(node_uris)}

    # Track directed edges to avoid duplicates.
    edge_set = set()

    def add_directed_edge(src_idx: int, dst_idx: int) -> None:
        edge_set.add((src_idx, dst_idx))

    # We want undirected edges, so add (u, v) and (v, u).
    def add_undirected_edge(u_idx: int, v_idx: int) -> None:
        add_directed_edge(u_idx, v_idx)
        add_directed_edge(v_idx, u_idx)

    if topology == "fully_connected":
        num_nodes = len(node_uris)
        for src_idx in range(num_nodes):
            for dst_idx in range(num_nodes):
                if src_idx != dst_idx:
                    add_directed_edge(src_idx, dst_idx)
    elif topology == "star":
        # Existing hasLocation + mapping enforcement behavior centered on room node.
        room_uri_ref = URIRef(room_uri)
        has_location = ctx.saref.hasLocation
        connected_indices = set()

        # Existing hasLocation logic
        for i, uri_str in enumerate(node_uris):
            if i == room_index:
                continue

            uri_ref = URIRef(uri_str)
            if (uri_ref, has_location, room_uri_ref) in ctx.graph:
                add_undirected_edge(i, room_index)
                connected_indices.add(i)

        # Enforce mapping-based edges for sensors
        for idx in sensor_indices.values():
            if idx not in connected_indices and idx != room_index:
                add_undirected_edge(idx, room_index)
                connected_indices.add(idx)

        # Enforce mapping-based edges for actuators
        actuators_map = mapping.get("actuators", {})
        for _, act_config in actuators_map.items():
            act_curie = act_config.get("actuator_uri") if isinstance(act_config, dict) else act_config
            if act_curie:
                full_uri = resolve_uri(act_curie)
                if full_uri in uri_to_idx:
                    idx = uri_to_idx[full_uri]
                    if idx not in connected_indices and idx != room_index:
                        add_undirected_edge(idx, room_index)
                        connected_indices.add(idx)
    else:
        raise ValueError(f"Unsupported topology '{topology}'. Use 'star' or 'fully_connected'.")

    edges = [[src, dst] for src, dst in sorted(edge_set)]

    if not edges:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T

    return GraphStructure(
        node_uris=node_uris,
        node_types=node_types,
        edge_index=edge_index,
        room_index=room_index,
        sensor_indices=sensor_indices,
        topology=topology
    )

