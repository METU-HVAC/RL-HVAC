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

def build_graph_structure(ctx: OntologyContext, mapping: Dict[str, Any]) -> GraphStructure:
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
    # We scan the graph for connections between our selected nodes
    # specifically focusing on saref:hasLocation to the room.
    
    edges = []
    
    # Pre-compute indices for fast lookup
    uri_to_idx = {uri: i for i, uri in enumerate(node_uris)}
    
    # We want undirected edges, so we'll add (u, v) and (v, u)
    def add_undirected_edge(u_idx, v_idx):
        edges.append([u_idx, v_idx])
        edges.append([v_idx, u_idx])

    # Check connections for each node against the graph
    # Optimization: Since we know we care about connections to the room,
    # we can explicitly check (node, hasLocation, room) for sensors/actuators.
    
    room_uri_ref = URIRef(room_uri)
    has_location = ctx.saref.hasLocation
    
    # Enforce edges for all mapped sensors and actuators to the room
    # This ensures connectivity even if triples are missing (e.g. outdoor sensors)
    
    # 1. Sensors
    for idx in sensor_indices.values():
        # Avoid duplicate edges if they were already added by the graph check
        # But our simple edge list might just add duplicates. 
        # For simplicity and correctness with "add_undirected_edge" above which appends, 
        # we can collect unique pairs or just rely on GNN framework handling it.
        # However, to be cleaner, let's track added edges.
        pass

    # Better approach: Clear edges and re-build with both logic combined or just enforcement.
    # The requirement says: "Modify... so that every sensor and actuator... is connected... regardless of whether saref:hasLocation exists"
    # It also says: "Keep existing hasLocation logic, but enforce these mapping-based edges anyway."
    
    # Let's use a set to track connections to avoid duplicates
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
    # We need to find indices for actuators. Since we didn't store a dict for them like sensor_indices,
    # we iterate node types or reconstruct lookup.
    # We constructed node_uris sequentially. 
    # Let's find actuator indices by type or by URI lookup if we had the config handy.
    
    # Re-iterate mapping to find indices
    actuators_map = mapping.get("actuators", {})
    for act_name, act_config in actuators_map.items():
        act_curie = act_config.get("actuator_uri") if isinstance(act_config, dict) else act_config
        if act_curie:
            full_uri = resolve_uri(act_curie)
            if full_uri in uri_to_idx:
                idx = uri_to_idx[full_uri]
                if idx not in connected_indices and idx != room_index:
                    add_undirected_edge(idx, room_index)
                    connected_indices.add(idx)

    if not edges:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T

    return GraphStructure(
        node_uris=node_uris,
        node_types=node_types,
        edge_index=edge_index,
        room_index=room_index,
        sensor_indices=sensor_indices
    )

