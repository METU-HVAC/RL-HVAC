from typing import List
from rdflib import URIRef
from .ontology_loader import OntologyContext

class KGQueryHelper:
    """
    Helper class for querying the Knowledge Graph using rdflib triple patterns.
    """

    def __init__(self, ctx: OntologyContext):
        """
        Initialize the KGQueryHelper with an ontology context.

        Args:
            ctx (OntologyContext): The context containing the graph and namespaces.
        """
        self.ctx = ctx

    def _expand_curie(self, curie: str) -> URIRef:
        """
        Helper to expand a CURIE string to a full URIRef.
        
        Args:
            curie (str): The CURIE string (e.g., "bld:RoomA403").
            
        Returns:
            URIRef: The expanded URI.
        """
        if ":" in curie:
            prefix, local = curie.split(":", 1)
            if prefix == "bld":
                return self.ctx.bld[local]
            elif prefix == "saref":
                return self.ctx.saref[local]
        
        # Fallback: if no known prefix or no colon, return as URIRef directly
        # logic could be expanded if more prefixes are needed
        return URIRef(curie)

    def sensor_belongs_to_room(self, sensor_uri: str, room_uri: str) -> bool:
        """
        Check if a sensor is located in a specific room.
        
        Args:
            sensor_uri (str): CURIE string for the sensor (e.g., "bld:RoomA403CO2Sensor_1").
            room_uri (str): CURIE string for the room (e.g., "bld:Room_A403").
            
        Returns:
            bool: True if <sensor> saref:hasLocation <room> exists in the graph.
        """
        s_uri = self._expand_curie(sensor_uri)
        r_uri = self._expand_curie(room_uri)
        
        # Check for existence of the triple (s, p, o)
        return (s_uri, self.ctx.saref.hasLocation, r_uri) in self.ctx.graph

    def get_sensors_in_room(self, room_curie: str) -> List[str]:
        """
        Get a list of sensor CURIEs located in a specific room.

        Args:
            room_curie (str): CURIE string for the room.

        Returns:
            list[str]: List of sensor CURIE strings found in that room.
        """
        r_uri = self._expand_curie(room_curie)
        sensors = []

        # Find all subjects s where (s, saref:hasLocation, r_uri) exists
        for s, p, o in self.ctx.graph.triples((None, self.ctx.saref.hasLocation, r_uri)):
            # Convert URI back to CURIE if possible
            s_str = str(s)
            bld_str = str(self.ctx.bld)
            
            if s_str.startswith(bld_str):
                # Strip base URI and add bld: prefix
                local_name = s_str[len(bld_str):]
                sensors.append(f"bld:{local_name}")
            else:
                # Fallback to full string if not in bld namespace
                sensors.append(s_str)
                
        return sensors

