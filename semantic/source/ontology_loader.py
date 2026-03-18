from dataclasses import dataclass
from rdflib import Graph, Namespace
import os

# Define Namespaces
SAREF = Namespace("https://saref.etsi.org/core/")
BLD = Namespace("http://example.org/building#")

@dataclass
class OntologyContext:
    """
    Context holding the loaded RDF graph and relevant namespaces.
    """
    graph: Graph
    saref: Namespace
    bld: Namespace

def load_building_graph(ontology_path: str) -> Graph:
    """
    Load an RDF graph from a Turtle file and bind namespaces.

    Args:
        ontology_path (str): Path to the ontology file (e.g., .ttl).

    Returns:
        Graph: The loaded rdflib Graph with prefixes bound.
        
    Raises:
        FileNotFoundError: If the ontology file is not found.
    """
    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"Ontology file not found at: {ontology_path}")

    g = Graph()
    try:
        g.parse(ontology_path, format="turtle")
    except Exception as e:
        raise ValueError(f"Failed to parse ontology file: {e}")

    # Bind prefixes for easier SPARQL querying and serialization
    g.bind("saref", SAREF)
    g.bind("bld", BLD)
    
    return g

def create_ontology_context(ontology_path: str) -> OntologyContext:
    """
    Convenience function to load the graph and wrap it in an OntologyContext.

    Args:
        ontology_path (str): Path to the ontology file.

    Returns:
        OntologyContext: The context containing graph and namespaces.
    """
    graph = load_building_graph(ontology_path)
    return OntologyContext(graph=graph, saref=SAREF, bld=BLD)

