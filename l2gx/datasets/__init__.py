from .as733 import AS733Dataset
from .base import BaseDataset
from .btc import BTCDataset, BTCReducedDataset
from .citeseer import CiteSeerDataset
from .cora import CoraDataset
from .dgraph import DGraphDataset
from .elliptic import EllipticDataset
from .mag240m import MAG240MDataset
from .orbitaal import ORBITAALDataset
from .pubmed import PubMedDataset
from .registry import DATASET_REGISTRY, get_dataset, register_dataset


def list_available_datasets():
    """
    Returns a list of registered dataset names.
    """
    return list(DATASET_REGISTRY.keys())


def get_dataset_info():
    """
    Returns detailed information about all available datasets.
    """
    return {
        "Cora": {
            "description": "Citation network dataset",
            "nodes": 2708,
            "edges": 5429,
            "features": 1433,
            "classes": 7,
            "type": "static",
        },
        "CiteSeer": {
            "description": "Citation network dataset",
            "nodes": 3327,
            "edges": 9104,
            "features": 3703,
            "classes": 6,
            "type": "static",
        },
        "PubMed": {
            "description": "Citation network dataset (PubMed abstracts)",
            "nodes": 19717,
            "edges": 88648,
            "features": 500,
            "classes": 3,
            "type": "static",
        },
        "as-733": {
            "description": "Autonomous systems temporal network",
            "nodes": 7716,
            "edges": 45645,
            "snapshots": 733,
            "type": "temporal",
        },
        "DGraph": {
            "description": "Financial fraud detection graph",
            "nodes": "~3M",
            "edges": "~4M",
            "features": "multiple",
            "type": "static",
        },
        "Elliptic": {
            "description": "Bitcoin transaction illicit detection",
            "nodes": 203769,
            "edges": 234355,
            "features": 166,
            "classes": 3,
            "type": "static",
        },
        "MAG240M": {
            "description": "Academic citation heterogeneous graph",
            "nodes": "244M+",
            "edges": "1.7B+",
            "features": "various",
            "type": "heterogeneous",
        },
        "MAG240M-Enhanced": {
            "description": "MAG240M with lazy loading and sampling",
            "nodes": "Configurable (subset of 244M+)",
            "edges": "Configurable (subset of 1.7B+)",
            "features": "128 (paper embeddings)",
            "classes": "Year-based",
            "type": "heterogeneous",
            "strategies": ["recent_papers", "random_papers", "field_papers", "temporal_window", "citation_subgraph"],
        },
        "ORBITAAL": {
            "description": "Bitcoin temporal transaction graph",
            "nodes": "252M (sample: 1K)",
            "edges": "785M (sample: 5K)",
            "features": "temporal, anomaly_labels",
            "classes": "anomaly detection",
            "type": "temporal",
        },
        "btc": {
            "description": "Bitcoin transaction graph with entity classification",
            "nodes": "252M",
            "edges": "~1B",
            "features": 8,
            "classes": 12,
            "type": "static",
        },
        "btc-reduced": {
            "description": "Bitcoin transaction graph (labeled nodes only)",
            "nodes": "~80K (0.03% of full dataset)",
            "edges": "~10K (between labeled nodes)",
            "features": 8,
            "classes": 11,
            "type": "static",
        },
    }


__all__ = [
    "DATASET_REGISTRY",
    "AS733Dataset",
    "BTCDataset",
    "BTCReducedDataset",
    "BaseDataset",
    "CiteSeerDataset",
    "CoraDataset",
    "DGraphDataset",
    "EllipticDataset",
    "EnhancedMAG240MDataset",
    "MAG240MDataset",
    "ORBITAALDataset",
    "PubMedDataset",
    "get_dataset",
    "get_dataset_info",
    "list_available_datasets",
    "register_dataset",
]
