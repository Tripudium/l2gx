from .registry import get_dataset, register_dataset, DATASET_REGISTRY
from .base import BaseDataset
from .as733 import AS733Dataset
from .cora import CoraDataset
from .citeseer import CiteSeerDataset
from .pubmed import PubMedDataset
from .dgraph import DGraphDataset
from .elliptic import EllipticDataset
from .mag240m import MAG240MDataset
from .orbitaal import ORBITAALDataset


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
            "type": "static"
        },
        "CiteSeer": {
            "description": "Citation network dataset",
            "nodes": 3327,
            "edges": 9104,
            "features": 3703,
            "classes": 6,
            "type": "static"
        },
        "PubMed": {
            "description": "Citation network dataset (PubMed abstracts)",
            "nodes": 19717,
            "edges": 88648,
            "features": 500,
            "classes": 3,
            "type": "static"
        },
        "as-733": {
            "description": "Autonomous systems temporal network",
            "nodes": 7716,
            "edges": 45645,
            "snapshots": 733,
            "type": "temporal"
        },
        "DGraph": {
            "description": "Financial fraud detection graph",
            "nodes": "~3M",
            "edges": "~4M",
            "features": "multiple",
            "type": "static"
        },
        "Elliptic": {
            "description": "Bitcoin transaction illicit detection",
            "nodes": 203769,
            "edges": 234355,
            "features": 166,
            "classes": 3,
            "type": "static"
        },
        "MAG240M": {
            "description": "Academic citation heterogeneous graph",
            "nodes": "244M+",
            "edges": "1.7B+",
            "features": "various",
            "type": "heterogeneous"
        },
        "ORBITAAL": {
            "description": "Bitcoin temporal transaction graph",
            "nodes": "252M (sample: 1K)",
            "edges": "785M (sample: 5K)", 
            "features": "temporal, anomaly_labels",
            "classes": "anomaly detection",
            "type": "temporal"
        }
    }


__all__ = [
    "get_dataset",
    "register_dataset",
    "DATASET_REGISTRY",
    "BaseDataset",
    "AS733Dataset",
    "CoraDataset",
    "CiteSeerDataset",
    "PubMedDataset",
    "DGraphDataset",
    "EllipticDataset",
    "MAG240MDataset",
    "ORBITAALDataset",
    "list_available_datasets",
    "get_dataset_info",
]
