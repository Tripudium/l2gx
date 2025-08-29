"""
Random graph models for testing patch generation algorithms.

This module provides implementations of various random graph models that
generate graphs with overlapping community structure, making them suitable
for testing patch-based graph embedding and alignment algorithms.

Models included:
- Community-Affiliation Graph Model (AGM)
- Overlapping Stochastic Block Model (OSBM)
- Random Overlapping Communities (ROC) Model
"""

from .agm import CommunityAffiliationGraphModel
from .osbm import OverlappingStochasticBlockModel
from .roc import RandomOverlappingCommunities

__all__ = [
    "CommunityAffiliationGraphModel",
    "OverlappingStochasticBlockModel",
    "RandomOverlappingCommunities",
]
