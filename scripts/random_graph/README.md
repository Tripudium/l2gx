# Random Graph Models for Patch Generation Testing

This module provides three random graph models specifically designed for testing patch-based graph embedding and alignment algorithms. Each model generates graphs with overlapping community structure that can be used to evaluate the performance of patch detection and alignment methods.

## Models Implemented

### 1. Community-Affiliation Graph Model (AGM)
**File:** `agm.py`

The AGM model creates graphs by first establishing a bipartite affiliation between nodes and communities, then connecting nodes that share community memberships.

**Key Features:**
- Flexible community size distributions (uniform, Poisson, power-law, Bernoulli)
- Controllable intra-community and background edge probabilities
- Well-defined community boundaries
- Fast generation

**Best For:**
- Testing basic patch detection algorithms
- Scenarios requiring clear community structure
- Benchmarking with well-separated communities

**Usage:**
```python
from agm import CommunityAffiliationGraphModel

agm = CommunityAffiliationGraphModel(
    n_nodes=150,
    n_communities=6,
    community_size_dist="uniform",
    community_size_params={"min_size": 20, "max_size": 35},
    intra_community_prob=0.3,
    background_prob=0.01,
    random_state=42
)

G = agm.generate_graph()
stats = agm.compute_statistics()
```

### 2. Overlapping Stochastic Block Model (OSBM)
**File:** `osbm.py`

The OSBM extends the classical Stochastic Block Model to allow nodes to belong to multiple communities, with edge probabilities determined by community memberships.

**Key Features:**
- Nodes can belong to multiple overlapping communities
- Edge probabilities based on shared community memberships
- Flexible community assignment schemes (random, balanced, power-law)
- Principled probabilistic framework

**Best For:**
- Testing alignment algorithms on block-structured data
- Evaluating performance with controlled overlap patterns
- Scenarios requiring theoretical guarantees

**Usage:**
```python
from osbm import OverlappingStochasticBlockModel

osbm = OverlappingStochasticBlockModel(
    n_nodes=120,
    n_communities=5,
    community_assignments="balanced",
    overlap_probability=0.25,
    intra_community_prob=0.3,
    inter_community_prob=0.02,
    random_state=42
)

G = osbm.generate_graph()
```

### 3. Petti-Vempala Random Overlapping Communities (ROC) Model
**File:** `petti_vempala_roc.py`

Implementation of the theoretical ROC model from Petti & Vempala's paper "Approximating Sparse Graphs: The Random Overlapping Communities Model". This model is specifically designed for theoretical analysis with controlled overlap structure.

**Key Features:**
- Each node belongs to exactly 1 or 2 communities
- Communities have roughly equal sizes  
- Principled edge probability structure (p_in, p_out)
- Designed for sparse graphs with theoretical guarantees

**Best For:**
- Theoretical analysis and algorithm evaluation
- Controlled experiments with known ground truth
- Testing community detection algorithms with provable guarantees

**Usage:**
```python
from petti_vempala_roc import PettiVempalaROC

roc = PettiVempalaROC(
    n_nodes=100,
    k_communities=4,
    p_in=0.3,
    p_out=0.05,
    overlap_fraction=0.1,
    random_state=42
)

G = roc.generate_graph()
```

### 4. General Overlapping Communities Model
**File:** `general_overlapping_communities.py`

A flexible model that creates communities through various growth processes, with realistic overlapping community structures and membership strengths.

**Key Features:**
- Multiple community generation methods (planted, grown, preferential, hybrid)
- Variable membership strengths (binary, uniform, beta, exponential)
- Flexible edge probability functions
- Realistic community growth patterns

**Best For:**
- Testing on realistic social network-like structures
- Scenarios with fuzzy community boundaries
- Evaluating algorithms on hierarchical community structures

**Usage:**
```python
from general_overlapping_communities import GeneralOverlappingCommunities

goc = GeneralOverlappingCommunities(
    n_nodes=200,
    n_communities=8,
    community_generation="preferential",
    average_community_size=25,
    overlap_factor=0.3,
    membership_strength_dist="beta",
    base_edge_prob=0.15,
    random_state=42
)

G = goc.generate_graph()
```

## Patch Integration

**File:** `patch_integration.py`

This module provides utilities to convert community structures from the random graph models into patch format compatible with your existing L2GX alignment algorithms.

**Key Components:**

### CommunityToPatchConverter
Converts communities to `Patch` objects and creates `TGraph` structures:

```python
from patch_integration import CommunityToPatchConverter

converter = CommunityToPatchConverter(min_patch_size=10)
patches = converter.communities_to_patches(graph, community_nodes, embeddings)
patch_graph = converter.create_patch_graph(patches)

# Now compatible with existing alignment algorithms
from l2gx.align import get_aligner
aligner = get_aligner("l2g")
aligner.align_patches(patch_graph)
```

### CommunityTestCase
Complete test case generator that creates graph, embeddings, and patches:

```python
from patch_integration import CommunityTestCase

test_case = CommunityTestCase(
    "agm",  # or "osbm", "roc"
    model_params={...},
    embedding_params={"embedding_dim": 64},
    random_state=42
)

graph, embeddings, patches, patch_graph = test_case.generate()

# Evaluate alignment performance against ground truth
metrics = test_case.evaluate_patch_recovery(detected_patches)
```

## Usage Examples

### Quick Start
```python
# Generate a test graph with overlapping communities
from agm import CommunityAffiliationGraphModel

model = CommunityAffiliationGraphModel(
    n_nodes=100, 
    n_communities=5,
    random_state=42
)
G = model.generate_graph()

# Visualize the result
model.visualize()

# Get statistics
stats = model.compute_statistics()
print(f"Generated {stats['n_edges']} edges with {stats['avg_communities_per_node']:.2f} avg communities per node")
```

### Integration with L2GX Pipeline
```python
from patch_integration import CommunityTestCase

# Create test case
test_case = CommunityTestCase("roc", {
    "n_nodes": 150,
    "n_communities": 6,
    "community_generation": "grown",
    "average_community_size": 25
})

# Generate complete test data
graph, embeddings, patches, patch_graph = test_case.generate()

# Test your alignment algorithm
from l2gx.align import get_aligner
aligner = get_aligner("geo", method="orthogonal")
aligner.align_patches(patch_graph, use_scale=True)

# Evaluate results
ground_truth = test_case.get_ground_truth_communities()
metrics = test_case.evaluate_patch_recovery(aligner.patches)
print(f"Recovery rate: {metrics['recovery_rate']:.2f}")
```

### Comprehensive Testing Suite
```python
from patch_integration import create_standard_test_suite

# Get standard test cases for all three models
test_suite = create_standard_test_suite()

for test_case in test_suite:
    graph, embeddings, patches, patch_graph = test_case.generate()
    
    # Run your alignment algorithm
    # ... test alignment performance
    
    # Evaluate against ground truth
    metrics = test_case.evaluate_patch_recovery(detected_patches)
    print(f"Model: {test_case.model_type}, Recovery: {metrics['recovery_rate']:.2f}")
```

## Demo Scripts

- **`demo_all_models.py`**: Comprehensive comparison of all models
- **`agm.py`, `osbm.py`, `petti_vempala_roc.py`, `general_overlapping_communities.py`**: Individual model demos (run as main)
- **`patch_integration.py`**: Patch integration demonstration

## Model Comparison

| Model | Generation Speed | Community Quality | Overlap Control | Parameter Complexity | Best Use Case |
|-------|-----------------|------------------|-----------------|-------------------|---------------|
| **AGM** | Fast | High | Medium | Low | Basic patch detection |
| **OSBM** | Medium | High | High | Medium | Block-structured alignment |
| **Petti-Vempala ROC** | Fast | High | Low | Low | Theoretical analysis |
| **General Overlapping** | Slow | Very High | Very High | High | Realistic community structures |

## Installation Requirements

The models require:
- `numpy`
- `networkx`
- `matplotlib` (for visualization)
- `scipy` (for some distributions)
- `pandas` (for demo scripts)

For patch integration:
- Your existing L2GX codebase
- `torch` (for TGraph compatibility)

## Files Overview

```
scripts/random_graph/
├── __init__.py                           # Module initialization
├── agm.py                               # Community-Affiliation Graph Model
├── osbm.py                              # Overlapping Stochastic Block Model  
├── petti_vempala_roc.py                 # Petti-Vempala ROC Model (theoretical)
├── general_overlapping_communities.py   # General Overlapping Communities Model
├── patch_integration.py                 # Integration with L2GX patch pipeline
├── demo_all_models.py                   # Comprehensive demo and comparison
└── README.md                            # This file
```

## Next Steps

1. **Test Generation**: Use these models to generate diverse test cases for your patch alignment algorithms
2. **Performance Evaluation**: Compare alignment performance across different community structures
3. **Parameter Tuning**: Experiment with different model parameters to create challenging test scenarios
4. **Ground Truth Validation**: Use the known community structure to validate embedding quality
5. **Scaling Studies**: Test how algorithms perform as graph size and community complexity increase

The models provide a comprehensive foundation for testing patch-based algorithms across a wide range of realistic and controllable community structures.