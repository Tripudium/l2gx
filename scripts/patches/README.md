# Patch Size Analysis

This directory contains analysis of average patch sizes for different subdivision configurations across Cora and PubMed datasets.

## Experiment Overview

We analyzed how subdividing graphs into different numbers of patches (2, 4, 6, 8, 10, 15, 20) affects the average patch size for:
- **Cora**: 2,708 nodes, 10,556 edges
- **PubMed**: 19,717 nodes, 88,648 edges

## Key Results

### Average Patch Sizes by Number of Patches

| Patches | Cora Avg Size | PubMed Avg Size | Cora Theoretical | PubMed Theoretical |
|---------|---------------|-----------------|------------------|-------------------|
| **2**   | 1,610 nodes   | 10,774 nodes    | 1,354 nodes      | 9,859 nodes       |
| **4**   | 1,067 nodes   | 5,319 nodes     | 677 nodes        | 4,929 nodes       |
| **6**   | 885 nodes     | 3,741 nodes     | 451 nodes        | 3,286 nodes       |
| **8**   | 810 nodes     | 2,936 nodes     | 339 nodes        | 2,465 nodes       |
| **10**  | 778 nodes     | 2,479 nodes     | 271 nodes        | 1,972 nodes       |
| **15**  | ⚠️ Failed     | 1,835 nodes     | 181 nodes        | 1,314 nodes       |
| **20**  | ⚠️ Failed     | 1,493 nodes     | 135 nodes        | 986 nodes         |

### Key Observations

1. **Patch Size Scaling**: Both datasets show predictable scaling where more patches result in smaller average patch sizes, but not linearly due to overlap requirements.

2. **Overlap Overhead**: Actual patch sizes are consistently **larger** than theoretical sizes (total_nodes/num_patches) due to overlapping regions between patches:
   - **Cora**: 1.19× to 2.87× larger than theoretical
   - **PubMed**: 1.08× to 1.51× larger than theoretical

3. **Dataset Differences**: 
   - **Cora** shows higher size efficiency ratios, meaning more overhead from overlaps
   - **PubMed** is more efficient at higher patch counts (closer to theoretical sizes)

4. **Limitations**: Cora fails at 15+ patches due to the graph being too small relative to the minimum overlap requirements (256 nodes per overlap).

### Size Efficiency Analysis

**Size Efficiency** = Actual Patch Size / Theoretical Patch Size

| Patches | Cora Efficiency | PubMed Efficiency |
|---------|-----------------|-------------------|
| 2       | 1.19×           | 1.09×             |
| 4       | 1.58×           | 1.08×             |
| 6       | 1.96×           | 1.14×             |
| 8       | 2.39×           | 1.19×             |
| 10      | 2.87×           | 1.26×             |
| 15      | Failed          | 1.40×             |
| 20      | Failed          | 1.51×             |

## Technical Details

### Patch Configuration
All experiments used consistent parameters:
- **Clustering method**: METIS
- **Min overlap**: 256 nodes
- **Target overlap**: 512 nodes
- **Sparsify method**: Resistance
- **Target patch degree**: 4

### Error Analysis
Cora failed at 15 and 20 patches due to:
```
The expanded size of the tensor (1) must match the existing size (0) at non-singleton dimension 0
```

This occurs when the required overlap size approaches the total graph size, making it impossible to create meaningful patches.

## Practical Implications

### For Cora (Small Graph: ~2.7K nodes)
- **Optimal range**: 2-10 patches
- **Recommended**: 4-8 patches for balance of granularity and efficiency
- **Avoid**: 15+ patches (computational failures)

### For PubMed (Large Graph: ~19.7K nodes)  
- **Full range viable**: 2-20 patches all work well
- **High efficiency**: Maintains good size efficiency even at 20 patches
- **Flexible**: Can scale to larger patch counts if needed

### General Guidelines
1. **Small graphs** (<5K nodes): Limit to ≤10 patches
2. **Large graphs** (>15K nodes): Can effectively use 15-20+ patches
3. **Overlap requirements** create a lower bound on viable patch counts
4. **Size efficiency decreases** with more patches, especially for smaller graphs

## Files

- `patch_size_analysis.py`: Main analysis script
- `results/patch_size_analysis.csv`: Detailed results for all configurations
- `results/patch_size_summary.csv`: Summary table with key metrics
- `patch_size_analysis.png`: Visualization of results

## Usage

```bash
# Run the complete patch size analysis
python patch_size_analysis.py

# This will:
# 1. Test both Cora and PubMed datasets
# 2. Try 2, 4, 6, 8, 10, 15, and 20 patches for each
# 3. Generate visualizations and save results
```

The analysis provides essential guidance for choosing appropriate patch counts in L2GX embedding experiments based on dataset size and computational constraints.