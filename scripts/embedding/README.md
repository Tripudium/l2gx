# L2G Embedding Scripts

This directory contains the complete pipeline for computing PubMed and Cora embeddings and generating the visualization shown in `l2g_embeddings.pdf`.

## Files

### Scripts
- `compute_embeddings.py` - Main script to compute L2G embeddings for both datasets
- `create_pdf.py` - Script to generate PDF visualizations from computed embeddings
- `run_embedding_config.py` - Configurable experiment runner using YAML configs
- `test_pipeline.py` - Test script to verify the pipeline works

### Generated Files (after running)
- `cora_results.npz` - Cora L2G embedding and metadata (2.8MB)
- `pubmed_results.npz` - PubMed L2G embedding and metadata (20.3MB)
- `l2g_embeddings.pdf` - Main visualization
- `l2g_embeddings_enhanced.pdf` - Alternative enhanced visualization

### Configuration Files
- `embedding_config_l2g.yaml` - L2G alignment configuration template
- `embedding_config_geo.yaml` - Geometric alignment configuration template  
- `embedding_config_hierarchical.yaml` - Hierarchical embedding configuration template
- `CONFIG_GUIDE.md` - Comprehensive guide to configuration options

## Usage

### Step 1: Compute Embeddings (Time-intensive)

**Warning**: This step takes a long time (several hours) due to high-quality VGAE training.

```bash
cd scripts/embedding
python compute_embeddings.py
```

This will:
1. Load Cora and PubMed datasets
2. Generate patches using METIS clustering
3. Train VGAE embeddings for each patch (10,000 epochs for quality)
4. Align patches using L2G with randomized Rademacher synchronization
5. Save results to `cora_results.npz` and `pubmed_results.npz`

### Step 2: Create Visualizations (Fast)

```bash
python create_pdf.py
```

This will:
1. Load the pre-computed embedding results
2. Apply UMAP dimensionality reduction for visualization
3. Use datashader to create enhanced, visible points
4. Generate `l2g_embeddings.pdf` and `l2g_embeddings_enhanced.pdf`

### Step 3: Run Configurable Experiments (Alternative)

Use YAML configuration files for flexible experimentation:

```bash
# Run L2G experiment with Rademacher sketching
python run_embedding_config.py embedding_config_l2g.yaml

# Run Geo experiment with 2 epochs  
python run_embedding_config.py embedding_config_geo.yaml

# Run hierarchical experiment
python run_embedding_config.py embedding_config_hierarchical.yaml
```

See `CONFIG_GUIDE.md` for detailed configuration options.

### Skip Step 1 (if you have the .npz files)

If you already have `cora_results.npz` and `pubmed_results.npz`, you can skip directly to Step 2.

## Algorithm Details

### L2G Embedding Pipeline
1. **Dataset Loading**: Load Cora (2,708 nodes) and PubMed (19,717 nodes) citation networks
2. **Patch Generation**: 
   - Cora: 10 patches with target degree 4
   - PubMed: 20 patches with target degree 5
   - Uses METIS clustering with resistance sparsification
3. **Local Embedding**: VGAE (Variational Graph Auto-Encoder)
   - 128-dimensional embeddings
   - 256-dimensional hidden layer
   - 10,000 training epochs for high quality
4. **Global Alignment**: L2G (Local2Global) with randomized Rademacher synchronization
   - Faster than standard eigenvalue decomposition
   - Maintains embedding quality

### Visualization Pipeline
1. **UMAP Projection**: Reduce 128D embeddings to 2D for visualization
   - Parameters tuned for spread: `n_neighbors=5, min_dist=0.5, spread=2.0`
2. **Enhanced Points**: Multiple techniques for visibility
   - Add offset points around each original point
   - Use smaller canvas resolution (300x300, 400x400)
   - Apply log scaling and high contrast colors
3. **Datashader Rendering**: High-performance visualization for large datasets
   - Anti-aliasing and proper color mixing
   - White background with bright, saturated colors

## Parameters

### Embedding Parameters
- **Embedding dimension**: 128
- **Hidden dimension**: 256 (2x embedding dimension)
- **Training epochs**: 10,000 (for publication quality)
- **Learning rate**: 0.001
- **Patience**: 20 (early stopping)

### Patch Parameters
- **Cora**: 10 patches, target degree 4, METIS clustering
- **PubMed**: 20 patches, target degree 5, METIS clustering
- **Overlap**: 256 minimum, 512 target
- **Sparsification**: Resistance-based with conductance weighting

### Visualization Parameters
- **UMAP**: `n_neighbors=5, min_dist=0.5, spread=2.0`
- **Colors**: Bright palette `['#0080FF', '#FF8000', '#00C000', ...]`
- **Canvas**: 400x400 for main version, 300x300 for enhanced version
- **Shading**: Log scaling with full alpha (255)

## Dependencies

- **Core**: numpy, torch, pathlib
- **L2GX**: All l2gx modules (datasets, patch, align, embedding, graphs)
- **Visualization**: matplotlib, polars, datashader, umap-learn
- **ML**: PyTorch Geometric (for datasets and VGAE)

## Output

The main output is `l2g_embeddings.pdf`, which shows:
- Side-by-side comparison of Cora (left) and PubMed (right)
- Enhanced, highly visible points colored by class
- Clean layout without axes, titles, or legends
- High-resolution (300 DPI) suitable for publication

## Performance Notes

- **Embedding computation**: Several hours (most time spent in VGAE training)
- **Visualization**: ~1 minute (UMAP + datashader rendering)
- **Memory usage**: ~2GB peak during PubMed embedding
- **Output size**: ~25MB total for both .npz files

## Troubleshooting

### Missing dependencies
```bash
pip install datashader umap-learn polars
```

### Import errors
Make sure you're running from the correct directory and that l2gx is properly installed.

### Out of memory
Reduce VGAE epochs or batch size if you encounter memory issues.