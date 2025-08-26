# Alignment

Alignment on a random, clustered point cloud with noise.

## Files

### Scripts
- `synthetic_data_experiment.py` - Main experiment script that generates the noise robustness data
- `data_generation.py` - Module containing functions for generating synthetic data and patches
- `plot_synthetic_data.py` - Script to generate the plot from CSV results

### Configuration and Data
- `data_config.yaml` - Configuration file specifying experiment parameters
- `noise_robustness_results.csv` - Pre-computed results from noise robustness experiments

**Note**: The experiment generates synthetic data on-the-fly, so no pre-saved data files are needed.

## Usage

### Option 1: Use Pre-computed Results

Generate the plot from the existing CSV data:
```bash
python plot_synthetic_data.py noise_robustness_results.csv
```

This will create `noise_robustness_plot_with_errorbars.pdf` in the current directory.

### Option 2: Run Full Experiment (Time-intensive)

To regenerate the noise robustness data from scratch:

```bash
# This will take a long time (tests 31 noise levels × 3 dimensions × 10 repetitions)
python synthetic_data_experiment.py
```

The experiment will:
1. Load configuration from `data_config.yaml`
2. Test noise levels from 0.0 to 0.3 (31 levels)
3. Test dimensions 32, 64, and 128
4. Run 10 repetitions per configuration
5. Save results to `noise_robustness_results/noise_robustness_results.csv`
6. Generate multiple plots in `noise_robustness_results/`


### Custom Plot Output

Specify custom output filename:
```bash
python plot_synthetic_data.py noise_robustness_results.csv -o my_plot.pdf
```

### Get Help
```bash
python plot_synthetic_data.py -h
```

## CSV Format

The script expects a CSV file with the following columns:
- `dimension` - The embedding dimension (32, 64, or 128)
- `noise_level` - The noise level (0.0 to 0.3)
- `avg_procrustes_error` - Average Procrustes error across repetitions
- `std_procrustes_error` - Standard deviation of Procrustes error

## Output

The script generates a PDF plot showing:
- Three lines for dimensions 32, 64, and 128
- Error bars showing standard deviation
- X-axis: Noise level
- Y-axis: Procrustes error
- Legend indicating dimension