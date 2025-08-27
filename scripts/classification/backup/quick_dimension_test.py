#!/usr/bin/env python3
"""
Quick test of dimension sweep experiment with reduced parameters for validation.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import the main experiment class
from dimension_sweep_experiment import DimensionSweepExperiment

def main():
    """Run quick test with reduced parameters."""
    print("Running quick dimension sweep test...")
    
    # Create experiment with reduced parameters
    experiment = DimensionSweepExperiment(output_dir="quick_test_results")
    
    # Override with smaller test parameters
    experiment.dimensions = [4, 16, 32]  # Just 3 dimensions
    experiment.n_runs = 2  # Just 2 runs per configuration
    
    print(f"Test parameters:")
    print(f"  Dimensions: {experiment.dimensions}")
    print(f"  Runs per config: {experiment.n_runs}")
    print(f"  Total experiments: {len(['full_graph', 'l2g_rademacher', 'hierarchical_l2g']) * len(experiment.dimensions) * experiment.n_runs}")
    
    try:
        # Run experiments
        experiment.run_all_experiments()
        
        # Analyze and plot
        summary = experiment.analyze_results()
        experiment.create_plots(summary)
        experiment.save_results()
        
        print("✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)