#!/usr/bin/env python3
"""
Test streaming patch generation on a small MAG240M subset.

This script demonstrates the lightweight streaming patch generation system
on a manageable subset of MAG240M data before deploying to high-performance platforms.

Usage:
    python test_streaming_patches.py [--num-papers 10000] [--num-patches 10]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from l2gx.datasets import get_dataset
from l2gx.patch.streaming import StreamingPatchGenerator, load_streaming_patches


def test_small_streaming_patches():
    """Test streaming patch generation on small MAG240M subset."""

    parser = argparse.ArgumentParser(description="Test streaming patches on MAG240M subset")
    parser.add_argument("--num-papers", type=int, default=5000,
                       help="Number of papers in subset (default: 5000)")
    parser.add_argument("--num-patches", type=int, default=5,
                       help="Number of patches to create (default: 5)")
    parser.add_argument("--min-year", type=int, default=2018,
                       help="Minimum paper year (default: 2018)")
    parser.add_argument("--patch-dir", type=str, default="patches_test",
                       help="Patch storage directory (default: patches_test)")
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreate patches even if they exist")

    args = parser.parse_args()

    print("MAG240M Streaming Patches Test")
    print("=" * 50)
    print("Parameters:")
    print(f"  Papers: {args.num_papers:,}")
    print(f"  Patches: {args.num_patches}")
    print(f"  Min year: {args.min_year}")
    print(f"  Patch dir: {args.patch_dir}")
    print()

    try:
        # Step 1: Load small MAG240M subset
        print("1. Loading MAG240M subset...")
        dataset = get_dataset(
            "MAG240M",
            subset_strategy="recent_papers",
            max_papers=args.num_papers,
            min_year=args.min_year,
            cache_subsets=True
        )

        print(f"   Dataset: {dataset}")

        stats = dataset.get_subset_statistics()
        print(f"   Subset: {stats.get('num_nodes', 0):,} nodes, {stats.get('num_edges', 0):,} edges")

        if stats.get('num_nodes', 0) == 0:
            print("   ⚠️  Empty dataset - check if MAG240M data is available")
            return False

        # Step 2: Create streaming patch generator
        print("\n2. Creating streaming patch generator...")
        patch_dir = Path(__file__).parent / args.patch_dir

        if args.force_recreate and patch_dir.exists():
            import shutil
            shutil.rmtree(patch_dir)
            print(f"   Removed existing patches: {patch_dir}")

        generator = StreamingPatchGenerator(
            dataset=dataset,
            num_patches=args.num_patches,
            patch_dir=patch_dir,
            verbose=True
        )

        # Step 3: Generate patches
        print("\n3. Generating patches...")
        patch_graph = generator.create_patches()

        print(f"   Patch graph: {patch_graph}")
        print(f"   Number of patches: {len(patch_graph.patches)}")

        # Step 4: Test patch access and properties
        print("\n4. Testing patch access...")

        total_nodes = 0
        for i, patch in enumerate(patch_graph.patches):
            patch_size = len(patch)
            total_nodes += patch_size

            print(f"   Patch {i}: {patch_size:,} nodes")

            # Test lazy loading
            if patch_size > 0:
                first_node = patch.nodes[0]
                print(f"     First node: {first_node}")

                # Test TGraph conversion
                tgraph = patch.to_tgraph()
                print(f"     TGraph: {tgraph.num_nodes} nodes, {tgraph.num_edges} edges")

                # Test coordinates (placeholder)
                coords_shape = patch.coordinates.shape
                print(f"     Coordinates: {coords_shape}")

        print(f"   Total nodes across patches: {total_nodes:,}")

        # Step 5: Test patch reloading
        print("\n5. Testing patch reloading...")

        # Clear current patches from memory (simulate restart)
        del patch_graph, generator

        # Reload patches
        reloaded_patch_graph = load_streaming_patches(patch_dir)
        print(f"   Reloaded patch graph: {reloaded_patch_graph}")
        print(f"   Number of patches: {len(reloaded_patch_graph.patches)}")

        # Test a few patches
        for i in range(min(3, len(reloaded_patch_graph.patches))):
            patch = reloaded_patch_graph.patches[i]
            print(f"   Reloaded patch {i}: {len(patch):,} nodes")

        # Step 6: Memory usage and file size analysis
        print("\n6. Storage analysis...")

        total_size = 0
        for file_path in patch_dir.rglob("*.parquet"):
            file_size = file_path.stat().st_size
            total_size += file_size
            print(f"   {file_path.name}: {file_size / 1024:.1f} KB")

        print(f"   Total storage: {total_size / (1024 * 1024):.2f} MB")
        print(f"   Storage per node: {total_size / total_nodes:.1f} bytes/node")

        # Step 7: Compatibility test with existing L2GX interface
        print("\n7. Testing L2GX compatibility...")

        print("   ✓ Patch graph has required attributes:")
        print(f"     - patches: {len(patch_graph.patches)}")
        print(f"     - overlap_nodes: {len(patch_graph.overlap_nodes)} pairs")

        print("   ✓ Patches are compatible with Patch interface:")
        for i, patch in enumerate(reloaded_patch_graph.patches[:2]):
            print(f"     - Patch {i}: nodes={len(patch.nodes)}, shape={patch.shape}")
            print(f"       has get_coordinates: {hasattr(patch, 'get_coordinates')}")
            print(f"       has get_coordinate: {hasattr(patch, 'get_coordinate')}")

        print("\n✅ Streaming patches test completed successfully!")
        print(f"   Patches saved to: {patch_dir}")
        print("   Ready for high-performance deployment")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure OGB library is installed: pip install ogb")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_usage_examples():
    """Show usage examples for different scenarios."""

    print("\nUsage Examples:")
    print("=" * 30)

    examples = [
        {
            "name": "Small test (quick)",
            "cmd": "python test_streaming_patches.py --num-papers 1000 --num-patches 3"
        },
        {
            "name": "Medium test",
            "cmd": "python test_streaming_patches.py --num-papers 10000 --num-patches 10"
        },
        {
            "name": "Large test (for HPC)",
            "cmd": "python test_streaming_patches.py --num-papers 100000 --num-patches 50"
        },
        {
            "name": "Force recreation",
            "cmd": "python test_streaming_patches.py --force-recreate --num-patches 5"
        }
    ]

    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  {example['cmd']}")


if __name__ == "__main__":
    # Show usage examples first
    if len(sys.argv) == 1:
        demonstrate_usage_examples()
        print()

    # Run the test
    success = test_small_streaming_patches()

    if not success:
        print("\n" + "="*50)
        print("Troubleshooting:")
        print("- Ensure OGB library is installed: pip install ogb")
        print("- Check MAG240M data is available (large download required)")
        print("- Try smaller subset size if memory issues occur")
        print("- Use --force-recreate to rebuild patches")

    sys.exit(0 if success else 1)
