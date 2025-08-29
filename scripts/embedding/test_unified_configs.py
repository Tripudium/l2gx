#!/usr/bin/env python3
"""
Test Script for Unified Configuration Framework

Tests that all unified configuration files work correctly.
"""

import subprocess
import sys
from pathlib import Path

def test_config(config_path: Path) -> bool:
    """Test a single configuration file."""
    print(f"\nTesting: {config_path.name}")
    print("-" * 40)
    
    cmd = [sys.executable, "run_embedding_config.py", str(config_path)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Check for success indicators in output
            if "EXPERIMENT COMPLETED" in result.stdout:
                print(f"✓ SUCCESS: {config_path.name}")
                # Extract embedding shape from output
                for line in result.stdout.split('\n'):
                    if "Generated embedding:" in line:
                        print(f"  {line.strip()}")
                        break
                return True
            else:
                print(f"✗ FAILED: Experiment did not complete")
                return False
        else:
            print(f"✗ FAILED: Return code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ FAILED: Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def main():
    """Test all unified configuration files."""
    print("TESTING UNIFIED CONFIGURATION FRAMEWORK")
    print("=" * 60)
    
    config_dir = Path("config")
    unified_configs = list(config_dir.glob("unified_*.yaml"))
    
    if not unified_configs:
        print("No unified configuration files found in config/")
        return 1
    
    print(f"Found {len(unified_configs)} unified configuration files:")
    for config in unified_configs:
        print(f"  - {config.name}")
    
    results = {}
    for config in unified_configs:
        results[config.name] = test_config(config)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    for config_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {config_name}")
    
    print(f"\nTotal: {successful}/{total} configurations passed")
    
    if successful == total:
        print("\n🎉 All unified configurations work correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - successful} configuration(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())