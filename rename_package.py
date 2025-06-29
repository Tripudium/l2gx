#!/usr/bin/env python3
"""
Package Renaming Script: L2Gv2 → L2GX

This script systematically renames the package from l2gv2 to l2gx throughout the codebase.
It handles:
- Python import statements
- Documentation strings
- README files
- Configuration files
- Comments and references
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_files_to_update(root_dir: Path) -> List[Path]:
    """Find all files that need to be updated"""
    extensions = {'.py', '.md', '.rst', '.txt', '.toml', '.cfg', '.ini', '.yaml', '.yml'}
    
    files_to_update = []
    
    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            # Skip certain directories
            skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
            if any(part in skip_dirs for part in file_path.parts):
                continue
            
            files_to_update.append(file_path)
    
    return files_to_update


def get_replacement_patterns() -> List[Tuple[str, str]]:
    """Get all the text patterns that need to be replaced"""
    return [
        # Python imports
        (r'from l2gv2', 'from l2gx'),
        (r'import l2gv2', 'import l2gx'),
        (r'l2gv2\.', 'l2gx.'),
        
        # Documentation and strings
        (r'L2Gv2', 'L2GX'),
        (r'L2GV2', 'L2GX'),
        (r'l2gv2', 'l2gx'),
        
        # Path references
        (r'/l2gv2/', '/l2gx/'),
        (r'\\l2gv2\\', '\\l2gx\\'),
        
        # Package names in setup files
        (r'"l2gv2"', '"l2gx"'),
        (r"'l2gv2'", "'l2gx'"),
        
        # URLs and repository references
        (r'L2Gv2', 'L2GX'),
    ]


def update_file_content(file_path: Path, patterns: List[Tuple[str, str]]) -> bool:
    """Update a single file with all replacement patterns"""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all replacements
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def update_directory_structure(root_dir: Path):
    """Update any remaining directory names if needed"""
    # The main directory was already renamed, check for any nested ones
    for dir_path in root_dir.rglob('*'):
        if dir_path.is_dir() and 'l2gv2' in dir_path.name.lower():
            new_name = dir_path.name.replace('l2gv2', 'l2gx').replace('L2Gv2', 'L2GX')
            new_path = dir_path.parent / new_name
            try:
                dir_path.rename(new_path)
                print(f"Renamed directory: {dir_path} → {new_path}")
            except Exception as e:
                print(f"Could not rename directory {dir_path}: {e}")


def main():
    """Main function to perform the renaming"""
    root_dir = Path('.')
    
    print("🔄 Starting L2Gv2 → L2GX package renaming...")
    print(f"Working directory: {root_dir.absolute()}")
    
    # Get replacement patterns
    patterns = get_replacement_patterns()
    print(f"Replacement patterns: {len(patterns)}")
    
    # Find files to update
    print("📁 Finding files to update...")
    files_to_update = find_files_to_update(root_dir)
    print(f"Found {len(files_to_update)} files to check")
    
    # Update files
    print("✏️  Updating file contents...")
    updated_count = 0
    
    for i, file_path in enumerate(files_to_update):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(files_to_update)} files processed")
        
        if update_file_content(file_path, patterns):
            updated_count += 1
            if updated_count <= 10:  # Show first 10 updated files
                print(f"  ✓ Updated: {file_path}")
            elif updated_count == 11:
                print(f"  ... (showing first 10, continuing silently)")
    
    # Update directory structure
    print("📂 Checking directory structure...")
    update_directory_structure(root_dir)
    
    # Summary
    print("\n✅ Renaming complete!")
    print(f"📊 Summary:")
    print(f"  - Files checked: {len(files_to_update)}")
    print(f"  - Files updated: {updated_count}")
    print(f"  - Package renamed: l2gv2 → l2gx")
    
    print("\n🔍 Next steps:")
    print("  1. Check that imports work: python -c 'import l2gx; print(\"Success!\")'")
    print("  2. Run tests to ensure everything works")
    print("  3. Update any external documentation")
    print("  4. Initialize new git repository")


if __name__ == "__main__":
    main()