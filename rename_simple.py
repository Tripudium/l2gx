#!/usr/bin/env python3
"""
Simple Package Renaming Script: L2Gv2 → L2GX
"""

import os
import sys
from pathlib import Path


def update_file_content(file_path: Path) -> bool:
    """Update a single file with simple string replacement"""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Simple string replacements
        replacements = [
            ('from l2gv2', 'from l2gx'),
            ('import l2gv2', 'import l2gx'),
            ('l2gv2.', 'l2gx.'),
            ('L2Gv2', 'L2GX'),
            ('L2GV2', 'L2GX'),
            ('"l2gv2"', '"l2gx"'),
            ("'l2gv2'", "'l2gx'"),
            ('/l2gv2/', '/l2gx/'),
        ]
        
        # Apply replacements
        for old, new in replacements:
            content = content.replace(old, new)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def main():
    """Main function"""
    print("🔄 Starting simple L2Gv2 → L2GX renaming...")
    
    # Focus on our main files
    important_dirs = ['l2gx', 'examples', 'tests', 'docs', 'rust_clustering']
    important_files = ['README.md', 'pyproject.toml', 'RUST_SETUP.md', 'setup.py']
    
    files_to_update = []
    
    # Add important files in root
    for filename in important_files:
        file_path = Path(filename)
        if file_path.exists():
            files_to_update.append(file_path)
    
    # Add files in important directories
    for dir_name in important_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for file_path in dir_path.rglob('*.py'):
                files_to_update.append(file_path)
            for file_path in dir_path.rglob('*.md'):
                files_to_update.append(file_path)
            for file_path in dir_path.rglob('*.toml'):
                files_to_update.append(file_path)
            for file_path in dir_path.rglob('*.rst'):
                files_to_update.append(file_path)
    
    print(f"Found {len(files_to_update)} files to update")
    
    updated_count = 0
    for file_path in files_to_update:
        if update_file_content(file_path):
            updated_count += 1
            print(f"✓ Updated: {file_path}")
    
    print(f"\n✅ Renaming complete! Updated {updated_count} files")


if __name__ == "__main__":
    main()