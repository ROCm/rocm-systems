#!/usr/bin/env python3
"""
Extract and parse .hip_fatbin section from a shared library.

Usage: extract_fatbin.py <librccl.so> [output_dir]

The .hip_fatbin section may contain multiple components:
- CLANG_OFFLOAD_BUNDLE: Standard offload bundle with embedded device ELFs
- CCOB: Compressed Code OBject (used by --offload-compress)

Each component is written to a separate file in output_dir.
"""

import sys
import os
import struct
import subprocess
import tempfile
from pathlib import Path

CLANG_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
CCOB_MAGIC = b"CCOB"


def extract_hip_fatbin(so_path, output_path):
    """Extract .hip_fatbin section using llvm-objcopy."""
    llvm_objcopy = "/opt/rocm/llvm/bin/llvm-objcopy"
    if not os.path.exists(llvm_objcopy):
        llvm_objcopy = "llvm-objcopy"  # Try PATH
    
    result = subprocess.run(
        [llvm_objcopy, f"--dump-section=.hip_fatbin={output_path}", so_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract .hip_fatbin: {result.stderr}")
    
    if not os.path.exists(output_path):
        raise RuntimeError("llvm-objcopy succeeded but output file not created")
    
    return os.path.getsize(output_path)


def parse_clang_bundle(data, offset):
    """
    Parse a CLANG_OFFLOAD_BUNDLE starting at offset.
    Returns (bundle_info, end_offset) where bundle_info is a dict with metadata.
    """
    if data[offset:offset+24] != CLANG_BUNDLE_MAGIC:
        return None, offset
    
    pos = offset + 24
    num_entries = struct.unpack_from('<Q', data, pos)[0]
    pos += 8
    
    entries = []
    max_end = pos  # Track the furthest byte used by any entry
    
    for i in range(num_entries):
        entry_offset = struct.unpack_from('<Q', data, pos)[0]
        pos += 8
        entry_size = struct.unpack_from('<Q', data, pos)[0]
        pos += 8
        name_len = struct.unpack_from('<Q', data, pos)[0]
        pos += 8
        name = data[pos:pos+name_len].decode('utf-8', errors='replace').rstrip('\x00')
        pos += name_len
        
        entries.append({
            'name': name,
            'offset': entry_offset,  # Relative to start of bundle
            'size': entry_size
        })
        
        entry_end = entry_offset + entry_size
        if entry_end > max_end:
            max_end = entry_end
    
    # The bundle size is from the magic to the end of the last entry
    # Entries use offsets relative to the start of the bundle (offset parameter)
    bundle_size = max_end
    
    return {
        'type': 'CLANG_OFFLOAD_BUNDLE',
        'offset': offset,
        'size': bundle_size,
        'num_entries': num_entries,
        'entries': entries,
        'header_end': pos  # Where entry descriptors end
    }, offset + bundle_size


def parse_ccob(data, offset):
    """
    Parse a CCOB (Compressed Code OBject) starting at offset.
    Returns (ccob_info, end_offset).
    
    CCOB format (based on observation):
    - Magic: "CCOB" (4 bytes)
    - Version: uint16 (2 bytes)
    - Flags: uint16 (2 bytes)
    - Followed by zstd compressed data
    """
    if data[offset:offset+4] != CCOB_MAGIC:
        return None, offset
    
    # Read header
    version = struct.unpack_from('<H', data, offset + 4)[0]
    flags = struct.unpack_from('<H', data, offset + 6)[0]
    
    # CCOB doesn't have an explicit size field in the header
    # We need to find the end by looking for the next magic or EOF
    # Look for next CCOB or CLANG_BUNDLE_MAGIC or EOF
    pos = offset + 8
    end_pos = len(data)
    
    while pos < len(data) - 4:
        if data[pos:pos+4] == CCOB_MAGIC:
            end_pos = pos
            break
        if data[pos:pos+24] == CLANG_BUNDLE_MAGIC:
            end_pos = pos
            break
        pos += 1
    
    ccob_size = end_pos - offset
    
    return {
        'type': 'CCOB',
        'offset': offset,
        'size': ccob_size,
        'version': version,
        'flags': flags
    }, end_pos


def parse_fatbin(data):
    """
    Parse the entire .hip_fatbin data and return list of components.
    """
    components = []
    pos = 0
    
    while pos < len(data):
        # Skip zero padding
        while pos < len(data) and data[pos] == 0:
            pos += 1
        
        if pos >= len(data):
            break
        
        # Try to parse as CLANG_OFFLOAD_BUNDLE
        if pos + 24 <= len(data) and data[pos:pos+24] == CLANG_BUNDLE_MAGIC:
            info, new_pos = parse_clang_bundle(data, pos)
            if info:
                components.append(info)
                pos = new_pos
                continue
        
        # Try to parse as CCOB
        if pos + 4 <= len(data) and data[pos:pos+4] == CCOB_MAGIC:
            info, new_pos = parse_ccob(data, pos)
            if info:
                components.append(info)
                pos = new_pos
                continue
        
        # Unknown format - report error
        preview = data[pos:pos+20].hex() if pos + 20 <= len(data) else data[pos:].hex()
        raise RuntimeError(f"Unknown format at offset 0x{pos:x}: {preview}...")
    
    return components


def write_components(data, components, output_dir):
    """Write each component to a separate file."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, comp in enumerate(components):
        comp_type = comp['type']
        offset = comp['offset']
        size = comp['size']
        
        if comp_type == 'CLANG_OFFLOAD_BUNDLE':
            filename = f"bundle_{i}.bin"
        elif comp_type == 'CCOB':
            filename = f"ccob_{i}.bin"
        else:
            filename = f"unknown_{i}.bin"
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(data[offset:offset+size])
        
        print(f"  Wrote {filepath} ({size} bytes)")
        comp['file'] = filepath


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <librccl.so> [output_dir]")
        sys.exit(1)
    
    so_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "fatbin_components"
    
    if not os.path.exists(so_path):
        print(f"Error: {so_path} not found")
        sys.exit(1)
    
    print(f"Extracting .hip_fatbin from {so_path}...")
    
    # Extract .hip_fatbin section
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        fatbin_path = tmp.name
    
    try:
        fatbin_size = extract_hip_fatbin(so_path, fatbin_path)
        print(f"  Extracted {fatbin_size} bytes ({fatbin_size/(1024*1024):.2f} MB)")
        
        # Read fatbin data
        with open(fatbin_path, 'rb') as f:
            data = f.read()
        
        # Parse components
        print(f"\nParsing .hip_fatbin...")
        components = parse_fatbin(data)
        
        print(f"\nFound {len(components)} component(s):")
        for i, comp in enumerate(components):
            print(f"\n  [{i}] {comp['type']}")
            print(f"      Offset: 0x{comp['offset']:x}")
            print(f"      Size: {comp['size']} bytes ({comp['size']/(1024*1024):.2f} MB)")
            
            if comp['type'] == 'CLANG_OFFLOAD_BUNDLE':
                print(f"      Entries: {comp['num_entries']}")
                for entry in comp['entries']:
                    print(f"        - {entry['name']}: offset=0x{entry['offset']:x}, size={entry['size']}")
            elif comp['type'] == 'CCOB':
                print(f"      Version: {comp['version']}")
                print(f"      Flags: 0x{comp['flags']:x}")
        
        # Check for unconsumed data
        total_consumed = sum(c['size'] for c in components)
        # Account for padding between components
        last_end = max(c['offset'] + c['size'] for c in components) if components else 0
        
        if last_end < len(data):
            # Check if remaining is just zeros
            remaining = data[last_end:]
            non_zero = sum(1 for b in remaining if b != 0)
            if non_zero > 0:
                print(f"\nWarning: {len(remaining)} bytes after last component ({non_zero} non-zero)")
            else:
                print(f"\nNote: {len(remaining)} bytes of zero padding after last component")
        
        # Write components
        print(f"\nWriting components to {output_dir}/...")
        write_components(data, components, output_dir)
        
        print(f"\nDone!")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(fatbin_path):
            os.unlink(fatbin_path)


if __name__ == "__main__":
    main()
