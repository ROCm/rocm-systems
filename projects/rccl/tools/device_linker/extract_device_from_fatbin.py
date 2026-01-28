#!/usr/bin/env python3
"""
Extract device code from a HIP object file's .hip_fatbin section.
Handles both uncompressed (__CLANG_OFFLOAD_BUNDLE__) and compressed (CCOB) formats.
"""

import struct
import subprocess
import sys
import os
import tempfile

def extract_device_from_obj(input_obj: str, output_device: str, target_arch: str = "gfx942"):
    """Extract device code from a HIP object file."""
    
    # Use llvm-objcopy to extract the .hip_fatbin section
    fatbin_bin = "/tmp/fatbin_extract_tmp.bin"
    cmd = ["/opt/rocm/llvm/bin/llvm-objcopy", "--dump-section=.hip_fatbin=" + fatbin_bin, input_obj]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting .hip_fatbin: {result.stderr}", file=sys.stderr)
        return False
    
    if not os.path.exists(fatbin_bin):
        print(f"Error: .hip_fatbin section not found in {input_obj}", file=sys.stderr)
        return False
    
    with open(fatbin_bin, 'rb') as f:
        magic = f.read(4)
        f.seek(0)
        data = f.read()
    
    device_data = None
    
    if magic == b'CCOB':
        # Compressed Code Object Bundle format
        print("Detected CCOB (compressed) format")
        device_data = extract_from_ccob(data, target_arch)
    elif magic == b'__CL':
        # Uncompressed __CLANG_OFFLOAD_BUNDLE__ format
        print("Detected uncompressed bundle format")
        device_data = extract_from_uncompressed(data, target_arch)
    else:
        print(f"Unknown fatbin format: {magic}", file=sys.stderr)
        return False
    
    if device_data is None:
        print(f"Failed to extract device code for {target_arch}", file=sys.stderr)
        return False
    
    with open(output_device, 'wb') as f:
        f.write(device_data)
    
    print(f"Extracted {len(device_data)} bytes to {output_device}")
    
    # Cleanup
    os.remove(fatbin_bin)
    return True


def extract_from_ccob(data: bytes, target_arch: str) -> bytes:
    """Extract device code from CCOB (Compressed Code Object Bundle) format."""
    # CCOB header format:
    # 4 bytes: magic "CCOB"
    # 2 bytes: version major
    # 2 bytes: version minor
    # 8 bytes: compressed size
    # 8 bytes: uncompressed size
    # 16 bytes: hash/checksum
    # Then zstd compressed data
    
    if len(data) < 40:
        return None
    
    magic = data[0:4]
    if magic != b'CCOB':
        return None
    
    version_major = struct.unpack_from('<H', data, 4)[0]
    version_minor = struct.unpack_from('<H', data, 6)[0]
    compressed_size = struct.unpack_from('<Q', data, 8)[0]
    uncompressed_size = struct.unpack_from('<Q', data, 16)[0]
    
    print(f"  CCOB version: {version_major}.{version_minor}")
    print(f"  Compressed: {compressed_size}, Uncompressed: {uncompressed_size}")
    
    # The compressed data starts after the header (32 bytes for CCOB v3.1)
    header_size = 32
    compressed_data = data[header_size:header_size + compressed_size]
    
    # Use system zstd to decompress
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zst') as tmp_in:
            tmp_in.write(compressed_data)
            tmp_in_path = tmp_in.name
        
        tmp_out_path = tmp_in_path.replace('.zst', '.bin')
        result = subprocess.run(['zstd', '-d', tmp_in_path, '-o', tmp_out_path, '-f'],
                                capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  zstd decompression failed: {result.stderr}", file=sys.stderr)
            return None
        
        with open(tmp_out_path, 'rb') as f:
            decompressed = f.read()
        
        os.remove(tmp_in_path)
        os.remove(tmp_out_path)
        print(f"  Decompressed {len(decompressed)} bytes")
    except Exception as e:
        print(f"  Decompression failed: {e}", file=sys.stderr)
        return None
    
    # The decompressed data should be in __CLANG_OFFLOAD_BUNDLE__ format
    return extract_from_uncompressed(decompressed, target_arch)


def extract_from_uncompressed(data: bytes, target_arch: str) -> bytes:
    """Extract device code from uncompressed __CLANG_OFFLOAD_BUNDLE__ format."""
    magic = data[0:24]
    if magic != b'__CLANG_OFFLOAD_BUNDLE__':
        print(f"  Invalid bundle magic: {magic[:24]}", file=sys.stderr)
        return None
    
    num_bundles = struct.unpack_from('<Q', data, 24)[0]
    print(f"  Number of bundles: {num_bundles}")
    
    offset = 32
    for i in range(num_bundles):
        bundle_offset = struct.unpack_from('<Q', data, offset)[0]
        bundle_size = struct.unpack_from('<Q', data, offset + 8)[0]
        str_len = struct.unpack_from('<Q', data, offset + 16)[0]
        target = data[offset + 24:offset + 24 + str_len].decode('ascii')
        offset += 24 + str_len
        
        print(f"  Bundle {i}: '{target}' offset={bundle_offset} size={bundle_size}")
        
        if target_arch in target and bundle_size > 0:
            return data[bundle_offset:bundle_offset + bundle_size]
    
    return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.o> <output_device.o> [target_arch]", file=sys.stderr)
        sys.exit(1)
    
    input_obj = sys.argv[1]
    output_device = sys.argv[2]
    target_arch = sys.argv[3] if len(sys.argv) > 3 else "gfx942"
    
    if extract_device_from_obj(input_obj, output_device, target_arch):
        sys.exit(0)
    else:
        sys.exit(1)
