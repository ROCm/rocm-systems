# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
A quick & rough script for testing the Cython bindings to the
hipFile C library. Reads a given file and copies it to an
output file, and then prints the hashes of the files.
"""

import hashlib
import os
import pathlib

from hipfile.hipMalloc import hipFree, hipMalloc

from hipfile import (
    Driver,
    FileHandle,
    Buffer,
    FileHandleType,
    get_version,
)

hipfile_version = get_version()

input_path = pathlib.Path("/mnt/ais/ext4/random_2MiB.bin")
output_path = pathlib.Path("/mnt/ais/ext4/output.bin")

print(f"hipFile Version: {hipfile_version}")
print(f"Driver Use Count Before: {Driver.use_count()}")

# Max to a 2GiB - 4KiB Buffer
# Note: Max IO in a single transaction is 2GiB - 4KiB as set by the Linux Kernel
#       Larger IOs will be quietly truncated.
size = min(input_path.stat().st_size, 2 * 1024 * 1024 * 1024 - 4 * 1024)
buffer = hipMalloc(size)
buffer_ptr = buffer.value  # pylint: disable=C0103  # False Positive
print(f"Buffer located at: {buffer_ptr} | {hex(buffer_ptr)}")

with Driver() as hipfile_driver:
    print(f"Driver Use Count After: {hipfile_driver.use_count()}")
    with Buffer.from_ctypes_void_p(buffer, size, 0) as registered_buffer:
        with FileHandle(
            input_path,
            os.O_RDWR | os.O_DIRECT | os.O_CREAT,
            handle_type=FileHandleType.OPAQUE_FD,
        ) as fh_input:
            with FileHandle(
                output_path, os.O_RDWR | os.O_DIRECT | os.O_CREAT | os.O_TRUNC
            ) as fh_output:
                print(f"Transferring {size} bytes...")
                bytes_read = fh_input.read(registered_buffer, size, 0, 0)
                print(f"Bytes Read: {bytes_read}")
                bytes_written = fh_output.write(registered_buffer, size, 0, 0)
                print(f"Bytes Written: {bytes_written}")

hipFree(buffer)

with open(input_path, "br") as file_in:
    hash_in = hashlib.sha256()
    chunk = file_in.read(1 * 1024 * 1024)  # 1 MiB
    while len(chunk) != 0:
        hash_in.update(chunk)
        chunk = file_in.read(1 * 1024 * 1024)  # 1 MiB
    print(f"Input File Hash: {hash_in.hexdigest()}")

with open(output_path, "br") as file_out:
    hash_out = hashlib.sha256()
    chunk = file_out.read(1 * 1024 * 1024)  # 1 MiB
    while len(chunk) != 0:
        hash_out.update(chunk)
        chunk = file_out.read(1 * 1024 * 1024)  # 1 MiB
    print(f"Output File Hash: {hash_out.hexdigest()}")
