# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
A quick & rough script for testing the Cython bindings to the
hipFile C library. Reads a given file and copies it to an
output file, and then compares the hashes of the files.

Exits non-zero if the round-trip did not reproduce the input byte for byte,
so CI can use this as a pass/fail hardware test.
"""

import argparse
import hashlib
import os
import pathlib
import sys

from hipfile.hipMalloc import hipFree, hipMalloc

from hipfile import (
    Driver,
    FileHandle,
    Buffer,
    FileHandleType,
    get_version,
)

DEFAULT_INPUT_PATH = "/mnt/ais/ext4/random_2MiB.bin"
DEFAULT_OUTPUT_PATH = "/mnt/ais/ext4/output.bin"

# Max IO in a single transaction is 2GiB - 4KiB as set by the Linux Kernel.
# Larger IOs will be quietly truncated.
MAX_TRANSFER_SIZE = 2 * 1024 * 1024 * 1024 - 4 * 1024

CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB


def parse_args():
    """Parse the input & output paths from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        type=pathlib.Path,
        help=f"File to read through hipFile (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=DEFAULT_OUTPUT_PATH,
        type=pathlib.Path,
        help=f"File to write through hipFile (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def sha256(path):
    """Return the hex SHA-256 digest of *path*, read in CHUNK_SIZE pieces."""
    digest = hashlib.sha256()
    with open(path, "br") as file:
        chunk = file.read(CHUNK_SIZE)
        while len(chunk) != 0:
            digest.update(chunk)
            chunk = file.read(CHUNK_SIZE)
    return digest.hexdigest()


def transfer(input_path, output_path):
    """Copy *input_path* to *output_path* via GPU memory using hipFile."""
    print(f"hipFile Version: {get_version()}")
    print(f"Driver Use Count Before: {Driver.use_count()}")

    size = min(input_path.stat().st_size, MAX_TRANSFER_SIZE)
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


def main():
    """Run the round-trip and report whether the copy is faithful."""
    args = parse_args()

    transfer(args.input, args.output)

    hash_in = sha256(args.input)
    hash_out = sha256(args.output)
    print(f"Input File Hash:  {hash_in}")
    print(f"Output File Hash: {hash_out}")

    if hash_in != hash_out:
        print(
            f"Files differ! Test failed. "
            f"SHA-256 of {args.input} and {args.output} do not match.",
            file=sys.stderr,
        )
        return 1

    print("Hashes match. Test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
