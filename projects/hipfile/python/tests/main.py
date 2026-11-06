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

# Size of a single read/write transaction. Keep it a multiple of the 4 KiB
# O_DIRECT block so every transfer but a trailing partial one stays eligible for
# the fast path -- hipFile silently falls back to the buffered backend when the
# offset or length is unaligned, which would defeat the point of this check.
# Must also stay under the kernel's 2 GiB - 4 KiB per-transaction ceiling, above
# which hipFile clamps the request and returns short.
CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB


def is_same_file(input_path, output_path):
    """Return whether both paths name the same file on disk.

    Compares the resolved paths, which normalizes relative spellings, ``..``
    segments and symlinks, and additionally compares inode identity when both
    already exist, which catches hard links and bind mounts that resolve to
    different paths.
    """
    if input_path.resolve() == output_path.resolve():
        return True
    try:
        return input_path.samefile(output_path)
    except OSError:
        # At least one does not exist yet, so they cannot be the same file.
        return False


def parse_args():
    """Parse the input & output paths from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="File to read through hipFile. Must live on an AIS-capable filesystem.",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="File to write through hipFile. Must live on an AIS-capable filesystem.",
    )
    args = parser.parse_args()
    if is_same_file(args.input, args.output):
        # The output is opened O_TRUNC, so a shared path destroys the input
        # before it is ever read and leaves both digests covering the same
        # wreckage -- which the round trip would then report as a pass.
        parser.error(
            f"input and output must name different files, but '{args.input}' "
            f"and '{args.output}' are the same file. Writing would destroy the "
            f"input before it is read."
        )
    return args


def sha256(path):
    """Return the hex SHA-256 digest of *path*, read in CHUNK_SIZE pieces."""
    digest = hashlib.sha256()
    with open(path, "br") as file:
        chunk = file.read(CHUNK_SIZE)
        while len(chunk) != 0:
            digest.update(chunk)
            chunk = file.read(CHUNK_SIZE)
    return digest.hexdigest()


def drain_write(fh_output, buffer, count, file_offset):
    """Write *count* bytes of *buffer* to *fh_output* at *file_offset*.

    ``hipFileWrite`` may transfer fewer bytes than asked for, so keep issuing
    writes -- advancing both the file and buffer offsets -- until the whole
    range is out.
    """
    written = 0
    while written < count:
        put = fh_output.write(buffer, count - written, file_offset + written, written)
        if put == 0:
            raise RuntimeError(
                f"Write to {fh_output.path} stalled at offset "
                f"{file_offset + written}: {written} of {count} bytes written."
            )
        written += put


def copy(fh_input, fh_output, buffer, size):
    """Copy *size* bytes from *fh_input* to *fh_output* through *buffer*.

    Both sides are driven by the byte counts the calls return rather than the
    counts requested: a short read must not cause the untouched tail of the
    buffer -- which holds whatever ``hipMalloc`` handed back, not file data --
    to be written out, and a short write must be finished rather than accepted.
    """
    offset = 0
    while offset < size:
        want = min(CHUNK_SIZE, size - offset)
        got = fh_input.read(buffer, want, offset, 0)
        if got == 0:
            raise RuntimeError(
                f"Unexpected EOF reading {fh_input.path} at offset {offset}: "
                f"{offset} of {size} bytes read."
            )
        drain_write(fh_output, buffer, got, offset)
        offset += got
    return offset


def transfer(input_path, output_path):
    """Copy *input_path* to *output_path* via GPU memory using hipFile."""
    print(f"hipFile Version: {get_version()}")
    print(f"Driver Use Count Before: {Driver.use_count()}")

    size = input_path.stat().st_size
    buffer = hipMalloc(CHUNK_SIZE)
    buffer_ptr = buffer.value  # pylint: disable=C0103  # False Positive
    print(f"Buffer located at: {buffer_ptr} | {hex(buffer_ptr)}")

    with Driver() as hipfile_driver:
        print(f"Driver Use Count After: {hipfile_driver.use_count()}")
        with Buffer.from_ctypes_void_p(buffer, CHUNK_SIZE, 0) as registered_buffer:
            with FileHandle(
                input_path,
                os.O_RDONLY | os.O_DIRECT,
                handle_type=FileHandleType.OPAQUE_FD,
            ) as fh_input:
                with FileHandle(
                    output_path, os.O_RDWR | os.O_DIRECT | os.O_CREAT | os.O_TRUNC
                ) as fh_output:
                    print(f"Transferring {size} bytes in {CHUNK_SIZE} byte chunks...")
                    transferred = copy(fh_input, fh_output, registered_buffer, size)
                    print(f"Bytes Transferred: {transferred}")

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
