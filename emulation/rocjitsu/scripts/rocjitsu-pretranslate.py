#!/usr/bin/env python3
"""Pre-translate the gfx1250 code objects in a ROCm install.

Translating a large device library takes minutes. The hotswap hook does it at
load time and remembers the result, which makes a long-running process pay once
-- but a container that starts, runs a job and exits pays every time. This script
moves the work to image-build or post-install time, where it is paid once for the
life of the image.

It finds gfx1250 code objects in the containers ROCm actually uses, extracts
them, and hands them to rj_pretranslate, which records each translation where the
hook looks first. Re-running is cheap: entries already held are reported and
skipped.

Extraction is entirely file based. Compressed Clang offload bundles are unpacked
with clang-offload-bundler, which needs no GPU -- an important detail, because
the ROCm install here keeps 224 gfx1250 containers in that form, all of them
GEMM libraries. Without the bundler they are counted and reported as skipped
rather than failing the run, so this still works where it is unavailable.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
EM_AMDGPU = 224
SHT_NOBITS = 8
CLANG_OFFLOAD_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
KPACK_MAGIC = b"KPAK"
COMPRESSED_BUNDLE_MAGIC = b"CCOB"
KPACK_ERROR_KERNEL_NOT_FOUND = 5

DEFAULT_TARGET = "gfx1250"
DEFAULT_ROOT = Path("/opt/rocm")
DEFAULT_KPACK_LIBRARY = "librocm_kpack.so.0"

# Anything this small cannot be a code object, and reading a header from every
# file in an install tree is the bulk of the scan.
MIN_CANDIDATE_BYTES = 64


# ---------------------------------------------------------------------------
# ELF
# ---------------------------------------------------------------------------


def is_amdgpu_elf(data: bytes) -> bool:
    return (
        len(data) >= 20
        and data.startswith(ELF_MAGIC)
        and data[4] == ELFCLASS64
        and data[5] == ELFDATA2LSB
        and struct.unpack_from("<H", data, 18)[0] == EM_AMDGPU
    )


def read_elf_section(path: Path, wanted: bytes) -> bytes | None:
    """Read one ELF section without materializing the host binary.

    Done here rather than by shelling out to llvm-objcopy so the script has no
    toolchain dependency: an image build that has ROCm installed does not
    necessarily have the LLVM tools, and needing them would be a surprising
    reason for pre-translation to be unavailable.
    """
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            elf_header = handle.read(64)
            if (
                len(elf_header) < 64
                or not elf_header.startswith(ELF_MAGIC)
                or elf_header[4] != ELFCLASS64
                or elf_header[5] != ELFDATA2LSB
            ):
                return None
            table_offset = struct.unpack_from("<Q", elf_header, 40)[0]
            entry_size, count, name_index = struct.unpack_from("<HHH", elf_header, 58)
            if entry_size < 64 or table_offset + entry_size > file_size:
                return None

            handle.seek(table_offset)
            first_header = handle.read(entry_size)
            if len(first_header) != entry_size:
                return None
            if count == 0:
                count = struct.unpack_from("<Q", first_header, 32)[0]
            if name_index == 0xFFFF:
                name_index = struct.unpack_from("<I", first_header, 40)[0]
            table_size = entry_size * count
            if (
                not count
                or name_index >= count
                or table_offset + table_size > file_size
            ):
                return None

            handle.seek(table_offset)
            table = handle.read(table_size)
            if len(table) != table_size:
                return None
            names_header = name_index * entry_size
            names_offset, names_size = struct.unpack_from(
                "<QQ", table, names_header + 24
            )
            if names_offset + names_size > file_size:
                return None
            handle.seek(names_offset)
            names = handle.read(names_size)
            if len(names) != names_size:
                return None

            for index in range(count):
                header = index * entry_size
                name_offset = struct.unpack_from("<I", table, header)[0]
                if name_offset >= len(names):
                    continue
                name_end = names.find(b"\0", name_offset)
                if name_end < 0 or names[name_offset:name_end] != wanted:
                    continue
                section_type = struct.unpack_from("<I", table, header + 4)[0]
                offset, size = struct.unpack_from("<QQ", table, header + 24)
                if section_type == SHT_NOBITS or not size or offset + size > file_size:
                    return None
                handle.seek(offset)
                content = handle.read(size)
                return content if len(content) == size else None
    except (OSError, struct.error, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# Clang offload bundles
# ---------------------------------------------------------------------------


def bundle_target_matches(entry_target: str, target: str) -> bool:
    match = re.search(r"--(gfx[0-9a-z]+)(?::[^\s]+)?$", entry_target)
    return match is not None and match.group(1) == target


def parse_clang_offload_bundles(data: bytes):
    """Yield (entry_target, payload) from concatenated Clang offload bundles."""
    search_offset = 0
    while True:
        bundle_offset = data.find(CLANG_OFFLOAD_BUNDLE_MAGIC, search_offset)
        if bundle_offset < 0:
            return
        cursor = bundle_offset + len(CLANG_OFFLOAD_BUNDLE_MAGIC)
        if cursor + 8 > len(data):
            return
        entry_count = struct.unpack_from("<Q", data, cursor)[0]
        cursor += 8
        if entry_count > (len(data) - cursor) // 24:
            return

        bundle_end = cursor
        for _ in range(entry_count):
            offset, size, target_size = struct.unpack_from("<QQQ", data, cursor)
            cursor += 24
            target_end = cursor + target_size
            if target_end > len(data):
                return
            entry_target = data[cursor:target_end].decode("utf-8", errors="replace")
            cursor = target_end
            payload_offset = bundle_offset + offset
            payload_end = payload_offset + size
            if payload_offset < bundle_offset or payload_end > len(data):
                return
            yield entry_target, data[payload_offset:payload_end]
            bundle_end = max(bundle_end, payload_end)
        search_offset = max(cursor, bundle_end)


# ---------------------------------------------------------------------------
# Compressed Clang offload bundles
# ---------------------------------------------------------------------------


def find_bundler(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    for candidate in (
        Path("/opt/rocm/lib/llvm/bin/clang-offload-bundler"),
        Path("/opt/rocm/llvm/bin/clang-offload-bundler"),
    ):
        if candidate.exists():
            return str(candidate)
    from shutil import which

    return which("clang-offload-bundler")


def unbundle_compressed(
    bundler: str, blob: bytes, target: str
) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]]]:
    """Return successful `target` payloads and per-entry failures.

    A compressed bundle cannot be walked with struct.unpack the way a plain one
    can, so this shells out. It does NOT need a GPU -- the container is just
    compressed, not device-resident -- which is what makes pre-translation of
    the Tensile GEMM libraries possible inside a Docker build.
    """
    with tempfile.TemporaryDirectory(prefix="rocjitsu-ccob-") as temporary:
        source = Path(temporary) / "bundle"
        source.write_bytes(blob)
        listed = subprocess.run(
            [bundler, "--list", "--type=o", f"--input={source}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if listed.returncode:
            raise RuntimeError(
                f"bundler list failed ({listed.returncode}): {listed.stderr.strip()}"
            )

        payloads: list[tuple[str, bytes]] = []
        failures: list[tuple[str, str]] = []
        for index, entry_target in enumerate(listed.stdout.split()):
            if not bundle_target_matches(entry_target, target):
                continue
            destination = Path(temporary) / f"payload{index}"
            extracted = subprocess.run(
                [
                    bundler,
                    "--unbundle",
                    "--type=o",
                    f"--targets={entry_target}",
                    f"--input={source}",
                    f"--output={destination}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if extracted.returncode != 0 or not destination.exists():
                failures.append(
                    (
                        entry_target,
                        f"bundler extraction failed ({extracted.returncode}): "
                        f"{extracted.stderr.strip()}",
                    )
                )
                continue
            data = destination.read_bytes()
            if data:
                payloads.append((entry_target, data))
        return payloads, failures


# ---------------------------------------------------------------------------
# KPACK
#
# Not optional: RCCL's gfx1250 device image -- the object that motivated all of
# this -- ships inside a .kpack archive and is reachable no other way.
# ---------------------------------------------------------------------------


class Kpack:
    """ctypes wrapper around the KPACK discovery API."""

    def __init__(self, library: str):
        self.lib = ctypes.CDLL(library)
        void_p = ctypes.c_void_p
        size_t = ctypes.c_size_t
        char_p = ctypes.c_char_p

        self.lib.kpack_open.argtypes = [char_p, ctypes.POINTER(void_p)]
        self.lib.kpack_open.restype = ctypes.c_int
        self.lib.kpack_close.argtypes = [void_p]
        self.lib.kpack_close.restype = None
        for name in ("architecture", "binary"):
            count = getattr(self.lib, f"kpack_get_{name}_count")
            count.argtypes = [void_p, ctypes.POINTER(size_t)]
            count.restype = ctypes.c_int
            value = getattr(self.lib, f"kpack_get_{name}")
            value.argtypes = [void_p, size_t, ctypes.POINTER(char_p)]
            value.restype = ctypes.c_int
        self.lib.kpack_get_kernel.argtypes = [
            void_p,
            char_p,
            char_p,
            ctypes.POINTER(void_p),
            ctypes.POINTER(size_t),
        ]
        self.lib.kpack_get_kernel.restype = ctypes.c_int
        self.lib.kpack_free_kernel.argtypes = [void_p]
        self.lib.kpack_free_kernel.restype = None

    def _strings(self, handle, get_count, get_value) -> list[str]:
        count = ctypes.c_size_t()
        if get_count(handle, ctypes.byref(count)):
            raise RuntimeError("KPACK count query failed")
        values = []
        for index in range(count.value):
            value = ctypes.c_char_p()
            if get_value(handle, index, ctypes.byref(value)) or value.value is None:
                raise RuntimeError(f"KPACK query {index} failed")
            values.append(value.value.decode())
        return values

    def objects(self, archive: Path, target: str):
        """Yield (member, architecture, bytes) for every `target` object."""
        handle = ctypes.c_void_p()
        if self.lib.kpack_open(os.fsencode(archive), ctypes.byref(handle)):
            raise RuntimeError(f"kpack_open failed: {archive}")
        try:
            architectures = self._strings(
                handle,
                self.lib.kpack_get_architecture_count,
                self.lib.kpack_get_architecture,
            )
            binaries = self._strings(
                handle, self.lib.kpack_get_binary_count, self.lib.kpack_get_binary
            )
            for architecture in sorted(architectures):
                if architecture.partition(":")[0] != target:
                    continue
                for binary in sorted(binaries):
                    pointer = ctypes.c_void_p()
                    size = ctypes.c_size_t()
                    result = self.lib.kpack_get_kernel(
                        handle,
                        binary.encode(),
                        architecture.encode(),
                        ctypes.byref(pointer),
                        ctypes.byref(size),
                    )
                    if result == KPACK_ERROR_KERNEL_NOT_FOUND:
                        continue
                    if result:
                        raise RuntimeError(
                            f"kpack_get_kernel failed ({result}): "
                            f"{archive}:{binary}:{architecture}"
                        )
                    try:
                        yield binary, architecture, ctypes.string_at(
                            pointer, size.value
                        )
                    finally:
                        self.lib.kpack_free_kernel(pointer)
        finally:
            self.lib.kpack_close(handle)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class Found:
    """One extracted code object and where it came from."""

    byte_count: int
    origin: dict[str, object]


@dataclass
class Discovery:
    objects: dict[str, Found] = field(default_factory=dict)
    extracted_objects: int = 0
    duplicate_objects: int = 0
    compressed_bundles: list[Path] = field(default_factory=list)
    compressed_unpacked: int = 0
    kpack_archives: list[Path] = field(default_factory=list)
    kpack_skipped: int = 0
    failures: list[dict[str, object]] = field(default_factory=list)

    def add_object(
        self,
        data: bytes,
        origin: dict[str, object],
        on_unique: Callable[[str, bytes, Found], None] | None,
    ) -> None:
        """Deduplicate one object and hand unique bytes to the batch consumer."""
        self.extracted_objects += 1
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.objects:
            self.duplicate_objects += 1
            return
        item = Found(len(data), origin)
        self.objects[digest] = item
        if on_unique is None:
            return
        try:
            on_unique(digest, data, item)
        except OSError as error:
            self.failures.append({"category": "stage", **origin, "error": str(error)})


def walk(roots: list[Path], failures: list[dict[str, object]]):
    seen: set[tuple[int, int]] = set()
    for root in roots:
        try:
            root_info = root.stat()
        except OSError as error:
            failures.append(
                {"category": "stat", "path": str(root), "error": str(error)}
            )
            continue
        if stat.S_ISREG(root_info.st_mode):
            yield root
            continue
        if not stat.S_ISDIR(root_info.st_mode):
            continue

        def record_traversal_error(error: OSError) -> None:
            failures.append(
                {
                    "category": "traversal",
                    "path": str(error.filename or root),
                    "error": str(error),
                }
            )

        for directory, _, names in os.walk(
            root, followlinks=False, onerror=record_traversal_error
        ):
            for name in names:
                path = Path(directory) / name
                try:
                    info = path.stat()
                except OSError as error:
                    failures.append(
                        {"category": "stat", "path": str(path), "error": str(error)}
                    )
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                # Hard links are common in an install tree; extracting the same
                # bytes twice would translate them twice.
                key = (info.st_dev, info.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                yield path


def discover(
    roots: list[Path],
    target: str,
    kpack: Kpack | None,
    bundler: str | None,
    on_unique: Callable[[str, bytes, Found], None] | None = None,
    kpack_intentionally_skipped: bool = False,
) -> Discovery:
    found = Discovery()

    def unpack_compressed(path: Path, blob: bytes) -> None:
        if bundler is None:
            return
        try:
            payloads, entry_failures = unbundle_compressed(bundler, blob, target)
        except (OSError, RuntimeError) as error:
            found.failures.append(
                {
                    "category": "compressed-bundle",
                    "path": str(path),
                    "error": str(error),
                }
            )
            return
        for entry_target, error in entry_failures:
            found.failures.append(
                {
                    "category": "compressed-bundle",
                    "path": str(path),
                    "target": entry_target,
                    "error": error,
                }
            )
        if payloads:
            found.compressed_unpacked += 1
        for index, (entry_target, payload) in enumerate(payloads):
            found.add_object(
                payload,
                {
                    "container": "compressed-bundle",
                    "path": str(path),
                    "entry": index,
                    "target": entry_target,
                },
                on_unique,
            )

    for path in walk(roots, found.failures):
        try:
            with path.open("rb") as handle:
                header = handle.read(64)
                if len(header) < MIN_CANDIDATE_BYTES:
                    continue
                if header.startswith(KPACK_MAGIC):
                    found.kpack_archives.append(path)
                    continue
                is_compressed = header.startswith(COMPRESSED_BUNDLE_MAGIC)
                if not is_compressed and not header.startswith(ELF_MAGIC):
                    continue
                if is_compressed and bundler is None:
                    found.compressed_bundles.append(path)
                    continue
                if is_compressed or is_amdgpu_elf(header):
                    data = header + handle.read()
                else:
                    data = None
        except OSError as error:
            found.failures.append(
                {"category": "read", "path": str(path), "error": str(error)}
            )
            continue

        if is_compressed:
            found.compressed_bundles.append(path)
            unpack_compressed(path, data)
            continue

        if data is not None and is_amdgpu_elf(data):
            # A standalone code object. Whether it is for this target is the
            # translator's verdict to give, not ours to guess from the name.
            found.add_object(data, {"container": "elf", "path": str(path)}, on_unique)
            continue

        fatbin = read_elf_section(path, b".hip_fatbin")
        if fatbin is None:
            continue
        if fatbin.startswith(COMPRESSED_BUNDLE_MAGIC):
            found.compressed_bundles.append(path)
            unpack_compressed(path, fatbin)
            continue
        for index, (entry_target, payload) in enumerate(
            parse_clang_offload_bundles(fatbin)
        ):
            if not bundle_target_matches(entry_target, target) or not payload:
                continue
            found.add_object(
                payload,
                {
                    "container": "hip-fatbin",
                    "path": str(path),
                    "entry": index,
                    "target": entry_target,
                },
                on_unique,
            )

    for archive in found.kpack_archives:
        if kpack is None:
            # An archive nobody meant to open is not a failure. --skip-kpack
            # documents those as accepted skips, so reporting them as errors and
            # exiting nonzero contradicts the option and breaks any build step
            # that checks the exit status. An archive that WAS meant to be
            # processed and could not be is still a failure.
            if kpack_intentionally_skipped:
                found.kpack_skipped += 1
                continue
            found.failures.append(
                {
                    "category": "kpack",
                    "path": str(archive),
                    "error": "no KPACK library; pass --kpack-library or --skip-kpack",
                }
            )
            continue
        try:
            for member, architecture, data in kpack.objects(archive, target):
                found.add_object(
                    data,
                    {
                        "container": "kpack",
                        "path": str(archive),
                        "member": member,
                        "architecture": architecture,
                    },
                    on_unique,
                )
        except (OSError, RuntimeError) as error:
            found.failures.append(
                {"category": "kpack", "path": str(archive), "error": str(error)}
            )
    return found


# ---------------------------------------------------------------------------
# Driving the tool
# ---------------------------------------------------------------------------


def locate_tool(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    # Beside this script's install prefix first, then the path. A ROCm install
    # ships both, and finding the one that belongs to this tree matters: the tool
    # and the runtime hook share entries only when they resolve the same
    # translator.
    beside = Path(__file__).resolve().parent.parent / "bin" / "rj_pretranslate"
    if beside.exists():
        return str(beside)
    from shutil import which

    return which("rj_pretranslate")


def run_tool(
    tool: str,
    store_root: str | None,
    paths: list[Path],
    force: bool,
    fail_on_skipped: bool,
    verbose: bool,
) -> tuple[int, str]:
    command = [tool]
    if store_root:
        command += ["--store-root", store_root]
    if force:
        command.append("--force")
    if fail_on_skipped:
        command.append("--fail-on-skipped")
    command += [str(path) for path in paths]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if verbose:
        sys.stdout.write(result.stdout)
        sys.stdout.flush()
    return result.returncode, result.stdout


def parse_tool_output(text: str) -> dict[str, int]:
    totals = {"written": 0, "held": 0, "skipped": 0, "failed": 0}
    for line in text.splitlines():
        if not line.startswith("summary "):
            continue
        for field_text in line.split()[1:]:
            name, _, value = field_text.partition("=")
            if name in totals and value.isdigit():
                totals[name] += int(value)
    return totals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to scan (default: %s)" % DEFAULT_ROOT,
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="GPU target (default: %(default)s)"
    )
    parser.add_argument(
        "--tool", help="path to rj_pretranslate (default: found beside this script)"
    )
    parser.add_argument(
        "--store-root",
        help="write here instead of the location derived from the translator",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="write provenance.jsonl and summary.json here",
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        help="create translation staging directories below this location",
    )
    parser.add_argument(
        "--kpack-library",
        default=DEFAULT_KPACK_LIBRARY,
        help="KPACK library to load (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-kpack",
        action="store_true",
        help="do not open KPACK archives; they are reported as skipped",
    )
    parser.add_argument(
        "--bundler",
        help="clang-offload-bundler to unpack compressed bundles "
        "(default: found under /opt/rocm or on PATH)",
    )
    parser.add_argument(
        "--skip-compressed",
        action="store_true",
        help="do not unpack compressed bundles; they are reported as skipped",
    )
    parser.add_argument(
        "--batch", type=int, default=8, help="objects per tool invocation (default: 8)"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="objects between progress lines (default: %(default)s; 0 disables)",
    )
    parser.add_argument(
        "--force", action="store_true", help="translate even objects already held"
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        default=True,
        help="accepted for packaging compatibility; the domain owns its key policy",
    )
    parser.add_argument(
        "--fail-on-skipped",
        action="store_true",
        help="exit nonzero if the translator rejects any discovered object",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be translated and stop",
    )
    parser.add_argument("--verbose", action="store_true", help="echo the tool's output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.batch < 1:
        print("error: --batch must be at least 1", file=sys.stderr)
        return 2
    roots = args.paths or [DEFAULT_ROOT]
    missing = [root for root in roots if not root.exists()]
    if missing:
        for root in missing:
            print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    kpack = None
    if not args.skip_kpack:
        try:
            kpack = Kpack(args.kpack_library)
        except OSError as error:
            print(
                f"warning: {args.kpack_library} unavailable ({error}); KPACK archives "
                "will be reported as failures. Pass --skip-kpack to accept that, but "
                "note that large device libraries such as RCCL's live only there.",
                file=sys.stderr,
            )

    bundler = None
    if not args.skip_compressed:
        bundler = find_bundler(args.bundler)
        if bundler is None:
            if args.bundler is not None:
                print(f"error: bundler not found: {args.bundler}", file=sys.stderr)
                return 2
            print(
                "warning: clang-offload-bundler not found; compressed bundles will be "
                "reported as skipped. On this ROCm tree those hold the gfx1250 GEMM "
                "libraries, so a cache built without it will be missing them.",
                file=sys.stderr,
            )

    totals = {"written": 0, "held": 0, "skipped": 0, "failed": 0}
    tool_failures = 0
    provenance: list[dict[str, object]] = []
    tool = None if args.dry_run else locate_tool(args.tool)
    if not args.dry_run and tool is None:
        print("error: rj_pretranslate not found; pass --tool", file=sys.stderr)
        return 2
    if args.temporary_root is not None:
        args.temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rocjitsu-pretranslate-", dir=args.temporary_root
    ) as temporary:
        staged: list[Path] = []
        processed = 0

        def flush_batch() -> None:
            nonlocal processed, staged, tool_failures
            if not staged:
                return
            assert tool is not None
            code, output = run_tool(
                tool,
                args.store_root,
                staged,
                args.force,
                args.fail_on_skipped,
                args.verbose,
            )
            for name, value in parse_tool_output(output).items():
                totals[name] += value
            if code != 0 and not args.verbose:
                sys.stderr.write(output)
            if code != 0:
                tool_failures += 1
            processed += len(staged)
            for staged_path in staged:
                staged_path.unlink(missing_ok=True)
            staged = []
            if args.progress_every and processed % args.progress_every == 0:
                print(f"pretranslate: {processed} objects", flush=True)

        def stage_unique(digest: str, data: bytes, item: Found) -> None:
            path = Path(temporary) / f"{digest}.co"
            path.write_bytes(data)
            staged.append(path)
            provenance.append(
                {"digest": digest, "bytes": item.byte_count, **item.origin}
            )
            if len(staged) == args.batch:
                flush_batch()

        found = discover(
            roots,
            args.target,
            kpack,
            bundler,
            None if args.dry_run else stage_unique,
            args.skip_kpack,
        )
        if not args.dry_run:
            flush_batch()
        unique = found.objects
        print(
            f"found {len(unique)} unique objects "
            f"({found.extracted_objects} extracted, "
            f"{found.duplicate_objects} duplicates), "
            f"{len(found.kpack_archives)} kpack archives, "
            f"{found.compressed_unpacked}/{len(found.compressed_bundles)} compressed "
            "bundles unpacked"
        )
        if args.dry_run:
            return 0 if not found.failures else 1
        if args.progress_every and processed % args.progress_every != 0:
            print(f"pretranslate: {processed}/{len(unique)} objects", flush=True)

    summary = {
        "target": args.target,
        "roots": [str(root) for root in roots],
        "store_root": args.store_root,
        "unique_objects": len(unique),
        "duplicate_objects": found.duplicate_objects,
        "kpack_archives": len(found.kpack_archives),
        "kpack_archives_skipped": found.kpack_skipped,
        "compressed_bundles": len(found.compressed_bundles),
        "compressed_bundles_unpacked": found.compressed_unpacked,
        "discovery_failures": len(found.failures),
        "fail_on_skipped": args.fail_on_skipped,
        "portable": True,
        "tool_failures": tool_failures,
        **totals,
    }
    print(
        "summary written={written} held={held} skipped={skipped} "
        "failed={failed}".format(**totals)
    )

    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        with (args.report_dir / "provenance.jsonl").open("w", encoding="utf-8") as out:
            for record in provenance:
                out.write(json.dumps(record, sort_keys=True) + "\n")
        (args.report_dir / "summary.json").write_text(
            json.dumps(
                {**summary, "failures": found.failures}, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )

    for failure in found.failures:
        print(f"error: {failure}", file=sys.stderr)
    failed = totals["failed"] != 0 or tool_failures != 0 or bool(found.failures)
    if args.fail_on_skipped and totals["skipped"] != 0:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
