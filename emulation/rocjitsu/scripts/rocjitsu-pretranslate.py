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
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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


def elf_section(data: bytes, wanted: bytes) -> bytes | None:
    """Return the contents of section `wanted`, or None.

    Done here rather than by shelling out to llvm-objcopy so the script has no
    toolchain dependency: an image build that has ROCm installed does not
    necessarily have the LLVM tools, and needing them would be a surprising
    reason for pre-translation to be unavailable.
    """
    try:
        if len(data) < 64 or not data.startswith(ELF_MAGIC) or data[4] != ELFCLASS64:
            return None
        table_offset = struct.unpack_from("<Q", data, 40)[0]
        entry_size, count, name_index = struct.unpack_from("<HHH", data, 58)
        if entry_size < 64 or table_offset + entry_size > len(data):
            return None
        if count == 0:
            count = struct.unpack_from("<Q", data, table_offset + 32)[0]
        if name_index == 0xFFFF:
            name_index = struct.unpack_from("<I", data, table_offset + 40)[0]
        if not count or name_index >= count:
            return None
        if table_offset + entry_size * count > len(data):
            return None

        names_header = table_offset + name_index * entry_size
        names_offset, names_size = struct.unpack_from("<QQ", data, names_header + 24)
        if names_offset + names_size > len(data):
            return None

        for index in range(count):
            header = table_offset + index * entry_size
            name_offset = struct.unpack_from("<I", data, header)[0]
            start = names_offset + name_offset
            if start >= names_offset + names_size:
                continue
            end = data.find(b"\0", start, names_offset + names_size)
            if end < 0 or data[start:end] != wanted:
                continue
            section_type = struct.unpack_from("<I", data, header + 4)[0]
            offset, size = struct.unpack_from("<QQ", data, header + 24)
            if section_type == SHT_NOBITS or not size or offset + size > len(data):
                return None
            return data[offset : offset + size]
    except (struct.error, ValueError):
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


def unbundle_compressed(bundler: str, blob: bytes, target: str) -> list[bytes]:
    """Return the `target` payloads inside one compressed bundle.

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
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if listed.returncode:
            return []

        payloads = []
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if extracted.returncode == 0 and destination.exists():
                data = destination.read_bytes()
                if data:
                    payloads.append(data)
        return payloads


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

    data: bytes
    origin: dict[str, object]


@dataclass
class Discovery:
    objects: list[Found] = field(default_factory=list)
    compressed_bundles: list[Path] = field(default_factory=list)
    compressed_unpacked: int = 0
    kpack_archives: list[Path] = field(default_factory=list)
    failures: list[dict[str, object]] = field(default_factory=list)


def walk(roots: list[Path]):
    seen: set[tuple[int, int]] = set()
    for root in roots:
        if root.is_file():
            yield root
            continue
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                path = Path(directory) / name
                try:
                    info = path.stat()
                except OSError:
                    continue
                if not os.path.isfile(path):
                    continue
                # Hard links are common in an install tree; extracting the same
                # bytes twice would translate them twice.
                key = (info.st_dev, info.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                yield path


def discover(
    roots: list[Path], target: str, kpack: Kpack | None, bundler: str | None
) -> Discovery:
    found = Discovery()
    # (path, container bytes) for every compressed bundle, resolved after the
    # walk so one missing tool does not abort the scan.
    compressed: list[tuple[Path, bytes]] = []
    for path in walk(roots):
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
                data = header + handle.read()
        except OSError as error:
            found.failures.append(
                {"category": "read", "path": str(path), "error": str(error)}
            )
            continue

        if is_compressed:
            found.compressed_bundles.append(path)
            compressed.append((path, data))
            continue

        if is_amdgpu_elf(data):
            # A standalone code object. Whether it is for this target is the
            # translator's verdict to give, not ours to guess from the name.
            found.objects.append(Found(data, {"container": "elf", "path": str(path)}))
            continue

        fatbin = elf_section(data, b".hip_fatbin")
        if fatbin is None:
            continue
        if fatbin.startswith(COMPRESSED_BUNDLE_MAGIC):
            found.compressed_bundles.append(path)
            compressed.append((path, fatbin))
            continue
        for index, (entry_target, payload) in enumerate(
            parse_clang_offload_bundles(fatbin)
        ):
            if not bundle_target_matches(entry_target, target) or not payload:
                continue
            found.objects.append(
                Found(
                    payload,
                    {
                        "container": "hip-fatbin",
                        "path": str(path),
                        "entry": index,
                        "target": entry_target,
                    },
                )
            )

    if bundler is not None:
        for path, blob in compressed:
            try:
                payloads = unbundle_compressed(bundler, blob, target)
            except OSError as error:
                found.failures.append(
                    {
                        "category": "compressed-bundle",
                        "path": str(path),
                        "error": str(error),
                    }
                )
                continue
            if payloads:
                found.compressed_unpacked += 1
            for index, payload in enumerate(payloads):
                found.objects.append(
                    Found(
                        payload,
                        {
                            "container": "compressed-bundle",
                            "path": str(path),
                            "entry": index,
                        },
                    )
                )

    for archive in found.kpack_archives:
        if kpack is None:
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
                found.objects.append(
                    Found(
                        data,
                        {
                            "container": "kpack",
                            "path": str(archive),
                            "member": member,
                            "architecture": architecture,
                        },
                    )
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
    verbose: bool,
) -> tuple[int, str]:
    command = [tool]
    if store_root:
        command += ["--store-root", store_root]
    if force:
        command.append("--force")
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


def parse_arguments() -> argparse.Namespace:
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
        "--dry-run",
        action="store_true",
        help="report what would be translated and stop",
    )
    parser.add_argument("--verbose", action="store_true", help="echo the tool's output")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
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
            print(
                "warning: clang-offload-bundler not found; compressed bundles will be "
                "reported as skipped. On this ROCm tree those hold the gfx1250 GEMM "
                "libraries, so a cache built without it will be missing them.",
                file=sys.stderr,
            )

    found = discover(roots, args.target, kpack, bundler)

    # Content-address before translating. An install tree holds the same object
    # under several names, and the whole point is not to do the work twice.
    unique: dict[str, Found] = {}
    duplicates = 0
    for item in found.objects:
        digest = hashlib.sha256(item.data).hexdigest()
        if digest in unique:
            duplicates += 1
            continue
        unique[digest] = item

    print(
        f"found {len(unique)} unique objects "
        f"({len(found.objects)} extracted, {duplicates} duplicates), "
        f"{len(found.kpack_archives)} kpack archives, "
        f"{found.compressed_unpacked}/{len(found.compressed_bundles)} compressed "
        "bundles unpacked"
    )
    if args.dry_run:
        return 0 if not found.failures else 1

    tool = locate_tool(args.tool)
    if tool is None:
        print("error: rj_pretranslate not found; pass --tool", file=sys.stderr)
        return 2

    totals = {"written": 0, "held": 0, "skipped": 0, "failed": 0}
    provenance: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="rocjitsu-pretranslate-") as temporary:
        staged: list[Path] = []
        order = sorted(unique.items())
        for index, (digest, item) in enumerate(order, start=1):
            path = Path(temporary) / f"{digest}.co"
            path.write_bytes(item.data)
            staged.append(path)
            provenance.append(
                {"digest": digest, "bytes": len(item.data), **item.origin}
            )

            at_end = index == len(order)
            if len(staged) < args.batch and not at_end:
                continue
            code, output = run_tool(
                tool, args.store_root, staged, args.force, args.verbose
            )
            for name, value in parse_tool_output(output).items():
                totals[name] += value
            if code != 0 and not args.verbose:
                sys.stderr.write(output)
            for path in staged:
                path.unlink(missing_ok=True)
            staged = []
            if args.progress_every and (index % args.progress_every == 0 or at_end):
                print(f"pretranslate: {index}/{len(order)} objects", flush=True)

    summary = {
        "target": args.target,
        "roots": [str(root) for root in roots],
        "store_root": args.store_root,
        "unique_objects": len(unique),
        "duplicate_objects": duplicates,
        "kpack_archives": len(found.kpack_archives),
        "compressed_bundles": len(found.compressed_bundles),
        "compressed_bundles_unpacked": found.compressed_unpacked,
        "discovery_failures": len(found.failures),
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
    return 0 if totals["failed"] == 0 and not found.failures else 1


if __name__ == "__main__":
    sys.exit(main())
