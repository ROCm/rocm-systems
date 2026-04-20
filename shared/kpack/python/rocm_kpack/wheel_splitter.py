"""
Wheel splitter for separating host and device code in Python wheels.

This module provides functionality to split a fat Python wheel (e.g., PyTorch
with embedded HIP device code) into:
- A host wheel: original package with device code stripped and kpack refs injected
- Device wheels: one per GPU architecture, containing .kpack archives

The device wheels overlay onto the host package's directory structure so that
at install time, the .kpack/ directory merges with the host package tree.
"""

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from email.parser import HeaderParser
from pathlib import Path

try:
    import rocm_bootstrap
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "rocm-bootstrap is required for wheel splitting.\n"
        "Install it with: pip install rocm-bootstrap"
    ) from e

from rocm_kpack.artifact_splitter import ExtractedKernel
from rocm_kpack.artifact_utils import (
    extract_architecture_from_target,
    scan_directory,
)
from rocm_kpack.binutils import BundledBinary, Toolchain
from rocm_kpack.compression import ZstdCompressor
from rocm_kpack.database_handlers import DatabaseHandler, get_database_handlers
from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.kpack_transform import (
    NotFatBinaryError,
    is_fat_binary,
    kpack_offload_binary,
)


def _normalize_arch(arch: str) -> str:
    """Normalize architecture name to canonical form (colons for xnack).

    Tensile filenames use hyphens (gfx90a-xnack+) while ELF bundle targets
    use colons (gfx90a:xnack+). Normalize to colons for consistency.
    """
    return arch.replace("-xnack", ":xnack")


def _arch_to_bundle_key(arch: str) -> str:
    """Strip xnack suffix to get the bare architecture (bundle key).

    xnack+/- variants collapse into the same device wheel as the bare arch.
    The individual kpack files preserve the full arch name.

    Examples:
        gfx942:xnack+ → gfx942
        gfx90a:xnack- → gfx90a
        gfx1100       → gfx1100
        gfx11         → gfx11
    """
    return re.sub(r":xnack[+-]$", "", arch)


class WheelSplitError(Exception):
    """Base error for wheel splitting operations."""

    pass


class InvalidWheelError(WheelSplitError):
    """Input is not a valid Python wheel."""

    pass


class NoFatBinariesError(WheelSplitError):
    """No fat binaries found in wheel."""

    pass


@dataclass
class WheelIdentity:
    """Parsed wheel filename and dist-info fields."""

    name: str  # e.g., "torch"
    version: str  # e.g., "2.10.0+rocm7.1"
    python_tag: str  # e.g., "cp313"
    abi_tag: str  # e.g., "cp313"
    platform_tag: str  # e.g., "manylinux_2_28_x86_64"
    dist_info_name: str  # e.g., "torch-2.10.0+rocm7.1.dist-info"


@dataclass
class FatBinaryInfo:
    """A fat binary found in the wheel."""

    absolute_path: Path
    overlay_relative_path: str  # relative to overlay root, e.g. "lib/libtorch_hip.so"


@dataclass
class SplitResult:
    """Result of a wheel split operation."""

    host_wheel_path: Path
    variant_wheel_path: Path | None = None
    device_wheel_paths: dict[str, Path] = field(default_factory=dict)
    architectures_found: list[str] = field(default_factory=list)
    fat_binaries_processed: int = 0


@dataclass
class TransformResult:
    """Result of transforming a single fat binary."""

    overlay_relative_path: str
    original_size: int
    new_size: int
    skipped: bool


@dataclass
class DatabaseFileRef:
    """A database file detected by a handler."""

    absolute_path: Path
    overlay_relative_path: str  # relative to overlay root
    architecture: str  # normalized (colons for xnack)
    handler_name: str


@dataclass
class ExtractedKernelRef:
    """Lightweight reference to an extracted kernel stored on disk.

    Used instead of ExtractedKernel for cross-process communication to avoid
    pickling large kernel_data bytes through IPC pipes. The parent process
    reads kernel data from disk when building kpack archives.
    """

    target_name: str
    kernel_file: Path  # Path to kernel data on disk
    kernel_size: int
    source_binary_relpath: str
    source_prefix: str
    architecture: str


@dataclass
class TransformArgs:
    """Arguments for a single fat binary transformation."""

    binary_path: Path
    temp_output: Path
    search_pattern: str
    kernel_name: str
    overlay_relative_path: str


def parse_wheel_identity(wheel_root: Path) -> WheelIdentity:
    """Parse wheel identity from the dist-info directory.

    Args:
        wheel_root: Root directory of the (exploded) wheel.

    Returns:
        WheelIdentity with parsed fields.

    Raises:
        InvalidWheelError: If the wheel structure is invalid.
    """
    dist_info_dirs = list(wheel_root.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise InvalidWheelError(
            f"Expected exactly one .dist-info directory in {wheel_root}, "
            f"found {len(dist_info_dirs)}"
        )
    dist_info_dir = dist_info_dirs[0]
    dist_info_name = dist_info_dir.name

    # Parse METADATA for authoritative Name and Version
    metadata_file = dist_info_dir / "METADATA"
    if not metadata_file.exists():
        raise InvalidWheelError(f"No METADATA file in {dist_info_dir}")
    metadata_text = metadata_file.read_text(encoding="utf-8")
    parser = HeaderParser()
    metadata = parser.parsestr(metadata_text)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise InvalidWheelError(f"METADATA missing Name or Version in {dist_info_dir}")

    # Parse WHEEL file for tags
    wheel_file = dist_info_dir / "WHEEL"
    if not wheel_file.exists():
        raise InvalidWheelError(f"No WHEEL file in {dist_info_dir}")
    wheel_text = wheel_file.read_text(encoding="utf-8")
    wheel_meta = parser.parsestr(wheel_text)

    # Get the first Tag: line (there may be multiple for multi-platform wheels)
    tag = wheel_meta.get("Tag")
    if not tag:
        raise InvalidWheelError(f"WHEEL file missing Tag in {dist_info_dir}")

    tag_parts = tag.split("-")
    if len(tag_parts) != 3:
        raise InvalidWheelError(
            f"Invalid wheel Tag format '{tag}', expected 'python-abi-platform'"
        )
    python_tag, abi_tag, platform_tag = tag_parts

    return WheelIdentity(
        name=name,
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        dist_info_name=dist_info_name,
    )


def compute_wheel_filename(name: str, version: str, identity: WheelIdentity) -> str:
    """Compute a PEP 427 wheel filename.

    Args:
        name: Distribution name (dashes are normalized to underscores).
        version: Version string.
        identity: WheelIdentity for tags.

    Returns:
        Filename like "amd_torch_device_gfx942-2.10.0+rocm7.1-cp313-cp313-manylinux_2_28_x86_64.whl"
    """
    normalized_name = name.replace("-", "_")
    return (
        f"{normalized_name}-{version}"
        f"-{identity.python_tag}-{identity.abi_tag}-{identity.platform_tag}.whl"
    )


def compute_dist_info_name(name: str, version: str) -> str:
    """Compute dist-info directory name per PEP 427.

    Args:
        name: Distribution name.
        version: Version string.

    Returns:
        e.g. "amd_torch_device_gfx942-2.10.0+rocm7.1.dist-info"
    """
    normalized_name = name.replace("-", "_")
    return f"{normalized_name}-{version}.dist-info"


def generate_record(
    wheel_root: Path, dist_info_dir_name: str, verbose: bool = False
) -> str:
    """Generate RECORD file content with SHA256 hashes for all files.

    Uses streaming hash computation to avoid loading large files into memory.

    Args:
        wheel_root: Root directory of the wheel.
        dist_info_dir_name: Name of the dist-info directory.
        verbose: Print progress for large files.

    Returns:
        RECORD file content as a string.
    """
    lines: list[str] = []
    for file_path in sorted(wheel_root.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(wheel_root).as_posix()
        # RECORD itself has no hash
        if rel_path == f"{dist_info_dir_name}/RECORD":
            lines.append(f"{rel_path},,")
            continue
        size = file_path.stat().st_size
        if verbose and size > 50 * 1024 * 1024:
            print(f"    Hashing: {rel_path} ({size / (1024*1024):.0f} MB)")
        # Stream hash to avoid loading entire file into memory
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                h.update(chunk)
        lines.append(f"{rel_path},sha256={h.hexdigest()},{size}")
    return "\n".join(lines) + "\n"


def _bundle_key_to_dist_name(device_package_prefix: str, bundle_key: str) -> str:
    """Convert a bundle key to a dist package name via rocm_bootstrap.

    Args:
        device_package_prefix: Package name prefix, e.g. "amd-torch-device".
        bundle_key: Bundle key, e.g. "gfx942", "gfx11", "gfx12_0".

    Returns:
        Dist name, e.g. "amd-torch-device-gfx942", "amd-torch-device-gfx12-0".
    """
    bundle = rocm_bootstrap.lookup_bundle(bundle_key)
    return rocm_bootstrap.device_dist_name(device_package_prefix, bundle)


def generate_device_metadata(
    host_identity: WheelIdentity,
    bundle_key: str,
    device_package_prefix: str,
    device_requires_dist: list[str],
) -> str:
    """Generate METADATA content for a device wheel.

    Args:
        host_identity: Identity of the host wheel.
        bundle_key: Bundle key, e.g. "gfx942", "gfx11", "gfx12_0".
        device_package_prefix: Package name prefix, e.g. "amd-torch-device".
        device_requires_dist: Additional Requires-Dist entries (may contain @GFXARCH@).

    Returns:
        METADATA file content.
    """
    device_name = _bundle_key_to_dist_name(device_package_prefix, bundle_key)
    bundle = rocm_bootstrap.lookup_bundle(bundle_key)
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {device_name}",
        f"Version: {host_identity.version}",
        f"Summary: AMD device kernels ({bundle.display_name}) for {host_identity.name}",
        f"Requires-Dist: {host_identity.name} == {host_identity.version}",
    ]
    for dep in device_requires_dist:
        expanded = dep.replace("@GFXARCH@", bundle_key)
        lines.append(f"Requires-Dist: {expanded}")
    return "\n".join(lines) + "\n"


def generate_wheel_file(identity: WheelIdentity) -> str:
    """Generate WHEEL file content for a device wheel.

    Args:
        identity: WheelIdentity for tags.

    Returns:
        WHEEL file content.
    """
    tag = f"{identity.python_tag}-{identity.abi_tag}-{identity.platform_tag}"
    return (
        f"Wheel-Version: 1.0\n"
        f"Generator: rocm-kpack-split-python-wheels\n"
        f"Root-Is-Purelib: false\n"
        f"Tag: {tag}\n"
    )


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree preserving metadata.

    Symlinks are preserved as symlinks (not followed).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src):
        src_dir = Path(dirpath)
        rel = src_dir.relative_to(src)
        dst_dir = dst / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in filenames:
            src_file = src_dir / fname
            dst_file = dst_dir / fname
            if src_file.is_symlink():
                link_target = os.readlink(src_file)
                os.symlink(link_target, dst_file)
            else:
                shutil.copy2(src_file, dst_file)


def zip_wheel(source_dir: Path, output_path: Path) -> None:
    """Zip a directory into a .whl file.

    Args:
        source_dir: Root directory to zip.
        output_path: Output .whl file path.
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir).as_posix()
                zf.write(file_path, arcname)


def _extract_single_binary(
    fat_binary_path: Path,
    overlay_relative_path: str,
    kernel_staging_dir: Path,
) -> list[ExtractedKernelRef]:
    """Extract device kernels from a single fat binary to disk.

    Module-level function for use with ProcessPoolExecutor. Writes kernel
    data to files under kernel_staging_dir instead of returning bytes,
    avoiding large IPC serialization overhead.

    Args:
        fat_binary_path: Absolute path to the fat binary.
        overlay_relative_path: Path relative to the overlay root.
        kernel_staging_dir: Directory to write kernel files into.

    Returns:
        List of ExtractedKernelRef objects with file paths.
    """
    binary = BundledBinary(fat_binary_path)
    code_object_index: dict[str, int] = defaultdict(int)
    refs: list[ExtractedKernelRef] = []

    # Create a subdirectory per binary to avoid filename collisions
    binary_name = Path(overlay_relative_path).name
    binary_staging = kernel_staging_dir / binary_name
    binary_staging.mkdir(parents=True, exist_ok=True)

    with binary.unbundle() as unbundled:
        for target_name, file_name in unbundled.target_list:
            arch = extract_architecture_from_target(target_name)
            if arch:
                src_path = unbundled.dest_dir / file_name
                index = code_object_index[arch]
                code_object_index[arch] += 1
                indexed_relpath = f"{overlay_relative_path}#{index}"

                # Move kernel file to staging (avoids a copy since unbundle
                # temp dir is cleaned up when the context manager exits)
                dest_name = f"{arch}_{index}_{file_name}"
                dest_path = binary_staging / dest_name
                shutil.copy2(src_path, dest_path)
                kernel_size = dest_path.stat().st_size

                refs.append(
                    ExtractedKernelRef(
                        target_name=target_name,
                        kernel_file=dest_path,
                        kernel_size=kernel_size,
                        source_binary_relpath=indexed_relpath,
                        source_prefix="",
                        architecture=arch,
                    )
                )

    return refs


def _transform_single_binary(
    binary_path: Path,
    temp_output: Path,
    search_pattern: str,
    kernel_name: str,
    overlay_relative_path: str,
    verbose: bool,
) -> TransformResult:
    """Transform a single fat binary in the host staging directory.

    This is a module-level function so it can be submitted to a thread pool.
    Each invocation operates on separate files with no shared mutable state.

    Args:
        binary_path: Path to the fat binary in the staging dir.
        temp_output: Path for the temporary transformed output.
        search_pattern: Kpack search pattern with @GFXARCH@ placeholder.
        kernel_name: Kernel name for the kpack reference.
        overlay_relative_path: For reporting.
        verbose: Print transformation details.

    Returns:
        TransformResult with size info.
    """
    original_size = binary_path.stat().st_size
    try:
        kpack_offload_binary(
            input_path=binary_path,
            output_path=temp_output,
            kpack_search_paths=[search_pattern],
            kernel_name=kernel_name,
            verbose=verbose,
        )

        os.replace(temp_output, binary_path)

        if not binary_path.exists():
            raise WheelSplitError(
                f"Binary disappeared after transformation: {binary_path}"
            )

        new_size = binary_path.stat().st_size
        return TransformResult(
            overlay_relative_path=overlay_relative_path,
            original_size=original_size,
            new_size=new_size,
            skipped=False,
        )

    except NotFatBinaryError:
        return TransformResult(
            overlay_relative_path=overlay_relative_path,
            original_size=original_size,
            new_size=original_size,
            skipped=True,
        )
    finally:
        if temp_output.exists():
            temp_output.unlink()


@dataclass
class KpackResult:
    """Result of creating a single kpack archive."""

    arch: str
    kpack_path: Path
    kpack_size: int
    num_kernels: int
    elapsed: float


def _create_single_kpack(
    arch: str,
    ref_list: list[ExtractedKernelRef],
    group_name: str,
    kpack_staging: Path,
    compression: str,
    compression_level: int,
) -> KpackResult:
    """Create a kpack archive for a single architecture.

    Module-level function for use with ProcessPoolExecutor. Each arch is
    fully independent — separate kernel files, separate output archive.
    """
    t0 = time.monotonic()

    compressor = None
    if compression == "zstd":
        compressor = ZstdCompressor(compression_level=compression_level)

    archive = PackedKernelArchive(
        group_name=group_name,
        gfx_arch_family=arch,
        gfx_arches=[arch],
        compressor=compressor,
    )

    for ref in ref_list:
        hsaco_data = ref.kernel_file.read_bytes()
        prepared = archive.prepare_kernel(
            relative_path=ref.source_binary_relpath,
            gfx_arch=ref.architecture,
            hsaco_data=hsaco_data,
        )
        archive.add_kernel(prepared)

    archive.finalize_archive()

    kpack_file = kpack_staging / f"{group_name}_{arch}.kpack"
    archive.write(kpack_file)

    if not kpack_file.exists() or kpack_file.stat().st_size == 0:
        raise WheelSplitError(f"Failed to create kpack archive: {kpack_file}")

    elapsed = time.monotonic() - t0
    return KpackResult(
        arch=arch,
        kpack_path=kpack_file,
        kpack_size=kpack_file.stat().st_size,
        num_kernels=len(ref_list),
        elapsed=elapsed,
    )


class WheelSplitter:
    """Splits a Python wheel into host + per-arch device wheels."""

    def __init__(
        self,
        device_package_prefix: str,
        overlay_root: str,
        toolchain: Toolchain,
        device_requires_dist: list[str] | None = None,
        database_handlers: list[DatabaseHandler] | None = None,
        compression: str = "zstd",
        compression_level: int = 3,
        verbose: bool = False,
        jobs: int = 1,
        generate_variant_wheel: bool = False,
        variant_label: str = "variant",
    ):
        """
        Args:
            device_package_prefix: Package name prefix for device wheels (e.g., "amd-torch-device").
            overlay_root: Directory within the wheel where .kpack/ is placed (e.g., "torch/").
            toolchain: Toolchain instance for binary operations.
            device_requires_dist: Additional Requires-Dist entries for device wheels.
                Supports @GFXARCH@ placeholder.
            database_handlers: Handlers for arch-specific database files (rocblas, hipblaslt, etc.).
            compression: Compression scheme for kpack archives ("none" or "zstd").
            compression_level: Zstd compression level (1-22).
            verbose: Enable verbose output.
            jobs: Number of parallel workers for extraction and transformation (default 1).
            generate_variant_wheel: If True, also generate a PEP 817 variant wheel with
                variant_properties markers. The classic wheel only uses extras.
            variant_label: Label appended to the variant wheel filename per PEP 817
                (e.g., "amd" produces ...-amd.whl). Default: "variant".
        """
        self.device_package_prefix = device_package_prefix
        # Normalize overlay_root: strip trailing slash, ensure no leading slash
        self.overlay_root = overlay_root.strip("/")
        self.toolchain = toolchain
        self.device_requires_dist = device_requires_dist or []
        self.database_handlers = database_handlers or []
        self.compression = compression
        self.compression_level = compression_level
        self.verbose = verbose
        self.jobs = max(1, jobs)
        self.generate_variant_wheel = generate_variant_wheel
        self.variant_label = variant_label

    def split(
        self,
        input_path: Path,
        output_dir: Path,
        output_format: str = "wheel",
    ) -> SplitResult:
        """Split a wheel into host and device components.

        Args:
            input_path: Path to input .whl file or exploded wheel directory.
            output_dir: Output directory for generated wheels.
            output_format: "wheel" for .whl files, "directory" for exploded dirs.

        Returns:
            SplitResult with paths to generated outputs.

        Raises:
            InvalidWheelError: If the input is not a valid wheel.
            NoFatBinariesError: If no fat binaries are found.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 0: Resolve input
        wheel_root, is_temporary = self._resolve_input(input_path)
        try:
            return self._split_impl(wheel_root, output_dir, output_format)
        finally:
            if is_temporary:
                shutil.rmtree(wheel_root, ignore_errors=True)

    def _resolve_input(self, input_path: Path) -> tuple[Path, bool]:
        """Resolve input to a working directory.

        Returns:
            (wheel_root, is_temporary) — is_temporary means we extracted to a temp
            dir that should be cleaned up.
        """
        if input_path.is_dir():
            return input_path, False
        elif input_path.is_file() and input_path.suffix == ".whl":
            # Extract to a temp directory adjacent to the input
            temp_dir = input_path.parent / (input_path.stem + ".tmp_extract")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir()
            if self.verbose:
                print(f"Extracting {input_path} to {temp_dir}")
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(temp_dir)
            return temp_dir, True
        else:
            raise InvalidWheelError(
                f"Input must be a .whl file or directory, got: {input_path}"
            )

    def _split_impl(
        self, wheel_root: Path, output_dir: Path, output_format: str
    ) -> SplitResult:
        """Core split implementation operating on an exploded wheel directory."""
        # Phase 1: Parse identity
        identity = parse_wheel_identity(wheel_root)
        if self.verbose:
            print(f"Wheel: {identity.name} {identity.version}")
            print(
                f"Tag: {identity.python_tag}-{identity.abi_tag}-{identity.platform_tag}"
            )

        overlay_dir = wheel_root / self.overlay_root
        if not overlay_dir.exists():
            raise InvalidWheelError(
                f"Overlay root '{self.overlay_root}' does not exist in wheel"
            )

        # Phase 2: Scan for fat binaries
        if self.verbose:
            print(f"\nPhase 2: Scanning for fat binaries under {self.overlay_root}/")
        fat_binaries = self._scan_fat_binaries(overlay_dir)
        if not fat_binaries:
            raise NoFatBinariesError(
                f"No fat binaries found under {self.overlay_root}/"
            )
        if self.verbose:
            print(f"Found {len(fat_binaries)} fat binaries")

        # Sort largest first so the process pool starts the critical path early
        fat_binaries.sort(key=lambda fb: fb.absolute_path.stat().st_size, reverse=True)

        # Phase 2b: Scan for database files (rocblas, hipblaslt, etc.)
        db_refs_by_arch: dict[str, list[DatabaseFileRef]] = defaultdict(list)
        if self.database_handlers:
            db_refs_by_arch = self._scan_database_files(overlay_dir)

        # Phase 3: Extract kernels from fat binaries
        if self.verbose:
            print(f"\nPhase 3: Extracting device kernels")
        kernel_staging_dir = output_dir / ".kernel_staging"
        kernel_staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            refs_by_arch = self._extract_kernels(fat_binaries, kernel_staging_dir)

            # Merge architecture sets from ELF extraction and database scanning
            all_arches: set[str] = set(refs_by_arch.keys())
            all_arches.update(db_refs_by_arch.keys())
            architectures = sorted(all_arches)

            if self.verbose:
                print(f"Architectures found: {', '.join(architectures)}")
                for arch in architectures:
                    parts = []
                    if arch in refs_by_arch:
                        parts.append(f"{len(refs_by_arch[arch])} kernels")
                    if arch in db_refs_by_arch:
                        parts.append(f"{len(db_refs_by_arch[arch])} db files")
                    print(f"  {arch}: {', '.join(parts)}")

            # Phase 4: Create kpack archives (staged to output_dir temporarily)
            group_name = identity.name.lower().replace("-", "_")
            kpack_by_arch = self._create_kpack_archives(
                refs_by_arch, group_name, output_dir
            )
        finally:
            shutil.rmtree(kernel_staging_dir, ignore_errors=True)

        # Phase 4b: Group by bundle key (collapse xnack variants)
        # kpack_by_arch may have keys like gfx942, gfx942:xnack+, gfx942:xnack-
        # These all collapse into bundle key "gfx942" with multiple kpack files.
        kpacks_by_bundle: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for arch, kpack_path in kpack_by_arch.items():
            bundle_key = _arch_to_bundle_key(arch)
            kpacks_by_bundle[bundle_key].append((arch, kpack_path))

        db_refs_by_bundle: dict[str, list[DatabaseFileRef]] = defaultdict(list)
        for arch, refs in db_refs_by_arch.items():
            bundle_key = _arch_to_bundle_key(arch)
            db_refs_by_bundle[bundle_key].extend(refs)

        bundle_keys = sorted(
            set(kpacks_by_bundle.keys()) | set(db_refs_by_bundle.keys())
        )

        if self.verbose and bundle_keys != architectures:
            print(f"  Bundle keys (after xnack collapse): {', '.join(bundle_keys)}")

        # Phase 5: Create host wheel
        host_filename = compute_wheel_filename(
            identity.name, identity.version, identity
        )
        host_staging = output_dir / (host_filename.removesuffix(".whl") + ".staging")
        if host_staging.exists():
            shutil.rmtree(host_staging)

        if self.verbose:
            print(f"\nPhase 5: Creating host wheel: {host_filename}")
            print(f"  Copying wheel contents to staging...")

        _copy_tree(wheel_root, host_staging)
        if self.verbose:
            print(f"  Copy complete. Transforming fat binaries...")
        self._transform_host_binaries(host_staging, fat_binaries, identity, group_name)

        # Remove database files from host staging
        if db_refs_by_arch:
            self._remove_database_files_from_host(host_staging, db_refs_by_arch)

        # Rewrite host METADATA (classic: extras only, no variant markers)
        self._rewrite_host_metadata(
            host_staging, identity, set(bundle_keys), include_variant_markers=False
        )

        # Regenerate RECORD for host wheel
        if self.verbose:
            print(f"  Generating RECORD (hashing all files)...")
        record_content = generate_record(
            host_staging, identity.dist_info_name, verbose=self.verbose
        )
        record_path = host_staging / identity.dist_info_name / "RECORD"
        record_path.write_text(record_content, encoding="utf-8")

        # Package host wheel
        if output_format == "wheel":
            host_output = output_dir / host_filename
            if self.verbose:
                print(f"  Zipping to {host_output}")
            zip_wheel(host_staging, host_output)
        else:
            host_output = output_dir / host_filename.removesuffix(".whl")
            if host_output.exists():
                shutil.rmtree(host_output)
            host_staging.rename(host_output)

        # Phase 5b: Generate variant wheel (PEP 817) if requested
        variant_output = None
        if self.generate_variant_wheel:
            variant_output = self._create_variant_wheel(
                host_staging if output_format == "wheel" else host_output,
                identity,
                set(bundle_keys),
                output_dir,
                output_format,
            )

        # Clean up host staging (deferred until after variant wheel uses it)
        if output_format == "wheel" and host_staging.exists():
            shutil.rmtree(host_staging)

        # Phase 6: Create device wheels (one per bundle key)
        device_paths: dict[str, Path] = {}
        for bundle_key in bundle_keys:
            device_path = self._create_device_wheel(
                bundle_key,
                identity,
                group_name,
                kpacks_by_bundle.get(bundle_key, []),
                db_refs_by_bundle.get(bundle_key, []),
                output_dir,
                output_format,
            )
            device_paths[bundle_key] = device_path

        return SplitResult(
            host_wheel_path=host_output,
            variant_wheel_path=variant_output,
            device_wheel_paths=device_paths,
            architectures_found=architectures,
            fat_binaries_processed=len(fat_binaries),
        )

    def _rewrite_host_metadata(
        self,
        host_staging: Path,
        identity: WheelIdentity,
        bundle_keys: set[str],
        include_variant_markers: bool = False,
    ) -> None:
        """Rewrite host METADATA to add extras and deps.

        Inserts into the existing METADATA header section:
          - Requires-Dist: rocm-bootstrap
          - Per-target Provides-Extra + Requires-Dist (extras-based)
          - Provides-Extra: all (with all device wheels)
          - If include_variant_markers: also adds variant_properties markers
        """
        metadata_path = host_staging / identity.dist_info_name / "METADATA"
        existing = metadata_path.read_text(encoding="utf-8")

        lines: list[str] = []
        version = identity.version

        # Add rocm-bootstrap dependency
        lines.append("Requires-Dist: rocm-bootstrap")

        # Collect target-level bundle keys (those are the ones we generate
        # extras and variant markers for).
        target_keys: list[str] = []
        for key in sorted(bundle_keys):
            bundle = rocm_bootstrap.lookup_bundle(key)
            if bundle.level == rocm_bootstrap.PackagingLevel.TARGET:
                target_keys.append(key)

        # For each target, compute packaging chain and generate extras
        all_device_dist_names: set[str] = set()
        for target_name in target_keys:
            chain = rocm_bootstrap.packaging_chain(target_name)
            lines.append(f"Provides-Extra: {target_name}")

            for chain_bundle in chain:
                if chain_bundle.key not in bundle_keys:
                    continue
                dist_name = rocm_bootstrap.device_dist_name(
                    self.device_package_prefix, chain_bundle
                )
                all_device_dist_names.add(dist_name)
                # NOTE: This format is parsed verbatim by
                # _add_variant_markers_to_metadata(). If you change it,
                # update the regex there too.
                lines.append(
                    f"Requires-Dist: {dist_name} == {version}; "
                    f'extra == "{target_name}"'
                )
                if include_variant_markers:
                    lines.append(
                        f"Requires-Dist: {dist_name} == {version}; "
                        f'"amd :: gfx_arch :: {target_name}" in variant_properties'
                    )

        # Also add non-target bundle keys (family, sub-family) to the "all" set
        for key in sorted(bundle_keys):
            bundle = rocm_bootstrap.lookup_bundle(key)
            if bundle.level != rocm_bootstrap.PackagingLevel.TARGET:
                dist_name = rocm_bootstrap.device_dist_name(
                    self.device_package_prefix, bundle
                )
                all_device_dist_names.add(dist_name)

        # "all" extra: includes every device wheel
        lines.append("Provides-Extra: all")
        for dist_name in sorted(all_device_dist_names):
            lines.append(f"Requires-Dist: {dist_name} == {version}; " f'extra == "all"')

        # Insert new headers before the body (after the last header line).
        # METADATA uses RFC 822 format: headers, blank line, body.
        # Find the first blank line that separates headers from body.
        header_end = existing.find("\n\n")
        if header_end != -1:
            header_section = existing[: header_end + 1]  # includes trailing \n
            body_section = existing[header_end + 1 :]  # includes leading \n
            new_content = header_section + "\n".join(lines) + "\n" + body_section
        else:
            # No body — just append
            if not existing.endswith("\n"):
                existing += "\n"
            new_content = existing + "\n".join(lines) + "\n"

        metadata_path.write_text(new_content, encoding="utf-8")

        if self.verbose:
            n_extras = len(target_keys) + 1  # +1 for "all"
            n_device = len(all_device_dist_names)
            label = "variant" if include_variant_markers else "classic"
            print(
                f"  Host metadata ({label}): {n_extras} extras, "
                f"{n_device} device wheels referenced"
            )

    def _create_variant_wheel(
        self,
        host_staging_or_dir: Path,
        identity: WheelIdentity,
        bundle_keys: set[str],
        output_dir: Path,
        output_format: str,
    ) -> Path:
        """Create a PEP 817 variant wheel from the classic host staging.

        The variant wheel has the same content as the classic host wheel but with:
        - variant_properties markers in METADATA alongside extras
        - variant.json in dist-info with provider/variant metadata

        For .whl output, host_staging_or_dir is the staging dir (still present).
        For directory output, it's the renamed output dir (we copy from it).
        """
        if self.verbose:
            print(f"\nPhase 5b: Creating variant wheel")

        # Create variant staging by copying classic host
        variant_staging = output_dir / f"{identity.name}-{self.variant_label}.staging"
        if variant_staging.exists():
            shutil.rmtree(variant_staging)
        _copy_tree(host_staging_or_dir, variant_staging)

        # Rewrite METADATA with variant markers (overwrite the classic METADATA)
        # We need to start from the original METADATA, not the already-rewritten one.
        # But the staging already has the classic rewrite. So we re-run the rewrite
        # on the copy — since _rewrite_host_metadata reads existing content and
        # inserts headers, we need to start fresh. Easiest: restore original METADATA
        # from the classic staging (which has extras already), then just add variant
        # markers. Actually, simpler: just re-read the already-rewritten classic
        # METADATA and add variant markers to it.
        self._add_variant_markers_to_metadata(variant_staging, identity, bundle_keys)

        # Add variant.json
        self._write_variant_json(variant_staging, identity, bundle_keys)

        # Regenerate RECORD
        record_content = generate_record(
            variant_staging, identity.dist_info_name, verbose=False
        )
        record_path = variant_staging / identity.dist_info_name / "RECORD"
        record_path.write_text(record_content, encoding="utf-8")

        # Package variant wheel
        # PEP 817 filename: {name}-{ver}-{py}-{abi}-{plat}-{variant_label}.whl
        variant_filename = compute_wheel_filename(
            identity.name, identity.version, identity
        ).replace(".whl", f"-{self.variant_label}.whl")

        if output_format == "wheel":
            variant_output = output_dir / variant_filename
            if self.verbose:
                print(f"  Zipping variant wheel to {variant_output}")
            zip_wheel(variant_staging, variant_output)
            shutil.rmtree(variant_staging)
        else:
            variant_output = output_dir / variant_filename.removesuffix(".whl")
            if variant_output.exists():
                shutil.rmtree(variant_output)
            variant_staging.rename(variant_output)

        if self.verbose:
            print(f"  Variant wheel: {variant_output}")

        return variant_output

    def _add_variant_markers_to_metadata(
        self,
        staging: Path,
        identity: WheelIdentity,
        bundle_keys: set[str],
    ) -> None:
        """Add variant_properties markers to an already-rewritten METADATA.

        The classic host METADATA already has extras-based Requires-Dist lines.
        This adds the corresponding variant_properties marker lines after each
        extras line for target-level bundle keys.
        """
        metadata_path = staging / identity.dist_info_name / "METADATA"
        existing = metadata_path.read_text(encoding="utf-8")
        version = identity.version

        # Collect target keys and their packaging chains
        new_lines: list[str] = []
        for line in existing.splitlines():
            new_lines.append(line)
            # After each extras-based Requires-Dist for a device wheel,
            # add the variant_properties version
            if not line.startswith("Requires-Dist:") or "extra ==" not in line:
                continue
            if 'extra == "all"' in line:
                continue
            # Extract the dist requirement and target name.
            # This regex must match the format generated by _rewrite_host_metadata():
            #   f"Requires-Dist: {dist_name} == {version}; "
            #   f'extra == "{target_name}"'
            # If that format changes, this regex must be updated to match.
            match = re.match(r'(Requires-Dist: .+ == .+); extra == "(\w+)"', line)
            if match:
                req_part = match.group(1)
                target_name = match.group(2)
                new_lines.append(
                    f"{req_part}; "
                    f'"amd :: gfx_arch :: {target_name}" in variant_properties'
                )

        metadata_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        if self.verbose:
            print(f"  Added variant markers to METADATA")

    def _write_variant_json(
        self,
        staging: Path,
        identity: WheelIdentity,
        bundle_keys: set[str],
    ) -> None:
        """Write variant.json to the dist-info directory.

        Contains PEP 817 variant provider and property metadata.
        """
        target_keys = []
        for key in sorted(bundle_keys):
            bundle = rocm_bootstrap.lookup_bundle(key)
            if bundle.level == rocm_bootstrap.PackagingLevel.TARGET:
                target_keys.append(key)

        variants = {}
        for target in target_keys:
            variants[target] = {"amd": {"gfx_arch": [target]}}

        variant_metadata = {
            "providers": {
                "amd": {
                    "install-time": True,
                    "plugin-api": "amd_variant_provider.plugin:AMDVariantPlugin",
                    "requires": ["amd-variant-provider >= 1.0"],
                }
            },
            "default-priorities": {
                "namespace": ["amd"],
                "property": {"amd": {"gfx_arch": target_keys}},
            },
            "variants": variants,
        }

        variant_json_path = staging / identity.dist_info_name / "variant.json"
        variant_json_path.write_text(
            json.dumps(variant_metadata, indent=2) + "\n", encoding="utf-8"
        )

        if self.verbose:
            print(f"  Wrote variant.json ({len(target_keys)} targets)")

    def _scan_fat_binaries(self, overlay_dir: Path) -> list[FatBinaryInfo]:
        """Scan for fat binaries under the overlay root.

        Uses a lightweight header scan (is_fat_binary) that reads only a few
        KB per file — section headers for ELF, PE section table for COFF.
        Non-binary files are rejected by magic number check without reading
        further. Works for both Linux (ELF) and Windows (PE/COFF) wheels.
        """
        fat_binaries: list[FatBinaryInfo] = []
        checked = 0
        for file_path, direntry in scan_directory(overlay_dir):
            if not direntry.is_file(follow_symlinks=False):
                continue
            checked += 1
            if is_fat_binary(file_path):
                rel_path = file_path.relative_to(overlay_dir).as_posix()
                fat_binaries.append(
                    FatBinaryInfo(
                        absolute_path=file_path,
                        overlay_relative_path=rel_path,
                    )
                )
                if self.verbose:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  Fat binary: {rel_path} ({size_mb:.1f} MB)")
        if self.verbose:
            print(f"  Scanned {checked} files, found {len(fat_binaries)} fat binaries")
        return fat_binaries

    def _scan_database_files(
        self, overlay_dir: Path
    ) -> dict[str, list[DatabaseFileRef]]:
        """Scan for arch-specific database files using registered handlers.

        Returns:
            Dict mapping normalized architecture to list of DatabaseFileRef.
        """
        if self.verbose:
            print(f"\nPhase 2b: Scanning for database files")
            print(f"  Handlers: {', '.join(h.name() for h in self.database_handlers)}")

        refs_by_arch: dict[str, list[DatabaseFileRef]] = defaultdict(list)
        counts_by_handler: dict[str, int] = defaultdict(int)
        size_by_handler: dict[str, int] = defaultdict(int)

        for file_path, direntry in scan_directory(overlay_dir):
            if not direntry.is_file(follow_symlinks=False):
                continue
            for handler in self.database_handlers:
                raw_arch = handler.detect(file_path, overlay_dir)
                if raw_arch is not None:
                    arch = _normalize_arch(raw_arch)
                    rel_path = file_path.relative_to(overlay_dir).as_posix()
                    refs_by_arch[arch].append(
                        DatabaseFileRef(
                            absolute_path=file_path,
                            overlay_relative_path=rel_path,
                            architecture=arch,
                            handler_name=handler.name(),
                        )
                    )
                    counts_by_handler[handler.name()] += 1
                    size_by_handler[handler.name()] += direntry.stat().st_size
                    break  # first handler wins

        if self.verbose:
            total_files = sum(counts_by_handler.values())
            total_size = sum(size_by_handler.values())
            print(
                f"  Found {total_files} database files ({total_size / (1024*1024):.0f} MB)"
            )
            for name in sorted(counts_by_handler):
                print(
                    f"    {name}: {counts_by_handler[name]} files "
                    f"({size_by_handler[name] / (1024*1024):.0f} MB)"
                )

        return refs_by_arch

    def _remove_database_files_from_host(
        self,
        host_staging: Path,
        db_refs_by_arch: dict[str, list[DatabaseFileRef]],
    ) -> None:
        """Remove database files from host staging directory."""
        overlay_dir = host_staging / self.overlay_root
        removed = 0
        removed_bytes = 0
        for refs in db_refs_by_arch.values():
            for ref in refs:
                host_file = overlay_dir / ref.overlay_relative_path
                if host_file.exists():
                    removed_bytes += host_file.stat().st_size
                    host_file.unlink()
                    removed += 1

        # Clean up empty directories left behind
        for dirpath, dirnames, filenames in os.walk(overlay_dir, topdown=False):
            if not filenames and not dirnames:
                try:
                    Path(dirpath).rmdir()
                except OSError:
                    pass  # not empty or root

        if self.verbose:
            print(
                f"  Removed {removed} database files from host "
                f"({removed_bytes / (1024*1024):.0f} MB)"
            )

    def _extract_kernels(
        self,
        fat_binaries: list[FatBinaryInfo],
        kernel_staging_dir: Path,
    ) -> dict[str, list[ExtractedKernelRef]]:
        """Extract device kernels from fat binaries to disk.

        When jobs > 1, fat binaries are extracted in parallel using a process
        pool. Kernel data is written to files under kernel_staging_dir to avoid
        pickling large byte buffers through IPC.

        Args:
            fat_binaries: List of fat binaries to extract.
            kernel_staging_dir: Directory for extracted kernel files.

        Returns:
            Dict mapping architecture to list of ExtractedKernelRef.
        """
        refs_by_arch: dict[str, list[ExtractedKernelRef]] = defaultdict(list)
        total = len(fat_binaries)

        phase_start = time.monotonic()

        if self.jobs <= 1:
            for idx, fb in enumerate(fat_binaries, 1):
                if self.verbose:
                    size_mb = fb.absolute_path.stat().st_size / (1024 * 1024)
                    print(
                        f"  [{idx}/{total}] Extracting: "
                        f"{fb.overlay_relative_path} ({size_mb:.1f} MB)"
                    )
                t0 = time.monotonic()
                refs = _extract_single_binary(
                    fb.absolute_path,
                    fb.overlay_relative_path,
                    kernel_staging_dir,
                )
                elapsed = time.monotonic() - t0
                for r in refs:
                    refs_by_arch[r.architecture].append(r)
                if self.verbose:
                    size_mb = fb.absolute_path.stat().st_size / (1024 * 1024)
                    archs = sorted({r.architecture for r in refs})
                    print(
                        f"    -> {len(refs)} kernels, {elapsed:.1f}s, "
                        f"{size_mb:.1f} MB, arches: {', '.join(archs)}"
                    )
        else:
            if self.verbose:
                print(f"  Using {self.jobs} parallel workers")
            with ProcessPoolExecutor(max_workers=self.jobs) as executor:
                future_to_fb: dict[object, tuple[FatBinaryInfo, float]] = {}
                for fb in fat_binaries:
                    future = executor.submit(
                        _extract_single_binary,
                        fb.absolute_path,
                        fb.overlay_relative_path,
                        kernel_staging_dir,
                    )
                    future_to_fb[future] = (fb, time.monotonic())

                completed = 0
                for future in as_completed(future_to_fb):
                    completed += 1
                    fb, submit_time = future_to_fb[future]
                    elapsed = time.monotonic() - submit_time
                    refs = future.result()
                    for r in refs:
                        refs_by_arch[r.architecture].append(r)
                    if self.verbose:
                        size_mb = fb.absolute_path.stat().st_size / (1024 * 1024)
                        archs = sorted({r.architecture for r in refs})
                        print(
                            f"  [{completed}/{total}] {fb.overlay_relative_path} "
                            f"({len(refs)} kernels, {elapsed:.1f}s, "
                            f"{size_mb:.1f} MB, arches: {', '.join(archs)})"
                        )

        if self.verbose:
            total_elapsed = time.monotonic() - phase_start
            total_kernels = sum(len(v) for v in refs_by_arch.values())
            print(
                f"  Extraction complete: {total_kernels} kernels in {total_elapsed:.1f}s"
            )

        return refs_by_arch

    def _create_kpack_archives(
        self,
        refs_by_arch: dict[str, list[ExtractedKernelRef]],
        group_name: str,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Create kpack archives, one per architecture.

        Reads kernel data from files referenced by ExtractedKernelRef objects.
        When jobs > 1, archives are created in parallel (each arch is independent).

        Returns:
            Dict of arch -> kpack file path (in a temp staging area under output_dir).
        """
        kpack_staging = output_dir / ".kpack_staging"
        kpack_staging.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(
                f"\nPhase 4: Creating kpack archives ({len(refs_by_arch)} architectures)"
            )

        phase_start = time.monotonic()
        kpack_paths: dict[str, Path] = {}

        # Sort by kernel count descending so largest archives start first
        sorted_arches = sorted(
            refs_by_arch.items(), key=lambda kv: len(kv[1]), reverse=True
        )

        if self.jobs <= 1:
            for arch, ref_list in sorted_arches:
                result = _create_single_kpack(
                    arch,
                    ref_list,
                    group_name,
                    kpack_staging,
                    self.compression,
                    self.compression_level,
                )
                kpack_paths[result.arch] = result.kpack_path
                if self.verbose:
                    print(
                        f"  {result.arch}: {result.num_kernels} kernels, "
                        f"{result.kpack_size} bytes, {result.elapsed:.1f}s"
                    )
        else:
            if self.verbose:
                print(f"  Using {self.jobs} parallel workers")
            total = len(sorted_arches)
            with ProcessPoolExecutor(max_workers=self.jobs) as executor:
                futures = {}
                for arch, ref_list in sorted_arches:
                    future = executor.submit(
                        _create_single_kpack,
                        arch,
                        ref_list,
                        group_name,
                        kpack_staging,
                        self.compression,
                        self.compression_level,
                    )
                    futures[future] = arch

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    kpack_paths[result.arch] = result.kpack_path
                    if self.verbose:
                        print(
                            f"  [{completed}/{total}] {result.arch}: "
                            f"{result.num_kernels} kernels, "
                            f"{result.kpack_size} bytes, {result.elapsed:.1f}s"
                        )

        if self.verbose:
            total_elapsed = time.monotonic() - phase_start
            total_size = sum(p.stat().st_size for p in kpack_paths.values())
            print(
                f"  Kpack complete: {len(kpack_paths)} archives, "
                f"{total_size / (1024*1024):.0f} MB in {total_elapsed:.1f}s"
            )

        return kpack_paths

    def _compute_kpack_search_pattern(
        self, overlay_relative_path: str, group_name: str
    ) -> str:
        """Compute the @GFXARCH@ search pattern from a binary to its kpack file.

        Args:
            overlay_relative_path: Path relative to overlay root (e.g., "lib/libtorch_hip.so").
            group_name: Group name for kpack files (e.g., "torch").

        Returns:
            Relative search pattern like "../.kpack/torch_@GFXARCH@.kpack".
        """
        parts = Path(overlay_relative_path).parts
        # Depth = number of directory levels (excluding the file itself)
        depth = len(parts) - 1

        kpack_pattern = f".kpack/{group_name}_@GFXARCH@.kpack"
        if depth == 0:
            return kpack_pattern
        up_path = "/".join([".."] * depth)
        return f"{up_path}/{kpack_pattern}"

    def _transform_host_binaries(
        self,
        host_staging: Path,
        fat_binaries: list[FatBinaryInfo],
        identity: WheelIdentity,
        group_name: str,
    ) -> None:
        """Transform fat binaries in the host staging directory.

        When jobs > 1, binary transformations run in parallel. Each binary
        is transformed independently (separate input/output files, no shared
        mutable state).
        """
        overlay_dir = host_staging / self.overlay_root

        # Build per-binary arguments
        all_args: list[TransformArgs] = []
        for fb in fat_binaries:
            binary_path = overlay_dir / fb.overlay_relative_path
            if not binary_path.exists():
                raise WheelSplitError(
                    f"Fat binary not found in host staging: {fb.overlay_relative_path}"
                )
            all_args.append(
                TransformArgs(
                    binary_path=binary_path,
                    temp_output=binary_path.with_suffix(
                        binary_path.suffix + ".kpacked"
                    ),
                    search_pattern=self._compute_kpack_search_pattern(
                        fb.overlay_relative_path, group_name
                    ),
                    kernel_name=fb.overlay_relative_path,
                    overlay_relative_path=fb.overlay_relative_path,
                )
            )

        if self.jobs <= 1:
            # Serial path
            for args in all_args:
                if self.verbose:
                    print(f"  Transforming: {args.overlay_relative_path}")
                    print(f"    Search pattern: {args.search_pattern}")
                result = _transform_single_binary(
                    args.binary_path,
                    args.temp_output,
                    args.search_pattern,
                    args.kernel_name,
                    args.overlay_relative_path,
                    self.verbose,
                )
                self._report_transform_result(result)
        else:
            # Parallel path
            if self.verbose:
                print(f"  Using {self.jobs} parallel workers for transformation")
            with ProcessPoolExecutor(max_workers=self.jobs) as executor:
                futures = {}
                for args in all_args:
                    future = executor.submit(
                        _transform_single_binary,
                        args.binary_path,
                        args.temp_output,
                        args.search_pattern,
                        args.kernel_name,
                        args.overlay_relative_path,
                        False,  # verbose=False in workers
                    )
                    futures[future] = args.overlay_relative_path

                completed = 0
                total = len(futures)
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    if self.verbose:
                        print(f"  [{completed}/{total}] ", end="")
                        self._report_transform_result(result)

    def _report_transform_result(self, result: TransformResult) -> None:
        """Print verbose output for a single binary transformation."""
        if not self.verbose:
            return
        if result.skipped:
            print(f"  Skipping (no device code): {result.overlay_relative_path}")
        else:
            delta = result.new_size - result.original_size
            if delta < 0:
                print(
                    f"  Transformed: {result.overlay_relative_path} "
                    f"({result.new_size} bytes, saved {-delta} bytes)"
                )
            else:
                print(
                    f"  Transformed: {result.overlay_relative_path} "
                    f"({result.new_size} bytes, +{delta} bytes overhead)"
                )

    def _create_device_wheel(
        self,
        bundle_key: str,
        host_identity: WheelIdentity,
        group_name: str,
        kpack_entries: list[tuple[str, Path]],
        db_refs: list[DatabaseFileRef],
        output_dir: Path,
        output_format: str,
    ) -> Path:
        """Create a device wheel for a specific bundle key.

        Args:
            bundle_key: Bundle key, e.g. "gfx942", "gfx11", "gfx12_0".
            kpack_entries: List of (arch, kpack_path) tuples. Multiple entries
                occur when xnack variants collapse into the same bundle key.
            db_refs: Database files to include for this bundle.

        Returns:
            Path to the generated device wheel (file or directory).
        """
        device_name = _bundle_key_to_dist_name(self.device_package_prefix, bundle_key)
        device_dist_info = compute_dist_info_name(device_name, host_identity.version)
        device_filename = compute_wheel_filename(
            device_name, host_identity.version, host_identity
        )

        if self.verbose:
            print(f"\nCreating device wheel: {device_filename}")

        # Create staging directory
        staging = output_dir / (device_filename.removesuffix(".whl") + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()

        # Place kpack file(s) under overlay_root/.kpack/
        # Multiple entries when xnack variants (gfx942, gfx942:xnack+, etc.)
        # collapse into the same bundle key.
        for arch, kpack_path in kpack_entries:
            kpack_dest_dir = staging / self.overlay_root / ".kpack"
            kpack_dest_dir.mkdir(parents=True, exist_ok=True)
            kpack_dest = kpack_dest_dir / f"{group_name}_{arch}.kpack"
            shutil.copy2(kpack_path, kpack_dest)

        # Copy database files preserving their overlay-relative paths
        for ref in db_refs:
            dest = staging / self.overlay_root / ref.overlay_relative_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ref.absolute_path, dest)

        # Create dist-info
        dist_info_dir = staging / device_dist_info
        dist_info_dir.mkdir()

        # METADATA
        metadata_content = generate_device_metadata(
            host_identity,
            bundle_key,
            self.device_package_prefix,
            self.device_requires_dist,
        )
        (dist_info_dir / "METADATA").write_text(metadata_content, encoding="utf-8")

        # WHEEL
        wheel_content = generate_wheel_file(host_identity)
        (dist_info_dir / "WHEEL").write_text(wheel_content, encoding="utf-8")

        # top_level.txt
        (dist_info_dir / "top_level.txt").write_text(
            f"{self.overlay_root}\n", encoding="utf-8"
        )

        # RECORD (must be last)
        record_content = generate_record(staging, device_dist_info)
        (dist_info_dir / "RECORD").write_text(record_content, encoding="utf-8")

        # Package
        if output_format == "wheel":
            device_output = output_dir / device_filename
            zip_wheel(staging, device_output)
            shutil.rmtree(staging)
        else:
            device_output = output_dir / device_filename.removesuffix(".whl")
            if device_output.exists():
                shutil.rmtree(device_output)
            staging.rename(device_output)

        if self.verbose:
            print(f"  Written: {device_output}")

        return device_output
