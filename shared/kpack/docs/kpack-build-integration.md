# kpack Build Integration Plan

## Overview

This document describes the integration of rocm-kpack into TheRock's build pipeline, focusing on a map/reduce architecture for splitting and recombining device code artifacts.

## Problem Statement

TheRock builds produce artifact directories containing mixed host and device code. These need to be:

1. Split into generic (host-only) and architecture-specific (device code) components
1. Recombined according to packaging topology for distribution
1. Organized so runtime can efficiently locate device code

## Architecture

### Key Design Decision: Manifest-Based Indirection

Instead of embedding full kpack search paths in host binaries, we use a two-level indirection:

1. Host binaries contain a relative path to a manifest file
1. The manifest lists available kpack files and their locations
1. The reduce phase updates the manifest without modifying host binaries

This provides flexibility in final assembly while keeping host code architecture-agnostic.

## Map Phase: Per-Build Artifact Splitting

Each architecture build produces artifacts that need splitting. The map phase processes these deterministically.

### Types of Device Code

The map phase handles two distinct types of device code:

1. **Fat Binaries**: Executables and libraries with embedded `.hip_fatbin` sections containing device code for multiple architectures. These need kpack extraction and transformation.

1. **Kernel Databases**: Pre-compiled kernel collections used by libraries like rocBLAS and hipBLASLt, stored as separate files:

   - `.hsaco` files: Compiled GPU kernel archives (160 KB - 3.7 MB each)
   - `.co` files: Individual kernel objects (40 KB - 590 KB each)
   - `.dat` files: MessagePack metadata indexes for lazy loading

   These files are already architecture-specific (e.g., `TensileLibrary_lazy_gfx1100.co`) and just need to be moved to the appropriate architecture artifact while preserving directory structure.

### Input

- Artifact directory from build (e.g., `/develop/therock-build/artifacts/rocblas_lib_gfx110X/`)
- Contains `artifact_manifest.txt` listing prefix directories
- Each prefix contains mixed host and device code

### Process

1. Read `artifact_manifest.txt` to identify prefix directories
1. For each prefix directory:
   - **For fat binaries** (e.g., in `bin/`, `lib/`):
     - Extract device code from `.hip_fatbin` sections
     - Auto-detect ISAs present in the binary
     - Generate one kpack file per ISA
     - Modify host binaries to reference kpack manifest path
   - **For kernel databases** (e.g., `lib/rocblas/library/`, `lib/hipblaslt/library/`):
     - Identify architecture-specific kernel files (.hsaco, .co, .dat)
     - Move to corresponding architecture artifact based on filename suffix
     - Preserve directory structure for database compatibility
1. Create kpack artifact directories following TheRock conventions
1. Generate kpack manifest (`.kpm` file) for this shard

### Output Structure

```
map-output/
├── rocblas_lib_generic/
│   ├── artifact_manifest.txt  # Preserved from input
│   └── {prefix}/
│       ├── .kpack/
│       │   └── rocblas_lib.kpm # Manifest for this component
│       ├── lib/
│       │   ├── librocblas.so  # Modified with .rocm_kpack_manifest marker
│       │   └── rocblas/
│       │       └── library/    # Kernel database directory (now empty)
│       └── bin/
│           └── rocblas-bench  # Modified with .rocm_kpack_manifest marker
├── rocblas_lib_gfx1100/
│   ├── artifact_manifest.txt
│   └── {prefix}/
│       ├── .kpack/
│       │   └── rocblas_lib_gfx1100.kpack  # From fat binaries
│       └── lib/
│           └── rocblas/
│               └── library/
│                   ├── TensileLibrary_lazy_gfx1100.dat
│                   ├── TensileLibrary_lazy_gfx1100.co
│                   └── *.hsaco  # Other gfx1100 kernel files
├── rocblas_lib_gfx1101/
│   ├── artifact_manifest.txt
│   └── {prefix}/
│       ├── .kpack/
│       │   └── rocblas_lib_gfx1101.kpack
│       └── lib/
│           └── rocblas/
│               └── library/
│                   ├── TensileLibrary_lazy_gfx1101.dat
│                   └── TensileLibrary_lazy_gfx1101.co
└── rocblas_lib_gfx1102/
    └── [similar structure]
```

Note: Kernel database files (.hsaco, .co, .dat) are moved to architecture-specific artifacts while preserving their directory structure. Fat binaries have their device code extracted into .kpack files.

### Manifest Format (.kpm)

Using MessagePack format for efficient runtime parsing:

```python
# Conceptual structure (actual format is binary MessagePack)
{
    "version": 1,
    "component": "miopen_lib",
    "kpack_files": [
        {
            "architecture": "gfx1100",
            "filename": "miopen_lib_gfx1100.kpack",  # Always in same .kpack/ directory
            "checksum": b"...",  # SHA256 as bytes
        },
        {
            "architecture": "gfx1101",
            "filename": "miopen_lib_gfx1101.kpack",
            "checksum": b"...",
        },
    ],
}
```

## Reduce Phase: Package Assembly

The reduce phase combines artifacts from all map phases according to packaging topology.

### Input

- Artifact directories from all map phase outputs
- Configuration file defining packaging topology

### Configuration Schema

Architecture grouping is driven by configuration rather than automatic detection. While consecutive architecture numbers often indicate SKU variants within the same IP generation (e.g., gfx1100, gfx1101, gfx1102), there are exceptions and edge cases that make automatic grouping unreliable. The mapping between build topology and packaging topology is therefore explicitly defined in a configuration file.

```yaml
version: 1.0

# Which build provides primary generic artifacts
primary_generic_source: gfx110X

# Architecture grouping for packages
architecture_groups:
  gfx110X:
    display_name: "ROCm gfx110X"
    architectures:
      - gfx1100
      - gfx1101
      - gfx1102

  gfx115X:
    display_name: "ROCm gfx115X"
    architectures:
      - gfx1150
      - gfx1151

# Component-specific overrides
component_overrides:
  rocblas:
    architecture_groups:
      gfx11-unified:
        architectures: [gfx1100, gfx1101, gfx1102, gfx1150, gfx1151]

# Validation rules
validation:
  error_on_duplicate_device_code: true
  verify_generic_artifacts_match: false
```

### Process

1. Download and flatten generic artifacts from primary source
1. Download and flatten kpack artifact directories according to architecture groups
1. Update/merge kpack manifests (`.kpm` files) to reflect complete distribution
1. Organize into package-ready directory structure

### Output Structure

```
package-staging/
├── gfx110X/
│   ├── {flattened-generic-prefixes}/
│   │   ├── .kpack/
│   │   │   ├── miopen_lib.kpm         # Updated manifest for full distribution
│   │   │   ├── miopen_lib_gfx1100.kpack
│   │   │   ├── miopen_lib_gfx1101.kpack
│   │   │   └── miopen_lib_gfx1102.kpack
│   │   └── bin/
│   │       └── binary1                # Still references miopen_lib.kpm
└── gfx115X/
    ├── {flattened-generic-prefixes}/
    │   ├── .kpack/
    │   │   ├── miopen_lib.kpm         # Different manifest for this package
    │   │   ├── miopen_lib_gfx1150.kpack
    │   │   └── miopen_lib_gfx1151.kpack
    │   └── bin/
    │       └── binary1
```

Note: Each build shard remains independently usable - its `.kpm` file references only the kpack files from that shard. The reduce phase creates comprehensive `.kpm` files for the complete distribution.

## Implementation Components

### New Tools

1. **`split_artifacts.py`** - Map phase tool

   - Input: Artifact directory
   - Output: Split generic + per-ISA kpacks
   - Deterministic, no configuration needed

1. **`recombine_artifacts.py`** - Reduce phase tool

   - Input: Multiple artifact directories + config
   - Output: Package-ready directory structure
   - Configuration-driven grouping

### Modified Components

1. **`ElfOffloadKpacker`** - Add manifest reference injection

   - Instead of `.rocm_kpack_ref` with direct kpack paths
   - Inject `.rocm_kpack_manifest` with path to `.kpm` file
   - Path format: `.kpack/{name}_{component}.kpm`

1. **Runtime (future)** - Manifest-aware kpack loading

   - Read manifest path from binary
   - Parse MessagePack manifest
   - Load kpack files from same `.kpack/` directory
   - Handle architecture fallback logic

## Integration with TheRock

### Build Flow

1. Standard TheRock builds produce artifacts (unchanged)
1. Map phase runs per build, splits artifacts
1. CI uploads split artifacts to S3
1. Package jobs download all artifacts
1. Reduce phase combines according to package type
1. Standard packaging tools create DEB/RPM/wheels

### Artifact Naming Convention

Following TheRock's pattern:

- Generic: `{name}_{component}_generic/` (host-only binaries with manifest references)
- Device: `{name}_{component}_gfx{arch}/` (architecture-specific kpack files)
- Manifest: `{name}_{component}.kpm` (always in `.kpack/` directory)

## Advantages of This Approach

1. **Host Code Stability**: Host binaries don't need modification during reduce phase
1. **Flexible Packaging**: Can reorganize kpacks without touching binaries
1. **Deterministic Map**: No configuration needed for splitting
1. **Configurable Reduce**: Packaging topology defined in version-controlled config
1. **Incremental Updates**: Can update manifest without full rebuild

## Manifest Path Resolution

### Solution: Pre-computed Relative Paths

When the splitting tool processes a binary, it knows:

- The prefix root (e.g., `/opt/rocm`)
- The binary's location within the prefix (e.g., `lib/libhip.so`)
- The manifest location (`.kpack/{name}_{component}.kpm`)

The tool pre-computes the exact relative path from the binary to the manifest and embeds it in the binary's `.rocm_kpack_manifest` section:

```
Binary at: lib/libhip.so
Manifest at: .kpack/hip_lib.kpm
Embedded path: ../.kpack/hip_lib.kpm

Binary at: lib/rocm/bin/hipcc
Manifest at: .kpack/hip_lib.kpm
Embedded path: ../../../.kpack/hip_lib.kpm
```

### Embedded Metadata Format

The `.rocm_kpack_manifest` section contains MessagePack data:

```python
{
    "component": "hip_lib",
    "manifest_path": "../.kpack/hip_lib.kpm",  # Pre-computed relative path
}
```

### Runtime Loading

```c
// Read embedded metadata from binary
metadata = read_rocm_kpack_manifest_section();
manifest_path = resolve_relative_path(binary_location, metadata.manifest_path);

// Process-level accounting to avoid re-parsing
static hashmap* loaded_manifests = NULL;
if (loaded_manifests && hashmap_contains(loaded_manifests, manifest_path)) {
    return hashmap_get(loaded_manifests, manifest_path);
}

// Load and cache the manifest
manifest = load_and_parse_kpm(manifest_path);
hashmap_put(loaded_manifests, manifest_path, manifest);
```

### Future Extensions

- Environment variable `ROCM_KPACK_PATH` for additional search directories (not implemented initially)
- Handling of symlinks and canonical paths (within controlled prefix, shouldn't be an issue)

## Runtime Lookup Mechanism

### Overview

Once a binary locates its `.kpm` manifest file, the runtime needs to load the appropriate kpack files based on the detected GPU architecture.

### Detailed Flow

1. **GPU Detection**:

   ```c
   // Detect current GPU architecture
   gpu_arch = detect_gpu_arch();  // Returns e.g., "gfx1100"
   ```

1. **Manifest Loading**:

   ```c
   // Load and parse MessagePack manifest
   manifest = load_kpm(manifest_path);
   // manifest contains list of available architectures and kpack filenames
   ```

1. **Architecture Selection**:

   ```c
   // Direct match
   kpack_entry = find_in_manifest(manifest, gpu_arch);

   // If no direct match, try fallback chain
   if (!kpack_entry) {
       fallback_arch = get_fallback_arch(gpu_arch);
       kpack_entry = find_in_manifest(manifest, fallback_arch);
   }
   ```

1. **Kpack Loading**:

   ```c
   // Construct full path (kpack files are in same .kpack/ directory as manifest)
   kpack_path = dirname(manifest_path) + "/" + kpack_entry.filename;

   // Open kpack archive
   kpack_handle = kpack_open(kpack_path);

   // Load specific kernel by ordinal
   kernel = kpack_get_kernel(kpack_handle, binary_path, kernel_ordinal);
   ```

### Architecture Fallback Rules

The runtime implements architecture fallback for compatibility:

```
gfx1101 → gfx1100 → gfx11-generic (if available)
gfx1102 → gfx1100 → gfx11-generic
gfx1151 → gfx1150 → gfx11-generic
```

Note: Fallback rules are GPU-family specific and encoded in the runtime, not the manifest.

**TODO**: These fallback rules need elaboration once we encounter libraries built with generic architectures. The exact mapping between specific ISAs and their generic equivalents will need to be determined based on actual build configurations and hardware capabilities.

### Lazy Loading Strategy

1. **On-Demand**: Kernels are loaded only when first requested, not at binary load time
1. **Caching**: Loaded kernels are cached in memory for reuse
1. **Memory Mapping**: Large kpack files can be mmap'd rather than fully loaded
1. **Decompression**: Happens per-kernel, not per-archive, to minimize memory usage

### Integration Points

The runtime integration happens at these points:

1. **HIP Runtime (CLR or comgr - TBD)**:

   - `__hipRegisterFatBinary()`: Detects HIPK magic, triggers manifest lookup
   - `digestFatBinary()`: Lazy loads kernels from kpack as needed

   Note: Whether to integrate into CLR or comgr is to be decided later. The examples here assume CLR but this decision is not critical to the overall design.

1. **Kernel Libraries**:

   - rocBLAS: No change needed - kernel database files are moved as-is
   - hipBLASLt: No change needed - kernel database files are moved as-is
   - Future: These could migrate to kpack format for consistency

### Error Handling

Currently, missing kernels cause segfaults with no error message. The improved error handling will:

```c
// Abort with clear error message instead of segfaulting
kernel = try_load_from_kpack(manifest, gpu_arch);
if (!kernel) {
    fprintf(stderr, "FATAL: ROCm kernel not found\n");
    fprintf(stderr, "  GPU: %s\n", gpu_arch);
    fprintf(stderr, "  Binary: %s\n", binary_path);
    fprintf(stderr, "  Manifest: %s\n", manifest_path);
    fprintf(stderr, "  Available architectures:");
    for (arch in manifest.architectures) {
        fprintf(stderr, " %s", arch);
    }
    fprintf(stderr, "\n");
    abort();  // Clear termination instead of segfault
}
```

Future improvements could include CPU fallback or degraded mode, but initially we'll fail fast with actionable error messages.

## Open Questions

1. **Validation Strategy**: What checks should reduce phase perform?

   - Required: No duplicate device code per architecture
   - Optional: Verify generic artifacts match across builds
   - Optional: Check kernel compatibility versions

1. **Error Recovery**: What happens if manifest or kpack files are missing?

   - Option A: Fall back to CPU implementation if available
   - Option B: Hard error - fail fast
   - Option C: Warning + degraded mode

## Python Wheel Splitting (Strawman)

### Overview

Python wheels containing ROCm components need similar host/device splitting as artifacts, but with wheel-specific packaging constraints. This enables distributing device code separately while maintaining pip-installable packages.

### Input Structure

Unlike artifacts which come from different architecture builds, wheels present as:

- Multiple fat architecture-specific wheels (e.g., `torch_rocm-2.0-gfx110X-linux_x86_64.whl`)
- Each wheel may contain device code for one or more gfx architectures
- Embedded kernel databases (e.g., `aotriton/` directories with per-arch kernels)

### Map Phase: Wheel Splitting

For each input wheel:

1. **Extract wheel contents**:

   ```python
   def split_wheel(input_wheel, output_dir, package_name):
       # Extract to temp directory
       with zipfile.ZipFile(input_wheel) as z:
           z.extractall(temp_dir)
   ```

1. **Process similar to artifacts**:

   - Identify fat binaries and extract device code
   - Locate kernel databases (e.g., `site-packages/aotriton/kernels/gfx1100/`)
   - Generate `.kpm` manifest files (in `_kpack/` directory)

1. **Create two output wheels**:

   **Host wheel** (device code removed):

   ```
   torch_rocm-2.0-linux_x86_64.whl
   ├── torch/
   │   ├── lib/
   │   │   └── libtorch.so  # Modified with .rocm_kpack_manifest
   │   └── _kpack/
   │       └── torch_rocm.kpm
   └── torch_rocm.dist-info/
   ```

   **Device wheel** (kpack module):

   ```
   torch_rocm_kpack_gfx1100-2.0-linux_x86_64.whl
   ├── _torch_rocm_kpack/  # No __init__.py - allows multiple device wheels to overlay
   │   └── _kpack/
   │       └── torch_rocm_gfx1100.kpack  # Fat binary device code only
   ├── aotriton/  # Kernel database at original path (not under _*_kpack)
   │   └── kernels/
   │       └── gfx1100/  # Architecture-specific kernels
   └── torch_rocm_kpack_gfx1100.dist-info/
   ```

   Note:

   - No `__init__.py` in `_torch_rocm_kpack/` allows multiple device wheels to overlay
   - Kernel databases remain at original paths relative to site-packages for proper overlay
   - Only fat binary device code (kpack files) goes in `_*_kpack` directory

1. **Wheel metadata adjustments**:

   ```python
   def update_wheel_metadata(wheel_dir, suffix=None):
       # Update METADATA and RECORD files
       # Add dependencies if device wheel depends on host
       # Adjust wheel name with suffix (e.g., -gfx1100)
       # Inject WheelNext metadata for auto-detection and dependency resolution
   ```

1. **WheelNext Integration**:

   WheelNext metadata will be injected into both host and device wheels to enable:

   - Automatic GPU detection and device wheel installation
   - Dynamic dependency resolution based on detected hardware
   - Coordinated version management between host and device wheels

   ```python
   def inject_wheelnext_metadata(wheel_dir, arch_info):
       # Add WheelNext-specific metadata files
       # Define hardware requirements and compatibility
       # Link host and device wheels together
   ```

### Kernel Database Handling in Wheels

Kernel databases are moved to device wheels at their original paths (not under `_*_kpack`):

```python
def process_wheel_kernel_databases(wheel_contents):
    # Find patterns like:
    # - aotriton/kernels/gfx1100/
    # - rocblas/library/TensileLibrary_*_gfx1100.co
    # Move to corresponding device wheel at SAME relative path
    # This ensures libraries can find their databases after overlay

    # Example: aotriton/kernels/gfx1100/ in device wheel
    #          NOT _torch_rocm_kpack/aotriton/kernels/gfx1100/
```

### Reduce Phase: Wheel Recombination

```python
def recombine_wheels(config, host_wheel, device_wheels, output_dir):
    # Similar to artifact recombination but:
    # 1. Unpack all wheels
    # 2. Merge according to topology config
    # 3. Update .kpm manifests
    # 4. Repackage as new wheel with all selected architectures
```

### Key Differences from Artifact Flow

1. **Packaging constraints**: Must maintain valid wheel structure and metadata
1. **Module organization**: Device code goes in `_{package}_kpack/` Python module
1. **Dependency management**: Device wheels may declare dependency on host wheel
1. **Naming conventions**: Wheel names must follow PEP standards with architecture suffixes

### Tool Integration

The wheel splitter will reuse most components from artifact splitting:

- Same `BundledBinary` class for kernel extraction
- Same kpack creation logic
- Same manifest generation
- Different packaging step (wheel vs artifact directory)

### Example Usage

```bash
# Map phase - split a fat wheel
python -m rocm_kpack.tools.split_wheel \
    --input torch_rocm-2.0-gfx110X-linux_x86_64.whl \
    --output-dir ./split-wheels/

# Reduce phase - recombine for distribution
python -m rocm_kpack.tools.recombine_wheels \
    --config wheel-topology.yaml \
    --host-wheel torch_rocm-2.0-linux_x86_64.whl \
    --device-wheels split-wheels/*_kpack_*.whl \
    --output-dir ./final-wheels/
```

### Open Questions for Wheel Splitting

1. **Wheel naming**: Exact suffix format for device wheels (e.g., `_kpack_gfx1100` vs `_gfx1100`)
1. **Dependency declaration**: Should device wheels depend on exact host wheel version?
1. **Module structure**: Should device code be importable or just data files?
1. **Metadata preservation**: How much of original wheel metadata to preserve?

Note: The exact ergonomics of wheel naming and dependencies will need refinement during implementation, but the core split/recombine pattern remains the same as artifacts.

## Implementation Plan

### Tool 1: split_artifacts.py

#### Purpose

Map phase tool to split artifact directories into generic (host-only) and architecture-specific components.

#### Command Line Interface

```bash
python -m rocm_kpack.tools.split_artifacts \
    --input-dir /path/to/artifact_dir \
    --output-dir /path/to/output \
    --component-name hip_lib \
    --verbose
```

#### Implementation Steps

1. **Artifact Processing**:

   ```python
   class ArtifactSplitter:
       def split(self, input_dir, output_dir, component_name):
           # 1. Read artifact_manifest.txt
           prefixes = self.read_manifest(input_dir)

           # 2. For each prefix, process files
           for prefix in prefixes:
               self.process_prefix(prefix, component_name)
   ```

1. **Binary Classification**:

   ```python
   def classify_files(self, prefix_dir):
       fat_binaries = []
       kernel_databases = {}

       for file_path in walk_directory(prefix_dir):
           if is_elf_binary(file_path):
               if has_hip_fatbin_section(file_path):
                   fat_binaries.append(file_path)
           elif is_kernel_database_file(file_path):
               # Extract architecture from filename
               arch = extract_arch_from_filename(file_path)
               kernel_databases.setdefault(arch, []).append(file_path)

       return fat_binaries, kernel_databases
   ```

1. **Relative Path Computation**:

   ```python
   def compute_relative_path(binary_path, component_name):
       """Compute relative path from binary to its .kpm manifest."""
       # binary_path is relative to prefix root
       # e.g., "lib/libhip.so" or "lib/rocm/bin/hipcc"
       depth = binary_path.count("/")

       # Build path: go up 'depth' levels, then into .kpack/
       up_path = "/".join([".."] * depth)
       manifest_name = f"{component_name}.kpm"
       return f"{up_path}/.kpack/{manifest_name}"
   ```

1. **Fat Binary Processing**:

   ```python
   def process_fat_binaries(self, binaries, component_name):
       kernels_by_arch = {}

       for binary_path in binaries:
           binary = BundledBinary(binary_path)

           # Extract kernels and group by architecture
           for kernel in binary.unbundle():
               arch = kernel.architecture
               kernels_by_arch.setdefault(arch, []).append(kernel)

           # Compute relative path from binary to manifest
           relative_path = compute_relative_path(binary_path, component_name)
           # Inject pre-computed path into binary
           self.inject_manifest_reference(binary_path, component_name, relative_path)

       # Create kpack files per architecture
       for arch, kernels in kernels_by_arch.items():
           self.create_kpack(arch, kernels, component_name)
   ```

1. **Kernel Database Handling**:

   ```python
   def move_kernel_databases(self, kernel_files_by_arch, output_dir):
       for arch, files in kernel_files_by_arch.items():
           # Create architecture artifact directory
           arch_artifact = f"{component_name}_{arch}"
           arch_dir = output_dir / arch_artifact

           # Preserve directory structure
           for file_path in files:
               relative_path = compute_relative_path(file_path)
               dest = arch_dir / relative_path
               dest.parent.mkdir(parents=True, exist_ok=True)
               shutil.move(file_path, dest)

           # Create artifact_manifest.txt
           self.write_artifact_manifest(arch_dir)
   ```

1. **Manifest Generation**:

   ```python
   def generate_kpm(self, component_name, available_arches):
       manifest = {"version": 1, "component": component_name, "kpack_files": []}

       for arch in available_arches:
           manifest["kpack_files"].append(
               {
                   "architecture": arch,
                   "filename": f"{component_name}_{arch}.kpack",
                   "checksum": compute_sha256(...),
               }
           )

       # Write as MessagePack
       return msgpack.packb(manifest)
   ```

#### Key Classes to Reuse/Extend from rocm-kpack

- `BundledBinary` - for kernel extraction
- `PackedKernelArchive` - for kpack creation
- `ElfOffloadKpacker` - for binary modification

#### Simple Reimplementations Needed

- Artifact manifest reading (just parse `artifact_manifest.txt`)
- Directory traversal (basic os.walk or pathlib)
- Artifact flattening (copy with prefix merging)

### Tool 2: recombine_artifacts.py

#### Purpose

Reduce phase tool to recombine split artifacts according to packaging topology.

#### Command Line Interface

```bash
python -m rocm_kpack.tools.recombine_artifacts \
    --config packaging-topology.yaml \
    --input-mapping gfx110X:/path/to/gfx110X_artifacts \
                    gfx120X:/path/to/gfx120X_artifacts \
    --output-dir /path/to/package-staging \
    --primary-generic gfx110X
```

#### Implementation Steps

1. **Configuration Loading**:

   ```python
   class PackagingConfig:
       def __init__(self, config_path):
           self.config = yaml.safe_load(open(config_path))
           self.validate_schema()

       def get_architecture_groups(self):
           return self.config["architecture_groups"]

       def get_primary_source(self):
           return self.config.get("primary_generic_source")
   ```

1. **Artifact Collection**:

   ```python
   class ArtifactCollector:
       def collect(self, input_mappings):
           artifacts = {}

           for build_name, path in input_mappings.items():
               # Find all artifact directories
               artifacts[build_name] = {
                   "generic": find_generic_artifacts(path),
                   "arch_specific": find_arch_artifacts(path),
               }

           return artifacts
   ```

1. **Flattening and Merging**:

   ```python
   def create_package_layout(self, arch_group, artifacts, config):
       output = self.output_dir / arch_group

       # 1. Copy generic artifacts from primary source
       primary = config.get_primary_source()
       for generic_artifact in artifacts[primary]["generic"]:
           flatten_artifact(generic_artifact, output)

       # 2. Collect architecture-specific artifacts
       for arch in config.get_architectures(arch_group):
           for build in artifacts:
               arch_artifact = find_artifact_for_arch(
                   artifacts[build]["arch_specific"], arch
               )
               if arch_artifact:
                   flatten_artifact(arch_artifact, output)

       # 3. Update/merge manifests
       self.update_manifests(output, arch_group)
   ```

1. **Manifest Merging**:

   ```python
   def update_manifests(self, package_dir, arch_group):
       kpack_dir = package_dir / ".kpack"

       # Find all .kpm files
       kpm_files = kpack_dir.glob("*.kpm")

       for kpm_path in kpm_files:
           # Load existing manifest
           manifest = msgpack.unpackb(kpm_path.read_bytes())

           # Update with all available kpack files in directory
           available_kpacks = kpack_dir.glob("*.kpack")
           manifest["kpack_files"] = []

           for kpack_path in available_kpacks:
               # Extract architecture from filename
               arch = extract_arch_from_kpack_name(kpack_path.name)
               manifest["kpack_files"].append(
                   {
                       "architecture": arch,
                       "filename": kpack_path.name,
                       "checksum": compute_sha256(kpack_path),
                   }
               )

           # Write updated manifest
           kpm_path.write_bytes(msgpack.packb(manifest))
   ```

1. **Validation**:

   ```python
   def validate_no_duplicates(self, package_dir):
       seen_kernels = {}

       for kpack_path in find_kpack_files(package_dir):
           archive = PackedKernelArchive.read(kpack_path)

           for binary, arch, kernel_id in archive.list_kernels():
               key = (binary, arch, kernel_id)
               if key in seen_kernels:
                   raise ValueError(
                       f"Duplicate kernel: {key} in {kpack_path} "
                       f"and {seen_kernels[key]}"
                   )
               seen_kernels[key] = kpack_path
   ```

#### Simple Artifact Operations

```python
def read_artifact_manifest(artifact_dir):
    """Read artifact_manifest.txt and return list of prefixes."""
    manifest_path = artifact_dir / "artifact_manifest.txt"
    return manifest_path.read_text().strip().splitlines()


def flatten_artifact(artifact_dir, output_dir):
    """Flatten artifact by copying all prefixes to output."""
    prefixes = read_artifact_manifest(artifact_dir)
    for prefix in prefixes:
        src = artifact_dir / prefix
        # Copy preserving structure
        shutil.copytree(src, output_dir, dirs_exist_ok=True)
```

### Testing Strategy

1. **Unit Tests**:

   - Test ISA detection from kernel files
   - Test manifest generation and parsing
   - Test kernel database file classification

1. **Integration Tests**:

   - Create sample artifact directories
   - Run split → recombine pipeline
   - Verify output structure matches expectation
   - Verify runtime can load kernels

1. **End-to-End Tests**:

   - Use real ROCm build artifacts
   - Split actual gfx110X build
   - Recombine with configuration
   - Test with HIP runtime

## Next Steps

1. Prototype split_artifacts.py with minimal functionality
1. Test with single artifact directory
1. Implement recombine_artifacts.py
1. Create integration test suite
1. Test with full TheRock build
