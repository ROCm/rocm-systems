# Tutorial: Splitting and Recombining ROCm Artifacts

## Overview

The ROCm build system produces **fat binary** artifacts containing GPU kernels for multiple architectures embedded in shared libraries. For distribution, these need to be:

1. **Split** - Separate GPU kernels from host code, organize by architecture
1. **Recombined** - Merge splits from multiple build shards into package groups

This enables:

- Smaller package sizes (users download only their GPU architecture)
- Faster builds (parallel builds for different GPU families)
- Better artifact reuse across builds

## Prerequisites

### Tools Required

- Python 3.9+ with rocm-kpack installed
- `clang-offload-bundler` from ROCm LLVM toolchain
- Sufficient disk space (~3x the input artifact size)

### Environment Setup

```bash
# Set environment variables for easy editing
export THEROCK_DIR=/path/to/therock
export RUN_ID=your-github-run-id

# Activate Python virtual environment
source /path/to/your/venv/bin/activate

# Locate clang-offload-bundler (usually in rocm installation)
BUNDLER=$(find ~/rocm -name "clang-offload-bundler" 2>/dev/null | head -1)
echo "Using bundler: $BUNDLER"
```

## Step 1: Download Artifacts

Download build artifacts from CI using `fetch_artifacts.py`:

```bash
# Example: Download artifacts for three architecture groups
python $THEROCK_DIR/build_tools/fetch_artifacts.py \
  --run-id $RUN_ID \
  --artifact-group gfx110X-dgpu \
  --output-dir /develop/artifacts/gfx110X-build

python $THEROCK_DIR/build_tools/fetch_artifacts.py \
  --run-id $RUN_ID \
  --artifact-group gfx1151 \
  --output-dir /develop/artifacts/gfx1151-build

python $THEROCK_DIR/build_tools/fetch_artifacts.py \
  --run-id $RUN_ID \
  --artifact-group gfx120X-all \
  --output-dir /develop/artifacts/gfx120X-build
```

**Note**: The artifact group names (gfx110X-dgpu, gfx1151, gfx120X-all) are just build job identifiers. The actual GPU architectures inside the binaries may differ and will be discovered during splitting.

## Step 2: Discovery - Understand Actual Architectures

Before splitting all artifacts, split a representative component to discover what GPU architectures are actually present:

```bash
# Create discovery output directory
mkdir -p /develop/tmp/discovery/gfx110X-build

# Split blas_lib from first shard to discover architectures
python python/rocm_kpack/tools/split_artifacts.py \
  --artifact-dir /develop/artifacts/gfx110X-build/blas_lib_gfx110X-dgpu \
  --output-dir /develop/tmp/discovery/gfx110X-build \
  --artifact-prefix blas_lib \
  --tmp-dir /develop/tmp \
  --split-databases rocblas hipblaslt \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

# Check what architectures were discovered
ls /develop/tmp/discovery/gfx110X-build/
# Output: blas_lib_generic blas_lib_gfx1100 blas_lib_gfx1101 blas_lib_gfx1102
```

Repeat for other shards:

```bash
# Discover architectures in gfx1151-build
mkdir -p /develop/tmp/discovery/gfx1151-build
python python/rocm_kpack/tools/split_artifacts.py \
  --artifact-dir /develop/artifacts/gfx1151-build/blas_lib_gfx1151 \
  --output-dir /develop/tmp/discovery/gfx1151-build \
  --artifact-prefix blas_lib \
  --tmp-dir /develop/tmp \
  --split-databases rocblas hipblaslt \
  --clang-offload-bundler "$BUNDLER"

ls /develop/tmp/discovery/gfx1151-build/
# Output: blas_lib_generic blas_lib_gfx1151

# Discover architectures in gfx120X-build
mkdir -p /develop/tmp/discovery/gfx120X-build
python python/rocm_kpack/tools/split_artifacts.py \
  --artifact-dir /develop/artifacts/gfx120X-build/blas_lib_gfx120X-all \
  --output-dir /develop/tmp/discovery/gfx120X-build \
  --artifact-prefix blas_lib \
  --tmp-dir /develop/tmp \
  --split-databases rocblas hipblaslt \
  --clang-offload-bundler "$BUNDLER"

ls /develop/tmp/discovery/gfx120X-build/
# Output: blas_lib_generic blas_lib_gfx1200 blas_lib_gfx1201
```

**Discovered Architectures:**

- **gfx110X-build**: gfx1100, gfx1101, gfx1102
- **gfx1151-build**: gfx1151
- **gfx120X-build**: gfx1200, gfx1201

## Step 3: Create Packaging Configuration

Based on discovered architectures, create a JSON configuration defining package groups:

```bash
cat > /develop/tmp/packaging-config.json << 'EOF'
{
  "primary_shard": "gfx110X-build",
  "architecture_groups": {
    "gfx110X": {
      "display_name": "ROCm gfx110X (RDNA3)",
      "architectures": [
        "gfx1100",
        "gfx1101",
        "gfx1102"
      ]
    },
    "gfx115X": {
      "display_name": "ROCm gfx115X (RDNA3.5)",
      "architectures": [
        "gfx1151"
      ]
    },
    "gfx120X": {
      "display_name": "ROCm gfx120X (RDNA4)",
      "architectures": [
        "gfx1200",
        "gfx1201"
      ]
    }
  },
  "validation": {
    "error_on_duplicate_device_code": true,
    "verify_generic_artifacts_match": false,
    "error_on_missing_architecture": false
  }
}
EOF
```

**Configuration Fields:**

- `primary_shard`: Which shard to use for generic (host-code) artifacts
- `architecture_groups`: Package groups and which GPU architectures go in each
- `validation`: Rules for handling edge cases during recombination

## Step 4: Split All Artifacts

Use batch mode to split all artifacts from each shard:

```bash
# Split all artifacts from gfx110X-build shard
python python/rocm_kpack/tools/split_artifacts.py \
  --batch-artifact-parent-dir /develop/artifacts/gfx110X-build \
  --output-dir /develop/tmp/split-artifacts/gfx110X-build \
  --split-databases rocblas hipblaslt \
  --tmp-dir /develop/tmp \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

# Split all artifacts from gfx1151-build shard
python python/rocm_kpack/tools/split_artifacts.py \
  --batch-artifact-parent-dir /develop/artifacts/gfx1151-build \
  --output-dir /develop/tmp/split-artifacts/gfx1151-build \
  --split-databases rocblas hipblaslt \
  --tmp-dir /develop/tmp \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

# Split all artifacts from gfx120X-build shard
python python/rocm_kpack/tools/split_artifacts.py \
  --batch-artifact-parent-dir /develop/artifacts/gfx120X-build \
  --output-dir /develop/tmp/split-artifacts/gfx120X-build \
  --split-databases rocblas hipblaslt \
  --tmp-dir /develop/tmp \
  --clang-offload-bundler "$BUNDLER" \
  --verbose
```

**What Batch Mode Does:**

For each arch-specific artifact in the shard directory:

1. Auto-detects artifact prefix from directory name (`blas_lib_gfx110X-dgpu` → `blas_lib`)
1. Skips generic artifacts (artifacts ending in `_generic`)
1. Scans for fat binaries (shared libraries with embedded GPU code)
1. Unbundles GPU kernels using `clang-offload-bundler`
1. Creates `.kpack` archive files per architecture
1. Strips GPU code from shared libraries (PROGBITS → NOBITS)
1. Adds `.rocm_kpack_ref` marker pointing to `.kpack` files
1. For databases (rocBLAS, hipBLASLt): separates by architecture
1. Creates **generic artifact** (host code + markers, NO .kpack files)
1. Creates **arch-specific artifacts** (ONLY GPU kernels + databases, NO host libraries)

**Output Structure:**

```
/develop/tmp/split-artifacts/
├── gfx110X-build/
│   ├── blas_lib_generic/           # Host code ONLY, no GPU kernels
│   ├── blas_lib_gfx1100/           # gfx1100 GPU kernels + databases ONLY
│   ├── blas_lib_gfx1101/
│   ├── blas_lib_gfx1102/
│   ├── fft_lib_generic/
│   ├── fft_lib_gfx1100/
│   └── ...
├── gfx1151-build/
│   ├── blas_lib_generic/
│   ├── blas_lib_gfx1151/
│   └── ...
└── gfx120X-build/
    ├── blas_lib_generic/
    ├── blas_lib_gfx1200/
    ├── blas_lib_gfx1201/
    └── ...
```

## Step 5: Verify Split Artifacts

Verify that splits meet all invariants:

```bash
# Verify each shard
python python/rocm_kpack/tools/verify_artifacts.py \
  --artifacts-dir /develop/tmp/split-artifacts/gfx110X-build \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

python python/rocm_kpack/tools/verify_artifacts.py \
  --artifacts-dir /develop/tmp/split-artifacts/gfx1151-build \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

python python/rocm_kpack/tools/verify_artifacts.py \
  --artifacts-dir /develop/tmp/split-artifacts/gfx120X-build \
  --clang-offload-bundler "$BUNDLER" \
  --verbose
```

**Verification Checks:**

1. ✓ All artifact manifests present and valid
1. ✓ Fat binaries converted (PROGBITS → NOBITS)
1. ✓ Architecture separation (no cross-contamination)
1. ✓ Kpack archives valid (KPAK magic, MessagePack TOC)

## Step 6: Recombine into Package Groups

Combine split artifacts from all shards according to the packaging configuration:

```bash
python python/rocm_kpack/tools/recombine_artifacts.py \
  --input-shards-dir /develop/tmp/split-artifacts \
  --config /develop/tmp/packaging-config.json \
  --output-dir /develop/tmp/recombined-artifacts \
  --verbose
```

**What Recombine Does:**

For each component and each architecture group:

1. Creates **generic artifact** (once per component):

   - Copies generic artifact from primary shard
   - Contains host shared libraries with `.rocm_kpack_ref` markers
   - NO .kpack files or architecture-specific databases

1. Creates **arch-specific artifacts** (one per architecture group):

   - Collects arch-specific artifacts from all shards
   - Copies `.kpack` files for all architectures in the group
   - Copies architecture-specific database files
   - Merges `.kpm` manifest files
   - NO host shared libraries

**Output:**

```
/develop/tmp/recombined-artifacts/
├── blas_lib_generic/      # Host code only (shared across all architectures)
├── blas_lib_gfx110X/      # gfx1100, gfx1101, gfx1102 kernels + databases
├── blas_lib_gfx115X/      # gfx1151 kernels + databases
├── blas_lib_gfx120X/      # gfx1200, gfx1201 kernels + databases
├── fft_lib_generic/
├── fft_lib_gfx110X/
├── fft_lib_gfx115X/
├── fft_lib_gfx120X/
└── ...
```

Each final package deployment requires:

- ONE generic artifact (e.g., `blas_lib_generic`)
- ONE arch-specific artifact for the target GPU (e.g., `blas_lib_gfx110X`)

## Step 7: Inspect Results

Examine a recombined artifact to verify correct separation:

```bash
# Check generic artifact structure
tree -L 4 /develop/tmp/recombined-artifacts/blas_lib_generic/

# Expected Output:
# blas_lib_generic/
# ├── artifact_manifest.txt
# └── math-libs/
#     └── BLAS/
#         ├── rocBLAS/
#         │   └── stage/
#         │       └── lib/
#         │           └── librocblas.so.5.2  # Host library with .rocm_kpack_ref marker
#         └── hipBLASLt/
#             └── stage/
#                 └── lib/
#                     └── libhipblaslt.so.0.9

# Check arch-specific artifact structure
tree -L 5 /develop/tmp/recombined-artifacts/blas_lib_gfx110X/

# Expected Output:
# blas_lib_gfx110X/
# ├── artifact_manifest.txt
# ├── kpack/
# │   └── stage/
# │       └── .kpack/
# │           ├── blas_lib_gfx1100.kpack
# │           ├── blas_lib_gfx1101.kpack
# │           ├── blas_lib_gfx1102.kpack
# │           └── blas_lib.kpm
# └── math-libs/
#     └── BLAS/
#         ├── rocBLAS/
#         │   └── stage/
#         │       └── lib/
#         │           └── rocblas/library/
#         │               ├── TensileLibrary_*_gfx1100.dat
#         │               ├── TensileLibrary_*_gfx1101.dat
#         │               └── TensileLibrary_*_gfx1102.dat
#         └── hipBLASLt/
#             └── ...

# Check manifest
cat /develop/tmp/recombined-artifacts/blas_lib_generic/artifact_manifest.txt
cat /develop/tmp/recombined-artifacts/blas_lib_gfx110X/artifact_manifest.txt

# Inspect .kpm manifest (MessagePack format)
python -c "
import msgpack
with open('/develop/tmp/recombined-artifacts/blas_lib_gfx110X/kpack/stage/.kpack/blas_lib.kpm', 'rb') as f:
    manifest = msgpack.unpack(f, raw=False)
    print('Component:', manifest['component_name'])
    print('Architectures:', list(manifest['kpack_files'].keys()))
    for arch, entry in manifest['kpack_files'].items():
        print(f'  {arch}: {entry[\"file\"]} ({entry[\"size\"]} bytes, {entry[\"kernel_count\"]} kernels)')
"

# Verify generic artifact has NO .kpack files
echo "Checking generic artifact has no .kpack files:"
find /develop/tmp/recombined-artifacts/blas_lib_generic -name "*.kpack" | wc -l
# Expected: 0

# Verify arch-specific artifact has NO host libraries (outside kpack reference structure)
echo "Checking arch-specific artifact has no host libraries:"
find /develop/tmp/recombined-artifacts/blas_lib_gfx110X -name "*.so*" | grep -v ".kpack" | wc -l
# Expected: 0
```

## Common Issues and Solutions

### Issue: clang-offload-bundler not found

**Error**: `Could not find tool 'clang-offload-bundler' on system path`

**Solution**: Specify path explicitly with `--clang-offload-bundler` flag:

```bash
find ~/rocm -name "clang-offload-bundler" 2>/dev/null
# Use the path in your split command
```

### Issue: Out of disk space

**Error**: `No space left on device`

**Solution**: The workflow requires ~3x input artifact size:

- Input artifacts: 38 GB
- Split artifacts: ~40 GB
- Recombined artifacts: ~30 GB
- Temporary files: ~10 GB

Use `/develop/tmp` for temporary files (not `/tmp`) per CLAUDE.md instructions.

### Issue: Binary not stripped or grew in size

**Warning**: `Binary was not stripped or grew in size: /path/to/binary`

**Explanation**: Some test binaries or specific libraries may not shrink after stripping device code. This is usually harmless and doesn't affect functionality. The warning appears when:

- Binary is already host-only (no device code to strip)
- Binary has very small or zero-size .hip_fatbin section
- Binary structure makes it difficult to remove sections efficiently

This doesn't indicate a failure - the process still completes successfully.

## Next Steps

After recombination, these artifacts are ready for:

1. **Package creation** - Create .deb/.rpm packages from recombined artifacts
1. **Distribution** - Upload to package repositories
1. **Installation testing** - Verify packages install and work on target GPUs

## Reference

- **Tool source**: `/develop/rocm-kpack/python/rocm_kpack/tools/`
- **Design document**: `/develop/rocm-kpack/docs/kpack-build-integration.md`

## Appendix: File Formats

### artifact_manifest.txt

Plain text file listing installation prefixes (one per line):

```
math-libs/BLAS/rocBLAS/stage
math-libs/BLAS/hipBLASLt/stage
kpack/stage
```

### .kpack (Kernel Pack Archive)

Binary archive format containing GPU kernels:

- Magic: `KPAK` (0x4b50414b)
- TOC: MessagePack table of contents
- Kernels: Compressed (zstd) or uncompressed code objects

### .kpm (Kernel Pack Manifest)

MessagePack format listing available kpack files:

```python
{
  'format_version': 1,
  'component_name': 'blas_lib',
  'prefix': 'kpack/stage',
  'kpack_files': {
    'gfx1100': {
      'file': 'blas_lib_gfx1100.kpack',
      'size': 3456789,
      'kernel_count': 3
    },
    ...
  }
}
```

### .rocm_kpack_ref (Kpack Reference Marker)

Plain text format pointing to kpack search paths:

```
../../../.kpack:../../.kpack:.kpack
```

This file is mapped into a PT_LOAD segment and referenced by the `__CudaFatBinaryWrapper` to locate kernels at runtime.
