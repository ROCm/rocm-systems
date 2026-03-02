# Multi-Architecture Packaging with Kpack

## 1. Introduction and Purpose

### Problem Statement

ROCm applications and libraries traditionally ship device code in two problematic forms:

1. **Fat Binaries**: Executables and shared libraries contain embedded `.hip_fatbin` sections with device code for all supported GPU architectures. A single binary may contain code for gfx900, gfx906, gfx908, gfx90a, gfx940, gfx941, gfx942, gfx1030, gfx1100, gfx1101, gfx1102, and more. Since device code comprises a significant portion of these binaries, supporting many architectures leads to substantial bloat.

1. **Ad-hoc Database Directories**: Libraries like rocBLAS, hipBLASLt, rocFFT, and AOTriton maintain kernel databases in various formats (SQLite, filesystem hierarchies, custom formats) with inconsistent organization and no standard loading interface.

These approaches create several problems:

- **Storage overhead**: Disk space consumed by unused device code for architectures not present on the system
- **Bandwidth waste**: Network transfers of unnecessary architecture-specific code during installation
- **Memory pressure**: Runtime memory consumption from loading unused device code
- **Distribution complexity**: Difficulty creating per-architecture packages without invasive build changes
- **Inconsistent interfaces**: Each library implements its own kernel database format and loading mechanism

### Solution Overview

**Kpack** provides a structured format for separating device code from host binaries by architecture, combined with a unified runtime loading interface. The system has two primary components:

1. **Offload Kpacker**: Transforms fat binaries and kernel databases into architecture-separated kpack archives
1. **Runtime Integration**: Lazy-loading infrastructure in CLR and kernel libraries to load device code on-demand

The transformation process extracts device code from binaries, organizes it into compressed per-architecture archives with structured table-of-contents (TOC), and converts the host binaries to reference external device code through a standard marker format.

### Key Benefits

- **Per-Architecture Distribution**: Ship only the device code needed for target GPUs
- **Lazy Loading**: Load device code only when required by detected hardware
- **Reduced Installation Size**: Install trees contain device code for only the local GPU architecture
- **Bandwidth Efficiency**: Download only relevant architectures
- **Unified Interface**: Standard API for loading device code across CLR and kernel libraries
- **Compression**: Efficient storage using per-kernel zstd compression
- **Backwards Compatibility**: Unmodified host API and ABI

### Scope

This document covers:

- **Offload bundle transformation**: Converting fat binaries (executables, shared libraries) to kpack format
- **Kernel library databases**: Transforming SQLite databases and custom formats used by FFT, BLAS, and ML libraries
- **Runtime integration**: CLR modifications for loading kpack device code
- **Build implications**: Integration points for ROCm, PyTorch, and third-party builds

______________________________________________________________________

## 2. KPACK Transformations

### 2a. Fat Bundled Binaries → Arch-Separated Kpack Archives (offload_kpacker)

The **offload_kpacker** transforms fat binaries into kpack'd form through a four-phase process: device code extraction, kpack archive creation, binary conversion, and output organization.

#### Overview of Transformation Pipeline

```
Input: Fat Binary (executable or shared library)
  ↓
[Phase 1] Extract device code and unbundle by architecture
  ↓
[Phase 2] Create kpack archive with compressed kernels
  ↓
[Phase 3] Convert binary (zero-page, add marker, update pointers)
  ↓
[Phase 4] Write host-only binary + .kpack/ archive
  ↓
Output: Converted binary + per-arch kpack file
```

#### Phase 1: Device Code Extraction and Unbundling

**Step 1.1: Detect Bundled Binaries**

The system uses `readelf -S` to check for the `.hip_fatbin` ELF section:

```bash
readelf -S binary | grep .hip_fatbin
```

Binaries with this section contain device code and require processing. Non-ELF files and host-only binaries are copied verbatim.

**Step 1.2: Extract `.hip_fatbin` Section**

```bash
objcopy --dump-section .hip_fatbin=fatbin.bin binary
```

The extracted section contains a bundled blob in clang-offload-bundler format, which may be:

- Uncompressed bundle (multiple architecture code objects concatenated)
- Compressed Code Object Bundle (CCOB) - zstd-compressed per-bundle format
- Direct ELF code object (single architecture)

**Step 1.3: List Available Architectures**

```bash
clang-offload-bundler --type=o --input=fatbin.bin --list
```

Output example:

```
hipv4-amdgcn-amd-amdhsa--gfx1100
hipv4-amdgcn-amd-amdhsa--gfx1101
hipv4-amdgcn-amd-amdhsa--gfx1102
```

**Step 1.4: Unbundle Per-Architecture**

For each architecture:

```bash
clang-offload-bundler --type=o --input=fatbin.bin \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1100 \
  --output=gfx1100.hsaco --unbundle
```

Result: One `.hsaco` file per architecture containing the device code.

#### Phase 2: Kpack Archive Creation

**Kpack Archive Format**

A kpack archive is a binary file with three sections:

```
┌─────────────────────────────────────────┐
│ Fixed Binary Header (16 bytes)         │
│  - Magic: "KPAK" (4 bytes)             │
│  - Version: uint32 (4 bytes)           │
│  - TOC Offset: uint64 (8 bytes)        │
├─────────────────────────────────────────┤
│ Padding to 64-byte boundary             │
├─────────────────────────────────────────┤
│ Blob Data (compression-scheme specific) │
│  - Zstd: [num_kernels][frame]*         │
│  - NoOp: Concatenated kernel bytes     │
├─────────────────────────────────────────┤
│ MessagePack TOC (at TOC Offset)         │
│  - Metadata + per-kernel TOC entries    │
└─────────────────────────────────────────┘
```

**Table of Contents Structure**

```python
{
    "format_version": 1,
    "group_name": "rocm",  # Build group identifier
    "gfx_arch_family": "gfx1100",  # Primary architecture family
    "gfx_arches": ["gfx1100", "gfx1101", "gfx1102"],  # All covered arches
    "compression_scheme": "zstd-per-kernel",
    # Compression-specific metadata (zstd-per-kernel)
    "zstd_offset": 64,  # Start of blob data
    "zstd_size": 245628,  # Total blob size
    # Per-binary kernel TOC
    "toc": {
        "bin/hipcc": {
            "gfx1100": {
                "type": "hsaco",
                "ordinal": 0,  # Index into compression blob
                "original_size": 7472,  # Decompressed size
            },
            "gfx1101": {"type": "hsaco", "ordinal": 1, "original_size": 7488},
        },
        "lib/libamdhip64.so.6": {
            "gfx1100": {"type": "hsaco", "ordinal": 2, "original_size": 138456}
        },
    },
}
```

**Compression Strategies**

1. **NoOp Compression**: Kernels concatenated without compression (baseline)

   - Blob: Direct concatenation of .hsaco files
   - TOC: Each entry stores byte offset and size

1. **Zstd Per-Kernel**: Each kernel compressed independently (default)

   - Blob structure:
     ```
     [num_kernels: uint32]
     [frame_0_size: uint32][zstd_frame_0]
     [frame_1_size: uint32][zstd_frame_1]
     ...
     ```
   - TOC: Each entry stores ordinal (frame index)
   - Enables O(1) random access: decompress only requested kernel
   - Default compression level: 3 (balance speed/compression)

**Ordinal-Based Indexing**

Kernels are referenced by ordinal (0-indexed) rather than byte offset, allowing compression schemes to use arbitrary internal layouts. The TOC maps `(binary_name, architecture)` to ordinal, and the compressor handles ordinal→bytes mapping.

#### Phase 3: Binary Conversion

Binary conversion transforms a fat binary into a host-only binary that references external device code. This four-step process modifies the ELF binary in-place.

**Step 3.1: Zero-Page `.hip_fatbin` Section**

**Goal**: Reclaim disk space by zeroing the device code section (using kernel zero-page optimizations on both ELF and PE/COFF).

**Conservative Zero-Page Algorithm**:

The `.hip_fatbin` section typically spans multiple memory pages. We zero only fully page-aligned regions while preserving unaligned prefix/suffix bytes:

```
Original Section:
├─────────────────────────────────────────────────┤
  ^                                             ^
  section_start                          section_end
  (may be unaligned)                     (may be unaligned)

After Zero-Paging:
├──┬────────────────────────────────────────┬───┤
 P₀  [zeroed region]                        Pₙ
 (preserve unaligned prefix)        (preserve unaligned suffix)
```

**Algorithm**:

1. Find the first page boundary ≥ section_start
1. Find the last page boundary ≤ section_end
1. Zero bytes between these boundaries
1. Preserve bytes before first boundary (prefix)
1. Preserve bytes after last boundary (suffix)

**ELF Consequences**:

Zeroing the section may split its containing PT_LOAD segment into up to 5 pieces:

```
Original PT_LOAD:
[───────────────────────────────────]

After Zero-Paging:
[prefix][zeroed pages][suffix] (split into 3 segments)
```

All relocations, dynamic section entries, and GOT entries pointing into the modified region must be updated to account for the new segment layout.

**Step 3.2: Add `.rocm_kpack_ref` Marker Section**

**Goal**: Embed MessagePack metadata that tells the runtime where to find device code.

**Marker Content**:

```python
{
    "kernel_name": "bin/hipcc",  # Path for TOC lookup
    "kpack_search_paths": [
        "../.kpack/rocm-gfx1100.kpack",  # Relative to binary
        "../.kpack/rocm-gfx1200.kpack",  # Relative to binary
    ],
}
```

**Creation**:

```bash
# Create marker section with MessagePack data
objcopy --add-section .rocm_kpack_ref=marker.bin \
        --set-section-flags .rocm_kpack_ref=noload,readonly \
        binary binary.marked
```

The section has `SHF_ALLOC` cleared initially (not loaded at runtime).

**Step 3.3: Map Marker Section to New PT_LOAD Segment**

**Goal**: Make the marker section accessible at runtime by mapping it to a new loadable segment.

**Process**:

1. Create a new PT_LOAD segment with appropriate permissions (R)
1. Set segment's `p_offset` to marker section's file offset
1. Set segment's `p_filesz` and `p_memsz` to marker section size
1. Assign a virtual address for the segment (typically after existing segments)
1. Update section header: set `SHF_ALLOC` flag
1. Update section header: set `sh_addr` to match segment's virtual address

**Result**: The marker section is now loaded into memory at runtime, accessible via its virtual address.

**Step 3.4: Update `__CudaFatBinaryWrapper.binary` Pointer**

**Goal**: Redirect the fat binary pointer from the (now-zeroed) `.hip_fatbin` section to the `.rocm_kpack_ref` section.

**`__CudaFatBinaryWrapper` Structure**:

```c
struct __CudaFatBinaryWrapper {
    uint32_t magic;       // 0x48495046 (HIPF) → 0x4B504948 (HIPK)
    uint32_t version;     // 1
    void* binary;         // Pointer to device code or metadata
    void* reserved1;
};
```

**Pointer Update Process**:

1. Locate `__CudaFatBinaryWrapper` instances in `.data` or `.rodata`

   - Search for HIPF magic: `0x48495046`
   - Verify version field: `0x00000001`

1. Read the `binary` pointer (offset +8 from magic)

1. Calculate new pointer value:

   ```
   new_pointer = .rocm_kpack_ref_vaddr + (old_pointer - .hip_fatbin_vaddr)
   ```

   This preserves any offset within the section (usually 0).

1. Write new pointer back to wrapper structure

1. Update any relocations referencing this pointer

**Relocation Handling**:

If the pointer is subject to dynamic relocation (e.g., in PIE executables or DYN libraries), update the relocation entry:

```c
// Example: R_X86_64_RELATIVE relocation
relocation.r_addend = new_pointer_value;
```

This ensures the dynamic linker applies the correct runtime adjustment.

**Step 3.5: Rewrite Magic (HIPF → HIPK)**

**Goal**: Signal to the runtime that this binary uses kpack device code.

**Magic Values**:

- `HIPF` (0x48495046): Fat binary with embedded device code
- `HIPK` (0x4B504948): Kpack'd binary with external device code

**Process**:

1. Locate all `__CudaFatBinaryWrapper` structures (found in step 3.4)
1. Overwrite magic field: `0x48495046` → `0x4B504948`
1. Leave all other fields unchanged

**Runtime Detection**:

```c
if (wrapper->magic == 0x4B504948) {  // HIPK
    // Parse wrapper->binary as MessagePack metadata
    // Load from .kpack file
} else if (wrapper->magic == 0x48495046) {  // HIPF
    // Parse wrapper->binary as fat binary
    // Extract via COMGR
}
```

#### Phase 4: Output Structure

The transformation produces a reorganized install tree:

```
output_root/
├── .kpack/
│   ├── rocm-gfx1100.kpack          # gfx1100 family device code
│   ├── rocm-gfx900.kpack           # gfx900 family device code
│   └── rocm-gfx942.kpack           # gfx942 family device code
├── bin/
│   ├── hipcc                        # Host-only + marker
│   └── rocm-smi                     # Unmodified (no device code)
├── lib/
│   ├── libamdhip64.so.6            # Host-only + marker
│   ├── librocblas.so.4             # Host-only + marker
│   └── libhsakmt.so.1              # Unmodified (no device code)
└── share/                           # Unmodified
    └── doc/
```

**Key Properties**:

- **Single kpack per architecture family**: All binaries for gfx1100/1101/1102 share one `rocm-gfx1100.kpack`
- **Relative search paths**: Binaries reference `../.kpack/` allowing relocatable installs
- **Transparent to unmodified code**: Host API and ABI unchanged

______________________________________________________________________

### 2b. Kernel Library Database Kpack Archives

Kernel libraries (rocFFT, rocBLAS, hipBLASLt, AOTriton) maintain large collections of pre-compiled kernels optimized for different problem sizes, data types, and hardware configurations. These libraries currently use custom database formats that kpack standardizes.

#### Adoption Timeline and Strategy

**Initial Rollout**: FFT libraries (rocFFT, hipFFT) will be the first to adopt kpack format. These libraries currently use SQLite databases that benefit significantly from kpack's structured TOC and compression.

**Deferred Migration**: Other kernel libraries (hipBLASLt, rocBLAS, AOTriton) will initially **not** be recoded to use kpack. These libraries already have on-disk layouts that support architecture-separated packaging:

- **hipBLASLt**: Directory-based structure with `gfx*/` subdirectories already enables per-architecture distribution
- **rocBLAS**: Tensile library format with separate architecture directories
- **AOTriton**: Kernel directory organization already architecture-separated

Since these libraries can already be split-packaged without kpack transformation, the immediate benefit is limited. They will continue using their legacy formats in the initial rollout.

**Future Migration**: As the kpack system matures and proves its benefits (compression, unified loading API, tooling ecosystem), these libraries will migrate away from their legacy formats to kpack. This migration will happen incrementally as the runtime infrastructure stabilizes and performance characteristics are validated.

**Rationale**: Focus initial effort on libraries with the most to gain (FFT's SQLite databases) while allowing existing split-packageable libraries to continue functioning without disruption.

#### Common Architecture

All kernel library transformations follow this pattern:

```
Input: Library-specific database format
  ↓
[Scan & Detect] Recognize database structure
  ↓
[Extract] Enumerate kernel artifacts by architecture
  ↓
[Pack] Add to kpack archive with structured TOC
  ↓
[Update Library] Modify library to load from kpack
  ↓
Output: Kpack archive + modified library code
```

#### Plugin System: DatabaseRecognizer

The kpack system provides an extensible plugin architecture for recognizing and processing different database formats:

```python
class DatabaseRecognizer(ABC):
    """Abstract interface for recognizing kernel database formats."""

    @abstractmethod
    def can_recognize(self, path: Path) -> bool:
        """Fast heuristic check (e.g., check filename, magic bytes)."""
        pass

    @abstractmethod
    def recognize(self, path: Path) -> Optional[KernelDatabase]:
        """Thorough validation and parsing. Returns KernelDatabase if valid."""
        pass


class KernelDatabase(ABC):
    """Abstract interface for a recognized kernel database."""

    @abstractmethod
    def get_architectures(self) -> List[str]:
        """Return list of GPU architectures (e.g., ['gfx1100', 'gfx90a'])."""
        pass

    @abstractmethod
    def get_kernel_artifacts(self) -> Iterator[KernelArtifact]:
        """Yield all kernel artifacts in the database."""
        pass


class KernelArtifact:
    """Represents a single kernel artifact."""

    relative_path: Path  # Path within database
    gfx_target: str  # Architecture (e.g., 'gfx1100')
    artifact_type: str  # 'hsaco', 'co', etc.
    metadata: Dict[str, Any]  # Library-specific metadata
```

**Recognizer Registry**:

```python
class RecognizerRegistry:
    def register(self, recognizer: DatabaseRecognizer):
        """Register a new database recognizer plugin."""

    def try_recognize(self, path: Path) -> Optional[KernelDatabase]:
        """Try all recognizers in order, return first match."""
```

Library-specific recognizers are registered at initialization:

```python
registry = RecognizerRegistry()
registry.register(RocFFTRecognizer())
registry.register(RocBLASRecognizer())
registry.register(HipBLASLtRecognizer())
registry.register(AOTritonRecognizer())
```

#### FFT Libraries (rocFFT, hipFFT)

**Current Format**: SQLite database with tables mapping problem configurations to kernel file paths.

**Database Structure**:

```sql
-- Simplified schema
CREATE TABLE kernels (
    arch TEXT,           -- 'gfx1100'
    length INTEGER,      -- FFT length
    precision TEXT,      -- 'single', 'double'
    kernel_path TEXT     -- Path to .co file
);
```

**Transformation Approach**:

1. **Scan**: FFTRecognizer detects SQLite files in `lib/rocfft/` or similar
1. **Extract**: Query database for all `(arch, kernel_path)` pairs
1. **Load Kernels**: Read each `.co` file from disk
1. **Pack**: Add to kpack TOC with structured key:
   ```python
   toc_key = f"fft/{arch}/{length}/{precision}/kernel"
   ```
1. **Update Library**: Modify rocFFT to query kpack instead of SQLite
   - Lookup: `kpack.get_kernel("fft/gfx1100/1024/single/kernel")`
   - Load returned HSACO into ROCm runtime

**Metadata Preservation**:

FFT-specific metadata (length, precision, batch size) is preserved in the TOC:

```python
{
    "toc": {
        "fft/gfx1100/1024/single/kernel": {
            "type": "co",
            "ordinal": 42,
            "metadata": {
                "length": 1024,
                "precision": "single",
                "factors": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            },
        }
    }
}
```

#### hipBLASLt

**Current Format**: Directory tree with hierarchical organization.

**Structure**:

```
hipblaslt/
└── library/
    └── TensileLibrary/
        ├── gfx1100/
        │   ├── gemm_kernel_1.co
        │   └── gemm_kernel_2.co
        └── gfx90a/
            └── gemm_kernel_1.co
```

**Transformation Approach**:

1. **Scan**: HipBLASLtRecognizer detects `TensileLibrary/` directory structure
1. **Extract**: Walk directory tree, identify architecture from path
1. **Parse Metadata**: Extract GEMM parameters from kernel names (M, N, K, datatype)
1. **Pack**: Add to kpack TOC:
   ```python
   toc_key = f"hipblaslt/{arch}/gemm_kernel_{id}"
   ```
1. **Update Library**: Modify hipBLASLt kernel selection logic
   - Match problem to kernel based on M/N/K
   - Load from: `kpack.get_kernel("hipblaslt/gfx1100/gemm_kernel_1")`

**Problem Matching**:

The TOC preserves GEMM parameters for kernel selection:

```python
{
    "hipblaslt/gfx1100/gemm_kernel_1": {
        "type": "co",
        "ordinal": 15,
        "metadata": {
            "M": 4096,
            "N": 4096,
            "K": 4096,
            "datatype": "FP16",
            "transpose_a": false,
            "transpose_b": false,
        },
    }
}
```

#### rocBLAS

**Current Format**: Tensile library with logic YAML files and kernel co-objects.

**Structure**:

```
rocblas/
└── library/
    ├── TensileLibrary.yaml     # Logic file (problem→kernel mapping)
    └── gfx1100/
        ├── kernel_1.co
        └── kernel_2.co
```

**Transformation Approach**:

1. **Scan**: RocBLASRecognizer detects `TensileLibrary.yaml` + `gfx*/` structure
1. **Parse Logic**: Read YAML to understand problem→kernel mapping
1. **Extract Kernels**: Identify all referenced `.co` files per architecture
1. **Pack**: Preserve Tensile naming:
   ```python
   toc_key = f"rocblas/{arch}/{tensile_kernel_name}"
   ```
1. **Update Library**: Modify Tensile library loader
   - Query logic remains same (YAML-based matching)
   - Kernel loading: `kpack.get_kernel(f"rocblas/{arch}/{kernel_name}")`

**Logic File Handling**:

The YAML logic file can be either:

- Packed into kpack TOC as metadata
- Kept as separate file (smaller, easier to update)

Tradeoff: Packing logic increases kpack size but simplifies deployment.

#### AOTriton

**Current Format**: AOT-compiled Triton kernels in directory structure.

**Structure**:

```
aotriton/
└── kernels/
    ├── gfx1100/
    │   ├── flash_attention_fwd.hsaco
    │   └── flash_attention_bwd.hsaco
    └── gfx90a/
        └── flash_attention_fwd.hsaco
```

**Transformation Approach**:

1. **Scan**: AOTritonRecognizer detects `kernels/gfx*/` pattern
1. **Extract**: Enumerate `.hsaco` files per architecture
1. **Pack**: Use kernel operation names:
   ```python
   toc_key = f"aotriton/{arch}/{operation_name}"
   ```
1. **Update Library**: Modify AOTriton runtime
   - Operation dispatch remains same
   - Kernel loading: `kpack.get_kernel(f"aotriton/{arch}/flash_attention_fwd")`

**Kernel Signatures**:

AOTriton kernels require type signatures for runtime dispatch. These are preserved in metadata:

```python
{
    "aotriton/gfx1100/flash_attention_fwd": {
        "type": "hsaco",
        "ordinal": 8,
        "metadata": {
            "operation": "flash_attention_fwd",
            "signature": "(f32[M,N], f32[N,K]) -> f32[M,K]",
            "tuning_params": {"block_m": 64, "block_n": 64},
        },
    }
}
```

#### Common Patterns

All library transformations share these design principles:

1. **Preserve Dispatch Logic**: Library's kernel selection algorithm unchanged
1. **Structured TOC Keys**: Hierarchical naming enables efficient lookup
1. **Metadata Retention**: Library-specific parameters stored in TOC
1. **Lazy Loading**: Load only kernels used by application
1. **Architecture Filtering**: Libraries query only their target architecture's kernels

______________________________________________________________________

## 3. Runtime Integrations

Runtime integration enables lazy loading of kpack device code when needed, rather than loading all architectures upfront. This section covers the C++ unpacking library, CLR modifications, and kernel library integration patterns.

### 3a. librocm_kpack.so Unpacking Library (Architecture)

The `librocm_kpack.so` shared library provides a standard C API for loading device code from kpack archives. It abstracts archive format details and provides efficient random access to compressed kernels.

**Implementation Note**: The unpacking library may be implemented as a **header-only C++ library** rather than a traditional shared library. This design choice offers several advantages:

- **Deployment Simplicity**: No separate .so file to distribute, version, or link against
- **Avoids Library Layering Issues**: CLR and kernel libraries can include the unpacking code directly without introducing circular dependencies or version conflicts
- **Multiple Contexts**: Easy to use in CLR runtime, kernel libraries, standalone tools, and test infrastructure without ABI concerns
- **Inline Optimization**: Compiler can optimize across unpacking library boundaries (e.g., inline decompression into caller)

**Tradeoff**: Header-only implementation increases compile times for dependent libraries and results in code duplication across binaries (each binary includes its own copy). However, given the relatively small size of the unpacking logic (~5-10KB compiled) and the significant deployment benefits, this tradeoff is favorable.

The C API presented below remains the same regardless of whether the library is implemented as header-only C++ or a traditional shared library. The header-only implementation would provide the C API through inline functions or template instantiations.

#### Public API Design

```c
// Opaque handle to open kpack archive
typedef struct kpack_archive* kpack_archive_t;

// Opaque handle to kernel artifact
typedef struct kpack_kernel* kpack_kernel_t;

// Error codes
typedef enum {
    KPACK_SUCCESS = 0,
    KPACK_ERROR_FILE_NOT_FOUND = 1,
    KPACK_ERROR_INVALID_FORMAT = 2,
    KPACK_ERROR_UNSUPPORTED_VERSION = 3,
    KPACK_ERROR_DECOMPRESSION_FAILED = 4,
    KPACK_ERROR_KERNEL_NOT_FOUND = 5,
    KPACK_ERROR_OUT_OF_MEMORY = 6
} kpack_error_t;

// Open a kpack archive for reading
kpack_error_t kpack_open(
    const char* path,           // Path to .kpack file
    kpack_archive_t* archive    // Output: archive handle
);

// Close archive and free resources
void kpack_close(kpack_archive_t archive);

// Query available architectures in archive
kpack_error_t kpack_get_architectures(
    kpack_archive_t archive,
    const char*** arches,       // Output: NULL-terminated array
    size_t* count               // Output: number of arches
);

// Query available kernels for a specific binary + architecture
kpack_error_t kpack_list_kernels(
    kpack_archive_t archive,
    const char* binary_name,    // e.g., "bin/hipcc"
    const char* arch,           // e.g., "gfx1100"
    const char*** kernel_names, // Output: NULL-terminated array
    size_t* count               // Output: number of kernels
);

// Load a kernel by name and architecture
kpack_error_t kpack_get_kernel(
    kpack_archive_t archive,
    const char* binary_name,    // e.g., "bin/hipcc"
    const char* arch,           // e.g., "gfx1100"
    const void** kernel_data,   // Output: pointer to kernel bytes
    size_t* kernel_size         // Output: size in bytes
);

// Free kernel data returned by kpack_get_kernel
void kpack_free_kernel(
    kpack_archive_t archive,
    const void* kernel_data
);

// Query TOC metadata for a kernel
kpack_error_t kpack_get_kernel_metadata(
    kpack_archive_t archive,
    const char* binary_name,
    const char* arch,
    const char** metadata_json, // Output: JSON string
    size_t* size                // Output: JSON string length
);
```

#### Integration Points

**CLR Integration**:

The CLR uses librocm_kpack.so during lazy loading:

1. Parse `.rocm_kpack_ref` marker section
1. Iterate through `kpack_search_paths`
1. `kpack_open()` first valid kpack file
1. `kpack_get_kernel()` for detected GPU architecture
1. Pass kernel bytes to `FatBinaryInfo::AddDevProgram()`
1. `kpack_close()` when binary unloads

**Kernel Library Integration**:

Libraries (rocFFT, rocBLAS, hipBLASLt) use librocm_kpack.so for kernel databases:

1. During library initialization, `kpack_open()` library-specific kpack
1. Query device architecture
1. `kpack_list_kernels()` to enumerate available kernels
1. Cache metadata for dispatch logic
1. On first use: `kpack_get_kernel()` for selected kernel
1. Load kernel into ROCm runtime
1. `kpack_close()` at library teardown

#### Thread Safety

**Guarantees**:

- Multiple threads can call `kpack_open()` concurrently (different archives)
- Multiple threads can query/load from same archive concurrently
- Internal locking protects shared state (decompression context, TOC cache)

**User Responsibilities**:

- User must not call `kpack_close()` while other threads use the archive
- User must serialize calls to `kpack_free_kernel()` for same kernel data

**Implementation Strategy**:

- Read-only operations (get_architectures, get_kernel) use shared locks
- TOC parsing and decompression context creation use exclusive locks
- Decompression is thread-safe (separate zstd context per operation)

#### Error Handling Strategy

**Fail-Fast Philosophy**:

- Invalid kpack format: Return error immediately (no partial parsing)
- Missing kernel: Return `KPACK_ERROR_KERNEL_NOT_FOUND` (don't search elsewhere)
- Decompression failure: Return error (don't try fallback)

**Error Context**:

```c
// Extended error information (optional, for debugging)
const char* kpack_get_last_error(kpack_archive_t archive);
```

Returns human-readable error message for last failed operation on this archive.

**Caller's Responsibility**:

- CLR: Try next path in `kpack_search_paths` on `KPACK_ERROR_FILE_NOT_FOUND`
- Libraries: Fall back to original kernel loading mechanism on any error

______________________________________________________________________

### 3b. CLR Pseudo-code

The CLR (Common Language Runtime) for HIP requires modifications to detect kpack'd binaries and load device code from external archives. This section shows both the POC implementation and the production design.

#### POC Implementation (PR #1755)

The proof-of-concept implementation demonstrates the core pattern with minimal changes. It uses environment variable hacks for device code location and a global map for lazy loading.

**Reference**: https://github.com/ROCm/rocm-systems/pull/1755

**HIPK Magic Detection (`__hipRegisterFatBinary`)**:

```c
// hip_platform.cpp
void** __hipRegisterFatBinary(const void* data) {
    const __CudaFatBinaryWrapper* wrapper =
        (const __CudaFatBinaryWrapper*)data;

    // Check for kpack'd binary (BEFORE dereferencing binary pointer!)
    if (wrapper->magic == 0x4B504948 /* HIPK */ && wrapper->version == 1) {
        // Parse MessagePack metadata at wrapper->binary
        KpackMetadata metadata;
        if (!parseKpackMetadata(wrapper->binary, &metadata)) {
            return nullptr;
        }

        // POC HACK: Use environment variable for device code path
        const char* device_code_path = getenv("HIP_KPACK_DEVICE_CODE");
        if (!device_code_path) {
            return nullptr;
        }

        // Load .hsaco file from disk
        char* hsaco_buffer;
        size_t file_size;
        if (!load_file(device_code_path, &hsaco_buffer, &file_size)) {
            return nullptr;
        }

        // Store device code in global map for lazy loading
        PlatformState::instance().registerKpackDeviceCode(
            wrapper, hsaco_buffer, file_size
        );

        // Register with initialized=false (defers loading until first use)
        bool success = false;
        FatBinaryInfo** fat_binary_info =
            PlatformState::instance().addFatBinary(wrapper, success);

        // Set *fat_binary_info = nullptr to trigger lazy load
        *fat_binary_info = nullptr;

        return (void**)fat_binary_info;
    }

    // Normal fat binary path (HIPF magic)
    // ... existing code ...
}
```

**Lazy Loading Hook (`StatCO::digestFatBinary`)**:

```c
// hip_code_object.cpp
hipError_t StatCO::digestFatBinary(const void* data, FatBinaryInfo*& programs) {
    // Check if already loaded
    if (programs != nullptr) {
        return hipSuccess;
    }

    // Check if this is kpack device code
    char* kpack_buffer = nullptr;
    size_t kpack_size = 0;
    if (PlatformState::instance().getKpackDeviceCode(data, kpack_buffer, kpack_size)) {
        // Lazy load triggered - runtime is now initialized
        if (g_devices.size() == 0) {
            return hipErrorNoDevice;
        }

        // Create FatBinaryInfo with nullptr image
        FatBinaryInfo* fb_info = new FatBinaryInfo(nullptr, nullptr);

        // Add device program for detected architecture
        for (size_t dev_idx = 0; dev_idx < g_devices.size(); dev_idx++) {
            hipError_t status = fb_info->AddDevProgram(
                g_devices[dev_idx],
                kpack_buffer,
                kpack_size,
                0
            );
            if (status != hipSuccess) {
                delete fb_info;
                return status;
            }
        }

        // Build program (load into GPU)
        for (size_t dev_idx = 0; dev_idx < g_devices.size(); dev_idx++) {
            hipError_t status = fb_info->BuildProgram(dev_idx);
            if (status != hipSuccess) {
                delete fb_info;
                return status;
            }
        }

        programs = fb_info;
        return hipSuccess;
    }

    // Normal fat binary path (extract via COMGR)
    // ... existing code ...
}
```

**POC Characteristics**:

- Environment variable `HIP_KPACK_DEVICE_CODE` points to .hsaco file
- Global map `kpack_device_code_map_` stores device code keyed by wrapper pointer
- MessagePack parsing implemented inline
- No kpack archive support (single .hsaco file only)
- Direct .hsaco loading bypasses COMGR unbundling

**POC Limitations**:

- Cannot load from kpack archives (only raw .hsaco)
- No search path resolution
- No per-binary TOC lookup
- Environment variable required (not production-ready)
- No error recovery or fallback

#### Production Design Evolution

The production implementation builds on the POC pattern with proper kpack file resolution, librocm_kpack.so integration, and robust error handling.

**HIPK Detection and Metadata Parsing**:

```c
// hip_platform.cpp
void** __hipRegisterFatBinary(const void* data) {
    const __CudaFatBinaryWrapper* wrapper =
        (const __CudaFatBinaryWrapper*)data;

    if (wrapper->magic == 0x4B504948 /* HIPK */ && wrapper->version == 1) {
        // Parse MessagePack metadata at wrapper->binary
        KpackMetadata metadata;
        if (!parseKpackMetadata(wrapper->binary, MAX_METADATA_SIZE, &metadata)) {
            LogError("Failed to parse kpack metadata");
            return nullptr;
        }

        // Store metadata in PlatformState for lazy load
        PlatformState::instance().registerKpackBinary(
            wrapper,
            metadata.kernel_name,
            metadata.kpack_search_paths
        );

        // Register with initialized=false (defers loading)
        bool success = false;
        FatBinaryInfo** fat_binary_info =
            PlatformState::instance().addFatBinary(wrapper, success);

        *fat_binary_info = nullptr;  // Trigger lazy load

        return (void**)fat_binary_info;
    }

    // ... normal path ...
}
```

**Kpack File Resolution**:

```c
// hip_platform.cpp
kpack_archive_t PlatformState::openKpackArchive(
    const std::vector<std::string>& search_paths,
    const char* binary_path  // For path resolution
) {
    for (const auto& search_path : search_paths) {
        // Resolve relative paths relative to binary location
        std::string full_path;
        if (search_path[0] == '/' || search_path[0] == '~') {
            // Absolute path
            full_path = search_path;
        } else {
            // Relative path: resolve relative to binary's directory
            std::string binary_dir = dirname(binary_path);
            full_path = binary_dir + "/" + search_path;
        }

        // Normalize path (resolve .., symlinks, etc.)
        char* resolved = realpath(full_path.c_str(), nullptr);
        if (!resolved) {
            continue;  // Path doesn't exist, try next
        }

        // Try to open kpack archive
        kpack_archive_t archive;
        kpack_error_t err = kpack_open(resolved, &archive);
        free(resolved);

        if (err == KPACK_SUCCESS) {
            return archive;  // Success!
        }

        // Log error and continue to next path
        LogWarning("Failed to open kpack at %s: %s",
                   full_path.c_str(), kpack_error_string(err));
    }

    // No valid kpack found
    LogError("No valid kpack found in search paths");
    return nullptr;
}
```

**Lazy Load with librocm_kpack.so**:

```c
// hip_code_object.cpp
hipError_t StatCO::digestFatBinary(const void* data, FatBinaryInfo*& programs) {
    if (programs != nullptr) {
        return hipSuccess;
    }

    // Check if this is kpack metadata
    std::string kernel_name;
    std::vector<std::string> search_paths;
    if (!PlatformState::instance().getKpackMetadata(
            data, kernel_name, search_paths)) {
        // Not a kpack binary, use normal path
        return digestFatBinaryNormal(data, programs);
    }

    // Open kpack archive
    kpack_archive_t archive = PlatformState::instance().openKpackArchive(
        search_paths, /* binary_path from /proc/self/maps or similar */
    );
    if (!archive) {
        LogError("Failed to open kpack archive");
        return hipErrorInvalidKernelFile;
    }

    // Ensure cleanup on all exit paths
    auto cleanup = [&]() { kpack_close(archive); };
    std::unique_ptr<void, decltype(cleanup)> guard(nullptr, cleanup);

    // Runtime should be initialized at lazy load time
    if (g_devices.size() == 0) {
        return hipErrorNoDevice;
    }

    // Create FatBinaryInfo
    FatBinaryInfo* fb_info = new FatBinaryInfo(nullptr, nullptr);

    // Load kernel for each device
    for (size_t dev_idx = 0; dev_idx < g_devices.size(); dev_idx++) {
        // Get device architecture
        const char* arch = g_devices[dev_idx]->isa().name();  // e.g., "gfx1100"

        // Load kernel from kpack
        const void* kernel_data;
        size_t kernel_size;
        kpack_error_t err = kpack_get_kernel(
            archive,
            kernel_name.c_str(),
            arch,
            &kernel_data,
            &kernel_size
        );

        if (err != KPACK_SUCCESS) {
            if (err == KPACK_ERROR_KERNEL_NOT_FOUND) {
                // Try generic fallback (e.g., gfx1100 → gfx11-generic)
                const char* generic_arch = getGenericArch(arch);
                if (generic_arch) {
                    err = kpack_get_kernel(
                        archive, kernel_name.c_str(), generic_arch,
                        &kernel_data, &kernel_size
                    );
                }
            }

            if (err != KPACK_SUCCESS) {
                LogError("Failed to load kernel for arch %s: %s",
                         arch, kpack_error_string(err));
                delete fb_info;
                return hipErrorInvalidKernelFile;
            }
        }

        // Add device program
        hipError_t status = fb_info->AddDevProgram(
            g_devices[dev_idx],
            (char*)kernel_data,
            kernel_size,
            0
        );

        // Free kernel data
        kpack_free_kernel(archive, kernel_data);

        if (status != hipSuccess) {
            delete fb_info;
            return status;
        }
    }

    // Build programs
    for (size_t dev_idx = 0; dev_idx < g_devices.size(); dev_idx++) {
        hipError_t status = fb_info->BuildProgram(dev_idx);
        if (status != hipSuccess) {
            delete fb_info;
            return status;
        }
    }

    programs = fb_info;
    return hipSuccess;
}
```

**Architecture Generic Fallback**:

```c
const char* getGenericArch(const char* specific_arch) {
    // Map specific targets to generic fallbacks
    // gfx1100, gfx1101, gfx1102 → gfx11-generic
    // gfx900, gfx906, gfx908, gfx90a → gfx9-generic

    if (strncmp(specific_arch, "gfx11", 5) == 0) {
        return "gfx11-generic";
    } else if (strncmp(specific_arch, "gfx10", 5) == 0) {
        return "gfx10-generic";
    } else if (strncmp(specific_arch, "gfx9", 4) == 0) {
        return "gfx9-generic";
    }

    return nullptr;  // No generic fallback
}
```

**Error Handling and Fallback**:

If kpack loading fails, the CLR can optionally fall back to embedded device code:

```c
hipError_t StatCO::digestFatBinary(const void* data, FatBinaryInfo*& programs) {
    // ... kpack loading attempt ...

    if (kpack_load_failed) {
        // Check if .hip_fatbin section still has data (not zero-paged)
        if (hasFatBinData(data)) {
            LogWarning("Kpack load failed, falling back to embedded device code");
            return digestFatBinaryNormal(data, programs);
        } else {
            LogError("Kpack load failed and no embedded device code available");
            return hipErrorInvalidKernelFile;
        }
    }

    // ... success path ...
}
```

**Production Design Benefits**:

- Proper path resolution (relative to binary location)
- Kpack archive support with TOC lookup
- Architecture-specific kernel loading
- Generic fallback for compatibility
- Robust error handling
- Optional fallback to embedded device code
- Thread-safe (librocm_kpack.so guarantees)

______________________________________________________________________

### 3c. Kernel Library Pseudo-code

Kernel libraries follow a similar pattern to CLR but query kpack archives based on problem parameters rather than just architecture.

#### General Pattern

```c
// Simplified example for rocFFT

typedef struct {
    kpack_archive_t archive;
    const char* library_name;  // "rocfft", "rocblas", etc.
    char current_arch[32];     // "gfx1100"
    // Cache of loaded kernels
    std::map<std::string, void*> kernel_cache;
} LibraryContext;

// Library initialization
hipError_t rocfft_initialize() {
    LibraryContext* ctx = allocate_context();

    // Detect GPU architecture
    hipDevice_t device;
    hipGetDevice(&device);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device);
    strncpy(ctx->current_arch, props.gcnArchName, sizeof(ctx->current_arch));

    // Open kpack archive for this library
    const char* kpack_path = find_library_kpack("rocfft", ctx->current_arch);
    if (!kpack_path) {
        // Fall back to legacy database
        return rocfft_initialize_legacy();
    }

    kpack_error_t err = kpack_open(kpack_path, &ctx->archive);
    if (err != KPACK_SUCCESS) {
        return rocfft_initialize_legacy();
    }

    return hipSuccess;
}

// Kernel selection and loading
hipError_t rocfft_execute_plan(rocfft_plan plan) {
    LibraryContext* ctx = get_context();

    // Determine kernel based on problem size, precision, etc.
    std::string kernel_key = select_kernel(
        plan->length,
        plan->precision,
        plan->batch_size
    );

    // Check cache
    if (ctx->kernel_cache.find(kernel_key) != ctx->kernel_cache.end()) {
        return launch_kernel(ctx->kernel_cache[kernel_key], plan);
    }

    // Load from kpack
    std::string toc_key =
        "rocfft/" + std::string(ctx->current_arch) + "/" + kernel_key;

    const void* kernel_data;
    size_t kernel_size;
    kpack_error_t err = kpack_get_kernel(
        ctx->archive,
        toc_key.c_str(),
        ctx->current_arch,
        &kernel_data,
        &kernel_size
    );

    if (err != KPACK_SUCCESS) {
        // Fall back to legacy loading
        return rocfft_load_kernel_legacy(kernel_key);
    }

    // Load kernel into ROCm runtime
    hipModule_t module;
    hipModuleLoadData(&module, kernel_data);

    hipFunction_t function;
    hipModuleGetFunction(&function, module, "fft_kernel");

    kpack_free_kernel(ctx->archive, kernel_data);

    // Cache for future use
    ctx->kernel_cache[kernel_key] = function;

    return launch_kernel(function, plan);
}

// Library teardown
void rocfft_cleanup() {
    LibraryContext* ctx = get_context();

    if (ctx->archive) {
        kpack_close(ctx->archive);
    }

    // Unload cached kernels
    for (auto& entry : ctx->kernel_cache) {
        hipModuleUnload(entry.second);
    }

    free_context(ctx);
}
```

#### Library-Specific TOC Lookup

Each library uses a structured TOC key format:

**rocFFT**:

```
rocfft/{arch}/{length}/{precision}/kernel
```

**rocBLAS**:

```
rocblas/{arch}/{gemm|axpy|...}/{datatype}/{transA}_{transB}
```

**hipBLASLt**:

```
hipblaslt/{arch}/gemm_{M}x{N}x{K}_{dtype}
```

**AOTriton**:

```
aotriton/{arch}/{operation_name}
```

#### Caching Strategy

Libraries should cache loaded kernels to avoid repeated decompression:

1. **First access**: Load from kpack, decompress, load into GPU, cache handle
1. **Subsequent access**: Reuse cached GPU handle
1. **Cache eviction**: Unload least-recently-used kernels if memory pressure

#### Performance Considerations

**Lazy Loading Overhead**:

- First kernel load: ~1-5ms (kpack lookup + decompression + GPU load)
- Cached access: ~microseconds (same as legacy path)

**Mitigation**:

- Preload common kernels at library initialization
- Use metadata to predict kernel usage patterns
- Async loading during idle GPU time

**Memory vs. Storage Tradeoff**:

- Loading all kernels upfront: Higher memory, faster execution
- Lazy loading: Lower memory, slight first-use latency
- Recommendation: Lazy load for large libraries (rocBLAS), preload for small (rocFFT)

______________________________________________________________________

## 4. Build and Packaging Implications

This section covers how kpack integrates into build systems and packaging workflows for ROCm, PyTorch, and third-party projects.

### 4.1. ROCm Build System (TheRock)

The ROCm build system (TheRock) integrates kpack through a multi-phase pipeline that separates generic (architecture-independent) builds from device-specific (per-architecture) builds, then merges them during a pre-packaging phase.

#### 4.1.1 Current Build Model (Baseline)

TheRock builds continue to work as they do today by building individual architecture families in separate invocations. This ensures backward compatibility and allows incremental adoption of kpack optimizations.

**Current Workflow**:

```
For each gfx family (gfx900, gfx1100, gfx1200, etc.):
  cmake -DTHEROCK_AMDGPU_FAMILIES=gfx110X -DTHEROCK_AMDGPU_DIST_ARCHITECTURES=<<all available>> ...
  ninja
  → Produces fat binaries with device code for gfx1100/1101/1102
```

Each build invocation compiles both host code (compiler, runtime libraries) and device code (GPU kernels), even though the host code is identical across architectures.

#### 4.1.2 Build Pipeline Optimization (Split Architecture Builds)

To eliminate redundant host code compilation, the build pipeline is split into two phases:

**Phase 1: Generic Build (Once for All Architectures)**

Build architecture-independent components once:

```bash
# Generic build: No device code, host-only
cmake -B build-generic \
      -S /therock \
      -DTHEROCK_BUILD_DEVICE_CODE=OFF \
      -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
      -DTHEROCK_ENABLE_COMPILER=ON \
      ... other host-only components ...

ninja -C build-generic
```

**Outputs**:

- Compiler binaries (`clang`, `lld`, `hipcc`)
- System libraries (host-only `libamdhip64.so`, `libhsa-runtime64.so`)
- Build artifacts needed for device code compilation (headers, CMake configs)

**Phase 2: Architecture-Specific Build Matrix**

For each architecture family, build only device-specific code using artifacts from Phase 1:

```bash
# For gfx110X family
cmake -B build-gfx110X \
      -S /therock \
      -DTHEROCK_AMDGPU_FAMILIES=gfx110X -DTHEROCK_AMDGPU_DIST_ARCHITECTURES=<<all available>> \
      -DTHEROCK_BUILD_DEVICE_CODE=ON \
      -DTHEROCK_GENERIC_BUILD_DIR=build-generic

ninja -C build-gfx110X

# For gfx120X family
cmake -B build-gfx120X \
      -S /therock \
      -DTHEROCK_AMDGPU_FAMILIES=gfx120X -DTHEROCK_AMDGPU_DIST_ARCHITECTURES=<<all available>> \
      -DTHEROCK_BUILD_DEVICE_CODE=ON \
      -DTHEROCK_GENERIC_BUILD_DIR=build-generic

ninja -C build-gfx120X
```

**Outputs** (per family):

- Fat binaries with `.hip_fatbin` sections (device code for family)
- Architecture-specific kernel libraries (rocFFT kernels for gfx110X)

**Build Matrix**:

```
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Generic Build (Once)                           │
│  - Compiler (clang, hipcc)                              │
│  - System libraries (host-only)                         │
│  - Headers, CMake configs                               │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ├─────────────────┬─────────────────┬──────
                 ▼                 ▼                 ▼
┌────────────────────────┐ ┌──────────────┐ ┌──────────────┐
│ Phase 2a: gfx1100      │ │ Phase 2b:    │ │ Phase 2c:    │
│  - gfx1100/1101/1102   │ │ gfx1200/1201 │ │ gfx900/906   │
│  - Device code only    │ │ Device code  │ │ Device code  │
└────────────────────────┘ └──────────────┘ └──────────────┘
```

**Benefit**: Generic build runs once instead of N times (where N = number of architecture families). For a typical build with 5 families, this eliminates significant redundant compilation and storage/transfer.

#### 4.1.3 Pre-Packaging Phase (Kpack Merge)

After the build matrix completes, a pre-packaging phase merges generic and architecture-specific artifacts using the kpack tooling.

**Input**:

- Generic artifacts from Phase 1 (host-only binaries, libraries)
- Architecture-specific artifacts from Phase 2 (fat binaries, kernel databases)

**Process**:

```bash
# Run kpack tooling on each architecture family
python -m rocm_kpack.tools.pack_tree \
    --input build-gfx110X/dist/rocm \
    --output kpack-artifacts/gfx110X \
    --group-name rocm \
    --gfx-arch-family gfx110X \
    --gfx-arches gfx1100,gfx1101,gfx1102

# Repeat for other families (gfx1200, gfx900, etc.)
```

**For each fat binary**:

1. Extract device code using `clang-offload-bundler`
1. Add device code to `.kpack/{group_name}-{gfx_arch_family}.kpack`
1. Convert binary to host-only with `.rocm_kpack_ref` marker
1. Zero-page `.hip_fatbin` section to reclaim disk space

**Output Structure** (per architecture family):

```
kpack-artifacts/gfx110X/
├── bin/
│   ├── hipcc                   # Host-only (copied from generic build)
│   └── rocm-smi                # Host-only (copied from generic build)
├── lib/
│   ├── libamdhip64.so.6        # Host-only + .rocm_kpack_ref marker
│   ├── librocblas.so.4         # Host-only + .rocm_kpack_ref marker
│   └── libhsa-runtime64.so.1   # Host-only (copied from generic build)
└── .kpack/
    └── rocm-gfx110X.kpack      # Device code for gfx1100/1101/1102
```

**Key Difference from Today**: Architecture-specific artifacts are now available as host-only binaries with external device code, rather than as fat binaries. Generic artifacts remain unchanged.

The above is a strawman for illustration. In reality, these are configured via GitHub action pipelines and tooling ergonomics/configurations will be setup to avoid hard-coding.

#### 4.1.4 Final Packaging (Native Packages, Wheels, Tarballs)

The final packaging phase consumes merged artifacts from the pre-packaging phase and generates distribution packages.

**Native Packages (DEB, RPM)**:

Packages are generated for each architecture family and a generic base:

```bash
# Generic base package (no device code)
rocm-blas-8.0.0_amd64.deb
  - Contains: host-only binaries, system libraries

# Architecture-specific packages (device code)
rocm-blas-gfx110X_8.0.0_amd64.deb
  - Contains: .kpack/rocm-blas-gfx110X.kpack
  - Provides: rocm-blas-device-code

rocm-blas-gfx120X_8.0.0_amd64.deb
  - Contains: .kpack/rocm-blas-gfx120X.kpack
  - Provides: rocm-blas-device-code
```

Again, this is simplified for illustration. Details will be worked out by the implementors.

**Installation Workflow**:

1. Install base package: `apt install rocm-blas-8.0.0`
1. Auto-detect GPU: `amd-smi --showproductname` → "gfx1100"
1. Install matching device code: `apt install rocm-blas-gfx110X`

**Python Wheels**:

Wheels follow a similar pattern with platform tags:

```
# Generic wheel (host-only)
rocm_blas-8.0.0-py3-none-manylinux_2_28_x86_64.whl

# Architecture-specific wheels
rocm_blas_gfx110X-8.0.0-py3-none-manylinux_2_28_x86_64.whl
rocm_blas_gfx120X-8.0.0-py3-none-manylinux_2_28_x86_64.whl
```

**Tarball Installations**:

Tarballs remain monolithic for ease of deployment, but contain kpack'd artifacts:

```
rocm-8.0.0-linux.tar.gz
├── bin/
│   └── hipcc                   # Host-only
├── lib/
│   └── librocblas.so.8         # Host-only + marker
└── .kpack/
    ├── rocm-gfx110X.kpack
    ├── rocm-gfx120X.kpack
    └── rocm-gfx90X.kpack
```

#### 4.1.5 Architecture Grouping Strategy

The number of architectures packed into each `.kpack` file is determined empirically based on compression efficiency and deployment complexity.

**Grouping Principles**:

1. **Architecture Families**: Group related architectures that share instruction sets

   - Example: `gfx1100`, `gfx1101`, `gfx1102` → `rocm-gfx110X.kpack`
   - Rationale: Similar ISA, kernels compress well together, users rarely need all three

1. **Compression Analysis**: Measure overhead of grouping vs. separate archives

   - RDNA (gfx12xx): Likely compress together efficiently due to ISA similarity
   - CDNA (gfx9xx): May benefit from separate archives due to ISA divergence and different optimization approaches

1. **Deployment Granularity**: Balance package count vs. download size

   - Too fine-grained: 20+ packages, complex dependency management
   - Too coarse-grained: Users download unnecessary device code

#### 4.1.6 CI/CD Strategy

Build and packaging workflows are tailored to the needs of pre-submits, nightlies, and releases.

**Pre-Submits (Fast Signal)**:

Goal: Provide fast feedback to developers (~30 minutes total).

```yaml
matrix:
  gfx_arch_family:
    - gfx1100  # RDNA 3 (developer workstations)
    - gfx90a   # CDNA 2 (datacenter, MI200 series)

steps:
  - Generic build (once)
  - Device code build (gfx1100, gfx90a)
  - Pre-packaging (kpack merge)
  - Smoke tests (basic functionality)
```

**Architectures Chosen**:

- Representative of common development targets
- Fast compile times
- Sufficient coverage for regressions

**Tests Run**:

- Host code unit tests (on generic artifacts)
- Device code smoke tests (one kernel launch per arch)
- Kpack tooling validation (round-trip unbundle→pack→load)

**Nightlies (Full Matrix)**:

Goal: Comprehensive testing across all supported architectures.

```yaml
matrix:
  gfx_arch_family:
    - gfx900, gfx906, gfx908, gfx90a
    - gfx1030, gfx1100, gfx1101, gfx1102
    - gfx1200, gfx1201
    - gfx940, gfx941, gfx942

steps:
  - Generic build (once)
  - Device code builds (all families, parallel)
  - Pre-packaging (all families)
  - Full test suite (per family)
  - Package generation (DEB, RPM, wheels, tarballs)
```

______________________________________________________________________

### 4.2. PyTorch + WheelNext

PyTorch releases integrate kpack through a post-processing workflow that splits monolithic wheels into base (host-only) wheels plus architecture-specific device code wheels.

#### 4.2.1 Current PyTorch Build Model

Today, PyTorch wheels for ROCm contain fat binaries with embedded device code for all supported architectures. A single wheel includes device code for RDNA, CDNA, and all intermediate architectures.

**Current Release**:

```
torch-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
  - Size: ~3GB
  - Contains: Fat binaries with device code for all gfx targets
  - Installation: Users download full wheel regardless of GPU
```

#### 4.2.2 Fat Build with Architecture Variants

For PyTorch releases, we continue building fat binaries but may partition architectures for time/size balance.

**Build Options**:

**Option 1: Single Fat Build (All Architectures)**

```bash
# Build PyTorch with all ROCm architectures
export PYTORCH_ROCM_ARCH="gfx90a;gfx940;gfx941;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
python setup.py bdist_wheel

# Output: Single wheel with all device code
torch-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
```

**Option 2: Split RDNA/CDNA (Balanced)**

```bash
# Build RDNA variant
export PYTORCH_ROCM_ARCH="gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
python setup.py bdist_wheel --variant=rdna

# Build CDNA variant
export PYTORCH_ROCM_ARCH="gfx90a;gfx940;gfx941;gfx942"
python setup.py bdist_wheel --variant=cdna

# Outputs:
torch-2.5.0+rocm6.2_rdna-cp311-cp311-linux_x86_64.whl
torch-2.5.0+rocm6.2_cdna-cp311-cp311-linux_x86_64.whl
```

The split granularity (1 to N wheels) will be determined based on build time and wheel size constraints.

#### 4.2.3 Wheel Post-Processing with `split_python_wheels`

After fat wheels are built, a new `split_python_wheels` tool post-processes them to extract device code into separate wheels.

**Tool Invocation**:

```bash
# Process a fat PyTorch wheel
python -m rocm_kpack.tools.split_python_wheels \
    --input torch-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl \
    --output-dir dist/ \
    --group-name torch

# Outputs:
dist/torch-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl              # Base wheel (host-only)
dist/torch_device_gfx110X-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
dist/torch_device_gfx120X-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
dist/torch_device_gfx90X-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
```

**Processing Steps**:

1. **Extract Wheel Contents**: Unzip the input wheel to temporary directory
1. **Scan for Artifacts**: Identify bundled binaries and kernel databases
   - Fat binaries: `torch/lib/libtorch_cuda.so`, `torch/lib/libtorch_hip.so`
   - Kernel databases: aotriton, etc
1. **Run Kpack Tooling**: Use `pack_tree` logic to extract device code per architecture
1. **Create Base Wheel**:
   - Convert binaries to host-only with `.rocm_kpack_ref` markers
   - Remove kernel database directories
   - Repackage as base wheel
1. **Create Device Wheels**: For each architecture family:
   - Create new wheel with `.kpack/{group_name}-{arch_family}.kpack`
   - Include kernel database files for that architecture (if present)
   - Repackage as architecture-specific wheel

**Example Base Wheel Structure**:

```
torch-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
├── torch/
│   ├── __init__.py
│   ├── lib/
│   │   ├── libtorch_cuda.so        # Host-only + .rocm_kpack_ref
│   │   └── libtorch_hip.so         # Host-only + .rocm_kpack_ref
│   └── ...
└── torch-2.5.0.dist-info/
    └── METADATA
```

**Example Device Wheel Structure**:

```
torch_device_gfx110X-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
└── torch/
    └── .kpack/
        └── torch-gfx110X.kpack     # Device code for gfx1100/1101/1102
```

The above is a strawman for illustration. The actual wheel structure, naming conventions, and metadata will be determined during implementation.

#### 4.2.4 WheelNext Integration (Dynamic Architecture Detection)

WheelNext provides a mechanism for wheels to declare optional dependencies that are dynamically resolved based on the target platform.

**Metadata in Base Wheel** (`METADATA` or `pyproject.toml`):

```toml
[project]
name = "torch"
version = "2.5.0+rocm6.2"

[project.optional-dependencies]
# Architecture-specific device code wheels
device-gfx110X = ["torch-device-gfx110X==2.5.0+rocm6.2"]
device-gfx120X = ["torch-device-gfx120X==2.5.0+rocm6.2"]
device-gfx90X = ["torch-device-gfx90X==2.5.0+rocm6.2"]

[tool.wheelnext]
# Dynamic dependency resolution based on detected GPU
auto-detect-extras = true
detection-script = "torch._detect_rocm_arch:get_arch_family"
```

**Detection Script** (`torch/_detect_rocm_arch.py`):

```python
import subprocess


def get_arch_family():
    """Detect GPU architecture family and return corresponding extra."""
    try:
        # Query GPU using rocminfo or similar
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        )

        # Parse output to extract gfx arch (e.g., "gfx1100")
        for line in result.stdout.splitlines():
            if "Name:" in line and "gfx" in line:
                arch = line.split()[-1]  # Extract "gfx1100"

                # Map to family
                if arch.startswith("gfx110"):
                    return "device-gfx110X"
                elif arch.startswith("gfx120"):
                    return "device-gfx120X"
                elif arch.startswith("gfx90"):
                    return "device-gfx90X"
                # ... other families

    except Exception:
        pass

    return None  # No GPU detected or unsupported
```

#### 4.2.5 Distribution Strategy

**PyPI Distribution**:

Upload all wheels to PyPI:

```
torch-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl              (Base)
torch-device-gfx110X-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
torch-device-gfx120X-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
torch-device-gfx90X-2.5.0+rocm6.2-cp311-cp311-linux_x86_64.whl
```

**Custom PyPI Index** (for ROCm-specific wheels):

Host on custom index to avoid polluting main PyPI:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

______________________________________________________________________

## 5. Future Optimizations

This section outlines potential improvements beyond the initial implementation.

### Compiler-Emitted Kpack Artifacts

**Current State**: The offload_kpacker performs post-hoc transformation of fat binaries, extracting device code and repackaging it into kpack archives. This requires compiling to fat binary format first, then running a separate transformation pass.

**Future Opportunity**: Teach the compiler toolchain to directly emit kpack artifacts during compilation, eliminating the need for post-hoc transformation entirely.

**Approach**:

1. **Compiler Frontend Integration**: Extend clang/hipcc to accept `--emit-kpack` flag

   ```bash
   hipcc --offload-arch=gfx1100 --offload-arch=gfx1101 \
         --emit-kpack=output.kpack \
         --emit-host-only=output.exe \
         kernel.hip
   ```

1. **Direct Archive Generation**: During compilation, the compiler:

   - Generates device code per architecture (existing behavior)
   - Writes device code directly to kpack archive (new)
   - Emits host-only binary with `.rocm_kpack_ref` marker (new)
   - No intermediate fat binary created

1. **Build Integration Benefits**:

   - Faster builds (no post-processing pass)
   - Simpler toolchain (fewer tools in pipeline)
   - Earlier error detection (kpack issues caught at compile time)
   - Reduced disk I/O (no temporary fat binaries)

1. **Incremental Adoption**: Can coexist with post-hoc transformation

   - Legacy builds continue using offload_kpacker
   - New builds opt into `--emit-kpack` flag
   - Gradual migration over time

**Tradeoff**: Requires compiler toolchain changes, may complicate build system integration initially. Post-hoc transformation remains necessary for legacy binaries and third-party distributions where source builds are not available.

### Dictionary-Trained Zstd Compression

**Current State**: Each kernel compressed independently with zstd at level 3.

**Opportunity**: Kernels within the same library often share common instruction patterns, prologue/epilogue code, and metadata structures.

**Approach**:

1. **Training Phase**: Analyze all kernels in a library to build a shared dictionary

   ```bash
   zstd --train -r lib/rocblas/*.hsaco -o rocblas.dict
   ```

1. **Compression Phase**: Compress each kernel using the shared dictionary

   ```bash
   zstd -D rocblas.dict kernel.hsaco
   ```

1. **TOC Storage**: Embed dictionary in kpack header, reference it in per-kernel metadata

1. **Decompression**: Load dictionary once, reuse for all kernel decompression

**Tradeoff**: Dictionary overhead (~100KB) amortized across many kernels. Not beneficial for binaries with few kernels.

**Advanced Cross-Frame Compression Techniques**:

Beyond dictionary training, additional compression opportunities exist by exploiting structural similarities across kernels:

1. **Semantic Block Sorting**: Reorder instruction blocks within kernels to maximize similarity

   - Group similar code sequences (prologue, epilogue, loop bodies) across kernels
   - Sort blocks by semantic purpose before compression
   - Example: All kernels' register save/restore prologues grouped together
   - Enables better cross-frame pattern matching in zstd

1. **Cross-Frame Reference Compression**: Exploit similarities across multiple kernel frames

   - Instead of compressing each kernel independently, compress sequences of related kernels
   - Example: Compress all GEMM kernels for different tile sizes as a sequence
   - Allows zstd to reference patterns from previous frames
   - Particularly effective for parameterized kernel families (same logic, different constants)

1. **Metadata Separation**: Extract and compress metadata separately from code

   - ELF headers, symbol tables, and debug info often highly redundant across kernels
   - Separate metadata into dedicated compression stream
   - Code-only compression can use more aggressive settings

1. **Instruction-Level Deduplication**: Identify and deduplicate common instruction sequences

   - Many kernels share identical subroutines (e.g., reduction, transpose)
   - Extract common sequences to shared "subroutine pool"
   - Kernels reference pool entries instead of duplicating code
   - Requires more complex decompression logic

**Implementation Strategy**:

These advanced techniques should be pursued **after** establishing the basic apples-to-apples per-kernel compression scheme. The initial implementation provides a baseline for comparison and validates the kpack infrastructure before adding compression complexity.

**Phased Rollout**:

1. **Phase 1** (current): Per-kernel zstd compression (baseline)
1. **Phase 2**: Dictionary-trained compression
1. **Phase 3**: Semantic block sorting + cross-frame compression
1. **Phase 4**: Full instruction-level deduplication

**Note**: Separate analysis exists for detailed evaluation of these techniques. The basic per-kernel scheme establishes the foundation and measurement baseline before pursuing advanced optimizations.

### Streaming Write Mode

**Current State**: Kpack archives built in-memory, then written to disk.

**Limitation**: Large libraries (rocBLAS, hipBLASLt with 10K+ kernels) consume significant memory during pack operation.

**Approach**:

1. **Streaming API**:

   ```python
   with KpackWriter(path, compression="zstd") as writer:
       for kernel in kernels:
           writer.add_kernel(name, arch, kernel_data)
   ```

1. **Implementation**:

   - Write header and blob data immediately
   - Accumulate TOC in memory (small compared to blobs)
   - Write TOC at finalization
   - Update header's TOC offset

1. **Memory Savings**: O(num_kernels) metadata vs. O(total_kernel_bytes) blob data

**Expected Benefit**: Enable packing of libraries with 100K+ kernels without memory constraints.

### Cross-Kernel Deduplication

**Observation**: Identical kernels may appear multiple times across different libraries or binaries.

**Approach**:

1. **Content-Addressing**: Hash each kernel (SHA256 or xxHash)

1. **Deduplication**: Store each unique kernel once, reference by hash

1. **TOC Structure**:

   ```python
   "toc": {
       "bin/hipcc/gfx1100": {"hash": "abc123...", "ordinal": 5},
       "lib/libhip.so/gfx1100": {"hash": "abc123...", "ordinal": 5},
   }
   ```

1. **Blob Storage**: Single copy at ordinal 5

**Tradeoff**: Increased TOC complexity, hash computation overhead.

### Kernel TOC Indexing Optimizations

**Current State**: TOC is flat MessagePack dictionary requiring full deserialization.

**Limitation**: Large TOC (100K+ kernels) slows down kpack_open() and initial queries.

**Approach**:

1. **Hierarchical TOC**:

   ```python
   {
       "binary_index": {
           "bin/hipcc": {offset: 1024, size: 256},
           "lib/libhip.so": {offset: 1280, size: 512},
       }
   }
   ```

1. **Lazy Deserialization**: Parse only relevant binary's TOC section

1. **Separate Index File**: Optional `.kpack.idx` sidecar for faster lookup

**Expected Benefit**: O(1) vs. O(num_binaries) TOC access for kpack with many binaries.

### Lazy TOC Loading

**Current State**: Entire TOC loaded into memory at kpack_open().

**Limitation**: Large TOC consumes memory even if only a few kernels accessed.

**Approach**:

1. **TOC Chunking**: Split TOC into chunks by binary or architecture
1. **Memory Mapping**: mmap() the kpack file, access TOC chunks on-demand
1. **LRU Cache**: Keep recently accessed TOC chunks in memory

**Expected Benefit**: Constant memory usage regardless of kpack size.

**Tradeoff**: Page fault overhead on first access to each chunk.

### mmap-Based Kernel Access

**Current State**: Kernels loaded into heap memory via read() and decompression.

**Opportunity**: For uncompressed kpacks, use mmap() for zero-copy access.

**Approach**:

1. **Check Compression**: If compression="none", use mmap() path
1. **mmap() kpack file**: Map entire file or kernel region
1. **Return pointer**: kpack_get_kernel() returns pointer into mmap'd region
1. **munmap() on close**: Clean up at kpack_close()

**Expected Benefit**: Eliminate memory copy for uncompressed kernels, faster load times.

**Tradeoff**: Only applicable to uncompressed kpacks (larger disk size).

### WheelNext Integration

**Context**: WheelNext is a proposed Python wheel format extension for platform-specific dependencies.

**Opportunity**: Distribute PyTorch wheels with kpack'd device code, download only target architecture.

**Approach**:

1. **Wheel Structure**:

   ```
   torch-2.0.0-py3-none-manylinux_2_17_x86_64.whl
   ├── torch/
   │   ├── lib/
   │   │   ├── libtorch.so (host-only + kpack markers)
   │   │   └── libtorch_cuda.so (host-only + kpack markers)
   │   └── .kpack/
   │       └── torch-{arch}.kpack (optional, downloaded per-arch)
   ```

1. **Installation Flow**:

   - `pip install torch`: Downloads base wheel (no device code)
   - Auto-detect GPU: `python -m torch.utils.hipconfig --arch`
   - Download arch-specific kpack: `pip install torch[gfx1100]`

1. **WheelNext Extension**:

   ```json
   {
       "extras_require": {
           "gfx1100": ["torch-gfx1100-kpack"],
           "gfx90a": ["torch-gfx90a-kpack"]
       }
   }
   ```

**Tradeoff**: Requires WheelNext adoption, pip tooling changes, mirror infrastructure.

______________________________________________________________________

## Appendix: File Format Specifications

### Kpack Archive Binary Format (Version 1)

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0x00   | 4    | magic           | "KPAK" (0x4B50414B)
0x04   | 4    | version         | uint32, currently 1
0x08   | 8    | toc_offset      | uint64, file offset to TOC
0x10   | 48   | padding         | Reserved (must be zero)
0x40   | var  | blob_data       | Compressed kernel data
EOF-N  | N    | toc             | MessagePack-encoded TOC
```

### MessagePack Metadata Format (`.rocm_kpack_ref`)

```
{
    "kernel_name": string,         # Path for TOC lookup
    "kpack_search_paths": [        # Ordered list of kpack locations
        string,                    # Relative or absolute paths
        ...
    ]
}
```

### ELF Section Layout (After Conversion)

```
Section Headers:
  [Nr] Name              Type     Addr     Off    Size
  ...
  [23] .hip_fatbin       PROGBITS 12340000 123000 500000  (zero-paged)
  [24] .rocm_kpack_ref   PROGBITS 12850000 623000 000800  (marker)
  ...

Program Headers:
  Type   Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  ...
  LOAD   0x123000 0x12340000 0x12340000 0x00000 0x00000 R   0x1000  (zeroed)
  LOAD   0x623000 0x12850000 0x12850000 0x00800 0x00800 R   0x1000  (marker)
  ...
```

______________________________________________________________________

## References

- **ROCm Documentation**: https://rocm.docs.amd.com/
- **HIP Programming Guide**: https://rocm.docs.amd.com/projects/HIP/
- **clang-offload-bundler**: LLVM tool for bundling/unbundling device code
- **MessagePack Specification**: https://msgpack.org/
- **Zstd Compression**: https://github.com/facebook/zstd
- **PR #1755 (POC Implementation)**: https://github.com/ROCm/rocm-systems/pull/1755
