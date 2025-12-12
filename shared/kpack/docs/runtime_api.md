# ROCm Kpack Runtime API

## Overview

The ROCm Kpack Runtime API provides a C library for loading and decompressing GPU kernels from `.kpack` archive files. This enables ROCm components (CLR runtime, kernel libraries) to dynamically load architecture-specific device code without embedding all variants in host binaries.

## Design Decisions

### Library Type

- **Static or shared library** via `BUILD_SHARED_LIBS` CMake option
- Default: Static library (`librocm_kpack.a`)
- Shared library (`librocm_kpack.so`) available with `-DBUILD_SHARED_LIBS=ON`
- C++17 implementation with C API for maximum compatibility

### Symbol Visibility

- **Hidden by default**: All symbols compiled with `-fvisibility=hidden` on POSIX
- **Explicit exports**: Only public API functions marked with `KPACK_API` are exported
- **Cross-platform**: Uses `__attribute__((visibility("default")))` on GCC/Clang,
  `__declspec(dllexport/dllimport)` on Windows
- **Shared library**: Exports only the 7 public API functions
- **Static library**: Compiled with hidden visibility for consistency across platforms

### Dependencies

- **msgpack-cxx**: Header-only MessagePack library for TOC parsing
- **libzstd**: Zstandard decompression library
- Both are external dependencies via `find_package()`

### Thread Safety

- Multiple threads can open different archives concurrently
- Multiple threads can query/load from same archive concurrently
- Internal locking protects shared state (decompression context, TOC cache)
- User must not call `kpack_close()` while other threads use the archive

### Error Handling

- **Fail-fast philosophy**: Return errors immediately, no silent failures
- **Error codes only**: Functions return descriptive error codes
- **No global state**: No thread-local or global error strings
- **OS error details**: Use `errno` (POSIX) or `GetLastError()` (Windows) for I/O error details
- No partial parsing or fallback behavior on corrupted data

## Kpack File Format

### Archive Structure

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

### Compression Schemes

**NoOp (baseline)**:

- Direct concatenation of `.hsaco` kernel files
- TOC stores byte offset and size for each kernel
- Used for testing and baseline comparison

**Zstd Per-Kernel (production)**:

- Each kernel compressed independently with zstd level 3
- Blob structure: `[num_kernels: uint32][frame_size: uint32][zstd_frame]...`
- TOC stores ordinal (frame index) for each kernel
- Enables O(1) random access: decompress only requested kernel

### TOC Format

MessagePack-encoded table of contents:

```python
{
    "format_version": 1,
    "group_name": "rocm",
    "gfx_arch_family": "gfx1100",
    "gfx_arches": ["gfx1100", "gfx1101", "gfx1102"],
    "compression_scheme": "zstd-per-kernel",
    "zstd_offset": 64,
    "zstd_size": 245628,
    "toc": {
        "lib/libamdhip64.so.6": {
            "gfx1100": {"type": "hsaco", "ordinal": 0, "original_size": 138456}
        }
    },
}
```

## API Reference

### Types and Constants

```c
// Opaque handle to an opened kpack archive
typedef struct kpack_archive* kpack_archive_t;

// Error codes
typedef enum {
    KPACK_SUCCESS = 0,
    KPACK_ERROR_INVALID_ARGUMENT,
    KPACK_ERROR_FILE_NOT_FOUND,
    KPACK_ERROR_INVALID_FORMAT,
    KPACK_ERROR_UNSUPPORTED_VERSION,
    KPACK_ERROR_KERNEL_NOT_FOUND,
    KPACK_ERROR_DECOMPRESSION_FAILED,
    KPACK_ERROR_OUT_OF_MEMORY,
    KPACK_ERROR_NOT_IMPLEMENTED,
} kpack_error_t;
```

### Archive Lifecycle

```c
// Open a kpack archive for reading
// Returns KPACK_SUCCESS on success, error code otherwise
kpack_error_t kpack_open(const char* path, kpack_archive_t* archive);

// Close archive and free all resources
// Must not be called while other threads are using the archive
void kpack_close(kpack_archive_t archive);
```

### Discovery and Querying

```c
// Get list of architectures available in archive
// Caller must free the returned array with kpack_free_string_array()
kpack_error_t kpack_get_architectures(
    kpack_archive_t archive,
    char*** arches,
    size_t* count
);

// Get list of binary names that have kernels
// Caller must free the returned array with kpack_free_string_array()
kpack_error_t kpack_get_binaries(
    kpack_archive_t archive,
    char*** binaries,
    size_t* count
);

// Free string array returned by query functions
void kpack_free_string_array(char** array, size_t count);
```

### Kernel Loading

```c
// Load kernel for specific binary and architecture
// Returns pointer to decompressed kernel data
// Caller must free with kpack_free_kernel()
kpack_error_t kpack_get_kernel(
    kpack_archive_t archive,
    const char* binary_name,
    const char* arch,
    void** kernel_data,
    size_t* kernel_size
);

// Free kernel data returned by kpack_get_kernel()
void kpack_free_kernel(kpack_archive_t archive, void* kernel_data);
```

All functions that can fail return a `kpack_error_t` code. For I/O errors (`KPACK_ERROR_IO_ERROR`), check `errno` on POSIX or `GetLastError()` on Windows for OS-level error details.

## Integration Points

### CLR Runtime Integration

The CLR (HIP runtime) needs to detect kpack references and lazy-load kernels:

1. **Detect HIPK magic** in `__hipRegisterFatBinary()`:

   - Check for `HIPK` magic (vs. `HIPF` for fat binaries)
   - Parse `.rocm_kpack_ref` ELF section for kpack path

1. **Lazy loading** in `StatCO::digestFatBinary()`:

   - Query current GPU architecture
   - Implement fallback logic: `gfx1101` → `gfx1100` → `gfx11-generic`
   - Call `kpack_get_kernel()` to load device code
   - Cache loaded kernels to avoid repeated decompression

### Kernel Library Integration

Libraries like rocBLAS, rocFFT, hipBLASLt:

1. **Library initialization**:

   - Open kpack archive from known location
   - Query available architectures
   - Select best match for current GPU

1. **Kernel selection**:

   - Map problem parameters to TOC key
   - Load kernel on-demand with `kpack_get_kernel()`
   - Cache frequently-used kernels

1. **Cleanup**:

   - Free kernels with `kpack_free_kernel()`
   - Close archive with `kpack_close()` on library shutdown

## Implementation Status

### Phase 1: Scaffolding (Current)

- [x] API design documented
- [x] Directory structure created
- [ ] CMake build system
- [ ] Header files with API declarations
- [ ] Stub implementations

### Phase 2: Core Functionality

- [ ] Archive opening and TOC parsing
- [ ] NoOp decompression (uncompressed archives)
- [ ] Zstd per-kernel decompression
- [ ] Error handling and reporting

### Phase 3: Advanced Features

- [ ] Thread-safe operations
- [ ] Kernel caching
- [ ] Performance optimization
- [ ] Integration tests with real kpack files

## Build Instructions

```bash
# Configure with default static library
cmake -B build -S /develop/rocm-kpack

# Or configure with shared library
cmake -B build -S /develop/rocm-kpack -DBUILD_SHARED_LIBS=ON

# Build
cmake --build build

# Run tests
ctest --test-dir build
```

## Testing

Unit tests verify:

- Archive opening with valid/invalid files
- TOC parsing with various formats
- Decompression correctness
- Error handling for all failure modes
- Thread safety under concurrent access

Integration tests use real kpack files from `test_assets/` created by the Python implementation.

## References

- Python reference implementation: `python/rocm_kpack/kpack.py`
- Kpack format specification: See TOC structure above
- MessagePack spec: https://msgpack.org/
- Zstandard spec: https://github.com/facebook/zstd
