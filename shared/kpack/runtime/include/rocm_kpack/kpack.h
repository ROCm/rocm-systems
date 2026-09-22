// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_KPACK_H
#define ROCM_KPACK_H

#include "kpack_export.h"
#include "kpack_types.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Archive Lifecycle
//

// Open a kpack archive for reading
//
// Opens the kpack file at the specified path, validates the header,
// and parses the table of contents (TOC).
//
// Args:
//   path: Path to the .kpack file
//   archive: Output parameter for archive handle
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if path is NULL or archive is NULL
//   KPACK_ERROR_FILE_NOT_FOUND if file doesn't exist
//   KPACK_ERROR_INVALID_FORMAT if file is not a valid kpack
//   KPACK_ERROR_UNSUPPORTED_VERSION if version is not supported
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
//   KPACK_ERROR_IO_ERROR on read failures
//   KPACK_ERROR_MSGPACK_PARSE_FAILED if TOC parsing fails
KPACK_API kpack_error_t kpack_open(const char* path, kpack_archive_t* archive);

// Close archive and free all resources
//
// Must not be called while other threads are using the archive.
// After this call, the archive handle is invalid.
//
// Args:
//   archive: Archive handle from kpack_open()
KPACK_API void kpack_close(kpack_archive_t archive);

//
// Discovery and Querying
//
// All query functions return pointers to memory owned by the archive.
// Returned pointers are valid until kpack_close() is called.
//

// Get number of architectures available in the archive
//
// Args:
//   archive: Archive handle
//   count: Output parameter for number of architectures
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
KPACK_API kpack_error_t kpack_get_architecture_count(kpack_archive_t archive,
                                                     size_t* count);

// Get architecture name by index
//
// Returns a pointer to an architecture string owned by the archive.
// The pointer is valid until kpack_close() is called.
//
// Args:
//   archive: Archive handle
//   index: Index in range [0, count)
//   arch: Output parameter for architecture string
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if pointers are NULL or index out of range
KPACK_API kpack_error_t kpack_get_architecture(kpack_archive_t archive,
                                               size_t index, const char** arch);

// Get number of binaries that have kernels in the archive
//
// Args:
//   archive: Archive handle
//   count: Output parameter for number of binaries
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
KPACK_API kpack_error_t kpack_get_binary_count(kpack_archive_t archive,
                                               size_t* count);

// Get binary name by index
//
// Returns a pointer to a binary path string owned by the archive.
// The pointer is valid until kpack_close() is called.
//
// Args:
//   archive: Archive handle
//   index: Index in range [0, count)
//   binary: Output parameter for binary path string
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if pointers are NULL or index out of range
KPACK_API kpack_error_t kpack_get_binary(kpack_archive_t archive, size_t index,
                                         const char** binary);

//
// Kernel Loading
//

// Load kernel for specific binary and architecture
//
// Decompresses and returns an allocated copy of the kernel data for the
// specified binary and architecture.
//
// Memory Management:
// - The returned pointer MUST be freed by the caller using kpack_free_kernel()
// - The pointer remains valid until freed, regardless of other operations
//   on the archive
//
// Thread Safety:
// - Thread-safe when called concurrently on the SAME archive handle
// - Multiple threads can safely call this function simultaneously
//
// Args:
//   archive: Archive handle
//   binary_name: Binary path (e.g., "lib/libamdhip64.so.6")
//   arch: Architecture name (e.g., "gfx942")
//   kernel_data: Output parameter for pointer to allocated kernel bytes
//   kernel_size: Output parameter for kernel size in bytes
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
//   KPACK_ERROR_KERNEL_NOT_FOUND if kernel doesn't exist
//   KPACK_ERROR_DECOMPRESSION_FAILED if decompression fails
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
//   KPACK_ERROR_IO_ERROR if file read fails
KPACK_API kpack_error_t kpack_get_kernel(kpack_archive_t archive,
                                         const char* binary_name,
                                         const char* arch, void** kernel_data,
                                         size_t* kernel_size);

// Free kernel data allocated by kpack_get_kernel
//
// Args:
//   kernel_data: Pointer returned by kpack_get_kernel, or NULL (no-op)
KPACK_API void kpack_free_kernel(void* kernel_data);

//
// Loader API
//
// These functions provide code object loading for runtime integration.
// They handle metadata parsing, path resolution, and archive management.
//

// Callback type for architecture enumeration
// Return true to continue enumeration, false to stop
typedef bool (*kpack_arch_callback_t)(const char* arch, void* user_data);

//
// Cache Lifecycle
//
// The cache provides thread-safe, high-performance code object loading.
// Environment variables are resolved once at cache creation time.
// Archives are opened lazily and kept open until cache destruction.
//

// Create a kpack cache
//
// The cache resolves environment variables at creation time:
// - ROCM_KPACK_PATH: Override search paths entirely
// - ROCM_KPACK_PATH_PREFIX: Prepend paths to search
// - ROCM_KPACK_DISABLE: If set, all load calls return NOT_IMPLEMENTED
// - ROCM_KPACK_DEBUG: Enable verbose logging to stderr
//
// Thread Safety:
// - This function is NOT thread-safe. Create cache before spawning threads.
// - The returned cache IS thread-safe for concurrent load operations.
//
// Args:
//   cache: Output parameter for cache handle
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if cache is NULL
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
KPACK_API kpack_error_t kpack_cache_create(kpack_cache_t* cache);

// Destroy a kpack cache and release all resources
//
// Closes all cached archives and frees memory.
// Must not be called while other threads are using the cache.
//
// Args:
//   cache: Cache handle from kpack_cache_create(), or NULL (no-op)
KPACK_API void kpack_cache_destroy(kpack_cache_t cache);

// Discover the file path of a binary from an address within it
//
// Given a pointer to any address within a loaded shared library or executable,
// returns the file path of that binary. This is used to resolve relative paths
// in kpack metadata.
//
// Platform-specific implementation:
// - Linux: Parses /proc/self/maps (dladdr cannot reliably resolve data
// segments)
// - Windows: Uses GetModuleHandleEx + GetModuleFileName
//
// Args:
//   address_in_binary: Any address within the target binary (e.g., from
//                      __CudaFatBinaryWrapper.binary for HIPK binaries)
//   path_out: Buffer to receive the file path
//   path_out_size: Size of path_out buffer
//   offset_out: Optional output for offset of address within the file (may be
//               NULL)
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if address_in_binary or path_out is NULL
//   KPACK_ERROR_PATH_DISCOVERY_FAILED if address is not in any known mapping
//   KPACK_ERROR_NOT_IMPLEMENTED on unsupported platforms
KPACK_API kpack_error_t
kpack_discover_binary_path(const void* address_in_binary, char* path_out,
                           size_t path_out_size, size_t* offset_out);

// Load a code object from kpack archives using HIPK metadata
//
// Parses HIPK msgpack metadata, locates the appropriate kpack archive(s),
// and returns the first matching code object for the given architecture list.
//
// Search Algorithm:
// For each architecture in priority order, searches all archives in the path
// list until a match is found. This ensures the highest-priority architecture
// is used even if it's only available in a later archive.
//
// Multi-TU Support:
// Libraries built with -fgpu-rdc have multiple __CudaFatBinaryWrapper
// structures, each corresponding to a different translation unit (TU). The
// co_index parameter identifies which code object to load (0-based). At build
// time, the kpack tooling assigns each code object an index and stores this
// in the wrapper's reserved1 field. At runtime, CLR passes this index through.
//
// Thread Safety:
// - Thread-safe when called with the same cache from multiple threads
// - Archives are cached and reused across calls
//
// Args:
//   cache: Cache handle from kpack_cache_create()
//   hipk_metadata: Pointer to msgpack-encoded HIPK metadata (from
//                  __CudaFatBinaryWrapper.binary when magic is HIPK)
//   binary_path: Path to the binary containing the HIPK metadata (used to
//                resolve relative kpack paths)
//   co_index: Code object index for multi-TU binaries (from wrapper reserved1
//             field). Use 0 for single-TU binaries.
//   arch_list: Array of architecture strings in priority order
//              (e.g., ["gfx942:xnack+", "gfx9-4-generic:xnack+"])
//   arch_count: Number of entries in arch_list
//   code_object_out: Output pointer to allocated code object bytes (caller
//                    must free via kpack_free_code_object)
//   code_object_size_out: Output for size of code object in bytes
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if cache or required pointers are NULL,
//                                or arch_count is 0
//   KPACK_ERROR_INVALID_METADATA if hipk_metadata is not valid msgpack
//   KPACK_ERROR_ARCHIVE_NOT_FOUND if no archive found at any search path
//   KPACK_ERROR_ARCH_NOT_FOUND if no architecture in arch_list found in
//                              any archive
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
//   KPACK_ERROR_NOT_IMPLEMENTED if ROCM_KPACK_DISABLE was set at cache creation
KPACK_API kpack_error_t kpack_load_code_object(
    kpack_cache_t cache, const void* hipk_metadata, const char* binary_path,
    uint32_t co_index, const char* const* arch_list, size_t arch_count,
    void** code_object_out, size_t* code_object_size_out);

// Free a code object allocated by kpack_load_code_object
//
// Args:
//   code_object: Pointer returned by kpack_load_code_object, or NULL (no-op)
KPACK_API void kpack_free_code_object(void* code_object);

// Enumerate architectures available in a kpack archive
//
// Opens the archive, invokes the callback for each architecture, then closes
// the archive. If the callback returns false, enumeration stops early.
//
// Args:
//   archive_path: Path to the kpack archive file
//   callback: Function called for each architecture (receives arch string and
//             user_data)
//   user_data: Opaque pointer passed to callback
//
// Returns:
//   KPACK_SUCCESS on success (including early termination via callback)
//   KPACK_ERROR_INVALID_ARGUMENT if archive_path or callback is NULL
//   KPACK_ERROR_FILE_NOT_FOUND if archive doesn't exist
//   Other errors from kpack_open
KPACK_API kpack_error_t kpack_enumerate_architectures(
    const char* archive_path, kpack_arch_callback_t callback, void* user_data);

#ifdef __cplusplus
}
#endif

#endif  // ROCM_KPACK_H
