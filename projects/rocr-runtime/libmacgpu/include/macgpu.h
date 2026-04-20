/*
 * macgpu.h — Public C API for the macOS userspace GPU driver shim.
 *
 * libmacgpu is a thin IOKit client that talks to ROCmGPU.dext (a
 * DriverKit PCI extension). It is the macOS counterpart to libhsakmt
 * on Linux: ROCR's MacOsDriver links against this library to get
 * register-level access to the GPU without each source file having to
 * learn IOKit.
 *
 * Scope:
 *   - Device discovery and lifecycle (open / close)
 *   - Device info (vendor/device, BAR map, VRAM size)
 *   - MMIO read/write (32-bit, BAR-indexed)
 *   - BAR mapping into the client process address space
 *   - DMA buffer allocation (with IOMMU-translated physical addresses)
 *
 * Not yet implemented (follow-up commits):
 *   - PCI config read/write
 *   - FLR (function-level reset)
 *   - MSI enable / interrupt wait
 *   - Queue creation (MEC/SDMA rings) — these layer on top of AllocDMA +
 *     MMIO writes and live in MacOsDriver rather than here.
 *
 * ABI stability: the selector numbers baked into rocmgpu_shared.h are
 * part of the DEXT contract. Bumping them requires coordinated updates
 * in ROCmGPUShared.h (DEXT side) and rocmgpu_shared.h (this library).
 */

#ifndef MACGPU_H_
#define MACGPU_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

typedef int32_t macgpu_status_t;

enum {
    MACGPU_SUCCESS             = 0,
    // Generic failure — DEXT returned an unexpected kIOReturn_*.
    MACGPU_ERROR               = -1,
    // No ROCmGPU DEXT is registered with IOKit (dext not installed or
    // not yet matched by a PCIe device). Non-fatal; driver discovery
    // treats this as "no device."
    MACGPU_ERROR_NOT_FOUND     = -2,
    // DEXT is present but IOServiceOpen() was refused. Usually missing
    // entitlements or the user client class is not being vended.
    MACGPU_ERROR_ACCESS_DENIED = -3,
    // Caller passed a null handle or out-of-range argument.
    MACGPU_ERROR_INVALID_ARG   = -4,
    // DMA alloc failed (DART-mapped memory exhausted, or the DEXT
    // refused the size/flags combination).
    MACGPU_ERROR_OUT_OF_MEMORY = -5,
};

// Returns a short description string for a status code. Thread-safe,
// returned pointer is valid for the life of the process.
const char* macgpu_status_string(macgpu_status_t status);

// ---------------------------------------------------------------------------
// Opaque device handle
// ---------------------------------------------------------------------------

typedef struct macgpu_device macgpu_device_t;

// Find and open the first ROCmGPU DEXT user client. On success writes
// an owning handle to *out_dev and returns MACGPU_SUCCESS. Caller must
// call macgpu_close() to release. Thread-safe.
macgpu_status_t macgpu_open(macgpu_device_t** out_dev);

// Close an open device handle. Safe to call on NULL (no-op). Unmaps
// any BARs or DMA buffers the caller forgot to release.
void macgpu_close(macgpu_device_t* dev);

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

typedef struct macgpu_bar_info {
    uint64_t size;         // BAR size in bytes (0 = not present)
    uint8_t  memory_index; // DriverKit memory index
    uint8_t  type;         // 0=memory, 1=IO, 2=not present
    uint8_t  is_64bit;
    uint8_t  prefetchable;
} macgpu_bar_info_t;

typedef struct macgpu_device_info {
    uint16_t vendor_id;
    uint16_t device_id;
    uint16_t subsystem_vendor_id;
    uint16_t subsystem_device_id;
    uint8_t  revision_id;
    macgpu_bar_info_t bars[6];
    uint64_t vram_size;
} macgpu_device_info_t;

// Query device info. Populates *out_info on success.
macgpu_status_t macgpu_get_info(macgpu_device_t* dev,
                                macgpu_device_info_t* out_info);

// ---------------------------------------------------------------------------
// MMIO
// ---------------------------------------------------------------------------

// Read a 32-bit MMIO register. `bar_index` is the PCI BAR (0-5),
// `byte_offset` is the offset into that BAR.
macgpu_status_t macgpu_mmio_read32(macgpu_device_t* dev,
                                   uint32_t bar_index,
                                   uint64_t byte_offset,
                                   uint32_t* out_value);

// Write a 32-bit MMIO register.
macgpu_status_t macgpu_mmio_write32(macgpu_device_t* dev,
                                    uint32_t bar_index,
                                    uint64_t byte_offset,
                                    uint32_t value);

// ---------------------------------------------------------------------------
// BAR mapping
// ---------------------------------------------------------------------------

// Map a PCI BAR into this process's address space. On success writes
// the CPU virtual address to *out_addr and the mapping size to
// *out_size. The mapping persists until macgpu_unmap_bar() or
// macgpu_close(). Only one mapping per BAR is supported; subsequent
// calls for the same BAR return the existing mapping.
macgpu_status_t macgpu_map_bar(macgpu_device_t* dev,
                               uint32_t bar_index,
                               void** out_addr,
                               uint64_t* out_size);

macgpu_status_t macgpu_unmap_bar(macgpu_device_t* dev, uint32_t bar_index);

// ---------------------------------------------------------------------------
// DMA buffers
// ---------------------------------------------------------------------------

// Scatter-gather entry — IOMMU-translated bus addresses the GPU can
// read from / write to.
typedef struct macgpu_dma_segment {
    uint64_t address;
    uint64_t length;
} macgpu_dma_segment_t;

typedef struct macgpu_dma_buffer {
    uint64_t buffer_id;              // Opaque handle for free_dma
    uint64_t size;                   // Actual allocation size
    void*    cpu_addr;               // Mapped virtual address in this process
    uint32_t segment_count;
    macgpu_dma_segment_t segments[64];
} macgpu_dma_buffer_t;

// Allocate a DMA-capable buffer. `flags` is a bitwise OR of
// ROCmGPUDMAFlags (e.g. kROCmGPU_DMA_Contiguous). The returned
// buffer is mapped into this process — cpu_addr is immediately usable.
macgpu_status_t macgpu_alloc_dma(macgpu_device_t* dev,
                                 uint64_t size,
                                 uint32_t flags,
                                 macgpu_dma_buffer_t* out_buf);

// Release a DMA buffer. `buffer_id` must have come from macgpu_alloc_dma
// on the same device. No-op on 0.
macgpu_status_t macgpu_free_dma(macgpu_device_t* dev, uint64_t buffer_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACGPU_H_
