/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */
#pragma once

/* Shared trace writer for out-of-tree HRR recording.
 * Plain C, used by both LD_PRELOAD interposer and Windows proxy DLL. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the trace writer. Reads HRR_* env vars.
 * Returns 1 if recording is enabled, 0 otherwise. */
int hrr_writer_init(void);

/* Flush and close the trace writer. */
void hrr_writer_shutdown(void);

/* Is recording active? */
int hrr_writer_enabled(void);

/* Record a malloc event */
void hrr_record_malloc(const void* ptr, size_t size, unsigned int flags);

/* Record a free event */
void hrr_record_free(const void* ptr);

/* Record a memcpy event. For H2D (kind=1), src data is captured. */
void hrr_record_memcpy(void* dst, const void* src, size_t size,
                       unsigned int kind, const void* stream);

/* Record a memset event */
void hrr_record_memset(void* dst, int value, size_t size, const void* stream);

/* Record a module load event with code object image */
void hrr_record_module_load(void* module, const void* image, size_t image_size);

/* Record a module unload event */
void hrr_record_module_unload(void* module);

/* Record a kernel launch event.
 * kernel_name: mangled kernel function name
 * code_object_image/size: for arg introspection (can be NULL if unknown) */
void hrr_record_kernel_launch(const char* kernel_name,
                              const void* code_object_image,
                              size_t code_object_size,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              uint32_t bx, uint32_t by, uint32_t bz,
                              uint32_t shared_mem,
                              const void* stream,
                              void** kernel_args);

/* Record a device synchronize event */
void hrr_record_device_sync(void);

/* Record a stream synchronize event */
void hrr_record_stream_sync(const void* stream);

#ifdef __cplusplus
}
#endif
