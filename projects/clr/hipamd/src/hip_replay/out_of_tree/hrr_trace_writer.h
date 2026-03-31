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

/* Record a kernel launch using the packed kernarg buffer (hipExtModuleLaunchKernel
 * 'extra' path). Uses arg offsets from parsed code object metadata to extract
 * each argument from the flat kernarg buffer. */
void hrr_record_kernel_launch_packed(const char* kernel_name,
                                      uint32_t gx, uint32_t gy, uint32_t gz,
                                      uint32_t bx, uint32_t by, uint32_t bz,
                                      uint32_t shared_mem,
                                      const void* stream,
                                      const void* packed_buf,
                                      size_t packed_size);

/* Record a device synchronize event */
void hrr_record_device_sync(void);

/* Record a stream synchronize event */
void hrr_record_stream_sync(const void* stream);

/* Function handle → kernel name registry.
 * Called from hipModuleGetFunction hook to associate the returned handle with
 * the kernel name. Used by hipExtModuleLaunchKernel to recover the name. */
void hrr_register_function(const void* func_handle, const char* kernel_name);

/* Look up a kernel name by its hipFunction_t handle.
 * Returns the registered name, or NULL if unknown. */
const char* hrr_lookup_function_name(const void* func_handle);

#ifdef __cplusplus
}
#endif
