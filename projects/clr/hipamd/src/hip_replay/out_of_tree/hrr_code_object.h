/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */
#pragma once

/* Standalone ELF/msgpack parser for AMD GPU code objects.
 * Extracts kernel argument metadata (pointer vs scalar) from
 * .note NT_AMDGPU_METADATA sections without any COMGR dependency.
 *
 * Used by the out-of-tree LD_PRELOAD interposer and Windows proxy DLL. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Argument kind from COMGR metadata value_kind */
typedef enum {
  HRR_ARG_VALUE          = 0,  /* by_value (scalar) */
  HRR_ARG_GLOBAL_BUFFER  = 1,  /* global_buffer (pointer) */
  HRR_ARG_HIDDEN         = 2,  /* hidden_* (runtime-injected) */
} hrr_arg_kind_t;

/* Parsed kernel argument descriptor */
typedef struct {
  hrr_arg_kind_t kind;
  uint16_t size;      /* byte size */
  uint16_t offset;    /* offset in kernarg segment */
} hrr_arg_desc_t;

/* Parsed kernel metadata */
typedef struct {
  char name[256];
  uint32_t num_args;
  hrr_arg_desc_t args[64];  /* max 64 args per kernel */
} hrr_kernel_meta_t;

/* Parse an ELF code object image and extract kernel metadata.
 * Returns number of kernels found (up to max_kernels).
 * Fills kernels[] array. Returns 0 on parse failure. */
int hrr_parse_code_object(const void* image, size_t image_size,
                          hrr_kernel_meta_t* kernels, int max_kernels);

/* Look up a kernel by name in a parsed metadata array.
 * Returns pointer to the matching entry or NULL. */
const hrr_kernel_meta_t* hrr_find_kernel(const hrr_kernel_meta_t* kernels,
                                         int num_kernels, const char* name);

#ifdef __cplusplus
}
#endif
