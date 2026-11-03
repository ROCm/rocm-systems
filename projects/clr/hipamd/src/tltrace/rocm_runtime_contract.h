/* Local POC: runtime-contract section 11 structs (proposals/runtime-contract.md @ 98deff2).
 * Declaration order differs from the document so it compiles as C. */
#ifndef ROCM_RUNTIME_CONTRACT_H
#define ROCM_RUNTIME_CONTRACT_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROCM_RUNTIME_CONTRACT_VERSION 1
enum { ROCM_DISPATCH_COMPLETED = 0, ROCM_DISPATCH_DROPPED = 1 };            /* done status */
enum { ROCM_PACKETS_SERIALIZE = 0x1 };                                      /* packets.flags */
enum { ROCM_DISPATCH_INTERNAL = 0x1, ROCM_DISPATCH_GRID_UNKNOWN = 0x2 };    /* info.flags */
enum { ROCM_MEM_HOST_VISIBLE = 0x1, ROCM_MEM_UNCACHED = 0x2,                /* gpu_alloc_fn */
       ROCM_MEM_EXPORTABLE = 0x4, ROCM_MEM_DEVICE_LOCAL = 0x8 };
enum { ROCM_ARG_UNKNOWN = 0, ROCM_ARG_VALUE = 1, ROCM_ARG_POINTER = 2, ROCM_ARG_HIDDEN = 3 };

typedef struct { uint32_t offset, size, kind; } rocm_arg_desc_t;  /* bytes within kernarg */

typedef struct {                     /* written by the runtime for each call */
  uint32_t version, size;
  uint32_t gpu_id, flags;            /* KFD gpu_id; ROCM_DISPATCH_* */
  uint64_t ring_va, kernel_object;
  const char* kernel_name;  uint64_t code_object_id;    /* mangled name, or null */
  uint32_t grid[3], workgroup[3];    /* in work-items, as in the AQL packet */
  uint32_t private_segment_size, group_segment_size;
  uint32_t vgpr_count, sgpr_count, lds_size, scratch_size;
  const void* kernarg;  uint32_t kernarg_size;
  const rocm_arg_desc_t* args;  uint32_t arg_count;
  uint64_t stream;                   /* runtime's stream handle; 0 = default stream or unknown */
} rocm_dispatch_info_t;

typedef struct {                     /* zeroed by the runtime, version/size set; filled by the selector */
  uint32_t version, size;
  const void* pre;   uint32_t pre_count;   /* 64-byte AQL packets */
  const void* post;  uint32_t post_count;
  uint32_t flags;                          /* SERIALIZE */
  uint64_t* join_key;                      /* where to write the kernel's packet index */
  uint64_t cookie;                         /* returned in dispatch_done_fn (for example, a slot) */
} rocm_dispatch_packets_t;

typedef struct {                     /* hook table: owned by the runtime, filled by the selector */
  uint32_t version, size;
  void (*emit_dispatch_packets_fn)(const rocm_dispatch_info_t*, rocm_dispatch_packets_t* out);
  void (*dispatch_done_fn)(const uint64_t* cookies, uint32_t count, uint32_t status);
} rocm_runtime_hooks_t;

typedef struct {                     /* services table: owned and filled by the runtime */
  uint32_t version, size;
  void* (*gpu_alloc_fn)(uint32_t gpu_id, size_t size, uint32_t flags);
  void  (*gpu_free_fn)(void* ptr);
  int   (*gpu_export_fn)(void* ptr, int* fd, uint64_t* offset);   /* EXPORTABLE memory */
} rocm_runtime_services_t;

#ifdef __cplusplus
}
#endif
#endif
