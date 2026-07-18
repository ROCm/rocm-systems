#include <hip/hip_runtime.h>

// Code-generation reproducers only: do not execute (both use address zero).

// Reduced from rocPRIM's RDNA4 atomic_store(__uint128_t *, __uint128_t).
// STORE_CNT completion does not release the gfx1250 XCNT source-register lock.
__global__ void rocprim_b128_store_unsafe() {
  asm volatile("v_mov_b64_e32 v[0:1], 0\n\t"
               "v_mov_b32_e32 v2, 0\n\t"
               "v_mov_b32_e32 v3, 0\n\t"
               "v_mov_b32_e32 v4, 0\n\t"
               "v_mov_b32_e32 v5, 0\n\t"
               "global_store_b128 v[0:1], v[2:5] off scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0x0\n\t"
               // Missing: s_wait_xcnt 0
               "v_mov_b32_e32 v2, 1"
               :
               :
               : "v0", "v1", "v2", "v3", "v4", "v5", "memory");
}

__global__ void rocprim_b128_store_fixed() {
  asm volatile("v_mov_b64_e32 v[0:1], 0\n\t"
               "v_mov_b32_e32 v2, 0\n\t"
               "v_mov_b32_e32 v3, 0\n\t"
               "v_mov_b32_e32 v4, 0\n\t"
               "v_mov_b32_e32 v5, 0\n\t"
               "global_store_b128 v[0:1], v[2:5] off scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0x0\n\t"
               "s_wait_xcnt 0x0\n\t"
               "v_mov_b32_e32 v2, 1"
               :
               :
               : "v0", "v1", "v2", "v3", "v4", "v5", "memory");
}
