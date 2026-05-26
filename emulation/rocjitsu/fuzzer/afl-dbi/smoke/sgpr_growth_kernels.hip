#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ __launch_bounds__(64)
    __attribute__((amdgpu_num_sgpr(8), amdgpu_num_vgpr(16))) void
    sgpr_growth_kernel(const uint32_t *input, uint32_t *output,
                       uint32_t selector) {
  const uint32_t tid = threadIdx.x;
  uint32_t s0 = selector + 0x11u;
  uint32_t s1 = selector ^ 0x23u;
  uint32_t s2 = selector + 0x35u;
  uint32_t s3 = selector ^ 0x47u;

  asm volatile("" : "+s"(s0), "+s"(s1), "+s"(s2), "+s"(s3) : : "memory");

  uint32_t v0;
  uint32_t v1;
  uint32_t v2;
  uint32_t v3;
  uint32_t v4;
  uint32_t v5;
  uint32_t v6;
  uint32_t v7;
  // Keep the VGPR allocation above the probe's VGPR needs, but make these
  // temporaries dead before the branch probe point. Vector defs after the
  // branch are EXEC-masked in rocjitsu liveness and would remain conservatively
  // live at the probe.
  asm volatile("v_mov_b32 %0, %8\n"
               "v_mov_b32 %1, %8\n"
               "v_mov_b32 %2, %8\n"
               "v_mov_b32 %3, %8\n"
               "v_mov_b32 %4, %8\n"
               "v_mov_b32 %5, %8\n"
               "v_mov_b32 %6, %8\n"
               "v_mov_b32 %7, %8\n"
               : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3), "=v"(v4),
                 "=v"(v5), "=v"(v6), "=v"(v7)
               : "s"(selector)
               : "memory");
  asm volatile("" : : "v"(v0), "v"(v1), "v"(v2), "v"(v3), "v"(v4),
                    "v"(v5), "v"(v6), "v"(v7)
               : "memory");

  const uint32_t guard = s0 ^ s1 ^ s2 ^ s3;
  asm volatile("s_cmp_lg_u32 %0, 0\n"
               "s_cbranch_scc0 1f\n"
               "s_nop 0\n"
               "1:\n" : : "s"(guard) : "memory");

  uint32_t value = input[tid & 63u] + tid;
  volatile uint32_t *volatile_output = output;
  if (((guard ^ input[tid & 63u]) & 1u) != 0) {
    value += s0 + s2;
    volatile_output[tid + 64u] = value;
  } else {
    value ^= s1 + s3;
    volatile_output[tid + 128u] = value;
  }

  asm volatile("" : "+s"(s0), "+s"(s1), "+s"(s2), "+s"(s3) : : "memory");

  output[tid] = value + s0 + s1 + s2 + s3;
}
