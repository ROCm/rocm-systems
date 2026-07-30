#include <hip/hip_runtime.h>

extern "C" {
__global__ void add_kernel(float* out, float* a, float* b) {
  size_t i = threadIdx.x;
  out[i] = a[i] + b[i];
}
__global__ void sub_kernel(float* out, float* a, float* b) {
  size_t i = threadIdx.x;
  out[i] = a[i] - b[i];
}
__global__ void mul_kernel(float* out, float* a, float* b) {
  size_t i = threadIdx.x;
  out[i] = a[i] * b[i];
}

// Globals exercised by hipLibraryGetGlobal / hipLibraryGetManaged tests in
// library_get_global.cc. Compiled offline (--cuda-device-only) so __managed__
// resolves via the auto-included __clang_hip_runtime_wrapper.h.
__device__ float d_var[32];
__managed__ float m_var[32];

__global__ void write_d_var() {
  size_t i = threadIdx.x;
  if (i < 32) d_var[i] = i + 1;
}
__global__ void read_d_var(float* out) {
  size_t i = threadIdx.x;
  if (i < 32) out[i] = d_var[i] + 1;
}
__global__ void read_modify_d_var(float* out) {
  size_t i = threadIdx.x;
  if (i < 32) {
    out[i] = d_var[i] + 1;
    d_var[i]++;
  }
}

__global__ void write_m_var() {
  size_t i = threadIdx.x;
  if (i < 32) m_var[i] = i + 1;
}
__global__ void read_m_var(float* out) {
  size_t i = threadIdx.x;
  if (i < 32) out[i] = m_var[i] + 1;
}
__global__ void read_modify_m_var(float* out) {
  size_t i = threadIdx.x;
  if (i < 32) {
    out[i] = m_var[i] + 1;
    m_var[i]++;
  }
}
}
