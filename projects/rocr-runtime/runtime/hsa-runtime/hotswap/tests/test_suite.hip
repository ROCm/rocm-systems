#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    printf("HIP error %d (%s) at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); \
    return 1; \
  } \
} while(0)

// ── original 7 kernels ──────────────────────────────────────────────────────

extern "C" __global__ void vector_add(float* out, const float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void saxpy(float* y, const float* x, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = alpha * x[i] + y[i];
}

extern "C" __global__ void relu(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

extern "C" __global__ void reduce_sum(float* out, const float* in, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

extern "C" __global__ void dot_product(float* out, const float* a, const float* b, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (i < n) ? a[i] * b[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

extern "C" __global__ void matrix_add_2d(float* out, const float* a, const float* b,
                                          int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row < rows && col < cols) {
    int i = row * cols + col;
    out[i] = a[i] + b[i];
  }
}

extern "C" __global__ void clamp_kernel(float* out, const float* in, float lo, float hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = in[i];
    out[i] = v < lo ? lo : (v > hi ? hi : v);
  }
}

// ── new 4 kernels ───────────────────────────────────────────────────────────

extern "C" __global__ void stencil_1d(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int im1 = (i > 0) ? i - 1 : 0;
    int ip1 = (i < n - 1) ? i + 1 : n - 1;
    out[i] = (in[im1] + 2.0f * in[i] + in[ip1]) * 0.25f;
  }
}

extern "C" __global__ void max_reduce(float* out, const float* in, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (i < n) ? in[i] : -1e30f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

extern "C" __global__ void sqrt_vals(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sqrtf(in[i]);
}

extern "C" __global__ void int_elementwise(int* out, const int* a, const int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * 2 + b[i];
}

// ── test functions ──────────────────────────────────────────────────────────

int run_test_vector_add(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_a = (float*)malloc(N*4), *h_b = (float*)malloc(N*4), *h_o = (float*)calloc(N,4);
  for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = i*2; }
  float *d_a, *d_b, *d_o;
  CHECK(hipMalloc(&d_a, N*4)); CHECK(hipMalloc(&d_b, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_a, h_a, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "vector_add"));
  int n=N; void* args[] = {&d_o, &d_a, &d_b, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) if(h_o[i]==h_a[i]+h_b[i]) ok++;
  printf("  vector_add:       %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a);hipFree(d_b);hipFree(d_o);hipModuleUnload(m);free(h_a);free(h_b);free(h_o);
  return ok==N?0:1;
}

int run_test_saxpy(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float alpha = 2.5f;
  float *h_x = (float*)malloc(N*4), *h_y = (float*)malloc(N*4), *h_o = (float*)calloc(N,4);
  for (int i = 0; i < N; i++) { h_x[i] = i+1; h_y[i] = (i+1)*0.5f; }
  float *d_x, *d_y;
  CHECK(hipMalloc(&d_x, N*4)); CHECK(hipMalloc(&d_y, N*4));
  CHECK(hipMemcpy(d_x, h_x, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_y, h_y, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "saxpy"));
  int n=N; void* args[] = {&d_y, &d_x, &alpha, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_y, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp = alpha * h_x[i] + (i+1)*0.5f;
    if(fabsf(h_o[i]-exp) < 0.01f) ok++;
  }
  printf("  saxpy:            %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_x);hipFree(d_y);hipModuleUnload(m);free(h_x);free(h_y);free(h_o);
  return ok==N?0:1;
}

int run_test_relu(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N,4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 128);
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, N*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "relu"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp = h_in[i] > 0 ? h_in[i] : 0;
    if(h_o[i]==exp) ok++;
  }
  printf("  relu:             %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return ok==N?0:1;
}

int run_test_reduce_sum(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(nb,4);
  float expected = 0;
  for (int i = 0; i < N; i++) { h_in[i] = i+1; expected += h_in[i]; }
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, nb*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, nb*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "reduce_sum"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, bs*4,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, nb*4, hipMemcpyDeviceToHost));
  float total = 0; for(int b=0;b<nb;b++) total += h_o[b];
  printf("  reduce_sum:       %.0f/%.0f %s\n", total, expected, total==expected?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return total==expected?0:1;
}

int run_test_dot_product(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_a = (float*)malloc(N*4), *h_b = (float*)malloc(N*4), *h_o = (float*)calloc(nb,4);
  float expected = 0;
  for (int i = 0; i < N; i++) { h_a[i] = i+1; h_b[i] = 1.0f; expected += h_a[i]; }
  float *d_a, *d_b, *d_o;
  CHECK(hipMalloc(&d_a, N*4)); CHECK(hipMalloc(&d_b, N*4)); CHECK(hipMalloc(&d_o, nb*4));
  CHECK(hipMemcpy(d_a, h_a, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, nb*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "dot_product"));
  int n=N; void* args[] = {&d_o, &d_a, &d_b, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, bs*4,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, nb*4, hipMemcpyDeviceToHost));
  float total = 0; for(int b=0;b<nb;b++) total += h_o[b];
  printf("  dot_product:      %.0f/%.0f %s\n", total, expected, fabsf(total-expected)<1.0f?"PASS":"FAIL");
  hipFree(d_a);hipFree(d_b);hipFree(d_o);hipModuleUnload(m);free(h_a);free(h_b);free(h_o);
  return fabsf(total-expected)<1.0f?0:1;
}

int run_test_matrix_add_2d(const char* co) {
  int rows = 16, cols = 32, N = rows * cols;
  int bsx = 32, nbx = (cols + bsx - 1) / bsx, nby = rows;
  float *h_a = (float*)malloc(N*4), *h_b = (float*)malloc(N*4), *h_o = (float*)calloc(N,4);
  for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = N - i; }
  float *d_a, *d_b, *d_o;
  CHECK(hipMalloc(&d_a, N*4)); CHECK(hipMalloc(&d_b, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_a, h_a, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, N*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "matrix_add_2d"));
  void* args[] = {&d_o, &d_a, &d_b, &rows, &cols};
  CHECK(hipModuleLaunchKernel(f, nbx,nby,1, bsx,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) if(h_o[i]==h_a[i]+h_b[i]) ok++;
  printf("  matrix_add_2d:    %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a);hipFree(d_b);hipFree(d_o);hipModuleUnload(m);free(h_a);free(h_b);free(h_o);
  return ok==N?0:1;
}

int run_test_clamp(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float lo = -50.0f, hi = 50.0f;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N,4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 128);
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, N*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "clamp_kernel"));
  int n=N; void* args[] = {&d_o, &d_in, &lo, &hi, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp = h_in[i] < lo ? lo : (h_in[i] > hi ? hi : h_in[i]);
    if(fabsf(h_o[i]-exp) < 1e-5f) ok++;
  }
  printf("  clamp:            %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return ok==N?0:1;
}

int run_test_stencil_1d(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N,4);
  for (int i = 0; i < N; i++) h_in[i] = (float)i;
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, N*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "stencil_1d"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    int im1 = (i > 0) ? i-1 : 0;
    int ip1 = (i < N-1) ? i+1 : N-1;
    float exp = (h_in[im1] + 2.0f*h_in[i] + h_in[ip1]) * 0.25f;
    if(fabsf(h_o[i]-exp) < 1e-4f) ok++;
  }
  printf("  stencil_1d:       %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return ok==N?0:1;
}

int run_test_max_reduce(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(nb,4);
  // Values with known per-block maxima
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 100);  // -100..155
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, nb*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, nb*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "max_reduce"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, bs*4,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, nb*4, hipMemcpyDeviceToHost));
  int ok=0;
  for(int b=0;b<nb;b++) {
    float exp = -1e30f;
    for(int t=0;t<bs;t++) { int idx=b*bs+t; if(idx<N) exp = fmaxf(exp, h_in[idx]); }
    if(fabsf(h_o[b]-exp) < 1e-4f) ok++;
  }
  printf("  max_reduce:       %d/%d %s\n", ok, nb, ok==nb?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return ok==nb?0:1;
}

int run_test_sqrt_vals(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N,4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);  // 1..256 (avoid sqrt(0))
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, N*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "sqrt_vals"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp = sqrtf(h_in[i]);
    if(fabsf(h_o[i]-exp) < 1e-4f) ok++;
  }
  printf("  sqrt_vals:        %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return ok==N?0:1;
}

int run_test_int_elementwise(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  int *h_a = (int*)malloc(N*4), *h_b = (int*)malloc(N*4), *h_o = (int*)calloc(N,4);
  for (int i = 0; i < N; i++) { h_a[i] = i+1; h_b[i] = N-i; }
  int *d_a, *d_b, *d_o;
  CHECK(hipMalloc(&d_a, N*4)); CHECK(hipMalloc(&d_b, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_a, h_a, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, N*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co)); CHECK(hipModuleGetFunction(&f, m, "int_elementwise"));
  int n=N; void* args[] = {&d_o, &d_a, &d_b, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) if(h_o[i] == h_a[i]*2 + h_b[i]) ok++;
  printf("  int_elementwise:  %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a);hipFree(d_b);hipFree(d_o);hipModuleUnload(m);free(h_a);free(h_b);free(h_o);
  return ok==N?0:1;
}

int main() {
  printf("=== gfx1250 → gfx942 Transpiler Test Suite ===\n\n");
  int fails = 0;
  fails += run_test_vector_add("/tmp/vector_add_gfx1250.elf");
  fails += run_test_saxpy("/tmp/saxpy_gfx1250.elf");
  fails += run_test_relu("/tmp/relu_gfx1250.elf");
  fails += run_test_reduce_sum("/tmp/reduce_sum_gfx1250.elf");
  fails += run_test_dot_product("/tmp/dot_product_gfx1250.elf");
  fails += run_test_matrix_add_2d("/tmp/matrix_add_2d_gfx1250.elf");
  fails += run_test_clamp("/tmp/clamp_kernel_gfx1250.elf");
  fails += run_test_stencil_1d("/tmp/stencil_1d_gfx1250.elf");
  fails += run_test_max_reduce("/tmp/max_reduce_gfx1250.elf");
  fails += run_test_sqrt_vals("/tmp/sqrt_vals_gfx1250.elf");
  fails += run_test_int_elementwise("/tmp/int_elementwise_gfx1250.elf");
  int total = 11;
  printf("\n%s: %d/%d tests passed\n", fails?"SOME FAILED":"ALL PASSED", total-fails, total);
  return fails;
}
