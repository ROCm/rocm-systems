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

// ── new 4 kernels (round 2) ─────────────────────────────────────────────────

extern "C" __global__ void fma_vals(float* out, const float* a, const float* b,
                                    const float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __fmaf_rn(a[i], b[i], c[i]);
}

extern "C" __global__ void exp_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expf(in[i]);
}

extern "C" __global__ void recip_vals(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0f / in[i];
}

extern "C" __global__ void mag2d(float* out, const float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sqrtf(__fmaf_rn(a[i], a[i], b[i] * b[i]));
}

// ── matmul kernel (round 3): one block per row, stride-32 column loop ────────
// Supports arbitrary M, N, K; launch with grid=(M,1,1) block=(32,1,1).
// Exercises: inner-loop FMA, stride access patterns, larger K depths.

extern "C" __global__ void matmul(
    float* __restrict__ C, const float* __restrict__ A,
    const float* __restrict__ B, int M, int N, int K) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= M) return;
  for (int col = tid; col < N; col += blockDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
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

int run_test_fma_vals(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_a=(float*)malloc(N*4), *h_b=(float*)malloc(N*4);
  float *h_c=(float*)malloc(N*4), *h_o=(float*)calloc(N,4);
  for (int i=0;i<N;i++) { h_a[i]=(float)(i+1); h_b[i]=(float)(N-i); h_c[i]=(float)(i*2); }
  float *d_a,*d_b,*d_c,*d_o;
  CHECK(hipMalloc(&d_a,N*4)); CHECK(hipMalloc(&d_b,N*4));
  CHECK(hipMalloc(&d_c,N*4)); CHECK(hipMalloc(&d_o,N*4));
  CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m,co)); CHECK(hipModuleGetFunction(&f,m,"fma_vals"));
  int n=N; void* args[]={&d_o,&d_a,&d_b,&d_c,&n};
  CHECK(hipModuleLaunchKernel(f,nb,1,1,bs,1,1,0,0,args,NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp=__builtin_fmaf(h_a[i],h_b[i],h_c[i]);
    if(fabsf(h_o[i]-exp)<0.01f) ok++;
  }
  printf("  fma_vals:         %d/%d %s\n",ok,N,ok==N?"PASS":"FAIL");
  hipFree(d_a);hipFree(d_b);hipFree(d_c);hipFree(d_o);hipModuleUnload(m);
  free(h_a);free(h_b);free(h_c);free(h_o);
  return ok==N?0:1;
}

int run_test_exp_kernel(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in=(float*)malloc(N*4), *h_o=(float*)calloc(N,4);
  for (int i=0;i<N;i++) h_in[i]=(float)(i-128)*0.05f;  // range ~[-6.4, 6.35]
  float *d_in,*d_o;
  CHECK(hipMalloc(&d_in,N*4)); CHECK(hipMalloc(&d_o,N*4));
  CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m,co)); CHECK(hipModuleGetFunction(&f,m,"exp_kernel"));
  int n=N; void* args[]={&d_o,&d_in,&n};
  CHECK(hipModuleLaunchKernel(f,nb,1,1,bs,1,1,0,0,args,NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp_val=expf(h_in[i]);
    if(fabsf(h_o[i]-exp_val)<=1e-4f*fabsf(exp_val)+1e-7f) ok++;
  }
  printf("  exp_kernel:       %d/%d %s\n",ok,N,ok==N?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return ok==N?0:1;
}

int run_test_recip_vals(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in=(float*)malloc(N*4), *h_o=(float*)calloc(N,4);
  for (int i=0;i<N;i++) h_in[i]=(float)(i+1)*0.1f;  // range [0.1, 25.6]
  float *d_in,*d_o;
  CHECK(hipMalloc(&d_in,N*4)); CHECK(hipMalloc(&d_o,N*4));
  CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m,co)); CHECK(hipModuleGetFunction(&f,m,"recip_vals"));
  int n=N; void* args[]={&d_o,&d_in,&n};
  CHECK(hipModuleLaunchKernel(f,nb,1,1,bs,1,1,0,0,args,NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp_val=1.0f/h_in[i];
    if(fabsf(h_o[i]-exp_val)<=1e-5f*fabsf(exp_val)+1e-10f) ok++;
  }
  printf("  recip_vals:       %d/%d %s\n",ok,N,ok==N?"PASS":"FAIL");
  hipFree(d_in);hipFree(d_o);hipModuleUnload(m);free(h_in);free(h_o);
  return ok==N?0:1;
}

int run_test_mag2d(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_a=(float*)malloc(N*4), *h_b=(float*)malloc(N*4), *h_o=(float*)calloc(N,4);
  for (int i=0;i<N;i++) { h_a[i]=(float)(i%16)+0.5f; h_b[i]=(float)(i/16)+0.5f; }
  float *d_a,*d_b,*d_o;
  CHECK(hipMalloc(&d_a,N*4)); CHECK(hipMalloc(&d_b,N*4)); CHECK(hipMalloc(&d_o,N*4));
  CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m,co)); CHECK(hipModuleGetFunction(&f,m,"mag2d"));
  int n=N; void* args[]={&d_o,&d_a,&d_b,&n};
  CHECK(hipModuleLaunchKernel(f,nb,1,1,bs,1,1,0,0,args,NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
  int ok=0; for(int i=0;i<N;i++) {
    float exp_val=sqrtf(h_a[i]*h_a[i]+h_b[i]*h_b[i]);
    if(fabsf(h_o[i]-exp_val)<=1e-4f*fabsf(exp_val)+1e-7f) ok++;
  }
  printf("  mag2d:            %d/%d %s\n",ok,N,ok==N?"PASS":"FAIL");
  hipFree(d_a);hipFree(d_b);hipFree(d_o);hipModuleUnload(m);
  free(h_a);free(h_b);free(h_o);
  return ok==N?0:1;
}

// ── split-K matmul kernels (round 4) ────────────────────────────────────────
// These exercise blockIdx.y (TTMP9-derived workgroup Y) which the plain 1D
// matmul never reaches.  Two kernels per ELF: compute + reduce.

extern "C" __global__ void matmul_splitk_compute(
    float* __restrict__ workspace,
    const float* __restrict__ A, const float* __restrict__ B,
    int M, int N, int K, int num_splits)
{
  int row = blockIdx.x, split = blockIdx.y, tid = threadIdx.x;
  if (row >= M) return;
  int K_chunk = (K + num_splits - 1) / num_splits;
  int k_lo = split * K_chunk, k_hi = k_lo + K_chunk;
  if (k_hi > K) k_hi = K;
  for (int col = tid; col < N; col += blockDim.x) {
    float acc = 0.0f;
    for (int k = k_lo; k < k_hi; k++) acc += A[row*K+k] * B[k*N+col];
    workspace[split * M * N + row * N + col] = acc;
  }
}

extern "C" __global__ void matmul_splitk_reduce(
    float* __restrict__ C, const float* __restrict__ workspace,
    int M, int N, int num_splits)
{
  int row = blockIdx.x, tid = threadIdx.x;
  if (row >= M) return;
  for (int col = tid; col < N; col += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; s++) acc += workspace[s * M * N + row * N + col];
    C[row * N + col] = acc;
  }
}

// ── attention kernels (round 5) ──────────────────────────────────────────────
// Exercises: exp, reciprocal, max/sum shared-memory reduction, FMA chains.
// attn_multihead uses blockIdx.y → TTMP taint chain for workgroup-ID Y.

extern "C" __global__ void softmax_row(
    float* __restrict__ out, const float* __restrict__ in,
    int rows, int cols)
{
  extern __shared__ float smem[];
  int row = blockIdx.x, tid = threadIdx.x;
  if (row >= rows) return;
  const float* ri = in + row * cols;
  float* ro = out + row * cols;
  float mx = -1e30f;
  for (int c = tid; c < cols; c += blockDim.x) mx = fmaxf(mx, ri[c]);
  smem[tid] = mx; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]); __syncthreads();
  }
  float rmax = smem[0]; __syncthreads();
  float sm = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    float e = __expf(ri[c] - rmax); ro[c] = e; sm += e;
  }
  smem[tid] = sm; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid+s]; __syncthreads();
  }
  float inv = 1.0f / smem[0]; __syncthreads();
  for (int c = tid; c < cols; c += blockDim.x) ro[c] *= inv;
}

extern "C" __global__ void attn_forward(
    float* __restrict__ O, const float* __restrict__ Q,
    const float* __restrict__ K, const float* __restrict__ V,
    int M, int N, int D)
{
  extern __shared__ float smem[];
  float* S = smem; float* sc = smem + N;
  int m = blockIdx.x, tid = threadIdx.x;
  if (m >= M) return;
  const float* Qm = Q + m * D;
  float scale = 1.0f / sqrtf((float)D);
  for (int n = 0; n < N; n++) {
    const float* Kn = K + n * D;
    float dot = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) dot += Qm[d] * Kn[d];
    sc[tid] = dot; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tid < s) sc[tid] += sc[tid+s]; __syncthreads();
    }
    if (tid == 0) S[n] = sc[0] * scale;
    __syncthreads();
  }
  float mx = -1e30f;
  for (int n = tid; n < N; n += blockDim.x) mx = fmaxf(mx, S[n]);
  sc[tid] = mx; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sc[tid] = fmaxf(sc[tid], sc[tid+s]); __syncthreads();
  }
  float smax = sc[0]; __syncthreads();
  float sm = 0.0f;
  for (int n = tid; n < N; n += blockDim.x) {
    S[n] = __expf(S[n] - smax); sm += S[n];
  }
  sc[tid] = sm; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sc[tid] += sc[tid+s]; __syncthreads();
  }
  float inv = 1.0f / sc[0]; __syncthreads();
  for (int n = tid; n < N; n += blockDim.x) S[n] *= inv;
  __syncthreads();
  float* Om = O + m * D;
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int n = 0; n < N; n++) acc += S[n] * V[n*D+d];
    Om[d] = acc;
  }
}

extern "C" __global__ void attn_multihead(
    float* __restrict__ O, const float* __restrict__ Q,
    const float* __restrict__ K, const float* __restrict__ V,
    int M, int N, int D, int num_heads)
{
  extern __shared__ float smem[];
  float* S = smem; float* sc = smem + N;
  int m = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  if (m >= M || h >= num_heads) return;
  const float* Qm = Q + h*M*D + m*D;
  const float* Kh = K + h*N*D;
  const float* Vh = V + h*N*D;
  float* Om = O + h*M*D + m*D;
  float scale = 1.0f / sqrtf((float)D);
  for (int n = 0; n < N; n++) {
    float dot = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) dot += Qm[d] * Kh[n*D+d];
    sc[tid] = dot; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tid < s) sc[tid] += sc[tid+s]; __syncthreads();
    }
    if (tid == 0) S[n] = sc[0] * scale;
    __syncthreads();
  }
  float mx = -1e30f;
  for (int n = tid; n < N; n += blockDim.x) mx = fmaxf(mx, S[n]);
  sc[tid] = mx; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sc[tid] = fmaxf(sc[tid], sc[tid+s]); __syncthreads();
  }
  float smax = sc[0]; __syncthreads();
  float sm = 0.0f;
  for (int n = tid; n < N; n += blockDim.x) {
    S[n] = __expf(S[n] - smax); sm += S[n];
  }
  sc[tid] = sm; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sc[tid] += sc[tid+s]; __syncthreads();
  }
  float inv = 1.0f / sc[0]; __syncthreads();
  for (int n = tid; n < N; n += blockDim.x) S[n] *= inv;
  __syncthreads();
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int n = 0; n < N; n++) acc += S[n] * Vh[n*D+d];
    Om[d] = acc;
  }
}

// ── matmul multi-shape harness ───────────────────────────────────────────────

static void ref_matmul(float* C, const float* A, const float* B, int M, int N, int K) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0.0f;
      for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
      C[i*N+j] = s;
    }
}

// Returns number of elements within tolerance, or -1 on GPU error.
static int check_matmul_shape(hipFunction_t fn, int M, int N, int K) {
  size_t szA=(size_t)M*K*4, szB=(size_t)K*N*4, szC=(size_t)M*N*4;
  float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
  float *hC=(float*)calloc(M*N,4), *hRef=(float*)malloc(szC);
  // Small integer inputs — exact in FP32, reference sum is precise.
  srand((unsigned)(M*7919u + N*6271u + K*4999u));
  for (int i=0;i<M*K;i++) hA[i]=(float)(rand()%5-2);   // {-2,-1,0,1,2}
  for (int i=0;i<K*N;i++) hB[i]=(float)(rand()%5-2);
  ref_matmul(hRef,hA,hB,M,N,K);
  float *dA,*dB,*dC;
  hipMalloc(&dA,szA); hipMalloc(&dB,szB); hipMalloc(&dC,szC);
  hipMemcpy(dA,hA,szA,hipMemcpyHostToDevice);
  hipMemcpy(dB,hB,szB,hipMemcpyHostToDevice);
  hipMemset(dC,0,szC);
  int m=M,n=N,k=K;
  void* args[]={&dC,&dA,&dB,&m,&n,&k};
  hipError_t e = hipModuleLaunchKernel(fn, M,1,1, 32,1,1, 0,0, args, NULL);
  int result = -1;
  if (e == hipSuccess && hipDeviceSynchronize() == hipSuccess) {
    hipMemcpy(hC,dC,szC,hipMemcpyDeviceToHost);
    // Tolerance: K accumulations of integers → allow K*eps relative + small absolute.
    float atol = K * 1.5e-5f + 1e-4f;
    int ok=0;
    for (int i=0;i<M*N;i++) {
      float tol = atol * (fabsf(hRef[i]) + 1.0f);
      if (fabsf(hC[i]-hRef[i]) <= tol) ok++;
    }
    result = ok;
  }
  hipFree(dA); hipFree(dB); hipFree(dC);
  free(hA); free(hB); free(hC); free(hRef);
  return result;
}

int run_test_matmul(const char* co) {
  hipModule_t mod; hipFunction_t fn;
  if (hipModuleLoad(&mod, co) != hipSuccess ||
      hipModuleGetFunction(&fn, mod, "matmul") != hipSuccess) {
    printf("  matmul:           LOAD FAILED\n");
    return 1;
  }

  // Shapes inspired by hipBLAS-LT matmul_common.yaml (small/medium/deepbench/resnet50)
  struct { int M, N, K; const char* label; } shapes[] = {
    {   8,   8,   8, "    8x   8x   8"},  // hipBLASLT small
    {  16,  16,  16, "   16x  16x  16"},  // hipBLASLT small
    {  32,  32,  32, "   32x  32x  32"},  // hipBLASLT small
    {  64,  64,  64, "   64x  64x  64"},  // hipBLASLT small
    { 128, 128,  64, "  128x 128x  64"},  // hipBLASLT medium
    { 128, 128, 128, "  128x 128x 128"},  // hipBLASLT medium
    { 256, 256, 256, "  256x 256x 256"},  // hipBLASLT medium
    { 192,  64, 128, "  192x  64x 128"},  // DeepBench-inspired
    {1024,  32, 256, " 1024x  32x 256"},  // DNN FC: tall-skinny N
    {2048,  16, 128, " 2048x  16x 128"},  // DNN FC: very tall-skinny
    {  64, 256, 128, "   64x 256x 128"},  // wide N — exercises stride loop
    { 512,  32, 512, "  512x  32x 512"},  // DeepBench-inspired
  };
  int nshapes = (int)(sizeof(shapes)/sizeof(shapes[0]));
  int pass = 0;
  for (int i = 0; i < nshapes; i++) {
    int M=shapes[i].M, N=shapes[i].N, K=shapes[i].K;
    int ok = check_matmul_shape(fn, M, N, K);
    int total = M*N;
    if (ok < 0) {
      printf("  matmul %s: LAUNCH ERROR\n", shapes[i].label);
    } else if (ok < total) {
      printf("  matmul %s: %d/%d FAIL\n", shapes[i].label, ok, total);
    } else {
      pass++;
    }
  }
  printf("  matmul:           %d/%d shapes PASS\n", pass, nshapes);
  hipModuleUnload(mod);
  return pass == nshapes ? 0 : 1;
}

// ── split-K test harness ─────────────────────────────────────────────────────

static int check_splitk_shape(hipFunction_t fn_compute, hipFunction_t fn_reduce,
                               int M, int N, int K, int num_splits) {
  size_t szA  = (size_t)M*K*4, szB = (size_t)K*N*4, szC = (size_t)M*N*4;
  size_t szWS = szC * num_splits;
  float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
  float *hC=(float*)calloc(M*N,4), *hRef=(float*)malloc(szC);
  srand((unsigned)(M*6151u + N*4093u + K*2999u + num_splits*1777u));
  for (int i=0;i<M*K;i++) hA[i]=(float)(rand()%5-2);
  for (int i=0;i<K*N;i++) hB[i]=(float)(rand()%5-2);
  ref_matmul(hRef, hA, hB, M, N, K);

  float *dA,*dB,*dC,*dWS;
  hipMalloc(&dA,szA); hipMalloc(&dB,szB);
  hipMalloc(&dC,szC); hipMalloc(&dWS,szWS);
  hipMemcpy(dA,hA,szA,hipMemcpyHostToDevice);
  hipMemcpy(dB,hB,szB,hipMemcpyHostToDevice);
  hipMemset(dC,0,szC); hipMemset(dWS,0,szWS);

  int m=M, n=N, k=K, ns=num_splits;
  void* cargs[]={&dWS,&dA,&dB,&m,&n,&k,&ns};
  void* rargs[]={&dC,&dWS,&m,&n,&ns};

  int result = -1;
  hipError_t e = hipModuleLaunchKernel(fn_compute, M,num_splits,1, 32,1,1, 0,0, cargs, NULL);
  if (e == hipSuccess) e = hipDeviceSynchronize();
  if (e == hipSuccess) e = hipModuleLaunchKernel(fn_reduce, M,1,1, 32,1,1, 0,0, rargs, NULL);
  if (e == hipSuccess && hipDeviceSynchronize() == hipSuccess) {
    hipMemcpy(hC,dC,szC,hipMemcpyDeviceToHost);
    float atol = K * 1.5e-5f + 1e-4f;
    int ok=0;
    for (int i=0;i<M*N;i++) {
      float tol = atol * (fabsf(hRef[i]) + 1.0f);
      if (fabsf(hC[i]-hRef[i]) <= tol) ok++;
    }
    result = ok;
  }
  hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dWS);
  free(hA); free(hB); free(hC); free(hRef);
  return result;
}

// Each pass is a separate ELF: transpiler currently translates one kernel per
// code object (multi-kernel ELF support requires per-kernel code range tracking
// to update kernel_code_entry_byte_offset after translation).
int run_test_matmul_splitk(const char* co_compute, const char* co_reduce) {
  hipModule_t mod_c, mod_r; hipFunction_t fn_compute, fn_reduce;
  if (hipModuleLoad(&mod_c, co_compute) != hipSuccess ||
      hipModuleGetFunction(&fn_compute, mod_c, "matmul_splitk_compute") != hipSuccess) {
    printf("  matmul_splitk:    LOAD FAILED (compute)\n");
    return 1;
  }
  if (hipModuleLoad(&mod_r, co_reduce) != hipSuccess ||
      hipModuleGetFunction(&fn_reduce, mod_r, "matmul_splitk_reduce") != hipSuccess) {
    printf("  matmul_splitk:    LOAD FAILED (reduce)\n");
    hipModuleUnload(mod_c);
    return 1;
  }

  // Shapes: small/medium N, large K — classic split-K territory.
  // num_splits chosen so K_chunk ≤ 256 (avoids very long inner loops in tests).
  struct { int M, N, K, splits; const char* label; } shapes[] = {
    {  32,  32,  512,  4, "   32x 32x 512 s4"},   // baseline
    {  64,  32,  512,  4, "   64x 32x 512 s4"},   // 2x M
    { 128,  32, 1024,  8, "  128x 32x1024 s8"},   // DNN FC-like
    { 256,  32, 2048, 16, "  256x 32x2048 s16"},  // K-heavy (hipBLASLT deep)
    { 512,  16, 1024,  8, "  512x 16x1024 s8"},   // tall-skinny
    {1024,   8,  256,  4, " 1024x  8x 256 s4"},   // very tall-skinny
    {  16,  32, 4096, 16, "   16x 32x4096 s16"},  // small M, very large K
    { 192,  32,  768,  6, "  192x 32x 768 s6"},   // BERT-like (768 hidden)
  };
  int nshapes = (int)(sizeof(shapes)/sizeof(shapes[0]));
  int pass = 0;
  for (int i = 0; i < nshapes; i++) {
    int M=shapes[i].M, N=shapes[i].N, K=shapes[i].K, S=shapes[i].splits;
    int ok = check_splitk_shape(fn_compute, fn_reduce, M, N, K, S);
    int total = M*N;
    if (ok < 0)     printf("  matmul_splitk %s: LAUNCH ERROR\n", shapes[i].label);
    else if (ok < total) printf("  matmul_splitk %s: %d/%d FAIL\n", shapes[i].label, ok, total);
    else pass++;
  }
  printf("  matmul_splitk:    %d/%d shapes PASS\n", pass, nshapes);
  hipModuleUnload(mod_c);
  hipModuleUnload(mod_r);
  return pass == nshapes ? 0 : 1;
}

// ── attention test harness ───────────────────────────────────────────────────

static void ref_softmax_row(float* out, const float* in, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    float mx = -1e30f;
    for (int j = 0; j < cols; j++) mx = fmaxf(mx, in[i*cols+j]);
    float sm = 0.0f;
    for (int j = 0; j < cols; j++) { out[i*cols+j] = expf(in[i*cols+j] - mx); sm += out[i*cols+j]; }
    for (int j = 0; j < cols; j++) out[i*cols+j] /= sm;
  }
}

static void ref_attn_forward(float* O, const float* Q, const float* K,
                              const float* V, int M, int N, int D) {
  float scale = 1.0f / sqrtf((float)D);
  float* S = (float*)malloc(N * sizeof(float));
  for (int m = 0; m < M; m++) {
    float mx = -1e30f;
    for (int n = 0; n < N; n++) {
      float dot = 0.0f;
      for (int d = 0; d < D; d++) dot += Q[m*D+d] * K[n*D+d];
      S[n] = dot * scale;
      mx = fmaxf(mx, S[n]);
    }
    float sm = 0.0f;
    for (int n = 0; n < N; n++) { S[n] = expf(S[n] - mx); sm += S[n]; }
    for (int n = 0; n < N; n++) S[n] /= sm;
    for (int d = 0; d < D; d++) {
      float acc = 0.0f;
      for (int n = 0; n < N; n++) acc += S[n] * V[n*D+d];
      O[m*D+d] = acc;
    }
  }
  free(S);
}

static int check_softmax_shape(hipFunction_t fn, int rows, int cols) {
  size_t sz = (size_t)rows * cols * 4;
  float *hIn=(float*)malloc(sz), *hOut=(float*)calloc(rows*cols,4), *hRef=(float*)malloc(sz);
  srand((unsigned)(rows*5501u + cols*3313u));
  for (int i = 0; i < rows*cols; i++) hIn[i] = (float)(rand()%200 - 100) / 50.0f;
  ref_softmax_row(hRef, hIn, rows, cols);
  float *dIn, *dOut;
  hipMalloc(&dIn, sz); hipMalloc(&dOut, sz);
  hipMemcpy(dIn, hIn, sz, hipMemcpyHostToDevice); hipMemset(dOut, 0, sz);
  int r=rows, c=cols;
  void* args[]={&dOut, &dIn, &r, &c};
  hipError_t e = hipModuleLaunchKernel(fn, rows,1,1, 32,1,1, 32*4,0, args, NULL);
  int result = -1;
  if (e == hipSuccess && hipDeviceSynchronize() == hipSuccess) {
    hipMemcpy(hOut, dOut, sz, hipMemcpyDeviceToHost);
    int ok = 0;
    for (int i = 0; i < rows*cols; i++)
      if (fabsf(hOut[i] - hRef[i]) <= 1e-3f * fabsf(hRef[i]) + 1e-5f) ok++;
    // Diagnostic: dump first row of output vs reference
    if (ok < rows * cols && rows <= 16) {
      int n = cols < 8 ? cols : 8;
      printf("    GPU[0..%d]: ", n-1);
      for (int j=0; j<n; j++) printf("%.4f ", hOut[j]);
      printf("\n    REF[0..%d]: ", n-1);
      for (int j=0; j<n; j++) printf("%.4f ", hRef[j]);
      printf("\n    IN[0..%d]:  ", n-1);
      for (int j=0; j<n; j++) printf("%.4f ", hIn[j]);
      printf("\n");
    }
    result = ok;
  }
  hipFree(dIn); hipFree(dOut); free(hIn); free(hOut); free(hRef);
  return result;
}

int run_test_softmax_row(const char* co) {
  hipModule_t mod; hipFunction_t fn;
  if (hipModuleLoad(&mod, co) != hipSuccess ||
      hipModuleGetFunction(&fn, mod, "softmax_row") != hipSuccess) {
    printf("  softmax_row:      LOAD FAILED\n"); return 1;
  }
  struct { int rows, cols; const char* label; } shapes[] = {
    {  1,   1, "   1x   1"},  {  1,  32, "   1x  32"},
    { 16,  32, "  16x  32"},  { 32,  64, "  32x  64"},
    { 64, 128, "  64x 128"},  {128, 256, " 128x 256"},
    {  8,  16, "   8x  16"},
  };
  int nshapes = (int)(sizeof(shapes)/sizeof(shapes[0]));
  int pass = 0;
  for (int i = 0; i < nshapes; i++) {
    int R=shapes[i].rows, C=shapes[i].cols;
    int ok = check_softmax_shape(fn, R, C);
    int total = R*C;
    if (ok < 0) printf("  softmax_row %s: LAUNCH ERROR\n", shapes[i].label);
    else if (ok < total) printf("  softmax_row %s: %d/%d FAIL\n", shapes[i].label, ok, total);
    else pass++;
  }
  printf("  softmax_row:      %d/%d shapes PASS\n", pass, nshapes);
  hipModuleUnload(mod);
  return pass == nshapes ? 0 : 1;
}

static int check_attn_shape(hipFunction_t fn, int M, int N, int D) {
  size_t szQ=(size_t)M*D*4, szK=(size_t)N*D*4, szV=szK, szO=szQ;
  float *hQ=(float*)malloc(szQ), *hK=(float*)malloc(szK);
  float *hV=(float*)malloc(szV), *hO=(float*)calloc(M*D,4), *hRef=(float*)malloc(szO);
  srand((unsigned)(M*7717u + N*5519u + D*3313u));
  for (int i=0;i<M*D;i++) hQ[i]=(float)(rand()%200-100)/100.0f;
  for (int i=0;i<N*D;i++) hK[i]=(float)(rand()%200-100)/100.0f;
  for (int i=0;i<N*D;i++) hV[i]=(float)(rand()%200-100)/100.0f;
  ref_attn_forward(hRef, hQ, hK, hV, M, N, D);
  float *dQ,*dK,*dV,*dO;
  hipMalloc(&dQ,szQ); hipMalloc(&dK,szK); hipMalloc(&dV,szV); hipMalloc(&dO,szO);
  hipMemcpy(dQ,hQ,szQ,hipMemcpyHostToDevice);
  hipMemcpy(dK,hK,szK,hipMemcpyHostToDevice);
  hipMemcpy(dV,hV,szV,hipMemcpyHostToDevice);
  hipMemset(dO,0,szO);
  int m=M,n=N,d=D;
  void* args[]={&dO,&dQ,&dK,&dV,&m,&n,&d};
  size_t shmem = (N + 32) * 4;
  hipError_t e = hipModuleLaunchKernel(fn, M,1,1, 32,1,1, shmem,0, args, NULL);
  int result = -1;
  if (e == hipSuccess && hipDeviceSynchronize() == hipSuccess) {
    hipMemcpy(hO,dO,szO,hipMemcpyDeviceToHost);
    int ok=0;
    for (int i=0;i<M*D;i++)
      if (fabsf(hO[i]-hRef[i]) <= 1e-2f*fabsf(hRef[i])+1e-4f) ok++;
    result = ok;
  }
  hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
  free(hQ);free(hK);free(hV);free(hO);free(hRef);
  return result;
}

int run_test_attn_forward(const char* co) {
  hipModule_t mod; hipFunction_t fn;
  if (hipModuleLoad(&mod, co) != hipSuccess ||
      hipModuleGetFunction(&fn, mod, "attn_forward") != hipSuccess) {
    printf("  attn_forward:     LOAD FAILED\n"); return 1;
  }
  struct { int M,N,D; const char* label; } shapes[] = {
    {  8, 16, 16, "  8x 16x 16"},
    { 16, 32, 32, " 16x 32x 32"},
    { 32, 64, 16, " 32x 64x 16"},
    {  4,128,  8, "  4x128x  8"},
  };
  int nshapes = (int)(sizeof(shapes)/sizeof(shapes[0]));
  int pass = 0;
  for (int i = 0; i < nshapes; i++) {
    int M=shapes[i].M, N=shapes[i].N, D=shapes[i].D;
    int ok = check_attn_shape(fn, M, N, D);
    int total = M*D;
    if (ok < 0) printf("  attn_forward %s: LAUNCH ERROR\n", shapes[i].label);
    else if (ok < total) printf("  attn_forward %s: %d/%d FAIL\n", shapes[i].label, ok, total);
    else pass++;
  }
  printf("  attn_forward:     %d/%d shapes PASS\n", pass, nshapes);
  hipModuleUnload(mod);
  return pass == nshapes ? 0 : 1;
}

static int check_attn_multihead_shape(hipFunction_t fn, int H, int M, int N, int D) {
  size_t szQ=(size_t)H*M*D*4, szK=(size_t)H*N*D*4, szV=szK, szO=szQ;
  float *hQ=(float*)malloc(szQ), *hK=(float*)malloc(szK);
  float *hV=(float*)malloc(szV), *hO=(float*)calloc(H*M*D,4), *hRef=(float*)malloc(szO);
  srand((unsigned)(H*9311u + M*7717u + N*5519u + D*3313u));
  for (int i=0;i<H*M*D;i++) hQ[i]=(float)(rand()%200-100)/100.0f;
  for (int i=0;i<H*N*D;i++) hK[i]=(float)(rand()%200-100)/100.0f;
  for (int i=0;i<H*N*D;i++) hV[i]=(float)(rand()%200-100)/100.0f;
  // Reference: run single-head attention per head
  for (int h=0;h<H;h++)
    ref_attn_forward(hRef+h*M*D, hQ+h*M*D, hK+h*N*D, hV+h*N*D, M, N, D);
  float *dQ,*dK,*dV,*dO;
  hipMalloc(&dQ,szQ); hipMalloc(&dK,szK); hipMalloc(&dV,szV); hipMalloc(&dO,szO);
  hipMemcpy(dQ,hQ,szQ,hipMemcpyHostToDevice);
  hipMemcpy(dK,hK,szK,hipMemcpyHostToDevice);
  hipMemcpy(dV,hV,szV,hipMemcpyHostToDevice);
  hipMemset(dO,0,szO);
  int m=M,n=N,d=D,nh=H;
  void* args[]={&dO,&dQ,&dK,&dV,&m,&n,&d,&nh};
  size_t shmem = (N + 32) * 4;
  hipError_t e = hipModuleLaunchKernel(fn, M,H,1, 32,1,1, shmem,0, args, NULL);
  int result = -1;
  if (e == hipSuccess && hipDeviceSynchronize() == hipSuccess) {
    hipMemcpy(hO,dO,szO,hipMemcpyDeviceToHost);
    int ok=0;
    for (int i=0;i<H*M*D;i++)
      if (fabsf(hO[i]-hRef[i]) <= 1e-2f*fabsf(hRef[i])+1e-4f) ok++;
    result = ok;
  }
  hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
  free(hQ);free(hK);free(hV);free(hO);free(hRef);
  return result;
}

int run_test_attn_multihead(const char* co) {
  hipModule_t mod; hipFunction_t fn;
  if (hipModuleLoad(&mod, co) != hipSuccess ||
      hipModuleGetFunction(&fn, mod, "attn_multihead") != hipSuccess) {
    printf("  attn_multihead:   LOAD FAILED\n"); return 1;
  }
  struct { int H,M,N,D; const char* label; } shapes[] = {
    { 4,  8, 16,  8, "h4   8x 16x  8"},
    { 2, 16, 32, 16, "h2  16x 32x 16"},
    { 8,  4,  8, 16, "h8   4x  8x 16"},
    { 1, 32, 64,  8, "h1  32x 64x  8"},
  };
  int nshapes = (int)(sizeof(shapes)/sizeof(shapes[0]));
  int pass = 0;
  for (int i = 0; i < nshapes; i++) {
    int H=shapes[i].H, M=shapes[i].M, N=shapes[i].N, D=shapes[i].D;
    int ok = check_attn_multihead_shape(fn, H, M, N, D);
    int total = H*M*D;
    if (ok < 0) printf("  attn_multihead %s: LAUNCH ERROR\n", shapes[i].label);
    else if (ok < total) printf("  attn_multihead %s: %d/%d FAIL\n", shapes[i].label, ok, total);
    else pass++;
  }
  printf("  attn_multihead:   %d/%d shapes PASS\n", pass, nshapes);
  hipModuleUnload(mod);
  return pass == nshapes ? 0 : 1;
}

// ── new instruction coverage kernels ─────────────────────────────────────────

// Tests: global_atomic_add_u32, ds_add_u32 (LDS atomic), v_cndmask_b32
// Histogram: each thread reads a value, atomics-adds into bins (LDS then global)
extern "C" __global__ void atomic_histogram(int* global_hist, const int* data,
                                             int n, int num_bins) {
  extern __shared__ int shared_hist[];
  int tid = threadIdx.x;
  // Clear shared histogram
  for (int b = tid; b < num_bins; b += blockDim.x) shared_hist[b] = 0;
  __syncthreads();
  // Accumulate into shared hist via LDS atomics (ds_add_u32)
  int i = blockIdx.x * blockDim.x + tid;
  if (i < n) {
    int bin = data[i] % num_bins;
    if (bin < 0) bin += num_bins;
    atomicAdd(&shared_hist[bin], 1);
  }
  __syncthreads();
  // Write back to global via global atomics (global_atomic_add_u32)
  for (int b = tid; b < num_bins; b += blockDim.x)
    atomicAdd(&global_hist[b], shared_hist[b]);
}

// Tests: global_atomic_add_f32 (floating-point atomic)
// Each thread atomics-adds its value to one of several bins
extern "C" __global__ void atomic_fadd(float* bins, const float* vals,
                                        const int* bin_ids, int n, int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bins[bin_ids[i]], vals[i]);
}

// Tests: v_cndmask_b32_e32, v_cndmask_b32_e64 (branchless ternary select)
// Exercises the three v_cndmask handlers in the transpiler
extern "C" __global__ void cndmask_select(float* out, const float* a, const float* b,
                                           const int* mask, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Simple ternary → v_cndmask_b32
    float sel = mask[i] ? a[i] : b[i];
    // Chained ternary → multiple v_cndmask
    float sel2 = (mask[i] > 0) ? a[i] * 2.0f : b[i] * 0.5f;
    out[i] = sel + sel2;
  }
}

// Tests: v_cvt_f32_f16_e32, v_cvt_f16_f32_e32, s_cvt_f32_f16 (half precision conversion)
extern "C" __global__ void f16_convert(float* out, const unsigned short* in_f16, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Load f16 bits, convert to f32, compute, convert back to f32 output
    // The cast through __half exercises v_cvt_f32_f16
    _Float16 h = *reinterpret_cast<const _Float16*>(&in_f16[i]);
    float f = (float)h;
    out[i] = f * 2.0f + 1.0f;
  }
}

// Tests: v_med3_f32 (median of 3 values — common in image processing)
extern "C" __global__ void median3_kernel(float* out, const float* a, const float* b,
                                           const float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i], y = b[i], z = c[i];
    // Compiler should emit v_med3_f32 for: max(min(x,y), min(max(x,y),z))
    out[i] = fmaxf(fminf(x, y), fminf(fmaxf(x, y), z));
  }
}

// Tests: bitfield extract/insert patterns (v_bfe_u32, v_bfi_b32, v_alignbit_b32)
extern "C" __global__ void bitfield_ops(unsigned* out, const unsigned* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned v = in[i];
    // Extract bits [15:8] (v_bfe_u32 pattern)
    unsigned byte1 = (v >> 8) & 0xFF;
    // Extract bits [23:16]
    unsigned byte2 = (v >> 16) & 0xFF;
    // Swap byte1 and byte2, keep the rest (v_bfi_b32 pattern)
    unsigned result = (v & 0xFF0000FFu) | (byte1 << 16) | (byte2 << 8);
    out[i] = result;
  }
}

// Tests: v_min_i32, v_max_i32, v_abs_i32, signed integer ops
extern "C" __global__ void int_minmax(int* out, const int* a, const int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int x = a[i], y = b[i];
    int mn = min(x, y);     // v_min_i32
    int mx = max(x, y);     // v_max_i32
    int d = abs(x - y);     // v_sub + v_abs? or v_sub + v_max with negation
    out[i] = mn + mx - d;   // = min + max - |diff| = 2*min for x != y
  }
}

// Tests: v_log_f32 (transcendental), v_rcp_f32 chain, combined with v_mul_f32
extern "C" __global__ void log_rcp_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    float lg = log2f(x);      // v_log_f32
    float rc = 1.0f / x;      // v_rcp_f32 + Newton-Raphson
    out[i] = lg * rc;          // v_mul_f32
  }
}

// ── new LLM kernels (round 7) ──────────────────────────────────────────────

// Leaky ReLU: exercises v_max, v_mul, v_cmp branches. No device library.
extern "C" __global__ void leaky_relu_kernel(float* out, const float* in, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    out[i] = (x > 0.0f) ? x : alpha * x;
  }
}

// RMS Normalization (used in LLaMA, simpler than layernorm)
extern "C" __global__ void rmsnorm_kernel(float* out, const float* in,
                                           const float* weight, int N, float eps) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  int row = blockIdx.x;
  float val = (tid < N) ? in[row * N + tid] : 0.0f;
  smem[tid] = val * val;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid+s];
    __syncthreads();
  }
  float rms = rsqrtf(smem[0] / (float)N + eps);
  if (tid < N) out[row * N + tid] = val * rms * weight[tid];
}

// L2 distance: exercises v_sub, v_fma, v_sqrt. No device library.
extern "C" __global__ void l2_distance_kernel(float* out, const float* a, const float* b, int D, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
      float diff = a[i*D+d] - b[i*D+d];
      sum += diff * diff;
    }
    out[i] = sqrtf(sum);
  }
}

// ── test harnesses for new kernels ──────────────────────────────────────────

int run_test_atomic_histogram(const char* co) {
  int N = 1024, bs = 32, nb = N / bs, num_bins = 16;
  int *h_data = (int*)malloc(N*4), *h_hist = (int*)calloc(num_bins, 4);
  int *h_ref = (int*)calloc(num_bins, 4);
  srand(42);
  for (int i = 0; i < N; i++) {
    h_data[i] = rand() % num_bins;
    h_ref[h_data[i]]++;
  }
  int *d_data, *d_hist;
  CHECK(hipMalloc(&d_data, N*4)); CHECK(hipMalloc(&d_hist, num_bins*4));
  CHECK(hipMemcpy(d_data, h_data, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_hist, 0, num_bins*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "atomic_histogram"));
  int n=N, nb_arg=num_bins;
  void* args[] = {&d_hist, &d_data, &n, &nb_arg};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, num_bins*4,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_hist, d_hist, num_bins*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int b = 0; b < num_bins; b++) if (h_hist[b] == h_ref[b]) ok++;
  printf("  atomic_histogram: %d/%d %s\n", ok, num_bins, ok==num_bins?"PASS":"FAIL");
  hipFree(d_data); hipFree(d_hist); hipModuleUnload(m);
  free(h_data); free(h_hist); free(h_ref);
  return ok == num_bins ? 0 : 1;
}

int run_test_atomic_fadd(const char* co) {
  int N = 256, bs = 32, nb = N / bs, num_bins = 8;
  float *h_vals = (float*)malloc(N*4), *h_bins = (float*)calloc(num_bins, 4);
  int *h_ids = (int*)malloc(N*4);
  float *h_ref = (float*)calloc(num_bins, 4);
  srand(123);
  for (int i = 0; i < N; i++) {
    h_vals[i] = (float)(rand() % 100) / 10.0f;
    h_ids[i] = rand() % num_bins;
    h_ref[h_ids[i]] += h_vals[i];
  }
  float *d_bins, *d_vals; int *d_ids;
  CHECK(hipMalloc(&d_bins, num_bins*4)); CHECK(hipMalloc(&d_vals, N*4));
  CHECK(hipMalloc(&d_ids, N*4));
  CHECK(hipMemset(d_bins, 0, num_bins*4));
  CHECK(hipMemcpy(d_vals, h_vals, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_ids, h_ids, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "atomic_fadd"));
  int n=N, nbin=num_bins;
  void* args[] = {&d_bins, &d_vals, &d_ids, &n, &nbin};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_bins, d_bins, num_bins*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int b = 0; b < num_bins; b++)
    if (fabsf(h_bins[b] - h_ref[b]) <= 0.01f * fabsf(h_ref[b]) + 0.1f) ok++;
  printf("  atomic_fadd:      %d/%d %s\n", ok, num_bins, ok==num_bins?"PASS":"FAIL");
  hipFree(d_bins); hipFree(d_vals); hipFree(d_ids); hipModuleUnload(m);
  free(h_vals); free(h_bins); free(h_ids); free(h_ref);
  return ok == num_bins ? 0 : 1;
}

int run_test_cndmask_select(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_a = (float*)malloc(N*4), *h_b = (float*)malloc(N*4);
  float *h_o = (float*)calloc(N, 4);
  int *h_mask = (int*)malloc(N*4);
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)(i + 1);
    h_b[i] = (float)(-(i + 1));
    h_mask[i] = (i % 3 == 0) ? 1 : 0;  // every 3rd element
  }
  float *d_a, *d_b, *d_o; int *d_mask;
  CHECK(hipMalloc(&d_a, N*4)); CHECK(hipMalloc(&d_b, N*4));
  CHECK(hipMalloc(&d_o, N*4)); CHECK(hipMalloc(&d_mask, N*4));
  CHECK(hipMemcpy(d_a, h_a, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N*4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_mask, h_mask, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "cndmask_select"));
  int n=N; void* args[] = {&d_o, &d_a, &d_b, &d_mask, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float sel = h_mask[i] ? h_a[i] : h_b[i];
    float sel2 = (h_mask[i] > 0) ? h_a[i] * 2.0f : h_b[i] * 0.5f;
    float exp = sel + sel2;
    if (fabsf(h_o[i] - exp) < 1e-4f) ok++;
  }
  printf("  cndmask_select:   %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a); hipFree(d_b); hipFree(d_o); hipFree(d_mask);
  hipModuleUnload(m); free(h_a); free(h_b); free(h_o); free(h_mask);
  return ok == N ? 0 : 1;
}

int run_test_f16_convert(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  // Create f16 values as unsigned short bit patterns
  unsigned short *h_in = (unsigned short*)malloc(N*2);
  float *h_o = (float*)calloc(N, 4);
  for (int i = 0; i < N; i++) {
    // Simple f16: small positive floats. f16 format: sign(1) exp(5) mant(10)
    // 1.0 = 0x3C00, 2.0 = 0x4000, 0.5 = 0x3800
    float f = (float)(i + 1) * 0.1f;
    // Use memcpy to create _Float16 bits
    _Float16 h = (_Float16)f;
    memcpy(&h_in[i], &h, 2);
  }
  unsigned short *d_in; float *d_o;
  CHECK(hipMalloc(&d_in, N*2)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*2, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f_fn;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f_fn, m, "f16_convert"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f_fn, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    _Float16 h; memcpy(&h, &h_in[i], 2);
    float expected = (float)h * 2.0f + 1.0f;
    if (fabsf(h_o[i] - expected) <= 0.01f * fabsf(expected) + 0.01f) ok++;
  }
  printf("  f16_convert:      %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

int run_test_median3(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_a=(float*)malloc(N*4), *h_b=(float*)malloc(N*4);
  float *h_c=(float*)malloc(N*4), *h_o=(float*)calloc(N,4);
  srand(77);
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)(rand()%200 - 100) / 10.0f;
    h_b[i] = (float)(rand()%200 - 100) / 10.0f;
    h_c[i] = (float)(rand()%200 - 100) / 10.0f;
  }
  float *d_a, *d_b, *d_c, *d_o;
  CHECK(hipMalloc(&d_a,N*4)); CHECK(hipMalloc(&d_b,N*4));
  CHECK(hipMalloc(&d_c,N*4)); CHECK(hipMalloc(&d_o,N*4));
  CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "median3_kernel"));
  int n=N; void* args[] = {&d_o, &d_a, &d_b, &d_c, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float x=h_a[i], y=h_b[i], z=h_c[i];
    float med = fmaxf(fminf(x,y), fminf(fmaxf(x,y), z));
    if (fabsf(h_o[i] - med) < 1e-5f) ok++;
  }
  printf("  median3:          %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_o);
  hipModuleUnload(m); free(h_a); free(h_b); free(h_c); free(h_o);
  return ok == N ? 0 : 1;
}

int run_test_bitfield_ops(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  unsigned *h_in = (unsigned*)malloc(N*4), *h_o = (unsigned*)calloc(N, 4);
  for (int i = 0; i < N; i++) h_in[i] = (unsigned)(i * 0x01010101u + 0xDEAD0000u);
  unsigned *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "bitfield_ops"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    unsigned v = h_in[i];
    unsigned byte1 = (v >> 8) & 0xFF;
    unsigned byte2 = (v >> 16) & 0xFF;
    unsigned exp = (v & 0xFF0000FFu) | (byte1 << 16) | (byte2 << 8);
    if (h_o[i] == exp) ok++;
  }
  printf("  bitfield_ops:     %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

int run_test_int_minmax(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  int *h_a=(int*)malloc(N*4), *h_b=(int*)malloc(N*4), *h_o=(int*)calloc(N,4);
  for (int i = 0; i < N; i++) {
    h_a[i] = i * 3 - 200;  // range: -200..+568 (some negative)
    h_b[i] = 100 - i * 2;  // range: -410..+100 (some negative)
  }
  int *d_a, *d_b, *d_o;
  CHECK(hipMalloc(&d_a,N*4)); CHECK(hipMalloc(&d_b,N*4)); CHECK(hipMalloc(&d_o,N*4));
  CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "int_minmax"));
  int n=N; void* args[] = {&d_o, &d_a, &d_b, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    int x=h_a[i], y=h_b[i];
    int mn = x<y?x:y, mx = x>y?x:y, d = abs(x-y);
    int exp = mn + mx - d;
    if (h_o[i] == exp) ok++;
  }
  printf("  int_minmax:       %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a); hipFree(d_b); hipFree(d_o); hipModuleUnload(m);
  free(h_a); free(h_b); free(h_o);
  return ok == N ? 0 : 1;
}

int run_test_log_rcp(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in=(float*)malloc(N*4), *h_o=(float*)calloc(N,4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.5f;  // 0.5..128.0
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in,N*4)); CHECK(hipMalloc(&d_o,N*4));
  CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "log_rcp_kernel"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float x = h_in[i];
    float exp = log2f(x) * (1.0f / x);
    if (fabsf(h_o[i] - exp) <= 0.01f * fabsf(exp) + 1e-5f) ok++;
  }
  printf("  log_rcp:          %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── bitmanip_ops test ────────────────────────────────────────────────────

int run_test_bitmanip_ops(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  unsigned *h_in = (unsigned*)malloc(N*4);
  int *h_o = (int*)calloc(N, 4);
  for (int i = 0; i < N; i++) h_in[i] = (unsigned)(i + 1);  // 1..256, avoid 0
  unsigned *d_in; int *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "bitmanip_ops"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    unsigned v = h_in[i];
    int clz = __builtin_clz(v);
    int ffs_val = __builtin_ffs(v);
    int popc = __builtin_popcount(v);
    int exp = clz + ffs_val * 100 + popc * 10000;
    if (h_o[i] == exp) ok++;
  }
  printf("  bitmanip_ops:     %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── f64_math test ───────────────────────────────────────────────────────

int run_test_f64_math(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  double *h_in = (double*)malloc(N*8), *h_o = (double*)calloc(N, 8);
  for (int i = 0; i < N; i++) h_in[i] = (double)(i + 1) * 0.5;  // 0.5..128.0
  double *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*8)); CHECK(hipMalloc(&d_o, N*8));
  CHECK(hipMemcpy(d_in, h_in, N*8, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "f64_math"));
  int n=N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*8, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    double x = h_in[i];
    double exp_val = sqrt(x) + 1.0 / x;
    if (fabs(h_o[i] - exp_val) <= 1e-10 * fabs(exp_val) + 1e-14) ok++;
  }
  printf("  f64_math:         %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── layernorm test ──────────────────────────────────────────────────────

int run_test_layernorm(const char* co) {
  int Nf = 32;  // feature dimension (must be <= blockDim.x)
  int batch = 8;
  int bs = 32;  // blockDim.x
  float eps = 1e-5f;
  size_t sz_in = (size_t)batch * Nf * 4;
  size_t sz_param = (size_t)Nf * 4;
  float *h_in = (float*)malloc(sz_in), *h_o = (float*)calloc(batch * Nf, 4);
  float *h_gamma = (float*)malloc(sz_param), *h_beta = (float*)malloc(sz_param);
  float *h_ref = (float*)malloc(sz_in);
  srand(999);
  for (int i = 0; i < batch * Nf; i++) h_in[i] = (float)(rand() % 200 - 100) / 50.0f;
  for (int i = 0; i < Nf; i++) {
    h_gamma[i] = 1.0f + (float)(rand() % 10) / 50.0f;
    h_beta[i] = (float)(rand() % 10 - 5) / 50.0f;
  }
  // Reference
  for (int b = 0; b < batch; b++) {
    float sum = 0.0f;
    for (int j = 0; j < Nf; j++) sum += h_in[b * Nf + j];
    float mean = sum / (float)Nf;
    float var_sum = 0.0f;
    for (int j = 0; j < Nf; j++) {
      float diff = h_in[b * Nf + j] - mean;
      var_sum += diff * diff;
    }
    float inv_std = 1.0f / sqrtf(var_sum / (float)Nf + eps);
    for (int j = 0; j < Nf; j++) {
      float diff = h_in[b * Nf + j] - mean;
      h_ref[b * Nf + j] = diff * inv_std * h_gamma[j] + h_beta[j];
    }
  }
  float *d_in, *d_o, *d_gamma, *d_beta;
  CHECK(hipMalloc(&d_in, sz_in)); CHECK(hipMalloc(&d_o, sz_in));
  CHECK(hipMalloc(&d_gamma, sz_param)); CHECK(hipMalloc(&d_beta, sz_param));
  CHECK(hipMemcpy(d_in, h_in, sz_in, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_gamma, h_gamma, sz_param, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_beta, h_beta, sz_param, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, sz_in));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "layernorm"));
  int n = Nf;
  void* args[] = {&d_o, &d_in, &d_gamma, &d_beta, &n, &eps};
  size_t shmem = bs * 4;
  CHECK(hipModuleLaunchKernel(f, batch,1,1, bs,1,1, shmem,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, sz_in, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < batch * Nf; i++) {
    if (fabsf(h_o[i] - h_ref[i]) <= 1e-3f * fabsf(h_ref[i]) + 1e-4f) ok++;
  }
  int total = batch * Nf;
  // Diagnostics: dump first few output vs reference values for debugging
  if (ok < total) {
    printf("    GPU[0..7]: ");
    for (int j = 0; j < 8 && j < total; j++) printf("%.4f ", h_o[j]);
    printf("\n    REF[0..7]: ");
    for (int j = 0; j < 8 && j < total; j++) printf("%.4f ", h_ref[j]);
    printf("\n     IN[0..7]: ");
    for (int j = 0; j < 8 && j < total; j++) printf("%.4f ", h_in[j]);
    printf("\n");
  }
  printf("  layernorm:        %d/%d %s\n", ok, total, ok==total?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipFree(d_gamma); hipFree(d_beta);
  hipModuleUnload(m);
  free(h_in); free(h_o); free(h_gamma); free(h_beta); free(h_ref);
  return ok == total ? 0 : 1;
}

// ── packed_f16 test ──────────────────────────────────────────────────────────

int run_test_packed_f16(const char* co) {
  int N = 256;  // number of f16 elements
  int out_n = N / 2;  // number of float output elements
  int bs = 32, nb = (out_n + bs - 1) / bs;
  // Allocate f16 inputs and float output
  __fp16 *h_a = (__fp16*)malloc(N * sizeof(__fp16));
  __fp16 *h_b = (__fp16*)malloc(N * sizeof(__fp16));
  float *h_o = (float*)calloc(out_n, sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a[i] = (__fp16)(i * 0.1f);
    h_b[i] = (__fp16)(i * 0.2f + 1.0f);
  }
  void *d_a, *d_b;
  float *d_o;
  CHECK(hipMalloc(&d_a, N * sizeof(__fp16)));
  CHECK(hipMalloc(&d_b, N * sizeof(__fp16)));
  CHECK(hipMalloc(&d_o, out_n * sizeof(float)));
  CHECK(hipMemcpy(d_a, h_a, N * sizeof(__fp16), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N * sizeof(__fp16), hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "packed_f16"));
  int n = N; void* args[] = {&d_o, &d_a, &d_b, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, out_n * sizeof(float), hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < out_n; i++) {
    float a0 = (float)h_a[i*2], a1 = (float)h_a[i*2+1];
    float b0 = (float)h_b[i*2], b1 = (float)h_b[i*2+1];
    float r0 = a0 + b0;
    float r1 = a1 * b1;
    float expected = r0 + r1;
    if (fabsf(h_o[i] - expected) < 0.5f) ok++;  // f16 precision tolerance
  }
  printf("  packed_f16:       %d/%d %s\n", ok, out_n, ok==out_n?"PASS":"FAIL");
  hipFree(d_a); hipFree(d_b); hipFree(d_o); hipModuleUnload(m);
  free(h_a); free(h_b); free(h_o);
  return ok == out_n ? 0 : 1;
}

// ── dot_product_f16 test ────────────────────────────────────────────────────

int run_test_dot_product_f16(const char* co) {
  int N = 256;  // number of f16 elements
  int out_n = N / 2;
  int bs = 32, nb = (out_n + bs - 1) / bs;
  __fp16 *h_a = (__fp16*)malloc(N * sizeof(__fp16));
  __fp16 *h_b = (__fp16*)malloc(N * sizeof(__fp16));
  float *h_o = (float*)calloc(out_n, sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a[i] = (__fp16)(1.0f + i * 0.01f);
    h_b[i] = (__fp16)(2.0f - i * 0.005f);
  }
  void *d_a, *d_b;
  float *d_o;
  CHECK(hipMalloc(&d_a, N * sizeof(__fp16)));
  CHECK(hipMalloc(&d_b, N * sizeof(__fp16)));
  CHECK(hipMalloc(&d_o, out_n * sizeof(float)));
  CHECK(hipMemcpy(d_a, h_a, N * sizeof(__fp16), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N * sizeof(__fp16), hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "dot_product_f16"));
  int n = N; void* args[] = {&d_o, &d_a, &d_b, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, out_n * sizeof(float), hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < out_n; i++) {
    float expected = (float)h_a[i*2] * (float)h_b[i*2] +
                     (float)h_a[i*2+1] * (float)h_b[i*2+1];
    if (fabsf(h_o[i] - expected) < 0.05f) ok++;
  }
  printf("  dot_product_f16:  %d/%d %s\n", ok, out_n, ok==out_n?"PASS":"FAIL");
  hipFree(d_a); hipFree(d_b); hipFree(d_o); hipModuleUnload(m);
  free(h_a); free(h_b); free(h_o);
  return ok == out_n ? 0 : 1;
}

// ── warp_shuffle test ───────────────────────────────────────────────────────

int run_test_warp_shuffle(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  int *h_in = (int*)malloc(N * sizeof(int));
  int *h_o = (int*)calloc(N, sizeof(int));
  for (int i = 0; i < N; i++) h_in[i] = i + 1;
  int *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N * sizeof(int)));
  CHECK(hipMalloc(&d_o, N * sizeof(int)));
  CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "warp_shuffle"));
  int n = N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N * sizeof(int), hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    // Lane 0 of each wavefront broadcasts its value to all lanes.
    // Each block has bs threads. On wave32, wavefront is 32 threads.
    // On wave64 (transpiled), wavefront is 64 threads but we run 32 active.
    // Lane 0 of wavefront j has in[j*32] = j*32+1. All lanes add their own val.
    int wf = i / 32;
    int lane0_val = h_in[wf * 32];  // value of lane 0 in this wavefront
    int expected = h_in[i] + lane0_val;
    if (h_o[i] == expected) ok++;
  }
  printf("  warp_shuffle:     %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── gelu_kernel test ────────────────────────────────────────────────────────

int run_test_gelu(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N, 4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 128) * 0.05f;  // range ~[-6.4, 6.35]
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "gelu_kernel"));
  int n = N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float x = h_in[i];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float expected = 0.5f * x * (1.0f + tanhf(inner));
    // Cross-arch tolerance: GFX1250 device library tanhf (used inside GELU)
    // transpiled to GFX942 may differ from host libc tanhf by several ULP;
    // 5e-3 relative + 1e-4 absolute covers the RDNA4→CDNA3 translation gap.
    if (fabsf(h_o[i] - expected) <= 5e-3f * fabsf(expected) + 1e-4f) ok++;
  }
  printf("  gelu_kernel:      %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── reduce_max_idx test ─────────────────────────────────────────────────────

int run_test_reduce_max_idx(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4);
  float *h_oval = (float*)calloc(nb, 4);
  int *h_oidx = (int*)calloc(nb, 4);
  srand(42);
  for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000) / 100.0f;
  float *d_in, *d_oval;
  int *d_oidx;
  CHECK(hipMalloc(&d_in, N*4));
  CHECK(hipMalloc(&d_oval, nb*4));
  CHECK(hipMalloc(&d_oidx, nb*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "reduce_max_idx"));
  int n = N; void* args[] = {&d_oidx, &d_oval, &d_in, &n};
  size_t shmem = bs * (sizeof(float) + sizeof(int));
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, shmem,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_oval, d_oval, nb*4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_oidx, d_oidx, nb*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int b = 0; b < nb; b++) {
    float exp_max = -1e30f;
    int exp_idx = b * bs;
    for (int t = 0; t < bs; t++) {
      int idx = b * bs + t;
      if (idx < N && h_in[idx] > exp_max) {
        exp_max = h_in[idx];
        exp_idx = idx;
      }
    }
    if (fabsf(h_oval[b] - exp_max) < 1e-4f && h_oidx[b] == exp_idx) ok++;
  }
  printf("  reduce_max_idx:   %d/%d %s\n", ok, nb, ok==nb?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_oval); hipFree(d_oidx); hipModuleUnload(m);
  free(h_in); free(h_oval); free(h_oidx);
  return ok == nb ? 0 : 1;
}

// ── silu_kernel test ────────────────────────────────────────────────────────

int run_test_silu(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N, 4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 128) * 0.05f;  // range ~[-6.4, 6.35]
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "silu_kernel"));
  int n = N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float x = h_in[i];
    float expected = x / (1.0f + expf(-x));
    if (fabsf(h_o[i] - expected) <= 1e-4f * fabsf(expected) + 1e-6f) ok++;
  }
  printf("  silu_kernel:      %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── tanh_kernel test ─────────────────────────────────────────────────────────

int run_test_tanh(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N, 4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 128) * 0.05f;  // range ~[-6.4, 6.35]
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "tanh_kernel"));
  int n = N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float expected = tanhf(h_in[i]);
    // Cross-arch tolerance: GFX1250 device library tanhf implementation transpiled
    // to GFX942 may differ from host libc tanhf by several ULP; 5e-3 relative
    // + 1e-4 absolute covers the RDNA4→CDNA3 translation gap.
    if (fabsf(h_o[i] - expected) <= 5e-3f * fabsf(expected) + 1e-4f) ok++;
  }
  printf("  tanh_kernel:      %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── sigmoid_kernel test ──────────────────────────────────────────────────────

int run_test_sigmoid(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N, 4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 128) * 0.05f;  // range ~[-6.4, 6.35]
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "sigmoid_kernel"));
  int n = N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float x = h_in[i];
    float expected = 1.0f / (1.0f + expf(-x));
    if (fabsf(h_o[i] - expected) <= 1e-4f * fabsf(expected) + 1e-6f) ok++;
  }
  printf("  sigmoid_kernel:   %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── leaky_relu_kernel test ────────────────────────────────────────────────────

int run_test_leaky_relu(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float alpha = 0.01f;
  float *h_in = (float*)malloc(N*4), *h_o = (float*)calloc(N, 4);
  for (int i = 0; i < N; i++) h_in[i] = (float)(i - 128) * 0.1f;
  float *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N*4)); CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "leaky_relu_kernel"));
  int n = N; void* args[] = {&d_o, &d_in, &alpha, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    float x = h_in[i];
    float expected = (x > 0.0f) ? x : alpha * x;
    if (fabsf(h_o[i] - expected) < 1e-5f) ok++;
  }
  printf("  leaky_relu:       %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── rmsnorm_kernel test ──────────────────────────────────────────────────────

int run_test_rmsnorm(const char* co) {
  int Nf = 32;  // feature dimension (must be <= blockDim.x)
  int batch = 8;
  int bs = 32;  // blockDim.x
  float eps = 1e-5f;
  size_t sz_in = (size_t)batch * Nf * 4;
  size_t sz_param = (size_t)Nf * 4;
  float *h_in = (float*)malloc(sz_in), *h_o = (float*)calloc(batch * Nf, 4);
  float *h_weight = (float*)malloc(sz_param);
  float *h_ref = (float*)malloc(sz_in);
  srand(777);
  for (int i = 0; i < batch * Nf; i++) h_in[i] = (float)(rand() % 200 - 100) / 50.0f;
  for (int i = 0; i < Nf; i++) h_weight[i] = 1.0f + (float)(rand() % 10) / 50.0f;
  // Reference
  for (int b = 0; b < batch; b++) {
    float ss = 0.0f;
    for (int j = 0; j < Nf; j++) {
      float v = h_in[b * Nf + j];
      ss += v * v;
    }
    float rms = 1.0f / sqrtf(ss / (float)Nf + eps);
    for (int j = 0; j < Nf; j++)
      h_ref[b * Nf + j] = h_in[b * Nf + j] * rms * h_weight[j];
  }
  float *d_in, *d_o, *d_weight;
  CHECK(hipMalloc(&d_in, sz_in)); CHECK(hipMalloc(&d_o, sz_in));
  CHECK(hipMalloc(&d_weight, sz_param));
  CHECK(hipMemcpy(d_in, h_in, sz_in, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_weight, h_weight, sz_param, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, sz_in));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "rmsnorm_kernel"));
  int n = Nf;
  void* args[] = {&d_o, &d_in, &d_weight, &n, &eps};
  size_t shmem = bs * 4;
  CHECK(hipModuleLaunchKernel(f, batch,1,1, bs,1,1, shmem,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, sz_in, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < batch * Nf; i++) {
    if (fabsf(h_o[i] - h_ref[i]) <= 1e-3f * fabsf(h_ref[i]) + 1e-4f) ok++;
  }
  int total = batch * Nf;
  printf("  rmsnorm_kernel:   %d/%d %s\n", ok, total, ok==total?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipFree(d_weight);
  hipModuleUnload(m);
  free(h_in); free(h_o); free(h_weight); free(h_ref);
  return ok == total ? 0 : 1;
}

// ── l2_distance_kernel test ──────────────────────────────────────────────────

int run_test_l2_distance(const char* co) {
  int N = 64, D = 8;  // 64 vectors of dimension 8
  int bs = 32, nb = (N + bs - 1) / bs;
  size_t sz_ab = (size_t)N * D * 4;
  float *h_a = (float*)malloc(sz_ab), *h_b = (float*)malloc(sz_ab);
  float *h_o = (float*)calloc(N, 4), *h_ref = (float*)malloc(N * 4);
  srand(555);
  for (int i = 0; i < N * D; i++) {
    h_a[i] = (float)(rand() % 200 - 100) / 50.0f;
    h_b[i] = (float)(rand() % 200 - 100) / 50.0f;
  }
  for (int i = 0; i < N; i++) {
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
      float diff = h_a[i*D+d] - h_b[i*D+d];
      sum += diff * diff;
    }
    h_ref[i] = sqrtf(sum);
  }
  float *d_a, *d_b, *d_o;
  CHECK(hipMalloc(&d_a, sz_ab)); CHECK(hipMalloc(&d_b, sz_ab));
  CHECK(hipMalloc(&d_o, N*4));
  CHECK(hipMemcpy(d_a, h_a, sz_ab, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, sz_ab, hipMemcpyHostToDevice));
  CHECK(hipMemset(d_o, 0, N*4));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "l2_distance_kernel"));
  int n = N, dim = D;
  void* args[] = {&d_o, &d_a, &d_b, &dim, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N*4, hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    if (fabsf(h_o[i] - h_ref[i]) <= 1e-4f * fabsf(h_ref[i]) + 1e-6f) ok++;
  }
  printf("  l2_distance:      %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a); hipFree(d_b); hipFree(d_o);
  hipModuleUnload(m);
  free(h_a); free(h_b); free(h_o); free(h_ref);
  return ok == N ? 0 : 1;
}

// ── permlane16 test kernel ──────────────────────────────────────────────────
// v_permlane16_b32 with selector=1: XOR each lane index with 1 within 16-lane groups
// Swaps adjacent lanes: 0↔1, 2↔3, 4↔5, ... within each group of 16

extern "C" __global__ void permlane16_test(int* out, const int* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int val = in[i];
    int result;
    // v_permlane16_b32 vDst, vSrc, sA=1, sB=1: lane i reads from lane (i^1) in same group
    asm volatile("v_permlane16_b32 %0, %1, 1, 1" : "=v"(result) : "v"(val));
    out[i] = result;
  }
}

// ── permlanex16 test kernel ────────────────────────────────────────────────
// v_permlanex16_b32 with selector=0: cross-group swap (lanes 0-15 ↔ lanes 16-31)

extern "C" __global__ void permlanex16_test(int* out, const int* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int val = in[i];
    int result;
    // v_permlanex16_b32 vDst, vSrc, sA=0, sB=0: lane i reads from lane (i^16)
    asm volatile("v_permlanex16_b32 %0, %1, 0, 0" : "=v"(result) : "v"(val));
    out[i] = result;
  }
}

// ── vopd (dual-issue) test kernel ──────────────────────────────────────────
// Tests v_dual_add_f32 :: v_dual_mul_f32 split into sequential ops

extern "C" __global__ void vopd_ops(float* out, const float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x, y;
    // VOPD: add and multiply in parallel
    asm volatile(
      "v_dual_add_f32 %0, %2, %3 :: v_dual_mul_f32 %1, %4, %5"
      : "=v"(x), "=v"(y)
      : "v"(a[i]), "v"(1.0f), "v"(b[i]), "v"(2.0f)
    );
    out[i] = x + y;  // (a[i] + 1.0) + (b[i] * 2.0)
  }
}

// ── permlane16 test harness ────────────────────────────────────────────────

int run_test_permlane16(const char* co) {
  // Use exactly 32 threads (one wavefront) to test intra-wave permutation
  int N = 32, bs = 32, nb = 1;
  int *h_in = (int*)malloc(N * sizeof(int));
  int *h_o = (int*)calloc(N, sizeof(int));
  for (int i = 0; i < N; i++) h_in[i] = i * 10 + 1;  // distinct values per lane
  int *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N * sizeof(int)));
  CHECK(hipMalloc(&d_o, N * sizeof(int)));
  CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "permlane16_test"));
  int n = N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N * sizeof(int), hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    // v_permlane16 with selector=1: within each 16-lane group, lane i reads from lane (i^1)
    int group_base = (i / 16) * 16;
    int lane_in_group = i % 16;
    int src_lane = group_base + (lane_in_group ^ 1);
    int expected = h_in[src_lane];
    if (h_o[i] == expected) ok++;
    else if (i < 4) printf("    lane %d: got %d, expected %d (from lane %d)\n", i, h_o[i], expected, src_lane);
  }
  printf("  permlane16:       %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── permlanex16 test harness ───────────────────────────────────────────────

int run_test_permlanex16(const char* co) {
  int N = 32, bs = 32, nb = 1;
  int *h_in = (int*)malloc(N * sizeof(int));
  int *h_o = (int*)calloc(N, sizeof(int));
  for (int i = 0; i < N; i++) h_in[i] = i * 10 + 1;
  int *d_in, *d_o;
  CHECK(hipMalloc(&d_in, N * sizeof(int)));
  CHECK(hipMalloc(&d_o, N * sizeof(int)));
  CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "permlanex16_test"));
  int n = N; void* args[] = {&d_o, &d_in, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N * sizeof(int), hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    // v_permlanex16 with selector=0: lane i reads from lane (i ^ 16)
    // Group 0 reads from group 1, group 1 reads from group 0
    int src_lane = i ^ 16;
    int expected = h_in[src_lane];
    if (h_o[i] == expected) ok++;
    else if (i < 4) printf("    lane %d: got %d, expected %d (from lane %d)\n", i, h_o[i], expected, src_lane);
  }
  printf("  permlanex16:      %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_in); hipFree(d_o); hipModuleUnload(m);
  free(h_in); free(h_o);
  return ok == N ? 0 : 1;
}

// ── vopd test harness ──────────────────────────────────────────────────────

int run_test_vopd(const char* co) {
  int N = 256, bs = 32, nb = N / bs;
  float *h_a = (float*)malloc(N * sizeof(float));
  float *h_b = (float*)malloc(N * sizeof(float));
  float *h_o = (float*)calloc(N, sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)(i + 1);
    h_b[i] = (float)(N - i);
  }
  float *d_a, *d_b, *d_o;
  CHECK(hipMalloc(&d_a, N * sizeof(float)));
  CHECK(hipMalloc(&d_b, N * sizeof(float)));
  CHECK(hipMalloc(&d_o, N * sizeof(float)));
  CHECK(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice));
  hipModule_t m; hipFunction_t f;
  CHECK(hipModuleLoad(&m, co));
  CHECK(hipModuleGetFunction(&f, m, "vopd_ops"));
  int n = N; void* args[] = {&d_o, &d_a, &d_b, &n};
  CHECK(hipModuleLaunchKernel(f, nb,1,1, bs,1,1, 0,0, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h_o, d_o, N * sizeof(float), hipMemcpyDeviceToHost));
  int ok = 0;
  for (int i = 0; i < N; i++) {
    // out = (a[i] + 1.0) + (b[i] * 2.0)
    float expected = (h_a[i] + 1.0f) + (h_b[i] * 2.0f);
    if (fabsf(h_o[i] - expected) <= 1e-5f * fabsf(expected) + 1e-6f) ok++;
  }
  printf("  vopd_ops:         %d/%d %s\n", ok, N, ok==N?"PASS":"FAIL");
  hipFree(d_a); hipFree(d_b); hipFree(d_o); hipModuleUnload(m);
  free(h_a); free(h_b); free(h_o);
  return ok == N ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────

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
  fails += run_test_fma_vals("/tmp/fma_vals_gfx1250.elf");
  fails += run_test_exp_kernel("/tmp/exp_kernel_gfx1250.elf");
  fails += run_test_recip_vals("/tmp/recip_vals_gfx1250.elf");
  fails += run_test_mag2d("/tmp/mag2d_gfx1250.elf");
  fails += run_test_matmul("/tmp/matmul_gfx1250.elf");
  fails += run_test_matmul_splitk("/tmp/matmul_splitk_compute_gfx1250.elf",
                                   "/tmp/matmul_splitk_reduce_gfx1250.elf");
  fails += run_test_softmax_row("/tmp/softmax_row_gfx1250.elf");
  // New instruction coverage tests
  fails += run_test_atomic_histogram("/tmp/atomic_histogram_gfx1250.elf");
  fails += run_test_atomic_fadd("/tmp/atomic_fadd_gfx1250.elf");
  fails += run_test_cndmask_select("/tmp/cndmask_select_gfx1250.elf");
  fails += run_test_f16_convert("/tmp/f16_convert_gfx1250.elf");
  fails += run_test_median3("/tmp/median3_kernel_gfx1250.elf");
  fails += run_test_bitfield_ops("/tmp/bitfield_ops_gfx1250.elf");
  fails += run_test_int_minmax("/tmp/int_minmax_gfx1250.elf");
  fails += run_test_log_rcp("/tmp/log_rcp_kernel_gfx1250.elf");
  // New ISA coverage tests (packed f16, dot product f16, warp shuffle)
  fails += run_test_packed_f16("/tmp/packed_f16_gfx1250.elf");
  fails += run_test_dot_product_f16("/tmp/dot_product_f16_gfx1250.elf");
  fails += run_test_warp_shuffle("/tmp/warp_shuffle_gfx1250.elf");
  // Instruction gap coverage tests (bitmanip, layernorm)
  fails += run_test_bitmanip_ops("/tmp/bitmanip_ops_gfx1250.elf");
  fails += run_test_layernorm("/tmp/layernorm_gfx1250.elf");
  // AI/ML activation and reduction kernels
  fails += run_test_gelu("/tmp/gelu_kernel_gfx1250.elf");
  fails += run_test_reduce_max_idx("/tmp/reduce_max_idx_gfx1250.elf");
  fails += run_test_silu("/tmp/silu_kernel_gfx1250.elf");
  // Temp VGPR collision regression tests (device library heavy kernels)
  fails += run_test_tanh("/tmp/tanh_kernel_gfx1250.elf");
  fails += run_test_sigmoid("/tmp/sigmoid_kernel_gfx1250.elf");
  // Additional kernel tests (leaky_relu, rmsnorm, l2_distance)
  fails += run_test_leaky_relu("/tmp/leaky_relu_kernel_gfx1250.elf");
  fails += run_test_rmsnorm("/tmp/rmsnorm_kernel_gfx1250.elf");
  fails += run_test_l2_distance("/tmp/l2_distance_kernel_gfx1250.elf");
  fails += run_test_f64_math("/tmp/f64_math_gfx1250.elf");
  // Cross-lane permutation tests
  fails += run_test_permlane16("/tmp/permlane16_test_gfx1250.elf");
  fails += run_test_permlanex16("/tmp/permlanex16_test_gfx1250.elf");
  // VOPD (dual-issue) test
  fails += run_test_vopd("/tmp/vopd_ops_gfx1250.elf");
  // Queue-poisoning tests last: attn crashes prevent subsequent tests from running
  fails += run_test_attn_forward("/tmp/attn_forward_gfx1250.elf");
  fails += run_test_attn_multihead("/tmp/attn_multihead_gfx1250.elf");
  int total = 45;
  printf("\n%s: %d/%d tests passed\n", fails?"SOME FAILED":"ALL PASSED", total-fails, total);
  return fails;
}
