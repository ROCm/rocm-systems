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
  fails += run_test_attn_forward("/tmp/attn_forward_gfx1250.elf");
  fails += run_test_attn_multihead("/tmp/attn_multihead_gfx1250.elf");
  int total = 20;
  printf("\n%s: %d/%d tests passed\n", fails?"SOME FAILED":"ALL PASSED", total-fails, total);
  return fails;
}
