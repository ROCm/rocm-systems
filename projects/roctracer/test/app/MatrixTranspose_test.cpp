/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <cstdlib>
using namespace std;
#else
#include <stdlib.h>
#endif

// roctx header file
#include <roctx.h>
// roctracer extension API
#include <roctracer_ext.h>

#ifdef __cplusplus
static thread_local const size_t msg_size = 512;
static thread_local char* msg_buf = NULL;
static thread_local char* message = NULL;
#else
static const size_t msg_size = 512;
static char* msg_buf = NULL;
static char* message = NULL;
#endif
void SPRINT(const char* fmt, ...) {
  if (msg_buf == NULL) {
    msg_buf = (char*)calloc(msg_size, 1);
    message = msg_buf;
  }

  va_list args;
  va_start(args, fmt);
  message += vsnprintf(message, msg_size - (message - msg_buf), fmt, args);
  va_end(args);
}
void SFLUSH() {
  if (msg_buf == NULL) abort();
  message = msg_buf;
  msg_buf[msg_size - 1] = 0;
  fprintf(stdout, "%s", msg_buf);
  fflush(stdout);
}

#if HIP_TEST
// hip header file
#include <hip/hip_runtime.h>
// Macro to call HIP API
#define CALL_HIP(call)                                                                             \
  do {                                                                                             \
    call;                                                                                          \
  } while (0);
#define CHECK_HIP(call)                                                                            \
  do {                                                                                             \
    hipError_t err = call;                                                                         \
    if (err != hipSuccess) {                                                                       \
      fprintf(stderr, "%s\n", hipGetErrorString(err));                                             \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)
#else
#define CALL_HIP(call)                                                                             \
  do {                                                                                             \
  } while (0)
#define CHECK_HIP(call)                                                                            \
  do {                                                                                             \
  } while (0)
#endif

#ifndef ITERATIONS
#define ITERATIONS 101
#endif
#define WIDTH 1024
#define NUM (WIDTH * WIDTH)
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

#if HIP_TEST
// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  out[y * width + x] = in[x * width + y];
}
#endif

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
  for (unsigned int j = 0; j < width; j++) {
    for (unsigned int i = 0; i < width; i++) {
      output[i * width + j] = input[j * width + i];
    }
  }
}

int iterations = ITERATIONS;
void init_tracing();
void start_tracing();
void stop_tracing();

int main() {
  float* Matrix;
  float* TransposeMatrix;
  float* cpuTransposeMatrix;

#if HIP_TEST
  float* gpuMatrix;
  float* gpuTransposeMatrix;
#endif

  int i;
  int errors = 0;

  init_tracing();

#if HIP_TEST
  int gpuCount = 1;
#if MGPU_TEST
  hipGetDeviceCount(&gpuCount);
  fprintf(stderr, "Number of GPUs: %d\n", gpuCount);
#endif
  iterations *= gpuCount;
#endif

  while (iterations-- > 0) {
    start_tracing();

#if HIP_TEST
    // set GPU
    const int devIndex = iterations % gpuCount;
    hipSetDevice(devIndex);

    hipDeviceProp_t devProp;
    CHECK_HIP(hipGetDeviceProperties(&devProp, 0));
    fprintf(stderr, "Device %d name: %s\n", devIndex, devProp.name);
#endif

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
      Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    CHECK_HIP(hipMalloc((void**)&gpuMatrix, NUM * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float)));

    // correlation reagion32
    roctracer_activity_push_external_correlation_id(31);
    // correlation reagion32
    roctracer_activity_push_external_correlation_id(32);

    // Memory transfer from host to device
    CHECK_HIP(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

    // correlation reagion33
    roctracer_activity_push_external_correlation_id(33);

    roctxMark("before hipLaunchKernel");
    roctxRangePush("hipLaunchKernel");

    // Lauching kernel from host
    CALL_HIP(hipLaunchKernelGGL(matrixTranspose,
                                dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                                dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0,
                                gpuTransposeMatrix, gpuMatrix, WIDTH));

    roctxMark("after hipLaunchKernel");

    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);

    // Memory transfer from device to host
    roctxRangePush("hipMemcpy");

    CHECK_HIP(
        hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));

    roctxRangePop();  // for "hipMemcpy"
    roctxRangePop();  // for "hipLaunchKernel"

    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);

    // CPU MatrixTranspose computation
#if HIP_TEST
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
      if (abs((double)TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
        errors++;
      }
    }
    if (errors != 0) {
      fprintf(stderr, "FAILED: %d errors\n", errors);
    } else {
      errors = 0;
      fprintf(stderr, "PASSED!\n");
    }
#endif

    // free the resources on device side
    CHECK_HIP(hipFree(gpuMatrix));
    CHECK_HIP(hipFree(gpuTransposeMatrix));

    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);
    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);
  }

  stop_tracing();

  return errors;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HIP Callbacks/Activity tracing
//
#if 1
#include <roctracer_hip.h>
#include <roctracer_hsa.h>
#include <roctracer_roctx.h>

#include <unistd.h>
#include <sys/syscall.h> /* For SYS_xxx definitions */

// Macro to check ROC-tracer calls status
#define CHECK_ROCTRACER(call)                                                                      \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      fprintf(stderr, "%s\n", roctracer_error_string());                                           \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

static inline uint32_t GetTid() { return syscall(__NR_gettid); }
static inline uint32_t GetPid() { return syscall(__NR_getpid); }


// Runtime API callback function
void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  (void)arg;

  if (domain == ACTIVITY_DOMAIN_ROCTX) {
    const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
    fprintf(stdout, "rocTX <\"%s pid(%d) tid(%d)\">\n", data->args.message, GetPid(), GetTid());
    return;
  }
  const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
  SPRINT("<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)> ",
         roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0), cid, data->correlation_id,
         (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit", GetPid(), GetTid());
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        SPRINT("dst(%p) src(%p) size(0x%x) kind(%u)", data->args.hipMemcpy.dst,
               data->args.hipMemcpy.src, (uint32_t)(data->args.hipMemcpy.sizeBytes),
               (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        SPRINT("ptr(%p) size(0x%x)", data->args.hipMalloc.ptr,
               (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        SPRINT("ptr(%p)", data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        SPRINT("kernel(\"%s\") stream(%p)", hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
               data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
  } else {
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        SPRINT("*ptr(0x%p)", *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
  }
  SPRINT("\n");
  SFLUSH();
}
// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

  SPRINT("\tActivity records:\n");
  while (record < end_record) {
    const char* name = roctracer_op_string(record->domain, record->op, record->kind);
    SPRINT("\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)", name, record->correlation_id,
           record->begin_ns, record->end_ns);
    if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
      SPRINT(" process_id(%u) thread_id(%u)", record->process_id, record->thread_id);
    } else if (record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
      SPRINT(" device_id(%d) queue_id(%lu)", record->device_id, record->queue_id);
      if (record->op == HIP_OP_ID_COPY) SPRINT(" bytes(0x%zx)", record->bytes);
    } else if (record->domain == ACTIVITY_DOMAIN_HSA_OPS) {
      SPRINT(" se(%u) cycle(%lu) pc(%lx)", record->pc_sample.se, record->pc_sample.cycle,
             record->pc_sample.pc);
    } else if (record->domain == ACTIVITY_DOMAIN_EXT_API) {
      SPRINT(" external_id(%lu)", record->external_id);
    } else {
      fprintf(stdout, "Bad domain %d\n\n", record->domain);
      abort();
    }
    SPRINT("\n");
    SFLUSH();

    CHECK_ROCTRACER(roctracer_next_record(record, &record));
  }
}

// Init tracing routine
void init_tracing() {
  fprintf(stderr, "# INIT #############################\n");
  // roctracer properties
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);
  // Allocating tracing pool
  roctracer_properties_t properties;
  memset(&properties, 0, sizeof(roctracer_properties_t));
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  CHECK_ROCTRACER(roctracer_open_pool(&properties));
  // Enable HIP API callbacks
  CHECK_ROCTRACER(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL));
  // Enable HIP activity tracing
#if HIP_API_ACTIVITY_ON
  CHECK_ROCTRACER(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
#endif
  CHECK_ROCTRACER(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  // Enable PC sampling
  CHECK_ROCTRACER(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1));
  // Enable rocTX
  CHECK_ROCTRACER(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL));
}

// Start tracing routine
void start_tracing() {
  fprintf(stderr, "# START (%d) #############################\n", iterations);
  // Start
  if ((iterations & 1) == 1)
    roctracer_start();
  else
    roctracer_stop();
}

// Stop tracing routine
void stop_tracing() {
  CHECK_ROCTRACER(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
#if HIP_API_ACTIVITY_ON
  CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
#endif
  CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
  CHECK_ROCTRACER(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));
  CHECK_ROCTRACER(roctracer_flush_activity());
  fprintf(stderr, "# STOP  #############################\n");
}
#else
void init_tracing() {}
void start_tracing() {}
void stop_tracing() {}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
