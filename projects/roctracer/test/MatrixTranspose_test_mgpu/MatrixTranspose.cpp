/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>

// roctracer extension API
#include <inc/roctracer.h>

// hip header file
#include <hip/hip_runtime.h>

#ifndef ITERATIONS
# define ITERATIONS 1
#endif
#define WIDTH 1024


#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];
}

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

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    int i;
    int errors;

    int gpuCount = 0;
    hipGetDeviceCount(&gpuCount);
    std::cout << "Number of GPUs: " << gpuCount << std::endl;

    init_tracing();

    while (iterations-- > 0) {
        start_tracing();

        Matrix = (float*)malloc(NUM * sizeof(float));
        TransposeMatrix = (float*)malloc(NUM * sizeof(float));
        cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

        // initialize the input data
        for (i = 0; i < NUM; i++) {
            Matrix[i] = (float)i * 10.0f;
        }

        for (i = 0; i < gpuCount; ++i) {
          // switch GPU.
          hipSetDevice(i);

          hipDeviceProp_t devProp;
          hipGetDeviceProperties(&devProp, 0);
          std::cout << "Device name " << devProp.name << std::endl;

          // allocate the memory on the device side
          hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
          hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

          // Memory transfer from host to device
          hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

          // Lauching kernel from host
          hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                          dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                          gpuMatrix, WIDTH);

          hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);

          hipStreamSynchronize(0);

          // free the resources on device side
          hipFree(gpuMatrix);
          hipFree(gpuTransposeMatrix);
        }

        // CPU MatrixTranspose computation
        matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

        // verify the results
        errors = 0;
        double eps = 1.0E-6;
        for (i = 0; i < NUM; i++) {
            if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
                errors++;
            }
        }
        if (errors != 0) {
            printf("FAILED: %d errors\n", errors);
        } else {
            printf("PASSED!\n");
        }

        // free the resources on host side
        free(Matrix);
        free(TransposeMatrix);
        free(cpuTransposeMatrix);

        stop_tracing();
    }

    return errors;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HIP Callbacks/Activity tracing
//
#if 1
#include <inc/roctracer_hip.h>
#include <inc/roctracer_hcc.h>

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      std::cerr << roctracer_error_string() << std::endl << std::flush;                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

// Runtime API callback function
void api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  std::cout << "### api_callback IN\n";
  (void)arg;

  //if (domain == ACTIVITY_DOMAIN_ROCTX) {
  //  const roctx_api_data_t* data = reinterpret_cast<const roctx_api_data_t*>(callback_data);
  //  fprintf(stdout, "ROCTX: \"%s\"\n", data->args.message);
  //  return;
  //}

  if (domain == ACTIVITY_DOMAIN_HCC_OPS) {
    fprintf(stdout, "HCC OPS\n");
    return;
  }

  if (domain == ACTIVITY_DOMAIN_HSA_API) {
    fprintf(stdout, "HSA API\n");
    return;
  }

  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  fprintf(stdout, "<%s id(%u)\tcorrelation_id(%lu) %s> ",
    roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0),
    cid,
    data->correlation_id,
    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        fprintf(stdout, "ptr(%p) size(0x%x)",
          data->args.hipMalloc.ptr,
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        fprintf(stdout, "ptr(%p)",
          data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        fprintf(stdout, "kernel(\"%s\") stream(%p)",
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
  } else {
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        fprintf(stdout, "*ptr(0x%p)",
          *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
  }
  fprintf(stdout, "\n"); fflush(stdout);
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
  std::cout << "### activity_callback IN\n";
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);
  fprintf(stdout, "\tActivity records:\n"); fflush(stdout);
  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    fprintf(stdout, "\tdomain(%u)", record->domain);
    fprintf(stdout, "\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)",
      name,
      record->correlation_id,
      record->begin_ns,
      record->end_ns
    );
    if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
      fprintf(stdout, " process_id(%u) thread_id(%u)",
        record->process_id,
        record->thread_id
      );
    } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
      fprintf(stdout, " device_id(%d) queue_id(%lu)",
        record->device_id,
        record->queue_id
      );
      if (record->op == HIP_OP_ID_COPY) fprintf(stdout, " bytes(0x%zx)", record->bytes);
    } else if (record->domain == ACTIVITY_DOMAIN_EXT_API) {
      fprintf(stdout, " external_id(%lu)",
        record->external_id
      );
    } else {
      fprintf(stderr, "Bad domain %d\n", record->domain);
      //abort();
    }
    fprintf(stdout, "\n");
    fflush(stdout);
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
}

// Init tracing routine
void init_tracing() {
  std::cout << "# INIT #############################" << std::endl << std::flush;
  // Allocating tracing pool
  roctracer_properties_t properties{};
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  properties.buffer_callback_arg = &properties;
  ROCTRACER_CALL(roctracer_open_pool(&properties));
  // Enable API callbacks
  ROCTRACER_CALL(roctracer_enable_callback(api_callback, NULL));
  // Enable activity tracing
  ROCTRACER_CALL(roctracer_enable_activity());
}

// Start tracing routine
void start_tracing() {
  std::cout << "# START (" << iterations << ") #############################" << std::endl << std::flush;
}

// Stop tracing routine
void stop_tracing() {
  ROCTRACER_CALL(roctracer_disable_callback());

  ROCTRACER_CALL(roctracer_disable_activity());
  ROCTRACER_CALL(roctracer_flush_activity());
  std::cout << "# STOP  #############################" << std::endl << std::flush;
}
#else
void init_tracing() {}
void start_tracing() {}
void stop_tracing() {}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
