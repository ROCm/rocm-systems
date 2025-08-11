# ROC Tracer / ROC-TX Libraries Specification
```
ROC Tracer API version 2
ROC-TX API version 1

- The rocTracer API is agnostic to specific runtime and may trace
the runtime API calls and asynchronous GPU activity.
- The rocTX API is provided for application code annotation.
```
## 1. High level overview
```
The goal of the implementation is to provide a runtime independent API
for tracing of runtime calls and asynchronous activity, like GPU kernel
dispatches and memory moves. The tracing includes callback API for
runtime API tracing and activity API for asynchronous activity records
logging.

Depending on particular runtime intercepting mechanism, the rocTracer
library can be dynamically linked, dynamically loaded by the runtime as
a plugin or some API wrapper can be loaded using LD_PRELOAD.
The library has a C API.

The rocTracer library is an API that intercepts runtime API calls and
traces asynchronous activity. The activity tracing results are recorded
in a ring buffer.

The rocTX contains application code instrumentation API to support high
level correlation of runtime API/activity events. The API includes mark
and nested ranges.
```
## 2. General API
### 2.1. Description
```
The library supports method for getting the error number and error string
of the last failed library API call. It allows to check the conformance
of used library API header and the library binary, the version macros and
API methods can be used.

Returning the error and error string methods:
•	roctracer_status_t – error code enumeration
•	roctracer_error_string – method for returning the error string

Library version:
•	ROCTRACER_VERSION_MAJOR – API major version macro
•	ROCTRACER_VERSION_MINOR – API minor version macro
•	roctracer_version_major – library major version
•	roctracer_version_minor – library minor version
```
### 2.2. Error codes and error string methods
```
Error code enumeration:
typedef enum {
   ROCTRACER_STATUS_SUCCESS = 0,
   ROCTRACER_STATUS_ERROR = 1,
   ROCTRACER_STATUS_UNINIT = 2,
   ROCTRACER_STATUS_BREAK = 3,
   ROCTRACER_STATUS_BAD_DOMAIN = 4,
   ROCTRACER_STATUS_BAD_PARAMETER = 5,
   ROCTRACER_STATUS_HIP_API_ERR = 6,
   ROCTRACER_STATUS_HCC_OPS_ERR = 7,
   ROCTRACER_STATUS_ROCTX_ERR = 8,
} roctracer_status_t;

Return error string:
const char* roctracer_error_string();
```
### 2.3. Library version
```
The library provides major and minor versions. Major version is for
incompatible API changes and minor version for bug fixes.

API version macros defined in the library API header ‘roctracer.h’:
ROCTRACER_VERSION_MAJOR
ROCTRACER_VERSION_MINOR

Methods to check library major and minor venison:
uint32_t roctracer_major_version();
uint32_t roctracer_minor_version();
```
## 3. Frontend API
### 3.1. Description
```
The rocTracer provides support for runtime API callbacks and activity
records logging. The APIs of different runtimes at different levels
are considered as different API domains with assigned domain IDs. For
example, language level and driver level. The API callbacks provide
the API calls arguments and are called on  two phases on “enter” and
on “exit”. The activity records are logged to the ring buffer and can
be associated with the respective API calls using the correlation ID.
Activity API can be used to enable collecting of the records with
timestamping data for API calls and asynchronous activity like the
kernel submits, memory copies and barriers

Tracing domains:
•	roctracer_domain_t – runtime API domains, HIP, HSA, etc…
•	roctracer_op_string – Return Op string by given domain and
                              activity Op code
•	roctracer_op_code –  Return Op code and kind by given string

Callback API:
•	roctracer_rtapi_callback_t  – runtime API callback type
•	roctracer_enable_op_callback – enable runtime API callback
                                       by domain and Op code
•	roctracer_enable_domain_callback – enable runtime API callback
                                           by domain for all Ops
•	roctracer_enable_callback – enable runtime API callback for
                                    all domains, all Ops
•	roctracer_disable_op_callback – disable runtime API callback
                                        by domain and Op code
•	roctracer_enable_op_callback – enable runtime API callback
                                       by domain for all Ops
•	roctracer_enable_op_callback – enable runtime API callback for
                                       all domains, all Ops

Activity API:
•	roctracer_record_t – activity record
•	roctracer_pool_t – records pool type
•	roctracer_allocator_t – tracer allocator type
•	roctracer_buffer_callback_t – pool callback type
•	roctracer_open_pool[_expl] – create records pool
•	roctracer_close_pool[_expl] – close records pool
•	roctracer_default_pool[_expl] – get/set default pool
•	roctracer_properties_t – tracer properties
•	roctracer_enable_op_activity[_expl] – enable activity records logging
•	roctracer_enable_domain_activity[_expl] – enable activity records logging
•	roctracer_enable_activity[_expl] – enable activity records logging
•	roctracer_disable_op_activity – disable activity records logging
•	roctracer_disable_domain_activity – disable activity records logging
•	roctracer_disable_activity – disable activity records logging
•	roctracer_flush_activity[_expl] – disable activity records logging
•	roctracer_next_record – return next record
•	roctracer_get_timestamp – return correlated GPU/CPU system timestamp

External correlation ID API:
•	roctracer_activity_push_external_correlation_id - push an external
                                                          correlation id for the calling thread
•	roctracer_activity_pop_external_correlation_id - pop an external
                                                         correlation id for the calling thread

Tracing control API:
•	roctracer_start – tracing start
•	roctracer_stop – tracer stop

```
### 3.2. Tracing Domains
```
Various tracing domains are supported. Each domain is assigned with
a domain ID. The domains include HSA, HIP runtime levels.

Traced API domains:
typedef enum {
   ACTIVITY_DOMAIN_HSA_API = 0,         // HSA API domain
   ACTIVITY_DOMAIN_HSA_OPS = 1,         // HSA async activity domain
   ACTIVITY_DOMAIN_HIP_API = 2,         // HIP API domain
   ACTIVITY_DOMAIN_HIP_OPS = 3,         // HIP async activity domain
   ACTIVITY_DOMAIN_KFD_API = 4,         // KFD API domain
   ACTIVITY_DOMAIN_EXT_API = 5,         // External ID domain
   ACTIVITY_DOMAIN_ROCTX   = 6,         // ROCTX domain
   ACTIVITY_DOMAIN_NUMBER = 7
} activity_domain_t;

Return name by given domain and Op code:
const char* roctracer_op_string(  // NULL returned on error and error number is set
   uint32_t domain,		  // tracing domain
   uint32_t op,	                  // activity op code
   uint32_t kind);                // activity kind
Return Op code and kind by given string:
roctracer_status_t roctracer_op_code(
    uint32_t domain,              // tracing domain
    const char* str,              // [in] op string
    uint32_t* op,                 // [out] op code
    uint32_t* kind);              // [out] op kind code if not NULL
```
### 3.3. Callback API
```
The tracer provides support for runtime API callbacks and activity records
logging. The API callbacks provide the API calls arguments and are called
on two phases on “enter”, on “exit”.

API phase passed to the callbacks:
typedef enum {
   ROCTRACER_API_PHASE_ENTER,
   ROCTRACER_API_PHASE_EXIT,
} roctracer_api_phase_t;

Runtime API callback type:
typedef void  (*roctracer_rtapi_callback_t)(
   uint32_t domain,   // runtime API domain
   uint32_t cid,      // API call ID
   const void* data,  // [in] callback data with correlation id and the call
		      // arguments
   void* arg);        // [in/out] user passed data

Enable runtime API callbacks:
roctracer_status_t roctracer_enable_op_callback(
   activity_domain_t domain,             // tracing domain
   uint32_t op,                          // API call ID
   activity_rtapi_callback_t callback,   // callback function pointer
   void* arg);                           // [in/out] callback arg

roctracer_status_t roctracer_enable_domain_callback(
   activity_domain_t domain,             // tracing domain
   activity_rtapi_callback_t callback,   // callback function pointer
    void* arg);                          // [in/out] callback arg


roctracer_status_t roctracer_enable_callback(
   activity_rtapi_callback_t callback,   // callback function pointer
   void* arg);                           // [in/out] callback arg

Disable runtime API callbacks:
roctracer_status_t roctracer_disable_op_callback(
    activity_domain_t domain,           // tracing domain
    uint32_t op);                       // API call ID

roctracer_status_t roctracer_disable_domain_callback(
    activity_domain_t domain);          // tracing domain

roctracer_status_t roctracer_disable_callback();
```

### 3.4 Activity API

The activity records are asynchronously logged to the pool and can be
associated with the respective API callbacks using the correlation ID.
Activity API can be used to enable collecting  the records with
timestamp data for API calls and GPU activity like kernel submits,
memory copies, and barriers.

```
// Correlation id
typedef uint64_t activity_correlation_id_t;

Activity record type:

// Activity record type
struct activity_record_t {
   uint32_t domain;                           // activity domain id
   activity_kind_t kind;                      // activity kind
   activity_op_t op;                          // activity op
   activity_correlation_id_t correlation_id;  // activity ID
   uint64_t begin_ns;                         // host begin timestamp
   uint64_t end_ns;                           // host end timestamp
   union {
      struct {
         int device_id;                       // device id
         uint64_t queue_id;                   // queue id
      };
      struct {
         uint32_t process_id;                 // device id
         uint32_t thread_id;                  // thread id
      };
      struct {
        activity_correlation_id_t external_id; // external correlation id
      };
   };
   size_t bytes;                              // data size bytes
};
```

> [!NOTE]
> rocprofiler reported device ids are in sync with node-ids reported by KFD(kernel). This can easily be verified by `rocm-smi` under `Node`.
> Please also note that this device id might not be in sync with the ones provided by `hipGetDeviceProperties` which includes CPU agents and starts from 0.

```
Return next record:
static inline int roctracer_next_record(
   const activity_record_t* record,         // [in] record ptr
   const activity_record_t** next);         // [out] next record ptr

Tracer allocator type:
typedef void (*roctracer_allocator_t)(
   char** ptr,       	// memory pointer
   size_t size,        // memory size
   void* arg);         // allocator arg

Pool callback type:
typedef void (*roctracer_buffer_callback_t)(
   const char* begin,   // [in] available buffered trace records
   const char* end,     // [in] end of buffered trace records
   void* arg);          // [in/out] callback arg

Tracer properties:
typedef struct {
   uint32_t mode;                                    // roctracer mode
   size_t buffer_size;                               // buffer size
                                                     // power of 2
   roctracer_allocator_t alloc_fun;                  // memory allocator
                                                     // function pointer
   void* alloc_arg;                                  // memory allocator
                                                     // function pointer
   roctracer_buffer_callback_t buffer_callback_fun;  // tracer record
                                                     // callback function
   void* buffer_callback_arg;                        // tracer record
                                                     // callback arg
} roctracer_properties_t;

Tracer memory pool handle type:
typedef void roctracer_pool_t;

Create tracer memory pool:
roctracer_status_t roctracer_open_pool(
   const roctracer_properties_t* properties); // tracer pool properties

roctracer_status_t roctracer_open_pool_expl(
   const roctracer_properties_t* properties, // tracer pool properties
   roctracer_pool_t** pool);                 // [out] returns tracer pool if
                                             // not NULL, otherwise sets the
                                             // default one if it is not set
                                             // yet; otherwise the error is
                                             // generated

Close tracer memory pool:
roctracer_status_t roctracer_close_pool();

roctracer_status_t roctracer_close_pool_expl(
   roctracer_pool_t* pool);          // memory pool, NULL means default pool

Return current default pool. Set new default pool if the argument is not NULL:
roctracer_pool_t* roctracer_default_pool();

roctracer_pool_t* roctracer_default_pool_expl(
   roctracer_pool_t* pool);          // new default pool if not NULL
```
Enable activity records logging:
```
roctracer_status_t roctracer_enable_op_activity(
   activity_domain_t domain,         // tracing domain
   uint32_t op);                     // activity op ID

roctracer_status_t roctracer_enable_op_activity_expl(
   activity_domain_t domain,         // tracing domain
   uint32_t op,                      // activity op ID
   roctracer_pool_t* pool);          // memory pool, NULL means default pool

roctracer_status_t roctracer_enable_domain_activity(
   activity_domain_t domain);        // tracing domain

roctracer_status_t roctracer_enable_domain_activity_expl(
   activity_domain_t domain,         // tracing domain
   roctracer_pool_t* pool);          // memory pool, NULL means default pool

roctracer_status_t roctracer_enable_activity();

roctracer_status_t roctracer_enable_activity_expl(
   roctracer_pool_t* pool);          // memory pool, NULL means default pool

Disable activity records logging:
roctracer_status_t roctracer_disable_op_activity(
   activity_domain_t domain,         // tracing domain
   uint32_t op);                     // activity op ID

roctracer_status_t roctracer_disable_domain_activity(
   activity_domain_t domain);        // tracing domain

roctracer_status_t roctracer_disable_activity();

Flush available activity records:
roctracer_status_t roctracer_flush_activity();

roctracer_status_t roctracer_flush_activity_expl(
   roctracer_pool_t* pool);          // memory pool, NULL means default pool

Return correlated GPU/CPU system timestamp:
roctracer_status_t roctracer_get_timestamp(
    uint64_t* timestamp);            // [out] return timestamp
```
External correlation ID API
```
The API provides activity records to associate rocTracer correlation IDs with
IDs provided by external APIs. The external ID records are identified by
ACTIVITY_DOMAIN_EXT_API domain value.
Using the ‘push’ method an external ID is pushed to a per CPU thread stack and
the ‘pop’ method can be used to remove the last pushed ID.
An external ID record is inserted before any generated rocTracer activity record
if the same CPU external ID stack is non-empty.

Notifies that the calling thread is entering an external API region.
Push an external correlation id for the calling thread.
roctracer_status_t roctracer_activity_push_external_correlation_id(
    activity_correlation_id_t id);        // external correlation id

Notifies that the calling thread is leaving an external API region.
Pop an external correlation id for the calling thread.
roctracer_status_t roctracer_activity_pop_external_correlation_id(
    activity_correlation_id_t* last_id);  // returns the last external correlation id
                                          // if not NULL
```
Tracing control API
```
Tracing start:
void roctracer_start();

Tracing stop:
void roctracer_stop();
```
## 4. rocTracer Usage Code Examples
### 4.1. HIP API ops, GPU Activity Tracing
```
#include <roctracer/roctracer_hip.h>

// HIP API callback function
void hip_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
   (void)arg;
   const hip_api_data_t* data = reinterpret_cast <const hip_api_data_t*>
     (callback_data);
   fprintf(stdout, "<%s id(%u)\tcorrelation_id(%lu) %s> ",
         roctracer_id_string(ACTIVITY_DOMAIN_HIP_API, cid),
         cid,
         data->correlation_id,
        (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");
   <some code  . . .>
}

// Activity tracing callback
void activity_callback(const char* begin, const char* end, void* arg) {
   const roctracer_record_t* record = reinterpret_cast<const
                                      roctracer_record_t*>(begin);
   const roctracer_record_t* end_record = reinterpret_cast<const
                                          roctracer_record_t*>(end);
   fprintf(stdout, "\tActivity records:\n");
   while (record < end_record) {
      const char * name = roctracer_op_string(record->domain,
                                              record->activity_id, 0);
      fprintf(stdout, "\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)
              device_id(%d) stream_id(%lu)\n",
              name,
              record->correlation_id,
              record->begin_ns,
              record->end_ns,
              record->device_id,
              record->stream_id
              );
      <some code . . .>
      ROCTRACER_CALL(roctracer_next_record(record, &record));
   }
}

int main() {
   // Allocating tracing pool
   roctracer_properties_t properties{};
   properties.buffer_size = 12;
   properties.buffer_callback_fun = activity_callback;
   ROCTRACER_CALL(roctracer_open_pool(&properties));

   // Enable HIP API callbacks. HIP_API_ID_ANY can be used to trace all HIP
   // API calls.
   ROCTRACER_CALL(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API,
                                     HIP_API_ID_hipModuleLaunchKernel,
                                     hip_api_callback, NULL));
   ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HIP_API,
                                     HIP_API_ID_hipModuleLaunchKernel));
   // Enable HIP kernel dispatch activity tracing
   ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HIP_OPS,
                                               HIP_OP_ID_DISPATCH));

   <test code>

   // Disable tracing and closing the pool
   ROCTRACER_CALL(roctracer_disable_callback());
   ROCTRACER_CALL(roctracer_disable_activity());
   ROCTRACER_CALL(roctracer_close_pool());
}
```
### 4.2. MatrixTranspose HIP sample with all APIs/activity tracing enabled
```
This shows a MatrixTranspose HIP sample with enabled tracing of
all HIP API and all GPU asynchronous activity.

/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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

// hip header file
#include <hip/hip_runtime.h>

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#define WIDTH 1024


#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
// hipLaunchParm provides the execution configuration
__global__ void matrixTranspose(hipLaunchParm lp, float* out, float* in,
                                const int width) {
   int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

   out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned
                                 int width) {
   for (unsigned int j = 0; j < width; j++) {
      for (unsigned int i = 0; i < width; i++) {
         output[i * width + j] = input[j * width + i];
      }
   }
}

int iterations = ITERATIONS;
void start_tracing();
void stop_tracing();

int main() {
   float* Matrix;
   float* TransposeMatrix;
   float* cpuTransposeMatrix;

   float* gpuMatrix;
   float* gpuTransposeMatrix;

   hipDeviceProp_t devProp;
   hipGetDeviceProperties(&devProp, 0);

   std::cout << "Device name " << devProp.name << std::endl;

   int i;
   int errors;

   while (iterations-- > 0) {
      start_tracing();

      Matrix = (float*)malloc(NUM * sizeof(float));
      TransposeMatrix = (float*)malloc(NUM * sizeof(float));
      cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

      // initialize the input data
      for (i = 0; i < NUM; i++) {
         Matrix[i] = (float)i * 10.0f;
      }

      // allocate the memory on the device side
      hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
      hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

      // Memory transfer from host to device
      hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float),
                hipMemcpyHostToDevice);

      // Lauching kernel from host
      hipLaunchKernel(matrixTranspose,
                      dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH /
                           THREADS_PER_BLOCK_Y),
                      dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0,
                      gpuTransposeMatrix, gpuMatrix, WIDTH);

      // Memory transfer from device to host
      hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float),
                hipMemcpyDeviceToHost);

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

      // free the resources on device side
      hipFree(gpuMatrix);
      hipFree(gpuTransposeMatrix);

      // free the resources on host side
      free(Matrix);
      free(TransposeMatrix);
      free(cpuTransposeMatrix);

      stop_tracing();
   }

   return errors;
}

/////////////////////////////////////////////////////////////////////////////
// HIP Callbacks/Activity tracing
/////////////////////////////////////////////////////////////////////////////
#include <roctracer/roctracer_hip.h>

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                               \
   do {                                                                    \
      int err = call;                                                      \
      if (err != 0) {                                                      \
         std::cerr << roctracer_error_string() << std::endl << std::flush; \
         abort();                                                          \
      }                                                                    \
   } while (0)

// HIP API callback function
void hip_api_callback(
   uint32_t domain,
   uint32_t cid,
   const void* callback_data,
   void* arg)
{
   (void)arg;
   const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>
                                (callback_data);
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
            fprintf(stdout, "ptr(%p),
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
//   hipMalloc id(3) correlation_id(1):
//   begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
   const roctracer_record_t* record = reinterpret_cast
                                      <const roctracer_record_t*>(begin);
   const roctracer_record_t* end_record = reinterpret_cast
                                      <const roctracer_record_t*>(end);
   fprintf(stdout, "\tActivity records:\n"); fflush(stdout);
   while (record < end_record) {
      const char * name = roctracer_op_string(record->domain,
                                              record->activity_id, 0);
      fprintf(stdout, "\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu) \
                      device_id(%d) stream_id(%lu)",
              name,
              record->correlation_id,
              record->begin_ns,
              record->end_ns,
              record->device_id,
              record->stream_id
              );
      if (record->kind == hc::HSA_OP_ID_COPY)
         fprintf(stdout, " bytes(0x%zx)", record->bytes);
      fprintf(stdout, "\n");
      fflush(stdout);
      ROCTRACER_CALL(roctracer_next_record(record, &record));
   }
}

// Start tracing routine
void start_tracing() {
   std::cout << "# START #############################" << std::endl
             << std::flush;
   // Allocating tracing pool
   roctracer_properties_t properties{};
   properties.buffer_size = 0x1000;
   properties.buffer_callback_fun = activity_callback;
   ROCTRACER_CALL(roctracer_open_pool(&properties));
   // Enable API callbacks, all domains
   ROCTRACER_CALL(roctracer_enable_callback(hip_api_callback, NULL));
   // Enable activity tracing, all domains
   ROCTRACER_CALL(roctracer_enable_activity());
}

// Stop tracing routine
void stop_tracing() {
   ROCTRACER_CALL(roctracer_disable_api_callback());
   ROCTRACER_CALL(roctracer_disable_api_activity());
   ROCTRACER_CALL(roctracer_close_pool());
   std::cout << "# STOP  #############################" << std::endl
             << std::flush;
}
/////////////////////////////////////////////////////////////////////////////
```
## 5. rocTX application code annotation API
```
Basic annotation API: markers and nested ranges.
// A marker created by given ASCII massage
void roctxMark(const char* message);

// Returns the 0 based level of a nested range being started by given message associated to this range.
// A negative value is returned on the error.
int roctxRangePush(const char* message);

// Marks the end of a nested range.
// Returns the 0 based level the range.
// A negative value is returned on the error.
int roctxRangePop();
```
