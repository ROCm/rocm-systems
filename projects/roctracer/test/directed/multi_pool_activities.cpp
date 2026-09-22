/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include <hip/hip_runtime.h>
#include <roctracer.h>
#include <roctracer_hip.h>

#include <iostream>

// This test checks that asynchronous activities can be enabled in distinct memory pools. It enables
// activity reporting for HIP kernel dispatches in one memory pool, and memory copy reporting in
// another memory pool. The output of this test to stdout should be a series of kernel dispatch
// records (10) followed by a series of memory copy records (10). The records should not be
// interleaved.

__global__ void kernel(void* global_memory) {}

namespace {

template <typename T> inline void CHECK(T status);

template <> inline void CHECK(hipError_t err) {
  if (err != hipSuccess) {
    std::cerr << hipGetErrorString(err) << std::endl;
    abort();
  }
}

template <> inline void CHECK(roctracer_status_t status) {
  if (status != ROCTRACER_STATUS_SUCCESS) {
    std::cerr << roctracer_error_string() << std::endl;
    abort();
  }
}

void buffer_callback(const char* begin, const char* end, void* arg) {
  for (const roctracer_record_t* record = (const roctracer_record_t*)begin;
       record != (const roctracer_record_t*)end; CHECK(roctracer_next_record(record, &record))) {
    fprintf(stdout, "\t:%s\t: correlation_id(%lu) time_ns(%lu:%lu)\n",
            roctracer_op_string(record->domain, record->op, record->kind), record->correlation_id,
            record->begin_ns, record->end_ns);
  }
}

}  // namespace

int main() {
  CHECK(hipSetDevice(0));

  roctracer_properties_t properties{};
  properties.buffer_callback_fun = buffer_callback;
  properties.buffer_callback_arg = nullptr;
  properties.buffer_size = 1024 * 1024;

  roctracer_pool_t* pool_1;
  CHECK(roctracer_open_pool_expl(&properties, &pool_1));
  CHECK(roctracer_enable_op_activity_expl(ACTIVITY_DOMAIN_HIP_OPS, HIP_OP_ID_DISPATCH, pool_1));

  roctracer_pool_t* pool_2;
  CHECK(roctracer_open_pool_expl(&properties, &pool_2));
  CHECK(roctracer_enable_op_activity_expl(ACTIVITY_DOMAIN_HIP_OPS, HIP_OP_ID_COPY, pool_2));
  CHECK(roctracer_enable_op_activity_expl(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMemcpy, pool_2));

  int host_array[256] = {0};
  int* device_memory;
  CHECK(hipMalloc(&device_memory, sizeof(host_array)));

  for (int i = 0; i < 10; ++i) {
    CHECK(hipMemcpy(device_memory, host_array, sizeof(host_array), hipMemcpyHostToDevice));
    kernel<<<1, 1>>>(device_memory);
  }
  CHECK(hipDeviceSynchronize());

  CHECK(roctracer_flush_activity_expl(pool_1));
  CHECK(roctracer_flush_activity_expl(pool_2));
  return 0;
}
