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
#define HIP_PROF_HIP_API_STRING 1
#include <roctracer_hip.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>

__global__ void kernel() {}

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

namespace {

uint32_t GetPid() {
  static auto pid = syscall(__NR_getpid);
  return pid;
}
uint32_t GetTid() {
  static thread_local auto tid = syscall(__NR_gettid);
  return tid;
}

void hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  const hip_api_data_t* data = static_cast<const hip_api_data_t*>(callback_data);
  fprintf(stdout, "<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)>\n",
          roctracer_op_string(domain, cid, 0), cid, data->correlation_id,
          (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit", GetPid(), GetTid());
}

void buffer_callback(const char* begin, const char* end, void* arg) {
  for (const roctracer_record_t* record = (const roctracer_record_t*)begin;
       record < (const roctracer_record_t*)end; CHECK(roctracer_next_record(record, &record))) {
    fprintf(stdout, "\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)\n",
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
  properties.buffer_size = 1024;
  CHECK(roctracer_open_pool(&properties));

  // 1: callbacks only
  CHECK(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, hip_api_callback, nullptr));
  CHECK(hipSetDevice(0));
  kernel<<<1, 1>>>();
  CHECK(hipDeviceSynchronize());
  CHECK(roctracer_flush_activity());

  // 2: callbacks and activities
  CHECK(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
  CHECK(hipSetDevice(0));
  kernel<<<1, 1>>>();
  CHECK(hipDeviceSynchronize());
  CHECK(roctracer_flush_activity());

  // 3: activities only
  CHECK(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
  CHECK(hipSetDevice(0));
  kernel<<<1, 1>>>();
  CHECK(hipDeviceSynchronize());
  CHECK(roctracer_flush_activity());

  // 4: callbacks only
  CHECK(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, hip_api_callback, nullptr));
  CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
  CHECK(hipSetDevice(0));
  kernel<<<1, 1>>>();
  CHECK(hipDeviceSynchronize());
  CHECK(roctracer_flush_activity());

  // 5: callbacks and activities
  CHECK(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
  CHECK(hipSetDevice(0));
  kernel<<<1, 1>>>();
  CHECK(hipDeviceSynchronize());
  CHECK(roctracer_flush_activity());

  // 6: callbacks only
  CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
  CHECK(hipSetDevice(0));
  kernel<<<1, 1>>>();
  CHECK(hipDeviceSynchronize());
  CHECK(roctracer_flush_activity());

  // 7: none
  CHECK(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
  CHECK(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
  CHECK(hipSetDevice(0));
  kernel<<<1, 1>>>();
  CHECK(hipDeviceSynchronize());
  CHECK(roctracer_flush_activity());

  return 0;
}