/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc.

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

#ifndef HIP_SRC_HIP_PROF_API_H
#define HIP_SRC_HIP_PROF_API_H

#include <atomic>
#include <cassert>
#include <iostream>
#include <shared_mutex>
#include <utility>

#include "hip/amd_detail/hip_prof_str.h"
#include "platform/prof_protocol.h"

struct hip_api_trace_data_t {
  hip_api_data_t api_data;
  uint64_t phase_enter_timestamp;
  uint64_t phase_data;

  void (*phase_enter)(hip_api_id_t operation_id, hip_api_trace_data_t* data);
  void (*phase_exit)(hip_api_id_t operation_id, hip_api_trace_data_t* data);
};

// HIP API callbacks spawner object macro
#define HIP_CB_SPAWNER_OBJECT(operation_id)                                                        \
  api_callbacks_spawner_t<HIP_API_ID_##operation_id> __api_tracer(                                 \
      [=](auto& api_data) { INIT_CB_ARGS_DATA(operation_id, api_data); });

template <hip_api_id_t operation_id> class api_callbacks_spawner_t {
 public:
  template <typename Functor> api_callbacks_spawner_t(Functor init_cb_args_data) {
    static_assert(operation_id >= HIP_API_ID_FIRST && operation_id <= HIP_API_ID_LAST,
                  "invalid HIP_API operation id");

    if (auto function = amd::activity_prof::report_activity.load(std::memory_order_relaxed);
        function &&
        (enabled_ = function(ACTIVITY_DOMAIN_HIP_API, operation_id, &trace_data_) == 0)) {
      amd::activity_prof::correlation_id = trace_data_.api_data.correlation_id;

      if (trace_data_.phase_enter != nullptr) {
        init_cb_args_data(trace_data_.api_data);
        trace_data_.phase_enter(operation_id, &trace_data_);
      }
    }
  }

  ~api_callbacks_spawner_t() {
    if (enabled_) {
      if (trace_data_.phase_exit != nullptr) trace_data_.phase_exit(operation_id, &trace_data_);
      amd::activity_prof::correlation_id = 0;
    }
  }

 private:
  bool enabled_{false};
  union {
    hip_api_trace_data_t trace_data_;
  };
};

template <> class api_callbacks_spawner_t<HIP_API_ID_NONE> {
 public:
  template <typename Functor> api_callbacks_spawner_t(Functor) {}
};
#endif  // HIP_SRC_HIP_PROF_API_H
