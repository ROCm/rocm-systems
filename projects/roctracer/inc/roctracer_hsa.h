/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef INC_ROCTRACER_HSA_H_
#define INC_ROCTRACER_HSA_H_
#include <mutex>

#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ext_amd.h>

#include "ext/prof_protocol.h"
#include "roctracer.h"

namespace roctracer {
namespace hsa_support {
template <int N>
class CbTable {
  public:
  typedef std::mutex mutex_t;

  CbTable() {
    std::lock_guard<mutex_t> lck(mutex_);
    for (int i = 0; i < N; i++) {
      callback_[i] = NULL;
      arg_[i] = NULL;
    }
  }

  void set(uint32_t id, activity_rtapi_callback_t callback, void* arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    callback_[id] = callback;
    arg_[id] = arg;
  }

  void get(uint32_t id, activity_rtapi_callback_t* callback, void** arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    *callback = callback_[id];
    *arg = arg_[id];
  }

  private:
  activity_rtapi_callback_t callback_[N];
  void* arg_[N];
  mutex_t mutex_;
};

extern CoreApiTable CoreApiTable_saved;
extern AmdExtTable AmdExtTable_saved;
extern ImageExtTable ImageExtTable_saved;
};
};

#include "inc/hsa_prof_str.h"
#endif // INC_ROCTRACER_HSA_H_
