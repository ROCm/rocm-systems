/******************************************************************************
MIT License

Copyright (c) 2018 ROCm Core Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

#ifndef _SRC_CORE_HSA_LISTENER_H
#define _SRC_CORE_HSA_LISTENER_H

#include <hsa.h>

#include "util/exception.h"

namespace rocprofiler {
extern decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;
extern decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
extern decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;

class HsaListener {
 public:
  hsa_status_t SetCallbacks(rocprofiler_hsa_callbacks_t callbacks, void* data) {
  }

 private:
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_HSA_LISTENER_H
