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


#include "roctracer.h"

#include <dlfcn.h>
#include <hsa/hsa.h>

#include <cassert>

using get_timestamp_t = decltype(roctracer_get_timestamp);
using hsa_init_t = decltype(hsa_init);
using hsa_shut_down_t = decltype(hsa_shut_down);

int main() {
  // CASE 1: HSA is not loaded.
  //
  {
    void* tracer_library = dlopen("libroctracer64.so", RTLD_LAZY);
    assert(tracer_library != nullptr);

    auto* get_timestamp =
        reinterpret_cast<get_timestamp_t*>(dlsym(tracer_library, "roctracer_get_timestamp"));
    assert(get_timestamp != nullptr);

    roctracer_timestamp_t timestamp;
    (*get_timestamp)(&timestamp);
    dlclose(tracer_library);
  }

  // CASE 2 Load the roctracer after hsa_init().
  //
  void* hsa_library = dlopen("libhsa-runtime64.so.1", RTLD_LAZY);
  assert(hsa_library != nullptr);

  auto* hsa_init = reinterpret_cast<hsa_init_t*>(dlsym(hsa_library, "hsa_init"));
  auto* hsa_shut_down = reinterpret_cast<hsa_shut_down_t*>(dlsym(hsa_library, "hsa_shut_down"));
  assert(hsa_init != nullptr && hsa_shut_down != nullptr);

  {
    (*hsa_init)();

    void* tracer_library = dlopen("libroctracer64.so", RTLD_LAZY);
    assert(tracer_library != nullptr);

    auto* get_timestamp =
        reinterpret_cast<get_timestamp_t*>(dlsym(tracer_library, "roctracer_get_timestamp"));
    assert(get_timestamp != nullptr);

    roctracer_timestamp_t timestamp;
    (*get_timestamp)(&timestamp);

    dlclose(tracer_library);
    (*hsa_shut_down)();
  }

  // CASE 3: Load and use the roctracer before hsa_init().
  //
  {
    void* tracer_library = dlopen("libroctracer64.so", RTLD_LAZY);
    assert(tracer_library != nullptr);

    auto* get_timestamp =
        reinterpret_cast<get_timestamp_t*>(dlsym(tracer_library, "roctracer_get_timestamp"));
    assert(get_timestamp != nullptr);

    roctracer_timestamp_t timestamp;
    (*get_timestamp)(&timestamp);

    (*hsa_init)();
    (*hsa_shut_down)();
    dlclose(tracer_library);
  }

  return 0;
}