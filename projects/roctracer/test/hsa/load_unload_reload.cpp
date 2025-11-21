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

#include <hsa/hsa.h>

#include <cassert>
#include <cstdlib>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    if ((x) != HSA_STATUS_SUCCESS) {                                                               \
      assert(false);                                                                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (false);

int main() {
  // Run 2 loops of {hsa_init(); hsa_iterate_agents(); hsa_shut_down()} to test that the
  // tracer tool correctly unloaded after the 1st iteration and then reloaded for the 2nd
  // iteration.
  for (int i = 0; i < 2; ++i) {
    hsa_init();

    CHECK(hsa_iterate_agents(
        [](hsa_agent_t agent, void*) {
          hsa_device_type_t type;
          return hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        },
        nullptr));

    hsa_shut_down();
  }
}