/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "common/rdc_capabilities.h"

#include <assert.h>
#include <errno.h>
#include <sys/capability.h>

namespace amd {
namespace rdc {

int GetCapability(cap_value_t cap, cap_flag_t cap_type, bool* enabled) {
  cap_t caps;

  assert(enabled != nullptr);

  if (enabled == nullptr) {
    return -1;
  }

  // Get process's current capabilities
  caps = cap_get_proc();
  if (caps == nullptr) {
    return errno;
  }

  cap_flag_value_t val;
  if (cap_get_flag(caps, cap, cap_type, &val) == -1) {
    int ret = errno;
    cap_free(caps);
    return ret;
  }

  if (cap_free(caps) == -1) {
    return errno;
  }

  *enabled = (val == CAP_SET ? true : false);

  return 0;
}

// !enable means disable;
int ModifyCapability(cap_value_t cap, cap_flag_t cap_type, bool enable) {
  cap_t caps;
  cap_value_t cap_list[1];

  // Get process's current capabilities
  caps = cap_get_proc();
  if (caps == nullptr) {
    return errno;
  }

  // the 1 in the call below is the size of the cap_list array
  cap_list[0] = cap;
  if (cap_set_flag(caps, cap_type, 1, cap_list, enable ? CAP_SET : CAP_CLEAR) == -1) {
    int ret = errno;
    cap_free(caps);
    return ret;
  }

  if (cap_set_proc(caps) == -1) {
    int ret = errno;
    cap_free(caps);
    return ret;
  }

  if (cap_free(caps) == -1) {
    return errno;
  }
  return 0;
}

}  // namespace rdc
}  // namespace amd
