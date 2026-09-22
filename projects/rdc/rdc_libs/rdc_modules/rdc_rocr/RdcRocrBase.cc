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

#include "rdc_modules/rdc_rocr/RdcRocrBase.h"

#include <string.h>

namespace amd {
namespace rdc {

RdcRocrBase::RdcRocrBase(void) {
  num_iteration_ = 1;
  cpu_device_.handle = -1;
  gpu_device1_.handle = -1;
  device_pool_.handle = 0;
  kern_arg_pool_.handle = 0;
  main_queue_ = nullptr;
  kernarg_buffer_ = nullptr;
  kernel_object_ = 0;
  memset(&aql_, 0, sizeof(aql_));
  set_requires_profile(-1);
  set_enable_interrupt(false);
  set_kernel_file_name("");
  set_verbosity(1);
  set_monitor_verbosity(0);
  set_title("unset_title");
  orig_hsa_enable_interrupt_ = nullptr;
}

RdcRocrBase::~RdcRocrBase() {}

}  // namespace rdc
}  // namespace amd
