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

#include "gfxip.h"
#include "vega10_ip_offset.h"

namespace rocprofiler::pc_sampler::gfxip {

void vega10_reg_offset_init(device_t& dev) {
  for (int i = 0; i < MAX_INSTANCE; ++i) {
    dev.reg_offset_[GC_HWIP][i] = (uint32_t*)(&(GC_BASE.instance[i]));
    dev.reg_offset_[OSSSYS_HWIP][i] = (uint32_t*)(&(OSSSYS_BASE.instance[i]));
  }
}

}  // namespace rocprofiler::pc_sampler::gfxip
