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
#ifndef RDC_MODULES_RDC_ROCR_MEMORYTEST_H_
#define RDC_MODULES_RDC_ROCR_MEMORYTEST_H_

#include "hsa/hsa.h"
#include "rdc_modules/rdc_rocr/TestBase.h"

namespace amd {
namespace rdc {
class MemoryTest : public TestBase {
 public:
  MemoryTest(uint32_t gpu_index, uint32_t node_id, GpuOutcome* outcome);

  // @Brief: Destructor for test case of MemoryTest
  virtual ~MemoryTest();

  // Each pool says what it did in a sentence of its own; a call that fails before any of them
  // is this GPU's failure, named with its status, and leaves by throwing.
  void MaxSingleAllocationTest(void);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_MEMORYTEST_H_
