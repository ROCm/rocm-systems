/******************************************************************************
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
*******************************************************************************/

#ifndef _SRC_CORE_HSA_QUEUE_H
#define _SRC_CORE_HSA_QUEUE_H

#include <atomic>

#include "core/queue.h"
#include "core/types.h"

namespace rocprofiler {

class HsaQueue : public Queue {
 public:
  HsaQueue(const util::AgentInfo* agent_info, hsa_queue_t* queue) : queue_(queue) {}

  void Submit(const packet_t* packet) {
    rocprofiler::util::HsaRsrcFactory::Instance().Submit(queue_, packet);
  }

 private:
  hsa_queue_t* queue_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_HSA_QUEUE_H
