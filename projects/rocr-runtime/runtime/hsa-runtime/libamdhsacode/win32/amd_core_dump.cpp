////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/util/utils.h"
#include "core/inc/runtime.h"
#include "./amd_hsa_code_util.hpp"
#include "core/inc/amd_core_dump.hpp"

namespace rocr {
namespace amd {
namespace coredump {
/* Implementation details */
namespace impl {

enum SegmentType { LOAD, NOTE };
struct SegmentBuilder;

struct SegmentInfo {
  SegmentType stype;
  uint64_t vaddr = 0;
  uint64_t size = 0;
  uint32_t flags = 0;
  SegmentBuilder* builder;
};

using SegmentsInfo = std::vector<SegmentInfo>;

struct SegmentBuilder {
  virtual ~SegmentBuilder() = default;
  /* Find which segments needs to be created.  */
  virtual hsa_status_t Collect(SegmentsInfo& segments) = 0;
  /* Called to read a given SegmentInfo's data.  */
  virtual hsa_status_t Read(void* buf, size_t buf_size, off_t offset) = 0;
};

struct LoadSegmentBuilder : public SegmentBuilder {
  LoadSegmentBuilder() {}
  ~LoadSegmentBuilder() {}

  hsa_status_t Collect(SegmentsInfo& segments) override {
    assert(!"Unimplemented!");
    return HSA_STATUS_ERROR;
  }

  hsa_status_t Read(void* buf, size_t buf_size, off_t offset) override {
    assert(!"Unimplemented!");
    return HSA_STATUS_ERROR;
  }

 private:
  int fd_ = -1;
};

hsa_status_t build_core_dump(const std::string& filename, const SegmentsInfo& segments,
                             size_t size_limit) {
  assert(!"Unimplemented!");
  return HSA_STATUS_ERROR;
}
}  // namespace impl
hsa_status_t dump_gpu_core() {
  assert(!"Unimplemented!");
  return HSA_STATUS_ERROR;
}
}  //  namespace coredump
}   //  namespace amd
}   //  namespace rocr
