// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_CORE_ARCHITECTURES_MI350_ARCHITECTURE_HPP_
#define SRC_CORE_ARCHITECTURES_MI350_ARCHITECTURE_HPP_

#include "core/architectures/mi300_architecture.hpp"

namespace aql_profile {

/// MI350 architecture (CDNA 3.5 - gfx950)
/// Extends MI300 with:
/// - Updated accumulator register IDs (accum hi=200)
/// - Higher SPM sample delay max (0x33)
/// - Inherits multi-XCC/AID topology and AID-aware counters from MI300
class Mi350Architecture : public Mi300Architecture {
 public:
  explicit Mi350Architecture(const AgentInfo* agent_info);
  virtual ~Mi350Architecture() = default;

  // Architecture query
  bool IsMI350() const override { return true; }

  // MI350-specific accumulator register IDs and SPM sample delay
  uint32_t GetAccumLowID() const override { return 1; }
  uint32_t GetAccumHiID() const override { return 200; }
  uint32_t GetSpmSampleDelayMax() const override { return 0x33; }

 protected:
  void InitializeConfig(const AgentInfo* agent_info) override;
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURES_MI350_ARCHITECTURE_HPP_
