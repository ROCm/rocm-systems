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

#ifndef SRC_CORE_ARCHITECTURES_MI100_ARCHITECTURE_HPP_
#define SRC_CORE_ARCHITECTURES_MI100_ARCHITECTURE_HPP_

#include "core/architectures/gfx9_architecture.hpp"

namespace aql_profile {

/// MI100 architecture (CDNA 1 - gfx908)
/// Specialized Gfx9 variant with:
/// - SPM dual-core support (has_spm_core1)
/// - Accumulator register support for SQ counters (accum low=1, hi=158)
/// - SPM sample delay max 0x34
class Mi100Architecture : public Gfx9Architecture {
 public:
  explicit Mi100Architecture(const AgentInfo* agent_info);
  virtual ~Mi100Architecture() = default;

  // Architecture queries
  bool IsMI100() const override { return true; }

  // MI100-specific accumulator register IDs for SQ counters
  uint32_t GetAccumLowID() const override { return 1; }
  uint32_t GetAccumHiID() const override { return 158; }
  uint32_t GetSpmSampleDelayMax() const override { return 0x34; }

 protected:
  void InitializeConfig(const AgentInfo* agent_info) override;
  void InitializeBlockTable() override;
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURES_MI100_ARCHITECTURE_HPP_
