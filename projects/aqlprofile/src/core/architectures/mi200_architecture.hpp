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

#ifndef SRC_CORE_ARCHITECTURES_MI200_ARCHITECTURE_HPP_
#define SRC_CORE_ARCHITECTURES_MI200_ARCHITECTURE_HPP_

#include "core/architectures/gfx9_architecture.hpp"

namespace aql_profile {

/// MI200 architecture (CDNA 2 - gfx90a)
/// Specialized Gfx9 variant with:
/// - SPM dual-core support (has_spm_core1)
/// - Accumulator register support (accum low=1, hi=185)
/// - SPM sample delay max 0x3e
class Mi200Architecture : public Gfx9Architecture {
 public:
  explicit Mi200Architecture(const AgentInfo* agent_info);
  virtual ~Mi200Architecture() = default;

  // Architecture queries
  bool IsMI200() const override { return true; }

  // MI200-specific accumulator register IDs and SPM sample delay
  uint32_t GetAccumLowID() const override { return 1; }
  uint32_t GetAccumHiID() const override { return 185; }
  uint32_t GetSpmSampleDelayMax() const override { return 0x3e; }

 protected:
  void InitializeConfig(const AgentInfo* agent_info) override;
  void InitializeBlockTable() override;
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURES_MI200_ARCHITECTURE_HPP_
