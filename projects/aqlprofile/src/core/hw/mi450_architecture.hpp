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

#ifndef SRC_CORE_ARCHITECTURES_MI450_ARCHITECTURE_HPP_
#define SRC_CORE_ARCHITECTURES_MI450_ARCHITECTURE_HPP_

#include "core/hw/gfx12_architecture.hpp"

namespace aql_profile {

/// MI450 architecture implementation (gfx1250).
/// Inherits Gfx12Architecture and overrides the block table for the MI450
/// block set.  xcc_per_aid is pre-patched to 4 by RegisterAgent before
/// AgentInfo is cached; this class reads it directly from agent_info.
class Mi450Architecture : public Gfx12Architecture {
 public:
  explicit Mi450Architecture(const AgentInfo* agent_info);
  ~Mi450Architecture() override = default;

 protected:
  void InitializeBlockTable() override;
};

}  // namespace aql_profile

#endif  // SRC_CORE_ARCHITECTURES_MI450_ARCHITECTURE_HPP_
