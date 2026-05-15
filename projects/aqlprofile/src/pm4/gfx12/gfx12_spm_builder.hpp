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

#ifndef SRC_PM4_GFX12_SPM_BUILDER_HPP_
#define SRC_PM4_GFX12_SPM_BUILDER_HPP_

#include "pm4/spm_builder.h"
#include "pm4/primitives_provider.hpp"

namespace pm4_builder {

/// Non-template SPM builder for GFX12, using PrimitivesProvider for runtime dispatch.
class Gfx12SpmBuilder : public SpmBuilder {
  CmdBuilder* builder_;
  const PrimitivesProvider* prim_;

 public:
  explicit Gfx12SpmBuilder(CmdBuilder* builder, const PrimitivesProvider* prim)
      : SpmBuilder(), builder_(builder), prim_(prim) {}

  void Begin(CmdBuffer* cmd_buffer, const SpmConfig* config,
             const counters_vector& counters_vec) override;
  void End(CmdBuffer* cmd_buffer, const SpmConfig* config) override;
};

}  // namespace pm4_builder

#endif  // SRC_PM4_GFX12_SPM_BUILDER_HPP_
