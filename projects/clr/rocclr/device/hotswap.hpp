/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE. */

#pragma once

#include <cstdlib>
#include <string>

namespace amd {
namespace hotswap {

// On when this tool is loaded via HSA_TOOLS_LIB (name must match ROCR LoadTools).
inline constexpr const char* kHotswapToolLib = "libhsa-hotswap.so";

inline bool Enabled() {
  const char* tools_lib = std::getenv("HSA_TOOLS_LIB");
  return tools_lib != nullptr &&
         std::string(tools_lib).find(kHotswapToolLib) != std::string::npos;
}

// Allowlist of (source -> device) gfx pairs the tool handles; only these are forwarded.
struct SourceTargetPair {
  const char* source;  // gfx processor, e.g. "gfx1250"
  const char* target;  // gfx processor, e.g. "gfx950"
};

inline constexpr SourceTargetPair kSupportedPairs[] = {
    {"gfx1250", "gfx1250"},
};

// True if (source_gfx -> target_gfx) is a supported pair.
inline bool IsSupportedPair(const std::string& source_gfx,
                            const std::string& target_gfx) {
  for (const SourceTargetPair& p : kSupportedPairs) {
    if (source_gfx == p.source && target_gfx == p.target) {
      return true;
    }
  }
  return false;
}

// True if ISA name names processor `gfx`, matched on a token boundary (gfx1250 != gfx12500).
inline bool IsaIsGfx(const std::string& isa_name, const std::string& gfx) {
  const std::string needle = "--" + gfx;
  const std::string::size_type pos = isa_name.find(needle);
  if (pos == std::string::npos) {
    return false;
  }
  const std::string::size_type after = pos + needle.size();
  return after == isa_name.size() || isa_name[after] == ':';
}

}  // namespace hotswap
}  // namespace amd
