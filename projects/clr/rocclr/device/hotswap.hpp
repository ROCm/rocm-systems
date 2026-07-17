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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace amd {
namespace hotswap {

// On when this tool is loaded via HSA_TOOLS_LIB (name must match ROCR LoadTools).
inline constexpr const char* kHotswapToolLib = "libamd_comgr_hotswap_tool.so";

inline bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !value[0]) return false;
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return normalized != "0" && normalized != "off" && normalized != "false" &&
         normalized != "no" && normalized != "n" && normalized != "f";
}

inline bool Disabled() { return EnvEnabled("HSA_HOTSWAP_DISABLE"); }

inline const char* PresentedIsaEnv() { return std::getenv("HSA_HOTSWAP_PRESENT_ISA"); }

inline const char* TargetIsaEnv() {
  const char* target = std::getenv("HSA_HOTSWAP_TARGET");
  if (!target || !target[0]) target = std::getenv("HSA_HOTSWAP_ISA_OVERRIDE");
  return target;
}

inline bool PresentationEnabled() {
  return EnvEnabled("HSA_HOTSWAP_PRESENT_ISA") && !Disabled();
}

inline bool ToolLoaded() {
  const char* tools_lib = std::getenv("HSA_TOOLS_LIB");
  return tools_lib != nullptr &&
         std::string(tools_lib).find(kHotswapToolLib) != std::string::npos;
}

inline bool Enabled() { return PresentationEnabled() || (!Disabled() && ToolLoaded()); }

// Allowlist for forwarding source-ISA fatbin bundles to a CLR-visible device ISA.
// In presentation mode the visible ISA is the presented source ISA; the physical
// execution ISA is resolved later by ROCr from HSA_HOTSWAP_TARGET.
struct SourcePresentedPair {
  const char* source;     // bundle gfx processor, e.g. "gfx1250"
  const char* presented;  // CLR-visible gfx processor, e.g. "gfx1250"
};

inline constexpr SourcePresentedPair kSourceForwardingPairs[] = {
    {"gfx1250", "gfx1250"},
};

// True if `source_gfx` may be forwarded for a device presented as `presented_gfx`.
inline bool IsSourceForwardingPair(const std::string& source_gfx,
                                   const std::string& presented_gfx) {
  for (const SourcePresentedPair& p : kSourceForwardingPairs) {
    if (source_gfx == p.source && presented_gfx == p.presented) {
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
