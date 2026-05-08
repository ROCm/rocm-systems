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

#include "core/architecture_init.hpp"
#include "core/architecture_registry.hpp"
#include "core/architectures/gfx9_architecture.hpp"
#include "core/architectures/gfx10_architecture.hpp"
#include "core/architectures/gfx11_architecture.hpp"
#include "core/architectures/gfx12_architecture.hpp"
#include "core/architectures/mi100_architecture.hpp"
#include "core/architectures/mi200_architecture.hpp"
#include "core/architectures/mi300_architecture.hpp"
#include "core/architectures/mi350_architecture.hpp"

#include <string_view>

namespace aql_profile {

// Helper to determine GPU ID from gfxip string
static std::string_view GetGfxIpPrefix(std::string_view gfxip) {
  // Order matters: check specific variants before generic ones
  if (gfxip.rfind("gfx908", 0) == 0) return "gfx908";  // MI100
  if (gfxip.rfind("gfx90a", 0) == 0) return "gfx90a";  // MI200
  if (gfxip.rfind("gfx94", 0) == 0) return "gfx94";    // MI300 series
  if (gfxip.rfind("gfx95", 0) == 0) return "gfx95";    // MI350 series
  if (gfxip.rfind("gfx900", 0) == 0) return "gfx900";  // Vega10
  if (gfxip.rfind("gfx902", 0) == 0) return "gfx902";  // Raven
  if (gfxip.rfind("gfx906", 0) == 0) return "gfx906";  // Vega20
  if (gfxip.rfind("gfx90", 0) == 0) return "gfx90";    // Generic GFX9
  if (gfxip.rfind("gfx10", 0) == 0) return "gfx10";    // Generic GFX10
  if (gfxip.rfind("gfx115", 0) == 0) return "gfx115";  // GFX11.5x (RDNA 3.5)
  if (gfxip.rfind("gfx11", 0) == 0) return "gfx11";    // Generic GFX11
  if (gfxip.rfind("gfx125", 0) == 0) return "gfx125";  // MI450 (gfx1250)
  if (gfxip.rfind("gfx12", 0) == 0) return "gfx12";    // Generic GFX12
  return "";
}

void InitializeArchitectureRegistry() {
  // This function intentionally left minimal
  // Architectures are created on-demand in CreateArchitectureForAgent
  // to avoid needing full AgentInfo at initialization time
}

HardwareArchitecture* CreateArchitectureForAgent(const AgentInfo* agent_info) {
  if (!agent_info) return nullptr;

  std::string_view gfxip = agent_info->gfxip;
  std::string_view prefix = GetGfxIpPrefix(gfxip);

  // Create appropriate architecture based on gfxip
  if (prefix == "gfx908") {
    return new Mi100Architecture(agent_info);
  } else if (prefix == "gfx90a") {
    return new Mi200Architecture(agent_info);
  } else if (prefix == "gfx94") {
    return new Mi300Architecture(agent_info);
  } else if (prefix == "gfx95") {
    return new Mi350Architecture(agent_info);
  } else if (prefix == "gfx900" || prefix == "gfx902" || prefix == "gfx906" || prefix == "gfx90") {
    return new Gfx9Architecture(agent_info);
  } else if (prefix == "gfx10") {
    return new Gfx10Architecture(agent_info);
  } else if (prefix == "gfx115") {
    return new Gfx115xArchitecture(agent_info);
  } else if (prefix == "gfx11") {
    return new Gfx11Architecture(agent_info);
  } else if (prefix == "gfx125") {
    return new Mi450Architecture(agent_info);
  } else if (prefix == "gfx12") {
    return new Gfx12Architecture(agent_info);
  }

  // Unknown architecture
  return nullptr;
}

}  // namespace aql_profile
