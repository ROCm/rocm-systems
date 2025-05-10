/* Copyright (c) 2008 - 2023 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "device/device.hpp"
#include "thread/monitor.hpp"
#include "utils/options.hpp"
#include "comgrctx.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>

#if defined(WITH_HSA_DEVICE)
#include "device/rocm/rocdevice.hpp"
extern amd::AppProfile* rocCreateAppProfile();
#endif

#if defined(WITH_PAL_DEVICE)
// namespace amd::pal {
extern bool PalDeviceLoad();
extern void PalDeviceUnload();
//}
#endif  // WITH_PAL_DEVICE

#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "thread/monitor.hpp"
#include "amdocl/cl_common.hpp"
#include "utils/options.hpp"
#include "utils/versions.hpp"  // AMD_PLATFORM_INFO

#if defined(HAVE_BLOWFISH_H)
#include "blowfish/oclcrypt.hpp"
#endif

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <set>
#include <algorithm>
#include <numeric>

namespace {

constexpr char hsaIsaNamePrefix[] = "amdgcn-amd-amdhsa--";

}  // namespace

namespace amd::device {
extern const char* BlitLinearSourceCode;
extern const char* BlitImageSourceCode;

bool VirtualDevice::ActiveWait() const { return device_().ActiveWait(); }

}  // namespace amd::device

static_assert(static_cast<uint32_t>(device::Memory::MemAccess::kMemAccessNone) ==
                  static_cast<uint32_t>(amd::Device::VmmAccess::kNone),
              "Mem Access Flag None mismatch between Device and Memory!");
static_assert(static_cast<uint32_t>(device::Memory::MemAccess::kMemAccessRead) ==
                  static_cast<uint32_t>(amd::Device::VmmAccess::kReadOnly),
              "Mem Access Flag Read mismatch between Device and Memory!");
static_assert(static_cast<uint32_t>(device::Memory::MemAccess::kMemAccessReadWrite) ==
                  static_cast<uint32_t>(amd::Device::VmmAccess::kReadWrite),
              "Mem Access Flag Read Write mismatch between Device and Memory!");

namespace amd {

amd::Monitor Device::lockP2P_("Lock P2P ON/OFF");
std::pair<const Isa*, const Isa*> Isa::supportedIsas() {
  constexpr amd::Isa::Feature NONE = amd::Isa::Feature::Unsupported;
  constexpr amd::Isa::Feature ANY = amd::Isa::Feature::Any;
  constexpr amd::Isa::Feature OFF = amd::Isa::Feature::Disabled;
  constexpr amd::Isa::Feature ON = amd::Isa::Feature::Enabled;

  static constexpr Isa supportedIsas_[] = {

      // NOTE: Add new targets by adding rows for each permutation of the SRAMECC
      // and XNACK target feature values. If the target does not support the
      // feature then only NONE is used. If it supports the feature than include
      // rows for ANY, OFF and ON (but not NONE).
      //
      // Use the Target ID syntax. This comprises the processor name, followed by
      // the target feature settings in alphebetic order separated by ':'. If a
      // target feature is omitted it means either it is not supported, or it has
      // the ANY value. If the target feature is disabled then use a '-' suffix,
      // and if enabled use a '+' suffix.
      //
      // If the HSAIL or AMD IL compilers do not support the target, then use
      // nullptr for the ID.
      //
      // -------------- Compiler ----------    - Runtime -    ---- IP ----   -- Target -- ----------
      // Target Properties ----------
      //                                        Supported        Version       Features
      //                                                                                    SIMD/
      //                                                                                    SIMD
      //                                                                                    Instr
      //                                                                                    Bank LDS
      //                                                                                    Mem
      // Target ID                 HSAIL ID     ROC    PAL      Maj/Min/Stp   SRAMECC XNACK CU Width
      // Width Width   Size    Banks    LDSAlignment
      {"gfx801", nullptr, true, true, 8, 0, 1, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx801:xnack-", nullptr, true, false, 8, 0, 1, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx801:xnack+", "gfx801", true, true, 8, 0, 1, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx802", "gfx802", true, true, 8, 0, 2, NONE, NONE, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx803", "gfx803", true, true, 8, 0, 3, NONE, NONE, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx805", nullptr, true, true, 8, 0, 5, NONE, NONE, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx810", nullptr, true, true, 8, 1, 0, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx810:xnack-", nullptr, true, false, 8, 1, 0, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx810:xnack+", "gfx810", true, true, 8, 1, 0, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx900", "gfx901", true, true, 9, 0, 0, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx900:xnack-", "gfx900", true, true, 9, 0, 0, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx900:xnack+", "gfx901", true, true, 9, 0, 0, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx902", "gfx903", true, true, 9, 0, 2, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx902:xnack-", "gfx902", true, true, 9, 0, 2, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx902:xnack+", "gfx903", true, true, 9, 0, 2, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx904", "gfx905", true, true, 9, 0, 4, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx904:xnack-", "gfx904", true, true, 9, 0, 4, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx904:xnack+", "gfx905", true, true, 9, 0, 4, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx906", "gfx907", true, true, 9, 0, 6, ANY, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx906:sramecc-", "gfx907", true, true, 9, 0, 6, OFF, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx906:sramecc+", nullptr, true, true, 9, 0, 6, ON, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx906:xnack-", "gfx906", true, true, 9, 0, 6, ANY, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx906:xnack+", "gfx907", true, true, 9, 0, 6, ANY, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx906:sramecc-:xnack-", "gfx906", true, true, 9, 0, 6, OFF, OFF, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx906:sramecc-:xnack+", "gfx907", true, true, 9, 0, 6, OFF, ON, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx906:sramecc+:xnack-", nullptr, true, true, 9, 0, 6, ON, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx906:sramecc+:xnack+", nullptr, true, true, 9, 0, 6, ON, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx908", nullptr, true, false, 9, 0, 8, ANY, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx908:sramecc-", nullptr, true, false, 9, 0, 8, OFF, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx908:sramecc+", nullptr, true, false, 9, 0, 8, ON, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx908:xnack-", nullptr, true, false, 9, 0, 8, ANY, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx908:xnack+", nullptr, true, false, 9, 0, 8, ANY, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx908:sramecc-:xnack-", nullptr, true, false, 9, 0, 8, OFF, OFF, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx908:sramecc-:xnack+", nullptr, true, false, 9, 0, 8, OFF, ON, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx908:sramecc+:xnack-", nullptr, true, false, 9, 0, 8, ON, OFF, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx908:sramecc+:xnack+", nullptr, true, false, 9, 0, 8, ON, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx909", nullptr, false, true, 9, 0, 2, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx909:xnack-", nullptr, false, true, 9, 0, 2, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx909:xnack+", nullptr, false, true, 9, 0, 2, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90a", nullptr, true, false, 9, 0, 10, ANY, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90a:sramecc-", nullptr, true, false, 9, 0, 10, OFF, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90a:sramecc+", nullptr, true, false, 9, 0, 10, ON, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90a:xnack-", nullptr, true, false, 9, 0, 10, ANY, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90a:xnack+", nullptr, true, false, 9, 0, 10, ANY, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90a:sramecc-:xnack-", nullptr, true, false, 9, 0, 10, OFF, OFF, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx90a:sramecc-:xnack+", nullptr, true, false, 9, 0, 10, OFF, ON, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx90a:sramecc+:xnack-", nullptr, true, false, 9, 0, 10, ON, OFF, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx90a:sramecc+:xnack+", nullptr, true, false, 9, 0, 10, ON, ON, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx942", nullptr, true, false, 9, 4, 2, ANY, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx942:sramecc-", nullptr, true, false, 9, 4, 2, OFF, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx942:sramecc+", nullptr, true, false, 9, 4, 2, ON, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx942:xnack-", nullptr, true, false, 9, 4, 2, ANY, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx942:xnack+", nullptr, true, false, 9, 4, 2, ANY, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx942:sramecc-:xnack-", nullptr, true, false, 9, 4, 2, OFF, OFF, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx942:sramecc-:xnack+", nullptr, true, false, 9, 4, 2, OFF, ON, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx942:sramecc+:xnack-", nullptr, true, false, 9, 4, 2, ON, OFF, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx942:sramecc+:xnack+", nullptr, true, false, 9, 4, 2, ON, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90c", nullptr, true, true, 9, 0, 12, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90c:xnack-", "gfx90c", true, true, 9, 0, 12, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx90c:xnack+", "gfx90d", true, true, 9, 0, 12, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx950", nullptr, true, false, 9, 5, 0, ANY, ANY, 4, 16, 1, 256, 160 * Ki, 64, 1280},
      {"gfx950:sramecc-", nullptr, true, false, 9, 5, 0, OFF, ANY, 4, 16, 1, 256, 160 * Ki, 64, 1280},
      {"gfx950:sramecc+", nullptr, true, false, 9, 5, 0, ON, ANY, 4, 16, 1, 256, 160 * Ki, 64, 1280},
      {"gfx950:xnack-", nullptr, true, false, 9, 5, 0, ANY, OFF, 4, 16, 1, 256, 160 * Ki, 64, 1280},
      {"gfx950:xnack+", nullptr, true, false, 9, 5, 0, ANY, ON, 4, 16, 1, 256, 160 * Ki, 64, 1280},
      {"gfx950:sramecc-:xnack-", nullptr, true, false, 9, 5, 0, OFF, OFF, 4, 16, 1, 256, 160 * Ki,
       64, 1280},
      {"gfx950:sramecc-:xnack+", nullptr, true, false, 9, 5, 0, OFF, ON, 4, 16, 1, 256, 160 * Ki,
       64, 1280},
      {"gfx950:sramecc+:xnack-", nullptr, true, false, 9, 5, 0, ON, OFF, 4, 16, 1, 256, 160 * Ki,
       64, 1280},
      {"gfx950:sramecc+:xnack+", nullptr, true, false, 9, 5, 0, ON, ON, 4, 16, 1, 256, 160 * Ki,
       64, 1280},
      {"gfx9-generic", nullptr, true, true, 9, 0, 0, NONE, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx9-generic:xnack-", nullptr, true, true, 9, 0, 0, NONE, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx9-generic:xnack+", nullptr, true, true, 9, 0, 0, NONE, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx9-4-generic", nullptr, true, true, 9, 4, 0, ANY, ANY, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx9-4-generic:sramecc-", nullptr, true, true, 9, 4, 0, OFF, ANY, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx9-4-generic:sramecc+", nullptr, true, true, 9, 4, 0, ON, ANY, 4, 16, 1, 256, 64 * Ki,
       32, 512},
      {"gfx9-4-generic:xnack-", nullptr, true, true, 9, 4, 0, ANY, OFF, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx9-4-generic:xnack+", nullptr, true, true, 9, 4, 0, ANY, ON, 4, 16, 1, 256, 64 * Ki, 32, 512},
      {"gfx9-4-generic:sramecc-:xnack-", nullptr, true, true, 9, 4, 0, OFF, OFF, 4, 16, 1, 256,
       64 * Ki, 32, 512},
      {"gfx9-4-generic:sramecc-:xnack+", nullptr, true, true, 9, 4, 0, OFF, ON, 4, 16, 1, 256,
       64 * Ki, 32, 512},
      {"gfx9-4-generic:sramecc+:xnack-", nullptr, true, true, 9, 4, 0, ON, OFF, 4, 16, 1, 256,
       64 * Ki, 32, 512},
      {"gfx9-4-generic:sramecc+:xnack+", nullptr, true, true, 9, 4, 0, ON, ON, 4, 16, 1, 256,
       64 * Ki, 32, 512},
      {"gfx1010", "gfx1010", true, true, 10, 1, 0, NONE, ANY, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1010:xnack-", "gfx1010", true, true, 10, 1, 0, NONE, OFF, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1010:xnack+", nullptr, true, true, 10, 1, 0, NONE, ON, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1011", "gfx1011", true, true, 10, 1, 1, NONE, ANY, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1011:xnack-", "gfx1011", true, true, 10, 1, 1, NONE, OFF, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1011:xnack+", nullptr, true, true, 10, 1, 1, NONE, ON, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1012", "gfx1012", true, true, 10, 1, 2, NONE, ANY, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1012:xnack-", "gfx1012", true, true, 10, 1, 2, NONE, OFF, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1012:xnack+", nullptr, true, true, 10, 1, 2, NONE, ON, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1013", "gfx1013", true, false, 10, 1, 3, NONE, ANY, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1013:xnack-", "gfx1013", true, false, 10, 1, 3, NONE, OFF, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1013:xnack+", nullptr, true, false, 10, 1, 3, NONE, ON, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx10-1-generic", nullptr, true, true, 10, 1, 0, NONE, ANY, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx10-1-generic:xnack-", nullptr, true, true, 10, 1, 0, NONE, OFF, 2, 32, 1, 256, 64 * Ki,
       32, 1024},
      {"gfx10-1-generic:xnack+", nullptr, true, true, 10, 1, 0, NONE, ON, 2, 32, 1, 256, 64 * Ki,
       32, 1024},
      {"gfx1030", "gfx1030", true, true, 10, 3, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1031", "gfx1031", true, true, 10, 3, 1, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1032", "gfx1032", true, true, 10, 3, 2, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1033", "gfx1033", true, false, 10, 3, 3, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1034", "gfx1034", true, true, 10, 3, 4, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1035", "gfx1035", true, true, 10, 3, 5, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1036", "gfx1036", true, true, 10, 3, 6, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx10-3-generic", nullptr, true, true, 10, 3, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1100", "gfx1100", true, true, 11, 0, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1101", "gfx1101", true, true, 11, 0, 1, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1102", "gfx1102", true, true, 11, 0, 2, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1103", "gfx1103", true, true, 11, 0, 3, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1150", "gfx1150", true, true, 11, 5, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1151", "gfx1151", true, true, 11, 5, 1, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1152", "gfx1152", true, true, 11, 5, 2, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1153", "gfx1153", true, true, 11, 5, 3, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx11-generic", nullptr, true, true, 11, 0, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1200", "gfx1200", true, true, 12, 0, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx1201", "gfx1201", true, true, 12, 0, 1, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
      {"gfx12-generic", nullptr, true, true, 12, 0, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
  };
  return std::make_pair(std::begin(supportedIsas_), std::end(supportedIsas_));
}

std::string Isa::processorName() const {
  std::string processor(targetId_);
  return processor.substr(0, processor.find(':'));
}

std::string Isa::isaName() const { return std::string(hsaIsaNamePrefix) + targetId(); }

bool Isa::isCompatible(const Isa& codeObjectIsa, const Isa& agentIsa) {
  bool isGeneric = std::strstr(codeObjectIsa.targetId(), "generic") != nullptr;
  if (isGeneric) {
    if (codeObjectIsa.versionMajor() != agentIsa.versionMajor() ||
        codeObjectIsa.versionMinor() > agentIsa.versionMinor() ||
        (codeObjectIsa.versionMinor() == agentIsa.versionMinor() &&
         codeObjectIsa.versionStepping() > agentIsa.versionStepping())) {
      return false;
    }
#ifdef DEBUG
    // Only check in DEBUG mode
    if (std::strstr(agentIsa.targetId(), "gfx906") != nullptr) {
      // For the generic target of gfx906, codeObjectIsa.isSrameccSupported() == false while
      // agentIsa.isSrameccSupported() = true
      assert(agentIsa.sramecc() != Feature::Any);
    } else {
      assert(codeObjectIsa.isSrameccSupported() == agentIsa.isSrameccSupported() &&
             agentIsa.sramecc() != Feature::Any);
    }
#endif
  } else {
    if (codeObjectIsa.versionMajor() != agentIsa.versionMajor() ||
        codeObjectIsa.versionMinor() != agentIsa.versionMinor() ||
        codeObjectIsa.versionStepping() != agentIsa.versionStepping())
      return false;
    assert(codeObjectIsa.isSrameccSupported() == agentIsa.isSrameccSupported() &&
           agentIsa.sramecc() != Feature::Any);
  }
  if ((codeObjectIsa.sramecc() == Feature::Enabled ||
       codeObjectIsa.sramecc() == Feature::Disabled) &&
      codeObjectIsa.sramecc() != agentIsa.sramecc())
    return false;

  assert(codeObjectIsa.isXnackSupported() == agentIsa.isXnackSupported() &&
         agentIsa.xnack() != Feature::Any);
  if ((codeObjectIsa.xnack() == Feature::Enabled || codeObjectIsa.xnack() == Feature::Disabled) &&
      codeObjectIsa.xnack() != agentIsa.xnack())
    return false;

  return true;
}

const Isa* Isa::findIsa(const char* isaName) {
  if (!isaName) return nullptr;
  const char* prefix = std::strstr(isaName, hsaIsaNamePrefix);
  if (prefix != isaName) return nullptr;
  const char* targetId = isaName + std::strlen(hsaIsaNamePrefix);
  auto supportedIsas_ = supportedIsas();
  auto isaIter = std::find_if(supportedIsas_.first, supportedIsas_.second, [&](const Isa& isa) {
    return std::strcmp(targetId, isa.targetId_) == 0;
  });
  return isaIter == supportedIsas_.second ? nullptr : isaIter;
}

const Isa* Isa::findIsa(uint32_t versionMajor, uint32_t versionMinor, uint32_t versionStepping,
                        Isa::Feature sramecc, Isa::Feature xnack) {
  auto supportedIsas_ = supportedIsas();
  auto isaIter = std::find_if(supportedIsas_.first, supportedIsas_.second, [&](const Isa& isa) {
    return versionMajor == isa.versionMajor_ && versionMinor == isa.versionMinor_ &&
           versionStepping == isa.versionStepping_ &&
           (isa.sramecc_ == amd::Isa::Feature::Unsupported || isa.sramecc_ == sramecc) &&
           (isa.xnack_ == amd::Isa::Feature::Unsupported || isa.xnack_ == xnack);
  });
  return isaIter == supportedIsas_.second ? nullptr : isaIter;
}

const Isa* Isa::begin() { return supportedIsas().first; }

const Isa* Isa::end() { return supportedIsas().second; }

std::vector<Device*>* Device::devices_ = nullptr;
AppProfile Device::appProfile_;

Context* Device::glb_ctx_ = nullptr;
// P2P Staging Lock
Monitor Device::p2p_stage_ops_(true);
Memory* Device::p2p_stage_ = nullptr;

cl_int Device::gpu_error_ = CL_SUCCESS;

std::shared_mutex MemObjMap::AllocatedLock_ ROCCLR_INIT_PRIORITY(101);
std::map<uintptr_t, amd::Memory*> MemObjMap::MemObjMap_ ROCCLR_INIT_PRIORITY(101);
std::map<uintptr_t, amd::Memory*> MemObjMap::VirtualMemObjMap_ ROCCLR_INIT_PRIORITY(101);
std::map<MemObjMap::IpcMemHandle, amd::Memory*> MemObjMap::IpcHandleMemObjMap_ ROCCLR_INIT_PRIORITY(
    101);


void MemObjMap::AddMemObj(const void* k, amd::Memory* v) {
  std::unique_lock lock(AllocatedLock_);
  auto rval = MemObjMap_.insert({reinterpret_cast<uintptr_t>(k), v});
  if (!rval.second) {
    DevLogPrintfError("Memobj map already has an entry for ptr: 0x%x",
                      reinterpret_cast<uintptr_t>(k));
  }
}

void MemObjMap::RemoveMemObj(const void* k) {
  std::unique_lock lock(AllocatedLock_);
  auto rval = MemObjMap_.erase(reinterpret_cast<uintptr_t>(k));
  guarantee(rval == 1, "Memobj map does not have ptr: 0x%x", reinterpret_cast<uintptr_t>(k));
}

amd::Memory* MemObjMap::FindMemObj(const void* k, size_t* offset) {
  std::shared_lock lock(AllocatedLock_);
  uintptr_t key = reinterpret_cast<uintptr_t>(k);
  auto it = MemObjMap_.upper_bound(key);
  if (it == MemObjMap_.begin()) {
    return nullptr;
  }

  --it;
  amd::Memory* mem = it->second;
  size_t mem_size = (mem->getMemFlags() & ROCCLR_MEM_PHYMEM) ? sizeof(mem->getUserData().hsa_handle)
                                                             : mem->getSize();
  if (key >= it->first && key < (it->first + mem_size)) {
    if (offset != nullptr) {
      *offset = key - it->first;
    }
    // the k is in the range
    return mem;
  } else {
    return nullptr;
  }
}

void MemObjMap::UpdateAccess(amd::Device* peerDev) {
  if (peerDev == nullptr) {
    return;
  }
  // Provides access to all memory allocated on peerDev but
  // hsa_amd_agents_allow_access was not called because there was no peer
  std::shared_lock lock(AllocatedLock_);
  for (auto it : MemObjMap_) {
    const std::vector<Device*>& devices = it.second->getContext().devices();
    if (devices.size() == 1 && devices[0] == peerDev) {
      device::Memory* devMem = it.second->getDeviceMemory(*devices[0]);
      if (!devMem->getAllowedPeerAccess()) {
        peerDev->deviceAllowAccess(reinterpret_cast<void*>(it.first));
        devMem->setAllowedPeerAccess(true);
      }
    }
  }
}

void MemObjMap::Purge(amd::Device* dev) {
  assert(dev != nullptr);
  std::unique_lock lock(AllocatedLock_);
  for (auto it = MemObjMap_.cbegin(); it != MemObjMap_.cend();) {
    amd::Memory* memObj = it->second;
    unsigned int flags = memObj->getMemFlags();
    const std::vector<Device*>& devices = memObj->getContext().devices();
    if (devices.size() == 1 && devices[0] == dev && !(flags & ROCCLR_MEM_INTERNAL_MEMORY)) {
      memObj->release();
      it = MemObjMap_.erase(it);
    } else {
      ++it;
    }
  }
}

void MemObjMap::AddVirtualMemObj(const void* k, amd::Memory* v) {
  std::unique_lock lock(AllocatedLock_);
  auto rval = VirtualMemObjMap_.insert({reinterpret_cast<uintptr_t>(k), v});
  if (!rval.second) {
    DevLogPrintfError("Virtual Memobj map already has an entry for ptr: 0x%x",
                      reinterpret_cast<uintptr_t>(k));
  }
}

void MemObjMap::RemoveVirtualMemObj(const void* k) {
  std::unique_lock lock(AllocatedLock_);
  auto rval = VirtualMemObjMap_.erase(reinterpret_cast<uintptr_t>(k));
  guarantee(rval == 1, "Virtual Memobj map does not have ptr: 0x%x",
            reinterpret_cast<uintptr_t>(k));
}

amd::Memory* MemObjMap::FindVirtualMemObj(const void* k) {
  std::shared_lock lock(AllocatedLock_);
  uintptr_t key = reinterpret_cast<uintptr_t>(k);
  auto it = VirtualMemObjMap_.upper_bound(key);
  if (it == VirtualMemObjMap_.begin()) {
    return nullptr;
  }

  --it;
  amd::Memory* mem = it->second;
  if (key >= it->first && key < (it->first + mem->getSize())) {
    // the k is in the range
    return mem;
  } else {
    return nullptr;
  }
}

void MemObjMap::AddIpcHandleMemObj(const IpcMemHandle& k, amd::Memory* v) {
  std::unique_lock lock(AllocatedLock_);
  auto rval = IpcHandleMemObjMap_.insert({k, v});
  if (!rval.second) {
    DevLogPrintfError(
        "Error adding entry for Memobj 0x%x in IpcHandle map. The handle already exists.", v);
  }
}

void MemObjMap::RemoveIpcHandleMemObj(amd::Memory* v) {
  std::unique_lock lock(AllocatedLock_);

  for (const auto it : IpcHandleMemObjMap_) {
    if (it.second == v) {
      IpcHandleMemObjMap_.erase(it.first);
      break;
    }
  }
}

amd::Memory* MemObjMap::FindIpcHandleMemObj(const IpcMemHandle& k) {
  std::shared_lock lock(AllocatedLock_);

  auto it = IpcHandleMemObjMap_.find(k);
  if (it == IpcHandleMemObjMap_.cend()) {
    return nullptr;
  }

  return it->second;
}

//==================================================================================================
bool Device::ValidateVirtualAddressRange(amd::Memory* vaddr_base_obj, amd::Memory* vaddr_sub_obj) {
  // Check if the start of the subbuffer is >= to base start.
  if (vaddr_base_obj->getSvmPtr() > vaddr_sub_obj->getSvmPtr()) {
    LogError("Sub buffer cannot start with addr lesser than base_start.");
    return false;
  }

  // Check if the new size belongs to the vaddr_base_obj range.
  address vaddr_base_end =
      reinterpret_cast<address>(vaddr_base_obj->getSvmPtr()) + vaddr_base_obj->getSize();
  address vaddr_sub_end =
      reinterpret_cast<address>(vaddr_sub_obj->getSvmPtr()) + vaddr_sub_obj->getSize();

  if (vaddr_sub_end > vaddr_base_end) {
    LogError("Sub buffer memory end cannot be greater than base_end. Return nullptr");
    return false;
  }

  return true;
}

//==================================================================================================
amd::Memory* Device::CreateVirtualBuffer(amd::Context& device_context, void* vptr, size_t size,
                                         int deviceId, int locationType, bool parent, bool kForceAlloc) {
  amd::Memory* vaddr_base_obj = nullptr;
  amd::Memory* vaddr_sub_obj = nullptr;
  constexpr bool kSysMemAlloc = false;
  constexpr bool kSkipAlloc = false;

  if (parent) {
    vaddr_base_obj = new (GlbCtx()) amd::Buffer(GlbCtx(), CL_MEM_VA_RANGE_AMD, size, vptr);
    if (vaddr_base_obj == nullptr) {
      LogError("failed to new a va range curr_mem_obj object!");
      return nullptr;
    }
    // This curr_mem_obj->create() does not create an actual memory but stores the memory info
    // with given vptr on ROCr backend.
    if (!vaddr_base_obj->create(nullptr, kSysMemAlloc, kSkipAlloc, kForceAlloc)) {
      LogError("failed to create a va range mem object");
      vaddr_base_obj->release();
      return nullptr;
    }

    amd::MemObjMap::AddVirtualMemObj(vaddr_base_obj->getSvmPtr(), vaddr_base_obj);
  } else {
    // If not parent, but sub-buffer/child, then validate the address range
    vaddr_base_obj = amd::MemObjMap::FindVirtualMemObj(vptr);
    if (vaddr_base_obj == nullptr) {
      LogPrintfError("Cannot find entry in VirtualMemObjMap: 0x%x \n", vptr);
      return nullptr;
    }
    assert(vaddr_base_obj->getMemFlags() & CL_MEM_VA_RANGE_AMD);

    size_t offset =
        (reinterpret_cast<address>(vptr) - reinterpret_cast<address>(vaddr_base_obj->getSvmPtr()));
    vaddr_sub_obj =
        new (device_context) amd::Buffer(device_context, CL_MEM_VA_RANGE_AMD, size, vptr);
    vaddr_sub_obj->SetParent(vaddr_base_obj);
    vaddr_sub_obj->setOrigin(offset);

    // This curr_mem_obj->create() does not create an actual memory but stores the memory info
    // with given vptr on ROCr backend.
    if (!vaddr_sub_obj->create(nullptr, kSysMemAlloc, kSkipAlloc, kForceAlloc)) {
      LogError("failed to create a va range mem object");
      vaddr_sub_obj->release();
      return nullptr;
    }

    vaddr_sub_obj->getUserData().deviceId = deviceId;
    vaddr_sub_obj->getUserData().locationType = locationType;

    if (!ValidateVirtualAddressRange(vaddr_base_obj, vaddr_sub_obj)) {
      LogError("Validation failed on address range, returning nullptr");
      return nullptr;
    }
  }

  if (vptr != nullptr) {
    // Assert to make sure that amd::Memory object has set the right ptr.
    guarantee(vptr == (parent ? vaddr_base_obj->getSvmPtr() : vaddr_sub_obj->getSvmPtr()),
              "amd::Memory object does not have the right ptr");
  }

  return parent ? vaddr_base_obj : vaddr_sub_obj;
}

//==================================================================================================
bool Device::DestroyVirtualBuffer(amd::Memory* vaddr_mem_obj) {
  // Argument nullptr check.
  if (vaddr_mem_obj == nullptr || vaddr_mem_obj->getSvmPtr() == nullptr) {
    LogPrintfError("Mem obj passed is nullptr, vaddr_mem_obj: %p \n", vaddr_mem_obj);
    return false;
  }

  if (vaddr_mem_obj->parent() != nullptr) {
    // If parent is not nullptr, this is the sub-buffer object.
    amd::Memory* vaddr_base_obj = amd::MemObjMap::FindVirtualMemObj(vaddr_mem_obj->getSvmPtr());
    if (vaddr_base_obj == nullptr) {
      LogPrintfError("Cannot find mem obj for ptr: 0x%x", vaddr_mem_obj->getSvmPtr());
      return false;
    }
    vaddr_base_obj->removeSubBuffer(vaddr_mem_obj);
  }

  return true;
}

Device::BlitProgram::~BlitProgram() {
  if (program_ != nullptr) {
    program_->release();
  }
}

bool Device::BlitProgram::create(amd::Device* device, const std::string& extraKernels,
                                 const std::string& extraOptions) {
  std::vector<amd::Device*> devices;
  devices.push_back(device);
  int32_t retval = CL_SUCCESS;
  std::string kernels(device::BlitLinearSourceCode);
  std::string image_kernels(device::BlitImageSourceCode);

  if (device->info().imageSupport_) {
    kernels += image_kernels;
  }

  if (!extraKernels.empty()) {
    kernels += extraKernels;
  }

  // Create a program with all blit kernels
  program_ = new Program(*context_, kernels.c_str(), Program::OpenCL_C);
  if (program_ == nullptr) {
    DevLogPrintfError("Program creation for Kernel: %s failed\n", kernels.c_str());
    return false;
  }

  // Build all kernels
  std::string opt = "-cl-internal-kernel ";
  if (!extraOptions.empty()) {
    opt += extraOptions;
  }
  if (!GPU_DUMP_BLIT_KERNELS) {
    opt += " -fno-enable-dump";
  }
  if (device->settings().kernel_arg_opt_) {
    opt += " -Wb,-amdgpu-kernarg-preload-count=8 ";
  }
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  opt += " -fsanitize=address ";
#endif
#endif
  if ((retval = program_->build(devices, opt.c_str(), nullptr, nullptr, GPU_DUMP_BLIT_KERNELS)) !=
      CL_SUCCESS) {
    DevLogPrintfError("Build failed for Kernel: %s with error code %d\n", kernels.c_str(), retval);
    return false;
  }
  if (!program_->load()) {
    DevLogPrintfError("Could not load the kernels: %s \n", kernels.c_str());
    return false;
  }

  return true;
}

bool Device::init() {
  assert(!Runtime::initialized() && "initialize only once");
  bool ret = false;
  devices_ = nullptr;
  appProfile_.init();


// IMPORTANT: Note that we are initialiing HSA stack first and then
// GPU stack. The order of initialization is signiicant and if changed
// amd::Device::registerDevice() must be accordingly modified.
#if defined(WITH_HSA_DEVICE)
  if ((GPU_ENABLE_PAL != 1) || flagIsDefault(GPU_ENABLE_PAL)) {
    // Return value of roc::Device::init()
    // If returned false, error initializing HSA stack.
    // If returned true, either HSA not installed or HSA stack
    //                   successfully initialized.
    ret = roc::Device::init();
    if (!ret) {
      // abort() commentted because this is the only indication
      // that KFD is not installed.
      // Ignore the failure and assume KFD is not installed.
      // abort();
      DevLogError("KFD is not installed \n");
      // Disable direct dispatch if ROC initialization wasn't successful
      AMD_DIRECT_DISPATCH = flagIsDefault(AMD_DIRECT_DISPATCH) ? false : AMD_DIRECT_DISPATCH;
    }
    if (!amd::IS_HIP) {
      ret |= roc::NullDevice::init();
    }
  }
#endif  // WITH_HSA_DEVICE
#if defined(WITH_PAL_DEVICE)
  if (GPU_ENABLE_PAL != 0) {
    if (GPU_ENABLE_PAL == 1) {
      // PAL path can't support direct dispatch, unless it's forced
      AMD_DIRECT_DISPATCH = flagIsDefault(AMD_DIRECT_DISPATCH) ? false : AMD_DIRECT_DISPATCH;
    }
    ret |= PalDeviceLoad();
  }
#endif  // WITH_PAL_DEVICE
  return ret;
}

void Device::tearDown() {
  if (devices_ != nullptr) {
    for (uint i = 0; i < devices_->size(); ++i) {
      delete devices_->at(i);
    }
    devices_->clear();
    delete devices_;
  }
#if defined(WITH_HSA_DEVICE)
  roc::Device::tearDown();
#endif  // WITH_HSA_DEVICE
#if defined(WITH_PAL_DEVICE)
  if (GPU_ENABLE_PAL != 0) {
    PalDeviceUnload();
  }
#endif  // WITH_PAL_DEVICE
}

Device::Device()
    : settings_(nullptr),
      online_(true),
      activeWait_(false),
      blitProgram_(nullptr),
      context_(nullptr),
      heap_buffer_(nullptr),
      initial_heap_buffer_(nullptr),
      arena_mem_obj_(nullptr),
      vaCacheAccess_(nullptr),
      vaCacheMap_(nullptr),
      index_(0) {
  memset(&info_, '\0', sizeof(info_));
  // By default consider just 1 xcc per device
  info_.numberOfXccs_ = 1;
}

Device::~Device() {
  if (heap_buffer_ != nullptr) {
    delete heap_buffer_;
    heap_buffer_ = nullptr;
  }

  if (initial_heap_buffer_ != nullptr) {
    delete initial_heap_buffer_;
    initial_heap_buffer_ = nullptr;
  }

  if (arena_mem_obj_ != nullptr) {
    arena_mem_obj_->release();
  }

  if (vaCacheMap_) {
    CondLog(vaCacheMap_->size() != 0, "Application didn't unmap all host memory!");
    delete vaCacheMap_;
  }

  for (auto memory : hostcall_allocated_memories_) {
    if (memory != nullptr) {
      amd::MemObjMap::RemoveMemObj(
          reinterpret_cast<void*>(memory->getDeviceMemory(*this, false)->virtualAddress()));
      memory->release();
    }
  }

  hostcall_allocated_memories_.clear();

  delete vaCacheAccess_;
  delete settings_;
  delete[] info_.extensions_;
}

bool Device::ValidateComgr() {
  // Check if Lightning compiler was requested
  constexpr bool kComgrVersioned = true;
  std::call_once(amd::Comgr::initialized, amd::Comgr::LoadLib, kComgrVersioned);
  return amd::Comgr::IsReady();
}

size_t GetMaxStackSize(const std::string& procName) {
  if (procName.find("gfx9") != std::string::npos || procName.find("gfx8") != std::string::npos) {
    return kMaxStackSize9X;
  } else if (procName.find("gfx11") != std::string::npos ||
             procName.find("gfx10") != std::string::npos) {
    return kMaxStackSize11X;
  } else {
    return kMaxStackSize12X;
  }
}

bool Device::create(const Isa& isa) {
  assert(!vaCacheAccess_ && !vaCacheMap_);
  isa_ = &isa;
  // VA Cache Ops Lock
  vaCacheAccess_ = new amd::Monitor(true);
  if (nullptr == vaCacheAccess_) {
    return false;
  }
  vaCacheMap_ = new std::map<uintptr_t, device::Memory*>();
  if (nullptr == vaCacheMap_) {
    return false;
  }
  // For OpenCl default stack size needs to be set to 16K
  if (!amd::IS_HIP) {
    stack_size_ = 16 * Ki;
  }
  maxStackSize_ = GetMaxStackSize(isa_->processorName());
  return true;
}

void Device::registerDevice() {
  assert(Runtime::singleThreaded() && "this is not thread-safe");

  if (devices_ == nullptr) {
    devices_ = new std::vector<Device*>;
  }

  if (info_.available_) {
    static bool defaultIsAssigned = false;
    if (!defaultIsAssigned && online_) {
      defaultIsAssigned = true;
      info_.type_ |= CL_DEVICE_TYPE_DEFAULT;
    }
  }
  if (isOnline()) {
    for (const auto& dev : devices()) {
      if (dev->isOnline()) {
        index_++;
      }
    }
  }
  devices_->push_back(this);
}

void Device::addVACache(device::Memory* memory) const {
  // Make sure system memory has direct access
  if (memory->isHostMemDirectAccess()) {
    // VA cache access must be serialised
    amd::ScopedLock lk(*vaCacheAccess_);
    void* start = memory->owner()->getHostMem();
    size_t offset;
    device::Memory* doubleMap = findMemoryFromVA(start, &offset);

    if (doubleMap == nullptr) {
      // Insert the new entry
      vaCacheMap_->insert(
          std::pair<uintptr_t, device::Memory*>(reinterpret_cast<uintptr_t>(start), memory));
    } else {
      LogError("Unexpected double map() call from the app!");
    }
  }
}

void Device::removeVACache(const device::Memory* memory) const {
  // Make sure system memory has direct access
  if (memory->isHostMemDirectAccess() && memory->owner()) {
    // VA cache access must be serialised
    amd::ScopedLock lk(*vaCacheAccess_);
    void* start = memory->owner()->getHostMem();
    vaCacheMap_->erase(reinterpret_cast<uintptr_t>(start));
  }
}

device::Memory* Device::findMemoryFromVA(const void* ptr, size_t* offset) const {
  // VA cache access must be serialised
  amd::ScopedLock lk(*vaCacheAccess_);

  uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
  auto it = vaCacheMap_->upper_bound(reinterpret_cast<uintptr_t>(ptr));
  if (it == vaCacheMap_->begin()) {
    return nullptr;
  }

  --it;
  device::Memory* mem = it->second;
  if (key >= it->first && key < (it->first + mem->size())) {
    // ptr is in the range
    *offset = key - it->first;
    return mem;
  }
  return nullptr;
}

bool Device::IsTypeMatching(cl_device_type type, bool offlineDevices) {
  if (!(isOnline() || offlineDevices)) {
    return false;
  }

  return (info_.type_ & type) != 0;
}

std::vector<Device*> Device::getDevices(cl_device_type type, bool offlineDevices) {
  std::vector<Device*> result;

  if (devices_ == nullptr) {
    return result;
  }

  // Create the list of available devices
  for (const auto& it : *devices_) {
    // Check if the device type is matched
    if (it->IsTypeMatching(type, offlineDevices)) {
      result.push_back(it);
    }
  }

  return result;
}

size_t Device::numDevices(cl_device_type type, bool offlineDevices) {
  size_t result = 0;

  if (devices_ == nullptr) {
    return 0;
  }

  for (const auto& it : *devices_) {
    // Check if the device type is matched
    if (it->IsTypeMatching(type, offlineDevices)) {
      ++result;
    }
  }

  return result;
}

bool Device::getDeviceIDs(cl_device_type deviceType, uint32_t numEntries, cl_device_id* devices,
                          uint32_t* numDevices, bool offlineDevices) {
  if (numDevices != nullptr && devices == nullptr) {
    *numDevices = (uint32_t)amd::Device::numDevices(deviceType, offlineDevices);
    return (*numDevices > 0) ? true : false;
  }
  assert(devices != nullptr && "check the code above");

  std::vector<amd::Device*> ret = amd::Device::getDevices(deviceType, offlineDevices);
  if (ret.size() == 0) {
    *not_null(numDevices) = 0;
    return false;
  }

  auto it = ret.cbegin();
  uint32_t count = std::min(numEntries, (uint32_t)ret.size());

  while (count--) {
    *devices++ = as_cl(*it++);
    --numEntries;
  }
  while (numEntries--) {
    *devices++ = (cl_device_id)0;
  }

  *not_null(numDevices) = (uint32_t)ret.size();
  return true;
}


bool Device::enableP2P(amd::Device* ptrDev) {
  assert(ptrDev != nullptr);
  amd::ScopedLock lock(lockP2P_);
  Device* peerDev = static_cast<Device*>(ptrDev);
  if (std::find(enabled_p2p_devices_.begin(), enabled_p2p_devices_.end(), peerDev) ==
      enabled_p2p_devices_.end()) {
    enabled_p2p_devices_.push_back(peerDev);
    // Update access to all old allocations
    amd::MemObjMap::UpdateAccess(static_cast<amd::Device*>(this));
  }
  return true;
}

bool Device::disableP2P(amd::Device* ptrDev) {
  assert(ptrDev != nullptr);
  amd::ScopedLock lock(lockP2P_);
  Device* peerDev = static_cast<Device*>(ptrDev);
  // if device is present then remove
  auto it = std::find(enabled_p2p_devices_.begin(), enabled_p2p_devices_.end(), peerDev);
  if (it != enabled_p2p_devices_.end()) {
    enabled_p2p_devices_.erase(it);
  }
  return true;
}

bool Device::UpdateStackSize(uint64_t stackSize) {
  if (stackSize > maxStackSize_) {
    return false;
  }
  stack_size_ = amd::alignUp(stackSize, 16);
  return true;
}

bool Device::UpdateInitialHeapSize(uint64_t initialHeapSize) {
  if (initialHeapSize >= info().globalMemSize_) {
    return false;
  }
  initial_heap_size_ = initialHeapSize;
  return true;
}

char* Device::getExtensionString() {
  std::stringstream extStream;
  size_t size;
  char* result = nullptr;

  // Generate the extension string
  for (uint i = 0; i < ClExtTotal; ++i) {
    if (settings().checkExtension(i)) {
      extStream << OclExtensionsString[i];
    }
  }

  size = extStream.str().size() + 1;

  // Create a single string with all extensions
  result = new char[size];
  if (result != nullptr) {
    memcpy(result, extStream.str().data(), (size - 1));
    result[size - 1] = 0;
  }

  return result;
}

// ================================================================================================
bool Device::IpcCreate(void* dev_ptr, size_t* mem_size, char* handle, size_t* mem_offset) const {
  amd::Memory* amd_mem_obj = amd::MemObjMap::FindMemObj(dev_ptr);
  if (amd_mem_obj == nullptr) {
    DevLogPrintfError("Cannot retrieve amd_mem_obj for dev_ptr: 0x%x", dev_ptr);
    return false;
  }

  // Get the original pointer from the amd::Memory object
  void* orig_dev_ptr = nullptr;
  if (amd_mem_obj->getSvmPtr() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getSvmPtr();
  } else if (amd_mem_obj->getHostMem() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getHostMem();
  } else {
    ShouldNotReachHere();
  }

  // Check if the dev_ptr is lesser than original dev_ptr
  if (orig_dev_ptr > dev_ptr) {
    // If this happens, then revisit FindMemObj logic
    DevLogPrintfError("Original dev_ptr: 0x%x cannot be greater than dev_ptr: 0x%x", orig_dev_ptr,
                      dev_ptr);
    return false;
  }

  // Calculate the memory offset from the original base ptr
  *mem_offset = reinterpret_cast<address>(dev_ptr) - reinterpret_cast<address>(orig_dev_ptr) +
                amd_mem_obj->getOffset();

  *mem_size = amd_mem_obj->getSize();

  auto dev_mem = static_cast<device::Memory*>(amd_mem_obj->getDeviceMemory(*this));
  auto result = dev_mem->ExportHandle(handle);

  return result;
}

// ================================================================================================
bool Device::IpcAttach(const char* handle, size_t mem_size, size_t mem_offset, unsigned int flags,
                       void** dev_ptr) const {
  amd::Memory* amd_mem_obj = nullptr;

  // Create an amd Memory object for the handle
  amd_mem_obj = new (context()) amd::IpcBuffer(context(), flags, mem_offset, mem_size, handle);
  if (amd_mem_obj == nullptr) {
    LogError("failed to create a mem object!");
    return false;
  }

  if (!amd_mem_obj->create(nullptr)) {
    LogError("failed to create a svm hidden buffer!");
    amd_mem_obj->release();
    return false;
  }

  auto mem_obj_exist = amd::MemObjMap::FindMemObj(amd_mem_obj->getSvmPtr());
  if (mem_obj_exist == nullptr) {
    // Add the original mem_ptr to the MemObjMap with newly created amd_mem_obj
    amd::MemObjMap::AddMemObj(amd_mem_obj->getSvmPtr(), amd_mem_obj);
  } else {
    amd_mem_obj->release();
    amd_mem_obj = mem_obj_exist;
    // Memory already exists, just retain the old one.
    amd_mem_obj->retain();
  }

  *dev_ptr = amd_mem_obj->getSvmPtr();

  return true;
}

// ================================================================================================
void Device::IpcDetach(amd::Memory* amd_mem_obj) const {
  // Get the original pointer from the amd::Memory object
  void* orig_dev_ptr = nullptr;
  if (amd_mem_obj->getSvmPtr() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getSvmPtr();
  } else if (amd_mem_obj->getHostMem() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getHostMem();
  } else {
    ShouldNotReachHere();
  }
  if (amd::MemObjMap::FindMemObj(orig_dev_ptr)) {
    amd::MemObjMap::RemoveMemObj(orig_dev_ptr);
  }
}

std::vector<amd::CommandQueue*> Device::getActiveQueues() {
  amd::ScopedLock lock(activeQueuesLock_);
  for (auto it = activeQueues.begin(); it != activeQueues.end();) {
    if ((*it)->referenceCount() == 0) {
      // It is being terminated in HostQueue::terminate().
      // We should not wait for commands in a queue being terminated.
      it = activeQueues.erase(it);
    } else {
      // In case the queue will be destroyed in Stream::Destroy().
      (*it)->retain();
      ++it;
    }
  }
  return std::vector<amd::CommandQueue*>(activeQueues.begin(), activeQueues.end());
}

// =================================================================================================
bool Device::GetHandleForAddressRange(void* dev_ptr, size_t size, void* handle) {
  // Check if the ptr is created through VMM APIs, if true we use different ROCr APIs.
  amd::Memory* amd_base_obj = amd::MemObjMap::FindVirtualMemObj(dev_ptr);
  bool VmmPtr = (amd_base_obj != nullptr) ? true : false;

  // Even if it is VMM ptr, check to make sure the memory is mapped. On hipMalloc'ed ptrs,
  // make sure the memory is allocated.
  amd::Memory* amd_mem_obj = amd::MemObjMap::FindMemObj(dev_ptr);
  if (amd_mem_obj == nullptr) {
    DevLogPrintfError("Cannot retrieve amd_mem_obj for dev_ptr: 0x%x", dev_ptr);
    return false;
  }

  device::Memory* dev_mem = amd_mem_obj->getDeviceMemory(*this);
  return dev_mem->GetFDHandleForMem(dev_ptr, size, VmmPtr, handle);
}

// ================================================================================================
void Device::TrackHostcallMemory(amd::Memory* memory) {
  hostcall_allocated_memories_.push_back(memory);
}

// ================================================================================================
void Device::RemoveHostcallMemory(amd::Memory* memory) {
  auto it =
      std::find(hostcall_allocated_memories_.begin(), hostcall_allocated_memories_.end(), memory);
  if (it != hostcall_allocated_memories_.end()) {
    hostcall_allocated_memories_.erase(it);
  }
}

}  // namespace amd

namespace amd::device {

Settings::Settings() : value_(0) {
  assert((ClExtTotal < (8 * sizeof(extensions_))) && "Too many extensions!");
  extensions_ = 0;
  supportRA_ = true;
  customHostAllocator_ = false;
  waitCommand_ = AMD_OCL_WAIT_COMMAND;
  supportDepthsRGB_ = false;
  fenceScopeAgent_ = AMD_OPT_FLUSH;

  // Amend certain flags for OpenCL
  if (!amd::IS_HIP) {
    if (flagIsDefault(GPU_SINGLE_ALLOC_PERCENT)) {
      GPU_SINGLE_ALLOC_PERCENT = 85;
    }
    if (flagIsDefault(GPU_FORCE_BLIT_COPY_SIZE)) {
      GPU_FORCE_BLIT_COPY_SIZE = 0;
    }
  }

  gwsInitSupported_ = true;
}

void Memory::saveMapInfo(const void* mapAddress, const amd::Coord3D origin,
                         const amd::Coord3D region, uint mapFlags, bool entire,
                         amd::Image* baseMip) {
  // Map/Unmap must be serialized.
  amd::ScopedLock lock(owner()->lockMemoryOps());

  WriteMapInfo info = {};
  WriteMapInfo* pInfo = &info;
  auto it = writeMapInfo_.find(mapAddress);
  if (it != writeMapInfo_.end()) {
    LogWarning("Double map of the same or overlapped region!");
    pInfo = &it->second;
  }

  if (mapFlags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)) {
    pInfo->origin_ = origin;
    pInfo->region_ = region;
    pInfo->entire_ = entire;
    pInfo->unmapWrite_ = true;
  }
  if (mapFlags & CL_MAP_READ) {
    pInfo->unmapRead_ = true;
  }
  pInfo->baseMip_ = baseMip;

  // Insert into the map if it's the first region
  if (++pInfo->count_ == 1) {
    writeMapInfo_.insert({mapAddress, info});
  }
}

ClBinary::ClBinary(const amd::Device& dev, BinaryImageFormat bifVer)
    : dev_(dev),
      binary_(nullptr),
      size_(0),
      flags_(0),
      origBinary_(nullptr),
      origSize_(0),
      encryptCode_(0),
      elfIn_(nullptr),
      elfOut_(nullptr),
      format_(bifVer) {}

ClBinary::~ClBinary() {
  release();

  delete elfIn_;
  delete elfOut_;
}

bool ClBinary::setElfTarget() {
  static const uint32_t Target = 21;
  assert(((0xFFFF8000 & Target) == 0) && "ASIC target ID >= 2^15");
  uint16_t elf_target = static_cast<uint16_t>(0x7FFF & Target);
  return elfOut()->setTarget(elf_target, amd::Elf::CAL_PLATFORM);
}

void ClBinary::init(amd::option::Options* optionsObj) {
  // option has higher priority than environment variable.
  if ((flags_ & BinarySourceMask) != BinaryRemoveSource) {
    // set to zero
    flags_ = (flags_ & (~BinarySourceMask));

    flags_ |= (optionsObj->oVariables->BinSOURCE ? BinarySaveSource : BinaryNoSaveSource);
  }

  if ((flags_ & BinaryLlvmirMask) != BinaryRemoveLlvmir) {
    // set to zero
    flags_ = (flags_ & (~BinaryLlvmirMask));

    flags_ |= (optionsObj->oVariables->BinLLVMIR ? BinarySaveLlvmir : BinaryNoSaveLlvmir);
  }

  if ((flags_ & BinaryIsaMask) != BinaryRemoveIsa) {
    // set to zero
    flags_ = (flags_ & (~BinaryIsaMask));
    flags_ |= ((optionsObj->oVariables->BinEXE) ? BinarySaveIsa : BinaryNoSaveIsa);
  }

  if ((flags_ & BinaryASMask) != BinaryRemoveAS) {
    // set to zero
    flags_ = (flags_ & (~BinaryASMask));
    flags_ |= ((optionsObj->oVariables->BinAS) ? BinarySaveAS : BinaryNoSaveAS);
  }
}

bool ClBinary::isRecompilable(std::string& llvmBinary, amd::Elf::ElfPlatform thePlatform) {
  /* It is recompilable if there is llvmir that was generated for
     the same platform (CPU or GPU) and with the same bitness.

     Note: the bitness has been checked in initClBinary(), no need
           to check it here.
   */
  if (llvmBinary.empty()) {
    DevLogError("LLVM Binary string is empty \n");
    return false;
  }

  uint16_t elf_target;
  amd::Elf::ElfPlatform platform;
  if (elfIn()->getTarget(elf_target, platform)) {
    if (platform == thePlatform) {
      return true;
    }
    if ((platform == amd::Elf::COMPLIB_PLATFORM) &&
        (((thePlatform == amd::Elf::CAL_PLATFORM) &&
          ((elf_target == (uint16_t)EM_HSAIL) || (elf_target == (uint16_t)EM_HSAIL_64))) ||
         ((thePlatform == amd::Elf::CPU_PLATFORM) &&
          ((elf_target == (uint16_t)EM_386) || (elf_target == (uint16_t)EM_X86_64))))) {
      return true;
    }
  }

  DevLogPrintfError("LLVM_Binary: %s is not recompilable \n", llvmBinary.c_str());
  return false;
}

void ClBinary::release() {
  if (isBinaryAllocated() && (binary_ != nullptr)) {
    delete[] binary_;
    binary_ = nullptr;
    flags_ &= ~BinaryAllocated;
  }
}

void ClBinary::saveBIFBinary(const char* binaryIn, size_t size) {
  char* image = new char[size];
  memcpy(image, binaryIn, size);

  setBinary(image, size, true);
  return;
}

bool ClBinary::createElfBinary(bool doencrypt, Program::type_t type) {
  release();

  size_t imageSize;
  char* image;
  assert(elfOut_ && "elfOut_ should be initialized in ClBinary::data()");

  // Insert Version string that builds this binary into .comment section
  const device::Info& devInfo = dev_.info();
  std::string buildVerInfo("@(#) ");
  if (devInfo.version_ != nullptr) {
    buildVerInfo.append(devInfo.version_);
    buildVerInfo.append(".  Driver version: ");
    buildVerInfo.append(devInfo.driverVersion_);
  } else {
    // char OpenCLVersion[256];
    // size_t sz;
    // int32_t ret= clGetPlatformInfo(AMD_PLATFORM, CL_PLATFORM_VERSION, 256, OpenCLVersion, &sz);
    // if (ret == CL_SUCCESS) {
    //     buildVerInfo.append(OpenCLVersion, sz);
    // }

    // If CAL is unavailable, just hard-code the OpenCL driver version
    buildVerInfo.append("OpenCL 1.1" AMD_PLATFORM_INFO);
  }

  elfOut_->addSection(amd::Elf::COMMENT, buildVerInfo.data(), buildVerInfo.size());
  switch (type) {
    case Program::TYPE_NONE: {
      elfOut_->setType(ET_NONE);
      break;
    }
    case Program::TYPE_COMPILED: {
      elfOut_->setType(ET_REL);
      break;
    }
    case Program::TYPE_LIBRARY: {
      elfOut_->setType(ET_DYN);
      break;
    }
    case Program::TYPE_EXECUTABLE: {
      elfOut_->setType(ET_EXEC);
      break;
    }
    default:
      assert(0 && "unexpected elf type");
  }

  if (!elfOut_->dumpImage(&image, &imageSize)) {
    DevLogError("Dump Image failed \n");
    return false;
  }

  if (tempFile_) {
    std::remove(fname_.c_str());
  }

#if defined(HAVE_BLOWFISH_H)
  if (doencrypt) {
    // Increase the size by 64 to accomodate extra headers
    int outBufSize = (int)(imageSize + 64);
    char* outBuf = new char[outBufSize];
    if (outBuf == nullptr) {
      return false;
    }
    memset(outBuf, '\0', outBufSize);

    int outBytes = 0;
    bool success = amd::oclEncrypt(0, image, imageSize, outBuf, outBufSize, &outBytes);
    delete[] image;
    if (!success) {
      delete[] outBuf;
      DevLogError("Cannot succesfully OCL Encrypt Image");
      return false;
    }
    image = outBuf;
    imageSize = outBytes;
  }
#endif

  setBinary(image, imageSize, true);
  return true;
}

Program::binary_t ClBinary::data() const { return {binary_, size_}; }

Program::finfo_t ClBinary::Datafd() const { return {fdesc_, foffset_}; }

std::string ClBinary::DataURI() const { return uri_; }

bool ClBinary::setBinary(const char* theBinary, size_t theBinarySize, bool allocated,
                         amd::Os::FileDesc fdesc, size_t foffset, std::string uri) {
  release();

  size_ = theBinarySize;
  binary_ = theBinary;
  if (allocated) {
    flags_ |= BinaryAllocated;
  }

  fdesc_ = fdesc;
  foffset_ = foffset;
  uri_ = uri;

  return true;
}

void ClBinary::setFlags(int encryptCode) {
  encryptCode_ = encryptCode;
  if (encryptCode != 0) {
    flags_ = (flags_ & (~(BinarySourceMask | BinaryLlvmirMask | BinaryIsaMask | BinaryASMask)));
    flags_ |= (BinaryRemoveSource | BinaryRemoveLlvmir | BinarySaveIsa | BinaryRemoveAS);
  }
}

bool ClBinary::decryptElf(const char* binaryIn, size_t size, char** decryptBin, size_t* decryptSize,
                          int* encryptCode) {
  *decryptBin = nullptr;
#if defined(HAVE_BLOWFISH_H)
  int outBufSize = 0;
  if (amd::isEncryptedBIF(binaryIn, (int)size, &outBufSize)) {
    char* outBuf = new (std::nothrow) char[outBufSize];
    if (outBuf == nullptr) {
      return false;
    }

    // Decrypt
    int outDataSize = 0;
    if (!amd::oclDecrypt(binaryIn, (int)size, outBuf, outBufSize, &outDataSize)) {
      delete[] outBuf;
      DevLogError("Cannot Decrypt Image \n");
      return false;
    }

    *decryptBin = reinterpret_cast<char*>(outBuf);
    *decryptSize = outDataSize;
    *encryptCode = 1;
  }
#endif
  return true;
}

bool ClBinary::setElfIn() {
  if (elfIn_) return true;

  if (binary_ == nullptr) {
    return false;
  }
  elfIn_ = new amd::Elf(ELFCLASSNONE, binary_, size_, nullptr, amd::Elf::ELF_C_READ);
  if ((elfIn_ == nullptr) || !elfIn_->isSuccessful()) {
    delete elfIn_;
    elfIn_ = nullptr;
    LogError("Creating input ELF object failed");
    return false;
  }

  return true;
}

void ClBinary::resetElfIn() {
  delete elfIn_;
  elfIn_ = nullptr;
}

bool ClBinary::setElfOut(unsigned char eclass, const char* outFile, bool tempFile) {
  elfOut_ = new amd::Elf(eclass, nullptr, 0, outFile, amd::Elf::ELF_C_WRITE);
  if ((elfOut_ == nullptr) || !elfOut_->isSuccessful()) {
    delete elfOut_;
    elfOut_ = nullptr;
    LogError("Creating output ELF object failed");
    return false;
  }

  fname_ = outFile;
  tempFile_ = tempFile;

  return setElfTarget();
}

void ClBinary::resetElfOut() {
  delete elfOut_;
  elfOut_ = nullptr;
}

bool ClBinary::loadLlvmBinary(std::string& llvmBinary,
                              amd::Elf::ElfSections& elfSectionType) const {
  // Check if current binary already has LLVMIR
  char* section = nullptr;
  size_t sz = 0;
  const amd::Elf::ElfSections SectionTypes[] = {amd::Elf::LLVMIR, amd::Elf::SPIR, amd::Elf::SPIRV};

  for (int i = 0; i < 3; ++i) {
    if (elfIn_->getSection(SectionTypes[i], &section, &sz) && section && sz > 0) {
      llvmBinary.append(section, sz);
      elfSectionType = SectionTypes[i];
      return true;
    }
  }

  DevLogPrintfError("Cannot Load LLVM Binary: %s \n", llvmBinary.c_str());
  return false;
}

bool ClBinary::isSPIR() const {
  char* section = nullptr;
  size_t sz = 0;
  if (elfIn_->getSection(amd::Elf::LLVMIR, &section, &sz) && section && sz > 0) return false;

  if (elfIn_->getSection(amd::Elf::SPIR, &section, &sz) && section && sz > 0) return true;

  return false;
}

bool ClBinary::isSPIRV() const {
  char* section = nullptr;
  size_t sz = 0;

  if (elfIn_->getSection(amd::Elf::SPIRV, &section, &sz) && section && sz > 0) {
    return true;
  }
  return false;
}

}  // namespace amd::device
