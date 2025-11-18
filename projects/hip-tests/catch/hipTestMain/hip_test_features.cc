/*
 * Copyright (C) Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "hip_test_features.hh"

#include <iostream>
#include <assert.h>
#include <map>
#include "hip_test_context.hh"

std::vector<std::unordered_set<std::string>> GCNArchFeatMap = {
    {"gfx90a", "gfx942", "gfx950"},  // CT_FEATURE_FINEGRAIN_HWSUPPORT
    {"gfx90a", "gfx942", "gfx950"},  // CT_FEATURE_HMM
    {"gfx90a", "gfx942", "gfx950"},  // CT_FEATURE_TEXTURES_NOT_SUPPORTED
};

#if HT_AMD
std::string TrimAndGetGFXName(const std::string& full_gfx_name) {
  std::string gfx_name("");

  // Split the first part of the delimiter
  std::string delimiter = ":";
  auto pos = full_gfx_name.find(delimiter);
  if (pos == std::string::npos) {
    gfx_name = full_gfx_name;
  } else {
    gfx_name = full_gfx_name.substr(0, pos);
  }

  assert(gfx_name.substr(0, 3) == "gfx");
  return gfx_name;
}
#endif

// Check if the GCN Maps
bool CheckIfFeatSupported(enum CTFeatures test_feat, std::string gcn_arch) {
#if HT_NVIDIA
  return true;  // returning true since feature check does not exist for NV.
#elif HT_AMD
  assert(test_feat >= 0 && test_feat < CTFeatures::CT_FEATURE_LAST);
  gcn_arch = TrimAndGetGFXName(gcn_arch);
  assert(gcn_arch != "");
  return (GCNArchFeatMap[test_feat].find(gcn_arch) != GCNArchFeatMap[test_feat].cend());
#else
  std::cout << "Platform has to be either AMD or NVIDIA, asserting..." << std::endl;
  assert(false);
#endif
}

// Return true if agentTarget has corresponding generic target which will be returned in
//  genericTarget;
// false, otherwise.
// Note: it will naturely return false on Nvidia device
bool getGenericTarget(const std::string& agentTarget, std::string& genericTarget) {
  // The map is subject to change per removing policy
  static std::map<std::string, std::string> genericTargetMap{
      // "gfx9-generic"
      {"gfx900", "gfx9-generic"},
      {"gfx902", "gfx9-generic"},
      {"gfx904", "gfx9-generic"},
      {"gfx906", "gfx9-generic"},
      {"gfx909", "gfx9-generic"},
      {"gfx90c", "gfx9-generic"},
      // "gfx9-4-generic
      {"gfx940", "gfx9-4-generic"},
      {"gfx941", "gfx9-4-generic"},
      {"gfx942", "gfx9-4-generic"},
      {"gfx950", "gfx9-4-generic"},
      // "gfx10-1-generic"
      {"gfx1010", "gfx10-1-generic"},
      {"gfx1011", "gfx10-1-generic"},
      {"gfx1012", "gfx10-1-generic"},
      {"gfx1013", "gfx10-1-generic"},
      // "gfx10-3-generic"
      {"gfx1030", "gfx10-3-generic"},
      {"gfx1031", "gfx10-3-generic"},
      {"gfx1032", "gfx10-3-generic"},
      {"gfx1033", "gfx10-3-generic"},
      {"gfx1034", "gfx10-3-generic"},
      {"gfx1035", "gfx10-3-generic"},
      {"gfx1036", "gfx10-3-generic"},
      // "gfx11-generic"
      {"gfx1100", "gfx11-generic"},
      {"gfx1101", "gfx11-generic"},
      {"gfx1102", "gfx11-generic"},
      {"gfx1103", "gfx11-generic"},
      {"gfx1150", "gfx11-generic"},
      {"gfx1151", "gfx11-generic"},
      // "gfx12-generic"
      {"gfx1200", "gfx12-generic"},
      {"gfx1201", "gfx12-generic"},
  };
  auto search = genericTargetMap.find(agentTarget);
  if (search == genericTargetMap.end()) return false;
  genericTarget = search->second;
  return true;
}

/*
Return true, if gcnArchName has corresponding generic target;
       false, otherwise.
If gcnArchName is nullptr, it will be queried from deviceId;
       otherwise, deviceId will be ignored.

The specific arches have the following mapping to generic targets,

Generic GFX11

--offload-arch=gfx11-generic - includes [gfx1100-gfx1103], gfx1150, gfx1151

Generic GFX10.3

--offload-arch=gfx10.3-generic - includes [gfx1030-gfx1036]

Generic GFX10.1

--offload-arch=gfx10.1-generic - includes [gfx1010-gfx1013]

Generic GFX9 / Consumer

--offload-arch=gfx9-generic - includes gfx900, gfx902, gfx904, gfx906, gfx909, gfx90c

Generic GFX9.4 / Data center

--offload-arch=gfx9-4-generic - includes gfx940, gfx941, gfx942, gfx950
*/
bool isGenericTargetSupported(char* gcnArchName, int deviceId) {
  hipDeviceProp_t props{};
  if (gcnArchName == nullptr) {
    if (hipGetDeviceProperties(&props, deviceId) != hipSuccess) return false;
    gcnArchName = props.gcnArchName;
  }
  std::string target{gcnArchName};
  std::string genericTarget{};
  auto pos = target.find(':');
  if (pos != std::string::npos) {
    target[pos] = 0;
    target.resize(pos);
  }
  return getGenericTarget(target, genericTarget);
}
