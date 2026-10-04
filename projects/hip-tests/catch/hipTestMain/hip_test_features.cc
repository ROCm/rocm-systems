/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_test_features.hh"

#include <map>
#include "hip_test_context.hh"

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
      // "gfx12-5-generic"
      {"gfx1250", "gfx12-5-generic"},
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
