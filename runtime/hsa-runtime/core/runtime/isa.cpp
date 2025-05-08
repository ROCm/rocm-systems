////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/isa.h"
#include "core/util/utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <utility>

namespace rocr {
namespace core {

bool Wavefront::GetInfo(
    const hsa_wavefront_info_t &attribute,
    void *value) const {
  if (!value) {
    return false;
  }

  switch (attribute) {
    case HSA_WAVEFRONT_INFO_SIZE: {
      *((uint32_t*)value) = num_threads_;
      return true;
    }
    default: {
      return false;
    }
  }
}

static __forceinline std::string strip_features(const std::string &isa_name) {
  return isa_name.substr(0, isa_name.find(':'));
}

/* static */
bool Isa::IsCompatible(const Isa &code_object_isa,
                       const Isa &agent_isa, unsigned int codeGenericVersion) {

  bool code_obj_isa_is_generic = false;
  auto generic_it = IsaRegistry::GetSupportedGenericVersions().find(
                                                 code_object_isa.GetIsaName());

  if (generic_it != IsaRegistry::GetSupportedGenericVersions().end()) {
    code_obj_isa_is_generic = true;
  }

  assert(code_object_isa.IsSrameccSupported() == agent_isa.IsSrameccSupported()
                                 && agent_isa.GetSramecc() != IsaFeature::Any);
  if ((code_object_isa.GetSramecc() == IsaFeature::Enabled ||
        code_object_isa.GetSramecc() == IsaFeature::Disabled) &&
      code_object_isa.GetSramecc() != agent_isa.GetSramecc())
    return false;

  assert(code_object_isa.IsXnackSupported() == agent_isa.IsXnackSupported() && agent_isa.GetXnack() != IsaFeature::Any);
  if ((code_object_isa.GetXnack() == IsaFeature::Enabled ||
        code_object_isa.GetXnack() == IsaFeature::Disabled) &&
      code_object_isa.GetXnack() != agent_isa.GetXnack())
    return false;

  if (code_obj_isa_is_generic) {
      // Verify the generic code object corresponds to the generic for
      // this isa agent.
      if (strip_features(agent_isa.GetIsaGeneric()) !=
                              strip_features(code_object_isa.GetIsaName())) {
        return false;
      }
      // Verify the generic code object version is greater than or equal to
      // the generic version for this isa agent.
      if (codeGenericVersion < generic_it->second) {
        return false;
      }
  } else if (code_object_isa.GetVersion() != agent_isa.GetVersion()) {
    return false;
  }

  return true;
}

std::string Isa::GetProcessorName() const {
  return strip_features(targetid_);
}

static __forceinline std::string prepend_isa_prefix(const std::string &isa_name) {
  constexpr char hsa_isa_name_prefix[] = "amdgcn-amd-amdhsa--";
  return hsa_isa_name_prefix + isa_name;
}

std::string Isa::GetIsaName() const {
  return prepend_isa_prefix(targetid_);
}

bool Isa::GetInfo(const hsa_isa_info_t &attribute, void *value) const {
  if (!value) {
    return false;
  }

  switch (attribute) {
    case HSA_ISA_INFO_NAME_LENGTH: {
      std::string isa_name = GetIsaName();
      *((uint32_t*)value) = static_cast<uint32_t>(isa_name.size() + 1);
      return true;
    }
    case HSA_ISA_INFO_NAME: {
      std::string isa_name = GetIsaName();
      memset(value, 0x0, isa_name.size() + 1);
      memcpy(value, isa_name.c_str(), isa_name.size());
      return true;
    }
    // deprecated.
    case HSA_ISA_INFO_CALL_CONVENTION_COUNT: {
      *((uint32_t*)value) = 1;
      return true;
    }
    // deprecated.
    case HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE: {
      *((uint32_t*)value) = 64;
      return true;
    }
    // deprecated.
    case HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT: {
      *((uint32_t*)value) = 40;
      return true;
    }
    case HSA_ISA_INFO_MACHINE_MODELS: {
      const bool machine_models[2] = {false, true};
      memcpy(value, machine_models, sizeof(machine_models));
      return true;
    }
    case HSA_ISA_INFO_PROFILES: {
      bool profiles[2] = {true, false};
      if (this->GetVersion() == Version(7, 0, 0) ||
          this->GetVersion() == Version(8, 0, 1)) {
        profiles[1] = true;
      }
      memcpy(value, profiles, sizeof(profiles));
      return true;
    }
    case HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES: {
      const bool rounding_modes[3] = {false, false, true};
      memcpy(value, rounding_modes, sizeof(rounding_modes));
      return true;
    }
    case HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES: {
      const bool rounding_modes[3] = {false, false, true};
      memcpy(value, rounding_modes, sizeof(rounding_modes));
      return true;
    }
    case HSA_ISA_INFO_FAST_F16_OPERATION: {
      if (this->GetMajorVersion() >= 8) {
        *((bool*)value) = true;
      } else {
        *((bool*)value) = false;
      }
      return true;
    }
    case HSA_ISA_INFO_WORKGROUP_MAX_DIM: {
      const uint16_t workgroup_max_dim[3] = {1024, 1024, 1024};
      memcpy(value, workgroup_max_dim, sizeof(workgroup_max_dim));
      return true;
    }
    case HSA_ISA_INFO_WORKGROUP_MAX_SIZE: {
      *((uint32_t*)value) = 1024;
      return true;
    }
    case HSA_ISA_INFO_GRID_MAX_DIM: {
      const hsa_dim3_t grid_max_dim = {INT32_MAX, UINT16_MAX, UINT16_MAX};
      memcpy(value, &grid_max_dim, sizeof(grid_max_dim));
      return true;
    }
    case HSA_ISA_INFO_GRID_MAX_SIZE: {
      *((uint64_t*)value) = UINT64_MAX;
      return true;
    }
    case HSA_ISA_INFO_FBARRIER_MAX_SIZE: {
      *((uint32_t*)value) = 32;
      return true;
    }
    default: {
      return false;
    }
  }
}

hsa_round_method_t Isa::GetRoundMethod(
    hsa_fp_type_t fp_type,
    hsa_flush_mode_t flush_mode) const {
  return HSA_ROUND_METHOD_SINGLE;
}

const Isa *IsaRegistry::GetIsa(const std::string &full_name) {
  auto isareg_iter = GetSupportedIsas().find(full_name);
  return isareg_iter == GetSupportedIsas().end() ?
                                              nullptr : &isareg_iter->second;
}

const Isa *IsaRegistry::GetIsa(const Isa::Version &version, IsaFeature sramecc, IsaFeature xnack) {
  auto isareg_iter = std::find_if(GetSupportedIsas().begin(),
                                  GetSupportedIsas().end(),
                                  [&](const IsaMap::value_type& isareg) {
                                    return isareg.second.GetVersion() == version &&
                                        (isareg.second.GetSramecc() == IsaFeature::Unsupported ||
                                         isareg.second.GetSramecc() == sramecc) &&
                                        (isareg.second.GetXnack() == IsaFeature::Unsupported ||
                                         isareg.second.GetXnack() == xnack);
                                  });
  return isareg_iter == GetSupportedIsas().end() ?
                                              nullptr : &isareg_iter->second;
}


// TODO: c++20 use constexpr or consteval
const std::unordered_map<std::string, unsigned int> &
IsaRegistry::GetSupportedGenericVersions() {
  static const
    std::unordered_map<std::string, unsigned int> * min_gen_versions =
                          new std::unordered_map<std::string, unsigned int> {
    {prepend_isa_prefix("gfx9-generic"), 1},
    {prepend_isa_prefix("gfx9-generic:xnack-"), 1},
    {prepend_isa_prefix("gfx9-generic:xnack+"), 1},
    {prepend_isa_prefix("gfx9-4-generic"), 1},
    {prepend_isa_prefix("gfx9-4-generic:xnack-"), 1},
    {prepend_isa_prefix("gfx9-4-generic:xnack+"), 1},
    {prepend_isa_prefix("gfx9-4-generic:sramecc-"), 1},
    {prepend_isa_prefix("gfx9-4-generic:sramecc+"), 1},
    {prepend_isa_prefix("gfx9-4-generic:sramecc-:xnack-"), 1},
    {prepend_isa_prefix("gfx9-4-generic:sramecc-:xnack+"), 1},
    {prepend_isa_prefix("gfx9-4-generic:sramecc+:xnack-"), 1},
    {prepend_isa_prefix("gfx9-4-generic:sramecc+:xnack+"), 1},
    {prepend_isa_prefix("gfx10-1-generic"), 1},
    {prepend_isa_prefix("gfx10-1-generic:xnack-"), 1},
    {prepend_isa_prefix("gfx10-1-generic:xnack+"), 1},
    {prepend_isa_prefix("gfx10-3-generic"), 1},
    {prepend_isa_prefix("gfx11-generic"), 1},
    {prepend_isa_prefix("gfx12-generic"), 1}
  };
  return *min_gen_versions;
}

const IsaRegistry::IsaMap& IsaRegistry::GetSupportedIsas() {
  // agent, and vendor name length limit excluding terminating nul character.
  constexpr size_t hsa_name_size = 63;
  // This allocation is meant to last until the last thread has exited.
  // It is intentionally not freed.
  static IsaMap* supported_isas = new IsaMap();

  if (supported_isas->size() > 0) {
    return *supported_isas;
  }
  
  auto parse_out_minor_ver = [&](const std::string& generic_name) -> int32_t {
      size_t dot_pos = generic_name.find('.');
      int32_t min;
      if (dot_pos != std::string::npos) {
          std::string minor_version_str = generic_name.substr(dot_pos + 1);
          size_t dash_pos = minor_version_str.find('-');
          if (dash_pos != std::string::npos) {
              minor_version_str = minor_version_str.substr(0, dash_pos);
          }
          min = std::stoi(minor_version_str);
      } else {
          min = 0xFF;
      }
      return min;
  };

// FIXME: Use static_assert when C++17 used.
#define ISAREG_ENTRY_GEN(name, maj, min, stp, sramecc, xnack, wavefrontsize, gen_name) \
 {                                                                                     \
  assert(std::char_traits<char>::length(name) <= hsa_name_size);                       \
  std::string isa_name = prepend_isa_prefix(name);                                     \
  (*supported_isas)[isa_name].targetid_ = name;                                           \
  (*supported_isas)[isa_name].version_ = Isa::Version(maj, min, stp);                     \
  (*supported_isas)[isa_name].sramecc_ = sramecc;                                         \
  (*supported_isas)[isa_name].xnack_ = xnack;                                             \
  (*supported_isas)[isa_name].wavefront_.num_threads_ = wavefrontsize;                    \
  std::string genericname(gen_name);                                                   \
  if (genericname.size() != 0) {                                                       \
    std::string gen_isa_name = prepend_isa_prefix(genericname);                        \
    (*supported_isas)[isa_name].generic_ = gen_isa_name;                                  \
    if ((*supported_isas).find(gen_isa_name) == (*supported_isas).end()) {                   \
      (*supported_isas)[gen_isa_name].targetid_ = genericname;                            \
      (*supported_isas)[gen_isa_name].version_ = Isa::Version(maj, parse_out_minor_ver(genericname), 0xFF); \
      (*supported_isas)[gen_isa_name].sramecc_ = sramecc;                                \
      (*supported_isas)[gen_isa_name].xnack_ = xnack;                                    \
      (*supported_isas)[gen_isa_name].wavefront_.num_threads_ = wavefrontsize;           \
    }                                                                                \
  }                                                                                  \
 }

  const IsaFeature unsupported = IsaFeature::Unsupported;
  const IsaFeature any = IsaFeature::Any;
  const IsaFeature disabled = IsaFeature::Disabled;
  const IsaFeature enabled = IsaFeature::Enabled;

  //               Target ID                 Version   SRAMECC      XNACK
  ISAREG_ENTRY_GEN("gfx700",                 7, 0, 0,  unsupported, unsupported, 64, "")
  ISAREG_ENTRY_GEN("gfx701",                 7, 0, 1,  unsupported, unsupported, 64, "")
  ISAREG_ENTRY_GEN("gfx702",                 7, 0, 2,  unsupported, unsupported, 64, "")
  ISAREG_ENTRY_GEN("gfx801",                 8, 0, 1,  unsupported, any,         64, "")
  ISAREG_ENTRY_GEN("gfx801:xnack-",          8, 0, 1,  unsupported, disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx801:xnack+",          8, 0, 1,  unsupported, enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx802",                 8, 0, 2,  unsupported, unsupported, 64, "")
  ISAREG_ENTRY_GEN("gfx803",                 8, 0, 3,  unsupported, unsupported, 64, "")
  ISAREG_ENTRY_GEN("gfx805",                 8, 0, 5,  unsupported, unsupported, 64, "")
  ISAREG_ENTRY_GEN("gfx810",                 8, 1, 0,  unsupported, any,         64, "")
  ISAREG_ENTRY_GEN("gfx810:xnack-",          8, 1, 0,  unsupported, disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx810:xnack+",          8, 1, 0,  unsupported, enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx900",                 9, 0, 0,  unsupported, any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx900:xnack-",          9, 0, 0,  unsupported, disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx900:xnack+",          9, 0, 0,  unsupported, enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx902",                 9, 0, 2,  unsupported, any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx902:xnack-",          9, 0, 2,  unsupported, disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx902:xnack+",          9, 0, 2,  unsupported, enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx904",                 9, 0, 4,  unsupported, any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx904:xnack-",          9, 0, 4,  unsupported, disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx904:xnack+",          9, 0, 4,  unsupported, enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx906",                 9, 0, 6,  any,         any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx906:xnack-",          9, 0, 6,  any,         disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx906:xnack+",          9, 0, 6,  any,         enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx906:sramecc-",        9, 0, 6,  any,         any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx906:sramecc+",        9, 0, 6,  any,         any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx906:sramecc-:xnack-", 9, 0, 6,  any,         disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx906:sramecc-:xnack+", 9, 0, 6,  any,         enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx906:sramecc+:xnack-", 9, 0, 6,  any,         disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx906:sramecc+:xnack+", 9, 0, 6,  any,         enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx908",                 9, 0, 8,  any,         any,         64, "")
  ISAREG_ENTRY_GEN("gfx908:xnack-",          9, 0, 8,  any,         disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx908:xnack+",          9, 0, 8,  any,         enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx908:sramecc-",        9, 0, 8,  disabled,    any,         64, "")
  ISAREG_ENTRY_GEN("gfx908:sramecc+",        9, 0, 8,  enabled,     any,         64, "")
  ISAREG_ENTRY_GEN("gfx908:sramecc-:xnack-", 9, 0, 8,  disabled,    disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx908:sramecc-:xnack+", 9, 0, 8,  disabled,    enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx908:sramecc+:xnack-", 9, 0, 8,  enabled,     disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx908:sramecc+:xnack+", 9, 0, 8,  enabled,     enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx909",                 9, 0, 9,  unsupported, any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx909:xnack-",          9, 0, 9,  unsupported, disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx909:xnack+",          9, 0, 9,  unsupported, enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx90a",                 9, 0, 10, any,         any,         64, "")
  ISAREG_ENTRY_GEN("gfx90a:xnack-",          9, 0, 10, any,         disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx90a:xnack+",          9, 0, 10, any,         enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx90a:sramecc-",        9, 0, 10, disabled,    any,         64, "")
  ISAREG_ENTRY_GEN("gfx90a:sramecc+",        9, 0, 10, enabled,     any,         64, "")
  ISAREG_ENTRY_GEN("gfx90a:sramecc-:xnack-", 9, 0, 10, disabled,    disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx90a:sramecc-:xnack+", 9, 0, 10, disabled,    enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx90a:sramecc+:xnack-", 9, 0, 10, enabled,     disabled,    64, "")
  ISAREG_ENTRY_GEN("gfx90a:sramecc+:xnack+", 9, 0, 10, enabled,     enabled,     64, "")
  ISAREG_ENTRY_GEN("gfx90c",                 9, 0, 12, unsupported, any,         64, "gfx9-generic")
  ISAREG_ENTRY_GEN("gfx90c:xnack-",          9, 0, 12, unsupported, disabled,    64, "gfx9-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx90c:xnack+",          9, 0, 12, unsupported, enabled,     64, "gfx9-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx942",                 9, 4, 2,  any,         any,         64, "gfx9-4-generic")
  ISAREG_ENTRY_GEN("gfx942:xnack-",          9, 4, 2,  any,         disabled,    64, "gfx9-4-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx942:xnack+",          9, 4, 2,  any,         enabled,     64, "gfx9-4-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx942:sramecc-",        9, 4, 2,  disabled,    any,         64, "gfx9-4-generic:sramecc-")
  ISAREG_ENTRY_GEN("gfx942:sramecc+",        9, 4, 2,  enabled,     any,         64, "gfx9-4-generic:sramecc+")
  ISAREG_ENTRY_GEN("gfx942:sramecc-:xnack-", 9, 4, 2,  disabled,    disabled,    64, "gfx9-4-generic:sramecc-:xnack-")
  ISAREG_ENTRY_GEN("gfx942:sramecc-:xnack+", 9, 4, 2,  disabled,    enabled,     64, "gfx9-4-generic:sramecc-:xnack+")
  ISAREG_ENTRY_GEN("gfx942:sramecc+:xnack-", 9, 4, 2,  enabled,     disabled,    64, "gfx9-4-generic:sramecc+:xnack-")
  ISAREG_ENTRY_GEN("gfx942:sramecc+:xnack+", 9, 4, 2,  enabled,     enabled,     64, "gfx9-4-generic:sramecc+:xnack+")
  ISAREG_ENTRY_GEN("gfx950",                 9, 5, 0,  any,         any,         64, "gfx9-4-generic")
  ISAREG_ENTRY_GEN("gfx950:xnack-",          9, 5, 0,  any,         disabled,    64, "gfx9-4-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx950:xnack+",          9, 5, 0,  any,         enabled,     64, "gfx9-4-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx950:sramecc-",        9, 5, 0,  disabled,    any,         64, "gfx9-4-generic:sramecc-")
  ISAREG_ENTRY_GEN("gfx950:sramecc+",        9, 5, 0,  enabled,     any,         64, "gfx9-4-generic:sramecc+")
  ISAREG_ENTRY_GEN("gfx950:sramecc-:xnack-", 9, 5, 0,  disabled,    disabled,    64, "gfx9-4-generic:sramecc-:xnack-")
  ISAREG_ENTRY_GEN("gfx950:sramecc-:xnack+", 9, 5, 0,  disabled,    enabled,     64, "gfx9-4-generic:sramecc-:xnack+")
  ISAREG_ENTRY_GEN("gfx950:sramecc+:xnack-", 9, 5, 0,  enabled,     disabled,    64, "gfx9-4-generic:sramecc+:xnack-")
  ISAREG_ENTRY_GEN("gfx950:sramecc+:xnack+", 9, 5, 0,  enabled,     enabled,     64, "gfx9-4-generic:sramecc+:xnack+")
  ISAREG_ENTRY_GEN("gfx1010",                10, 1, 0, unsupported, any,         32, "gfx10-1-generic")
  ISAREG_ENTRY_GEN("gfx1010:xnack-",         10, 1, 0, unsupported, disabled,    32, "gfx10-1-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx1010:xnack+",         10, 1, 0, unsupported, enabled,     32, "gfx10-1-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx1011",                10, 1, 1, unsupported, any,         32, "gfx10-1-generic")
  ISAREG_ENTRY_GEN("gfx1011:xnack-",         10, 1, 1, unsupported, disabled,    32, "gfx10-1-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx1011:xnack+",         10, 1, 1, unsupported, enabled,     32, "gfx10-1-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx1012",                10, 1, 2, unsupported, any,         32, "gfx10-1-generic")
  ISAREG_ENTRY_GEN("gfx1012:xnack-",         10, 1, 2, unsupported, disabled,    32, "gfx10-1-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx1012:xnack+",         10, 1, 2, unsupported, enabled,     32, "gfx10-1-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx1013",                10, 1, 3, unsupported, any,         32, "gfx10-1-generic")
  ISAREG_ENTRY_GEN("gfx1013:xnack-",         10, 1, 3, unsupported, disabled,    32, "gfx10-1-generic:xnack-")
  ISAREG_ENTRY_GEN("gfx1013:xnack+",         10, 1, 3, unsupported, enabled,     32, "gfx10-1-generic:xnack+")
  ISAREG_ENTRY_GEN("gfx1030",                10, 3, 0, unsupported, unsupported, 32, "gfx10-3-generic")
  ISAREG_ENTRY_GEN("gfx1031",                10, 3, 1, unsupported, unsupported, 32, "gfx10-3-generic")
  ISAREG_ENTRY_GEN("gfx1032",                10, 3, 2, unsupported, unsupported, 32, "gfx10-3-generic")
  ISAREG_ENTRY_GEN("gfx1033",                10, 3, 3, unsupported, unsupported, 32, "gfx10-3-generic")
  ISAREG_ENTRY_GEN("gfx1034",                10, 3, 4, unsupported, unsupported, 32, "gfx10-3-generic")
  ISAREG_ENTRY_GEN("gfx1035",                10, 3, 5, unsupported, unsupported, 32, "gfx10-3-generic")
  ISAREG_ENTRY_GEN("gfx1036",                10, 3, 6, unsupported, unsupported, 32, "gfx10-3-generic")
  ISAREG_ENTRY_GEN("gfx1100",                11, 0, 0, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1101",                11, 0, 1, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1102",                11, 0, 2, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1103",                11, 0, 3, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1150",                11, 5, 0, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1151",                11, 5, 1, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1152",                11, 5, 2, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1153",                11, 5, 3, unsupported, unsupported, 32, "gfx11-generic")
  ISAREG_ENTRY_GEN("gfx1200",                12, 0, 0, unsupported, unsupported, 32, "gfx12-generic")
  ISAREG_ENTRY_GEN("gfx1201",                12, 0, 1, unsupported, unsupported, 32, "gfx12-generic")
#undef ISAREG_ENTRY_GEN

  return *supported_isas;
}

} // namespace core
} // namespace rocr
