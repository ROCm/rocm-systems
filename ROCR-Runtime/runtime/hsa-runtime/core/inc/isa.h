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

#ifndef HSA_RUNTIME_CORE_ISA_H_
#define HSA_RUNTIME_CORE_ISA_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include "core/inc/amd_hsa_code.hpp"

namespace rocr {
namespace core {

/// @class Wavefront.
/// @brief Wavefront.
class Wavefront final: public amd::hsa::common::Signed<0xA02483F1AD7F101C> {
public:
  /// @brief Default destructor.
  ~Wavefront() {}

  /// @returns Handle equivalent of @p object.
  static hsa_wavefront_t Handle(const Wavefront *object) {
    hsa_wavefront_t handle = { reinterpret_cast<uint64_t>(object) };
    return handle;
  }

  /// @returns Object equivalent of @p handle.
  static Wavefront *Object(const hsa_wavefront_t &handle) {
    Wavefront *object = amd::hsa::common::ObjectAt<Wavefront>(handle.handle);
    return object;
  }

  /// @brief Query value of requested @p attribute and record it in @p value.
  bool GetInfo(const hsa_wavefront_info_t &attribute, void *value) const;

private:
  uint32_t num_threads_;
  /// @brief Default constructor.
  Wavefront() : num_threads_(0) {}
  Wavefront(uint32_t num_threads) : num_threads_(num_threads) {}

  /// @brief Wavefront's friends.
  friend class Isa;
  friend class IsaRegistry;
};

enum class IsaFeature : uint8_t {
  Unsupported,
  Any,
  Disabled,
  Enabled,
};

/// @class Isa.
/// @brief Instruction Set Architecture.
class Isa final: public amd::hsa::common::Signed<0xB13594F2BD8F212D> {
 public:
  /// @brief Isa's version type.
  typedef std::tuple<int32_t, int32_t, int32_t> Version;

  /// @brief Default destructor.
  ~Isa() = default;

  /// @returns Handle equivalent of @p isa_object.
  static hsa_isa_t Handle(const Isa *isa_object) {
    hsa_isa_t isa_handle = { reinterpret_cast<uint64_t>(isa_object) };
    return isa_handle;
  }

  /// @returns Object equivalent of @p isa_handle.
  static Isa *Object(const hsa_isa_t &isa_handle) {
    Isa *isa_object = amd::hsa::common::ObjectAt<Isa>(isa_handle.handle);
    return isa_object;
  }

  /// @returns True if @p code_object_isa and @p agent_isa are compatible,
  /// false otherwise.
  static bool IsCompatible(const Isa &code_object_isa,
                      const Isa &agent_isa, unsigned int codeGenericVersion);

  /// @returns This Isa's version.
  const Version &GetVersion() const {
    return version_;
  }
  /// @returns This Isa's generic target.
  const std::string & GetIsaGeneric() const {return generic_;}


  /// @returns SRAM ECC feature status.
  IsaFeature GetSramecc() const {
    return sramecc_;
  }

  /// @returns XNACK feature status.
  IsaFeature GetXnack() const {
    return xnack_;
  }

  /// @returns This Isa's supported wavefront.
  const Wavefront &GetWavefront() const {
    return wavefront_;
  }

  /// @returns True if SRAMECC feature is supported, false otherwise.
  bool IsSrameccSupported() const {
    return sramecc_ != IsaFeature::Unsupported;
  }

  /// @returns True if XNACK feature is supported, false otherwise.
  bool IsXnackSupported() const {
    return xnack_ != IsaFeature::Unsupported;
  }

  /// @returns This Isa's major version.
  int32_t GetMajorVersion() const {
    return std::get<0>(version_);
  }

  /// @returns This Isa's minor version.
  int32_t GetMinorVersion() const {
    return std::get<1>(version_);
  }

  /// @returns This Isa's stepping.
  int32_t GetStepping() const {
    return std::get<2>(version_);
  }

  /// @brief Isa is always in valid state.
  bool IsValid() const {
    return true;
  }

  /// @returns This Isa's processor name.
  std::string GetProcessorName() const;

  /// @returns This Isa's name consisting of the target triple and target ID.
  std::string GetIsaName() const;

  /// @brief Query value of requested @p attribute and record it in @p value.
  bool GetInfo(const hsa_isa_info_t &attribute, void *value) const;

  /// @returns Round method (single or double) used to implement the floating-
  /// point multiply add instruction (mad) for a given combination of @p fp_type
  /// and @p flush_mode.
  hsa_round_method_t GetRoundMethod(
      hsa_fp_type_t fp_type,
      hsa_flush_mode_t flush_mode) const;

  /// @brief Default constructor.
  Isa()
      : version_(Version(-1, -1, -1)),
        sramecc_(IsaFeature::Unsupported),
        xnack_(IsaFeature::Unsupported) {}
  private:

  // @brief Isa's target ID name.
  std::string targetid_;

  // @brief Isa's generic version, if it exists. "" otherwise.
  std::string generic_;

  /// @brief Isa's version.
  Version version_;

  /// @brief SRAMECC feature.
  IsaFeature sramecc_;

  /// @brief XNACK feature.
  IsaFeature xnack_;

  /// @brief Isa's supported wavefront.
  Wavefront wavefront_;

  /// @brief Isa's friends.
  friend class IsaRegistry;
}; // class Isa

/// @class IsaRegistry.
/// @brief Instruction Set Architecture Registry.
class IsaRegistry final {
 public:
  /// @returns Isa for requested @p full_name, null pointer if not supported.
  static const Isa *GetIsa(const std::string &full_name);

  /// @returns Isa for requested @p version, null pointer if not supported.
  static const Isa *GetIsa(const Isa::Version &version,
                           IsaFeature sramecc = IsaFeature::Any,
                           IsaFeature xnack = IsaFeature::Any);
  static const std::unordered_map<std::string, unsigned int> &
                                                GetSupportedGenericVersions();
 private:
  /// @brief IsaRegistry's map type.
  typedef std::unordered_map<std::string, Isa> IsaMap;

  /// @brief  Default constructor
  IsaRegistry() = delete;

  /// @brief Default destructor
  ~IsaRegistry() = default;

  /// @returns Supported instruction set architectures.
  static const IsaMap& GetSupportedIsas();
}; // class IsaRegistry

} // namespace core
} // namespace rocr

#endif // HSA_RUNTIME_CORE_ISA_HPP_
