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

#ifndef HSA_RUNTIME_CORE_INC_AMD_FILTER_DEVICE_H_
#define HSA_RUNTIME_CORE_INC_AMD_FILTER_DEVICE_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <sstream>

// Forward declaration of the HsaNodeProperties.
struct _HsaNodeProperties;
using HsaNodeProperties = _HsaNodeProperties;

namespace rocr {
namespace AMD {

// ROCr allows users to filter and reorder various Gpu devices that are
// present on ROCm system. This ability is made available via environment
// variable ROCR_VISIBLE_DEVICES (RVD). Users are allowed to specify a list
// of Gpu Identifiers separated by comma delimiter as the value of this env
// variable.
//
// On a ROCm platform instance, a Gpu device could be identified by its:
//
//    Index - Position at which ROCr reports it upon device enumeration
//    UUID  - A string that is unique and is immutable i.e. tags Gpu
//            instance across systems and power cycles. UUID values
//            are defined to begin with "GPU-" prefix
//
//    @note: Not all Gpu devices will report valid UUID's. For example,
//    Only devices from Gfx9 and later will encode valid UUID's. To account
//    for this and other reasons, the UUID string "GPU-XX" is defined as
//    indicating those devices. Users can still select those Gpu devices
//    by using their enumeration index
//
//  Users are allowed to select a device by specifying its UUID string in
//  full or part. A UUID string that does not uniquely match an agent's
//  valid UUID prefix is interpreted as terminating. The UUID string
//  "GPU-XX" will not match and therefore will terminate
//
//  RVD interpreter treats an empty token list as filtering all devices.
//    Users can use this mode to report ZERO Gpu devices
//
//  RVD interpreter treats a token as Illegal if can't be evaluated into an
//    instance of Device UUID or Enumeration Index
//
//  RVD interpreter treats a Legal instance of Enumeration Index as Terminating
//    if any ONE of the following conditions apply:
//      Value of index lies outside the interval [0 - (numGpuDevices - 1)]
//      Value of index maps to a device that has been previously selected
//
//  RVD interpreter treats a Legal instance of Device UUID as Terminating
//    if any ONE of the following conditions apply:
//      Value of UUID is the literal "GPU-XX"
//      Value of UUID matches ZERO devices on system
//      Value of UUID matches TWO or more devices on system
//      Value of UUID maps to a device that has been previously selected
//
//  RVD interpreter builds the list of Gpu devices to surface using tokens
//    that are Legal and NOT Terminating
//
//  Following are some examples of RVD value strings and their intepretation
//  on a ROCm system with four Gpu devices. Assume for now the UUID's of the
//  four Gpu devices are:
//    Gpu-0: "GPU-BABABABABABABABA"
//    Gpu-1: "GPU-ABBAABBAABBAABBA"
//    Gpu-2: "GPU-BABAABBAABBABABA"
//    Gpu-3: "GPU-ABBABABABABAABBA"
//
//    Surface ZERO devices
//    A1) ROCR_VISIBLE_DEVICES=""
//    A2) ROCR_VISIBLE_DEVICES="-1"
//    A3) ROCR_VISIBLE_DEVICES="GPU-XX"
//
//    Surface Gpu-3 and Gpu-0 devices in that order
//    B) ROCR_VISIBLE_DEVICES="3,GPU-BABABABABABABABA,4"
//
//    Surface Gpu-1 and Gpu-2 devices in that order
//    C) ROCR_VISIBLE_DEVICES="1,GPU-ABBAABBAABBAABBA,GPU-XX"
//
//    Surface Gpu-3 and Gpu-2 devices in that order
//    D) ROCR_VISIBLE_DEVICES="3,GPU-BABAABBA,GPU-XX"
//
class RvdFilter {
 public:
  /// @brief Constructor
  RvdFilter() {}

  // @brief Destructor.
  ~RvdFilter() {}

  /// @brief Determine if user has specified environment variable
  /// ROCR_VISIBLE_DEVICES (RVD) to filter and reorder Gpu devices
  ///
  /// @return TRUE if user has defined the env RVD
  static bool FilterDevices();

  /// @brief Determine if user has specified environment variable
  /// ROCR_VISIBLE_DEVICES (RVD) to filter out all Gpu devices i.e.
  /// surface ZERO devices
  ///
  /// @return TRUE if user has specified ZERO to be surfaced
  bool SelectZeroDevices();

  /// @brief Builds the list of tokens specified by user to filter
  /// and reorder Gpu devices. A token represents either a Gpu's
  /// enumeration index or its UUID value. It is possible for the
  /// list to have no tokens i.e. user has selected zero devices
  void BuildRvdTokenList();

  /// @brief Build the list of Gpu device UUIDs as enumerated by ROCt
  ///
  /// @param numNodes Number of ROCm devices present on system, includes
  /// both Cpu and Gpu's devices
  void BuildDeviceUuidList(const std::vector<HsaNodeProperties>& node_props);

  /// @brief Build the list of Gpu devices that will be enumerated to user
  ///
  /// @return Number of Gpu devices to surface upon devices enumeration
  uint32_t BuildUsrDeviceList();

  /// @brief Processes UUID token and returns its enumeration index
  ///
  /// @param token RVD token encoding a device's UUID value
  /// @return int32_t if it is valid, -1 otherwise
  int32_t ProcessUuidToken(const std::string& token);

  /// @brief Get the number of Gpu devices that will be surface
  /// upon device enumeration
  ///
  /// @uint32_t Number of devices to enumerate including possibly
  /// ZERO devices
  uint32_t GetUsrDeviceListSize();

  /// @brief Return the rank of queried Gpu device. If queried device
  /// is surfaced the number of Gpu devices that will be surface
  /// upon device enumeration
  ///
  /// @int32_t -1 if queried device is not surfaced, else a value in
  /// the range [0 - (numGpus - 1)]
  int32_t GetUsrDeviceRank(uint32_t roctIdx);

#ifndef NDEBUG
  /// @brief Set debug UUID values to Gpu devices. This is intended to
  /// help debug and test RVD module functionality
  void SetDeviceUuidList();

  /// @brief Print the list of Uuids of Gpu devices present on system
  void PrintDeviceUuidList();

  /// @brief Print the list of Gpu devices per their enumeration order
  void PrintUsrDeviceList();

  /// @brief Print the list of tokens specified by user to filter
  /// and reorder Gpu devices
  void PrintRvdTokenList();
#endif

 private:
  /// @brief List of tokens specified by user to select and reorder
  std::vector<std::string> rvdTokenList_;

  /// @brief Ordered list of ROCt enumerated Gpu device's UUID values
  std::vector<std::string> devUuidList_;

  /// @brief Ordered list of ROCr enumerated Gpu devices
  std::map<uint32_t, int32_t> usrDeviceList_;

};  // End of class RvdFilter

}  // namespace amd
}  // namespace rocr

#endif  // header guard - HSA_RUNTIME_CORE_INC_AMD_FILTER_DEVICE_H_
