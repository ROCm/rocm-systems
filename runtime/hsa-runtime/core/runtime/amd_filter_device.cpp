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

#include "core/inc/amd_filter_device.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <climits>

#include "hsakmt/hsakmt.h"

#include "core/util/utils.h"
#include "core/inc/runtime.h"
#include "core/inc/amd_cpu_agent.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"

namespace rocr {
namespace AMD {

bool RvdFilter::FilterDevices() {
  return core::Runtime::runtime_singleton_->flag().filter_visible_gpus();
}

bool RvdFilter::SelectZeroDevices() {
  const std::string& envVal = core::Runtime::runtime_singleton_->flag().visible_gpus();
  return envVal.empty();
}

void RvdFilter::BuildRvdTokenList() {
  // Determine if user has chosen ZERO devices to be surfaced
  const std::string& envVal = core::Runtime::runtime_singleton_->flag().visible_gpus();
  if (envVal.empty()) {
    return;
  }

  // Parse env value into tokens separated by comma (',') delimiter
  std::string token;
  char separator = ',';
  std::stringstream stream(envVal);
  while (getline(stream, token, separator)) {
    std::transform(token.begin(), token.end(), token.begin(), ::toupper);
    token = trim(token);
    rvdTokenList_.push_back(token);
  }
}

void RvdFilter::BuildDeviceUuidList(uint32_t numNodes) {
  HSAKMT_STATUS status;
  HsaNodeProperties props = {0};
  for (HSAuint32 idx = 0; idx < numNodes; idx++) {
    // Query for node properties and ignore Cpu devices
    status = HSAKMT_CALL(hsaKmtGetNodeProperties(idx, &props));
    if (status != HSAKMT_STATUS_SUCCESS) {
      continue;
    }
    if (props.NumFComputeCores == 0) {
      continue;
    }

    // For devices whose UUID is zero build a string that
    // will not match user provided value
    if (props.UniqueID == 0) {
      devUuidList_.push_back("Invalid-UUID");
      continue;
    }

    // For devices that support valid UUID values capture UUID
    // value into a upper case hex string of length 16 including
    // leading zeros if necessary
    std::stringstream stream;
    stream << "GPU-" << std::setfill('0') << std::setw(sizeof(uint64_t) * 2) << std::hex
           << props.UniqueID;
    std::string uuidVal(stream.str());
    std::transform(uuidVal.begin(), uuidVal.end(), uuidVal.begin(), ::toupper);
    devUuidList_.push_back(std::move(uuidVal));
  }
}

int32_t RvdFilter::ProcessUuidToken(const std::string& token) {
  // Determine if token exceeds max length of a UUID string
  uint32_t tokenLen = token.length();
  if ((tokenLen < 5) || (tokenLen > 20)) {
    return -1;
  }

  // Track the number of devices user token matches
  int32_t devIdx = -1;
  int32_t compareVal = -1;
  uint32_t numGpus = devUuidList_.size();
  for (uint32_t idx = 0; idx < numGpus; idx++) {
    uint32_t uuidLen = devUuidList_[idx].length();

    // Token could match UUID of another device
    if (tokenLen > uuidLen)
      continue;

    // Token could match as substring of device UUID
    compareVal = token.compare(0, tokenLen, devUuidList_[idx], 0, tokenLen);

    // Check if user Uuid matches with ROCt Uuid
    if (compareVal == 0) {
      if (devIdx != -1) {
        return -1;
      }
      devIdx = idx;
    }
  }

  // Return value includes possibility of both
  // finding or not finding a device
  return devIdx;
}

uint32_t RvdFilter::BuildUsrDeviceList() {
  // Get number of Gpu devices and user specified tokens
  uint32_t numGpus = devUuidList_.size();
  uint32_t loopCnt = std::min(numGpus, uint32_t(rvdTokenList_.size()));

  // Evaluate tokens into device index or UUID values
  int32_t usrIdx = 0;
  int32_t devIdx = -1;
  for (uint32_t idx = 0; idx < loopCnt; idx++) {
    // User token to be evaluated as UUID or device index
    std::string& token = rvdTokenList_[idx];

    // Token encodes a UUID valaue
    if (token.at(0) == 'G') {
      devIdx = ProcessUuidToken(token);
      if (devIdx == -1) {
        return usrDeviceList_.size();
      }

      // Token encodes device index
    } else {
      char* end = nullptr;
      const char* tmp = token.c_str();
      devIdx = std::strtol(tmp, &end, 0);
      if (*end != '\0') {
        return usrDeviceList_.size();
      }
    }

    // Rvd Token evaluates to wrong device index
    if ((devIdx < 0) || (devIdx >= numGpus)) {
      return usrDeviceList_.size();
    }

    // Determine if device index is previously seen
    // Such indices are interpreted as terminators
    bool exists = (usrDeviceList_.find(devIdx) != usrDeviceList_.end());
    if (exists) {
      return usrDeviceList_.size();
    }

    // Add index to the list of devices that will be
    // surfaced upon device enumeration
    usrDeviceList_[devIdx] = usrIdx++;
  }

  return usrDeviceList_.size();
}

uint32_t RvdFilter::GetUsrDeviceListSize() { return usrDeviceList_.size(); }

int32_t RvdFilter::GetUsrDeviceRank(uint32_t roctIdx) {
  const auto& it = usrDeviceList_.find(roctIdx);
  if (it != usrDeviceList_.end()) {
    return it->second;
  }
  return -1;
}

#ifndef NDEBUG
void RvdFilter::SetDeviceUuidList() {
  uint64_t dbgUuid[] = {0xBABABABABABABABA, 0xBABABABABABAABBA, 0xBABABABAABBAABBA,
                        0xBABAABBAABBAABBA, 0xABBAABBAABBAABBA, 0xABBAABBAABBABABA,
                        0xABBAABBABABABABA, 0xABBABABABABABABA};

  // Override or Set Uuid values for the first four devices
  uint32_t numGpus = devUuidList_.size();
  uint32_t numUuids = (sizeof(dbgUuid) / sizeof(uint64_t));
  for (uint32_t idx = 0; (idx < numGpus && (idx < numUuids)); idx++) {
    std::stringstream stream;

    // For devices whose UUID is zero
    if (dbgUuid[idx] == 0) {
      stream << "GPU-XX";
      continue;
    }

    // For devices that support valid UUID values
    stream << "GPU-" << std::setfill('0') << std::setw(sizeof(uint64_t) * 2) << std::hex
           << dbgUuid[idx];
    std::string uuidVal(stream.str());
    std::transform(uuidVal.begin(), uuidVal.end(), uuidVal.begin(), ::toupper);
    devUuidList_[idx] = std::move(uuidVal);
  }
}

void RvdFilter::PrintDeviceUuidList() {
  uint32_t numGpus = devUuidList_.size();
  for (uint32_t idx = 0; idx < numGpus; idx++) {
    std::cout << "Dev[" << idx << "]: " << devUuidList_[idx];
    std::cout << std::endl << std::flush;
  }
}

void RvdFilter::PrintUsrDeviceList() {
  // Flip the map values as value indicates surface rank
  for (auto const& elem : usrDeviceList_) {
    std::cout << "UsrDev[" << elem.second << "]: " << elem.first;
    std::cout << std::endl << std::flush;
  }
}

void RvdFilter::PrintRvdTokenList() {
  uint32_t numTokens = rvdTokenList_.size();
  for (uint32_t idx = 0; idx < numTokens; idx++) {
    std::cout << "Token[" << idx << "]: " << rvdTokenList_[idx];
    std::cout << std::endl << std::flush;
  }
}
#endif

}  // namespace amd
}  // namespace rocr
