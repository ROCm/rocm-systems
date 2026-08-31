// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/nic/amd_smi_ainic_device.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

namespace amd::smi {
amdsmi_status_t AMDSmiAINICDevice::amd_query_nic_info(AINICInfo& info) const {
  info = ai_nic_info_;
  return AMDSMI_STATUS_SUCCESS;
}
}  // namespace amd::smi
