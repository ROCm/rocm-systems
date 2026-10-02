// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/gim_util.h"

using cuid::gim::GimAsicInfo;
using cuid::gim::GimClient;
using cuid::gim::GimDeviceEntry;

extern "C" bool __real__ZN4cuid3gim9GimClient12is_availableEv();
extern "C" amdcuid_status_t
__real__ZN4cuid3gim9GimClient11get_devicesERSt6vectorINS0_14GimDeviceEntryESaIS3_EE(
    GimClient*, std::vector<GimDeviceEntry>&);
extern "C" amdcuid_status_t
__real__ZN4cuid3gim9GimClient21get_asic_info_for_bdfERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERNS0_11GimAsicInfoE(
    GimClient*, const std::string&, GimAsicInfo&);

namespace {
// One device per line: "<bdf> <vendor> <device> <revision> <serial>" in hex,
// or "<bdf> -" for a device whose ASIC information GIM cannot report.
constexpr const char* kDevices = "/sys/.cuid-path-gim";

bool gim() { return access(kDevices, F_OK) == 0; }

std::vector<std::vector<std::string>> devices() {
  std::vector<std::vector<std::string>> out;
  std::ifstream input(kDevices);
  for (std::string line; std::getline(input, line);) {
    std::istringstream fields(line);
    std::vector<std::string> device;
    for (std::string field; fields >> field;) device.push_back(field);
    if (!device.empty()) out.push_back(device);
  }
  return out;
}
}  // namespace

extern "C" bool __wrap__ZN4cuid3gim9GimClient12is_availableEv() {
  return gim() || __real__ZN4cuid3gim9GimClient12is_availableEv();
}

extern "C" amdcuid_status_t
__wrap__ZN4cuid3gim9GimClient11get_devicesERSt6vectorINS0_14GimDeviceEntryESaIS3_EE(
    GimClient* client, std::vector<GimDeviceEntry>& out) {
  if (!gim())
    return __real__ZN4cuid3gim9GimClient11get_devicesERSt6vectorINS0_14GimDeviceEntryESaIS3_EE(
        client, out);
  out.clear();
  uint64_t handle = 0;
  for (const auto& device : devices()) out.push_back({++handle, device[0], false});
  return AMDCUID_STATUS_SUCCESS;
}

extern "C" amdcuid_status_t
__wrap__ZN4cuid3gim9GimClient21get_asic_info_for_bdfERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERNS0_11GimAsicInfoE(
    GimClient* client, const std::string& bdf, GimAsicInfo& info) {
  if (!gim())
    return __real__ZN4cuid3gim9GimClient21get_asic_info_for_bdfERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERNS0_11GimAsicInfoE(
        client, bdf, info);
  for (const auto& device : devices()) {
    if (device[0] != bdf || device.size() != 5) continue;
    info.vendor_id = static_cast<uint32_t>(std::stoul(device[1], nullptr, 16));
    info.device_id = std::stoull(device[2], nullptr, 16);
    info.rev_id = static_cast<uint32_t>(std::stoul(device[3], nullptr, 16));
    info.asic_serial = device[4];
    return AMDCUID_STATUS_SUCCESS;
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}
