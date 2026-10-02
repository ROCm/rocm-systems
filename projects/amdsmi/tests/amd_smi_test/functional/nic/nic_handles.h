// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_NIC_NIC_HANDLES_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_NIC_NIC_HANDLES_H_

#include <cstdint>
#include <vector>

#include "amd_smi/amdsmi.h"

// Every NIC handle across all sockets; empty if none present.
inline auto nic_handles() -> std::vector<amdsmi_processor_handle> {
  auto nics = std::vector<amdsmi_processor_handle>{};
  auto socket_count = uint32_t{0};
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
    return nics;
  }
  auto sockets = std::vector<amdsmi_socket_handle>(socket_count);
  if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return nics;
  }
  for (const auto socket : sockets) {
    auto count = uint32_t{0};
    if ((amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_NIC, nullptr,
                                              &count) != AMDSMI_STATUS_SUCCESS) ||
        (count == 0)) {
      continue;
    }
    auto handles = std::vector<amdsmi_processor_handle>(count);
    if (amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_NIC, handles.data(),
                                             &count) == AMDSMI_STATUS_SUCCESS) {
      nics.insert(nics.end(), handles.begin(), handles.end());
    }
  }
  return nics;
}

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_NIC_NIC_HANDLES_H_
