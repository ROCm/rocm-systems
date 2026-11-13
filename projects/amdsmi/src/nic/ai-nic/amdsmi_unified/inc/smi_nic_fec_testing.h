// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_NIC_FEC_TESTING_H_
#define AMDSMI_UNIFIED_NIC_FEC_TESTING_H_

#include <cstddef>
#include <cstdint>

#include "smi_nic_transport.h"

// Library-local test seam: decodes one ETHTOOL_MSG_FEC_GET reply, netlink header included, the way
// get_fec_statistics() does. Reports ENOTSUP in a build without libnl-3. Kept apart from
// smi_nic_testing.h so the netlink layer does not include the interface and system headers.
amd::smi::nic::transport::Result<amd::smi::nic::transport::FecStatistics_t>
smi_nic_fec_stats_from_reply_for_testing(const uint8_t* reply, size_t size);

#endif  // AMDSMI_UNIFIED_NIC_FEC_TESTING_H_
