// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Live devlink health probe.
 *
 * Unlike test_nic_telemetry_suite (which runs against fakes), this exercises the
 * real DevlinkNetlinkClient against a live kernel. It enumerates the health
 * reporters for one devlink device (a PCI BDF) and prints them, so the netlink
 * dump + attribute parsing can be validated against actual hardware.
 *
 *   sudo ./probe_devlink_health 0000:63:00.0
 *
 * Cross-check against: devlink -j health show
 */

#include <cstdio>
#include <cstdlib>
#include <string>

#include "smi_devlink_netlink.h"

namespace dl = amd::nic::netlink;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <pci-bdf>   e.g. %s 0000:63:00.0\n", argv[0], argv[0]);
    return 2;
  }
  const std::string bdf = argv[1];

  auto client = dl::create_devlink_client();
  const auto result = client->get_health_reporters(bdf);

  if (!result.success) {
    std::printf("get_health_reporters(%s) failed: error_code=%d\n", bdf.c_str(), result.error_code);
    return 1;
  }

  std::printf("device %s: %zu reporter(s)\n", bdf.c_str(), result.value.size());
  for (const auto& r : result.value) {
    std::printf("  reporter=%-16s state=%-7s error_count=%u\n", r.name,
                r.healthy ? "healthy" : "error", r.error_count);
  }
  if (result.value.empty()) {
    std::printf("  (no reporters — device exposes none, reported as unsupported)\n");
  }
  return 0;
}
