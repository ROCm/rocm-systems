// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Live devlink port-split probe.
 *
 * Exercises the real DevlinkNetlinkClient against a live kernel: queries the
 * port-split state for one devlink device (a PCI BDF) so the DEVLINK_CMD_PORT_GET
 * dump + attribute parsing can be validated against actual hardware.
 *
 *   sudo ./probe_devlink_port 0000:63:00.0
 *
 * Cross-check against: devlink port show
 */

#include <cstdio>
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
  const auto result = client->get_port_split(bdf);

  if (!result.success) {
    std::printf("device %s: no port object (error_code=%d, reported as unsupported)\n", bdf.c_str(),
                result.error_code);
    return result.error_code == 0 ? 0 : 1;
  }

  std::printf("device %s: splittable=%s split_count=%u\n", bdf.c_str(),
              result.value.splittable ? "true" : "false", result.value.split_count);
  return 0;
}
