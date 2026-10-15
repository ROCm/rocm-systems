// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
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

#include <cstdint>
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

  // UINT8_MAX / UINT32_MAX mean the kernel did not report the attribute.
  const auto splittable = result.value.splittable;
  const auto split_count = result.value.split_count;
  const char* splittable_text = (splittable == UINT8_MAX) ? "N/A" : (splittable ? "true" : "false");
  const auto split_count_text =
      (split_count == UINT32_MAX) ? std::string("N/A") : std::to_string(split_count);
  std::printf("device %s: splittable=%s split_count=%s\n", bdf.c_str(), splittable_text,
              split_count_text.c_str());
  return 0;
}
