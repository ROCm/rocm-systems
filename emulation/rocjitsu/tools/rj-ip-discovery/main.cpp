// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief Writes the IP discovery table a guest driver reads from a device.
///
/// @details The driver can be told to load this table from a file instead of
/// from the device, which is how a generated one is tried without first teaching
/// the device to publish it. Drop the output at
/// `/lib/firmware/amdgpu/ip_discovery.bin` in a guest and boot with
/// `amdgpu.discovery=2`.

#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery_profile.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: rj-ip-discovery <output-path>\n";
    return EXIT_FAILURE;
  }

  const rocjitsu::IpDiscoveryBuild built =
      rocjitsu::build_ip_discovery_table(rocjitsu::gfx1250_discovery_spec());
  if (!built.ok()) {
    std::cerr << "rj-ip-discovery: cannot build a table: " << built.problem << "\n";
    return EXIT_FAILURE;
  }
  const std::vector<std::byte> &table = built.table;
  const rocjitsu::IpDiscoveryValidation checked = rocjitsu::validate_ip_discovery_table(table);
  if (!checked.valid) {
    std::cerr << "rj-ip-discovery: refusing to write an unusable table: " << checked.problem
              << "\n";
    return EXIT_FAILURE;
  }
  // The format permits a larger table than the driver will load from a file, so
  // a well-formed one can still be unusable by the path this tool writes for.
  if (table.size() > rocjitsu::kDiscoveryTableBytes) {
    std::cerr << "rj-ip-discovery: refusing to write " << table.size() << " bytes, more than the "
              << rocjitsu::kDiscoveryTableBytes << " the driver loads\n";
    return EXIT_FAILURE;
  }

  std::ofstream out(argv[1], std::ios::binary);
  out.write(reinterpret_cast<const char *>(table.data()),
            static_cast<std::streamsize>(table.size()));
  if (!out) {
    std::cerr << "rj-ip-discovery: cannot write " << argv[1] << "\n";
    return EXIT_FAILURE;
  }

  std::cerr << "rj-ip-discovery: wrote " << table.size() << " bytes to " << argv[1] << "\n";
  return EXIT_SUCCESS;
}
