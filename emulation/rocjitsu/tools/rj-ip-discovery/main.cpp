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

#include "rocjitsu/vm/amdgpu/pci/gpu_generation_registry.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
  const rocjitsu::GpuGenerationRegistry &generations = rocjitsu::gpu_generations();
  if (!generations.ok()) {
    std::cerr << "rj-ip-discovery: the generations this build offers are inconsistent: "
              << *generations.error() << "\n";
    return EXIT_FAILURE;
  }
  if (argc != 3) {
    std::cerr << "usage: rj-ip-discovery <generation> <output-path>\n"
              << "  known generations: " << generations.known_ids() << "\n";
    return EXIT_FAILURE;
  }

  // Resolved through the registry the device uses, so a table written here and
  // a table the device publishes come from one factory. A guest that saw
  // different hardware depending on which path delivered its table would be
  // worse off than with either alone, and naming the part is what makes those
  // two the same question.
  const rocjitsu::GpuGenerationDescriptor *generation = generations.find(std::string_view(argv[1]));
  if (generation == nullptr) {
    std::cerr << "rj-ip-discovery: no generation named " << argv[1] << "; known generations are "
              << generations.known_ids() << "\n";
    return EXIT_FAILURE;
  }

  const rocjitsu::IpDiscoveryBuild built =
      rocjitsu::build_ip_discovery_table(generation->discovery_factory({}));
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

  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char *>(table.data()),
            static_cast<std::streamsize>(table.size()));
  if (!out) {
    std::cerr << "rj-ip-discovery: cannot write " << argv[2] << "\n";
    return EXIT_FAILURE;
  }

  std::cerr << "rj-ip-discovery: wrote " << table.size() << " bytes to " << argv[2] << "\n";
  return EXIT_SUCCESS;
}
