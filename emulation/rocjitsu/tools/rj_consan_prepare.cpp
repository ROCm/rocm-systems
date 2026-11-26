// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/prepare.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

namespace {

bool write(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
  if (!std::in_range<std::streamsize>(bytes.size())) {
    return false;
  }
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  file.close();
  return !file.fail();
}

int run(int argc, char **argv) {
  std::vector<std::string_view> entries;
  std::filesystem::path destination;
  if (argc < 5) {
    std::cerr << "Usage: rj_consan_prepare ORIGINAL TARGET ENTRY NEW-DIRECTORY "
                 "[--flat-policy=likely|strict]\n"
                 "   or: rj_consan_prepare ORIGINAL TARGET NEW-DIRECTORY --entry NAME "
                 "[--entry NAME] [--flat-policy=likely|strict]\n";
    return 2;
  }
  int index = 5;
  if (argc == 5 || std::string_view(argv[4]) != "--entry") {
    entries.push_back(argv[3]);
    destination = argv[4];
  } else {
    destination = argv[3];
    for (index = 4; index + 1 < argc; index += 2) {
      if (std::string_view(argv[index]) != "--entry") {
        std::cerr << "Expected --entry NAME\n";
        return 2;
      }
      entries.push_back(argv[index + 1]);
    }
  }
  rocjitsu::consan::FlatPolicy policy = rocjitsu::consan::FlatPolicy::Likely;
  if (index < argc) {
    const std::string_view option(argv[index]);
    if (index + 1 != argc ||
        (option != "--flat-policy=likely" && option != "--flat-policy=strict")) {
      std::cerr << "Expected --flat-policy=likely or --flat-policy=strict\n";
      return 2;
    }
    if (option == "--flat-policy=strict") {
      policy = rocjitsu::consan::FlatPolicy::Strict;
    }
  }
  const rocjitsu::consan::TargetProfile *target = rocjitsu::consan::target_profile(argv[2]);
  if (target == nullptr) {
    std::cerr << "Preparation rejected input: unsupported concrete native LDS target\n";
    return 2;
  }
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  const std::ifstream::pos_type position = file.tellg();
  const std::streamoff size = position;
  const rocjitsu::consan::InputLimits limits;
  if (!file || size <= 0 || !std::in_range<size_t>(size) || !std::in_range<std::streamsize>(size) ||
      std::cmp_greater(size, limits.image_bytes)) {
    std::cerr << "Cannot read bounded original code object\n";
    return 2;
  }
  std::vector<uint8_t> original(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(original.data()), static_cast<std::streamsize>(size));
  if (!file) {
    std::cerr << "Short original code-object read\n";
    return 2;
  }
  rocjitsu::consan::InputError error;
  const rocjitsu::FailureOr<rocjitsu::consan::PreparedImageArtifact> artifact =
      rocjitsu::consan::prepare_image(original, *target, entries, error, policy);
  if (artifact.failed()) {
    std::cerr << "Preparation rejected input: " << error.message << '\n';
    return 2;
  }
  const rocjitsu::FailureOr<std::vector<uint8_t>> facts =
      rocjitsu::consan::encode_image_facts(artifact.value().facts);
  if (facts.failed()) {
    std::cerr << "Cannot encode validated static facts\n";
    return 3;
  }
  std::error_code status;
  if (!std::filesystem::create_directory(destination, status)) {
    std::cerr << "Artifact directory must be new: "
              << (status ? status.message() : "already exists") << '\n';
    return 3;
  }
  // Incomplete writes never publish the ready marker. The runtime requires it
  // and still validates both exact image identities and the facts checksum.
  if (!write(destination / "prepared.hsaco", artifact.value().image) ||
      !write(destination / "facts.bin", facts.value()) ||
      !write(destination / "ready", rocjitsu::consan::kProducerContract)) {
    std::cerr << "Artifact write/close failed\n";
    return 3;
  }
  constexpr std::string_view kHex = "0123456789abcdef";
  std::cout << "selection_profile=";
  for (const uint8_t byte : artifact.value().facts.selection_profile) {
    std::cout << kHex[byte >> 4] << kHex[byte & 15];
  }
  std::cout << '\n';
  for (const rocjitsu::consan::ImageEntryFacts &entry : artifact.value().facts.entries) {
    std::cout << "entry=" << entry.binding.name << " key=" << entry.key
              << " disposition=" << (entry.selected ? "selected" : "native") << '\n';
    if (!entry.selected) {
      continue;
    }
    const rocjitsu::consan::Manifest &manifest = *entry.selected;
    std::cout << "prepared_bytes=" << manifest.prepared.bytes << " sites=" << manifest.sites.size()
              << " entry_offset=" << manifest.entry.prepared_entry_offset
              << " sgprs=" << manifest.entry.prepared_sgprs
              << " vgprs=" << manifest.entry.prepared_vgprs << '\n';
    for (const rocjitsu::consan::SiteFacts &site : manifest.sites) {
      std::cout << "site=" << site.id << " original=" << site.original_offset
                << " guest=" << site.prepared_guest_offset << " probe=" << site.probe_offset
                << " probe_bytes=" << site.probe_size << '\n';
    }
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
