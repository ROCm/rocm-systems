// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/prepare.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

namespace {

bool write(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
  if (!std::in_range<std::streamsize>(bytes.size()))
    return false;
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  file.close();
  return !file.fail();
}

int run(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: rj_consan_prepare ORIGINAL.hsaco TARGET ENTRY NEW-ARTIFACT-DIRECTORY\n";
    return 2;
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
  const rocjitsu::FailureOr<rocjitsu::consan::PreparedArtifact> artifact =
      rocjitsu::consan::prepare(original, *target, argv[3], error);
  if (artifact.failed()) {
    std::cerr << "Preparation rejected input: " << error.message << '\n';
    return 2;
  }
  const rocjitsu::FailureOr<std::vector<uint8_t>> facts =
      rocjitsu::consan::encode_manifest(artifact.value().manifest);
  if (facts.failed()) {
    std::cerr << "Cannot encode validated static facts\n";
    return 3;
  }
  const std::filesystem::path destination(argv[4]);
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
  const rocjitsu::consan::Manifest &manifest = artifact.value().manifest;
  std::cout << "prepared_bytes=" << manifest.prepared.bytes << " sites=" << manifest.sites.size()
            << " entry_offset=" << manifest.entry.prepared_entry_offset
            << " sgprs=" << manifest.entry.prepared_sgprs
            << " vgprs=" << manifest.entry.prepared_vgprs << '\n';
  for (const rocjitsu::consan::SiteFacts &site : manifest.sites)
    std::cout << "site=" << site.id << " original=" << site.original_offset
              << " guest=" << site.prepared_guest_offset << " probe=" << site.probe_offset
              << " probe_bytes=" << site.probe_size << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
