// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/application_catalog.h"
#include "rocjitsu/code/patch/consan/prepare.h"
#include "util/log.h"

#include <algorithm>

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
  if (argc == 2 && std::string_view(argv[1]) == "--producer-contract") {
    constexpr std::string_view digits = "0123456789abcdef";
    for (const uint8_t byte : rocjitsu::consan::kProducerContract)
      std::cout << digits[byte >> 4] << digits[byte & 15];
    std::cout << '\n';
    return std::cout ? 0 : 3;
  }
  if (argc == 3 && std::string_view(argv[1]) == "--validate-application-catalog") {
    util::StringDiagnostic diagnostic;
    const rocjitsu::FailureOr<rocjitsu::consan::ApplicationCatalog> catalog =
        rocjitsu::consan::load_application_catalog(argv[2], diagnostic.emitter());
    if (catalog.failed()) {
      std::cerr << diagnostic.message() << '\n';
      return 2;
    }
    std::cout << "images=" << catalog.value().images.size() << '\n';
    return std::cout ? 0 : 3;
  }
  std::vector<std::string_view> entries;
  std::filesystem::path destination;
  rocjitsu::consan::ScopeKind scope = rocjitsu::consan::ScopeKind::NativeLdsNoSynchronization;
  bool profile_seen = false;
  bool require_all = false;
  if (argc == 5 && !std::string_view(argv[4]).starts_with("--")) {
    entries.push_back(argv[3]);
    destination = argv[4];
  } else if (argc >= 4) {
    destination = argv[3];
    for (int index = 4; index < argc; ++index) {
      const std::string_view option(argv[index]);
      if (option == "--require-all" && !require_all) {
        require_all = true;
        continue;
      }
      if (index + 1 >= argc) {
        std::cerr << "Expected a value for " << option << '\n';
        return 2;
      }
      const std::string_view value(argv[++index]);
      if (option == "--profile" && !profile_seen &&
          (value == "lds" || value == "execution-trace")) {
        profile_seen = true;
        scope = value == "execution-trace"
                    ? rocjitsu::consan::ScopeKind::ExecutionTrace
                    : rocjitsu::consan::ScopeKind::NativeLdsNoSynchronization;
      } else if (option == "--entry") {
        entries.push_back(value);
      } else {
        std::cerr << "Expected --entry NAME or one --profile lds|execution-trace\n";
        return 2;
      }
    }
  } else {
    std::cerr << "Usage: rj_consan_prepare ORIGINAL TARGET NEW-DIRECTORY [--profile "
                 "lds|execution-trace] [--require-all] [--entry NAME ...]\n"
                 "   or: rj_consan_prepare ORIGINAL TARGET ENTRY NEW-DIRECTORY\n";
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
  const rocjitsu::FailureOr<rocjitsu::consan::PreparationResult> preparation =
      rocjitsu::consan::prepare_image_best_effort(original, *target, entries, error, scope);
  if (preparation.failed()) {
    std::cerr << "Preparation rejected input: " << error.message << '\n';
    return 2;
  }
  const rocjitsu::consan::PreparationResult &result = preparation.value();
  const size_t skipped = static_cast<size_t>(
      std::ranges::count_if(result.entries, [](const rocjitsu::consan::EntryPreparation &entry) {
        return !entry.instrumented && !entry.no_probes_needed;
      }));
  if (require_all && skipped != 0) {
    const std::vector<rocjitsu::consan::EntryPreparation>::const_iterator first =
        std::ranges::find_if(result.entries, [](const rocjitsu::consan::EntryPreparation &entry) {
          return !entry.instrumented && !entry.no_probes_needed;
        });
    std::cerr << "Preparation rejected incomplete requested coverage: " << skipped
              << " requested entries are blocked; " << first->entry << ": " << first->reason.message
              << '\n';
    return 2;
  }
  std::vector<uint8_t> encoded_facts;
  if (result.artifact) {
    const rocjitsu::FailureOr<std::vector<uint8_t>> facts =
        rocjitsu::consan::encode_image_facts(result.artifact->facts);
    if (facts.failed()) {
      std::cerr << "Cannot encode validated static facts\n";
      return 3;
    }
    encoded_facts = facts.value();
  }
  const std::string report = rocjitsu::consan::render_preparation_report(result);
  if (report.empty()) {
    std::cerr << "Cannot encode consistent preparation outcomes\n";
    return 3;
  }
  std::error_code status;
  if (!std::filesystem::create_directory(destination, status)) {
    std::cerr << "Artifact directory must be new: "
              << (status ? status.message() : "already exists") << '\n';
    return 3;
  }
  const std::span<const uint8_t> report_bytes(reinterpret_cast<const uint8_t *>(report.data()),
                                              report.size());
  if (result.artifact) {
    // Incomplete writes never publish the ready marker. Report publication is
    // part of preparation success, while facts retain the exact selected set.
    if (!write(destination / "prepared.hsaco", result.artifact->image) ||
        !write(destination / "facts.bin", encoded_facts) ||
        !write(destination / "preparation.json", report_bytes) ||
        !write(destination / "ready", rocjitsu::consan::kProducerContract)) {
      std::cerr << "Artifact write/close failed\n";
      return 3;
    }
  } else {
    // Native-only output is an explicit no-instrumentation outcome. Configure
    // original.hsaco as a native image; it has no selected facts or capture.
    if (!write(destination / "original.hsaco", original) ||
        !write(destination / "preparation.json", report_bytes)) {
      std::cerr << "Artifact write/close failed\n";
      return 3;
    }
  }
  if (skipped != 0)
    util::Logger::warn("ConSan requested=", result.entries.size(), " instrumented=",
                       std::ranges::count_if(result.entries,
                                             [](const rocjitsu::consan::EntryPreparation &entry) {
                                               return entry.instrumented;
                                             }),
                       " blocked=", skipped, "; see ", (destination / "preparation.json").string());
  if (!result.artifact) {
    std::cout << "outcome=native_only original=" << (destination / "original.hsaco").string()
              << '\n';
    return std::cout ? 0 : 3;
  }
  const rocjitsu::consan::PreparedImageArtifact &artifact = *result.artifact;
  constexpr std::string_view kHex = "0123456789abcdef";
  std::cout << "selection_profile=";
  for (const uint8_t byte : artifact.facts.selection_profile) {
    std::cout << kHex[byte >> 4] << kHex[byte & 15];
  }
  std::cout << '\n';
  for (const rocjitsu::consan::ImageEntryFacts &entry : artifact.facts.entries) {
    std::cout << "entry=" << entry.binding.name << " key=" << entry.key
              << " disposition=" << (entry.selected ? "selected" : "native") << '\n';
    if (!entry.selected) {
      continue;
    }
    const rocjitsu::consan::Manifest &manifest = *entry.selected;
    std::cout << "prepared_bytes=" << manifest.prepared.bytes << " sites=" << manifest.sites.size()
              << " boundaries=" << manifest.boundaries.size()
              << " entry_offset=" << manifest.entry.prepared_entry_offset
              << " sgprs=" << manifest.entry.prepared_sgprs
              << " vgprs=" << manifest.entry.prepared_vgprs << '\n';
    for (const rocjitsu::consan::SiteFacts &site : manifest.sites) {
      std::cout << "site=" << site.id << " original=" << site.original_offset
                << " guest=" << site.prepared_guest_offset << " probe=" << site.probe_offset
                << " probe_bytes=" << site.probe_size << '\n';
    }
  }
  return std::cout ? 0 : 3;
}

} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
