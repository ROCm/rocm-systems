// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace rocjitsu;

constexpr size_t kMaxImageBytes = 256 * 1024 * 1024;
constexpr std::string_view kUsage =
    "Usage: rj_code_mutate --input FILE --kernel ENTRY [--kind DropBarrier]\n"
    "       --list | --output NEW-FILE [--exact TEXT-OFFSET] [--ordinal N]\n"
    "       [--target GFX-NAME]\n"
    "       rj_code_mutate --capabilities | --help\n\n"
    "List complete logical barriers for the exact kernel entry in this input.\n"
    "Mutate one barrier with DropBarrier (the only available kind). --exact is\n"
    "the first member's .text-relative byte offset from --list, decimal or 0x hex.\n"
    "Exact takes precedence over the zero-based logical --ordinal (default 0);\n"
    "invalid exact selection never falls back. Split pairs count as one barrier.\n"
    "--target only asserts the source identity; it cannot override the ELF.\n"
    "Input: extracted little-endian AMDHSA ET_DYN ELF64, at most 256 MiB.\n"
    "Complete builtin split pairs: RDNA4 workgroup; CDNA5 workgroup/cluster.\n"
    "Full s_barrier singletons: RDNA3/CDNA3/CDNA4. See --capabilities for linked\n"
    "concrete target/wave admission. Partial/named/dynamic barriers, shared\n"
    "owners/helpers and multiple .text sections are not supported.\n"
    "Trap-handler barrier IDs are rejected. Cluster barriers require a clustered\n"
    "dispatch that activates cluster semantics.\n"
    "Output must be new; existing files, aliases and in-place mutation are refused.\n"
    "Host transformation only; no GPU execution or detector qualification.\n";

bool parse_number(std::string_view text, uint64_t &number) {
  int base = 10;
  if (text.starts_with("0x")) {
    text.remove_prefix(2);
    base = 16;
  }
  const std::from_chars_result parsed =
      std::from_chars(text.data(), text.data() + text.size(), number, base);
  return !text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

// Exclusive creation also protects aliases and destinations created after argument parsing.
// A failed write/close removes our incomplete output and never reports a mutant.
bool write_new_file(const char *path, std::span<const uint8_t> image) {
  std::FILE *file = std::fopen(path, "wbx");
  if (!file)
    return false;
  const bool written = std::fwrite(image.data(), 1, image.size(), file) == image.size();
  const bool closed = std::fclose(file) == 0;
  if (!written || !closed) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return false;
  }
  return true;
}

int reject(std::string_view message) {
  std::cerr << "rj_code_mutate: " << message << '\n';
  return 2;
}

std::string_view error_code_name(FaultMutationErrorCode code) {
  switch (code) {
  case FaultMutationErrorCode::None:
    return "None";
  case FaultMutationErrorCode::BadIdentity:
    return "BadIdentity";
  case FaultMutationErrorCode::UnsupportedTarget:
    return "UnsupportedTarget";
  case FaultMutationErrorCode::UnsupportedForm:
    return "UnsupportedForm";
  case FaultMutationErrorCode::AmbiguousSelection:
    return "AmbiguousSelection";
  case FaultMutationErrorCode::IncompleteSelection:
    return "IncompleteSelection";
  case FaultMutationErrorCode::InvalidParameters:
    return "InvalidParameters";
  case FaultMutationErrorCode::ValidationFailure:
    return "ValidationFailure";
  case FaultMutationErrorCode::NoCandidates:
    return "NoCandidates";
  case FaultMutationErrorCode::OrdinalOutOfRange:
    return "OrdinalOutOfRange";
  }
  return "Unknown";
}

int reject(const FaultMutationError &error, std::string_view operation) {
  std::cerr << "rj_code_mutate: " << operation << " failed: code=" << error_code_name(error.code)
            << " offset=" << error.offset << ' ' << error.message << '\n';
  return 2;
}

int finish_stdout() {
  std::cout.flush();
  if (!std::cout) {
    std::cerr << "rj_code_mutate: cannot write/flush stdout\n";
    return 3;
  }
  return 0;
}

int run(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << kUsage;
    return finish_stdout();
  }
  if (argc == 2 && std::string_view(argv[1]) == "--capabilities") {
    for (const FaultMutationCapability &capability : fault_mutation_capabilities()) {
      const IsaGpuTargetDescription *target =
          default_isa_target_registry().find_gpu_target(capability.target);
      std::cout << "kind=DropBarrier target=" << target->code_object_id
                << " wave=" << capability.wave_size << " form="
                << (capability.form == StaticBarrierForm::BuiltinSplitPair ? "split-pair"
                                                                           : "full-singleton")
                << " admission=" << capability.admission << '\n';
    }
    return finish_stdout();
  }
  const char *input = nullptr;
  const char *output = nullptr;
  std::string_view kernel, kind = "DropBarrier", target_assertion;
  std::optional<uint64_t> exact;
  size_t ordinal = 0;
  bool list = false;
  std::vector<std::string_view> seen;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (std::ranges::find(seen, option) != seen.end())
      return reject("duplicate option: " + std::string(option));
    seen.push_back(option);
    if (option == "--list") {
      list = true;
      continue;
    }
    if (option != "--input" && option != "--output" && option != "--kernel" && option != "--kind" &&
        option != "--target" && option != "--exact" && option != "--ordinal")
      return reject("unknown option: " + std::string(option));
    if (++index == argc || std::string_view(argv[index]).empty() ||
        std::string_view(argv[index]).starts_with("--"))
      return reject("missing value for " + std::string(option));
    const std::string_view value(argv[index]);
    if (option == "--input")
      input = argv[index];
    else if (option == "--output")
      output = argv[index];
    else if (option == "--kernel")
      kernel = value;
    else if (option == "--kind")
      kind = value;
    else if (option == "--target")
      target_assertion = value;
    else {
      uint64_t number = 0;
      if (!parse_number(value, number) || (option == "--ordinal" && number > SIZE_MAX))
        return reject("invalid unsigned integer for " + std::string(option));
      if (option == "--exact")
        exact = number;
      else
        ordinal = static_cast<size_t>(number);
    }
  }
  if (!input || kernel.empty() ||
      (list ? output != nullptr || exact.has_value() ||
                  std::ranges::find(seen, "--ordinal") != seen.end()
            : output == nullptr)) {
    std::cerr << kUsage;
    return reject("require input, exact kernel, and either --list or --output");
  }
  if (kind != "DropBarrier")
    return reject("unsupported mutation kind: " + std::string(kind));
  const util::FailureOr<std::vector<char>> bytes = detail::read_file_bytes(input, kMaxImageBytes);
  if (bytes.failed() || bytes.value().empty())
    return reject("cannot read bounded input code object (regular file, 1..256 MiB)");
  const FaultMutationContext context(
      std::span(reinterpret_cast<const uint8_t *>(bytes.value().data()), bytes.value().size()));
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory = discover_drop_barriers(context, kernel, error);
  if (inventory.failed())
    return reject(error, "discovery");
  const IsaGpuTargetDescription *target =
      default_isa_target_registry().find_gpu_target(context.source().target_id());
  if (!target_assertion.empty() && target_assertion != target->code_object_id)
    return reject("target assertion does not match source ELF identity");
  if (list) {
    size_t logical_ordinal = 0;
    for (const StaticBarrier &barrier : inventory.value().barriers()) {
      std::cout << "ordinal=" << logical_ordinal++ << " exact=" << barrier.members.front().offset
                << " target=" << target->code_object_id
                << " section=" << inventory.value().section().index << " form="
                << (barrier.form == StaticBarrierForm::BuiltinSplitPair ? "split-pair"
                                                                        : "full-singleton")
                << " id=" << barrier.barrier_id << " scope="
                << (barrier.scope == StaticBarrierScope::Workgroup ? "workgroup" : "cluster")
                << " members=";
      for (size_t index = 0; index < barrier.members.size(); ++index) {
        if (index)
          std::cout << ',';
        std::cout << barrier.members[index].offset << ':' << barrier.members[index].bytes.size();
      }
      std::cout << '\n';
    }
    return finish_stdout();
  }
  DropBarrierSelector selector = {.exact = std::nullopt, .ordinal = ordinal};
  if (exact) {
    // Resolve this input's coordinate once. A miss must not select the ordinal
    // fallback or masquerade as a library handle from another source.
    const std::span<const StaticBarrier> barriers = inventory.value().barriers();
    const std::span<const StaticBarrier>::iterator found =
        std::ranges::find_if(barriers, [&](const StaticBarrier &barrier) {
          return barrier.members.front().offset == *exact;
        });
    if (found == barriers.end())
      return reject({.code = FaultMutationErrorCode::InvalidParameters,
                     .offset = *exact,
                     .message = "no complete barrier starts at .text offset " +
                                std::to_string(*exact) + " for this entry"},
                    "selection");
    selector.ordinal = static_cast<size_t>(found - barriers.begin());
  }
  const FailureOr<DropBarrierSelection> selection =
      select_drop_barrier(inventory.value(), selector, error);
  if (selection.failed())
    return reject(error, "selection");
  const FailureOr<DropBarrierResult> mutant = apply_drop_barrier(context, selection.value(), error);
  if (mutant.failed())
    return reject(error, "mutation");
  if (!write_new_file(output, mutant.value().edit_result.image)) {
    std::cerr << "rj_code_mutate: cannot create/write/close new output file\n";
    return 3;
  }
  for (const std::string &assumption : mutant.value().assumptions)
    std::cerr << "rj_code_mutate: assumption: " << assumption << '\n';
  std::cerr.flush();
  if (!std::cerr)
    return 3;
  std::cout << "kind=DropBarrier target=" << target->code_object_id
            << " exact=" << selection.value().barrier().members.front().offset
            << " logical_mutations=" << mutant.value().logical_mutation_count
            << " members=" << selection.value().barrier().members.size()
            << " output_bytes=" << mutant.value().edit_result.image.size() << '\n';
  return finish_stdout();
}
} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
