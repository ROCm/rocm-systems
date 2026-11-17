// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <charconv>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
using namespace rocjitsu;

constexpr size_t kMaxImageBytes = 256 * 1024 * 1024;
constexpr std::string_view kUsage =
    "Usage: rj_code_mutate --input FILE --kernel ENTRY [--kind DropBarrier|LdsWrongAddress]\n"
    "       --list | --output NEW-FILE [--exact TEXT-OFFSET] [--ordinal N]\n"
    "       [--target GFX-NAME] [--address-vgpr N]\n"
    "       rj_code_mutate --capabilities | --help\n\n"
    "List sites for the selected kind and exact kernel entry in this input.\n"
    "--exact is the .text-relative byte offset from --list, decimal or 0x hex:\n"
    "the first member of a barrier, or the LDS instruction. Exact takes precedence\n"
    "over the zero-based --ordinal (default 0); invalid exact never falls back.\n"
    "Split pairs count as one barrier.\n"
    "--target only asserts the source identity; it cannot override the ELF.\n"
    "Input: extracted little-endian AMDHSA ET_DYN ELF64, at most 256 MiB.\n"
    "Complete builtin split pairs: RDNA4 workgroup; CDNA5 workgroup/cluster.\n"
    "Full s_barrier singletons: RDNA3/CDNA3/CDNA4. See --capabilities for linked\n"
    "concrete target/wave admission. Drop rejects partial/named/dynamic barriers\n"
    "and shared owners/helpers. Multiple .text sections are not supported.\n"
    "Trap-handler barrier IDs are rejected. Cluster barriers require a clustered\n"
    "dispatch that activates cluster semantics.\n"
    "LdsWrongAddress admits gfx1100/gfx1200/gfx1201 wave32/wave64 scalar and two-address\n"
    "loads/stores, gfx942/gfx950 wave64 scalar and read2/write2 forms, and\n"
    "gfx1250 wave32 scalar loads/stores and gfx1251 wave32 scalar/two-address forms.\n"
    "Scalar widths: b32/b64/b96/b128.\n"
    "Two-address widths: b32/b64, including stride64 forms, except on gfx1250.\n"
    "Selected subword and transpose forms are also supported.\n"
    "See --capabilities for exact forms. It requires a distinct\n"
    "allocated --address-vgpr for mutation.\n"
    "This argument is an encoded selector; allocation uses the address bank.\n"
    "LDS output lists every affected owner when a helper is shared.\n"
    "Discovery can succeed while replacement growth or branch range prevents emission.\n"
    "The caller supplies its runtime value.\n"
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

int reject(const FaultMutationError &error, std::string_view operation,
           std::optional<uint64_t> known_offset = std::nullopt) {
  std::cerr << "rj_code_mutate: " << operation << " failed: code=" << error_code_name(error.code);
  // Library offsets do not encode presence; callers supply a known CLI coordinate explicitly.
  if (known_offset.has_value())
    std::cerr << " offset=" << *known_offset;
  std::cerr << ' ' << error.message << '\n';
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
    for (const LdsCapability &capability : lds_wrong_address_capabilities()) {
      const IsaGpuTargetDescription *target =
          default_isa_target_registry().find_gpu_target(capability.target);
      std::cout << "kind=LdsWrongAddress target=" << target->code_object_id
                << " wave=" << capability.wave_size << " form=" << capability.form
                << " admission=proven-all-owners\n";
    }
    return finish_stdout();
  }
  const char *input = nullptr;
  const char *output = nullptr;
  std::string_view kernel, kind = "DropBarrier", target_assertion;
  std::optional<uint64_t> exact;
  std::optional<uint32_t> address_vgpr;
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
        option != "--target" && option != "--exact" && option != "--ordinal" &&
        option != "--address-vgpr")
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
      if (!parse_number(value, number) || (option == "--ordinal" && number > SIZE_MAX) ||
          (option == "--address-vgpr" && number > UINT32_MAX))
        return reject("invalid unsigned integer for " + std::string(option));
      if (option == "--exact")
        exact = number;
      else if (option == "--address-vgpr")
        address_vgpr = static_cast<uint32_t>(number);
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
  if (kind != "DropBarrier" && kind != "LdsWrongAddress")
    return reject("unsupported mutation kind: " + std::string(kind));
  if ((kind == "DropBarrier" && address_vgpr) || (list && address_vgpr))
    return reject("--address-vgpr applies only to LdsWrongAddress mutation");
  const util::FailureOr<std::vector<char>> bytes = detail::read_file_bytes(input, kMaxImageBytes);
  if (bytes.failed() || bytes.value().empty())
    return reject("cannot read bounded input code object (regular file, 1..256 MiB)");
  const FaultMutationContext context(
      std::span(reinterpret_cast<const uint8_t *>(bytes.value().data()), bytes.value().size()));
  FaultMutationError error;
  if (kind == "LdsWrongAddress") {
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(context, kernel, error);
    if (inventory.failed())
      return reject(error, "discovery");
    const IsaGpuTargetDescription *target =
        default_isa_target_registry().find_gpu_target(context.source().target_id());
    if (!target_assertion.empty() && target_assertion != target->code_object_id)
      return reject("target assertion does not match source ELF identity");
    if (list) {
      size_t index = 0;
      for (const LdsSite &site : inventory.value().sites()) {
        std::cout << "ordinal=" << index++ << " exact=" << site.offset
                  << " address_vgpr=" << site.address_vgpr << " address_bank=" << site.address_bank
                  << " target=" << target->code_object_id << " form=" << site.form
                  << " affected_owners=" << site.affected_owners.size();
        for (const FaultMutationOwner &owner : site.affected_owners)
          std::cout << " owner=" << owner.entry << ':' << owner.entry_offset << ':'
                    << owner.descriptor_file_offset;
        std::cout << '\n';
      }
      return finish_stdout();
    }
    LdsSelector selector = {.exact = std::nullopt, .ordinal = ordinal};
    if (exact) {
      const std::span<const LdsSite> sites = inventory.value().sites();
      const std::span<const LdsSite>::iterator found =
          std::ranges::find(sites, *exact, &LdsSite::offset);
      if (found == sites.end())
        return reject({.code = FaultMutationErrorCode::InvalidParameters,
                       .offset = *exact,
                       .message = "no LDS site at the requested offset"},
                      "selection", exact);
      selector.ordinal = static_cast<size_t>(found - sites.begin());
    }
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), selector, error);
    if (selection.failed())
      return reject(error, "selection");
    const FailureOr<LdsResult> mutant =
        apply_lds_wrong_address(context, selection.value(), address_vgpr, error);
    if (mutant.failed())
      return reject(error, "mutation");
    if (!write_new_file(output, mutant.value().image)) {
      std::cerr << "rj_code_mutate: cannot create/write/close new output file\n";
      return 3;
    }
    std::cout << "kind=LdsWrongAddress target=" << target->code_object_id
              << " exact=" << selection.value().site().offset << " address_vgpr=" << *address_vgpr
              << " logical_mutations=1 output_bytes=" << mutant.value().image.size()
              << " affected_owners=" << selection.value().site().affected_owners.size();
    for (const FaultMutationOwner &owner : selection.value().site().affected_owners)
      std::cout << " owner=" << owner.entry << ':' << owner.entry_offset << ':'
                << owner.descriptor_file_offset;
    std::cout << '\n';
    return finish_stdout();
  }
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
                    "selection", exact);
    selector.ordinal = static_cast<size_t>(found - barriers.begin());
  }
  const FailureOr<DropBarrierSelection> selection =
      select_drop_barrier(inventory.value(), selector, error);
  if (selection.failed())
    return reject(error, "selection");
  const FailureOr<DropBarrierResult> mutant = apply_drop_barrier(context, selection.value(), error);
  if (mutant.failed())
    return reject(error, "mutation");
  if (!write_new_file(output, mutant.value().image)) {
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
            << " output_bytes=" << mutant.value().image.size() << '\n';
  return finish_stdout();
}
} // namespace

int main(int argc, char **argv) {
  // Let EPIPE/EFBIG reach the checked stream and file paths, including cleanup.
#ifdef SIGPIPE
  if (std::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
    std::cerr << "rj_code_mutate: cannot ignore SIGPIPE\n";
    return 3;
  }
#endif
#ifdef SIGXFSZ
  if (std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR) {
    std::cerr << "rj_code_mutate: cannot ignore SIGXFSZ\n";
    return 3;
  }
#endif
  return run(argc, argv);
}
