// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/cli.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"
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
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace rocjitsu;

constexpr size_t kMaxImageBytes = 256 * 1024 * 1024;
namespace cli = rocjitsu::cli;

constexpr std::string_view kMutationHelp =
    "\n"
    "List sites for the selected kind and exact kernel entry in this input.\n"
    "--exact is the .text-relative byte offset from --list, decimal or 0x hex:\n"
    "the first member of a barrier, or the LDS instruction. For atomic sites, it selects\n"
    "the complete atomic instruction. Exact takes precedence\n"
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
    "AtomicWrongAddress lists all owned atomic sites before form admission.\n"
    "Mutation currently supports gfx1250/gfx1251 wave32 and gfx1200/gfx1201\n"
    "wave32/wave64 flat/global/buffer forms; see --capabilities. It requires a positive\n"
    "--address-delta, at most 8388607 bytes, divisible by four and preserving\n"
    "the operation's natural alignment (8 bytes for 64-bit, 4 for 32-bit and\n"
    "packed pairs; CAS uses the memory element width). The caller must supply\n"
    "an aligned original address and valid source and mutant addresses. Only the\n"
    "immediate offset changes; data, returning behavior, modifiers and waits remain.\n"
    "An unsupported selected form fails without renumbering atomic ordinals.\n"
    "CDNA5 and RDNA4 buffer atomics require a nonnegative original immediate offset.\n"
    "Output must be new; existing files, aliases and in-place mutation are refused.\n"
    "Host transformation only; no GPU execution or detector qualification.\n";

util::FailureOr<uint64_t> parse_number(std::string_view text, util::DiagnosticEmitter diagnostic) {
  if (!text.starts_with("0x")) {
    return cli::parse_uint(text, diagnostic);
  }
  text.remove_prefix(2);
  uint64_t number = 0;
  const std::from_chars_result parsed =
      std::from_chars(text.data(), text.data() + text.size(), number, 16);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    return diagnostic.emit() << "expected an unsigned decimal or 0x hexadecimal integer";
  }
  return number;
}

// Mutation requests reject repeated options and empty values rather than using
// the shared parser's last-value-wins policy, which could silently select a new site.
template <typename T> class MutationFlag final : public cli::Flag {
public:
  explicit MutationFlag(cli::FlagInfo info) : Flag(std::move(info)) {}
  const std::optional<T> &value() const { return value_; }
  bool allows_switch_mode() const override { return std::is_same_v<T, bool>; }
  std::string format_value() const override {
    if (!value_) {
      return "unset";
    }
    if constexpr (std::is_same_v<T, std::string>) {
      return *value_;
    } else if constexpr (std::is_same_v<T, bool>) {
      return *value_ ? "true" : "false";
    } else {
      return std::to_string(*value_);
    }
  }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    if (value_) {
      return diagnostic.emit() << "duplicate option";
    }
    if (text.empty() || text.starts_with("--")) {
      return diagnostic.emit() << "requires a nonempty value";
    }
    if constexpr (std::is_same_v<T, std::string>) {
      value_ = std::string(text);
    } else if constexpr (std::is_same_v<T, bool>) {
      if (text != "true") {
        return diagnostic.emit() << "expected a bare switch or '=true'";
      }
      value_ = true;
    } else {
      const util::FailureOr<uint64_t> number = parse_number(text, diagnostic);
      if (number.failed()) {
        return util::Result::failure();
      }
      value_ = number.value();
    }
    return util::Result::success();
  }
  std::optional<T> value_;
};

// Exclusive creation also protects aliases and destinations created after argument parsing.
// A failed write/close removes our incomplete output and never reports a mutant.
bool write_new_file(const char *path, std::span<const uint8_t> image) {
  std::FILE *file = std::fopen(path, "wbx");
  if (!file) {
    return false;
  }
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
  if (known_offset.has_value()) {
    std::cerr << " offset=" << *known_offset;
  }
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
  cli::Command command("rj_code_mutate");
  command.set_positional_usage("");
  auto input_flag =
      std::make_shared<MutationFlag<std::string>>(cli::FlagInfo{.name = "--input",
                                                                .aliases = {},
                                                                .description = "Input code object",
                                                                .value_placeholder = "FILE"});
  auto kernel_flag = std::make_shared<MutationFlag<std::string>>(
      cli::FlagInfo{.name = "--kernel",
                    .aliases = {},
                    .description = "Exact descriptor-derived entry",
                    .value_placeholder = "ENTRY"});
  auto output_flag = std::make_shared<MutationFlag<std::string>>(
      cli::FlagInfo{.name = "--output",
                    .aliases = {},
                    .description = "Create a new mutant file",
                    .value_placeholder = "NEW-FILE"});
  auto kind_flag = std::make_shared<MutationFlag<std::string>>(
      cli::FlagInfo{.name = "--kind",
                    .aliases = {},
                    .description = "Mutation kind (default: DropBarrier)",
                    .value_placeholder = "KIND"});
  auto target_flag = std::make_shared<MutationFlag<std::string>>(
      cli::FlagInfo{.name = "--target",
                    .aliases = {},
                    .description = "Assert the source target",
                    .value_placeholder = "GFX-NAME"});
  auto exact_flag = std::make_shared<MutationFlag<uint64_t>>(
      cli::FlagInfo{.name = "--exact",
                    .aliases = {},
                    .description = "Exact site .text byte offset",
                    .value_placeholder = "TEXT-OFFSET"});
  auto ordinal_flag = std::make_shared<MutationFlag<uint64_t>>(
      cli::FlagInfo{.name = "--ordinal",
                    .aliases = {},
                    .description = "Site ordinal (default: 0)",
                    .value_placeholder = "N"});
  auto address_vgpr_flag = std::make_shared<MutationFlag<uint64_t>>(
      cli::FlagInfo{.name = "--address-vgpr",
                    .aliases = {},
                    .description = "Encoded LDS address VGPR selector",
                    .value_placeholder = "N"});
  auto address_delta_flag = std::make_shared<MutationFlag<uint64_t>>(
      cli::FlagInfo{.name = "--address-delta",
                    .aliases = {},
                    .description = "Positive atomic byte delta preserving natural alignment",
                    .value_placeholder = "N"});
  auto list_flag = std::make_shared<MutationFlag<bool>>(
      cli::FlagInfo{.name = "--list",
                    .aliases = {},
                    .description = "List sites for the selected mutation kind",
                    .value_mode = cli::ValueMode::Switch});
  auto capabilities_flag = std::make_shared<MutationFlag<bool>>(
      cli::FlagInfo{.name = "--capabilities",
                    .aliases = {},
                    .description = "List supported target/wave forms and exit",
                    .value_mode = cli::ValueMode::Switch});
  auto help_flag =
      std::make_shared<MutationFlag<bool>>(cli::FlagInfo{.name = "--help",
                                                         .aliases = {},
                                                         .description = "Show help and exit",
                                                         .value_mode = cli::ValueMode::Switch});
  auto report = [](std::string_view message) { std::cerr << message << '\n'; };
  util::DiagnosticEmitter diagnostic(report);
  if (command
          .add_flags("Mutation",
                     {input_flag, kernel_flag, output_flag, kind_flag, target_flag, exact_flag,
                      ordinal_flag, address_vgpr_flag, address_delta_flag, list_flag},
                     diagnostic)
          .failed() ||
      command.add_flags("Information", {capabilities_flag, help_flag}, diagnostic).failed()) {
    return 2;
  }
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  const util::FailureOr<cli::ParsedArguments> parsed = cli::parse(command, arguments, diagnostic);
  if (parsed.failed()) {
    return 2;
  }
  if (!parsed.value().positional.empty() || parsed.value().saw_separator) {
    return reject("positional arguments are not supported");
  }
  if ((help_flag->value() || capabilities_flag->value()) && argc != 2) {
    return reject("--help and --capabilities must be used alone");
  }
  if (help_flag->value()) {
    std::cout << cli::format_help(command) << kMutationHelp;
    return finish_stdout();
  }
  if (capabilities_flag->value()) {
    for (const FaultMutationCapability &capability : fault_mutation_capabilities()) {
      const IsaTargetBinding *target = default_isa_target_registry().find_target(capability.target);
      std::cout << "kind=DropBarrier target=" << target->id << " wave=" << capability.wave_size
                << " form="
                << (capability.form == StaticBarrierForm::BuiltinSplitPair ? "split-pair"
                                                                           : "full-singleton")
                << " admission=" << capability.admission << '\n';
    }
    for (const LdsCapability &capability : lds_wrong_address_capabilities()) {
      const IsaTargetBinding *target = default_isa_target_registry().find_target(capability.target);
      std::cout << "kind=LdsWrongAddress target=" << target->id << " wave=" << capability.wave_size
                << " form=" << capability.form << " admission=proven-all-owners\n";
    }
    for (const AtomicCapability &capability : atomic_wrong_address_capabilities()) {
      const IsaTargetBinding *target = default_isa_target_registry().find_target(capability.target);
      std::cout << "kind=AtomicWrongAddress target=" << target->id
                << " wave=" << capability.wave_size << " form=" << capability.form
                << " admission=proven-all-owners\n";
    }
    return finish_stdout();
  }
  const std::string input = input_flag->value().value_or("");
  const std::string output = output_flag->value().value_or("");
  const std::string kernel = kernel_flag->value().value_or("");
  const std::string kind = kind_flag->value().value_or("DropBarrier");
  const std::string target_assertion = target_flag->value().value_or("");
  const std::optional<uint64_t> exact = exact_flag->value();
  const uint64_t requested_ordinal = ordinal_flag->value().value_or(0);
  if (requested_ordinal > SIZE_MAX) {
    return reject("invalid unsigned integer for --ordinal");
  }
  const size_t ordinal = static_cast<size_t>(requested_ordinal);
  std::optional<uint32_t> address_vgpr = std::nullopt;
  if (address_vgpr_flag->value()) {
    const uint64_t requested_address_vgpr = *address_vgpr_flag->value();
    if (requested_address_vgpr > UINT32_MAX) {
      return reject("invalid unsigned integer for --address-vgpr");
    }
    address_vgpr = static_cast<uint32_t>(requested_address_vgpr);
  }
  std::optional<uint32_t> address_delta = std::nullopt;
  if (address_delta_flag->value()) {
    const uint64_t requested_address_delta = *address_delta_flag->value();
    if (requested_address_delta > UINT32_MAX)
      return reject("invalid unsigned integer for --address-delta");
    address_delta = static_cast<uint32_t>(requested_address_delta);
  }
  const bool list = list_flag->value().value_or(false);
  if (input.empty() || kernel.empty() ||
      (list ? output_flag->value().has_value() || exact.has_value() ||
                  ordinal_flag->value().has_value()
            : output.empty())) {
    std::cerr << cli::format_help(command) << kMutationHelp;
    return reject("require input, exact kernel, and either --list or --output");
  }
  if (kind != "DropBarrier" && kind != "LdsWrongAddress" && kind != "AtomicWrongAddress") {
    return reject("unsupported mutation kind: " + kind);
  }
  if ((kind != "LdsWrongAddress" && address_vgpr) || (list && address_vgpr)) {
    return reject("--address-vgpr applies only to LdsWrongAddress mutation");
  }
  if ((kind != "AtomicWrongAddress" && address_delta) || (list && address_delta)) {
    return reject("--address-delta applies only to AtomicWrongAddress mutation");
  }
  const util::FailureOr<std::vector<char>> bytes =
      detail::read_file_bytes(input.c_str(), kMaxImageBytes);
  if (bytes.failed() || bytes.value().empty()) {
    return reject("cannot read bounded input code object (regular file, 1..256 MiB)");
  }
  const FaultMutationContext context(
      std::span(reinterpret_cast<const uint8_t *>(bytes.value().data()), bytes.value().size()));
  FaultMutationError error;
  if (kind == "AtomicWrongAddress") {
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(context, kernel, error);
    if (inventory.failed())
      return reject(error, "discovery");
    const IsaTargetBinding *target =
        default_isa_target_registry().find_target(context.source().target_id());
    if (!target_assertion.empty() && target_assertion != target->id)
      return reject("target assertion does not match source ELF identity");
    if (list) {
      size_t index = 0;
      for (const AtomicSite &site : inventory.value().sites()) {
        std::cout << "ordinal=" << index++ << " exact=" << site.offset
                  << " size=" << site.instruction_size << " byte_offset=";
        if (site.byte_offset)
          std::cout << *site.byte_offset;
        else
          std::cout << "unknown";
        std::cout << " address_supported=" << site.address_supported << " target=" << target->id
                  << " form=" << site.form << " affected_owners=" << site.affected_owners.size();
        for (const FaultMutationOwner &owner : site.affected_owners)
          std::cout << " owner=" << owner.entry << ':' << owner.entry_offset << ':'
                    << owner.descriptor_file_offset;
        std::cout << '\n';
      }
      return finish_stdout();
    }
    AtomicSelector selector = {.exact = std::nullopt, .ordinal = ordinal};
    if (exact) {
      const std::span<const AtomicSite> sites = inventory.value().sites();
      const std::span<const AtomicSite>::iterator found =
          std::ranges::find(sites, *exact, &AtomicSite::offset);
      if (found == sites.end())
        return reject({.code = FaultMutationErrorCode::InvalidParameters,
                       .offset = *exact,
                       .message = "no atomic site at the requested offset"},
                      "selection", exact);
      selector.ordinal = static_cast<size_t>(found - sites.begin());
    }
    const FailureOr<AtomicSelection> selection =
        select_atomic_wrong_address(inventory.value(), selector, error);
    if (selection.failed())
      return reject(error, "selection");
    const FailureOr<AtomicResult> mutant =
        apply_atomic_wrong_address(context, selection.value(), address_delta, error);
    if (mutant.failed())
      return reject(error, "mutation");
    if (!write_new_file(output.c_str(), mutant.value().image)) {
      std::cerr << "rj_code_mutate: cannot create/write/close new output file\n";
      return 3;
    }
    std::cout << "kind=AtomicWrongAddress target=" << target->id
              << " exact=" << selection.value().site().offset << " address_delta=" << *address_delta
              << " logical_mutations=" << mutant.value().logical_mutation_count
              << " output_bytes=" << mutant.value().image.size()
              << " affected_owners=" << selection.value().site().affected_owners.size();
    for (const FaultMutationOwner &owner : selection.value().site().affected_owners)
      std::cout << " owner=" << owner.entry << ':' << owner.entry_offset << ':'
                << owner.descriptor_file_offset;
    std::cout << '\n';
    return finish_stdout();
  }
  if (kind == "LdsWrongAddress") {
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(context, kernel, error);
    if (inventory.failed()) {
      return reject(error, "discovery");
    }
    const IsaTargetBinding *target =
        default_isa_target_registry().find_target(context.source().target_id());
    if (!target_assertion.empty() && target_assertion != target->id) {
      return reject("target assertion does not match source ELF identity");
    }
    if (list) {
      size_t index = 0;
      for (const LdsSite &site : inventory.value().sites()) {
        std::cout << "ordinal=" << index++ << " exact=" << site.offset
                  << " address_vgpr=" << site.address_vgpr << " address_bank=" << site.address_bank
                  << " target=" << target->id << " form=" << site.form
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
      if (found == sites.end()) {
        return reject({.code = FaultMutationErrorCode::InvalidParameters,
                       .offset = *exact,
                       .message = "no LDS site at the requested offset"},
                      "selection", exact);
      }
      selector.ordinal = static_cast<size_t>(found - sites.begin());
    }
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), selector, error);
    if (selection.failed()) {
      return reject(error, "selection");
    }
    const FailureOr<LdsResult> mutant =
        apply_lds_wrong_address(context, selection.value(), address_vgpr, error);
    if (mutant.failed()) {
      return reject(error, "mutation");
    }
    if (!write_new_file(output.c_str(), mutant.value().image)) {
      std::cerr << "rj_code_mutate: cannot create/write/close new output file\n";
      return 3;
    }
    std::cout << "kind=LdsWrongAddress target=" << target->id
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
  if (inventory.failed()) {
    return reject(error, "discovery");
  }
  const IsaTargetBinding *target =
      default_isa_target_registry().find_target(context.source().target_id());
  if (!target_assertion.empty() && target_assertion != target->id) {
    return reject("target assertion does not match source ELF identity");
  }
  if (list) {
    size_t logical_ordinal = 0;
    for (const StaticBarrier &barrier : inventory.value().barriers()) {
      std::cout << "ordinal=" << logical_ordinal++ << " exact=" << barrier.members.front().offset
                << " target=" << target->id << " section=" << inventory.value().section().index
                << " form="
                << (barrier.form == StaticBarrierForm::BuiltinSplitPair ? "split-pair"
                                                                        : "full-singleton")
                << " id=" << barrier.barrier_id << " scope="
                << (barrier.scope == StaticBarrierScope::Workgroup ? "workgroup" : "cluster")
                << " members=";
      for (size_t index = 0; index < barrier.members.size(); ++index) {
        if (index) {
          std::cout << ',';
        }
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
    if (found == barriers.end()) {
      return reject({.code = FaultMutationErrorCode::InvalidParameters,
                     .offset = *exact,
                     .message = "no complete barrier starts at .text offset " +
                                std::to_string(*exact) + " for this entry"},
                    "selection", exact);
    }
    selector.ordinal = static_cast<size_t>(found - barriers.begin());
  }
  const FailureOr<DropBarrierSelection> selection =
      select_drop_barrier(inventory.value(), selector, error);
  if (selection.failed()) {
    return reject(error, "selection");
  }
  const FailureOr<DropBarrierResult> mutant = apply_drop_barrier(context, selection.value(), error);
  if (mutant.failed()) {
    return reject(error, "mutation");
  }
  if (!write_new_file(output.c_str(), mutant.value().image)) {
    std::cerr << "rj_code_mutate: cannot create/write/close new output file\n";
    return 3;
  }
  for (const std::string &assumption : mutant.value().assumptions) {
    std::cerr << "rj_code_mutate: assumption: " << assumption << '\n';
  }
  std::cerr.flush();
  if (!std::cerr) {
    return 3;
  }
  std::cout << "kind=DropBarrier target=" << target->id
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
