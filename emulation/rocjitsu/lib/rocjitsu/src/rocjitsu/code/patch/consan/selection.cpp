// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/prepare.h"

#include "rocjitsu/code/consan/image/artifact.h"
#include "util/diagnostic.h"
#include "util/log.h"
#include "util/utf8.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace rocjitsu::consan {
namespace {

bool unsupported_entry(InputErrorCode code) {
  switch (code) {
  case InputErrorCode::UnsupportedEntry:
  case InputErrorCode::UnsupportedAccess:
  case InputErrorCode::UnsupportedSynchronization:
  case InputErrorCode::UnsupportedInstruction:
  case InputErrorCode::UnsupportedFlow:
    return true;
  default:
    return false;
  }
}

std::string_view reason_name(InputErrorCode code) {
  switch (code) {
  case InputErrorCode::None:
    return "none";
  case InputErrorCode::InvalidImage:
    return "invalid_image";
  case InputErrorCode::UnsupportedTarget:
    return "unsupported_target";
  case InputErrorCode::UnsupportedEntry:
    return "unsupported_entry";
  case InputErrorCode::UnsupportedAccess:
    return "unsupported_access";
  case InputErrorCode::UnsupportedSynchronization:
    return "unsupported_synchronization";
  case InputErrorCode::UnsupportedInstruction:
    return "unsupported_instruction";
  case InputErrorCode::UnsupportedFlow:
    return "unsupported_flow";
  case InputErrorCode::UnsupportedControlTransferRange:
    return "unsupported_control_transfer_range";
  case InputErrorCode::DecodeFailure:
    return "decode_failure";
  case InputErrorCode::ResourceLimit:
    return "resource_limit";
  }
  return "unknown";
}

void json_string(std::ostream &out, std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  out << '"';
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\')
      out << '\\' << ch;
    else if (ch < 32)
      out << "\\u00" << digits[ch >> 4] << digits[ch & 15];
    else
      out << ch;
  }
  out << '"';
}

std::string_view stage_name(PreparationStage stage) {
  switch (stage) {
  case PreparationStage::Applicability:
    return "applicability";
  case PreparationStage::Qualification:
    return "qualification";
  case PreparationStage::Lowering:
    return "lowering";
  }
  return "qualification";
}

void json_digest(std::ostream &out, const Digest &digest) {
  constexpr char digits[] = "0123456789abcdef";
  out << '"';
  for (const uint8_t byte : digest)
    out << digits[byte >> 4] << digits[byte & 15];
  out << '"';
}

} // namespace

FailureOr<PreparationResult> prepare_image_best_effort(std::span<const uint8_t> original,
                                                       const TargetProfile &target,
                                                       std::span<const std::string_view> entries,
                                                       InputError &error, ScopeKind scope) {
  error = {};
  const InputLimits limits;
  if (entries.size() > kImageEntryLimit || original.size() > limits.image_bytes) {
    error = {.code = InputErrorCode::ResourceLimit,
             .text_offset = {},
             .message = "selected input copies exceed bounded image analysis budget"};
    return Result::failure();
  }
  const TargetProfile *canonical = target_profile(target.name);
  if (canonical == nullptr || *canonical != target) {
    error = {.code = InputErrorCode::UnsupportedTarget,
             .text_offset = {},
             .message = "unsupported or inconsistent native LDS target profile"};
    return Result::failure();
  }
  if (scope != ScopeKind::NativeLdsNoSynchronization && scope != ScopeKind::ExecutionTrace) {
    error = {.code = InputErrorCode::UnsupportedEntry,
             .text_offset = {},
             .message = "unknown preparation profile"};
    return Result::failure();
  }
  std::unordered_set<std::string_view> names;
  for (const std::string_view entry : entries) {
    if (!util::is_valid_utf8(entry) || entry.find('\0') != std::string_view::npos) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "entry name is not valid UTF-8 for JSON output"};
      return Result::failure();
    }
    if (!names.insert(entry).second) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "duplicate selected entry name"};
      return Result::failure();
    }
  }
  // Native inspection establishes a complete valid original before any local
  // qualification failure can be treated as an unsupported entry.
  util::StringDiagnostic diagnostic;
  const FailureOr<NativeImageInventory> inventory =
      inspect_native_image(original, target.name, diagnostic.emitter());
  if (inventory.failed()) {
    error = {
        .code = InputErrorCode::InvalidImage, .text_offset = {}, .message = diagnostic.message()};
    return Result::failure();
  }
  std::vector<std::string_view> automatic_entries;
  if (entries.empty()) {
    automatic_entries.reserve(inventory.value().entries.size());
    for (const ImageEntryFacts &entry : inventory.value().entries)
      automatic_entries.push_back(entry.binding.name);
    entries = automatic_entries;
  }
  if (original.size() > limits.image_bytes / entries.size()) {
    error = {.code = InputErrorCode::ResourceLimit,
             .text_offset = {},
             .message = "selected input copies exceed bounded image analysis budget"};
    return Result::failure();
  }
  for (const std::string_view entry : entries) {
    if (std::ranges::find(inventory.value().entries, entry, [](const ImageEntryFacts &facts) {
          return facts.binding.name;
        }) == inventory.value().entries.end()) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "requested kernel entry '" + std::string(entry) + "' was not found"};
      return Result::failure();
    }
  }
  PreparationResult result = {.original = inventory.value().identity,
                              .target = target.id,
                              .requested_scope = scope,
                              .inventory_entries =
                                  static_cast<uint32_t>(inventory.value().entries.size()),
                              .entries = {},
                              .artifact = std::nullopt};
  std::vector<std::string_view> selected;
  for (const std::string_view name : entries) {
    const std::vector<ImageEntryFacts>::const_iterator binding =
        std::ranges::find(inventory.value().entries, name,
                          [](const ImageEntryFacts &facts) { return facts.binding.name; });
    EntryPreparation entry = {
        .entry = std::string(name), .key = binding->key, .instrumented = false, .reason = {}};
    if (scope != ScopeKind::ExecutionTrace) {
      const FailureOr<std::optional<NoProbesNeededProof>> applicability =
          inspect_no_lds_memory(original, target, binding->key, error);
      if (applicability.failed())
        return Result::failure();
      if (applicability.value()) {
        entry.no_probes_needed = applicability.value();
        entry.stage = PreparationStage::Applicability;
        entry.reason.message = "closed reachable code has no LDS memory accesses";
        util::Logger::consan("no LDS probes needed for entry ", name);
        result.entries.push_back(std::move(entry));
        continue;
      }
    }
    InputError reason;
    const FailureOr<InputProgram> program =
        build_image_entry_program(original, target, name, reason, {}, scope);
    if (program.failed()) {
      if (!unsupported_entry(reason.code)) {
        error = std::move(reason);
        return Result::failure();
      }
      entry.reason = std::move(reason);
      util::Logger::consan("retaining native entry ", name, ": ", entry.reason.message);
    } else {
      entry.instrumented = true;
      entry.stage = PreparationStage::Lowering;
      selected.push_back(name);
      util::Logger::consan("qualified entry ", name);
    }
    result.entries.push_back(std::move(entry));
  }
  if (!selected.empty()) {
    FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, target, selected, error, scope);
    if (artifact.succeeded()) {
      result.artifact = std::move(artifact.value());
    } else {
      if (error.code != InputErrorCode::UnsupportedControlTransferRange)
        return Result::failure();
      std::vector<std::string_view> accepted;
      for (EntryPreparation &entry : result.entries) {
        if (!entry.instrumented)
          continue;
        accepted.push_back(entry.entry);
        InputError candidate_error;
        FailureOr<PreparedImageArtifact> candidate =
            prepare_image(original, target, accepted, candidate_error, scope);
        if (candidate.succeeded()) {
          result.artifact = std::move(candidate.value());
          continue;
        }
        if (candidate_error.code != InputErrorCode::UnsupportedControlTransferRange) {
          error = std::move(candidate_error);
          return Result::failure();
        }
        accepted.pop_back();
        entry.instrumented = false;
        entry.reason = std::move(candidate_error);
        util::Logger::consan("retaining native entry ", entry.entry, ": ", entry.reason.message);
      }
      error = {};
    }
  }
  if (result.artifact) {
    for (const EntryPreparation &entry : result.entries) {
      if (!entry.no_probes_needed)
        continue;
      const std::vector<ImageEntryFacts>::iterator native =
          std::ranges::find(result.artifact->facts.entries, entry.key, &ImageEntryFacts::key);
      if (native == result.artifact->facts.entries.end() ||
          native->disposition != EntryDisposition::Native) {
        error = {.code = InputErrorCode::InvalidImage,
                 .text_offset = {},
                 .message = "no-probes-needed certificate has no owning native image entry"};
        return Result::failure();
      }
      native->no_probes_needed = entry.no_probes_needed;
    }
    util::StringDiagnostic diagnostic;
    if (verify_artifact_images(result.artifact->facts, original, result.artifact->image,
                               diagnostic.emitter())
            .failed()) {
      error = {.code = InputErrorCode::InvalidImage,
               .text_offset = {},
               .message = "invalid prepared no-probes-needed evidence: " + diagnostic.message()};
      return Result::failure();
    }
  }
  return result;
}

std::string render_preparation_report(const PreparationResult &result) {
  if (result.inventory_entries > kImageEntryLimit ||
      result.entries.size() > result.inventory_entries)
    return {};
  for (const EntryPreparation &entry : result.entries) {
    if (entry.instrumented && entry.no_probes_needed)
      return {};
    if (entry.no_probes_needed &&
        (result.requested_scope == ScopeKind::ExecutionTrace ||
         entry.no_probes_needed->version != 1 ||
         entry.no_probes_needed->scope != ScopeKind::NativeLdsNoSynchronization ||
         entry.no_probes_needed->instruction_count == 0 ||
         entry.no_probes_needed->code_bytes % 4 != 0 ||
         entry.no_probes_needed->code_bytes / 4 < entry.no_probes_needed->instruction_count ||
         std::ranges::all_of(entry.no_probes_needed->code_sha256,
                             [](uint8_t byte) { return byte == 0; }) ||
         entry.stage != PreparationStage::Applicability ||
         entry.reason.code != InputErrorCode::None || entry.reason.text_offset))
      return {};
  }
  const size_t instrumented = static_cast<size_t>(std::ranges::count_if(
      result.entries, [](const EntryPreparation &entry) { return entry.instrumented; }));
  const size_t no_probes_needed =
      static_cast<size_t>(std::ranges::count_if(result.entries, [](const EntryPreparation &entry) {
        return entry.no_probes_needed.has_value();
      }));
  const size_t blocked = result.entries.size() - instrumented - no_probes_needed;
  std::ostringstream out;
  out << "{\"schema_version\":2,\"outcome\":\""
      << (result.artifact ? "instrumented" : "native_only") << "\",\"original_sha256\":\"";
  constexpr char digits[] = "0123456789abcdef";
  for (const uint8_t byte : result.original.sha256)
    out << digits[byte >> 4] << digits[byte & 15];
  out << "\",\"selection_profile\":";
  if (result.artifact)
    json_digest(out, result.artifact->facts.selection_profile);
  else
    out << "null";
  out << ",\"original_bytes\":" << result.original.bytes << ",\"target\":" << result.target
      << ",\"requested_scope\":\""
      << (result.requested_scope == ScopeKind::ExecutionTrace ? "execution_trace" : "lds")
      << "\",\"inventory_entries\":" << result.inventory_entries
      << ",\"not_requested\":" << result.inventory_entries - result.entries.size()
      << ",\"requested\":" << result.entries.size() << ",\"instrumented\":" << instrumented
      << ",\"no_probes_needed\":" << no_probes_needed << ",\"blocked\":" << blocked
      << ",\"skipped\":" << blocked << ",\"requested_coverage\":\""
      << (blocked == 0 ? "complete"
                       : (instrumented + no_probes_needed == 0 ? "not_performed" : "partial"))
      << "\",\"entries\":[";
  bool first = true;
  for (const EntryPreparation &entry : result.entries) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"entry\":";
    json_string(out, entry.entry);
    out << ",\"entry_key\":" << entry.key << ",\"disposition\":\""
        << (entry.instrumented ? "instrumented"
                               : (entry.no_probes_needed ? "no_probes_needed" : "blocked"))
        << "\",\"stage\":\"" << stage_name(entry.stage) << "\",\"reason_code\":";
    json_string(out, reason_name(entry.reason.code));
    out << ",\"text_offset\":";
    if (entry.reason.text_offset)
      out << *entry.reason.text_offset;
    else
      out << "null";
    out << ",\"reason\":";
    json_string(out, entry.reason.message);
    out << ",\"no_probes_needed\":";
    if (entry.no_probes_needed) {
      const NoProbesNeededProof &proof = *entry.no_probes_needed;
      out << "{\"version\":" << proof.version
          << ",\"scope\":\"lds\",\"instruction_count\":" << proof.instruction_count
          << ",\"code_bytes\":" << proof.code_bytes << ",\"code_sha256\":";
      json_digest(out, proof.code_sha256);
      out << '}';
    } else {
      out << "null";
    }
    out << '}';
  }
  out << "]}\n";
  return out.str();
}

} // namespace rocjitsu::consan
