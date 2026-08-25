// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/data_hazard/report.h"

#include "hash_utils.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace rocjitsu::plugins::data_hazard {

namespace {

/// Escapes a string for embedding in a JSON document. Bytes above ASCII are
/// passed through, which is correct for the UTF-8 the formatters produce.
void append_json_string(std::string &out, const std::string &value) {
  out += '"';
  for (const char c : value) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[7];
        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
        out += buf;
      } else {
        out += c;
      }
      break;
    }
  }
  out += '"';
}

void append_indent(std::string &out, int depth) { out.append(static_cast<size_t>(depth) * 2, ' '); }

void append_uint_field(std::string &out, int depth, const char *key, uint64_t value, bool comma) {
  append_indent(out, depth);
  append_json_string(out, key);
  out += ": ";
  out += std::to_string(value);
  out += comma ? ",\n" : "\n";
}

void append_bool_field(std::string &out, int depth, const char *key, bool value, bool comma) {
  append_indent(out, depth);
  append_json_string(out, key);
  out += ": ";
  out += value ? "true" : "false";
  out += comma ? ",\n" : "\n";
}

void append_string_field(std::string &out, int depth, const char *key, const std::string &value,
                         bool comma) {
  append_indent(out, depth);
  append_json_string(out, key);
  out += ": ";
  append_json_string(out, value);
  out += comma ? ",\n" : "\n";
}

} // namespace

std::string format_raw_isa_hex(const std::array<uint32_t, 4> &raw_isa) {
  char buf[36];
  std::snprintf(buf, sizeof(buf), "%08x %08x %08x %08x", raw_isa[0], raw_isa[1], raw_isa[2],
                raw_isa[3]);
  return buf;
}

std::string dedup_message_key(const std::string &message) {
  std::string normalized = message;

  // Messages that embed a per-lane memory address collapse to one entry so a
  // hazard at a given PC is not reported once per address slot.
  struct AddrPattern {
    const char *prefix;
    const char *addr_end;
  };
  static constexpr AddrPattern kAddrPatterns[] = {
      {"RAW hazard: LDS address 0x", " (size "},
      {"RAW hazard: global address 0x", " read before async store completes"},
  };
  for (const auto &p : kAddrPatterns) {
    if (normalized.rfind(p.prefix, 0) != 0)
      continue;
    const size_t addr_begin = std::char_traits<char>::length(p.prefix);
    const size_t addr_end = normalized.find(p.addr_end, addr_begin);
    if (addr_end != std::string::npos && addr_end > addr_begin)
      normalized.replace(addr_begin, addr_end - addr_begin, "<addr>");
    break;
  }

  // Cross-wave race messages name concrete waves and addresses; those are
  // runtime instances of one shader-level race.
  static constexpr char kLdsRacePrefix[] = "LDS data race: Wave ";
  if (normalized.rfind(kLdsRacePrefix, 0) == 0) {
    auto replace_between = [&normalized](size_t begin, size_t end, const char *replacement) {
      if (begin != std::string::npos && end != std::string::npos && end > begin)
        normalized.replace(begin, end - begin, replacement);
    };

    const size_t first_wave_begin = sizeof(kLdsRacePrefix) - 1;
    const size_t first_wave_end = normalized.find(" writes LDS address 0x", first_wave_begin);
    replace_between(first_wave_begin, first_wave_end, "<wave>");

    const size_t first_addr_marker = normalized.find(" writes LDS address 0x");
    if (first_addr_marker != std::string::npos) {
      const size_t first_addr_begin =
          first_addr_marker + std::char_traits<char>::length(" writes LDS address 0x");
      const size_t first_addr_end = normalized.find(" (size ", first_addr_begin);
      replace_between(first_addr_begin, first_addr_end, "<addr>");
    }

    const size_t second_wave_marker = normalized.find(" and Wave ");
    if (second_wave_marker != std::string::npos) {
      const size_t second_wave_begin =
          second_wave_marker + std::char_traits<char>::length(" and Wave ");
      const size_t second_wave_end = normalized.find(' ', second_wave_begin);
      replace_between(second_wave_begin, second_wave_end, "<wave>");
    }

    const size_t second_addr_marker =
        normalized.find(" LDS address 0x", normalized.find(" and Wave "));
    if (second_addr_marker != std::string::npos) {
      const size_t second_addr_begin =
          second_addr_marker + std::char_traits<char>::length(" LDS address 0x");
      const size_t second_addr_end = normalized.find(" (size ", second_addr_begin);
      replace_between(second_addr_begin, second_addr_end, "<addr>");
    }
  }

  // Every message ends with "(...at pc 0xHEX, instruction N)". N is a
  // per-execution ordinal, so the same shader instruction executed by several
  // waves would otherwise produce distinct keys.
  static constexpr char kInstSuffix[] = ", instruction ";
  const size_t inst_pos = normalized.rfind(kInstSuffix);
  if (inst_pos != std::string::npos && !normalized.empty() && normalized.back() == ')') {
    const size_t digits_start = inst_pos + sizeof(kInstSuffix) - 1;
    const size_t paren_pos = normalized.size() - 1;
    bool all_digits = digits_start < paren_pos;
    for (size_t i = digits_start; i < paren_pos && all_digits; ++i)
      all_digits = (std::isdigit(static_cast<unsigned char>(normalized[i])) != 0);
    if (all_digits)
      normalized.erase(inst_pos, paren_pos - inst_pos);
  }

  return normalized;
}

size_t WarningCollector::DedupKeyHash::operator()(const DedupKey &key) const noexcept {
  size_t seed = 0;
  hazard_core::hash_combine(seed, key.dispatch_id);
  hazard_core::hash_combine(seed, key.pc);
  hazard_core::hash_combine(seed, key.message);
  hazard_core::hash_combine(seed, key.suggestion);
  return seed;
}

void WarningCollector::add(const HazardWarning &warning) {
  bool is_new = true;
  std::function<void(const HazardWarning &)> callback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (verbose_) {
      warnings_.push_back(warning);
    } else {
      DedupKey key{warning.dispatch_id, warning.pc, dedup_message_key(warning.message),
                   warning.suggestion};
      auto it = dedup_index_.find(key);
      if (it == dedup_index_.end()) {
        dedup_index_.emplace(std::move(key), warnings_.size());
        warnings_.push_back(warning);
      } else {
        warnings_[it->second].suppressed_occurrences++;
        is_new = false;
      }
    }
    callback = on_new_warning_;
  }

  // Run outside the lock: the callback writes through the plugin sink, which is
  // an unrelated lock order this class should not pull inside its own.
  if (is_new && callback)
    callback(warning);
}

void WarningCollector::set_on_new_warning(std::function<void(const HazardWarning &)> callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  on_new_warning_ = std::move(callback);
}

void WarningCollector::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  warnings_.clear();
  dedup_index_.clear();
}

std::vector<HazardWarning> WarningCollector::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return warnings_;
}

std::string WarningCollector::to_json() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string out;
  if (warnings_.empty())
    return "[]\n";

  out += "[\n";
  for (size_t i = 0; i < warnings_.size(); ++i) {
    const HazardWarning &w = warnings_[i];
    const std::string raw_isa = format_raw_isa_hex(w.raw_isa);
    const std::string source_raw_isa = format_raw_isa_hex(w.source_raw_isa);

    append_indent(out, 1);
    out += "{\n";
    append_uint_field(out, 2, "instruction_id", w.instruction_id, true);
    append_uint_field(out, 2, "wave_id", w.wave_id, true);
    append_uint_field(out, 2, "pc", w.pc, true);
    append_uint_field(out, 2, "dispatch_id", w.dispatch_id, true);
    append_uint_field(out, 2, "cluster_id", w.cluster_id, true);
    append_uint_field(out, 2, "workgroup_id", w.workgroup_id, true);

    append_indent(out, 2);
    append_json_string(out, "consumer");
    out += ": {\n";
    append_uint_field(out, 3, "instruction_id", w.instruction_id, true);
    append_uint_field(out, 3, "pc", w.pc, true);
    append_uint_field(out, 3, "dispatch_id", w.dispatch_id, true);
    append_uint_field(out, 3, "cluster_id", w.cluster_id, true);
    append_uint_field(out, 3, "workgroup_id", w.workgroup_id, true);
    append_uint_field(out, 3, "wave_id", w.wave_id, true);
    append_string_field(out, 3, "instruction_text", w.instruction_text, true);
    append_string_field(out, 3, "raw_isa", raw_isa, false);
    append_indent(out, 2);
    out += "},\n";

    append_indent(out, 2);
    append_json_string(out, "producer");
    out += ": {\n";
    append_uint_field(out, 3, "instruction_id", w.source_instruction_id, true);
    append_uint_field(out, 3, "pc", w.source_pc, true);
    append_bool_field(out, 3, "has_descriptor", w.has_source_instruction, true);
    append_string_field(out, 3, "instruction_text", w.source_instruction_text, true);
    append_string_field(out, 3, "raw_isa", source_raw_isa, false);
    append_indent(out, 2);
    out += "},\n";

    append_string_field(out, 2, "instruction_text", w.instruction_text, true);
    append_string_field(out, 2, "raw_isa", raw_isa, true);
    append_string_field(out, 2, "source_instruction_text", w.source_instruction_text, true);
    append_string_field(out, 2, "source_raw_isa", source_raw_isa, true);
    append_uint_field(out, 2, "source_pc", w.source_pc, true);
    append_string_field(out, 2, "message", w.message, true);
    append_string_field(out, 2, "suggestion", w.suggestion, true);
    append_uint_field(out, 2, "suppressed_occurrences", w.suppressed_occurrences, false);

    append_indent(out, 1);
    out += (i + 1 == warnings_.size()) ? "}\n" : "},\n";
  }
  out += "]\n";
  return out;
}

std::string format_warning_line(const HazardWarning &warning) {
  std::ostringstream out;
  out << "[data_hazard] pc=0x" << std::hex << warning.pc << std::dec << " wave=" << warning.wave_id
      << ": " << warning.message;
  if (!warning.suggestion.empty())
    out << " | " << warning.suggestion;
  if (warning.suppressed_occurrences != 0)
    out << " (+" << warning.suppressed_occurrences << " more)";
  return out.str();
}

std::string WarningCollector::to_summary() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::ostringstream out;
  out << "[data_hazard] " << warnings_.size() << " hazard" << (warnings_.size() == 1 ? "" : "s")
      << " detected\n";
  // Each hazard was already streamed when it was first collected, so only
  // repeat the ones whose duplicate count was still unknown at that point.
  for (const HazardWarning &w : warnings_) {
    if (w.suppressed_occurrences != 0)
      out << format_warning_line(w) << '\n';
  }
  return out.str();
}

bool WarningCollector::write_json_file(const std::string &path) const {
  const std::string json = to_json();
  std::ofstream file(path);
  if (!file)
    return false;
  file << json;
  return static_cast<bool>(file);
}

void CollectingWarningSink::emit_warning(const hazard_core::EngineWarning &warning) const {
  const hazard_core::HazardFinding &finding = warning.finding;
  const hazard_core::InstructionDescriptor &inst = finding.instruction;

  HazardWarning out;
  out.instruction_id = inst.instruction_id;
  out.wave_id = inst.execution.wave_id;
  out.pc = inst.pc;
  out.message = warning.message;
  out.suggestion = warning.suggestion;
  out.dispatch_id = inst.execution.dispatch_id;
  out.cluster_id = inst.execution.cluster_id;
  out.workgroup_id = inst.execution.workgroup_id;
  out.raw_isa = inst.raw_isa;
  if (formatter_ != nullptr)
    out.instruction_text = formatter_->format_instruction(inst);

  if (warning.has_source || finding.has_source_instruction) {
    out.has_source_instruction = true;
    out.source_pc = warning.source_pc;
    out.source_raw_isa = warning.source_raw_isa;

    // A structured producer descriptor is more precise than the raw words the
    // engine carries alongside the warning, so prefer it when present.
    if (finding.has_source_instruction) {
      out.source_instruction_id = finding.source_instruction.instruction_id;
      out.source_pc = finding.source_instruction.pc;
      out.source_raw_isa = finding.source_instruction.raw_isa;
    }
    if (formatter_ != nullptr) {
      out.source_instruction_text = formatter_->format_instruction(
          finding.has_source_instruction ? finding.source_instruction
                                         : hazard_core::InstructionDescriptor{});
    }
  }

  collector_.add(out);
}

} // namespace rocjitsu::plugins::data_hazard
