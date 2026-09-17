// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <istream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace rocjitsu::test {

/// Controls whether a race test expects one or one-or-more findings.
enum class FindingCount {
  One,
  OneOrMore,
};

/// Describes the observable race report expected from a test kernel.
///
/// Tests set only the fields relevant to the behavior they protect. The
/// matcher compares those fields with a race record after validating the
/// requested finding count. Instruction fields refer to the two marked
/// instructions and, optionally, an instruction between those markers in the
/// human-readable race trace.
struct RaceExpectation {
  FindingCount findings = FindingCount::One;

  // Normalized report fields. Type and access use exact matches. Kernel is a
  // substring expected in both the resolved display name and ELF symbol.
  const char *kernel = nullptr;
  const char *type = nullptr;
  const char *access = nullptr;

  // Negative context values mean that the field is not checked.
  int wave = -1;
  int lane = -1;

  // Substrings expected in the marked producer, intervening instructions, and
  // marked consumer sections of the human-readable trace.
  const char *producer = nullptr;
  const char *between = nullptr;
  const char *consumer = nullptr;
};

struct RaceRecord {
  std::string kernel;
  std::string symbol;
  int dispatch = -1;
  std::string type;
  std::string access;
  int reg = -1;
  int wave = -1;
  int lane = -1;
  std::string workgroup;
  std::string conflict;
  std::string message;
};

struct RaceLogParseResult {
  std::vector<RaceRecord> records;
  std::string error;

  [[nodiscard]] bool ok() const { return error.empty(); }
};

struct TraceSections {
  std::string header;
  std::string producer;
  std::vector<std::string> between;
  std::string consumer;
  std::size_t marker_count = 0;
};

struct RaceExpectationMatchResult {
  std::vector<std::string> errors;

  [[nodiscard]] bool ok() const { return errors.empty(); }

  [[nodiscard]] std::string message() const {
    std::ostringstream output;
    for (std::size_t index = 0; index < errors.size(); ++index) {
      if (index != 0)
        output << '\n';
      output << errors[index];
    }
    return output.str();
  }
};

namespace detail {

inline RaceLogParseResult parseFailure(std::string error) {
  return RaceLogParseResult{.records = {}, .error = std::move(error)};
}

inline bool parseInteger(std::string_view text, int &value) {
  if (text.empty())
    return false;
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto [parsed_end, error] = std::from_chars(begin, end, value);
  return error == std::errc{} && parsed_end == end;
}

inline std::string lineError(std::size_t line_number, const std::string &message) {
  return "race log line " + std::to_string(line_number) + ": " + message;
}

inline void appendMismatch(std::vector<std::string> &errors, const std::string &field,
                           const std::string &actual, const std::string &expected) {
  errors.push_back(field + " mismatch: expected '" + expected + "', got '" + actual + "'");
}

inline void appendMissing(std::vector<std::string> &errors, const std::string &field) {
  errors.push_back("missing " + field);
}

inline bool contains(const std::string &value, const std::string &substring) {
  return value.find(substring) != std::string::npos;
}

class RaceHeaderParser {
public:
  explicit RaceHeaderParser(std::size_t line_number) : line_number_(line_number) {}

  [[nodiscard]] std::optional<std::string> parse(std::string_view header_text) {
    std::istringstream header{std::string(header_text)};
    std::string field;
    while (header >> field) {
      const std::size_t equals = field.find('=');
      if (equals == std::string::npos || equals == 0 || equals + 1 == field.size()) {
        return lineError(line_number_, "malformed header field '" + field + "'");
      }

      const std::string_view key(field.data(), equals);
      const std::string_view value(field.data() + equals + 1, field.size() - equals - 1);
      if (auto error = parseField(key, value))
        return error;
    }

    if (!std::all_of(seen_.begin(), seen_.end(), [](bool seen) { return seen; }))
      return lineError(line_number_, "RACE header is missing one or more required fields");
    return std::nullopt;
  }

  [[nodiscard]] RaceRecord takeRecord() && { return std::move(record_); }

private:
  struct FieldDescriptor {
    std::string_view key;
    std::string RaceRecord::*string_member;
    int RaceRecord::*integer_member;
  };

  inline static constexpr std::array<FieldDescriptor, 10> fields_{{
      {"kernel", &RaceRecord::kernel, nullptr},
      {"symbol", &RaceRecord::symbol, nullptr},
      {"dispatch", nullptr, &RaceRecord::dispatch},
      {"type", &RaceRecord::type, nullptr},
      {"access", &RaceRecord::access, nullptr},
      {"reg", nullptr, &RaceRecord::reg},
      {"wave", nullptr, &RaceRecord::wave},
      {"lane", nullptr, &RaceRecord::lane},
      {"wg", &RaceRecord::workgroup, nullptr},
      {"conflict", &RaceRecord::conflict, nullptr},
  }};

  [[nodiscard]] std::optional<std::string> parseField(std::string_view key,
                                                      std::string_view value) {
    for (std::size_t index = 0; index < fields_.size(); ++index) {
      const FieldDescriptor &descriptor = fields_[index];
      if (descriptor.key != key)
        continue;
      if (seen_[index])
        return lineError(line_number_, "duplicate header field '" + std::string(key) + "'");
      seen_[index] = true;

      if (descriptor.string_member != nullptr) {
        record_.*descriptor.string_member = value;
      } else if (!parseInteger(value, record_.*descriptor.integer_member)) {
        return lineError(line_number_,
                         "invalid " + std::string(key) + " value '" + std::string(value) + "'");
      }
      return std::nullopt;
    }

    // Ignore well-formed unknown fields so the producer can extend the format.
    return std::nullopt;
  }

  std::size_t line_number_;
  RaceRecord record_;
  std::array<bool, fields_.size()> seen_{};
};

} // namespace detail

/// Parse structured RACE blocks from an already-open stream.
///
/// Non-RACE lines (such as plugin summaries and dispatch diagnostics) are
/// ignored. Once a RACE header is observed, its complete required field set and
/// END_RACE terminator are mandatory. A valid zero-finding log must still
/// contain recognizable race-plugin output, such as a kernel dispatch line.
inline RaceLogParseResult parseRaceLog(std::istream &input) {
  std::vector<RaceRecord> records;
  std::string line;
  std::size_t line_number = 0;
  bool saw_plugin_output = false;
  while (std::getline(input, line)) {
    ++line_number;
    if (line == "END_RACE")
      return detail::parseFailure(detail::lineError(line_number, "unexpected END_RACE"));
    if (!line.starts_with("RACE ")) {
      saw_plugin_output = saw_plugin_output || line.starts_with("[rocjitsu] Kernel dispatch:") ||
                          line.find("ROCJITSU RACE DETECTION SUMMARY") != std::string::npos;
      continue;
    }
    saw_plugin_output = true;

    detail::RaceHeaderParser header_parser(line_number);
    if (auto error = header_parser.parse(std::string_view(line).substr(5)))
      return detail::parseFailure(std::move(*error));
    RaceRecord record = std::move(header_parser).takeRecord();

    bool terminated = false;
    while (std::getline(input, line)) {
      ++line_number;
      if (line == "END_RACE") {
        terminated = true;
        break;
      }
      if (line.starts_with("RACE ")) {
        return detail::parseFailure(
            detail::lineError(line_number, "nested RACE header before END_RACE"));
      }
      if (!record.message.empty())
        record.message += '\n';
      record.message += line;
    }
    if (!terminated) {
      return detail::parseFailure(
          detail::lineError(line_number, "RACE block reached EOF before END_RACE"));
    }
    records.push_back(std::move(record));
  }

  if (input.bad())
    return detail::parseFailure("failed while reading race log");
  if (!saw_plugin_output)
    return detail::parseFailure("race log contains no recognizable race-plugin output");
  return RaceLogParseResult{.records = std::move(records), .error = {}};
}

inline RaceLogParseResult parseRaceLogFile(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open())
    return detail::parseFailure("could not open race log '" + path.string() + "'");
  return parseRaceLog(file);
}

inline RaceLogParseResult parseRaceLogFromEnvironment() {
  const char *directory = std::getenv("RJ_SINK_DIR");
  if (directory == nullptr || directory[0] == '\0')
    return detail::parseFailure("RJ_SINK_DIR is not set");
  return parseRaceLogFile(std::filesystem::path(directory) / "race.log");
}

inline std::string stripTraceMarkPrefix(const std::string &line) {
  const std::size_t mark = line.find("==>");
  if (mark == std::string::npos)
    return line;

  const std::string rest = line.substr(mark + 3);
  const std::size_t address = rest.find("0x");
  if (address == std::string::npos)
    return rest;

  std::size_t instruction = rest.find(' ', address);
  if (instruction != std::string::npos)
    instruction = rest.find_first_not_of(' ', instruction);
  if (instruction == std::string::npos)
    return rest;
  return rest.substr(instruction);
}

inline TraceSections parseTrace(const RaceRecord &record) {
  TraceSections trace;
  std::istringstream stream(record.message);
  std::string line;
  while (std::getline(stream, line)) {
    if (trace.header.empty()) {
      trace.header = line;
      continue;
    }

    if (line.find("==>") != std::string::npos) {
      if (trace.marker_count == 0)
        trace.producer = stripTraceMarkPrefix(line);
      else
        trace.consumer = stripTraceMarkPrefix(line);
      ++trace.marker_count;
    } else if (trace.marker_count == 1) {
      trace.between.push_back(line);
    }
  }
  return trace;
}

inline bool anyContains(const std::vector<std::string> &lines, const std::string &substring) {
  return std::any_of(lines.begin(), lines.end(),
                     [&](const std::string &line) { return detail::contains(line, substring); });
}

inline RaceExpectationMatchResult matchRaceRecord(const RaceRecord &record,
                                                  const RaceExpectation &expected) {
  RaceExpectationMatchResult result;
  if (expected.kernel != nullptr) {
    if (record.kernel.empty())
      detail::appendMissing(result.errors, "kernel name");
    else if (record.kernel == "?")
      result.errors.push_back("kernel name is unresolved");
    if (record.symbol.empty())
      detail::appendMissing(result.errors, "kernel symbol");
    else if (record.symbol == "?")
      result.errors.push_back("kernel symbol is unresolved");
    if (record.dispatch < 1)
      result.errors.push_back("dispatch id must be positive");
    if (!detail::contains(record.kernel, expected.kernel)) {
      result.errors.push_back("kernel name '" + record.kernel + "' does not contain '" +
                              expected.kernel + "'");
    }
    if (!detail::contains(record.symbol, expected.kernel)) {
      result.errors.push_back("kernel symbol '" + record.symbol + "' does not contain '" +
                              expected.kernel + "'");
    }
  }
  if (expected.type != nullptr && record.type != expected.type)
    detail::appendMismatch(result.errors, "race type", record.type, expected.type);
  if (expected.access != nullptr && record.access != expected.access)
    detail::appendMismatch(result.errors, "race access", record.access, expected.access);
  if (expected.wave >= 0 && record.wave != expected.wave) {
    detail::appendMismatch(result.errors, "wave", std::to_string(record.wave),
                           std::to_string(expected.wave));
  }
  if (expected.lane >= 0 && record.lane != expected.lane) {
    detail::appendMismatch(result.errors, "lane", std::to_string(record.lane),
                           std::to_string(expected.lane));
  }

  const TraceSections trace = parseTrace(record);
  if (expected.type != nullptr && !detail::contains(trace.header, expected.type)) {
    result.errors.push_back("trace header does not contain race type '" +
                            std::string(expected.type) + "'");
  }
  if (expected.wave >= 0 &&
      !detail::contains(trace.header, "wave " + std::to_string(expected.wave))) {
    result.errors.push_back("trace header does not contain expected wave");
  }
  if (expected.lane >= 0 &&
      !detail::contains(trace.header, "lane " + std::to_string(expected.lane))) {
    result.errors.push_back("trace header does not contain expected lane");
  }
  if (expected.producer != nullptr && !detail::contains(trace.producer, expected.producer)) {
    result.errors.push_back("producer trace '" + trace.producer + "' does not contain '" +
                            expected.producer + "'");
  }
  if (expected.between != nullptr && !anyContains(trace.between, expected.between)) {
    result.errors.push_back("intervening trace does not contain '" + std::string(expected.between) +
                            "'");
  }
  if (expected.consumer != nullptr && !detail::contains(trace.consumer, expected.consumer)) {
    result.errors.push_back("consumer trace '" + trace.consumer + "' does not contain '" +
                            expected.consumer + "'");
  }
  return result;
}

inline RaceExpectationMatchResult matchRaceExpectation(const std::vector<RaceRecord> &records,
                                                       const RaceExpectation &expected) {
  if (expected.findings == FindingCount::One) {
    if (records.size() != 1) {
      return RaceExpectationMatchResult{
          .errors = {"expected exactly one race, got " + std::to_string(records.size())}};
    }
    return matchRaceRecord(records.front(), expected);
  }

  if (records.empty())
    return RaceExpectationMatchResult{.errors = {"expected at least one race, got none"}};

  RaceExpectationMatchResult closest = matchRaceRecord(records.front(), expected);
  if (closest.ok())
    return closest;
  for (std::size_t index = 1; index < records.size(); ++index) {
    RaceExpectationMatchResult candidate = matchRaceRecord(records[index], expected);
    if (candidate.ok())
      return candidate;
    if (candidate.errors.size() < closest.errors.size())
      closest = std::move(candidate);
  }

  closest.errors.insert(closest.errors.begin(),
                        "none of " + std::to_string(records.size()) +
                            " races matched the expectation; closest record:");
  return closest;
}

} // namespace rocjitsu::test
