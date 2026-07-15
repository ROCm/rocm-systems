// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/code/executable.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

using namespace rocjitsu;

namespace {

constexpr int kUsageError = 1;
constexpr int kInputError = 2;
constexpr int kHazardDetected = 4;

struct TargetInfo {
  std::string_view name;
  rj_code_target_id_t target;
};

constexpr std::array<TargetInfo, 6> kSupportedTargets = {{
    {"gfx942", ROCJITSU_CODE_TARGET_GFX942},
    {"gfx950", ROCJITSU_CODE_TARGET_GFX950},
    {"gfx1100", ROCJITSU_CODE_TARGET_GFX1100},
    {"gfx1200", ROCJITSU_CODE_TARGET_GFX1200},
    {"gfx1201", ROCJITSU_CODE_TARGET_GFX1201},
    {"gfx1250", ROCJITSU_CODE_TARGET_GFX1250},
}};

struct CliOptions {
  std::vector<std::string> input_paths;
  std::optional<rj_code_target_id_t> target;
  uint32_t code_object_index = 0;
  bool code_object_index_set = false;
  std::optional<uint64_t> kernel_entry;
  size_t max_diagnostics = 32;
  bool all_code_objects = false;
  bool list_code_objects = false;
  bool recursive = false;
  bool skip_unsupported = false;
  bool summary_only = false;
  bool stop_after_first_diagnostic = false;
  bool no_fail = false;
  bool show_help = false;
};

struct SelectedCodeObject {
  std::unique_ptr<Executable> executable;
  const AmdGpuCodeObject *code_object = nullptr;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  uint32_t index = 0;
};

struct ScanTotals {
  uint32_t inputs = 0;
  uint32_t skipped = 0;
  uint32_t code_objects = 0;
  size_t diagnostics = 0;
  bool diagnostics_truncated = false;
  bool hazards = false;
};

void print_supported_targets(std::ostream &os) {
  for (size_t i = 0; i < kSupportedTargets.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << kSupportedTargets[i].name;
  }
}

void print_help() {
  std::cout << "Usage: rj_waitcheck INPUT... [options]\n\n"
            << "Options:\n"
            << "  --target TARGET          Select target from executable inputs\n"
            << "  --code-object-index N    Code-object index for the selected target (default: 0)\n"
            << "  --kernel-entry OFFSET    Analyze only the kernel at this .text byte offset\n"
            << "  --all-code-objects       Analyze all supported code objects\n"
            << "  --list-code-objects      List supported code objects and exit\n"
            << "  --recursive              Expand directory inputs into recursive file sweeps\n"
            << "  --skip-unsupported       Skip unparsable or unsupported inputs\n"
            << "  --max-diagnostics N      Limit collected and printed diagnostics (default: 32)\n"
            << "  --stop-after-first-diagnostic\n"
            << "                           Stop each code object after the first observed hazard\n"
            << "  --summary-only           Print only final batch totals\n"
            << "  --no-fail                Return success even when hazards are reported\n"
            << "  --help                   Show this help\n\n"
            << "Supported target names: ";
  print_supported_targets(std::cout);
  std::cout << ".\n";
}

[[nodiscard]] std::string_view target_name(rj_code_target_id_t target) {
  for (const TargetInfo &info : kSupportedTargets) {
    if (info.target == target)
      return info.name;
  }
  return "unsupported";
}

[[nodiscard]] std::optional<rj_code_target_id_t> parse_target(std::string_view value) {
  for (const TargetInfo &info : kSupportedTargets) {
    if (value == info.name)
      return info.target;
  }
  return std::nullopt;
}

[[nodiscard]] bool parse_u32(std::string_view text, uint32_t &value) {
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value, base);
  return ec == std::errc{} && ptr == end;
}

[[nodiscard]] bool parse_u64(std::string_view text, uint64_t &value) {
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value, base);
  return ec == std::errc{} && ptr == end;
}

[[nodiscard]] bool require_value(int argc, char **argv, int &index, std::string_view flag,
                                 std::string_view &value) {
  if (index + 1 >= argc) {
    std::cerr << "missing value for " << flag << "\n";
    return false;
  }
  ++index;
  value = argv[index];
  return true;
}

[[nodiscard]] bool parse_args(int argc, char **argv, CliOptions &options) {
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    std::string_view value;

    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return true;
    }
    if (arg == "--list-code-objects") {
      options.list_code_objects = true;
      continue;
    }
    if (arg == "--all-code-objects") {
      options.all_code_objects = true;
      continue;
    }
    if (arg == "--recursive") {
      options.recursive = true;
      continue;
    }
    if (arg == "--skip-unsupported") {
      options.skip_unsupported = true;
      continue;
    }
    if (arg == "--summary-only") {
      options.summary_only = true;
      continue;
    }
    if (arg == "--stop-after-first-diagnostic") {
      options.stop_after_first_diagnostic = true;
      continue;
    }
    if (arg == "--no-fail") {
      options.no_fail = true;
      continue;
    }
    if (arg == "--target") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto target = parse_target(value);
      if (!target) {
        std::cerr << "invalid target: " << value << "\n";
        return false;
      }
      options.target = *target;
      continue;
    }
    if (arg == "--code-object-index") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_u32(value, options.code_object_index)) {
        std::cerr << "invalid code-object index: " << value << "\n";
        return false;
      }
      options.code_object_index_set = true;
      continue;
    }
    if (arg == "--kernel-entry") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      uint64_t kernel_entry = 0;
      if (!parse_u64(value, kernel_entry)) {
        std::cerr << "invalid kernel entry: " << value << "\n";
        return false;
      }
      options.kernel_entry = kernel_entry;
      continue;
    }
    if (arg == "--max-diagnostics") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      uint32_t max_diagnostics = 0;
      if (!parse_u32(value, max_diagnostics)) {
        std::cerr << "invalid max diagnostics: " << value << "\n";
        return false;
      }
      options.max_diagnostics = max_diagnostics;
      continue;
    }

    if (!arg.empty() && arg.front() != '-') {
      options.input_paths.emplace_back(arg);
      continue;
    }

    std::cerr << "unknown option: " << arg << "\n";
    return false;
  }

  if (options.all_code_objects && options.code_object_index_set) {
    std::cerr << "--code-object-index cannot be used with --all-code-objects\n";
    return false;
  }
  if (options.all_code_objects && options.kernel_entry) {
    std::cerr << "--kernel-entry cannot be used with --all-code-objects\n";
    return false;
  }

  return true;
}

[[nodiscard]] bool expand_recursive_inputs(CliOptions &options, std::string &error) {
  if (!options.recursive)
    return true;

  std::vector<std::string> expanded;
  for (const std::string &input_path : options.input_paths) {
    const std::filesystem::path path(input_path);
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec)) {
      expanded.push_back(input_path);
      continue;
    }

    std::vector<std::string> directory_files;
    std::filesystem::recursive_directory_iterator it(
        path, std::filesystem::directory_options::skip_permission_denied, ec);
    if (ec) {
      error = input_path + ": failed to recurse input directory: " + ec.message();
      return false;
    }

    const std::filesystem::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
      if (ec) {
        ec.clear();
        continue;
      }
      if (!it->is_regular_file(ec)) {
        ec.clear();
        continue;
      }
      directory_files.push_back(it->path().string());
    }
    std::sort(directory_files.begin(), directory_files.end());
    expanded.insert(expanded.end(), directory_files.begin(), directory_files.end());
  }

  if (expanded.empty()) {
    error = "no regular files found under recursive input paths";
    return false;
  }
  options.input_paths = std::move(expanded);
  return true;
}

[[nodiscard]] std::string hex_offset(uint64_t value) {
  std::ostringstream os;
  os << "0x" << std::hex << value;
  return os.str();
}

[[nodiscard]] std::string count_label(size_t count, bool lower_bound) {
  std::ostringstream os;
  if (lower_bound)
    os << ">=";
  os << count;
  return os.str();
}

void list_code_objects(const Executable &executable, std::string_view input_path,
                       bool include_path) {
  for (const TargetInfo &info : kSupportedTargets) {
    const uint32_t count = executable.num_code_objects(info.target);
    if (include_path)
      std::cout << input_path << ":";
    std::cout << info.name << ": " << count << "\n";
  }
}

[[nodiscard]] SelectedCodeObject
select_code_object(const CliOptions &options, const std::string &input_path, std::string &error) {
  SelectedCodeObject selected;
  selected.executable = std::make_unique<Executable>(input_path);
  if (!selected.executable->is_valid()) {
    error = "failed to parse input executable or code object";
    selected.executable.reset();
    return selected;
  }

  if (options.target) {
    selected.code_object =
        selected.executable->code_object(*options.target, options.code_object_index);
    if (!selected.code_object) {
      std::ostringstream os;
      os << "failed to select " << target_name(*options.target) << " code object "
         << options.code_object_index;
      error = os.str();
      selected.executable.reset();
      return selected;
    }
    selected.target = *options.target;
    selected.index = options.code_object_index;
    return selected;
  }

  std::optional<rj_code_target_id_t> target_with_objects;
  uint32_t target_count = 0;
  for (const TargetInfo &info : kSupportedTargets) {
    if (selected.executable->num_code_objects(info.target) == 0)
      continue;
    target_with_objects = info.target;
    ++target_count;
  }

  if (target_count == 0) {
    error = "no supported code objects found";
    selected.executable.reset();
    return selected;
  }
  if (target_count > 1) {
    error = "multiple supported targets found; pass --target";
    selected.executable.reset();
    return selected;
  }

  selected.code_object =
      selected.executable->code_object(*target_with_objects, options.code_object_index);
  if (!selected.code_object) {
    std::ostringstream os;
    os << "failed to select " << target_name(*target_with_objects) << " code object "
       << options.code_object_index;
    error = os.str();
    selected.executable.reset();
    return selected;
  }
  selected.target = *target_with_objects;
  selected.index = options.code_object_index;
  return selected;
}

void print_diagnostics(const std::string &input_path, rj_code_target_id_t target,
                       uint32_t code_object_index, const WaitcheckReport &report,
                       size_t max_diagnostics) {
  const size_t limit = std::min(max_diagnostics, report.diagnostics.size());
  for (size_t i = 0; i < limit; ++i) {
    const auto &diag = report.diagnostics[i];
    std::cout << input_path << ":" << target_name(target) << "[" << code_object_index
              << "]:" << diag.section_name << "+" << hex_offset(diag.section_offset) << ": "
              << diag.message << "\n";
    std::cout << "  producer " << diag.section_name << "+"
              << hex_offset(diag.producer_section_offset) << ": " << diag.producer_instruction
              << "\n";
    std::cout << "  consumer " << diag.section_name << "+" << hex_offset(diag.section_offset)
              << ": " << diag.instruction << "\n";
  }
  if (report.diagnostics.size() > limit) {
    std::cout << "omitted " << (report.diagnostics.size() - limit) << " additional diagnostic(s)\n";
  } else if (report.diagnostics_truncated) {
    if (report.diagnostics_observed > limit) {
      std::cout << "omitted at least " << (report.diagnostics_observed - limit)
                << " diagnostic(s) after limit\n";
    } else {
      std::cout << "omitted additional diagnostic(s) after limit\n";
    }
  }
}

void print_summary(const std::string &input_path, rj_code_target_id_t target,
                   uint32_t code_object_index, std::optional<uint64_t> kernel_entry,
                   const WaitcheckReport &report) {
  std::cout << "rj_waitcheck: " << input_path << ":" << target_name(target) << "["
            << code_object_index << "]";
  if (kernel_entry)
    std::cout << ":kernel=.text+" << hex_offset(*kernel_entry);
  std::cout << ": instructions=" << count_label(report.instructions_analyzed, report.stopped_early)
            << " memory-events=" << count_label(report.memory_events_tracked, report.stopped_early)
            << " diagnostics="
            << count_label(report.diagnostics_observed, report.diagnostics_truncated) << "\n";
}

void skip_input(const std::string &input_path, std::string_view reason, ScanTotals &totals,
                bool quiet = false) {
  ++totals.skipped;
  if (quiet)
    return;
  std::cout << "rj_waitcheck: " << input_path << ": skipped: " << reason << "\n";
}

[[nodiscard]] bool analyze_code_object(const CliOptions &options, const std::string &input_path,
                                       rj_code_target_id_t target, uint32_t code_object_index,
                                       const AmdGpuCodeObject &code_object, ScanTotals &totals,
                                       std::string &error) {
  const rj_code_arch_t arch = waitcheck_arch_for_target(target);
  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    error = "target is not supported by waitcheck: " + std::string(target_name(target));
    return false;
  }

  WaitcheckOptions analysis_options;
  analysis_options.max_diagnostics = options.max_diagnostics;
  analysis_options.stop_after_first_diagnostic = options.stop_after_first_diagnostic;
  WaitcheckReport report =
      options.kernel_entry
          ? analyze_waitcnts_for_kernel(code_object, arch, *options.kernel_entry, analysis_options)
          : analyze_waitcnts(code_object, arch, analysis_options);
  if (!report.supported) {
    error = "waitcheck analysis failed for " + std::string(target_name(target)) + "[" +
            std::to_string(code_object_index) + "]";
    if (!report.analysis_error.empty())
      error += ": " + report.analysis_error;
    return false;
  }

  ++totals.code_objects;
  totals.diagnostics += report.diagnostics_observed;
  totals.diagnostics_truncated = totals.diagnostics_truncated || report.diagnostics_truncated;
  totals.hazards = totals.hazards || !report.passed();

  if (!options.summary_only) {
    print_summary(input_path, target, code_object_index, options.kernel_entry, report);
    print_diagnostics(input_path, target, code_object_index, report, options.max_diagnostics);
  }
  return true;
}

[[nodiscard]] bool scan_selected_code_object(const CliOptions &options,
                                             const std::string &input_path, ScanTotals &totals,
                                             std::string &error) {
  std::string selection_error;
  SelectedCodeObject selected = select_code_object(options, input_path, selection_error);
  if (!selected.executable) {
    if (options.skip_unsupported) {
      skip_input(input_path, selection_error, totals, options.summary_only);
      return true;
    }
    error = input_path + ": " + selection_error;
    return false;
  }

  if (!analyze_code_object(options, input_path, selected.target, selected.index,
                           *selected.code_object, totals, error)) {
    if (options.skip_unsupported) {
      skip_input(input_path, error, totals, options.summary_only);
      return true;
    }
    error = input_path + ": " + error;
    return false;
  }
  return true;
}

[[nodiscard]] bool scan_all_code_objects(const CliOptions &options, const std::string &input_path,
                                         ScanTotals &totals, std::string &error) {
  Executable executable(input_path);
  if (!executable.is_valid()) {
    if (options.skip_unsupported) {
      skip_input(input_path, "failed to parse input executable or code object", totals,
                 options.summary_only);
      return true;
    }
    error = input_path + ": failed to parse input executable or code object";
    return false;
  }

  bool found = false;
  for (const TargetInfo &info : kSupportedTargets) {
    if (options.target && *options.target != info.target)
      continue;

    const uint32_t count = executable.num_code_objects(info.target);
    for (uint32_t index = 0; index < count; ++index) {
      const AmdGpuCodeObject *code_object = executable.code_object(info.target, index);
      if (!code_object) {
        std::ostringstream os;
        os << input_path << ": failed to select " << info.name << " code object " << index;
        error = os.str();
        return false;
      }
      found = true;
      if (!analyze_code_object(options, input_path, info.target, index, *code_object, totals,
                               error)) {
        if (options.skip_unsupported) {
          skip_input(input_path, error, totals, options.summary_only);
          continue;
        }
        error = input_path + ": " + error;
        return false;
      }
    }
  }

  if (!found) {
    const std::string reason =
        options.target ? "no " + std::string(target_name(*options.target)) + " code objects found"
                       : "no supported code objects found";
    if (options.skip_unsupported) {
      skip_input(input_path, reason, totals, options.summary_only);
      return true;
    }
    error = input_path + ": " + reason;
    return false;
  }

  return true;
}

} // namespace

int main(int argc, char **argv) {
  CliOptions options;
  if (!parse_args(argc, argv, options))
    return kUsageError;

  if (options.show_help) {
    print_help();
    return 0;
  }

  if (options.input_paths.empty()) {
    std::cerr << "input path is required\n";
    return kUsageError;
  }
  std::string expansion_error;
  if (!expand_recursive_inputs(options, expansion_error)) {
    std::cerr << expansion_error << "\n";
    return kInputError;
  }

  if (options.list_code_objects) {
    const bool include_path = options.input_paths.size() > 1;
    ScanTotals totals;
    for (const std::string &input_path : options.input_paths) {
      ++totals.inputs;
      Executable executable(input_path);
      if (!executable.is_valid()) {
        if (options.skip_unsupported) {
          skip_input(input_path, "failed to parse input executable or code object", totals);
          continue;
        }
        std::cerr << input_path << ": failed to parse input executable or code object\n";
        return kInputError;
      }
      list_code_objects(executable, input_path, include_path);
    }
    return 0;
  }

  const bool batch_mode = options.input_paths.size() > 1 || options.all_code_objects ||
                          options.skip_unsupported || options.recursive || options.summary_only;
  ScanTotals totals;
  for (const std::string &input_path : options.input_paths) {
    ++totals.inputs;
    std::string error;
    const bool ok = options.all_code_objects
                        ? scan_all_code_objects(options, input_path, totals, error)
                        : scan_selected_code_object(options, input_path, totals, error);
    if (!ok) {
      std::cerr << error << "\n";
      return kInputError;
    }
  }

  if (batch_mode) {
    std::cout << "rj_waitcheck: scanned inputs=" << totals.inputs << " skipped=" << totals.skipped
              << " code-objects=" << totals.code_objects
              << " diagnostics=" << count_label(totals.diagnostics, totals.diagnostics_truncated)
              << "\n";
  }

  if (totals.hazards && !options.no_fail)
    return kHazardDetected;
  return 0;
}
