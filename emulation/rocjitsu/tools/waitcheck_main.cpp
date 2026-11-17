// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/cli.h"
#include "rocjitsu/code/analysis/waitcheck.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <unistd.h>

using namespace rocjitsu;

std::string_view waitcheck_target_name(rj_code_target_id_t target) {
  const auto *gpu = default_isa_target_registry().find_gpu_target(target);
  return gpu ? gpu->code_object_id : "unknown";
}

namespace {

constexpr int kUsageError = 1;
constexpr int kInputError = 2;
constexpr int kHazardDetected = 4;
constexpr uint32_t kMaxJobs = 16;
constexpr uint32_t kMaxSlowestKernels = 1000;
constexpr size_t kKernelBatchSize = 4;
constexpr size_t kAllocatorTrimKernelCadence = 8;

enum class ProgressMode : uint8_t { Auto, Always, Never };

constexpr std::array<rj_code_target_id_t, 9> kSupportedTargets = {
    ROCJITSU_CODE_TARGET_GFX942,  ROCJITSU_CODE_TARGET_GFX950,  ROCJITSU_CODE_TARGET_GFX1100,
    ROCJITSU_CODE_TARGET_GFX1150, ROCJITSU_CODE_TARGET_GFX1151, ROCJITSU_CODE_TARGET_GFX1200,
    ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251,
};

struct CliOptions {
  std::vector<std::string> input_paths;
  std::optional<rj_code_target_id_t> target;
  uint32_t code_object_index = 0;
  bool code_object_index_set = false;
  std::optional<uint64_t> kernel_entry;
  std::optional<uint32_t> wave_size;
  size_t max_diagnostics = 32;
  bool max_diagnostics_set = false;
  std::optional<std::string> diagnostics_jsonl_path;
  std::optional<std::string> counter_parity_jsonl_path;
  std::optional<std::string> extract_code_object_path;
  std::optional<std::string> extract_code_objects_dir;
  bool all_code_objects = false;
  bool list_code_objects = false;
  bool list_kernels = false;
  bool recursive = false;
  bool exhaustive = false;
  bool skip_unsupported = false;
  bool summary_only = false;
  bool stop_after_first_diagnostic = false;
  bool check_counter_parity = false;
  bool no_fail = false;
  bool show_help = false;
  ProgressMode progress = ProgressMode::Auto;
  uint32_t jobs = 1;
  uint32_t slowest_kernels = 0;
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
  uint32_t ignored = 0;
  uint32_t analysis_errors = 0;
  uint32_t code_objects_discovered = 0;
  uint32_t code_objects = 0;
  size_t kernels_discovered = 0;
  size_t kernels_analyzed = 0;
  size_t instructions_analyzed = 0;
  size_t memory_events_tracked = 0;
  size_t diagnostics = 0;
  bool diagnostics_lower_bound = false;
  size_t counter_parity_fields = 0;
  size_t counter_parity_exact = 0;
  size_t counter_underaccounting = 0;
  size_t counter_unmodeled_waits = 0;
  size_t counter_parity_indeterminate_groups = 0;
  bool hazards = false;
  struct SlowKernel {
    std::chrono::nanoseconds elapsed{};
    std::string input_path;
    std::string target;
    uint32_t code_object_index = 0;
    std::string kernel_name;
    uint64_t kernel_entry = 0;
  };
  std::vector<SlowKernel> slowest_kernels;
};

struct ProgressTotals {
  size_t code_objects = 0;
  size_t kernels = 0;
};

struct CodeObjectAnalysis {
  WaitcheckReport report;
  std::string error;
  std::vector<ScanTotals::SlowKernel> slow_kernels;
};

class ProgressDisplay {
public:
  explicit ProgressDisplay(ProgressMode mode)
      : enabled_(mode == ProgressMode::Always ||
                 (mode == ProgressMode::Auto && isatty(STDERR_FILENO) != 0)) {}

  [[nodiscard]] bool enabled() const { return enabled_; }

  void begin_discovery() {
    if (!enabled_)
      return;
    const std::lock_guard lock(mutex_);
    std::cerr << "\rrj_waitcheck: discovering kernels...\033[K" << std::flush;
    line_active_ = true;
  }

  void set_totals(ProgressTotals totals) {
    const std::lock_guard lock(mutex_);
    totals_ = totals;
    render(true);
  }

  void begin_code_object(const std::string &input_path, std::string_view target,
                         uint32_t code_object_index) {
    if (!enabled_)
      return;
    const std::lock_guard lock(mutex_);
    const std::filesystem::path path(input_path);
    current_ = path.filename().string();
    if (current_.empty())
      current_ = input_path;
    current_ += ":" + std::string(target) + "[" + std::to_string(code_object_index) + "]";
    ++active_;
    render(false);
  }

  void kernel_analyzed() {
    const std::lock_guard lock(mutex_);
    ++kernels_processed_;
    render(false);
  }

  void analysis_processed(bool code_object_completed) {
    const std::lock_guard lock(mutex_);
    if (active_ != 0)
      --active_;
    if (code_object_completed)
      ++code_objects_processed_;
    render(code_object_completed);
  }

  void before_message() {
    if (!enabled_)
      return;
    const std::lock_guard lock(mutex_);
    before_message_locked();
  }

  void finish() {
    if (!enabled_)
      return;
    const std::lock_guard lock(mutex_);
    if (!line_active_)
      render(true);
    before_message_locked();
  }

private:
  void before_message_locked() {
    if (!line_active_)
      return;
    std::cerr << "\n";
    line_active_ = false;
  }

  void render(bool force) {
    if (!enabled_)
      return;
    const size_t total_work = totals_.kernels + totals_.code_objects;
    const size_t completed_work = kernels_processed_ + code_objects_processed_;
    const size_t percent =
        total_work == 0 ? 100 : std::min<size_t>(100, completed_work * 100 / total_work);
    if (!force && last_percent_ && *last_percent_ == percent)
      return;
    last_percent_ = percent;
    std::cerr << "\rrj_waitcheck: " << std::setw(3) << percent << "% kernels " << kernels_processed_
              << "/" << totals_.kernels << " code-objects " << code_objects_processed_ << "/"
              << totals_.code_objects << " active=" << active_;
    if (!current_.empty())
      std::cerr << " " << current_;
    std::cerr << "\033[K" << std::flush;
    line_active_ = true;
  }

  bool enabled_ = false;
  bool line_active_ = false;
  ProgressTotals totals_;
  size_t code_objects_processed_ = 0;
  size_t kernels_processed_ = 0;
  size_t active_ = 0;
  std::optional<size_t> last_percent_;
  std::string current_;
  std::mutex mutex_;
};

template <typename Function> void parallel_for(size_t count, uint32_t jobs, Function &&function) {
  if (count == 0)
    return;
  const size_t worker_count = std::min<size_t>(count, jobs);
  if (worker_count == 1) {
    for (size_t index = 0; index < count; ++index)
      function(index);
    return;
  }

  std::atomic<size_t> next_index = 0;
  std::exception_ptr first_exception;
  std::mutex exception_mutex;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&] {
      try {
        while (true) {
          const size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
          if (index >= count)
            return;
          function(index);
        }
      } catch (...) {
        const std::lock_guard lock(exception_mutex);
        if (!first_exception)
          first_exception = std::current_exception();
        next_index.store(count, std::memory_order_relaxed);
      }
    });
  }
  for (std::thread &worker : workers)
    worker.join();
  if (first_exception)
    std::rethrow_exception(first_exception);
}

void print_supported_targets(std::ostream &os) {
  for (size_t i = 0; i < kSupportedTargets.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << waitcheck_target_name(kSupportedTargets[i]);
  }
}

[[nodiscard]] std::string_view access_name(WaitcheckAccessKind access) {
  switch (access) {
  case WaitcheckAccessKind::Use:
    return "use";
  case WaitcheckAccessKind::Def:
    return "def";
  case WaitcheckAccessKind::MemoryOrder:
    return "memory-order";
  case WaitcheckAccessKind::ProgramEnd:
    return "program-end";
  case WaitcheckAccessKind::ControlTransfer:
    return "control-transfer";
  }
  return "unknown";
}

[[nodiscard]] std::string_view register_class_name(RegClass reg_class) {
  switch (reg_class) {
  case RegClass::SGPR:
    return "sgpr";
  case RegClass::VGPR:
    return "vgpr";
  case RegClass::ACC_VGPR:
    return "acc-vgpr";
  case RegClass::EXEC:
    return "exec";
  case RegClass::VCC:
    return "vcc";
  case RegClass::SCC:
    return "scc";
  case RegClass::M0:
    return "m0";
  case RegClass::FLAT_SCRATCH:
    return "flat-scratch";
  case RegClass::TTMP:
    return "ttmp";
  case RegClass::PC:
    return "pc";
  }
  return "unknown";
}

[[nodiscard]] std::string_view instruction_mnemonic(std::string_view disassembly) {
  return disassembly.substr(0, disassembly.find_first_of(" \t"));
}

[[nodiscard]] std::string
counter_parity_catalog_key(rj_code_target_id_t target,
                           const WaitcheckCounterUnderaccountingDiagnostic &diagnostic) {
  std::ostringstream os;
  os << waitcheck_target_name(target) << "/"
     << (diagnostic.has_required_dependency ? "modeled" : "unmodeled") << "/"
     << wait_counter_name(diagnostic.counter) << "/";
  if (diagnostic.has_required_dependency) {
    os << access_name(diagnostic.access) << "/" << register_class_name(diagnostic.reg.cls) << "/"
       << instruction_mnemonic(diagnostic.producer_instruction) << "/";
  } else {
    os << "none/none/none/";
  }
  os << instruction_mnemonic(diagnostic.consumer_instruction);
  return os.str();
}

void write_json_string(std::ostream &os, std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  os << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\b':
      os << "\\b";
      break;
    case '\f':
      os << "\\f";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      if (ch < 0x20) {
        os << "\\u00" << hex[ch >> 4u] << hex[ch & 0xfu];
      } else {
        os << static_cast<char>(ch);
      }
      break;
    }
  }
  os << '"';
}

[[nodiscard]] std::string hex_value(uint64_t value) {
  std::ostringstream os;
  os << "0x" << std::hex << value;
  return os.str();
}

[[nodiscard]] std::string shell_quote(std::string_view value) {
  std::string result = "'";
  for (char ch : value) {
    if (ch == '\'')
      result += "'\\''";
    else
      result += ch;
  }
  result += '\'';
  return result;
}

template <typename Diagnostic>
[[nodiscard]] const WaitcheckKernelInfo *
matching_kernel(std::span<const WaitcheckKernelInfo> kernels, const Diagnostic &diagnostic) {
  if (!diagnostic.has_kernel)
    return kernels.size() == 1 ? &kernels.front() : nullptr;
  const auto match = std::ranges::find_if(kernels, [&](const WaitcheckKernelInfo &kernel) {
    return kernel.entry_offset == diagnostic.kernel_entry_offset &&
           kernel.name == diagnostic.kernel_name;
  });
  return match == kernels.end() ? nullptr : &*match;
}

class DiagnosticJsonlWriter {
public:
  DiagnosticJsonlWriter(std::ostream &output, std::string tool_path,
                        std::optional<uint32_t> wave_size)
      : output_(output), tool_path_(std::move(tool_path)), wave_size_(wave_size) {}

  void write(std::string_view input_path, rj_code_target_id_t target, uint32_t code_object_index,
             std::span<const WaitcheckKernelInfo> kernels,
             std::span<const WaitcheckDiagnostic> diagnostics) {
    if (diagnostics.empty())
      return;
    const std::lock_guard lock(mutex_);
    for (const WaitcheckDiagnostic &diagnostic : diagnostics) {
      const WaitcheckKernelInfo *kernel = matching_kernel(kernels, diagnostic);
      const uint32_t wave_size = kernel ? kernel->wavefront_size
                                        : wave_size_.value_or(waitcheck_default_wavefront_size(
                                              waitcheck_arch_for_target(target)));
      output_ << "{\"schema\":\"rj-waitcheck-diagnostic-v1\",\"input\":";
      write_json_string(output_, input_path);
      output_ << ",\"target\":";
      write_json_string(output_, waitcheck_target_name(target));
      output_ << ",\"code_object_index\":" << code_object_index << ",\"kernel_name\":";
      if (kernel)
        write_json_string(output_, kernel->name);
      else
        output_ << "null";
      output_ << ",\"kernel_entry\":";
      if (kernel)
        output_ << kernel->entry_offset;
      else
        output_ << "null";
      output_ << ",\"kernel_entry_hex\":";
      if (kernel)
        write_json_string(output_, hex_value(kernel->entry_offset));
      else
        output_ << "null";
      output_ << ",\"wavefront_size\":";
      output_ << wave_size;
      output_ << ",\"counter\":";
      write_json_string(output_, wait_counter_name(diagnostic.counter));
      output_ << ",\"access\":";
      write_json_string(output_, access_name(diagnostic.access));
      output_ << ",\"register\":{\"class\":";
      write_json_string(output_, register_class_name(diagnostic.reg.cls));
      output_ << ",\"index\":" << diagnostic.reg.index
              << ",\"width\":" << static_cast<uint32_t>(diagnostic.reg.width)
              << "},\"required_count\":" << diagnostic.required_count << ",\"section\":";
      write_json_string(output_, diagnostic.section_name);
      output_ << ",\"section_offset\":" << diagnostic.section_offset << ",\"section_offset_hex\":";
      write_json_string(output_, hex_value(diagnostic.section_offset));
      output_ << ",\"file_offset\":" << diagnostic.file_offset
              << ",\"producer_section_offset\":" << diagnostic.producer_section_offset
              << ",\"producer_section_offset_hex\":";
      write_json_string(output_, hex_value(diagnostic.producer_section_offset));
      output_ << ",\"producer_file_offset\":" << diagnostic.producer_file_offset
              << ",\"instruction\":";
      write_json_string(output_, diagnostic.instruction);
      output_ << ",\"producer_instruction\":";
      write_json_string(output_, diagnostic.producer_instruction);
      output_ << ",\"message\":";
      write_json_string(output_, diagnostic.message);

      const std::array<std::string, 7> fixed_repro_argv = {
          tool_path_,
          "--target",
          std::string(waitcheck_target_name(target)),
          "--code-object-index",
          std::to_string(code_object_index),
          "--no-fail",
          "--max-diagnostics"};
      output_ << ",\"repro_argv\":[";
      bool first = true;
      for (const std::string &argument : fixed_repro_argv) {
        if (!first)
          output_ << ',';
        first = false;
        write_json_string(output_, argument);
      }
      output_ << ',';
      write_json_string(output_, "4294967295");
      if (kernel) {
        output_ << ',';
        write_json_string(output_, "--kernel-entry");
        output_ << ',';
        write_json_string(output_, hex_value(kernel->entry_offset));
      }
      output_ << ",\"--wave-size\",";
      write_json_string(output_, std::to_string(wave_size));
      output_ << ',';
      write_json_string(output_, "--");
      output_ << ',';
      write_json_string(output_, input_path);
      output_ << "],\"repro_command\":";
      std::string repro_command;
      for (const std::string &argument : fixed_repro_argv) {
        if (!repro_command.empty())
          repro_command += ' ';
        repro_command += shell_quote(argument);
      }
      repro_command += " '4294967295'";
      if (kernel) {
        repro_command += " '--kernel-entry' ";
        repro_command += shell_quote(hex_value(kernel->entry_offset));
      }
      repro_command += " '--wave-size' " + shell_quote(std::to_string(wave_size));
      repro_command += " -- " + shell_quote(input_path);
      write_json_string(output_, repro_command);
      output_ << "}\n";
    }
    output_.flush();
  }

private:
  std::ostream &output_;
  std::string tool_path_;
  std::optional<uint32_t> wave_size_;
  std::mutex mutex_;
};

class CounterParityJsonlWriter {
public:
  CounterParityJsonlWriter(std::ostream &output, std::string tool_path,
                           std::optional<uint32_t> wave_size)
      : output_(output), tool_path_(std::move(tool_path)), wave_size_(wave_size) {}

  void write(std::string_view input_path, rj_code_target_id_t target, uint32_t code_object_index,
             std::span<const WaitcheckKernelInfo> kernels,
             std::span<const WaitcheckCounterUnderaccountingDiagnostic> diagnostics) {
    if (diagnostics.empty())
      return;
    const std::lock_guard lock(mutex_);
    for (const auto &diagnostic : diagnostics) {
      const WaitcheckKernelInfo *kernel = matching_kernel(kernels, diagnostic);
      const uint32_t wave_size = kernel ? kernel->wavefront_size
                                        : wave_size_.value_or(waitcheck_default_wavefront_size(
                                              waitcheck_arch_for_target(target)));
      output_ << "{\"schema\":\"rj-waitcheck-counter-parity-v1\",\"kind\":";
      write_json_string(output_, diagnostic.has_required_dependency
                                     ? "modeled-counter-underaccounting"
                                     : "unmodeled-emitted-wait");
      output_ << ",\"catalog_key\":";
      write_json_string(output_, counter_parity_catalog_key(target, diagnostic));
      output_ << ",\"root_cause_status\":\"untriaged\"";
      output_ << ",\"input\":";
      write_json_string(output_, input_path);
      output_ << ",\"target\":";
      write_json_string(output_, waitcheck_target_name(target));
      output_ << ",\"code_object_index\":" << code_object_index << ",\"kernel_name\":";
      if (diagnostic.has_kernel)
        write_json_string(output_, diagnostic.kernel_name);
      else if (kernel)
        write_json_string(output_, kernel->name);
      else
        output_ << "null";
      output_ << ",\"kernel_entry\":";
      if (diagnostic.has_kernel)
        output_ << diagnostic.kernel_entry_offset;
      else if (kernel)
        output_ << kernel->entry_offset;
      else
        output_ << "null";
      output_ << ",\"kernel_entry_hex\":";
      if (diagnostic.has_kernel)
        write_json_string(output_, hex_value(diagnostic.kernel_entry_offset));
      else if (kernel)
        write_json_string(output_, hex_value(kernel->entry_offset));
      else
        output_ << "null";
      output_ << ",\"wavefront_size\":";
      output_ << wave_size;
      output_ << ",\"counter\":";
      write_json_string(output_, wait_counter_name(diagnostic.counter));
      output_ << ",\"emitted_count\":" << diagnostic.emitted_count
              << ",\"required_count\":" << diagnostic.required_count
              << ",\"has_required_dependency\":"
              << (diagnostic.has_required_dependency ? "true" : "false") << ",\"access\":";
      if (diagnostic.has_required_dependency)
        write_json_string(output_, access_name(diagnostic.access));
      else
        output_ << "null";
      output_ << ",\"register\":";
      if (diagnostic.has_required_dependency) {
        output_ << "{\"class\":";
        write_json_string(output_, register_class_name(diagnostic.reg.cls));
        output_ << ",\"index\":" << diagnostic.reg.index
                << ",\"width\":" << static_cast<uint32_t>(diagnostic.reg.width) << "}";
      } else {
        output_ << "null";
      }
      output_ << ",\"section\":";
      write_json_string(output_, diagnostic.section_name);
      output_ << ",\"wait_section_offset\":" << diagnostic.wait_section_offset
              << ",\"wait_section_offset_hex\":";
      write_json_string(output_, hex_value(diagnostic.wait_section_offset));
      output_ << ",\"wait_file_offset\":" << diagnostic.wait_file_offset
              << ",\"wait_instruction\":";
      write_json_string(output_, diagnostic.wait_instruction);
      output_ << ",\"consumer_section_offset\":" << diagnostic.consumer_section_offset
              << ",\"consumer_section_offset_hex\":";
      write_json_string(output_, hex_value(diagnostic.consumer_section_offset));
      output_ << ",\"consumer_file_offset\":" << diagnostic.consumer_file_offset
              << ",\"consumer_class\":";
      write_json_string(output_, instruction_mnemonic(diagnostic.consumer_instruction));
      output_ << ",\"consumer_instruction\":";
      write_json_string(output_, diagnostic.consumer_instruction);
      output_ << ",\"producer_section_offset\":";
      if (diagnostic.has_required_dependency)
        output_ << diagnostic.producer_section_offset;
      else
        output_ << "null";
      output_ << ",\"producer_section_offset_hex\":";
      if (diagnostic.has_required_dependency)
        write_json_string(output_, hex_value(diagnostic.producer_section_offset));
      else
        output_ << "null";
      output_ << ",\"producer_file_offset\":";
      if (diagnostic.has_required_dependency)
        output_ << diagnostic.producer_file_offset;
      else
        output_ << "null";
      output_ << ",\"producer_instruction\":";
      if (diagnostic.has_required_dependency)
        write_json_string(output_, diagnostic.producer_instruction);
      else
        output_ << "null";
      output_ << ",\"producer_class\":";
      if (diagnostic.has_required_dependency)
        write_json_string(output_, instruction_mnemonic(diagnostic.producer_instruction));
      else
        output_ << "null";
      output_ << ",\"message\":";
      write_json_string(output_, diagnostic.message);

      const uint64_t kernel_entry = diagnostic.has_kernel ? diagnostic.kernel_entry_offset
                                                          : (kernel ? kernel->entry_offset : 0);
      // A shared physical wait must be reproduced with every entry, since a
      // single entry can falsely make its emitted count appear too strong.
      const bool has_kernel = kernels.size() == 1 && kernel != nullptr;
      const std::array<std::string, 8> fixed_repro_argv = {
          tool_path_,
          "--target",
          std::string(waitcheck_target_name(target)),
          "--code-object-index",
          std::to_string(code_object_index),
          "--check-counter-parity",
          "--no-fail",
          "--max-diagnostics"};
      output_ << ",\"repro_argv\":[";
      bool first = true;
      for (const std::string &argument : fixed_repro_argv) {
        if (!first)
          output_ << ',';
        first = false;
        write_json_string(output_, argument);
      }
      output_ << ',';
      write_json_string(output_, "4294967295");
      if (has_kernel) {
        output_ << ',';
        write_json_string(output_, "--kernel-entry");
        output_ << ',';
        write_json_string(output_, hex_value(kernel_entry));
      }
      output_ << ",\"--wave-size\",";
      write_json_string(output_, std::to_string(wave_size));
      output_ << ',';
      write_json_string(output_, "--");
      output_ << ',';
      write_json_string(output_, input_path);
      output_ << "],\"repro_command\":";
      std::string repro_command;
      for (const std::string &argument : fixed_repro_argv) {
        if (!repro_command.empty())
          repro_command += ' ';
        repro_command += shell_quote(argument);
      }
      repro_command += " '4294967295'";
      if (has_kernel) {
        repro_command += " '--kernel-entry' ";
        repro_command += shell_quote(hex_value(kernel_entry));
      }
      repro_command += " '--wave-size' " + shell_quote(std::to_string(wave_size));
      repro_command += " -- " + shell_quote(input_path);
      write_json_string(output_, repro_command);
      output_ << "}\n";
    }
    output_.flush();
  }

private:
  std::ostream &output_;
  std::string tool_path_;
  std::optional<uint32_t> wave_size_;
  std::mutex mutex_;
};

[[nodiscard]] std::optional<rj_code_target_id_t> parse_target(std::string_view value) {
  for (rj_code_target_id_t target : kSupportedTargets) {
    if (value == waitcheck_target_name(target))
      return target;
  }
  return std::nullopt;
}

// Waitcheck accepts hexadecimal offsets and counts as well as decimal values.
class CountFlag final : public cli::Flag {
public:
  CountFlag(cli::FlagInfo info, uint64_t minimum, uint64_t maximum, std::string label)
      : Flag(std::move(info)), minimum_(minimum), maximum_(maximum), label_(std::move(label)) {}
  std::optional<uint64_t> value() const { return value_; }
  std::string format_value() const override { return value_ ? std::to_string(*value_) : "unset"; }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    std::string_view digits = text;
    const bool hex = digits.starts_with("0x") || digits.starts_with("0X");
    if (hex)
      digits.remove_prefix(2);
    uint64_t value = 0;
    auto [end, ec] =
        std::from_chars(digits.data(), digits.data() + digits.size(), value, hex ? 16 : 10);
    if (ec != std::errc{} || end != digits.data() + digits.size() || value < minimum_ ||
        value > maximum_)
      return diagnostic.emit() << "invalid " << label_ << ": " << text << " (expected " << minimum_
                               << "-" << maximum_ << ")";
    value_ = value;
    return util::Result::success();
  }
  uint64_t minimum_;
  uint64_t maximum_;
  std::string label_;
  std::optional<uint64_t> value_;
};

// Absence is distinct from an explicitly empty target, path, or wave selector.
class NonemptyFlag final : public cli::Flag {
public:
  using Flag::Flag;
  const std::optional<std::string> &value() const { return value_; }
  std::string format_value() const override { return value_.value_or("unset"); }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    if (text.empty())
      return diagnostic.emit() << "value cannot be empty";
    value_ = text;
    return util::Result::success();
  }
  std::optional<std::string> value_;
};

[[nodiscard]] bool parse_args(int argc, char **argv, CliOptions &options) {
  cli::Command command{"rj_waitcheck"};
  command.set_positional_usage("INPUT...");
  auto report = [](std::string_view message) { std::cerr << message << "\n"; };
  util::DiagnosticEmitter diagnostic(report);
  const auto switch_flag = [](std::string name, std::string description,
                              std::vector<std::string> aliases = {}) {
    return std::make_shared<cli::BoolFlag>(cli::FlagInfo{
        std::move(name), std::move(aliases), std::move(description), cli::ValueMode::Switch});
  };
  const auto text_flag = [](std::string name, std::string placeholder, std::string description) {
    return std::make_shared<NonemptyFlag>(cli::FlagInfo{std::move(name),
                                                        {},
                                                        std::move(description),
                                                        cli::ValueMode::Required,
                                                        std::move(placeholder)});
  };
  const auto count = [](std::string name, std::string description, std::string label, uint64_t min,
                        uint64_t max, std::vector<std::string> aliases = {}) {
    return std::make_shared<CountFlag>(cli::FlagInfo{std::move(name), std::move(aliases),
                                                     std::move(description),
                                                     cli::ValueMode::Required, "N"},
                                       min, max, std::move(label));
  };
  auto target = text_flag("--target", "TARGET", "Select target from executable inputs");
  auto index =
      count("--code-object-index", "Code-object index for the selected target (default: 0)",
            "code-object index", 0, UINT32_MAX);
  auto entry = count("--kernel-entry", "Analyze kernels at this .text byte offset", "kernel entry",
                     0, UINT64_MAX);
  auto wave = text_flag("--wave-size", "32|64",
                        "With --kernel-entry, select a descriptor wave size or supply one for a "
                        "descriptorless entry");
  auto all = switch_flag("--all-code-objects", "Analyze all supported code objects");
  auto list_objects = switch_flag("--list-code-objects", "List supported code objects and exit");
  auto extract = text_flag("--extract-code-object", "PATH",
                           "Write the selected AMDGPU ELF from a supported container");
  auto extract_all = text_flag("--extract-code-objects", "DIR",
                               "Write every selected AMDGPU ELF after parsing one input once");
  auto list_kernels =
      switch_flag("--list-kernels", "List code objects and kernels as JSONL and exit");
  auto recursive = switch_flag("--recursive", "Expand directories recursively");
  auto exhaustive =
      switch_flag("--exhaustive", "Strict recursive target sweep with completeness totals");
  auto progress = switch_flag("--progress", "Show exhaustive kernel progress even without a TTY");
  auto no_progress = switch_flag("--no-progress", "Disable exhaustive kernel progress");
  auto jobs = count("--jobs", "Analyze up to N kernels concurrently within each all-code-object "
                                "input (default: 1)",
                    "job count", 1, kMaxJobs, {"-j"});
  auto slowest = count("--slowest-kernels", "Report the N slowest individually analyzed kernels",
                       "slow-kernel count", 0, kMaxSlowestKernels);
  auto skip = switch_flag("--skip-unsupported", "Skip unparsable or unsupported inputs");
  auto limit = count("--max-diagnostics", "Limit collected and printed diagnostics (default: 32)",
                     "max diagnostics", 0, UINT32_MAX);
  auto diagnostics =
      text_flag("--diagnostics-jsonl", "PATH", "Write all per-kernel diagnostics as JSONL");
  auto parity_output =
      text_flag("--counter-parity-jsonl", "PATH", "Write all counter-parity findings as JSONL");
  auto first = switch_flag("--stop-after-first-diagnostic",
                           "Stop each code object after the first observed hazard");
  auto parity =
      switch_flag("--check-counter-parity", "Compare emitted waits with modeled requirements");
  auto summary = switch_flag("--summary-only", "Print only final batch totals");
  auto no_fail = switch_flag("--no-fail", "Return success even when hazards are reported");
  auto help = switch_flag("--help", "Show this help", {"-h"});
  if (command
          .add_flags("Options",
                     {target,        index,       entry,        wave,      all,        list_objects,
                      extract,       extract_all, list_kernels, recursive, exhaustive, progress,
                      no_progress,   jobs,        slowest,      skip,      limit,      diagnostics,
                      parity_output, first,       parity,       summary,   no_fail,    help},
                     diagnostic)
          .failed())
    return false;

  // Preserve the compact -jN spelling and the order of progress overrides.
  std::vector<std::string_view> arguments;
  const cli::BoolFlag *last_progress = nullptr;
  bool positional_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (!positional_only && argument.starts_with("-j") && argument.size() > 2 &&
        argument[2] != '=') {
      arguments.push_back("-j");
      arguments.push_back(argument.substr(2));
      continue;
    }
    arguments.push_back(argument);
    if (positional_only)
      continue;
    if (argument == "--") {
      positional_only = true;
      continue;
    }
    const auto equals = argument.find('=');
    const auto flag = command.find_flag(argument.substr(0, equals));
    if (flag == progress || flag == no_progress)
      last_progress = flag == progress ? progress.get() : no_progress.get();
    if (flag == help && equals == std::string_view::npos)
      break;
    if (flag && flag->info().value_mode == cli::ValueMode::Required &&
        equals == std::string_view::npos && i + 1 < argc)
      arguments.push_back(argv[++i]);
  }
  auto parsed = cli::parse(command, arguments, diagnostic);
  if (parsed.failed())
    return false;
  options.input_paths = std::move(parsed.value().positional);
  if (help->value()) {
    options.show_help = true;
    std::cout << cli::format_help(command) << "\nSupported target names: ";
    print_supported_targets(std::cout);
    std::cout << ".\n";
    return true;
  }
  if (target->value()) {
    options.target = parse_target(*target->value());
    if (!options.target) {
      std::cerr << "invalid target: " << *target->value() << "\n";
      return false;
    }
  }
  if (wave->value()) {
    if (*wave->value() != "32" && *wave->value() != "64") {
      std::cerr << "--wave-size requires 32 or 64\n";
      return false;
    }
    options.wave_size = *wave->value() == "32" ? 32 : 64;
  }
  options.code_object_index_set = index->value().has_value();
  options.code_object_index = index->value().value_or(0);
  options.kernel_entry = entry->value();
  options.list_code_objects = list_objects->value();
  options.extract_code_object_path = extract->value();
  options.extract_code_objects_dir = extract_all->value();
  options.list_kernels = list_kernels->value();
  options.exhaustive = exhaustive->value();
  options.recursive = recursive->value() || options.exhaustive;
  options.all_code_objects = all->value() || options.exhaustive;
  if (last_progress)
    options.progress = (last_progress == progress.get()) == last_progress->value()
                           ? ProgressMode::Always
                           : ProgressMode::Never;
  options.jobs = jobs->value().value_or(options.jobs);
  options.slowest_kernels = slowest->value().value_or(0);
  options.skip_unsupported = skip->value();
  options.max_diagnostics_set = limit->value().has_value();
  options.max_diagnostics = limit->value().value_or(options.max_diagnostics);
  options.diagnostics_jsonl_path = diagnostics->value();
  options.counter_parity_jsonl_path = parity_output->value();
  options.stop_after_first_diagnostic = first->value();
  options.check_counter_parity = parity->value() || options.counter_parity_jsonl_path.has_value();
  options.summary_only = summary->value();
  options.no_fail = no_fail->value();

  if (options.wave_size && !options.kernel_entry) {
    std::cerr << "--wave-size requires --kernel-entry\n";
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
  if (options.exhaustive && !options.target) {
    std::cerr << "--exhaustive requires --target\n";
    return false;
  }
  if (options.exhaustive && options.skip_unsupported) {
    std::cerr << "--skip-unsupported cannot be used with --exhaustive\n";
    return false;
  }
  if (options.exhaustive && options.stop_after_first_diagnostic) {
    std::cerr << "--stop-after-first-diagnostic cannot be used with --exhaustive\n";
    return false;
  }
  if (options.exhaustive && options.list_code_objects) {
    std::cerr << "--list-code-objects cannot be used with --exhaustive\n";
    return false;
  }
  if (options.exhaustive && options.list_kernels) {
    std::cerr << "--list-kernels cannot be used with --exhaustive\n";
    return false;
  }
  if (options.list_code_objects && options.list_kernels) {
    std::cerr << "--list-code-objects cannot be used with --list-kernels\n";
    return false;
  }
  if (options.extract_code_object_path && options.extract_code_objects_dir) {
    std::cerr << "--extract-code-object and --extract-code-objects cannot be combined\n";
    return false;
  }
  if (options.extract_code_object_path &&
      (options.input_paths.size() != 1 || options.all_code_objects || options.recursive ||
       options.list_code_objects || options.list_kernels || options.kernel_entry ||
       options.wave_size || options.diagnostics_jsonl_path || options.counter_parity_jsonl_path ||
       options.check_counter_parity || options.skip_unsupported)) {
    std::cerr << "--extract-code-object requires one input and cannot be combined with analysis, "
                 "listing, or batch options\n";
    return false;
  }
  if (options.extract_code_objects_dir &&
      (options.input_paths.size() != 1 || options.all_code_objects || options.recursive ||
       options.list_code_objects || options.list_kernels || options.code_object_index_set ||
       options.kernel_entry || options.wave_size || options.diagnostics_jsonl_path ||
       options.counter_parity_jsonl_path || options.check_counter_parity)) {
    std::cerr << "--extract-code-objects requires one input and cannot be combined with "
                 "analysis, listing, or index options\n";
    return false;
  }
  if ((options.list_code_objects || options.list_kernels) &&
      (options.diagnostics_jsonl_path || options.counter_parity_jsonl_path)) {
    std::cerr << "JSONL analysis outputs cannot be used with listing modes\n";
    return false;
  }
  if (!options.exhaustive && options.progress == ProgressMode::Always) {
    std::cerr << "--progress requires --exhaustive\n";
    return false;
  }
  if (options.slowest_kernels != 0 && !options.all_code_objects) {
    std::cerr << "--slowest-kernels requires --all-code-objects or --exhaustive\n";
    return false;
  }
  if (options.diagnostics_jsonl_path && !options.all_code_objects) {
    std::cerr << "--diagnostics-jsonl requires --all-code-objects or --exhaustive\n";
    return false;
  }
  if (options.diagnostics_jsonl_path && options.max_diagnostics_set) {
    std::cerr << "--diagnostics-jsonl cannot be combined with --max-diagnostics\n";
    return false;
  }
  if (options.diagnostics_jsonl_path && options.stop_after_first_diagnostic) {
    std::cerr << "--diagnostics-jsonl cannot be combined with --stop-after-first-diagnostic\n";
    return false;
  }
  if (options.counter_parity_jsonl_path && !options.all_code_objects) {
    std::cerr << "--counter-parity-jsonl requires --all-code-objects or --exhaustive\n";
    return false;
  }
  if (options.counter_parity_jsonl_path && options.max_diagnostics_set) {
    std::cerr << "--counter-parity-jsonl cannot be combined with --max-diagnostics\n";
    return false;
  }
  if (options.counter_parity_jsonl_path && options.stop_after_first_diagnostic) {
    std::cerr << "--counter-parity-jsonl cannot be combined with "
                 "--stop-after-first-diagnostic\n";
    return false;
  }
  if (options.diagnostics_jsonl_path || options.counter_parity_jsonl_path)
    options.max_diagnostics = std::numeric_limits<size_t>::max();

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
    const auto traversal = options.exhaustive
                               ? std::filesystem::directory_options::none
                               : std::filesystem::directory_options::skip_permission_denied;
    std::filesystem::recursive_directory_iterator it(path, traversal, ec), end;
    while (!ec && it != end) {
      if (it->is_regular_file(ec))
        directory_files.push_back(it->path().string());
      if (ec && options.exhaustive) {
        error = it->path().string() + ": failed to inspect directory entry: " + ec.message();
        return false;
      }
      it.increment(ec);
    }
    if (ec) {
      error = input_path + ": failed to recurse input directory: " + ec.message();
      return false;
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

[[nodiscard]] ProgressTotals discover_progress_totals(const CliOptions &options) {
  ProgressTotals totals;
  for (const std::string &input_path : options.input_paths) {
    Executable executable(input_path);
    if (!executable.is_valid())
      continue;

    for (rj_code_target_id_t target : kSupportedTargets) {
      if (options.target && *options.target != target)
        continue;
      const uint32_t count = executable.num_code_objects(target);
      for (uint32_t index = 0; index < count; ++index) {
        const AmdGpuCodeObject *code_object = executable.code_object(target, index);
        if (!code_object)
          continue;
        ++totals.code_objects;
        totals.kernels += waitcheck_kernels(*code_object).size();
      }
    }
  }
  return totals;
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
  for (rj_code_target_id_t target : kSupportedTargets) {
    const uint32_t count = executable.num_code_objects(target);
    if (include_path)
      std::cout << input_path << ":";
    std::cout << waitcheck_target_name(target) << ": " << count << "\n";
  }
}

void list_kernels(const Executable &executable, std::string_view input_path,
                  const CliOptions &options) {
  for (rj_code_target_id_t target : kSupportedTargets) {
    if (options.target && *options.target != target)
      continue;
    const uint32_t count = executable.num_code_objects(target);
    for (uint32_t index = 0; index < count; ++index) {
      if (options.code_object_index_set && index != options.code_object_index)
        continue;
      const AmdGpuCodeObject *code_object = executable.code_object(target, index);
      if (!code_object)
        continue;
      const std::vector<WaitcheckKernelInfo> kernels = waitcheck_kernels_exact(*code_object);
      const std::vector<WaitcheckKernelInfo> groups = waitcheck_kernels(*code_object);
      std::cout << "{\"schema\":\"rj-waitcheck-corpus-inventory-v1\",\"kind\":";
      write_json_string(std::cout, "code-object");
      std::cout << ",\"input\":";
      write_json_string(std::cout, input_path);
      std::cout << ",\"target\":";
      write_json_string(std::cout, waitcheck_target_name(target));
      std::cout << ",\"code_object_index\":" << index << ",\"kernel_count\":" << kernels.size()
                << ",\"analysis_group_count\":" << groups.size() << "}\n";
      for (const WaitcheckKernelInfo &kernel : kernels) {
        const auto group = std::ranges::find_if(groups, [&](const WaitcheckKernelInfo &candidate) {
          return candidate.entry_offset == kernel.entry_offset &&
                 candidate.wavefront_size == kernel.wavefront_size;
        });
        if (group == groups.end())
          throw std::logic_error("exact kernel descriptor has no analysis group");
        std::cout << "{\"schema\":\"rj-waitcheck-corpus-inventory-v1\",\"kind\":";
        write_json_string(std::cout, "kernel");
        std::cout << ",\"input\":";
        write_json_string(std::cout, input_path);
        std::cout << ",\"target\":";
        write_json_string(std::cout, waitcheck_target_name(target));
        std::cout << ",\"code_object_index\":" << index << ",\"kernel_name\":";
        write_json_string(std::cout, kernel.name);
        std::cout << ",\"descriptor_vaddr\":" << kernel.descriptor_vaddr
                  << ",\"kernel_entry\":" << kernel.entry_offset << ",\"kernel_entry_hex\":";
        write_json_string(std::cout, hex_offset(kernel.entry_offset));
        std::cout << ",\"code_size\":" << kernel.code_size
                  << ",\"wavefront_size\":" << kernel.wavefront_size
                  << ",\"analysis_group_name\":";
        write_json_string(std::cout, group->name);
        std::cout << ",\"analysis_group_descriptor_vaddrs\":[" << group->descriptor_vaddr;
        for (uint64_t alias : group->alias_descriptor_vaddrs)
          std::cout << ',' << alias;
        std::cout << "]}\n";
      }
    }
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
      os << "failed to select " << waitcheck_target_name(*options.target) << " code object "
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
  for (rj_code_target_id_t target : kSupportedTargets) {
    if (selected.executable->num_code_objects(target) == 0)
      continue;
    target_with_objects = target;
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
    os << "failed to select " << waitcheck_target_name(*target_with_objects) << " code object "
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
    std::cout << input_path << ":" << waitcheck_target_name(target) << "[" << code_object_index
              << "]";
    if (diag.has_kernel)
      std::cout << ":kernel=" << diag.kernel_name << "@.text+"
                << hex_offset(diag.kernel_entry_offset);
    std::cout << ":" << diag.section_name << "+" << hex_offset(diag.section_offset) << ": "
              << diag.message << "\n";
    std::cout << "  producer " << diag.section_name << "+"
              << hex_offset(diag.producer_section_offset) << ": " << diag.producer_instruction
              << "\n";
    std::cout << "  consumer " << diag.section_name << "+" << hex_offset(diag.section_offset)
              << ": " << diag.instruction << "\n";
  }
  if (report.diagnostics_observed > limit) {
    std::cout << "omitted " << (report.stopped_early ? "at least " : "")
              << (report.diagnostics_observed - limit) << " diagnostic(s) after limit\n";
  } else if (report.stopped_early) {
    std::cout << "analysis stopped early; additional diagnostics may exist\n";
  }
}

void print_counter_underaccounting(const std::string &input_path, rj_code_target_id_t target,
                                   uint32_t code_object_index, const WaitcheckReport &report,
                                   size_t max_diagnostics) {
  const size_t limit = std::min(max_diagnostics, report.counter_underaccounting_diagnostics.size());
  for (size_t i = 0; i < limit; ++i) {
    const auto &diag = report.counter_underaccounting_diagnostics[i];
    std::cout << input_path << ":" << waitcheck_target_name(target) << "[" << code_object_index
              << "]";
    if (diag.has_kernel)
      std::cout << ":kernel=" << diag.kernel_name << "@.text+"
                << hex_offset(diag.kernel_entry_offset);
    std::cout << ":" << diag.section_name << "+" << hex_offset(diag.wait_section_offset)
              << ": counter under-accounting: " << diag.message << "\n";
    std::cout << "  wait " << diag.section_name << "+" << hex_offset(diag.wait_section_offset)
              << ": " << diag.wait_instruction << "\n";
    if (diag.has_required_dependency) {
      std::cout << "  producer " << diag.section_name << "+"
                << hex_offset(diag.producer_section_offset) << ": " << diag.producer_instruction
                << "\n";
    } else {
      std::cout << "  producer <no modeled dependency>\n";
    }
    std::cout << "  consumer " << diag.section_name << "+"
              << hex_offset(diag.consumer_section_offset) << ": " << diag.consumer_instruction
              << "\n";
  }
  if (report.counter_underaccounting_diagnostics.size() > limit) {
    std::cout << "omitted " << (report.counter_underaccounting_diagnostics.size() - limit)
              << " additional counter-underaccounting diagnostic(s)\n";
  } else if (report.counter_parity_diagnostics_truncated) {
    std::cout << "omitted additional counter-underaccounting diagnostic(s) after limit\n";
  }
}

void print_summary(const std::string &input_path, rj_code_target_id_t target,
                   uint32_t code_object_index, std::optional<uint64_t> kernel_entry,
                   const WaitcheckReport &report) {
  std::cout << "rj_waitcheck: " << input_path << ":" << waitcheck_target_name(target) << "["
            << code_object_index << "]";
  if (kernel_entry)
    std::cout << ":kernel=.text+" << hex_offset(*kernel_entry);
  std::cout << ": instructions=" << count_label(report.instructions_analyzed, report.stopped_early)
            << " memory-events=" << count_label(report.memory_events_tracked, report.stopped_early)
            << " diagnostics=" << count_label(report.diagnostics_observed, report.stopped_early);
  if (report.counter_parity_fields_checked != 0 || report.counter_underaccounting_observed != 0) {
    std::cout << " parity-fields=" << report.counter_parity_fields_checked
              << " parity-exact=" << report.counter_parity_exact << " counter-underaccounting="
              << (report.counter_underaccounting_observed - report.counter_unmodeled_wait_observed)
              << " unmodeled-waits=" << report.counter_unmodeled_wait_observed
              << " parity-indeterminate=" << report.counter_parity_indeterminate_groups;
  }
  std::cout << "\n";
}

void skip_input(const std::string &input_path, std::string_view reason, ScanTotals &totals,
                bool quiet = false) {
  ++totals.skipped;
  if (quiet)
    return;
  std::cout << "rj_waitcheck: " << input_path << ": skipped: " << reason << "\n";
}

void ignore_input(const std::string &input_path, std::string_view reason, ScanTotals &totals,
                  bool quiet = false) {
  ++totals.ignored;
  if (quiet)
    return;
  std::cout << "rj_waitcheck: " << input_path << ": ignored: " << reason << "\n";
}

void record_analysis_error(const std::string &input_path, std::string_view reason,
                           ScanTotals &totals, ProgressDisplay *progress) {
  ++totals.analysis_errors;
  if (progress)
    progress->before_message();
  std::cerr << "rj_waitcheck: " << input_path << ": analysis error: " << reason << "\n";
}

[[nodiscard]] CodeObjectAnalysis run_code_object_analysis(const CliOptions &options,
                                                          rj_code_target_id_t target,
                                                          uint32_t code_object_index,
                                                          const AmdGpuCodeObject &code_object,
                                                          ProgressDisplay *progress) {
  CodeObjectAnalysis result;
  const rj_code_arch_t arch = waitcheck_arch_for_target(target);
  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    result.report.supported = false;
    result.error =
        "target is not supported by waitcheck: " + std::string(waitcheck_target_name(target));
    return result;
  }

  WaitcheckOptions analysis_options;
  analysis_options.max_diagnostics = options.max_diagnostics;
  analysis_options.stop_after_first_diagnostic = options.stop_after_first_diagnostic;
  analysis_options.check_counter_parity = options.check_counter_parity;
  analysis_options.max_counter_parity_diagnostics = options.max_diagnostics;
  analysis_options.fallback_wavefront_size = options.wave_size;
  if (progress)
    analysis_options.kernel_analyzed_callback = [&] { progress->kernel_analyzed(); };
  try {
    if (options.wave_size && options.kernel_entry) {
      const auto kernels = waitcheck_kernels(code_object);
      if (kernels.empty()) {
        result.report =
            analyze_waitcnts_for_kernel(code_object, arch, *options.kernel_entry, analysis_options);
      } else {
        const auto kernel = std::ranges::find_if(kernels, [&](const WaitcheckKernelInfo &info) {
          return info.entry_offset == *options.kernel_entry &&
                 info.wavefront_size == *options.wave_size;
        });
        if (kernel == kernels.end()) {
          result.report.supported = false;
          result.report.analysis_error = "kernel entry with the requested wave size was not found";
        } else {
          result.report = analyze_waitcnts_for_kernel(code_object, arch, *kernel, analysis_options);
        }
      }
    } else
      result.report = options.kernel_entry
                          ? analyze_waitcnts_for_kernel(code_object, arch, *options.kernel_entry,
                                                        analysis_options)
                          : analyze_waitcnts(code_object, arch, analysis_options);
  } catch (const std::exception &ex) {
    result.report.supported = false;
    result.report.analysis_error = std::string("unexpected analysis failure: ") + ex.what();
  } catch (...) {
    result.report.supported = false;
    result.report.analysis_error = "unexpected non-standard analysis failure";
  }
  if (!result.report.supported || !result.report.complete) {
    result.error = "waitcheck analysis failed for " + std::string(waitcheck_target_name(target)) +
                   "[" + std::to_string(code_object_index) + "]";
    if (!result.report.analysis_error.empty())
      result.error += ": " + result.report.analysis_error;
  }
  return result;
}

[[nodiscard]] CodeObjectAnalysis
run_kernel_batch_analysis(const CliOptions &options, const std::string &input_path,
                          rj_code_target_id_t target, uint32_t code_object_index,
                          const AmdGpuCodeObject &code_object,
                          std::span<const WaitcheckKernelInfo> kernels, ProgressDisplay *progress) {
  CodeObjectAnalysis result;
  const rj_code_arch_t arch = waitcheck_arch_for_target(target);
  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    result.report.supported = false;
    result.error =
        "target is not supported by waitcheck: " + std::string(waitcheck_target_name(target));
    return result;
  }

  WaitcheckOptions analysis_options;
  analysis_options.max_diagnostics = options.max_diagnostics;
  analysis_options.stop_after_first_diagnostic = options.stop_after_first_diagnostic;
  analysis_options.check_counter_parity = options.check_counter_parity;
  analysis_options.max_counter_parity_diagnostics = options.max_diagnostics;
  analysis_options.fallback_wavefront_size = options.wave_size;
  if (progress)
    analysis_options.kernel_analyzed_callback = [&] { progress->kernel_analyzed(); };
  if (options.slowest_kernels != 0)
    analysis_options.kernel_timing_callback = [&](const WaitcheckKernelInfo &kernel,
                                                  std::chrono::nanoseconds elapsed) {
      result.slow_kernels.push_back({elapsed, input_path,
                                     std::string(waitcheck_target_name(target)), code_object_index,
                                     kernel.name, kernel.entry_offset});
    };

  try {
    result.report = analyze_waitcnts_for_kernels(code_object, arch, kernels, analysis_options);
  } catch (const std::exception &ex) {
    result.report.supported = false;
    result.report.analysis_error = std::string("unexpected analysis failure: ") + ex.what();
  } catch (...) {
    result.report.supported = false;
    result.report.analysis_error = "unexpected non-standard analysis failure";
  }
  if (!result.report.supported || !result.report.complete) {
    result.error = "waitcheck analysis failed for " + std::string(waitcheck_target_name(target)) +
                   "[" + std::to_string(code_object_index) + "]";
    if (!result.report.analysis_error.empty())
      result.error += ": " + result.report.analysis_error;
  }
  return result;
}

[[nodiscard]] bool merge_code_object_analysis(const CliOptions &options,
                                              const std::string &input_path,
                                              rj_code_target_id_t target,
                                              uint32_t code_object_index,
                                              const CodeObjectAnalysis &result, ScanTotals &totals,
                                              std::string &error) {
  const WaitcheckReport &report = result.report;
  ++totals.code_objects_discovered;
  totals.kernels_discovered += report.kernels_discovered;
  if (!report.supported || !report.complete) {
    error = result.error;
    return false;
  }

  ++totals.code_objects;
  totals.kernels_analyzed += report.kernels_analyzed;
  totals.instructions_analyzed += report.instructions_analyzed;
  totals.memory_events_tracked += report.memory_events_tracked;
  totals.diagnostics += report.diagnostics_observed;
  totals.diagnostics_lower_bound = totals.diagnostics_lower_bound || report.stopped_early;
  totals.counter_parity_fields += report.counter_parity_fields_checked;
  totals.counter_parity_exact += report.counter_parity_exact;
  totals.counter_underaccounting += report.counter_underaccounting_observed;
  totals.counter_unmodeled_waits += report.counter_unmodeled_wait_observed;
  totals.counter_parity_indeterminate_groups += report.counter_parity_indeterminate_groups;
  totals.hazards = totals.hazards || report.diagnostics_observed != 0;

  if (!options.summary_only) {
    print_summary(input_path, target, code_object_index, options.kernel_entry, report);
    if (!options.diagnostics_jsonl_path)
      print_diagnostics(input_path, target, code_object_index, report, options.max_diagnostics);
    if (options.check_counter_parity && !options.counter_parity_jsonl_path) {
      print_counter_underaccounting(input_path, target, code_object_index, report,
                                    options.max_diagnostics);
    }
  }
  return true;
}

[[nodiscard]] bool analyze_code_object(const CliOptions &options, const std::string &input_path,
                                       rj_code_target_id_t target, uint32_t code_object_index,
                                       const AmdGpuCodeObject &code_object, ScanTotals &totals,
                                       std::string &error, ProgressDisplay *progress = nullptr) {
  const CodeObjectAnalysis result =
      run_code_object_analysis(options, target, code_object_index, code_object, progress);
  return merge_code_object_analysis(options, input_path, target, code_object_index, result, totals,
                                    error);
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

void merge_kernel_analysis(CodeObjectAnalysis &code_object_result,
                           const CodeObjectAnalysis &kernel_result, size_t max_diagnostics) {
  WaitcheckReport &combined = code_object_result.report;
  const WaitcheckReport &kernel = kernel_result.report;
  if (combined.arch == ROCJITSU_CODE_ARCH_INVALID)
    combined.arch = kernel.arch;
  combined.instructions_analyzed += kernel.instructions_analyzed;
  combined.memory_events_tracked += kernel.memory_events_tracked;
  combined.kernels_discovered += kernel.kernels_discovered;
  combined.kernels_analyzed += kernel.kernels_analyzed;
  combined.diagnostics_observed += kernel.diagnostics_observed;
  combined.diagnostics_truncated = combined.diagnostics_truncated || kernel.diagnostics_truncated;
  combined.stopped_early = combined.stopped_early || kernel.stopped_early;
  combined.counter_parity_wait_groups += kernel.counter_parity_wait_groups;
  combined.counter_parity_fields_checked += kernel.counter_parity_fields_checked;
  combined.counter_parity_exact += kernel.counter_parity_exact;
  combined.counter_underaccounting_observed += kernel.counter_underaccounting_observed;
  combined.counter_unmodeled_wait_observed += kernel.counter_unmodeled_wait_observed;
  combined.counter_parity_indeterminate_groups += kernel.counter_parity_indeterminate_groups;
  combined.counter_parity_diagnostics_truncated =
      combined.counter_parity_diagnostics_truncated || kernel.counter_parity_diagnostics_truncated;

  for (const WaitcheckDiagnostic &diagnostic : kernel.diagnostics) {
    if (combined.diagnostics.size() >= max_diagnostics) {
      combined.diagnostics_truncated = true;
      break;
    }
    combined.diagnostics.push_back(diagnostic);
  }
  for (const auto &diagnostic : kernel.counter_underaccounting_diagnostics) {
    if (combined.counter_underaccounting_diagnostics.size() >= max_diagnostics) {
      combined.counter_parity_diagnostics_truncated = true;
      break;
    }
    combined.counter_underaccounting_diagnostics.push_back(diagnostic);
  }

  combined.complete &= kernel.complete;
  if ((!kernel.supported || !kernel.complete) && code_object_result.error.empty()) {
    combined.supported &= kernel.supported;
    combined.analysis_error = kernel.analysis_error;
    code_object_result.error = kernel_result.error;
  }
}

[[nodiscard]] bool slower_kernel_less(const ScanTotals::SlowKernel &lhs,
                                      const ScanTotals::SlowKernel &rhs) {
  if (lhs.elapsed != rhs.elapsed)
    return lhs.elapsed > rhs.elapsed;
  return std::tie(lhs.input_path, lhs.target, lhs.code_object_index, lhs.kernel_entry,
                  lhs.kernel_name) < std::tie(rhs.input_path, rhs.target, rhs.code_object_index,
                                              rhs.kernel_entry, rhs.kernel_name);
}

void record_slow_kernel(ScanTotals &totals, ScanTotals::SlowKernel kernel, size_t limit) {
  if (limit == 0)
    return;
  totals.slowest_kernels.push_back(std::move(kernel));
  std::ranges::sort(totals.slowest_kernels, slower_kernel_less);
  if (totals.slowest_kernels.size() > limit)
    totals.slowest_kernels.resize(limit);
}

void print_slowest_kernels(const ScanTotals &totals) {
  std::cout << "rj_waitcheck: slowest-kernels count=" << totals.slowest_kernels.size() << "\n";
  for (size_t i = 0; i < totals.slowest_kernels.size(); ++i) {
    const ScanTotals::SlowKernel &kernel = totals.slowest_kernels[i];
    const double elapsed_ms = std::chrono::duration<double, std::milli>(kernel.elapsed).count();
    std::cout << "rj_waitcheck: slowest-kernel rank=" << (i + 1) << " elapsed-ms=" << std::fixed
              << std::setprecision(3) << elapsed_ms << " input=" << std::quoted(kernel.input_path)
              << " target=" << kernel.target << " code-object=" << kernel.code_object_index
              << " entry=" << hex_offset(kernel.kernel_entry)
              << " name=" << std::quoted(kernel.kernel_name) << "\n";
  }
}

[[nodiscard]] bool scan_all_code_objects(const CliOptions &options, const std::string &input_path,
                                         ScanTotals &totals, std::string &error,
                                         DiagnosticJsonlWriter *diagnostic_writer,
                                         CounterParityJsonlWriter *counter_parity_writer,
                                         ProgressDisplay *progress = nullptr) {
  Executable executable(input_path);
  if (!executable.is_valid()) {
    if (options.exhaustive && (executable.read_failed() || executable.gpu_payload_recognized())) {
      record_analysis_error(input_path,
                            executable.read_failed() ? "could not read input"
                                                     : "recognized GPU input failed to parse",
                            totals, progress);
      return true;
    }
    if (options.exhaustive) {
      ignore_input(input_path, "not a supported executable or code object", totals,
                   options.summary_only);
      return true;
    }
    if (options.skip_unsupported) {
      skip_input(input_path, "failed to parse input executable or code object", totals,
                 options.summary_only);
      return true;
    }
    error = input_path + ": failed to parse input executable or code object";
    return false;
  }

  struct CodeObjectTask {
    rj_code_target_id_t target;
    uint32_t index;
    const AmdGpuCodeObject *code_object;
    std::vector<WaitcheckKernelInfo> kernels;
    std::vector<size_t> work_item_indices;
  };
  struct WorkItem {
    size_t code_object_task = 0;
    size_t first_kernel = 0;
    size_t kernel_count = 0;
  };
  std::vector<CodeObjectTask> code_object_tasks;
  std::vector<WorkItem> work_items;
  for (rj_code_target_id_t target : kSupportedTargets) {
    if (options.target && *options.target != target)
      continue;

    const uint32_t count = executable.num_code_objects(target);
    for (uint32_t index = 0; index < count; ++index) {
      const AmdGpuCodeObject *code_object = executable.code_object(target, index);
      if (!code_object) {
        std::ostringstream os;
        os << input_path << ": failed to select " << waitcheck_target_name(target)
           << " code object " << index;
        if (options.exhaustive) {
          record_analysis_error(input_path, os.str(), totals, progress);
          continue;
        }
        error = os.str();
        return false;
      }
      code_object_tasks.push_back(
          {target, index, code_object, waitcheck_kernels(*code_object), {}});
    }
  }

  if (code_object_tasks.empty()) {
    const std::string reason =
        options.target
            ? "no " + std::string(waitcheck_target_name(*options.target)) + " code objects found"
            : "no supported code objects found";
    if (options.exhaustive) {
      ignore_input(input_path, reason, totals, options.summary_only);
      return true;
    }
    if (options.skip_unsupported) {
      skip_input(input_path, reason, totals, options.summary_only);
      return true;
    }
    error = input_path + ": " + reason;
    return false;
  }

  // Interleave small bounded kernel batches from different code objects.
  // Generated libraries often place many pathological special-function
  // kernels next to one another, so a large adjacent batch can pin every
  // worker. Four kernels retain analyzer/decoder reuse while bounding that
  // imbalance. Lossless writers use one kernel per item for unambiguous
  // identity.
  const size_t kernel_batch_size =
      diagnostic_writer || counter_parity_writer ? 1 : kKernelBatchSize;
  const size_t allocator_trim_work_cadence =
      std::max<size_t>(1, kAllocatorTrimKernelCadence / kernel_batch_size);
  size_t max_batch_count = 0;
  for (size_t task_index = 0; task_index < code_object_tasks.size(); ++task_index) {
    CodeObjectTask &task = code_object_tasks[task_index];
    // Parity for one physical wait is shared across every entry that reaches
    // it. Keep all entries in one analyzer so a stronger requirement in one
    // entry can veto a weaker threshold proposed by another.
    if (task.kernels.empty() || options.stop_after_first_diagnostic ||
        (options.check_counter_parity && task.kernels.size() > 1)) {
      task.work_item_indices.push_back(work_items.size());
      const size_t kernel_count = options.stop_after_first_diagnostic ? 0 : task.kernels.size();
      work_items.push_back({task_index, 0, kernel_count});
    } else {
      max_batch_count = std::max(max_batch_count,
                                 (task.kernels.size() + kernel_batch_size - 1) / kernel_batch_size);
    }
  }
  for (size_t batch_index = 0; batch_index < max_batch_count; ++batch_index) {
    for (size_t task_index = 0; task_index < code_object_tasks.size(); ++task_index) {
      CodeObjectTask &task = code_object_tasks[task_index];
      const size_t first_kernel = batch_index * kernel_batch_size;
      if (options.stop_after_first_diagnostic || first_kernel >= task.kernels.size())
        continue;
      task.work_item_indices.push_back(work_items.size());
      work_items.push_back({task_index, first_kernel,
                            std::min(kernel_batch_size, task.kernels.size() - first_kernel)});
    }
  }

  std::vector<CodeObjectAnalysis> work_results(work_items.size());
  std::vector<std::atomic<size_t>> remaining_work(code_object_tasks.size());
  for (size_t task_index = 0; task_index < code_object_tasks.size(); ++task_index)
    remaining_work[task_index].store(code_object_tasks[task_index].work_item_indices.size(),
                                     std::memory_order_relaxed);
  try {
    parallel_for(work_items.size(), options.jobs, [&](size_t work_index) {
      const WorkItem &work = work_items[work_index];
      const CodeObjectTask &task = code_object_tasks[work.code_object_task];
      if (progress)
        progress->begin_code_object(input_path, waitcheck_target_name(task.target), task.index);
      if (work.kernel_count != 0) {
        work_results[work_index] = run_kernel_batch_analysis(
            options, input_path, task.target, task.index, *task.code_object,
            std::span<const WaitcheckKernelInfo>{task.kernels}.subspan(work.first_kernel,
                                                                       work.kernel_count),
            progress);
      } else {
        work_results[work_index] =
            run_code_object_analysis(options, task.target, task.index, *task.code_object, progress);
      }
      if (diagnostic_writer) {
        if (work_results[work_index].report.supported && work_results[work_index].report.complete) {
          diagnostic_writer->write(input_path, task.target, task.index,
                                   std::span<const WaitcheckKernelInfo>{task.kernels}.subspan(
                                       work.first_kernel, work.kernel_count),
                                   work_results[work_index].report.diagnostics);
        }
        // Lossless runs can materialize many diagnostics for one kernel. Drop
        // both their strings and the vector allocation before trimming the
        // worker arena; clear() alone retains one high-water capacity in every
        // work-result slot until the whole executable finishes.
        std::vector<WaitcheckDiagnostic>().swap(work_results[work_index].report.diagnostics);
      }
      if (counter_parity_writer) {
        if (work_results[work_index].report.supported && work_results[work_index].report.complete) {
          counter_parity_writer->write(
              input_path, task.target, task.index,
              std::span<const WaitcheckKernelInfo>{task.kernels}.subspan(work.first_kernel,
                                                                         work.kernel_count),
              work_results[work_index].report.counter_underaccounting_diagnostics);
        }
        std::vector<WaitcheckCounterUnderaccountingDiagnostic>().swap(
            work_results[work_index].report.counter_underaccounting_diagnostics);
      }
      // Batch CFGs and diagnostic payloads can be much larger than their
      // code-object images. Return their freed pages before the worker takes
      // another batch so glibc arenas do not retain a pathological high-water
      // mark per thread.
#if defined(__GLIBC__)
      // malloc_trim takes the process-wide allocator lock. Trimming after
      // every small work item serializes the worker pool. A bounded
      // kernel-count cadence still returns pathological CFG arenas during a
      // sweep without putting allocator maintenance on the hot path.
      if (work.kernel_count != 0 && work_index % allocator_trim_work_cadence == 0)
        malloc_trim(0);
#endif
      const bool code_object_completed =
          remaining_work[work.code_object_task].fetch_sub(1, std::memory_order_relaxed) == 1;
      if (progress)
        progress->analysis_processed(code_object_completed);
    });
  } catch (const std::exception &ex) {
    error = input_path + ": parallel analysis failed: " + ex.what();
    return false;
  }

  for (const CodeObjectTask &task : code_object_tasks) {
    CodeObjectAnalysis result;
    if (task.kernels.empty() || options.stop_after_first_diagnostic) {
      result = std::move(work_results[task.work_item_indices.front()]);
    } else {
      for (size_t work_index : task.work_item_indices) {
        merge_kernel_analysis(result, work_results[work_index], options.max_diagnostics);
        for (ScanTotals::SlowKernel &slow_kernel : work_results[work_index].slow_kernels)
          record_slow_kernel(totals, std::move(slow_kernel), options.slowest_kernels);
      }
    }
    if (diagnostic_writer && (!result.report.supported || !result.report.complete)) {
      // A completed kernel may have emitted a valid JSONL finding before a
      // different kernel made the containing object incomplete. Keep the
      // summary and hazard status aligned with those emitted records.
      for (size_t work_index : task.work_item_indices) {
        const WaitcheckReport &partial = work_results[work_index].report;
        if (!partial.supported || !partial.complete)
          continue;
        totals.diagnostics += partial.diagnostics_observed;
        totals.hazards = totals.hazards || partial.diagnostics_observed != 0;
        totals.diagnostics_lower_bound = true;
      }
    }
    if (merge_code_object_analysis(options, input_path, task.target, task.index, result, totals,
                                   error))
      continue;
    if (options.exhaustive) {
      record_analysis_error(input_path, error, totals, progress);
      continue;
    }
    if (options.skip_unsupported) {
      skip_input(input_path, error, totals, options.summary_only);
      continue;
    }
    error = input_path + ": " + error;
    return false;
  }

  return true;
}

} // namespace

[[nodiscard]] bool extracted_file_matches(const std::filesystem::path &path,
                                          const AmdGpuCodeObject &code_object) {
  std::error_code error;
  if (std::filesystem::file_size(path, error) != code_object.image_size() || error)
    return false;
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return false;
  std::array<char, 1u << 16> buffer{};
  const char *bytes = code_object.image_data();
  size_t offset = 0;
  while (offset < code_object.image_size()) {
    const size_t count = std::min(buffer.size(), code_object.image_size() - offset);
    input.read(buffer.data(), static_cast<std::streamsize>(count));
    if (input.gcount() != static_cast<std::streamsize>(count) ||
        !std::equal(buffer.begin(), buffer.begin() + count, bytes + offset))
      return false;
    offset += count;
  }
  return true;
}

[[nodiscard]] bool extraction_output_aliases_input(const std::filesystem::path &input_path,
                                                   const std::filesystem::path &output_path,
                                                   bool &aliases, std::string &error) {
  std::error_code path_error;
  const bool equivalent = std::filesystem::equivalent(input_path, output_path, path_error);
  path_error.clear();
  const auto canonical_input =
      std::filesystem::weakly_canonical(std::filesystem::absolute(input_path), path_error);
  if (path_error) {
    error = input_path.string() + ": " + path_error.message();
    return false;
  }
  const auto canonical_output =
      std::filesystem::weakly_canonical(std::filesystem::absolute(output_path), path_error);
  if (path_error) {
    error = output_path.string() + ": " + path_error.message();
    return false;
  }
  aliases = equivalent || canonical_input == canonical_output;
  return true;
}

[[nodiscard]] bool write_extracted_code_object(const std::filesystem::path &output_path,
                                               const AmdGpuCodeObject &code_object,
                                               std::string &error, bool replace_existing = true) {
  if (!replace_existing) {
    std::error_code exists_error;
    const auto status = std::filesystem::symlink_status(output_path, exists_error);
    if (exists_error == std::errc::no_such_file_or_directory)
      exists_error.clear();
    if (exists_error) {
      error = "failed to inspect extraction output: " + exists_error.message();
      return false;
    }
    if (std::filesystem::is_symlink(status)) {
      error = "extraction output is a symlink: " + output_path.string();
      return false;
    }
    if (std::filesystem::exists(status)) {
      if (extracted_file_matches(output_path, code_object))
        return true;
      error = "extraction output already exists with different bytes: " + output_path.string();
      return false;
    }
  }
  const std::filesystem::path directory =
      output_path.has_parent_path() ? output_path.parent_path() : std::filesystem::path{"."};
  const std::string pattern =
      (directory / ("." + output_path.filename().string() + ".tmp-XXXXXX")).string();
  std::vector<char> temporary_name(pattern.begin(), pattern.end());
  temporary_name.push_back('\0');
  const int descriptor = ::mkstemp(temporary_name.data());
  if (descriptor < 0) {
    error = "failed to create extraction output: " + output_path.string();
    return false;
  }
  const std::filesystem::path temporary_path{temporary_name.data()};
  const auto *bytes = reinterpret_cast<const uint8_t *>(code_object.image_data());
  const size_t size = code_object.image_size();
  size_t written = 0;
  while (written < size) {
    const size_t chunk_size = std::min<size_t>(size - written, 1u << 20);
    const ssize_t count = ::write(descriptor, bytes + written, chunk_size);
    if (count > 0) {
      written += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    break;
  }
  const bool closed = ::close(descriptor) == 0;
  if (written != size || !closed) {
    std::error_code ignored;
    std::filesystem::remove(temporary_path, ignored);
    error = "failed to write extraction output: " + output_path.string();
    return false;
  }
  if (!replace_existing) {
    if (::link(temporary_path.c_str(), output_path.c_str()) == 0) {
      std::error_code ignored;
      std::filesystem::remove(temporary_path, ignored);
      return true;
    }
    const int link_error = errno;
    std::error_code ignored;
    std::filesystem::remove(temporary_path, ignored);
    if (link_error == EEXIST && extracted_file_matches(output_path, code_object))
      return true;
    error = "failed to publish extraction output without replacing an existing file: " +
            output_path.string() + ": " + std::generic_category().message(link_error);
    return false;
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary_path, output_path, rename_error);
  if (rename_error) {
    std::error_code ignored;
    std::filesystem::remove(temporary_path, ignored);
    error = "failed to publish extraction output: " + rename_error.message();
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  CliOptions options;
  if (!parse_args(argc, argv, options))
    return kUsageError;

  if (options.show_help)
    return 0;

  if (options.input_paths.empty()) {
    std::cerr << "input path is required\n";
    return kUsageError;
  }
  std::string expansion_error;
  if (!expand_recursive_inputs(options, expansion_error)) {
    std::cerr << expansion_error << "\n";
    return kInputError;
  }

  if (options.extract_code_object_path) {
    const auto &input_path = options.input_paths.front();
    const std::filesystem::path output_path{*options.extract_code_object_path};
    bool aliases = false;
    std::string path_error;
    if (!extraction_output_aliases_input(input_path, output_path, aliases, path_error)) {
      std::cerr << path_error << "\n";
      return kInputError;
    }
    if (aliases) {
      std::cerr << "extraction output aliases input: " << output_path << "\n";
      return kInputError;
    }
    std::string selection_error;
    const SelectedCodeObject selected = select_code_object(options, input_path, selection_error);
    if (!selected.code_object) {
      std::cerr << input_path << ": " << selection_error << "\n";
      return kInputError;
    }
    std::string write_error;
    if (!write_extracted_code_object(output_path, *selected.code_object, write_error)) {
      std::cerr << write_error << "\n";
      return kInputError;
    }
    std::cout << "rj_waitcheck: extracted " << waitcheck_target_name(selected.target) << '['
              << selected.index << "] to " << output_path << "\n";
    return 0;
  }

  if (options.extract_code_objects_dir) {
    const std::filesystem::path input_path{options.input_paths.front()};
    const std::filesystem::path output_dir{*options.extract_code_objects_dir};
    Executable executable(input_path.string());
    if (!executable.is_valid()) {
      if (options.skip_unsupported)
        return 0;
      std::cerr << input_path << ": failed to parse input executable or code object\n";
      return kInputError;
    }
    size_t selected = 0;
    for (rj_code_target_id_t target : kSupportedTargets) {
      if (!options.target || *options.target == target)
        selected += executable.num_code_objects(target);
    }
    if (selected == 0) {
      if (options.skip_unsupported)
        return 0;
      std::cerr << input_path << ": no supported code objects found\n";
      return kInputError;
    }
    std::error_code directory_error;
    std::filesystem::create_directories(output_dir, directory_error);
    if (directory_error) {
      std::cerr << "failed to create extraction directory: " << directory_error.message() << "\n";
      return kInputError;
    }
    for (rj_code_target_id_t target : kSupportedTargets) {
      if (options.target && *options.target != target)
        continue;
      const uint32_t count = executable.num_code_objects(target);
      for (uint32_t index = 0; index < count; ++index) {
        const AmdGpuCodeObject *code_object = executable.code_object(target, index);
        if (!code_object) {
          std::cerr << input_path << ": failed to select " << waitcheck_target_name(target) << '['
                    << index << "]\n";
          return kInputError;
        }
        std::ostringstream name;
        name << waitcheck_target_name(target) << '-' << std::setfill('0') << std::setw(4) << index
             << ".hsaco";
        const std::filesystem::path output_path = output_dir / name.str();
        bool aliases = false;
        std::string error;
        if (!extraction_output_aliases_input(input_path, output_path, aliases, error)) {
          std::cerr << error << "\n";
          return kInputError;
        }
        if (aliases) {
          std::cerr << "extraction output aliases input: " << output_path << "\n";
          return kInputError;
        }
        if (!write_extracted_code_object(output_path, *code_object, error, false)) {
          std::cerr << error << "\n";
          return kInputError;
        }
        std::cout
            << "{\"schema\":\"rj-waitcheck-extraction-v1\",\"kind\":\"code-object\",\"target\":";
        write_json_string(std::cout, waitcheck_target_name(target));
        std::cout << ",\"code_object_index\":" << index << ",\"output\":";
        write_json_string(std::cout, output_path.string());
        std::cout << "}\n";
      }
    }
    return 0;
  }

  std::vector<std::filesystem::path> protected_paths(options.input_paths.begin(),
                                                     options.input_paths.end());
  for (const auto &report : {options.diagnostics_jsonl_path, options.counter_parity_jsonl_path}) {
    if (!report)
      continue;
    for (const auto &path : protected_paths) {
      std::error_code error;
      const bool equivalent = std::filesystem::equivalent(path, *report, error);
      error.clear();
      const auto existing =
          std::filesystem::weakly_canonical(std::filesystem::absolute(path), error);
      if (error) {
        std::cerr << path << ": " << error.message() << "\n";
        return kInputError;
      }
      const auto output =
          std::filesystem::weakly_canonical(std::filesystem::absolute(*report), error);
      if (error) {
        std::cerr << *report << ": " << error.message() << "\n";
        return kInputError;
      }
      if (equivalent || existing == output) {
        std::cerr << "JSONL output path aliases an input or another report: " << *report << "\n";
        return kInputError;
      }
    }
    protected_paths.emplace_back(*report);
  }
  std::ofstream diagnostic_output;
  std::unique_ptr<DiagnosticJsonlWriter> diagnostic_writer;
  if (options.diagnostics_jsonl_path) {
    diagnostic_output.open(*options.diagnostics_jsonl_path,
                           std::ios::binary | std::ios::out | std::ios::trunc);
    if (!diagnostic_output) {
      std::cerr << "failed to open diagnostics JSONL output: " << *options.diagnostics_jsonl_path
                << "\n";
      return kInputError;
    }
    diagnostic_writer = std::make_unique<DiagnosticJsonlWriter>(
        diagnostic_output, std::string(argv[0]), options.wave_size);
  }
  std::ofstream counter_parity_output;
  std::unique_ptr<CounterParityJsonlWriter> counter_parity_writer;
  if (options.counter_parity_jsonl_path) {
    counter_parity_output.open(*options.counter_parity_jsonl_path,
                               std::ios::binary | std::ios::out | std::ios::trunc);
    if (!counter_parity_output) {
      std::cerr << "failed to open counter-parity JSONL output: "
                << *options.counter_parity_jsonl_path << "\n";
      return kInputError;
    }
    counter_parity_writer = std::make_unique<CounterParityJsonlWriter>(
        counter_parity_output, std::string(argv[0]), options.wave_size);
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

  if (options.list_kernels) {
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
      list_kernels(executable, input_path, options);
    }
    return 0;
  }

  ProgressDisplay progress(options.progress);
  ProgressDisplay *progress_ptr = nullptr;
  if (options.exhaustive && progress.enabled()) {
    progress_ptr = &progress;
    progress.begin_discovery();
    progress.set_totals(discover_progress_totals(options));
  }

  const bool batch_mode = options.input_paths.size() > 1 || options.all_code_objects ||
                          options.skip_unsupported || options.recursive || options.summary_only;
  ScanTotals totals;
  for (const std::string &input_path : options.input_paths) {
    ++totals.inputs;
    std::string error;
    const bool ok =
        options.all_code_objects
            ? scan_all_code_objects(options, input_path, totals, error, diagnostic_writer.get(),
                                    counter_parity_writer.get(), progress_ptr)
            : scan_selected_code_object(options, input_path, totals, error);
    if (!ok) {
      if (progress_ptr)
        progress.before_message();
      std::cerr << error << "\n";
      return kInputError;
    }
  }
  if (progress_ptr)
    progress.finish();
  for (auto *output : {&diagnostic_output, &counter_parity_output}) {
    if (output->is_open()) {
      output->close();
      if (!*output) {
        std::cerr << "failed to write JSONL output\n";
        return kInputError;
      }
    }
  }

  if (options.exhaustive) {
    std::cout << "rj_waitcheck: exhaustive files=" << totals.inputs << " ignored=" << totals.ignored
              << " code-objects=" << totals.code_objects << "/" << totals.code_objects_discovered
              << " kernels=" << totals.kernels_analyzed << "/" << totals.kernels_discovered
              << " instructions=" << totals.instructions_analyzed
              << " memory-events=" << totals.memory_events_tracked
              << " diagnostics=" << count_label(totals.diagnostics, totals.diagnostics_lower_bound);
    if (options.check_counter_parity) {
      std::cout << " parity-fields=" << totals.counter_parity_fields
                << " parity-exact=" << totals.counter_parity_exact << " counter-underaccounting="
                << (totals.counter_underaccounting - totals.counter_unmodeled_waits)
                << " unmodeled-waits=" << totals.counter_unmodeled_waits
                << " parity-indeterminate=" << totals.counter_parity_indeterminate_groups;
    }
    std::cout << " analysis-errors=" << totals.analysis_errors << "\n";
  } else if (batch_mode) {
    std::cout << "rj_waitcheck: scanned inputs=" << totals.inputs << " skipped=" << totals.skipped
              << " code-objects=" << totals.code_objects
              << " diagnostics=" << count_label(totals.diagnostics, totals.diagnostics_lower_bound);
    if (options.check_counter_parity) {
      std::cout << " counter-underaccounting="
                << (totals.counter_underaccounting - totals.counter_unmodeled_waits)
                << " unmodeled-waits=" << totals.counter_unmodeled_waits;
    }
    std::cout << "\n";
  }
  if (options.slowest_kernels != 0)
    print_slowest_kernels(totals);

  if (options.exhaustive && (totals.analysis_errors != 0 || totals.code_objects_discovered == 0 ||
                             totals.code_objects != totals.code_objects_discovered ||
                             totals.kernels_analyzed != totals.kernels_discovered))
    return kInputError;

  if (totals.hazards && !options.no_fail)
    return kHazardDetected;
  return 0;
}
