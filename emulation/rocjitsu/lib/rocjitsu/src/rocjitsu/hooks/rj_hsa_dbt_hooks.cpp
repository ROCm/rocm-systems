// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbt_hooks.cpp
/// @brief HSA tools load-time DBT hook for translating AMDGPU code objects.
///
/// @details DBT guest mode is split across KFD discovery and HSA execution.
/// GuestKfd appends one synthetic guest GPU to KFD topology so ROCR discovers
/// the agent requested by the application, while the real host GPU remains
/// available for execution. ROCR loads this shared library through
/// `HSA_TOOLS_LIB` during `hsa_init()`. The hook selects the configured host
/// agent, presents guest-facing agent handles where applications enumerate or
/// query devices, rewrites execution-facing calls back to the host agent, and
/// invokes rocjitsu DBT before ROCR sees a memory-backed guest code object.
/// The MVP is deliberately strict: when translation is requested and fails, the
/// hook returns an HSA error instead of retrying the original reader, because
/// the original ELF may target a different GPU ISA.

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/hooks/sidecar_registry.h"
#include "rocjitsu/hooks/virtual_lds.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/arena_alloc.h"
#include "util/intrusive_list.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <execinfo.h>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <shared_mutex>
#include <signal.h>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

namespace {

using rocjitsu::AmdGpuCodeObject;
using rocjitsu::arch_for_elf_mach;
using rocjitsu::arch_lds_bytes;
using rocjitsu::BinaryTranslator;
using rocjitsu::BinaryTranslatorOptions;
using rocjitsu::DiagnosticKind;
using rocjitsu::DiagnosticSeverity;
using rocjitsu::EF_AMDGPU_MACH;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1100;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950;
using rocjitsu::Elf64_Ehdr;
using rocjitsu::Elf64_Shdr;
using rocjitsu::elf_mach_for_arch;
using rocjitsu::elf_mach_name;
using rocjitsu::ELFCLASS64;
using rocjitsu::EM_AMDGPU;
using rocjitsu::has_error_diagnostic;
using rocjitsu::has_skipped_kernel;
using rocjitsu::KernargExtensionLayout;
using rocjitsu::KernargExtensionMetadata;
using rocjitsu::KernargExtensionPayloadLayout;
using rocjitsu::KernargExtensionPayloadWrite;
using rocjitsu::kVirtualLdsFlagRuntimeStateBlock;
using rocjitsu::kVirtualLdsFlagWorkgroupIdX;
using rocjitsu::kVirtualLdsFlagWorkgroupIdY;
using rocjitsu::kVirtualLdsFlagWorkgroupIdZ;
using rocjitsu::kVirtualLdsMetadataSectionName;
using rocjitsu::kVirtualLdsRuntimeStatePayloadName;
using rocjitsu::make_kernarg_extension_layout;
using rocjitsu::parse_kernarg_extension_metadata;
using rocjitsu::parse_virtual_lds_metadata;
using rocjitsu::plan_virtual_lds_dispatch;
using rocjitsu::ProcessorRevision;
using rocjitsu::SHN_UNDEF;
using rocjitsu::SHT_NULL;
using rocjitsu::SHT_STRTAB;
using rocjitsu::TranslationDiagnostic;
using rocjitsu::VirtualLdsDispatchDecision;
using rocjitsu::VirtualLdsDispatchGeometry;
using rocjitsu::VirtualLdsDispatchKernelFacts;
using rocjitsu::VirtualLdsDispatchPacketFields;
using rocjitsu::VirtualLdsDispatchState;
using rocjitsu::write_kernarg_extension_wrapper;
using rocjitsu::hooks::completed_virtual_lds_dispatch;
using rocjitsu::hooks::empty_virtual_lds_buffers;
using rocjitsu::hooks::first_unsupportable_virtual_lds_kernel;
using rocjitsu::hooks::parse_virtual_lds_hook_metadata;
using rocjitsu::hooks::record_virtual_lds_completion_signal;
using rocjitsu::hooks::release_virtual_lds_buffers;
using rocjitsu::hooks::VirtualLdsDispatchAllocator;
using rocjitsu::hooks::VirtualLdsDispatchBuffers;
using rocjitsu::hooks::VirtualLdsRuntimeRegistry;

enum HookLogLevel : int {
  kLogDisabled = 0,
  kLogInfo = 1,
  kLogVerbose = 2,
  kLogDebug = 3,
};

std::atomic<int> g_log_level{kLogDisabled};
std::atomic<bool> g_signal_backtrace_enabled{false};
std::atomic<bool> g_signal_backtrace_installed{false};
struct sigaction g_previous_sigsegv {};
struct sigaction g_previous_sigabrt {};

/// @brief Parsed ISA target used by DBT and HSA agent matching.
struct TargetInfo {
  std::string_view name;
  rj_code_arch_t arch;
  uint32_t mach;
};

constexpr std::array<uint32_t, 5> kAcceptedConcreteTargetMachs = {
    EF_AMDGPU_MACH_AMDGCN_GFX942, EF_AMDGPU_MACH_AMDGCN_GFX950, EF_AMDGPU_MACH_AMDGCN_GFX1100,
    EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1201};

constexpr std::array<TargetInfo, 4> kArchAliases = {{
    {"cdna3", ROCJITSU_CODE_ARCH_CDNA3, elf_mach_for_arch(ROCJITSU_CODE_ARCH_CDNA3)},
    {"cdna4", ROCJITSU_CODE_ARCH_CDNA4, elf_mach_for_arch(ROCJITSU_CODE_ARCH_CDNA4)},
    {"rdna3", ROCJITSU_CODE_ARCH_RDNA3, elf_mach_for_arch(ROCJITSU_CODE_ARCH_RDNA3)},
    {"rdna4", ROCJITSU_CODE_ARCH_RDNA4, elf_mach_for_arch(ROCJITSU_CODE_ARCH_RDNA4)},
}};

/// @brief Return the byte offset immediately after an API-table field.
///
/// @details ROCR uses `ApiTableVersion::minor_id` as the HSA-tools table size.
/// Checking the end offset before reading a tail field lets the hook run against
/// older runtimes whose table stops before recently-added AMD extension entries.
template <typename Table, typename Field>
constexpr uint32_t api_table_field_end_offset(size_t field_offset, Field Table::*) {
  return static_cast<uint32_t>(field_offset + sizeof(Field));
}

/// @brief Return true when @p table is large enough to contain @p field.
template <typename Table, typename Field>
[[nodiscard]] bool api_table_has_field(const Table *table, size_t field_offset,
                                       Field Table::*field) {
  return table != nullptr &&
         table->version.minor_id >= api_table_field_end_offset(field_offset, field);
}

static_assert(api_table_field_end_offset(offsetof(AmdExtTable, hsa_amd_memory_async_batch_copy_fn),
                                         &AmdExtTable::hsa_amd_memory_async_batch_copy_fn) == 648);
static_assert(api_table_field_end_offset(offsetof(AmdExtTable, hsa_amd_agent_preload_fn),
                                         &AmdExtTable::hsa_amd_agent_preload_fn) == 656);

/// @brief Number of recent AQL packet IDs rescanned by the polling fallback.
///
/// @details HSA's write index can become visible before the packet body is
/// stable. Rescanning the tail is safe because packet rewriting is idempotent:
/// an already-rewritten slot is recognized by its rocjitsu-owned kernarg/state
/// pointers, and old slot allocations are retired behind completion signals
/// before a reused queue slot receives new buffers. Doorbell-triggered scans
/// still maintain the contiguous cursor used for the normal HSA signal path.
constexpr uint64_t kVirtualLdsScannerTailPackets = 512;

/// @brief AMD memory-pool enum values mirrored locally for internal allocation.
///
/// @details Keep these constants out of msgpack or YAML metadata paths: runtime
/// virtual-LDS state is carried by the rocjitsu ELF section and AQL rewrite.
/// @brief Runtime configuration consumed by the HSA tools hook.
struct HookConfig {
  TargetInfo target{};
  std::optional<TargetInfo> source_override;
  std::optional<TargetInfo> guest_target;
  uint32_t host_gpu_id = 0;
  int log_level = kLogDisabled;
  bool signal_backtrace = false;
  /// @brief Guest and host revisions from the DBT guest config. gfx1250 A0 and
  /// B0 share an ELF machine ID, so these select the same-target translation
  /// profile.
  ProcessorRevision guest_revision = ProcessorRevision::Unspecified;
  ProcessorRevision host_revision = ProcessorRevision::Unspecified;
};

/// @brief Parse a config ISA name or architecture alias into a DBT target.
[[nodiscard]] std::optional<TargetInfo> parse_target(std::string_view value) {
  for (uint32_t mach : kAcceptedConcreteTargetMachs) {
    std::string_view name = elf_mach_name(mach);
    if (value == name) {
      const rj_code_arch_t arch = arch_for_elf_mach(mach);
      if (arch != ROCJITSU_CODE_ARCH_INVALID)
        return TargetInfo{name, arch, mach};
    }
  }
  for (const TargetInfo &target : kArchAliases) {
    if (value == target.name)
      return target;
  }
  return std::nullopt;
}

/// @brief Map a config silicon revision to the DBT translator revision.
[[nodiscard]] ProcessorRevision
processor_revision_from_config(rocjitsu::config::DbtSiliconRevision revision) {
  switch (revision) {
  case rocjitsu::config::DbtSiliconRevision::Unspecified:
    return ProcessorRevision::Unspecified;
  case rocjitsu::config::DbtSiliconRevision::Gfx1250A0:
    return ProcessorRevision::Gfx1250A0;
  case rocjitsu::config::DbtSiliconRevision::Gfx1250B0:
    return ProcessorRevision::Gfx1250B0;
  }
  return ProcessorRevision::Unspecified;
}

/// @brief Clamp a user-provided hook log level to the supported range.
[[nodiscard]] int clamp_log_level(int value) {
  if (value < kLogDisabled)
    return kLogDisabled;
  if (value > kLogDebug)
    return kLogDebug;
  return value;
}

/// @brief Chain to the signal handler that was installed before rocjitsu.
void invoke_previous_signal_handler(int signo, siginfo_t *info, void *context) {
  const struct sigaction &previous = signo == SIGABRT ? g_previous_sigabrt : g_previous_sigsegv;
  if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
    previous.sa_sigaction(signo, info, context);
    return;
  }
  if (previous.sa_handler == SIG_IGN)
    return;
  if (previous.sa_handler != SIG_DFL && previous.sa_handler != nullptr) {
    previous.sa_handler(signo);
    return;
  }
  (void)::sigaction(signo, &previous, nullptr);
  (void)::raise(signo);
}

void prewarm_signal_backtrace() {
  void *frames[1];
  int count = ::backtrace(frames, 1);
  int fd = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
  if (fd >= 0) {
    ::backtrace_symbols_fd(frames, count, fd);
    ::close(fd);
  }
}

/// @brief Print a best-effort stack trace for fatal hook signals.
///
/// @details The handler uses prewarmed unwinder paths, but glibc backtrace APIs
/// are not async-signal-safe. Keep this diagnostic opt-in.
void signal_backtrace_handler(int signo, siginfo_t *info, void *context) {
  if (!g_signal_backtrace_enabled.exchange(false, std::memory_order_relaxed)) {
    invoke_previous_signal_handler(signo, info, context);
    return;
  }

  const char header[] = "\n[rocjitsu-hooks] signal backtrace\n";
  const ssize_t written = ::write(STDERR_FILENO, header, sizeof(header) - 1);
  (void)written;
  void *frames[128];
  int count = ::backtrace(frames, 128);
  ::backtrace_symbols_fd(frames, count, STDERR_FILENO);
  invoke_previous_signal_handler(signo, info, context);
}

/// @brief Install fatal-signal backtrace handling when enabled by config.
void maybe_install_signal_backtrace(bool enabled) {
  if (!enabled)
    return;

  prewarm_signal_backtrace();

  struct sigaction action {};
  action.sa_sigaction = signal_backtrace_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_SIGINFO;

  struct sigaction previous_sigsegv {};
  struct sigaction previous_sigabrt {};
  if (::sigaction(SIGSEGV, &action, &previous_sigsegv) != 0)
    return;
  if (::sigaction(SIGABRT, &action, &previous_sigabrt) != 0) {
    (void)::sigaction(SIGSEGV, &previous_sigsegv, nullptr);
    return;
  }

  g_previous_sigsegv = previous_sigsegv;
  g_previous_sigabrt = previous_sigabrt;
  g_signal_backtrace_installed.store(true, std::memory_order_release);
  g_signal_backtrace_enabled.store(true, std::memory_order_release);
}

/// @brief Restore fatal-signal handlers installed before rocjitsu.
void restore_signal_backtrace_handlers() {
  g_signal_backtrace_enabled.store(false, std::memory_order_release);
  if (!g_signal_backtrace_installed.exchange(false, std::memory_order_acq_rel))
    return;
  (void)::sigaction(SIGSEGV, &g_previous_sigsegv, nullptr);
  (void)::sigaction(SIGABRT, &g_previous_sigabrt, nullptr);
}

/// @brief Load and validate DBT hook configuration from the runtime config file.
[[nodiscard]] std::optional<HookConfig> parse_config() {
  std::optional<rocjitsu::config::DbtGuestConfig> dbt_guest;
  try {
    dbt_guest = rocjitsu::config::load_dbt_guest_config_from_runtime_config();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to load runtime config: %s\n", error.what());
    return std::nullopt;
  }
  if (!dbt_guest) {
    std::fprintf(stderr, "[rocjitsu-hooks] runtime config is required for the DBT HSA hook\n");
    return std::nullopt;
  }

  if (!dbt_guest->enabled) {
    std::fprintf(stderr, "[rocjitsu-hooks] runtime config does not enable dbt_guest mode\n");
    return std::nullopt;
  }

  auto target = parse_target(dbt_guest->host.isa);
  if (!target) {
    std::fprintf(stderr, "[rocjitsu-hooks] invalid dbt_guest.host_isa='%s'\n",
                 dbt_guest->host.isa.c_str());
    return std::nullopt;
  }
  auto guest = parse_target(dbt_guest->guest_isa);
  if (!guest) {
    std::fprintf(stderr, "[rocjitsu-hooks] invalid dbt_guest.guest_isa='%s'\n",
                 dbt_guest->guest_isa.c_str());
    return std::nullopt;
  }

  HookConfig config;
  config.target = *target;
  config.source_override = *guest;
  config.guest_target = *guest;
  config.host_gpu_id = dbt_guest->host.gpu_id;
  config.log_level = clamp_log_level(dbt_guest->log_level);
  config.signal_backtrace = dbt_guest->signal_backtrace;
  config.guest_revision = processor_revision_from_config(dbt_guest->guest_revision);
  config.host_revision = processor_revision_from_config(dbt_guest->host_revision);
  return config;
}

/// @brief Emit a hook log message through the rocjitsu logger.
void log_message(int required_level, const char *format, ...) {
  if (g_log_level.load(std::memory_order_relaxed) < required_level)
    return;

  std::array<char, 512> message{};
  va_list args;
  va_start(args, format);
  std::vsnprintf(message.data(), message.size(), format, args);
  va_end(args);

  util::Logger::dbt_hooks(message.data());
}

/// @brief Return true when public hook logging will emit @p required_level.
[[nodiscard]] bool log_level_enabled(int required_level) {
  return g_log_level.load(std::memory_order_relaxed) >= required_level;
}

/// @brief Return true when public debug logging will include virtual-LDS dispatch traces.
[[nodiscard]] bool trace_virtual_lds_dispatch_enabled() { return log_level_enabled(kLogDebug); }

/// @brief Emit a virtual-LDS dispatch trace line through public hook logging.
void trace_virtual_lds_dispatch(const char *format, ...) {
  if (!trace_virtual_lds_dispatch_enabled())
    return;

  std::array<char, 512> message{};
  va_list args;
  va_start(args, format);
  std::vsnprintf(message.data(), message.size(), format, args);
  va_end(args);
  util::Logger::dbt_hooks(message.data());
}

[[nodiscard]] bool trace_virtual_lds_kernarg_enabled() { return log_level_enabled(kLogDebug); }

[[nodiscard]] bool trace_virtual_lds_userargs_enabled() { return log_level_enabled(kLogDebug); }

[[nodiscard]] bool host_range_is_readable(uint64_t address, size_t size) {
  if (address == 0 || size == 0)
    return false;
  if (address > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(size - 1))
    return false;

  FILE *maps = std::fopen("/proc/self/maps", "r");
  if (maps == nullptr)
    return false;

  const uint64_t end = address + static_cast<uint64_t>(size);
  char line[512];
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long long start = 0;
    unsigned long long stop = 0;
    char perms[5] = {};
    if (std::sscanf(line, "%llx-%llx %4s", &start, &stop, perms) != 3)
      continue;
    if (perms[0] != 'r')
      continue;
    if (address >= start && end <= stop) {
      std::fclose(maps);
      return true;
    }
  }

  std::fclose(maps);
  return false;
}

void trace_virtual_lds_userargs_pointer(uint64_t packet_id, uint32_t offset, uint64_t pointer) {
  constexpr size_t kUserArgsTraceBytes = 128;
  if (!trace_virtual_lds_userargs_enabled())
    return;

  if (!host_range_is_readable(pointer, kUserArgsTraceBytes)) {
    std::fprintf(stderr,
                 "[rocjitsu-vlds-userargs] packet=%llu kernarg_off=%u ptr=0x%llx unreadable\n",
                 static_cast<unsigned long long>(packet_id), offset,
                 static_cast<unsigned long long>(pointer));
    return;
  }

  std::fprintf(stderr, "[rocjitsu-vlds-userargs] packet=%llu kernarg_off=%u ptr=0x%llx bytes=",
               static_cast<unsigned long long>(packet_id), offset,
               static_cast<unsigned long long>(pointer));
  const auto *bytes = reinterpret_cast<const uint8_t *>(pointer);
  for (size_t i = 0; i < kUserArgsTraceBytes; ++i)
    std::fprintf(stderr, "%02x", bytes[i]);
  std::fprintf(stderr, "\n");
}

void trace_virtual_lds_kernarg(uint64_t packet_id, const void *kernarg, size_t size) {
  if (!trace_virtual_lds_kernarg_enabled())
    return;

  constexpr size_t kMaxTraceBytes = 96;
  const size_t trace_bytes = std::min(size, kMaxTraceBytes);
  std::fprintf(stderr, "[rocjitsu-vlds-kernarg] packet=%llu size=%zu bytes=",
               static_cast<unsigned long long>(packet_id), size);
  const auto *bytes = static_cast<const uint8_t *>(kernarg);
  for (size_t i = 0; i < trace_bytes; ++i)
    std::fprintf(stderr, "%02x", bytes[i]);
  if (trace_bytes != size)
    std::fprintf(stderr, "...");
  std::fprintf(stderr, "\n");

  // Tensile UserArgs kernels commonly pass a host-side argument block pointer
  // in the tail of the kernarg preload window. Dump the pointed-to block only
  // when explicitly requested and only for mapped readable host addresses.
  if (trace_virtual_lds_userargs_enabled()) {
    uint64_t previous_pointer = 0;
    for (uint32_t offset = 0; offset + sizeof(uint64_t) <= size; offset += sizeof(uint32_t)) {
      uint64_t pointer = 0;
      std::memcpy(&pointer, bytes + offset, sizeof(pointer));
      if (pointer == 0 || pointer == previous_pointer)
        continue;
      if (pointer < (uint64_t{1} << 32u))
        continue;
      trace_virtual_lds_userargs_pointer(packet_id, offset, pointer);
      previous_pointer = pointer;
    }
  }
}

/// @brief Return a compact architecture-family name for diagnostics.
[[nodiscard]] const char *arch_name(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "cdna3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "cdna4";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "rdna3";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "rdna4";
  default:
    return "invalid";
  }
}

/// @brief Return a stable diagnostic severity name.
[[nodiscard]] const char *diagnostic_severity_name(DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  }
  return "diagnostic";
}

/// @brief Return a stable diagnostic kind name.
[[nodiscard]] const char *diagnostic_kind_name(DiagnosticKind kind) {
  switch (kind) {
  case DiagnosticKind::UnsupportedGuestArch:
    return "unsupported-guest-arch";
  case DiagnosticKind::KernelDescriptor:
    return "kernel-descriptor";
  case DiagnosticKind::Legalization:
    return "legalization";
  case DiagnosticKind::ExpandMissing:
    return "expand-missing";
  case DiagnosticKind::ExpandFailed:
    return "expand-failed";
  case DiagnosticKind::ResourceLimit:
    return "resource-limit";
  case DiagnosticKind::KernelSkipped:
    return "kernel-skipped";
  }
  return "unknown";
}

/// @brief Print one structured DBT diagnostic in the same compact style as the CLI.
void print_diagnostic(FILE *stream, const TranslationDiagnostic &diagnostic) {
  std::fprintf(stream, "[rocjitsu-dbt] %s: %s", diagnostic_severity_name(diagnostic.severity),
               diagnostic_kind_name(diagnostic.kind));
  if (diagnostic.guest_offset)
    std::fprintf(stream, " .text+0x%llx",
                 static_cast<unsigned long long>(*diagnostic.guest_offset));
  if (!diagnostic.mnemonic.empty())
    std::fprintf(stream, " %s", diagnostic.mnemonic.c_str());
  std::fprintf(stream, ": %s\n", diagnostic.message.c_str());
  for (const std::string &item : diagnostic.required_work)
    std::fprintf(stream, "[rocjitsu-dbt]   required: %s\n", item.c_str());
}

/// @brief Print errors unconditionally and lower-severity diagnostics when requested.
void print_diagnostics(FILE *stream, std::span<const TranslationDiagnostic> diagnostics,
                       bool include_warnings) {
  for (const TranslationDiagnostic &diagnostic : diagnostics) {
    if (diagnostic.severity == DiagnosticSeverity::Error || include_warnings ||
        diagnostic.kind == DiagnosticKind::KernelSkipped)
      print_diagnostic(stream, diagnostic);
  }
}

/// @brief ISA family and exact ELF machine value detected from a code object.
struct DetectedElfTarget {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t mach = 0;
};

/// @brief Detect the rocjitsu ISA family and exact ELF MACH from an AMDGPU ELF header.
///
/// @details HSA code-object readers are opaque once created, so source ISA
/// detection has to use the ELF bytes captured at reader creation time. The
/// helper checks only ELF identity, machine type, and `EF_AMDGPU_MACH`. The
/// exact MACH is kept because ROCR rejects same-family-but-different-stepping
/// code objects such as gfx1200 ELFs loaded on gfx1201 agents.
[[nodiscard]] DetectedElfTarget detect_target_from_elf(const uint8_t *bytes, size_t size) {
  if (bytes == nullptr || size < sizeof(Elf64_Ehdr))
    return {};

  Elf64_Ehdr header{};
  std::memcpy(&header, bytes, sizeof(header));
  if (std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) != 0)
    return {};
  if (header.e_ident[rocjitsu::EI_CLASS] != ELFCLASS64 || header.e_machine != EM_AMDGPU)
    return {};

  const uint32_t mach = header.e_flags & EF_AMDGPU_MACH;
  return DetectedElfTarget{arch_for_elf_mach(mach), mach};
}

/// @brief Process-local map from HSA code-object reader handles to ELF bytes.
///
/// @details `hsa_executable_load_agent_code_object()` receives only an opaque
/// reader handle. The create wrapper records memory-backed reader bytes here so
/// the load wrapper can translate the original ELF. The registry uses rocjitsu's
/// intrusive list and fixed-block arena so this C ABI path can report registry
/// exhaustion as an HSA status instead of depending on throwing STL allocation.
/// Entries for application readers are non-owning and rely on the application's
/// reader lifetime. Entries for hidden translated readers own a vector so ROCR's
/// memory-reader pointer remains valid while the translated load is in progress.
class CodeObjectReaderRegistry {
public:
  /// @brief Return the singleton registry used by all hook wrappers.
  static CodeObjectReaderRegistry &instance() {
    static CodeObjectReaderRegistry registry;
    return registry;
  }

  /// @brief Stable byte snapshot returned by lookup().
  ///
  /// @details Application-created memory readers are non-owning because ROCR's
  /// public API requires the caller to keep those bytes valid for the reader
  /// lifetime. Translated readers carry shared ownership so a concurrent destroy
  /// cannot free rocjitsu-owned ELF storage while a load is in progress.
  struct ReaderBytes {
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> owned;

    [[nodiscard]] explicit operator bool() const { return bytes != nullptr; }
  };

  /// @brief Record bytes backing a code-object reader.
  /// @param reader HSA reader handle used as the lookup key.
  /// @param bytes Start of the ELF image.
  /// @param size Size of the ELF image in bytes.
  /// @param owned Optional owned storage for translated ELF bytes.
  [[nodiscard]] bool store(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size,
                           std::shared_ptr<const std::vector<uint8_t>> owned) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        entry->bytes = bytes;
        entry->size = size;
        entry->owned = std::move(owned);
        return true;
      }
    }

    void *storage = entry_pool_.try_allocate(sizeof(Entry));
    if (storage == nullptr)
      return false;
    auto *entry = new (storage) Entry(reader.handle, bytes, size, owned);
    entries_.push_front(*entry);
    return true;
  }

  /// @brief Find bytes previously recorded for @p reader.
  /// @returns Snapshot with bytes populated when the reader is known.
  ReaderBytes lookup(hsa_code_object_reader_t reader) {
    std::shared_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        return ReaderBytes{entry->bytes, entry->size, entry->owned};
      }
    }
    return {};
  }

  /// @brief Remove one reader entry and release owned translated bytes if any.
  void remove(hsa_code_object_reader_t reader) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        it = entries_.erase(it);
        destroy_entry(entry);
        return;
      }
      ++it;
    }
  }

  /// @brief Clear all reader entries during tool unload.
  void clear() {
    std::unique_lock lock(mutex_);
    while (!entries_.empty()) {
      auto it = entries_.begin();
      auto *entry = static_cast<Entry *>(it.node_pointer());
      entries_.erase(it);
      destroy_entry(entry);
    }
  }

private:
  /// @brief One code-object reader entry tracked by reader handle.
  struct Entry : util::IListNode<Entry> {
    Entry(uint64_t h, const uint8_t *b, size_t s, std::shared_ptr<const std::vector<uint8_t>> o)
        : handle(h), bytes(b), size(s), owned(std::move(o)) {}

    uint64_t handle = 0;
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> owned;
  };

  /// @brief Destroy one reader entry and release optional owned ELF storage.
  void destroy_entry(Entry *entry) {
    entry->~Entry();
    entry_pool_.deallocate(entry);
  }

  mutable std::shared_mutex mutex_;
  util::ArenaAlloc<sizeof(Entry), 256, alignof(Entry)> entry_pool_;
  util::IntrusiveList<Entry> entries_;
};

/// @brief Tracks executable guest-agent loads that must resolve symbols on the host agent.
class ExecutableAgentRegistry {
public:
  /// @brief Return the process-local executable-agent registry.
  static ExecutableAgentRegistry &instance() {
    static ExecutableAgentRegistry registry;
    return registry;
  }

  /// @brief Remember that @p executable loaded guest code for @p guest on @p host.
  void record(hsa_executable_t executable, hsa_agent_t guest, hsa_agent_t host) {
    std::lock_guard lock(mutex_);
    map_[key(executable, guest)] = host.handle;
  }

  /// @brief Map a guest executable-agent pair back to the host agent used for loading.
  hsa_agent_t map_agent(hsa_executable_t executable, hsa_agent_t agent) {
    std::lock_guard lock(mutex_);
    auto it = map_.find(key(executable, agent));
    if (it == map_.end())
      return agent;
    return hsa_agent_t{it->second};
  }

  /// @brief Drop all recorded agent mappings for one executable.
  void erase_executable(hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    const uint64_t prefix = executable.handle;
    for (auto it = map_.begin(); it != map_.end();) {
      if (it->first.first == prefix)
        it = map_.erase(it);
      else
        ++it;
    }
  }

  /// @brief Clear all executable-agent mappings during hook unload.
  void clear() {
    std::lock_guard lock(mutex_);
    map_.clear();
  }

private:
  using Key = std::pair<uint64_t, uint64_t>;

  /// @brief Hash function for executable/agent handle pairs.
  struct KeyHash {
    size_t operator()(Key key) const {
      size_t h = std::hash<uint64_t>{}(key.first);
      h ^= std::hash<uint64_t>{}(key.second) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  /// @brief Build the map key from an executable and agent handle.
  static Key key(hsa_executable_t executable, hsa_agent_t agent) {
    return {executable.handle, agent.handle};
  }

  std::mutex mutex_;
  std::unordered_map<Key, uint64_t, KeyHash> map_;
};

/// @brief Record memory-backed HSA code-object bytes for later DBT translation.
hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);

/// @brief Warn for file-backed readers, which the MVP cannot translate safely.
hsa_status_t HSA_API rj_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);

/// @brief Remove tracked reader bytes and forward reader destruction.
hsa_status_t HSA_API rj_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);

/// @brief Skip ROCR shutdown in guest mode and uninstall rocjitsu wrappers.
hsa_status_t HSA_API rj_shut_down();

/// @brief Shadow public HSA agent iteration with the guest replacing the host.
hsa_status_t HSA_API rj_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void *data),
                                       void *data);

/// @brief Translate guest code objects and load the translated ELF on the host agent.
hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);

/// @brief Preserve guest ISA iteration so framework fatbin selection picks guest images.
hsa_status_t HSA_API rj_agent_iterate_isas(hsa_agent_t agent,
                                           hsa_status_t (*callback)(hsa_isa_t isa, void *data),
                                           void *data);

/// @brief Create real host queues when the application asks for a guest queue.
hsa_status_t HSA_API rj_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                     void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                     void *data, uint32_t private_segment_size,
                                     uint32_t group_segment_size, hsa_queue_t **queue);

/// @brief Forward destruction for queues returned by guest queue creation.
hsa_status_t HSA_API rj_queue_destroy(hsa_queue_t *queue);

/// @brief Detach virtual-LDS buffers before destroying an application-owned signal.
hsa_status_t HSA_API rj_signal_destroy(hsa_signal_t signal);

/// @brief Rewrite queued dispatch packets before forwarding a relaxed doorbell store.
void HSA_API rj_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value);

/// @brief Rewrite queued dispatch packets before forwarding a release doorbell store.
void HSA_API rj_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);

/// @brief Return host memory regions for guest region iteration.
hsa_status_t HSA_API rj_agent_iterate_regions(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void *data), void *data);

/// @brief Assign memory access to the host agent when the caller passes the guest.
hsa_status_t HSA_API rj_memory_assign_agent(void *ptr, hsa_agent_t agent,
                                            hsa_access_permission_t access);

/// @brief Drop executable-agent mappings before forwarding executable destruction.
hsa_status_t HSA_API rj_executable_destroy(hsa_executable_t executable);

/// @brief Remap guest symbol queries to the host load agent.
hsa_status_t HSA_API rj_executable_get_symbol(hsa_executable_t executable, const char *module_name,
                                              const char *symbol_name, hsa_agent_t agent,
                                              int32_t call_convention,
                                              hsa_executable_symbol_t *symbol);

/// @brief Remap by-name guest symbol queries to the host load agent.
hsa_status_t HSA_API rj_executable_get_symbol_by_name(hsa_executable_t executable,
                                                      const char *symbol_name,
                                                      const hsa_agent_t *agent,
                                                      hsa_executable_symbol_t *symbol);

/// @brief Observe loaded kernel-object addresses for virtual-LDS dispatch metadata.
hsa_status_t HSA_API rj_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                                                   hsa_executable_symbol_info_t attribute,
                                                   void *value);

/// @brief Define executable globals against the host load agent.
hsa_status_t HSA_API rj_executable_agent_global_variable_define(hsa_executable_t executable,
                                                                hsa_agent_t agent,
                                                                const char *variable_name,
                                                                void *address);

/// @brief Iterate host-agent symbols while reporting the guest agent to callbacks.
hsa_status_t HSA_API rj_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data);

/// @brief Wrap the deprecated agentless symbol iterator to record symbol names.
hsa_status_t HSA_API rj_executable_iterate_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void *), void *data);

/// @brief Enumerate guest memory pools for public guest-agent discovery.
hsa_status_t HSA_API rj_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data);

/// @brief Return memory-pool properties without changing guest-facing identity.
hsa_status_t HSA_API rj_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                                 hsa_amd_memory_pool_info_t attribute, void *value);

/// @brief Allocate from a matching host pool when the caller passes a guest pool.
hsa_status_t HSA_API rj_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                                 uint32_t flags, void **ptr);

/// @brief Free host-backed memory allocated through mapped pools.
hsa_status_t HSA_API rj_amd_memory_pool_free(void *ptr);

/// @brief Enable profiling on the mapped host queue.
hsa_status_t HSA_API rj_amd_profiling_set_profiler_enabled(hsa_queue_t *queue, int enable);

/// @brief Query dispatch timestamps through the mapped host agent.
hsa_status_t HSA_API rj_amd_profiling_get_dispatch_time(hsa_agent_t agent, hsa_signal_t signal,
                                                        hsa_amd_profiling_dispatch_time_t *time);

/// @brief Convert agent ticks through the mapped host agent.
hsa_status_t HSA_API rj_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent,
                                                                    uint64_t agent_tick,
                                                                    uint64_t *system_tick);

/// @brief Query agent/pool relationships through mapped host handles.
hsa_status_t HSA_API rj_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                       hsa_amd_memory_pool_t memory_pool,
                                                       hsa_amd_agent_memory_pool_info_t attribute,
                                                       void *value);

/// @brief Allow memory access for mapped host agents, removing duplicates.
hsa_status_t HSA_API rj_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t *agents,
                                                const uint32_t *flags, const void *ptr);

/// @brief Map source and destination agents for async memory copies.
hsa_status_t HSA_API rj_amd_memory_async_copy(void *dst, hsa_agent_t dst_agent, const void *src,
                                              hsa_agent_t src_agent, size_t size,
                                              uint32_t num_dep_signals,
                                              const hsa_signal_t *dep_signals,
                                              hsa_signal_t completion_signal);

/// @brief Map source and destination agents for engine-selected async copies.
hsa_status_t HSA_API rj_amd_memory_async_copy_on_engine(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t *dep_signals, hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma);

/// @brief Map the copy agent for rectangular async copies.
hsa_status_t HSA_API rj_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t *dst, const hsa_dim3_t *dst_offset, const hsa_pitched_ptr_t *src,
    const hsa_dim3_t *src_offset, const hsa_dim3_t *range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal);

/// @brief Query copy-engine status using mapped host agents.
hsa_status_t HSA_API rj_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                                      uint32_t *engine_ids_mask);

/// @brief Query preferred copy engines using mapped host agents.
hsa_status_t HSA_API rj_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent,
                                                             hsa_agent_t src_agent,
                                                             uint32_t *engine_ids_mask);

/// @brief Map agent arrays before locking host memory for GPU access.
hsa_status_t HSA_API rj_amd_memory_lock(void *host_ptr, size_t size, hsa_agent_t *agents,
                                        int num_agent, void **agent_ptr);

/// @brief Forward memory-fill operations while preserving trace visibility.
hsa_status_t HSA_API rj_amd_memory_fill(void *ptr, uint32_t value, size_t count);

/// @brief Map pointer-info owner and accessible agents from host back to guest.
hsa_status_t HSA_API rj_amd_pointer_info(const void *ptr, hsa_amd_pointer_info_t *info,
                                         void *(*alloc)(size_t), uint32_t *num_agents_accessible,
                                         hsa_agent_t **accessible);

/// @brief Map agent arrays and guest pools before locking host memory to a pool.
hsa_status_t HSA_API rj_amd_memory_lock_to_pool(void *host_ptr, size_t size, hsa_agent_t *agents,
                                                int num_agent, hsa_amd_memory_pool_t pool,
                                                uint32_t flags, void **agent_ptr);

/// @brief Prefetch SVM memory to the host agent when the caller passes the guest.
hsa_status_t HSA_API rj_amd_svm_prefetch_async(void *ptr, size_t size, hsa_agent_t agent,
                                               uint32_t num_dep_signals,
                                               const hsa_signal_t *dep_signals,
                                               hsa_signal_t completion_signal);

/// @brief Rewrite virtual-memory access descriptors from guest to host.
hsa_status_t HSA_API rj_amd_vmem_set_access(void *va, size_t size,
                                            const hsa_amd_memory_access_desc_t *desc,
                                            size_t desc_cnt);

/// @brief Query virtual-memory access through the host agent.
hsa_status_t HSA_API rj_amd_vmem_get_access(void *va, hsa_access_permission_t *perms,
                                            hsa_agent_t agent_handle);

/// @brief Apply async scratch limits to the real host agent.
hsa_status_t HSA_API rj_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold);

/// @brief Map every embedded agent in AMD batch-copy operations.
hsa_status_t HSA_API rj_amd_memory_async_batch_copy(const hsa_amd_memory_copy_op_t *copy_ops,
                                                    uint32_t num_copy_ops, uint32_t num_dep_signals,
                                                    const hsa_signal_t *dep_signals);

/// @brief Preload runtime state on the real host agent.
hsa_status_t HSA_API rj_amd_agent_preload(hsa_agent_t agent, uint64_t flags);

/// @brief Create and track AMD queue-intercept queues on the real host agent.
hsa_status_t HSA_API rj_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue);

/// @brief Clear cached guest-to-host memory-pool mappings on HSA unload.
void clear_memory_pool_mapper();

/// @brief Clear cached guest-to-host agent mapping on HSA unload.
void clear_agent_mapper();

/// @brief Release virtual-LDS dispatch buffers and drop tracked queue state.
void clear_virtual_lds_dispatch_queues();

/// @brief Original HSA table entries saved for internal hook queries but not patched.
///
/// @details Each entry is:
/// `X(name, table_ptr, present, field, type)`.
/// - `name` is the `RjHsaLayer` getter name and `original_<name>_` member stem.
/// - `table_ptr` is the API-table member pointer that owns `field`.
/// - `present` guards table access when a table or versioned field may be absent.
/// - `field` is the HSA table function-pointer field to save.
/// - `type` is the exact function-pointer type stored in `original_<name>_`.
#define RJ_HSA_SAVED_ONLY_ENTRIES(X)                                                               \
  X(agent_get_info, core_, true, hsa_agent_get_info_fn, decltype(hsa_agent_get_info) *)            \
  X(isa_get_info_alt, core_, true, hsa_isa_get_info_alt_fn, decltype(hsa_isa_get_info_alt) *)      \
  X(queue_load_write_index_relaxed, core_, true, hsa_queue_load_write_index_relaxed_fn,            \
    decltype(hsa_queue_load_write_index_relaxed) *)                                                \
  X(signal_create, core_, true, hsa_signal_create_fn, decltype(hsa_signal_create) *)               \
  X(signal_load_scacquire, core_, true, hsa_signal_load_scacquire_fn,                              \
    decltype(hsa_signal_load_scacquire) *)                                                         \
  X(amd_queue_intercept_register, amd_ext_,                                                        \
    api_table_has_field(amd_ext_, offsetof(AmdExtTable, hsa_amd_queue_intercept_register_fn),      \
                        &AmdExtTable::hsa_amd_queue_intercept_register_fn),                        \
    hsa_amd_queue_intercept_register_fn, hsa_amd_queue_intercept_register_fn_t)

/// @brief HSA table entries patched by the DBT hook.
///
/// @details Each entry is:
/// `X(name, table_ptr, present, patch_if_original, field, wrapper, type)`.
/// - `name` is the `RjHsaLayer` getter name and `original_<name>_` member stem.
/// - `table_ptr` is the API-table member pointer that owns `field`.
/// - `present` guards table access when a table or versioned field may be absent.
/// - `patch_if_original` means install only writes `wrapper` when the saved
///   original function pointer is non-null. Core entries set this false because
///   they are required after `validate_table()`. AMD extension entries set this
///   true because ROCR may expose the table field but leave the function absent.
/// - `field` is the HSA table function-pointer field to save/patch/restore.
/// - `wrapper` is the rocjitsu replacement function assigned during install.
/// - `type` is the exact function-pointer type stored in `original_<name>_`.
#define RJ_HSA_PATCH_ENTRIES(X)                                                                    \
  X(shut_down, core_, true, false, hsa_shut_down_fn, rj_shut_down, decltype(hsa_shut_down) *)      \
  X(iterate_agents, core_, true, false, hsa_iterate_agents_fn, rj_iterate_agents,                  \
    decltype(hsa_iterate_agents) *)                                                                \
  X(agent_iterate_isas, core_, true, false, hsa_agent_iterate_isas_fn, rj_agent_iterate_isas,      \
    decltype(hsa_agent_iterate_isas) *)                                                            \
  X(queue_create, core_, true, false, hsa_queue_create_fn, rj_queue_create,                        \
    decltype(hsa_queue_create) *)                                                                  \
  X(queue_destroy, core_, true, false, hsa_queue_destroy_fn, rj_queue_destroy,                     \
    decltype(hsa_queue_destroy) *)                                                                 \
  X(signal_destroy, core_, true, false, hsa_signal_destroy_fn, rj_signal_destroy,                  \
    decltype(hsa_signal_destroy) *)                                                                \
  X(signal_store_relaxed, core_, true, true, hsa_signal_store_relaxed_fn, rj_signal_store_relaxed, \
    decltype(hsa_signal_store_relaxed) *)                                                          \
  X(signal_store_screlease, core_, true, true, hsa_signal_store_screlease_fn,                      \
    rj_signal_store_screlease, decltype(hsa_signal_store_screlease) *)                             \
  X(agent_iterate_regions, core_, true, false, hsa_agent_iterate_regions_fn,                       \
    rj_agent_iterate_regions, decltype(hsa_agent_iterate_regions) *)                               \
  X(memory_assign_agent, core_, true, false, hsa_memory_assign_agent_fn, rj_memory_assign_agent,   \
    decltype(hsa_memory_assign_agent) *)                                                           \
  X(executable_destroy, core_, true, false, hsa_executable_destroy_fn, rj_executable_destroy,      \
    decltype(hsa_executable_destroy) *)                                                            \
  X(executable_get_symbol, core_, true, false, hsa_executable_get_symbol_fn,                       \
    rj_executable_get_symbol, decltype(hsa_executable_get_symbol) *)                               \
  X(executable_get_symbol_by_name, core_, true, false, hsa_executable_get_symbol_by_name_fn,       \
    rj_executable_get_symbol_by_name, decltype(hsa_executable_get_symbol_by_name) *)               \
  X(executable_symbol_get_info, core_, true, false, hsa_executable_symbol_get_info_fn,             \
    rj_executable_symbol_get_info, decltype(hsa_executable_symbol_get_info) *)                     \
  X(executable_agent_global_variable_define, core_, true, false,                                   \
    hsa_executable_agent_global_variable_define_fn, rj_executable_agent_global_variable_define,    \
    decltype(hsa_executable_agent_global_variable_define) *)                                       \
  X(executable_iterate_agent_symbols, core_, true, false, hsa_executable_iterate_agent_symbols_fn, \
    rj_executable_iterate_agent_symbols, decltype(hsa_executable_iterate_agent_symbols) *)         \
  X(executable_iterate_symbols, core_, true, false, hsa_executable_iterate_symbols_fn,             \
    rj_executable_iterate_symbols, decltype(hsa_executable_iterate_symbols) *)                     \
  X(create_from_file, core_, true, false, hsa_code_object_reader_create_from_file_fn,              \
    rj_code_object_reader_create_from_file, decltype(hsa_code_object_reader_create_from_file) *)   \
  X(create_from_memory, core_, true, false, hsa_code_object_reader_create_from_memory_fn,          \
    rj_code_object_reader_create_from_memory,                                                      \
    decltype(hsa_code_object_reader_create_from_memory) *)                                         \
  X(destroy, core_, true, false, hsa_code_object_reader_destroy_fn, rj_code_object_reader_destroy, \
    decltype(hsa_code_object_reader_destroy) *)                                                    \
  X(load_agent_code_object, core_, true, false, hsa_executable_load_agent_code_object_fn,          \
    rj_executable_load_agent_code_object, decltype(hsa_executable_load_agent_code_object) *)       \
  X(amd_memory_pool_get_info, amd_ext_, amd_ext_ != nullptr, true,                                 \
    hsa_amd_memory_pool_get_info_fn, rj_amd_memory_pool_get_info,                                  \
    hsa_amd_memory_pool_get_info_fn_t)                                                             \
  X(amd_agent_iterate_memory_pools, amd_ext_, amd_ext_ != nullptr, true,                           \
    hsa_amd_agent_iterate_memory_pools_fn, rj_amd_agent_iterate_memory_pools,                      \
    hsa_amd_agent_iterate_memory_pools_fn_t)                                                       \
  X(amd_memory_pool_allocate, amd_ext_, amd_ext_ != nullptr, true,                                 \
    hsa_amd_memory_pool_allocate_fn, rj_amd_memory_pool_allocate,                                  \
    hsa_amd_memory_pool_allocate_fn_t)                                                             \
  X(amd_memory_pool_free, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_pool_free_fn,        \
    rj_amd_memory_pool_free, hsa_amd_memory_pool_free_fn_t)                                        \
  X(amd_profiling_set_profiler_enabled, amd_ext_, amd_ext_ != nullptr, true,                       \
    hsa_amd_profiling_set_profiler_enabled_fn, rj_amd_profiling_set_profiler_enabled,              \
    hsa_amd_profiling_set_profiler_enabled_fn_t)                                                   \
  X(amd_profiling_get_dispatch_time, amd_ext_, amd_ext_ != nullptr, true,                          \
    hsa_amd_profiling_get_dispatch_time_fn, rj_amd_profiling_get_dispatch_time,                    \
    hsa_amd_profiling_get_dispatch_time_fn_t)                                                      \
  X(amd_profiling_convert_tick_to_system_domain, amd_ext_, amd_ext_ != nullptr, true,              \
    hsa_amd_profiling_convert_tick_to_system_domain_fn,                                            \
    rj_amd_profiling_convert_tick_to_system_domain,                                                \
    hsa_amd_profiling_convert_tick_to_system_domain_fn_t)                                          \
  X(amd_agent_memory_pool_get_info, amd_ext_, amd_ext_ != nullptr, true,                           \
    hsa_amd_agent_memory_pool_get_info_fn, rj_amd_agent_memory_pool_get_info,                      \
    hsa_amd_agent_memory_pool_get_info_fn_t)                                                       \
  X(amd_agents_allow_access, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_agents_allow_access_fn,  \
    rj_amd_agents_allow_access, hsa_amd_agents_allow_access_fn_t)                                  \
  X(amd_memory_async_copy, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_async_copy_fn,      \
    rj_amd_memory_async_copy, hsa_amd_memory_async_copy_fn_t)                                      \
  X(amd_memory_async_copy_on_engine, amd_ext_, amd_ext_ != nullptr, true,                          \
    hsa_amd_memory_async_copy_on_engine_fn, rj_amd_memory_async_copy_on_engine,                    \
    hsa_amd_memory_async_copy_on_engine_fn_t)                                                      \
  X(amd_memory_async_copy_rect, amd_ext_, amd_ext_ != nullptr, true,                               \
    hsa_amd_memory_async_copy_rect_fn, rj_amd_memory_async_copy_rect,                              \
    hsa_amd_memory_async_copy_rect_fn_t)                                                           \
  X(amd_memory_copy_engine_status, amd_ext_, amd_ext_ != nullptr, true,                            \
    hsa_amd_memory_copy_engine_status_fn, rj_amd_memory_copy_engine_status,                        \
    hsa_amd_memory_copy_engine_status_fn_t)                                                        \
  X(amd_memory_lock, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_lock_fn,                  \
    rj_amd_memory_lock, hsa_amd_memory_lock_fn_t)                                                  \
  X(amd_memory_lock_to_pool, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_lock_to_pool_fn,  \
    rj_amd_memory_lock_to_pool, hsa_amd_memory_lock_to_pool_fn_t)                                  \
  X(amd_svm_prefetch_async, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_svm_prefetch_async_fn,    \
    rj_amd_svm_prefetch_async, hsa_amd_svm_prefetch_async_fn_t)                                    \
  X(amd_pointer_info, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_pointer_info_fn,                \
    rj_amd_pointer_info, hsa_amd_pointer_info_fn_t)                                                \
  X(amd_vmem_set_access, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_vmem_set_access_fn,          \
    rj_amd_vmem_set_access, hsa_amd_vmem_set_access_fn_t)                                          \
  X(amd_vmem_get_access, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_vmem_get_access_fn,          \
    rj_amd_vmem_get_access, hsa_amd_vmem_get_access_fn_t)                                          \
  X(amd_agent_set_async_scratch_limit, amd_ext_, amd_ext_ != nullptr, true,                        \
    hsa_amd_agent_set_async_scratch_limit_fn, rj_amd_agent_set_async_scratch_limit,                \
    hsa_amd_agent_set_async_scratch_limit_fn_t)                                                    \
  X(amd_memory_get_preferred_copy_engine, amd_ext_, amd_ext_ != nullptr, true,                     \
    hsa_amd_memory_get_preferred_copy_engine_fn, rj_amd_memory_get_preferred_copy_engine,          \
    hsa_amd_memory_get_preferred_copy_engine_fn_t)                                                 \
  X(amd_memory_fill, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_fill_fn,                  \
    rj_amd_memory_fill, hsa_amd_memory_fill_fn_t)                                                  \
  X(amd_memory_async_batch_copy, amd_ext_,                                                         \
    api_table_has_field(amd_ext_, offsetof(AmdExtTable, hsa_amd_memory_async_batch_copy_fn),       \
                        &AmdExtTable::hsa_amd_memory_async_batch_copy_fn),                         \
    true, hsa_amd_memory_async_batch_copy_fn, rj_amd_memory_async_batch_copy,                      \
    hsa_amd_memory_async_batch_copy_fn_t)                                                          \
  X(amd_agent_preload, amd_ext_,                                                                   \
    api_table_has_field(amd_ext_, offsetof(AmdExtTable, hsa_amd_agent_preload_fn),                 \
                        &AmdExtTable::hsa_amd_agent_preload_fn),                                   \
    true, hsa_amd_agent_preload_fn, rj_amd_agent_preload, hsa_amd_agent_preload_fn_t)              \
  X(amd_queue_intercept_create, amd_ext_,                                                          \
    api_table_has_field(amd_ext_, offsetof(AmdExtTable, hsa_amd_queue_intercept_create_fn),        \
                        &AmdExtTable::hsa_amd_queue_intercept_create_fn),                          \
    true, hsa_amd_queue_intercept_create_fn, rj_amd_queue_intercept_create,                        \
    hsa_amd_queue_intercept_create_fn_t)

/// @brief Process-local HSA API table patch state for the rocjitsu DBT tool.
///
/// @details Tool chaining depends on saving the function pointers that are
/// present at `OnLoad()` time and calling those saved pointers from wrappers.
/// They may already point at another tool's wrapper. `OnUnload()` restores only
/// entries that still point at rocjitsu wrappers so later tools are not
/// accidentally overwritten.
class RjHsaLayer {
public:
  /// @brief Validate the incoming table, save original entries, and install wrappers.
  bool install(HsaApiTable *table, HookConfig config) {
    std::lock_guard lock(mutex_);
    if (active_) {
      std::fprintf(stderr, "[rocjitsu-hooks] OnLoad called while hook is already active\n");
      return false;
    }
    if (!validate_table(table))
      return false;

    table_ = table;
    core_ = table->core_;
    amd_ext_ = table->amd_ext_;
    g_log_level.store(config.log_level, std::memory_order_relaxed);
    config_ = std::move(config);

#define RJ_SAVE_SAVED_ONLY(name, table_ptr, present, field, type)                                  \
  if (present)                                                                                     \
    original_##name##_ = (table_ptr)->field;
    RJ_HSA_SAVED_ONLY_ENTRIES(RJ_SAVE_SAVED_ONLY)
#undef RJ_SAVE_SAVED_ONLY

#define RJ_SAVE_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)           \
  if (present)                                                                                     \
    original_##name##_ = (table_ptr)->field;
    RJ_HSA_PATCH_ENTRIES(RJ_SAVE_PATCH)
#undef RJ_SAVE_PATCH

    if (original_create_from_file_ == nullptr || original_create_from_memory_ == nullptr ||
        original_destroy_ == nullptr || original_load_agent_code_object_ == nullptr ||
        original_iterate_agents_ == nullptr || original_agent_get_info_ == nullptr ||
        original_agent_iterate_isas_ == nullptr || original_isa_get_info_alt_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-hooks] HSA core table contains null code-object entries\n");
      clear_unlocked();
      return false;
    }

#define RJ_INSTALL_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)        \
  if ((present) && (!(patch_if_original) || original_##name##_ != nullptr))                        \
    (table_ptr)->field = wrapper;
    RJ_HSA_PATCH_ENTRIES(RJ_INSTALL_PATCH)
#undef RJ_INSTALL_PATCH

    active_ = true;

    log_message(kLogInfo, "installed DBT hook target=%s arch=%s mach=0x%x",
                config_->target.name.data(), arch_name(config_->target.arch), config_->target.mach);
    return true;
  }

  /// @brief Restore rocjitsu wrappers if still installed and clear owned state.
  void uninstall() {
    bool had_state = false;
    {
      std::lock_guard lock(mutex_);
      had_state = active_ || core_ != nullptr || amd_ext_ != nullptr;
      if (active_ && core_ != nullptr) {
        log_message(kLogVerbose, "uninstall begin");
#define RJ_RESTORE_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)        \
  if ((present) && (table_ptr)->field == wrapper)                                                  \
    (table_ptr)->field = original_##name##_;
        RJ_HSA_PATCH_ENTRIES(RJ_RESTORE_PATCH)
#undef RJ_RESTORE_PATCH
      }
      active_ = false;
    }

    CodeObjectReaderRegistry::instance().clear();
    ExecutableAgentRegistry::instance().clear();
    clear_agent_mapper();
    VirtualLdsRuntimeRegistry::instance().clear();
    clear_virtual_lds_dispatch_queues();
    // Drop the installed runtime-API table so a later OnLoad cannot race readers
    // against stale entry points and post-unload callbacks become no-ops.
    rocjitsu::hooks::clear_virtual_lds_runtime_api();
    clear_memory_pool_mapper();
    if (!had_state)
      return;

    std::lock_guard lock(mutex_);
    log_message(kLogVerbose, "uninstall end");
    clear_unlocked();
  }

  /// @brief Return the active hook log level, or disabled when uninstalled.
  [[nodiscard]] int log_level() const {
    std::lock_guard lock(mutex_);
    return config_ ? config_->log_level : kLogDisabled;
  }

  /// @brief Return a copy of the active hook configuration.
  [[nodiscard]] std::optional<HookConfig> config() const {
    std::lock_guard lock(mutex_);
    return config_;
  }

#define RJ_DEFINE_SAVED_ONLY_GETTER(name, table_ptr, present, field, type)                         \
  [[nodiscard]] type name() const {                                                                \
    std::lock_guard lock(mutex_);                                                                  \
    return original_##name##_;                                                                     \
  }
  RJ_HSA_SAVED_ONLY_ENTRIES(RJ_DEFINE_SAVED_ONLY_GETTER)
#undef RJ_DEFINE_SAVED_ONLY_GETTER

#define RJ_DEFINE_PATCH_GETTER(name, table_ptr, present, patch_if_original, field, wrapper, type)  \
  [[nodiscard]] type name() const {                                                                \
    std::lock_guard lock(mutex_);                                                                  \
    return original_##name##_;                                                                     \
  }
  RJ_HSA_PATCH_ENTRIES(RJ_DEFINE_PATCH_GETTER)
#undef RJ_DEFINE_PATCH_GETTER

private:
  /// @brief Check that ROCR supplied the HSA entry points used by this hook.
  [[nodiscard]] static bool validate_table(HsaApiTable *table) {
    if (table == nullptr || table->core_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-hooks] invalid HSA API table passed to OnLoad\n");
      return false;
    }

    constexpr size_t required_size =
        offsetof(CoreApiTable, hsa_executable_iterate_agent_symbols_fn) +
        sizeof(CoreApiTable::hsa_executable_iterate_agent_symbols_fn);
    if (table->core_->version.minor_id < required_size) {
      std::fprintf(stderr,
                   "[rocjitsu-hooks] HSA core table too small: got %u bytes, need %zu bytes\n",
                   table->core_->version.minor_id, required_size);
      return false;
    }
    return true;
  }

  /// @brief Clear saved table pointers and runtime configuration.
  void clear_unlocked() {
    active_ = false;
    g_log_level.store(kLogDisabled, std::memory_order_relaxed);
    restore_signal_backtrace_handlers();
    table_ = nullptr;
    core_ = nullptr;
    amd_ext_ = nullptr;
    config_.reset();

#define RJ_CLEAR_SAVED_ONLY(name, table_ptr, present, field, type) original_##name##_ = nullptr;
    RJ_HSA_SAVED_ONLY_ENTRIES(RJ_CLEAR_SAVED_ONLY)
#undef RJ_CLEAR_SAVED_ONLY

#define RJ_CLEAR_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)          \
  original_##name##_ = nullptr;
    RJ_HSA_PATCH_ENTRIES(RJ_CLEAR_PATCH)
#undef RJ_CLEAR_PATCH
  }

  mutable std::mutex mutex_;
  HsaApiTable *table_ = nullptr;
  CoreApiTable *core_ = nullptr;
  AmdExtTable *amd_ext_ = nullptr;
  std::optional<HookConfig> config_;
  bool active_ = false;

#define RJ_DECLARE_SAVED_ONLY(name, table_ptr, present, field, type)                               \
  type original_##name##_ = nullptr;
  RJ_HSA_SAVED_ONLY_ENTRIES(RJ_DECLARE_SAVED_ONLY)
#undef RJ_DECLARE_SAVED_ONLY

#define RJ_DECLARE_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)        \
  type original_##name##_ = nullptr;
  RJ_HSA_PATCH_ENTRIES(RJ_DECLARE_PATCH)
#undef RJ_DECLARE_PATCH
};

/// @brief Return the singleton hook state used by every wrapper.
RjHsaLayer &layer() {
  static RjHsaLayer state;
  return state;
}

/// @brief Parse a uint32_t from a NUL-terminated sysfs text value.
[[nodiscard]] bool parse_u32_text(const char *text, uint32_t *out) {
  if (!text || !out)
    return false;
  errno = 0;
  char *end = nullptr;
  unsigned long parsed = std::strtoul(text, &end, 0);
  if (errno != 0 || end == text || parsed > UINT32_MAX)
    return false;
  while (*end == '\n' || *end == '\r' || *end == ' ' || *end == '\t')
    ++end;
  if (*end != '\0')
    return false;
  *out = static_cast<uint32_t>(parsed);
  return true;
}

/// @brief Read a uint32_t from a sysfs-style file.
[[nodiscard]] std::optional<uint32_t> read_u32_file(const std::string &path) {
  FILE *file = std::fopen(path.c_str(), "r");
  if (!file)
    return std::nullopt;

  std::array<char, 64> buffer{};
  char *line = std::fgets(buffer.data(), static_cast<int>(buffer.size()), file);
  std::fclose(file);
  if (!line)
    return std::nullopt;

  uint32_t value = 0;
  if (!parse_u32_text(buffer.data(), &value))
    return std::nullopt;
  return value;
}

/// @brief Search one KFD topology root for the node id owning @p gpu_id.
[[nodiscard]] std::optional<uint32_t> node_id_for_kfd_gpu_id_in_root(const char *root,
                                                                     uint32_t gpu_id) {
  DIR *dir = opendir(root);
  if (!dir)
    return std::nullopt;

  std::optional<uint32_t> result;
  while (dirent *entry = readdir(dir)) {
    if (entry->d_name[0] == '.')
      continue;

    uint32_t node_id = 0;
    if (!parse_u32_text(entry->d_name, &node_id))
      continue;

    std::string gpu_id_path = std::string(root) + "/" + entry->d_name + "/gpu_id";
    std::optional<uint32_t> node_gpu_id = read_u32_file(gpu_id_path);
    if (node_gpu_id && *node_gpu_id == gpu_id) {
      result = node_id;
      break;
    }
  }
  closedir(dir);
  return result;
}

/// @brief Translate a configured KFD gpu_id to ROCR's HSA driver node id.
///
/// @details The config uses the KFD topology `gpu_id` because it is stable
/// across ROCR agent handle creation and is the id KFD ioctls use. HSA does
/// not expose that value directly, so the hook reads the redirected topology
/// tree and later compares agents by `HSA_AMD_AGENT_INFO_DRIVER_NODE_ID`.
[[nodiscard]] std::optional<uint32_t> node_id_for_kfd_gpu_id(uint32_t gpu_id) {
  constexpr std::array<const char *, 2> kTopologyNodeRoots = {
      "/sys/devices/virtual/kfd/kfd/topology/nodes", "/sys/class/kfd/kfd/topology/nodes"};
  for (const char *root : kTopologyNodeRoots) {
    if (std::optional<uint32_t> node_id = node_id_for_kfd_gpu_id_in_root(root, gpu_id))
      return node_id;
  }
  return std::nullopt;
}

/// @brief Maps the configured guest HSA agent to the selected host execution agent.
///
/// @details Discovery is lazy because HSA tools are installed before ROCR has
/// necessarily enumerated agents. The guest agent remains visible to callers,
/// but calls that would execute or allocate through it are redirected to host_.
class AgentMapper {
public:
  /// @brief Return the process-wide agent mapper.
  static AgentMapper &instance() {
    static AgentMapper mapper;
    return mapper;
  }

  /// @brief Replace the guest agent with the selected host agent.
  hsa_agent_t map(hsa_agent_t agent) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    if (has_mapping_ && agent.handle == guest_.handle)
      return host_;
    return agent;
  }

  /// @brief Return true when @p agent is the configured guest agent.
  bool is_guest(hsa_agent_t agent) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    return has_mapping_ && agent.handle == guest_.handle;
  }

  /// @brief Return the host agent selected to execute guest work.
  hsa_agent_t host_for_guest() {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    return has_mapping_ ? host_ : hsa_agent_t{};
  }

  /// @brief Return the visible guest agent, if a mapping was discovered.
  hsa_agent_t guest_agent() {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    return has_mapping_ ? guest_ : hsa_agent_t{};
  }

  /// @brief Replace the selected host agent with the visible guest agent.
  hsa_agent_t guest_for_host(hsa_agent_t agent) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    if (has_mapping_ && agent.handle == host_.handle)
      return guest_;
    return agent;
  }

  /// @brief Drop cached agent mapping so unload/reload can rediscover it.
  void clear() {
    std::lock_guard lock(mutex_);
    discovered_ = false;
    has_mapping_ = false;
    guest_ = {};
    host_ = {};
    ++generation_;
  }

private:
  /// @brief Agent discovery callback state.
  struct AgentSearchData {
    AgentMapper *self = nullptr;
    TargetInfo guest{};
    TargetInfo host{};
    std::optional<uint32_t> host_node_id;
    hsa_agent_t guest_agent{};
    hsa_agent_t host_agent{};
  };

  /// @brief ISA discovery callback state for matching configured targets.
  struct IsaSearchData {
    AgentMapper *self = nullptr;
    TargetInfo target{};
    bool found = false;
  };

  /// @brief Return true when an HSA ISA name names the requested target.
  static bool target_matches_isa_name(std::string_view isa_name, std::string_view target) {
    constexpr std::string_view prefix = "amdgcn-amd-amdhsa--";
    if (isa_name.starts_with(prefix))
      isa_name.remove_prefix(prefix.size());
    if (!isa_name.starts_with(target))
      return false;
    if (isa_name.size() == target.size())
      return true;
    return isa_name[target.size()] == ':' || isa_name[target.size()] == '\0';
  }

  /// @brief HSA ISA iteration callback used while matching an agent target.
  static hsa_status_t isa_callback(hsa_isa_t isa, void *data) {
    auto *search = static_cast<IsaSearchData *>(data);
    auto *get_info = layer().isa_get_info_alt();
    if (!get_info)
      return HSA_STATUS_ERROR;

    uint32_t name_length = 0;
    if (get_info(isa, HSA_ISA_INFO_NAME_LENGTH, &name_length) != HSA_STATUS_SUCCESS ||
        name_length == 0)
      return HSA_STATUS_SUCCESS;
    std::vector<char> name(name_length + 1, 0);
    if (get_info(isa, HSA_ISA_INFO_NAME, name.data()) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;

    if (target_matches_isa_name(std::string_view(name.data()), search->target.name)) {
      search->found = true;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Return true when @p agent advertises an ISA matching @p target.
  bool agent_has_target(hsa_agent_t agent, TargetInfo target) {
    auto *iterate_isas = layer().agent_iterate_isas();
    if (!iterate_isas)
      return false;
    IsaSearchData search{this, target, false};
    hsa_status_t status = iterate_isas(agent, isa_callback, &search);
    return search.found || status == HSA_STATUS_INFO_BREAK;
  }

  /// @brief Read ROCR's KFD topology node id for @p agent.
  std::optional<uint32_t> agent_driver_node_id(hsa_agent_t agent) {
    auto *get_info = layer().agent_get_info();
    if (!get_info)
      return std::nullopt;

    uint32_t node_id = 0;
    hsa_status_t status =
        get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID), &node_id);
    if (status != HSA_STATUS_SUCCESS)
      return std::nullopt;
    return node_id;
  }

  /// @brief Return true when @p agent belongs to @p expected_node_id.
  bool agent_has_driver_node(hsa_agent_t agent, uint32_t expected_node_id) {
    std::optional<uint32_t> node_id = agent_driver_node_id(agent);
    return node_id && *node_id == expected_node_id;
  }

  /// @brief Return true when @p agent is the configured host execution agent.
  bool agent_matches_host(hsa_agent_t agent, const AgentSearchData &search) {
    if (!agent_has_target(agent, search.host))
      return false;
    if (search.host_node_id)
      return agent_has_driver_node(agent, *search.host_node_id);
    return true;
  }

  /// @brief HSA agent iteration callback that records guest and host agents.
  static hsa_status_t agent_callback(hsa_agent_t agent, void *data) {
    auto *search = static_cast<AgentSearchData *>(data);
    if (search->guest_agent.handle == 0 && search->self->agent_has_target(agent, search->guest))
      search->guest_agent = agent;
    if (search->host_agent.handle == 0 && search->self->agent_matches_host(agent, *search))
      search->host_agent = agent;
    if (search->guest_agent.handle != 0 && search->host_agent.handle != 0)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Lazily discover the guest-to-host agent mapping.
  void ensure_discovered() {
    uint64_t generation = 0;
    {
      std::unique_lock lock(mutex_);
      while (discovering_)
        discovery_cv_.wait(lock);
      if (discovered_)
        return;
      discovering_ = true;
      generation = generation_;
    }

    auto config = layer().config();
    hsa_agent_t discovered_guest{};
    hsa_agent_t discovered_host{};
    bool has_mapping = false;
    bool attempted_agent_search = false;

    if (config && config->guest_target) {
      auto *iterate_agents = layer().iterate_agents();
      if (iterate_agents) {
        std::optional<uint32_t> host_node_id;
        bool can_search = true;
        if (config->host_gpu_id != 0) {
          host_node_id = node_id_for_kfd_gpu_id(config->host_gpu_id);
          if (!host_node_id) {
            std::fprintf(stderr,
                         "[rocjitsu-hooks] failed to find topology node for host KFD gpu_id=%u\n",
                         config->host_gpu_id);
            can_search = false;
          }
        }

        if (can_search) {
          AgentSearchData search{this, *config->guest_target, config->target, host_node_id};
          hsa_status_t status = iterate_agents(agent_callback, &search);
          attempted_agent_search = true;
          has_mapping = (status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK) &&
                        search.guest_agent.handle != 0 && search.host_agent.handle != 0 &&
                        search.guest_agent.handle != search.host_agent.handle;
          discovered_guest = search.guest_agent;
          discovered_host = search.host_agent;
        }
      }
    }

    bool published = false;
    {
      std::lock_guard lock(mutex_);
      if (generation == generation_) {
        guest_ = discovered_guest;
        host_ = discovered_host;
        has_mapping_ = has_mapping;
        discovered_ = true;
        published = true;
      }
      discovering_ = false;
    }
    discovery_cv_.notify_all();
    if (!published)
      return;

    if (has_mapping) {
      log_message(kLogInfo, "mapped guest agent=%llu to host agent=%llu",
                  static_cast<unsigned long long>(discovered_guest.handle),
                  static_cast<unsigned long long>(discovered_host.handle));
      if (config->log_level > kLogDisabled) {
        std::optional<uint32_t> selected_node_id = agent_driver_node_id(discovered_host);
        std::fprintf(stderr,
                     "[rocjitsu-hooks] selected host agent=%llu for guest agent=%llu "
                     "host_gpu_id=%u host_node_id=%u\n",
                     static_cast<unsigned long long>(discovered_host.handle),
                     static_cast<unsigned long long>(discovered_guest.handle), config->host_gpu_id,
                     selected_node_id.value_or(0));
      }
    } else if (attempted_agent_search) {
      std::fprintf(stderr, "[rocjitsu-hooks] failed to find guest/host HSA agents for DBT\n");
    }
  }

  std::mutex mutex_;
  std::condition_variable discovery_cv_;
  bool discovering_ = false;
  bool discovered_ = false;
  bool has_mapping_ = false;
  uint64_t generation_ = 0;
  hsa_agent_t guest_{};
  hsa_agent_t host_{};
};

/// @brief Maps guest memory pools to matching pools on the selected host agent.
class MemoryPoolMapper {
public:
  /// @brief Return the process-wide memory pool mapper.
  static MemoryPoolMapper &instance() {
    static MemoryPoolMapper mapper;
    return mapper;
  }

  /// @brief Replace a guest pool with the matching host pool, when known.
  hsa_amd_memory_pool_t map(hsa_amd_memory_pool_t pool) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    auto it = guest_to_host_.find(pool.handle);
    if (it == guest_to_host_.end())
      return pool;
    return hsa_amd_memory_pool_t{it->second};
  }

  /// @brief Drop cached pool mappings so unload/reload can rediscover them.
  void clear() {
    std::lock_guard lock(mutex_);
    discovered_ = false;
    guest_to_host_.clear();
  }

private:
  /// @brief Memory pool attributes used to match guest and host pools.
  struct PoolInfo {
    hsa_amd_memory_pool_t pool{};
    uint32_t segment = 0;
    uint32_t global_flags = 0;
    bool runtime_alloc_allowed = false;
    uint32_t location = 0;
  };

  /// @brief Collected memory pools for one HSA agent.
  struct PoolList {
    hsa_amd_memory_pool_get_info_fn_t get_info = nullptr;
    std::vector<PoolInfo> pools;
  };

  /// @brief Callback that records a pool and its matching attributes.
  static hsa_status_t collect_pool(hsa_amd_memory_pool_t pool, void *data) {
    auto *list = static_cast<PoolList *>(data);
    if (list == nullptr || list->get_info == nullptr)
      return HSA_STATUS_ERROR;

    PoolInfo info{};
    info.pool = pool;
    if (list->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &info.segment) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;
    (void)list->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &info.global_flags);
    (void)list->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                         &info.runtime_alloc_allowed);
    (void)list->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_LOCATION, &info.location);
    list->pools.push_back(info);
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Return true when guest and host pools are interchangeable for forwarding.
  static bool same_kind(const PoolInfo &guest, const PoolInfo &host) {
    return guest.segment == host.segment && guest.global_flags == host.global_flags &&
           guest.runtime_alloc_allowed == host.runtime_alloc_allowed &&
           guest.location == host.location;
  }

  /// @brief Find the first host pool matching @p guest.
  static std::optional<PoolInfo> find_match(const PoolInfo &guest,
                                            std::span<const PoolInfo> host_pools) {
    for (const PoolInfo &host : host_pools) {
      if (same_kind(guest, host))
        return host;
    }
    return std::nullopt;
  }

  /// @brief Lazily discover guest-to-host memory-pool mappings.
  void ensure_discovered() {
    {
      std::lock_guard lock(mutex_);
      if (discovered_)
        return;
    }

    auto *iterate_pools = layer().amd_agent_iterate_memory_pools();
    auto *get_info = layer().amd_memory_pool_get_info();

    hsa_agent_t guest = AgentMapper::instance().guest_agent();
    hsa_agent_t host = AgentMapper::instance().host_for_guest();
    if (guest.handle == 0 || host.handle == 0)
      return;

    std::unordered_map<uint64_t, uint64_t> discovered;
    bool mapped_any_pool = false;
    if (iterate_pools != nullptr && get_info != nullptr) {
      PoolList guest_pools;
      PoolList host_pools;
      guest_pools.get_info = get_info;
      host_pools.get_info = get_info;
      hsa_status_t guest_status = iterate_pools(guest, collect_pool, &guest_pools);
      hsa_status_t host_status = iterate_pools(host, collect_pool, &host_pools);
      if (guest_status == HSA_STATUS_SUCCESS && host_status == HSA_STATUS_SUCCESS) {
        for (const PoolInfo &guest_pool : guest_pools.pools) {
          std::optional<PoolInfo> host_pool = find_match(guest_pool, host_pools.pools);
          if (!host_pool)
            continue;
          discovered[guest_pool.pool.handle] = host_pool->pool.handle;
          mapped_any_pool = true;
          log_message(kLogDebug, "mapped guest pool=%llu to host pool=%llu",
                      static_cast<unsigned long long>(guest_pool.pool.handle),
                      static_cast<unsigned long long>(host_pool->pool.handle));
        }
      }
    }
    if (!mapped_any_pool) {
      // Pool iteration can fail transiently while the agent mapper is still
      // converging or ROCR is initializing extension state. Do not publish an
      // empty cache: the next mapped-pool call should retry discovery instead
      // of forwarding guest pool handles forever.
      return;
    }

    {
      std::lock_guard lock(mutex_);
      if (discovered_)
        return;
      guest_to_host_ = std::move(discovered);
      discovered_ = true;
    }
  }

  std::mutex mutex_;
  bool discovered_ = false;
  std::unordered_map<uint64_t, uint64_t> guest_to_host_;
};

/// @brief Clear memory-pool mappings when the HSA layer unloads.
void clear_memory_pool_mapper() { MemoryPoolMapper::instance().clear(); }

/// @brief Clear agent mappings when the HSA layer unloads.
void clear_agent_mapper() { AgentMapper::instance().clear(); }

/// @brief Map one execution-facing agent from the visible guest handle to the host handle.
[[nodiscard]] hsa_agent_t mapped_agent(hsa_agent_t agent) {
  return AgentMapper::instance().map(agent);
}

/// @brief Map one guest memory pool to the matching host memory pool.
[[nodiscard]] hsa_amd_memory_pool_t mapped_memory_pool(hsa_amd_memory_pool_t pool) {
  return MemoryPoolMapper::instance().map(pool);
}

/// @brief Function-call argument that must be forwarded as a mapped HSA agent.
struct MappedAgentArg {
  hsa_agent_t value{};
};

/// @brief Function-call argument that must be forwarded as a mapped AMD memory pool.
struct MappedPoolArg {
  hsa_amd_memory_pool_t value{};
};

/// @brief Create a typed forwarding marker for agent arguments.
[[nodiscard]] MappedAgentArg mapped_agent_arg(hsa_agent_t agent) {
  return MappedAgentArg{mapped_agent(agent)};
}

/// @brief Create a typed forwarding marker for memory-pool arguments.
[[nodiscard]] MappedPoolArg mapped_pool_arg(hsa_amd_memory_pool_t pool) {
  return MappedPoolArg{mapped_memory_pool(pool)};
}

/// @brief Unwrap a non-mapped forwarding argument without changing its value category.
template <typename Arg> decltype(auto) forward_amd_arg(Arg &&arg) { return std::forward<Arg>(arg); }

/// @brief Unwrap an explicitly mapped agent marker for AMD-extension forwarding.
inline hsa_agent_t forward_amd_arg(MappedAgentArg arg) { return arg.value; }

/// @brief Unwrap an explicitly mapped pool marker for AMD-extension forwarding.
inline hsa_amd_memory_pool_t forward_amd_arg(MappedPoolArg arg) { return arg.value; }

/// @brief Forward an AMD-extension call after call-site arguments have declared mapping policy.
///
/// @details This helper is intentionally small: each wrapper still names the API
/// semantics and any logging/output rewriting, but execution-facing handles are
/// visibly marked with `mapped_agent_arg` or `mapped_pool_arg` at the call site.
template <typename Original, typename... Args>
hsa_status_t forward_amd_call(Original original, Args &&...args) {
  return original(forward_amd_arg(std::forward<Args>(args))...);
}

hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(code_object, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr && code_object != nullptr) {
    if (!CodeObjectReaderRegistry::instance().store(
            *code_object_reader, static_cast<const uint8_t *>(code_object), size, {})) {
      if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
        (void)original_destroy(*code_object_reader);
      *code_object_reader = {};
      std::fprintf(stderr, "[rocjitsu-hooks] failed to track memory-backed code-object reader\n");
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    log_message(kLogDebug, "registered reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), size);
  }
  return status;
}

hsa_status_t HSA_API rj_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_file();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr) {
    // `hsa_executable_load_agent_code_object` only receives the opaque reader,
    // not the original file descriptor. Copy the file bytes while we still have
    // the descriptor so HIP extension modules that use file-backed readers take
    // the same DBT path as memory-backed framework code objects.
    struct stat statbuf {};
    if (file >= 0 && fstat(file, &statbuf) == 0 && statbuf.st_size > 0) {
      std::shared_ptr<std::vector<uint8_t>> owned;
      try {
        owned = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(statbuf.st_size));
      } catch (const std::bad_alloc &) {
        // The HSA tools ABI cannot propagate C++ allocation failures. Fall
        // through to the existing diagnostic and let ROCR keep the reader.
      }
      if (owned) {
        size_t done = 0;
        while (done < owned->size()) {
          const ssize_t read =
              pread(file, owned->data() + done, owned->size() - done, static_cast<off_t>(done));
          if (read <= 0)
            break;
          done += static_cast<size_t>(read);
        }
        if (done == owned->size() &&
            CodeObjectReaderRegistry::instance().store(*code_object_reader, owned->data(),
                                                       owned->size(), owned)) {
          log_message(kLogDebug, "registered file-backed reader=%llu bytes=%zu",
                      static_cast<unsigned long long>(code_object_reader->handle), owned->size());
          return status;
        }
      }
    }
    std::fprintf(stderr,
                 "[rocjitsu-hooks] file-backed code-object reader=%llu could not be copied for "
                 "DBT translation\n",
                 static_cast<unsigned long long>(code_object_reader->handle));
  }
  return status;
}

hsa_status_t HSA_API rj_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  log_message(kLogVerbose, "reader_destroy reader=%llu",
              static_cast<unsigned long long>(code_object_reader.handle));
  CodeObjectReaderRegistry::instance().remove(code_object_reader);

  auto *original = layer().destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  hsa_status_t status = original(code_object_reader);
  log_message(kLogVerbose, "reader_destroy status=%d", static_cast<int>(status));
  return status;
}

/// @brief Map every agent in a caller-owned array into a temporary vector.
std::vector<hsa_agent_t> map_agent_array(const hsa_agent_t *agents, size_t count) {
  std::vector<hsa_agent_t> mapped;
  if (!agents || count == 0)
    return mapped;
  mapped.reserve(count);
  for (size_t i = 0; i < count; ++i)
    mapped.push_back(AgentMapper::instance().map(agents[i]));
  return mapped;
}

/// @brief Forwardable mapped agent list with duplicate mapped handles removed.
struct MappedAgentList {
  std::vector<hsa_agent_t> agents;
  bool changed = false;
};

/// @brief Map a caller-owned agent array and keep only the first mapped handle.
MappedAgentList map_unique_agent_array(const hsa_agent_t *agents, size_t count) {
  MappedAgentList mapped;
  if (!agents || count == 0)
    return mapped;

  mapped.agents.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    hsa_agent_t agent = AgentMapper::instance().map(agents[i]);
    mapped.changed = mapped.changed || agent.handle != agents[i].handle;

    bool duplicate = false;
    for (hsa_agent_t existing : mapped.agents) {
      if (existing.handle == agent.handle) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      // Guest and host handles may both appear in "all agents" arrays. After
      // mapping the visible guest back to the host, forwarding duplicates can
      // make ROCR reject access setup; retain the first entry's policy.
      mapped.changed = true;
      continue;
    }
    mapped.agents.push_back(agent);
  }
  return mapped;
}

/// @brief Forwardable allow-access agent list plus optional per-agent flags.
struct MappedAccessAgents {
  std::vector<hsa_agent_t> agents;
  std::vector<uint32_t> flags;
  bool changed = false;
};

/// @brief Map and deduplicate agents passed to hsa_amd_agents_allow_access.
MappedAccessAgents map_access_agent_array(const hsa_agent_t *agents, uint32_t count,
                                          const uint32_t *flags) {
  MappedAccessAgents mapped;
  if (!agents || count == 0)
    return mapped;

  mapped.agents.reserve(count);
  if (flags)
    mapped.flags.reserve(count);

  for (uint32_t i = 0; i < count; ++i) {
    hsa_agent_t agent = AgentMapper::instance().map(agents[i]);
    mapped.changed = mapped.changed || agent.handle != agents[i].handle;

    bool duplicate = false;
    for (hsa_agent_t existing : mapped.agents) {
      if (existing.handle == agent.handle) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      // Mapping the visible guest to its host can make an all-agent list contain
      // the selected host twice. Forwarding unique agents avoids asking ROCR to
      // install duplicate access records for the same allocation.
      mapped.changed = true;
      continue;
    }

    mapped.agents.push_back(agent);
    if (flags)
      mapped.flags.push_back(flags[i]);
  }

  return mapped;
}

/// @brief Forwardable virtual-memory access descriptor list.
struct MappedAccessDescs {
  std::vector<hsa_amd_memory_access_desc_t> descs;
  bool changed = false;
};

/// @brief Map vmem access descriptors and deduplicate mapped agent handles.
MappedAccessDescs map_unique_access_descs(const hsa_amd_memory_access_desc_t *desc,
                                          size_t desc_cnt) {
  MappedAccessDescs mapped;
  if (!desc || desc_cnt == 0)
    return mapped;

  mapped.descs.reserve(desc_cnt);
  for (size_t i = 0; i < desc_cnt; ++i) {
    hsa_amd_memory_access_desc_t entry = desc[i];
    entry.agent_handle = mapped_agent(entry.agent_handle);
    mapped.changed = mapped.changed || entry.agent_handle.handle != desc[i].agent_handle.handle;

    bool duplicate = false;
    for (const auto &existing : mapped.descs) {
      if (existing.agent_handle.handle == entry.agent_handle.handle) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      // Keep the first descriptor for a mapped host agent. Supplying the same
      // agent twice is invalid on some ROCR paths even if permissions match.
      mapped.changed = true;
      continue;
    }
    mapped.descs.push_back(entry);
  }
  return mapped;
}

/// @brief Agent fields used by one supported AMD batch-copy operation shape.
struct BatchCopyAgentShape {
  bool src_scalar = false;
  bool dst_scalar = false;
  bool dst_list = false;
};

/// @brief Return the current ROCR-owned agent fields for a batch-copy descriptor.
///
/// @details `hsa_amd_memory_copy_op_t` stores scalar and list fields in unions.
/// Current ROCR uses a scalar `src_agent` for every supported operation and uses
/// `dst_agent_list` for multi-entry/broadcast forms. `src_agent_list` is
/// documented as reserved, so probing it as a pointer misclassifies ordinary
/// scalar source-agent handles as caller-owned arrays.
BatchCopyAgentShape batch_copy_agent_shape(const hsa_amd_memory_copy_op_t &op) {
  switch (static_cast<hsa_amd_memory_copy_op_type_t>(op.type)) {
  case HSA_AMD_MEMORY_COPY_OP_LINEAR:
  case HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP:
  case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC:
  case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST:
  case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST:
    return BatchCopyAgentShape{
        .src_scalar = true,
        .dst_scalar = op.num_entries == 0,
        .dst_list = op.num_entries > 0,
    };
  case HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST:
    return BatchCopyAgentShape{
        .src_scalar = true,
        .dst_scalar = false,
        .dst_list = true,
    };
  }

  // Unknown future operation types have an unknown union contract. Forward them
  // unchanged so the runtime can reject or handle them without rocjitsu guessing.
  return {};
}

/// @brief Remap pointer-info accessible agents to guest handles and remove duplicates.
///
/// @details ROCR owns the returned array and exposes only a mutable pointer plus
/// count. Shrinking the array in place preserves that ownership while avoiding
/// the common host+guest pair collapsing into two copies of the visible guest.
void map_pointer_info_accessible_agents(uint32_t *num_agents_accessible, hsa_agent_t **accessible) {
  if (num_agents_accessible == nullptr || accessible == nullptr || *accessible == nullptr)
    return;

  hsa_agent_t *agents = *accessible;
  uint32_t out_count = 0;
  for (uint32_t i = 0; i < *num_agents_accessible; ++i) {
    hsa_agent_t mapped = AgentMapper::instance().guest_for_host(agents[i]);
    bool duplicate = false;
    for (uint32_t out_idx = 0; out_idx < out_count; ++out_idx) {
      if (agents[out_idx].handle == mapped.handle) {
        duplicate = true;
        break;
      }
    }
    if (duplicate)
      continue;
    agents[out_count++] = mapped;
  }
  *num_agents_accessible = out_count;
}

/// @brief Return true when an AQL packet header names a kernel dispatch.
[[nodiscard]] uint32_t aql_packet_type(uint16_t header) {
  constexpr uint32_t kPacketTypeMask = (1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u;
  return (header >> HSA_PACKET_HEADER_TYPE) & kPacketTypeMask;
}

/// @brief Return true when an AQL packet header names a kernel dispatch.
[[nodiscard]] bool is_kernel_dispatch_packet(const hsa_kernel_dispatch_packet_t &packet) {
  return aql_packet_type(packet.header) == HSA_PACKET_TYPE_KERNEL_DISPATCH;
}

/// @brief Return a copy of @p header with the packet type forced to INVALID.
///
/// @details Used only as a transient "not ready" marker while the ring rewrite
/// mutates a not-yet-doorbelled slot: the header is set to INVALID, the packet
/// body is edited, then the original header is republished. A command processor
/// that observes an INVALID slot treats it as "the producer has not finished
/// writing this packet" and waits (it does NOT skip the slot), so this must only
/// ever be a transient value that is overwritten before the doorbell is rung.
/// HSA_PACKET_TYPE_INVALID is 1 (HSA_PACKET_TYPE_VENDOR_SPECIFIC is 0), so this
/// clears the type field and writes the INVALID code.
[[nodiscard]] uint16_t make_invalid_packet_header(uint16_t header) {
  constexpr uint16_t kTypeMask =
      static_cast<uint16_t>(((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u) << HSA_PACKET_HEADER_TYPE);
  return static_cast<uint16_t>((header & ~kTypeMask) |
                               (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE));
}

/// @brief Queue-doorbell rewrite state for virtual-LDS dispatch selection.
///
/// @details This registry remains in the HSA interposition layer because it
/// coordinates queue creation/destruction, packet-interceptor callbacks,
/// doorbell hooks, the active RjHsaLayer API table, and runtime tracing. The
/// reusable allocation and packet-rewrite primitives live in virtual_lds.h/cpp;
/// moving this class there would make that lower-level module depend on the
/// process-global hook layer and its callback lifecycle.
class VirtualLdsDispatchQueueRegistry {
public:
  static VirtualLdsDispatchQueueRegistry &instance() {
    static VirtualLdsDispatchQueueRegistry registry;
    return registry;
  }

  /// @brief Track a real host queue returned to the application.
  void record_queue(hsa_queue_t *queue, hsa_agent_t host_agent,
                    bool uses_packet_interceptor = false) {
    if (queue == nullptr || queue->base_address == nullptr || queue->size == 0 ||
        queue->doorbell_signal.handle == 0 || host_agent.handle == 0) {
      return;
    }

    QueueState state;
    state.queue = queue;
    state.host_agent = host_agent;
    state.doorbell_signal = queue->doorbell_signal.handle;
    if (auto config = layer().config())
      state.host_lds_bytes = arch_lds_bytes(config->target.arch);
    // host_agent is the mapped agent: for a guest queue rj_queue_create remapped
    // the guest agent to the guest's execution host, so a match here means this is
    // the queue that runs guest work. Only that queue's dispatches are rewritten
    // and only its oversized-no-metadata launches fail closed; unrelated agents'
    // queues are tracked for doorbell forwarding but otherwise left untouched.
    const hsa_agent_t guest_host = AgentMapper::instance().host_for_guest();
    state.is_guest_queue = guest_host.handle != 0 && host_agent.handle == guest_host.handle;
    state.uses_packet_interceptor = uses_packet_interceptor;
    state.slots.resize(queue->size);

    std::lock_guard lock(mutex_);
    // Intercept queues are rewritten synchronously in the ROCR packet callback.
    // The polling scanner is only a fallback for non-intercept queues that use
    // the normal doorbell path; starting it for intercept-only workloads leaves
    // an unnecessary background reader active during process teardown.
    if (!uses_packet_interceptor)
      ensure_scanner_started_locked();
    const uint64_t queue_key = reinterpret_cast<uintptr_t>(queue);
    auto old = queues_by_ptr_.find(queue_key);
    if (old != queues_by_ptr_.end()) {
      queues_by_doorbell_.erase(old->second.doorbell_signal);
      release_queue(old->second);
    }
    const uint32_t host_lds_bytes = state.host_lds_bytes;
    queues_by_doorbell_[state.doorbell_signal] = queue_key;
    queues_by_ptr_[queue_key] = std::move(state);
    log_message(kLogDebug, "tracked queue=%p doorbell=%llu size=%u host_agent=%llu lds_limit=%u",
                static_cast<void *>(queue),
                static_cast<unsigned long long>(queue->doorbell_signal.handle), queue->size,
                static_cast<unsigned long long>(host_agent.handle), host_lds_bytes);
    trace_virtual_lds_dispatch(
        "tracked queue=%p type=%u size=%u doorbell=%llu host_agent=%llu lds_limit=%u "
        "interceptor=%d "
        "load_write_index=%d",
        static_cast<void *>(queue), static_cast<unsigned>(queue->type), queue->size,
        static_cast<unsigned long long>(queue->doorbell_signal.handle),
        static_cast<unsigned long long>(host_agent.handle), host_lds_bytes, uses_packet_interceptor,
        layer().queue_load_write_index_relaxed() != nullptr);
  }

  /// @brief Release all slot-owned resources for a queue being destroyed.
  void erase_queue(hsa_queue_t *queue) {
    if (queue == nullptr)
      return;
    std::lock_guard lock(mutex_);
    const uint64_t queue_key = reinterpret_cast<uintptr_t>(queue);
    auto it = queues_by_ptr_.find(queue_key);
    if (it == queues_by_ptr_.end())
      return;
    queues_by_doorbell_.erase(it->second.doorbell_signal);
    release_queue(it->second);
    queues_by_ptr_.erase(it);
  }

  /// @brief Stop retaining an application signal immediately before its destruction.
  ///
  /// @details The dispatch packet borrows an application-supplied completion
  /// signal; ownership never transfers to rocjitsu. A conforming application may
  /// destroy that signal as soon as it observes dispatch completion, before a
  /// later packet gives us another opportunity to poll it. Detach every matching
  /// borrowed handle while it is still valid and record completion explicitly.
  ///
  /// Scanner slot buffers remain in place until their AQL slot is reused because
  /// the slot record also prevents the scanner from rewriting a stale completed
  /// packet. Interceptor buffers are already retired, so reap them immediately.
  void retire_destroyed_completion_signal(hsa_signal_t signal) {
    if (signal.handle == 0)
      return;

    // O(queues x slots) per hsa_signal_destroy. Acceptable because virtual-LDS
    // dispatches (the only ones that borrow a completion signal) are rare relative
    // to plain signal churn, and the scan is bounded by the queue ring size. If a
    // workload ever makes this hot, index borrowed signals by handle so destroy
    // becomes O(1); today the simpler full sweep keeps the borrow bookkeeping in
    // one place.
    std::lock_guard lock(mutex_);
    for (auto &[queue_key, state] : queues_by_ptr_) {
      (void)queue_key;
      for (VirtualLdsDispatchBuffers &buffers : state.slots)
        mark_destroyed_completion_signal(buffers, signal);
      for (VirtualLdsDispatchBuffers &buffers : state.retired_buffers)
        mark_destroyed_completion_signal(buffers, signal);
      release_completed_retired_buffers(state);
    }
  }

  /// @brief Rewrite any kernel dispatch packet made visible by this doorbell.
  void rewrite_before_doorbell(hsa_signal_t signal, hsa_signal_value_t value) {
    if (value < 0)
      return;
    std::lock_guard lock(mutex_);
    auto doorbell_it = queues_by_doorbell_.find(signal.handle);
    static std::atomic<uint32_t> doorbell_trace_count{0};
    if (doorbell_trace_count.fetch_add(1, std::memory_order_relaxed) < 128) {
      trace_virtual_lds_dispatch("doorbell signal=%llu value=%lld tracked=%d",
                                 static_cast<unsigned long long>(signal.handle),
                                 static_cast<long long>(value),
                                 doorbell_it != queues_by_doorbell_.end());
    }
    if (doorbell_it == queues_by_doorbell_.end())
      return;
    auto queue_it = queues_by_ptr_.find(doorbell_it->second);
    if (queue_it == queues_by_ptr_.end())
      return;

    QueueState &state = queue_it->second;
    if (state.queue == nullptr || state.queue->base_address == nullptr || state.queue->size == 0)
      return;
    if (state.uses_packet_interceptor)
      return;

    const uint64_t packet_id = static_cast<uint64_t>(value);
    // A producer (single- OR multi-producer) can reserve and publish several
    // consecutive packets and then ring once with the final packet ID. Rewriting
    // only the doorbell-named packet would let an earlier still-unrewritten
    // predecessor (e.g. a virtual-LDS dispatch at next_packet_id) reach the CP
    // unrevised, and advancing next_packet_id past it would make the scanner skip
    // it too. So rewrite the whole newly-published range [next_packet_id,
    // packet_id]. note_packet_ready advances the contiguous frontier and remembers
    // ready-but-non-contiguous successors, so an out-of-order producer that leaves
    // an unready hole does not advance the cursor past it -- but once the hole
    // closes the cursor catches up across every already-ready successor instead of
    // stranding below the ready suffix. The range must fit the ring to bound the
    // scan.
    if (packet_id >= state.next_packet_id && packet_id - state.next_packet_id < state.queue->size) {
      rewrite_packet_range(state, state.next_packet_id, packet_id + 1, true);
      return;
    }

    // Doorbell ID below the frontier (a lagging out-of-order ring) or a jump
    // larger than the ring: rewrite at least the named packet. The background
    // scanner covers any packets between the frontier and this one.
    const bool ready = rewrite_packet(state, packet_id);
    if (ready && packet_id >= state.next_packet_id) {
      // A jump larger than the ring means the intervening packets have wrapped
      // (their slots reused), so step the cursor past this packet directly and
      // discard any now-stale ready_ahead entries below the new frontier rather
      // than trying to make the cursor contiguous across the wrapped gap.
      state.next_packet_id = packet_id + 1;
      state.ready_ahead.erase(state.ready_ahead.begin(),
                              state.ready_ahead.lower_bound(state.next_packet_id));
    }
  }

  /// @brief Release all tracked queues during HSA tool unload.
  void clear() {
    // Stop the scanner and hand its jthread to a local, then JOIN WITH NO LOCK
    // HELD. The scanner thread takes mutex_ inside scan_ready_packets(), and a
    // concurrent record_queue() holds mutex_ while it waits on
    // scanner_lifecycle_mutex_ in ensure_scanner_started_locked(). Joining while
    // holding either mutex_ or scanner_lifecycle_mutex_ would therefore close a
    // deadlock cycle (clear waits for the scanner to exit, the scanner waits for
    // mutex_, record_queue holds mutex_ and waits for the lifecycle mutex). Set
    // stopping_ and request_stop under the lifecycle mutex so record_queue cannot
    // resurrect the thread, move the handle out, release the lock, then join.
    std::jthread scanner_to_join;
    {
      std::lock_guard lifecycle(scanner_lifecycle_mutex_);
      stopping_ = true;
      scanner_.request_stop();
      scanner_to_join = std::move(scanner_);
    }
    if (scanner_to_join.joinable())
      scanner_to_join.join();

    std::lock_guard lock(mutex_);
    for (auto &[queue, state] : queues_by_ptr_) {
      (void)queue;
      release_queue(state);
    }
    queues_by_ptr_.clear();
    queues_by_doorbell_.clear();
    VirtualLdsDispatchAllocator::instance().clear();
    // Allow the scanner to start again if the registry is reused after a clear
    // (e.g. an in-process reload). Safe under mutex_ because start is gated by
    // stopping_ under the lifecycle mutex.
    {
      std::lock_guard lifecycle(scanner_lifecycle_mutex_);
      stopping_ = false;
    }
  }

  /// @brief Rewrite a contiguous packet batch delivered by ROCR's intercept queue.
  void rewrite_intercept_packets(hsa_queue_t *queue, const void *pkts, uint64_t pkt_count,
                                 uint64_t user_pkt_index,
                                 hsa_amd_queue_intercept_packet_writer_t writer) {
    if (writer == nullptr || pkts == nullptr || pkt_count == 0)
      return;
    static_assert(sizeof(hsa_kernel_dispatch_packet_t) == 64,
                  "AQL packets must remain 64 bytes for intercept rewriting");

    std::vector<hsa_kernel_dispatch_packet_t> packets(static_cast<size_t>(pkt_count));
    std::memcpy(packets.data(), pkts, packets.size() * sizeof(hsa_kernel_dispatch_packet_t));

    {
      std::lock_guard lock(mutex_);
      auto queue_it = queues_by_ptr_.find(reinterpret_cast<uintptr_t>(queue));
      if (queue_it == queues_by_ptr_.end()) {
        writer(packets.data(), pkt_count);
        return;
      }

      QueueState &state = queue_it->second;
      // Reclaim buffers from earlier intercept dispatches whose completion signal
      // has fired. The intercept path has no slot bookkeeping and no scanner (the
      // scanner skips intercept-managed queues), so without a sweep here a queue
      // that issues virtual-LDS dispatches accumulates retired_buffers until the
      // queue is destroyed. Sweeping at batch entry bounds retention to "until the
      // next dispatch on this queue", matching the slot path.
      release_completed_retired_buffers(state);
      trace_virtual_lds_dispatch("intercept batch queue=%p begin=%llu count=%llu",
                                 static_cast<void *>(queue),
                                 static_cast<unsigned long long>(user_pkt_index),
                                 static_cast<unsigned long long>(pkt_count));
      for (uint64_t index = 0; index < pkt_count; ++index) {
        hsa_kernel_dispatch_packet_t &packet = packets[static_cast<size_t>(index)];
        if (!is_kernel_dispatch_packet(packet))
          continue;
        ensure_intercept_packet_private_size(packet, user_pkt_index + index);
        const uint64_t packet_id = user_pkt_index + index;
        auto prepared = prepare_virtual_lds_dispatch(state, packet, packet_id, "intercept");
        if (!prepared) {
          // nullopt here means KeepNormal: no virtual-LDS plan or a below-threshold
          // launch whose normal descriptor is safe. Any sidecar-required dispatch
          // that could not be prepared aborts inside prepare_virtual_lds_dispatch,
          // so this never silently submits an oversized normal packet. (Do NOT
          // publish an INVALID header: the command processor treats INVALID as
          // "producer has not finished writing this slot" and stalls the queue.)
          continue;
        }
        apply_virtual_lds_dispatch(packet, *prepared);
        trace_prepared_virtual_lds_dispatch(*prepared, packet_id, "intercept");
        state.retired_buffers.push_back(std::exchange(prepared->buffers, {}));
      }
    }

    writer(packets.data(), pkt_count);
  }

private:
  struct QueueState {
    hsa_queue_t *queue = nullptr;
    hsa_agent_t host_agent{};
    uint64_t doorbell_signal = 0;
    uint64_t next_packet_id = 0;
    // Packet IDs observed ready at or beyond next_packet_id but not yet contiguous
    // with the cursor (an out-of-order producer rang a later packet while an
    // earlier hole was still INVALID). When the hole closes, note_packet_ready
    // drains the contiguous run from here so the cursor catches up across every
    // already-ready successor instead of stranding below the ready suffix. Bounded
    // by the ring size: an id leaves the set as soon as the cursor passes it.
    std::set<uint64_t> ready_ahead;
    uint32_t host_lds_bytes = 0;
    bool uses_packet_interceptor = false;
    // True only for the queue running guest work on the guest's execution host
    // agent. host_lds_bytes is derived from the configured host target arch, so
    // its limit is only meaningful for this queue. Other agents' queues are
    // tracked (so their doorbell stores still forward) but their dispatches are
    // never rewritten and an oversized packet on them is the driver's concern.
    bool is_guest_queue = false;
    std::vector<VirtualLdsDispatchBuffers> slots;
    std::vector<VirtualLdsDispatchBuffers> retired_buffers;
  };

  /// @brief Delivery-independent result of preparing one virtual-LDS dispatch.
  ///
  /// @details Packet discovery paths have different publication rules, but the
  /// sidecar decision, allocation, wrapper construction, and completion lifetime
  /// are identical. Keeping those decisions here prevents the intercept and ring
  /// paths from drifting as virtual-LDS metadata evolves.
  struct PreparedVirtualLdsDispatch {
    std::string kernel_name;
    uint64_t original_kernel_object = 0;
    uint64_t virtual_kernel_object = 0;
    uint64_t requested_lds = 0;
    uint32_t original_group_segment_size = 0;
    uint32_t original_private_segment_size = 0;
    uint32_t virtual_private_segment_size = 0;
    VirtualLdsDispatchGeometry geometry;
    uint64_t state_address = 0;
    hsa_signal_t completion_signal{};
    VirtualLdsDispatchBuffers buffers;
  };

  static void release_queue(QueueState &state) {
    for (VirtualLdsDispatchBuffers &buffers : state.slots)
      release_virtual_lds_buffers(buffers);
    for (VirtualLdsDispatchBuffers &buffers : state.retired_buffers)
      release_virtual_lds_buffers(buffers);
    state.retired_buffers.clear();
  }

  static void release_completed_retired_buffers(QueueState &state) {
    auto out = state.retired_buffers.begin();
    for (auto it = state.retired_buffers.begin(); it != state.retired_buffers.end(); ++it) {
      if (completed_virtual_lds_dispatch(*it)) {
        release_virtual_lds_buffers(*it);
        continue;
      }
      if (out != it)
        *out = *it;
      ++out;
    }
    state.retired_buffers.erase(out, state.retired_buffers.end());
  }

  /// @brief Detach one borrowed completion signal that the application is destroying.
  static void mark_destroyed_completion_signal(VirtualLdsDispatchBuffers &buffers,
                                               hsa_signal_t signal) {
    // Internally-created signals are destroyed only by release_virtual_lds_buffers
    // through the saved original HSA entry point, so they never enter this hook.
    // Keep the ownership check explicit in case a handle is exposed unexpectedly.
    if (buffers.owns_completion_signal || buffers.completion_signal.handle != signal.handle)
      return;

    buffers.completion_signal = {};
    buffers.completion_signal_was_pending = false;
    buffers.completion_signal_destroyed = true;
  }

  /// @brief Return true when a queue's host architecture has a modeled LDS limit.
  [[nodiscard]] static bool has_host_lds_limit(const QueueState &state) {
    return state.host_lds_bytes != 0;
  }

  /// @brief Return true when @p bytes exceeds the queue host's modeled LDS limit.
  [[nodiscard]] static bool exceeds_host_lds_limit(const QueueState &state, uint64_t bytes) {
    return has_host_lds_limit(state) && bytes > state.host_lds_bytes;
  }

  static void retire_slot_buffers(QueueState &state, uint32_t slot) {
    VirtualLdsDispatchBuffers &slot_buffers = state.slots[slot];
    if (empty_virtual_lds_buffers(slot_buffers))
      return;
    // AQL queue slots may be reused before the previously dispatched kernel has
    // retired. Move old virtual-LDS backing allocations to the retired list and
    // let completion-signal cleanup release them once CP has finished the
    // dispatch. Signal-less packets remain live until queue destruction.
    state.retired_buffers.push_back(slot_buffers);
    slot_buffers = {};
  }

  static void publish_packet_header(hsa_kernel_dispatch_packet_t &packet, uint16_t header) {
    std::atomic_ref<uint16_t>(packet.header).store(header, std::memory_order_release);
  }

  [[nodiscard]] static uint32_t descriptor_private_segment_size(uint64_t kernel_object) {
    return VirtualLdsRuntimeRegistry::instance().private_segment_size_for_object(kernel_object);
  }

  static void ensure_intercept_packet_private_size(hsa_kernel_dispatch_packet_t &packet,
                                                   uint64_t packet_id) {
    const uint32_t descriptor_private = descriptor_private_segment_size(packet.kernel_object);
    if (descriptor_private <= packet.private_segment_size)
      return;

    // Unlike ensure_queue_packet_private_size (the ring path), no INVALID header
    // toggle is needed here: the intercept path edits a private std::vector copy
    // that is handed to the CP via writer(...) only after the whole batch is
    // rewritten, so no consumer can observe a half-written packet. Do not add a
    // header fence to "match" the ring path.
    const uint32_t original_private = packet.private_segment_size;
    packet.private_segment_size = descriptor_private;
    trace_virtual_lds_dispatch("patched intercept private packet=%llu object=0x%llx private=%u->%u",
                               static_cast<unsigned long long>(packet_id),
                               static_cast<unsigned long long>(packet.kernel_object),
                               original_private, descriptor_private);
  }

  static void ensure_queue_packet_private_size(hsa_kernel_dispatch_packet_t &packet,
                                               uint64_t packet_id, uint32_t slot, uint16_t header) {
    const uint32_t descriptor_private = descriptor_private_segment_size(packet.kernel_object);
    if (descriptor_private <= packet.private_segment_size)
      return;

    const uint32_t original_private = packet.private_segment_size;
    // Mark the slot not-ready (INVALID) while mutating the body, then republish
    // the original header. Header 0 would be VENDOR_SPECIFIC — a processable
    // type — so a consumer could act on a half-written packet; INVALID tells the
    // consumer the producer has not finished. Route every transient header
    // toggle through make_invalid_packet_header for consistency with the ring
    // rewrite path.
    publish_packet_header(packet, make_invalid_packet_header(header));
    packet.private_segment_size = descriptor_private;
    publish_packet_header(packet, header);
    trace_virtual_lds_dispatch(
        "patched packet private packet=%llu slot=%u object=0x%llx private=%u->%u",
        static_cast<unsigned long long>(packet_id), slot,
        static_cast<unsigned long long>(packet.kernel_object), original_private,
        descriptor_private);
  }

  static void note_packet_ready(QueueState &state, uint64_t packet_id, bool ready) {
    // Only packets at or beyond the contiguous cursor can advance it; a ready
    // packet below it was already accounted for.
    if (!ready || packet_id < state.next_packet_id)
      return;

    // Remember this packet as ready even if it is not the one the cursor is
    // waiting for. An out-of-order producer can ring a later packet while an
    // earlier hole is still INVALID, so the cursor must be able to catch up across
    // this successor once the hole closes -- otherwise it strands below the
    // already-ready suffix and a later doorbell range can skip a packet.
    if (packet_id != state.next_packet_id) {
      state.ready_ahead.insert(packet_id);
      return;
    }

    // The cursor's packet is ready: advance across it and every contiguous
    // successor already observed ready in a prior doorbell.
    ++state.next_packet_id;
    auto it = state.ready_ahead.begin();
    while (it != state.ready_ahead.end() && *it == state.next_packet_id) {
      ++state.next_packet_id;
      it = state.ready_ahead.erase(it);
    }
  }

  [[nodiscard]] static VirtualLdsDispatchState
  make_virtual_lds_dispatch_state(uint64_t backing_address,
                                  const VirtualLdsDispatchGeometry &geometry) {
    return VirtualLdsDispatchState{.backing_base = backing_address,
                                   .stride_x = geometry.stride_x,
                                   .stride_y = geometry.stride_y,
                                   .stride_z = geometry.stride_z};
  }

  [[nodiscard]] static bool
  write_virtual_lds_kernarg_wrapper(void *wrapper, const KernargExtensionLayout &layout,
                                    const hsa_kernel_dispatch_packet_t &packet,
                                    const VirtualLdsDispatchState &dispatch_state) {
    if (wrapper == nullptr)
      return false;
    const uint64_t original_kernarg_pointer = reinterpret_cast<uintptr_t>(packet.kernarg_address);
    const KernargExtensionPayloadWrite payload{
        .data = &dispatch_state,
        .size = static_cast<uint32_t>(sizeof(dispatch_state)),
    };
    return write_kernarg_extension_wrapper(
        std::span<uint8_t>(static_cast<uint8_t *>(wrapper), layout.wrapper_size), layout,
        packet.kernarg_address, original_kernarg_pointer, std::span{&payload, 1});
  }

  /// @brief Abort the process after a virtual-LDS sidecar dispatch cannot be prepared.
  ///
  /// @details Once plan_virtual_lds_dispatch selects UseSidecar (or Reject), the
  /// requested LDS exceeds the host limit, so the normal descriptor is unsafe for
  /// THIS dispatch even when it is a live (non-stubbed) descriptor -- e.g. a kernel
  /// whose static LDS fits but whose dynamic (packet) group segment overflows.
  /// Falling back to the normal descriptor would submit a host-faulting oversized
  /// launch, and a persistent INVALID header would stall the queue forever. Neither
  /// is acceptable, and there is no per-dispatch recovery, so fail loudly and stop.
  [[noreturn]] static void abort_virtual_lds_dispatch(const char *reason, const char *delivery,
                                                      uint64_t packet_id, const char *kernel_name,
                                                      uint64_t requested_lds,
                                                      uint32_t host_lds_bytes) {
    std::fprintf(stderr,
                 "[rocjitsu-hooks] fatal: virtual-LDS sidecar dispatch could not be prepared "
                 "(%s) delivery=%s packet=%llu kernel=%s requested_lds=%llu host_lds=%u; the "
                 "normal descriptor is oversized for this launch and cannot be dispatched\n",
                 reason, delivery, static_cast<unsigned long long>(packet_id),
                 kernel_name ? kernel_name : "<unknown>",
                 static_cast<unsigned long long>(requested_lds), host_lds_bytes);
    std::abort();
  }

  /// @brief Prepare a virtual-LDS rewrite without publishing packet fields.
  ///
  /// @details This is the single policy path shared by ROCR intercept batches and
  /// direct queue scanning. The returned buffers transfer to the delivery adapter,
  /// which publishes fields using the synchronization required by its packet source.
  /// A nullopt result means "dispatch the packet on its normal descriptor" and is
  /// returned ONLY when the normal descriptor is safe: there is no virtual-LDS plan
  /// and the packet fits the host LDS limit, or the request is below threshold
  /// (KeepNormal). Any case where the normal descriptor would be oversized -- an
  /// unregistered kernel whose group segment exceeds the host limit, or a plan that
  /// selects a sidecar (UseSidecar/Reject) followed by a preparation failure --
  /// calls abort_virtual_lds_dispatch rather than returning nullopt, so it never
  /// silently falls back to a host-faulting launch.
  [[nodiscard]] static std::optional<PreparedVirtualLdsDispatch>
  prepare_virtual_lds_dispatch(QueueState &state, const hsa_kernel_dispatch_packet_t &packet,
                               uint64_t packet_id, const char *delivery) {
    auto resolved =
        VirtualLdsRuntimeRegistry::instance().find_by_kernel_object(packet.kernel_object);
    if (!resolved) {
      // No virtual-LDS variant for this kernel object. On the guest's own queue a
      // group segment exceeding the host LDS limit has no sidecar to fall back to
      // and the normal descriptor would fault the host, so fail closed (consistent
      // with the post-plan paths). host_lds_bytes is derived from the configured
      // host target arch, so this check is only meaningful for the guest queue; an
      // unrelated agent with a larger real LDS capacity must keep its packet
      // unchanged.
      if (state.is_guest_queue && exceeds_host_lds_limit(state, packet.group_segment_size)) {
        abort_virtual_lds_dispatch("kernel has no virtual-LDS variant but its group segment "
                                   "exceeds the host LDS limit",
                                   delivery, packet_id, "<unregistered>", packet.group_segment_size,
                                   state.host_lds_bytes);
      }
      return std::nullopt;
    }

    const rocjitsu::hooks::VirtualLdsKernelRuntimeMetadata &metadata = resolved->metadata;
    const VirtualLdsDispatchPacketFields packet_fields{
        .group_segment_size = packet.group_segment_size,
        .private_segment_size = packet.private_segment_size,
        .workgroup_size_x = packet.workgroup_size_x,
        .workgroup_size_y = packet.workgroup_size_y,
        .workgroup_size_z = packet.workgroup_size_z,
        .grid_size_x = packet.grid_size_x,
        .grid_size_y = packet.grid_size_y,
        .grid_size_z = packet.grid_size_z,
    };
    const VirtualLdsDispatchKernelFacts kernel_facts{
        .static_lds_bytes = metadata.static_lds_bytes,
        .normal_private_segment_size = metadata.normal_private_segment_size,
        .virtual_private_segment_size = metadata.virtual_private_segment_size,
        .has_workgroup_id_x = (metadata.flags & kVirtualLdsFlagWorkgroupIdX) != 0,
        .has_workgroup_id_y = (metadata.flags & kVirtualLdsFlagWorkgroupIdY) != 0,
        .has_workgroup_id_z = (metadata.flags & kVirtualLdsFlagWorkgroupIdZ) != 0,
    };
    const auto rewrite =
        plan_virtual_lds_dispatch(packet_fields, kernel_facts, state.host_lds_bytes);
    if (rewrite.decision == VirtualLdsDispatchDecision::KeepNormal) {
      trace_virtual_lds_dispatch(
          "virtual-LDS below threshold delivery=%s packet=%llu kernel=%s static=%u "
          "packet_group=%u requested=%llu",
          delivery, static_cast<unsigned long long>(packet_id), metadata.kernel_name.c_str(),
          metadata.static_lds_bytes, packet.group_segment_size,
          static_cast<unsigned long long>(rewrite.requested_lds));
      return std::nullopt;
    }
    if (rewrite.decision == VirtualLdsDispatchDecision::Reject) {
      // A dispatch-time-only rejection (per-packet grid geometry overflowing the
      // runtime-state ABI). The load-time check cannot see packet dimensions, so
      // this is reachable. The request exceeds the host LDS limit, so the normal
      // descriptor cannot service this launch either -- abort rather than fault.
      abort_virtual_lds_dispatch(rewrite.diagnostic.c_str(), delivery, packet_id,
                                 metadata.kernel_name.c_str(), rewrite.requested_lds,
                                 state.host_lds_bytes);
    }

    if ((metadata.flags & kVirtualLdsFlagRuntimeStateBlock) == 0 ||
        (metadata.kernarg_size != 0 && packet.kernarg_address == nullptr)) {
      // Sidecar is required (requested LDS exceeds host) but the runtime-state ABI
      // is missing or the app supplied no kernarg buffer to wrap. These are fixed
      // per kernel and validated at load, but the null-kernarg case is packet
      // data; either way the normal descriptor is oversized, so fail closed.
      abort_virtual_lds_dispatch(
          (metadata.flags & kVirtualLdsFlagRuntimeStateBlock) == 0
              ? "kernel lacks the runtime-state-block ABI required for a sidecar dispatch"
              : "dispatch supplied no kernarg buffer to wrap for the sidecar",
          delivery, packet_id, metadata.kernel_name.c_str(), rewrite.requested_lds,
          state.host_lds_bytes);
    }
    const KernargExtensionPayloadLayout payload{
        .size = static_cast<uint32_t>(sizeof(VirtualLdsDispatchState)),
        .alignment = alignof(uint64_t),
    };
    const auto wrapper_layout =
        make_kernarg_extension_layout(metadata.kernarg_size, std::span{&payload, 1});
    if (!wrapper_layout || wrapper_layout->payload_offsets.size() != 1 ||
        wrapper_layout->payload_offsets.front() != metadata.backing_pointer_kernarg_offset) {
      // The wrapper layout is fixed per kernel and validated at load; a mismatch
      // here means the descriptor and metadata disagree, so the sidecar cannot be
      // built and the normal descriptor is oversized for this launch.
      abort_virtual_lds_dispatch("sidecar kernarg wrapper layout does not match the descriptor",
                                 delivery, packet_id, metadata.kernel_name.c_str(),
                                 rewrite.requested_lds, state.host_lds_bytes);
    }

    release_completed_retired_buffers(state);
    VirtualLdsDispatchBuffers buffers;
    if (!VirtualLdsDispatchAllocator::instance().allocate(state.host_agent,
                                                          rewrite.geometry.backing_bytes,
                                                          wrapper_layout->wrapper_size, buffers)) {
      // Backing/kernarg allocation is a genuinely dispatch-time-only failure (OOM
      // or pool discovery). There is no per-dispatch recovery and the normal
      // descriptor is oversized, so abort rather than fault the GPU.
      abort_virtual_lds_dispatch("sidecar backing/kernarg allocation failed", delivery, packet_id,
                                 metadata.kernel_name.c_str(), rewrite.requested_lds,
                                 state.host_lds_bytes);
    }

    const VirtualLdsDispatchState dispatch_state = make_virtual_lds_dispatch_state(
        reinterpret_cast<uintptr_t>(buffers.backing), rewrite.geometry);
    if (!write_virtual_lds_kernarg_wrapper(buffers.kernarg, *wrapper_layout, packet,
                                           dispatch_state)) {
      release_virtual_lds_buffers(buffers);
      abort_virtual_lds_dispatch("sidecar kernarg wrapper could not be written", delivery,
                                 packet_id, metadata.kernel_name.c_str(), rewrite.requested_lds,
                                 state.host_lds_bytes);
    }
    trace_virtual_lds_kernarg(packet_id, buffers.kernarg, metadata.kernarg_size);

    // Completion-signal creation edits the packet. Do it on a private copy so
    // queue-backed callers can publish every changed field under an invalid header.
    hsa_kernel_dispatch_packet_t packet_with_signal = packet;
    buffers.virtual_kernel_object = resolved->virtual_kernel_object;
    record_virtual_lds_completion_signal(buffers, packet_with_signal);

    return PreparedVirtualLdsDispatch{
        .kernel_name = metadata.kernel_name,
        .original_kernel_object = packet.kernel_object,
        .virtual_kernel_object = resolved->virtual_kernel_object,
        .requested_lds = rewrite.requested_lds,
        .original_group_segment_size = packet.group_segment_size,
        .original_private_segment_size = packet.private_segment_size,
        .virtual_private_segment_size = rewrite.private_segment_size,
        .geometry = rewrite.geometry,
        .state_address = reinterpret_cast<uintptr_t>(static_cast<uint8_t *>(buffers.kernarg) +
                                                     metadata.backing_pointer_kernarg_offset),
        .completion_signal = packet_with_signal.completion_signal,
        .buffers = buffers,
    };
  }

  /// @brief Publish the fields owned by virtual-LDS dispatch rewriting.
  static void apply_virtual_lds_dispatch(hsa_kernel_dispatch_packet_t &packet,
                                         const PreparedVirtualLdsDispatch &prepared) {
    packet.kernel_object = prepared.virtual_kernel_object;
    packet.kernarg_address = prepared.buffers.kernarg;
    packet.group_segment_size = 0;
    packet.private_segment_size = prepared.virtual_private_segment_size;
    packet.completion_signal = prepared.completion_signal;
  }

  static void trace_prepared_virtual_lds_dispatch(const PreparedVirtualLdsDispatch &prepared,
                                                  uint64_t packet_id, const char *delivery) {
    trace_virtual_lds_dispatch(
        "rewrote delivery=%s packet=%llu kernel=%s original_object=0x%llx virtual_object=0x%llx "
        "packet_group=%u private=%u->%u requested=%llu groups=%u,%u,%u strides=%u,%u,%u "
        "backing_bytes=%zu kernarg=%p backing=%p state=0x%llx",
        delivery, static_cast<unsigned long long>(packet_id), prepared.kernel_name.c_str(),
        static_cast<unsigned long long>(prepared.original_kernel_object),
        static_cast<unsigned long long>(prepared.virtual_kernel_object),
        prepared.original_group_segment_size, prepared.original_private_segment_size,
        prepared.virtual_private_segment_size,
        static_cast<unsigned long long>(prepared.requested_lds), prepared.geometry.groups_x,
        prepared.geometry.groups_y, prepared.geometry.groups_z, prepared.geometry.stride_x,
        prepared.geometry.stride_y, prepared.geometry.stride_z, prepared.geometry.backing_bytes,
        prepared.buffers.kernarg, prepared.buffers.backing,
        static_cast<unsigned long long>(prepared.state_address));
  }

  static void rewrite_packet_range(QueueState &state, uint64_t begin_packet_id,
                                   uint64_t end_packet_id, bool advance_contiguous_cursor) {
    // The HSA queue can contain invalid/no-op holes before later ready dispatches.
    // Keep scanning the visible range so an oversized-LDS kernel is not missed behind
    // such a hole, but only advance the contiguous cursor across packets that were
    // actually ready when observed.
    for (uint64_t packet_id = begin_packet_id; packet_id < end_packet_id; ++packet_id) {
      const bool ready = rewrite_packet(state, packet_id);
      if (advance_contiguous_cursor)
        note_packet_ready(state, packet_id, ready);
    }
  }

  /// @returns true when the packet header is ready and the scanner may advance.
  static bool rewrite_packet(QueueState &state, uint64_t packet_id) {
    const uint32_t slot = static_cast<uint32_t>(packet_id % state.queue->size);
    VirtualLdsDispatchBuffers &slot_buffers = state.slots[slot];

    auto *packets = static_cast<hsa_kernel_dispatch_packet_t *>(state.queue->base_address);
    hsa_kernel_dispatch_packet_t &packet = packets[slot];
    const uint16_t header =
        std::atomic_ref<uint16_t>(packet.header).load(std::memory_order_acquire);
    const uint32_t type = aql_packet_type(header);
    if (type == HSA_PACKET_TYPE_INVALID)
      return false;

    if (!is_kernel_dispatch_packet(packet)) {
      retire_slot_buffers(state, slot);
      return true;
    }

    if (slot_buffers.kernarg != nullptr && packet.kernarg_address == slot_buffers.kernarg)
      return true;

    retire_slot_buffers(state, slot);
    ensure_queue_packet_private_size(packet, packet_id, slot, header);
    auto prepared = prepare_virtual_lds_dispatch(state, packet, packet_id, "queue");
    if (!prepared) {
      // nullopt here means KeepNormal: no virtual-LDS plan or a below-threshold
      // launch whose normal descriptor is safe. Any sidecar-required dispatch that
      // could not be prepared aborts inside prepare_virtual_lds_dispatch, so this
      // never silently submits an oversized normal packet. Do NOT publish an
      // INVALID header: the command processor treats INVALID as "producer has not
      // finished writing this slot" and stalls the queue permanently.
      return true;
    }
    publish_packet_header(packet, make_invalid_packet_header(header));
    apply_virtual_lds_dispatch(packet, *prepared);
    publish_packet_header(packet, header);
    trace_prepared_virtual_lds_dispatch(*prepared, packet_id, "queue");
    slot_buffers = std::exchange(prepared->buffers, {});
    return true;
  }

  void scan_ready_packets() {
    auto *load_write_index = layer().queue_load_write_index_relaxed();
    if (load_write_index == nullptr) {
      static std::atomic<bool> reported{false};
      bool expected = false;
      if (reported.compare_exchange_strong(expected, true, std::memory_order_relaxed))
        trace_virtual_lds_dispatch("scanner disabled: hsa_queue_load_write_index_relaxed missing");
      return;
    }

    std::lock_guard lock(mutex_);
    for (auto &[queue_key, state] : queues_by_ptr_) {
      (void)queue_key;
      if (state.queue == nullptr || state.queue->base_address == nullptr || state.queue->size == 0)
        continue;
      if (state.uses_packet_interceptor)
        continue;
      const uint64_t write_index = load_write_index(state.queue);
      if (write_index != 0) {
        static std::atomic<uint32_t> scan_trace_count{0};
        if (scan_trace_count.fetch_add(1, std::memory_order_relaxed) < 128) {
          trace_virtual_lds_dispatch("scan queue=%p next=%llu write=%llu",
                                     static_cast<void *>(state.queue),
                                     static_cast<unsigned long long>(state.next_packet_id),
                                     static_cast<unsigned long long>(write_index));
        }
      }
      // Only rewrite packets the doorbell hook has not already published to the
      // command processor. rewrite_before_doorbell() advances next_packet_id past
      // every slot it rewrote BEFORE the original doorbell store makes those slots
      // visible to the CP; re-touching a slot below next_packet_id here would mutate
      // (and briefly INVALID-toggle) a packet the CP may already be consuming — a
      // data race against the consumer. Clamp the scan's lower edge to
      // next_packet_id so the scanner only ever handles not-yet-doorbelled packets
      // (the safety-net case where a submission's doorbell was not observed).
      const uint64_t tail_packets = std::min(write_index, kVirtualLdsScannerTailPackets);
      const uint64_t scan_begin = std::max(write_index - tail_packets, state.next_packet_id);
      if (scan_begin >= write_index)
        continue;
      rewrite_packet_range(state, scan_begin, write_index, false);
    }
  }

  // Called while holding mutex_. Acquires scanner_lifecycle_mutex_ second; clear()
  // takes the lifecycle mutex WITHOUT mutex_ held, so the two paths share the same
  // (data-then-lifecycle) nesting order and cannot deadlock. The lifecycle mutex
  // guards the jthread handle and stopping_ against a concurrent clear() that is
  // joining the thread.
  void ensure_scanner_started_locked() {
    std::lock_guard lifecycle(scanner_lifecycle_mutex_);
    if (stopping_ || scanner_.joinable())
      return;
    scanner_ = std::jthread([this](std::stop_token stop) {
      using namespace std::chrono_literals;
      trace_virtual_lds_dispatch("scanner started load_write_index=%d",
                                 layer().queue_load_write_index_relaxed() != nullptr);
      while (!stop.stop_requested()) {
        scan_ready_packets();
        std::this_thread::sleep_for(25us);
      }
    });
  }

  // Guards the queue maps and the whole doorbell/intercept rewrite path. The HSA
  // runtime calls made under this lock (memory-pool allocate, agents_allow_access,
  // signal_create/load in prepare_virtual_lds_dispatch and
  // record_virtual_lds_completion_signal) are the *saved original* ROCr entry
  // points captured in set_virtual_lds_runtime_api, NOT the rocjitsu wrappers, so
  // they cannot re-enter this mutex. Future changes MUST keep routing those calls
  // through the saved originals; routing them through the wrappers would deadlock.
  std::mutex mutex_;
  std::unordered_map<uint64_t, QueueState> queues_by_ptr_;
  std::unordered_map<uint64_t, uint64_t> queues_by_doorbell_;
  // Serializes scanner start (ensure_scanner_started_locked) against stop/join
  // (clear). Separate from mutex_ because clear() must join the scanner thread,
  // which itself takes mutex_ — joining under mutex_ would deadlock.
  std::mutex scanner_lifecycle_mutex_;
  bool stopping_ = false;
  std::jthread scanner_;
};

void clear_virtual_lds_dispatch_queues() { VirtualLdsDispatchQueueRegistry::instance().clear(); }

/// @brief ROCR intercept-queue callback that rewrites virtual-LDS packets pre-submit.
void virtual_lds_packet_interceptor(const void *pkts, uint64_t pkt_count, uint64_t user_pkt_index,
                                    void *data, hsa_amd_queue_intercept_packet_writer_t writer) {
  auto *queue = static_cast<hsa_queue_t *>(data);
  static std::atomic<uint32_t> intercept_trace_count{0};
  if (intercept_trace_count.fetch_add(1, std::memory_order_relaxed) < 128) {
    uint16_t first_header = 0;
    if (pkts != nullptr && pkt_count != 0)
      std::memcpy(&first_header, pkts, sizeof(first_header));
    trace_virtual_lds_dispatch(
        "intercept callback queue=%p packets=%llu user_index=%llu first_header=0x%x",
        static_cast<void *>(queue), static_cast<unsigned long long>(pkt_count),
        static_cast<unsigned long long>(user_pkt_index), first_header);
  }
  if (queue == nullptr) {
    if (writer != nullptr)
      writer(pkts, pkt_count);
    return;
  }
  VirtualLdsDispatchQueueRegistry::instance().rewrite_intercept_packets(queue, pkts, pkt_count,
                                                                        user_pkt_index, writer);
}

/// @brief Attach the virtual-LDS packet interceptor to an AMD intercept queue.
/// @returns true iff the interceptor was successfully registered. Callers must not
/// record the queue as intercept-managed unless this succeeds — a queue flagged
/// uses_packet_interceptor is skipped by BOTH the doorbell and scanner rewrite
/// paths, so a failed registration would silently disable all packet rewriting for
/// it and every oversized-LDS dispatch would fault.
[[nodiscard]] bool register_virtual_lds_packet_interceptor(hsa_queue_t *queue) {
  auto *register_interceptor = layer().amd_queue_intercept_register();
  if (register_interceptor == nullptr || queue == nullptr)
    return false;
  const hsa_status_t status = register_interceptor(queue, virtual_lds_packet_interceptor, queue);
  if (status != HSA_STATUS_SUCCESS) {
    log_message(kLogInfo, "failed to register virtual-LDS queue interceptor queue=%p status=%d",
                static_cast<void *>(queue), static_cast<int>(status));
    trace_virtual_lds_dispatch("queue interceptor registration failed queue=%p status=%d",
                               static_cast<void *>(queue), static_cast<int>(status));
    return false;
  }
  return true;
}

hsa_status_t HSA_API rj_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void *data),
                                       void *data) {
  auto *original = layer().iterate_agents();
  if (!original)
    return HSA_STATUS_ERROR;

  hsa_agent_t guest = AgentMapper::instance().guest_agent();
  hsa_agent_t host = AgentMapper::instance().host_for_guest();
  if (guest.handle == 0 || host.handle == 0)
    return original(callback, data);

  struct ShadowIteration {
    hsa_status_t (*callback)(hsa_agent_t, void *) = nullptr;
    void *data = nullptr;
    hsa_agent_t guest{};
    hsa_agent_t host{};
  } shadow{callback, data, guest, host};

  auto shadow_callback = [](hsa_agent_t agent, void *opaque) -> hsa_status_t {
    auto *shadow = static_cast<ShadowIteration *>(opaque);
    if (agent.handle == shadow->host.handle) {
      // Public enumeration is the replacement boundary: applications see the
      // guest agent in the selected host's ordinal slot, while ROCR keeps the
      // host agent alive for translated execution.
      return shadow->callback(shadow->guest, shadow->data);
    }
    if (agent.handle == shadow->guest.handle) {
      // The synthetic KFD node may appear before or after the real host in
      // ROCR enumeration. Always suppress its own slot so public clients see
      // exactly one guest agent, emitted where the selected host appeared.
      return HSA_STATUS_SUCCESS;
    }
    return shadow->callback(agent, shadow->data);
  };

  log_message(kLogDebug, "iterate_agents shadow host=%llu guest=%llu",
              static_cast<unsigned long long>(host.handle),
              static_cast<unsigned long long>(guest.handle));
  return original(shadow_callback, &shadow);
}

hsa_status_t HSA_API rj_agent_iterate_isas(hsa_agent_t agent,
                                           hsa_status_t (*callback)(hsa_isa_t isa, void *data),
                                           void *data) {
  auto *original = layer().agent_iterate_isas();
  if (!original)
    return HSA_STATUS_ERROR;

  // HIP/CLR derives the application-visible device ISA from this callback.
  // Keep the synthetic agent guest-facing so fatbin selection picks gfx950 code
  // objects; execution-facing hooks translate and map those loads to the host.
  log_message(kLogDebug, "agent_iterate_isas agent=%llu",
              static_cast<unsigned long long>(agent.handle));
  return original(agent, callback, data);
}

hsa_status_t HSA_API rj_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                     void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                     void *data, uint32_t private_segment_size,
                                     uint32_t group_segment_size, hsa_queue_t **queue) {
  auto *original = layer().queue_create();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  log_message(kLogVerbose, "queue_create agent=%llu mapped=%llu size=%u",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle), size);

  hsa_status_t status = HSA_STATUS_ERROR;
  auto *intercept_create = layer().amd_queue_intercept_create();
  auto *intercept_register = layer().amd_queue_intercept_register();
  if (intercept_create != nullptr && intercept_register != nullptr && size >= 3) {
    status = intercept_create(mapped, size, type, callback, data, private_segment_size,
                              group_segment_size, queue);
    if (status == HSA_STATUS_SUCCESS && queue != nullptr) {
      // Record the queue as intercept-managed ONLY if the interceptor actually
      // attaches. If registration fails, record it as a normal queue so the
      // doorbell/scanner rewrite path still services its oversized-LDS dispatches
      // instead of leaving it silently un-rewritten.
      const bool intercepting = register_virtual_lds_packet_interceptor(*queue);
      VirtualLdsDispatchQueueRegistry::instance().record_queue(*queue, mapped, intercepting);
      return status;
    }
    log_message(kLogInfo, "intercept queue_create failed status=%d; falling back",
                static_cast<int>(status));
  }

  status =
      original(mapped, size, type, callback, data, private_segment_size, group_segment_size, queue);
  if (status == HSA_STATUS_SUCCESS && queue != nullptr)
    VirtualLdsDispatchQueueRegistry::instance().record_queue(*queue, mapped);
  return status;
}

hsa_status_t HSA_API rj_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue) {
  auto *original = layer().amd_queue_intercept_create();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent_handle);
  log_message(kLogVerbose, "queue_intercept_create agent=%llu mapped=%llu size=%u",
              static_cast<unsigned long long>(agent_handle.handle),
              static_cast<unsigned long long>(mapped.handle), size);
  const hsa_status_t status =
      original(mapped, size, type, callback, data, private_segment_size, group_segment_size, queue);
  if (status == HSA_STATUS_SUCCESS && queue != nullptr) {
    // Only mark intercept-managed when the interceptor attaches; otherwise record
    // as a normal queue so the doorbell/scanner rewrite path still covers it.
    const bool intercepting = register_virtual_lds_packet_interceptor(*queue);
    VirtualLdsDispatchQueueRegistry::instance().record_queue(*queue, mapped, intercepting);
  }
  return status;
}

hsa_status_t HSA_API rj_queue_destroy(hsa_queue_t *queue) {
  auto *original = layer().queue_destroy();
  if (!original)
    return HSA_STATUS_ERROR;
  log_message(kLogVerbose, "queue_destroy queue=%p id=%llu", static_cast<void *>(queue),
              queue ? static_cast<unsigned long long>(queue->id) : 0);
  VirtualLdsDispatchQueueRegistry::instance().erase_queue(queue);
  hsa_status_t status = original(queue);
  log_message(kLogVerbose, "queue_destroy status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_signal_destroy(hsa_signal_t signal) {
  auto *original = layer().signal_destroy();
  if (!original)
    return HSA_STATUS_ERROR;

  // Detach the borrowed handle before ROCR invalidates it. This also serializes
  // against the queue scanner and intercept callbacks, both of which use the
  // registry mutex while polling completion state.
  VirtualLdsDispatchQueueRegistry::instance().retire_destroyed_completion_signal(signal);
  return original(signal);
}

void HSA_API rj_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  VirtualLdsDispatchQueueRegistry::instance().rewrite_before_doorbell(signal, value);
  auto *original = layer().signal_store_relaxed();
  if (original != nullptr)
    original(signal, value);
}

void HSA_API rj_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  VirtualLdsDispatchQueueRegistry::instance().rewrite_before_doorbell(signal, value);
  auto *original = layer().signal_store_screlease();
  if (original != nullptr)
    original(signal, value);
}

hsa_status_t HSA_API rj_agent_iterate_regions(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void *data), void *data) {
  auto *original = layer().agent_iterate_regions();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  log_message(kLogDebug, "agent_iterate_regions agent=%llu mapped=%llu",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle));
  return original(mapped, callback, data);
}

hsa_status_t HSA_API rj_memory_assign_agent(void *ptr, hsa_agent_t agent,
                                            hsa_access_permission_t access) {
  auto *original = layer().memory_assign_agent();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  log_message(kLogVerbose, "memory_assign_agent ptr=%p agent=%llu mapped=%llu access=%d", ptr,
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle), static_cast<int>(access));
  return original(ptr, mapped, access);
}

hsa_status_t HSA_API rj_shut_down() {
  auto config = layer().config();
  if (config && config->guest_target) {
    // Guest mode does not call the real ROCR shutdown. Later language-runtime
    // teardown can still run HSA cleanup paths, so keep rocjitsu's API-table
    // mappings installed until process exit.
    log_message(kLogVerbose, "skipping real hsa_shut_down in guest mode");
    return HSA_STATUS_SUCCESS;
  }

  auto *original = layer().shut_down();
  if (!original)
    return HSA_STATUS_ERROR;
  return original();
}

hsa_status_t HSA_API rj_executable_destroy(hsa_executable_t executable) {
  log_message(kLogVerbose, "executable_destroy exec=%llu",
              static_cast<unsigned long long>(executable.handle));
  VirtualLdsRuntimeRegistry::instance().erase_executable(executable);
  ExecutableAgentRegistry::instance().erase_executable(executable);
  auto *original = layer().executable_destroy();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_status_t status = original(executable);
  log_message(kLogVerbose, "executable_destroy status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_executable_get_symbol(hsa_executable_t executable, const char *module_name,
                                              const char *symbol_name, hsa_agent_t agent,
                                              int32_t call_convention,
                                              hsa_executable_symbol_t *symbol) {
  auto *original = layer().executable_get_symbol();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = ExecutableAgentRegistry::instance().map_agent(executable, agent);
  mapped = AgentMapper::instance().map(mapped);
  log_message(kLogVerbose, "get_symbol exec=%llu agent=%llu mapped=%llu symbol=%s",
              static_cast<unsigned long long>(executable.handle),
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle), symbol_name ? symbol_name : "");
  hsa_status_t status =
      original(executable, module_name, symbol_name, mapped, call_convention, symbol);
  if (status == HSA_STATUS_SUCCESS && symbol_name != nullptr && symbol != nullptr)
    VirtualLdsRuntimeRegistry::instance().record_symbol(executable, symbol_name, *symbol);
  return status;
}

hsa_status_t HSA_API rj_executable_get_symbol_by_name(hsa_executable_t executable,
                                                      const char *symbol_name,
                                                      const hsa_agent_t *agent,
                                                      hsa_executable_symbol_t *symbol) {
  auto *original = layer().executable_get_symbol_by_name();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped_agent{};
  const hsa_agent_t *mapped_ptr = nullptr;
  if (agent != nullptr) {
    mapped_agent = ExecutableAgentRegistry::instance().map_agent(executable, *agent);
    mapped_agent = AgentMapper::instance().map(mapped_agent);
    mapped_ptr = &mapped_agent;
  }
  log_message(kLogVerbose, "get_symbol_by_name exec=%llu agent=%llu mapped=%llu symbol=%s",
              static_cast<unsigned long long>(executable.handle),
              static_cast<unsigned long long>(agent ? agent->handle : 0),
              static_cast<unsigned long long>(mapped_agent.handle), symbol_name ? symbol_name : "");
  hsa_status_t status = original(executable, symbol_name, mapped_ptr, symbol);
  if (status == HSA_STATUS_SUCCESS && symbol_name != nullptr && symbol != nullptr)
    VirtualLdsRuntimeRegistry::instance().record_symbol(executable, symbol_name, *symbol);
  return status;
}

hsa_status_t HSA_API rj_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                                                   hsa_executable_symbol_info_t attribute,
                                                   void *value) {
  auto *original = layer().executable_symbol_get_info();
  if (!original)
    return HSA_STATUS_ERROR;

  hsa_status_t status = original(executable_symbol, attribute, value);
  if (status == HSA_STATUS_SUCCESS && attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT &&
      value != nullptr) {
    uint64_t kernel_object = 0;
    std::memcpy(&kernel_object, value, sizeof(kernel_object));
    uint32_t private_segment_size = 0;
    (void)original(executable_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                   &private_segment_size);
    VirtualLdsRuntimeRegistry::instance().note_kernel_object(executable_symbol, kernel_object,
                                                             private_segment_size);
  }
  return status;
}

/// @brief Query the original ROCR symbol-name fields without recursing into this hook.
[[nodiscard]] std::optional<std::string>
query_executable_symbol_name(hsa_executable_symbol_t executable_symbol) {
  auto *original = layer().executable_symbol_get_info();
  if (!original)
    return std::nullopt;

  uint32_t name_length = 0;
  hsa_status_t status =
      original(executable_symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
  if (status != HSA_STATUS_SUCCESS || name_length == 0)
    return std::nullopt;

  std::string symbol_name(name_length, '\0');
  status = original(executable_symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, symbol_name.data());
  if (status != HSA_STATUS_SUCCESS)
    return std::nullopt;
  return symbol_name;
}

hsa_status_t HSA_API rj_executable_agent_global_variable_define(hsa_executable_t executable,
                                                                hsa_agent_t agent,
                                                                const char *variable_name,
                                                                void *address) {
  auto *original = layer().executable_agent_global_variable_define();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = ExecutableAgentRegistry::instance().map_agent(executable, agent);
  mapped = AgentMapper::instance().map(mapped);
  log_message(kLogVerbose, "global_variable_define exec=%llu agent=%llu mapped=%llu name=%s",
              static_cast<unsigned long long>(executable.handle),
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle), variable_name ? variable_name : "");
  return original(executable, mapped, variable_name, address);
}

/// @brief Callback context for restoring the caller-visible agent in symbol iteration.
struct IterateAgentSymbolsData {
  hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t,
                           void *) = nullptr;
  void *data = nullptr;
  hsa_agent_t guest{};
};

/// @brief Agent-symbol callback wrapper that reports the original guest agent.
hsa_status_t HSA_API rj_iterate_agent_symbols_callback(hsa_executable_t executable, hsa_agent_t,
                                                       hsa_executable_symbol_t symbol, void *data) {
  auto *wrapped = static_cast<IterateAgentSymbolsData *>(data);
  if (auto symbol_name = query_executable_symbol_name(symbol)) {
    // Clients often obtain kernel handles by iteration and then immediately ask
    // for KERNEL_OBJECT inside their callback. Record the name before handing
    // the symbol to the client so the KERNEL_OBJECT hook can resolve the
    // virtual-LDS metadata for this symbol path too.
    VirtualLdsRuntimeRegistry::instance().record_symbol(executable, *symbol_name, symbol);
  }
  return wrapped->callback(executable, wrapped->guest, symbol, wrapped->data);
}

hsa_status_t HSA_API rj_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data) {
  auto *original = layer().executable_iterate_agent_symbols();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = ExecutableAgentRegistry::instance().map_agent(executable, agent);
  mapped = AgentMapper::instance().map(mapped);
  // Always route through the recording callback, even when the agent mapping is
  // identity. The wrapper records each symbol's name so the KERNEL_OBJECT hook can
  // later resolve its virtual-LDS sidecar; skipping it on the identity path would
  // let iterated symbols get a normal kernel object with no executable/name
  // association, leaving an oversized virtual-LDS dispatch stuck on its normal
  // descriptor. The client still sees the original agent it passed in.
  IterateAgentSymbolsData wrapped{callback, data, agent};
  return original(executable, mapped, rj_iterate_agent_symbols_callback, &wrapped);
}

struct IterateSymbolsData {
  hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void *) = nullptr;
  void *data = nullptr;
};

/// @brief Deprecated agentless iterate-symbols callback that records each name.
hsa_status_t HSA_API rj_iterate_symbols_callback(hsa_executable_t executable,
                                                 hsa_executable_symbol_t symbol, void *data) {
  auto *wrapped = static_cast<IterateSymbolsData *>(data);
  if (auto symbol_name = query_executable_symbol_name(symbol)) {
    // Same rationale as the agent-symbols wrapper: record the name before the
    // client can query KERNEL_OBJECT so the virtual-LDS sidecar stays resolvable
    // for symbols discovered through the legacy (deprecated) iterator too.
    VirtualLdsRuntimeRegistry::instance().record_symbol(executable, *symbol_name, symbol);
  }
  return wrapped->callback(executable, symbol, wrapped->data);
}

/// @brief Wrap the deprecated agentless symbol iterator to record symbol names.
hsa_status_t HSA_API rj_executable_iterate_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void *), void *data) {
  auto *original = layer().executable_iterate_symbols();
  if (!original)
    return HSA_STATUS_ERROR;
  IterateSymbolsData wrapped{callback, data};
  return original(executable, rj_iterate_symbols_callback, &wrapped);
}

hsa_status_t HSA_API rj_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
  auto *original = layer().amd_agent_iterate_memory_pools();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(agent);
  log_message(kLogDebug, "amd_agent_iterate_memory_pools agent=%llu mapped=%llu",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle));
  return original(AgentMapper::instance().is_guest(agent) ? agent : mapped, callback, data);
}

hsa_status_t HSA_API rj_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                                 hsa_amd_memory_pool_info_t attribute,
                                                 void *value) {
  auto *original = layer().amd_memory_pool_get_info();
  if (!original)
    return HSA_STATUS_ERROR;
  log_message(kLogDebug, "amd_memory_pool_get_info pool=%llu attr=%u",
              static_cast<unsigned long long>(memory_pool.handle),
              static_cast<unsigned>(attribute));
  return original(memory_pool, attribute, value);
}

hsa_status_t HSA_API rj_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                                 uint32_t flags, void **ptr) {
  auto *original = layer().amd_memory_pool_allocate();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_amd_memory_pool_t mapped_pool = mapped_memory_pool(memory_pool);
  log_message(kLogVerbose, "amd_memory_pool_allocate pool=%llu mapped=%llu size=%zu flags=0x%x",
              static_cast<unsigned long long>(memory_pool.handle),
              static_cast<unsigned long long>(mapped_pool.handle), size, flags);
  hsa_status_t status = forward_amd_call(original, MappedPoolArg{mapped_pool}, size, flags, ptr);
  log_message(kLogVerbose, "amd_memory_pool_allocate status=%d ptr=%p", static_cast<int>(status),
              ptr ? *ptr : nullptr);
  return status;
}

hsa_status_t HSA_API rj_amd_memory_pool_free(void *ptr) {
  auto *original = layer().amd_memory_pool_free();
  if (!original)
    return HSA_STATUS_ERROR;
  log_message(kLogVerbose, "amd_memory_pool_free ptr=%p", ptr);
  hsa_status_t status = original(ptr);
  log_message(kLogVerbose, "amd_memory_pool_free status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_amd_profiling_set_profiler_enabled(hsa_queue_t *queue, int enable) {
  auto *original = layer().amd_profiling_set_profiler_enabled();
  if (!original)
    return HSA_STATUS_ERROR;
  log_message(kLogVerbose, "amd_profiling_set_profiler_enabled queue=%p enable=%d",
              static_cast<void *>(queue), enable);
  return original(queue, enable);
}

hsa_status_t HSA_API rj_amd_profiling_get_dispatch_time(hsa_agent_t agent, hsa_signal_t signal,
                                                        hsa_amd_profiling_dispatch_time_t *time) {
  auto *original = layer().amd_profiling_get_dispatch_time();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(agent);
  hsa_status_t status = forward_amd_call(original, MappedAgentArg{mapped}, signal, time);
  log_message(kLogVerbose,
              "amd_profiling_get_dispatch_time agent=%llu mapped=%llu signal=%llu status=%d "
              "start=%llu end=%llu",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle),
              static_cast<unsigned long long>(signal.handle), static_cast<int>(status),
              static_cast<unsigned long long>(time ? time->start : 0),
              static_cast<unsigned long long>(time ? time->end : 0));
  return status;
}

hsa_status_t HSA_API rj_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent,
                                                                    uint64_t agent_tick,
                                                                    uint64_t *system_tick) {
  auto *original = layer().amd_profiling_convert_tick_to_system_domain();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(agent);
  hsa_status_t status = forward_amd_call(original, MappedAgentArg{mapped}, agent_tick, system_tick);
  log_message(kLogVerbose,
              "amd_profiling_convert_tick_to_system_domain agent=%llu mapped=%llu status=%d "
              "agent_tick=%llu system_tick=%llu",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle), static_cast<int>(status),
              static_cast<unsigned long long>(agent_tick),
              static_cast<unsigned long long>(system_tick ? *system_tick : 0));
  return status;
}

hsa_status_t HSA_API rj_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                       hsa_amd_memory_pool_t memory_pool,
                                                       hsa_amd_agent_memory_pool_info_t attribute,
                                                       void *value) {
  auto *original = layer().amd_agent_memory_pool_get_info();
  if (!original)
    return HSA_STATUS_ERROR;
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (memory_pool.handle == 0) {
    // ROCR converts the AMD pool handle back to an internal region before
    // answering agent/pool access queries. Keep null synthetic handles from
    // crossing that ABI boundary; they are invalid AMD memory pools.
    return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  }
  hsa_agent_t mapped = mapped_agent(agent);
  hsa_amd_memory_pool_t mapped_pool = mapped_memory_pool(memory_pool);
  log_message(
      kLogDebug,
      "amd_agent_memory_pool_get_info agent=%llu mapped=%llu pool=%llu mapped=%llu "
      "attr=%u",
      static_cast<unsigned long long>(agent.handle), static_cast<unsigned long long>(mapped.handle),
      static_cast<unsigned long long>(memory_pool.handle),
      static_cast<unsigned long long>(mapped_pool.handle), static_cast<unsigned>(attribute));
  return forward_amd_call(original, MappedAgentArg{mapped}, MappedPoolArg{mapped_pool}, attribute,
                          value);
}

hsa_status_t HSA_API rj_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t *agents,
                                                const uint32_t *flags, const void *ptr) {
  auto *original = layer().amd_agents_allow_access();
  if (!original)
    return HSA_STATUS_ERROR;
  auto mapped = map_access_agent_array(agents, num_agents, flags);
  const uint32_t forwarded_count =
      mapped.changed ? static_cast<uint32_t>(mapped.agents.size()) : num_agents;
  const hsa_agent_t *forwarded_agents = mapped.changed ? mapped.agents.data() : agents;
  const uint32_t *forwarded_flags = mapped.changed && flags ? mapped.flags.data() : flags;
  log_message(kLogVerbose, "amd_agents_allow_access ptr=%p count=%u forwarded=%u mapped=%d", ptr,
              num_agents, forwarded_count, mapped.changed ? 1 : 0);
  hsa_status_t status = original(forwarded_count, forwarded_agents, forwarded_flags, ptr);
  log_message(kLogVerbose, "amd_agents_allow_access status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_amd_memory_async_copy(void *dst, hsa_agent_t dst_agent, const void *src,
                                              hsa_agent_t src_agent, size_t size,
                                              uint32_t num_dep_signals,
                                              const hsa_signal_t *dep_signals,
                                              hsa_signal_t completion_signal) {
  auto *original = layer().amd_memory_async_copy();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped_dst = mapped_agent(dst_agent);
  hsa_agent_t mapped_src = mapped_agent(src_agent);
  log_message(kLogVerbose,
              "amd_memory_async_copy dst_agent=%llu mapped=%llu src_agent=%llu mapped=%llu "
              "size=%zu",
              static_cast<unsigned long long>(dst_agent.handle),
              static_cast<unsigned long long>(mapped_dst.handle),
              static_cast<unsigned long long>(src_agent.handle),
              static_cast<unsigned long long>(mapped_src.handle), size);
  return forward_amd_call(original, dst, MappedAgentArg{mapped_dst}, src,
                          MappedAgentArg{mapped_src}, size, num_dep_signals, dep_signals,
                          completion_signal);
}

hsa_status_t HSA_API rj_amd_memory_async_copy_on_engine(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t *dep_signals, hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
  auto *original = layer().amd_memory_async_copy_on_engine();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped_dst = mapped_agent(dst_agent);
  hsa_agent_t mapped_src = mapped_agent(src_agent);
  log_message(kLogVerbose,
              "amd_memory_async_copy_on_engine dst_agent=%llu mapped=%llu src_agent=%llu "
              "mapped=%llu size=%zu engine=%u",
              static_cast<unsigned long long>(dst_agent.handle),
              static_cast<unsigned long long>(mapped_dst.handle),
              static_cast<unsigned long long>(src_agent.handle),
              static_cast<unsigned long long>(mapped_src.handle), size,
              static_cast<unsigned>(engine_id));
  return forward_amd_call(original, dst, MappedAgentArg{mapped_dst}, src,
                          MappedAgentArg{mapped_src}, size, num_dep_signals, dep_signals,
                          completion_signal, engine_id, force_copy_on_sdma);
}

hsa_status_t HSA_API rj_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t *dst, const hsa_dim3_t *dst_offset, const hsa_pitched_ptr_t *src,
    const hsa_dim3_t *src_offset, const hsa_dim3_t *range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal) {
  auto *original = layer().amd_memory_async_copy_rect();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(copy_agent);
  log_message(kLogVerbose, "amd_memory_async_copy_rect agent=%llu mapped=%llu dir=%u",
              static_cast<unsigned long long>(copy_agent.handle),
              static_cast<unsigned long long>(mapped.handle), static_cast<unsigned>(dir));
  return forward_amd_call(original, dst, dst_offset, src, src_offset, range, MappedAgentArg{mapped},
                          dir, num_dep_signals, dep_signals, completion_signal);
}

hsa_status_t HSA_API rj_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                                      uint32_t *engine_ids_mask) {
  auto *original = layer().amd_memory_copy_engine_status();
  if (!original)
    return HSA_STATUS_ERROR;
  return forward_amd_call(original, mapped_agent_arg(dst_agent), mapped_agent_arg(src_agent),
                          engine_ids_mask);
}

hsa_status_t HSA_API rj_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent,
                                                             hsa_agent_t src_agent,
                                                             uint32_t *engine_ids_mask) {
  auto *original = layer().amd_memory_get_preferred_copy_engine();
  if (!original)
    return HSA_STATUS_ERROR;
  return forward_amd_call(original, mapped_agent_arg(dst_agent), mapped_agent_arg(src_agent),
                          engine_ids_mask);
}

hsa_status_t HSA_API rj_amd_memory_lock(void *host_ptr, size_t size, hsa_agent_t *agents,
                                        int num_agent, void **agent_ptr) {
  auto *original = layer().amd_memory_lock();
  if (!original)
    return HSA_STATUS_ERROR;
  MappedAgentList mapped = num_agent > 0
                               ? map_unique_agent_array(agents, static_cast<size_t>(num_agent))
                               : MappedAgentList{};
  hsa_agent_t *forwarded_agents = mapped.changed ? mapped.agents.data() : agents;
  int forwarded_count = mapped.changed ? static_cast<int>(mapped.agents.size()) : num_agent;
  return original(host_ptr, size, forwarded_agents, forwarded_count, agent_ptr);
}

hsa_status_t HSA_API rj_amd_memory_lock_to_pool(void *host_ptr, size_t size, hsa_agent_t *agents,
                                                int num_agent, hsa_amd_memory_pool_t pool,
                                                uint32_t flags, void **agent_ptr) {
  auto *original = layer().amd_memory_lock_to_pool();
  if (!original)
    return HSA_STATUS_ERROR;
  MappedAgentList mapped = num_agent > 0
                               ? map_unique_agent_array(agents, static_cast<size_t>(num_agent))
                               : MappedAgentList{};
  hsa_agent_t *forwarded_agents = mapped.changed ? mapped.agents.data() : agents;
  int forwarded_count = mapped.changed ? static_cast<int>(mapped.agents.size()) : num_agent;
  return forward_amd_call(original, host_ptr, size, forwarded_agents, forwarded_count,
                          mapped_pool_arg(pool), flags, agent_ptr);
}

hsa_status_t HSA_API rj_amd_memory_fill(void *ptr, uint32_t value, size_t count) {
  auto *original = layer().amd_memory_fill();
  if (!original)
    return HSA_STATUS_ERROR;
  log_message(kLogVerbose, "amd_memory_fill ptr=%p value=0x%x count=%zu", ptr, value, count);
  hsa_status_t status = original(ptr, value, count);
  log_message(kLogVerbose, "amd_memory_fill status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_amd_pointer_info(const void *ptr, hsa_amd_pointer_info_t *info,
                                         void *(*alloc)(size_t), uint32_t *num_agents_accessible,
                                         hsa_agent_t **accessible) {
  auto *original = layer().amd_pointer_info();
  if (!original)
    return HSA_STATUS_ERROR;
  log_message(kLogVerbose, "amd_pointer_info ptr=%p", ptr);
  hsa_status_t status = original(ptr, info, alloc, num_agents_accessible, accessible);
  if (status == HSA_STATUS_SUCCESS) {
    if (info != nullptr &&
        info->size >= offsetof(hsa_amd_pointer_info_t, agentOwner) + sizeof(info->agentOwner))
      info->agentOwner = AgentMapper::instance().guest_for_host(info->agentOwner);
    map_pointer_info_accessible_agents(num_agents_accessible, accessible);
  }
  log_message(kLogVerbose, "amd_pointer_info status=%d owner=%llu accessible=%u",
              static_cast<int>(status),
              static_cast<unsigned long long>(info ? info->agentOwner.handle : 0),
              num_agents_accessible ? *num_agents_accessible : 0);
  return status;
}

hsa_status_t HSA_API rj_amd_svm_prefetch_async(void *ptr, size_t size, hsa_agent_t agent,
                                               uint32_t num_dep_signals,
                                               const hsa_signal_t *dep_signals,
                                               hsa_signal_t completion_signal) {
  auto *original = layer().amd_svm_prefetch_async();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(agent);
  log_message(kLogVerbose, "amd_svm_prefetch_async ptr=%p size=%zu agent=%llu mapped=%llu", ptr,
              size, static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle));
  return forward_amd_call(original, ptr, size, MappedAgentArg{mapped}, num_dep_signals, dep_signals,
                          completion_signal);
}

hsa_status_t HSA_API rj_amd_vmem_set_access(void *va, size_t size,
                                            const hsa_amd_memory_access_desc_t *desc,
                                            size_t desc_cnt) {
  auto *original = layer().amd_vmem_set_access();
  if (!original)
    return HSA_STATUS_ERROR;
  MappedAccessDescs mapped = map_unique_access_descs(desc, desc_cnt);
  const hsa_amd_memory_access_desc_t *forwarded_desc = mapped.changed ? mapped.descs.data() : desc;
  size_t forwarded_count = mapped.changed ? mapped.descs.size() : desc_cnt;
  log_message(kLogVerbose, "amd_vmem_set_access va=%p size=%zu desc_cnt=%zu", va, size, desc_cnt);
  return original(va, size, forwarded_desc, forwarded_count);
}

hsa_status_t HSA_API rj_amd_vmem_get_access(void *va, hsa_access_permission_t *perms,
                                            hsa_agent_t agent_handle) {
  auto *original = layer().amd_vmem_get_access();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(agent_handle);
  log_message(kLogVerbose, "amd_vmem_get_access va=%p agent=%llu mapped=%llu", va,
              static_cast<unsigned long long>(agent_handle.handle),
              static_cast<unsigned long long>(mapped.handle));
  return forward_amd_call(original, va, perms, MappedAgentArg{mapped});
}

hsa_status_t HSA_API rj_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold) {
  auto *original = layer().amd_agent_set_async_scratch_limit();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(agent);
  log_message(kLogVerbose, "amd_agent_set_async_scratch_limit agent=%llu mapped=%llu threshold=%zu",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle), threshold);
  return forward_amd_call(original, MappedAgentArg{mapped}, threshold);
}

hsa_status_t HSA_API rj_amd_memory_async_batch_copy(const hsa_amd_memory_copy_op_t *copy_ops,
                                                    uint32_t num_copy_ops, uint32_t num_dep_signals,
                                                    const hsa_signal_t *dep_signals) {
  auto *original = layer().amd_memory_async_batch_copy();
  if (!original)
    return HSA_STATUS_ERROR;
  if (!copy_ops || num_copy_ops == 0)
    return original(copy_ops, num_copy_ops, num_dep_signals, dep_signals);

  std::vector<hsa_amd_memory_copy_op_t> mapped(copy_ops, copy_ops + num_copy_ops);
  std::vector<std::vector<hsa_agent_t>> mapped_dst_lists;
  mapped_dst_lists.reserve(num_copy_ops);
  log_message(kLogVerbose, "amd_memory_async_batch_copy ops=%u deps=%u", num_copy_ops,
              num_dep_signals);
  for (uint32_t i = 0; i < num_copy_ops; ++i) {
    auto &op = mapped[i];
    const BatchCopyAgentShape shape = batch_copy_agent_shape(op);

    if (shape.src_scalar)
      op.src_agent = mapped_agent(op.src_agent);
    if (shape.dst_scalar)
      op.dst_agent = mapped_agent(op.dst_agent);
    if (shape.dst_list && op.dst_agent_list != nullptr) {
      mapped_dst_lists.push_back(map_agent_array(op.dst_agent_list, op.num_entries));
      op.dst_agent_list = mapped_dst_lists.back().data();
    }
  }
  return original(mapped.data(), num_copy_ops, num_dep_signals, dep_signals);
}

hsa_status_t HSA_API rj_amd_agent_preload(hsa_agent_t agent, uint64_t flags) {
  auto *original = layer().amd_agent_preload();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = mapped_agent(agent);
  log_message(kLogVerbose, "amd_agent_preload agent=%llu mapped=%llu flags=0x%llx",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle),
              static_cast<unsigned long long>(flags));
  return forward_amd_call(original, MappedAgentArg{mapped}, flags);
}

/// @brief Create an HSA reader from translated ELF bytes and keep the storage alive.
[[nodiscard]] hsa_status_t create_translated_reader(std::vector<uint8_t> translated,
                                                    hsa_code_object_reader_t *translated_reader) {
  auto *owned_storage = new (std::nothrow) std::vector<uint8_t>(std::move(translated));
  if (owned_storage == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  std::shared_ptr<const std::vector<uint8_t>> owned(owned_storage);

  auto *original_create = layer().create_from_memory();
  if (original_create == nullptr) {
    return HSA_STATUS_ERROR;
  }

  const hsa_status_t status = original_create(owned->data(), owned->size(), translated_reader);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  if (!CodeObjectReaderRegistry::instance().store(*translated_reader, owned->data(), owned->size(),
                                                  owned)) {
    if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
      (void)original_destroy(*translated_reader);
    *translated_reader = {};
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object) {
  auto *original_load = layer().load_agent_code_object();
  if (original_load == nullptr)
    return HSA_STATUS_ERROR;

  auto config = layer().config();
  if (!config) {
    // `OnUnload()` should not normally race active ROCR callbacks, but returning
    // an HSA error here is safer than dereferencing cleared hook state.
    std::fprintf(stderr, "[rocjitsu-hooks] DBT hook layer is inactive during code-object load\n");
    return HSA_STATUS_ERROR;
  }

  const bool guest_load = AgentMapper::instance().is_guest(agent);
  log_message(kLogVerbose, "load_agent_code_object exec=%llu agent=%llu guest=%d reader=%llu",
              static_cast<unsigned long long>(executable.handle),
              static_cast<unsigned long long>(agent.handle), guest_load ? 1 : 0,
              static_cast<unsigned long long>(code_object_reader.handle));

  // Only the configured guest agent's loads belong to the DBT pipeline. An
  // unrelated GPU's native code object must not be reinterpreted as the guest
  // ISA, translated for the configured host, or subjected to rocjitsu metadata
  // validation/registration -- doing so could misdecode it and load a wrong
  // artifact on that agent. Forward it verbatim before any reader lookup, target
  // detection, source override, translation, or registry mutation.
  if (!guest_load)
    return original_load(executable, agent, code_object_reader, options, loaded_code_object);

  hsa_agent_t load_agent = AgentMapper::instance().host_for_guest();
  if (load_agent.handle == 0)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  CodeObjectReaderRegistry::ReaderBytes reader_bytes =
      CodeObjectReaderRegistry::instance().lookup(code_object_reader);
  if (!reader_bytes) {
    // A guest load with no registered memory bytes cannot be translated (the DBT
    // path needs the source image); fail rather than load an untranslated guest
    // object on the host.
    log_message(kLogInfo, "no memory bytes registered for reader=%llu",
                static_cast<unsigned long long>(code_object_reader.handle));
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  }
  const uint8_t *bytes = reader_bytes.bytes;
  size_t size = reader_bytes.size;

  const DetectedElfTarget detected = detect_target_from_elf(bytes, size);
  DetectedElfTarget source_target = detected;
  const bool detected_already_target =
      detected.arch == config->target.arch && detected.mach == config->target.mach;
  if (config->source_override && !detected_already_target) {
    source_target.arch = config->source_override->arch;
    source_target.mach = config->source_override->mach;
  }
  if (source_target.arch == ROCJITSU_CODE_ARCH_INVALID || source_target.mach == 0) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to detect source ISA from code-object ELF\n");
    return HSA_STATUS_ERROR;
  }

  // gfx1250 A0 and B0 share an ELF machine ID, so a same-arch/same-mach match
  // does not identify the revision. When both revisions are configured, use the
  // selected same-ISA translation profile. Otherwise fail closed.
  const bool gfx1250_same_target = source_target.arch == ROCJITSU_CODE_ARCH_GFX1250 &&
                                   config->target.arch == ROCJITSU_CODE_ARCH_GFX1250 &&
                                   source_target.mach == config->target.mach;
  const bool have_gfx1250_revisions = config->guest_revision != ProcessorRevision::Unspecified &&
                                      config->host_revision != ProcessorRevision::Unspecified;
  if (gfx1250_same_target && !have_gfx1250_revisions) {
    std::fprintf(stderr,
                 "[rocjitsu-hooks] gfx1250 same-target load requires guest and host silicon "
                 "revisions in the DBT guest config; failing load\n");
    return HSA_STATUS_ERROR;
  }

  if (!gfx1250_same_target && source_target.arch == config->target.arch &&
      source_target.mach == config->target.mach) {
    log_message(kLogInfo,
                "source target %s arch %s already matches requested target; passing through",
                elf_mach_name(source_target.mach), arch_name(source_target.arch));
    // Parse and validate hook metadata BEFORE loading, so an unusable code object
    // is rejected without first mutating executable state, and so the malformed-
    // metadata policy matches the translated path (reject, not ignore).
    auto hook_metadata = parse_virtual_lds_hook_metadata(std::span<const uint8_t>(bytes, size));
    if (!hook_metadata) {
      std::fprintf(stderr,
                   "[rocjitsu-hooks] target code object has malformed virtual-LDS metadata\n");
      return HSA_STATUS_ERROR;
    }
    if (auto bad = first_unsupportable_virtual_lds_kernel(*hook_metadata)) {
      std::fprintf(stderr,
                   "[rocjitsu-hooks] virtual-LDS kernel '%s' cannot be dispatched on this host "
                   "(missing runtime-state ABI or inconsistent kernarg wrapper); failing load\n",
                   bad->c_str());
      return HSA_STATUS_ERROR;
    }
    hsa_status_t status =
        original_load(executable, load_agent, code_object_reader, options, loaded_code_object);
    log_message(kLogVerbose, "load_agent_code_object already-target status=%d",
                static_cast<int>(status));
    if (status == HSA_STATUS_SUCCESS)
      ExecutableAgentRegistry::instance().record(executable, agent, load_agent);
    if (status == HSA_STATUS_SUCCESS) {
      const hsa_loaded_code_object_t loaded =
          loaded_code_object != nullptr ? *loaded_code_object : hsa_loaded_code_object_t{};
      VirtualLdsRuntimeRegistry::instance().record_load(executable, agent, load_agent, loaded,
                                                        std::move(*hook_metadata));
    }
    return status;
  }

  BinaryTranslatorOptions translator_options;
  // Runtime DBT loads large framework code objects where some kernel symbols may
  // never be dispatched in the current process. Keep the code object loadable if
  // an independent kernel fails translation; the skipped-kernel diagnostic names
  // the symbol that is redirected to a target no-op stub.
  translator_options.skip_failed_kernels = true;
  // gfx1250 A0 and B0 share a machine ID. Select the same-ISA translation
  // profile from the configured revisions.
  translator_options.input_revision = config->guest_revision;
  translator_options.output_revision = config->host_revision;

  std::vector<uint8_t> translated_elf;
  log_message(kLogInfo, "translating reader=%llu %s/%s -> %s/%s mach=0x%x",
              static_cast<unsigned long long>(code_object_reader.handle),
              elf_mach_name(source_target.mach), arch_name(source_target.arch),
              config->target.name.data(), arch_name(config->target.arch), config->target.mach);

  AmdGpuCodeObject source_object(bytes, size);
  if (!source_object.is_valid()) {
    std::fprintf(stderr, "[rocjitsu-hooks] source bytes are not a valid AMDGPU code object\n");
    return HSA_STATUS_ERROR;
  }

  rocjitsu::TranslatedCodeObject translated;
  try {
    BinaryTranslator translator(source_target.arch, config->target.arch, config->target.mach,
                                translator_options);
    translated = translator.translate(source_object);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[rocjitsu-hooks] code-object translation threw an exception: %s\n",
                 error.what());
    return HSA_STATUS_ERROR;
  } catch (...) {
    std::fprintf(stderr, "[rocjitsu-hooks] code-object translation threw an unknown exception\n");
    return HSA_STATUS_ERROR;
  }

  print_diagnostics(stderr, translated.diagnostics, config->log_level > kLogDisabled);
  if (translated.elf_bytes.empty() || has_error_diagnostic(translated.diagnostics)) {
    std::fprintf(stderr, "[rocjitsu-hooks] translation failed; refusing original code object\n");
    return HSA_STATUS_ERROR;
  }

  // A skipped kernel is redirected to an s_endpgm stub. Dispatching that stub
  // would COMPLETE NORMALLY without producing the kernel's outputs, so the application
  // would observe success with stale/garbage output instead of a failure. There is no
  // safe way to let such a kernel be dispatched, so fail the whole code-object
  // load rather than hand back an executable that can silently produce wrong
  // results. (This trades the "keep the module loadable when an unused kernel
  // fails to translate" behavior for correctness; a dispatched skipped kernel
  // must never appear to succeed.)
  if (has_skipped_kernel(translated.diagnostics)) {
    std::fprintf(stderr,
                 "[rocjitsu-hooks] code object contains a kernel that could not be translated and "
                 "would silently complete if dispatched; refusing the load\n");
    return HSA_STATUS_ERROR;
  }

  translated_elf = std::move(translated.elf_bytes);

  auto hook_metadata = parse_virtual_lds_hook_metadata(
      std::span<const uint8_t>(translated_elf.data(), translated_elf.size()));
  if (!hook_metadata) {
    std::fprintf(stderr,
                 "[rocjitsu-hooks] translated code object has malformed virtual-LDS metadata\n");
    return HSA_STATUS_ERROR;
  }
  if (auto bad = first_unsupportable_virtual_lds_kernel(*hook_metadata)) {
    std::fprintf(stderr,
                 "[rocjitsu-hooks] virtual-LDS kernel '%s' cannot be dispatched on this host "
                 "(missing runtime-state ABI or inconsistent kernarg wrapper); failing load\n",
                 bad->c_str());
    return HSA_STATUS_ERROR;
  }

  hsa_code_object_reader_t translated_reader{};
  hsa_status_t status = create_translated_reader(std::move(translated_elf), &translated_reader);
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to create translated code-object reader: %d\n",
                 static_cast<int>(status));
    return status;
  }

  status = original_load(executable, load_agent, translated_reader, options, loaded_code_object);
  log_message(kLogVerbose, "load_agent_code_object translated load_agent=%llu status=%d",
              static_cast<unsigned long long>(load_agent.handle), static_cast<int>(status));
  CodeObjectReaderRegistry::instance().remove(translated_reader);
  if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
    (void)original_destroy(translated_reader);

  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-hooks] translated code-object load failed: %d\n",
                 static_cast<int>(status));
  } else {
    ExecutableAgentRegistry::instance().record(executable, agent, load_agent);
  }
  if (status == HSA_STATUS_SUCCESS) {
    const hsa_loaded_code_object_t loaded =
        loaded_code_object != nullptr ? *loaded_code_object : hsa_loaded_code_object_t{};
    VirtualLdsRuntimeRegistry::instance().record_load(executable, agent, load_agent, loaded,
                                                      std::move(*hook_metadata));
  }
  return status;
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_EXPORT
#endif

/// @brief ROCR HSA tools entry point.
///
/// @details Saves the incoming `CoreApiTable` function pointers and installs
/// DBT load-time wrappers when the runtime config enables `dbt_guest`. The
/// failed tool list is not modified; ROCR owns that state and passes it for
/// diagnostics only.
extern "C" RJ_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                      uint64_t failed_tool_count,
                                      const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;

  auto config = parse_config();
  if (!config)
    return false;
  const bool signal_backtrace = config->signal_backtrace;
  if (!layer().install(table, std::move(*config)))
    return false;
  // Virtual-LDS resource management lives in its own hook component. Give it
  // only the original runtime entries so allocations cannot recurse through
  // the wrappers installed above.
  rocjitsu::hooks::set_virtual_lds_runtime_api({
      .iterate_agents = layer().iterate_agents(),
      .signal_create = layer().signal_create(),
      .signal_destroy = layer().signal_destroy(),
      .signal_load_scacquire = layer().signal_load_scacquire(),
      .iterate_memory_pools = layer().amd_agent_iterate_memory_pools(),
      .memory_pool_get_info = layer().amd_memory_pool_get_info(),
      .memory_pool_allocate = layer().amd_memory_pool_allocate(),
      .memory_pool_free = layer().amd_memory_pool_free(),
      .agents_allow_access = layer().amd_agents_allow_access(),
  });
  maybe_install_signal_backtrace(signal_backtrace);
  return true;
}

/// @brief ROCR HSA tools unload entry point.
///
/// @details Restores rocjitsu wrappers that are still installed and clears
/// process-local reader state owned by the hook. ROCR also resets the API table
/// after unloading tools, but the hook does its own cleanup so tests and future
/// in-process reload paths do not retain stale translated ELF buffers.
extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
