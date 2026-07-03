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
#include "rocjitsu/code/dbt/virtual_lds_metadata.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "util/arena_alloc.h"
#include "util/intrusive_list.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <execinfo.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <signal.h>
#include <span>
#include <string>
#include <string_view>
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
using rocjitsu::kVirtualLdsFlagBackingPointerInDispatchPacket;
using rocjitsu::kVirtualLdsFlagRuntimeStateBlock;
using rocjitsu::kVirtualLdsFlagWorkgroupIdX;
using rocjitsu::kVirtualLdsFlagWorkgroupIdY;
using rocjitsu::kVirtualLdsFlagWorkgroupIdZ;
using rocjitsu::kVirtualLdsMetadataSectionName;
using rocjitsu::parse_virtual_lds_metadata;
using rocjitsu::read_virtual_lds_descriptor_dispatch_metadata;
using rocjitsu::SHN_UNDEF;
using rocjitsu::SHT_NULL;
using rocjitsu::SHT_STRTAB;
using rocjitsu::TranslationDiagnostic;
using rocjitsu::VirtualLdsKernelMetadata;

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

/// @brief CDNA3/gfx942 hardware LDS allocation limit in bytes.
///
/// @details The minimal HSA headers used by this hook do not expose an agent
/// group-segment limit query. Runtime DBT currently targets gfx942 for the
/// gfx950 guest path, so dispatch selection uses the gfx942 64 KiB LDS limit.
constexpr uint32_t kCdna3HardwareLdsBytes = 64u * 1024u;

/// @brief Number of recent AQL packet IDs rescanned by the polling fallback.
///
/// @details HSA's write index can become visible before the full packet body is
/// stable. The scanner therefore revisits a small tail window instead of treating
/// one early observation as final. Doorbell-triggered scans still maintain the
/// contiguous cursor used for the normal HSA signal path.
constexpr uint64_t kVirtualLdsScannerTailPackets = 512;

/// @brief AMD memory-pool enum values mirrored locally for internal allocation.
///
/// @details Keep these constants out of msgpack or YAML metadata paths: runtime
/// virtual-LDS state is carried by the rocjitsu ELF section and AQL rewrite.
constexpr uint32_t kHsaAmdSegmentGlobal = 0;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagKernargInit = 1u;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagFineGrained = 2u;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagCoarseGrained = 4u;
constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagExtendedScopeFineGrained = 8u;

#define RJ_AMD_EXT_HAS_FIELD(table, field)                                                         \
  ((table) != nullptr &&                                                                           \
   (table)->version.minor_id >= offsetof(AmdExtTable, field) + sizeof((table)->field))

/// @brief Runtime configuration consumed by the HSA tools hook.
struct HookConfig {
  TargetInfo target{};
  std::optional<TargetInfo> source_override;
  std::optional<TargetInfo> guest_target;
  uint32_t host_gpu_id = 0;
  int log_level = kLogDisabled;
  bool signal_backtrace = false;
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

/// @brief Print a best-effort stack trace for fatal hook signals.
void signal_backtrace_handler(int signo, siginfo_t *info, void *context) {
  if (!g_signal_backtrace_enabled.exchange(false, std::memory_order_relaxed)) {
    invoke_previous_signal_handler(signo, info, context);
    return;
  }

  const char header[] = "\n[rocjitsu-hooks] signal backtrace\n";
  [[maybe_unused]] const ssize_t written = ::write(STDERR_FILENO, header, sizeof(header) - 1);
  void *frames[128];
  int count = ::backtrace(frames, 128);
  ::backtrace_symbols_fd(frames, count, STDERR_FILENO);
  invoke_previous_signal_handler(signo, info, context);
}

/// @brief Install fatal-signal backtrace handling when enabled by config.
void maybe_install_signal_backtrace(bool enabled) {
  if (!enabled)
    return;

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

  auto target = parse_target(dbt_guest->host_isa);
  if (!target) {
    std::fprintf(stderr, "[rocjitsu-hooks] invalid dbt_guest.host_isa='%s'\n",
                 dbt_guest->host_isa.c_str());
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
  config.host_gpu_id = dbt_guest->host_gpu_id;
  config.log_level = clamp_log_level(dbt_guest->log_level);
  config.signal_backtrace = dbt_guest->signal_backtrace;
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

/// @brief Return true when an environment flag is explicitly enabled.
[[nodiscard]] bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

/// @brief Return true when direct virtual-LDS dispatch tracing is explicitly enabled.
///
/// @details The normal hook logger is configuration-driven and may be routed away from
/// stderr by the launcher. This trace is intentionally environment-gated and writes
/// directly to stderr so runtime AQL packet-rewrite failures can be diagnosed without
/// changing the public logging defaults.
[[nodiscard]] bool trace_virtual_lds_dispatch_enabled() {
  static const bool enabled = env_flag_enabled("ROCJITSU_TRACE_VIRTUAL_LDS_DISPATCH");
  return enabled;
}

/// @brief Emit an env-gated direct virtual-LDS dispatch trace line.
void trace_virtual_lds_dispatch(const char *format, ...) {
  if (!trace_virtual_lds_dispatch_enabled())
    return;

  std::fprintf(stderr, "[rocjitsu-vlds] ");
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

[[nodiscard]] bool trace_virtual_lds_kernarg_enabled() {
  static const bool enabled = env_flag_enabled("ROCJITSU_TRACE_VIRTUAL_LDS_KERNARG");
  return enabled;
}

[[nodiscard]] bool trace_virtual_lds_userargs_enabled() {
  static const bool enabled = env_flag_enabled("ROCJITSU_TRACE_VIRTUAL_LDS_USERARGS");
  return enabled;
}

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

/// @brief Write code-object bytes for offline DBT debugging when requested.
///
/// @details Large framework libraries can embed hundreds of code objects, and
/// a failing launch path often only exposes a mangled kernel name. This
/// diagnostic dump is env-gated. By default it only fires when the translator
/// skipped at least one symbol; `ROCJITSU_DBT_DUMP_ALL=1` intentionally broadens
/// it to every translated code object so runtime-only descriptor problems can be
/// matched back to the exact source/translated ELF pair.
void dump_code_object_if_requested(std::string_view label, hsa_code_object_reader_t reader,
                                   std::span<const uint8_t> bytes,
                                   std::span<const std::string> skipped_symbols) {
  const bool dump_all = env_flag_enabled("ROCJITSU_DBT_DUMP_ALL");
  if (((skipped_symbols.empty() && !dump_all) || bytes.empty()))
    return;

  const char *dump_dir = std::getenv("ROCJITSU_DBT_DUMP_DIR");
  if (dump_dir == nullptr || dump_dir[0] == '\0')
    return;

  std::string base(dump_dir);
  if (!base.empty() && base.back() != '/')
    base.push_back('/');
  base += std::string(label) + "-reader-" + std::to_string(reader.handle);

  const std::string object_path = base + ".hsaco";
  FILE *object_file = std::fopen(object_path.c_str(), "wb");
  if (object_file == nullptr) {
    log_message(kLogInfo, "failed to open DBT dump %s: errno=%d", object_path.c_str(), errno);
    return;
  }
  const size_t written = std::fwrite(bytes.data(), 1, bytes.size(), object_file);
  const int close_status = std::fclose(object_file);
  if (written != bytes.size() || close_status != 0) {
    log_message(kLogInfo, "failed to write DBT dump %s: wrote=%zu size=%zu close=%d errno=%d",
                object_path.c_str(), written, bytes.size(), close_status, errno);
    return;
  }

  if (!skipped_symbols.empty()) {
    const std::string symbols_path = base + ".skipped.txt";
    FILE *symbols_file = std::fopen(symbols_path.c_str(), "w");
    if (symbols_file != nullptr) {
      for (const std::string &symbol : skipped_symbols)
        std::fprintf(symbols_file, "%s\n", symbol.c_str());
      std::fclose(symbols_file);
    }
  }
  log_message(kLogInfo, "dumped DBT %s code object to %s", std::string(label).c_str(),
              object_path.c_str());
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

  /// @brief Record bytes backing a code-object reader.
  /// @param reader HSA reader handle used as the lookup key.
  /// @param bytes Start of the ELF image.
  /// @param size Size of the ELF image in bytes.
  /// @param owned Optional owned storage for translated ELF bytes.
  [[nodiscard]] bool store(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size,
                           std::vector<uint8_t> *owned) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        delete entry->owned;
        entry->bytes = bytes;
        entry->size = size;
        entry->owned = owned;
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
  /// @returns true when @p bytes and @p size were populated.
  bool lookup(hsa_code_object_reader_t reader, const uint8_t **bytes, size_t *size) {
    std::shared_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        *bytes = entry->bytes;
        *size = entry->size;
        return true;
      }
    }
    return false;
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
    Entry(uint64_t h, const uint8_t *b, size_t s, std::vector<uint8_t> *o)
        : handle(h), bytes(b), size(s), owned(o) {}

    uint64_t handle = 0;
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::vector<uint8_t> *owned = nullptr;
  };

  /// @brief Destroy one reader entry and release optional owned ELF storage.
  void destroy_entry(Entry *entry) {
    delete entry->owned;
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

[[nodiscard]] bool elf_range_in_bounds(size_t image_size, uint64_t offset, uint64_t size) {
  const uint64_t limit = static_cast<uint64_t>(image_size);
  return offset <= limit && size <= limit - offset;
}

/// @brief Parse rocjitsu virtual-LDS records from a translated ELF image.
///
/// @details ROCR never consumes this section. The hook parses the section table
/// directly so it can keep dispatch facts for later AQL translation without
/// reading or editing AMD msgpack notes.
[[nodiscard]] std::optional<std::vector<VirtualLdsKernelMetadata>>
parse_virtual_lds_metadata_section(std::span<const uint8_t> image) {
  if (image.size() < sizeof(Elf64_Ehdr))
    return std::vector<VirtualLdsKernelMetadata>{};

  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) != 0 ||
      header.e_ident[rocjitsu::EI_CLASS] != ELFCLASS64 || header.e_machine != EM_AMDGPU) {
    return std::vector<VirtualLdsKernelMetadata>{};
  }
  if (header.e_shoff == 0 || header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shnum == 0)
    return std::vector<VirtualLdsKernelMetadata>{};
  if (header.e_shstrndx == SHN_UNDEF || header.e_shstrndx >= header.e_shnum)
    return std::nullopt;
  if (!elf_range_in_bounds(image.size(), header.e_shoff,
                           static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr))) {
    return std::nullopt;
  }

  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));

  const Elf64_Shdr &shstrtab = sections[header.e_shstrndx];
  if (shstrtab.sh_type != SHT_STRTAB ||
      !elf_range_in_bounds(image.size(), shstrtab.sh_offset, shstrtab.sh_size)) {
    return std::nullopt;
  }
  const char *section_names = reinterpret_cast<const char *>(image.data() + shstrtab.sh_offset);

  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type == SHT_NULL || section.sh_name >= shstrtab.sh_size)
      continue;
    const size_t max_name = static_cast<size_t>(shstrtab.sh_size - section.sh_name);
    const std::string_view name(section_names + section.sh_name,
                                strnlen(section_names + section.sh_name, max_name));
    if (name != kVirtualLdsMetadataSectionName)
      continue;
    if (!elf_range_in_bounds(image.size(), section.sh_offset, section.sh_size))
      return std::nullopt;
    return parse_virtual_lds_metadata(image.subspan(static_cast<size_t>(section.sh_offset),
                                                    static_cast<size_t>(section.sh_size)));
  }

  return std::vector<VirtualLdsKernelMetadata>{};
}

[[nodiscard]] std::string normalize_kernel_symbol_name(std::string_view symbol_name) {
  constexpr std::string_view kDescriptorSuffix = ".kd";
  if (symbol_name.ends_with(kDescriptorSuffix))
    symbol_name.remove_suffix(kDescriptorSuffix.size());
  return std::string(symbol_name);
}

/// @brief Process-local virtual-LDS metadata resolved through HSA symbol APIs.
///
/// @details Load-time DBT can embed static facts in `.rocjitsu.lds`, but the
/// runtime only learns the loaded kernel-object address when the application
/// queries `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT`. This registry bridges
/// those two phases. Later dispatch translation can consult the resolved normal
/// and virtual kernel-object addresses without touching msgpack metadata.
class VirtualLdsRuntimeRegistry {
public:
  struct ResolvedKernel {
    uint64_t executable = 0;
    VirtualLdsKernelMetadata metadata;
    uint64_t normal_kernel_object = 0;
    uint64_t virtual_kernel_object = 0;
  };

  static VirtualLdsRuntimeRegistry &instance() {
    static VirtualLdsRuntimeRegistry registry;
    return registry;
  }

  void record_load(hsa_executable_t executable, hsa_agent_t guest_agent, hsa_agent_t host_agent,
                   hsa_loaded_code_object_t loaded_code_object,
                   std::vector<VirtualLdsKernelMetadata> metadata) {
    if (metadata.empty())
      return;
    std::lock_guard lock(mutex_);
    loads_.push_back(LoadEntry{.executable = executable.handle,
                               .guest_agent = guest_agent.handle,
                               .host_agent = host_agent.handle,
                               .loaded_code_object = loaded_code_object.handle,
                               .metadata = std::move(metadata)});
    log_message(kLogInfo, "registered %zu virtual-LDS records exec=%llu loaded=%llu",
                loads_.back().metadata.size(),
                static_cast<unsigned long long>(loads_.back().executable),
                static_cast<unsigned long long>(loads_.back().loaded_code_object));
  }

  void record_symbol(hsa_executable_t executable, std::string_view symbol_name,
                     hsa_executable_symbol_t symbol) {
    if (symbol.handle == 0 || symbol_name.empty())
      return;
    const std::string kernel_name = normalize_kernel_symbol_name(symbol_name);
    std::lock_guard lock(mutex_);
    for (size_t load_index = loads_.size(); load_index > 0; --load_index) {
      LoadEntry &load = loads_[load_index - 1];
      if (load.executable != executable.handle)
        continue;
      auto record =
          std::ranges::find_if(load.metadata, [&](const VirtualLdsKernelMetadata &metadata) {
            return metadata.kernel_name == kernel_name;
          });
      if (record == load.metadata.end())
        continue;
      symbols_[symbol.handle] =
          ResolvedKernel{.executable = executable.handle, .metadata = *record};
      // The load-time metadata vector is only needed to associate future symbol
      // queries by name. Once this symbol is resolved, dispatch rewriting uses
      // the copied ResolvedKernel record, so avoid keeping a duplicate
      // string-bearing record alive until executable teardown.
      load.metadata.erase(record);
      if (load.metadata.empty())
        loads_.erase(loads_.begin() + static_cast<std::ptrdiff_t>(load_index - 1));
      log_message(kLogDebug, "associated virtual-LDS symbol=%llu kernel=%s",
                  static_cast<unsigned long long>(symbol.handle), kernel_name.c_str());
      return;
    }
  }

  void note_kernel_object(hsa_executable_symbol_t symbol, uint64_t kernel_object) {
    if (symbol.handle == 0 || kernel_object == 0)
      return;
    std::lock_guard lock(mutex_);
    auto it = symbols_.find(symbol.handle);
    if (it == symbols_.end())
      return;

    ResolvedKernel &resolved = it->second;
    if (resolved.metadata.normal_descriptor_vaddr == 0 ||
        kernel_object < resolved.metadata.normal_descriptor_vaddr) {
      return;
    }

    const uint64_t load_base = kernel_object - resolved.metadata.normal_descriptor_vaddr;
    resolved.normal_kernel_object = kernel_object;
    resolved.virtual_kernel_object = load_base + resolved.metadata.virtual_descriptor_vaddr;
    log_message(kLogDebug,
                "resolved virtual-LDS kernel=%s normal_object=0x%llx virtual_object=0x%llx "
                "static_lds=%u base_sgpr=%u",
                resolved.metadata.kernel_name.c_str(),
                static_cast<unsigned long long>(resolved.normal_kernel_object),
                static_cast<unsigned long long>(resolved.virtual_kernel_object),
                resolved.metadata.static_lds_bytes, resolved.metadata.virtual_lds_base_sgpr);
  }

  /// @brief Return the virtual descriptor object when the normal descriptor cannot run on host.
  ///
  /// @details Static LDS larger than the host hardware limit is never a valid
  /// normal dispatch on gfx942. Exposing the virtual sidecar at symbol-query time
  /// avoids a race with ROCR/direct-doorbell packet construction: the runtime
  /// builds the AQL packet from the already-virtual descriptor and reports zero
  /// static group segment size for the same symbol.
  [[nodiscard]] std::optional<uint64_t>
  static_oversized_virtual_kernel_object(hsa_executable_symbol_t symbol) {
    if (symbol.handle == 0)
      return std::nullopt;
    std::lock_guard lock(mutex_);
    auto it = symbols_.find(symbol.handle);
    if (it == symbols_.end())
      return std::nullopt;
    const ResolvedKernel &resolved = it->second;
    if (resolved.metadata.static_lds_bytes <= kCdna3HardwareLdsBytes ||
        resolved.virtual_kernel_object == 0 ||
        resolved.virtual_kernel_object == resolved.normal_kernel_object) {
      return std::nullopt;
    }
    return resolved.virtual_kernel_object;
  }

  /// @brief Return the private segment size belonging to the virtual descriptor.
  ///
  /// @details When rocjitsu exposes the virtual descriptor from the kernel-object
  /// symbol query, every descriptor-derived symbol attribute that ROCR may use to
  /// build an AQL packet must describe the same virtual kernel. In particular,
  /// private scratch is descriptor-specific after DBT rewrites that spill values.
  [[nodiscard]] std::optional<uint32_t>
  static_oversized_virtual_private_segment_size(hsa_executable_symbol_t symbol) {
    const auto virtual_kernel_object = static_oversized_virtual_kernel_object(symbol);
    if (!virtual_kernel_object)
      return std::nullopt;

    rocr::llvm::amdhsa::kernel_descriptor_t virtual_descriptor{};
    std::memcpy(&virtual_descriptor, reinterpret_cast<const void *>(*virtual_kernel_object),
                sizeof(virtual_descriptor));
    return virtual_descriptor.private_segment_fixed_size;
  }

  /// @brief Return true when static LDS should be hidden from ROCR for this symbol.
  [[nodiscard]] bool uses_static_oversized_virtual_descriptor(hsa_executable_symbol_t symbol) {
    if (symbol.handle == 0)
      return false;
    std::lock_guard lock(mutex_);
    auto it = symbols_.find(symbol.handle);
    if (it == symbols_.end())
      return false;
    const ResolvedKernel &resolved = it->second;
    return resolved.metadata.static_lds_bytes > kCdna3HardwareLdsBytes &&
           resolved.metadata.virtual_descriptor_vaddr != 0 &&
           resolved.metadata.virtual_descriptor_vaddr != resolved.metadata.normal_descriptor_vaddr;
  }

  /// @brief Find virtual-LDS metadata for a descriptor object in an AQL packet.
  ///
  /// @details Dispatch packets carry the loaded kernel-object address, not an
  /// HSA symbol handle. Symbol resolution records both addresses after ROCR
  /// reports the normal descriptor object. Static-oversized kernels may already
  /// have the virtual descriptor in the packet because rocjitsu exposes it from
  /// `hsa_executable_symbol_get_info`, so both addresses must resolve to the
  /// same rewrite plan.
  [[nodiscard]] std::optional<ResolvedKernel> find_by_kernel_object(uint64_t kernel_object) {
    if (kernel_object == 0)
      return std::nullopt;
    std::lock_guard lock(mutex_);
    for (const auto &[symbol_handle, resolved] : symbols_) {
      (void)symbol_handle;
      if ((resolved.normal_kernel_object == kernel_object ||
           resolved.virtual_kernel_object == kernel_object) &&
          resolved.virtual_kernel_object != 0)
        return resolved;
    }
    return std::nullopt;
  }

  void erase_executable(hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    for (auto it = loads_.begin(); it != loads_.end();) {
      if (it->executable == executable.handle)
        it = loads_.erase(it);
      else
        ++it;
    }
    for (auto it = symbols_.begin(); it != symbols_.end();) {
      if (it->second.executable == executable.handle)
        it = symbols_.erase(it);
      else
        ++it;
    }
  }

  void clear() {
    std::lock_guard lock(mutex_);
    loads_.clear();
    symbols_.clear();
  }

private:
  struct LoadEntry {
    uint64_t executable = 0;
    uint64_t guest_agent = 0;
    uint64_t host_agent = 0;
    uint64_t loaded_code_object = 0;
    std::vector<VirtualLdsKernelMetadata> metadata;
  };

  std::mutex mutex_;
  std::vector<LoadEntry> loads_;
  std::unordered_map<uint64_t, ResolvedKernel> symbols_;
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
                                                             hsa_amd_sdma_engine_id_t *engine_id);

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
  X(amd_queue_intercept_register, amd_ext_,                                                        \
    amd_ext_ != nullptr && RJ_AMD_EXT_HAS_FIELD(amd_ext_, hsa_amd_queue_intercept_register_fn),    \
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
    amd_ext_ != nullptr && RJ_AMD_EXT_HAS_FIELD(amd_ext_, hsa_amd_memory_async_batch_copy_fn),     \
    true, hsa_amd_memory_async_batch_copy_fn, rj_amd_memory_async_batch_copy,                      \
    hsa_amd_memory_async_batch_copy_fn_t)                                                          \
  X(amd_agent_preload, amd_ext_,                                                                   \
    amd_ext_ != nullptr && RJ_AMD_EXT_HAS_FIELD(amd_ext_, hsa_amd_agent_preload_fn), true,         \
    hsa_amd_agent_preload_fn, rj_amd_agent_preload, hsa_amd_agent_preload_fn_t)                    \
  X(amd_queue_intercept_create, amd_ext_,                                                          \
    amd_ext_ != nullptr && RJ_AMD_EXT_HAS_FIELD(amd_ext_, hsa_amd_queue_intercept_create_fn),      \
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
    VirtualLdsRuntimeRegistry::instance().clear();
    clear_virtual_lds_dispatch_queues();
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

private:
  /// @brief Agent discovery callback state.
  struct AgentSearchData {
    AgentMapper *self = nullptr;
    TargetInfo guest{};
    TargetInfo host{};
    std::optional<uint32_t> host_node_id;
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
    if (search->self->agent_has_target(agent, search->guest))
      search->self->guest_ = agent;
    if (search->self->host_.handle == 0 && search->self->agent_matches_host(agent, *search))
      search->self->host_ = agent;
    if (search->self->guest_.handle != 0 && search->self->host_.handle != 0)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Lazily discover the guest-to-host agent mapping.
  void ensure_discovered() {
    std::lock_guard lock(mutex_);
    if (discovered_)
      return;
    discovered_ = true;

    auto config = layer().config();
    if (!config || !config->guest_target)
      return;

    auto *iterate_agents = layer().iterate_agents();
    if (!iterate_agents)
      return;

    std::optional<uint32_t> host_node_id;
    if (config->host_gpu_id != 0) {
      host_node_id = node_id_for_kfd_gpu_id(config->host_gpu_id);
      if (!host_node_id) {
        std::fprintf(stderr,
                     "[rocjitsu-hooks] failed to find topology node for host KFD gpu_id=%u\n",
                     config->host_gpu_id);
        return;
      }
    }

    AgentSearchData search{this, *config->guest_target, config->target, host_node_id};
    hsa_status_t status = iterate_agents(agent_callback, &search);
    has_mapping_ = (status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK) &&
                   guest_.handle != 0 && host_.handle != 0 && guest_.handle != host_.handle;
    if (has_mapping_) {
      log_message(kLogInfo, "mapped guest agent=%llu to host agent=%llu",
                  static_cast<unsigned long long>(guest_.handle),
                  static_cast<unsigned long long>(host_.handle));
      if (config->log_level > kLogDisabled) {
        std::optional<uint32_t> selected_node_id = agent_driver_node_id(host_);
        std::fprintf(stderr,
                     "[rocjitsu-hooks] selected host agent=%llu for guest agent=%llu "
                     "host_gpu_id=%u host_node_id=%u\n",
                     static_cast<unsigned long long>(host_.handle),
                     static_cast<unsigned long long>(guest_.handle), config->host_gpu_id,
                     selected_node_id.value_or(0));
      }
    } else {
      std::fprintf(stderr, "[rocjitsu-hooks] failed to find guest/host HSA agents for DBT\n");
    }
  }

  std::mutex mutex_;
  bool discovered_ = false;
  bool has_mapping_ = false;
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

    std::unordered_map<uint64_t, uint64_t> discovered;
    if (iterate_pools != nullptr && get_info != nullptr && guest.handle != 0 && host.handle != 0) {
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
          log_message(kLogDebug, "mapped guest pool=%llu to host pool=%llu",
                      static_cast<unsigned long long>(guest_pool.pool.handle),
                      static_cast<unsigned long long>(host_pool->pool.handle));
        }
      }
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

hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(code_object, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr && code_object != nullptr) {
    if (!CodeObjectReaderRegistry::instance().store(
            *code_object_reader, static_cast<const uint8_t *>(code_object), size, nullptr)) {
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
    // File-backed readers do not expose stable ELF bytes through the later
    // load-agent callback. Hook the create path anyway so users get a direct
    // warning instead of an unexplained INVALID_ISA pass-through failure.
    std::fprintf(stderr,
                 "[rocjitsu-hooks] file-backed code-object reader=%llu is not translated; use "
                 "hsa_code_object_reader_create_from_memory for DBT hook translation\n",
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

/// @brief Return true when an AQL packet header names a kernel dispatch.
[[nodiscard]] uint32_t aql_packet_type(uint16_t header) {
  constexpr uint32_t kPacketTypeMask = (1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u;
  return (header >> HSA_PACKET_HEADER_TYPE) & kPacketTypeMask;
}

/// @brief Return true when an AQL packet header names a kernel dispatch.
[[nodiscard]] bool is_kernel_dispatch_packet(const hsa_kernel_dispatch_packet_t &packet) {
  return aql_packet_type(packet.header) == HSA_PACKET_TYPE_KERNEL_DISPATCH;
}

/// @brief GPU-visible virtual-LDS state consumed by the translated entry prologue.
struct VirtualLdsDispatchState {
  uint64_t backing_base = 0;
  uint32_t stride_x = 0;
  uint32_t stride_y = 0;
  uint32_t stride_z = 0;
  uint32_t reserved = 0;
};

static_assert(sizeof(VirtualLdsDispatchState) == 24);

/// @brief Per-dispatch buffers owned by a queue slot after virtual-LDS rewrite.
struct VirtualLdsDispatchBuffers {
  void *kernarg = nullptr;
  void *backing = nullptr;
  void *state = nullptr;
  uint64_t virtual_kernel_object = 0;
};

/// @brief Free one runtime allocation through the original AMD extension entry.
void free_virtual_lds_allocation(void *ptr) {
  if (ptr == nullptr)
    return;
  auto *free_fn = layer().amd_memory_pool_free();
  if (free_fn == nullptr)
    return;
  (void)free_fn(ptr);
}

/// @brief Release any virtual-LDS buffers still attached to a queue slot.
void release_virtual_lds_buffers(VirtualLdsDispatchBuffers &buffers) {
  free_virtual_lds_allocation(buffers.kernarg);
  free_virtual_lds_allocation(buffers.backing);
  free_virtual_lds_allocation(buffers.state);
  buffers = {};
}

/// @brief Internal allocator for virtual-LDS backing storage and optional extended kernargs.
class VirtualLdsDispatchAllocator {
public:
  static VirtualLdsDispatchAllocator &instance() {
    static VirtualLdsDispatchAllocator allocator;
    return allocator;
  }

  /// @brief Allocate GPU-visible backing memory plus optional kernarg/state copies.
  [[nodiscard]] bool allocate(hsa_agent_t host_agent, size_t backing_bytes, size_t kernarg_bytes,
                              size_t state_bytes, VirtualLdsDispatchBuffers &buffers) {
    if (host_agent.handle == 0 || backing_bytes == 0)
      return false;

    const bool need_kernarg_pool = kernarg_bytes != 0 || state_bytes != 0;
    const auto pools = pools_for_agent(host_agent, need_kernarg_pool);
    if (!pools || !pools->has_backing_pool || (need_kernarg_pool && !pools->has_kernarg_pool)) {
      log_message(kLogInfo,
                  "virtual-LDS dispatch cannot find backing/kernarg pools for host agent=%llu",
                  static_cast<unsigned long long>(host_agent.handle));
      trace_virtual_lds_dispatch(
          "virtual-LDS pool discovery failed host_agent=%llu has_pools=%d has_backing=%d "
          "has_kernarg=%d need_kernarg=%d",
          static_cast<unsigned long long>(host_agent.handle), pools.has_value(),
          pools ? pools->has_backing_pool : false, pools ? pools->has_kernarg_pool : false,
          need_kernarg_pool);
      return false;
    }

    void *backing = nullptr;
    if (!allocate_from_pool(pools->backing_pool, backing_bytes, &backing))
      return false;
    if (!allow_agent_access(host_agent, backing)) {
      free_virtual_lds_allocation(backing);
      return false;
    }

    void *kernarg = nullptr;
    if (need_kernarg_pool) {
      if (kernarg_bytes != 0) {
        if (!allocate_from_pool(pools->kernarg_pool, kernarg_bytes, &kernarg)) {
          free_virtual_lds_allocation(backing);
          return false;
        }
        if (!allow_agent_access(host_agent, kernarg)) {
          free_virtual_lds_allocation(kernarg);
          free_virtual_lds_allocation(backing);
          return false;
        }
      }
    }

    void *state = nullptr;
    if (state_bytes != 0) {
      if (!allocate_from_pool(pools->kernarg_pool, state_bytes, &state)) {
        free_virtual_lds_allocation(kernarg);
        free_virtual_lds_allocation(backing);
        return false;
      }
      if (!allow_agent_access(host_agent, state)) {
        free_virtual_lds_allocation(state);
        free_virtual_lds_allocation(kernarg);
        free_virtual_lds_allocation(backing);
        return false;
      }
    }

    buffers.backing = backing;
    buffers.kernarg = kernarg;
    buffers.state = state;
    return true;
  }

  /// @brief Drop cached pool discovery on HSA unload.
  void clear() {
    std::lock_guard lock(mutex_);
    pools_by_agent_.clear();
  }

private:
  struct Pools {
    hsa_amd_memory_pool_t backing_pool{};
    hsa_amd_memory_pool_t kernarg_pool{};
    bool has_backing_pool = false;
    bool has_kernarg_pool = false;
  };

  struct PoolSearch {
    hsa_amd_memory_pool_get_info_fn_t get_info = nullptr;
    Pools pools;
  };

  struct GlobalKernargPoolSearch {
    hsa_amd_agent_iterate_memory_pools_fn_t iterate_pools = nullptr;
    hsa_amd_memory_pool_get_info_fn_t get_info = nullptr;
    hsa_amd_memory_pool_t kernarg_pool{};
    bool found = false;
  };

  /// @brief Collect candidate host pools for internal dispatch allocations.
  static hsa_status_t collect_pool(hsa_amd_memory_pool_t pool, void *data) {
    auto *search = static_cast<PoolSearch *>(data);
    if (search == nullptr || search->get_info == nullptr)
      return HSA_STATUS_ERROR;

    uint32_t segment = 0;
    uint32_t flags = 0;
    bool runtime_alloc_allowed = false;
    if (search->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;
    (void)search->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    (void)search->get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                           &runtime_alloc_allowed);

    if (segment != kHsaAmdSegmentGlobal || !runtime_alloc_allowed)
      return HSA_STATUS_SUCCESS;

    const bool kernarg = (flags & kHsaAmdMemoryPoolGlobalFlagKernargInit) != 0;
    const bool global_backing =
        (flags &
         (kHsaAmdMemoryPoolGlobalFlagFineGrained | kHsaAmdMemoryPoolGlobalFlagCoarseGrained |
          kHsaAmdMemoryPoolGlobalFlagExtendedScopeFineGrained)) != 0;

    if (kernarg && !search->pools.has_kernarg_pool) {
      search->pools.kernarg_pool = pool;
      search->pools.has_kernarg_pool = true;
    }

    // Prefer ordinary global pools for LDS backing storage. Kernarg pools are
    // meant for ABI argument blocks, while the backing buffer can be large and
    // lives for the whole dispatch.
    if (!kernarg && global_backing && !search->pools.has_backing_pool) {
      search->pools.backing_pool = pool;
      search->pools.has_backing_pool = true;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Search one agent for a runtime-allocatable kernarg pool.
  static hsa_status_t collect_agent_kernarg_pool(hsa_agent_t agent, void *data) {
    auto *search = static_cast<GlobalKernargPoolSearch *>(data);
    if (search == nullptr || search->iterate_pools == nullptr || search->get_info == nullptr ||
        search->found) {
      return HSA_STATUS_SUCCESS;
    }

    PoolSearch agent_pools;
    agent_pools.get_info = search->get_info;
    (void)search->iterate_pools(agent, collect_pool, &agent_pools);
    if (agent_pools.pools.has_kernarg_pool) {
      search->kernarg_pool = agent_pools.pools.kernarg_pool;
      search->found = true;
    }
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Find a process-visible kernarg pool when the GPU agent does not expose one.
  [[nodiscard]] static std::optional<hsa_amd_memory_pool_t>
  find_global_kernarg_pool(hsa_amd_agent_iterate_memory_pools_fn_t iterate_pools,
                           hsa_amd_memory_pool_get_info_fn_t get_info) {
    auto *iterate_agents = layer().iterate_agents();
    if (iterate_agents == nullptr || iterate_pools == nullptr || get_info == nullptr)
      return std::nullopt;

    GlobalKernargPoolSearch search;
    search.iterate_pools = iterate_pools;
    search.get_info = get_info;
    (void)iterate_agents(collect_agent_kernarg_pool, &search);
    if (!search.found)
      return std::nullopt;
    return search.kernarg_pool;
  }

  [[nodiscard]] std::optional<Pools> pools_for_agent(hsa_agent_t host_agent,
                                                     bool need_kernarg_pool) {
    {
      std::lock_guard lock(mutex_);
      auto it = pools_by_agent_.find(host_agent.handle);
      if (it != pools_by_agent_.end() && (!need_kernarg_pool || it->second.has_kernarg_pool))
        return it->second;
    }

    auto *iterate_pools = layer().amd_agent_iterate_memory_pools();
    auto *get_info = layer().amd_memory_pool_get_info();
    if (iterate_pools == nullptr || get_info == nullptr)
      return std::nullopt;

    PoolSearch search;
    search.get_info = get_info;
    const hsa_status_t status = iterate_pools(host_agent, collect_pool, &search);
    if (status == HSA_STATUS_SUCCESS && need_kernarg_pool && !search.pools.has_kernarg_pool) {
      if (auto kernarg_pool = find_global_kernarg_pool(iterate_pools, get_info)) {
        search.pools.kernarg_pool = *kernarg_pool;
        search.pools.has_kernarg_pool = true;
        trace_virtual_lds_dispatch(
            "virtual-LDS using global kernarg pool host_agent=%llu kernarg_pool=%llu",
            static_cast<unsigned long long>(host_agent.handle),
            static_cast<unsigned long long>(kernarg_pool->handle));
      }
    }
    if (status != HSA_STATUS_SUCCESS || !search.pools.has_backing_pool ||
        (need_kernarg_pool && !search.pools.has_kernarg_pool)) {
      trace_virtual_lds_dispatch(
          "virtual-LDS iterate pools failed host_agent=%llu status=%d has_backing=%d "
          "backing_pool=%llu has_kernarg=%d kernarg_pool=%llu need_kernarg=%d",
          static_cast<unsigned long long>(host_agent.handle), static_cast<int>(status),
          search.pools.has_backing_pool,
          static_cast<unsigned long long>(search.pools.backing_pool.handle),
          search.pools.has_kernarg_pool,
          static_cast<unsigned long long>(search.pools.kernarg_pool.handle), need_kernarg_pool);
      return std::nullopt;
    }

    std::lock_guard lock(mutex_);
    auto [it, inserted] = pools_by_agent_.insert_or_assign(host_agent.handle, search.pools);
    (void)inserted;
    return it->second;
  }

  [[nodiscard]] static bool allocate_from_pool(hsa_amd_memory_pool_t pool, size_t size,
                                               void **ptr) {
    auto *allocate = layer().amd_memory_pool_allocate();
    if (allocate == nullptr || ptr == nullptr)
      return false;
    *ptr = nullptr;
    const hsa_status_t status = allocate(pool, size, 0, ptr);
    if (status != HSA_STATUS_SUCCESS || *ptr == nullptr) {
      log_message(kLogInfo, "virtual-LDS allocation failed pool=%llu size=%zu status=%d",
                  static_cast<unsigned long long>(pool.handle), size, static_cast<int>(status));
      trace_virtual_lds_dispatch("virtual-LDS memory_pool_allocate failed pool=%llu size=%zu "
                                 "status=%d ptr=%p",
                                 static_cast<unsigned long long>(pool.handle), size,
                                 static_cast<int>(status), ptr ? *ptr : nullptr);
      return false;
    }
    return true;
  }

  [[nodiscard]] static bool allow_agent_access(hsa_agent_t host_agent, const void *ptr) {
    auto *allow_access = layer().amd_agents_allow_access();
    if (allow_access == nullptr || ptr == nullptr)
      return false;
    const hsa_status_t status = allow_access(1, &host_agent, nullptr, ptr);
    if (status != HSA_STATUS_SUCCESS) {
      log_message(kLogInfo, "virtual-LDS allocation access failed agent=%llu ptr=%p status=%d",
                  static_cast<unsigned long long>(host_agent.handle), ptr,
                  static_cast<int>(status));
      trace_virtual_lds_dispatch("virtual-LDS allow_access failed agent=%llu ptr=%p status=%d",
                                 static_cast<unsigned long long>(host_agent.handle), ptr,
                                 static_cast<int>(status));
      return false;
    }
    return true;
  }

  std::mutex mutex_;
  std::unordered_map<uint64_t, Pools> pools_by_agent_;
};

/// @brief Queue-doorbell rewrite state for virtual-LDS dispatch selection.
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
    queues_by_doorbell_[state.doorbell_signal] = queue_key;
    queues_by_ptr_[queue_key] = std::move(state);
    log_message(kLogDebug, "tracked queue=%p doorbell=%llu size=%u host_agent=%llu",
                static_cast<void *>(queue),
                static_cast<unsigned long long>(queue->doorbell_signal.handle), queue->size,
                static_cast<unsigned long long>(host_agent.handle));
    trace_virtual_lds_dispatch(
        "tracked queue=%p type=%u size=%u doorbell=%llu host_agent=%llu interceptor=%d "
        "load_write_index=%d",
        static_cast<void *>(queue), static_cast<unsigned>(queue->type), queue->size,
        static_cast<unsigned long long>(queue->doorbell_signal.handle),
        static_cast<unsigned long long>(host_agent.handle), uses_packet_interceptor,
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
    if (state.queue->type == HSA_QUEUE_TYPE_SINGLE && packet_id >= state.next_packet_id &&
        packet_id - state.next_packet_id < state.queue->size) {
      rewrite_packet_range(state, state.next_packet_id, packet_id + 1, true);
      return;
    }

    // Multi queues may ring arbitrary packet IDs. Also handle unusual single
    // queue jumps by at least rewriting the packet named by the doorbell value.
    const bool ready = rewrite_packet(state, packet_id);
    if (ready && packet_id >= state.next_packet_id)
      state.next_packet_id = packet_id + 1;
  }

  /// @brief Release all tracked queues during HSA tool unload.
  void clear() {
    scanner_.request_stop();
    if (scanner_.joinable())
      scanner_.join();

    std::lock_guard lock(mutex_);
    for (auto &[queue, state] : queues_by_ptr_) {
      (void)queue;
      release_queue(state);
    }
    queues_by_ptr_.clear();
    queues_by_doorbell_.clear();
    VirtualLdsDispatchAllocator::instance().clear();
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
      for (uint64_t index = 0; index < pkt_count; ++index) {
        hsa_kernel_dispatch_packet_t &packet = packets[static_cast<size_t>(index)];
        const uint16_t header = packet.header;
        if (!is_kernel_dispatch_packet(packet))
          continue;
        if (packet.group_segment_size > kCdna3HardwareLdsBytes) {
          trace_virtual_lds_dispatch(
              "intercept oversized candidate packet=%llu header=0x%x object=0x%llx "
              "packet_group=%u kernarg=%p",
              static_cast<unsigned long long>(user_pkt_index + index), header,
              static_cast<unsigned long long>(packet.kernel_object), packet.group_segment_size,
              packet.kernarg_address);
        }

        auto plan = rewrite_plan_for_packet(packet);
        if (!plan) {
          if (packet.group_segment_size > kCdna3HardwareLdsBytes) {
            trace_virtual_lds_dispatch(
                "no virtual-LDS plan intercept_packet=%llu header=0x%x object=0x%llx "
                "packet_group=%u",
                static_cast<unsigned long long>(user_pkt_index + index), header,
                static_cast<unsigned long long>(packet.kernel_object), packet.group_segment_size);
          }
          continue;
        }

        // AQL's group_segment_size is the total per-workgroup group memory
        // request. It must already cover the descriptor's fixed group segment
        // plus any dynamic group segment variables, so do not add the fixed
        // size again when sizing the virtual backing allocation.
        const uint64_t requested_lds =
            std::max<uint64_t>(plan->static_lds_bytes, packet.group_segment_size);
        if (requested_lds <= kCdna3HardwareLdsBytes)
          continue;

        const bool packet_pointer = plan->backing_pointer_in_dispatch_packet;
        const uint64_t pointer_offset = plan->backing_pointer_kernarg_offset;
        if (packet_pointer) {
          if (pointer_offset != offsetof(hsa_kernel_dispatch_packet_t, reserved2)) {
            log_message(kLogInfo,
                        "virtual-LDS metadata has invalid dispatch-packet pointer offset "
                        "kernel=%s",
                        plan->kernel_name.c_str());
            continue;
          }
        } else if (pointer_offset < plan->kernarg_size ||
                   pointer_offset >
                       std::numeric_limits<size_t>::max() - sizeof(VirtualLdsDispatchState)) {
          log_message(kLogInfo, "virtual-LDS metadata has invalid kernarg pointer offset kernel=%s",
                      plan->kernel_name.c_str());
          trace_virtual_lds_dispatch(
              "virtual-LDS intercept invalid pointer offset packet=%llu kernel=%s "
              "kernarg_size=%u pointer_offset=%llu",
              static_cast<unsigned long long>(user_pkt_index + index), plan->kernel_name.c_str(),
              plan->kernarg_size, static_cast<unsigned long long>(pointer_offset));
          continue;
        }

        const auto geometry = compute_virtual_lds_geometry(packet, *plan, requested_lds);
        if (!geometry) {
          trace_virtual_lds_dispatch(
              "virtual-LDS intercept geometry failed packet=%llu kernel=%s requested=%llu "
              "grid=%u,%u,%u workgroup=%u,%u,%u",
              static_cast<unsigned long long>(user_pkt_index + index), plan->kernel_name.c_str(),
              static_cast<unsigned long long>(requested_lds), packet.grid_size_x,
              packet.grid_size_y, packet.grid_size_z, packet.workgroup_size_x,
              packet.workgroup_size_y, packet.workgroup_size_z);
          continue;
        }

        if (!packet_pointer && plan->kernarg_size != 0 && packet.kernarg_address == nullptr) {
          log_message(kLogInfo, "virtual-LDS intercept dispatch has null source kernarg kernel=%s",
                      plan->kernel_name.c_str());
          trace_virtual_lds_dispatch("virtual-LDS intercept null kernarg packet=%llu kernel=%s",
                                     static_cast<unsigned long long>(user_pkt_index + index),
                                     plan->kernel_name.c_str());
          continue;
        }

        const size_t kernarg_bytes =
            packet_pointer ? 0
                           : static_cast<size_t>(pointer_offset + sizeof(VirtualLdsDispatchState));
        const size_t state_bytes = packet_pointer ? sizeof(VirtualLdsDispatchState) : 0;
        const size_t backing_bytes = geometry->backing_bytes;
        VirtualLdsDispatchBuffers buffers;
        if (!VirtualLdsDispatchAllocator::instance().allocate(
                state.host_agent, backing_bytes, kernarg_bytes, state_bytes, buffers)) {
          trace_virtual_lds_dispatch(
              "virtual-LDS intercept allocation failed packet=%llu kernel=%s backing=%zu "
              "kernarg=%zu state=%zu host_agent=%llu",
              static_cast<unsigned long long>(user_pkt_index + index), plan->kernel_name.c_str(),
              backing_bytes, kernarg_bytes, state_bytes,
              static_cast<unsigned long long>(state.host_agent.handle));
          continue;
        }

        const uint64_t backing_address = reinterpret_cast<uintptr_t>(buffers.backing);
        const VirtualLdsDispatchState dispatch_state =
            make_virtual_lds_dispatch_state(backing_address, *geometry);
        const uint64_t state_address =
            packet_pointer ? reinterpret_cast<uintptr_t>(buffers.state)
                           : reinterpret_cast<uintptr_t>(static_cast<uint8_t *>(buffers.kernarg) +
                                                         pointer_offset);
        if (packet_pointer) {
          std::memcpy(buffers.state, &dispatch_state, sizeof(dispatch_state));
          packet.reserved2 = state_address;
          // ROCR intercept queues hand the writer a packet copy, but the
          // dispatch-packet SGPR may still name the application-visible queue
          // slot. Mirror the rocjitsu-private field there as well so the
          // translated prologue sees the same state pointer regardless of
          // which packet image CP uses for dispatch_ptr.
          if (state.queue != nullptr && state.queue->base_address != nullptr &&
              state.queue->size != 0) {
            auto *queue_packets =
                static_cast<hsa_kernel_dispatch_packet_t *>(state.queue->base_address);
            queue_packets[(user_pkt_index + index) % state.queue->size].reserved2 = state_address;
          }
        } else {
          std::memset(buffers.kernarg, 0, kernarg_bytes);
          if (plan->kernarg_size != 0)
            std::memcpy(buffers.kernarg, packet.kernarg_address, plan->kernarg_size);
          trace_virtual_lds_kernarg(user_pkt_index + index, buffers.kernarg, plan->kernarg_size);
          std::memcpy(static_cast<uint8_t *>(buffers.kernarg) + pointer_offset, &dispatch_state,
                      sizeof(dispatch_state));
          packet.kernarg_address = buffers.kernarg;
        }

        const uint64_t original_kernel_object = packet.kernel_object;
        const uint32_t original_group_segment_size = packet.group_segment_size;
        const uint32_t original_private_segment_size = packet.private_segment_size;
        packet.kernel_object = plan->virtual_kernel_object;
        // The virtual descriptor has zero fixed LDS, and the backing buffer covers
        // both static and dynamic group segment bytes. Leaving dynamic LDS in the
        // packet would still ask hardware to allocate LDS and defeat the switch.
        packet.group_segment_size = 0;
        // Keep packet scratch metadata aligned with the virtual descriptor. The
        // DBT sidecar may introduce private spills, and ROCR's scratch accounting
        // uses the AQL packet field when handling dispatch-time scratch events.
        packet.private_segment_size = plan->virtual_private_segment_size;
        buffers.virtual_kernel_object = plan->virtual_kernel_object;
        state.retired_buffers.push_back(buffers);
        trace_virtual_lds_dispatch(
            "rewrote intercept_packet=%llu kernel=%s original_object=0x%llx "
            "virtual_object=0x%llx static=%u packet_group=%u private=%u->%u requested=%llu "
            "groups=%u,%u,%u strides=%u,%u,%u backing_bytes=%zu kernarg=%p backing=%p "
            "state=0x%llx "
            "packet_pointer=%d",
            static_cast<unsigned long long>(user_pkt_index + index), plan->kernel_name.c_str(),
            static_cast<unsigned long long>(original_kernel_object),
            static_cast<unsigned long long>(plan->virtual_kernel_object), plan->static_lds_bytes,
            original_group_segment_size, original_private_segment_size,
            plan->virtual_private_segment_size, static_cast<unsigned long long>(requested_lds),
            geometry->groups_x, geometry->groups_y, geometry->groups_z, geometry->stride_x,
            geometry->stride_y, geometry->stride_z, backing_bytes, packet.kernarg_address,
            buffers.backing, static_cast<unsigned long long>(state_address), packet_pointer);
      }
    }

    writer(packets.data(), pkt_count);
  }

private:
  struct PacketRewritePlan {
    std::string kernel_name;
    uint64_t virtual_kernel_object = 0;
    uint32_t static_lds_bytes = 0;
    uint32_t virtual_private_segment_size = 0;
    uint32_t kernarg_size = 0;
    uint32_t backing_pointer_kernarg_offset = 0;
    bool backing_pointer_in_dispatch_packet = false;
    bool runtime_state_block = false;
    bool has_workgroup_id_x = false;
    bool has_workgroup_id_y = false;
    bool has_workgroup_id_z = false;
  };

  struct VirtualLdsDispatchGeometry {
    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    size_t backing_bytes = 0;
  };

  struct QueueState {
    hsa_queue_t *queue = nullptr;
    hsa_agent_t host_agent{};
    uint64_t doorbell_signal = 0;
    uint64_t next_packet_id = 0;
    bool uses_packet_interceptor = false;
    std::vector<VirtualLdsDispatchBuffers> slots;
    std::vector<VirtualLdsDispatchBuffers> retired_buffers;
  };

  static void release_queue(QueueState &state) {
    for (VirtualLdsDispatchBuffers &buffers : state.slots)
      release_virtual_lds_buffers(buffers);
    for (VirtualLdsDispatchBuffers &buffers : state.retired_buffers)
      release_virtual_lds_buffers(buffers);
    state.retired_buffers.clear();
  }

  static void retire_slot_buffers(QueueState &state, uint32_t slot) {
    VirtualLdsDispatchBuffers &slot_buffers = state.slots[slot];
    if (slot_buffers.kernarg == nullptr && slot_buffers.backing == nullptr &&
        slot_buffers.state == nullptr)
      return;
    // AQL queue slots may be reused before the previously dispatched kernel has
    // retired. Keep old virtual-LDS backing allocations alive until the queue is
    // destroyed rather than freeing them on slot reuse.
    state.retired_buffers.push_back(slot_buffers);
    slot_buffers = {};
  }

  static void publish_packet_header(hsa_kernel_dispatch_packet_t &packet, uint16_t header) {
    std::atomic_ref<uint16_t>(packet.header).store(header, std::memory_order_release);
  }

  static void note_packet_ready(QueueState &state, uint64_t packet_id, bool ready) {
    if (ready && packet_id == state.next_packet_id)
      ++state.next_packet_id;
  }

  [[nodiscard]] static std::optional<uint32_t> ceil_div_u32(uint32_t value, uint16_t divisor) {
    if (divisor == 0)
      return std::nullopt;
    return (value + divisor - 1u) / divisor;
  }

  [[nodiscard]] static std::optional<uint32_t> checked_mul_u32(uint32_t lhs, uint32_t rhs) {
    const uint64_t product = static_cast<uint64_t>(lhs) * rhs;
    if (product > std::numeric_limits<uint32_t>::max())
      return std::nullopt;
    return static_cast<uint32_t>(product);
  }

  [[nodiscard]] static std::optional<VirtualLdsDispatchGeometry>
  compute_virtual_lds_geometry(const hsa_kernel_dispatch_packet_t &packet,
                               const PacketRewritePlan &plan, uint64_t requested_lds) {
    if (!plan.runtime_state_block || requested_lds == 0 ||
        requested_lds > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }

    auto groups_x = ceil_div_u32(packet.grid_size_x, packet.workgroup_size_x);
    auto groups_y = ceil_div_u32(packet.grid_size_y, packet.workgroup_size_y);
    auto groups_z = ceil_div_u32(packet.grid_size_z, packet.workgroup_size_z);
    if (!groups_x || !groups_y || !groups_z)
      return std::nullopt;

    if ((*groups_x > 1 && !plan.has_workgroup_id_x) ||
        (*groups_y > 1 && !plan.has_workgroup_id_y) ||
        (*groups_z > 1 && !plan.has_workgroup_id_z)) {
      trace_virtual_lds_dispatch(
          "virtual-LDS missing workgroup-id SGPR kernel=%s groups=%u,%u,%u has_id=%d,%d,%d",
          plan.kernel_name.c_str(), *groups_x, *groups_y, *groups_z, plan.has_workgroup_id_x,
          plan.has_workgroup_id_y, plan.has_workgroup_id_z);
      return std::nullopt;
    }

    const auto allocation_stride_x = static_cast<uint32_t>(requested_lds);
    auto allocation_stride_y = checked_mul_u32(*groups_x, allocation_stride_x);
    if (!allocation_stride_y)
      return std::nullopt;
    auto allocation_stride_z = checked_mul_u32(*groups_y, *allocation_stride_y);
    if (!allocation_stride_z)
      return std::nullopt;
    auto backing = checked_mul_u32(*groups_z, *allocation_stride_z);
    if (!backing && *groups_x != 0 && *groups_y != 0 && *groups_z != 0)
      return std::nullopt;

    // The runtime state records the increments consumed by the translated entry
    // prologue, not merely the dense backing-buffer layout. Leave singleton
    // dimensions with a zero increment so stale or unavailable workgroup-id
    // SGPRs cannot perturb the base pointer for 1-D/2-D launches. The backing
    // allocation still uses the dense allocation strides above, so multi-group
    // dimensions keep a private LDS slice per workgroup.
    const uint32_t runtime_stride_x = *groups_x > 1 ? allocation_stride_x : 0;
    const uint32_t runtime_stride_y = *groups_y > 1 ? *allocation_stride_y : 0;
    const uint32_t runtime_stride_z = *groups_z > 1 ? *allocation_stride_z : 0;

    // A zero-grid dispatch should not execute the prologue, but still switch to
    // the virtual descriptor so resource validation sees zero hardware LDS.
    const uint32_t backing_bytes = backing.value_or(0);
    return VirtualLdsDispatchGeometry{
        .groups_x = *groups_x,
        .groups_y = *groups_y,
        .groups_z = *groups_z,
        .stride_x = runtime_stride_x,
        .stride_y = runtime_stride_y,
        .stride_z = runtime_stride_z,
        .backing_bytes = static_cast<size_t>(std::max(backing_bytes, allocation_stride_x)),
    };
  }

  [[nodiscard]] static VirtualLdsDispatchState
  make_virtual_lds_dispatch_state(uint64_t backing_address,
                                  const VirtualLdsDispatchGeometry &geometry) {
    return VirtualLdsDispatchState{.backing_base = backing_address,
                                   .stride_x = geometry.stride_x,
                                   .stride_y = geometry.stride_y,
                                   .stride_z = geometry.stride_z};
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

  static std::optional<PacketRewritePlan>
  descriptor_reserved_rewrite_plan(const hsa_kernel_dispatch_packet_t &packet,
                                   const rocr::llvm::amdhsa::kernel_descriptor_t &descriptor) {
    const auto metadata = read_virtual_lds_descriptor_dispatch_metadata(descriptor);
    if (!metadata)
      return std::nullopt;
    if (packet.kernel_object > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;

    const int64_t normal_object = static_cast<int64_t>(packet.kernel_object);
    if ((metadata->virtual_descriptor_delta > 0 &&
         normal_object >
             std::numeric_limits<int64_t>::max() - metadata->virtual_descriptor_delta) ||
        (metadata->virtual_descriptor_delta < 0 &&
         normal_object <
             std::numeric_limits<int64_t>::min() - metadata->virtual_descriptor_delta)) {
      return std::nullopt;
    }
    const uint64_t virtual_kernel_object =
        static_cast<uint64_t>(normal_object + metadata->virtual_descriptor_delta);
    rocr::llvm::amdhsa::kernel_descriptor_t virtual_descriptor{};
    std::memcpy(&virtual_descriptor, reinterpret_cast<const void *>(virtual_kernel_object),
                sizeof(virtual_descriptor));
    return PacketRewritePlan{
        .kernel_name = "<descriptor-reserved>",
        .virtual_kernel_object = virtual_kernel_object,
        .static_lds_bytes = descriptor.group_segment_fixed_size,
        .virtual_private_segment_size = virtual_descriptor.private_segment_fixed_size,
        .kernarg_size = metadata->kernarg_size,
        .backing_pointer_kernarg_offset = metadata->backing_pointer_kernarg_offset,
        .backing_pointer_in_dispatch_packet =
            (metadata->flags & kVirtualLdsFlagBackingPointerInDispatchPacket) != 0,
        .runtime_state_block = (metadata->flags & kVirtualLdsFlagRuntimeStateBlock) != 0,
        .has_workgroup_id_x = (metadata->flags & kVirtualLdsFlagWorkgroupIdX) != 0,
        .has_workgroup_id_y = (metadata->flags & kVirtualLdsFlagWorkgroupIdY) != 0,
        .has_workgroup_id_z = (metadata->flags & kVirtualLdsFlagWorkgroupIdZ) != 0};
  }

  static std::optional<PacketRewritePlan>
  rewrite_plan_for_packet(const hsa_kernel_dispatch_packet_t &packet) {
    if (auto resolved =
            VirtualLdsRuntimeRegistry::instance().find_by_kernel_object(packet.kernel_object)) {
      const VirtualLdsKernelMetadata &metadata = resolved->metadata;
      rocr::llvm::amdhsa::kernel_descriptor_t virtual_descriptor{};
      std::memcpy(&virtual_descriptor,
                  reinterpret_cast<const void *>(resolved->virtual_kernel_object),
                  sizeof(virtual_descriptor));
      return PacketRewritePlan{
          .kernel_name = metadata.kernel_name,
          .virtual_kernel_object = resolved->virtual_kernel_object,
          .static_lds_bytes = metadata.static_lds_bytes,
          .virtual_private_segment_size = virtual_descriptor.private_segment_fixed_size,
          .kernarg_size = metadata.kernarg_size,
          .backing_pointer_kernarg_offset = metadata.backing_pointer_kernarg_offset,
          .backing_pointer_in_dispatch_packet =
              (metadata.flags & kVirtualLdsFlagBackingPointerInDispatchPacket) != 0,
          .runtime_state_block = (metadata.flags & kVirtualLdsFlagRuntimeStateBlock) != 0,
          .has_workgroup_id_x = (metadata.flags & kVirtualLdsFlagWorkgroupIdX) != 0,
          .has_workgroup_id_y = (metadata.flags & kVirtualLdsFlagWorkgroupIdY) != 0,
          .has_workgroup_id_z = (metadata.flags & kVirtualLdsFlagWorkgroupIdZ) != 0,
      };
    }

    rocr::llvm::amdhsa::kernel_descriptor_t descriptor{};
    std::memcpy(&descriptor, reinterpret_cast<const void *>(packet.kernel_object),
                sizeof(descriptor));
    return descriptor_reserved_rewrite_plan(packet, descriptor);
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
    if (slot_buffers.backing != nullptr && slot_buffers.kernarg == nullptr &&
        slot_buffers.virtual_kernel_object != 0 &&
        packet.kernel_object == slot_buffers.virtual_kernel_object &&
        packet.reserved2 == reinterpret_cast<uintptr_t>(slot_buffers.state)) {
      return true;
    }

    retire_slot_buffers(state, slot);
    auto plan = rewrite_plan_for_packet(packet);
    if (!plan) {
      if (packet.group_segment_size > kCdna3HardwareLdsBytes) {
        trace_virtual_lds_dispatch(
            "no virtual-LDS plan packet=%llu slot=%u header=0x%x object=0x%llx "
            "packet_group=%u",
            static_cast<unsigned long long>(packet_id), slot, header,
            static_cast<unsigned long long>(packet.kernel_object), packet.group_segment_size);
      }
      return true;
    }

    // AQL's group_segment_size is the total per-workgroup group memory request.
    // It must already cover the descriptor's fixed group segment plus any
    // dynamic group segment variables, so do not add the fixed size again when
    // sizing the virtual backing allocation.
    const uint64_t requested_lds =
        std::max<uint64_t>(plan->static_lds_bytes, packet.group_segment_size);
    if (requested_lds <= kCdna3HardwareLdsBytes) {
      if (packet.group_segment_size > kCdna3HardwareLdsBytes) {
        trace_virtual_lds_dispatch(
            "virtual-LDS plan below threshold packet=%llu slot=%u kernel=%s static=%u "
            "packet_group=%u requested=%llu",
            static_cast<unsigned long long>(packet_id), slot, plan->kernel_name.c_str(),
            plan->static_lds_bytes, packet.group_segment_size,
            static_cast<unsigned long long>(requested_lds));
      }
      return true;
    }

    const bool packet_pointer = plan->backing_pointer_in_dispatch_packet;
    const uint64_t pointer_offset = plan->backing_pointer_kernarg_offset;
    if (packet_pointer) {
      if (pointer_offset != offsetof(hsa_kernel_dispatch_packet_t, reserved2)) {
        log_message(kLogInfo,
                    "virtual-LDS metadata has invalid dispatch-packet pointer offset kernel=%s",
                    plan->kernel_name.c_str());
        return true;
      }
    } else if (pointer_offset < plan->kernarg_size ||
               pointer_offset >
                   std::numeric_limits<size_t>::max() - sizeof(VirtualLdsDispatchState)) {
      log_message(kLogInfo, "virtual-LDS metadata has invalid kernarg pointer offset kernel=%s",
                  plan->kernel_name.c_str());
      return true;
    }

    const auto geometry = compute_virtual_lds_geometry(packet, *plan, requested_lds);
    if (!geometry) {
      trace_virtual_lds_dispatch(
          "virtual-LDS geometry failed packet=%llu slot=%u kernel=%s requested=%llu "
          "grid=%u,%u,%u workgroup=%u,%u,%u",
          static_cast<unsigned long long>(packet_id), slot, plan->kernel_name.c_str(),
          static_cast<unsigned long long>(requested_lds), packet.grid_size_x, packet.grid_size_y,
          packet.grid_size_z, packet.workgroup_size_x, packet.workgroup_size_y,
          packet.workgroup_size_z);
      return true;
    }

    if (!packet_pointer && plan->kernarg_size != 0 && packet.kernarg_address == nullptr) {
      log_message(kLogInfo, "virtual-LDS dispatch has null source kernarg kernel=%s",
                  plan->kernel_name.c_str());
      return true;
    }

    const size_t kernarg_bytes =
        packet_pointer ? 0 : static_cast<size_t>(pointer_offset + sizeof(VirtualLdsDispatchState));
    const size_t state_bytes = packet_pointer ? sizeof(VirtualLdsDispatchState) : 0;
    const size_t backing_bytes = geometry->backing_bytes;
    VirtualLdsDispatchBuffers buffers;
    if (!VirtualLdsDispatchAllocator::instance().allocate(state.host_agent, backing_bytes,
                                                          kernarg_bytes, state_bytes, buffers)) {
      return true;
    }

    const uint64_t backing_address = reinterpret_cast<uintptr_t>(buffers.backing);
    const VirtualLdsDispatchState dispatch_state =
        make_virtual_lds_dispatch_state(backing_address, *geometry);
    const uint64_t state_address =
        packet_pointer
            ? reinterpret_cast<uintptr_t>(buffers.state)
            : reinterpret_cast<uintptr_t>(static_cast<uint8_t *>(buffers.kernarg) + pointer_offset);

    const uint64_t original_kernel_object = packet.kernel_object;
    const uint32_t original_group_segment_size = packet.group_segment_size;
    const uint32_t original_private_segment_size = packet.private_segment_size;
    publish_packet_header(packet, 0);
    packet.kernel_object = plan->virtual_kernel_object;
    if (packet_pointer) {
      std::memcpy(buffers.state, &dispatch_state, sizeof(dispatch_state));
      packet.reserved2 = state_address;
    } else {
      std::memset(buffers.kernarg, 0, kernarg_bytes);
      if (plan->kernarg_size != 0)
        std::memcpy(buffers.kernarg, packet.kernarg_address, plan->kernarg_size);
      trace_virtual_lds_kernarg(packet_id, buffers.kernarg, plan->kernarg_size);
      std::memcpy(static_cast<uint8_t *>(buffers.kernarg) + pointer_offset, &dispatch_state,
                  sizeof(dispatch_state));
      packet.kernarg_address = buffers.kernarg;
    }
    // The virtual descriptor has zero fixed LDS, and the backing buffer covers
    // both static and dynamic group segment bytes. Leaving dynamic LDS in the
    // packet would still ask hardware to allocate LDS and defeat the switch.
    packet.group_segment_size = 0;
    // Keep packet scratch metadata aligned with the virtual descriptor. The DBT
    // sidecar may introduce private spills, and ROCR's scratch accounting uses
    // the AQL packet field when handling dispatch-time scratch events.
    packet.private_segment_size = plan->virtual_private_segment_size;
    buffers.virtual_kernel_object = plan->virtual_kernel_object;
    slot_buffers = buffers;
    publish_packet_header(packet, header);

    log_message(kLogDebug,
                "rewrote virtual-LDS dispatch kernel=%s packet=%llu slot=%u lds=%llu "
                "virtual_object=0x%llx kernarg=%p backing=%p",
                plan->kernel_name.c_str(), static_cast<unsigned long long>(packet_id), slot,
                static_cast<unsigned long long>(requested_lds),
                static_cast<unsigned long long>(plan->virtual_kernel_object), buffers.kernarg,
                buffers.backing);
    trace_virtual_lds_dispatch(
        "rewrote packet=%llu slot=%u kernel=%s original_object=0x%llx virtual_object=0x%llx "
        "static=%u packet_group=%u private=%u->%u requested=%llu groups=%u,%u,%u "
        "strides=%u,%u,%u backing_bytes=%zu kernarg=%p backing=%p state=0x%llx "
        "packet_pointer=%d",
        static_cast<unsigned long long>(packet_id), slot, plan->kernel_name.c_str(),
        static_cast<unsigned long long>(original_kernel_object),
        static_cast<unsigned long long>(plan->virtual_kernel_object), plan->static_lds_bytes,
        original_group_segment_size, original_private_segment_size,
        plan->virtual_private_segment_size, static_cast<unsigned long long>(requested_lds),
        geometry->groups_x, geometry->groups_y, geometry->groups_z, geometry->stride_x,
        geometry->stride_y, geometry->stride_z, backing_bytes, packet.kernarg_address,
        buffers.backing, static_cast<unsigned long long>(state_address), packet_pointer);
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
      const uint64_t tail_packets = std::min(write_index, kVirtualLdsScannerTailPackets);
      rewrite_packet_range(state, write_index - tail_packets, write_index, false);
    }
  }

  void ensure_scanner_started_locked() {
    if (scanner_.joinable())
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

  std::mutex mutex_;
  std::unordered_map<uint64_t, QueueState> queues_by_ptr_;
  std::unordered_map<uint64_t, uint64_t> queues_by_doorbell_;
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
void register_virtual_lds_packet_interceptor(hsa_queue_t *queue) {
  auto *register_interceptor = layer().amd_queue_intercept_register();
  if (register_interceptor == nullptr || queue == nullptr)
    return;
  const hsa_status_t status = register_interceptor(queue, virtual_lds_packet_interceptor, queue);
  if (status != HSA_STATUS_SUCCESS) {
    log_message(kLogInfo, "failed to register virtual-LDS queue interceptor queue=%p status=%d",
                static_cast<void *>(queue), static_cast<int>(status));
    trace_virtual_lds_dispatch("queue interceptor registration failed queue=%p status=%d",
                               static_cast<void *>(queue), static_cast<int>(status));
  }
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
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  log_message(kLogDebug, "agent_iterate_isas agent=%llu mapped=%llu",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle));
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
      VirtualLdsDispatchQueueRegistry::instance().record_queue(*queue, mapped, true);
      register_virtual_lds_packet_interceptor(*queue);
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
    VirtualLdsDispatchQueueRegistry::instance().record_queue(*queue, mapped, true);
    register_virtual_lds_packet_interceptor(*queue);
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
    VirtualLdsRuntimeRegistry::instance().note_kernel_object(executable_symbol, kernel_object);
    if (auto virtual_kernel_object =
            VirtualLdsRuntimeRegistry::instance().static_oversized_virtual_kernel_object(
                executable_symbol)) {
      std::memcpy(value, &*virtual_kernel_object, sizeof(*virtual_kernel_object));
      trace_virtual_lds_dispatch("symbol kernel_object uses virtual descriptor symbol=%llu "
                                 "normal=0x%llx virtual=0x%llx",
                                 static_cast<unsigned long long>(executable_symbol.handle),
                                 static_cast<unsigned long long>(kernel_object),
                                 static_cast<unsigned long long>(*virtual_kernel_object));
    }
  } else if (status == HSA_STATUS_SUCCESS &&
             attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE &&
             value != nullptr &&
             VirtualLdsRuntimeRegistry::instance().uses_static_oversized_virtual_descriptor(
                 executable_symbol)) {
    const uint32_t virtual_group_segment_size = 0;
    std::memcpy(value, &virtual_group_segment_size, sizeof(virtual_group_segment_size));
    trace_virtual_lds_dispatch("symbol group segment size uses virtual descriptor symbol=%llu",
                               static_cast<unsigned long long>(executable_symbol.handle));
  } else if (status == HSA_STATUS_SUCCESS &&
             attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE &&
             value != nullptr) {
    if (auto virtual_private_segment_size =
            VirtualLdsRuntimeRegistry::instance().static_oversized_virtual_private_segment_size(
                executable_symbol)) {
      std::memcpy(value, &*virtual_private_segment_size, sizeof(*virtual_private_segment_size));
      trace_virtual_lds_dispatch(
          "symbol private segment size uses virtual descriptor symbol=%llu private=%u",
          static_cast<unsigned long long>(executable_symbol.handle), *virtual_private_segment_size);
    }
  }
  return status;
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
  if (mapped.handle == agent.handle)
    return original(executable, mapped, callback, data);
  IterateAgentSymbolsData wrapped{callback, data, agent};
  return original(executable, mapped, rj_iterate_agent_symbols_callback, &wrapped);
}

hsa_status_t HSA_API rj_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
  auto *original = layer().amd_agent_iterate_memory_pools();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
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
  hsa_amd_memory_pool_t mapped_pool = MemoryPoolMapper::instance().map(memory_pool);
  log_message(kLogVerbose, "amd_memory_pool_allocate pool=%llu mapped=%llu size=%zu flags=0x%x",
              static_cast<unsigned long long>(memory_pool.handle),
              static_cast<unsigned long long>(mapped_pool.handle), size, flags);
  hsa_status_t status = original(mapped_pool, size, flags, ptr);
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
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  hsa_status_t status = original(mapped, signal, time);
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
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  hsa_status_t status = original(mapped, agent_tick, system_tick);
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
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  hsa_amd_memory_pool_t mapped_pool = MemoryPoolMapper::instance().map(memory_pool);
  log_message(
      kLogDebug,
      "amd_agent_memory_pool_get_info agent=%llu mapped=%llu pool=%llu mapped=%llu "
      "attr=%u",
      static_cast<unsigned long long>(agent.handle), static_cast<unsigned long long>(mapped.handle),
      static_cast<unsigned long long>(memory_pool.handle),
      static_cast<unsigned long long>(mapped_pool.handle), static_cast<unsigned>(attribute));
  return original(mapped, mapped_pool, attribute, value);
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
  hsa_agent_t mapped_dst = AgentMapper::instance().map(dst_agent);
  hsa_agent_t mapped_src = AgentMapper::instance().map(src_agent);
  log_message(kLogVerbose,
              "amd_memory_async_copy dst_agent=%llu mapped=%llu src_agent=%llu mapped=%llu "
              "size=%zu",
              static_cast<unsigned long long>(dst_agent.handle),
              static_cast<unsigned long long>(mapped_dst.handle),
              static_cast<unsigned long long>(src_agent.handle),
              static_cast<unsigned long long>(mapped_src.handle), size);
  return original(dst, mapped_dst, src, mapped_src, size, num_dep_signals, dep_signals,
                  completion_signal);
}

hsa_status_t HSA_API rj_amd_memory_async_copy_on_engine(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t *dep_signals, hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
  auto *original = layer().amd_memory_async_copy_on_engine();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped_dst = AgentMapper::instance().map(dst_agent);
  hsa_agent_t mapped_src = AgentMapper::instance().map(src_agent);
  log_message(kLogVerbose,
              "amd_memory_async_copy_on_engine dst_agent=%llu mapped=%llu src_agent=%llu "
              "mapped=%llu size=%zu engine=%u",
              static_cast<unsigned long long>(dst_agent.handle),
              static_cast<unsigned long long>(mapped_dst.handle),
              static_cast<unsigned long long>(src_agent.handle),
              static_cast<unsigned long long>(mapped_src.handle), size,
              static_cast<unsigned>(engine_id));
  return original(dst, mapped_dst, src, mapped_src, size, num_dep_signals, dep_signals,
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
  hsa_agent_t mapped = AgentMapper::instance().map(copy_agent);
  log_message(kLogVerbose, "amd_memory_async_copy_rect agent=%llu mapped=%llu dir=%u",
              static_cast<unsigned long long>(copy_agent.handle),
              static_cast<unsigned long long>(mapped.handle), static_cast<unsigned>(dir));
  return original(dst, dst_offset, src, src_offset, range, mapped, dir, num_dep_signals,
                  dep_signals, completion_signal);
}

hsa_status_t HSA_API rj_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                                      uint32_t *engine_ids_mask) {
  auto *original = layer().amd_memory_copy_engine_status();
  if (!original)
    return HSA_STATUS_ERROR;
  return original(AgentMapper::instance().map(dst_agent), AgentMapper::instance().map(src_agent),
                  engine_ids_mask);
}

hsa_status_t HSA_API rj_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent,
                                                             hsa_agent_t src_agent,
                                                             hsa_amd_sdma_engine_id_t *engine_id) {
  auto *original = layer().amd_memory_get_preferred_copy_engine();
  if (!original)
    return HSA_STATUS_ERROR;
  return original(AgentMapper::instance().map(dst_agent), AgentMapper::instance().map(src_agent),
                  engine_id);
}

hsa_status_t HSA_API rj_amd_memory_lock(void *host_ptr, size_t size, hsa_agent_t *agents,
                                        int num_agent, void **agent_ptr) {
  auto *original = layer().amd_memory_lock();
  if (!original)
    return HSA_STATUS_ERROR;
  auto mapped = num_agent > 0 ? map_agent_array(agents, static_cast<size_t>(num_agent))
                              : std::vector<hsa_agent_t>{};
  return original(host_ptr, size, mapped.empty() ? agents : mapped.data(), num_agent, agent_ptr);
}

hsa_status_t HSA_API rj_amd_memory_lock_to_pool(void *host_ptr, size_t size, hsa_agent_t *agents,
                                                int num_agent, hsa_amd_memory_pool_t pool,
                                                uint32_t flags, void **agent_ptr) {
  auto *original = layer().amd_memory_lock_to_pool();
  if (!original)
    return HSA_STATUS_ERROR;
  auto mapped = num_agent > 0 ? map_agent_array(agents, static_cast<size_t>(num_agent))
                              : std::vector<hsa_agent_t>{};
  hsa_amd_memory_pool_t mapped_pool = MemoryPoolMapper::instance().map(pool);
  return original(host_ptr, size, mapped.empty() ? agents : mapped.data(), num_agent, mapped_pool,
                  flags, agent_ptr);
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
    if (num_agents_accessible != nullptr && accessible != nullptr && *accessible != nullptr) {
      for (uint32_t i = 0; i < *num_agents_accessible; ++i)
        (*accessible)[i] = AgentMapper::instance().guest_for_host((*accessible)[i]);
    }
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
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  log_message(kLogVerbose, "amd_svm_prefetch_async ptr=%p size=%zu agent=%llu mapped=%llu", ptr,
              size, static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle));
  return original(ptr, size, mapped, num_dep_signals, dep_signals, completion_signal);
}

hsa_status_t HSA_API rj_amd_vmem_set_access(void *va, size_t size,
                                            const hsa_amd_memory_access_desc_t *desc,
                                            size_t desc_cnt) {
  auto *original = layer().amd_vmem_set_access();
  if (!original)
    return HSA_STATUS_ERROR;
  std::vector<hsa_amd_memory_access_desc_t> mapped;
  if (desc && desc_cnt > 0) {
    mapped.assign(desc, desc + desc_cnt);
    for (auto &entry : mapped)
      entry.agent_handle = AgentMapper::instance().map(entry.agent_handle);
  }
  log_message(kLogVerbose, "amd_vmem_set_access va=%p size=%zu desc_cnt=%zu", va, size, desc_cnt);
  return original(va, size, mapped.empty() ? desc : mapped.data(), desc_cnt);
}

hsa_status_t HSA_API rj_amd_vmem_get_access(void *va, hsa_access_permission_t *perms,
                                            hsa_agent_t agent_handle) {
  auto *original = layer().amd_vmem_get_access();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent_handle);
  log_message(kLogVerbose, "amd_vmem_get_access va=%p agent=%llu mapped=%llu", va,
              static_cast<unsigned long long>(agent_handle.handle),
              static_cast<unsigned long long>(mapped.handle));
  return original(va, perms, mapped);
}

hsa_status_t HSA_API rj_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold) {
  auto *original = layer().amd_agent_set_async_scratch_limit();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  log_message(kLogVerbose, "amd_agent_set_async_scratch_limit agent=%llu mapped=%llu threshold=%zu",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle), threshold);
  return original(mapped, threshold);
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
    op.src_agent = AgentMapper::instance().map(op.src_agent);
    if (op.num_entries == 0) {
      op.dst_agent = AgentMapper::instance().map(op.dst_agent);
      continue;
    }
    if (op.dst_agent_list != nullptr) {
      mapped_dst_lists.push_back(map_agent_array(op.dst_agent_list, op.num_entries));
      op.dst_agent_list = mapped_dst_lists.back().data();
    } else {
      op.dst_agent = AgentMapper::instance().map(op.dst_agent);
    }
  }
  return original(mapped.data(), num_copy_ops, num_dep_signals, dep_signals);
}

hsa_status_t HSA_API rj_amd_agent_preload(hsa_agent_t agent, uint64_t flags) {
  auto *original = layer().amd_agent_preload();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  log_message(kLogVerbose, "amd_agent_preload agent=%llu mapped=%llu flags=0x%llx",
              static_cast<unsigned long long>(agent.handle),
              static_cast<unsigned long long>(mapped.handle),
              static_cast<unsigned long long>(flags));
  return original(mapped, flags);
}

/// @brief Create an HSA reader from translated ELF bytes and keep the storage alive.
[[nodiscard]] hsa_status_t create_translated_reader(std::vector<uint8_t> translated,
                                                    hsa_code_object_reader_t *translated_reader) {
  auto *owned = new (std::nothrow) std::vector<uint8_t>(std::move(translated));
  if (owned == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto *original_create = layer().create_from_memory();
  if (original_create == nullptr) {
    delete owned;
    return HSA_STATUS_ERROR;
  }

  const hsa_status_t status = original_create(owned->data(), owned->size(), translated_reader);
  if (status != HSA_STATUS_SUCCESS) {
    delete owned;
    return status;
  }

  if (!CodeObjectReaderRegistry::instance().store(*translated_reader, owned->data(), owned->size(),
                                                  owned)) {
    if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
      (void)original_destroy(*translated_reader);
    *translated_reader = {};
    delete owned;
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
  if (config->guest_target && !guest_load) {
    hsa_status_t status =
        original_load(executable, agent, code_object_reader, options, loaded_code_object);
    log_message(kLogVerbose, "load_agent_code_object host/pass-through status=%d",
                static_cast<int>(status));
    return status;
  }

  hsa_agent_t load_agent = guest_load ? AgentMapper::instance().host_for_guest() : agent;
  if (guest_load && load_agent.handle == 0)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  const uint8_t *bytes = nullptr;
  size_t size = 0;
  if (!CodeObjectReaderRegistry::instance().lookup(code_object_reader, &bytes, &size)) {
    log_message(kLogInfo, "no memory bytes registered for reader=%llu",
                static_cast<unsigned long long>(code_object_reader.handle));
    if (guest_load)
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
    return original_load(executable, load_agent, code_object_reader, options, loaded_code_object);
  }

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

  if (source_target.arch == config->target.arch && source_target.mach == config->target.mach) {
    log_message(kLogInfo,
                "source target %s arch %s already matches requested target; passing through",
                elf_mach_name(source_target.mach), arch_name(source_target.arch));
    hsa_status_t status =
        original_load(executable, load_agent, code_object_reader, options, loaded_code_object);
    log_message(kLogVerbose, "load_agent_code_object already-target status=%d",
                static_cast<int>(status));
    if (status == HSA_STATUS_SUCCESS && guest_load)
      ExecutableAgentRegistry::instance().record(executable, agent, load_agent);
    return status;
  }

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
  BinaryTranslatorOptions translator_options;
  // Runtime DBT loads large framework code objects where some kernel symbols may
  // never be dispatched in the current process. Keep the code object loadable if
  // an independent kernel fails translation; the skipped-kernel diagnostic names
  // the symbol that is redirected to a target no-op stub.
  translator_options.skip_failed_kernels = true;
  BinaryTranslator translator(source_target.arch, config->target.arch, config->target.mach,
                              translator_options);
  translated = translator.translate(source_object);

  dump_code_object_if_requested("source", code_object_reader, std::span<const uint8_t>(bytes, size),
                                translated.skipped_kernel_symbols);
  dump_code_object_if_requested("translated", code_object_reader, translated.elf_bytes,
                                translated.skipped_kernel_symbols);

  print_diagnostics(stderr, translated.diagnostics, config->log_level > kLogDisabled);
  if (translated.elf_bytes.empty() || has_error_diagnostic(translated.diagnostics)) {
    std::fprintf(stderr, "[rocjitsu-hooks] translation failed; refusing original code object\n");
    return HSA_STATUS_ERROR;
  }

  auto virtual_lds_metadata = parse_virtual_lds_metadata_section(
      std::span<const uint8_t>(translated.elf_bytes.data(), translated.elf_bytes.size()));
  if (!virtual_lds_metadata) {
    std::fprintf(stderr,
                 "[rocjitsu-hooks] translated code object has malformed virtual-LDS metadata\n");
    return HSA_STATUS_ERROR;
  }

  hsa_code_object_reader_t translated_reader{};
  hsa_status_t status =
      create_translated_reader(std::move(translated.elf_bytes), &translated_reader);
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
  } else if (guest_load) {
    ExecutableAgentRegistry::instance().record(executable, agent, load_agent);
  }
  if (status == HSA_STATUS_SUCCESS && !virtual_lds_metadata->empty()) {
    const hsa_loaded_code_object_t loaded =
        loaded_code_object != nullptr ? *loaded_code_object : hsa_loaded_code_object_t{};
    VirtualLdsRuntimeRegistry::instance().record_load(executable, agent, load_agent, loaded,
                                                      std::move(*virtual_lds_metadata));
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
