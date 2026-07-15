// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_waitcheck_hooks.cpp
/// @brief ROCR HSA tools hook for checking the final AMDGPU code object loaded.

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/code/amdgpu_code_object.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

using CreateFromFileWithOffsetSizeFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                        hsa_code_object_reader_t *);

struct AmdLoaderTablePrefix {
  void *query_host_address = nullptr;
  void *query_segment_descriptors = nullptr;
  void *query_executable = nullptr;
  void *executable_iterate_loaded_code_objects = nullptr;
  void *loaded_code_object_get_info = nullptr;
  CreateFromFileWithOffsetSizeFn create_from_file_with_offset_size = nullptr;
};

struct AnalysisResult {
  bool checked = false;
  bool passed = true;
  bool analysis_failed = false;
};

struct ReaderEntry {
  void store_memory(uint64_t reader_handle, const uint8_t *data, size_t data_size) {
    handle = reader_handle;
    active = true;
    analyzed = false;
    result = {};
    owned.clear();
    bytes = data;
    size = data_size;
  }

  void store_owned(uint64_t reader_handle, std::vector<uint8_t> data) {
    handle = reader_handle;
    active = true;
    analyzed = false;
    result = {};
    owned = std::move(data);
    bytes = owned.data();
    size = owned.size();
  }

  void reset() {
    active = false;
    analyzed = false;
    result = {};
    bytes = nullptr;
    size = 0;
    owned.clear();
  }

  std::mutex mutex;
  uint64_t handle = 0;
  bool active = false;
  bool analyzed = false;
  const uint8_t *bytes = nullptr;
  size_t size = 0;
  std::vector<uint8_t> owned;
  AnalysisResult result;
};

struct HookStats {
  std::atomic<uint64_t> loads{0};
  std::atomic<uint64_t> checked{0};
  std::atomic<uint64_t> passed{0};
  std::atomic<uint64_t> hazards{0};
  std::atomic<uint64_t> analysis_errors{0};
  std::atomic<uint64_t> unavailable{0};

  void reset() {
    loads.store(0, std::memory_order_relaxed);
    checked.store(0, std::memory_order_relaxed);
    passed.store(0, std::memory_order_relaxed);
    hazards.store(0, std::memory_order_relaxed);
    analysis_errors.store(0, std::memory_order_relaxed);
    unavailable.store(0, std::memory_order_relaxed);
  }
};

HookStats g_stats;
std::atomic<bool> g_summary_registered{false};
std::atomic<bool> g_summary_printed{false};
thread_local bool g_in_waitcheck = false;

[[nodiscard]] bool env_enabled(const char *name, bool default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr)
    return default_value;
  const std::string_view text(value);
  return !(text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF");
}

void print_summary_once() {
  if (!env_enabled("ROCJITSU_WAITCHECK_SUMMARY", false) ||
      g_summary_printed.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  std::fprintf(
      stderr,
      "rocjitsu-waitcheck: summary loads=%llu checked=%llu passed=%llu hazards=%llu "
      "analysis_errors=%llu unavailable=%llu\n",
      static_cast<unsigned long long>(g_stats.loads.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.checked.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.passed.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.hazards.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.analysis_errors.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.unavailable.load(std::memory_order_relaxed)));
}

[[nodiscard]] const char *target_name(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX942:
    return "gfx942";
  case ROCJITSU_CODE_TARGET_GFX950:
    return "gfx950";
  case ROCJITSU_CODE_TARGET_GFX1100:
    return "gfx1100";
  case ROCJITSU_CODE_TARGET_GFX1200:
    return "gfx1200";
  case ROCJITSU_CODE_TARGET_GFX1201:
    return "gfx1201";
  case ROCJITSU_CODE_TARGET_GFX1250:
    return "gfx1250";
  default:
    return "unsupported";
  }
}

void print_report(const rocjitsu::AmdGpuCodeObject &code_object,
                  const rocjitsu::WaitcheckReport &report) {
  if (report.passed())
    return;

  if (report.diagnostics_truncated) {
    std::fprintf(stderr, "rocjitsu-waitcheck: at least %zu waitcnt hazard(s) in %s code object\n",
                 report.diagnostics_observed, target_name(code_object.target_id()));
  } else {
    std::fprintf(stderr, "rocjitsu-waitcheck: %zu waitcnt hazard(s) in %s code object\n",
                 report.diagnostics_observed, target_name(code_object.target_id()));
  }

  constexpr size_t kMaxDiagnostics = 32;
  const size_t limit = std::min(kMaxDiagnostics, report.diagnostics.size());
  for (size_t i = 0; i < limit; ++i) {
    const auto &diagnostic = report.diagnostics[i];
    std::fprintf(stderr, "rocjitsu-waitcheck: %s+0x%llx: %s; producer %s+0x%llx: %s\n",
                 diagnostic.section_name.c_str(),
                 static_cast<unsigned long long>(diagnostic.section_offset),
                 diagnostic.message.c_str(), diagnostic.section_name.c_str(),
                 static_cast<unsigned long long>(diagnostic.producer_section_offset),
                 diagnostic.producer_instruction.c_str());
    std::fprintf(stderr, "rocjitsu-waitcheck:   consumer: %s\n", diagnostic.instruction.c_str());
  }
  if (report.diagnostics.size() > limit) {
    std::fprintf(stderr, "rocjitsu-waitcheck: omitted %zu additional diagnostic(s)\n",
                 report.diagnostics.size() - limit);
  } else if (report.diagnostics_truncated) {
    std::fprintf(stderr, "rocjitsu-waitcheck: omitted additional diagnostic(s) after limit\n");
  }
}

void print_analysis_failure(const rocjitsu::AmdGpuCodeObject &code_object,
                            const rocjitsu::WaitcheckReport &report) {
  std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed for %s code object",
               target_name(code_object.target_id()));
  if (!report.analysis_error.empty())
    std::fprintf(stderr, ": %s", report.analysis_error.c_str());
  std::fprintf(stderr, "\n");
}

[[nodiscard]] AnalysisResult check_code_object(const void *code_object, size_t size) {
  if (!env_enabled("ROCJITSU_WAITCHECK", true) || code_object == nullptr || size == 0 ||
      g_in_waitcheck) {
    return {};
  }

  g_in_waitcheck = true;
  AnalysisResult result;
  try {
    rocjitsu::AmdGpuCodeObject parsed(static_cast<const uint8_t *>(code_object), size);
    const rj_code_arch_t arch = rocjitsu::waitcheck_arch_for_target(parsed.target_id());
    if (parsed.is_valid() && arch != ROCJITSU_CODE_ARCH_INVALID) {
      result.checked = true;
      rocjitsu::WaitcheckOptions options;
      options.max_diagnostics = 32;
      auto report = rocjitsu::analyze_waitcnts(parsed, arch, options);
      if (!report.supported) {
        print_analysis_failure(parsed, report);
        result.passed = false;
        result.analysis_failed = true;
        g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
      } else {
        print_report(parsed, report);
        result.passed = report.passed();
      }
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed: %s\n", error.what());
    result.analysis_failed = true;
    g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
  } catch (...) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed\n");
    result.analysis_failed = true;
    g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
  }
  g_in_waitcheck = false;

  if (result.checked) {
    g_stats.checked.fetch_add(1, std::memory_order_relaxed);
    if (result.passed)
      g_stats.passed.fetch_add(1, std::memory_order_relaxed);
    else if (!result.analysis_failed)
      g_stats.hazards.fetch_add(1, std::memory_order_relaxed);
  }
  return result;
}

class ReaderRegistry {
public:
  static ReaderRegistry &instance() {
    static ReaderRegistry registry;
    return registry;
  }

  void store_memory(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size) {
    std::lock_guard registry_lock(mutex_);
    ReaderEntry *entry = find_or_free_unlocked(reader.handle);
    if (entry == nullptr) {
      std::fprintf(stderr, "rocjitsu-waitcheck: reader registry is full\n");
      return;
    }
    std::lock_guard entry_lock(entry->mutex);
    entry->store_memory(reader.handle, bytes, size);
  }

  void store_owned(hsa_code_object_reader_t reader, std::vector<uint8_t> bytes) {
    std::lock_guard registry_lock(mutex_);
    ReaderEntry *entry = find_or_free_unlocked(reader.handle);
    if (entry == nullptr) {
      std::fprintf(stderr, "rocjitsu-waitcheck: reader registry is full\n");
      return;
    }
    std::lock_guard entry_lock(entry->mutex);
    entry->store_owned(reader.handle, std::move(bytes));
  }

  [[nodiscard]] ReaderEntry *lookup(hsa_code_object_reader_t reader) {
    std::lock_guard lock(mutex_);
    return find_unlocked(reader.handle);
  }

  void remove(hsa_code_object_reader_t reader) {
    std::lock_guard registry_lock(mutex_);
    if (ReaderEntry *entry = find_unlocked(reader.handle); entry != nullptr) {
      std::lock_guard entry_lock(entry->mutex);
      entry->reset();
    }
  }

  void clear() {
    std::lock_guard registry_lock(mutex_);
    for (ReaderEntry &entry : entries_) {
      std::lock_guard entry_lock(entry.mutex);
      entry.reset();
    }
  }

private:
  [[nodiscard]] ReaderEntry *find_unlocked(uint64_t handle) {
    for (ReaderEntry &entry : entries_) {
      if (entry.active && entry.handle == handle)
        return &entry;
    }
    return nullptr;
  }

  [[nodiscard]] ReaderEntry *find_or_free_unlocked(uint64_t handle) {
    if (ReaderEntry *entry = find_unlocked(handle); entry != nullptr)
      return entry;
    for (ReaderEntry &entry : entries_) {
      if (!entry.active)
        return &entry;
    }
    return nullptr;
  }

  static constexpr size_t kMaxReaders = 1024;
  std::mutex mutex_;
  std::array<ReaderEntry, kMaxReaders> entries_;
};

[[nodiscard]] std::optional<std::vector<uint8_t>>
read_regular_file_range(hsa_file_t file, size_t range_offset, size_t range_size) {
  struct stat status {};
  if (fstat(file, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0)
    return std::nullopt;
  if (static_cast<uintmax_t>(status.st_size) > std::numeric_limits<size_t>::max())
    return std::nullopt;
  const size_t file_size = static_cast<size_t>(status.st_size);
  if (range_size == 0 || range_offset > file_size || range_size > file_size - range_offset)
    return std::nullopt;
  if (range_offset > static_cast<size_t>(std::numeric_limits<off_t>::max()))
    return std::nullopt;

  std::vector<uint8_t> data(range_size);
  size_t offset = 0;
  while (offset < data.size()) {
    const size_t absolute_offset = range_offset + offset;
    if (absolute_offset > static_cast<size_t>(std::numeric_limits<off_t>::max()))
      return std::nullopt;
    const size_t chunk =
        std::min(data.size() - offset, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t read_size =
        pread(file, data.data() + offset, chunk, static_cast<off_t>(absolute_offset));
    if (read_size < 0) {
      if (errno == EINTR)
        continue;
      return std::nullopt;
    }
    if (read_size == 0)
      return std::nullopt;
    offset += static_cast<size_t>(read_size);
  }
  return data;
}

[[nodiscard]] std::optional<std::vector<uint8_t>> read_regular_file(hsa_file_t file) {
  struct stat status {};
  if (fstat(file, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0)
    return std::nullopt;
  if (static_cast<uintmax_t>(status.st_size) > std::numeric_limits<size_t>::max())
    return std::nullopt;
  return read_regular_file_range(file, 0, static_cast<size_t>(status.st_size));
}

template <typename ReadFn>
[[nodiscard]] std::optional<std::vector<uint8_t>> try_read_code_object(ReadFn &&read) noexcept {
  try {
    return std::forward<ReadFn>(read)();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "rocjitsu-waitcheck: failed to snapshot code object: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr, "rocjitsu-waitcheck: failed to snapshot code object\n");
  }
  return std::nullopt;
}

hsa_status_t HSA_API waitcheck_system_get_extension_table(uint16_t extension,
                                                          uint16_t version_major,
                                                          uint16_t version_minor, void *table);
hsa_status_t HSA_API waitcheck_system_get_major_extension_table(uint16_t extension,
                                                                uint16_t version_major,
                                                                size_t table_length, void *table);
hsa_status_t HSA_API waitcheck_reader_create_from_file(hsa_file_t file,
                                                       hsa_code_object_reader_t *reader);
hsa_status_t HSA_API waitcheck_reader_create_from_memory(const void *code_object, size_t size,
                                                         hsa_code_object_reader_t *reader);
hsa_status_t HSA_API waitcheck_reader_destroy(hsa_code_object_reader_t reader);
hsa_status_t HSA_API waitcheck_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);
hsa_status_t HSA_API waitcheck_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *reader);

class WaitcheckHsaLayer {
public:
  bool install(HsaApiTable *table) {
    std::lock_guard lock(mutex_);
    if (active_) {
      std::fprintf(stderr, "rocjitsu-waitcheck: OnLoad called while hook is active\n");
      return false;
    }
    if (!validate_table(table))
      return false;

    core_ = table->core_;
    original_get_extension_table_ = core_->hsa_system_get_extension_table_fn;
    original_get_major_extension_table_ = core_->hsa_system_get_major_extension_table_fn;
    original_create_from_file_ = core_->hsa_code_object_reader_create_from_file_fn;
    original_create_from_memory_ = core_->hsa_code_object_reader_create_from_memory_fn;
    original_destroy_ = core_->hsa_code_object_reader_destroy_fn;
    original_load_ = core_->hsa_executable_load_agent_code_object_fn;

    if (original_get_extension_table_ == nullptr ||
        original_get_major_extension_table_ == nullptr || original_create_from_file_ == nullptr ||
        original_create_from_memory_ == nullptr || original_destroy_ == nullptr ||
        original_load_ == nullptr) {
      std::fprintf(stderr, "rocjitsu-waitcheck: HSA core table contains null required entries\n");
      clear_unlocked();
      return false;
    }

    core_->hsa_system_get_extension_table_fn = waitcheck_system_get_extension_table;
    core_->hsa_system_get_major_extension_table_fn = waitcheck_system_get_major_extension_table;
    core_->hsa_code_object_reader_create_from_file_fn = waitcheck_reader_create_from_file;
    core_->hsa_code_object_reader_create_from_memory_fn = waitcheck_reader_create_from_memory;
    core_->hsa_code_object_reader_destroy_fn = waitcheck_reader_destroy;
    core_->hsa_executable_load_agent_code_object_fn = waitcheck_executable_load_agent_code_object;
    active_ = true;
    g_stats.reset();
    g_summary_printed.store(false, std::memory_order_relaxed);
    if (!g_summary_registered.exchange(true, std::memory_order_relaxed))
      (void)std::atexit(print_summary_once);
    if (env_enabled("ROCJITSU_WAITCHECK_SUMMARY", false))
      std::fprintf(stderr, "rocjitsu-waitcheck: HSA tools hook installed\n");
    return true;
  }

  void uninstall() {
    {
      std::lock_guard lock(mutex_);
      if (active_ && core_ != nullptr) {
        if (core_->hsa_system_get_extension_table_fn == waitcheck_system_get_extension_table)
          core_->hsa_system_get_extension_table_fn = original_get_extension_table_;
        if (core_->hsa_system_get_major_extension_table_fn ==
            waitcheck_system_get_major_extension_table) {
          core_->hsa_system_get_major_extension_table_fn = original_get_major_extension_table_;
        }
        if (core_->hsa_code_object_reader_create_from_file_fn == waitcheck_reader_create_from_file)
          core_->hsa_code_object_reader_create_from_file_fn = original_create_from_file_;
        if (core_->hsa_code_object_reader_create_from_memory_fn ==
            waitcheck_reader_create_from_memory) {
          core_->hsa_code_object_reader_create_from_memory_fn = original_create_from_memory_;
        }
        if (core_->hsa_code_object_reader_destroy_fn == waitcheck_reader_destroy)
          core_->hsa_code_object_reader_destroy_fn = original_destroy_;
        if (core_->hsa_executable_load_agent_code_object_fn ==
            waitcheck_executable_load_agent_code_object) {
          core_->hsa_executable_load_agent_code_object_fn = original_load_;
        }
      }
      active_ = false;
    }

    ReaderRegistry::instance().clear();
    print_summary_once();

    std::lock_guard lock(mutex_);
    clear_unlocked();
  }

  [[nodiscard]] decltype(hsa_system_get_extension_table) *get_extension_table() const {
    std::lock_guard lock(mutex_);
    return original_get_extension_table_;
  }
  [[nodiscard]] decltype(hsa_system_get_major_extension_table) *get_major_extension_table() const {
    std::lock_guard lock(mutex_);
    return original_get_major_extension_table_;
  }
  [[nodiscard]] decltype(hsa_code_object_reader_create_from_file) *create_from_file() const {
    std::lock_guard lock(mutex_);
    return original_create_from_file_;
  }
  [[nodiscard]] decltype(hsa_code_object_reader_create_from_memory) *create_from_memory() const {
    std::lock_guard lock(mutex_);
    return original_create_from_memory_;
  }
  [[nodiscard]] decltype(hsa_code_object_reader_destroy) *destroy() const {
    std::lock_guard lock(mutex_);
    return original_destroy_;
  }
  [[nodiscard]] decltype(hsa_executable_load_agent_code_object) *load() const {
    std::lock_guard lock(mutex_);
    return original_load_;
  }
  [[nodiscard]] CreateFromFileWithOffsetSizeFn file_range_create() const {
    std::lock_guard lock(mutex_);
    return original_file_range_create_;
  }
  void save_file_range_create(CreateFromFileWithOffsetSizeFn original) {
    std::lock_guard lock(mutex_);
    if (original != nullptr && original != waitcheck_reader_create_from_file_with_offset_size)
      original_file_range_create_ = original;
  }

private:
  [[nodiscard]] static bool validate_table(HsaApiTable *table) {
    if (table == nullptr || table->core_ == nullptr) {
      std::fprintf(stderr, "rocjitsu-waitcheck: invalid HSA API table passed to OnLoad\n");
      return false;
    }
    constexpr size_t required_size =
        offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
        sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn);
    if (table->core_->version.minor_id < required_size) {
      std::fprintf(stderr,
                   "rocjitsu-waitcheck: HSA core table too small: got %u bytes, need %zu bytes\n",
                   table->core_->version.minor_id, required_size);
      return false;
    }
    return true;
  }

  void clear_unlocked() {
    active_ = false;
    core_ = nullptr;
    original_get_extension_table_ = nullptr;
    original_get_major_extension_table_ = nullptr;
    original_create_from_file_ = nullptr;
    original_create_from_memory_ = nullptr;
    original_destroy_ = nullptr;
    original_load_ = nullptr;
    original_file_range_create_ = nullptr;
  }

  mutable std::mutex mutex_;
  bool active_ = false;
  CoreApiTable *core_ = nullptr;
  decltype(hsa_system_get_extension_table) *original_get_extension_table_ = nullptr;
  decltype(hsa_system_get_major_extension_table) *original_get_major_extension_table_ = nullptr;
  decltype(hsa_code_object_reader_create_from_file) *original_create_from_file_ = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *original_create_from_memory_ = nullptr;
  decltype(hsa_code_object_reader_destroy) *original_destroy_ = nullptr;
  decltype(hsa_executable_load_agent_code_object) *original_load_ = nullptr;
  CreateFromFileWithOffsetSizeFn original_file_range_create_ = nullptr;
};

WaitcheckHsaLayer &layer() {
  static WaitcheckHsaLayer state;
  return state;
}

[[nodiscard]] bool amd_loader_table_has_file_range_reader(size_t table_length) {
  constexpr size_t reader_offset =
      offsetof(AmdLoaderTablePrefix, create_from_file_with_offset_size);
  return table_length >= reader_offset + sizeof(CreateFromFileWithOffsetSizeFn);
}

void patch_amd_loader_extension_table(uint16_t extension, uint16_t version_major,
                                      size_t table_length, void *table) {
  if (extension != HSA_EXTENSION_AMD_LOADER || version_major != 1 || table == nullptr ||
      !amd_loader_table_has_file_range_reader(table_length)) {
    return;
  }
  auto *loader = static_cast<AmdLoaderTablePrefix *>(table);
  layer().save_file_range_create(loader->create_from_file_with_offset_size);
  if (loader->create_from_file_with_offset_size != nullptr)
    loader->create_from_file_with_offset_size = waitcheck_reader_create_from_file_with_offset_size;
}

[[nodiscard]] size_t deprecated_amd_loader_table_length(uint16_t version_minor) {
  return version_minor >= 2 ? sizeof(AmdLoaderTablePrefix) : 0;
}

hsa_status_t HSA_API waitcheck_system_get_extension_table(uint16_t extension,
                                                          uint16_t version_major,
                                                          uint16_t version_minor, void *table) {
  auto *original = layer().get_extension_table();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status = original(extension, version_major, version_minor, table);
  if (status == HSA_STATUS_SUCCESS) {
    patch_amd_loader_extension_table(extension, version_major,
                                     deprecated_amd_loader_table_length(version_minor), table);
  }
  return status;
}

hsa_status_t HSA_API waitcheck_system_get_major_extension_table(uint16_t extension,
                                                                uint16_t version_major,
                                                                size_t table_length, void *table) {
  auto *original = layer().get_major_extension_table();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status = original(extension, version_major, table_length, table);
  if (status == HSA_STATUS_SUCCESS)
    patch_amd_loader_extension_table(extension, version_major, table_length, table);
  return status;
}

hsa_status_t HSA_API waitcheck_reader_create_from_file(hsa_file_t file,
                                                       hsa_code_object_reader_t *reader) {
  auto *original = layer().create_from_file();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  std::optional<std::vector<uint8_t>> bytes;
  if (env_enabled("ROCJITSU_WAITCHECK", true))
    bytes = try_read_code_object([&] { return read_regular_file(file); });
  const hsa_status_t status = original(file, reader);
  if (status == HSA_STATUS_SUCCESS && reader != nullptr && bytes)
    ReaderRegistry::instance().store_owned(*reader, std::move(*bytes));
  return status;
}

hsa_status_t HSA_API waitcheck_reader_create_from_memory(const void *code_object, size_t size,
                                                         hsa_code_object_reader_t *reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status = original(code_object, size, reader);
  if (status == HSA_STATUS_SUCCESS && reader != nullptr && code_object != nullptr && size != 0 &&
      env_enabled("ROCJITSU_WAITCHECK", true)) {
    ReaderRegistry::instance().store_memory(*reader, static_cast<const uint8_t *>(code_object),
                                            size);
  }
  return status;
}

hsa_status_t HSA_API waitcheck_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *reader) {
  auto *original = layer().file_range_create();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  std::optional<std::vector<uint8_t>> bytes;
  if (env_enabled("ROCJITSU_WAITCHECK", true))
    bytes = try_read_code_object([&] { return read_regular_file_range(file, offset, size); });
  const hsa_status_t status = original(file, offset, size, reader);
  if (status == HSA_STATUS_SUCCESS && reader != nullptr && bytes)
    ReaderRegistry::instance().store_owned(*reader, std::move(*bytes));
  return status;
}

hsa_status_t HSA_API waitcheck_reader_destroy(hsa_code_object_reader_t reader) {
  ReaderRegistry::instance().remove(reader);
  auto *original = layer().destroy();
  return original == nullptr ? HSA_STATUS_ERROR_NOT_INITIALIZED : original(reader);
}

hsa_status_t HSA_API waitcheck_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object) {
  auto *original = layer().load();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  g_stats.loads.fetch_add(1, std::memory_order_relaxed);
  ReaderEntry *entry = ReaderRegistry::instance().lookup(reader);
  if (entry != nullptr) {
    std::lock_guard entry_lock(entry->mutex);
    if (!entry->active || entry->handle != reader.handle) {
      g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      return original(executable, agent, reader, options, loaded_code_object);
    }
    if (!entry->analyzed) {
      entry->result = check_code_object(entry->bytes, entry->size);
      entry->analyzed = true;
    }
    if (entry->result.checked && !entry->result.passed &&
        env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
  } else if (env_enabled("ROCJITSU_WAITCHECK", true)) {
    g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
  }
  return original(executable, agent, reader, options, loaded_code_object);
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RJ_WAITCHECK_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_WAITCHECK_HOOK_EXPORT
#endif

extern "C" RJ_WAITCHECK_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                                uint64_t failed_tool_count,
                                                const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;
  return layer().install(table);
}

extern "C" RJ_WAITCHECK_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
