// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_waitcheck_hooks.cpp
/// @brief ROCR HSA tools hook for checking the final AMDGPU code object loaded.

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/hooks/hsa_tool_lifetime.h"

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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

using CreateFromFileWithOffsetSizeFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                        hsa_code_object_reader_t *);
using LoadedCodeObjectGetInfoFn = hsa_status_t (*)(hsa_loaded_code_object_t, uint32_t, void *);

inline constexpr uint32_t kLoadedCodeObjectInfoLoadDelta = 8;

struct AmdLoaderTablePrefix {
  void *query_host_address = nullptr;
  void *query_segment_descriptors = nullptr;
  void *query_executable = nullptr;
  void *executable_iterate_loaded_code_objects = nullptr;
  LoadedCodeObjectGetInfoFn loaded_code_object_get_info = nullptr;
  CreateFromFileWithOffsetSizeFn create_from_file_with_offset_size = nullptr;
};

enum class RuntimeMode { Dispatch, Eager };

[[nodiscard]] RuntimeMode runtime_mode() {
  const char *value = std::getenv("ROCJITSU_WAITCHECK_MODE");
  if (value != nullptr && std::string_view(value) == "eager")
    return RuntimeMode::Eager;
  return RuntimeMode::Dispatch;
}

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
  std::atomic<uint64_t> dispatches{0};
  std::atomic<uint64_t> checked{0};
  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> passed{0};
  std::atomic<uint64_t> hazards{0};
  std::atomic<uint64_t> analysis_errors{0};
  std::atomic<uint64_t> unavailable{0};

  void reset() {
    loads.store(0, std::memory_order_relaxed);
    dispatches.store(0, std::memory_order_relaxed);
    checked.store(0, std::memory_order_relaxed);
    cache_hits.store(0, std::memory_order_relaxed);
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
      "rocjitsu-waitcheck: summary loads=%llu dispatches=%llu checked=%llu cache_hits=%llu "
      "passed=%llu hazards=%llu analysis_errors=%llu unavailable=%llu\n",
      static_cast<unsigned long long>(g_stats.loads.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.dispatches.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.checked.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.cache_hits.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.passed.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.hazards.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.analysis_errors.load(std::memory_order_relaxed)),
      static_cast<unsigned long long>(g_stats.unavailable.load(std::memory_order_relaxed)));
}

void print_report(const rocjitsu::AmdGpuCodeObject &code_object,
                  const rocjitsu::WaitcheckReport &report, std::string_view kernel_name = {}) {
  if (report.passed())
    return;

  if (report.diagnostics_truncated) {
    std::fprintf(stderr,
                 "rocjitsu-waitcheck: at least %zu waitcnt hazard(s) in %s code object%s%.*s\n",
                 report.diagnostics_observed, rj_code_target_name(code_object.target_id()),
                 kernel_name.empty() ? "" : " kernel ", static_cast<int>(kernel_name.size()),
                 kernel_name.empty() ? "" : kernel_name.data());
  } else {
    std::fprintf(stderr, "rocjitsu-waitcheck: %zu waitcnt hazard(s) in %s code object%s%.*s\n",
                 report.diagnostics_observed, rj_code_target_name(code_object.target_id()),
                 kernel_name.empty() ? "" : " kernel ", static_cast<int>(kernel_name.size()),
                 kernel_name.empty() ? "" : kernel_name.data());
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
                            const rocjitsu::WaitcheckReport &report,
                            std::string_view kernel_name = {}) {
  std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed for %s code object%s%.*s",
               rj_code_target_name(code_object.target_id()), kernel_name.empty() ? "" : " kernel ",
               static_cast<int>(kernel_name.size()), kernel_name.empty() ? "" : kernel_name.data());
  if (!report.analysis_error.empty())
    std::fprintf(stderr, ": %s", report.analysis_error.c_str());
  std::fprintf(stderr, "\n");
}

void print_analysis_incomplete(const rocjitsu::AmdGpuCodeObject &code_object,
                               const rocjitsu::WaitcheckReport &report,
                               std::string_view kernel_name = {}) {
  std::fprintf(stderr, "rocjitsu-waitcheck: analysis incomplete for %s code object%s%.*s",
               rj_code_target_name(code_object.target_id()), kernel_name.empty() ? "" : " kernel ",
               static_cast<int>(kernel_name.size()), kernel_name.empty() ? "" : kernel_name.data());
  if (!report.incomplete_reason.empty())
    std::fprintf(stderr, ": %s", report.incomplete_reason.c_str());
  std::fprintf(stderr, " (observations=%zu)\n", report.incomplete_observations);
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
      } else if (!report.analysis_complete) {
        if (report.diagnostics_observed != 0)
          print_report(parsed, report);
        print_analysis_incomplete(parsed, report);
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

[[nodiscard]] AnalysisResult check_kernel(const rocjitsu::AmdGpuCodeObject &code_object,
                                          const rocjitsu::WaitcheckKernelInfo &kernel) {
  if (!env_enabled("ROCJITSU_WAITCHECK", true) || g_in_waitcheck)
    return {};

  g_in_waitcheck = true;
  AnalysisResult result;
  try {
    const rj_code_arch_t arch = rocjitsu::waitcheck_arch_for_target(code_object.target_id());
    if (code_object.is_valid() && arch != ROCJITSU_CODE_ARCH_INVALID) {
      result.checked = true;
      rocjitsu::WaitcheckOptions options;
      options.max_diagnostics = 32;
      auto report = rocjitsu::analyze_waitcnts_for_kernel(code_object, arch, kernel, options);
      if (!report.supported) {
        print_analysis_failure(code_object, report, kernel.name);
        result.passed = false;
        result.analysis_failed = true;
        g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
      } else if (!report.analysis_complete) {
        if (report.diagnostics_observed != 0)
          print_report(code_object, report, kernel.name);
        print_analysis_incomplete(code_object, report, kernel.name);
        result.passed = false;
        result.analysis_failed = true;
        g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
      } else {
        print_report(code_object, report, kernel.name);
        result.passed = report.passed();
      }
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed for kernel %s: %s\n",
                 kernel.name.c_str(), error.what());
    result.analysis_failed = true;
    g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
  } catch (...) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed for kernel %s\n",
                 kernel.name.c_str());
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
    // HSA API wrappers can be called by HIP module destructors after ordinary
    // C++ function-local statics have begun destruction. Keep hook state alive
    // until the process address space is torn down; OnUnload still clears it.
    static ReaderRegistry *registry = new ReaderRegistry();
    return *registry;
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

struct KernelCheck {
  [[nodiscard]] AnalysisResult run() {
    std::lock_guard lock(mutex);
    if (analyzed) {
      g_stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
      return result;
    }
    result = check_kernel(*code_object, info);
    analyzed = true;
    return result;
  }

  std::mutex mutex;
  std::shared_ptr<rocjitsu::AmdGpuCodeObject> code_object;
  rocjitsu::WaitcheckKernelInfo info;
  bool analyzed = false;
  AnalysisResult result;
};

struct LoadedCodeObjectEntry {
  uint64_t executable_handle = 0;
  hsa_loaded_code_object_t loaded{};
  std::vector<std::shared_ptr<KernelCheck>> kernels;
  bool activated = false;
};

class DispatchRegistry {
public:
  static DispatchRegistry &instance() {
    static DispatchRegistry *registry = new DispatchRegistry();
    return *registry;
  }

  void store(hsa_executable_t executable, hsa_loaded_code_object_t loaded, const uint8_t *bytes,
             size_t size) {
    if (bytes == nullptr || size == 0)
      return;
    try {
      auto code_object = std::make_shared<rocjitsu::AmdGpuCodeObject>(bytes, size);
      if (!code_object->is_valid() || rocjitsu::waitcheck_arch_for_target(
                                          code_object->target_id()) == ROCJITSU_CODE_ARCH_INVALID) {
        return;
      }

      auto entry = std::make_shared<LoadedCodeObjectEntry>();
      entry->executable_handle = executable.handle;
      entry->loaded = loaded;
      for (rocjitsu::WaitcheckKernelInfo &info : rocjitsu::waitcheck_kernels(*code_object)) {
        auto kernel = std::make_shared<KernelCheck>();
        kernel->code_object = code_object;
        kernel->info = std::move(info);
        entry->kernels.push_back(std::move(kernel));
      }
      if (entry->kernels.empty())
        return;

      std::lock_guard lock(mutex_);
      loaded_.push_back(std::move(entry));
    } catch (const std::exception &error) {
      std::fprintf(stderr, "rocjitsu-waitcheck: failed to index loaded code object: %s\n",
                   error.what());
      g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
      std::fprintf(stderr, "rocjitsu-waitcheck: failed to index loaded code object\n");
      g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void activate(hsa_executable_t executable, LoadedCodeObjectGetInfoFn get_info) {
    if (get_info == nullptr) {
      g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    std::vector<std::shared_ptr<LoadedCodeObjectEntry>> entries;
    {
      std::lock_guard lock(mutex_);
      for (const auto &entry : loaded_) {
        if (entry->executable_handle == executable.handle && !entry->activated)
          entries.push_back(entry);
      }
    }

    for (const auto &entry : entries) {
      int64_t load_delta = 0;
      if (get_info(entry->loaded, kLoadedCodeObjectInfoLoadDelta, &load_delta) !=
          HSA_STATUS_SUCCESS) {
        g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
        continue;
      }

      std::lock_guard lock(mutex_);
      for (const auto &kernel : entry->kernels) {
        const uint64_t descriptor = kernel->info.descriptor_vaddr;
        uint64_t runtime_address = 0;
        if (load_delta >= 0) {
          const uint64_t positive_delta = static_cast<uint64_t>(load_delta);
          if (descriptor > std::numeric_limits<uint64_t>::max() - positive_delta) {
            g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          runtime_address = descriptor + positive_delta;
        } else {
          const uint64_t magnitude = static_cast<uint64_t>(-(load_delta + 1)) + 1;
          if (descriptor < magnitude) {
            g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          runtime_address = descriptor - magnitude;
        }
        if (runtime_address == 0) {
          g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        kernels_.insert_or_assign(runtime_address, kernel);
      }
      entry->activated = true;
    }
  }

  [[nodiscard]] std::shared_ptr<KernelCheck> lookup(uint64_t kernel_object) {
    std::lock_guard lock(mutex_);
    auto it = kernels_.find(kernel_object);
    return it == kernels_.end() ? nullptr : it->second;
  }

  void remove(hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    for (auto it = loaded_.begin(); it != loaded_.end();) {
      if ((*it)->executable_handle != executable.handle) {
        ++it;
        continue;
      }
      for (const auto &kernel : (*it)->kernels) {
        for (auto map_it = kernels_.begin(); map_it != kernels_.end();) {
          if (map_it->second == kernel)
            map_it = kernels_.erase(map_it);
          else
            ++map_it;
        }
      }
      it = loaded_.erase(it);
    }
  }

  void clear() {
    std::lock_guard lock(mutex_);
    kernels_.clear();
    loaded_.clear();
  }

private:
  std::mutex mutex_;
  std::vector<std::shared_ptr<LoadedCodeObjectEntry>> loaded_;
  std::unordered_map<uint64_t, std::shared_ptr<KernelCheck>> kernels_;
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
hsa_status_t HSA_API waitcheck_executable_freeze(hsa_executable_t executable, const char *options);
hsa_status_t HSA_API waitcheck_executable_destroy(hsa_executable_t executable);
hsa_status_t HSA_API waitcheck_queue_create(hsa_agent_t agent, uint32_t size,
                                            hsa_queue_type32_t type,
                                            void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                            void *data, uint32_t private_segment_size,
                                            uint32_t group_segment_size, hsa_queue_t **queue);
hsa_status_t HSA_API waitcheck_amd_queue_intercept_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue);
hsa_status_t HSA_API waitcheck_amd_queue_create(hsa_agent_t agent,
                                                hsa_amd_queue_create_desc_t *descs,
                                                uint32_t num_descs);
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
    amd_ext_ = table->amd_ext_;
    original_get_extension_table_ = core_->hsa_system_get_extension_table_fn;
    original_get_major_extension_table_ = core_->hsa_system_get_major_extension_table_fn;
    original_create_from_file_ = core_->hsa_code_object_reader_create_from_file_fn;
    original_create_from_memory_ = core_->hsa_code_object_reader_create_from_memory_fn;
    original_destroy_ = core_->hsa_code_object_reader_destroy_fn;
    original_load_ = core_->hsa_executable_load_agent_code_object_fn;
    original_freeze_ = core_->hsa_executable_freeze_fn;
    original_executable_destroy_ = core_->hsa_executable_destroy_fn;
    original_queue_create_ = core_->hsa_queue_create_fn;
    original_queue_destroy_ = core_->hsa_queue_destroy_fn;
    const bool amd_queue_create_table_valid =
        amd_ext_ != nullptr &&
        amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_create_fn) +
                                          sizeof(AmdExtTable::hsa_amd_queue_create_fn);
    original_amd_queue_create_ =
        amd_queue_create_table_valid ? amd_ext_->hsa_amd_queue_create_fn : nullptr;
    mode_ = runtime_mode();

    if (mode_ == RuntimeMode::Dispatch) {
      constexpr size_t intercept_table_size =
          offsetof(AmdExtTable, hsa_amd_queue_intercept_register_fn) +
          sizeof(AmdExtTable::hsa_amd_queue_intercept_register_fn);
      if (table->amd_ext_ == nullptr || table->amd_ext_->version.minor_id < intercept_table_size ||
          table->amd_ext_->hsa_amd_queue_intercept_create_fn == nullptr ||
          table->amd_ext_->hsa_amd_queue_intercept_register_fn == nullptr) {
        std::fprintf(stderr,
                     "rocjitsu-waitcheck: queue interception unavailable; using eager mode\n");
        mode_ = RuntimeMode::Eager;
      } else {
        original_intercept_create_ = table->amd_ext_->hsa_amd_queue_intercept_create_fn;
        original_intercept_register_ = table->amd_ext_->hsa_amd_queue_intercept_register_fn;
      }
    }

    if (original_get_extension_table_ == nullptr ||
        original_get_major_extension_table_ == nullptr || original_create_from_file_ == nullptr ||
        original_create_from_memory_ == nullptr || original_destroy_ == nullptr ||
        original_load_ == nullptr ||
        (mode_ == RuntimeMode::Dispatch &&
         (original_freeze_ == nullptr || original_executable_destroy_ == nullptr ||
          original_queue_create_ == nullptr || original_queue_destroy_ == nullptr))) {
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
    if (mode_ == RuntimeMode::Dispatch) {
      core_->hsa_executable_freeze_fn = waitcheck_executable_freeze;
      core_->hsa_executable_destroy_fn = waitcheck_executable_destroy;
      core_->hsa_queue_create_fn = waitcheck_queue_create;
      amd_ext_->hsa_amd_queue_intercept_create_fn = waitcheck_amd_queue_intercept_create;
      if (original_amd_queue_create_ != nullptr)
        amd_ext_->hsa_amd_queue_create_fn = waitcheck_amd_queue_create;
    }
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
        if (core_->hsa_executable_freeze_fn == waitcheck_executable_freeze)
          core_->hsa_executable_freeze_fn = original_freeze_;
        if (core_->hsa_executable_destroy_fn == waitcheck_executable_destroy)
          core_->hsa_executable_destroy_fn = original_executable_destroy_;
        if (core_->hsa_queue_create_fn == waitcheck_queue_create)
          core_->hsa_queue_create_fn = original_queue_create_;
        if (amd_ext_ != nullptr &&
            amd_ext_->hsa_amd_queue_intercept_create_fn == waitcheck_amd_queue_intercept_create) {
          amd_ext_->hsa_amd_queue_intercept_create_fn = original_intercept_create_;
        }
        if (amd_ext_ != nullptr &&
            amd_ext_->hsa_amd_queue_create_fn == waitcheck_amd_queue_create) {
          amd_ext_->hsa_amd_queue_create_fn = original_amd_queue_create_;
        }
      }
      active_ = false;
    }

    ReaderRegistry::instance().clear();
    DispatchRegistry::instance().clear();
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
  [[nodiscard]] decltype(hsa_executable_freeze) *freeze() const {
    std::lock_guard lock(mutex_);
    return original_freeze_;
  }
  [[nodiscard]] decltype(hsa_executable_destroy) *executable_destroy() const {
    std::lock_guard lock(mutex_);
    return original_executable_destroy_;
  }
  [[nodiscard]] decltype(hsa_queue_destroy) *queue_destroy() const {
    std::lock_guard lock(mutex_);
    return original_queue_destroy_;
  }
  [[nodiscard]] hsa_amd_queue_create_fn_t amd_queue_create() const {
    std::lock_guard lock(mutex_);
    return original_amd_queue_create_;
  }
  [[nodiscard]] hsa_amd_queue_set_priority_fn_t queue_set_priority() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_set_priority_fn;
  }
  [[nodiscard]] hsa_amd_queue_cu_set_mask_fn_t queue_cu_set_mask() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_cu_set_mask_fn;
  }
  [[nodiscard]] hsa_amd_queue_intercept_create_fn_t intercept_create() const {
    std::lock_guard lock(mutex_);
    return original_intercept_create_;
  }
  [[nodiscard]] hsa_amd_queue_intercept_register_fn_t intercept_register() const {
    std::lock_guard lock(mutex_);
    return original_intercept_register_;
  }
  [[nodiscard]] LoadedCodeObjectGetInfoFn loaded_code_object_get_info() const {
    std::lock_guard lock(mutex_);
    return original_loaded_code_object_get_info_;
  }
  [[nodiscard]] RuntimeMode mode() const {
    std::lock_guard lock(mutex_);
    return mode_;
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
  void save_loaded_code_object_get_info(LoadedCodeObjectGetInfoFn original) {
    std::lock_guard lock(mutex_);
    if (original != nullptr)
      original_loaded_code_object_get_info_ = original;
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
    amd_ext_ = nullptr;
    original_get_extension_table_ = nullptr;
    original_get_major_extension_table_ = nullptr;
    original_create_from_file_ = nullptr;
    original_create_from_memory_ = nullptr;
    original_destroy_ = nullptr;
    original_load_ = nullptr;
    original_freeze_ = nullptr;
    original_executable_destroy_ = nullptr;
    original_queue_create_ = nullptr;
    original_queue_destroy_ = nullptr;
    original_amd_queue_create_ = nullptr;
    original_intercept_create_ = nullptr;
    original_intercept_register_ = nullptr;
    original_loaded_code_object_get_info_ = nullptr;
    original_file_range_create_ = nullptr;
    mode_ = RuntimeMode::Eager;
  }

  mutable std::mutex mutex_;
  bool active_ = false;
  CoreApiTable *core_ = nullptr;
  AmdExtTable *amd_ext_ = nullptr;
  decltype(hsa_system_get_extension_table) *original_get_extension_table_ = nullptr;
  decltype(hsa_system_get_major_extension_table) *original_get_major_extension_table_ = nullptr;
  decltype(hsa_code_object_reader_create_from_file) *original_create_from_file_ = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *original_create_from_memory_ = nullptr;
  decltype(hsa_code_object_reader_destroy) *original_destroy_ = nullptr;
  decltype(hsa_executable_load_agent_code_object) *original_load_ = nullptr;
  decltype(hsa_executable_freeze) *original_freeze_ = nullptr;
  decltype(hsa_executable_destroy) *original_executable_destroy_ = nullptr;
  decltype(hsa_queue_create) *original_queue_create_ = nullptr;
  decltype(hsa_queue_destroy) *original_queue_destroy_ = nullptr;
  hsa_amd_queue_create_fn_t original_amd_queue_create_ = nullptr;
  hsa_amd_queue_intercept_create_fn_t original_intercept_create_ = nullptr;
  hsa_amd_queue_intercept_register_fn_t original_intercept_register_ = nullptr;
  LoadedCodeObjectGetInfoFn original_loaded_code_object_get_info_ = nullptr;
  CreateFromFileWithOffsetSizeFn original_file_range_create_ = nullptr;
  RuntimeMode mode_ = RuntimeMode::Eager;
};

WaitcheckHsaLayer &layer() {
  static WaitcheckHsaLayer *state = new WaitcheckHsaLayer();
  return *state;
}

[[nodiscard]] bool amd_loader_table_has_file_range_reader(size_t table_length) {
  constexpr size_t reader_offset =
      offsetof(AmdLoaderTablePrefix, create_from_file_with_offset_size);
  return table_length >= reader_offset + sizeof(CreateFromFileWithOffsetSizeFn);
}

[[nodiscard]] bool amd_loader_table_has_loaded_code_object_info(size_t table_length) {
  constexpr size_t info_offset = offsetof(AmdLoaderTablePrefix, loaded_code_object_get_info);
  return table_length >= info_offset + sizeof(LoadedCodeObjectGetInfoFn);
}

void patch_amd_loader_extension_table(uint16_t extension, uint16_t version_major,
                                      size_t table_length, void *table) {
  if (extension != HSA_EXTENSION_AMD_LOADER || version_major != 1 || table == nullptr) {
    return;
  }
  auto *loader = static_cast<AmdLoaderTablePrefix *>(table);
  if (amd_loader_table_has_loaded_code_object_info(table_length))
    layer().save_loaded_code_object_get_info(loader->loaded_code_object_get_info);
  if (amd_loader_table_has_file_range_reader(table_length)) {
    layer().save_file_range_create(loader->create_from_file_with_offset_size);
    if (loader->create_from_file_with_offset_size != nullptr) {
      loader->create_from_file_with_offset_size =
          waitcheck_reader_create_from_file_with_offset_size;
    }
  }
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
  if (layer().mode() == RuntimeMode::Dispatch) {
    hsa_loaded_code_object_t captured_loaded{};
    hsa_loaded_code_object_t *runtime_output =
        loaded_code_object == nullptr ? &captured_loaded : loaded_code_object;
    const hsa_status_t status = original(executable, agent, reader, options, runtime_output);
    if (status != HSA_STATUS_SUCCESS || !env_enabled("ROCJITSU_WAITCHECK", true))
      return status;

    if (entry == nullptr) {
      g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      return status;
    }

    // The HSA contract keeps the reader and its backing memory alive through
    // executable destruction. Index directly from it after a successful load;
    // AmdGpuCodeObject takes the one persistent copy needed for lazy dispatch.
    std::lock_guard entry_lock(entry->mutex);
    if (!entry->active || entry->handle != reader.handle || entry->bytes == nullptr ||
        entry->size == 0) {
      g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      return status;
    }
    DispatchRegistry::instance().store(executable, *runtime_output, entry->bytes, entry->size);
    return status;
  }

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

hsa_status_t HSA_API waitcheck_executable_freeze(hsa_executable_t executable, const char *options) {
  auto *original = layer().freeze();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status = original(executable, options);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  LoadedCodeObjectGetInfoFn get_info = layer().loaded_code_object_get_info();
  if (get_info == nullptr) {
    AmdLoaderTablePrefix loader{};
    auto *get_table = layer().get_major_extension_table();
    if (get_table != nullptr &&
        get_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader) == HSA_STATUS_SUCCESS) {
      layer().save_loaded_code_object_get_info(loader.loaded_code_object_get_info);
      get_info = loader.loaded_code_object_get_info;
    }
  }
  DispatchRegistry::instance().activate(executable, get_info);
  return status;
}

hsa_status_t HSA_API waitcheck_executable_destroy(hsa_executable_t executable) {
  auto *original = layer().executable_destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status = original(executable);
  if (status == HSA_STATUS_SUCCESS)
    DispatchRegistry::instance().remove(executable);
  return status;
}

void waitcheck_queue_intercept(const void *packets, uint64_t packet_count, uint64_t, void *,
                               hsa_amd_queue_intercept_packet_writer_t writer) {
  static_assert(sizeof(hsa_kernel_dispatch_packet_t) == 64);
  if (packets == nullptr || writer == nullptr)
    return;

  const auto *packet_bytes = static_cast<const uint8_t *>(packets);
  constexpr uint16_t type_mask = static_cast<uint16_t>((1U << HSA_PACKET_HEADER_WIDTH_TYPE) - 1U);
  for (uint64_t i = 0; i < packet_count; ++i) {
    const auto *packet = reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(
        packet_bytes + i * sizeof(hsa_kernel_dispatch_packet_t));
    const uint16_t type =
        static_cast<uint16_t>((packet->header >> HSA_PACKET_HEADER_TYPE) & type_mask);
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
      continue;

    g_stats.dispatches.fetch_add(1, std::memory_order_relaxed);
    std::shared_ptr<KernelCheck> kernel =
        DispatchRegistry::instance().lookup(packet->kernel_object);
    if (kernel == nullptr) {
      if (env_enabled("ROCJITSU_WAITCHECK", true))
        g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    const AnalysisResult result = kernel->run();
    if (result.checked && !result.passed && env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
      std::fprintf(stderr, "rocjitsu-waitcheck: refusing to dispatch kernel %s; aborting process\n",
                   kernel->info.name.c_str());
      std::fflush(stderr);
      std::abort();
    }
  }
  writer(packets, packet_count);
}

hsa_status_t HSA_API waitcheck_queue_create(hsa_agent_t agent, uint32_t size,
                                            hsa_queue_type32_t type,
                                            void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                            void *data, uint32_t private_segment_size,
                                            uint32_t group_segment_size, hsa_queue_t **queue) {
  auto create = layer().intercept_create();
  auto register_intercept = layer().intercept_register();
  if (create == nullptr || register_intercept == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  const hsa_status_t status =
      create(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
  if (status != HSA_STATUS_SUCCESS || queue == nullptr || *queue == nullptr)
    return status;

  const hsa_status_t register_status =
      register_intercept(*queue, waitcheck_queue_intercept, nullptr);
  if (register_status == HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;

  if (auto *destroy = layer().queue_destroy(); destroy != nullptr)
    (void)destroy(*queue);
  *queue = nullptr;
  return register_status;
}

hsa_status_t HSA_API waitcheck_amd_queue_intercept_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue) {
  return waitcheck_queue_create(agent, size, type, callback, data, private_segment_size,
                                group_segment_size, queue);
}

hsa_status_t HSA_API waitcheck_amd_queue_create(hsa_agent_t agent,
                                                hsa_amd_queue_create_desc_t *descs,
                                                uint32_t num_descs) {
  auto *original_create = layer().amd_queue_create();
  if (original_create == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  const hsa_status_t status = original_create(agent, descs, num_descs);
  if (descs == nullptr)
    return status;

  hsa_status_t replacement_error = HSA_STATUS_SUCCESS;
  const auto note_replacement_error = [&](hsa_status_t error) {
    if (replacement_error == HSA_STATUS_SUCCESS)
      replacement_error = error;
  };
  const auto destroy_queue = [](hsa_queue_t *queue) {
    auto *destroy = layer().queue_destroy();
    return queue == nullptr || destroy == nullptr ? HSA_STATUS_ERROR : destroy(queue);
  };

  for (uint32_t index = 0; index < num_descs; ++index) {
    hsa_amd_queue_create_desc_t &desc = descs[index];
    hsa_queue_t *plain_queue = desc.queue;
    if (plain_queue == nullptr || desc.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE)
      continue;

    const hsa_amd_compute_queue_params_t &compute = desc.engine.compute;
    const uint32_t packet_count =
        desc.queue_size_bytes / static_cast<uint32_t>(sizeof(hsa_kernel_dispatch_packet_t));
    hsa_queue_t *intercept_queue = nullptr;
    hsa_status_t intercept_status =
        waitcheck_queue_create(agent, packet_count, compute.type, desc.callback, desc.callback_data,
                               compute.private_segment_size, UINT32_MAX, &intercept_queue);

    if (intercept_status == HSA_STATUS_SUCCESS && desc.priority != HSA_AMD_QUEUE_PRIORITY_NORMAL) {
      auto *set_priority = layer().queue_set_priority();
      intercept_status = set_priority == nullptr ? HSA_STATUS_ERROR_INVALID_QUEUE_CREATION
                                                 : set_priority(intercept_queue, desc.priority);
    }
    if (intercept_status == HSA_STATUS_SUCCESS && compute.cu_mask_count != 0u) {
      auto *set_cu_mask = layer().queue_cu_set_mask();
      intercept_status = set_cu_mask == nullptr
                             ? HSA_STATUS_ERROR_INVALID_QUEUE_CREATION
                             : set_cu_mask(intercept_queue, compute.cu_mask_count, compute.cu_mask);
    }

    if (intercept_status == HSA_STATUS_SUCCESS) {
      const hsa_status_t destroy_status = destroy_queue(plain_queue);
      if (destroy_status == HSA_STATUS_SUCCESS) {
        desc.queue = intercept_queue;
        continue;
      }
      intercept_status = destroy_status;
    }

    if (intercept_queue != nullptr)
      (void)destroy_queue(intercept_queue);
    (void)destroy_queue(plain_queue);
    desc.queue = nullptr;
    note_replacement_error(intercept_status);
  }

  return status != HSA_STATUS_SUCCESS ? status : replacement_error;
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
  if (!rocjitsu::hooks::retain_hsa_tool_dso())
    return false;
  return layer().install(table);
}

extern "C" RJ_WAITCHECK_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
