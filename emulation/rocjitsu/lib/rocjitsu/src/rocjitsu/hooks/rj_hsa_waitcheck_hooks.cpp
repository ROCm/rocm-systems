// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_waitcheck_hooks.cpp
/// @brief ROCR HSA tools hook for checking the final AMDGPU code object loaded.

#include "hsa/hsa_api_trace.h"
#include "rocjitsu/hooks/hsa_tool_lifetime.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/waitcheck.h"
#include "rocjitsu/hooks/hsa_api_function_patch.h"
#include "rocjitsu/hooks/hsa_code_object_file_snapshot.h"

#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/target_registry.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/stat.h>

namespace {

using CreateFromFileWithOffsetSizeFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                        hsa_code_object_reader_t *);
using LoadedCodeObjectGetInfoFn = hsa_status_t (*)(hsa_loaded_code_object_t, uint32_t, void *);

inline constexpr uint32_t kLoadedCodeObjectInfoLoadDelta = 8;
inline constexpr uint32_t kLoadedCodeObjectInfoStorageType = 4;
inline constexpr uint32_t kLoadedCodeObjectInfoMemoryBase = 5;
inline constexpr uint32_t kLoadedCodeObjectInfoMemorySize = 6;
inline constexpr uint32_t kLoadedCodeObjectInfoFile = 7;
inline constexpr uint32_t kLoadedCodeObjectInfoUriLength = 11;
inline constexpr uint32_t kLoadedCodeObjectInfoUri = 12;

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
  bool hazards_found = false;
};

struct ReaderEntry {
  std::mutex mutex;
  bool analyzed = false;
  const uint8_t *bytes = nullptr;
  size_t size = 0;
  rocjitsu::CodeObjectFileSnapshot owned;
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

[[nodiscard]] std::string target_name(const rocjitsu::AmdGpuCodeObject &code_object) {
  if (!code_object.target_triple().empty())
    return code_object.target_triple();
  const auto *target =
      rocjitsu::default_isa_target_registry().find_gpu_target(code_object.target_id());
  return target == nullptr ? "unknown" : std::string(target->code_object_id);
}

void print_report(const rocjitsu::AmdGpuCodeObject &code_object,
                  const rocjitsu::WaitcheckReport &report, std::string_view kernel_name = {}) {
  if (report.passed())
    return;

  const std::string target = target_name(code_object);

  if (report.stopped_early) {
    std::fprintf(
        stderr, "rocjitsu-waitcheck: at least %zu waitcnt hazard(s) in %s code object%s%.*s\n",
        report.diagnostics_observed, target.c_str(), kernel_name.empty() ? "" : " kernel ",
        static_cast<int>(kernel_name.size()), kernel_name.empty() ? "" : kernel_name.data());
  } else {
    std::fprintf(stderr, "rocjitsu-waitcheck: %zu waitcnt hazard(s) in %s code object%s%.*s\n",
                 report.diagnostics_observed, target.c_str(), kernel_name.empty() ? "" : " kernel ",
                 static_cast<int>(kernel_name.size()),
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
  if (report.diagnostics_observed > limit) {
    std::fprintf(stderr, "rocjitsu-waitcheck: omitted %s%zu diagnostic(s) after limit\n",
                 report.stopped_early ? "at least " : "", report.diagnostics_observed - limit);
  } else if (report.stopped_early) {
    std::fprintf(stderr,
                 "rocjitsu-waitcheck: analysis stopped early; additional diagnostics may exist\n");
  }
}

void print_analysis_failure(const rocjitsu::AmdGpuCodeObject &code_object,
                            const rocjitsu::WaitcheckReport &report,
                            std::string_view kernel_name = {}) {
  const std::string target = target_name(code_object);
  std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed for %s code object%s%.*s",
               target.c_str(), kernel_name.empty() ? "" : " kernel ",
               static_cast<int>(kernel_name.size()), kernel_name.empty() ? "" : kernel_name.data());
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
      result.hazards_found = report.diagnostics_observed != 0;
      if (!report.supported || !report.complete) {
        if (result.hazards_found)
          print_report(parsed, report);
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
    result.checked = true;
    result.passed = false;
    result.analysis_failed = true;
    g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
  } catch (...) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed\n");
    result.checked = true;
    result.passed = false;
    result.analysis_failed = true;
    g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
  }
  g_in_waitcheck = false;

  if (result.checked) {
    g_stats.checked.fetch_add(1, std::memory_order_relaxed);
    if (result.passed)
      g_stats.passed.fetch_add(1, std::memory_order_relaxed);
    if (result.hazards_found)
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
      result.hazards_found = report.diagnostics_observed != 0;
      if (!report.supported || !report.complete) {
        if (result.hazards_found)
          print_report(code_object, report, kernel.name);
        print_analysis_failure(code_object, report, kernel.name);
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
    result.checked = true;
    result.passed = false;
    result.analysis_failed = true;
    g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
  } catch (...) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed for kernel %s\n",
                 kernel.name.c_str());
    result.checked = true;
    result.passed = false;
    result.analysis_failed = true;
    g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
  }
  g_in_waitcheck = false;

  if (result.checked) {
    g_stats.checked.fetch_add(1, std::memory_order_relaxed);
    if (result.passed)
      g_stats.passed.fetch_add(1, std::memory_order_relaxed);
    if (result.hazards_found)
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
    auto entry = std::make_shared<ReaderEntry>();
    entry->bytes = bytes;
    entry->size = size;
    std::lock_guard lock(mutex_);
    entries_.insert_or_assign(reader.handle, std::move(entry));
  }

  void store_owned(hsa_code_object_reader_t reader, rocjitsu::CodeObjectFileSnapshot bytes) {
    auto entry = std::make_shared<ReaderEntry>();
    entry->owned = std::move(bytes);
    entry->bytes = entry->owned->data();
    entry->size = entry->owned->size();
    std::lock_guard lock(mutex_);
    entries_.insert_or_assign(reader.handle, std::move(entry));
  }

  [[nodiscard]] std::shared_ptr<ReaderEntry> lookup(hsa_code_object_reader_t reader) {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(reader.handle);
    return it == entries_.end() ? nullptr : it->second;
  }

  void remove(hsa_code_object_reader_t reader) {
    std::lock_guard lock(mutex_);
    entries_.erase(reader.handle);
  }
  void clear() {
    std::lock_guard lock(mutex_);
    entries_.clear();
  }

private:
  std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<ReaderEntry>> entries_;
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
  uint64_t agent_handle = 0;
  hsa_loaded_code_object_t loaded{};
  std::vector<std::shared_ptr<KernelCheck>> kernels;
  bool ready = false;
  std::thread::id load_thread;
  bool load_wrapper_complete = false;
  bool activating = false;
  bool activated = false;
};

class DispatchRegistry {
public:
  static DispatchRegistry &instance() {
    static DispatchRegistry *registry = new DispatchRegistry();
    return *registry;
  }

  struct PreparedLoad {
    std::shared_ptr<LoadedCodeObjectEntry> entry;
    bool passed = true;
  };

  PreparedLoad prepare(hsa_executable_t executable, hsa_agent_t agent, const uint8_t *bytes,
                       size_t size) {
    if (bytes == nullptr || size == 0)
      return {};
    try {
      auto code_object = std::make_shared<rocjitsu::AmdGpuCodeObject>(bytes, size);
      if (!code_object->is_valid() || rocjitsu::waitcheck_arch_for_target(
                                          code_object->target_id()) == ROCJITSU_CODE_ARCH_INVALID) {
        return {};
      }

      auto entry = std::make_shared<LoadedCodeObjectEntry>();
      entry->executable_handle = executable.handle;
      entry->agent_handle = agent.handle;
      entry->load_thread = std::this_thread::get_id();
      const auto add_kernel = [&](rocjitsu::WaitcheckKernelInfo info) {
        auto kernel = std::make_shared<KernelCheck>();
        kernel->code_object = code_object;
        kernel->info = std::move(info);
        entry->kernels.push_back(std::move(kernel));
      };
      for (auto &info : rocjitsu::waitcheck_kernels_exact(*code_object))
        add_kernel(std::move(info));
      if (entry->kernels.empty())
        return {};

      bool cooperative_active = false;
      {
        std::lock_guard lock(mutex_);
        // Cooperative creation also checks pending loads. Publish the candidate
        // before either side checks it, but keep it out of dispatch lookup until
        // the underlying HSA load succeeds.
        loaded_.push_back(entry);
        cooperative_active = cooperative_creates_.contains(agent.handle) ||
                             std::ranges::any_of(cooperative_queues_, [&](const auto &queue) {
                               return queue.second.agent_handle == agent.handle;
                             });
      }
      return {entry, !cooperative_active || check_all(entry->kernels)};
    } catch (const std::exception &error) {
      std::fprintf(stderr, "rocjitsu-waitcheck: failed to index loaded code object: %s\n",
                   error.what());
      g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
      std::fprintf(stderr, "rocjitsu-waitcheck: failed to index loaded code object\n");
      g_stats.analysis_errors.fetch_add(1, std::memory_order_relaxed);
    }
    return {.entry = nullptr, .passed = false};
  }

  void complete(const PreparedLoad &prepared, hsa_loaded_code_object_t loaded, bool success) {
    if (!prepared.entry)
      return;
    LoadedCodeObjectGetInfoFn frozen_get_info = nullptr;
    {
      std::lock_guard lock(mutex_);
      if (!success) {
        std::erase(loaded_, prepared.entry);
        pending_loads_changed_.notify_all();
        return;
      }
      prepared.entry->loaded = loaded;
      prepared.entry->ready = true;
      if (auto it = frozen_executables_.find(prepared.entry->executable_handle);
          it != frozen_executables_.end())
        frozen_get_info = it->second;
    }
    // Freeze may have completed while the real loader was still returning.
    if (frozen_get_info != nullptr)
      activate({prepared.entry->executable_handle}, frozen_get_info, false);
    {
      std::lock_guard lock(mutex_);
      prepared.entry->load_wrapper_complete = true;
    }
    pending_loads_changed_.notify_all();
  }

  void activate(hsa_executable_t executable, LoadedCodeObjectGetInfoFn get_info,
                bool record_freeze = true) {
    if (get_info == nullptr) {
      g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    std::vector<std::shared_ptr<LoadedCodeObjectEntry>> entries;
    {
      std::unique_lock lock(mutex_);
      if (record_freeze)
        frozen_executables_[executable.handle] = get_info;
      else if (auto it = frozen_executables_.find(executable.handle);
               it == frozen_executables_.end() || it->second != get_info)
        return;
      if (record_freeze) {
        // The real loader may have released ROCR's executable lock while an
        // earlier tool's load wrapper is still returning. A successful freeze
        // must not expose that entry to dispatch before our load wrapper has
        // published its descriptor mappings. Reentrant freeze on the same
        // loading thread is completed by complete() after the load returns.
        const auto caller = std::this_thread::get_id();
        pending_loads_changed_.wait(lock, [&] {
          return std::ranges::none_of(loaded_, [&](const auto &entry) {
            return entry->executable_handle == executable.handle &&
                   entry->load_thread != caller && !entry->load_wrapper_complete;
          });
        });
      }
      for (const auto &entry : loaded_) {
        if (entry->executable_handle == executable.handle && entry->ready && !entry->activating &&
            !entry->activated) {
          entry->activating = true;
          entries.push_back(entry);
        }
      }
    }

    for (const auto &entry : entries) {
      int64_t load_delta = 0;
      if (get_info(entry->loaded, kLoadedCodeObjectInfoLoadDelta, &load_delta) !=
          HSA_STATUS_SUCCESS) {
        std::lock_guard lock(mutex_);
        entry->activating = false;
        g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
        continue;
      }

      std::lock_guard lock(mutex_);
      if (std::ranges::find(loaded_, entry) == loaded_.end())
        continue;
      for (const auto &kernel : entry->kernels) {
        const auto register_descriptor = [&](uint64_t descriptor) {
          uint64_t runtime_address = 0;
          if (load_delta >= 0) {
            const uint64_t positive_delta = static_cast<uint64_t>(load_delta);
            if (descriptor > std::numeric_limits<uint64_t>::max() - positive_delta) {
              g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
              return;
            }
            runtime_address = descriptor + positive_delta;
          } else {
            const uint64_t magnitude = static_cast<uint64_t>(-(load_delta + 1)) + 1;
            if (descriptor < magnitude) {
              g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
              return;
            }
            runtime_address = descriptor - magnitude;
          }
          if (runtime_address == 0) {
            g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
            return;
          }
          kernels_.insert_or_assign(runtime_address, kernel);
        };
        register_descriptor(kernel->info.descriptor_vaddr);
      }
      entry->activated = true;
      entry->activating = false;
    }
  }

  [[nodiscard]] std::shared_ptr<KernelCheck> lookup(uint64_t kernel_object) {
    std::lock_guard lock(mutex_);
    auto it = kernels_.find(kernel_object);
    return it == kernels_.end() ? nullptr : it->second;
  }

  void remove(hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    frozen_executables_.erase(executable.handle);
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
    pending_loads_changed_.notify_all();
  }

  bool begin_cooperative_create(hsa_agent_t agent, bool check_waits) {
    std::vector<std::shared_ptr<KernelCheck>> kernels;
    {
      std::lock_guard lock(mutex_);
      ++cooperative_creates_[agent.handle];
      if (check_waits)
        for (const auto &entry : loaded_)
          if (entry->agent_handle == agent.handle)
            kernels.insert(kernels.end(), entry->kernels.begin(), entry->kernels.end());
    }
    return !check_waits || check_all(kernels);
  }

  void finish_cooperative_create(hsa_agent_t agent, hsa_queue_t *queue) {
    std::lock_guard lock(mutex_);
    if (queue) {
      auto &active = cooperative_queues_[queue];
      active.agent_handle = agent.handle;
      ++active.refs;
    }
    auto pending = cooperative_creates_.find(agent.handle);
    if (pending != cooperative_creates_.end() && --pending->second == 0)
      cooperative_creates_.erase(pending);
  }

  void release_cooperative_queue(hsa_queue_t *queue) {
    std::lock_guard lock(mutex_);
    auto it = cooperative_queues_.find(queue);
    if (it != cooperative_queues_.end() && --it->second.refs == 0)
      cooperative_queues_.erase(it);
  }

  void clear() {
    std::lock_guard lock(mutex_);
    kernels_.clear();
    loaded_.clear();
    frozen_executables_.clear();
    cooperative_queues_.clear();
    cooperative_creates_.clear();
    pending_loads_changed_.notify_all();
  }

private:
  struct CooperativeQueue {
    uint64_t agent_handle = 0;
    size_t refs = 0;
  };
  static bool check_all(const std::vector<std::shared_ptr<KernelCheck>> &kernels) {
    bool passed = true;
    for (const auto &kernel : kernels)
      passed &= kernel->run().passed;
    return passed;
  }
  std::unordered_map<hsa_queue_t *, CooperativeQueue> cooperative_queues_;
  std::unordered_map<uint64_t, size_t> cooperative_creates_;
  std::mutex mutex_;
  std::condition_variable pending_loads_changed_;
  std::vector<std::shared_ptr<LoadedCodeObjectEntry>> loaded_;
  std::unordered_map<uint64_t, LoadedCodeObjectGetInfoFn> frozen_executables_;
  std::unordered_map<uint64_t, std::shared_ptr<KernelCheck>> kernels_;
};

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

hsa_status_t HSA_API waitcheck_queue_destroy(hsa_queue_t *queue);
hsa_status_t HSA_API waitcheck_queue_get_info(hsa_queue_t *queue,
                                              hsa_queue_info_attribute_t attribute, void *value);

// Core API slots owned by WaitCheck. The final column is the runtime condition
// for installing the replacement; every entry is captured so wrappers and
// dispatch setup can still call the chain predecessor.
#define WAITCHECK_HSA_CORE_FUNCTIONS(X)                                                            \
  X(get_extension_table, hsa_system_get_extension_table_fn, waitcheck_system_get_extension_table,  \
    decltype(hsa_system_get_extension_table) *, true)                                              \
  X(get_major_extension_table, hsa_system_get_major_extension_table_fn,                            \
    waitcheck_system_get_major_extension_table, decltype(hsa_system_get_major_extension_table) *,  \
    true)                                                                                          \
  X(create_from_file, hsa_code_object_reader_create_from_file_fn,                                  \
    waitcheck_reader_create_from_file, decltype(hsa_code_object_reader_create_from_file) *, true)  \
  X(create_from_memory, hsa_code_object_reader_create_from_memory_fn,                              \
    waitcheck_reader_create_from_memory, decltype(hsa_code_object_reader_create_from_memory) *,    \
    true)                                                                                          \
  X(destroy, hsa_code_object_reader_destroy_fn, waitcheck_reader_destroy,                          \
    decltype(hsa_code_object_reader_destroy) *, true)                                              \
  X(load, hsa_executable_load_agent_code_object_fn, waitcheck_executable_load_agent_code_object,   \
    decltype(hsa_executable_load_agent_code_object) *, true)                                       \
  X(freeze, hsa_executable_freeze_fn, waitcheck_executable_freeze,                                 \
    decltype(hsa_executable_freeze) *, mode_ == RuntimeMode::Dispatch)                             \
  X(executable_destroy, hsa_executable_destroy_fn, waitcheck_executable_destroy,                   \
    decltype(hsa_executable_destroy) *, mode_ == RuntimeMode::Dispatch)                            \
  X(queue_create, hsa_queue_create_fn, waitcheck_queue_create, decltype(hsa_queue_create) *,       \
    mode_ == RuntimeMode::Dispatch)                                                                \
  X(queue_destroy, hsa_queue_destroy_fn, waitcheck_queue_destroy, decltype(hsa_queue_destroy) *,   \
    mode_ == RuntimeMode::Dispatch)

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
#define WAITCHECK_CAPTURE_CORE(name, field, wrapper, type, install_if)                             \
  name##_.capture(&core_->field);
    WAITCHECK_HSA_CORE_FUNCTIONS(WAITCHECK_CAPTURE_CORE)
#undef WAITCHECK_CAPTURE_CORE
    const bool amd_queue_create_table_valid =
        amd_ext_ != nullptr &&
        amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_create_fn) +
                                          sizeof(AmdExtTable::hsa_amd_queue_create_fn);
    if (amd_queue_create_table_valid)
      amd_queue_create_.capture(&amd_ext_->hsa_amd_queue_create_fn);
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
        if (amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_get_info_fn) +
                                              sizeof(AmdExtTable::hsa_amd_queue_get_info_fn))
          queue_info_.capture(&amd_ext_->hsa_amd_queue_get_info_fn);
        intercept_create_.capture(&amd_ext_->hsa_amd_queue_intercept_create_fn);
        intercept_register_.capture(&amd_ext_->hsa_amd_queue_intercept_register_fn);
      }
    }

    if (get_extension_table_.original() == nullptr ||
        get_major_extension_table_.original() == nullptr ||
        create_from_file_.original() == nullptr || create_from_memory_.original() == nullptr ||
        destroy_.original() == nullptr || load_.original() == nullptr ||
        (mode_ == RuntimeMode::Dispatch &&
         (freeze_.original() == nullptr || executable_destroy_.original() == nullptr ||
          queue_create_.original() == nullptr || queue_destroy_.original() == nullptr))) {
      std::fprintf(stderr, "rocjitsu-waitcheck: HSA core table contains null required entries\n");
      clear_unlocked();
      return false;
    }

#define WAITCHECK_INSTALL_CORE(name, field, wrapper, type, install_if)                             \
  if (install_if)                                                                                  \
    name##_.install(wrapper);
    WAITCHECK_HSA_CORE_FUNCTIONS(WAITCHECK_INSTALL_CORE)
#undef WAITCHECK_INSTALL_CORE
    if (mode_ == RuntimeMode::Dispatch) {
      if (queue_info_.original())
        queue_info_.install(waitcheck_queue_get_info);
      intercept_create_.install(waitcheck_amd_queue_intercept_create);
      if (amd_queue_create_.original() != nullptr)
        amd_queue_create_.install(waitcheck_amd_queue_create);
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
#define WAITCHECK_RESTORE_CORE(name, field, wrapper, type, install_if) name##_.restore();
        WAITCHECK_HSA_CORE_FUNCTIONS(WAITCHECK_RESTORE_CORE)
#undef WAITCHECK_RESTORE_CORE
        queue_info_.restore();
        intercept_create_.restore();
        amd_queue_create_.restore();
      }
      active_ = false;
    }

    ReaderRegistry::instance().clear();
    DispatchRegistry::instance().clear();
    print_summary_once();

    std::lock_guard lock(mutex_);
    clear_unlocked();
  }

#define WAITCHECK_CORE_GETTER(name, field, wrapper, type, install_if)                              \
  [[nodiscard]] type name() const {                                                                \
    std::lock_guard lock(mutex_);                                                                  \
    return name##_.original();                                                                     \
  }
  WAITCHECK_HSA_CORE_FUNCTIONS(WAITCHECK_CORE_GETTER)
#undef WAITCHECK_CORE_GETTER
  void track_queue(hsa_queue_t *queue) {
    std::lock_guard lock(mutex_);
    queues_.insert(queue);
  }
  void untrack_queue(hsa_queue_t *queue) {
    std::lock_guard lock(mutex_);
    queues_.erase(queue);
  }
  hsa_status_t queue_info(hsa_queue_t *queue, hsa_queue_info_attribute_t attribute, void *value) {
    decltype(hsa_amd_queue_get_info) *original;
    {
      std::lock_guard lock(mutex_);
      if (attribute == HSA_AMD_QUEUE_INFO_DOORBELL_ID && queues_.contains(queue))
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      original = queue_info_.original();
    }
    return original ? original(queue, attribute, value) : HSA_STATUS_ERROR_NOT_INITIALIZED;
  }
  [[nodiscard]] decltype(AmdExtTable::hsa_amd_queue_create_fn) amd_queue_create() const {
    std::lock_guard lock(mutex_);
    return amd_queue_create_.original();
  }
  [[nodiscard]] decltype(AmdExtTable::hsa_amd_queue_set_priority_fn) queue_set_priority() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_set_priority_fn;
  }
  [[nodiscard]] decltype(AmdExtTable::hsa_amd_queue_cu_set_mask_fn) queue_cu_set_mask() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_cu_set_mask_fn;
  }
  [[nodiscard]] decltype(AmdExtTable::hsa_amd_queue_intercept_create_fn) intercept_create() const {
    std::lock_guard lock(mutex_);
    return intercept_create_.original();
  }
  [[nodiscard]] decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn)
  intercept_register() const {
    std::lock_guard lock(mutex_);
    return intercept_register_.original();
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
#define WAITCHECK_CLEAR_CORE(name, field, wrapper, type, install_if) name##_.clear();
    WAITCHECK_HSA_CORE_FUNCTIONS(WAITCHECK_CLEAR_CORE)
#undef WAITCHECK_CLEAR_CORE
    amd_queue_create_.clear();
    queues_.clear();
    queue_info_.clear();
    intercept_create_.clear();
    intercept_register_.clear();
    original_loaded_code_object_get_info_ = nullptr;
    original_file_range_create_ = nullptr;
    mode_ = RuntimeMode::Eager;
  }

  std::unordered_set<hsa_queue_t *> queues_;
  rocjitsu::hooks::HsaApiFunctionPatch<decltype(hsa_amd_queue_get_info) *> queue_info_;
  mutable std::mutex mutex_;
  bool active_ = false;
  CoreApiTable *core_ = nullptr;
  AmdExtTable *amd_ext_ = nullptr;
#define WAITCHECK_DECLARE_CORE(name, field, wrapper, type, install_if)                             \
  rocjitsu::hooks::HsaApiFunctionPatch<type> name##_;
  WAITCHECK_HSA_CORE_FUNCTIONS(WAITCHECK_DECLARE_CORE)
#undef WAITCHECK_DECLARE_CORE
  rocjitsu::hooks::HsaApiFunctionPatch<decltype(AmdExtTable::hsa_amd_queue_create_fn)>
      amd_queue_create_;
  rocjitsu::hooks::HsaApiFunctionPatch<decltype(AmdExtTable::hsa_amd_queue_intercept_create_fn)>
      intercept_create_;
  rocjitsu::hooks::HsaApiFunctionPatch<decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn)>
      intercept_register_;
  LoadedCodeObjectGetInfoFn original_loaded_code_object_get_info_ = nullptr;
  CreateFromFileWithOffsetSizeFn original_file_range_create_ = nullptr;
  RuntimeMode mode_ = RuntimeMode::Eager;
};

#undef WAITCHECK_HSA_CORE_FUNCTIONS

WaitcheckHsaLayer &layer() {
  static WaitcheckHsaLayer *state = new WaitcheckHsaLayer();
  return *state;
}

[[nodiscard]] LoadedCodeObjectGetInfoFn resolve_loaded_code_object_get_info() {
  if (auto *get_info = layer().loaded_code_object_get_info(); get_info != nullptr)
    return get_info;
  AmdLoaderTablePrefix loader{};
  auto *get_table = layer().get_major_extension_table();
  if (get_table == nullptr ||
      get_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader) != HSA_STATUS_SUCCESS)
    return nullptr;
  layer().save_loaded_code_object_get_info(loader.loaded_code_object_get_info);
  return loader.loaded_code_object_get_info;
}

enum class LoadedImageIdentity { Same, Changed, Unavailable };

[[nodiscard]] LoadedImageIdentity loaded_image_identity(hsa_loaded_code_object_t loaded,
                                                        const uint8_t *expected, size_t size) {
  auto *get_info = resolve_loaded_code_object_get_info();
  if (get_info == nullptr || expected == nullptr || size == 0)
    return LoadedImageIdentity::Unavailable;
  uint32_t storage_type = 0;
  if (get_info(loaded, kLoadedCodeObjectInfoStorageType, &storage_type) != HSA_STATUS_SUCCESS)
    return LoadedImageIdentity::Unavailable;

  if (storage_type == 2) { // HSA memory reader
    uint64_t base = 0;
    uint64_t loaded_size = 0;
    if (get_info(loaded, kLoadedCodeObjectInfoMemoryBase, &base) != HSA_STATUS_SUCCESS ||
        get_info(loaded, kLoadedCodeObjectInfoMemorySize, &loaded_size) != HSA_STATUS_SUCCESS ||
        base == 0)
      return LoadedImageIdentity::Unavailable;
    if (loaded_size != size)
      return LoadedImageIdentity::Changed;
    const auto *bytes = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(base));
    return std::memcmp(bytes, expected, size) == 0 ? LoadedImageIdentity::Same
                                                   : LoadedImageIdentity::Changed;
  }
  if (storage_type == 1) { // HSA file reader
    int file = -1;
    if (get_info(loaded, kLoadedCodeObjectInfoFile, &file) != HSA_STATUS_SUCCESS || file < 0)
      return LoadedImageIdentity::Unavailable;
    struct stat status {};
    if (fstat(file, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0 ||
        static_cast<uintmax_t>(status.st_size) > std::numeric_limits<size_t>::max())
      return LoadedImageIdentity::Unavailable;
    size_t offset = 0;
    size_t loaded_size = static_cast<size_t>(status.st_size);
    uint32_t uri_length = 0;
    if (get_info(loaded, kLoadedCodeObjectInfoUriLength, &uri_length) != HSA_STATUS_SUCCESS ||
        uri_length > 8192)
      return LoadedImageIdentity::Unavailable;
    std::string uri(uri_length + 1, '\0');
    if (get_info(loaded, kLoadedCodeObjectInfoUri, uri.data()) != HSA_STATUS_SUCCESS ||
        uri[uri_length] != '\0')
      return LoadedImageIdentity::Unavailable;
    uri.resize(uri_length);
    size_t range = uri.find("#offset=");
    if (range == std::string::npos)
      range = uri.find("?offset=");
    if (range != std::string::npos) {
      const char *start = uri.c_str() + range + 8;
      char *end = nullptr;
      errno = 0;
      const unsigned long long parsed_offset = std::strtoull(start, &end, 0);
      if (errno != 0 || end == start || std::strncmp(end, "&size=", 6) != 0 ||
          parsed_offset > std::numeric_limits<size_t>::max())
        return LoadedImageIdentity::Unavailable;
      start = end + 6;
      errno = 0;
      const unsigned long long parsed_size = std::strtoull(start, &end, 0);
      if (errno != 0 || end == start || *end != '\0' ||
          parsed_size > std::numeric_limits<size_t>::max())
        return LoadedImageIdentity::Unavailable;
      offset = static_cast<size_t>(parsed_offset);
      loaded_size = static_cast<size_t>(parsed_size);
    }
    if (loaded_size != size)
      return LoadedImageIdentity::Changed;
    const auto bytes = rocjitsu::snapshot_code_object_file_range(file, offset, size);
    if (!bytes)
      return LoadedImageIdentity::Unavailable;
    return bytes->size() == size && std::memcmp(bytes->data(), expected, size) == 0
               ? LoadedImageIdentity::Same
               : LoadedImageIdentity::Changed;
  }
  return LoadedImageIdentity::Unavailable;
}

[[nodiscard]] hsa_status_t unchecked_loaded_image_status(LoadedImageIdentity identity) {
  g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
  std::fprintf(stderr, "rocjitsu-waitcheck: loaded code object image %s; check unavailable\n",
               identity == LoadedImageIdentity::Changed ? "differs from reader" : "is unavailable");
  return env_enabled("ROCJITSU_WAITCHECK_FAIL", false) ? HSA_STATUS_ERROR_INVALID_CODE_OBJECT
                                                        : HSA_STATUS_SUCCESS;
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
  rocjitsu::CodeObjectFileSnapshot bytes;
  if (env_enabled("ROCJITSU_WAITCHECK", true))
    bytes = rocjitsu::snapshot_code_object_file(file);
  const hsa_status_t status = original(file, reader);
  if (status == HSA_STATUS_SUCCESS && reader != nullptr && bytes)
    ReaderRegistry::instance().store_owned(*reader, std::move(bytes));
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
  rocjitsu::CodeObjectFileSnapshot bytes;
  if (env_enabled("ROCJITSU_WAITCHECK", true))
    bytes = rocjitsu::snapshot_code_object_file_range(file, offset, size);
  const hsa_status_t status = original(file, offset, size, reader);
  if (status == HSA_STATUS_SUCCESS && reader != nullptr && bytes)
    ReaderRegistry::instance().store_owned(*reader, std::move(bytes));
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
  auto entry = ReaderRegistry::instance().lookup(reader);
  if (layer().mode() == RuntimeMode::Dispatch) {
    if (!env_enabled("ROCJITSU_WAITCHECK", true))
      return original(executable, agent, reader, options, loaded_code_object);

    hsa_loaded_code_object_t captured_loaded{};
    hsa_loaded_code_object_t *runtime_output =
        loaded_code_object == nullptr ? &captured_loaded : loaded_code_object;
    if (entry == nullptr) {
      const hsa_status_t status = original(executable, agent, reader, options, runtime_output);
      if (status == HSA_STATUS_SUCCESS)
        g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      return status;
    }

    // The HSA contract keeps the reader and its backing memory alive through
    // executable destruction. Prepare directly from it before the real load;
    // AmdGpuCodeObject takes the persistent copy needed for lazy dispatch.
    std::lock_guard entry_lock(entry->mutex);
    if (entry->bytes == nullptr || entry->size == 0) {
      const hsa_status_t status = original(executable, agent, reader, options, runtime_output);
      if (status == HSA_STATUS_SUCCESS)
        g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
      return status;
    }

    auto &registry = DispatchRegistry::instance();
    const auto prepared = registry.prepare(executable, agent, entry->bytes, entry->size);
    if (!prepared.passed && env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
      registry.complete(prepared, {}, false);
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    const hsa_status_t status = original(executable, agent, reader, options, runtime_output);
    if (status != HSA_STATUS_SUCCESS) {
      registry.complete(prepared, {}, false);
      return status;
    }
    const auto identity = loaded_image_identity(*runtime_output, entry->bytes, entry->size);
    if (identity != LoadedImageIdentity::Same) {
      registry.complete(prepared, {}, false);
      return unchecked_loaded_image_status(identity);
    }
    registry.complete(prepared, *runtime_output, true);
    return status;
  }

  hsa_loaded_code_object_t captured_loaded{};
  hsa_loaded_code_object_t *runtime_output =
      loaded_code_object == nullptr ? &captured_loaded : loaded_code_object;
  if (entry != nullptr) {
    std::lock_guard entry_lock(entry->mutex);
    if (!entry->analyzed) {
      entry->result = check_code_object(entry->bytes, entry->size);
      entry->analyzed = true;
    }
    if (entry->result.checked && !entry->result.passed &&
        env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    const hsa_status_t status = original(executable, agent, reader, options, runtime_output);
    if (status != HSA_STATUS_SUCCESS || !env_enabled("ROCJITSU_WAITCHECK", true))
      return status;
    const auto identity = loaded_image_identity(*runtime_output, entry->bytes, entry->size);
    return identity == LoadedImageIdentity::Same ? status : unchecked_loaded_image_status(identity);
  } else if (env_enabled("ROCJITSU_WAITCHECK", true)) {
    g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
  }
  return original(executable, agent, reader, options, runtime_output);
}

hsa_status_t HSA_API waitcheck_executable_freeze(hsa_executable_t executable, const char *options) {
  auto *original = layer().freeze();
  if (original == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status = original(executable, options);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  LoadedCodeObjectGetInfoFn get_info = resolve_loaded_code_object_get_info();
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
                               hsa_amd_queue_intercept_packet_writer writer) {
  static_assert(sizeof(hsa_kernel_dispatch_packet_t) == 64);
  if (packets == nullptr || writer == nullptr)
    return;
  if (!env_enabled("ROCJITSU_WAITCHECK", true)) {
    writer(packets, packet_count);
    return;
  }

  const auto *packet_bytes = static_cast<const uint8_t *>(packets);
  constexpr uint16_t type_mask = static_cast<uint16_t>((1U << HSA_PACKET_HEADER_WIDTH_TYPE) - 1U);
  for (uint64_t i = 0; i < packet_count; ++i) {
    const auto *packet = reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(
        packet_bytes + i * sizeof(hsa_kernel_dispatch_packet_t));
    const uint16_t type =
        static_cast<uint16_t>((packet->header >> HSA_PACKET_HEADER_TYPE) & type_mask);
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      if (type != HSA_PACKET_TYPE_VENDOR_SPECIFIC)
        continue;
      const auto format = packet_bytes[i * sizeof(*packet) + sizeof(uint16_t)];
      // Format 4 is the long-dispatch form of format 3. Older installed
      // headers only name format 3; both carry kernel_object at byte 32.
      constexpr uint8_t kExtendedLongDispatch = 4;
      if (format != HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH && format != kExtendedLongDispatch)
        continue;
    }
    static_assert(offsetof(hsa_amd_ext_kernel_dispatch_packet_t, kernel_object) ==
                  offsetof(hsa_kernel_dispatch_packet_t, kernel_object));

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

hsa_status_t HSA_API waitcheck_queue_destroy(hsa_queue_t *queue) {
  auto *destroy = layer().queue_destroy();
  if (!destroy)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const auto status = destroy(queue);
  if (status == HSA_STATUS_SUCCESS) {
    layer().untrack_queue(queue);
    DispatchRegistry::instance().release_cooperative_queue(queue);
  }
  return status;
}

hsa_status_t HSA_API waitcheck_queue_get_info(hsa_queue_t *queue,
                                              hsa_queue_info_attribute_t attribute, void *value) {
  return layer().queue_info(queue, attribute, value);
}

hsa_status_t HSA_API waitcheck_queue_create(hsa_agent_t agent, uint32_t size,
                                            hsa_queue_type32_t type,
                                            void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                            void *data, uint32_t private_segment_size,
                                            uint32_t group_segment_size, hsa_queue_t **queue) {
  if (!env_enabled("ROCJITSU_WAITCHECK", true)) {
    auto *create = layer().queue_create();
    return create == nullptr ? HSA_STATUS_ERROR_NOT_INITIALIZED
                             : create(agent, size, type, callback, data, private_segment_size,
                                      group_segment_size, queue);
  }
  if (type == HSA_QUEUE_TYPE_COOPERATIVE) {
    auto &registry = DispatchRegistry::instance();
    auto create = layer().queue_create();
    if (!create)
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    const bool checking = env_enabled("ROCJITSU_WAITCHECK", true);
    if (!registry.begin_cooperative_create(agent, checking) &&
        env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
      registry.finish_cooperative_create(agent, nullptr);
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    const auto status =
        create(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
    registry.finish_cooperative_create(agent,
                                       status == HSA_STATUS_SUCCESS && queue ? *queue : nullptr);
    return status;
  }
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
  if (register_status == HSA_STATUS_SUCCESS) {
    layer().track_queue(*queue);
    return HSA_STATUS_SUCCESS;
  }

  if (auto *destroy = layer().queue_destroy(); destroy != nullptr)
    (void)destroy(*queue);
  *queue = nullptr;
  return register_status;
}

hsa_status_t HSA_API waitcheck_amd_queue_intercept_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue) {
  if (!env_enabled("ROCJITSU_WAITCHECK", true)) {
    auto *create = layer().intercept_create();
    return create == nullptr ? HSA_STATUS_ERROR_NOT_INITIALIZED
                             : create(agent, size, type, callback, data, private_segment_size,
                                      group_segment_size, queue);
  }
  return waitcheck_queue_create(agent, size, type, callback, data, private_segment_size,
                                group_segment_size, queue);
}

hsa_status_t HSA_API waitcheck_amd_queue_create(hsa_agent_t agent,
                                                hsa_amd_queue_create_desc_t *descs,
                                                uint32_t num_descs) {
  auto *original_create = layer().amd_queue_create();
  if (original_create == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!env_enabled("ROCJITSU_WAITCHECK", true))
    return original_create(agent, descs, num_descs);

  auto &registry = DispatchRegistry::instance();
  const auto cooperative = [](const hsa_amd_queue_create_desc_t &desc) {
    return desc.engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE &&
           desc.engine.compute.type == HSA_QUEUE_TYPE_COOPERATIVE;
  };
  constexpr uint16_t kDevicePlacement = HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF |
                                         HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR;
  const bool checking = env_enabled("ROCJITSU_WAITCHECK", true);
  // This API reports the first error while preserving queues created for other
  // descriptors. A fail-closed preflight may reject just one descriptor, so
  // process the batch in order before making any preflight decision for another.
  if (descs != nullptr && num_descs > 1 && checking &&
      env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
    hsa_status_t first_error = HSA_STATUS_SUCCESS;
    for (uint32_t i = 0; i < num_descs; ++i) {
      const hsa_status_t status = waitcheck_amd_queue_create(agent, &descs[i], 1);
      if (first_error == HSA_STATUS_SUCCESS && status != HSA_STATUS_SUCCESS)
        first_error = status;
    }
    return first_error;
  }
  if (descs != nullptr && checking && env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
    for (uint32_t i = 0; i < num_descs; ++i) {
      if (descs[i].engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE || cooperative(descs[i]) ||
          (descs[i].flags & kDevicePlacement) == 0)
        continue;
      for (uint32_t n = 0; n < num_descs; ++n)
        descs[n].queue = nullptr;
      return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
    }
  }
  for (uint32_t i = 0; descs != nullptr && i < num_descs; ++i)
    descs[i].queue = nullptr;
  uint32_t pending = 0;
  for (uint32_t i = 0; descs && i < num_descs; ++i) {
    if (!cooperative(descs[i]))
      continue;
    ++pending;
    if (!registry.begin_cooperative_create(agent, checking) &&
        env_enabled("ROCJITSU_WAITCHECK_FAIL", false)) {
      for (uint32_t n = 0; n < num_descs; ++n)
        descs[n].queue = nullptr;
      for (uint32_t n = 0; n < pending; ++n)
        registry.finish_cooperative_create(agent, nullptr);
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
  }
  const hsa_status_t status = original_create(agent, descs, num_descs);
  for (uint32_t i = 0; descs && i < num_descs; ++i) {
    if (cooperative(descs[i]))
      registry.finish_cooperative_create(agent, descs[i].queue);
  }
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
    if (plain_queue == nullptr || desc.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE ||
        desc.engine.compute.type == HSA_QUEUE_TYPE_COOPERATIVE)
      continue;
    if (desc.flags & kDevicePlacement) {
      // An intercept queue has a host staging ring. Keep the original queue so
      // CLR's device-wide signal ordering assumption remains valid.
      if (checking) {
        g_stats.unavailable.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr,
                     "rocjitsu-waitcheck: device-memory queue cannot be intercepted; "
                     "dispatch checking unavailable for this queue\n");
      }
      continue;
    }

    // A preceding HSA tool may already have replaced this descriptor queue
    // with an intercept queue. Register on that queue so its packet callbacks
    // and queue identity remain intact.
    auto *register_intercept = layer().intercept_register();
    const hsa_status_t reuse_status =
        register_intercept == nullptr
            ? HSA_STATUS_ERROR_NOT_INITIALIZED
            : register_intercept(plain_queue, waitcheck_queue_intercept, nullptr);
    if (reuse_status == HSA_STATUS_SUCCESS) {
      layer().track_queue(plain_queue);
      continue;
    }
    if (reuse_status != HSA_STATUS_ERROR_INVALID_QUEUE) {
      (void)destroy_queue(plain_queue);
      desc.queue = nullptr;
      note_replacement_error(reuse_status);
      continue;
    }

    // A plain descriptor queue cannot accept a ROCr packet callback. Release
    // its hardware slot before creating the intercept queue; some agents have
    // only one free slot for this descriptor.
    const hsa_status_t destroy_status = destroy_queue(plain_queue);
    desc.queue = nullptr;
    if (destroy_status != HSA_STATUS_SUCCESS) {
      note_replacement_error(destroy_status);
      continue;
    }

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
      desc.queue = intercept_queue;
      continue;
    }

    if (intercept_queue != nullptr)
      (void)waitcheck_queue_destroy(intercept_queue);
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
