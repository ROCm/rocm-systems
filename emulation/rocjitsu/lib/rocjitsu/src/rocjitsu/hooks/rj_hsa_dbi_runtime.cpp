// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbi_runtime.cpp
/// @brief HSA tools hook that instruments code objects as ROCr loads them.
///
/// @details Loaded through `HSA_TOOLS_LIB`. While a probe is set, every code
/// object loaded with hsa_executable_load_agent_code_object() goes through
/// instrument_each_block(), and the patched object is loaded in place of the
/// original. An object that is not instrumented, or whose reader bytes could
/// not be read, loads unchanged and is counted under the reason. Nothing happens
/// at dispatch.
///
/// Only the testing variant can set a probe (rj_test_dbi_set_probe()), so the
/// production library currently instruments nothing. Loads through
/// hsa_executable_load_program_code_object() and the deprecated
/// hsa_executable_load_code_object() are not intercepted and never instrumented.
///
/// Reader capture, the API-table snapshot and the owned-bytes load follow the
/// gfx1250 hotswap hook (gfx1250_b0_a0_hotswap.cpp).

#include "hsa/hsa_api_trace.h"
#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/hooks/dbi_runtime_sites.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iterator>
#include <memory>
#include <mutex>
#include <ranges>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::hooks {
namespace {

using Blob = std::shared_ptr<const std::vector<uint8_t>>;
using VendorReaderCreate = hsa_status_t (*)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *);

// The AMD loader extension table up to the reader constructor this hook wraps.
struct VendorLoaderTable {
  void (*query_host_address)();
  void (*query_segment_descriptors)();
  void (*query_executable)();
  void (*iterate_loaded_code_objects)();
  void (*loaded_code_object_get_info)();
  VendorReaderCreate create_reader_from_file;
};
static_assert(offsetof(VendorLoaderTable, create_reader_from_file) == 5 * sizeof(void (*)()));

void log_error(const char *format, ...) noexcept {
  va_list args;
  va_start(args, format);
  flockfile(stderr);
  std::fputs("[rocjitsu-dbi] error: ", stderr);
  std::vfprintf(stderr, format, args);
  std::fputc('\n', stderr);
  funlockfile(stderr);
  va_end(args);
}

template <typename Fn> hsa_status_t hsa_boundary(const char *operation, Fn &&fn) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc &) {
    log_error("operation=%s exception=std::bad_alloc", operation);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  } catch (const std::exception &error) {
    log_error("operation=%s exception=std::exception what=%s", operation, error.what());
    return HSA_STATUS_ERROR;
  } catch (...) {
    log_error("operation=%s exception=unknown", operation);
    return HSA_STATUS_ERROR;
  }
}

hsa_status_t HSA_API reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t *reader);
hsa_status_t HSA_API reader_create_from_memory(const void *code_object, size_t size,
                                               hsa_code_object_reader_t *reader);
hsa_status_t HSA_API reader_destroy(hsa_code_object_reader_t reader);
hsa_status_t HSA_API executable_destroy(hsa_executable_t executable);
hsa_status_t HSA_API load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                            hsa_code_object_reader_t reader, const char *options,
                                            hsa_loaded_code_object_t *loaded);
hsa_status_t HSA_API system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                      size_t table_length, void *table);

struct OriginalApi {
  decltype(hsa_code_object_reader_create_from_file) *create_file = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *create_memory = nullptr;
  decltype(hsa_code_object_reader_destroy) *destroy_reader = nullptr;
  decltype(hsa_executable_destroy) *destroy_executable = nullptr;
  decltype(hsa_executable_load_agent_code_object) *load_agent = nullptr;
  decltype(hsa_system_get_major_extension_table) *get_extension_table = nullptr;
};

// The probe and the object it resolves in, replaced as a unit.
struct LoadedProbe {
  AmdGpuCodeObject object;
  std::string symbol;
};

struct LoadCounters {
  std::atomic<uint64_t> loads{0};
  std::atomic<uint64_t> unknown_readers{0};
  std::array<std::atomic<uint64_t>, kDbiLoadDecisionCount> decisions{};
  std::atomic<uint64_t> sites{0};
};

struct HookState {
  std::mutex lifecycle_mutex;
  CoreApiTable *core = nullptr;
  // Immutable snapshots: a callback keeps the one it loaded alive for its whole
  // call, so a concurrent reinstall or probe change never mutates it underneath.
  std::atomic<std::shared_ptr<const OriginalApi>> active_api{nullptr};
  std::atomic<std::shared_ptr<const LoadedProbe>> probe{nullptr};
  std::atomic<VendorReaderCreate> vendor_reader{nullptr};

  std::mutex storage_mutex;
  std::unordered_map<uint64_t, Blob> readers;
  std::unordered_map<uint64_t, std::vector<Blob>> executables;

  LoadCounters counters;
};

// Never destructed, so loaded bytes outlive static destructors that may run
// before a profiler's exit-time finalizer reads them. The hotswap hook's
// g_state documents the case.
HookState &g_state = *new HookState();

void count(std::atomic<uint64_t> &counter, uint64_t amount = 1) noexcept {
  counter.fetch_add(amount, std::memory_order_relaxed);
}

void store_reader(hsa_code_object_reader_t reader, Blob bytes) {
  std::lock_guard lock(g_state.storage_mutex);
  g_state.readers[reader.handle] = std::move(bytes);
}

Blob lookup_reader(hsa_code_object_reader_t reader) {
  std::lock_guard lock(g_state.storage_mutex);
  const auto it = g_state.readers.find(reader.handle);
  return it == g_state.readers.end() ? nullptr : it->second;
}

Blob take_reader(hsa_code_object_reader_t reader) {
  std::lock_guard lock(g_state.storage_mutex);
  auto node = g_state.readers.extract(reader.handle);
  return node.empty() ? nullptr : std::move(node.mapped());
}

void retain(hsa_executable_t executable, Blob bytes) {
  std::lock_guard lock(g_state.storage_mutex);
  g_state.executables[executable.handle].push_back(std::move(bytes));
}

void unretain(hsa_executable_t executable, const Blob &bytes) {
  std::lock_guard lock(g_state.storage_mutex);
  const auto map_it = g_state.executables.find(executable.handle);
  if (map_it == g_state.executables.end())
    return;
  std::vector<Blob> &buffers = map_it->second;
  const auto found = std::ranges::find(buffers, bytes);
  if (found != buffers.end())
    buffers.erase(found);
  if (buffers.empty())
    g_state.executables.erase(map_it);
}

std::vector<Blob> take_retained(hsa_executable_t executable) {
  std::lock_guard lock(g_state.storage_mutex);
  auto node = g_state.executables.extract(executable.handle);
  return node.empty() ? std::vector<Blob>{} : std::move(node.mapped());
}

void restore_retained(hsa_executable_t executable, std::vector<Blob> buffers) {
  if (buffers.empty())
    return;
  std::lock_guard lock(g_state.storage_mutex);
  std::vector<Blob> &retained = g_state.executables[executable.handle];
  retained.insert(retained.end(), std::make_move_iterator(buffers.begin()),
                  std::make_move_iterator(buffers.end()));
}

bool install(HsaApiTable *table) {
  std::lock_guard lock(g_state.lifecycle_mutex);
  if (g_state.core != nullptr) {
    log_error("OnLoad refused: hook is already installed");
    return false;
  }
  if (table == nullptr || table->core_ == nullptr) {
    log_error("OnLoad refused: missing HSA API table");
    return false;
  }

  CoreApiTable *core = table->core_;
  constexpr size_t required_size =
      offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
      sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn);
  if (core->version.minor_id < required_size) {
    log_error("OnLoad refused: HSA core API table is too small size=%u required=%zu",
              core->version.minor_id, required_size);
    return false;
  }

  const OriginalApi original{
      core->hsa_code_object_reader_create_from_file_fn,
      core->hsa_code_object_reader_create_from_memory_fn,
      core->hsa_code_object_reader_destroy_fn,
      core->hsa_executable_destroy_fn,
      core->hsa_executable_load_agent_code_object_fn,
      core->hsa_system_get_major_extension_table_fn,
  };
  if (original.create_file == nullptr || original.create_memory == nullptr ||
      original.destroy_reader == nullptr || original.destroy_executable == nullptr ||
      original.load_agent == nullptr || original.get_extension_table == nullptr) {
    log_error("OnLoad refused: HSA core API table has a missing required entry");
    return false;
  }

  // The snapshot is the only fallible step, so it comes before any state is
  // committed: a failed install must leave the hook installable. Bytes retained
  // by an earlier runtime generation are kept, because a profiler record from
  // that generation may still point into them.
  auto snapshot = std::make_shared<const OriginalApi>(original);

  g_state.vendor_reader.store(nullptr, std::memory_order_release);
  g_state.core = core;
  g_state.active_api.store(std::move(snapshot), std::memory_order_release);
  core->hsa_code_object_reader_create_from_file_fn = reader_create_from_file;
  core->hsa_code_object_reader_create_from_memory_fn = reader_create_from_memory;
  core->hsa_code_object_reader_destroy_fn = reader_destroy;
  core->hsa_executable_destroy_fn = executable_destroy;
  core->hsa_executable_load_agent_code_object_fn = load_agent_code_object;
  core->hsa_system_get_major_extension_table_fn = system_get_major_extension_table;
  return true;
}

void uninstall() {
  std::lock_guard lock(g_state.lifecycle_mutex);
  CoreApiTable *core = g_state.core;
  const std::shared_ptr<const OriginalApi> original =
      g_state.active_api.exchange(nullptr, std::memory_order_acq_rel);
  if (core != nullptr && original != nullptr) {
    if (core->hsa_code_object_reader_create_from_file_fn == reader_create_from_file)
      core->hsa_code_object_reader_create_from_file_fn = original->create_file;
    if (core->hsa_code_object_reader_create_from_memory_fn == reader_create_from_memory)
      core->hsa_code_object_reader_create_from_memory_fn = original->create_memory;
    if (core->hsa_code_object_reader_destroy_fn == reader_destroy)
      core->hsa_code_object_reader_destroy_fn = original->destroy_reader;
    if (core->hsa_executable_destroy_fn == executable_destroy)
      core->hsa_executable_destroy_fn = original->destroy_executable;
    if (core->hsa_executable_load_agent_code_object_fn == load_agent_code_object)
      core->hsa_executable_load_agent_code_object_fn = original->load_agent;
    if (core->hsa_system_get_major_extension_table_fn == system_get_major_extension_table)
      core->hsa_system_get_major_extension_table_fn = original->get_extension_table;
  }
  // ROCr destroys its loader after OnUnload, so loaded bytes stay retained.
  g_state.vendor_reader.store(nullptr, std::memory_order_release);
  g_state.core = nullptr;
}

Blob copy_bytes(const void *data, size_t size) {
  if (data == nullptr || size == 0)
    return nullptr;
  const auto *begin = static_cast<const uint8_t *>(data);
  return std::make_shared<const std::vector<uint8_t>>(begin, begin + size);
}

// Reads @p requested_size bytes at @p offset, or the rest of the file when 0.
Blob read_file_region(hsa_file_t file, size_t offset, size_t requested_size) {
  if (file < 0)
    return nullptr;
  struct stat file_info {};
  if (fstat(file, &file_info) != 0 || file_info.st_size <= 0)
    return nullptr;
  const size_t file_size = static_cast<size_t>(file_info.st_size);
  if (offset > file_size)
    return nullptr;
  const size_t remaining = file_size - offset;
  const size_t size = requested_size == 0 ? remaining : requested_size;
  if (size == 0 || size > remaining)
    return nullptr;

  std::shared_ptr<std::vector<uint8_t>> bytes = std::make_shared<std::vector<uint8_t>>(size);
  size_t done = 0;
  while (done < size) {
    const ssize_t read_count =
        pread(file, bytes->data() + done, size - done, static_cast<off_t>(offset + done));
    if (read_count < 0 && errno == EINTR)
      continue;
    if (read_count <= 0)
      return nullptr;
    done += static_cast<size_t>(read_count);
  }
  return bytes;
}

// A reader whose bytes cannot be captured still works: its loads go through
// unchanged and are counted as unknown readers. Any bytes already stored under
// its handle belong to an earlier reader whose destroy the hook never saw, such
// as one that outlived OnUnload, so they are dropped.
void capture_reader(hsa_code_object_reader_t reader, Blob bytes) {
  if (bytes != nullptr)
    store_reader(reader, std::move(bytes));
  else
    take_reader(reader);
}

std::string join_errors(const std::vector<std::string> &errors) {
  std::string joined;
  for (const std::string &error : errors) {
    if (!joined.empty())
      joined += "; ";
    joined += error;
  }
  return joined;
}

hsa_status_t load_owned_bytes(hsa_executable_t executable, hsa_agent_t agent, const Blob &bytes,
                              const char *options, hsa_loaded_code_object_t *loaded,
                              const OriginalApi &api) {
  hsa_code_object_reader_t owned_reader{};
  const hsa_status_t reader_status = api.create_memory(bytes->data(), bytes->size(), &owned_reader);
  if (reader_status != HSA_STATUS_SUCCESS)
    return reader_status;

  // ROCr aliases a memory reader's bytes into the loaded code object, so they
  // must live until the executable is destroyed. Reserve their slot before the
  // load so a successful load can never be followed by a failed retain.
  retain(executable, bytes);
  const hsa_status_t status = api.load_agent(executable, agent, owned_reader, options, loaded);
  if (status != HSA_STATUS_SUCCESS)
    unretain(executable, bytes);
  (void)api.destroy_reader(owned_reader);
  return status;
}

hsa_status_t HSA_API reader_create_from_memory(const void *code_object, size_t size,
                                               hsa_code_object_reader_t *reader) {
  return hsa_boundary("hsa_code_object_reader_create_from_memory", [&] {
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    const hsa_status_t status = api->create_memory(code_object, size, reader);
    if (status == HSA_STATUS_SUCCESS)
      capture_reader(*reader, copy_bytes(code_object, size));
    return status;
  });
}

hsa_status_t HSA_API reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t *reader) {
  return hsa_boundary("hsa_code_object_reader_create_from_file", [&] {
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    const hsa_status_t status = api->create_file(file, reader);
    if (status == HSA_STATUS_SUCCESS)
      capture_reader(*reader, read_file_region(file, 0, 0));
    return status;
  });
}

hsa_status_t vendor_reader_create(hsa_file_t file, size_t offset, size_t size,
                                  hsa_code_object_reader_t *reader) {
  return hsa_boundary("amd_loader_code_object_reader_create_from_file", [&] {
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const VendorReaderCreate original = g_state.vendor_reader.load(std::memory_order_acquire);
    if (original == nullptr)
      return HSA_STATUS_ERROR;
    const hsa_status_t status = original(file, offset, size, reader);
    if (status == HSA_STATUS_SUCCESS)
      capture_reader(*reader, read_file_region(file, offset, size));
    return status;
  });
}

// HIP creates its file-backed readers through the AMD loader extension rather
// than the core table, so the extension's reader constructor is wrapped too.
hsa_status_t HSA_API system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                      size_t table_length, void *table) {
  return hsa_boundary("hsa_system_get_major_extension_table", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    const hsa_status_t status =
        api->get_extension_table(extension, version_major, table_length, table);
    constexpr size_t reader_field_end =
        offsetof(VendorLoaderTable, create_reader_from_file) + sizeof(VendorReaderCreate);
    if (status != HSA_STATUS_SUCCESS || table == nullptr || extension != HSA_EXTENSION_AMD_LOADER ||
        version_major != 1 || table_length < reader_field_end)
      return status;

    auto *loader = static_cast<VendorLoaderTable *>(table);
    const VendorReaderCreate reader = loader->create_reader_from_file;
    if (reader != nullptr) {
      VendorReaderCreate expected = nullptr;
      g_state.vendor_reader.compare_exchange_strong(expected, reader, std::memory_order_acq_rel);
      loader->create_reader_from_file = vendor_reader_create;
    }
    return status;
  });
}

hsa_status_t HSA_API reader_destroy(hsa_code_object_reader_t reader) {
  return hsa_boundary("hsa_code_object_reader_destroy", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    // Taken out first, so a reader created with the same handle after the
    // destroy keeps its own bytes.
    Blob bytes = take_reader(reader);
    const hsa_status_t status = api->destroy_reader(reader);
    if (status != HSA_STATUS_SUCCESS && bytes != nullptr)
      store_reader(reader, std::move(bytes));
    return status;
  });
}

hsa_status_t HSA_API executable_destroy(hsa_executable_t executable) {
  return hsa_boundary("hsa_executable_destroy", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    // Taken out before the destroy and freed after it: ROCr reads the bytes
    // until it returns, and an executable created with the same handle once
    // it has must not lose its own.
    std::vector<Blob> retained = take_retained(executable);
    const hsa_status_t status = api->destroy_executable(executable);
    if (status != HSA_STATUS_SUCCESS)
      restore_retained(executable, std::move(retained));
    return status;
  });
}

hsa_status_t HSA_API load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                            hsa_code_object_reader_t reader, const char *options,
                                            hsa_loaded_code_object_t *loaded) {
  return hsa_boundary("hsa_executable_load_agent_code_object", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    count(g_state.counters.loads);
    const std::shared_ptr<const LoadedProbe> probe = g_state.probe.load(std::memory_order_acquire);
    if (probe == nullptr)
      return api->load_agent(executable, agent, reader, options, loaded);
    const Blob source = lookup_reader(reader);
    if (source == nullptr) {
      count(g_state.counters.unknown_readers);
      return api->load_agent(executable, agent, reader, options, loaded);
    }

    DbiInstrumentation result =
        instrument_each_block(*source, DbiProbe{&probe->object, probe->symbol});
    count(g_state.counters.decisions[static_cast<size_t>(result.decision)]);
    if (result.decision == DbiLoadDecision::PatchFailed)
      log_error("loading a %zu-byte code object unchanged (%s): %s", source->size(),
                dbi_load_decision_name(result.decision), join_errors(result.errors).c_str());
    if (result.decision != DbiLoadDecision::Instrumented)
      return api->load_agent(executable, agent, reader, options, loaded);

    count(g_state.counters.sites, result.sites);
    const Blob patched = std::make_shared<const std::vector<uint8_t>>(std::move(result.elf_bytes));
    const hsa_status_t status = load_owned_bytes(executable, agent, patched, options, loaded, *api);
    if (status != HSA_STATUS_SUCCESS)
      log_error("loading an instrumented code object failed status=%d", static_cast<int>(status));
    return status;
  });
}

} // namespace
} // namespace rocjitsu::hooks

extern "C" RJ_API_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                     uint64_t failed_tool_count,
                                     const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;
  try {
    return rocjitsu::hooks::install(table);
  } catch (const std::exception &error) {
    rocjitsu::hooks::log_error("OnLoad failed: exception=std::exception what=%s", error.what());
    return false;
  } catch (...) {
    rocjitsu::hooks::log_error("OnLoad failed: exception=unknown");
    return false;
  }
}

extern "C" RJ_API_EXPORT void OnUnload() {
  try {
    rocjitsu::hooks::uninstall();
  } catch (const std::exception &error) {
    rocjitsu::hooks::log_error("OnUnload failed: exception=std::exception what=%s", error.what());
  } catch (...) {
    rocjitsu::hooks::log_error("OnUnload failed: exception=unknown");
  }
}

#if defined(RJ_DBI_RUNTIME_TEST_HOOKS)
// Test-only entry points, exported by the testing variant's version script and
// compiled out of the production library.

/// Sets the probe called at every site, or clears it when @p elf is null.
/// Returns false, leaving the previous probe, when @p elf is not a code object.
extern "C" RJ_API_EXPORT bool rj_test_dbi_set_probe(const void *elf, size_t size,
                                                    const char *symbol) {
  using rocjitsu::hooks::LoadedProbe;
  if (elf == nullptr) {
    rocjitsu::hooks::g_state.probe.store(nullptr, std::memory_order_release);
    return true;
  }
  if (symbol == nullptr)
    return false;
  auto probe = std::make_shared<const LoadedProbe>(
      LoadedProbe{rocjitsu::AmdGpuCodeObject(static_cast<const uint8_t *>(elf), size), symbol});
  if (!probe->object.is_valid())
    return false;
  rocjitsu::hooks::g_state.probe.store(std::move(probe), std::memory_order_release);
  return true;
}

/// Calls to hsa_executable_load_agent_code_object() seen while installed.
extern "C" RJ_API_EXPORT uint64_t rj_test_dbi_loads() {
  return rocjitsu::hooks::g_state.counters.loads.load(std::memory_order_relaxed);
}

/// Loads whose reader the hook did not capture, which load unchanged.
extern "C" RJ_API_EXPORT uint64_t rj_test_dbi_unknown_readers() {
  return rocjitsu::hooks::g_state.counters.unknown_readers.load(std::memory_order_relaxed);
}

/// Loads that reached instrument_each_block() and got @p decision, a
/// DbiLoadDecision value.
extern "C" RJ_API_EXPORT uint64_t rj_test_dbi_decisions(int decision) {
  if (decision < 0 || static_cast<size_t>(decision) >= rocjitsu::hooks::kDbiLoadDecisionCount)
    return 0;
  return rocjitsu::hooks::g_state.counters.decisions[static_cast<size_t>(decision)].load(
      std::memory_order_relaxed);
}

/// Sites in the objects that were instrumented.
extern "C" RJ_API_EXPORT uint64_t rj_test_dbi_sites() {
  return rocjitsu::hooks::g_state.counters.sites.load(std::memory_order_relaxed);
}

/// Instrumented objects held for executables that are not yet destroyed.
extern "C" RJ_API_EXPORT size_t rj_test_dbi_retained_buffer_count() {
  rocjitsu::hooks::HookState &state = rocjitsu::hooks::g_state;
  std::lock_guard lock(state.storage_mutex);
  size_t buffers = 0;
  for (const std::vector<rocjitsu::hooks::Blob> &retained : std::views::values(state.executables))
    buffers += retained.size();
  return buffers;
}

/// Clears the probe, the counters, captured readers and retained bytes, which
/// production keeps for the life of the process.
extern "C" RJ_API_EXPORT void rj_test_dbi_reset() {
  rocjitsu::hooks::HookState &state = rocjitsu::hooks::g_state;
  state.probe.store(nullptr, std::memory_order_release);
  {
    std::lock_guard lock(state.storage_mutex);
    state.readers.clear();
    state.executables.clear();
  }
  state.counters.loads.store(0, std::memory_order_relaxed);
  state.counters.unknown_readers.store(0, std::memory_order_relaxed);
  for (std::atomic<uint64_t> &decision : state.counters.decisions)
    decision.store(0, std::memory_order_relaxed);
  state.counters.sites.store(0, std::memory_order_relaxed);
}
#endif // RJ_DBI_RUNTIME_TEST_HOOKS
