// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_a0_hotswap.cpp
/// @brief Minimal eager-only HSA hook for gfx1250 B0-to-A0 translation.

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Blob = std::shared_ptr<std::vector<uint8_t>>;
using VendorReaderCreate = hsa_status_t (*)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *);

constexpr uint32_t kAmdAgentInfoAsicRevision = 0xA012;

// Minimal mirror through the sole AMD loader entry intercepted by this hook.
struct VendorLoaderTable {
  void (*query_host_address)();
  void (*query_segment_descriptors)();
  void (*query_executable)();
  void (*iterate_loaded_code_objects)();
  void (*loaded_code_object_get_info)();
  VendorReaderCreate create_reader_from_file;
};
static_assert(offsetof(VendorLoaderTable, create_reader_from_file) == 5 * sizeof(void (*)()));

bool verbose_logging() {
  const char *value = std::getenv("HSA_HOTSWAP_VERBOSE");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void log(const char *format, ...) {
  if (!verbose_logging())
    return;
  std::fputs("[hsa-hotswap-rj] ", stderr);
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fputc('\n', stderr);
}

template <typename Fn> hsa_status_t hsa_boundary(Fn &&fn) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc &) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
}

template <typename Fn> class ScopeGuard {
public:
  explicit ScopeGuard(Fn fn) : fn_(std::move(fn)) {}
  ScopeGuard(const ScopeGuard &) = delete;
  ScopeGuard &operator=(const ScopeGuard &) = delete;
  ~ScopeGuard() noexcept {
    try {
      fn_();
    } catch (...) {
    }
  }

private:
  Fn fn_;
};

template <typename Fn> ScopeGuard<Fn> make_scope_guard(Fn fn) {
  return ScopeGuard<Fn>(std::move(fn));
}

hsa_status_t HSA_API reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t *reader);
hsa_status_t HSA_API reader_create_from_memory(const void *code_object, size_t size,
                                               hsa_code_object_reader_t *reader);
hsa_status_t HSA_API reader_destroy(hsa_code_object_reader_t reader);
hsa_status_t HSA_API executable_destroy(hsa_executable_t executable);
hsa_status_t HSA_API load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                            hsa_code_object_reader_t reader, const char *options,
                                            hsa_loaded_code_object_t *loaded);
hsa_status_t HSA_API load_program_code_object(hsa_executable_t executable,
                                              hsa_code_object_reader_t reader, const char *options,
                                              hsa_loaded_code_object_t *loaded);
hsa_status_t HSA_API load_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                      hsa_code_object_t code_object, const char *options);
hsa_status_t HSA_API system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                      size_t table_length, void *table);

struct OriginalApi {
  decltype(hsa_code_object_reader_create_from_file) *create_file = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *create_memory = nullptr;
  decltype(hsa_code_object_reader_destroy) *destroy_reader = nullptr;
  decltype(hsa_executable_destroy) *destroy_executable = nullptr;
  decltype(hsa_executable_load_agent_code_object) *load_agent = nullptr;
  decltype(hsa_executable_load_program_code_object) *load_program = nullptr;
  decltype(hsa_executable_load_code_object) *load_deprecated = nullptr;
  decltype(hsa_system_get_major_extension_table) *get_extension_table = nullptr;
  decltype(hsa_iterate_agents) *iterate_agents = nullptr;
  decltype(hsa_agent_get_info) *agent_get_info = nullptr;
  decltype(hsa_agent_iterate_isas) *agent_iterate_isas = nullptr;
  decltype(hsa_isa_get_info_alt) *isa_get_info = nullptr;
};

struct HookState {
  std::mutex lifecycle_mutex;
  CoreApiTable *core = nullptr;
  // The saved lower API is published as an IMMUTABLE, never-overwritten snapshot.
  // Each install() builds a fresh const OriginalApi on the heap and publishes it;
  // uninstall() swaps in nullptr. A hot-path callback loads the shared_ptr into a
  // local, which keeps that exact table alive for the whole call even if a
  // concurrent OnUnload/OnLoad swaps a new generation in underneath -- so a
  // reinstall can never mutate the table an in-flight callback is dereferencing.
  // (Publishing only a raw pointer to a mutated-in-place member, as before, was a
  // data race across the reinstall window.)
  std::atomic<std::shared_ptr<const OriginalApi>> active_api{nullptr};
  std::atomic<VendorReaderCreate> vendor_reader{nullptr};

  std::mutex storage_mutex;
  std::unordered_map<uint64_t, Blob> readers;
  std::unordered_map<uint64_t, std::vector<Blob>> executables;
};

// ROCr calls OnUnload before releasing this tool, so one DSO-local state owns the
// saved API and the buffers that must outlive code-object load calls.
//
// INTENTIONALLY LEAKED (never destructed): a heap object referenced by a function-
// local reference, so it has no static destructor. A plain namespace-scope object
// would run its map destructors at process exit -- and that ordering is unsafe here.
// A supported profiler startup (rocprofv3 force-configures rocprofiler and registers
// an atexit finalizer BEFORE the application's hsa_init loads this hook) means our
// state is constructed AFTER rocprofiler's. Reverse-order exit teardown would then
// destroy g_state -- freeing translated bytes -- before rocprofiler's finalizer
// processes its code-object records that alias those bytes, a use-after-free.
// RTLD_NODELETE only defers the DSO's unmap to process exit; it does not make a
// static object indestructible. Leaking the state makes the retained bytes truly
// process-lifetime: reclaimed only by the OS at exit, after all consumers are gone.
HookState &g_state = *new HookState();

bool store_reader(hsa_code_object_reader_t reader, Blob bytes) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.readers[reader.handle] = std::move(bytes);
    return true;
  } catch (...) {
    return false;
  }
}

Blob lookup_reader(hsa_code_object_reader_t reader) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    const auto it = g_state.readers.find(reader.handle);
    return it == g_state.readers.end() ? nullptr : it->second;
  } catch (...) {
    return nullptr;
  }
}

void erase_reader(hsa_code_object_reader_t reader) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.readers.erase(reader.handle);
  } catch (...) {
  }
}

bool retain(hsa_executable_t executable, Blob bytes) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.executables[executable.handle].push_back(std::move(bytes));
    return true;
  } catch (...) {
    return false;
  }
}

// Drop a single reserved blob for @p executable (matched by identity), erasing the
// map entry when its last blob is removed. Used to roll back a reservation when the
// lower load fails.
void unretain(hsa_executable_t executable, const Blob &bytes) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    const auto map_it = g_state.executables.find(executable.handle);
    if (map_it == g_state.executables.end())
      return;
    auto &buffers = map_it->second;
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
      if (it->get() == bytes.get()) {
        buffers.erase(it);
        break;
      }
    }
    if (buffers.empty())
      g_state.executables.erase(map_it);
  } catch (...) {
  }
}

void release(hsa_executable_t executable) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.executables.erase(executable.handle);
  } catch (...) {
  }
}

bool install(HsaApiTable *table) {
  std::lock_guard lock(g_state.lifecycle_mutex);
  if (g_state.core != nullptr || table == nullptr || table->core_ == nullptr)
    return false;

  CoreApiTable *core = table->core_;
  constexpr size_t required_size =
      offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
      sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn);
  if (core->version.minor_id < required_size)
    return false;

  OriginalApi original{
      core->hsa_code_object_reader_create_from_file_fn,
      core->hsa_code_object_reader_create_from_memory_fn,
      core->hsa_code_object_reader_destroy_fn,
      core->hsa_executable_destroy_fn,
      core->hsa_executable_load_agent_code_object_fn,
      core->hsa_executable_load_program_code_object_fn,
      core->hsa_executable_load_code_object_fn,
      core->hsa_system_get_major_extension_table_fn,
      core->hsa_iterate_agents_fn,
      core->hsa_agent_get_info_fn,
      core->hsa_agent_iterate_isas_fn,
      core->hsa_isa_get_info_alt_fn,
  };
  if (original.create_file == nullptr || original.create_memory == nullptr ||
      original.destroy_reader == nullptr || original.destroy_executable == nullptr ||
      original.load_agent == nullptr || original.load_program == nullptr ||
      original.load_deprecated == nullptr || original.get_extension_table == nullptr ||
      original.iterate_agents == nullptr || original.agent_get_info == nullptr ||
      original.agent_iterate_isas == nullptr || original.isa_get_info == nullptr)
    return false;

  // Do NOT free the previous generation's translated backing storage here. Those
  // buffers can still be referenced by consumers whose lifetime is NOT bounded by
  // the HSA runtime generation -- notably rocprofiler-register, which is not in
  // tool_libs_ and finalizes its code-object records (each holding a memory_base
  // into these bytes) at process-exit atexit, not at hsa_shut_down. A reinstall
  // (next hsa_init after a shutdown that left a live executable) would otherwise
  // free bytes an old-generation profiler record still points into. The buffers are
  // therefore process-lifetime: released only at executable destroy (release()) or
  // when the process exits. RTLD_NODELETE keeps this DSO (and g_state) mapped across
  // generations, so retaining across reinstall is sound. This bounds growth by the
  // number of never-destroyed executables across runtime generations, which is
  // small; correctness (no dangling profiler/debugger pointer) takes precedence.
  // Build the immutable snapshot FIRST -- this is the only fallible step. If the
  // allocation throws, we must not have committed any install state: g_state.core
  // stays null so a later install is not permanently rejected by the "already
  // installed" guard above (the earlier order assigned core before this alloc, so a
  // throw left the hook sticky-uninstallable). A callback that already loaded the
  // previous generation's shared_ptr keeps that table alive for its whole call, so
  // publishing a fresh snapshot never mutates a table another thread dereferences.
  auto snapshot = std::make_shared<const OriginalApi>(original);

  g_state.vendor_reader.store(nullptr, std::memory_order_release);
  g_state.core = core;
  g_state.active_api.store(std::move(snapshot), std::memory_order_release);
  core->hsa_code_object_reader_create_from_file_fn = reader_create_from_file;
  core->hsa_code_object_reader_create_from_memory_fn = reader_create_from_memory;
  core->hsa_code_object_reader_destroy_fn = reader_destroy;
  core->hsa_executable_destroy_fn = executable_destroy;
  core->hsa_executable_load_agent_code_object_fn = load_agent_code_object;
  core->hsa_executable_load_program_code_object_fn = load_program_code_object;
  core->hsa_executable_load_code_object_fn = load_code_object;
  core->hsa_system_get_major_extension_table_fn = system_get_major_extension_table;
  log("installed eager gfx1250 B0-to-A0 hook");
  return true;
}

void uninstall() {
  std::lock_guard lock(g_state.lifecycle_mutex);
  CoreApiTable *core = g_state.core;
  // Swap the published snapshot out for nullptr. Any callback still holding the
  // previous shared_ptr keeps its table alive until that call returns; this only
  // stops NEW callbacks from finding a table.
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
    if (core->hsa_executable_load_program_code_object_fn == load_program_code_object)
      core->hsa_executable_load_program_code_object_fn = original->load_program;
    if (core->hsa_executable_load_code_object_fn == load_code_object)
      core->hsa_executable_load_code_object_fn = original->load_deprecated;
    if (core->hsa_system_get_major_extension_table_fn == system_get_major_extension_table)
      core->hsa_system_get_major_extension_table_fn = original->get_extension_table;
  }
  // ROCr destroys its loader after OnUnload but before closing tool DSOs.
  // Keep code-object backing storage alive until that later DSO close.
  g_state.vendor_reader.store(nullptr, std::memory_order_release);
  g_state.core = nullptr;
}

Blob copy_bytes(const void *data, size_t size) {
  if (data == nullptr || size == 0)
    return nullptr;
  try {
    const auto *begin = static_cast<const uint8_t *>(data);
    return std::make_shared<std::vector<uint8_t>>(begin, begin + size);
  } catch (...) {
    return nullptr;
  }
}

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

  Blob bytes;
  try {
    bytes = std::make_shared<std::vector<uint8_t>>(size);
  } catch (...) {
    return nullptr;
  }
  size_t done = 0;
  while (done < size) {
    const ssize_t count =
        pread(file, bytes->data() + done, size - done, static_cast<off_t>(offset + done));
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return nullptr;
    done += static_cast<size_t>(count);
  }
  return bytes;
}

hsa_status_t capture_reader(const OriginalApi &api, hsa_code_object_reader_t *reader, Blob bytes) {
  if (bytes != nullptr && store_reader(*reader, std::move(bytes)))
    return HSA_STATUS_SUCCESS;
  (void)api.destroy_reader(*reader);
  *reader = {};
  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

bool is_gfx1250(const Blob &bytes) {
  if (bytes == nullptr || bytes->size() < sizeof(rocjitsu::Elf64_Ehdr))
    return false;
  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(&header, bytes->data(), sizeof(header));
  return std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) == 0 &&
         header.e_ident[rocjitsu::EI_CLASS] == rocjitsu::ELFCLASS64 &&
         header.e_machine == rocjitsu::EM_AMDGPU &&
         (header.e_flags & rocjitsu::EF_AMDGPU_MACH) == rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250;
}

enum class AgentStepping { kOther, kA0, kB0OrLater, kUnknown };

AgentStepping classify_agent(const OriginalApi &api, hsa_agent_t agent) {
  auto *iterate = api.agent_iterate_isas;
  auto *get_isa_info = api.isa_get_info;
  auto *get_agent_info = api.agent_get_info;
  if (iterate == nullptr || get_isa_info == nullptr || get_agent_info == nullptr)
    return AgentStepping::kUnknown;

  struct IsaData {
    decltype(hsa_isa_get_info_alt) *get_info;
    std::string name;
    bool failed = false;
  } data{get_isa_info, {}, false};
  const hsa_status_t status = iterate(
      agent,
      [](hsa_isa_t isa, void *opaque) -> hsa_status_t {
        auto *out = static_cast<IsaData *>(opaque);
        uint32_t length = 0;
        if (out->get_info(isa, HSA_ISA_INFO_NAME_LENGTH, &length) != HSA_STATUS_SUCCESS ||
            length == 0) {
          out->failed = true;
          return HSA_STATUS_ERROR;
        }
        out->name.resize(length);
        if (out->get_info(isa, HSA_ISA_INFO_NAME, out->name.data()) != HSA_STATUS_SUCCESS) {
          out->failed = true;
          return HSA_STATUS_ERROR;
        }
        if (!out->name.empty() && out->name.back() == '\0')
          out->name.pop_back();
        return HSA_STATUS_INFO_BREAK;
      },
      &data);
  if (data.failed || status != HSA_STATUS_INFO_BREAK)
    return status == HSA_STATUS_SUCCESS ? AgentStepping::kOther : AgentStepping::kUnknown;

  std::string_view target = data.name;
  const size_t dash = target.rfind('-');
  if (dash != std::string_view::npos)
    target.remove_prefix(dash + 1);
  const size_t colon = target.find(':');
  if (colon != std::string_view::npos)
    target = target.substr(0, colon);
  if (target != "gfx1250")
    return AgentStepping::kOther;

  uint32_t revision = 0;
  if (get_agent_info(agent, static_cast<hsa_agent_info_t>(kAmdAgentInfoAsicRevision), &revision) !=
      HSA_STATUS_SUCCESS)
    return AgentStepping::kUnknown;
  return revision == 0 ? AgentStepping::kA0 : AgentStepping::kB0OrLater;
}

bool any_agent_could_be_a0(const OriginalApi &api) {
  auto *iterate = api.iterate_agents;
  if (iterate == nullptr)
    return true;
  struct IterateData {
    const OriginalApi *api;
    bool found = false;
  } data{&api, false};
  const hsa_status_t status = iterate(
      [](hsa_agent_t agent, void *opaque) -> hsa_status_t {
        auto *data = static_cast<IterateData *>(opaque);
        const AgentStepping stepping = classify_agent(*data->api, agent);
        if (stepping == AgentStepping::kA0 || stepping == AgentStepping::kUnknown) {
          data->found = true;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &data);
  return data.found || (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK);
}

hsa_status_t load_owned_bytes(hsa_executable_t executable, hsa_agent_t agent, const Blob &bytes,
                              const char *options, hsa_loaded_code_object_t *loaded,
                              const OriginalApi &api,
                              decltype(hsa_executable_load_agent_code_object) *original_load) {
  if (bytes == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  auto *create = api.create_memory;
  auto *destroy = api.destroy_reader;
  if (create == nullptr || destroy == nullptr || original_load == nullptr)
    return HSA_STATUS_ERROR;

  hsa_code_object_reader_t owned_reader{};
  const hsa_status_t reader_status = create(bytes->data(), bytes->size(), &owned_reader);
  if (reader_status != HSA_STATUS_SUCCESS)
    return reader_status;
  auto reader_guard = make_scope_guard([&] { (void)destroy(owned_reader); });

  // RESERVE the ownership slot BEFORE entering the lower loader. ROCr aliases (does
  // not copy) the ELF pointer -- CodeObjectReaderImpl::SetMemory stores it directly
  // and the loader hands code->ElfData() to the LoadedCodeObjectImpl -- so once a
  // load succeeds, ROCr references these bytes until the executable is destroyed.
  // The map insertion is fallible (allocation); doing it AFTER a successful load
  // would mean a successful publish could be followed by a failed retain, leaving
  // ROCr with an unowned raw pointer. Reserving first makes retention a no-fail fact
  // by the time the load returns success.
  if (!retain(executable, bytes))
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // On FAILURE, drop the reservation. A failure may be a pre-publication rejection
  // (null/invalid executable, profile/ISA mismatch) where ROCr never referenced the
  // bytes, so keeping the blob would strand it under a handle no destroy releases --
  // unbounded growth on repeated invalid loads. The rejection status is not
  // distinguishable from a post-publication failure, so we drop on all failures.
  // KNOWN LIMITATION: if ROCr publishes the loaded object and then a LATER stage
  // (segment/symbol/relocation/trampoline) fails, it does not roll back the appended
  // object, so it briefly holds a pointer into bytes we drop here. Closing that
  // window needs a transactional lower loader (or a published-object query) upstream
  // in ROCr; it is not fixable from the hook without a status-guessing heuristic that
  // reintroduces the unbounded-growth bug.
  const hsa_status_t status = original_load(executable, agent, owned_reader, options, loaded);
  if (status != HSA_STATUS_SUCCESS)
    unretain(executable, bytes);
  return status;
}

hsa_status_t HSA_API reader_create_from_memory(const void *code_object, size_t size,
                                               hsa_code_object_reader_t *reader) {
  return hsa_boundary([&] {
    // Reject a null output pointer before forwarding: the lower/vendor API may write
    // through it without checking, so fail fast the way ROCr's own layer does.
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->create_memory;
    const hsa_status_t status = original(code_object, size, reader);
    if (status != HSA_STATUS_SUCCESS)
      return status;

    return capture_reader(*api, reader, copy_bytes(code_object, size));
  });
}

hsa_status_t HSA_API reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t *reader) {
  return hsa_boundary([&] {
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->create_file;
    const hsa_status_t status = original(file, reader);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    return capture_reader(*api, reader, read_file_region(file, 0, 0));
  });
}

hsa_status_t vendor_reader_create(hsa_file_t file, size_t offset, size_t size,
                                  hsa_code_object_reader_t *reader) {
  return hsa_boundary([&] {
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto original = g_state.vendor_reader.load(std::memory_order_acquire);
    if (original == nullptr)
      return HSA_STATUS_ERROR;
    const hsa_status_t status = original(file, offset, size, reader);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    return capture_reader(*api, reader, read_file_region(file, offset, size));
  });
}

hsa_status_t HSA_API system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                      size_t table_length, void *table) {
  return hsa_boundary([&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->get_extension_table;
    const hsa_status_t status = original(extension, version_major, table_length, table);
    constexpr size_t reader_field_end =
        offsetof(VendorLoaderTable, create_reader_from_file) + sizeof(VendorReaderCreate);
    if (status != HSA_STATUS_SUCCESS || table == nullptr || extension != HSA_EXTENSION_AMD_LOADER ||
        version_major != 1 || table_length < reader_field_end)
      return status;

    auto *loader = static_cast<VendorLoaderTable *>(table);
    auto reader = loader->create_reader_from_file;
    if (reader != nullptr) {
      auto expected = static_cast<VendorReaderCreate>(nullptr);
      g_state.vendor_reader.compare_exchange_strong(expected, reader, std::memory_order_acq_rel);
      loader->create_reader_from_file = vendor_reader_create;
    }
    return status;
  });
}

hsa_status_t HSA_API reader_destroy(hsa_code_object_reader_t reader) {
  return hsa_boundary([&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->destroy_reader;
    const hsa_status_t status = original(reader);
    if (status == HSA_STATUS_SUCCESS)
      erase_reader(reader);
    return status;
  });
}

hsa_status_t HSA_API executable_destroy(hsa_executable_t executable) {
  return hsa_boundary([&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->destroy_executable;
    const hsa_status_t status = original(executable);
    if (status == HSA_STATUS_SUCCESS)
      release(executable);
    return status;
  });
}

hsa_status_t HSA_API load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                            hsa_code_object_reader_t reader, const char *options,
                                            hsa_loaded_code_object_t *loaded) {
  return hsa_boundary([&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original_load = api->load_agent;

    const AgentStepping stepping = classify_agent(*api, agent);
    if (stepping == AgentStepping::kOther || stepping == AgentStepping::kB0OrLater)
      return original_load(executable, agent, reader, options, loaded);
    if (stepping == AgentStepping::kUnknown)
      return HSA_STATUS_ERROR_INVALID_AGENT;

    Blob source = lookup_reader(reader);
    if (source == nullptr)
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
    if (!is_gfx1250(source))
      return load_owned_bytes(executable, agent, source, options, loaded, *api, original_load);

    // gfx1250 A0 and B0 code objects carry the same machine identity. Mode 2 is
    // a B0-input environment, so gfx1250 input targeting an A0 agent uses the
    // fixed B0-to-A0 profile.
    uint8_t *translated_data = nullptr;
    size_t translated_size = 0;
    const rj_status_t translate_status = rj_gfx1250_b0_to_a0_translate(
        source->data(), source->size(), &translated_data, &translated_size);
    if (translate_status != ROCJITSU_STATUS_SUCCESS || translated_data == nullptr ||
        translated_size == 0) {
      rj_gfx1250_b0_to_a0_free(translated_data);
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    Blob translated = copy_bytes(translated_data, translated_size);
    rj_gfx1250_b0_to_a0_free(translated_data);
    if (translated == nullptr)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    const hsa_status_t load_status =
        load_owned_bytes(executable, agent, translated, options, loaded, *api, original_load);
    log("eager translated input_bytes=%zu output_bytes=%zu status=%d", source->size(),
        translated_size, static_cast<int>(load_status));
    return load_status;
  });
}

hsa_status_t HSA_API load_program_code_object(hsa_executable_t executable,
                                              hsa_code_object_reader_t reader, const char *options,
                                              hsa_loaded_code_object_t *loaded) {
  return hsa_boundary([&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->load_program;
    if (any_agent_could_be_a0(*api))
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    return original(executable, reader, options, loaded);
  });
}

hsa_status_t HSA_API load_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                      hsa_code_object_t code_object, const char *options) {
  return hsa_boundary([&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->load_deprecated;
    const AgentStepping stepping = classify_agent(*api, agent);
    if (stepping == AgentStepping::kA0 || stepping == AgentStepping::kUnknown)
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    return original(executable, agent, code_object, options);
  });
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_EXPORT
#endif

extern "C" RJ_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                      uint64_t failed_tool_count,
                                      const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;
  try {
    return install(table);
  } catch (...) {
    return false;
  }
}

extern "C" RJ_HOOK_EXPORT void OnUnload() {
  try {
    uninstall();
  } catch (...) {
  }
}

#if defined(RJ_HOTSWAP_TEST_HOOKS)
// Test-only: total number of translated backing buffers currently retained across
// all executables. Lets a unit test assert the storage-retention lifecycle --
// buffers survive OnUnload() AND a runtime-generation reinstall (install() does not
// clear them, because an old-generation profiler record may still alias them), and
// are released only at executable destroy or process exit. Never present in the
// shipped DSO: its version script exports only OnLoad/OnUnload; the testable build
// adds rj_test_*.
extern "C" RJ_HOOK_EXPORT size_t rj_test_retained_executable_buffer_count() {
  std::lock_guard lock(g_state.storage_mutex);
  size_t count = 0;
  for (const auto &[handle, buffers] : g_state.executables) {
    (void)handle;
    count += buffers.size();
  }
  return count;
}

// Test-only: drop all retained storage. Production storage is process-lifetime (not
// cleared on reinstall), so a test fixture needs this to isolate the retention
// lifecycle between test cases; production never calls it.
extern "C" RJ_HOOK_EXPORT void rj_test_clear_retained_storage() {
  std::lock_guard lock(g_state.storage_mutex);
  g_state.readers.clear();
  g_state.executables.clear();
}
#endif // RJ_HOTSWAP_TEST_HOOKS
