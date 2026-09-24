// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>

// Keep this fixture independent from observer_abi_v13.h. These declarations
// model the native C++ surface consumed by an external FFM v13 backend, so the
// adapter and its test backend cannot accidentally agree on the same bad ABI
// mirror. In particular, use FFM's bitfields rather than the adapter's raw-byte
// representation.
namespace foreign_ffm_v13 {

using EntityId = std::uint64_t;
using FfmResourceType = std::uint32_t;
using FfmWaitType = std::uint32_t;
using FfmWaitName = std::uint32_t;

inline constexpr std::uint32_t FFM_MAX_WAVE_SIZE = 64;
inline constexpr std::uint32_t FFM_OBSERVER_PLUGIN_OLDEST_SUPPORTED_API_VERSION = 5;
inline constexpr std::uint32_t FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION = 13;

struct FfmObserverInstruction {
  std::uint64_t pc;
  const std::uint32_t raw_isa[4];
};

struct FfmDispatchInfo {
  EntityId dispatch_id;
};

struct FfmDispatchMetadata {
  FfmDispatchInfo dispatch_info;
  std::uint32_t vgpr_count;
  std::uint32_t sgpr_count;
  std::uint32_t lds_size_bytes;
  std::uint32_t wave_size;
  std::uint32_t num_waves_per_wg;
  std::uint32_t grid_size[3];
  std::uint32_t workgroup_size[3];
  const char *dispatch_name;
};

struct FfmClusterInfo {
  FfmDispatchInfo dispatch_info;
  EntityId cluster_id;
};

struct FfmWorkgroupInfo {
  FfmClusterInfo cluster_info;
  EntityId workgroup_id;
};

struct FfmWavegroupInfo {
  FfmWorkgroupInfo workgroup_info;
  EntityId wavegroup_id;
};

struct FfmWaveInfo {
  FfmWorkgroupInfo workgroup_info;
  EntityId wavegroup_id;
  EntityId wave_id;
};

struct FfmWaitInfo {
  FfmWaitType wait_type;
  FfmWaitName wait_name;
};

struct FfmInstructionCounters {
  std::uint64_t valu_count;
  std::uint64_t salu_count;
  std::uint64_t smem_count;
  std::uint64_t lds_count;
  std::uint64_t flat_count;
  std::uint64_t tex_count;
  std::uint64_t global_scratch_load;
  std::uint64_t global_scratch_store;
  std::uint64_t xdl_valu_count;
};

struct FfmInstructionInfo {
  EntityId instruction_id;
  FfmWaveInfo wave_info;
  FfmObserverInstruction instruction;
  FfmInstructionCounters instruction_counters;
  FfmWaitInfo wait_info;
};

struct FfmMemoryAccess {
  EntityId instruction_id;
  FfmWaveInfo wave_info;
  std::uint64_t exec_mask;
  std::uint32_t wave_size;
  std::uint64_t addresses[FFM_MAX_WAVE_SIZE];
  std::uint32_t data_size_bytes;
  FfmResourceType resource_type;
  std::uint8_t is_atomic : 1;
  std::uint8_t is_read : 1;
  std::uint8_t is_write : 1;
};

struct FfmTdmMemoryAccess {
  EntityId instruction_id;
  FfmWaveInfo wave_info;
  std::uint32_t num_addresses;
  const std::uint64_t *addresses;
  std::uint32_t data_size_bytes;
  std::uint8_t is_read : 1;
  std::uint8_t is_write : 1;
  std::uint32_t tile_dim0;
  std::uint32_t tile_dim1;
  std::uint32_t data_size;
  std::int64_t tensor_dim0_stride;
  std::int64_t tensor_dim1_stride;
};

struct FfmResourceAccess;
struct FfmBarrier;

using FfmLogLevel = std::uint32_t;
inline constexpr FfmLogLevel FFM_LOG_DEBUG = 0;
inline constexpr FfmLogLevel FFM_LOG_INFO = 1;
inline constexpr FfmLogLevel FFM_LOG_WARN = 2;
inline constexpr FfmLogLevel FFM_LOG_ERROR = 3;

struct FfmHostApi {
  std::uint32_t api_version;
  void (*log)(FfmLogLevel level, const char *msg);
};

struct FfmObserverPluginApi {
  std::uint32_t api_version;
  const char *name;
  void (*on_init)(const FfmHostApi *host);
  void (*on_dispatch_begin)(const FfmDispatchMetadata *dispatch);
  void (*on_dispatch_end)(const FfmDispatchMetadata *dispatch);
  void (*on_cluster_begin)(const FfmClusterInfo *cluster);
  void (*on_cluster_end)(const FfmClusterInfo *cluster);
  void (*on_workgroup_begin)(const FfmWorkgroupInfo *workgroup);
  void (*on_workgroup_end)(const FfmWorkgroupInfo *workgroup);
  void (*on_wavegroup_begin)(const FfmWavegroupInfo *wavegroup);
  void (*on_wavegroup_end)(const FfmWavegroupInfo *wavegroup);
  void (*on_wave_begin)(const FfmWaveInfo *wave);
  void (*on_wave_end)(const FfmWaveInfo *wave);
  void (*on_instruction)(const FfmInstructionInfo *instruction);
  void (*on_resource_access)(const FfmResourceAccess *access);
  void (*on_barrier_signal)(const FfmBarrier *barrier);
  void (*on_barrier_wait)(const FfmBarrier *barrier);
  void (*on_barrier_complete)(const FfmBarrier *barrier);
  void (*on_shutdown)();
  void (*on_memory_access)(const FfmMemoryAccess *access);
  void (*on_tdm_memory_access)(const FfmTdmMemoryAccess *access);
};

static_assert(std::is_standard_layout_v<FfmMemoryAccess>);
static_assert(std::is_trivially_copyable_v<FfmMemoryAccess>);
static_assert(sizeof(FfmMemoryAccess) == 592);
static_assert(offsetof(FfmMemoryAccess, resource_type) == 580);
static_assert(sizeof(FfmDispatchMetadata) == 64);
static_assert(offsetof(FfmDispatchMetadata, dispatch_name) == 56);
static_assert(sizeof(FfmTdmMemoryAccess) == 104);
static_assert(offsetof(FfmTdmMemoryAccess, data_size_bytes) == 64);
static_assert(offsetof(FfmTdmMemoryAccess, tile_dim0) == 72);
static_assert(offsetof(FfmTdmMemoryAccess, tensor_dim0_stride) == 88);
static_assert(sizeof(FfmHostApi) == 16);
static_assert(offsetof(FfmHostApi, log) == 8);
static_assert(sizeof(FfmObserverPluginApi) == 168);
static_assert(offsetof(FfmObserverPluginApi, on_instruction) == 104);
static_assert(offsetof(FfmObserverPluginApi, on_shutdown) == 144);
static_assert(offsetof(FfmObserverPluginApi, on_memory_access) == 152);
static_assert(offsetof(FfmObserverPluginApi, on_tdm_memory_access) == 160);

} // namespace foreign_ffm_v13

// Model the native v8 payloads separately so the compatibility test proves
// that the adapter preserves the legacy prefixes instead of merely agreeing
// with its own v13 supersets.
namespace foreign_ffm_v8 {

using FfmClusterInfo = foreign_ffm_v13::FfmClusterInfo;
using FfmDispatchInfo = foreign_ffm_v13::FfmDispatchInfo;
using FfmInstructionInfo = foreign_ffm_v13::FfmInstructionInfo;
using FfmMemoryAccess = foreign_ffm_v13::FfmMemoryAccess;
using FfmResourceAccess = foreign_ffm_v13::FfmResourceAccess;
using FfmBarrier = foreign_ffm_v13::FfmBarrier;
using FfmWavegroupInfo = foreign_ffm_v13::FfmWavegroupInfo;
using FfmWaveInfo = foreign_ffm_v13::FfmWaveInfo;
using FfmWorkgroupInfo = foreign_ffm_v13::FfmWorkgroupInfo;

struct FfmDispatchMetadata {
  FfmDispatchInfo dispatch_info;
  std::uint32_t vgpr_count;
  std::uint32_t sgpr_count;
  std::uint32_t lds_size_bytes;
  std::uint32_t wave_size;
  std::uint32_t num_waves_per_wg;
  std::uint32_t grid_size[3];
  std::uint32_t workgroup_size[3];
};

struct FfmTdmMemoryAccess {
  foreign_ffm_v13::EntityId instruction_id;
  FfmWaveInfo wave_info;
  std::uint32_t num_addresses;
  const std::uint64_t *addresses;
  std::uint32_t data_size_bytes;
  std::uint8_t is_read : 1;
  std::uint8_t is_write : 1;
};

struct FfmHostApi {
  std::uint32_t api_version;
};

struct FfmObserverPluginApi {
  std::uint32_t api_version;
  const char *name;
  void (*on_init)(const FfmHostApi *host);
  void (*on_dispatch_begin)(const FfmDispatchMetadata *dispatch);
  void (*on_dispatch_end)(const FfmDispatchMetadata *dispatch);
  void (*on_cluster_begin)(const FfmClusterInfo *cluster);
  void (*on_cluster_end)(const FfmClusterInfo *cluster);
  void (*on_workgroup_begin)(const FfmWorkgroupInfo *workgroup);
  void (*on_workgroup_end)(const FfmWorkgroupInfo *workgroup);
  void (*on_wavegroup_begin)(const FfmWavegroupInfo *wavegroup);
  void (*on_wavegroup_end)(const FfmWavegroupInfo *wavegroup);
  void (*on_wave_begin)(const FfmWaveInfo *wave);
  void (*on_wave_end)(const FfmWaveInfo *wave);
  void (*on_instruction)(const FfmInstructionInfo *instruction);
  void (*on_resource_access)(const FfmResourceAccess *access);
  void (*on_barrier_signal)(const FfmBarrier *barrier);
  void (*on_barrier_wait)(const FfmBarrier *barrier);
  void (*on_barrier_complete)(const FfmBarrier *barrier);
  void (*on_shutdown)();
  void (*on_memory_access)(const FfmMemoryAccess *access);
  void (*on_tdm_memory_access)(const FfmTdmMemoryAccess *access);
};

static_assert(sizeof(FfmDispatchMetadata) == 56);
static_assert(offsetof(FfmDispatchMetadata, workgroup_size) == 40);
static_assert(sizeof(FfmTdmMemoryAccess) == 72);
static_assert(offsetof(FfmTdmMemoryAccess, data_size_bytes) == 64);
static_assert(sizeof(FfmHostApi) == 4);
static_assert(sizeof(FfmObserverPluginApi) == 168);
static_assert(offsetof(FfmObserverPluginApi, on_instruction) == 104);
static_assert(offsetof(FfmObserverPluginApi, on_shutdown) == 144);
static_assert(offsetof(FfmObserverPluginApi, on_memory_access) == 152);
static_assert(offsetof(FfmObserverPluginApi, on_tdm_memory_access) == 160);

} // namespace foreign_ffm_v8

namespace {

using namespace foreign_ffm_v13;

std::mutex trace_mutex;
void (*host_log)(FfmLogLevel, const char *) = nullptr;
std::thread host_log_worker;

std::uint8_t memory_flag_byte(bool is_atomic, bool is_read, bool is_write) {
  FfmMemoryAccess access{};
  access.is_atomic = is_atomic;
  access.is_read = is_read;
  access.is_write = is_write;
  return reinterpret_cast<const std::uint8_t *>(&access)[584];
}

std::uint8_t tdm_flag_byte(bool is_read, bool is_write) {
  FfmTdmMemoryAccess access{};
  access.is_read = is_read;
  access.is_write = is_write;
  return reinterpret_cast<const std::uint8_t *>(&access)[68];
}

bool has_expected_native_bitfield_layout() {
  return memory_flag_byte(false, false, false) == 0x00 &&
         memory_flag_byte(true, false, false) == 0x01 &&
         memory_flag_byte(false, true, false) == 0x02 &&
         memory_flag_byte(false, false, true) == 0x04 &&
         memory_flag_byte(true, true, true) == 0x07 && tdm_flag_byte(false, false) == 0x00 &&
         tdm_flag_byte(true, false) == 0x01 && tdm_flag_byte(false, true) == 0x02 &&
         tdm_flag_byte(true, true) == 0x03;
}

bool has_expected_v8_native_bitfield_layout() {
  foreign_ffm_v8::FfmTdmMemoryAccess access{};
  access.is_read = true;
  access.is_write = true;
  return reinterpret_cast<const std::uint8_t *>(&access)[68] == 0x03;
}

void trace(const std::string &line) noexcept {
  try {
    std::lock_guard<std::mutex> lock(trace_mutex);
    const char *path = std::getenv("ROCJITSU_PERFSIM_FAKE_TRACE");
    if (!path || !*path)
      return;
    std::ofstream output(path, std::ios::app);
    output << line << '\n';
  } catch (...) {
  }
}

bool mode_is(const char *value) {
  const char *mode = std::getenv("ROCJITSU_PERFSIM_FAKE_MODE");
  return mode && std::strcmp(mode, value) == 0;
}

void on_init(const FfmHostApi *host) {
  const std::uint32_t version = host ? host->api_version : 0;
  if (version < FFM_OBSERVER_PLUGIN_OLDEST_SUPPORTED_API_VERSION ||
      version > FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION) {
    trace("init_rejected " + std::to_string(version));
    return;
  }
  host_log = host->log;
  if (mode_is("v12_only") || mode_is("v11_only"))
    trace(host_log ? "host_log present" : "host_log absent");
  if ((mode_is("host_log") || mode_is("v12_only")) && host_log)
    host_log(FFM_LOG_WARN, "fake backend initialized");
  if (mode_is("async_host_log") && host_log) {
    auto logger = host_log;
    host_log_worker =
        std::thread([logger]() { logger(FFM_LOG_WARN, "fake backend worker initialized"); });
  }
  trace("init " + std::to_string(version));
}

void append_dispatch(std::ostringstream &out, const FfmDispatchMetadata *dispatch) {
  if (!dispatch) {
    out << " null";
    return;
  }
  out << ' ' << dispatch->dispatch_info.dispatch_id << ' ' << dispatch->vgpr_count << ' '
      << dispatch->sgpr_count << ' ' << dispatch->lds_size_bytes << ' ' << dispatch->wave_size
      << ' ' << dispatch->num_waves_per_wg;
  for (uint32_t value : dispatch->grid_size)
    out << ' ' << value;
  for (uint32_t value : dispatch->workgroup_size)
    out << ' ' << value;
  out << ' ' << (dispatch->dispatch_name ? dispatch->dispatch_name : "<null>");
}

void on_dispatch_begin(const FfmDispatchMetadata *dispatch) {
  std::ostringstream out;
  out << "begin";
  append_dispatch(out, dispatch);
  trace(out.str());
}

void on_dispatch_end(const FfmDispatchMetadata *dispatch) {
  std::ostringstream out;
  out << "end";
  append_dispatch(out, dispatch);
  trace(out.str());
}

void on_init_v8(const foreign_ffm_v8::FfmHostApi *host) {
  const std::uint32_t version = host ? host->api_version : 0;
  if (version != 8) {
    trace("init_rejected " + std::to_string(version));
    return;
  }
  host_log = nullptr;
  trace("init " + std::to_string(version));
}

void append_dispatch_v8(std::ostringstream &out,
                        const foreign_ffm_v8::FfmDispatchMetadata *dispatch) {
  if (!dispatch) {
    out << " null";
    return;
  }
  out << ' ' << dispatch->dispatch_info.dispatch_id << ' ' << dispatch->vgpr_count << ' '
      << dispatch->sgpr_count << ' ' << dispatch->lds_size_bytes << ' ' << dispatch->wave_size
      << ' ' << dispatch->num_waves_per_wg;
  for (uint32_t value : dispatch->grid_size)
    out << ' ' << value;
  for (uint32_t value : dispatch->workgroup_size)
    out << ' ' << value;
}

void on_dispatch_begin_v8(const foreign_ffm_v8::FfmDispatchMetadata *dispatch) {
  std::ostringstream out;
  out << "begin";
  append_dispatch_v8(out, dispatch);
  trace(out.str());
}

void on_dispatch_end_v8(const foreign_ffm_v8::FfmDispatchMetadata *dispatch) {
  std::ostringstream out;
  out << "end";
  append_dispatch_v8(out, dispatch);
  trace(out.str());
}

void append_wave(std::ostringstream &out, const FfmWaveInfo &wave) {
  out << wave.workgroup_info.cluster_info.dispatch_info.dispatch_id << ' '
      << wave.workgroup_info.cluster_info.cluster_id << ' ' << wave.workgroup_info.workgroup_id
      << ' ' << wave.wavegroup_id << ' ' << wave.wave_id;
}

void on_instruction(const FfmInstructionInfo *instruction) {
  std::ostringstream out;
  out << "instruction ";
  if (!instruction) {
    out << "null";
    trace(out.str());
    return;
  }
  append_wave(out, instruction->wave_info);
  out << ' ' << instruction->instruction_id << ' ' << instruction->instruction.pc << std::hex;
  for (uint32_t word : instruction->instruction.raw_isa)
    out << ' ' << word;
  out << std::dec << ' ' << instruction->instruction_counters.valu_count << ' '
      << instruction->instruction_counters.salu_count << ' '
      << instruction->instruction_counters.smem_count << ' '
      << instruction->instruction_counters.lds_count << ' '
      << instruction->instruction_counters.flat_count << ' '
      << instruction->instruction_counters.tex_count << ' '
      << instruction->instruction_counters.global_scratch_load << ' '
      << instruction->instruction_counters.global_scratch_store << ' '
      << instruction->instruction_counters.xdl_valu_count << ' '
      << static_cast<int>(instruction->wait_info.wait_type) << ' '
      << static_cast<int>(instruction->wait_info.wait_name);
  trace(out.str());
}

void on_memory_access(const FfmMemoryAccess *access) {
  std::ostringstream out;
  out << "memory ";
  if (!access) {
    out << "null";
    trace(out.str());
    return;
  }
  append_wave(out, access->wave_info);
  out << ' ' << access->instruction_id << ' ' << access->exec_mask << ' ' << access->wave_size
      << ' ' << access->data_size_bytes << ' ' << static_cast<int>(access->resource_type) << ' '
      << static_cast<int>(access->is_atomic) << ' ' << static_cast<int>(access->is_read) << ' '
      << static_cast<int>(access->is_write);
  for (uint32_t lane = 0; lane < access->wave_size; ++lane)
    out << ' ' << access->addresses[lane];
  trace(out.str());
}

void on_tdm_memory_access(const FfmTdmMemoryAccess *access) {
  std::ostringstream out;
  out << "tdm ";
  if (!access) {
    out << "null";
    trace(out.str());
    return;
  }
  append_wave(out, access->wave_info);
  out << ' ' << access->instruction_id << ' ' << access->num_addresses << ' '
      << access->data_size_bytes << ' ' << static_cast<int>(access->is_read) << ' '
      << static_cast<int>(access->is_write) << ' ' << access->tile_dim0 << ' ' << access->tile_dim1
      << ' ' << access->data_size << ' ' << access->tensor_dim0_stride << ' '
      << access->tensor_dim1_stride;
  for (uint32_t i = 0; i < access->num_addresses; ++i)
    out << ' ' << access->addresses[i];
  trace(out.str());
}

void on_tdm_memory_access_v8(const foreign_ffm_v8::FfmTdmMemoryAccess *access) {
  std::ostringstream out;
  out << "tdm ";
  if (!access) {
    out << "null";
    trace(out.str());
    return;
  }
  append_wave(out, access->wave_info);
  out << ' ' << access->instruction_id << ' ' << access->num_addresses << ' '
      << access->data_size_bytes << ' ' << static_cast<int>(access->is_read) << ' '
      << static_cast<int>(access->is_write);
  for (uint32_t i = 0; i < access->num_addresses; ++i)
    out << ' ' << access->addresses[i];
  trace(out.str());
}

void on_shutdown() {
  if (host_log_worker.joinable())
    host_log_worker.join();
  if (mode_is("host_log") && host_log)
    host_log(FFM_LOG_ERROR, "fake backend shutting down");
  host_log = nullptr;
  trace("shutdown");
}

FfmObserverPluginApi api{};
foreign_ffm_v8::FfmObserverPluginApi api_v8{};

template <typename Callback> void maybe_remove(Callback &callback, const char *name) {
  if (mode_is(name))
    callback = nullptr;
}

} // namespace

extern "C" __attribute__((visibility("default"))) foreign_ffm_v13::FfmObserverPluginApi *
ffm_observer_plugin_get_api(uint32_t host_api_version) {
  trace("get_api " + std::to_string(host_api_version));
  if (mode_is("reject_all") || !has_expected_native_bitfield_layout())
    return nullptr;

  if (mode_is("v8_only")) {
    if (host_api_version != 8 || !has_expected_v8_native_bitfield_layout())
      return nullptr;
    api_v8 = {};
    api_v8.api_version = 8;
    api_v8.name = "rocjitsu-perfsim-fake-v8";
    api_v8.on_init = on_init_v8;
    api_v8.on_dispatch_begin = on_dispatch_begin_v8;
    api_v8.on_dispatch_end = on_dispatch_end_v8;
    api_v8.on_instruction = on_instruction;
    api_v8.on_shutdown = on_shutdown;
    api_v8.on_memory_access = on_memory_access;
    api_v8.on_tdm_memory_access = on_tdm_memory_access_v8;
    return reinterpret_cast<foreign_ffm_v13::FfmObserverPluginApi *>(&api_v8);
  }

  const std::uint32_t accepted_request = mode_is("v12_only") ? 12 : mode_is("v11_only") ? 11 : 13;
  if (host_api_version != accepted_request)
    return nullptr;

  api = {};
  api.api_version = mode_is("v12_only")          ? 12
                    : mode_is("v11_only")        ? 11
                    : mode_is("old_version")     ? 12
                    : mode_is("too_old_version") ? 7
                    : mode_is("bad_version")     ? 14
                                                 : 13;
  api.name = "rocjitsu-perfsim-fake";
  api.on_init = on_init;
  api.on_dispatch_begin = on_dispatch_begin;
  api.on_dispatch_end = on_dispatch_end;
  api.on_instruction = on_instruction;
  api.on_shutdown = on_shutdown;
  api.on_memory_access = on_memory_access;
  api.on_tdm_memory_access = on_tdm_memory_access;

  maybe_remove(api.on_init, "missing_on_init");
  maybe_remove(api.on_dispatch_begin, "missing_on_dispatch_begin");
  maybe_remove(api.on_dispatch_end, "missing_on_dispatch_end");
  maybe_remove(api.on_instruction, "missing_on_instruction");
  maybe_remove(api.on_memory_access, "missing_on_memory_access");
  maybe_remove(api.on_tdm_memory_access, "missing_on_tdm_memory_access");
  maybe_remove(api.on_shutdown, "missing_on_shutdown");
  return &api;
}

__attribute__((destructor)) static void on_unload() {
  if (host_log_worker.joinable())
    host_log_worker.join();
  trace("unload");
}
