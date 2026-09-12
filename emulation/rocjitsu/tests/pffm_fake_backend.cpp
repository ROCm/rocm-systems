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
#include <type_traits>

// Keep this fixture independent from observer_abi_v8.h. These declarations
// model the native C++ surface consumed by an external FFM v8 backend, so the
// adapter and its test backend cannot accidentally agree on the same bad ABI
// mirror. In particular, use FFM's bitfields rather than the adapter's raw-byte
// representation.
namespace foreign_ffm_v8 {

using EntityId = std::uint64_t;
using FfmResourceType = std::uint32_t;
using FfmWaitType = std::uint32_t;
using FfmWaitName = std::uint32_t;

inline constexpr std::uint32_t FFM_MAX_WAVE_SIZE = 64;

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
};

struct FfmResourceAccess;
struct FfmBarrier;

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

static_assert(std::is_standard_layout_v<FfmMemoryAccess>);
static_assert(std::is_trivially_copyable_v<FfmMemoryAccess>);
static_assert(sizeof(FfmMemoryAccess) == 592);
static_assert(offsetof(FfmMemoryAccess, resource_type) == 580);
static_assert(sizeof(FfmTdmMemoryAccess) == 72);
static_assert(offsetof(FfmTdmMemoryAccess, data_size_bytes) == 64);
static_assert(sizeof(FfmObserverPluginApi) == 168);
static_assert(offsetof(FfmObserverPluginApi, on_instruction) == 104);
static_assert(offsetof(FfmObserverPluginApi, on_shutdown) == 144);
static_assert(offsetof(FfmObserverPluginApi, on_memory_access) == 152);
static_assert(offsetof(FfmObserverPluginApi, on_tdm_memory_access) == 160);

} // namespace foreign_ffm_v8

namespace {

using namespace foreign_ffm_v8;

std::mutex trace_mutex;

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

void trace(const std::string &line) noexcept {
  try {
    std::lock_guard<std::mutex> lock(trace_mutex);
    const char *path = std::getenv("ROCJITSU_PFFM_FAKE_TRACE");
    if (!path || !*path)
      return;
    std::ofstream output(path, std::ios::app);
    output << line << '\n';
  } catch (...) {
  }
}

bool mode_is(const char *value) {
  const char *mode = std::getenv("ROCJITSU_PFFM_FAKE_MODE");
  return mode && std::strcmp(mode, value) == 0;
}

void on_init(const FfmHostApi *host) {
  trace("init " + std::to_string(host ? host->api_version : 0));
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
      << static_cast<int>(access->is_write);
  for (uint32_t i = 0; i < access->num_addresses; ++i)
    out << ' ' << access->addresses[i];
  trace(out.str());
}

void on_shutdown() { trace("shutdown"); }

FfmObserverPluginApi api{};

template <typename Callback> void maybe_remove(Callback &callback, const char *name) {
  if (mode_is(name))
    callback = nullptr;
}

} // namespace

extern "C" __attribute__((visibility("default"))) foreign_ffm_v8::FfmObserverPluginApi *
ffm_observer_plugin_get_api(uint32_t host_api_version) {
  trace("get_api " + std::to_string(host_api_version));
  if (mode_is("reject_all") || !has_expected_native_bitfield_layout() || host_api_version != 8)
    return nullptr;

  api = {};
  api.api_version = mode_is("old_version") ? 7 : (mode_is("bad_version") ? 9 : 8);
  api.name = "rocjitsu-pffm-fake";
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

__attribute__((destructor)) static void on_unload() { trace("unload"); }
