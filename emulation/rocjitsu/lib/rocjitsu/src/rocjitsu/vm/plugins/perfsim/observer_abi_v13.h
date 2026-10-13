// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#if !defined(__linux__)
#error "The private FFM v13 ABI mirror is supported only on Linux"
#endif

#if !defined(__GXX_ABI_VERSION)
#error "The private FFM v13 ABI mirror requires the GCC/Clang Itanium C++ ABI"
#endif

namespace rocjitsu::plugins::perfsim::observer_abi_v13 {

inline constexpr std::uint32_t FFM_OBSERVER_PLUGIN_OLDEST_SUPPORTED_API_VERSION = 5;
inline constexpr std::uint32_t FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION = 13;
inline constexpr std::uint32_t FFM_MAX_WAVE_SIZE = 64;

using EntityId = std::uint64_t;

struct FfmObserverInstruction {
  std::uint64_t pc;
  const std::uint32_t raw_isa[4];
};

using FfmResourceType = std::uint32_t;
inline constexpr FfmResourceType FFM_RESOURCE_UNKNOWN = 0;
inline constexpr FfmResourceType FFM_RESOURCE_SGPR = 1;
inline constexpr FfmResourceType FFM_RESOURCE_VGPR = 2;
inline constexpr FfmResourceType FFM_RESOURCE_LDS = 3;
inline constexpr FfmResourceType FFM_RESOURCE_SCRATCH = 4;
inline constexpr FfmResourceType FFM_RESOURCE_GLOBAL = 5;
inline constexpr FfmResourceType FFM_RESOURCE_MREG = 6;
inline constexpr FfmResourceType FFM_RESOURCE_TYPE_COUNT = 7;

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

using FfmWaitType = std::uint32_t;
inline constexpr FfmWaitType FFM_WAIT_TYPE_NONE = 0;
inline constexpr FfmWaitType FFM_WAIT_TYPE_LOAD = 1;
inline constexpr FfmWaitType FFM_WAIT_TYPE_SAMPLE = 2;
inline constexpr FfmWaitType FFM_WAIT_TYPE_LDS = 3;
inline constexpr FfmWaitType FFM_WAIT_TYPE_VM = 4;
inline constexpr FfmWaitType FFM_WAIT_TYPE_BARRIER = 5;

using FfmWaitName = std::uint32_t;
inline constexpr FfmWaitName FFM_WAIT_NAME_NONE = 0;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_LOADCNT = 1;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_XCNT = 2;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_LOADCNT_DSCNT = 3;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_SAMPLECNT = 4;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_DSCNT = 5;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_STORECNT_DSCNT = 6;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_KMCNT = 7;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_IDLE = 8;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAITCNT = 9;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_WAIT_TENSORCNT = 10;
inline constexpr FfmWaitName FFM_WAIT_NAME_S_BARRIER_WAIT = 11;

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

inline constexpr std::uint8_t kMemoryAtomicFlag = 0x01;
inline constexpr std::uint8_t kMemoryReadFlag = 0x02;
inline constexpr std::uint8_t kMemoryWriteFlag = 0x04;

constexpr std::uint8_t encode_memory_flags(bool is_atomic, bool is_read, bool is_write) {
  return static_cast<std::uint8_t>((is_atomic ? kMemoryAtomicFlag : 0) |
                                   (is_read ? kMemoryReadFlag : 0) |
                                   (is_write ? kMemoryWriteFlag : 0));
}

struct FfmMemoryAccess {
  EntityId instruction_id;
  FfmWaveInfo wave_info;
  std::uint64_t exec_mask;
  std::uint32_t wave_size;
  std::uint64_t addresses[FFM_MAX_WAVE_SIZE];
  std::uint32_t data_size_bytes;
  FfmResourceType resource_type;
  std::uint8_t flags;
};

inline constexpr std::uint8_t kTdmReadFlag = 0x01;
inline constexpr std::uint8_t kTdmWriteFlag = 0x02;

constexpr std::uint8_t encode_tdm_flags(bool is_read, bool is_write) {
  return static_cast<std::uint8_t>((is_read ? kTdmReadFlag : 0) | (is_write ? kTdmWriteFlag : 0));
}

struct FfmTdmMemoryAccess {
  EntityId instruction_id;
  FfmWaveInfo wave_info;
  std::uint32_t num_addresses;
  const std::uint64_t *addresses;
  std::uint32_t data_size_bytes;
  std::uint8_t flags;
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

// Only the stable instruction-observer prefix used by this adapter is mirrored.
// Later graphics and experimental callbacks deliberately remain outside
// RocJITsu. The prefix through on_tdm_memory_access has remained stable.
struct FfmObserverPluginApiPrefix {
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

struct FfmObserverPluginApi;
using FfmObserverPluginGetApiFn = FfmObserverPluginApi *(*)(std::uint32_t host_api_version);

// Calls cross an independently declared C++ ABI mirror. The layouts are
// checked below, but Clang's function sanitizer compares nominal source types
// and therefore cannot validate these foreign calls.
template <typename Function, typename... Args>
#if defined(__clang__)
__attribute__((no_sanitize("function")))
#endif
decltype(auto)
invoke_foreign_abi(Function function, Args &&...args) {
  return function(std::forward<Args>(args)...);
}

static_assert(CHAR_BIT == 8);
static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(short) == 2);
static_assert(sizeof(int) == 4);
static_assert(sizeof(long) == 8);
static_assert(sizeof(long long) == 8);
static_assert(sizeof(void *) == 8);
static_assert(sizeof(void (*)()) == 8);

#define ROCJITSU_PERFSIM_ASSERT_RECORD(Type)                                                       \
  static_assert(std::is_standard_layout_v<Type>);                                                  \
  static_assert(std::is_trivially_copyable_v<Type>)

ROCJITSU_PERFSIM_ASSERT_RECORD(FfmObserverInstruction);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmDispatchInfo);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmDispatchMetadata);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmClusterInfo);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmWorkgroupInfo);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmWavegroupInfo);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmWaveInfo);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmWaitInfo);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmInstructionCounters);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmInstructionInfo);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmMemoryAccess);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmTdmMemoryAccess);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmHostApi);
ROCJITSU_PERFSIM_ASSERT_RECORD(FfmObserverPluginApiPrefix);

#undef ROCJITSU_PERFSIM_ASSERT_RECORD

static_assert(sizeof(FfmObserverInstruction) == 24);
static_assert(alignof(FfmObserverInstruction) == 8);
static_assert(offsetof(FfmObserverInstruction, pc) == 0);
static_assert(offsetof(FfmObserverInstruction, raw_isa) == 8);

static_assert(sizeof(FfmDispatchInfo) == 8);
static_assert(alignof(FfmDispatchInfo) == 8);
static_assert(offsetof(FfmDispatchInfo, dispatch_id) == 0);

static_assert(sizeof(FfmDispatchMetadata) == 64);
static_assert(alignof(FfmDispatchMetadata) == 8);
static_assert(offsetof(FfmDispatchMetadata, dispatch_info) == 0);
static_assert(offsetof(FfmDispatchMetadata, vgpr_count) == 8);
static_assert(offsetof(FfmDispatchMetadata, sgpr_count) == 12);
static_assert(offsetof(FfmDispatchMetadata, lds_size_bytes) == 16);
static_assert(offsetof(FfmDispatchMetadata, wave_size) == 20);
static_assert(offsetof(FfmDispatchMetadata, num_waves_per_wg) == 24);
static_assert(offsetof(FfmDispatchMetadata, grid_size) == 28);
static_assert(offsetof(FfmDispatchMetadata, workgroup_size) == 40);
static_assert(offsetof(FfmDispatchMetadata, dispatch_name) == 56);

static_assert(sizeof(FfmClusterInfo) == 16);
static_assert(alignof(FfmClusterInfo) == 8);
static_assert(offsetof(FfmClusterInfo, dispatch_info) == 0);
static_assert(offsetof(FfmClusterInfo, cluster_id) == 8);

static_assert(sizeof(FfmWorkgroupInfo) == 24);
static_assert(alignof(FfmWorkgroupInfo) == 8);
static_assert(offsetof(FfmWorkgroupInfo, cluster_info) == 0);
static_assert(offsetof(FfmWorkgroupInfo, workgroup_id) == 16);

static_assert(sizeof(FfmWavegroupInfo) == 32);
static_assert(alignof(FfmWavegroupInfo) == 8);
static_assert(offsetof(FfmWavegroupInfo, workgroup_info) == 0);
static_assert(offsetof(FfmWavegroupInfo, wavegroup_id) == 24);

static_assert(sizeof(FfmWaveInfo) == 40);
static_assert(alignof(FfmWaveInfo) == 8);
static_assert(offsetof(FfmWaveInfo, workgroup_info) == 0);
static_assert(offsetof(FfmWaveInfo, wavegroup_id) == 24);
static_assert(offsetof(FfmWaveInfo, wave_id) == 32);

static_assert(sizeof(FfmWaitInfo) == 8);
static_assert(alignof(FfmWaitInfo) == 4);
static_assert(offsetof(FfmWaitInfo, wait_type) == 0);
static_assert(offsetof(FfmWaitInfo, wait_name) == 4);

static_assert(sizeof(FfmInstructionCounters) == 72);
static_assert(alignof(FfmInstructionCounters) == 8);
static_assert(offsetof(FfmInstructionCounters, valu_count) == 0);
static_assert(offsetof(FfmInstructionCounters, salu_count) == 8);
static_assert(offsetof(FfmInstructionCounters, smem_count) == 16);
static_assert(offsetof(FfmInstructionCounters, lds_count) == 24);
static_assert(offsetof(FfmInstructionCounters, flat_count) == 32);
static_assert(offsetof(FfmInstructionCounters, tex_count) == 40);
static_assert(offsetof(FfmInstructionCounters, global_scratch_load) == 48);
static_assert(offsetof(FfmInstructionCounters, global_scratch_store) == 56);
static_assert(offsetof(FfmInstructionCounters, xdl_valu_count) == 64);

static_assert(sizeof(FfmInstructionInfo) == 152);
static_assert(alignof(FfmInstructionInfo) == 8);
static_assert(offsetof(FfmInstructionInfo, instruction_id) == 0);
static_assert(offsetof(FfmInstructionInfo, wave_info) == 8);
static_assert(offsetof(FfmInstructionInfo, instruction) == 48);
static_assert(offsetof(FfmInstructionInfo, instruction_counters) == 72);
static_assert(offsetof(FfmInstructionInfo, wait_info) == 144);

static_assert(sizeof(FfmMemoryAccess) == 592);
static_assert(alignof(FfmMemoryAccess) == 8);
static_assert(offsetof(FfmMemoryAccess, instruction_id) == 0);
static_assert(offsetof(FfmMemoryAccess, wave_info) == 8);
static_assert(offsetof(FfmMemoryAccess, exec_mask) == 48);
static_assert(offsetof(FfmMemoryAccess, wave_size) == 56);
static_assert(offsetof(FfmMemoryAccess, addresses) == 64);
static_assert(offsetof(FfmMemoryAccess, data_size_bytes) == 576);
static_assert(offsetof(FfmMemoryAccess, resource_type) == 580);
static_assert(offsetof(FfmMemoryAccess, flags) == 584);

static_assert(sizeof(FfmTdmMemoryAccess) == 104);
static_assert(alignof(FfmTdmMemoryAccess) == 8);
static_assert(offsetof(FfmTdmMemoryAccess, instruction_id) == 0);
static_assert(offsetof(FfmTdmMemoryAccess, wave_info) == 8);
static_assert(offsetof(FfmTdmMemoryAccess, num_addresses) == 48);
static_assert(offsetof(FfmTdmMemoryAccess, addresses) == 56);
static_assert(offsetof(FfmTdmMemoryAccess, data_size_bytes) == 64);
static_assert(offsetof(FfmTdmMemoryAccess, flags) == 68);
static_assert(offsetof(FfmTdmMemoryAccess, tile_dim0) == 72);
static_assert(offsetof(FfmTdmMemoryAccess, tile_dim1) == 76);
static_assert(offsetof(FfmTdmMemoryAccess, data_size) == 80);
static_assert(offsetof(FfmTdmMemoryAccess, tensor_dim0_stride) == 88);
static_assert(offsetof(FfmTdmMemoryAccess, tensor_dim1_stride) == 96);

static_assert(sizeof(FfmHostApi) == 16);
static_assert(alignof(FfmHostApi) == 8);
static_assert(offsetof(FfmHostApi, api_version) == 0);
static_assert(offsetof(FfmHostApi, log) == 8);

static_assert(sizeof(FfmObserverPluginApiPrefix) == 168);
static_assert(alignof(FfmObserverPluginApiPrefix) == 8);
static_assert(offsetof(FfmObserverPluginApiPrefix, api_version) == 0);
static_assert(offsetof(FfmObserverPluginApiPrefix, name) == 8);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_init) == 16);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_dispatch_begin) == 24);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_dispatch_end) == 32);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_cluster_begin) == 40);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_cluster_end) == 48);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_workgroup_begin) == 56);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_workgroup_end) == 64);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_wavegroup_begin) == 72);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_wavegroup_end) == 80);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_wave_begin) == 88);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_wave_end) == 96);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_instruction) == 104);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_resource_access) == 112);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_barrier_signal) == 120);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_barrier_wait) == 128);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_barrier_complete) == 136);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_shutdown) == 144);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_memory_access) == 152);
static_assert(offsetof(FfmObserverPluginApiPrefix, on_tdm_memory_access) == 160);

static_assert(encode_memory_flags(false, false, false) == 0x00);
static_assert(encode_memory_flags(true, false, false) == 0x01);
static_assert(encode_memory_flags(false, true, false) == 0x02);
static_assert(encode_memory_flags(true, true, false) == 0x03);
static_assert(encode_memory_flags(false, false, true) == 0x04);
static_assert(encode_memory_flags(true, false, true) == 0x05);
static_assert(encode_memory_flags(false, true, true) == 0x06);
static_assert(encode_memory_flags(true, true, true) == 0x07);
static_assert(encode_tdm_flags(false, false) == 0x00);
static_assert(encode_tdm_flags(true, false) == 0x01);
static_assert(encode_tdm_flags(false, true) == 0x02);
static_assert(encode_tdm_flags(true, true) == 0x03);

} // namespace rocjitsu::plugins::perfsim::observer_abi_v13
