// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/perfsim/plugin.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/encodings.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/plugins/perfsim/observer_abi_v13.h"
#include "util/dynamic_loader.h"

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace rocjitsu::plugins::perfsim {
namespace {

using namespace observer_abi_v13;

constexpr size_t kDefaultMaxStagedBytes = 256 * 1024 * 1024;
constexpr uint32_t kOldestCompatibleApiVersion = 8;
static_assert(FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION == 13,
              "the Perfsim adapter constructs FFM API v13 payload supersets");
static_assert(kOldestCompatibleApiVersion >= FFM_OBSERVER_PLUGIN_OLDEST_SUPPORTED_API_VERSION);
static_assert(kOldestCompatibleApiVersion <= FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION);

class InstanceClaim {
public:
  InstanceClaim() {
    if (claimed().test_and_set(std::memory_order_acq_rel))
      throw std::runtime_error("only one Perfsim plugin instance may be active in a process");
    owns_claim_ = true;
  }

  ~InstanceClaim() {
    if (owns_claim_)
      claimed().clear(std::memory_order_release);
  }

  InstanceClaim(const InstanceClaim &) = delete;
  InstanceClaim &operator=(const InstanceClaim &) = delete;

private:
  static std::atomic_flag &claimed() {
    static std::atomic_flag value = ATOMIC_FLAG_INIT;
    return value;
  }

  bool owns_claim_ = false;
};

struct BackendLibraryRegistry {
  std::mutex mutex;
  std::unordered_map<std::string, util::LibraryHandle> by_path;
};

BackendLibraryRegistry &backend_libraries() {
  // The backend mapping intentionally outlives every adapter instance. Some
  // Perfsim builds perform LLVM-global cleanup from an ELF destructor, which is
  // unsafe while the RocJITsu host remains alive.
  static auto *registry = new BackendLibraryRegistry;
  return *registry;
}

util::LibraryHandle retain_backend_library(const std::string &path) {
  BackendLibraryRegistry &registry = backend_libraries();
  std::lock_guard lock(registry.mutex);

  if (const auto iter = registry.by_path.find(path); iter != registry.by_path.end())
    return iter->second;

  util::LibraryHandle handle = util::open_library(path.c_str());
  if (!handle)
    throw std::runtime_error(
        std::format("cannot load Perfsim backend '{}': {}", path, util::last_library_error()));
  registry.by_path.emplace(path, handle);
  return handle;
}

class RetainedSharedLibrary {
public:
  explicit RetainedSharedLibrary(const std::string &path) : handle_(retain_backend_library(path)) {}

  RetainedSharedLibrary(const RetainedSharedLibrary &) = delete;
  RetainedSharedLibrary &operator=(const RetainedSharedLibrary &) = delete;

  util::LibraryHandle get() const { return handle_; }

private:
  util::LibraryHandle handle_ = nullptr;
};

struct AdapterConfig {
  std::string library_path;
  size_t max_staged_bytes = kDefaultMaxStagedBytes;
};

AdapterConfig parse_config(const char *config_json) {
  if (!config_json)
    throw std::invalid_argument("Perfsim plugin configuration is null");

  flexbuffers::Builder builder;
  flatbuffers::Parser parser;
  if (!parser.ParseFlexBuffer(config_json, nullptr, &builder))
    throw std::invalid_argument("Perfsim plugin configuration is not valid JSON");

  const auto root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsMap())
    throw std::invalid_argument("Perfsim plugin configuration must be an object");
  const auto config = root.AsMap();
  const auto path = config["library_path"];
  if (!path.IsString() || path.AsString().size() == 0)
    throw std::invalid_argument("Perfsim plugin requires a non-empty string 'library_path'");
  AdapterConfig result;
  result.library_path.assign(path.AsString().c_str(), path.AsString().size());
  if (!std::filesystem::path(result.library_path).is_absolute())
    throw std::invalid_argument("Perfsim plugin 'library_path' must be absolute");

  const auto max_staged_bytes = config["max_staged_bytes"];
  if (!max_staged_bytes.IsNull()) {
    if (!max_staged_bytes.IsIntOrUint() ||
        (max_staged_bytes.IsInt() && max_staged_bytes.AsInt64() <= 0)) {
      throw std::invalid_argument("Perfsim plugin 'max_staged_bytes' must be a positive integer");
    }
    const uint64_t value = max_staged_bytes.AsUInt64();
    if (value == 0)
      throw std::invalid_argument("Perfsim plugin 'max_staged_bytes' must be a positive integer");
    if (value > std::numeric_limits<size_t>::max())
      throw std::invalid_argument("Perfsim plugin 'max_staged_bytes' is too large");
    result.max_staged_bytes = static_cast<size_t>(value);
  }
  return result;
}

uint64_t physical_wave_key(uint32_t compute_unit_id, uint32_t wavefront_id) {
  return (static_cast<uint64_t>(compute_unit_id) << 32) | wavefront_id;
}

uint64_t cluster_id(const std::array<uint32_t, 3> &coordinate) {
  // FFM performs this packing in uint32_t. Preserve its intentional wrapping
  // before widening to EntityId.
  uint32_t packed = coordinate[0];
  packed += coordinate[1] * uint32_t{1000};
  packed += coordinate[2] * uint32_t{1000000};
  return packed;
}

FfmWaveInfo make_wave_info(uint32_t dispatch_id, const std::array<uint32_t, 3> &coordinate,
                           uint32_t wave_in_group) {
  const EntityId cid = cluster_id(coordinate);
  return FfmWaveInfo{FfmWorkgroupInfo{FfmClusterInfo{FfmDispatchInfo{dispatch_id}, cid}, 0},
                     wave_in_group % 4, wave_in_group};
}

bool is_cdna5_smem(const Instruction &inst) {
  return inst.encoding_id() >= cdna5::encoding::kSmem &&
         inst.encoding_id() <= cdna5::encoding::kSmemHi7;
}

FfmInstructionCounters instruction_counters(const Instruction &inst) {
  FfmInstructionCounters counters{};
  const std::string_view mnemonic = inst.mnemonic();

  if (inst.is_mfma() || mnemonic.starts_with("v_wmma_") || mnemonic.starts_with("v_smfmac_") ||
      mnemonic.starts_with("v_swmmac_")) {
    counters.valu_count = 1;
    counters.xdl_valu_count = 1;
  } else if (mnemonic.starts_with("ds_")) {
    counters.lds_count = 1;
  } else if (mnemonic.starts_with("flat_")) {
    // FFM deliberately leaves flat_count at zero. FLAT issues through TEX and
    // is also charged to LDS because its resource is unresolved at issue time.
    counters.lds_count = 1;
    counters.tex_count = 1;
  } else if (mnemonic.starts_with("global_") || mnemonic.starts_with("buffer_") ||
             mnemonic.starts_with("tbuffer_") || mnemonic.starts_with("image_") ||
             mnemonic.starts_with("scratch_") || mnemonic.starts_with("tensor_") ||
             mnemonic.starts_with("cluster_")) {
    counters.tex_count = 1;
    if (mnemonic.starts_with("global_") || mnemonic.starts_with("scratch_") ||
        mnemonic.starts_with("cluster_")) {
      if (mnemonic.find("store") != std::string_view::npos)
        counters.global_scratch_store = 1;
      else
        counters.global_scratch_load = 1;
    }
  } else if (is_cdna5_smem(inst)) {
    counters.smem_count = 1;
  } else if (mnemonic.starts_with("s_")) {
    counters.salu_count = 1;
  } else if (mnemonic.starts_with("v_")) {
    counters.valu_count = 1;
  }

  return counters;
}

FfmWaitInfo wait_info(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_wait_loadcnt")
    return {FFM_WAIT_TYPE_LOAD, FFM_WAIT_NAME_S_WAIT_LOADCNT};
  if (mnemonic == "s_wait_xcnt")
    return {FFM_WAIT_TYPE_LOAD, FFM_WAIT_NAME_S_WAIT_XCNT};
  if (mnemonic == "s_wait_loadcnt_dscnt")
    return {FFM_WAIT_TYPE_LOAD, FFM_WAIT_NAME_S_WAIT_LOADCNT_DSCNT};
  if (mnemonic == "s_wait_samplecnt")
    return {FFM_WAIT_TYPE_SAMPLE, FFM_WAIT_NAME_S_WAIT_SAMPLECNT};
  if (mnemonic == "s_wait_dscnt")
    return {FFM_WAIT_TYPE_LDS, FFM_WAIT_NAME_S_WAIT_DSCNT};
  if (mnemonic == "s_wait_storecnt_dscnt")
    return {FFM_WAIT_TYPE_LDS, FFM_WAIT_NAME_S_WAIT_STORECNT_DSCNT};
  if (mnemonic == "s_wait_kmcnt")
    return {FFM_WAIT_TYPE_LDS, FFM_WAIT_NAME_S_WAIT_KMCNT};
  if (mnemonic == "s_wait_tensorcnt")
    return {FFM_WAIT_TYPE_LDS, FFM_WAIT_NAME_S_WAIT_TENSORCNT};
  if (mnemonic == "s_wait_idle")
    return {FFM_WAIT_TYPE_VM, FFM_WAIT_NAME_S_WAIT_IDLE};
  if (mnemonic == "s_waitcnt")
    return {FFM_WAIT_TYPE_VM, FFM_WAIT_NAME_S_WAITCNT};
  if (mnemonic == "s_barrier_wait")
    return {FFM_WAIT_TYPE_BARRIER, FFM_WAIT_NAME_S_BARRIER_WAIT};
  return {FFM_WAIT_TYPE_NONE, FFM_WAIT_NAME_NONE};
}

bool ffm_omits_memory_access(std::string_view mnemonic) {
  return mnemonic == "ds_append" || mnemonic == "ds_consume" || mnemonic == "ds_load_tr4_b64" ||
         mnemonic == "ds_load_tr6_b96" || mnemonic == "ds_load_tr8_b64" ||
         mnemonic == "ds_atomic_async_barrier_arrive_b64" ||
         mnemonic == "ds_atomic_barrier_arrive_rtn_b64" || mnemonic == "global_load_block" ||
         mnemonic == "global_store_block";
}

uint32_t ffm_memory_data_size(const amdgpu::MemoryAccessObservation &access) {
  const uint64_t bytes = access.bytes_per_lane();
  if (access.decoded_space == amdgpu::DecodedMemorySpace::LOCAL ||
      access.mnemonic.starts_with("global_"))
    return static_cast<uint32_t>(std::max<uint64_t>(4, bytes));
  if (access.mnemonic.starts_with("buffer_") || access.mnemonic.starts_with("tbuffer_"))
    return static_cast<uint32_t>(std::max<uint64_t>(1, bytes / 4) * 4);
  return static_cast<uint32_t>(bytes);
}

struct BeginEvent {
  FfmDispatchMetadata metadata{};
};

struct EndEvent {
  FfmDispatchMetadata metadata{};
};

struct InstructionEvent {
  EntityId instruction_id = 0;
  FfmWaveInfo wave_info{};
  uint64_t pc = 0;
  std::array<uint32_t, 4> raw_isa{};
  FfmInstructionCounters counters{};
  FfmWaitInfo wait{FFM_WAIT_TYPE_NONE, FFM_WAIT_NAME_NONE};
};

struct TdmEvent {
  EntityId instruction_id = 0;
  FfmWaveInfo wave_info{};
  std::unique_ptr<uint64_t[]> addresses;
  uint32_t num_addresses = 0;
  uint32_t data_size_bytes = 0;
  uint32_t tile_dim0 = 0;
  uint32_t tile_dim1 = 0;
  uint32_t data_size = 0;
  int64_t tensor_dim0_stride = 0;
  int64_t tensor_dim1_stride = 0;
  bool is_read = false;
  bool is_write = false;
};

using EventPayload =
    std::variant<BeginEvent, InstructionEvent, FfmMemoryAccess, TdmEvent, EndEvent>;

struct OrderedEvent {
  uint32_t dispatch_id = 0;
  EventPayload payload;
};

struct StagedEvent {
  OrderedEvent event;
  size_t dynamic_bytes = 0;
};
static_assert(std::is_nothrow_move_constructible_v<StagedEvent>);
static_assert(std::is_nothrow_move_assignable_v<StagedEvent>);

struct EventChunk {
  std::vector<StagedEvent> events;
  size_t capacity_bytes = 0;
};

// Match the original adapter's allocation granularity while charging each
// retained chunk's actual capacity against the configured budget.
constexpr size_t kEventChunkCapacity = 4096;

size_t staged_dynamic_size(const OrderedEvent &event) {
  if (const auto *tdm = std::get_if<TdmEvent>(&event.payload)) {
    const uint64_t bytes = static_cast<uint64_t>(tdm->num_addresses) * sizeof(uint64_t);
    if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
      if (bytes > std::numeric_limits<size_t>::max())
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(bytes);
  }
  return 0;
}

struct PerfsimWavefrontState final : WavefrontState {
  FfmWaveInfo wave_info{};
  uint32_t compute_unit_id = 0;
  uint32_t physical_wavefront_id = 0;
  uint32_t rocjitsu_workgroup_id = 0;
  uint32_t queue_id = 0;
  uint32_t process_id = 0;
  uint32_t lds_base = 0;
  uint32_t next_instruction_id = 0;
  uint64_t current_instruction_id = 0;
  uint64_t current_pc = 0;
  std::array<uint32_t, 4> current_instruction_encoding{};
  uint32_t current_encoding_dwords = 0;
  FfmWaitInfo current_wait{FFM_WAIT_TYPE_NONE, FFM_WAIT_NAME_NONE};
  bool has_current_instruction = false;
  bool saw_instruction = false;
  bool last_instruction_terminates = false;
};

struct DispatchState {
  FfmDispatchMetadata metadata{};
  std::string dispatch_name;
  bool metadata_seen = false;
  bool begun = false;
  bool ended = false;
  bool supported = true;
  bool diagnostic_emitted = false;
  std::string rejection_reason;
  size_t live_waves = 0;
};

std::optional<std::string> validate_dispatch(const KernelDispatchInfo &info) {
  if (info.code_target != ROCJITSU_CODE_TARGET_GFX1250)
    return std::format("code target is not gfx1250 ({})", static_cast<int>(info.code_target));
  if (info.wave_size != 32)
    return std::format("gfx1250 dispatch has unsupported wave size {}", info.wave_size);
  if (info.cluster_size_x != 1 || info.cluster_size_y != 1 || info.cluster_size_z != 1)
    return std::format("multi-workgroup cluster [{},{},{}] is unsupported", info.cluster_size_x,
                       info.cluster_size_y, info.cluster_size_z);
  if (info.workgroup_size_x == 0 || info.workgroup_size_y == 0 || info.workgroup_size_z == 0 ||
      info.grid_size_x == 0 || info.grid_size_y == 0 || info.grid_size_z == 0)
    return "grid and workgroup dimensions must be nonzero";
  if (info.wfs_per_workgroup == 0)
    return "dispatch has no waves per workgroup";
  return std::nullopt;
}

FfmDispatchMetadata make_dispatch_metadata(const KernelDispatchInfo &info) {
  FfmDispatchMetadata metadata{};
  metadata.dispatch_info.dispatch_id = info.dispatch_id;
  metadata.vgpr_count = info.vgprs_per_wf;
  // The FFM contract reports the fixed MI400 SGPR file width here, not the
  // kernel's allocated SGPR count.
  metadata.sgpr_count = 128;
  metadata.lds_size_bytes = info.lds_size_bytes;
  metadata.wave_size = info.wave_size;
  metadata.num_waves_per_wg = info.wfs_per_workgroup;
  metadata.grid_size[0] = info.grid_size_x;
  metadata.grid_size[1] = info.grid_size_y;
  metadata.grid_size[2] = info.grid_size_z;
  metadata.workgroup_size[0] = info.workgroup_size_x;
  metadata.workgroup_size[1] = info.workgroup_size_y;
  metadata.workgroup_size[2] = info.workgroup_size_z;
  return metadata;
}

} // namespace

struct PerfsimPlugin::Impl {
  explicit Impl(PerfsimPlugin &owner, const char *config_json)
      : owner(owner), config(parse_config(config_json)), library(config.library_path) {
    const auto get_api = util::lookup_symbol<FfmObserverPluginGetApiFn>(
        library.get(), "ffm_observer_plugin_get_api");
    if (!get_api)
      throw std::runtime_error(
          std::format("Perfsim backend '{}' is missing ffm_observer_plugin_get_api: {}",
                      config.library_path, util::last_library_error()));

    FfmObserverPluginApi *raw_api = nullptr;
    for (uint32_t requested_version = FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION;;
         --requested_version) {
      raw_api = invoke_foreign_abi(get_api, requested_version);
      if (raw_api) {
        uint32_t returned_version = 0;
        std::memcpy(&returned_version, raw_api, sizeof(returned_version));
        if (returned_version < kOldestCompatibleApiVersion ||
            returned_version > requested_version) {
          throw std::runtime_error(std::format(
              "Perfsim backend returned incompatible API version {} for version {} request; "
              "supported versions are {} through {}",
              returned_version, requested_version, kOldestCompatibleApiVersion,
              FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION));
        }
        negotiated_api_version = returned_version;
        std::memcpy(&api, raw_api, sizeof(api));
        break;
      }
      if (requested_version == kOldestCompatibleApiVersion)
        break;
    }
    if (!raw_api)
      throw std::runtime_error(
          std::format("Perfsim backend rejected supported FFM API versions {} through {}",
                      kOldestCompatibleApiVersion, FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION));

    require_callback(api.on_init, "on_init");
    require_callback(api.on_dispatch_begin, "on_dispatch_begin");
    require_callback(api.on_dispatch_end, "on_dispatch_end");
    require_callback(api.on_instruction, "on_instruction");
    require_callback(api.on_memory_access, "on_memory_access");
    require_callback(api.on_tdm_memory_access, "on_tdm_memory_access");
    require_callback(api.on_shutdown, "on_shutdown");
  }

  ~Impl() {
    shutdown();
    // The retained backend remains mapped after on_shutdown and for the rest
    // of the process lifetime.
  }

  template <typename Callback> void require_callback(Callback callback, const char *name) {
    if (!callback)
      throw std::runtime_error(
          std::format("Perfsim backend is missing required callback {}", name));
  }

  static std::mutex &host_log_mutex() {
    // The backend mapping is intentionally process-lived and may retain the
    // callback pointer. Keep its serialization object process-lived as well.
    static auto *value = new std::mutex;
    return *value;
  }

  static Impl *&host_log_instance() {
    static Impl *value = nullptr;
    return value;
  }

  static const char *host_log_level_name(FfmLogLevel level) {
    switch (level) {
    case FFM_LOG_DEBUG:
      return "debug";
    case FFM_LOG_INFO:
      return "info";
    case FFM_LOG_WARN:
      return "warn";
    case FFM_LOG_ERROR:
      return "error";
    default:
      return "unknown";
    }
  }

  static void host_log(FfmLogLevel level, const char *message) noexcept {
    try {
      std::lock_guard lock(host_log_mutex());
      Impl *instance = host_log_instance();
      if (!instance)
        return;
      instance->owner.sink().write(
          std::format("[perfsim:{}] {}\n", host_log_level_name(level), message ? message : ""));
    } catch (...) {
      // The logger is a C ABI callback and must not propagate an exception into
      // the backend. A failed diagnostic must not abort emulation.
    }
  }

  void activate_host_log() {
    std::lock_guard lock(host_log_mutex());
    assert(host_log_instance() == nullptr);
    host_log_instance() = this;
  }

  void deactivate_host_log() {
    std::lock_guard lock(host_log_mutex());
    if (host_log_instance() == this)
      host_log_instance() = nullptr;
  }

  void write_sink(std::string_view message) {
    std::lock_guard lock(host_log_mutex());
    owner.sink().write(message);
  }

  void init() {
    if (initialized || shutdown_called)
      return;
    const bool supports_host_log = negotiated_api_version >= 12;
    if (supports_host_log)
      activate_host_log();
    const FfmHostApi host{negotiated_api_version, supports_host_log ? &host_log : nullptr};
    try {
      invoke_foreign_abi(api.on_init, &host);
    } catch (...) {
      if (supports_host_log)
        deactivate_host_log();
      throw;
    }
    initialized = true;
  }

  void shutdown() {
    if (shutdown_called)
      return;

    std::vector<uint32_t> incomplete_dispatches;
    for (const auto &[dispatch_id, state] : dispatches)
      if (state.begun && !state.ended)
        incomplete_dispatches.push_back(dispatch_id);
    for (uint32_t dispatch_id : incomplete_dispatches)
      reject(dispatch_id, "dispatch was incomplete at plugin shutdown");
    replay_blockers.clear();
    drain_epoch(/*shutdown=*/true);
    physical_waves.clear();

    if (initialized)
      invoke_foreign_abi(api.on_shutdown);
    deactivate_host_log();
    shutdown_called = true;
  }

  void reject(uint32_t dispatch_id, std::string reason) {
    auto &state = dispatches[dispatch_id];
    if (!state.supported)
      return;
    state.supported = false;
    state.rejection_reason = std::move(reason);
    replay_blockers.erase(dispatch_id);
    purge_events(dispatch_id);
    drain_epoch(/*shutdown=*/false);
  }

  DispatchState *active_dispatch(uint32_t dispatch_id) {
    auto iter = dispatches.find(dispatch_id);
    if (iter == dispatches.end() || !iter->second.supported || !iter->second.begun ||
        iter->second.ended)
      return nullptr;
    return &iter->second;
  }

  PerfsimWavefrontState *find_wave(uint32_t compute_unit_id, uint32_t wavefront_id) {
    const auto iter = physical_waves.find(physical_wave_key(compute_unit_id, wavefront_id));
    return iter == physical_waves.end() ? nullptr : iter->second;
  }

  void replay(const OrderedEvent &event) const {
    std::visit(
        [this](const auto &payload) {
          using T = std::decay_t<decltype(payload)>;
          if constexpr (std::is_same_v<T, BeginEvent>) {
            invoke_foreign_abi(api.on_dispatch_begin, &payload.metadata);
          } else if constexpr (std::is_same_v<T, EndEvent>) {
            invoke_foreign_abi(api.on_dispatch_end, &payload.metadata);
          } else if constexpr (std::is_same_v<T, InstructionEvent>) {
            const FfmObserverInstruction instruction{
                payload.pc,
                {payload.raw_isa[0], payload.raw_isa[1], payload.raw_isa[2], payload.raw_isa[3]}};
            const FfmInstructionInfo info{payload.instruction_id, payload.wave_info, instruction,
                                          payload.counters, payload.wait};
            invoke_foreign_abi(api.on_instruction, &info);
          } else if constexpr (std::is_same_v<T, FfmMemoryAccess>) {
            invoke_foreign_abi(api.on_memory_access, &payload);
          } else if constexpr (std::is_same_v<T, TdmEvent>) {
            FfmTdmMemoryAccess access{};
            access.instruction_id = payload.instruction_id;
            access.wave_info = payload.wave_info;
            access.num_addresses = payload.num_addresses;
            access.addresses = payload.addresses.get();
            access.data_size_bytes = payload.data_size_bytes;
            access.flags = encode_tdm_flags(payload.is_read, payload.is_write);
            access.tile_dim0 = payload.tile_dim0;
            access.tile_dim1 = payload.tile_dim1;
            access.data_size = payload.data_size;
            access.tensor_dim0_stride = payload.tensor_dim0_stride;
            access.tensor_dim1_stride = payload.tensor_dim1_stride;
            invoke_foreign_abi(api.on_tdm_memory_access, &access);
          }
        },
        event.payload);
  }

  bool can_stage(uint32_t dispatch_id, size_t bytes) {
    const auto iter = dispatches.find(dispatch_id);
    if (iter == dispatches.end() || !iter->second.supported)
      return false;
    if (bytes > config.max_staged_bytes || staged_bytes > config.max_staged_bytes - bytes) {
      reject(dispatch_id,
             std::format("staging budget of {} bytes exceeded", config.max_staged_bytes));
      return false;
    }
    return true;
  }

  bool has_staging_slot() const {
    return !event_chunks.empty() &&
           event_chunks.back().events.size() < event_chunks.back().events.capacity();
  }

  bool can_stage_event(uint32_t dispatch_id, size_t dynamic_bytes) {
    const size_t slot_bytes = has_staging_slot() ? 0 : sizeof(StagedEvent);
    if (slot_bytes > config.max_staged_bytes ||
        dynamic_bytes > config.max_staged_bytes - slot_bytes) {
      reject(dispatch_id,
             std::format("staging budget of {} bytes exceeded", config.max_staged_bytes));
      return false;
    }
    return can_stage(dispatch_id, slot_bytes + dynamic_bytes);
  }

  bool record_event(OrderedEvent event) {
    const size_t dynamic_bytes = staged_dynamic_size(event);
    if (!can_stage_event(event.dispatch_id, dynamic_bytes))
      return false;

    if (has_staging_slot()) {
      event_chunks.back().events.push_back({std::move(event), dynamic_bytes});
      staged_bytes += dynamic_bytes;
      return true;
    }

    const size_t available_for_capacity = config.max_staged_bytes - staged_bytes - dynamic_bytes;
    const size_t requested_capacity =
        std::min(kEventChunkCapacity, available_for_capacity / sizeof(StagedEvent));
    assert(requested_capacity != 0);

    EventChunk chunk;
    chunk.events.reserve(requested_capacity);
    if (chunk.events.capacity() > available_for_capacity / sizeof(StagedEvent)) {
      reject(event.dispatch_id,
             std::format("staging budget of {} bytes exceeded", config.max_staged_bytes));
      return false;
    }
    chunk.capacity_bytes = chunk.events.capacity() * sizeof(StagedEvent);
    chunk.events.push_back({std::move(event), dynamic_bytes});
    event_chunks.push_back(std::move(chunk));
    staged_bytes += event_chunks.back().capacity_bytes + dynamic_bytes;
    return true;
  }

  void release_chunk(size_t index) {
    assert(index < event_chunks.size());
    assert(event_chunks[index].events.empty());
    assert(event_chunks[index].capacity_bytes <= staged_bytes);
    staged_bytes -= event_chunks[index].capacity_bytes;
    event_chunks.erase(event_chunks.begin() + index);
  }

  void compact_event_chunks() {
    size_t index = 0;
    while (index < event_chunks.size()) {
      while (index + 1 < event_chunks.size() &&
             event_chunks[index].events.size() < event_chunks[index].events.capacity()) {
        auto &destination = event_chunks[index].events;
        auto &source = event_chunks[index + 1].events;
        const size_t move_count =
            std::min(destination.capacity() - destination.size(), source.size());
        destination.insert(destination.end(), std::make_move_iterator(source.begin()),
                           std::make_move_iterator(source.begin() + move_count));
        source.erase(source.begin(), source.begin() + move_count);
        if (source.empty()) {
          release_chunk(index + 1);
          continue;
        }
      }
      if (event_chunks[index].events.empty()) {
        release_chunk(index);
        continue;
      }
      ++index;
    }
  }

  void purge_events(uint32_t dispatch_id) {
    size_t removed_dynamic_bytes = 0;
    for (EventChunk &chunk : event_chunks) {
      std::erase_if(chunk.events, [&](const StagedEvent &staged) {
        if (staged.event.dispatch_id != dispatch_id)
          return false;
        removed_dynamic_bytes += staged.dynamic_bytes;
        return true;
      });
    }
    assert(removed_dynamic_bytes <= staged_bytes);
    staged_bytes -= removed_dynamic_bytes;
    compact_event_chunks();
  }

  void drain_epoch(bool shutdown) {
    if (!replay_blockers.empty())
      return;

    while (!event_chunks.empty()) {
      EventChunk chunk = std::move(event_chunks.front());
      event_chunks.pop_front();
      assert(chunk.capacity_bytes <= staged_bytes);
      staged_bytes -= chunk.capacity_bytes;
      for (StagedEvent &staged : chunk.events) {
        assert(staged.dynamic_bytes <= staged_bytes);
        staged_bytes -= staged.dynamic_bytes;
        OrderedEvent &event = staged.event;
        const auto iter = dispatches.find(event.dispatch_id);
        if (iter != dispatches.end() && iter->second.supported && iter->second.ended)
          replay(event);
      }
    }
    assert(staged_bytes == 0);

    for (auto &[dispatch_id, state] : dispatches) {
      if (!state.supported && !state.diagnostic_emitted) {
        write_sink(std::format("[rocjitsu:perfsim] skipped dispatch {}: {}\n", dispatch_id,
                               state.rejection_reason));
        state.diagnostic_emitted = true;
      }
    }
    for (auto iter = dispatches.begin(); iter != dispatches.end();) {
      if (shutdown || iter->second.ended)
        iter = dispatches.erase(iter);
      else
        ++iter;
    }
  }

  void record_memory(PerfsimWavefrontState &wave, const amdgpu::MemoryAccessObservation &access,
                     uint64_t lane_mask, FfmResourceType resource,
                     std::span<const uint64_t> addresses) {
    if (lane_mask == 0)
      return;

    FfmMemoryAccess event{};
    event.instruction_id = wave.current_instruction_id;
    event.wave_info = wave.wave_info;
    event.exec_mask = lane_mask;
    event.wave_size = access.wavefront_size;
    for (uint32_t lane = 0; lane < access.wavefront_size; ++lane)
      if ((lane_mask & (uint64_t{1} << lane)) != 0)
        event.addresses[lane] = addresses[lane];
    event.data_size_bytes = ffm_memory_data_size(access);
    event.resource_type = resource;
    const bool is_atomic = access.atomic_op != amdgpu::AtomicOp::NONE;
    event.flags =
        encode_memory_flags(is_atomic, access.is_load || is_atomic, !access.is_load || is_atomic);
    record_event({access.dispatch_id, std::move(event)});
  }

  bool record_local_memory(PerfsimWavefrontState &wave,
                           const amdgpu::MemoryAccessObservation &access, uint64_t lane_mask,
                           std::span<const uint64_t> addresses) {
    std::array<uint64_t, FFM_MAX_WAVE_SIZE> local_addresses{};
    for (uint32_t lane = 0; lane < access.wavefront_size; ++lane) {
      if ((lane_mask & (uint64_t{1} << lane)) == 0)
        continue;
      if (addresses[lane] < wave.lds_base) {
        reject(access.dispatch_id, "LDS address precedes the workgroup allocation base");
        return false;
      }
      local_addresses[lane] = addresses[lane] - wave.lds_base;
    }
    record_memory(wave, access, lane_mask, FFM_RESOURCE_LDS, local_addresses);
    return true;
  }

  void memory_access(const amdgpu::MemoryAccessObservation &access) {
    // Current FFM does not emit ON_MEMORY_ACCESS for scalar memory.
    if (access.route == amdgpu::MemoryRoute::SCALAR)
      return;
    // FFM executes these LDS operations without recording a regular-memory
    // event. Preserve the instruction callback but suppress RocJITsu's routing
    // observation.
    if (ffm_omits_memory_access(access.mnemonic))
      return;

    DispatchState *dispatch = active_dispatch(access.dispatch_id);
    if (!dispatch) {
      reject(access.dispatch_id, "memory callback arrived outside an active dispatch");
      return;
    }

    PerfsimWavefrontState *wave = find_wave(access.compute_unit_id, access.wavefront_id);
    if (!wave ||
        wave->wave_info.workgroup_info.cluster_info.dispatch_info.dispatch_id !=
            access.dispatch_id ||
        wave->rocjitsu_workgroup_id != access.workgroup_id || wave->queue_id != access.queue_id ||
        wave->process_id != access.process_id || !wave->has_current_instruction ||
        wave->current_pc != access.pc) {
      reject(access.dispatch_id, "memory callback could not be matched to its issuing wave");
      return;
    }

    if (access.decoded_space == amdgpu::DecodedMemorySpace::SCRATCH) {
      reject(access.dispatch_id, "dedicated SCRATCH instruction addresses are not FFM-compatible");
      return;
    }
    if (access.route == amdgpu::MemoryRoute::UNKNOWN ||
        access.decoded_space == amdgpu::DecodedMemorySpace::UNKNOWN) {
      reject(access.dispatch_id, "memory route or decoded address space is unknown");
      return;
    }
    if (access.wavefront_size != 32 || access.addresses.size() != access.wavefront_size ||
        (!access.pre_routing_addresses.empty() &&
         access.pre_routing_addresses.size() != access.wavefront_size) ||
        (!access.secondary_addresses.empty() &&
         access.secondary_addresses.size() != access.wavefront_size) ||
        (!access.element_lane_masks.empty() &&
         access.element_lane_masks.size() != access.elements_per_lane) ||
        access.element_size_bytes == 0 || access.elements_per_lane == 0 ||
        access.bytes_per_lane() > std::numeric_limits<uint32_t>::max()) {
      reject(access.dispatch_id, "memory observation has malformed dimensions or addresses");
      return;
    }

    const uint64_t wave_mask = access.wavefront_lane_mask();
    if ((access.active_lane_mask & ~wave_mask) != 0 ||
        (access.architectural_exec_lane_mask & ~wave_mask) != 0 ||
        (access.architectural_exec_lane_mask & ~access.active_lane_mask) != 0 ||
        (access.valid_lane_mask & ~wave_mask) != 0 ||
        (access.valid_lane_mask & ~access.active_lane_mask) != 0 ||
        (access.request_lane_mask & ~access.valid_lane_mask) != 0 ||
        (access.scratch_lane_mask & ~access.request_lane_mask) != 0 ||
        (access.flat_local_lane_mask & ~access.request_lane_mask) != 0 ||
        (access.flat_dds_lane_mask & ~access.request_lane_mask) != 0 ||
        (access.flat_local_lane_mask & access.scratch_lane_mask) != 0 ||
        (access.flat_dds_lane_mask & access.scratch_lane_mask) != 0 ||
        (access.flat_dds_lane_mask & access.flat_local_lane_mask) != 0) {
      reject(access.dispatch_id, "memory observation has inconsistent lane masks");
      return;
    }
    // FFM has no per-element validity field. Its native gfx1250 VBUFFER
    // producer selects a lane when any component is in bounds, then reports
    // the instruction's nominal dword width (including for partially-OOB
    // B64/B96/B128 accesses). Validate RocJITsu's richer masks here, but
    // intentionally forward request_lane_mask and the nominal width below to
    // preserve the direct-FFM callback contract.
    for (uint64_t element_mask : access.element_lane_masks) {
      if ((element_mask & ~access.valid_lane_mask) != 0) {
        reject(access.dispatch_id, "memory observation has an invalid per-element lane mask");
        return;
      }
    }

    const uint64_t requests = access.request_lane_mask;
    if (requests == 0)
      return;

    switch (access.decoded_space) {
    case amdgpu::DecodedMemorySpace::LOCAL:
      if (access.route != amdgpu::MemoryRoute::LOCAL || access.normalized_to_local ||
          access.scratch_lane_mask != 0 || access.flat_local_lane_mask != 0 ||
          access.flat_dds_lane_mask != 0 || !access.pre_routing_addresses.empty()) {
        reject(access.dispatch_id, "explicit LDS observation has inconsistent routing");
        return;
      }
      {
        uint64_t callback_requests = requests;
        if (access.mnemonic == "ds_load_tr16_b128") {
          callback_requests = access.architectural_exec_lane_mask;
          if ((callback_requests & ~requests) != 0) {
            reject(access.dispatch_id,
                   "DS TR16 architectural EXEC is not a subset of requesting lanes");
            return;
          }
        }
        if (!record_local_memory(*wave, access, callback_requests, access.addresses))
          return;
        if (!access.secondary_addresses.empty() &&
            !record_local_memory(*wave, access, callback_requests, access.secondary_addresses))
          return;
      }
      return;

    case amdgpu::DecodedMemorySpace::GLOBAL:
      if (access.route != amdgpu::MemoryRoute::GLOBAL || access.normalized_to_local ||
          access.scratch_lane_mask != 0 || access.flat_local_lane_mask != 0 ||
          access.flat_dds_lane_mask != 0 || !access.pre_routing_addresses.empty() ||
          !access.secondary_addresses.empty()) {
        reject(access.dispatch_id, "explicit global observation has inconsistent routing");
        return;
      }
      record_memory(*wave, access, requests, FFM_RESOURCE_GLOBAL, access.addresses);
      return;

    case amdgpu::DecodedMemorySpace::FLAT:
      if (access.route != amdgpu::MemoryRoute::LOCAL &&
          access.route != amdgpu::MemoryRoute::GLOBAL) {
        reject(access.dispatch_id, "FLAT observation has an unsupported route");
        return;
      }
      {
        const uint32_t first_lane = static_cast<uint32_t>(std::countr_zero(requests));
        const uint64_t flat_shared_lane_mask =
            access.flat_local_lane_mask | access.flat_dds_lane_mask;
        const bool first_lane_is_shared =
            (flat_shared_lane_mask & (uint64_t{1} << first_lane)) != 0;
        if (access.route == amdgpu::MemoryRoute::LOCAL) {
          if (!access.normalized_to_local || !first_lane_is_shared ||
              access.pre_routing_addresses.size() != access.wavefront_size) {
            reject(access.dispatch_id,
                   "FLAT local route lacks consistent per-lane aperture metadata");
            return;
          }
        } else if (access.normalized_to_local || first_lane_is_shared ||
                   !access.pre_routing_addresses.empty()) {
          reject(access.dispatch_id, "FLAT global route has inconsistent aperture metadata");
          return;
        }
        if (!access.secondary_addresses.empty()) {
          reject(access.dispatch_id, "FLAT observation unexpectedly has secondary addresses");
          return;
        }

        const auto callback_addresses =
            access.pre_routing_addresses.empty() ? access.addresses : access.pre_routing_addresses;
        const uint64_t dds_requests = requests & access.flat_dds_lane_mask;
        if (dds_requests != 0 && (!access.is_load || access.atomic_op != amdgpu::AtomicOp::NONE)) {
          reject(access.dispatch_id, "FLAT DDS store or atomic is not FFM-compatible");
          return;
        }
        const uint64_t local_requests = requests & flat_shared_lane_mask;
        const uint64_t scratch_requests = requests & access.scratch_lane_mask;
        const uint64_t global_requests = requests & ~(local_requests | scratch_requests);
        if (access.atomic_op != amdgpu::AtomicOp::NONE && scratch_requests != 0) {
          reject(access.dispatch_id, "FLAT atomic resolving to scratch is not FFM-compatible");
          return;
        }

        struct ResourceRecord {
          uint64_t lane_mask;
          FfmResourceType resource;
        };
        std::array<ResourceRecord, 3> records{};
        size_t record_count = 0;
        const auto add_record = [&](uint64_t lane_mask, FfmResourceType resource) {
          if (lane_mask != 0)
            records[record_count++] = {lane_mask, resource};
        };
        add_record(global_requests, FFM_RESOURCE_GLOBAL);
        add_record(scratch_requests, FFM_RESOURCE_SCRATCH);
        add_record(local_requests, FFM_RESOURCE_LDS);
        std::sort(records.begin(), records.begin() + record_count,
                  [](const ResourceRecord &lhs, const ResourceRecord &rhs) {
                    return std::countr_zero(lhs.lane_mask) < std::countr_zero(rhs.lane_mask);
                  });
        for (size_t i = 0; i < record_count; ++i)
          record_memory(*wave, access, records[i].lane_mask, records[i].resource,
                        callback_addresses);
      }
      return;

    case amdgpu::DecodedMemorySpace::SCALAR:
    case amdgpu::DecodedMemorySpace::SCRATCH:
    case amdgpu::DecodedMemorySpace::UNKNOWN:
      break;
    }
    reject(access.dispatch_id, "unsupported memory observation");
  }

  void tensor_dma_memory_access(const amdgpu::TensorDmaMemoryAccessObservation &access) {
    DispatchState *dispatch = active_dispatch(access.dispatch_id);
    if (!dispatch) {
      reject(access.dispatch_id, "tensor-DMA callback arrived outside an active dispatch");
      return;
    }

    PerfsimWavefrontState *wave = find_wave(access.compute_unit_id, access.wavefront_id);
    if (!wave ||
        wave->wave_info.workgroup_info.cluster_info.dispatch_info.dispatch_id !=
            access.dispatch_id ||
        wave->rocjitsu_workgroup_id != access.workgroup_id || wave->queue_id != access.queue_id ||
        wave->process_id != access.process_id || !wave->has_current_instruction ||
        wave->current_pc != access.pc) {
      reject(access.dispatch_id, "tensor-DMA callback could not be matched to its issuing wave");
      return;
    }

    const bool is_load = access.mnemonic == "tensor_load_to_lds";
    const bool is_store = access.mnemonic == "tensor_store_from_lds";
    if ((!is_load && !is_store) || access.is_load != is_load || access.addresses.empty() ||
        !access.addresses.valid() || !std::has_single_bit(access.element_size_bytes) ||
        access.element_size_bytes > 8 || access.data_size > 3 ||
        access.element_size_bytes != (uint32_t{1} << access.data_size) || access.tile_dim0 == 0 ||
        access.tensor_dim0_stride < 0 || access.tensor_dim1_stride < 0 ||
        access.addresses.size() > std::numeric_limits<uint32_t>::max()) {
      reject(access.dispatch_id, "tensor-DMA observation is malformed");
      return;
    }
    if (access.addresses.size() > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
      reject(access.dispatch_id,
             std::format("staging budget of {} bytes exceeded", config.max_staged_bytes));
      return;
    }
    if (!can_stage_event(access.dispatch_id, access.addresses.size() * sizeof(uint64_t))) {
      return;
    }

    TdmEvent event;
    event.instruction_id = wave->current_instruction_id;
    event.wave_info = wave->wave_info;
    event.num_addresses = static_cast<uint32_t>(access.addresses.size());
    event.addresses = std::make_unique_for_overwrite<uint64_t[]>(event.num_addresses);
    if (!access.addresses.copy_to(
            std::span<uint64_t>(event.addresses.get(), event.num_addresses))) {
      reject(access.dispatch_id, "tensor-DMA observation is malformed");
      return;
    }
    event.data_size_bytes = access.element_size_bytes;
    event.tile_dim0 = access.tile_dim0;
    event.tile_dim1 = access.tile_dim1;
    event.data_size = access.data_size;
    event.tensor_dim0_stride = access.tensor_dim0_stride;
    event.tensor_dim1_stride = access.tensor_dim1_stride;
    event.is_read = is_load;
    event.is_write = is_store;
    record_event({access.dispatch_id, std::move(event)});
  }

  PerfsimPlugin &owner;
  InstanceClaim instance_claim;
  AdapterConfig config;
  RetainedSharedLibrary library;
  FfmObserverPluginApiPrefix api{};
  uint32_t negotiated_api_version = 0;
  bool initialized = false;
  bool shutdown_called = false;
  std::unordered_map<uint32_t, DispatchState> dispatches;
  // Only supported in-flight dispatches block ordered replay. Rejected
  // dispatch state remains until its real end callback, but it retains no
  // events and cannot indefinitely hold completed supported work.
  std::unordered_set<uint32_t> replay_blockers;
  std::unordered_map<uint64_t, PerfsimWavefrontState *> physical_waves;
  std::deque<EventChunk> event_chunks;
  size_t staged_bytes = 0;
};

PerfsimPlugin::PerfsimPlugin(const char *config_json)
    : ExecutionPlugin("perfsim"), impl_(std::make_unique<Impl>(*this, config_json)) {}

PerfsimPlugin::~PerfsimPlugin() { impl_->shutdown(); }

void PerfsimPlugin::onInit() { impl_->init(); }

void PerfsimPlugin::onShutdown() { impl_->shutdown(); }

void PerfsimPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  auto [iter, inserted] = impl_->dispatches.try_emplace(info.dispatch_id);
  DispatchState &state = iter->second;
  if (!inserted && (state.metadata_seen || state.begun)) {
    impl_->reject(info.dispatch_id, "duplicate dispatch metadata");
    return;
  }
  state.metadata = make_dispatch_metadata(info);
  state.dispatch_name = info.kernelNameOrUnknown();
  state.metadata.dispatch_name = state.dispatch_name.c_str();
  state.metadata_seen = true;
  if (const auto reason = validate_dispatch(info))
    impl_->reject(info.dispatch_id, *reason);
}

void PerfsimPlugin::onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
  DispatchState &state = impl_->dispatches[dispatch_id];
  if (!state.metadata_seen)
    impl_->reject(dispatch_id, "dispatch began without metadata");
  if (state.begun && !state.ended) {
    impl_->reject(dispatch_id, "dispatch began more than once");
    return;
  }
  state.begun = true;
  state.ended = false;
  if (!state.supported)
    return;
  impl_->replay_blockers.insert(dispatch_id);
  impl_->record_event({dispatch_id, BeginEvent{state.metadata}});
}

void PerfsimPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  auto iter = impl_->dispatches.find(dispatch_id);
  if (iter == impl_->dispatches.end() || !iter->second.begun) {
    impl_->reject(dispatch_id, "dispatch ended without a matching begin");
    iter = impl_->dispatches.find(dispatch_id);
  }
  DispatchState &state = iter->second;
  if (state.ended) {
    impl_->reject(dispatch_id, "dispatch ended more than once");
    return;
  }
  if (state.live_waves != 0)
    impl_->reject(dispatch_id, "dispatch ended with live wavefronts");
  state.ended = true;
  impl_->replay_blockers.erase(dispatch_id);
  if (state.supported)
    impl_->record_event({dispatch_id, EndEvent{state.metadata}});
  impl_->drain_epoch(/*shutdown=*/false);
}

void PerfsimPlugin::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  DispatchState *dispatch = impl_->active_dispatch(wf.dispatch_id());
  if (!dispatch) {
    impl_->reject(wf.dispatch_id(), "wavefront dispatched outside an active dispatch");
    return;
  }
  if (wf.wf_size() != 32 || wf.cluster_size() != 1 || wf.cluster_rank() != 0) {
    impl_->reject(wf.dispatch_id(), "wavefront does not satisfy gfx1250 one-workgroup cluster");
    return;
  }

  const uint32_t compute_unit_id = static_cast<uint32_t>(wf.cu().id());
  const uint64_t physical_key = physical_wave_key(compute_unit_id, wf.wf_id());
  if (const auto existing = impl_->physical_waves.find(physical_key);
      existing != impl_->physical_waves.end()) {
    const uint32_t old_dispatch = static_cast<uint32_t>(
        existing->second->wave_info.workgroup_info.cluster_info.dispatch_info.dispatch_id);
    impl_->reject(old_dispatch, "physical wavefront slot was reused before halt");
    impl_->reject(wf.dispatch_id(), "physical wavefront slot was reused before halt");
    impl_->physical_waves.erase(existing);
    return;
  }

  auto state = std::make_unique<PerfsimWavefrontState>();
  state->wave_info = make_wave_info(wf.dispatch_id(), wf.wg_coord(), wf.wave_in_group());
  state->compute_unit_id = compute_unit_id;
  state->physical_wavefront_id = wf.wf_id();
  state->rocjitsu_workgroup_id = wf.wg_id();
  state->queue_id = wf.queue_id();
  state->process_id = wf.process_id();
  state->lds_base = wf.lds_base();
  auto *raw_state = state.get();

  ++dispatch->live_waves;
  impl_->physical_waves.emplace(physical_key, raw_state);
  wf.set_plugin_state(slot_index(), std::move(state));
}

void PerfsimPlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  const uint32_t compute_unit_id = static_cast<uint32_t>(wf.cu().id());
  const uint64_t physical_key = physical_wave_key(compute_unit_id, wf.wf_id());
  const auto wave_iter = impl_->physical_waves.find(physical_key);
  if (wave_iter == impl_->physical_waves.end()) {
    impl_->reject(wf.dispatch_id(), "halted wavefront has no Perfsim identity");
    return;
  }

  PerfsimWavefrontState &wave = *wave_iter->second;
  const uint32_t dispatch_id =
      static_cast<uint32_t>(wave.wave_info.workgroup_info.cluster_info.dispatch_info.dispatch_id);
  auto dispatch_iter = impl_->dispatches.find(dispatch_id);
  if (dispatch_iter != impl_->dispatches.end()) {
    DispatchState &dispatch = dispatch_iter->second;
    if (wf.instruction_execution_failed())
      impl_->reject(dispatch_id, "wavefront halted after an instruction execution failure");
    if (!wave.saw_instruction || !wave.last_instruction_terminates)
      impl_->reject(dispatch_id, "wavefront halted without a terminating instruction");
    if (dispatch.live_waves != 0)
      --dispatch.live_waves;
  }
  impl_->physical_waves.erase(wave_iter);
}

void PerfsimPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                     amdgpu::Wavefront &wf) {
  record_instruction(pc, inst, wf, {});
}

void PerfsimPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                     amdgpu::Wavefront &wf,
                                                     std::span<const uint32_t> fetch_window) {
  record_instruction(pc, inst, wf, fetch_window);
}

void PerfsimPlugin::record_instruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf,
                                       std::span<const uint32_t> fetch_window) {
  const auto dispatch_iter = impl_->dispatches.find(wf.dispatch_id());
  if (dispatch_iter != impl_->dispatches.end() && !dispatch_iter->second.supported)
    return;

  const uint32_t compute_unit_id = static_cast<uint32_t>(wf.cu().id());
  PerfsimWavefrontState *wave = impl_->find_wave(compute_unit_id, wf.wf_id());
  if (!wave ||
      wave->wave_info.workgroup_info.cluster_info.dispatch_info.dispatch_id != wf.dispatch_id() ||
      wave->rocjitsu_workgroup_id != wf.wg_id()) {
    impl_->reject(wf.dispatch_id(), "instruction callback could not be matched to its wavefront");
    return;
  }

  const int encoding_bytes = inst.size();
  if (encoding_bytes <= 0 || encoding_bytes % 4 != 0 || encoding_bytes > 16 ||
      !inst.raw_encoding()) {
    impl_->reject(wf.dispatch_id(),
                  std::format("instruction encoding is not representable by FFM v{}",
                              impl_->negotiated_api_version));
    return;
  }
  if (inst.mnemonic().starts_with("scratch_")) {
    impl_->reject(wf.dispatch_id(), "dedicated SCRATCH instructions are not FFM-compatible");
    return;
  }
  if (!fetch_window.empty() && fetch_window.size() != 4) {
    impl_->reject(wf.dispatch_id(), "instruction fetch window is not four dwords");
    return;
  }

  InstructionEvent event;
  event.wave_info = wave->wave_info;
  event.pc = pc;
  const uint32_t encoding_dwords = static_cast<uint32_t>(encoding_bytes / 4);
  std::array<uint32_t, 4> instruction_encoding{};
  std::copy_n(inst.raw_encoding(), encoding_dwords, instruction_encoding.begin());
  if (fetch_window.empty())
    event.raw_isa = instruction_encoding;
  else
    std::copy(fetch_window.begin(), fetch_window.end(), event.raw_isa.begin());
  event.counters = instruction_counters(inst);
  event.wait = wait_info(inst);

  // FFM decrements its per-wave ordinal when a wait yields and reissues at the
  // same PC. RocJITsu's hook has no source ordinal, but a consecutive callback
  // with the same wait classification, PC, and encoding is unambiguous: wait
  // instructions cannot branch to themselves. Preserve that repeated ordinal
  // without collapsing a real loop over a non-wait instruction.
  const bool repeated_wait = event.wait.wait_type != FFM_WAIT_TYPE_NONE &&
                             wave->has_current_instruction && wave->current_pc == pc &&
                             wave->current_encoding_dwords == encoding_dwords &&
                             wave->current_instruction_encoding == instruction_encoding &&
                             wave->current_wait.wait_type == event.wait.wait_type &&
                             wave->current_wait.wait_name == event.wait.wait_name;
  event.instruction_id = repeated_wait ? wave->current_instruction_id : wave->next_instruction_id++;

  wave->current_instruction_id = event.instruction_id;
  wave->current_pc = pc;
  wave->current_instruction_encoding = instruction_encoding;
  wave->current_encoding_dwords = encoding_dwords;
  wave->current_wait = event.wait;
  wave->has_current_instruction = true;
  wave->saw_instruction = true;
  wave->last_instruction_terminates = (inst.flags() & PROGRAM_TERMINATOR) != 0;
  impl_->record_event({wf.dispatch_id(), std::move(event)});
}

void PerfsimPlugin::onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) {
  impl_->memory_access(access);
}

void PerfsimPlugin::onAmdgpuTensorDmaMemoryAccess(
    const amdgpu::TensorDmaMemoryAccessObservation &access) {
  impl_->tensor_dma_memory_access(access);
}

} // namespace rocjitsu::plugins::perfsim
