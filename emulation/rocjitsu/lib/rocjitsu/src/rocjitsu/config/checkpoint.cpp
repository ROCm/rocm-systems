// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/config/gfx10_checkpoint_resources.h"
#include "rocjitsu/config/gfx10_checkpoint_scheduler.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "rocjitsu/vm/amdgpu/workgroup_key.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "checkpoint_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "util/diagnostic.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {
namespace config {

namespace {

bool checkpoint_cu_owned_lds(rj_code_arch_t arch) {
  return arch != ROCJITSU_CODE_ARCH_RDNA1 && arch != ROCJITSU_CODE_ARCH_RDNA2 &&
         arch != ROCJITSU_CODE_ARCH_RV32I && arch != ROCJITSU_CODE_ARCH_RV64I;
}

flatbuffers::Optional<fb::GraphicsWait> graphics_wait(const amdgpu::Wavefront &wave) {
  if (!amdgpu::has_typed_export_waits(wave.cu().arch()))
    return flatbuffers::nullopt;
  if (wave.state() == amdgpu::WfState::EXPORT_WAIT)
    return fb::GraphicsWait_Export;
  if (wave.state() == amdgpu::WfState::PARAM_WAIT)
    return fb::GraphicsWait_Parameters;
  return fb::GraphicsWait_None;
}

util::Result restore_graphics_status(amdgpu::Wavefront &wave, const fb::WavefrontState &saved,
                                     bool predates_grants, util::DiagnosticEmitter emit_error) {
  uint32_t status = saved.status();
  const auto arch = wave.cu().arch();
  const auto waiting = saved.graphics_wait();
  bool use_launch_grants = !waiting && predates_grants;
  if (waiting) {
    if (!amdgpu::has_typed_export_waits(arch))
      return emit_error.emit() << "Graphics wait on an incompatible checkpoint architecture";
    switch (*waiting) {
    case fb::GraphicsWait_None:
      break;
    case fb::GraphicsWait_LaunchDefaults:
      use_launch_grants = true;
      break;
    case fb::GraphicsWait_Export:
    case fb::GraphicsWait_Parameters:
      if (wave.state() != amdgpu::WfState::RUNNING)
        return emit_error.emit() << "Inconsistent wave state for a checkpoint graphics wait";
      wave.set_state(*waiting == fb::GraphicsWait_Export ? amdgpu::WfState::EXPORT_WAIT
                                                         : amdgpu::WfState::PARAM_WAIT);
      break;
    default:
      return emit_error.emit() << "Invalid checkpoint graphics wait";
    }
  }
  if (use_launch_grants && amdgpu::has_typed_export_waits(arch)) {
    // Standalone waves started with both frontend grants. GFX11.5 also supported
    // independent grants, so keep a partial grant from a legacy frontend.
    constexpr uint32_t grants =
        amdgpu::Wavefront::kStatusExportReadyMask | amdgpu::Wavefront::kStatusParamReadyMask;
    if (arch != ROCJITSU_CODE_ARCH_RDNA3_5 || (status & grants) == 0)
      status |= wave.status_raw() & grants;
  }
  wave.set_status_raw(status);
  return util::Result::success();
}

const char *checkpoint_group_target(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return "RDNA4";
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return "CDNA5";
  return arch_to_string(arch);
}

bool checkpoint_tracks_workgroups(rj_code_arch_t arch) {
  return arch_is_cdna_4_or_lower(arch) || arch == ROCJITSU_CODE_ARCH_RDNA3 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         amdgpu::Wavefront::supports_sleep_wakeup_credit(arch);
}

// Check the group graph before modifying a CU, or opening the output file on save.
util::Result validate_checkpoint_groups(const fb::ComputeUnitState &state, uint32_t slots,
                                        const flatbuffers::Vector<uint32_t> *cursors,
                                        rj_code_arch_t arch, util::DiagnosticEmitter emit_error) {
  const auto *target = checkpoint_group_target(arch);
  const auto *groups = state.rdna4_workgroups();
  if ((cursors != nullptr) != (groups != nullptr))
    return emit_error.emit() << "Inconsistent " << target << " checkpoint workgroup metadata";
  struct Group {
    const fb::Rdna4CheckpointGroup *record;
    uint32_t seen = 0;
    uint32_t process = 0, queue = 0;
  };
  std::unordered_map<uint64_t, Group> by_key;
  if (groups) {
    for (const auto *group : *groups) {
      const auto live = group->live_members();
      if (!live || live > slots || (!group->multiwave() && live != 1) ||
          group->workgroup_signals() >= live || group->trap_signals() >= live ||
          !by_key.emplace(amdgpu::wg_key(group->dispatch_id(), group->wg_id()), Group{group})
               .second)
        return emit_error.emit() << "Invalid " << target
                                 << " checkpoint workgroup count or duplicate identity";
    }
  }
  if (const auto *waves = state.wavefronts()) {
    std::unordered_set<uint32_t> seen_slots;
    for (const auto *wave : *waves) {
      const auto waiting = wave->waiting_barrier();
      const auto complete = wave->barrier_complete_mask();
      const bool split_barriers =
          arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
      if ((!split_barriers && (waiting || complete)) || complete > 3 || (waiting && !groups))
        return emit_error.emit() << "Invalid " << target
                                 << " checkpoint wave barrier or wake state";
      if (!groups) {
        // RDNA4 has required identity for sleep since this layout was introduced.
        // Older CDNA5 files stored wave-local timers and may carry a wake
        // credit. Restore those fields even when no group table was recorded.
        if (arch == ROCJITSU_CODE_ARCH_RDNA4 && (wave->sleep_cycles() || wave->sleep_wakeup()))
          return emit_error.emit()
                 << "Legacy RDNA4 checkpoint lacks workgroup identity for sleeping wave";
        continue;
      }
      if (wave->wf_id() >= slots || !seen_slots.insert(wave->wf_id()).second)
        return emit_error.emit() << "Invalid " << target << " checkpoint wave slot";
      if (std::find(cursors->begin(), cursors->end(), wave->dispatch_id()) != cursors->end())
        return emit_error.emit() << target << " checkpoint dispatch cursor aliases a live wave";
      auto it = by_key.find(amdgpu::wg_key(wave->dispatch_id(), wave->wg_id()));
      if (waiting > 2 || (waiting && (complete & (1u << (waiting - 1)))) ||
          (waiting && (it == by_key.end() || !it->second.record->multiwave())))
        return emit_error.emit() << "Invalid " << target
                                 << " checkpoint wave barrier or wake state";
      if (it == by_key.end())
        continue; // Directly dispatched standalone wave.
      auto &group = it->second;
      if (group.seen && (group.process != wave->process_id() || group.queue != wave->queue_id()))
        return emit_error.emit() << "Inconsistent " << target
                                 << " checkpoint workgroup process or queue";
      group.process = wave->process_id();
      group.queue = wave->queue_id();
      ++group.seen;
    }
  }
  for (const auto &[key, group] : by_key)
    if (group.seen != group.record->live_members())
      return emit_error.emit() << target
                               << " checkpoint workgroup member count differs from saved waves";
  return util::Result::success();
}

const flatbuffers::Vector<uint32_t> *
checkpoint_dispatch_cursors(const fb::SimulationCheckpoint &checkpoint) {
  if (const auto *cp = checkpoint.command_processor(); cp && cp->next_dispatch_ids())
    return cp->next_dispatch_ids();
  return nullptr;
}

util::Result validate_checkpoint_workgroups(const fb::SimulationCheckpoint &checkpoint,
                                            const SoC &soc, util::DiagnosticEmitter emit_error) {
  const auto *units = checkpoint.compute_units();
  // Resource-only checkpoints may carry CP cursors without a workgroup table.
  const bool has_group_records =
      units && std::any_of(units->begin(), units->end(),
                           [](const auto *unit) { return unit && unit->rdna4_workgroups(); });
  const auto *cursors = has_group_records ? checkpoint_dispatch_cursors(checkpoint) : nullptr;
  const bool groups_supported = checkpoint_tracks_workgroups(soc.arch());
  const auto *target = checkpoint_group_target(soc.arch());
  if (cursors && (!groups_supported || cursors->size() != soc.xcds().size()))
    return emit_error.emit() << "Invalid " << target << " checkpoint dispatch cursor count";
  size_t record_index = 0;
  for (size_t xcd_index = 0; xcd_index < soc.xcds().size(); ++xcd_index) {
    auto *xcd = soc.xcds()[xcd_index];
    if (cursors) {
      const uint32_t cursor = cursors->Get(xcd_index);
      if (cursor < xcd_index + 1 || (cursor - xcd_index - 1) % soc.num_xcds() != 0)
        return emit_error.emit() << "Invalid " << target << " checkpoint dispatch cursor for XCD "
                                 << xcd_index;
    }
    for (auto *se : xcd->shader_engines())
      for (uint32_t cu_index = 0; cu_index < se->num_compute_units(); ++cu_index, ++record_index) {
        const auto *unit =
            units && record_index < units->size() ? units->Get(record_index) : nullptr;
        if (!unit) {
          if (cursors)
            return emit_error.emit() << "Missing " << target << " checkpoint compute unit";
          continue;
        }
        if (!groups_supported && unit->rdna4_workgroups())
          return emit_error.emit()
                 << "GFX12 group metadata on incompatible checkpoint architecture";
        if (validate_checkpoint_groups(*unit, se->compute_unit(cu_index)->num_wf_slots(), cursors,
                                       soc.arch(), emit_error)
                .failed())
          return util::Result::failure();
      }
  }
  if (cursors && units && record_index != units->size())
    return emit_error.emit() << "Invalid " << target << " checkpoint compute unit count";
  return util::Result::success();
}

/// @brief First scalar selector that named a TTMP before TTMPs became their own
/// register file.
/// @details The decoder used to alias selectors 108..123 into the wavefront SGPR
/// block on every architecture, so in a checkpoint written then those slots hold
/// exactly the TTMP file. Selectors that high are unreachable as ordinary SGPR
/// operands, so reading them back as TTMPs needs no per-architecture test.
constexpr uint32_t kLegacyTtmpSgprBase = 108;

util::FailureOr<flatbuffers::Offset<flatbuffers::Vector<uint8_t>>>
serialize_vgpr_block(flatbuffers::FlatBufferBuilder &builder, const amdgpu::ComputeUnitCore &cu,
                     uint32_t base, uint32_t lane_count, util::DiagnosticEmitter emit_error) {
  if ((lane_count != 32 && lane_count != 64) || lane_count > cu.vgpr_storage_lane_count())
    return emit_error.emit() << "Invalid VGPR checkpoint lane count";
  const size_t register_bytes = static_cast<size_t>(lane_count) * sizeof(uint32_t);
  const size_t block_bytes = static_cast<size_t>(cu.vgpr_allocation_block_size()) * register_bytes;
  uint8_t *serialized = nullptr;
  const auto offset = builder.CreateUninitializedVector<uint8_t>(block_bytes, &serialized);
  size_t offset_bytes = 0;
  bool invalid_storage = false;
  cu.for_each_raw_vgpr(base, cu.vgpr_allocation_block_size(), [&](std::span<const uint32_t> lanes) {
    if (lanes.size() < lane_count) {
      invalid_storage = true;
      return;
    }
    std::ranges::copy_n(reinterpret_cast<const uint8_t *>(lanes.data()), register_bytes,
                        serialized + offset_bytes);
    offset_bytes += register_bytes;
  });
  if (invalid_storage)
    return emit_error.emit() << "VGPR storage is narrower than the checkpoint wave";
  return offset;
}

/// Restore sparse checkpoint data into a freshly allocated, zeroed VGPR block.
/// Zero source registers are skipped to preserve lazy backing; this is a full
/// restore, rather than a merge, only while the destination begins entirely
/// zeroed.
util::Result restore_vgpr_block_into_zeroed_storage(amdgpu::ComputeUnitCore &cu, uint32_t base,
                                                    const flatbuffers::Vector<uint8_t> &stored,
                                                    uint32_t stored_lane_count,
                                                    uint32_t architectural_wave_size,
                                                    util::DiagnosticEmitter emit_error) {
  const auto bytes = std::span(reinterpret_cast<const std::byte *>(stored.data()), stored.size());
  const uint32_t register_count = cu.vgpr_allocation_block_size();

  uint32_t lane_count = stored_lane_count;
  if (lane_count == 0) {
    const uint32_t candidates[] = {architectural_wave_size, cu.vgpr_storage_lane_count(),
                                   cu.wf_size()};
    for (uint32_t candidate : candidates) {
      const size_t candidate_bytes =
          static_cast<size_t>(register_count) * candidate * sizeof(uint32_t);
      if ((candidate == 32 || candidate == 64) && stored.size() == candidate_bytes) {
        lane_count = candidate;
        break;
      }
    }
  }
  if ((lane_count != 32 && lane_count != 64) || lane_count > cu.vgpr_storage_lane_count())
    return emit_error.emit() << "Invalid VGPR checkpoint lane count";
  const size_t register_bytes = static_cast<size_t>(lane_count) * sizeof(uint32_t);
  if (stored.size() != static_cast<size_t>(register_count) * register_bytes)
    return emit_error.emit() << "Invalid VGPR checkpoint payload size";
  for (uint32_t reg = 0; reg < register_count; ++reg) {
    cu.restore_raw_vgprs_into_zeroed_storage(
        base + reg, 1, bytes.subspan(static_cast<size_t>(reg) * register_bytes, register_bytes));
  }
  return util::Result::success();
}

/// @brief Serialize the SoC configuration into a FlatBuffer SimulationConfig.
flatbuffers::Offset<fb::SimulationConfig>
serialize_config(flatbuffers::FlatBufferBuilder &builder, const SoC &soc,
                 const simdojo::SimulationEngine::Config &engine_config,
                 uint32_t cpu_dispatch_threads, uint32_t cpu_thread_budget,
                 std::span<const ExecutionThreadChoice> thread_allocations,
                 bool legacy_auto_dispatch, int32_t async_helper_threads) {
  auto arch_str = builder.CreateString(arch_to_string(soc.arch()));
  auto exec_mode_str = builder.CreateString(
      soc.exec_mode() == simdojo::ExecMode::CLOCKED ? "clocked" : "functional");

  // Extract configuration from the live component tree.
  uint32_t num_xcds = soc.num_xcds();
  uint32_t num_iods = soc.num_iods();
  uint32_t num_ses = 0;
  uint32_t num_cus = 0;
  uint32_t cus_per_shader_array = 0;
  flatbuffers::Offset<fb::ComputeUnitConfig> fb_cu;

  if (num_xcds > 0) {
    auto *xcd = soc.xcd(0);
    num_ses = xcd->num_shader_engines();
    if (num_ses > 0) {
      auto *se = xcd->shader_engine(0);
      num_cus = se->num_compute_units();
      if (num_cus > 0) {
        cus_per_shader_array = se->compute_unit(0)->cus_per_shader_array();
        const auto &cu_cfg = se->compute_unit(0)->config();
        // Store an explicit zero even though it is the stable wire default. It
        // means unbounded for a new checkpoint, while field absence remains the
        // legacy signal that restore should use ComputeUnitCore's native default.
        builder.ForceDefaults(true);
        fb_cu = fb::CreateComputeUnitConfig(
            builder, cu_cfg.num_wf_slots, cu_cfg.sgprs_per_wf, cu_cfg.vgprs_per_wf,
            cu_cfg.lds_size_kb, cu_cfg.functional_quantum,
            se->compute_unit(0)->scratch_slots_per_cu(), cu_cfg.gws_wait_limit,
            arithmetic_to_fb(builder, cu_cfg.arithmetic));
        builder.ForceDefaults(false);
      }
    }
  }

  auto fb_se = fb::CreateShaderEngineConfig(builder, num_cus, fb_cu, cus_per_shader_array);
  auto fb_xcd = fb::CreateXcdConfig(builder, num_ses, fb_se);
  auto fb_gpu = fb::CreateAmdgpuConfig(builder, num_xcds, num_iods, fb_xcd);
  auto fb_vm = fb::CreateVirtualMachineConfig(builder, arch_str, fb_gpu);

  std::vector<flatbuffers::Offset<fb::ExecutionThreadChoice>> choices;
  for (const auto &choice : thread_allocations)
    choices.push_back(
        fb::CreateExecutionThreadChoice(builder, choice.engines, choice.dispatch, choice.helpers));
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fb::ExecutionThreadChoice>>>
      fb_choices;
  if (!legacy_auto_dispatch)
    fb_choices = builder.CreateVector(choices);
  // Preserve automatic requests, including zero engines/budget.
  // Field absence continues to identify legacy serial-dispatch checkpoints.
  builder.ForceDefaults(true);
  auto result = fb::CreateSimulationConfig(
      builder, engine_config.max_ticks, engine_config.num_threads, exec_mode_str, fb_vm, 0, 0,
      cpu_dispatch_threads, cpu_thread_budget, fb_choices, async_helper_threads);
  builder.ForceDefaults(false);
  return result;
}

/// @brief Reconstruct a VirtualMachine::Config from a stored FlatBuffer config.
util::FailureOr<VirtualMachine::Config>
config_from_checkpoint(const fb::SimulationConfig *fb_config, util::DiagnosticEmitter emit_error) {
  VirtualMachine::Config vm_config{};

  if (!fb_config || !fb_config->vm())
    return emit_error.emit() << "Checkpoint missing simulation config";

  auto *vm = fb_config->vm();
  if (vm->arch())
    vm_config.soc.arch = parse_arch(vm->arch()->str());
  if (vm_config.soc.arch == ROCJITSU_CODE_ARCH_INVALID)
    return emit_error.emit() << "Checkpoint has missing or invalid architecture";
  vm_config.soc.exec_mode =
      parse_exec_mode(fb_config->exec_mode() ? fb_config->exec_mode()->str() : "");

  if (auto *gpu = vm->gpu()) {
    vm_config.soc.num_xcds = gpu->num_xcds();
    vm_config.soc.num_iods = gpu->num_iods();
    if (auto *xcd = gpu->xcd()) {
      vm_config.soc.xcd.num_shader_engines = xcd->num_shader_engines();
      if (auto *se = xcd->shader_engine()) {
        vm_config.soc.xcd.shader_engine.num_compute_units = se->num_compute_units();
        // Legacy checkpoints lack this geometry. Keep it unknown so WGP-ID
        // reads remain unsupported instead of reporting an SE-local index.
        vm_config.soc.xcd.shader_engine.cus_per_shader_array = se->cus_per_shader_array();
        if (auto *cu = se->compute_unit()) {
          auto &cu_cfg = vm_config.soc.xcd.shader_engine.compute_unit;
          auto arithmetic = arithmetic_from_fb(cu->arithmetic(), emit_error);
          if (arithmetic.failed())
            return util::Result::failure();
          cu_cfg.arithmetic = arithmetic.value();
          cu_cfg.num_wf_slots = cu->num_wf_slots();
          cu_cfg.sgprs_per_wf = cu->sgprs_per_wf();
          cu_cfg.vgprs_per_wf = cu->vgprs_per_wf();
          cu_cfg.lds_size_kb = cu->lds_size_kb();
          // An absent field is a legacy checkpoint. Keep Config's native
          // default instead of changing the FlatBuffers wire default: changing
          // a scalar default makes old and new schemas non-conforming, and an
          // old omitted zero would otherwise be reinterpreted as 1024.
          if (flatbuffers::IsFieldPresent(cu, fb::ComputeUnitConfig::VT_FUNCTIONAL_QUANTUM))
            cu_cfg.functional_quantum = cu->functional_quantum();
          cu_cfg.gws_wait_limit = cu->gws_wait_limit();
          vm_config.soc.scratch_slots_per_cu = cu->scratch_slots_per_cu();
        }
      }
    }
  }

  if (vm_config.soc.num_xcds == 0)
    vm_config.soc.num_xcds = 1;
  if (vm_config.soc.xcd.num_shader_engines == 0)
    vm_config.soc.xcd.num_shader_engines = 1;
  if (vm_config.soc.xcd.shader_engine.num_compute_units == 0)
    vm_config.soc.xcd.shader_engine.num_compute_units = 4;

  return vm_config;
}

} // namespace

util::Result save_checkpoint(const std::string &path, const SoC &soc, uint64_t tick,
                             const simdojo::SimulationEngine::Config &engine_config,
                             uint32_t cpu_dispatch_threads, util::DiagnosticEmitter emit_error,
                             uint32_t cpu_thread_budget,
                             std::span<const ExecutionThreadChoice> thread_allocations,
                             bool legacy_auto_dispatch, int32_t async_helper_threads) {
  // The format stores one topology template, not each engine's CU membership.
  // Equal total CU counts are insufficient: restoring would move per-CU state.
  if (soc.num_xcds() > 0) {
    const auto *first_xcd = soc.xcd(0);
    const auto num_ses = first_xcd->num_shader_engines();
    const auto num_cus = num_ses ? first_xcd->shader_engine(0)->num_compute_units() : 0;
    for (const auto *xcd : soc.xcds()) {
      if (xcd->num_shader_engines() != num_ses)
        return emit_error.emit() << "Cannot checkpoint nonuniform shader-engine topology";
      for (const auto *se : xcd->shader_engines()) {
        if (se->num_compute_units() != num_cus)
          return emit_error.emit() << "Cannot checkpoint nonuniform compute-unit topology";
      }
    }
  }

  // The serialization traversal must cover every CU, including those nested
  // outside the XCD/SE hierarchy accepted by the checkpoint format.
  std::unordered_set<const amdgpu::ComputeUnitCore *> serialized_cus;
  for (const auto *xcd : soc.xcds())
    for (const auto *se : xcd->shader_engines())
      for (uint32_t i = 0; i < se->num_compute_units(); ++i)
        serialized_cus.insert(se->compute_unit(i));
  auto covers_tree = [&](auto &&self, const simdojo::Component &component) -> bool {
    if (auto *cu = dynamic_cast<const amdgpu::ComputeUnitCore *>(&component))
      if (serialized_cus.erase(cu) != 1)
        return false;
    if (auto *composite = dynamic_cast<const simdojo::CompositeComponent *>(&component))
      for (const auto &child : composite->children())
        if (!self(self, *child))
          return false;
    return true;
  };
  // VM construction may have moved these components out of the SoC itself.
  if (!covers_tree(covers_tree, soc.component_root()) || !serialized_cus.empty())
    return emit_error.emit() << "Cannot checkpoint topology: serialization does not cover every CU";

  const bool gfx12 = amdgpu::Wavefront::supports_sleep_wakeup_credit(soc.arch());
  const bool groups_supported = checkpoint_tracks_workgroups(soc.arch());
  flatbuffers::FlatBufferBuilder builder(1024 * 1024);

  // Serialize compute unit states across all XCDs and their shader engines.
  std::unordered_set<uint32_t> live_dispatch_ids;
  std::vector<flatbuffers::Offset<fb::ComputeUnitState>> cu_offsets;
  for (auto *xcd : soc.xcds()) {
    for (auto *se : xcd->shader_engines()) {
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci) {
        auto *cu = se->compute_unit(ci);
        if (cu->gws_device().has_gds_checkpoint_state())
          return emit_error.emit()
                 << "Cannot checkpoint GDS memory or ordered-count state: indexed GDS data and "
                    "outstanding process reservations require a live owner";
        if (cu->gws_device().tracked_process_count() != 0)
          return emit_error.emit() << "Cannot checkpoint GWS resource state: the shared table "
                                      "is not part of the checkpoint format";
        const bool gfx10 = Gfx10CheckpointResources::supported(*cu);
        flatbuffers::Offset<fb::Gfx10CuResources> cu_resources;
        flatbuffers::Offset<fb::Gfx10CuScheduling> cu_scheduling;
        if (gfx10) {
          auto resources = Gfx10CheckpointResources::save_cu(builder, *cu, emit_error);
          if (resources.failed())
            return util::Result::failure();
          cu_resources = resources.value();
          auto scheduling = Gfx10CheckpointScheduler::save_cu(builder, *cu, emit_error);
          if (scheduling.failed())
            return util::Result::failure();
          cu_scheduling = scheduling.value();
        }
        const bool cdna3_wave = cu->arch() == ROCJITSU_CODE_ARCH_CDNA3;
        const bool cu_owned_lds = checkpoint_cu_owned_lds(cu->arch());
        const auto lds_cursor =
            cu_owned_lds ? std::optional{cu->reserved_lds_bytes()} : std::nullopt;
        uint64_t lds_end =
            cu_owned_lds ? std::max<uint64_t>(*lds_cursor, cu->lds().materialized_size_bytes()) : 0;
        if (cu_owned_lds && cu->lds_allocation_pinned())
          return emit_error.emit() << "Cannot checkpoint a pinned LDS allocation";
        std::vector<flatbuffers::Offset<fb::WavefrontState>> wf_offsets;
        std::vector<uint32_t> register_blocks;
        for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
          const auto *w = cu->wf(i);
          // Only checkpoint active (non-halted) wavefronts. Idle slots
          // have no register allocations and nothing meaningful to save.
          if (!w || w->is_halted())
            continue;

          // The group record contains only workgroup/trap barrier counters.
          // A nonzero named-barrier state means its allocation or membership
          // would be lost, changing the recipients of a later S_WAKEUP.
          if (gfx12 && w->barrier_state(1) != 0)
            return emit_error.emit()
                   << "Cannot checkpoint a GFX12 workgroup with allocated named barriers";
          const bool rdna2 = cu->arch() == ROCJITSU_CODE_ARCH_RDNA2;
          // Only the SoC's built-in identity mapping and direct standalone memory
          // are present in this file. A PASID alone cannot restore a frontend VM.
          const bool internal =
              w->address_space() && w->address_space() == soc.internal_address_space();
          const auto *access = w->vm_access();
          if ((!internal && (w->address_space() || access || (rdna2 && w->process_id()))) ||
              (access && (!access->is_current() ||
                          access->cache_namespace().address_space != w->address_space())))
            return emit_error.emit() << "Cannot checkpoint a resident wave with an "
                                        "external or stale address space";

          // A mid-handler checkpoint also needs the interrupted EXEC, any pending
          // debugger edit to the resume mask, and saved STATUS.PRIV. The record
          // omits these fields and the trap/debug reasons that own the stopped wave.
          //
          // Queue suspension belongs to the live command processor. A restored
          // checkpoint has no registered queue that can resume this wave.
          if (w->runtime_suspended())
            return emit_error.emit() << "Cannot checkpoint a wave suspended by its runtime queue";
          // A GFX10 wave stopped by its own S_SETHALT has no external owner.
          const bool shader_stop = gfx10 && w->self_halted() && w->debug_halted() &&
                                   !w->debug_suspended() && !cu->debug_active();
          if (cu_owned_lds && w->state() != amdgpu::WfState::RUNNING) {
            const bool recorded_barrier = gfx12 && w->state() == amdgpu::WfState::BARRIER &&
                                          w->checkpoint_waiting_barrier() != 0;
            const auto parked = graphics_wait(*w);
            if (!recorded_barrier && (!parked || *parked == fb::GraphicsWait_None))
              return emit_error.emit()
                     << "Cannot checkpoint AMDGPU state without its scheduler record on "
                     << cu->name();
          }
          if (cu_owned_lds &&
              (w->ready_cycle() > cu->cycle_count() || w->fatal_exception_pending()))
            return emit_error.emit()
                   << "Cannot checkpoint a delayed or exception-pending AMDGPU wave on "
                   << cu->name();
          // GFX10 has a separate resource record for restorable scratch allocations.
          if (cu_owned_lds && w->scratch_lane_size())
            return emit_error.emit()
                   << "Cannot checkpoint a resident private scratch allocation on " << cu->name();
          if (cu_owned_lds &&
              (!w->wait_counters().empty() ||
               (w->memory_wait_scoreboard() && !w->memory_wait_scoreboard()->empty())))
            return emit_error.emit()
                   << "Cannot checkpoint pending AMDGPU memory or wait state on " << cu->name();
          if (w->memory_trap_pending())
            return emit_error.emit() << "Cannot checkpoint a wave awaiting a memory trap";
          if (w->in_trap_handler() || (w->debug_stopped() && !shader_stop))
            return emit_error.emit()
                   << "Cannot checkpoint " << cu->name() << " wf" << w->wf_id()
                   << ": the wave is in a trap handler or stopped for a debugger, and that state "
                      "is not part of the checkpoint format";

          if (rdna2) {
            const auto block = w->vgpr_alloc().base / cu->vgpr_allocation_block_size();
            if (block != w->sgpr_alloc().base / cu->sgpr_allocation_block_size())
              return emit_error.emit()
                     << "Cannot checkpoint separately placed RDNA2 register blocks";
            register_blocks.push_back(block);
          }
          const bool cdna4 = cu->arch() == ROCJITSU_CODE_ARCH_CDNA4;
          const bool rounded_sgprs = arch_is_cdna_4_or_lower(cu->arch()) && !cdna3_wave;
          uint32_t sgpr_words = w->num_sgprs();
          if (rdna2)
            sgpr_words = cu->sgpr_allocation_block_size();
          else if (cdna3_wave)
            sgpr_words = std::min(cdna3::physical_sgpr_count(sgpr_words), w->debug_sgpr_capacity());
          else if (rounded_sgprs)
            sgpr_words =
                amdgpu::Wavefront::rounded_sgpr_count(sgpr_words, cu->config().sgprs_per_wf);
          std::vector<uint32_t> sgprs(sgpr_words);
          cu->sgpr_file().copy_to(w->sgpr_alloc().base, sgpr_words,
                                  std::as_writable_bytes(std::span(sgprs)));
          auto sgprs_vec = builder.CreateVector(sgprs);
          util::FailureOr<flatbuffers::Offset<flatbuffers::Vector<uint8_t>>> vgprs_vec =
              serialize_vgpr_block(builder, *cu, w->vgpr_alloc().base, w->wf_size(), emit_error);
          if (vgprs_vec.failed())
            return util::Result::failure();

          // TTMPs are their own file, so they are not covered by sgprs_vec.
          std::array<uint32_t, 16> ttmps{};
          for (uint32_t t = 0; t < ttmps.size(); ++t)
            ttmps[t] = w->ttmp(t);
          auto ttmps_vec = builder.CreateVector(ttmps.data(), ttmps.size());

          // Dispatch identity. The flat wg_id above cannot stand in for it --
          // unflattening needs the grid dimensions, which belong to the
          // dispatch packet and are not checkpointed -- and trap entry publishes
          // exactly these three values in TTMP8/9/10 for the CWSR record to
          // match.
          const auto &wg_coord = w->wg_coord();
          auto wg_coord_vec = builder.CreateVector(wg_coord.data(), wg_coord.size());

          flatbuffers::Offset<fb::Gfx10WaveResources> wave_resources;
          flatbuffers::Offset<fb::Gfx10WaveScheduling> wave_scheduling;
          if (gfx10) {
            auto resources = Gfx10CheckpointResources::save_wave(builder, *cu, *w,
                                                                 rdna2 && internal, emit_error);
            if (resources.failed())
              return util::Result::failure();
            wave_resources = resources.value();
            auto scheduling = Gfx10CheckpointScheduler::save_wave(builder, *cu, *w, emit_error);
            if (scheduling.failed())
              return util::Result::failure();
            wave_scheduling = scheduling.value();
          }
          if (groups_supported)
            live_dispatch_ids.insert(w->dispatch_id());
          const fb::LdsWindow lds_window(w->lds_base(), w->lds_size());
          if (cu_owned_lds) {
            const uint64_t end = uint64_t{w->lds_base()} + w->lds_size();
            if (&w->lds() != &cu->lds() || end > cu->lds().size_bytes() || end > UINT32_MAX)
              return emit_error.emit() << "Invalid checkpoint LDS placement";
            if (w->lds_size())
              lds_end = std::max(lds_end, end);
          }
          const fb::WaveExceptionRegisters exceptions(
              w->trapsts(), w->gfx12_excp_flag_user_extra_raw(), w->gfx12_trap_ctrl_raw());
          std::vector<flatbuffers::Offset<fb::ShaderExportRow>> export_rows;
          for (const auto &row : w->shader_exports()) {
            std::vector<uint32_t> values;
            values.reserve(w->wf_size() * 4);
            for (unsigned lane = 0; lane < w->wf_size(); ++lane)
              values.insert(values.end(), row.values[lane].begin(), row.values[lane].end());
            export_rows.push_back(
                fb::CreateShaderExportRow(builder, row.target, row.row, row.components, row.done,
                                          row.valid_lanes, builder.CreateVector(values)));
          }
          const auto exports = export_rows.empty() ? 0 : builder.CreateVector(export_rows);
          fb::WavefrontStateBuilder record(builder);
          if (gfx10) {
            record.add_gfx10_resources(wave_resources);
            record.add_gfx10_scheduling(wave_scheduling);
          } else
            record.add_exceptions(&exceptions);
          record.add_wf_id(w->wf_id());
          record.add_wg_id(w->wg_id());
          record.add_dispatch_id(w->dispatch_id());
          if (groups_supported) {
            record.add_process_id(w->process_id());
            record.add_queue_id(w->queue_id());
            record.add_wave_in_group(w->wave_in_group());
          }
          record.add_pc(w->pc);
          record.add_exec(w->exec_raw());
          record.add_vcc(w->vcc());
          record.add_m0(w->m0());
          record.add_halted(w->is_halted());
          record.add_status(w->status_raw());
          if (const auto wait = graphics_wait(*w))
            record.add_graphics_wait(*wait);
          record.add_sgprs(sgprs_vec);
          record.add_vgprs(vgprs_vec.value());
          record.add_mode(w->mode_raw());
          record.add_wave_sched_mode(w->wave_sched_mode_raw());
          record.add_ttmps(ttmps_vec);
          record.add_wg_coord(wg_coord_vec);
          record.add_kernel_wave_size(w->kernel_wave_size());
          record.add_vgpr_lane_count(w->wf_size());
          record.add_setreg_vgpr_msb_hazard(w->setreg_vgpr_msb_hazard());
          if (rdna2) {
            record.add_rdna2_process_id(w->process_id());
            record.add_rdna2_internal_address_space(internal);
          }
          if (cu_owned_lds) {
            record.add_lds_window(&lds_window);
            record.add_flat_scratch(w->scratch_base());
          }
          if (cdna3_wave)
            record.add_requested_sgprs(w->num_sgprs());
          if (rounded_sgprs || rdna2)
            record.add_num_sgprs(w->num_sgprs());
          if (!gfx10)
            record.add_num_vgprs(w->num_vgprs());
          if (cdna4 || cdna3_wave || cu->arch() == ROCJITSU_CODE_ARCH_CDNA2)
            record.add_accum_vgpr_base(w->accum_vgpr_base());
          record.add_sleep_cycles(w->sleep_cycles());
          if (gfx12) {
            record.add_sleep_wakeup(w->sleep_wakeup());
            record.add_barrier_complete_mask(unsigned(w->workgroup_barrier_complete()) |
                                             (unsigned(w->trap_barrier_complete()) << 1));
            record.add_waiting_barrier(w->checkpoint_waiting_barrier());
          }
          if (cu->arch() == ROCJITSU_CODE_ARCH_RDNA3 || cu->arch() == ROCJITSU_CODE_ARCH_RDNA3_5)
            record.add_ib_sts2_gfx11(w->ib_sts2_gfx11() & ~(1u << 11));
          if (!rdna2)
            record.add_uses_internal_address_space(internal);
          record.add_shader_exports(exports);
          wf_offsets.push_back(record.Finish());
        }

        flatbuffers::Offset<flatbuffers::Vector<const fb::Rdna4CheckpointGroup *>> group_vec;
        if (groups_supported) {
          std::vector<fb::Rdna4CheckpointGroup> groups;
          for (const auto &g : cu->rdna4_checkpoint_groups())
            groups.emplace_back(g.dispatch_id, g.wg_id, g.live_members, g.workgroup_signals,
                                g.trap_signals, g.multiwave);
          group_vec = builder.CreateVectorOfStructs(groups);
        }
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> lds;
        if (cu_owned_lds && (!wf_offsets.empty() || lds_end)) {
          if (lds_end > cu->lds().size_bytes() || lds_end > UINT32_MAX)
            return emit_error.emit() << "Invalid checkpoint LDS reservation";
          uint8_t *bytes = nullptr;
          const uint32_t payload_size = static_cast<uint32_t>(lds_end);
          lds = builder.CreateUninitializedVector<uint8_t>(payload_size, &bytes);
          const amdgpu::Lds &backing = cu->lds();
          backing.read(0u, bytes, payload_size);
        }
        auto name = builder.CreateString(cu->name());
        auto wfs_vec = builder.CreateVector(wf_offsets);
        const auto &levels = cu->performance_levels();
        const auto saved_levels = std::ranges::any_of(levels, [](auto count) { return count != 0; })
                                      ? builder.CreateVector(levels.data(), levels.size())
                                      : 0;
        auto cus = fb::CreateComputeUnitState(
            builder, name, wfs_vec, 0, cu->config().functional_quantum, true,
            cu->config().memory_wait_diagnostics != amdgpu::MemoryWaitDiagnostics::Off,
            cu->config().xcnt_diagnostics != amdgpu::MemoryWaitDiagnostics::Off,
            cu->config().gws_wait_limit, arithmetic_to_fb(builder, cu->config().arithmetic),
            cu_resources, register_blocks.empty() ? 0 : builder.CreateVector(register_blocks), lds,
            lds_cursor, cu_scheduling, cu->cycle_count(), group_vec,
            gfx10 ? std::nullopt : std::optional{cu->model_realtime_timestamp()}, saved_levels);

        cu_offsets.push_back(cus);
      }
    }
  }

  // Serialize command processor state (first XCD's CP).
  flatbuffers::Offset<fb::CommandProcessorState> cp_offset;
  if (!soc.xcds().empty()) {
    auto *cp = soc.xcds()[0]->command_processor();
    auto name = builder.CreateString(cp->name());
    std::vector<uint32_t> cursors;
    for (uint32_t i = 0; i < soc.num_xcds(); ++i) {
      const auto *xcd_cp = soc.xcd(i)->command_processor();
      if (xcd_cp->has_pending_aql_dispatch_work())
        return emit_error.emit() << "Cannot checkpoint AQL workgroups awaiting launch "
                                    "or a pending completion signal";
      auto next = xcd_cp->next_dispatch_id_for_topology(i, soc.num_xcds());
      // Direct CU dispatch can install an ID without consuming it from the CP.
      while (groups_supported && live_dispatch_ids.contains(next))
        next = amdgpu::CommandProcessor::step_dispatch_id(next, i + 1, soc.num_xcds());
      cursors.push_back(next);
    }
    const auto cursor_vec = builder.CreateVector(cursors);
    cp_offset = fb::CreateCommandProcessorState(builder, name, cp->dispatched_count(),
                                                cp->next_cu_index(), cursor_vec);
  }

  // Serialize GPU memory pages.
  std::vector<flatbuffers::Offset<fb::MemoryPage>> page_offsets;
  soc.memory()->for_each_page([&](uint64_t addr, const auto &page) {
    auto data_vec = builder.CreateVector(page.data(), page.size());
    page_offsets.push_back(fb::CreateMemoryPage(builder, addr, data_vec));
  });

  auto cu_vec = builder.CreateVector(cu_offsets);
  auto pages_vec = builder.CreateVector(page_offsets);
  auto mem_state = fb::CreateGpuMemoryState(builder, pages_vec);
  auto config_offset = serialize_config(builder, soc, engine_config, cpu_dispatch_threads,
                                        cpu_thread_budget, thread_allocations,
                                        legacy_auto_dispatch && cpu_dispatch_threads == 0 &&
                                            cpu_thread_budget == 0 && thread_allocations.empty(),
                                        async_helper_threads);

  auto checkpoint =
      fb::CreateSimulationCheckpoint(builder, tick, config_offset, cu_vec, cp_offset, mem_state);
  builder.Finish(checkpoint);
  if (groups_supported &&
      validate_checkpoint_workgroups(*fb::GetSimulationCheckpoint(builder.GetBufferPointer()), soc,
                                     emit_error)
          .failed())
    return util::Result::failure();

  // Write to file.
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open())
    return emit_error.emit() << "Cannot open checkpoint file for writing: " << path;
  f.write(reinterpret_cast<const char *>(builder.GetBufferPointer()), builder.GetSize());
  f.flush();
  if (!f)
    return emit_error.emit() << "Failed to write checkpoint file: " << path;
  f.close();
  if (!f)
    return emit_error.emit() << "Failed to close checkpoint file: " << path;
  return util::Result::success();
}

util::FailureOr<LoadedConfig> restore_checkpoint(const std::string &path,
                                                 util::DiagnosticEmitter emit_error) {
  // Read binary file.
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    return emit_error.emit() << "Cannot open checkpoint file: " << path;
  auto pos = f.tellg();
  if (pos <= 0)
    return emit_error.emit() << "Empty or unreadable checkpoint file: " << path;
  auto size = static_cast<size_t>(pos);
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(size);
  if (!f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size)))
    return emit_error.emit() << "Failed to read checkpoint file: " << path;

  flatbuffers::Verifier verifier(buf.data(), buf.size());
  if (!fb::VerifySimulationCheckpointBuffer(verifier))
    return emit_error.emit() << "Invalid checkpoint format: " << path;
  auto *checkpoint = fb::GetSimulationCheckpoint(buf.data());

  // Rebuild SoC and engine config from the stored configuration.
  if (!checkpoint->config())
    return emit_error.emit() << "Checkpoint missing config section: " << path;
  auto *fb_config = checkpoint->config();
  util::FailureOr<VirtualMachine::Config> maybe_vm_config =
      config_from_checkpoint(fb_config, emit_error);
  if (maybe_vm_config.failed())
    return util::Result::failure();
  auto vm_config = std::move(maybe_vm_config).value();
  simdojo::SimulationEngine::Config engine_config{};
  engine_config.max_ticks = fb_config->max_ticks();
  engine_config.num_threads = fb_config->num_threads();

  LoadedConfig result;
  result.exec_mode = vm_config.soc.exec_mode;
  result.requested_engine_threads = engine_config.num_threads;
  result.cpu_dispatch_threads =
      flatbuffers::IsFieldPresent(fb_config, fb::SimulationConfig::VT_CPU_DISPATCH_THREADS)
          ? fb_config->cpu_dispatch_threads()
          : 1u;
  result.cpu_thread_budget = fb_config->cpu_thread_budget();
  result.async_helper_threads =
      flatbuffers::IsFieldPresent(fb_config, fb::SimulationConfig::VT_ASYNC_HELPER_THREADS)
          ? fb_config->async_helper_threads()
          : 0;
  result.legacy_auto_dispatch =
      !fb_config->thread_allocations() && result.cpu_dispatch_threads == 0;
  if (fb_config->thread_allocations())
    for (const auto *choice : *fb_config->thread_allocations())
      result.thread_allocations.push_back(
          {choice->num_threads(), choice->cpu_dispatch_threads(), choice->async_helper_threads()});
  const auto &xcd = vm_config.soc.xcd;
  auto checked_capacity = [emit_error](uint64_t n) -> util::FailureOr<uint32_t> {
    if (n > std::numeric_limits<uint32_t>::max())
      return emit_error.emit() << "Checkpoint execution capacity exceeds uint32 range";
    return static_cast<uint32_t>(n);
  };
  util::FailureOr<uint32_t> maybe_cus =
      checked_capacity(uint64_t{xcd.num_shader_engines} * xcd.shader_engine.num_compute_units);
  if (maybe_cus.failed())
    return util::Result::failure();
  const uint32_t cus = maybe_cus.value();
  // The stored topology is uniform: each nonempty XCD contributes CUs-1 workers
  // to the shared pool, whose inclusive width also counts one caller.
  util::FailureOr<uint32_t> maybe_capacity =
      checked_capacity(1 + uint64_t{vm_config.soc.num_xcds} * (std::max(cus, 1u) - 1));
  if (maybe_capacity.failed())
    return util::Result::failure();
  const uint32_t capacity = maybe_capacity.value();
  const uint32_t host_threads = amdgpu::available_host_threads();
  util::FailureOr<ExecutionThreadAllocation> allocation = resolve_execution_threads(
      {result.cpu_thread_budget, result.requested_engine_threads, result.cpu_dispatch_threads,
       result.async_helper_threads},
      host_threads, vm_config.soc.num_xcds, std::span(&capacity, 1),
      configured_async_mma_supported(vm_config.soc.arch), result.thread_allocations, emit_error,
      result.exec_mode == simdojo::ExecMode::CLOCKED);
  if (allocation.failed())
    return util::Result::failure();
  result.execution_threads = std::move(allocation).value();
  // Before allocation tables, explicit zero selected a dispatch-only host
  // budget. An absent vector identifies that policy; a present empty vector
  // intentionally selects the new serial fallback.
  if (result.legacy_auto_dispatch && result.exec_mode == simdojo::ExecMode::FUNCTIONAL) {
    auto budgets = resolve_cpu_dispatch_thread_budgets(0, host_threads, 1);
    result.execution_threads.dispatch = {std::min(budgets.front(), std::max(capacity, 1u))};
  }
  if (!engine_config.num_threads)
    engine_config.num_threads = result.execution_threads.engines;

  result.async_resources = make_async_execution_resources(result.execution_threads.helpers);
  vm_config.soc.xcd.shader_engine.compute_unit.async_resources = result.async_resources;

  // Rebuild the SoC root expected by LoadedConfig and create_from_loaded().
  auto soc = std::make_unique<SoC>("gpu_soc", vm_config.soc);
  auto *soc_ptr = soc.get();
  auto *mem_ptr = soc->memory();

  if (const auto *cursors = checkpoint_dispatch_cursors(*checkpoint)) {
    if (cursors->size() != soc_ptr->num_xcds())
      return emit_error.emit() << "Invalid checkpoint dispatch ID cursor count";
    for (uint32_t i = 0; i < cursors->size(); ++i) {
      if (!soc_ptr->xcd(i)->command_processor()->restore_dispatch_id_sequence(i, cursors->size(),
                                                                              cursors->Get(i)))
        return emit_error.emit() << "Invalid checkpoint dispatch ID cursor for XCD " << i;
    }
  }

  if (validate_checkpoint_workgroups(*checkpoint, *soc, emit_error).failed())
    return util::Result::failure();
  // Restore GPU memory pages.
  if (auto *mem_state = checkpoint->memory()) {
    if (auto *pages = mem_state->pages()) {
      for (auto *page : *pages) {
        if (page->data() && page->data()->size() > 0) {
          mem_ptr->load_image(page->data()->data(), page->data()->size(), page->address());
        }
      }
    }
  }

  // Collect all CUs across XCDs and their shader engines for indexed restoration.
  std::vector<amdgpu::ComputeUnitCore *> all_cus;
  for (auto *xcd : soc_ptr->xcds()) {
    for (auto *se : xcd->shader_engines()) {
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci)
        all_cus.push_back(se->compute_unit(ci));
    }
  }

  // Restore wavefront state in compute units.
  if (auto *cu_states = checkpoint->compute_units()) {
    for (size_t i = 0; i < cu_states->size() && i < all_cus.size(); ++i) {
      auto *cu_state = cu_states->Get(i);
      if (!cu_state)
        continue;
      auto *cu = all_cus[i];
      const auto *cu_resources = cu_state->gfx10_resources();
      const auto *cu_scheduling = cu_state->gfx10_scheduling();
      if ((cu_scheduling && !cu_resources) ||
          (cu_resources && !Gfx10CheckpointResources::supported(*cu)))
        return emit_error.emit() << "Unexpected or incomplete GFX10 CU checkpoint records";
      if (cu_resources &&
          Gfx10CheckpointResources::restore_cu(*cu_resources, *cu, emit_error).failed())
        return util::Result::failure();
      if (cu_scheduling)
        Gfx10CheckpointScheduler::restore_cu(*cu_scheduling, *cu);
      // GFX10 checkpoints written before the shared counter field keep it in
      // the scheduling record. A CU without either field starts at zero.
      const auto *levels = cu_state->shader_perf_levels();
      if (!levels && cu_scheduling)
        levels = cu_scheduling->perf_levels();
      if (levels && !cu->restore_performance_levels({levels->data(), levels->size()}))
        return emit_error.emit() << "Invalid checkpoint performance-counter count";
      cu->restore_cycle_count(cu_state->cycle_count());
      if (auto realtime = cu_state->model_realtime())
        cu->restore_model_realtime(*realtime);
      if (cu_state->gws_wait_limit())
        cu->set_gws_wait_limit(*cu_state->gws_wait_limit());
      if (cu_state->arithmetic()) {
        auto arithmetic = arithmetic_from_fb(cu_state->arithmetic(), emit_error);
        if (arithmetic.failed())
          return util::Result::failure();
        cu->restore_arithmetic_policy(arithmetic.value());
      }
      if (cu_state->functional_quantum_present())
        cu->set_functional_quantum(cu_state->functional_quantum());
      using amdgpu::MemoryWaitDiagnostics;
      const auto memory_checks = cu_state->memory_wait_checks();
      const auto xcnt_checks = cu_state->xcnt_checks();
      cu->restore_wait_diagnostics(
          memory_checks
              ? (*memory_checks ? MemoryWaitDiagnostics::Warn : MemoryWaitDiagnostics::Off)
              : cu->config().memory_wait_diagnostics,
          xcnt_checks ? (*xcnt_checks ? MemoryWaitDiagnostics::Warn : MemoryWaitDiagnostics::Off)
                      : cu->config().xcnt_diagnostics);
      const auto *groups = cu_state->rdna4_workgroups();
      if (groups)
        for (const auto *g : *groups)
          cu->restore_rdna4_checkpoint_group({g->dispatch_id(), g->wg_id(), g->live_members(),
                                              g->workgroup_signals(), g->trap_signals(),
                                              g->multiwave()});
      const bool restore_lds = checkpoint_cu_owned_lds(cu->arch());
      const auto *lds = cu_state->lds();
      const auto cursor = cu_state->lds_cursor();
      if (cursor && (!restore_lds || (*cursor && (!lds || *cursor > lds->size()))))
        return emit_error.emit() << "Invalid checkpoint LDS allocator cursor";
      if (lds) {
        if (!restore_lds || lds->size() > cu->lds().size_bytes() ||
            (!cursor && lds->size() % amdgpu::lds_allocation_granule(cu->arch()) != 0) ||
            !cu->restore_lds_cursor(cursor.value_or(lds->size())))
          return emit_error.emit() << "Invalid checkpoint LDS reservation";
        cu->lds().write(0u, lds->data(), lds->size());
      }
      const bool rdna2 = cu->arch() == ROCJITSU_CODE_ARCH_RDNA2;
      const auto *blocks = rdna2 ? cu_state->rdna2_register_blocks() : nullptr;
      const auto *wf_states = cu_state->wavefronts();
      if (blocks && blocks->size() != (wf_states ? wf_states->size() : 0))
        return emit_error.emit() << "Invalid RDNA2 checkpoint register block count";
      if (wf_states) {
        for (uint32_t wi = 0; wi < wf_states->size(); ++wi) {
          const auto *wf_state = wf_states->Get(wi);
          const auto pid = wf_state->rdna2_process_id();
          const auto *sgprs = wf_state->sgprs();
          const uint32_t sgpr_capacity = cu->config().sgprs_per_wf;
          uint32_t num_sgprs = sgprs ? sgprs->size() : sgpr_capacity;
          uint32_t restore_sgpr_words = num_sgprs;
          const auto *window = wf_state->lds_window();
          if (restore_lds && !wf_state->halted() && (!window || !lds))
            return emit_error.emit() << "Cannot restore active wave without saved LDS state";
          if (window &&
              (!restore_lds || !lds || window->base() > cu->lds().size_bytes() ||
               (window->size() && uint64_t{window->base()} + window->size() > lds->size())))
            return emit_error.emit() << "Invalid checkpoint LDS window";
          const uint32_t stored_sgprs =
              wf_state->sgprs() ? wf_state->sgprs()->size() : cu->config().sgprs_per_wf;
          num_sgprs = wf_state->requested_sgprs().value_or(stored_sgprs);
          if (!num_sgprs || num_sgprs > stored_sgprs ||
              stored_sgprs > cu->sgpr_allocation_block_size())
            return emit_error.emit() << "Invalid checkpoint SGPR allocation";
          const auto *wave_resources = wf_state->gfx10_resources();
          const auto *wave_scheduling = wf_state->gfx10_scheduling();
          if ((wave_resources != nullptr) != (cu_resources != nullptr) ||
              (wave_scheduling != nullptr) != (cu_scheduling != nullptr))
            return emit_error.emit() << "Inconsistent GFX10 wave checkpoint records";
          if (cu->arch() == ROCJITSU_CODE_ARCH_RDNA1 && !wave_resources) {
            if (wf_state->halted())
              continue;
            return emit_error.emit() << "Cannot restore active RDNA1 wave: checkpoint omitted "
                                        "its VGPR and LDS allocations";
          }
          uint32_t num_vgprs =
              wave_resources ? wave_resources->vgpr_count() : cu->config().vgprs_per_wf;
          if (wave_resources &&
              (!num_vgprs || num_vgprs > cu->vgpr_allocation_block_size() || wf_state->halted()))
            return emit_error.emit() << "Invalid GFX10 wave allocation";
          if (rdna2) {
            if (wf_state->halted())
              continue;
            if (!pid)
              return emit_error.emit() << "Cannot restore active RDNA2 wave: checkpoint omitted "
                                          "the owning process";
            if (!wave_resources)
              return emit_error.emit() << "Cannot restore active RDNA2 wave: checkpoint omitted "
                                          "LDS and scratch allocation state";
            if (*pid != 0 && !wf_state->rdna2_internal_address_space())
              return emit_error.emit() << "RDNA2 checkpoint process lacks its address space";
            const auto requested = wf_state->num_sgprs();
            if (!requested)
              return emit_error.emit() << "Cannot restore active RDNA2 wave: checkpoint omitted "
                                          "its register allocation";
            num_sgprs = *requested;
            if (!num_sgprs || num_sgprs > cu->sgpr_allocation_block_size() || !num_vgprs ||
                num_vgprs > cu->vgpr_allocation_block_size())
              return emit_error.emit() << "Invalid RDNA2 checkpoint register allocation";
            if (!sgprs || sgprs->size() != cu->sgpr_allocation_block_size())
              return emit_error.emit() << "Invalid RDNA2 checkpoint SGPR payload size";
            if (!wf_state->vgprs() || !wf_state->kernel_wave_size() ||
                wf_state->vgpr_lane_count() != wf_state->kernel_wave_size())
              return emit_error.emit() << "Invalid RDNA2 checkpoint VGPR payload or lane count";
          }
          uint32_t accum_base = 0;
          if (cu->arch() != ROCJITSU_CODE_ARCH_CDNA4 && !Gfx10CheckpointResources::supported(*cu)) {
            if (const auto total = wf_state->num_vgprs()) {
              if (*total > cu->vgpr_allocation_block_size())
                return emit_error.emit() << "Invalid checkpoint VGPR allocation";
              num_vgprs = *total;
            }
            if (cu->arch() == ROCJITSU_CODE_ARCH_CDNA2 || cu->arch() == ROCJITSU_CODE_ARCH_CDNA3) {
              accum_base = wf_state->accum_vgpr_base();
              if (accum_base > amdgpu::ACC_VGPR_OFFSET || accum_base % 4 != 0 ||
                  (accum_base && !wf_state->num_vgprs()))
                return emit_error.emit()
                       << "Invalid " << (cu->arch() == ROCJITSU_CODE_ARCH_CDNA2 ? "CDNA2" : "CDNA3")
                       << " checkpoint AccVGPR boundary";
            }
          }
          if (arch_is_cdna_4_or_lower(cu->arch()) && cu->arch() != ROCJITSU_CODE_ARCH_CDNA3) {
            if (const auto requested = wf_state->num_sgprs()) {
              num_sgprs = *requested;
              restore_sgpr_words = amdgpu::Wavefront::rounded_sgpr_count(num_sgprs, sgpr_capacity);
              if (!num_sgprs || num_sgprs > sgpr_capacity || !sgprs ||
                  sgprs->size() != restore_sgpr_words)
                return emit_error.emit() << "Invalid "
                                         << (cu->arch() == ROCJITSU_CODE_ARCH_CDNA1   ? "CDNA1"
                                             : cu->arch() == ROCJITSU_CODE_ARCH_CDNA2 ? "CDNA2"
                                                                                      : "CDNA4")
                                         << " checkpoint SGPR allocation";
            } else {
              // A pre-TTMP-file record may also contain trap selectors after its SGPRs.
              num_sgprs = std::min(num_sgprs, sgpr_capacity);
              restore_sgpr_words = num_sgprs;
            }
          }
          if (cu->arch() == ROCJITSU_CODE_ARCH_CDNA4) {
            const auto saved_total = wf_state->num_vgprs();
            accum_base = wf_state->accum_vgpr_base();
            num_vgprs = saved_total.value_or(num_vgprs);
            if (num_vgprs == 0 || num_vgprs > cu->vgpr_allocation_block_size())
              return emit_error.emit() << "Invalid CDNA4 checkpoint VGPR allocation";
            // A descriptor can place the AccVGPR boundary above the total when
            // a wave has only ordinary registers. Keep that register value.
            if ((!saved_total && accum_base != 0) || accum_base > amdgpu::ACC_VGPR_OFFSET ||
                accum_base % 4 != 0)
              return emit_error.emit() << "Invalid CDNA4 checkpoint AccVGPR boundary";
          }

          const uint32_t wave_size =
              wf_state->kernel_wave_size() == 0 ? cu->wf_size() : wf_state->kernel_wave_size();
          auto *wf = cu->dispatch_wf_at(wf_state->wf_id(), wf_state->wg_id(), wf_state->pc(),
                                        num_sgprs, num_vgprs, wave_size,
                                        blocks ? std::optional{blocks->Get(wi)} : std::nullopt);
          if (!wf)
            return emit_error.emit()
                   << (blocks
                           ? "Failed to restore wavefront into its recorded slot or register block"
                           : "Failed to restore wavefront into its recorded slot");

          if (rdna2) {
            wf->set_process_id(*pid);
            if (wf_state->rdna2_internal_address_space())
              wf->set_address_space(soc_ptr->internal_address_space());
          }
          wf->set_dispatch_id(wf_state->dispatch_id());
          if (window) {
            wf->set_lds_base(window->base());
            wf->set_lds_size(window->size());
          }
          if (wf_state->ib_sts2_gfx11()) {
            if ((cu->arch() != ROCJITSU_CODE_ARCH_RDNA3 &&
                 cu->arch() != ROCJITSU_CODE_ARCH_RDNA3_5) ||
                (wf_state->ib_sts2_gfx11() & ~0x703u))
              return emit_error.emit() << "Invalid GFX11 checkpoint IB_STS2";
            wf->set_ib_sts2_gfx11(wf_state->ib_sts2_gfx11());
          }
          wf->set_accum_vgpr_base(accum_base);
          wf->set_exec_raw(wf_state->exec());
          wf->set_m0(wf_state->m0());
          // Halted wavefronts are never saved (see save_checkpoint skip above),
          // so halted() is always false here. Keep the branch for
          // future-proofing.
          wf->set_state(wf_state->halted() ? amdgpu::WfState::HALTED : amdgpu::WfState::RUNNING);
          if (!rdna2) {
            const auto internal = wf_state->uses_internal_address_space();
            if (!internal && groups && wf_state->process_id() != 0)
              return emit_error.emit()
                     << "Legacy GFX12 checkpoint lacks the resident process address-space binding";
            if (internal.value_or(false)) {
              if (!soc_ptr->internal_address_space())
                return emit_error.emit() << "Checkpoint requires the standalone VM's address space";
              wf->set_address_space(soc_ptr->internal_address_space());
            }
          }
          if (groups) {
            wf->set_process_id(wf_state->process_id());
            wf->set_queue_id(wf_state->queue_id());
            wf->set_wave_in_group(wf_state->wave_in_group());
            wf->restore_rdna4_checkpoint_barriers(wf_state->barrier_complete_mask(),
                                                  wf_state->waiting_barrier());
          }
          // The resource-only writer saved both CU LDS and dispatch cursors before it
          // modeled grants. Older frontend-aware files have only dispatch cursors.
          const auto *cp_state = checkpoint->command_processor();
          const bool predates_grants =
              cursor.has_value() || !cp_state || !cp_state->next_dispatch_ids();
          if (restore_graphics_status(*wf, *wf_state, predates_grants, emit_error).failed())
            return util::Result::failure();
          if (const auto *saved = wf_state->shader_exports()) {
            std::vector<amdgpu::ShaderExportRow> rows;
            std::set<std::pair<uint8_t, uint8_t>> seen;
            for (const auto *entry : *saved) {
              if (!amdgpu::has_typed_export_waits(cu->arch()) || entry->target() >= 64 ||
                  entry->components() > 15 ||
                  (wf->wf_size() < 64 && (entry->valid_lanes() >> wf->wf_size())) ||
                  entry->values()->size() != wf->wf_size() * 4 ||
                  !seen.emplace(entry->target(), entry->row()).second)
                return emit_error.emit() << "Invalid wave shader export row in checkpoint";
              auto &row = rows.emplace_back(
                  amdgpu::ShaderExportRow{entry->target(), entry->row(), entry->components(),
                                          entry->done(), entry->valid_lanes()});
              for (unsigned lane = 0; lane < wf->wf_size(); ++lane)
                for (unsigned channel = 0; channel < 4; ++channel)
                  row.values[lane][channel] = entry->values()->Get(4 * lane + channel);
            }
            wf->restore_shader_exports(std::move(rows));
          }
          wf->set_sleep_cycles(wf_state->sleep_cycles());
          if (wf->supports_sleep_wakeup_credit())
            wf->set_sleep_wakeup(wf_state->sleep_wakeup());
          if (wave_resources &&
              (Gfx10CheckpointResources::restore_wave(*wave_resources, *cu, *wf, emit_error)
                   .failed() ||
               (wave_scheduling &&
                Gfx10CheckpointScheduler::restore_wave(*wave_scheduling, *cu, *wf, emit_error)
                    .failed())))
            return util::Result::failure();
          if (const auto *exceptions = wf_state->exceptions()) {
            const bool gfx12 =
                cu->arch() == ROCJITSU_CODE_ARCH_RDNA4 || cu->arch() == ROCJITSU_CODE_ARCH_CDNA5;
            if ((!gfx12 && (exceptions->gfx12_user_extra() || exceptions->gfx12_trap_ctrl())) ||
                (exceptions->gfx12_user_extra() & ~(3u << 30)))
              return emit_error.emit() << "Invalid checkpoint wave exception registers";
            wf->set_trapsts(exceptions->trapsts());
            wf->set_gfx12_excp_flag_user_extra_raw(exceptions->gfx12_user_extra());
            wf->set_gfx12_trap_ctrl_raw(exceptions->gfx12_trap_ctrl());
          }
          wf->set_mode_raw(wf_state->mode());
          wf->set_wave_sched_mode_raw(wf_state->wave_sched_mode());
          if (wf_state->setreg_vgpr_msb_hazard())
            wf->arm_setreg_vgpr_msb_hazard();

          if (sgprs != nullptr) {
            // Dispatch allocated a logically zero block. Skip zero source
            // registers to preserve lazy backing for untouched waves.
            for (size_t r = 0; r < std::min<size_t>(restore_sgpr_words, sgprs->size()); ++r) {
              const uint32_t value = sgprs->Get(static_cast<unsigned>(r));
              if (value != 0)
                cu->write_sgpr(wf->sgpr_alloc().base + static_cast<uint32_t>(r), value);
            }
          }

          // Older checkpoints stored the SGPR allocation and VCC independently.
          // Restore the explicit condition code last if they disagree.
          if (wf->vcc() != wf_state->vcc())
            wf->set_vcc_raw(wf_state->vcc());
          if (const auto scratch = wf_state->flat_scratch();
              scratch && *scratch != wf->scratch_base())
            wf->set_scratch_base(*scratch);

          if (auto *ttmps = wf_state->ttmps()) {
            for (uint32_t t = 0; t < ttmps->size() && t < 16; ++t)
              wf->set_ttmp(t, ttmps->Get(t));
          } else if (sgprs != nullptr && sgprs->size() > kLegacyTtmpSgprBase) {
            // Written before TTMPs were split out of the SGPR file. Back then
            // the decoder aliased scalar selectors 108..123 into the SGPR block
            // on every architecture, so those slots are exactly the TTMP file
            // and the migration needs no per-arch test. Leaving them behind
            // loses real launch state: RDNA4 and GFX1250 seed TTMP6/7/9 with
            // workgroup ids at dispatch, and TTMP operands no longer read the
            // SGPR slots the old checkpoint restores them into.
            const uint32_t available = std::min<uint32_t>(16, sgprs->size() - kLegacyTtmpSgprBase);
            for (uint32_t t = 0; t < available; ++t)
              wf->set_ttmp(t, sgprs->Get(kLegacyTtmpSgprBase + t));
          }

          // Dispatch identity. A record written before this field existed has
          // to fall back to something defined, and the only other place the
          // coordinate appears is the TTMP file: on the architectures that do
          // not carry workgroup ids in TTMP6/7/9, trap entry writes x/y/z into
          // TTMP8/9/10 and the CWSR codec reads them from there, so those slots
          // are authoritative for any wave that reached a trap handler. Waves
          // that never trapped leave them zero, which is what the coordinate
          // restored to before this field was added; nothing else in an old
          // record can improve on that.
          if (auto *wg_coord = wf_state->wg_coord(); wg_coord != nullptr && wg_coord->size() >= 3) {
            wf->set_wg_coord(wg_coord->Get(0), wg_coord->Get(1), wg_coord->Get(2));
          } else if (!isa_properties(cu->arch()).uses_ttmp_workgroup_ids) {
            wf->set_wg_coord(wf->ttmp(8), wf->ttmp(9), wf->ttmp(10));
          }

          // dispatch_wf_at() has just allocated this block, so RegisterFile::allocate()'s
          // zero-state postcondition satisfies the sparse restore helper's precondition.
          if (auto *vgprs = wf_state->vgprs()) {
            if (restore_vgpr_block_into_zeroed_storage(*cu, wf->vgpr_alloc().base, *vgprs,
                                                       wf_state->vgpr_lane_count(), wf->wf_size(),
                                                       emit_error)
                    .failed())
              return util::Result::failure();
          }
        }
      }
    }
  }

  // Return the same root shape as the JSON configuration loader.
  result.engine_config = engine_config;
  result.build_result.root = std::move(soc);
  result.build_result.memory = mem_ptr;
  return result;
}

} // namespace config
} // namespace rocjitsu
