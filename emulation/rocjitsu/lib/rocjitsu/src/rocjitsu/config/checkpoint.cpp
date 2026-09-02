// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "checkpoint_generated.h"
#include "flatbuffers/flatbuffers.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace rocjitsu {
namespace config {

namespace {

/// @brief First scalar selector that named a TTMP before TTMPs became their own
/// register file.
/// @details The decoder used to alias selectors 108..123 into the wavefront SGPR
/// block on every architecture, so in a checkpoint written then those slots hold
/// exactly the TTMP file. Selectors that high are unreachable as ordinary SGPR
/// operands, so reading them back as TTMPs needs no per-architecture test.
constexpr uint32_t kLegacyTtmpSgprBase = 108;

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
serialize_vgpr_block(flatbuffers::FlatBufferBuilder &builder, const amdgpu::ComputeUnitCore &cu,
                     uint32_t base, uint32_t lane_count) {
  if ((lane_count != 32 && lane_count != 64) || lane_count > cu.vgpr_storage_lane_count())
    throw std::runtime_error("Invalid VGPR checkpoint lane count");
  const size_t register_bytes = static_cast<size_t>(lane_count) * sizeof(uint32_t);
  const size_t block_bytes = static_cast<size_t>(cu.vgpr_allocation_block_size()) * register_bytes;
  uint8_t *serialized = nullptr;
  const auto offset = builder.CreateUninitializedVector<uint8_t>(block_bytes, &serialized);
  size_t offset_bytes = 0;
  cu.for_each_raw_vgpr(base, cu.vgpr_allocation_block_size(), [&](std::span<const uint32_t> lanes) {
    if (lanes.size() < lane_count)
      throw std::runtime_error("VGPR storage is narrower than the checkpoint wave");
    std::copy_n(reinterpret_cast<const uint8_t *>(lanes.data()), register_bytes,
                serialized + offset_bytes);
    offset_bytes += register_bytes;
  });
  return offset;
}

/// Restore sparse checkpoint data into a freshly allocated, zeroed VGPR block.
/// Zero source registers are skipped to preserve lazy backing; this is a full
/// restore, rather than a merge, only while the destination begins entirely
/// zeroed.
void restore_vgpr_block_into_zeroed_storage(amdgpu::ComputeUnitCore &cu, uint32_t base,
                                            const flatbuffers::Vector<uint8_t> &stored,
                                            uint32_t stored_lane_count,
                                            uint32_t architectural_wave_size) {
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
    throw std::runtime_error("Invalid VGPR checkpoint lane count");
  const size_t register_bytes = static_cast<size_t>(lane_count) * sizeof(uint32_t);
  if (stored.size() != static_cast<size_t>(register_count) * register_bytes)
    throw std::runtime_error("Invalid VGPR checkpoint payload size");
  for (uint32_t reg = 0; reg < register_count; ++reg) {
    cu.restore_raw_vgprs_into_zeroed_storage(
        base + reg, 1, bytes.subspan(static_cast<size_t>(reg) * register_bytes, register_bytes));
  }
}

/// @brief Serialize the SoC configuration into a FlatBuffer SimulationConfig.
flatbuffers::Offset<fb::SimulationConfig>
serialize_config(flatbuffers::FlatBufferBuilder &builder, const SoC &soc,
                 const simdojo::SimulationEngine::Config &engine_config) {
  auto arch_str = builder.CreateString(arch_to_string(soc.arch()));
  auto exec_mode_str = builder.CreateString(
      soc.exec_mode() == simdojo::ExecMode::CLOCKED ? "clocked" : "functional");

  // Extract configuration from the live component tree.
  uint32_t num_xcds = soc.num_xcds();
  uint32_t num_iods = soc.num_iods();
  uint32_t num_ses = 0;
  uint32_t num_cus = 0;
  flatbuffers::Offset<fb::ComputeUnitConfig> fb_cu;

  if (num_xcds > 0) {
    auto *xcd = soc.xcd(0);
    num_ses = xcd->num_shader_engines();
    if (num_ses > 0) {
      auto *se = xcd->shader_engine(0);
      num_cus = se->num_compute_units();
      if (num_cus > 0) {
        const auto &cu_cfg = se->compute_unit(0)->config();
        fb_cu = fb::CreateComputeUnitConfig(builder, cu_cfg.num_wf_slots, cu_cfg.sgprs_per_wf,
                                            cu_cfg.vgprs_per_wf, cu_cfg.lds_size_kb);
      }
    }
  }

  auto fb_se = fb::CreateShaderEngineConfig(builder, num_cus, fb_cu);
  auto fb_xcd = fb::CreateXcdConfig(builder, num_ses, fb_se);
  auto fb_gpu = fb::CreateAmdgpuConfig(builder, num_xcds, num_iods, fb_xcd);
  auto fb_vm = fb::CreateVirtualMachineConfig(builder, arch_str, fb_gpu);

  return fb::CreateSimulationConfig(builder, engine_config.max_ticks, engine_config.num_threads,
                                    exec_mode_str, fb_vm);
}

/// @brief Reconstruct a VirtualMachine::Config from a stored FlatBuffer config.
VirtualMachine::Config config_from_checkpoint(const fb::SimulationConfig *fb_config) {
  VirtualMachine::Config vm_config{};

  if (!fb_config || !fb_config->vm())
    throw std::runtime_error("Checkpoint missing simulation config");

  auto *vm = fb_config->vm();
  if (vm->arch())
    vm_config.soc.arch = parse_arch(vm->arch()->str());
  if (vm_config.soc.arch == ROCJITSU_CODE_ARCH_INVALID)
    throw std::runtime_error("Checkpoint has missing or invalid architecture");
  vm_config.soc.exec_mode =
      parse_exec_mode(fb_config->exec_mode() ? fb_config->exec_mode()->str() : "");

  if (auto *gpu = vm->gpu()) {
    vm_config.soc.num_xcds = gpu->num_xcds();
    vm_config.soc.num_iods = gpu->num_iods();
    if (auto *xcd = gpu->xcd()) {
      vm_config.soc.xcd.num_shader_engines = xcd->num_shader_engines();
      if (auto *se = xcd->shader_engine()) {
        vm_config.soc.xcd.shader_engine.num_compute_units = se->num_compute_units();
        if (auto *cu = se->compute_unit()) {
          auto &cu_cfg = vm_config.soc.xcd.shader_engine.compute_unit;
          cu_cfg.num_wf_slots = cu->num_wf_slots();
          cu_cfg.sgprs_per_wf = cu->sgprs_per_wf();
          cu_cfg.vgprs_per_wf = cu->vgprs_per_wf();
          cu_cfg.lds_size_kb = cu->lds_size_kb();
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

void save_checkpoint(const std::string &path, const SoC &soc, uint64_t tick,
                     const simdojo::SimulationEngine::Config &engine_config) {
  flatbuffers::FlatBufferBuilder builder(1024 * 1024);

  // Serialize compute unit states across all XCDs and their shader engines.
  std::vector<flatbuffers::Offset<fb::ComputeUnitState>> cu_offsets;
  for (auto *xcd : soc.xcds()) {
    for (auto *se : xcd->shader_engines()) {
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci) {
        auto *cu = se->compute_unit(ci);
        std::vector<flatbuffers::Offset<fb::WavefrontState>> wf_offsets;
        for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
          const auto *w = cu->wf(i);
          // Only checkpoint active (non-halted) wavefronts. Idle slots
          // have no register allocations and nothing meaningful to save.
          if (w->is_halted())
            continue;

          // The record holds the architectural registers and the TTMPs, but
          // none of the trap/debug state around them: in_trap_handler, the
          // saved EXEC and STATUS the handler will restore, TRAPSTS, and the
          // halted/suspended reasons. Restoring a wave captured mid-handler
          // from such a record would skip the EXEC restore and the privileged
          // STATUS write on the way out and resume the application with the
          // handler's state installed -- silently wrong, and hard to trace
          // back here. Refuse instead of writing a checkpoint that cannot be
          // restored faithfully.
          //
          // debug_stopped(), not debug_paused(): a wave whose only pause reason
          // is the runtime's (queue_percentage 0) carries no trap or debugger
          // state, and the CP re-derives queue suspension on restore. Refusing
          // it would make an ordinary throttled queue unable to checkpoint, and
          // blame a debugger that is not attached.
          if (w->in_trap_handler() || w->debug_stopped())
            throw std::runtime_error(
                "Cannot checkpoint " + cu->name() + " wf" + std::to_string(w->wf_id()) +
                ": the wave is in a trap handler or stopped for a debugger, and that state "
                "is not part of the checkpoint format");

          auto sgprs_vec =
              builder.CreateVector(cu->sgpr_data(w->sgpr_alloc().base), w->num_sgprs());
          auto vgprs_vec = serialize_vgpr_block(builder, *cu, w->vgpr_alloc().base, w->wf_size());

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

          auto wfs = fb::CreateWavefrontState(
              builder, w->wf_id(), w->wg_id(), w->pc, w->exec_raw(), w->vcc(), w->m0(),
              w->is_halted(), w->status_raw(), sgprs_vec, vgprs_vec, w->mode_raw(),
              w->wave_sched_mode_raw(), ttmps_vec, wg_coord_vec, w->kernel_wave_size(),
              w->wf_size(), w->setreg_vgpr_msb_hazard());
          wf_offsets.push_back(wfs);
        }

        auto name = builder.CreateString(cu->name());
        auto wfs_vec = builder.CreateVector(wf_offsets);
        auto cus = fb::CreateComputeUnitState(builder, name, wfs_vec, 0);
        cu_offsets.push_back(cus);
      }
    }
  }

  // Serialize command processor state (first XCD's CP).
  flatbuffers::Offset<fb::CommandProcessorState> cp_offset;
  if (!soc.xcds().empty()) {
    auto *cp = soc.xcds()[0]->command_processor();
    auto name = builder.CreateString(cp->name());
    cp_offset =
        fb::CreateCommandProcessorState(builder, name, cp->dispatched_count(), cp->next_cu_index());
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
  auto config_offset = serialize_config(builder, soc, engine_config);

  auto checkpoint =
      fb::CreateSimulationCheckpoint(builder, tick, config_offset, cu_vec, cp_offset, mem_state);
  builder.Finish(checkpoint);

  // Write to file.
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Cannot open checkpoint file for writing: " + path);
  f.write(reinterpret_cast<const char *>(builder.GetBufferPointer()), builder.GetSize());
}

LoadedConfig restore_checkpoint(const std::string &path) {
  // Read binary file.
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    throw std::runtime_error("Cannot open checkpoint file: " + path);
  auto pos = f.tellg();
  if (pos <= 0)
    throw std::runtime_error("Empty or unreadable checkpoint file: " + path);
  auto size = static_cast<size_t>(pos);
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(size);
  if (!f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size)))
    throw std::runtime_error("Failed to read checkpoint file: " + path);

  flatbuffers::Verifier verifier(buf.data(), buf.size());
  if (!fb::VerifySimulationCheckpointBuffer(verifier))
    throw std::runtime_error("Invalid checkpoint format: " + path);
  auto *checkpoint = fb::GetSimulationCheckpoint(buf.data());

  // Rebuild SoC and engine config from the stored configuration.
  if (!checkpoint->config())
    throw std::runtime_error("Checkpoint missing config section: " + path);
  auto *fb_config = checkpoint->config();
  VirtualMachine::Config vm_config = config_from_checkpoint(fb_config);
  simdojo::SimulationEngine::Config engine_config{};
  engine_config.max_ticks = fb_config->max_ticks();
  engine_config.num_threads = fb_config->num_threads();

  // Rebuild the SoC root expected by LoadedConfig and create_from_loaded().
  auto soc = std::make_unique<SoC>("gpu_soc", vm_config.soc);
  auto *soc_ptr = soc.get();
  auto *mem_ptr = soc->memory();

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
      if (auto *wf_states = cu_state->wavefronts()) {
        for (auto *wf_state : *wf_states) {
          uint32_t num_sgprs =
              wf_state->sgprs() ? wf_state->sgprs()->size() : cu->config().sgprs_per_wf;
          uint32_t num_vgprs = cu->config().vgprs_per_wf;

          const uint32_t wave_size =
              wf_state->kernel_wave_size() == 0 ? cu->wf_size() : wf_state->kernel_wave_size();
          auto *wf = cu->dispatch_wf_at(wf_state->wf_id(), wf_state->wg_id(), wf_state->pc(),
                                        num_sgprs, num_vgprs, wave_size);
          if (!wf)
            throw std::runtime_error("Failed to restore wavefront into its recorded slot");

          wf->set_exec_raw(wf_state->exec());
          wf->set_vcc_raw(wf_state->vcc());
          wf->set_m0(wf_state->m0());
          // Halted wavefronts are never saved (see save_checkpoint skip above),
          // so halted() is always false here. Keep the branch for future-proofing.
          wf->set_state(wf_state->halted() ? amdgpu::WfState::HALTED : amdgpu::WfState::RUNNING);
          wf->set_status_raw(wf_state->status());
          wf->set_mode_raw(wf_state->mode());
          wf->set_wave_sched_mode_raw(wf_state->wave_sched_mode());
          if (wf_state->setreg_vgpr_msb_hazard())
            wf->arm_setreg_vgpr_msb_hazard();

          const auto *sgprs = wf_state->sgprs();
          if (sgprs != nullptr) {
            for (size_t r = 0; r < sgprs->size() && r < wf->num_sgprs(); ++r) {
              cu->write_sgpr(wf->sgpr_alloc().base + static_cast<uint32_t>(r),
                             sgprs->Get(static_cast<unsigned>(r)));
            }
          }

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
          if (auto *vgprs = wf_state->vgprs())
            restore_vgpr_block_into_zeroed_storage(*cu, wf->vgpr_alloc().base, *vgprs,
                                                   wf_state->vgpr_lane_count(), wf->wf_size());
        }
      }
    }
  }

  // Return the same root shape as the JSON configuration loader.
  LoadedConfig result;
  result.engine_config = engine_config;
  result.exec_mode = vm_config.soc.exec_mode;
  result.build_result.root = std::move(soc);
  result.build_result.memory = mem_ptr;
  return result;
}

} // namespace config
} // namespace rocjitsu
