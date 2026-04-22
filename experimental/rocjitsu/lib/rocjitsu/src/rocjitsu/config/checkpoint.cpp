// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "checkpoint_generated.h"
#include "flatbuffers/flatbuffers.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace rocjitsu {
namespace config {

namespace {

/// @brief Serialize the SoC configuration into a FlatBuffer SimulationConfig.
flatbuffers::Offset<fb::SimulationConfig>
serialize_config(flatbuffers::FlatBufferBuilder &builder, const SoC &soc,
                 const simdojo::SimulationEngine::Config &engine_config) {
  auto arch_str = builder.CreateString(arch_to_string(soc.arch()));

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
                                            cu_cfg.vgprs_per_wf, cu_cfg.lds_size_kb,
                                            cu_cfg.functional_quantum);
      }
    }
  }

  auto fb_se = fb::CreateShaderEngineConfig(builder, num_cus, fb_cu);
  auto fb_xcd = fb::CreateXcdConfig(builder, num_ses, fb_se);
  auto fb_gpu = fb::CreateAmdgpuConfig(builder, num_xcds, num_iods, fb_xcd);
  auto fb_vm = fb::CreateVirtualMachineConfig(builder, arch_str, fb_gpu);

  return fb::CreateSimulationConfig(builder, engine_config.max_ticks, engine_config.num_threads, 0,
                                    fb_vm);
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
          cu_cfg.functional_quantum = cu->functional_quantum();
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

          auto sgprs_vec =
              builder.CreateVector(cu->sgpr_data(w->sgpr_alloc().base), w->num_sgprs());
          size_t vgpr_bytes = static_cast<size_t>(w->num_vgprs()) *
                              static_cast<size_t>(w->wf_size()) * sizeof(uint32_t);
          auto vgprs_vec = builder.CreateVector(cu->vgpr_data(w->vgpr_alloc().base), vgpr_bytes);

          auto wfs = fb::CreateWavefrontState(builder, w->wf_id(), w->wg_id(), w->pc, w->exec(),
                                              w->vcc(), w->m0(), w->is_halted(), w->status_raw(),
                                              sgprs_vec, vgprs_vec);
          wf_offsets.push_back(wfs);
        }

        auto name = builder.CreateString(cu->name());
        auto wfs_vec = builder.CreateVector(wf_offsets);
        auto cus = fb::CreateComputeUnitState(builder, name, wfs_vec, cu->next_wf_index());
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

  auto *checkpoint = fb::GetSimulationCheckpoint(buf.data());
  if (!checkpoint)
    throw std::runtime_error("Invalid checkpoint format: " + path);

  // Rebuild SoC and engine config from the stored configuration.
  if (!checkpoint->config())
    throw std::runtime_error("Checkpoint missing config section: " + path);
  auto *fb_config = checkpoint->config();
  VirtualMachine::Config vm_config = config_from_checkpoint(fb_config);
  simdojo::SimulationEngine::Config engine_config{};
  engine_config.max_ticks = fb_config->max_ticks();
  engine_config.num_threads = fb_config->num_threads();

  // Build a VirtualMachine from the old-format config, then extract SoC as root.
  auto vm = std::make_unique<VirtualMachine>(vm_config);
  auto *soc_ptr = vm->soc();
  auto *mem_ptr = vm->memory();

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

          auto *wf = cu->dispatch_wf(wf_state->wg_id(), wf_state->pc(), num_sgprs, num_vgprs);
          if (!wf)
            throw std::runtime_error("Failed to dispatch wavefront during checkpoint restoration");

          wf->set_exec(wf_state->exec());
          wf->set_vcc(wf_state->vcc());
          wf->set_m0(wf_state->m0());
          // Halted wavefronts are never saved (see save_checkpoint skip above),
          // so halted() is always false here. Keep the branch for future-proofing.
          wf->set_state(wf_state->halted() ? amdgpu::WfState::HALTED : amdgpu::WfState::RUNNING);
          wf->set_status_raw(wf_state->status());

          if (auto *sgprs = wf_state->sgprs()) {
            for (size_t r = 0; r < sgprs->size() && r < wf->num_sgprs(); ++r) {
              cu->write_sgpr(wf->sgpr_alloc().base + static_cast<uint32_t>(r),
                             sgprs->Get(static_cast<unsigned>(r)));
            }
          }

          if (auto *vgprs = wf_state->vgprs()) {
            size_t vgpr_bytes = static_cast<size_t>(wf->num_vgprs()) *
                                static_cast<size_t>(wf->wf_size()) * sizeof(uint32_t);
            size_t copy_size = std::min<size_t>(vgprs->size(), vgpr_bytes);
            std::memcpy(cu->vgpr_data(wf->vgpr_alloc().base), vgprs->data(), copy_size);
          }
        }
      }
    }
  }

  // Return as LoadedConfig with the VirtualMachine as root (legacy checkpoint path).
  LoadedConfig result;
  result.engine_config = engine_config;
  result.build_result.root = std::move(vm);
  result.build_result.memory = mem_ptr;
  return result;
}

} // namespace config
} // namespace rocjitsu
