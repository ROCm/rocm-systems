// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_sim.h
/// @brief Minimal single-CU simulator harness shared by the DBI simulator tests.
///
/// Lays out a kernel descriptor plus code in GPU memory (AMDHSA ABI), dispatches
/// one workgroup, runs to completion, and reads VGPRs back from the halt
/// snapshot. A wavefront frees its register file at s_endpgm, so the final
/// register state has to come from a HaltSnapshotPlugin captured at halt rather
/// than from the CU.
///
/// @warning The descriptor this dispatches is **not** the patched code object's.
/// write_kernel() synthesizes a fresh one with 256 VGPRs and 104 SGPRs, taking
/// only private_segment_fixed_size from the caller. So a test that patches an
/// ELF whose descriptor advertises a small allocation still executes with the
/// full register file: the orchestrator's register-ownership gates are exercised
/// statically, at patch time, and never by execution here.

#pragma once

#include "../aql_queue.h"
#include "../halt_snapshot_plugin.h"
#include "embedded_schema.h"

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace test {

/// @brief One-CU simulator (CDNA3, CDNA4, or RDNA4, selected by the constructor
///        arch) for executing a patched DBI kernel.
class DbiSim {
public:
  /// @param arch Config/VM arch string: "cdna3", "cdna4", or "rdna4".
  /// @param wave_size 64 for CDNA, 32 for RDNA4.
  DbiSim(std::string_view arch, uint32_t wave_size) : wave_size_(wave_size) {
    const std::string json =
        std::string(R"({"max_ticks":100000,"num_threads":1,"vm":{"arch":")") + std::string(arch) +
        R"("},)"
        R"("topology":{"root":{"name":"soc","type":"soc","children":[)"
        R"({"name":"vram","type":"gpu_memory"},)"
        R"({"name":"xcd0","type":"xcd","children":[)"
        R"({"name":"l2","type":"l2_cache"},)"
        R"({"name":"cp","type":"command_processor"},)"
        R"({"name":"se0","type":"shader_engine","children":[)"
        R"({"name":"cu[0:1]","type":"compute_unit","config":[)"
        R"({"key":"num_wf_slots","value":"10"},)"
        R"({"key":"sgprs_per_wf","value":"800"},)"
        R"({"key":"vgprs_per_wf","value":"256"},)"
        R"({"key":"lds_size_kb","value":"64"})"
        R"(]}]}]}]},"links":[)"
        R"({"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},)"
        R"({"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10})"
        R"(]}})";
    loaded_ = config::load_config_from_string(json, kEmbeddedSchema);
    soc_ = loaded_.soc();
    mem_ = loaded_.memory();
    engine_ = std::make_unique<simdojo::SimulationEngine>(loaded_.engine_config);
    engine_->topology().set_root(loaded_.take_root());
    loaded_.wire_links(engine_->topology());
    engine_->create();
    plugin_group_ = make_halt_snapshot_group(&snapshot_plugin_);
    soc_->set_plugin_group(plugin_group_);
  }

  amdgpu::CommandProcessor *cp() { return soc_->xcd(0)->command_processor(); }

  /// @brief Write a kernel_descriptor_t (entry at code start) followed by
  ///        @p code, with @p private_bytes of per-lane scratch.
  /// @return The kernel_object address.
  /// @note The descriptor is synthesized, not the patched object's -- see the
  ///   file-level warning.
  uint64_t write_kernel(uint64_t addr, const std::vector<uint32_t> &code, uint32_t private_bytes) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    ((256 / 8) - 1));
    // 104 SGPRs is ample for the probe link pair s[30:31] and envelope temps.
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    ((104 / 8) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    kd.private_segment_fixed_size = private_bytes;

    mem_->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem_->load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * sizeof(uint32_t),
                     addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  /// @brief Dispatch @p code over one wave and return, for each register in
  ///        @p regs, its per-lane values after the kernel halts.
  ///
  /// @details Reading several registers from one dispatch rather than one each
  /// keeps assertions about them statements about a single execution.
  ///
  /// @return One vector per entry of @p regs, each @p wave_size long; empty when
  ///   no wave halted (the kernel did not run to completion). Callers must check
  ///   the size before indexing per lane.
  std::vector<std::vector<uint32_t>> run_and_read_vgprs(const std::vector<uint32_t> &code,
                                                        uint32_t private_bytes,
                                                        const std::vector<uint32_t> &regs) {
    const uint64_t ko = write_kernel(0x1000, code, private_bytes);
    // Callers reuse one DbiSim for several dispatches, and each AqlQueue leaves
    // its registration behind on the CP, so every run needs its own queue --
    // its own id, since two live queues sharing one on a CP are rejected (fan-out
    // routes shards back by (queue_id, process_id)), and its own ring and pointer
    // page, since queues sharing a ring would let one doorbell be fetched and
    // dispatched once per registration.
    const uint32_t queue_id = ++queue_seq_;
    const uint64_t ring = AqlQueue::DEFAULT_RING_ADDR + uint64_t{queue_id} * 0x100000ULL;
    AqlQueue queue(mem_, cp(), ring, AqlQueue::DEFAULT_RING_SIZE, ring + 0x10000, ring + 0x10008,
                   ring + 0x10010, /*xcd_fanout=*/false, /*queue_id=*/queue_id);
    queue.dispatch(ko, /*grid_size_x=*/wave_size_, /*workgroup_size_x=*/wave_size_);
    engine_->run();

    if (snapshot_plugin_->snapshots().empty())
      return std::vector<std::vector<uint32_t>>(regs.size());
    const WavefrontSnapshot &wf = snapshot_plugin_->snapshots().front();

    std::vector<std::vector<uint32_t>> out(regs.size());
    for (size_t i = 0; i < regs.size(); ++i) {
      out[i].resize(wave_size_);
      for (uint32_t lane = 0; lane < wave_size_; ++lane)
        out[i][lane] = wf.vgpr(regs[i], lane);
    }
    return out;
  }

  /// @brief run_and_read_vgprs() for a single register.
  std::vector<uint32_t> run_and_read_vgpr(const std::vector<uint32_t> &code, uint32_t private_bytes,
                                          uint32_t reg) {
    return run_and_read_vgprs(code, private_bytes, {reg}).front();
  }

private:
  uint32_t wave_size_ = 64;
  uint32_t queue_seq_ = 0;
  config::LoadedConfig loaded_;
  SoC *soc_ = nullptr;
  amdgpu::GpuMemory *mem_ = nullptr;
  std::shared_ptr<ExecutionPluginGroup> plugin_group_;
  HaltSnapshotPlugin *snapshot_plugin_ = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
};

} // namespace test
} // namespace rocjitsu
