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
///
/// The kernarg segment pointer is one place the two descriptors do have to
/// agree, since the DBI entry prologue reads it at a slot fixed at patch time
/// from the patched descriptor's enable bits. set_kernarg() is what makes the
/// synthesized descriptor carry the same properties word, so the CP writes the
/// pointer where the prologue expects it rather than where it happens to land.
///
/// Wave size is a place they do **not** agree: the synthesized descriptor never
/// sets ENABLE_WAVEFRONT_SIZE32, so kernel_wavefront_size() reports 64 and an
/// RDNA target dispatches Wave64 however its patched ELF was planned. The
/// wave_size constructor argument only says how many lanes to read back. A test
/// whose correctness depends on the executing wave size cannot get it here.

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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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

  /// @brief Address the kernarg image passed to set_kernarg() is loaded at, and
  ///        therefore the value the CP places in the kernarg SGPR pair.
  static constexpr uint64_t KERNARG_ADDR = 0x40000;

  /// @brief Supply the bytes dispatched as the kernarg segment.
  /// @details Loaded at KERNARG_ADDR on every subsequent run. @p properties goes
  /// into the synthesized descriptor's kernel_code_properties, and the default
  /// declares the kernarg pointer as the only user SGPR, putting it at s[0:1].
  /// Pass more properties only to move the pointer deliberately; @p
  /// user_sgpr_count must then be the total those properties imply, since a
  /// descriptor whose count disagrees with its enable bits places the system
  /// SGPRs somewhere no real kernel would read.
  ///
  /// A DbiSim that is never given kernargs leaves the word 0, which takes the
  /// CP's fallback for internal test dispatches, the else branch of the enable-bit
  /// walk in CommandProcessor::init_wavefront_regs.
  /// That path writes nothing when the dispatch carries no kernarg address.
  void set_kernarg(std::vector<uint8_t> bytes,
                   uint32_t properties =
                       rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                   uint32_t user_sgpr_count = 2) {
    kernarg_bytes_ = std::move(bytes);
    kernarg_properties_ = properties;
    user_sgpr_count_ = user_sgpr_count;
  }

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
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, user_sgpr_count_);
    kd.kernel_code_properties = kernarg_properties_;
    kd.kernarg_size = static_cast<uint32_t>(kernarg_bytes_.size());
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
    const WavefrontSnapshot *wf = run(code, private_bytes);
    if (wf == nullptr)
      return std::vector<std::vector<uint32_t>>(regs.size());

    std::vector<std::vector<uint32_t>> out(regs.size());
    for (size_t i = 0; i < regs.size(); ++i) {
      out[i].resize(wave_size_);
      for (uint32_t lane = 0; lane < wave_size_; ++lane)
        out[i][lane] = wf->vgpr(regs[i], lane);
    }
    return out;
  }

  /// @brief run_and_read_vgprs() for a single register.
  std::vector<uint32_t> run_and_read_vgpr(const std::vector<uint32_t> &code, uint32_t private_bytes,
                                          uint32_t reg) {
    return run_and_read_vgprs(code, private_bytes, {reg}).front();
  }

  /// @brief Dispatch @p code over one wave and return the 64-bit SGPR pair based
  ///        at architectural index @p base after the kernel halts.
  /// @return std::nullopt when no wave halted, so an absent value is
  ///   distinguishable from a genuine zero pair.
  std::optional<uint64_t> run_and_read_sgpr64(const std::vector<uint32_t> &code,
                                              uint32_t private_bytes, uint32_t base) {
    const WavefrontSnapshot *wf = run(code, private_bytes);
    if (wf == nullptr)
      return std::nullopt;
    return wf->sgpr64(base);
  }

private:
  /// @brief Write the kernel and its kernargs, dispatch one wave, and run to
  ///        completion.
  /// @return The halt snapshot, or nullptr when no wave halted (the kernel did
  ///   not run to completion). The plugin never drops a snapshot, so a DbiSim
  ///   reused for several dispatches still reports the first one here.
  const WavefrontSnapshot *run(const std::vector<uint32_t> &code, uint32_t private_bytes) {
    const uint64_t ko = write_kernel(0x1000, code, private_bytes);
    if (!kernarg_bytes_.empty())
      mem_->load_image(kernarg_bytes_.data(), kernarg_bytes_.size(), KERNARG_ADDR);
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
    queue.dispatch(ko, /*grid_size_x=*/wave_size_, /*workgroup_size_x=*/wave_size_,
                   kernarg_bytes_.empty() ? 0 : KERNARG_ADDR);
    engine_->run();

    if (snapshot_plugin_->snapshots().empty())
      return nullptr;
    return &snapshot_plugin_->snapshots().front();
  }

  uint32_t wave_size_ = 64;
  uint32_t queue_seq_ = 0;
  uint32_t kernarg_properties_ = 0;
  uint32_t user_sgpr_count_ = 2;
  std::vector<uint8_t> kernarg_bytes_;
  config::LoadedConfig loaded_;
  SoC *soc_ = nullptr;
  amdgpu::GpuMemory *mem_ = nullptr;
  std::shared_ptr<ExecutionPluginGroup> plugin_group_;
  HaltSnapshotPlugin *snapshot_plugin_ = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
};

} // namespace test
} // namespace rocjitsu
