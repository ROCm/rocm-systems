// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_CTS_CTS_COMMON_H_
#define ROCJITSU_TESTS_CTS_CTS_COMMON_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace rocjitsu::cts {

struct IsaConfig {
  rj_code_arch_t arch;
  std::string_view name;
  uint32_t wf_size;
};

inline constexpr IsaConfig kIsaConfigs[] = {
    {ROCJITSU_CODE_ARCH_CDNA1, "cdna1", 64}, {ROCJITSU_CODE_ARCH_CDNA2, "cdna2", 64},
    {ROCJITSU_CODE_ARCH_CDNA3, "cdna3", 64}, {ROCJITSU_CODE_ARCH_CDNA4, "cdna4", 64},
    {ROCJITSU_CODE_ARCH_RDNA1, "rdna1", 32}, {ROCJITSU_CODE_ARCH_RDNA2, "rdna2", 32},
    {ROCJITSU_CODE_ARCH_RDNA3, "rdna3", 32}, {ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", 32},
    {ROCJITSU_CODE_ARCH_RDNA4, "rdna4", 32}, {ROCJITSU_CODE_ARCH_GFX1250, "gfx1250", 32},
};

inline constexpr uint32_t kSgprsPerWf = 106;
inline constexpr uint32_t kVgprsPerWf = 512;
inline constexpr uint32_t kLdsSizeKb = 64;

class CtsFixture {
public:
  explicit CtsFixture(const IsaConfig &cfg)
      : gpu_mem_(std::string(cfg.name) + "_cts_mem"), l2_(std::string(cfg.name) + "_cts_l2"),
        cfg_(cfg) {
    amdgpu::ComputeUnitCore::Config cu_cfg{};
    cu_cfg.arch = cfg.arch;
    cu_cfg.num_wf_slots = 1;
    cu_cfg.sgprs_per_wf = kSgprsPerWf;
    cu_cfg.vgprs_per_wf = kVgprsPerWf;
    cu_cfg.lds_size_kb = kLdsSizeKb;
    cu_ =
        amdgpu::ComputeUnitCore::create(std::string(cfg.name) + "_cts_cu", cu_cfg, &gpu_mem_, &l2_);
    decoder_ = Decoder::create(cfg.arch);
    wf_ = cu_->dispatch_wf(0, 0, kSgprsPerWf, kVgprsPerWf);
  }

  void write_sgpr(uint32_t idx, uint32_t value) {
    cu_->write_sgpr(wf_->sgpr_alloc().base + idx, value);
  }

  uint32_t read_sgpr(uint32_t idx) const { return cu_->read_sgpr(wf_->sgpr_alloc().base + idx); }

  void write_vgpr(uint32_t idx, uint32_t lane, uint32_t value) {
    cu_->write_vgpr(wf_->vgpr_alloc().base + idx, lane, value);
  }

  uint32_t read_vgpr(uint32_t idx, uint32_t lane) const {
    return cu_->read_vgpr(wf_->vgpr_alloc().base + idx, lane);
  }

  bool read_scc() const { return wf_->read_scc(); }
  uint64_t read_vcc() const { return wf_->vcc(); }
  void set_vcc(uint64_t val) { wf_->set_vcc(val); }
  uint64_t read_exec() const { return wf_->exec(); }
  void set_exec(uint64_t val) { wf_->set_exec(val); }

  bool decode_execute(const uint32_t *words) {
    auto *inst = decoder_->decode(words);
    if (!inst)
      return false;
    cu_->execute_instruction(inst, *wf_);
    if (inst->is_memory_op()) {
      cu_->route_memory_inst(inst, *wf_);
    } else {
      delete inst;
    }
    return true;
  }

  void reset() {
    cu_->reset_all_wf();
    wf_ = cu_->dispatch_wf(0, 0, kSgprsPerWf, kVgprsPerWf);
  }

  void write_lds32(uint32_t addr, uint32_t value) { cu_->lds().write32(addr, value); }
  uint32_t read_lds32(uint32_t addr) const { return cu_->lds().read32(addr); }
  void write_lds64(uint32_t addr, uint64_t value) { cu_->lds().write64(addr, value); }
  uint64_t read_lds64(uint32_t addr) const { return cu_->lds().read64(addr); }
  void clear_lds() { cu_->lds().clear(); }

  const IsaConfig &isa_config() const { return cfg_; }
  uint32_t wf_size() const { return cfg_.wf_size; }

private:
  amdgpu::GpuMemory gpu_mem_;
  amdgpu::L2Cache l2_;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wf_ = nullptr;
  IsaConfig cfg_;
};

} // namespace rocjitsu::cts

#endif // ROCJITSU_TESTS_CTS_CTS_COMMON_H_
