// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_cndmask_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for v_cndmask_b32
/// (CDNA4 VOP2 opcode 0): dst[lane] = (VCC bit) ? vsrc1 : src0. VCC is an input
/// side-channel that drives the per-lane select. The test seeds distinct
/// src0/vsrc1 values and sweeps several VCC patterns (which lanes pick vsrc1).
/// Each (vcc) case runs TWICE in the same process -- once forcing the scalar
/// body, once the SIMD fast path, with identical seed/inputs/EXEC/VCC -- and the
/// two 64-lane result arrays are asserted equal with EXPECT_EQ
/// (util::set_force_scalar_for_testing flips the gate in-process).
/// In-process inactive EXEC lanes must keep the dst sentinel.

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;

constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop2_cndmask_mem"), l2("vop2_cndmask_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop2_cndmask", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint64_t seed, uint64_t exec, uint64_t vcc) {
    std::mt19937_64 rng(seed);
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      cu->write_vgpr(vbase + 0, lane, static_cast<uint32_t>(rng())); // src0
      cu->write_vgpr(vbase + 1, lane, static_cast<uint32_t>(rng())); // vsrc1
      cu->write_vgpr(vbase + 2, lane, DST_SENTINEL);                 // dst
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc);
  }

  std::array<uint32_t, WF_SIZE> snapshot_dst() const {
    std::array<uint32_t, WF_SIZE> out{};
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      out[lane] = cu->read_vgpr(vbase + 2, lane);
    return out;
  }

  std::array<uint32_t, WF_SIZE> run(Instruction *inst, uint64_t seed, uint64_t exec, uint64_t vcc) {
    seed_inputs(seed, exec, vcc);
    cu->execute_instruction(inst, *wf);
    return snapshot_dst();
  }
};

const uint64_t kVccPatterns[] = {
    0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xAAAAAAAAAAAAAAAAULL,
    0x5555555555555555ULL, 0x0123456789ABCDEFULL, 0xF0F0F0F00F0F0F0FULL,
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

void check_case(uint64_t exec) {
  ForceScalarGuard gate_guard;

  constexpr uint64_t SEED = 0xCD'1234'5678'9AB0ULL;

  auto run_mode = [&](bool force_scalar, uint64_t vcc) -> std::array<uint32_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t enc = vop2_encode(/*opcode=*/0, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_cndmask_b32: decode failed";
    auto out = fx.run(inst, SEED, exec, vcc);
    delete inst;
    return out;
  };

  for (uint64_t vcc : kVccPatterns) {
    const auto scalar_out = run_mode(/*force_scalar=*/true, vcc);
    const auto simd_out = run_mode(/*force_scalar=*/false, vcc);

    EXPECT_EQ(scalar_out, simd_out) << "v_cndmask_b32 vcc=0x" << std::hex << vcc << " (exec=0x"
                                    << exec << "): SIMD path diverged from scalar body";

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = (exec >> lane) & 1ULL;
      if (!active) {
        EXPECT_EQ(simd_out[lane], DST_SENTINEL)
            << "vcc=0x" << std::hex << vcc << ": clobbered inactive lane " << std::dec << lane;
        EXPECT_EQ(scalar_out[lane], DST_SENTINEL)
            << "vcc=0x" << std::hex << vcc << ": clobbered inactive lane " << std::dec << lane;
      }
    }
  }
}

TEST(Vop2CndmaskSimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_case(/*exec=*/~0ULL);
}

TEST(Vop2CndmaskSimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_case(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
