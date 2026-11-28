// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded half-unary bindings, guest controls and register layouts.
// Independent numerical coverage lives in lib/arith/tests/math/.
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "unary_binding.h"
#include "util/data_types.h"
#include "util/simd.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>

namespace rocjitsu::amdgpu {
namespace {
// Check VM routing/storage against scalar APIs, whose arithmetic has separate
// literal/reference tests in lib/arith. FREXP exponent keeps a standard reference.
TEST(OlderHalfUnary, Bindings) {
  struct Family {
    rocjitsu::IsaFamily arch;
    const IsaExecutionBackend *backend;
    struct Operation {
      int model;
      uint16_t vop1, vop3;
    };
    std::initializer_list<Operation> operations;
  };
  const Family families[] = {
      {rocjitsu::IsaFamily::Cdna1,
       &cdna1::execution_backend(),
       {
           {0, cdna1::kVTruncF16Vop1, cdna1::kVTruncF16Vop3},
           {1, cdna1::kVCeilF16Vop1, cdna1::kVCeilF16Vop3},
           {2, cdna1::kVRndneF16Vop1, cdna1::kVRndneF16Vop3},
           {3, cdna1::kVFloorF16Vop1, cdna1::kVFloorF16Vop3},
           {4, cdna1::kVFrexpMantF16Vop1, cdna1::kVFrexpMantF16Vop3},
           {5, cdna1::kVFrexpExpI16F16Vop1, cdna1::kVFrexpExpI16F16Vop3},
           {6, cdna1::kVRcpF16Vop1, cdna1::kVRcpF16Vop3},
           {7, cdna1::kVRsqF16Vop1, cdna1::kVRsqF16Vop3},
           {8, cdna1::kVSqrtF16Vop1, cdna1::kVSqrtF16Vop3},
           {9, cdna1::kVExpF16Vop1, cdna1::kVExpF16Vop3},
           {10, cdna1::kVLogF16Vop1, cdna1::kVLogF16Vop3},
           {11, cdna1::kVSinF16Vop1, cdna1::kVSinF16Vop3},
           {12, cdna1::kVCosF16Vop1, cdna1::kVCosF16Vop3},
           {13, cdna1::kVFractF16Vop1, cdna1::kVFractF16Vop3},
       }},
      {rocjitsu::IsaFamily::Cdna2,
       &cdna2::execution_backend(),
       {
           {0, cdna2::kVTruncF16Vop1, cdna2::kVTruncF16Vop3},
           {1, cdna2::kVCeilF16Vop1, cdna2::kVCeilF16Vop3},
           {2, cdna2::kVRndneF16Vop1, cdna2::kVRndneF16Vop3},
           {3, cdna2::kVFloorF16Vop1, cdna2::kVFloorF16Vop3},
           {4, cdna2::kVFrexpMantF16Vop1, cdna2::kVFrexpMantF16Vop3},
           {5, cdna2::kVFrexpExpI16F16Vop1, cdna2::kVFrexpExpI16F16Vop3},
           {6, cdna2::kVRcpF16Vop1, cdna2::kVRcpF16Vop3},
           {7, cdna2::kVRsqF16Vop1, cdna2::kVRsqF16Vop3},
           {8, cdna2::kVSqrtF16Vop1, cdna2::kVSqrtF16Vop3},
           {9, cdna2::kVExpF16Vop1, cdna2::kVExpF16Vop3},
           {10, cdna2::kVLogF16Vop1, cdna2::kVLogF16Vop3},
           {11, cdna2::kVSinF16Vop1, cdna2::kVSinF16Vop3},
           {12, cdna2::kVCosF16Vop1, cdna2::kVCosF16Vop3},
           {13, cdna2::kVFractF16Vop1, cdna2::kVFractF16Vop3},
       }},
      {rocjitsu::IsaFamily::Cdna3,
       &cdna3::execution_backend(),
       {
           {0, cdna3::kVTruncF16Vop1, cdna3::kVTruncF16Vop3},
           {1, cdna3::kVCeilF16Vop1, cdna3::kVCeilF16Vop3},
           {2, cdna3::kVRndneF16Vop1, cdna3::kVRndneF16Vop3},
           {3, cdna3::kVFloorF16Vop1, cdna3::kVFloorF16Vop3},
           {4, cdna3::kVFrexpMantF16Vop1, cdna3::kVFrexpMantF16Vop3},
           {5, cdna3::kVFrexpExpI16F16Vop1, cdna3::kVFrexpExpI16F16Vop3},
           {6, cdna3::kVRcpF16Vop1, cdna3::kVRcpF16Vop3},
           {7, cdna3::kVRsqF16Vop1, cdna3::kVRsqF16Vop3},
           {8, cdna3::kVSqrtF16Vop1, cdna3::kVSqrtF16Vop3},
           {9, cdna3::kVExpF16Vop1, cdna3::kVExpF16Vop3},
           {10, cdna3::kVLogF16Vop1, cdna3::kVLogF16Vop3},
           {11, cdna3::kVSinF16Vop1, cdna3::kVSinF16Vop3},
           {12, cdna3::kVCosF16Vop1, cdna3::kVCosF16Vop3},
           {13, cdna3::kVFractF16Vop1, cdna3::kVFractF16Vop3},
       }},
      {rocjitsu::IsaFamily::Cdna4,
       &cdna4::execution_backend(),
       {
           {0, cdna4::kVTruncF16Vop1, cdna4::kVTruncF16Vop3},
           {1, cdna4::kVCeilF16Vop1, cdna4::kVCeilF16Vop3},
           {2, cdna4::kVRndneF16Vop1, cdna4::kVRndneF16Vop3},
           {3, cdna4::kVFloorF16Vop1, cdna4::kVFloorF16Vop3},
           {4, cdna4::kVFrexpMantF16Vop1, cdna4::kVFrexpMantF16Vop3},
           {5, cdna4::kVFrexpExpI16F16Vop1, cdna4::kVFrexpExpI16F16Vop3},
           {6, cdna4::kVRcpF16Vop1, cdna4::kVRcpF16Vop3},
           {7, cdna4::kVRsqF16Vop1, cdna4::kVRsqF16Vop3},
           {8, cdna4::kVSqrtF16Vop1, cdna4::kVSqrtF16Vop3},
           {9, cdna4::kVExpF16Vop1, cdna4::kVExpF16Vop3},
           {10, cdna4::kVLogF16Vop1, cdna4::kVLogF16Vop3},
           {11, cdna4::kVSinF16Vop1, cdna4::kVSinF16Vop3},
           {12, cdna4::kVCosF16Vop1, cdna4::kVCosF16Vop3},
           {13, cdna4::kVFractF16Vop1, cdna4::kVFractF16Vop3},
       }},
      {rocjitsu::IsaFamily::Rdna1,
       &rdna1::execution_backend(),
       {
           {0, rdna1::kVTruncF16Vop1, rdna1::kVTruncF16Vop3},
           {1, rdna1::kVCeilF16Vop1, rdna1::kVCeilF16Vop3},
           {2, rdna1::kVRndneF16Vop1, rdna1::kVRndneF16Vop3},
           {3, rdna1::kVFloorF16Vop1, rdna1::kVFloorF16Vop3},
           {4, rdna1::kVFrexpMantF16Vop1, rdna1::kVFrexpMantF16Vop3},
           {5, rdna1::kVFrexpExpI16F16Vop1, rdna1::kVFrexpExpI16F16Vop3},
           {6, rdna1::kVRcpF16Vop1, rdna1::kVRcpF16Vop3},
           {7, rdna1::kVRsqF16Vop1, rdna1::kVRsqF16Vop3},
           {8, rdna1::kVSqrtF16Vop1, rdna1::kVSqrtF16Vop3},
           {9, rdna1::kVExpF16Vop1, rdna1::kVExpF16Vop3},
           {10, rdna1::kVLogF16Vop1, rdna1::kVLogF16Vop3},
           {11, rdna1::kVSinF16Vop1, rdna1::kVSinF16Vop3},
           {12, rdna1::kVCosF16Vop1, rdna1::kVCosF16Vop3},
           {13, rdna1::kVFractF16Vop1, rdna1::kVFractF16Vop3},
       }},
      {rocjitsu::IsaFamily::Rdna2,
       &rdna2::execution_backend(),
       {
           {0, rdna2::kVTruncF16Vop1, rdna2::kVTruncF16Vop3},
           {1, rdna2::kVCeilF16Vop1, rdna2::kVCeilF16Vop3},
           {2, rdna2::kVRndneF16Vop1, rdna2::kVRndneF16Vop3},
           {3, rdna2::kVFloorF16Vop1, rdna2::kVFloorF16Vop3},
           {4, rdna2::kVFrexpMantF16Vop1, rdna2::kVFrexpMantF16Vop3},
           {5, rdna2::kVFrexpExpI16F16Vop1, rdna2::kVFrexpExpI16F16Vop3},
           {6, rdna2::kVRcpF16Vop1, rdna2::kVRcpF16Vop3},
           {7, rdna2::kVRsqF16Vop1, rdna2::kVRsqF16Vop3},
           {8, rdna2::kVSqrtF16Vop1, rdna2::kVSqrtF16Vop3},
           {9, rdna2::kVExpF16Vop1, rdna2::kVExpF16Vop3},
           {10, rdna2::kVLogF16Vop1, rdna2::kVLogF16Vop3},
           {11, rdna2::kVSinF16Vop1, rdna2::kVSinF16Vop3},
           {12, rdna2::kVCosF16Vop1, rdna2::kVCosF16Vop3},
           {13, rdna2::kVFractF16Vop1, rdna2::kVFractF16Vop3},
       }},
      {rocjitsu::IsaFamily::Rdna3_5,
       &rdna3_5::execution_backend(),
       {
           {0, rdna3_5::kVTruncF16Vop1, rdna3_5::kVTruncF16Vop3},
           {1, rdna3_5::kVCeilF16Vop1, rdna3_5::kVCeilF16Vop3},
           {2, rdna3_5::kVRndneF16Vop1, rdna3_5::kVRndneF16Vop3},
           {3, rdna3_5::kVFloorF16Vop1, rdna3_5::kVFloorF16Vop3},
           {4, rdna3_5::kVFrexpMantF16Vop1, rdna3_5::kVFrexpMantF16Vop3},
           {5, rdna3_5::kVFrexpExpI16F16Vop1, rdna3_5::kVFrexpExpI16F16Vop3},
           {6, rdna3_5::kVRcpF16Vop1, rdna3_5::kVRcpF16Vop3},
           {7, rdna3_5::kVRsqF16Vop1, rdna3_5::kVRsqF16Vop3},
           {8, rdna3_5::kVSqrtF16Vop1, rdna3_5::kVSqrtF16Vop3},
           {9, rdna3_5::kVExpF16Vop1, rdna3_5::kVExpF16Vop3},
           {10, rdna3_5::kVLogF16Vop1, rdna3_5::kVLogF16Vop3},
           {11, rdna3_5::kVSinF16Vop1, rdna3_5::kVSinF16Vop3},
           {12, rdna3_5::kVCosF16Vop1, rdna3_5::kVCosF16Vop3},
           {13, rdna3_5::kVFractF16Vop1, rdna3_5::kVFractF16Vop3},
       }},
      {rocjitsu::IsaFamily::Cdna5,
       &cdna5::execution_backend(),
       {
           {0, cdna5::kVTruncF16Vop1, cdna5::kVTruncF16Vop3},
           {1, cdna5::kVCeilF16Vop1, cdna5::kVCeilF16Vop3},
           {2, cdna5::kVRndneF16Vop1, cdna5::kVRndneF16Vop3},
           {3, cdna5::kVFloorF16Vop1, cdna5::kVFloorF16Vop3},
           {4, cdna5::kVFrexpMantF16Vop1, cdna5::kVFrexpMantF16Vop3},
           {5, cdna5::kVFrexpExpI16F16Vop1, cdna5::kVFrexpExpI16F16Vop3},
           {6, cdna5::kVRcpF16Vop1, cdna5::kVRcpF16Vop3},
           {7, cdna5::kVRsqF16Vop1, cdna5::kVRsqF16Vop3},
           {8, cdna5::kVSqrtF16Vop1, cdna5::kVSqrtF16Vop3},
           {9, cdna5::kVExpF16Vop1, cdna5::kVExpF16Vop3},
           {10, cdna5::kVLogF16Vop1, cdna5::kVLogF16Vop3},
           {11, cdna5::kVSinF16Vop1, cdna5::kVSinF16Vop3},
           {12, cdna5::kVCosF16Vop1, cdna5::kVCosF16Vop3},
           {13, cdna5::kVFractF16Vop1, cdna5::kVFractF16Vop3},
       }},
  };
  const uint16_t values[] = {0x3e00, 0xbe00, 0xb401, 0xc100, 0x8001,
                             1,      0x7d23, 0xfc00, 0x7bff, 0xfbff};
  enum class Form { Vop1, Vop3, Sdwa, Dpp, SuppressedDpp, Dpp8 };
  struct Scenario {
    Form form;
    uint32_t mode;
    unsigned omod;
    bool clamp, high, abs, neg;
  };
  const Scenario scenarios[] = {
      {Form::Vop1, 0, 0, false, false, false, false},
      {Form::Vop1, 0xf0, 0, false, false, false, false},
      {Form::Vop1, 0xf0, 0, false, true, false, false},
      // FRACT(0xb401): nearest/upward -> 0x3a00, downward/zero -> 0x39ff.
      {Form::Vop3, 0xf0, 0, false, false, false, false},
      {Form::Vop3, 0xf4, 0, false, false, false, false},
      {Form::Vop3, 0xf8, 0, false, false, false, false},
      {Form::Vop3, 0xfc, 0, false, false, false, false},
      {Form::Vop3, 0, 1, false, false, false, false},
      {Form::Vop3, 4, 2, false, false, false, false},
      {Form::Vop3, 12, 1, false, false, false, false},
      {Form::Vop3, 0x800000, 2, false, false, false, false},
      {Form::Vop3, 0x200, 1, false, false, false, false},
      {Form::Vop3, 0x80, 1, false, false, false, false},
      {Form::Vop3, 8, 3, false, false, false, false},
      {Form::Vop3, 0, 0, true, false, false, false},
      {Form::Vop3, 0x100, 0, true, false, false, false},
      {Form::Vop3, 0xf0, 0, false, true, false, false},
      {Form::Vop3, 0, 0, false, false, true, true},
      {Form::Sdwa, 0, 1, false, true, false, false},
      {Form::Dpp, 0xf0, 0, false, false, false, false},
      {Form::Dpp, 0xf0, 0, false, true, false, false},
      {Form::Dpp8, 0xf0, 0, false, true, false, false},
      {Form::SuppressedDpp, 0xf0, 0, false, false, false, false},
  };
  // The four numerical contracts are older profiles, CDNA3, gfx1250, and CDNA4.
  const rocjitsu::test::Unary32 calls[4][14] = {
      {rj_arith_v_trunc_f16_w32_gfx908, rj_arith_v_ceil_f16_w32_gfx908,
       rj_arith_v_rndne_f16_w32_gfx908, rj_arith_v_floor_f16_w32_gfx908,
       rj_arith_v_frexp_mant_f16_w32_gfx908, nullptr, rj_arith_v_rcp_f16_w32_gfx908,
       rj_arith_v_rsq_f16_w32_gfx908, rj_arith_v_sqrt_f16_w32_gfx908, rj_arith_v_exp_f16_w32_gfx908,
       rj_arith_v_log_f16_w32_gfx908, rj_arith_v_sin_f16_w32_gfx908, rj_arith_v_cos_f16_w32_gfx908,
       rj_arith_v_fract_f16_w32_gfx908},
      {rj_arith_v_trunc_f16_w32_gfx942, rj_arith_v_ceil_f16_w32_gfx942,
       rj_arith_v_rndne_f16_w32_gfx942, rj_arith_v_floor_f16_w32_gfx942,
       rj_arith_v_frexp_mant_f16_w32_gfx942, nullptr, rj_arith_v_rcp_f16_w32_gfx942,
       rj_arith_v_rsq_f16_w32_gfx942, rj_arith_v_sqrt_f16_w32_gfx942, rj_arith_v_exp_f16_w32_gfx942,
       rj_arith_v_log_f16_w32_gfx942, rj_arith_v_sin_f16_w32_gfx942, rj_arith_v_cos_f16_w32_gfx942,
       rj_arith_v_fract_f16_w32_gfx942},
      {rj_arith_v_trunc_f16_w32_gfx12, rj_arith_v_ceil_f16_w32_gfx12,
       rj_arith_v_rndne_f16_w32_gfx12, rj_arith_v_floor_f16_w32_gfx12,
       rj_arith_v_frexp_mant_f16_w32_gfx12, nullptr, rj_arith_v_rcp_f16_w32_gfx1250,
       rj_arith_v_rsq_f16_w32_gfx1250, rj_arith_v_sqrt_f16_w32_gfx1250,
       rj_arith_v_exp_f16_w32_gfx1250, rj_arith_v_log_f16_w32_gfx1250,
       rj_arith_v_sin_f16_w32_gfx1250, rj_arith_v_cos_f16_w32_gfx1250,
       rj_arith_v_fract_f16_w32_gfx12},
      {rj_arith_v_trunc_f16_w32_gfx942, rj_arith_v_ceil_f16_w32_gfx942,
       rj_arith_v_rndne_f16_w32_gfx942, rj_arith_v_floor_f16_w32_gfx942,
       rj_arith_v_frexp_mant_f16_w32_gfx942, nullptr, rj_arith_v_rcp_f16_w32_gfx950,
       rj_arith_v_rsq_f16_w32_gfx950, rj_arith_v_sqrt_f16_w32_gfx950, rj_arith_v_exp_f16_w32_gfx950,
       rj_arith_v_log_f16_w32_gfx950, rj_arith_v_sin_f16_w32_gfx950, rj_arith_v_cos_f16_w32_gfx950,
       rj_arith_v_fract_f16_w32_gfx942},
  };
  for (const auto &family : families) {
    ScopedIsaExecutionBackend backend(family.backend);
    const bool cdna =
        family.arch == rocjitsu::IsaFamily::Cdna1 || family.arch == rocjitsu::IsaFamily::Cdna2 ||
        family.arch == rocjitsu::IsaFamily::Cdna3 || family.arch == rocjitsu::IsaFamily::Cdna4;
    const bool gfx1250 = family.arch == rocjitsu::IsaFamily::Cdna5;
    const bool true16 = cdna || gfx1250 || family.arch == rocjitsu::IsaFamily::Rdna3_5;
    for (int lanes : {32, 64}) {
      if ((cdna && lanes == 32) || (gfx1250 && lanes == 64)) {
        continue;
      }
      for (auto semantics : {ArithmeticSemantics::Compatibility, ArithmeticSemantics::Loose}) {
        ComputeUnitCore::Config config{};
        config.arch = family.arch;
        config.arithmetic = {semantics};
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 104;
        config.vgprs_per_wf = 8;
        config.lds_size_kb = 64;
        GpuMemory memory("half_round_memory");
        L2Cache cache("half_round_cache");
        auto cu = ComputeUnitCore::create("half_round", config, &memory, &cache);
        auto *wave = cu->dispatch_wf(0, 0, 104, 8, lanes);
        ASSERT_NE(wave, nullptr);
        auto decoder = Decoder::create(family.arch);
        const auto base = wave->vgpr_alloc().base;
        for (const auto &operation : family.operations) {
          const int op = operation.model;
          for (const auto &test : scenarios) {
            const bool vop3 = test.form == Form::Vop3, sdwa = test.form == Form::Sdwa;
            const bool dpp8 = test.form == Form::Dpp8;
            const bool dpp = test.form == Form::Dpp || test.form == Form::SuppressedDpp || dpp8;
            if ((dpp8 || (test.high && (test.form == Form::Vop1 || dpp))) && !gfx1250) {
              continue;
            }
            if (sdwa && (gfx1250 || family.arch == rocjitsu::IsaFamily::Rdna3_5)) {
              continue;
            }
            const bool high = test.high && (true16 || sdwa);
            const unsigned omod = test.omod;
            const bool clamp = test.clamp;
            wave->set_mode_raw(test.mode);
            wave->set_exec(dpp ? (lanes == 64 ? ~0ULL : 0xffffffffULL) : 0x80000003800003ffULL);
            wave->set_vgpr_write_mask(dpp ? 0x80000003800003ffULL : ~0ULL);
            std::array<uint32_t, 64> original{};
            for (int lane = 0; lane < lanes; ++lane) {
              original[lane] =
                  uint32_t(values[lane % 10]) | (uint32_t(values[(lane + 3) % 10]) << 16);
              cu->write_vgpr(base, lane, original[lane]);
            }
            std::array<uint32_t, 2> words =
                vop3
                    ? std::array<uint32_t, 2>{(cdna ? 0xd0000000U : 0xd4000000U) |
                                                  (uint32_t(operation.vop3) << 16) |
                                                  (high ? 9U << 11 : 0U) | (clamp ? 1U << 15 : 0U) |
                                                  (test.abs ? 1U << 8 : 0U),
                                              256U | (omod << 27) | (test.neg ? 1U << 29 : 0U)}
                    : std::array<uint32_t, 2>{0x7e000000U | (uint32_t(operation.vop1) << 9) | 256U,
                                              0};
            if (sdwa) {
              // llvm-mc: WORD_1 source/destination, preserve unused half, mul:2.
              words = {0x7e000000U | (uint32_t(operation.vop1) << 9) | 249U, 0x00055500U};
            } else if (dpp) {
              // quad_perm:[1,0,3,2]; the second form suppresses every destination row.
              words = {0x7e000000U | (uint32_t(operation.vop1) << 9) | 250U,
                       test.form == Form::SuppressedDpp ? 0x0f00b100U : 0xff00b100U};
            }
            if (gfx1250 && !vop3) {
              // e32 true16 puts the half selector in each VGPR index's bit7.
              if (high) {
                words[0] |= 0x80U << 17;
                if (!dpp) {
                  words[0] |= 0x80U;
                } else {
                  words[1] |= 0x80U;
                }
              }
              if (dpp8) {
                uint32_t selectors = 0;
                for (int i = 0; i < 8; ++i) {
                  selectors |= uint32_t(i ^ 1) << (3 * i);
                }
                words[0] = (words[0] & ~0x1ffU) | SRC_DPP8_FI_1;
                words[1] = (high ? 0x80U : 0U) | (selectors << 8);
              }
            }
            auto decoded = decoder->decode_window(words);
            ASSERT_TRUE(decoded.succeeded());
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            for (int lane = 0; lane < lanes; ++lane) {
              uint32_t expected = original[lane];
              const bool written = test.form != Form::SuppressedDpp &&
                                   (((wave->exec() & wave->vgpr_write_mask()) >> lane) & 1);
              if (written) {
                uint16_t input = uint16_t(original[dpp ? lane ^ 1 : lane] >> (high ? 16 : 0));
                uint32_t result;
                if (op == 5) {
                  if (test.abs) {
                    input &= 0x7fff;
                  }
                  if (test.neg) {
                    input ^= 0x8000;
                  }
                  if (!(wave->fp_denorm_mode_f16_f64() & 1) && (input & 0x7fff) < 0x400) {
                    input &= 0x8000;
                  }
                  const float x = util::f16_to_f32(input);
                  int exponent = 0;
                  if (x != 0 && std::isfinite(x)) {
                    std::frexp(x, &exponent);
                  }
                  result = uint16_t(exponent);
                } else {
                  const unsigned denorm = wave->fp_denorm_mode_f16_f64();
                  // Compare the same semantics: Loose may ignore directed intermediate
                  // rounding. Native half math also ignores guest ROUND in Compatibility.
                  uint64_t flags =
                      (semantics == ArithmeticSemantics::Loose ? RJ_ARITH_SEMANTICS_LOOSE
                                                               : RJ_ARITH_SEMANTICS_COMPATIBILITY) |
                      (op >= 6 && op <= 12 ? 0 : uint64_t(wave->fp_round_mode_f16_f64()) << 24);
                  flags |= (denorm & 1) ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
                  flags |= (denorm & 2) ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
                  flags |= wave->ieee_mode() ? RJ_ARITH_FP_IEEE_MODE : 0;
                  flags |= !gfx1250 && !wave->dx10_clamp() ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0;
                  flags |= wave->fp16_ovfl() ? RJ_ARITH_FP16_OVFL : 0;
                  const uint64_t modifiers =
                      uint64_t(omod) * RJ_ARITH_ALU_OMOD_2 | (clamp ? RJ_ARITH_ALU_CLAMP : 0) |
                      (test.abs ? RJ_ARITH_ALU_ABS_A : 0) | (test.neg ? RJ_ARITH_ALU_NEG_A : 0) |
                      (sdwa && (family.arch == rocjitsu::IsaFamily::Cdna3 ||
                                family.arch == rocjitsu::IsaFamily::Cdna4)
                           ? RJ_ARITH_ALU_SDWA
                           : 0);
                  const int profile = family.arch == rocjitsu::IsaFamily::Cdna4   ? 3
                                      : gfx1250                                   ? 2
                                      : family.arch == rocjitsu::IsaFamily::Cdna3 ? 1
                                                                                  : 0;
                  result = uint32_t(rocjitsu::test::evaluate_unary32(calls[profile][op], flags,
                                                                     modifiers, input));
                }
                expected = high ? (result << 16) | (original[lane] & 0xffffU)
                                : (cdna ? 0U : original[lane] & 0xffff0000U) | result;
              }
              const uint32_t actual = cu->read_vgpr(base, lane);
              if (semantics == ArithmeticSemantics::Loose && op != 5 && written) {
                const uint32_t keep = high ? 0xffffU : 0xffff0000U;
                EXPECT_EQ(actual & keep, expected & keep);
                const float got = util::f16_to_f32(uint16_t(actual >> (high ? 16 : 0)));
                const float want = util::f16_to_f32(uint16_t(expected >> (high ? 16 : 0)));
                if (std::isnan(want)) {
                  EXPECT_TRUE(std::isnan(got));
                } else if (std::isinf(want)) {
                  EXPECT_EQ(got, want);
                } else {
                  EXPECT_LE(std::abs(double(got) - want), 0x1p-24 + 1.1e-3 * std::abs(want));
                }
              } else {
                EXPECT_EQ(actual, expected) << family.arch << ',' << op << ',' << int(test.form)
                                            << ',' << test.mode << ',' << lane;
              }
            }
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
