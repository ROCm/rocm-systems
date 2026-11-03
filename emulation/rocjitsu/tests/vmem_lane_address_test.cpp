// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vector-memory lane addresses: the vector forms against the per-lane loops they replace.
//
// Part 1 checks each AVX2 body against its scalar body slot for slot. Part 2
// runs the real address calculations on randomized registers, descriptors and
// EXEC masks and compares every output with a copy of the per-lane loop each
// one used before the vector forms.

#include "rocjitsu/isa/arch/amdgpu/cdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_lanes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/flat_address.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rocjitsu/isa/isa_family.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace {

using namespace rocjitsu;
namespace ac = amdgpu::addr_calc;

constexpr uint64_t wave_mask(uint32_t wf_size) {
  return wf_size >= 64 ? ~uint64_t{0} : (uint64_t{1} << wf_size) - 1;
}

/// EXEC patterns that matter to lane loops: full, empty, single, sparse, halves, random.
uint64_t random_exec(std::mt19937_64 &rng, uint32_t wf_size) {
  const uint64_t full = wave_mask(wf_size);
  switch (rng() % 7) {
  case 0:
  case 1:
    return full;
  case 2:
    return 0;
  case 3:
    return uint64_t{1} << (rng() % wf_size);
  case 4:
    return rng() & rng() & full;
  case 5:
    return (rng() % 2 ? 0xFFFF'FFFFull : 0xFFFF'FFFF'0000'0000ull) & full;
  default:
    return rng() & full;
  }
}

/// 32-bit lane values biased toward carries and sign changes.
uint32_t random_u32(std::mt19937_64 &rng) {
  switch (rng() % 5) {
  case 0:
    return 0xFFFF'FFFFu - static_cast<uint32_t>(rng() % 64);
  case 1:
    return static_cast<uint32_t>(rng() % 4096);
  case 2:
    return 0x7FFF'FFFFu + static_cast<uint32_t>(rng() % 3) - 1;
  default:
    return static_cast<uint32_t>(rng());
  }
}

uint64_t random_u64(std::mt19937_64 &rng) {
  switch (rng() % 4) {
  case 0:
    return ~uint64_t{0} - rng() % 4096; // wraps past 2^64
  case 1:
    return 0x7FFF'0000'0000ull + rng() % (uint64_t{1} << 33);
  default:
    return rng();
  }
}

// ---------------------------------------------------------------------------
// Part 1: AVX2 bodies against scalar bodies.
// ---------------------------------------------------------------------------

#if RJ_HOST_HAS_AVX2

template <uint32_t WF> void check_global_forms(std::mt19937_64 &rng) {
  for (int round = 0; round < 4000; ++round) {
    std::array<uint32_t, 64> lo{}, hi{};
    for (uint32_t lane = 0; lane < 64; ++lane) {
      lo[lane] = random_u32(rng);
      hi[lane] = random_u32(rng);
    }
    // Bits past the wave must not be read as lanes.
    const uint64_t exec = random_exec(rng, WF) | (WF < 64 ? rng() << (WF % 64) : 0);
    const uint64_t bias = random_u64(rng);
    const uint32_t scale = std::array<uint32_t, 6>{1, 2, 4, 8, 12, 16}[rng() % 6];
    std::array<uint64_t, 64> want{}, got{};

    want.fill(0xA5A5);
    got.fill(0xA5A5);
    ac::detail::lane_addresses_u32_scalar(lo.data(), exec, WF, bias, want.data());
    ac::detail::lane_addresses_u32_avx2<WF>(lo.data(), exec, bias, got.data());
    ASSERT_EQ(want, got) << "u32 exec=" << exec;

    ac::detail::lane_addresses_s32_scalar(lo.data(), exec, WF, bias, scale, want.data());
    ac::detail::lane_addresses_s32_avx2<WF>(lo.data(), exec, bias, scale, got.data());
    ASSERT_EQ(want, got) << "s32 exec=" << exec << " scale=" << scale;

    ac::detail::lane_addresses_u64_scalar(lo.data(), hi.data(), exec, WF, bias, want.data());
    ac::detail::lane_addresses_u64_avx2<WF>(lo.data(), hi.data(), exec, bias, got.data());
    ASSERT_EQ(want, got) << "u64 exec=" << exec;

    // A window around one lane's address catches it and its neighbours in the
    // same 4 GiB; the lane bits model gfx1250's encoded private addresses.
    const uint64_t keep = rng() % 2 ? ~uint64_t{0} : ~(uint64_t{WF - 1} << (WF == 64 ? 51 : 52));
    const uint64_t base = (want[rng() % WF] & keep) - rng() % (uint64_t{1} << 33);
    EXPECT_EQ(ac::detail::lanes_in_window_scalar(want.data(), exec & wave_mask(WF), WF, keep, base),
              ac::detail::lanes_in_window_avx2<WF>(want.data(), exec & wave_mask(WF), keep, base));
  }
}

TEST(VmemLaneAddressTest, GlobalFormsMatchScalarBodies) {
  std::mt19937_64 rng(1);
  check_global_forms<32>(rng);
  check_global_forms<64>(rng);
}

#endif // RJ_HOST_HAS_AVX2

// ---------------------------------------------------------------------------
// Part 2: the address calculations against their former per-lane loops.
// ---------------------------------------------------------------------------

struct Machine {
  amdgpu::GpuMemory mem{"vmem_lane_mem"};
  amdgpu::L2Cache l2{"vmem_lane_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;

  Machine(rocjitsu::IsaFamily arch, uint32_t wave_size) {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 104;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("vmem_lane_cu", cfg, &mem, &l2);
    wf = cu->dispatch_wf(0, 0, 104, 16, wave_size);
  }
  ~Machine() {
    if (wf)
      wf->halt();
  }

  uint32_t vgpr(uint32_t reg, uint32_t lane) const {
    return static_cast<uint32_t>(
        reinterpret_cast<const uint32_t *>(cu->raw_vgpr_data(wf->vgpr_alloc().base + reg))[lane]);
  }
  void set_vgpr(uint32_t reg, uint32_t lane, uint32_t value) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, value);
  }
  void set_sgpr(uint32_t reg, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + reg, value);
  }
  void set_sgpr64(uint32_t reg, uint64_t value) {
    set_sgpr(reg, static_cast<uint32_t>(value));
    set_sgpr(reg + 1, static_cast<uint32_t>(value >> 32));
  }
  void randomize_vgprs(std::mt19937_64 &rng) {
    for (uint32_t reg = 0; reg < 4; ++reg)
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
        set_vgpr(reg, lane, random_u32(rng));
  }
  /// Wave32 EXEC keeps bits past the wave; the address loops must ignore them.
  void set_exec(std::mt19937_64 &rng) {
    const uint32_t n = wf->wf_size();
    wf->set_exec_raw(random_exec(rng, n) | (n < 64 && rng() % 4 == 0 ? rng() << n : 0));
  }
};

void expect_same(const amdgpu::VectorMemState &want, const amdgpu::VectorMemState &got,
                 const char *what, int round) {
  SCOPED_TRACE(testing::Message() << what << " round " << round);
  ASSERT_EQ(want.exec_mask, got.exec_mask);
  ASSERT_EQ(want.lane_mask, got.lane_mask);
  ASSERT_EQ(want.per_lane_addr, got.per_lane_addr);
  ASSERT_EQ(want.element_lane_masks.size(), got.element_lane_masks.size());
  for (size_t elem = 0; elem < want.element_lane_masks.size(); ++elem)
    ASSERT_EQ(want.element_lane_masks[elem], got.element_lane_masks[elem]) << "element " << elem;
  ASSERT_EQ(want.scratch_swizzle, got.scratch_swizzle);
  ASSERT_EQ(want.requires_scratch_backing, got.requires_scratch_backing);
  ASSERT_EQ(want.scratch_addr_stride, got.scratch_addr_stride);
  ASSERT_EQ(want.scratch_lane_mask, got.scratch_lane_mask);
}

/// A lane whose upper word lands on the private aperture, the window base, or random.
uint32_t random_high_word(std::mt19937_64 &rng, uint32_t private_hi) {
  switch (rng() % 4) {
  case 0:
    return private_hi;
  case 1:
    return private_hi - 1; // carries into the aperture with a large low word
  default:
    return random_u32(rng);
  }
}

// CDNA4 FLAT (seg 0) and GLOBAL (seg 2) -------------------------------------

TEST(VmemLaneAddressTest, Cdna4FlatAndGlobalMatchPerLaneLoop) {
  std::mt19937_64 rng(3);
  Machine m(rocjitsu::IsaFamily::Cdna4, 64);
  ASSERT_NE(m.wf, nullptr);
  auto &wf = *m.wf;
  int private_rounds = 0;
  for (int round = 0; round < 3000; ++round) {
    m.randomize_vgprs(rng);
    m.set_exec(rng);
    const uint32_t private_hi = rng() % 4 ? 0x1000u + static_cast<uint32_t>(rng() % 4) : 0;
    wf.set_apertures(0, 0, uint64_t{private_hi} << 32, (uint64_t{private_hi} << 32) | 0xFFFF'FFFF,
                     false);
    wf.set_scratch_base(random_u64(rng) & ~uint64_t{3});
    for (uint32_t lane = 0; lane < 64; ++lane)
      if (rng() % 2)
        m.set_vgpr(1, lane, random_high_word(rng, private_hi));
    const uint64_t saddr = random_u64(rng);
    m.set_sgpr64(20, saddr);

    cdna4::FlatMachineInst inst{};
    inst.seg = rng() % 2 ? 2 : 0;
    inst.addr = 0;
    inst.offset = static_cast<uint32_t>(rng() % 4096);
    inst.pad_12 = rng() % 2;
    inst.saddr = inst.seg == 2 && rng() % 2 ? 20 : 0x7F;

    amdgpu::VectorMemState got(amdgpu::GLOBAL_MEM);
    cdna4::flat_calculate_addresses(inst, wf, got);

    // The per-lane loop: every FLAT lane goes through the private capture.
    amdgpu::VectorMemState want(amdgpu::GLOBAL_MEM);
    const uint64_t exec = wf.exec();
    want.lane_mask = want.exec_mask = exec;
    const int64_t offset = inst.seg == 2
                               ? static_cast<int32_t>((inst.offset | inst.pad_12 << 12) << 19) >> 19
                               : inst.offset & 0xFFF;
    for (uint32_t lane = 0; lane < 64; ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      const uint64_t pair = uint64_t{m.vgpr(1, lane)} << 32 | m.vgpr(0, lane);
      if (inst.seg == 2) {
        want.per_lane_addr[lane] =
            (inst.saddr != 0x7F ? saddr + m.vgpr(0, lane) : pair) + static_cast<uint64_t>(offset);
        continue;
      }
      want.per_lane_addr[lane] =
          amdgpu::addr_calc::capture_flat_private_address(wf, want, pair + offset, lane);
    }
    expect_same(want, got, inst.seg == 2 ? "global" : "flat", round);
    private_rounds += want.scratch_lane_mask != 0;
  }
  EXPECT_GT(private_rounds, 100);
}

// RDNA4 VGLOBAL and VFLAT ---------------------------------------------------

TEST(VmemLaneAddressTest, Rdna4GlobalAndFlatMatchPerLaneLoop) {
  std::mt19937_64 rng(4);
  int private_lanes = 0;
  for (uint32_t wave : {32u, 64u}) {
    Machine m(rocjitsu::IsaFamily::Rdna4, wave);
    ASSERT_NE(m.wf, nullptr);
    auto &wf = *m.wf;
    ASSERT_EQ(wf.wf_size(), wave);
    for (int round = 0; round < 3000; ++round) {
      m.randomize_vgprs(rng);
      m.set_exec(rng);
      const uint32_t private_hi = rng() % 4 ? 0x2000u + static_cast<uint32_t>(rng() % 4) : 0;
      wf.set_apertures(0, 0, uint64_t{private_hi} << 32, (uint64_t{private_hi} << 32) | 0xFFFF'FFFF,
                       false);
      wf.set_scratch_base(random_u64(rng));
      wf.set_scratch_lane_size(static_cast<uint32_t>(rng() % 0x10000));
      for (uint32_t lane = 0; lane < wave; ++lane)
        if (rng() % 2)
          m.set_vgpr(1, lane, random_high_word(rng, private_hi));
      const uint64_t saddr = random_u64(rng);
      m.set_sgpr64(20, saddr);
      const bool flat = rng() % 2;
      const uint32_t saddr_sel = rng() % 2 ? 20 : (rng() % 2 ? 0x7C : 0x7F);
      const uint32_t ioffset = static_cast<uint32_t>(rng()) & 0xFFFFFF;

      amdgpu::VectorMemState got(amdgpu::GLOBAL_MEM);
      if (flat) {
        rdna4::VflatMachineInst inst{};
        inst.saddr = saddr_sel;
        inst.vaddr = 0;
        inst.ioffset = ioffset;
        rdna4::flat_calculate_addresses(inst, wf, got);
      } else {
        rdna4::VglobalMachineInst inst{};
        inst.saddr = saddr_sel;
        inst.vaddr = 0;
        inst.ioffset = ioffset;
        rdna4::flat_calculate_addresses(inst, wf, got);
      }

      amdgpu::VectorMemState want(amdgpu::GLOBAL_MEM);
      const uint64_t exec = wf.exec();
      want.lane_mask = want.exec_mask = exec;
      const int32_t offset = static_cast<int32_t>(ioffset << 8) >> 8;
      const bool has_saddr = saddr_sel != 0x7C && saddr_sel != 0x7F;
      for (uint32_t lane = 0; lane < wave; ++lane) {
        if (!(exec & (1ULL << lane)))
          continue;
        const uint64_t vaddr =
            has_saddr ? m.vgpr(0, lane) : uint64_t{m.vgpr(1, lane)} << 32 | m.vgpr(0, lane);
        const uint64_t pointer = (has_saddr ? saddr : 0) + vaddr;
        if (!flat) {
          want.per_lane_addr[lane] = pointer + static_cast<uint64_t>(int64_t{offset});
          continue;
        }
        // GFX12 selects the segment from the pointer, before the immediate.
        private_lanes += amdgpu::translate_flat_address(
                             wf, pointer, lane, amdgpu::FlatPrivateLayout::Interleaved, offset)
                             .private_address;
        want.per_lane_addr[lane] =
            amdgpu::addr_calc::capture_flat_private_address(wf, want, pointer, lane, offset);
      }
      expect_same(want, got, flat ? "vflat" : "vglobal", round);
    }
  }
  EXPECT_GT(private_lanes, 1000);
}

// CDNA5 VGLOBAL and VFLAT ---------------------------------------------------

TEST(VmemLaneAddressTest, Cdna5GlobalAndFlatMatchPerLaneLoop) {
  std::mt19937_64 rng(5);
  Machine m(rocjitsu::IsaFamily::Cdna5, 32);
  ASSERT_NE(m.wf, nullptr);
  auto &wf = *m.wf;
  const uint32_t wave = wf.wf_size();
  const unsigned shift = wave == 64 ? 51 : 52;
  const uint64_t lane_bits = uint64_t{wave - 1} << shift;
  int private_rounds = 0;
  for (int round = 0; round < 3000; ++round) {
    m.randomize_vgprs(rng);
    m.set_exec(rng);
    const uint64_t scratch_base = (0x7000'0000'0000ull + (rng() % (1ull << 40))) & ~lane_bits;
    wf.set_scratch_base(scratch_base);
    wf.set_scratch_lane_size(rng() % 4 ? 0x400 : 0);
    const bool flat = rng() % 2;
    // FLAT lanes encoded as private addresses: base + offset with a lane number
    // in the lane bits. GLOBAL must not use that encoding.
    for (uint32_t lane = 0; flat && lane < wave; ++lane) {
      if (rng() % 2)
        continue;
      const uint64_t encoded = (scratch_base + rng() % (uint64_t{1} << 33) - (1ull << 31)) |
                               (uint64_t{rng() % wave} << shift);
      m.set_vgpr(0, lane, static_cast<uint32_t>(encoded));
      m.set_vgpr(1, lane, static_cast<uint32_t>(encoded >> 32));
    }
    const uint64_t saddr = random_u64(rng);
    m.set_sgpr64(20, saddr);
    const bool has_saddr = rng() % 2;
    const bool scale_offset = has_saddr && rng() % 2;
    const uint32_t ioffset = static_cast<uint32_t>(rng()) & 0xFFFFFF;
    const uint32_t num_elems = 1 + static_cast<uint32_t>(rng() % 4);

    auto fresh = [&] {
      amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
      d.elem_size = 4;
      d.num_elems = num_elems;
      return d;
    };
    amdgpu::VectorMemState got = fresh();
    if (flat) {
      cdna5::VflatMachineInst inst{};
      inst.saddr = has_saddr ? 20 : cdna5::OPR_SREG_NULL;
      inst.scale_offset = scale_offset;
      inst.ioffset = ioffset;
      cdna5::flat_calculate_addresses(inst, wf, got);
    } else {
      cdna5::VglobalMachineInst inst{};
      inst.saddr = has_saddr ? 20 : cdna5::OPR_SREG_NULL;
      inst.scale_offset = scale_offset;
      inst.ioffset = ioffset;
      cdna5::flat_calculate_addresses(inst, wf, got);
    }

    amdgpu::VectorMemState want = fresh();
    const uint64_t exec = wf.exec();
    want.lane_mask = want.exec_mask = exec;
    const int64_t offset = cdna5::signed_ioffset(ioffset);
    const uint32_t scale = scale_offset ? 4 * num_elems : 1;
    for (uint32_t lane = 0; lane < wave; ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      const uint64_t vaddr = has_saddr ? amdgpu::flat_vector_offset(m.vgpr(0, lane), true, scale)
                                       : uint64_t{m.vgpr(1, lane)} << 32 | m.vgpr(0, lane);
      uint64_t addr = (has_saddr ? saddr : 0) + vaddr + offset;
      if (flat) {
        const auto translated =
            amdgpu::translate_flat_address(wf, addr, lane, amdgpu::FlatPrivateLayout::EncodedLane);
        if (translated.private_address) {
          addr = translated.value;
          want.scratch_swizzle = true;
          want.requires_scratch_backing = true;
          want.scratch_addr_stride = wave * sizeof(uint32_t);
          want.scratch_lane_mask |= uint64_t{1} << lane;
        }
      }
      want.per_lane_addr[lane] = addr;
    }
    expect_same(want, got, flat ? "vflat" : "vglobal", round);
    private_rounds += want.scratch_lane_mask != 0;
  }
  EXPECT_GT(private_rounds, 100);
}

} // namespace
