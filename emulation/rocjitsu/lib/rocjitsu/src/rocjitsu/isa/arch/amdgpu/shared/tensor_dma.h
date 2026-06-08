// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

namespace tensor_dma_detail {

constexpr int kSgprNull = 124;
constexpr uint32_t kGlobalHighBitsMask = (1u << 25) - 1u;
constexpr uint64_t kDsBarrierPendingMask = (1ull << 29) - 1ull;
constexpr uint32_t kDsBarrierPhaseShift = 29;
constexpr uint32_t kDsBarrierInitCountShift = 32;

template <size_t N>
uint64_t read_bits(const std::array<uint32_t, N> &words, uint32_t bit_offset, uint32_t bit_count) {
  uint64_t value = 0;
  for (uint32_t bit = 0; bit < bit_count; ++bit) {
    const uint32_t absolute_bit = bit_offset + bit;
    const uint32_t word = absolute_bit / 32;
    if (word >= N)
      break;
    if (words[word] & (1u << (absolute_bit % 32)))
      value |= 1ull << bit;
  }
  return value;
}

template <size_t N>
std::array<uint32_t, N> read_sgpr_group(const Wavefront &wf, int reg, bool allow_null) {
  std::array<uint32_t, N> words{};
  if (reg == kSgprNull) {
    if (allow_null)
      return words;
    throw util::UnimplementedInst("tensor DMA null descriptor operand");
  }
  if (reg < 0 || reg > 105 || static_cast<size_t>(reg) + N > 106)
    throw util::UnimplementedInst("tensor DMA non-SGPR descriptor operand");
  const uint32_t base = wf.sgpr_alloc().base + static_cast<uint32_t>(reg);
  for (size_t i = 0; i < N; ++i)
    words[i] = wf.cu().read_sgpr(base + static_cast<uint32_t>(i));
  return words;
}

struct TensorDmaDescriptor {
  std::array<uint32_t, 4> d0{};
  std::array<uint32_t, 8> d1{};
  std::array<uint32_t, 4> d2{};
  std::array<uint32_t, 4> d3{};
  uint64_t global_base = 0;
  uint32_t lds_base = 0;
  uint32_t elem_size = 0;
  std::array<uint32_t, 5> tensor_dims{};
  std::array<uint32_t, 5> tile_dims{};
  std::array<uint64_t, 4> global_strides{};
  uint32_t pad_interval = 0;
  uint32_t pad_amount = 0;
  uint32_t lds_increment = 0;
  uint64_t global_increment = 0;
  uint32_t iteration_count = 1;
  uint32_t atomic_barrier_addr = 0;
  uint32_t valid_indices = 0;
  std::array<uint32_t, 16> gather_indices{};
  bool gather = false;
  bool gather_indices_32bit = false;
  bool atomic_barrier = false;
  bool iterate = false;
  bool pad = false;

  uint32_t rank() const {
    if (gather) {
      if (tensor_dims[1] != 0)
        return 2;
      if (tensor_dims[0] != 0 || tile_dims[0] != 0 || valid_indices != 0)
        return 1;
      return 0;
    }
    for (uint32_t i = static_cast<uint32_t>(tile_dims.size()); i > 0; --i) {
      if (tile_dims[i - 1] != 0)
        return i;
    }
    return 0;
  }
};

inline uint64_t default_stride(const TensorDmaDescriptor &desc, uint32_t stride_index) {
  uint64_t stride = 1;
  for (uint32_t dim = 0; dim <= stride_index; ++dim)
    stride *= std::max(desc.tensor_dims[dim], 1u);
  return stride;
}

inline TensorDmaDescriptor parse_descriptor(std::array<uint32_t, 4> d0, std::array<uint32_t, 8> d1,
                                            std::array<uint32_t, 4> d2,
                                            std::array<uint32_t, 4> d3) {
  TensorDmaDescriptor desc;
  desc.d0 = d0;
  desc.d1 = d1;
  desc.d2 = d2;
  desc.d3 = d3;

  // Descriptor bit layout follows LLVM MLIR's gfx1250 TDM lowering in
  // mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp.
  desc.gather = (d0[0] & (1u << 30)) != 0;
  desc.gather_indices_32bit = (d0[0] & (1u << 31)) != 0;
  desc.lds_base = d0[1];
  desc.global_base =
      static_cast<uint64_t>(d0[2]) | (static_cast<uint64_t>(d0[3] & kGlobalHighBitsMask) << 32);

  desc.elem_size = 1u << static_cast<uint32_t>(read_bits(d1, 16, 2));
  desc.atomic_barrier = read_bits(d1, 18, 1) != 0;
  desc.iterate = !desc.gather && read_bits(d1, 19, 1) != 0;
  desc.pad = read_bits(d1, 20, 1) != 0;
  if (desc.atomic_barrier)
    desc.atomic_barrier_addr = static_cast<uint32_t>(read_bits(d1, 32, 16) << 3);
  if (desc.pad) {
    desc.pad_interval = 1u << (static_cast<uint32_t>(read_bits(d1, 22, 3)) + 1);
    desc.pad_amount = static_cast<uint32_t>(read_bits(d1, 25, 7)) + 1;
  }

  desc.tensor_dims[0] = static_cast<uint32_t>(read_bits(d1, 48, 32));
  desc.tensor_dims[1] = static_cast<uint32_t>(read_bits(d1, 80, 32));
  desc.tensor_dims[2] = static_cast<uint32_t>(read_bits(d2, 0, 32));
  if (desc.iterate)
    desc.lds_increment = static_cast<uint32_t>(read_bits(d2, 32, 32));
  else
    desc.tensor_dims[3] = static_cast<uint32_t>(read_bits(d2, 32, 32));
  desc.tensor_dims[4] = static_cast<uint32_t>(read_bits(d3, 48, 32));

  desc.tile_dims[0] = static_cast<uint32_t>(read_bits(d1, 112, 16));
  if (desc.gather)
    desc.valid_indices = static_cast<uint32_t>(read_bits(d1, 128, 16));
  else
    desc.tile_dims[1] = static_cast<uint32_t>(read_bits(d1, 128, 16));
  desc.tile_dims[2] = static_cast<uint32_t>(read_bits(d1, 144, 16));
  if (desc.iterate)
    desc.iteration_count = static_cast<uint32_t>(read_bits(d2, 112, 16)) + 1;
  else
    desc.tile_dims[3] = static_cast<uint32_t>(read_bits(d2, 112, 16));
  desc.tile_dims[4] = static_cast<uint32_t>(read_bits(d3, 80, 16));

  desc.global_strides[0] = read_bits(d1, 160, 48);
  desc.global_strides[1] = read_bits(d1, 208, 48);
  if (desc.iterate)
    desc.global_increment = read_bits(d2, 64, 48);
  else
    desc.global_strides[2] = read_bits(d2, 64, 48);
  desc.global_strides[3] = read_bits(d3, 0, 48);
  for (uint32_t i = 0; i < desc.global_strides.size(); ++i) {
    if (desc.global_strides[i] == 0)
      desc.global_strides[i] = default_stride(desc, i);
  }
  if (desc.gather) {
    if (desc.gather_indices_32bit) {
      for (uint32_t i = 0; i < 4; ++i)
        desc.gather_indices[i] = d2[i];
      for (uint32_t i = 0; i < 4; ++i)
        desc.gather_indices[i + 4] = d3[i];
    } else {
      for (uint32_t i = 0; i < 4; ++i) {
        desc.gather_indices[i * 2] = d2[i] & 0xffffu;
        desc.gather_indices[i * 2 + 1] = (d2[i] >> 16) & 0xffffu;
        desc.gather_indices[i * 2 + 8] = d3[i] & 0xffffu;
        desc.gather_indices[i * 2 + 9] = (d3[i] >> 16) & 0xffffu;
      }
    }
  }
  return desc;
}

inline void validate_supported_descriptor(const TensorDmaDescriptor &desc, bool store_from_lds) {
  // LLVM MLIR's AMDGPU dialect documents padding only for memory-to-LDS copies.
  if (desc.pad && store_from_lds)
    throw util::UnimplementedInst("tensor DMA padded store descriptor");
  if (desc.elem_size != 1 && desc.elem_size != 2 && desc.elem_size != 4 && desc.elem_size != 8)
    throw util::UnimplementedInst("tensor DMA element size");
  const uint32_t rank = desc.rank();
  if (!desc.gather && desc.gather_indices_32bit)
    throw util::UnimplementedInst("tensor DMA gather index-size bit without gather");
  if (desc.gather) {
    if (rank == 0)
      return;
    if (rank > 2)
      throw util::UnimplementedInst("tensor DMA gather rank");
    if (desc.tile_dims[0] == 0)
      throw util::UnimplementedInst("tensor DMA gather tile dimension");
    if (desc.valid_indices == 0)
      throw util::UnimplementedInst("tensor DMA gather valid indices");
    const uint32_t max_indices = desc.gather_indices_32bit ? 8 : 16;
    if (desc.valid_indices > max_indices)
      throw util::UnimplementedInst("tensor DMA gather index count");
    return;
  }
  if (desc.iterate && (rank < 2 || rank > 3))
    throw util::UnimplementedInst("tensor DMA iterate rank");
  for (uint32_t dim = 0; dim < rank; ++dim) {
    if (desc.tile_dims[dim] == 0)
      throw util::UnimplementedInst("tensor DMA sparse tile dimensions");
  }
}

inline void copy_bytes(const TensorDmaDescriptor &desc, Wavefront &wf, uint64_t global_element,
                       uint64_t lds_element, bool in_bounds, bool store_from_lds) {
  auto *memory = wf.cu().memory();
  if (!memory)
    throw util::UnimplementedInst("tensor DMA without GPU memory");

  const uint64_t global_addr = desc.global_base + global_element * desc.elem_size;
  uint64_t lds_byte = lds_element * desc.elem_size;
  if (desc.pad) {
    const uint32_t pad_interval_bytes = desc.pad_interval * sizeof(uint32_t);
    const uint32_t pad_amount_bytes = desc.pad_amount * sizeof(uint32_t);
    lds_byte += (lds_byte / pad_interval_bytes) * pad_amount_bytes;
  }

  const uint32_t lds_addr = wf.lds_base() + desc.lds_base + static_cast<uint32_t>(lds_byte);
  for (uint32_t byte = 0; byte < desc.elem_size; ++byte) {
    if (store_from_lds) {
      if (in_bounds)
        memory->write8(global_addr + byte, wf.cu().lds().read8(lds_addr + byte));
    } else {
      const uint8_t value = in_bounds ? memory->read8(global_addr + byte) : 0;
      wf.cu().lds().write8(lds_addr + byte, value);
    }
  }
}

inline void copy_gather_tensor(const TensorDmaDescriptor &desc, Wavefront &wf,
                               bool store_from_lds) {
  const uint32_t rank = desc.rank();
  if (rank == 0)
    return;

  for (uint32_t idx = 0; idx < desc.valid_indices; ++idx) {
    const uint32_t gather_index = desc.gather_indices[idx];
    for (uint32_t coord0 = 0; coord0 < desc.tile_dims[0]; ++coord0) {
      bool in_bounds = true;
      uint64_t global_element = coord0;
      if (rank == 1) {
        global_element += gather_index;
        if (desc.tensor_dims[0] != 0 && global_element >= desc.tensor_dims[0])
          in_bounds = false;
      } else {
        global_element += static_cast<uint64_t>(gather_index) * desc.global_strides[0];
        if (desc.tensor_dims[0] != 0 && coord0 >= desc.tensor_dims[0])
          in_bounds = false;
        if (desc.tensor_dims[1] != 0 && gather_index >= desc.tensor_dims[1])
          in_bounds = false;
      }

      const uint64_t lds_element =
          static_cast<uint64_t>(idx) * desc.tile_dims[0] + static_cast<uint64_t>(coord0);
      copy_bytes(desc, wf, global_element, lds_element, in_bounds, store_from_lds);
    }
  }
}

inline void copy_dense_tensor(const TensorDmaDescriptor &desc, Wavefront &wf, bool store_from_lds) {
  const uint32_t rank = desc.rank();
  if (rank == 0)
    return;

  uint64_t element_count = 1;
  for (uint32_t dim = 0; dim < rank; ++dim)
    element_count *= desc.tile_dims[dim];

  const uint32_t iteration_count = desc.iterate ? desc.iteration_count : 1;
  for (uint32_t iter = 0; iter < iteration_count; ++iter) {
    for (uint64_t linear = 0; linear < element_count; ++linear) {
      uint64_t remaining = linear;
      uint64_t global_element = static_cast<uint64_t>(iter) * desc.global_increment;
      uint64_t lds_element = static_cast<uint64_t>(iter) * desc.lds_increment;
      uint64_t lds_stride = 1;
      bool in_bounds = true;

      for (uint32_t dim = 0; dim < rank; ++dim) {
        const uint32_t tile_dim = desc.tile_dims[dim];
        const uint32_t coord = static_cast<uint32_t>(remaining % tile_dim);
        remaining /= tile_dim;
        if (desc.tensor_dims[dim] != 0 && coord >= desc.tensor_dims[dim])
          in_bounds = false;
        global_element += coord * (dim == 0 ? 1 : desc.global_strides[dim - 1]);
        lds_element += coord * lds_stride;
        lds_stride *= tile_dim;
      }

      copy_bytes(desc, wf, global_element, lds_element, in_bounds, store_from_lds);
    }
  }
}

inline void copy_tensor(const TensorDmaDescriptor &desc, Wavefront &wf, bool store_from_lds) {
  validate_supported_descriptor(desc, store_from_lds);
  if (desc.gather)
    copy_gather_tensor(desc, wf, store_from_lds);
  else
    copy_dense_tensor(desc, wf, store_from_lds);
}

inline void arrive_atomic_barrier(const TensorDmaDescriptor &desc, Wavefront &wf) {
  // Layout follows LLVM MLIR's ds_barrier_state lowering:
  // [63:32] init count, [31:29] phase, [28:0] pending count.
  const uint32_t addr = wf.lds_base() + desc.atomic_barrier_addr;
  const uint64_t state = wf.cu().lds().read64(addr);
  const uint64_t init_count = state >> kDsBarrierInitCountShift;
  uint64_t phase = (state >> kDsBarrierPhaseShift) & 0x7ull;
  uint64_t pending = state & kDsBarrierPendingMask;

  if (pending == 0) {
    phase = (phase + 7) & 0x7ull;
    pending = init_count;
  } else {
    --pending;
  }

  wf.cu().lds().write64(addr, (init_count << kDsBarrierInitCountShift) |
                                  (phase << kDsBarrierPhaseShift) | pending);
}

template <typename Inst>
TensorDmaDescriptor read_descriptor(const Inst &inst, const Wavefront &wf) {
  return parse_descriptor(read_sgpr_group<4>(wf, inst.vaddr0.encoding_value(), false),
                          read_sgpr_group<8>(wf, inst.vaddr1.encoding_value(), false),
                          read_sgpr_group<4>(wf, inst.vaddr2.encoding_value(), true),
                          read_sgpr_group<4>(wf, inst.vaddr3.encoding_value(), true));
}

} // namespace tensor_dma_detail

template <typename Inst> void execute_tensor_load_to_lds(const Inst &inst, Wavefront &wf) {
  const auto desc = tensor_dma_detail::read_descriptor(inst, wf);
  tensor_dma_detail::copy_tensor(desc, wf, false);
  if (desc.atomic_barrier)
    tensor_dma_detail::arrive_atomic_barrier(desc, wf);
}

template <typename Inst> void execute_tensor_store_from_lds(const Inst &inst, Wavefront &wf) {
  const auto desc = tensor_dma_detail::read_descriptor(inst, wf);
  tensor_dma_detail::copy_tensor(desc, wf, true);
  if (desc.atomic_barrier)
    tensor_dma_detail::arrive_atomic_barrier(desc, wf);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_
