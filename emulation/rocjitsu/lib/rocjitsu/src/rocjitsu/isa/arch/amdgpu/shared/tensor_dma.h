// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds_barrier_cell.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rocjitsu {
namespace amdgpu {

namespace tensor_dma_detail {

// The descriptor layout is specified by the AMD CDNA5 ISA, sections 10.11.1
// through 10.11.6, and is also present in the ROCm HIP header
// hip/amd_detail/amd_gfx1250_TDM.h. Descriptors produced from MLIR follow
// LLVM's AMDGPUToROCDL make_dma_descriptor lowering.
//
// rocJITsu consumes encoded descriptor values literally: null optional groups
// read as zero, zero strides alias coordinates, gather is rank two, and a zero
// active tensor extent masks the whole transfer. Masked loads zero-fill LDS;
// masked stores suppress global writes while completion effects still retire.
constexpr int kSgprNull = 124;
constexpr uint32_t kGlobalHighBitsMask = (1u << 25) - 1u;

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
  auto group = amdgpu::RegisterAccess(wf).read_sgpr_region(base, static_cast<uint32_t>(N));
  if (!group.valid())
    throw util::UnimplementedInst("tensor DMA descriptor outside wavefront SGPR block");
  for (size_t i = 0; i < N; ++i)
    words[i] = group.dword(static_cast<uint32_t>(i));
  return words;
}

struct TensorDmaDescriptor {
  std::array<uint32_t, 4> d0{};
  std::array<uint32_t, 8> d1{};
  std::array<uint32_t, 4> d2{};
  std::array<uint32_t, 4> d3{};
  uint32_t count = 0;
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
  uint32_t tensor_rank = 0;
  std::array<uint32_t, 16> gather_indices{};
  bool gather = false;
  bool gather_indices_32bit = false;
  bool atomic_barrier = false;
  bool iterate = false;
  bool pad = false;

  bool active() const { return count != 0; }

  uint32_t rank() const { return tensor_rank; }
};

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
  desc.count = d0[0] & 0x3u;
  desc.gather_indices_32bit = (d0[0] & (1u << 30)) != 0;
  desc.gather = (d0[0] & (1u << 31)) != 0;
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

  if (desc.gather) {
    // CDNA5 gather mode is always a 2D row gather/scatter. tensor_dim1 is the
    // row bound even when its value is zero.
    desc.tensor_rank = 2;
  } else {
    for (uint32_t dim = static_cast<uint32_t>(desc.tile_dims.size()); dim > 0; --dim) {
      if (desc.tile_dims[dim - 1] != 0) {
        desc.tensor_rank = dim;
        break;
      }
    }
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

inline uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs ? std::numeric_limits<uint64_t>::max()
                                                          : lhs + rhs;
}

inline uint64_t saturating_multiply(uint64_t lhs, uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs
             ? std::numeric_limits<uint64_t>::max()
             : lhs * rhs;
}

struct TensorDmaAxis {
  uint32_t dimension = 0;
  uint32_t extent = 0;
  uint64_t stride = 0;
};

// Iteration advances the global address linearly, but bounds checks need a
// logical tensor origin. This layout orders non-unit axes by storage stride and
// supports a greedy inverse only when every axis starts beyond the occupied
// span of all faster axes. Empty, single-iteration, and zero-increment
// descriptors need no inverse. Rejecting other advancing layouts is a
// rocJITsu support boundary rather than an ISA restriction.
class TensorDmaLayout {
public:
  explicit TensorDmaLayout(const TensorDmaDescriptor &desc) : rank_(desc.rank()) {
    for (uint32_t dim = 0; dim < rank_; ++dim) {
      const uint32_t extent = desc.tensor_dims[dim];
      empty_ |= extent == 0;
      if (extent > 1) {
        axes_[axis_count_++] = {.dimension = dim,
                                .extent = extent,
                                .stride = dim == 0 ? 1 : desc.global_strides[dim - 1]};
      }
    }

    // Logical dimension order and increasing memory-stride order may differ.
    // Validation and inversion must use the same storage-ordered basis.
    // Extent-one axes have only coordinate zero and consume no address span.
    // Use a bounded insertion sort: GCC 13 can report a false-positive
    // -Warray-bounds error when std::sort is optimized for this five-element array.
    for (uint32_t axis_idx = 1; axis_idx < axis_count_; ++axis_idx) {
      const TensorDmaAxis axis = axes_[axis_idx];
      uint32_t insertion_idx = axis_idx;
      while (insertion_idx > 0) {
        const auto &previous = axes_[insertion_idx - 1];
        const bool axis_precedes_previous =
            axis.stride < previous.stride ||
            (axis.stride == previous.stride && axis.dimension < previous.dimension);
        if (!axis_precedes_previous)
          break;
        axes_[insertion_idx] = previous;
        --insertion_idx;
      }
      axes_[insertion_idx] = axis;
    }
  }

  uint32_t rank() const { return rank_; }
  bool empty() const { return empty_; }

  void validate_iteration_inverse() const {
    if (empty_)
      return;

    uint64_t occupied_span = 1;
    for (uint32_t axis_idx = 0; axis_idx < axis_count_; ++axis_idx) {
      const auto &axis = axes_[axis_idx];
      // Each axis must begin beyond the complete span of all faster axes for
      // greedy mixed-radix inversion to be unique.
      if (axis.stride < occupied_span)
        throw util::UnimplementedInst("tensor DMA iterate non-invertible strides");

      const uint64_t additional_span =
          saturating_multiply(axis.stride, static_cast<uint64_t>(axis.extent - 1));
      occupied_span = saturating_add(occupied_span, additional_span);
    }
  }

  std::array<uint64_t, 5> origin_from_linear_offset(uint64_t linear_offset) const {
    std::array<uint64_t, 5> origin{};
    if (linear_offset == 0)
      return origin;

    for (uint32_t axis_idx = axis_count_; axis_idx > 0; --axis_idx) {
      const auto &axis = axes_[axis_idx - 1];
      origin[axis.dimension] = linear_offset / axis.stride;
      linear_offset %= axis.stride;
    }
    // Dimension zero has implicit stride one. If it was omitted because its
    // extent is one, preserve any padding remainder there so bounds masking
    // rejects offsets outside the tensor domain.
    if (linear_offset != 0)
      origin[0] = linear_offset;
    return origin;
  }

private:
  std::array<TensorDmaAxis, 5> axes_{};
  uint32_t rank_ = 0;
  uint32_t axis_count_ = 0;
  bool empty_ = false;
};

inline void validate_supported_descriptor(const TensorDmaDescriptor &desc,
                                          const TensorDmaLayout &layout) {
  // The in-tree HIP descriptor API and LLVM lowering only produce the boolean
  // count encodings 0 (disabled) and 1 (active).
  if (desc.count > 1)
    throw util::UnimplementedInst("tensor DMA count encoding");
  if (desc.elem_size != 1 && desc.elem_size != 2 && desc.elem_size != 4 && desc.elem_size != 8)
    throw util::UnimplementedInst("tensor DMA element size");
  const uint32_t rank = layout.rank();
  if (!desc.gather && desc.gather_indices_32bit)
    throw util::UnimplementedInst("tensor DMA gather index-size bit without gather");
  if (desc.gather) {
    if (rank != 2)
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
  if (desc.iterate && desc.iteration_count > 1 && desc.global_increment != 0)
    layout.validate_iteration_inverse();
  for (uint32_t dim = 0; dim < rank; ++dim) {
    if (desc.tile_dims[dim] == 0)
      throw util::UnimplementedInst("tensor DMA sparse tile dimensions");
  }
}

inline void copy_bytes(const TensorDmaDescriptor &desc, Wavefront &wf, uint64_t global_element,
                       uint64_t lds_element, bool in_bounds, bool store_from_lds) {
  if (!wf.has_gpu_memory())
    throw util::UnimplementedInst("tensor DMA without GPU memory");

  const uint64_t global_addr = desc.global_base + global_element * desc.elem_size;
  uint64_t lds_byte = lds_element * desc.elem_size;
  // The ISA applies descriptor padding only to memory-to-LDS transfers.
  // Stores read the ordinary dense LDS stream and ignore the padding fields.
  if (desc.pad && !store_from_lds) {
    const uint32_t pad_interval_bytes = desc.pad_interval * sizeof(uint32_t);
    const uint32_t pad_amount_bytes = desc.pad_amount * sizeof(uint32_t);
    lds_byte += (lds_byte / pad_interval_bytes) * pad_amount_bytes;
  }

  const uint32_t lds_addr = wf.lds_base() + desc.lds_base + static_cast<uint32_t>(lds_byte);
  std::array<uint8_t, 8> bytes{};
  auto element_bytes = std::span<uint8_t>(bytes).first(desc.elem_size);
  if (store_from_lds) {
    if (!in_bounds)
      return;
    for (uint32_t byte = 0; byte < desc.elem_size; ++byte)
      element_bytes[byte] = wf.lds().read8(lds_addr + byte);
    wf.write_gpu_memory(global_addr, element_bytes);
    return;
  }

  if (in_bounds)
    wf.read_gpu_memory(global_addr, element_bytes);
  for (uint32_t byte = 0; byte < desc.elem_size; ++byte)
    wf.lds().write8(lds_addr + byte, element_bytes[byte]);
}

inline void copy_gather_tensor(const TensorDmaDescriptor &desc, const TensorDmaLayout &layout,
                               Wavefront &wf, bool store_from_lds) {
  for (uint32_t idx = 0; idx < desc.valid_indices; ++idx) {
    const uint32_t gather_index = desc.gather_indices[idx];
    for (uint32_t coord0 = 0; coord0 < desc.tile_dims[0]; ++coord0) {
      bool in_bounds = !layout.empty();
      const uint64_t global_element =
          coord0 + static_cast<uint64_t>(gather_index) * desc.global_strides[0];
      if (coord0 >= desc.tensor_dims[0])
        in_bounds = false;
      if (gather_index >= desc.tensor_dims[1])
        in_bounds = false;

      const uint64_t lds_element =
          static_cast<uint64_t>(idx) * desc.tile_dims[0] + static_cast<uint64_t>(coord0);
      copy_bytes(desc, wf, global_element, lds_element, in_bounds, store_from_lds);
    }
  }
}

inline void copy_dense_tensor(const TensorDmaDescriptor &desc, const TensorDmaLayout &layout,
                              Wavefront &wf, bool store_from_lds) {
  const uint32_t rank = layout.rank();
  if (rank == 0)
    return;

  uint64_t element_count = 1;
  for (uint32_t dim = 0; dim < rank; ++dim)
    element_count *= desc.tile_dims[dim];

  const uint32_t iteration_count = desc.iterate ? desc.iteration_count : 1;
  for (uint32_t iter = 0; iter < iteration_count; ++iter) {
    const uint64_t iteration_offset = static_cast<uint64_t>(iter) * desc.global_increment;
    const auto iteration_origin = desc.iterate && !layout.empty()
                                      ? layout.origin_from_linear_offset(iteration_offset)
                                      : std::array<uint64_t, 5>{};
    for (uint64_t linear = 0; linear < element_count; ++linear) {
      uint64_t remaining = linear;
      uint64_t global_element = iteration_offset;
      uint64_t lds_element = static_cast<uint64_t>(iter) * desc.lds_increment;
      uint64_t lds_stride = 1;
      bool in_bounds = !layout.empty();

      for (uint32_t dim = 0; dim < rank; ++dim) {
        const uint32_t tile_dim = desc.tile_dims[dim];
        const uint32_t coord = static_cast<uint32_t>(remaining % tile_dim);
        remaining /= tile_dim;
        const uint64_t tensor_coord = iteration_origin[dim] + coord;
        if (tensor_coord >= desc.tensor_dims[dim])
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
  const TensorDmaLayout layout(desc);
  validate_supported_descriptor(desc, layout);
  if (desc.gather)
    copy_gather_tensor(desc, layout, wf, store_from_lds);
  else
    copy_dense_tensor(desc, layout, wf, store_from_lds);
}

inline void arrive_atomic_barrier(const TensorDmaDescriptor &desc, Wavefront &wf) {
  const uint32_t addr = wf.lds_base() + desc.atomic_barrier_addr;
  const uint64_t state = wf.lds().read64(addr);
  wf.lds().write64(addr, lds_barrier_cell_update_arrive(state));
}

class ScopedWaitCounter {
public:
  ScopedWaitCounter(Wavefront &wf, WaitCounterType type) : wf_(wf), type_(type) {
    wf_.wait_counters().increment(type_);
  }

  ScopedWaitCounter(const ScopedWaitCounter &) = delete;
  ScopedWaitCounter &operator=(const ScopedWaitCounter &) = delete;
  ScopedWaitCounter(ScopedWaitCounter &&) = delete;
  ScopedWaitCounter &operator=(ScopedWaitCounter &&) = delete;

  ~ScopedWaitCounter() { wf_.release_wait_counter(type_); }

private:
  Wavefront &wf_;
  WaitCounterType type_;
};

template <typename Inst>
TensorDmaDescriptor read_descriptor(const Inst &inst, const Wavefront &wf) {
  return parse_descriptor(read_sgpr_group<4>(wf, inst.vaddr0.encoding_value(), false),
                          read_sgpr_group<8>(wf, inst.vaddr1.encoding_value(), false),
                          read_sgpr_group<4>(wf, inst.vaddr2.encoding_value(), true),
                          read_sgpr_group<4>(wf, inst.vaddr3.encoding_value(), true));
}

template <typename Inst>
void execute_tensor_dma(const Inst &inst, Wavefront &wf, bool store_from_lds) {
  ScopedWaitCounter counter(wf, WaitCounterType::TENSORCNT);
  const auto desc = read_descriptor(inst, wf);
  if (!desc.active())
    return;
  copy_tensor(desc, wf, store_from_lds);
  if (desc.atomic_barrier)
    arrive_atomic_barrier(desc, wf);
}

} // namespace tensor_dma_detail

template <typename Inst> void execute_tensor_load_to_lds(const Inst &inst, Wavefront &wf) {
  tensor_dma_detail::execute_tensor_dma(inst, wf, false);
}

template <typename Inst> void execute_tensor_store_from_lds(const Inst &inst, Wavefront &wf) {
  tensor_dma_detail::execute_tensor_dma(inst, wf, true);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_
