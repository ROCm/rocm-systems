// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds_barrier_cell.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/bit.h"
#include "util/result.h"

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
util::FailureOr<std::array<uint32_t, N>> read_sgpr_group(const Wavefront &wf, int reg,
                                                         bool allow_null) {
  std::array<uint32_t, N> words{};
  if (reg == kSgprNull) {
    if (allow_null)
      return words;
    return util::Result::failure();
  }
  if (reg < 0 || reg > 105 || static_cast<size_t>(reg) + N > 106) [[unlikely]]
    return util::Result::failure();
  const uint32_t base = wf.sgpr_alloc().base + static_cast<uint32_t>(reg);
  auto group = amdgpu::RegisterAccess(wf).read_sgpr_region(base, static_cast<uint32_t>(N));
  if (!group.valid()) [[unlikely]]
    return util::Result::failure();
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

  util::Result validate_iteration_inverse() const {
    if (empty_)
      return util::Result::success();

    uint64_t occupied_span = 1;
    for (uint32_t axis_idx = 0; axis_idx < axis_count_; ++axis_idx) {
      const auto &axis = axes_[axis_idx];
      // Each axis must begin beyond the complete span of all faster axes for
      // greedy mixed-radix inversion to be unique.
      if (axis.stride < occupied_span) [[unlikely]]
        return util::Result::failure();

      const uint64_t additional_span =
          saturating_multiply(axis.stride, static_cast<uint64_t>(axis.extent - 1));
      occupied_span = saturating_add(occupied_span, additional_span);
    }
    return util::Result::success();
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

inline util::Result validate_supported_descriptor(const TensorDmaDescriptor &desc,
                                                  const TensorDmaLayout &layout) {
  // The in-tree HIP descriptor API and LLVM lowering only produce the boolean
  // count encodings 0 (disabled) and 1 (active).
  if (desc.count > 1) [[unlikely]]
    return util::Result::failure();
  if (desc.elem_size != 1 && desc.elem_size != 2 && desc.elem_size != 4 && desc.elem_size != 8)
      [[unlikely]]
    return util::Result::failure();
  const uint32_t rank = layout.rank();
  if (!desc.gather && desc.gather_indices_32bit) [[unlikely]]
    return util::Result::failure();
  if (desc.gather) {
    if (rank != 2) [[unlikely]]
      return util::Result::failure();
    if (desc.tile_dims[0] == 0) [[unlikely]]
      return util::Result::failure();
    if (desc.valid_indices == 0) [[unlikely]]
      return util::Result::failure();
    const uint32_t max_indices = desc.gather_indices_32bit ? 8 : 16;
    if (desc.valid_indices > max_indices) [[unlikely]]
      return util::Result::failure();
    return util::Result::success();
  }
  if (desc.iterate && (rank < 2 || rank > 3)) [[unlikely]]
    return util::Result::failure();
  if (desc.iterate && desc.iteration_count > 1 && desc.global_increment != 0) {
    if (layout.validate_iteration_inverse().failed()) [[unlikely]]
      return util::Result::failure();
  }
  uint64_t element_count = 1;
  for (uint32_t dim = 0; dim < rank; ++dim) {
    const auto product = util::checked_mul(element_count, uint64_t{desc.tile_dims[dim]});
    if (!product || *product == 0) [[unlikely]]
      return util::Result::failure();
    element_count = *product;
  }
  return util::Result::success();
}

/// The byte @p lds_element of the transfer's LDS element stream lands on,
/// relative to the wave's LDS allocation rather than absolute in LDS.
inline uint64_t lds_element_offset(const TensorDmaDescriptor &desc, uint64_t lds_element,
                                   bool store_from_lds) {
  uint64_t lds_byte = lds_element * desc.elem_size;
  // The ISA applies descriptor padding only to memory-to-LDS transfers.
  // Stores read the ordinary dense LDS stream and ignore the padding fields.
  if (desc.pad && !store_from_lds) {
    const uint32_t pad_interval_bytes = desc.pad_interval * sizeof(uint32_t);
    const uint32_t pad_amount_bytes = desc.pad_amount * sizeof(uint32_t);
    lds_byte += (lds_byte / pad_interval_bytes) * pad_amount_bytes;
  }
  return static_cast<uint64_t>(desc.lds_base) + lds_byte;
}

/// A stretch of the transfer's LDS element stream that is consecutive before
/// padding skews it: @p element_count elements from @p first_element.
struct TensorDmaLdsRun {
  uint64_t first_element = 0;
  uint64_t element_count = 0;
};

/// Visits the runs of LDS elements the transfer covers, in the order the copy
/// loops below walk them. A dense tile leaves one run per iteration, because
/// the tile coordinates are the mixed-radix digits of the element's position in
/// its tile, and gather packs its rows into a single run. Tensor bounds do not
/// enter into it: a load writes every element of the tile, filling the ones
/// outside the tensor with zeros.
template <typename Fn>
void for_each_lds_run(const TensorDmaDescriptor &desc, const TensorDmaLayout &layout, Fn visit) {
  if (desc.gather) {
    visit(TensorDmaLdsRun{0, static_cast<uint64_t>(desc.valid_indices) *
                                 static_cast<uint64_t>(desc.tile_dims[0])});
    return;
  }

  const uint32_t rank = layout.rank();
  if (rank == 0)
    return;

  uint64_t element_count = 1;
  for (uint32_t dim = 0; dim < rank; ++dim)
    element_count *= desc.tile_dims[dim];

  const uint32_t iteration_count = desc.iterate ? desc.iteration_count : 1;
  for (uint32_t iter = 0; iter < iteration_count; ++iter)
    visit(TensorDmaLdsRun{static_cast<uint64_t>(iter) * desc.lds_increment, element_count});
}

inline void copy_bytes(const TensorDmaDescriptor &desc, Wavefront &wf, uint64_t global_element,
                       uint64_t lds_element, bool in_bounds, bool store_from_lds) {
  if (!wf.has_gpu_memory())
    throw util::UnimplementedInst("tensor DMA without GPU memory");

  const uint64_t global_addr = desc.global_base + global_element * desc.elem_size;
  const uint32_t lds_addr =
      wf.lds_base() + static_cast<uint32_t>(lds_element_offset(desc, lds_element, store_from_lds));
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

inline util::Result copy_tensor(const TensorDmaDescriptor &desc, Wavefront &wf,
                                bool store_from_lds) {
  const TensorDmaLayout layout(desc);
  if (validate_supported_descriptor(desc, layout).failed()) [[unlikely]]
    return util::Result::failure();
  if (layout.rank() != 0 && !wf.has_gpu_memory()) [[unlikely]]
    return util::Result::failure();
  if (desc.gather)
    copy_gather_tensor(desc, layout, wf, store_from_lds);
  else
    copy_dense_tensor(desc, layout, wf, store_from_lds);
  return util::Result::success();
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
util::FailureOr<TensorDmaDescriptor> read_descriptor(const Inst &inst, const Wavefront &wf) {
  auto d0 = read_sgpr_group<4>(wf, inst.vaddr0.encoding_value(), false);
  if (d0.failed()) [[unlikely]]
    return util::Result::failure();
  auto d1 = read_sgpr_group<8>(wf, inst.vaddr1.encoding_value(), false);
  if (d1.failed()) [[unlikely]]
    return util::Result::failure();
  auto d2 = read_sgpr_group<4>(wf, inst.vaddr2.encoding_value(), true);
  if (d2.failed()) [[unlikely]]
    return util::Result::failure();
  auto d3 = read_sgpr_group<4>(wf, inst.vaddr3.encoding_value(), true);
  if (d3.failed()) [[unlikely]]
    return util::Result::failure();
  return parse_descriptor(d0.value(), d1.value(), d2.value(), d3.value());
}

template <typename Inst>
util::Result execute_tensor_dma(const Inst &inst, Wavefront &wf, bool store_from_lds) {
  ScopedWaitCounter counter(wf, WaitCounterType::TENSORCNT);
  const auto desc = read_descriptor(inst, wf);
  if (desc.failed()) [[unlikely]]
    return util::Result::failure();
  if (!desc.value().active())
    return util::Result::success();
  if (copy_tensor(desc.value(), wf, store_from_lds).failed()) [[unlikely]]
    return util::Result::failure();
  if (desc.value().atomic_barrier)
    arrive_atomic_barrier(desc.value(), wf);
  return util::Result::success();
}

} // namespace tensor_dma_detail

template <typename Inst> util::Result execute_tensor_load_to_lds(const Inst &inst, Wavefront &wf) {
  return tensor_dma_detail::execute_tensor_dma(inst, wf, false);
}

template <typename Inst>
util::Result execute_tensor_store_from_lds(const Inst &inst, Wavefront &wf) {
  return tensor_dma_detail::execute_tensor_dma(inst, wf, true);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_TENSOR_DMA_H_
