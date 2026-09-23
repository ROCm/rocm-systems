// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds_barrier_cell.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/tensor_dma_memory_access_observation.h"
#include "util/bit.h"
#include "util/result.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

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

template <size_t WordCount>
uint64_t read_bits(const std::array<uint32_t, WordCount> &words, uint32_t bit_offset,
                   uint32_t bit_count) {
  uint64_t value = 0;
  for (uint32_t bit = 0; bit < bit_count; ++bit) {
    const uint32_t absolute_bit = bit_offset + bit;
    const uint32_t word = absolute_bit / 32;
    if (word >= WordCount)
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

class TensorDmaDescriptor {
public:
  std::array<uint32_t, 4> d0{};
  std::array<uint32_t, 8> d1{};
  std::array<uint32_t, 4> d2{};
  std::array<uint32_t, 4> d3{};
  uint32_t count = 0;
  uint64_t global_base = 0;
  uint32_t lds_base = 0;
  uint32_t data_size = 0;
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

  desc.data_size = static_cast<uint32_t>(read_bits(d1, 16, 2));
  desc.elem_size = 1u << desc.data_size;
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
      for (uint32_t index = 0; index < 4; ++index)
        desc.gather_indices[index] = d2[index];
      for (uint32_t index = 0; index < 4; ++index)
        desc.gather_indices[index + 4] = d3[index];
    } else {
      for (uint32_t index = 0; index < 4; ++index) {
        desc.gather_indices[index * 2] = d2[index] & 0xffffu;
        desc.gather_indices[index * 2 + 1] = (d2[index] >> 16) & 0xffffu;
        desc.gather_indices[index * 2 + 8] = d3[index] & 0xffffu;
        desc.gather_indices[index * 2 + 9] = (d3[index] >> 16) & 0xffffu;
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
        const TensorDmaAxis &previous = axes_[insertion_idx - 1];
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
      const TensorDmaAxis &axis = axes_[axis_idx];
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
      const TensorDmaAxis &axis = axes_[axis_idx - 1];
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

class TensorDmaTransferElement {
public:
  uint64_t global_address = 0;
  uint32_t lds_address = 0;
  std::array<uint8_t, 8> bytes{};
  std::size_t completed_bytes = 0;
  bool in_bounds = false;
};

class TensorDmaState final : public DynamicInstState {
public:
  TensorDmaState(TensorDmaDescriptor descriptor, bool store)
      : desc(std::move(descriptor)), store_from_lds(store) {
    tag_ = TENSOR_DMA;
  }

  TensorDmaDescriptor desc;
  std::optional<GpuVmAccess> access;
  std::vector<TensorDmaTransferElement> elements;
  std::size_t next_element = 0;
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  bool store_from_lds = false;
  bool completion_committed = false;
  bool observation_reported = false;
};

inline void append_copy(TensorDmaState &state, Wavefront &wf, uint64_t global_element,
                        uint64_t lds_element, bool in_bounds) {
  const TensorDmaDescriptor &desc = state.desc;
  const uint64_t global_addr = desc.global_base + global_element * desc.elem_size;
  uint64_t lds_byte = lds_element * desc.elem_size;
  // The ISA applies descriptor padding only to memory-to-LDS transfers.
  // Stores read the ordinary dense LDS stream and ignore the padding fields.
  if (desc.pad && !state.store_from_lds) {
    const uint32_t pad_interval_bytes = desc.pad_interval * sizeof(uint32_t);
    const uint32_t pad_amount_bytes = desc.pad_amount * sizeof(uint32_t);
    lds_byte += (lds_byte / pad_interval_bytes) * pad_amount_bytes;
  }

  const uint32_t lds_addr = wf.lds_base() + desc.lds_base + static_cast<uint32_t>(lds_byte);
  TensorDmaTransferElement element{
      .global_address = global_addr, .lds_address = lds_addr, .in_bounds = in_bounds};
  if (state.store_from_lds && in_bounds) {
    for (uint32_t byte = 0; byte < desc.elem_size; ++byte)
      element.bytes[byte] = wf.lds().read8(lds_addr + byte);
  }
  state.elements.push_back(element);
}

template <typename Visitor>
inline void for_each_gather_tensor_element(const TensorDmaDescriptor &desc,
                                           const TensorDmaLayout &layout, Visitor &visit) {
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
      visit(global_element, lds_element, in_bounds);
    }
  }
}

template <typename Visitor>
inline void for_each_dense_tensor_element(const TensorDmaDescriptor &desc,
                                          const TensorDmaLayout &layout, Visitor &visit) {
  const uint32_t rank = layout.rank();
  if (rank == 0)
    return;

  uint64_t element_count = 1;
  for (uint32_t dim = 0; dim < rank; ++dim)
    element_count *= desc.tile_dims[dim];

  const uint32_t iteration_count = desc.iterate ? desc.iteration_count : 1;
  for (uint32_t iter = 0; iter < iteration_count; ++iter) {
    const uint64_t iteration_offset = static_cast<uint64_t>(iter) * desc.global_increment;
    const std::array<uint64_t, 5> iteration_origin =
        desc.iterate && !layout.empty() ? layout.origin_from_linear_offset(iteration_offset)
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

      visit(global_element, lds_element, in_bounds);
    }
  }
}

template <typename Visitor>
inline void for_each_tensor_element(const TensorDmaDescriptor &desc, const TensorDmaLayout &layout,
                                    Visitor &&visit) {
  if (desc.gather)
    for_each_gather_tensor_element(desc, layout, visit);
  else
    for_each_dense_tensor_element(desc, layout, visit);
}

inline util::Result prepare_tensor(TensorDmaState &state, Wavefront &wf) {
  const TensorDmaDescriptor &desc = state.desc;
  const TensorDmaLayout layout(desc);
  if (validate_supported_descriptor(desc, layout).failed()) [[unlikely]]
    return util::Result::failure();
  if (layout.rank() != 0 && !wf.has_gpu_memory()) [[unlikely]]
    return util::Result::failure();
  auto append_element = [&](uint64_t global_element, uint64_t lds_element, bool in_bounds) {
    append_copy(state, wf, global_element, lds_element, in_bounds);
  };
  for_each_tensor_element(desc, layout, append_element);

  if (wf.address_space()) {
    state.access = wf.snapshot_vm_access();
    if (!state.access)
      state.outcome = VmAccessOutcome::Faulted;
  }
  return util::Result::success();
}

inline bool materialize_observed_addresses(const void *context, std::span<uint64_t> addresses) {
  if (!context)
    return false;
  const auto &desc = *static_cast<const TensorDmaDescriptor *>(context);
  const TensorDmaLayout layout(desc);
  size_t next_address = 0;
  bool fits = true;
  auto collect_address = [&](uint64_t global_element, uint64_t, bool in_bounds) {
    if (!in_bounds)
      return;
    if (next_address == addresses.size()) {
      fits = false;
      return;
    }
    addresses[next_address++] = desc.global_base + global_element * desc.elem_size;
  };
  for_each_tensor_element(desc, layout, collect_address);
  return fits && next_address == addresses.size();
}

inline void report_completed_tensor_dma(const Instruction &inst, Wavefront &wf,
                                        TensorDmaState &state) {
  if (state.outcome != VmAccessOutcome::Complete || state.observation_reported)
    return;

  state.observation_reported = true;
  if (!wf.cu().observes_tensor_dma_memory_access())
    return;

  const size_t observed_address_count = static_cast<size_t>(
      std::count_if(state.elements.begin(), state.elements.end(),
                    [](const TensorDmaTransferElement &element) { return element.in_bounds; }));
  if (observed_address_count == 0)
    return;

  const TensorDmaMemoryAccessObservation access{
      .mnemonic = inst.mnemonic(),
      .pc = wf.pc,
      .compute_unit_id = wf.cu().id(),
      .dispatch_id = wf.dispatch_id(),
      .queue_id = wf.queue_id(),
      .workgroup_id = wf.wg_id(),
      .wavefront_id = wf.wf_id(),
      .process_id = wf.process_id(),
      .element_size_bytes = state.desc.elem_size,
      .is_load = !state.store_from_lds,
      .addresses = TensorDmaAddressView::deferred(observed_address_count, &state.desc,
                                                  &materialize_observed_addresses),
      .tile_dim0 = state.desc.tile_dims[0],
      .tile_dim1 = state.desc.gather ? state.desc.valid_indices : state.desc.tile_dims[1],
      .data_size = state.desc.data_size,
      .tensor_dim0_stride =
          static_cast<int64_t>(state.desc.global_strides[0] * state.desc.elem_size),
      .tensor_dim1_stride =
          static_cast<int64_t>(state.desc.global_strides[1] * state.desc.elem_size),
  };
  wf.cu().report_tensor_dma_memory_access(access);
}

inline void arrive_atomic_barrier(const TensorDmaDescriptor &desc, Wavefront &wf) {
  const uint32_t addr = wf.lds_base() + desc.atomic_barrier_addr;
  const uint64_t state = wf.lds().read64(addr);
  wf.lds().write64(addr, lds_barrier_cell_update_arrive(state));
}

// A declined batch has no observable effect. Only the element access below
// publishes faults or advances resumable progress, preserving precise addresses.
inline size_t try_coalesced_transfer(TensorDmaState &state, Wavefront &wf) {
  constexpr size_t kCopyBufferBytes = 256;
  constexpr uint64_t kPageBytes = IdentityAddressSpaceTranslator::kPageSize;
  const auto &first = state.elements[state.next_element];
  if ((!state.access && wf.process_id() != 0) || !first.in_bounds || first.completed_bytes != 0)
    return 0;
  const size_t size = state.desc.elem_size;
  size_t count = 1;
  while (state.next_element + count < state.elements.size() &&
         (count + 1) * size <= kCopyBufferBytes &&
         (count + 1) * size <= kPageBytes - (first.global_address & (kPageBytes - 1))) {
    const auto &next = state.elements[state.next_element + count];
    if (!next.in_bounds || next.completed_bytes != 0 ||
        next.global_address != first.global_address + count * size ||
        next.lds_address != first.lds_address + count * size)
      break;
    ++count;
  }
  if (count == 1)
    return 0;
  std::array<std::byte, kCopyBufferBytes> bytes;
  auto span = std::span(bytes).first(count * size);
  if (state.store_from_lds) {
    for (size_t i = 0; i < count; ++i)
      std::memcpy(bytes.data() + i * size, state.elements[state.next_element + i].bytes.data(),
                  size);
  }
  bool copied;
  if (state.access) {
    copied = state.store_from_lds ? state.access->try_write_contiguous(first.global_address, span)
                                  : state.access->try_read_contiguous(first.global_address, span);
  } else if (state.store_from_lds) {
    copied = wf.write_gpu_memory(first.global_address,
                                 {reinterpret_cast<const uint8_t *>(bytes.data()), span.size()}) ==
             VmAccessOutcome::Complete;
  } else {
    copied = wf.read_gpu_memory(first.global_address, {reinterpret_cast<uint8_t *>(bytes.data()),
                                                       span.size()}) == VmAccessOutcome::Complete;
  }
  if (!copied)
    return 0;
  for (size_t i = 0; i < count; ++i) {
    auto &element = state.elements[state.next_element + i];
    if (!state.store_from_lds)
      std::memcpy(element.bytes.data(), bytes.data() + i * size, size);
    element.completed_bytes = size;
  }
  return count;
}

template <bool Coalesce = true>
inline VmAccessOutcome resume_tensor_dma_state(TensorDmaState &state, Wavefront &wf) {
  if (state.outcome != VmAccessOutcome::Complete && state.outcome != VmAccessOutcome::Unavailable)
    return state.outcome;

  if (state.access && !state.access->info().ready && state.next_element == 0 &&
      (state.elements.empty() || state.elements.front().completed_bytes == 0)) {
    state.access = wf.snapshot_vm_access();
    if (!state.access)
      return state.outcome = VmAccessOutcome::Faulted;
  }

  while (state.next_element < state.elements.size()) {
    if constexpr (Coalesce) {
      if (const size_t count = try_coalesced_transfer(state, wf)) {
        state.next_element += count;
        continue;
      }
    }
    TensorDmaTransferElement &element = state.elements[state.next_element];
    if (!element.in_bounds) {
      ++state.next_element;
      continue;
    }

    const uint32_t size = state.desc.elem_size;
    VmAccessOutcome outcome = VmAccessOutcome::Complete;
    if (state.access) {
      if (state.store_from_lds) {
        outcome = state.access->write(
            element.global_address,
            std::span<const std::byte>(reinterpret_cast<const std::byte *>(element.bytes.data()),
                                       size),
            element.completed_bytes);
      } else {
        outcome = state.access->read(
            element.global_address,
            std::span<std::byte>(reinterpret_cast<std::byte *>(element.bytes.data()), size),
            element.completed_bytes);
      }
    } else if (state.store_from_lds) {
      outcome = wf.write_gpu_memory(element.global_address,
                                    std::span<const uint8_t>(element.bytes).first(size));
    } else {
      outcome =
          wf.read_gpu_memory(element.global_address, std::span<uint8_t>(element.bytes).first(size));
    }
    if (outcome != VmAccessOutcome::Complete) {
      state.outcome = outcome;
      return outcome;
    }
    ++state.next_element;
  }

  if (!state.completion_committed) {
    if (!state.store_from_lds) {
      for (const TensorDmaTransferElement &element : state.elements) {
        for (uint32_t byte = 0; byte < state.desc.elem_size; ++byte)
          wf.lds().write8(element.lds_address + byte, element.bytes[byte]);
      }
    }
    if (state.desc.atomic_barrier)
      arrive_atomic_barrier(state.desc, wf);
    state.completion_committed = true;
  }
  state.outcome = VmAccessOutcome::Complete;
  return state.outcome;
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
util::Result execute_tensor_dma(Inst &inst, Wavefront &wf, bool store_from_lds) {
  ScopedWaitCounter counter(wf, WaitCounterType::TENSORCNT);
  if (inst.data() != nullptr && inst.data()->tag() == TENSOR_DMA &&
      inst.template data_as<TensorDmaState>()->outcome != VmAccessOutcome::Unavailable)
    inst.set_data(nullptr);
  if (inst.data() == nullptr) {
    const auto desc = read_descriptor(inst, wf);
    if (desc.failed()) [[unlikely]]
      return util::Result::failure();
    std::unique_ptr<TensorDmaState> state =
        std::make_unique<TensorDmaState>(desc.value(), store_from_lds);
    if (desc.value().active() && prepare_tensor(*state, wf).failed()) [[unlikely]]
      return util::Result::failure();
    inst.set_data(std::move(state));
  }
  TensorDmaState &state = *inst.template data_as<TensorDmaState>();
  if (!state.desc.active())
    return util::Result::success();
  state.outcome = resume_tensor_dma_state(state, wf);
  report_completed_tensor_dma(inst, wf, state);
  return util::Result::success();
}

} // namespace tensor_dma_detail

inline bool is_tensor_dma_instruction(const Instruction &inst) {
  return inst.data() != nullptr && inst.data()->tag() == TENSOR_DMA;
}

inline VmAccessOutcome tensor_dma_outcome(const Instruction &inst) {
  return inst.data_as<tensor_dma_detail::TensorDmaState>()->outcome;
}

inline VmAccessOutcome resume_tensor_dma(Instruction &inst, Wavefront &wf) {
  tensor_dma_detail::TensorDmaState &state = *inst.data_as<tensor_dma_detail::TensorDmaState>();
  state.outcome = tensor_dma_detail::resume_tensor_dma_state(state, wf);
  tensor_dma_detail::report_completed_tensor_dma(inst, wf, state);
  return state.outcome;
}

template <typename Inst> util::Result execute_tensor_load_to_lds(Inst &inst, Wavefront &wf) {
  return tensor_dma_detail::execute_tensor_dma(inst, wf, false);
}

template <typename Inst> util::Result execute_tensor_store_from_lds(Inst &inst, Wavefront &wf) {
  return tensor_dma_detail::execute_tensor_dma(inst, wf, true);
}

} // namespace amdgpu
} // namespace rocjitsu
