// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/graphics/image_metadata.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <stdexcept>

namespace rocjitsu::amdgpu {

namespace {

// Bounds only: no metadata or pixel is read during optional preparation. The
// metadata envelope includes its padded layer; the pixel envelope includes only
// original direct texels. A clear outside either span releases the lease before
// continuing through ordinary VM access at that exact operation.
template <bool IncludeClearBlocks = false>
std::optional<std::array<VmRamRange, 2>> image_metadata_ram_ranges(const VectorMemState &d) {
  const auto &image = *d.image_metadata;
  const uint32_t bytes = d.elem_size;
  if (image.mip_levels != 1 || !image.width || !image.height || image.width > 4096 ||
      image.height > 4096 || !d.wf_size || d.wf_size > 64 ||
      (image.depth ? (bytes != 4 || (image.swizzle != 24 && image.swizzle != 28))
                   : (!std::has_single_bit(bytes) || bytes > 16 ||
                      (image.swizzle != 27 && image.swizzle != 31))))
    return std::nullopt;
  const uint32_t tap_count = d.image_sample ? d.image_sample->tap_count : 1;
  if (!tap_count || tap_count > ImageSampleAccess::kMaxTaps ||
      (d.image_sample && tap_count > d.image_sample->taps.size()) ||
      (!d.is_load && d.store_data.size() < size_t{d.wf_size} * bytes))
    return std::nullopt;
  std::optional<uint32_t> layer;
  uint64_t first_texel = UINT64_MAX, last_texel = 0;
  uint64_t first_block = UINT64_MAX, last_block = 0;
  for (uint32_t tap = 0; tap < tap_count; ++tap) {
    const auto &addresses = d.image_sample ? d.image_sample->taps[tap].addresses : d.per_lane_addr;
    const auto &coordinates =
        d.image_sample ? d.image_sample->taps[tap].coordinates : image.coordinates;
    const auto &layers = d.image_sample ? d.image_sample->taps[tap].layers : image.layers;
    const uint64_t mask = d.image_sample ? d.image_sample->taps[tap].lane_mask : d.lane_mask;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(mask & (uint64_t{1} << lane)))
        continue;
      if ((coordinates[lane] & 0xffff) >= image.width ||
          (coordinates[lane] >> 16) >= image.height || addresses[lane] > UINT64_MAX - bytes ||
          (image.depth && layers[lane]) || (layer && *layer != layers[lane]))
        return std::nullopt;
      layer = layers[lane];
      first_texel = std::min(first_texel, addresses[lane]);
      last_texel = std::max(last_texel, addresses[lane] + bytes);
      if constexpr (IncludeClearBlocks) {
        const auto block = image_metadata_detail::pixel_swizzle_block_offset(
            coordinates[lane] & 0xffff, coordinates[lane] >> 16, image.width, bytes, image.swizzle);
        first_block = std::min(first_block, block);
        last_block = std::max(last_block, block);
      }
    }
  }
  if (!layer ||
      (*layer && (!image.slice_size || image.slice_size > (UINT64_MAX - image.base) / *layer)))
    return std::nullopt;
  const auto mip = image_mip_layout(false, image.swizzle, bytes, image.width, image.height, 1, 0);
  if (!mip || (*layer && image.slice_size != mip->slice_size))
    return std::nullopt;
  const uint32_t pixel_log2 = image_block_log2(false, image.swizzle);
  const uint64_t pixel_base =
      image_layer_base(false, image.base, image.slice_size, *layer, bytes, image.swizzle) &
      ~((uint64_t{1} << pixel_log2) - 1);
  if (mip->slice_size > UINT64_MAX - pixel_base || first_texel < pixel_base ||
      last_texel > pixel_base + mip->slice_size)
    return std::nullopt;
  if constexpr (IncludeClearBlocks) {
    const uint64_t block_size = uint64_t{1} << pixel_log2;
    if (last_block > mip->slice_size || block_size > mip->slice_size - last_block)
      return std::nullopt;
    first_texel = std::min(first_texel, pixel_base + first_block);
    last_texel = std::max(last_texel, pixel_base + last_block + block_size);
  }
  const uint32_t metadata_log2 = image.depth ? 17 : image.pipe_aligned ? 14 : 12;
  const uint32_t pixel_bits = image.depth ? 21 : metadata_log2 + 8 - std::countr_zero(bytes);
  const uint32_t xb = (pixel_bits + 1) / 2, yb = pixel_bits / 2;
  const uint64_t metadata_size = ((uint64_t{image.width} + (1u << xb) - 1) >> xb) *
                                     ((uint64_t{image.height} + (1u << yb) - 1) >> yb)
                                 << metadata_log2;
  const uint64_t metadata_base = image.metadata & ~((uint64_t{1} << metadata_log2) - 1);
  if (*layer && metadata_size > (UINT64_MAX - metadata_base) / *layer)
    return std::nullopt;
  const uint64_t metadata_layer = metadata_base + *layer * metadata_size;
  if (metadata_size > UINT64_MAX - metadata_layer)
    return std::nullopt;
  return std::array<VmRamRange, 2>{
      {{metadata_layer, metadata_size}, {first_texel, last_texel - first_texel}}};
}

// Both ordinary VM accesses and an admitted RAM transaction execute this same
// tap/lane/key/pixel/direct-copy sequence. Static errors defer exception storage
// until the optional mapping admission has been released.
template <bool CacheAddress = false, typename Memory>
const char *transfer_image_metadata(const Memory &memory, VectorMemState &d) {
  const auto &image = *d.image_metadata;
  const uint32_t tap_count = d.image_sample ? d.image_sample->tap_count : 1;
  const size_t tap_bytes = d.wf_size * d.elem_size;
  std::array<std::optional<Gfx11DccMipLayout>, 16> mip_layouts{};
  image_metadata_detail::MetadataAddressCache address_cache(
      image.metadata, image.width, image.height, d.elem_size, image.swizzle, image.depth,
      image.pipe_aligned);
  for (uint32_t tap = 0; tap < tap_count; ++tap) {
    const auto &addresses = d.image_sample ? d.image_sample->taps[tap].addresses : d.per_lane_addr;
    const auto &coordinates =
        d.image_sample ? d.image_sample->taps[tap].coordinates : image.coordinates;
    const uint64_t lane_mask = d.image_sample ? d.image_sample->taps[tap].lane_mask : d.lane_mask;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(lane_mask & (uint64_t{1} << lane)))
        continue;
      const uint32_t x = coordinates[lane] & 0xffff;
      const uint32_t y = coordinates[lane] >> 16;
      const char *error = [&]() {
        if (image.mip_levels > 1) {
          const uint32_t level = d.image_sample ? image.tap_levels[tap][lane] : image.levels[lane];
          if (image.depth || level >= mip_layouts.size())
            return "unsupported GFX11 metadata mip level";
          auto &mip = mip_layouts[level];
          if (!mip)
            mip = gfx11_dcc_mip_layout(image.swizzle, d.elem_size, image.width, image.height,
                                       image.mip_levels, level, image.pipe_aligned);
          if (!mip)
            return "unsupported GFX11 DCC mip layout";
          const uint32_t layer =
              d.image_sample ? d.image_sample->taps[tap].layers[lane] : image.layers[lane];
          return image_metadata_detail::materialize_gfx11_dcc_mip(
              memory, image.base, image.metadata, x, y, d.elem_size, image.swizzle,
              image.pipe_aligned, layer, *mip);
        }
        if constexpr (CacheAddress) {
          if (memory.has_ram()) {
            const uint32_t layer = image.depth      ? 0
                                   : d.image_sample ? d.image_sample->taps[tap].layers[lane]
                                                    : image.layers[lane];
            const auto address = address_cache.lookup(x, y, layer);
            if (!address)
              return image.depth ? "unsupported GFX11 HTILE surface layout"
                                 : "unsupported GFX11 DCC surface layout";
            return image.depth ? image_metadata_detail::materialize_gfx11_htile_at_address(
                                     memory, image.base, *address, x, y, image.width, image.height,
                                     d.elem_size, image.swizzle)
                               : image_metadata_detail::materialize_gfx11_dcc_at_address(
                                     memory, image.base, *address, x, y, image.width, image.height,
                                     d.elem_size, image.swizzle, layer, image.slice_size);
          }
        }
        return image.depth ? image_metadata_detail::materialize_gfx11_htile(
                                 memory, image.base, image.metadata, x, y, image.width,
                                 image.height, d.elem_size, image.swizzle)
                           : image_metadata_detail::materialize_gfx11_dcc(
                                 memory, image.base, image.metadata, x, y, image.width,
                                 image.height, d.elem_size, image.swizzle, image.pipe_aligned,
                                 d.image_sample ? d.image_sample->taps[tap].layers[lane]
                                                : image.layers[lane],
                                 image.slice_size);
      }();
      if (error)
        return error;
      if (d.is_load) {
        if (memory.read(addresses[lane],
                        std::as_writable_bytes(
                            std::span{d.response_data.data() + tap * tap_bytes + lane * d.elem_size,
                                      d.elem_size})) != VmAccessOutcome::Complete)
          return "image read failed";
      } else if (memory.write(addresses[lane],
                              std::as_bytes(std::span{d.store_data.data() + lane * d.elem_size,
                                                      d.elem_size})) != VmAccessOutcome::Complete)
        return "image write failed";
    }
  }
  return nullptr;
}

} // namespace

VmAccessOutcome GlobalMemPipeline::initiate_image_metadata_access(Instruction &inst,
                                                                  Wavefront &wf) {
  auto &d = *inst.data_as<VectorMemState>();
  // Metadata transitions touch an entire compression block. Serialize them
  // with other CUs and publish dirty cache contents before accessing backing
  // memory, as for a device-wide atomic operation. The functional renderer
  // always leaves the resulting block in the uncompressed encoding.
  if (!l2_)
    return VmAccessOutcome::Faulted;
  if (collect_step_metadata_ && deferable_instruction_ == &inst && d.is_load &&
      d.translated.access && d.translated.access->info().ready &&
      d.translated.access->info().legacy_cache_compatible &&
      d.translated.access->supports_ram_word_reads() && !wf.raw_cu().debug_active() &&
      wf.raw_cu().plugin_group().empty()) {
    // A voluntary issue stall: no backing access has occurred. issued_ keeps
    // ownership and wait tokens; the same wave cannot issue its successor.
    step_staged_instruction_ = &inst;
    return VmAccessOutcome::Unavailable;
  }
  std::optional<std::array<VmRamRange, 2>> ram_ranges;
  std::unique_ptr<VmRamLeaseRequest> ram_request;
  struct DestroyPreparedRequest {
    std::unique_ptr<VmRamLeaseRequest> &request;
    ~DestroyPreparedRequest() {
      const int saved_errno = errno;
      request.reset();
      errno = saved_errno;
    }
  } destroy_prepared{ram_request};
  if (d.translated.access && d.translated.access->supports_ram_word_reads() &&
      !wf.raw_cu().debug_active() && wf.raw_cu().plugin_group().empty()) {
    ram_ranges = image_metadata_ram_ranges(d);
    if (ram_ranges) {
      const int saved_errno = errno;
      ram_request = d.translated.access->prepare_ram_lease(*ram_ranges);
      errno = saved_errno;
    }
  }
  // Prepared storage outlives the device boundary. Acquire mapping ownership
  // only inside that boundary, and release it before reporting any error.
  auto boundary = l2_->coherence_domain()->acquire_atomic_boundary();
  if (boundary.outcome() != VmAccessOutcome::Complete)
    return boundary.outcome();
  const uint32_t tap_count = d.image_sample ? d.image_sample->tap_count : 1;
  const size_t tap_bytes = d.wf_size * d.elem_size;
  if (d.is_load)
    d.response_data.assign(tap_count * tap_bytes, 0);
  try {
    const auto access = wf.snapshot_vm_access();
    if (!access)
      return VmAccessOutcome::Faulted;
    const auto &memory = *access;
    const char *error = nullptr;
    bool admitted = false;
    if (ram_request && memory.shares_access_state(*d.translated.access)) {
      const int saved_errno = errno;
      admitted = ram_request->try_acquire();
      errno = saved_errno;
      if (admitted) {
        // This owner releases once on refusal, completion, exception or the
        // first uncovered access. No later cleanup repeats the transition.
        const image_metadata_detail::FallbackRamAccess ram(memory, *ram_ranges, *ram_request);
        // Instruction buffers are normally separate heap storage; explicitly
        // refuse any raw registration that aliases the selected destination.
        const auto host_buffer = d.is_load ? std::span{d.response_data} : std::span{d.store_data};
        const auto host_begin = reinterpret_cast<uintptr_t>(host_buffer.data());
        for (size_t i = 0; i < ram_ranges->size(); ++i) {
          const auto bytes = ram_request->bytes(i);
          const auto begin = reinterpret_cast<uintptr_t>(bytes.data());
          if (host_begin < begin + bytes.size() && begin < host_begin + host_buffer.size())
            admitted = false;
        }
        if (admitted)
          error = transfer_image_metadata<true>(ram, d);
      }
    }
    if (!admitted)
      error = transfer_image_metadata(memory, d);
    if (error)
      throw std::runtime_error(error);
  } catch (const std::runtime_error &) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    reject_vector_memory_access(d);
  }
  return VmAccessOutcome::Complete;
}

void GlobalMemPipeline::finish_step_batch() {
  {
    struct PreserveErrno {
      int value = errno;
      ~PreserveErrno() { errno = value; }
    } preserve_errno;
    struct Prepared {
      uint64_t id = 0;
      std::array<VmRamRange, 2> ranges{};
      std::optional<GpuVmAccess> access;
      std::unique_ptr<VmRamLeaseRequest> request;
      std::vector<uint8_t> response;
    };
    const auto candidates =
        std::ranges::count_if(issued_, [](const auto &entry) { return entry.step_batch; });
    const uint64_t last_id = next_issue_id_ - 1;
    std::vector<Prepared> prepared(candidates > 1 ? candidates : 0);
    size_t selected = 0;
    // The allocation above may reenter cancellation. Snapshot only identities
    // still present afterward, never references or iterators into issued_.
    for (const auto &entry : issued_) {
      if (entry.step_batch && entry.issue_id <= last_id && selected < prepared.size())
        prepared[selected++].id = entry.issue_id;
    }
    const auto find_entry = [&](uint64_t id) -> PipelineEntry * {
      const auto found = std::ranges::find(issued_, id, &PipelineEntry::issue_id);
      return found == issued_.end() ? nullptr : &*found;
    };
    for (auto &item : prepared) {
      size_t response_size = 0;
      {
        const auto *entry = find_entry(item.id);
        if (!entry)
          continue;
        const auto &d = *entry->inst->data_as<VectorMemState>();
        const auto ranges = image_metadata_ram_ranges<true>(d);
        if (!ranges)
          break;
        item.ranges = *ranges;
        item.access = d.translated.access;
        response_size =
            size_t{d.image_sample ? d.image_sample->tap_count : 1} * d.wf_size * d.elem_size;
      }
      // The unlocked factory and response allocation may cancel any queued
      // wave. Their receivers are independent of the instruction's lifetime.
      {
        PreserveErrno factory_errno;
        item.request = item.access->prepare_ram_lease(item.ranges);
      }
      if (!item.request)
        break;
      if (find_entry(item.id)) {
        PreserveErrno allocation_errno;
        item.response.assign(response_size, 0);
      }
    }
    const auto ready = std::ranges::count_if(prepared, [&](const auto &item) {
      return item.request && !item.response.empty() && find_entry(item.id);
    });
    if (ready > 1) {
      auto boundary = l2_->coherence_domain()->try_acquire_clean_boundary();
      if (boundary.outcome() == VmAccessOutcome::Complete) {
        for (auto &item : prepared) {
          auto *entry = find_entry(item.id);
          if (!entry || !item.request || item.response.empty())
            continue;
          auto &d = *entry->inst->data_as<VectorMemState>();
          const auto access = entry->wf->snapshot_vm_access();
          if (entry->wf->dispatch_generation() != entry->wave_generation || !access ||
              !access->shares_access_state(*item.access))
            break;
          if (!item.request->try_acquire())
            break;
          struct Release {
            VmRamLeaseRequest &request;
            ~Release() { request.release(); }
          } release{*item.request};
          bool disjoint = true;
          for (size_t i = 0; i < item.ranges.size(); ++i) {
            const auto bytes = item.request->bytes(i);
            const auto begin = reinterpret_cast<uintptr_t>(bytes.data());
            const auto check = [&](const auto &response) {
              const auto host = reinterpret_cast<uintptr_t>(response.data());
              return host >= begin + bytes.size() || begin >= host + response.size();
            };
            for (const auto &other : prepared) {
              disjoint &= check(other.response);
              if (const auto *live = find_entry(other.id))
                disjoint &= check(live->inst->data_as<VectorMemState>()->response_data);
            }
          }
          if (!disjoint)
            break;
          const image_metadata_detail::RamAccess ram(item.ranges, *item.request);
          // All callbacks/allocations are excluded from this interval. Keep
          // both old and new buffers alive until the device boundary is gone.
          d.response_data.swap(item.response);
          boundary.advance_data_epoch();
          const char *error = transfer_image_metadata<true>(ram, d);
          if (error) {
            entry->wf->report_instruction_execution_error(
                InstructionExecutionError::UnsupportedOperandValue);
            reject_vector_memory_access(d);
          }
          entry->transfer_ready = true;
        }
      }
    }
    // Boundary, request guards, request objects and all buffers/capacity are
    // destroyed before restoring errno and entering architectural completion.
  }
  tick_impl(true);
}

} // namespace rocjitsu::amdgpu
