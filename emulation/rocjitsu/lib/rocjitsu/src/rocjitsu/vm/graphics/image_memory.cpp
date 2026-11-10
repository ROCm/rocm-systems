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

namespace rocjitsu::amdgpu {

namespace {

ImageMetadataResult transfer_image_metadata(const GpuVmAccess &memory, VectorMemState &d) {
  auto &image = *d.image_metadata;
  const uint32_t tap_count = d.graphics_sample ? d.graphics_sample->tap_count : 1;
  const size_t tap_bytes = d.wf_size * d.elem_size;
  std::array<std::optional<Gfx11DccMipLayout>, 16> mip_layouts{};
  for (uint32_t tap = 0; tap < tap_count; ++tap) {
    const auto &addresses =
        d.graphics_sample ? d.graphics_sample->taps[tap].addresses : d.per_lane_addr;
    const auto &coordinates =
        d.graphics_sample ? d.graphics_sample->taps[tap].coordinates : image.coordinates;
    const uint64_t lane_mask =
        d.graphics_sample ? d.graphics_sample->taps[tap].lane_mask : d.lane_mask;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      const uint32_t transfer = tap * d.wf_size + lane;
      if (transfer < image.next_transfer || !(lane_mask & (uint64_t{1} << lane)))
        continue;
      const uint32_t x = coordinates[lane] & 0xffff;
      const uint32_t y = coordinates[lane] >> 16;
      const ImageMetadataResult result = [&]() -> ImageMetadataResult {
        if (image.mip_levels > 1) {
          const uint32_t level =
              d.graphics_sample ? image.tap_levels[tap][lane] : image.levels[lane];
          if (image.depth || level >= mip_layouts.size())
            return util::Result::failure();
          auto &mip = mip_layouts[level];
          if (!mip)
            mip = gfx11_dcc_mip_layout(image.swizzle, d.elem_size, image.width, image.height,
                                       image.mip_levels, level, image.pipe_aligned);
          if (!mip)
            return util::Result::failure();
          const uint32_t layer =
              d.graphics_sample ? d.graphics_sample->taps[tap].layers[lane] : image.layers[lane];
          return materialize_gfx11_dcc_mip(memory, image.base, image.metadata, x, y, d.elem_size,
                                           image.swizzle, image.pipe_aligned, layer, *mip);
        }
        return image.depth
                   ? materialize_gfx11_htile(memory, image.base, image.metadata, x, y, image.width,
                                             image.height, d.elem_size, image.swizzle)
                   : materialize_gfx11_dcc(
                         memory, image.base, image.metadata, x, y, image.width, image.height,
                         d.elem_size, image.swizzle, image.pipe_aligned,
                         d.graphics_sample ? d.graphics_sample->taps[tap].layers[lane]
                                           : image.layers[lane],
                         image.slice_size);
      }();
      if (result.failed() || result.value() != VmAccessOutcome::Complete)
        return result;
      const VmAccessOutcome outcome =
          d.is_load ? memory.read(addresses[lane],
                                  std::as_writable_bytes(std::span{
                                      d.response_data.data() + tap * tap_bytes + lane * d.elem_size,
                                      d.elem_size}))
                    : memory.write(addresses[lane],
                                   std::as_bytes(std::span{d.store_data.data() + lane * d.elem_size,
                                                           d.elem_size}));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      image.next_transfer = transfer + 1;
    }
  }
  return VmAccessOutcome::Complete;
}

} // namespace

VmAccessOutcome GlobalMemPipeline::initiate_image_metadata_access(VectorMemState &d,
                                                                  Wavefront &wf) {
  if (!l2_)
    return VmAccessOutcome::Faulted;
  const auto access = wf.snapshot_vm_access();
  if (!access)
    return VmAccessOutcome::Faulted;
  auto boundary = l2_->coherence_domain()->acquire_atomic_boundary();
  if (boundary.outcome() != VmAccessOutcome::Complete)
    return boundary.outcome();
  if (d.is_load) {
    const auto taps = d.graphics_sample ? d.graphics_sample->tap_count : 1;
    d.response_data.resize(size_t(taps) * d.wf_size * d.elem_size);
  }
  const ImageMetadataResult result = transfer_image_metadata(*access, d);
  if (result.failed()) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return VmAccessOutcome::Faulted;
  }
  return result.value();
}
} // namespace rocjitsu::amdgpu
