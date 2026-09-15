// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/scratch_pci_device.h"

#include <algorithm>
#include <utility>

namespace rocjitsu {
namespace {

// A guest reaches device registers with ordinary loads and stores, so a
// transport can deliver any width the guest instruction used. Real register
// files answer a fixed set and fault on the rest; rejecting the others here
// keeps a transport bug from looking like a device that quietly tolerates
// nonsense.
bool is_supported_width(std::size_t width) {
  return width == 1 || width == 2 || width == 4 || width == 8;
}

} // namespace

ScratchPciDevice::ScratchPciDevice(std::string name, simdojo::PciId id, BarAccessTrace *trace)
    : simdojo::PciDevice(std::move(name), id), trace_(trace), storage_(kBarSize) {}

std::vector<simdojo::BarSpec> ScratchPciDevice::bars() const {
  simdojo::BarSpec bar;
  bar.index = kBarIndex;
  bar.size = kBarSize;
  bar.mem = true;
  return {bar};
}

int64_t ScratchPciDevice::bar_access(int bar, std::span<std::byte> buf, uint64_t offset,
                                     bool write) {
  const bool in_range = offset <= kBarSize && buf.size() <= kBarSize - offset;
  const bool acceptable = bar == kBarIndex && in_range && is_supported_width(buf.size());

  if (trace_ != nullptr) {
    // Every access this device accepts is modeled -- it is a scratch buffer, so
    // there is no register it answers with a default. A refusal is therefore
    // never a gap in the model, and reporting it as one would put work in the
    // burndown list that no hardware corresponds to.
    if (acceptable) {
      trace_->record(bar, offset, buf.size(), write, /*modeled=*/true);
    } else {
      trace_->record_rejected(bar, offset, buf.size(), write);
    }
  }
  if (!acceptable) {
    return -1;
  }

  const auto begin = storage_.begin() + static_cast<std::ptrdiff_t>(offset);
  if (write) {
    std::ranges::copy(buf, begin);
  } else {
    std::copy_n(begin, buf.size(), buf.begin());
  }
  return static_cast<int64_t>(buf.size());
}

void ScratchPciDevice::dma_map(const simdojo::DmaRegion & /*region*/) { ++mapped_regions_; }

void ScratchPciDevice::dma_unmap(const simdojo::DmaRegion & /*region*/) {
  if (mapped_regions_ != 0) {
    --mapped_regions_;
  }
}

void ScratchPciDevice::reset(simdojo::ResetKind /*kind*/) {
  std::ranges::fill(storage_, std::byte{0});
}

} // namespace rocjitsu
