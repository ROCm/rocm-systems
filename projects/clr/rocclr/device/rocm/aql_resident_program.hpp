// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>

namespace amd::roc::aql_resident {

// Host/device relocation contract: write bindings[bindingSlot] + bindingOffset
// as one 64-bit word at resident_base + targetOffset. Entries are 16 bytes;
// the GPU update kernel and host compiler must use this exact layout.
struct BindingFixup {
  uint32_t targetOffset;
  uint32_t bindingSlot;
  uint64_t bindingOffset;
};
static_assert(sizeof(BindingFixup) == 16);
static_assert(offsetof(BindingFixup, bindingOffset) == 8);

// Only compiler-declared mutable bytes may be touched by a GPU fixup.
// Offsets are relative to one resident physical-queue variant.
struct MutableRange {
  uint32_t offset;
  uint32_t length;
};

enum class PlanStatus { Ok, Bounds, Alignment, Binding, Overlap, Overflow };

// Immutable after successful compilation. This class owns the checked fixup
// metadata, not GPU allocations or launch inputs. The latter belong to the
// queue publication/submission layer. No HIP or HSA dependency is needed here.
class BindingPlan {
 public:
  // Failure leaves the prior plan unchanged. Allocation failures propagate to
  // the adapter, which must handle them before publishing any submission.
  PlanStatus compile(uint64_t imageBytes, uint32_t bindingCount,
                     const std::vector<MutableRange>& mutableRanges,
                     const std::vector<BindingFixup>& entries);
  // Validate launch values before a GPU sees them; in particular, address-plus-
  // offset fixups may not wrap. Raw scalar words should use bindingOffset=0.
  PlanStatus validateBindings(const std::vector<uint64_t>& bindings) const;
  const std::vector<BindingFixup>& entries() const { return entries_; }
  uint64_t imageBytes() const { return imageBytes_; }
  uint32_t bindingCount() const { return bindingCount_; }

 private:
  uint64_t imageBytes_ = 0;
  uint32_t bindingCount_ = 0;
  std::vector<BindingFixup> entries_;
};

using Packet = std::array<uint32_t, 16>;

// Compiled resident packet template. Input comes from the validated fast
// packet encoder, not arbitrary user memory. Packet headers/control flow are
// invariant; supported dispatch fields and SET payloads become raw 64-bit
// binding values. This also covers scalar/preload words without guessing types.
class PacketTemplate {
 public:
  PlanStatus compile(const std::vector<Packet>& packets);
  PlanStatus bind(const std::vector<Packet>& packets, std::vector<uint64_t>& values) const;
  const std::vector<uint8_t>& image() const { return image_; }
  const BindingPlan& plan() const { return plan_; }
  uint32_t packetCount() const { return static_cast<uint32_t>(packets_.size()); }
  uint32_t fixupOffset() const { return packetCount() * 64; }
 private:
  std::vector<Packet> packets_;
  // Immutable validation spans are compiled once; bind need not rediscover
  // mutability with one branch per packet word. Adjacent spans may be merged.
  struct ImmutableSpan { uint32_t offset, bytes; };
  std::vector<ImmutableSpan> immutableSpans_;
  BindingPlan plan_;
  std::vector<uint8_t> image_;
};

}  // namespace amd::roc::aql_resident
