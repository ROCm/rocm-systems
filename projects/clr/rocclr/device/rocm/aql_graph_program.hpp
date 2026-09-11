// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "device/device.hpp"
#include "aql_fast_packets.hpp"
#include "aql_resident_program.hpp"
#include "aql_resident_storage.hpp"
#include "aql_resident_fixup.hpp"
#include "aql_resident_submission.hpp"
#include "aql_resident_upload.hpp"
#include "aql_argument_bindings.hpp"
#include "aql_binding_update.hpp"
#include <map>
#include <mutex>

namespace amd::roc::aql_resident {
// Resident command-buffer model
// -----------------------------
// This opt-in backend requires support for the vendor packet layouts below;
// selecting a matching ISA alone is not a capability negotiation. Enable with
// HIP_GRAPH_AQL_IB_MODE=1 and HSA_NO_SCRATCH_RECLAIM=1 only on a compatible
// runtime/device stack. Default HIP behavior is unchanged when the mode is off.
// HIP's graph scheduler still owns dependency analysis, node filtering, stream
// selection and kernel-argument snapshots. This backend replaces only eligible
// flat kernel batches: compilation validates their descriptors and emits a
// register/dispatch program, an execution boundary and a terminal jump.
// Unsupported packet types, dynamic stack, oversized static scratch, preload
// payloads not available to the encoder, and per-dispatch signals keep ordinary
// AQL submission. A failed allocation before publication can also fall back;
// metadata-prefetch queues and device-memory packet rings retain their existing
// publication path until resident-root publication supports those queue modes.
// a failure after submission must never replay the work on the fallback path.
//
// The compiled packet headers and control flow are immutable. BindingPlan
// identifies the only mutable qwords and validates every target range. Updating
// a graph stages host binding values; it does not write memory a running GPU
// could be reading. Each physical queue gets its own resident image and applied
// binding history. This permits concurrent launches on different queues while
// preserving one in-order mutation/execution stream for each image.
//
// Replay optionally submits a GPU relocation kernel, then a dirty-target root
// jump. One queue reservation keeps them adjacent; publication exposes bodies
// before headers and the head last. The dirty-target jump orders relocation
// writes against packet fetch. Root system acquire/release bracket the batch;
// dispatch barriers retain dependencies within it. A final execution boundary
// precedes the terminal jump and root completion.
//
// The root completion signal retains Launch, which owns this program, its
// resident variant, and any upload lease. Graph destruction or a later update
// therefore cannot release allocations still read by submitted work. Uploads
// become reusable only after completion; cache limits do not limit in-flight
// launches. The mutex serializes host compilation/update/publication bookkeeping,
// not GPU completion. Static scratch is reserved when physical queues are made
// and kept resident by the non-reclaim policy; this backend never retries a
// partially executed program to grow scratch.
class GraphProgram final : public amd::AqlBatchImage,
                           public std::enable_shared_from_this<GraphProgram> {
 public:
  static std::shared_ptr<GraphProgram> create(Device& device,
      const amd::AlignedVector64<uint8_t>& packets, const std::vector<uint32_t>& headers);
  bool update(const amd::AlignedVector64<uint8_t>& packets,
              const std::vector<uint32_t>& headers) override;
  SubmitResult submit(VirtualGPU& gpu, bool blocking);
 private:
  explicit GraphProgram(Device& device);
  bool encode(const amd::AlignedVector64<uint8_t>& packets, std::vector<Packet>& output,
              std::vector<uint32_t>* argumentTargets = nullptr) const;
  struct Variant {
    std::shared_ptr<ResidentImage> image;
    std::unique_ptr<BindingKernel> kernel;
    std::shared_ptr<UploadPool> uploads;
    VirtualGPU* kernelQueue = nullptr;
    uint64_t generation = 0;
    // Last successfully submitted contents of this queue's mutable image.
    // Empty means the original published template's initial bindings.
    std::vector<uint64_t> appliedBindings;
    std::vector<uint64_t> stagedBindings;
  };
  struct Launch;
  amd::SharedReference<amd::Device> owner_;
  Device* device_;
  std::mutex mutex_;
  std::vector<uint32_t> headers_;
  std::vector<uint64_t> kernelObjects_;
  std::vector<aql_fast::Descriptor> descriptors_;
  amd::AlignedVector64<uint8_t> pending_;
  std::vector<uint64_t> bindings_;
  std::vector<uint64_t> initialBindings_;
  PacketTemplate template_;
  ArgumentBindingPlan argumentBindings_;
  // Resident packet program plus full and argument-only relocation tables.
  std::vector<uint8_t> imageBytes_;
  size_t argumentFixupOffset_ = 0;
  std::shared_ptr<ResidentImage> initialImage_;
  std::map<uint64_t, Variant> variants_;
  uint64_t generation_ = 0;
  bool dirty_ = false;
};
}  // namespace amd::roc::aql_resident
