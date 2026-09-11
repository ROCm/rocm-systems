// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_graph_program.hpp"
#include "aql_static_scratch.hpp"
#include "rocdevice.hpp"
#include "rocvirtual.hpp"
#include "rocrctx.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace amd::roc::aql_resident {
struct GraphProgram::Launch final : amd::AqlBatchImage {
  std::shared_ptr<GraphProgram> program;
  std::shared_ptr<ResidentImage> image;
  std::unique_ptr<UploadPool::Lease> upload;
};

GraphProgram::GraphProgram(Device& device) : owner_(device), device_(&device) {}

bool GraphProgram::encode(const amd::AlignedVector64<uint8_t>& packets, std::vector<Packet>& output,
                          std::vector<uint32_t>* argumentTargets) const {
  static_assert(offsetof(hsa_kernel_dispatch_packet_t, kernarg_address) == 40);
  std::vector<Packet> encoded;
  encoded.reserve(headers_.size()*2+2);
  for (size_t i = 0; i < headers_.size(); ++i) {
    hsa_kernel_dispatch_packet_t dispatch;
    std::memcpy(&dispatch, packets.data()+i*64, sizeof(dispatch));
    dispatch.header = static_cast<uint16_t>(headers_[i]);
    dispatch.setup = static_cast<uint16_t>(headers_[i] >> 16);
    if (!fitsStaticScratch(dispatch.private_segment_size,
                           (descriptors_[i].properties & (1u << 10)) != 0)) return false;
    const uint16_t next = i+1 < headers_.size() ? static_cast<uint16_t>(headers_[i+1]) : 0;
    dispatch.header = aql_fast::coalesceAgentRelease(dispatch.header, next, i+1 == headers_.size());
    const size_t first = encoded.size();
    if (aql_fast::appendDispatch(dispatch, descriptors_[i], nullptr, 0, encoded) !=
        aql_fast::Result::Encoded) return false;
    if (argumentTargets) {
      uint32_t offset = ArgumentBindingPlan::kNoArgument;
      if (descriptors_[i].properties & (1u << 3)) {
        const uint32_t count = (encoded[first][0] >> 28)+1;
        offset = static_cast<uint32_t>(first*64 + (16-count)*4);
      }
      argumentTargets->push_back(offset);
    }
  }
  encoded.push_back(aql_fast::completionBarrier());
  Packet terminal{}; terminal[0] = 7u << 16;
  encoded.push_back(terminal);
  output.swap(encoded);
  return true;
}

std::shared_ptr<GraphProgram> GraphProgram::create(Device& device,
    const amd::AlignedVector64<uint8_t>& packets, const std::vector<uint32_t>& headers) try {
  const auto& isa = device.isa();
  if (isa.versionMajor() != 12 || isa.versionMinor() != 0 || isa.versionStepping() != 1 ||
      headers.empty() || headers.size() > 32766 || packets.size() != headers.size()*64) return {};
  auto program = std::shared_ptr<GraphProgram>(new GraphProgram(device));
  program->headers_ = headers;
  program->pending_ = packets;
  for (size_t i = 0; i < headers.size(); ++i) {
    if ((headers[i] & 0xff) != HSA_PACKET_TYPE_KERNEL_DISPATCH) return {};
    hsa_kernel_dispatch_packet_t dispatch;
    std::memcpy(&dispatch, packets.data()+i*64, sizeof(dispatch));
    if (device.KernelMap().find(dispatch.kernel_object) == device.KernelMap().end()) return {};
    const void* descriptor = nullptr;
    if (Device::loaderQueryHostAddress(reinterpret_cast<void*>(dispatch.kernel_object), &descriptor)
          != HSA_STATUS_SUCCESS || !descriptor) return {};
    aql_fast::Descriptor copy;
    std::memcpy(&copy, descriptor, sizeof(copy));
    program->descriptors_.push_back(copy);
    program->kernelObjects_.push_back(dispatch.kernel_object);
  }
  std::vector<Packet> encoded;
  std::vector<uint32_t> argumentTargets;
  if (!program->encode(packets, encoded, &argumentTargets) || program->template_.compile(encoded) != PlanStatus::Ok ||
      program->template_.bind(encoded, program->bindings_) != PlanStatus::Ok ||
      !program->argumentBindings_.compile(packets, argumentTargets, program->template_.plan())) return {};
  program->initialBindings_ = program->bindings_;
  program->imageBytes_ = program->template_.image();
  program->argumentFixupOffset_ = program->imageBytes_.size();
  const auto& argumentSlots = program->argumentBindings_.slots();
  program->imageBytes_.resize((program->argumentFixupOffset_ +
                              argumentSlots.size()*sizeof(BindingFixup) + 63) & ~size_t(63));
  for (size_t i = 0; i < argumentSlots.size(); ++i) {
    // PacketTemplate allocates one relocation per raw binding, in slot order.
    const uint32_t slot = argumentSlots[i];
    if (slot >= program->template_.plan().entries().size()) return {};
    auto entry = program->template_.plan().entries()[slot];
    if (entry.bindingSlot != slot) return {};
    entry.bindingSlot = static_cast<uint32_t>(i);
    std::memcpy(program->imageBytes_.data()+program->argumentFixupOffset_+i*sizeof(entry), &entry, sizeof(entry));
  }
  if (ResidentImage::publish(device, program->imageBytes_, program->initialImage_) !=
      HSA_STATUS_SUCCESS) return {};
  if (std::getenv("HIP_AQL_IB_AUDIT")) {
    std::fprintf(stderr, "HIP_RESIDENT_COMPILE kernels=%zu packets=%u image_bytes=%zu fixups=%zu\n",
        headers.size(), program->template_.packetCount(), program->imageBytes_.size(),
        program->template_.plan().entries().size());
  }
  return program;
} catch (const std::bad_alloc&) { return {}; }

bool GraphProgram::update(const amd::AlignedVector64<uint8_t>& packets,
                          const std::vector<uint32_t>& headers) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (headers != headers_ || packets.size() != pending_.size()) return false;
  for (size_t i = 0; i < headers.size(); ++i) {
    hsa_kernel_dispatch_packet_t dispatch;
    std::memcpy(&dispatch, packets.data()+i*64, sizeof(dispatch));
    if (dispatch.kernel_object != kernelObjects_[i] || dispatch.completion_signal.handle) return false;
  }
  if (packets != pending_) {
    pending_ = packets;
    dirty_ = true;
  }
  return true;
}

SubmitResult GraphProgram::submit(VirtualGPU& gpu, bool blocking) try {
  std::lock_guard<std::mutex> lock(mutex_);
  if (&gpu.dev() != device_ || !gpu.gpu_queue()) return SubmitResult::Unsupported;
  if (dirty_) {
    bool changed = false;
    const bool projected = argumentBindings_.bind(pending_, bindings_, changed);
    if (!projected) {
      std::vector<Packet> encoded;
      std::vector<uint64_t> bindings;
      if (!encode(pending_, encoded) || template_.bind(encoded, bindings) != PlanStatus::Ok ||
          !argumentBindings_.rebaseValidated(pending_)) {
        return SubmitResult::Unsupported;
      }
      if (bindings != bindings_) { bindings_.swap(bindings); changed = true; }
    }
    if (changed) ++generation_;
    dirty_ = false;
    if (std::getenv("HIP_AQL_IB_AUDIT")) {
      std::fprintf(stderr, "HIP_RESIDENT_BIND direct=%d changed=%d\n", projected, changed);
    }
  }
  // ROCr allocates monotonically increasing HSA queue IDs; pointer reuse does
  // not alias a previous queue generation. One image per actual physical queue.
  auto& variant = variants_[gpu.gpu_queue()->id];
  if (!variant.image) {
    if (initialImage_) variant.image = std::move(initialImage_);
    else if (ResidentImage::publish(*device_, imageBytes_, variant.image) != HSA_STATUS_SUCCESS) {
      return SubmitResult::Unsupported;
    }
  }
  auto launch = std::make_shared<Launch>();
  launch->program = shared_from_this();
  launch->image = variant.image;
  hsa_kernel_dispatch_packet_t fixup{};
  BindingUpdate update;
  const bool needsGeneration = variant.generation != generation_;
  if (needsGeneration) {
    const auto& previous = variant.appliedBindings.empty() ? initialBindings_ : variant.appliedBindings;
    if (update.prepare(template_.plan(), previous, bindings_, argumentBindings_.slots()) != PlanStatus::Ok) {
      return SubmitResult::Unsupported;
    }
    // Allocate before publishing any packet. Updating host bookkeeping after
    // submission must not throw and accidentally request ordinary fallback.
    variant.stagedBindings = bindings_;
  }
  const bool needsFixup = update.count() != 0;
  if (needsFixup) {
    if (!variant.kernel || variant.kernelQueue != &gpu) {
      variant.kernel = BindingKernel::create(gpu);
      variant.kernelQueue = &gpu;
      if (!variant.kernel) return SubmitResult::Unsupported;
    }
    if (!variant.uploads) variant.uploads = std::make_shared<UploadPool>(*device_);
    const size_t uploadBytes = update.bytes();
    launch->upload = variant.uploads->acquire(uploadBytes);
    if (!launch->upload || Hsa::memory_copy(launch->upload->base(), update.data(bindings_), uploadBytes)
          != HSA_STATUS_SUCCESS) return SubmitResult::Unsupported;
    const uintptr_t entries = update.sparse()
        ? reinterpret_cast<uintptr_t>(launch->upload->base()) + update.entriesOffset()
        : reinterpret_cast<uintptr_t>(variant.image->base()) +
            (update.arguments() ? argumentFixupOffset_ : template_.fixupOffset());
    if (!variant.kernel->prepare(gpu, reinterpret_cast<uintptr_t>(launch->upload->base()),
        entries, *variant.image, update.count(), fixup)) return SubmitResult::Unsupported;
  }
  const auto result = QueueSubmitter::submit(gpu, *variant.image, template_.packetCount(),
                                            needsFixup ? &fixup : nullptr, launch, blocking);
  if (result != SubmitResult::Unsupported) {
    variant.generation = generation_;
    if (needsGeneration) variant.appliedBindings.swap(variant.stagedBindings);
  }
  if (std::getenv("HIP_AQL_IB_AUDIT")) {
    std::fprintf(stderr, "HIP_RESIDENT_SUBMIT queue=%lu generation=%lu fixup=%d result=%d\n",
        gpu.gpu_queue()->id, generation_, needsFixup, static_cast<int>(result));
    std::fprintf(stderr, "HIP_RESIDENT_FIXUP sparse=%d arguments=%d entries=%u upload_bytes=%zu\n",
                 update.sparse(), update.arguments(), update.count(), update.bytes());
  }
  return result;
} catch (const std::bad_alloc&) { return SubmitResult::Unsupported; }
}  // namespace amd::roc::aql_resident
