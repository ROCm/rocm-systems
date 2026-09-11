// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "device/device.hpp"
#include <hsa/hsa.h>

namespace amd::roc {
class VirtualGPU;
namespace aql_resident {
class ResidentImage;
enum class SubmitResult { Unsupported, Submitted, Failed };

// Narrow queue adapter. The caller holds VirtualGPU::execution(), finishes all
// allocation/validation first and retains the image, upload, executable and
// argument snapshots in resources. No fallback is permitted after Submitted
// or Failed; Unsupported is returned only before queue publication.
class QueueSubmitter {
 public:
  static SubmitResult submit(VirtualGPU& gpu, const ResidentImage& image,
                             uint32_t packetCount,
                             const hsa_kernel_dispatch_packet_t* preparedFixup,
                             const std::shared_ptr<amd::AqlBatchImage>& resources,
                             bool blocking);
};
}  // namespace aql_resident
}  // namespace amd::roc
