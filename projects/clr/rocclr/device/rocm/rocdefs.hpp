/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
namespace amd::roc {

//! Alignment restriction for the pinned memory
static constexpr size_t PinnedMemoryAlignment = 4 * Ki;

//! Specific defines for images for Dynamic Parallelism
static constexpr uint DeviceQueueMaskSize = 32;

//! Set to match the number of pipes, which is 8.
static constexpr uint kMaxAsyncQueues = 8;

constexpr bool kSkipCpuWait = true;

enum HwQueueEngine : uint32_t {
  Compute = 0,
  SdmaD2H = 1,
  SdmaH2D = 2,
  SdmaD2D = 3,
  SdmaP2P = 4,
  //! ABCE submission. Ordering is only implied when HwQueueTracker also sees
  //! the same exact coordinator ring as the previous ABCE submission.
  SdmaAbce = 5,
  Unknown = 6
};

//! Returns true if the engine is an SDMA engine (any type)
inline bool IsSdmaEngine(HwQueueEngine engine) {
  return engine >= HwQueueEngine::SdmaD2H && engine <= HwQueueEngine::SdmaAbce;
}

inline const char* EngineOpName(HwQueueEngine engine) {
  switch (engine) {
    case HwQueueEngine::SdmaD2H: return "D2H";
    case HwQueueEngine::SdmaH2D: return "H2D";
    case HwQueueEngine::SdmaD2D:
    case HwQueueEngine::SdmaP2P: return "D2D/P2P";
    case HwQueueEngine::SdmaAbce: return "ABCE";
    case HwQueueEngine::Compute: return "Compute";
    default:                     return "Unknown";
  }
}

}  // namespace amd::roc
