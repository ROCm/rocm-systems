// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_queue_binding_factory.h
/// @brief Queue binding factory for the shared SDMA scheduler.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/sdma_packet_processor.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"

#include <cstdint>
#include <functional>
#include <memory>

namespace rocjitsu::amdgpu {

using SdmaCallbackFactory = std::function<SdmaPacketCallbacks(const SdmaQueueContext &context)>;
using SdmaProgressObserverFactory =
    std::function<SdmaQueueProgressObserver(const SdmaQueueContext &context)>;

/// @brief Reusable binding factory for queues owned by one SoC SDMA scheduler.
///
/// @details The binding factory owns neither the scheduler nor callback targets. Its
/// owner must outlive every queue binding it creates.
class SdmaQueueBindingFactory final : public QueueBindingFactory {
public:
  /// @brief Create a queue binding for one SoC-owned scheduler.
  SdmaQueueBindingFactory(SdmaQueueScheduler &scheduler, SdmaCallbackFactory callback_factory = {},
                          SdmaProgressObserverFactory progress_observer_factory = {});

  [[nodiscard]] QueueBindingCreateResult
  create_binding(const QueueRegistrationRequest &request) override;

private:
  SdmaQueueScheduler *scheduler_ = nullptr;
  SdmaCallbackFactory callback_factory_;
  SdmaProgressObserverFactory progress_observer_factory_;
};

/// @brief Create a reusable frontend-neutral SDMA queue binding factory.
[[nodiscard]] std::shared_ptr<SdmaQueueBindingFactory>
make_sdma_queue_binding_factory(SdmaQueueScheduler &scheduler,
                                SdmaCallbackFactory callback_factory = {});

} // namespace rocjitsu::amdgpu
