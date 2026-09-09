// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"

#include <chrono>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

class SdmaQueueBinding final : public QueueBinding {
public:
  SdmaQueueBinding(SdmaQueueScheduler &scheduler, SdmaQueueScheduler::Handle handle)
      : scheduler_(scheduler), handle_(handle) {}

  ~SdmaQueueBinding() override {
    if (handle_)
      (void)scheduler_.detach(handle_);
  }

  QueuePrepareCloseStatus prepare_close() noexcept override {
    if (!handle_)
      return QueuePrepareCloseStatus::Ready;
    const QueuePrepareCloseStatus status = scheduler_.prepare_detach(handle_);
    if (status == QueuePrepareCloseStatus::Ready)
      handle_ = {};
    return status;
  }

  QueueReconfigureStatus reconfigure(const QueueReconfigureRequest &request) override {
    return handle_ ? scheduler_.update(handle_, request.ring_base_address, request.ring_size_bytes,
                                       request.scheduling_percentage)
                   : QueueReconfigureStatus::Stale;
  }

  QueueSubmissionStatus submit_producer(uint64_t producer_cursor) override {
    return handle_ ? scheduler_.notify(handle_, producer_cursor) : QueueSubmissionStatus::Faulted;
  }

private:
  SdmaQueueScheduler &scheduler_;
  SdmaQueueScheduler::Handle handle_;
};

} // namespace

SdmaQueueBindingFactory::SdmaQueueBindingFactory(
    SdmaQueueScheduler &scheduler, SdmaCallbackFactory callback_factory,
    SdmaProgressObserverFactory progress_observer_factory)
    : scheduler_(&scheduler), callback_factory_(std::move(callback_factory)),
      progress_observer_factory_(std::move(progress_observer_factory)) {}

QueueBindingCreateResult
SdmaQueueBindingFactory::create_binding(const QueueRegistrationRequest &request) {
  if (request.type != QueueType::Sdma || request.packet_format != QueuePacketFormat::Sdma)
    return {};
  const SdmaQueueContext context{.process_id = request.identity.process_id,
                                 .queue_id = request.identity.queue_id,
                                 .engine_id = request.engine_id};
  SdmaPacketCallbacks callbacks =
      callback_factory_ ? callback_factory_(context) : SdmaPacketCallbacks{};
  if (!callbacks.deliver_interrupt) {
    callbacks.deliver_interrupt = [interrupt_sink = request.identity.interrupt_sink,
                                   process_id = request.identity.process_id](uint32_t event_id) {
      interrupt_sink.deliver(process_id, event_id);
      return VmAccessOutcome::Complete;
    };
  }
  if (!callbacks.acquire_cache_maintenance) {
    callbacks.acquire_cache_maintenance =
        [coherence = scheduler_->cache_coherence()](SdmaCacheOperation operation) {
          return coherence->acquire_cache_maintenance(operation);
        };
  }
  if (!callbacks.timestamp) {
    callbacks.timestamp = [] {
      const std::chrono::steady_clock::duration now =
          std::chrono::steady_clock::now().time_since_epoch();
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    };
  }
  const SdmaQueueScheduler::Handle handle = scheduler_->attach(
      {.ring = {.address_space = request.identity.address_space,
                .ring_base = request.ring.base_address,
                .ring_bytes = request.ring.size_bytes,
                .read_pointer_address = request.ring.consumer_pointer_address,
                .initial_cursor = request.initial_consumer_cursor},
       .context = context,
       .doorbell_offset = request.doorbell.offset,
       .doorbell_base = request.doorbell.host_base,
       .last_doorbell = request.doorbell.last_value,
       .host_accessible = request.doorbell.mode == QueueDoorbellMode::HostPolled},
      std::move(callbacks),
      progress_observer_factory_ ? progress_observer_factory_(context)
                                 : SdmaQueueProgressObserver{});
  if (!handle)
    return {};
  try {
    return {.status = QueueBindingCreateStatus::Bound,
            .binding = std::make_unique<SdmaQueueBinding>(*scheduler_, handle)};
  } catch (...) {
    (void)scheduler_->detach(handle);
    throw;
  }
}

std::shared_ptr<SdmaQueueBindingFactory>
make_sdma_queue_binding_factory(SdmaQueueScheduler &scheduler,
                                SdmaCallbackFactory callback_factory) {
  return std::make_shared<SdmaQueueBindingFactory>(scheduler, std::move(callback_factory));
}

} // namespace rocjitsu::amdgpu
