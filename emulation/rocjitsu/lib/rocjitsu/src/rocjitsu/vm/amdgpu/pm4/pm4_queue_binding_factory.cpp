// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4/pm4_queue_binding_factory.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4_queue_controller.h"

#include <memory>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

class KfdPm4ComputeQueueBinding final : public QueueBinding {
public:
  KfdPm4ComputeQueueBinding(CommandProcessor &command_processor, uint32_t queue_id,
                            uint32_t process_id)
      : command_processor_(command_processor), queue_id_(queue_id), process_id_(process_id) {}
  ~KfdPm4ComputeQueueBinding() override {
    command_processor_.unregister_drm_queue(queue_id_, process_id_);
  }
  QueuePrepareCloseStatus prepare_close() noexcept override {
    return command_processor_.prepare_pm4_compute_queue(queue_id_, process_id_);
  }
  QueueReconfigureStatus reconfigure(const QueueReconfigureRequest &request) override {
    return command_processor_.update_pm4_compute_queue(queue_id_, process_id_, request);
  }
  QueueSubmissionStatus submit_producer(uint64_t) override {
    // Native KFD queues publish through their mapped host doorbell.
    return QueueSubmissionStatus::Faulted;
  }

private:
  CommandProcessor &command_processor_;
  uint32_t queue_id_ = 0;
  uint32_t process_id_ = 0;
};

class Pm4QueueBinding final : public QueueBinding {
public:
  Pm4QueueBinding(CommandProcessor &command_processor, uint64_t registration_id)
      : command_processor_(command_processor), registration_id_(registration_id) {}

  ~Pm4QueueBinding() override {
    if (registration_id_ != 0)
      (void)command_processor_.unregister_pm4_queue_registration(registration_id_);
  }

  QueuePrepareCloseStatus prepare_close() noexcept override {
    if (registration_id_ == 0)
      return QueuePrepareCloseStatus::Ready;
    const QueuePrepareCloseStatus status =
        command_processor_.prepare_unregister_pm4_queue_registration(registration_id_);
    if (status != QueuePrepareCloseStatus::Ready)
      return status;
    registration_id_ = 0;
    return QueuePrepareCloseStatus::Ready;
  }

  QueueReconfigureStatus reconfigure(const QueueReconfigureRequest &request) override {
    return registration_id_ != 0
               ? command_processor_.update_pm4_queue_registration(registration_id_, request)
               : QueueReconfigureStatus::Stale;
  }

  QueueSubmissionStatus submit_producer(uint64_t producer_cursor) override {
    return registration_id_ != 0
               ? command_processor_.notify_pm4_queue_doorbell(registration_id_, producer_cursor)
               : QueueSubmissionStatus::Faulted;
  }

private:
  CommandProcessor &command_processor_;
  uint64_t registration_id_ = 0;
};

// KFD's native ring uses the full compute executor; MES transport queues below
// retain their synchronous packet callbacks and explicit producer notifications.
class KfdPm4ComputeQueueBindingFactory final : public QueueBindingFactory {
public:
  explicit KfdPm4ComputeQueueBindingFactory(CommandProcessor &command_processor)
      : command_processor_(command_processor) {}

  QueueBindingCreateResult create_binding(const QueueRegistrationRequest &request) override {
    if (request.type != QueueType::Compute || request.packet_format != QueuePacketFormat::Pm4 ||
        request.doorbell.mode != QueueDoorbellMode::HostPolled)
      return {};
    if (!command_processor_.register_pm4_compute_queue(request))
      return {};
    try {
      return {.status = QueueBindingCreateStatus::Bound,
              .binding = std::make_unique<KfdPm4ComputeQueueBinding>(
                  command_processor_, request.identity.queue_id, request.identity.process_id)};
    } catch (...) {
      command_processor_.unregister_drm_queue(request.identity.queue_id,
                                              request.identity.process_id);
      throw;
    }
  }

private:
  CommandProcessor &command_processor_;
};

class Pm4QueueBindingFactory final : public QueueBindingFactory {
public:
  Pm4QueueBindingFactory(CommandProcessor &command_processor, Pm4PacketCallbacks packet_callbacks)
      : command_processor_(command_processor), packet_callbacks_(std::move(packet_callbacks)) {}

  QueueBindingCreateResult create_binding(const QueueRegistrationRequest &request) override {
    if (request.type != QueueType::Compute || request.packet_format != QueuePacketFormat::Pm4)
      return {};
    if (request.doorbell.mode != QueueDoorbellMode::Explicit)
      return {};
    const uint64_t registration_id = command_processor_.register_pm4_queue({
        .address_space = request.identity.address_space,
        .ring_base = request.ring.base_address,
        .ring_size_bytes = request.ring.size_bytes,
        .consumer_pointer_address = request.ring.consumer_pointer_address,
        .initial_consumer_cursor = request.initial_consumer_cursor,
        .packet_callbacks = packet_callbacks_,
    });
    if (registration_id == 0)
      return {};
    try {
      return {.status = QueueBindingCreateStatus::Bound,
              .binding = std::make_unique<Pm4QueueBinding>(command_processor_, registration_id)};
    } catch (...) {
      (void)command_processor_.unregister_pm4_queue_registration(registration_id);
      throw;
    }
  }

private:
  CommandProcessor &command_processor_;
  Pm4PacketCallbacks packet_callbacks_;
};

} // namespace

std::shared_ptr<QueueBindingFactory>
make_pm4_queue_binding_factory(CommandProcessor &command_processor,
                               Pm4PacketCallbacks packet_callbacks) {
  return std::make_shared<Pm4QueueBindingFactory>(command_processor, std::move(packet_callbacks));
}

std::shared_ptr<QueueBindingFactory>
make_kfd_pm4_compute_queue_binding_factory(CommandProcessor &command_processor) {
  return std::make_shared<KfdPm4ComputeQueueBindingFactory>(command_processor);
}

} // namespace rocjitsu::amdgpu
