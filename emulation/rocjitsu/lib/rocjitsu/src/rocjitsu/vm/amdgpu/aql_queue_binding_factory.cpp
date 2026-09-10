// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/aql_queue_binding_factory.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include <memory>
namespace rocjitsu::amdgpu {
namespace {

class AqlQueueBinding final : public QueueBinding {
public:
  AqlQueueBinding(CommandProcessor &command_processor, uint64_t registration_id)
      : command_processor_(command_processor), registration_id_(registration_id) {}

  ~AqlQueueBinding() override {
    if (registration_id_)
      (void)command_processor_.unregister_queue_registration(registration_id_);
  }

  QueuePrepareCloseStatus prepare_close() noexcept override {
    if (registration_id_ == 0)
      return QueuePrepareCloseStatus::Ready;
    const QueuePrepareCloseStatus status =
        command_processor_.prepare_unregister_queue_registration(registration_id_);
    if (status == QueuePrepareCloseStatus::Ready)
      registration_id_ = 0;
    return status;
  }

  QueueReconfigureStatus reconfigure(const QueueReconfigureRequest &request) override {
    if (!valid_aql_packet_ring(request.ring_base_address, request.ring_size_bytes))
      return QueueReconfigureStatus::Invalid;
    if (!registration_id_ || !command_processor_.update_queue_registration(
                                 registration_id_, request.ring_base_address,
                                 request.ring_size_bytes, request.scheduling_percentage)) {
      return QueueReconfigureStatus::Stale;
    }
    return request.scheduling_percentage == 0 ? QueueReconfigureStatus::Disabled
                                              : QueueReconfigureStatus::Applied;
  }

  QueueSubmissionStatus submit_producer(uint64_t producer_cursor) override {
    if (!registration_id_)
      return QueueSubmissionStatus::Faulted;
    command_processor_.notify_queue_doorbell(registration_id_, producer_cursor);
    return QueueSubmissionStatus::Accepted;
  }

private:
  CommandProcessor &command_processor_;
  uint64_t registration_id_ = 0;
};

class AqlQueueBindingFactory final : public QueueBindingFactory {
public:
  explicit AqlQueueBindingFactory(CommandProcessor &command_processor)
      : command_processor_(command_processor) {}

  QueueBindingCreateResult create_binding(const QueueRegistrationRequest &request) override {
    if (request.type != QueueType::Compute || request.packet_format != QueuePacketFormat::Aql)
      return {};
    if (!valid_aql_queue_layout(request.ring.base_address, request.ring.size_bytes,
                                request.ring.consumer_pointer_address,
                                request.ring.producer_pointer_address)) {
      return {};
    }
    const uint64_t registration_id = command_processor_.register_queue({
        .address_space = request.identity.address_space,
        .interrupt_sink = request.identity.interrupt_sink,
        .process_id = request.identity.process_id,
        .queue_id = request.identity.queue_id,
        .ring_base_va = request.ring.base_address,
        .ring_size = request.ring.size_bytes,
        .read_ptr_va = request.ring.consumer_pointer_address,
        .write_ptr_va = request.ring.producer_pointer_address,
        .doorbell_offset = request.doorbell.offset,
        .doorbell_base = request.doorbell.host_base,
        .doorbell_va = request.doorbell.address,
        .last_doorbell = request.doorbell.last_value,
        .doorbell_mode = request.doorbell.mode,
        .uses_kfd_queue_abi = request.doorbell.mode == QueueDoorbellMode::HostPolled,
        .queue_desc_va = request.queue_descriptor_address,
        .exception_status_va = request.exception_status_address,
        .exception_event_id = request.exception_event_id,
        .xcd_fanout = request.xcd_fanout,
    });
    if (registration_id == 0)
      return {};
    try {
      return {.status = QueueBindingCreateStatus::Bound,
              .binding = std::make_unique<AqlQueueBinding>(command_processor_, registration_id)};
    } catch (...) {
      (void)command_processor_.unregister_queue_registration(registration_id);
      throw;
    }
  }

private:
  CommandProcessor &command_processor_;
};

} // namespace

std::shared_ptr<QueueBindingFactory>
make_aql_queue_binding_factory(CommandProcessor &command_processor) {
  return std::make_shared<AqlQueueBindingFactory>(command_processor);
}

} // namespace rocjitsu::amdgpu
