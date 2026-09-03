// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor_queue_backend.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include <memory>

namespace rocjitsu::amdgpu {
namespace {

class CommandProcessorQueueBackend final : public QueueBackend {
public:
  explicit CommandProcessorQueueBackend(CommandProcessor &command_processor)
      : command_processor_(command_processor) {}

  void attach(const QueueCreateInfo &info) override {
    registration_id_ = command_processor_.register_queue({
        .address_space = info.address_space,
        .interrupt_sink = info.interrupt_sink,
        .process_id = info.process_id,
        .queue_id = info.queue_id,
        .ring_base_va = info.ring_base_va,
        .ring_size = info.ring_size,
        .read_ptr_va = info.read_ptr_va,
        .write_ptr_va = info.write_ptr_va,
        .initial_read_pointer = info.initial_read_pointer,
        .doorbell_offset = info.doorbell_offset,
        .doorbell_base = info.doorbell_base,
        .doorbell_va = info.doorbell_va,
        .last_doorbell = info.last_doorbell,
        .host_accessible = info.host_accessible,
        .is_sdma = info.kind == QueueKind::Sdma,
        .queue_desc_va = info.queue_desc_va,
        .exception_status_va = info.exception_status_va,
        .exception_event_id = info.exception_event_id,
        .xcd_fanout = info.xcd_fanout,
    });
  }

  void rollback_attach(const QueueCreateInfo &) noexcept override {
    // CommandProcessor::register_queue() provides the strong guarantee and
    // removes any fan-out replicas before propagating a failure.
  }

  void update(uint32_t queue_id, uint32_t process_id, uint64_t ring_base_va, uint32_t ring_size,
              uint32_t queue_percentage) override {
    command_processor_.update_queue(queue_id, process_id, ring_base_va, ring_size,
                                    queue_percentage);
  }

  QueueDoorbellDisposition notify_doorbell(uint32_t queue_id, uint32_t process_id,
                                           uint64_t value) override {
    (void)queue_id;
    (void)process_id;
    command_processor_.notify_queue_doorbell(registration_id_, value);
    return QueueDoorbellDisposition::Complete;
  }

  void detach(uint32_t queue_id, uint32_t process_id) noexcept override {
    command_processor_.unregister_queue(queue_id, process_id);
    registration_id_ = 0;
  }

private:
  CommandProcessor &command_processor_;
  uint64_t registration_id_ = 0;
};

} // namespace

std::shared_ptr<QueueBackend>
make_command_processor_queue_backend(CommandProcessor &command_processor) {
  return std::make_shared<CommandProcessorQueueBackend>(command_processor);
}

} // namespace rocjitsu::amdgpu
