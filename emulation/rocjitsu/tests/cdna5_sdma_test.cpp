// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/pm4_queue_controller.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

constexpr uint32_t kSdmaOpCopy = 1;
constexpr uint32_t kSdmaOpFence = 5;
constexpr uint32_t kSdmaOpPollRegmem = 8;
constexpr uint32_t kSdmaOpConstFill = 11;
constexpr uint32_t kSdmaOpWrite = 2;
constexpr uint32_t kSdmaOpAtomic = 10;
constexpr uint32_t kSdmaOpTimestamp = 13;
constexpr uint32_t kSdmaOpGcr = 17;
constexpr uint32_t kSdmaSubopCopyLinear = 0;
constexpr uint32_t kSdmaSubopFence64 = 2;
constexpr uint32_t kSdmaSubopPollMem64 = 5;

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds timeout = std::chrono::seconds(1)) {
  const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return true;
}

enum class SdmaSubmissionWait {
  RetirementOrTerminal,
  None,
};

class TransientAqlAddressSpace final : public amdgpu::AddressSpaceTranslator,
                                       public amdgpu::PhysicalMemoryAccess {
public:
  amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                        amdgpu::VmAccessKind) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = amdgpu::VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = bytes_.size() - address,
                        .mtype = amdgpu::Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain, uint64_t address,
                               std::span<std::byte> bytes) override {
    if (address == tracked_read_address_)
      ++tracked_read_attempts_;
    if (next_read_outcome_ && address == next_read_outcome_->address) {
      const amdgpu::VmAccessOutcome outcome = next_read_outcome_->outcome;
      next_read_outcome_.reset();
      return outcome;
    }
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return amdgpu::VmAccessOutcome::Faulted;
    std::copy_n(bytes_.begin() + static_cast<ptrdiff_t>(address), bytes.size(), bytes.begin());
    run_hook(read_hook_, address);
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain, uint64_t address,
                                std::span<const std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return amdgpu::VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<ptrdiff_t>(address));
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::AtomicLoadResult atomic_load(amdgpu::VmMemoryDomain, uint64_t address,
                                       uint32_t width) override {
    atomic_load_addresses_.push_back(address);
    if (next_atomic_load_outcome_ && address == next_atomic_load_outcome_->address) {
      const amdgpu::VmAccessOutcome outcome = next_atomic_load_outcome_->outcome;
      next_atomic_load_outcome_.reset();
      return {.outcome = outcome};
    }
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return {.outcome = amdgpu::VmAccessOutcome::Faulted};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    run_hook(atomic_load_hook_, address);
    return {.outcome = amdgpu::VmAccessOutcome::Complete, .value = value};
  }

  amdgpu::VmAccessOutcome atomic_store(amdgpu::VmMemoryDomain, uint64_t address, uint32_t width,
                                       uint64_t value) override {
    if (address == tracked_atomic_store_address_)
      ++tracked_atomic_store_attempts_;
    if (fail_atomic_store_at_ && address == *fail_atomic_store_at_) {
      fail_atomic_store_at_.reset();
      run_hook(atomic_store_hook_, address);
      return amdgpu::VmAccessOutcome::Unavailable;
    }
    if ((width != 4 && width != 8) || address > bytes_.size() || width > bytes_.size() - address)
      return amdgpu::VmAccessOutcome::Faulted;
    std::memcpy(bytes_.data() + address, &value, width);
    run_hook(atomic_store_hook_, address);
    return amdgpu::VmAccessOutcome::Complete;
  }

  template <typename T> void store(uint64_t address, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  uint64_t load_u64(uint64_t address) const {
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  void fail_next_read_at(uint64_t address) {
    return_next_read_at(address, amdgpu::VmAccessOutcome::Unavailable);
  }
  void return_next_read_at(uint64_t address, amdgpu::VmAccessOutcome outcome) {
    tracked_read_address_ = address;
    next_read_outcome_ = ReadFailure{.address = address, .outcome = outcome};
  }
  void fail_next_atomic_store_at(uint64_t address) {
    tracked_atomic_store_address_ = address;
    fail_atomic_store_at_ = address;
  }
  void return_next_atomic_load_at(uint64_t address, amdgpu::VmAccessOutcome outcome) {
    next_atomic_load_outcome_ = AtomicLoadFailure{.address = address, .outcome = outcome};
  }
  void run_after_next_read_at(uint64_t address, std::function<void()> callback) {
    read_hook_ = AccessHook{.address = address, .callback = std::move(callback)};
  }
  void run_after_next_atomic_load_at(uint64_t address, std::function<void()> callback) {
    atomic_load_hook_ = AccessHook{.address = address, .callback = std::move(callback)};
  }
  void run_after_next_atomic_store_at(uint64_t address, std::function<void()> callback) {
    atomic_store_hook_ = AccessHook{.address = address, .callback = std::move(callback)};
  }
  uint32_t atomic_load_attempts_at(uint64_t address) const {
    return static_cast<uint32_t>(
        std::count(atomic_load_addresses_.begin(), atomic_load_addresses_.end(), address));
  }
  uint32_t tracked_read_attempts() const { return tracked_read_attempts_; }
  uint32_t tracked_atomic_store_attempts() const { return tracked_atomic_store_attempts_; }

private:
  struct ReadFailure {
    uint64_t address;
    amdgpu::VmAccessOutcome outcome;
  };

  struct AtomicLoadFailure {
    uint64_t address;
    amdgpu::VmAccessOutcome outcome;
  };

  struct AccessHook {
    uint64_t address;
    std::function<void()> callback;
  };

  static void run_hook(std::optional<AccessHook> &hook, uint64_t address) {
    if (!hook || hook->address != address)
      return;
    std::function<void()> callback = std::move(hook->callback);
    hook.reset();
    callback();
  }

  std::array<std::byte, 4096> bytes_{};
  std::optional<ReadFailure> next_read_outcome_;
  std::optional<uint64_t> fail_atomic_store_at_;
  std::optional<AtomicLoadFailure> next_atomic_load_outcome_;
  std::optional<AccessHook> read_hook_;
  std::optional<AccessHook> atomic_load_hook_;
  std::optional<AccessHook> atomic_store_hook_;
  std::vector<uint64_t> atomic_load_addresses_;
  uint64_t tracked_read_address_ = std::numeric_limits<uint64_t>::max();
  uint64_t tracked_atomic_store_address_ = std::numeric_limits<uint64_t>::max();
  uint32_t tracked_read_attempts_ = 0;
  uint32_t tracked_atomic_store_attempts_ = 0;
};

class TransientAqlQueueForTest {
public:
  explicit TransientAqlQueueForTest(Gfx1250Sim &sim, amdgpu::QueueDoorbellMode doorbell_mode =
                                                         amdgpu::QueueDoorbellMode::VmPolled)
      : sim_(sim) {
    hsa_barrier_and_packet_t packet{};
    packet.header = HSA_PACKET_TYPE_BARRIER_AND;
    backing_ = std::make_shared<TransientAqlAddressSpace>();
    backing_->store(kRingVa, packet);
    backing_->store(kRingVa + sizeof(packet), packet);
    backing_->store(kReadPointerVa, uint64_t{0});
    backing_->store(kWritePointerVa, uint64_t{1});
    backing_->store(kDoorbellVa, uint64_t{1});
    address_space_ = sim_.soc->gpu_vm().register_translated(kProcessId, backing_, backing_);
    if (!address_space_)
      throw std::runtime_error("cannot register transient AQL test address space");

    amdgpu::AqlQueueConfig queue{};
    queue.address_space = address_space_;
    queue.process_id = kProcessId;
    queue.queue_id = kQueueId;
    queue.ring_base_va = kRingVa;
    queue.ring_size = 2 * sizeof(packet);
    queue.read_ptr_va = kReadPointerVa;
    queue.write_ptr_va = kWritePointerVa;
    queue.doorbell_va = doorbell_mode == amdgpu::QueueDoorbellMode::VmPolled ? kDoorbellVa : 0;
    queue.doorbell_mode = doorbell_mode;
    queue.uses_kfd_queue_abi = false;
    registration_id_ = sim_.cp()->register_queue(std::move(queue));
    if (registration_id_ == 0)
      throw std::runtime_error("cannot register transient AQL test queue");
  }

  ~TransientAqlQueueForTest() {
    sim_.cp()->unregister_queue(kQueueId, kProcessId);
    (void)sim_.soc->gpu_vm().unregister_address_space(address_space_);
  }

  void fail_next_packet_read() { backing_->fail_next_read_at(kRingVa); }
  void fail_next_read_pointer_store() { backing_->fail_next_atomic_store_at(kReadPointerVa); }
  void fail_next_write_pointer_load(amdgpu::VmAccessOutcome outcome) {
    backing_->return_next_atomic_load_at(kWritePointerVa, outcome);
  }
  void fail_next_dependency_value_load(amdgpu::VmAccessOutcome outcome) {
    backing_->return_next_atomic_load_at(kSignalValueVa, outcome);
  }
  void fail_next_kernel_descriptor_read(amdgpu::VmAccessOutcome outcome) {
    backing_->return_next_read_at(kKernelObjectVa, outcome);
  }
  std::shared_ptr<TransientAqlAddressSpace> make_invalid_replacement() const {
    auto replacement = std::make_shared<TransientAqlAddressSpace>();
    hsa_kernel_dispatch_packet_t invalid{};
    invalid.header = HSA_PACKET_TYPE_INVALID;
    replacement->store(kRingVa, invalid);
    replacement->store(kReadPointerVa, uint64_t{0});
    replacement->store(kWritePointerVa, uint64_t{1});
    replacement->store(kDoorbellVa, uint64_t{1});
    return replacement;
  }
  void set_replacement_dependency(const std::shared_ptr<TransientAqlAddressSpace> &replacement,
                                  int64_t value) {
    replacement->store(kSignalValueVa, value);
  }
  void fail_replacement_kernel_descriptor_read(
      const std::shared_ptr<TransientAqlAddressSpace> &replacement,
      amdgpu::VmAccessOutcome outcome) {
    replacement->return_next_read_at(kKernelObjectVa, outcome);
  }
  void replace_after_next_write_pointer_load(
      const std::shared_ptr<TransientAqlAddressSpace> &replacement) {
    backing_->run_after_next_atomic_load_at(kWritePointerVa, replacement_callback(replacement));
  }
  void
  replace_after_next_packet_read(const std::shared_ptr<TransientAqlAddressSpace> &replacement) {
    backing_->run_after_next_read_at(kRingVa, replacement_callback(replacement));
  }
  void
  replace_after_next_dependency_load(const std::shared_ptr<TransientAqlAddressSpace> &replacement) {
    backing_->run_after_next_atomic_load_at(kSignalValueVa, replacement_callback(replacement));
  }
  uint64_t read_pointer_on(const std::shared_ptr<TransientAqlAddressSpace> &backing) const {
    return backing->load_u64(kReadPointerVa);
  }
  void set_barrier_dependency(int64_t value, uint64_t completion_signal = 0) {
    hsa_barrier_and_packet_t packet{};
    packet.header = HSA_PACKET_TYPE_BARRIER_AND;
    packet.dep_signal[0].handle = kSignalVa;
    packet.completion_signal.handle = completion_signal;
    backing_->store(kRingVa, packet);
    backing_->store(kSignalValueVa, value);
  }
  void set_kernel_dispatch() {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t descriptor{};
    descriptor.kernel_code_entry_byte_offset = sizeof(descriptor);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    1);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    12);
    backing_->store(kKernelObjectVa, descriptor);
    backing_->store(kKernelObjectVa + sizeof(descriptor), S_ENDPGM_GFX12);

    hsa_kernel_dispatch_packet_t packet{};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    packet.setup = 1;
    packet.workgroup_size_x = 1;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.grid_size_x = 1;
    packet.grid_size_y = 1;
    packet.grid_size_z = 1;
    packet.kernel_object = kKernelObjectVa;
    backing_->store(kRingVa, packet);
  }
  uint64_t read_pointer() const { return backing_->load_u64(kReadPointerVa); }
  void set_write_pointer(uint64_t value) { backing_->store(kWritePointerVa, value); }
  void notify(uint64_t producer_cursor) {
    sim_.cp()->notify_queue_doorbell(registration_id_, producer_cursor);
  }
  uint32_t write_pointer_load_attempts() const {
    return backing_->atomic_load_attempts_at(kWritePointerVa);
  }
  uint32_t dependency_value_load_attempts() const {
    return backing_->atomic_load_attempts_at(kSignalValueVa);
  }
  uint32_t dependency_handle_load_attempts() const {
    return backing_->atomic_load_attempts_at(kRingVa +
                                             offsetof(hsa_barrier_and_packet_t, dep_signal));
  }
  uint32_t completion_handle_load_attempts() const {
    return backing_->atomic_load_attempts_at(kRingVa +
                                             offsetof(hsa_barrier_and_packet_t, completion_signal));
  }
  uint32_t kernel_descriptor_read_attempts() const { return backing_->tracked_read_attempts(); }
  uint32_t packet_read_attempts() const { return backing_->tracked_read_attempts(); }
  uint32_t read_pointer_store_attempts() const { return backing_->tracked_atomic_store_attempts(); }
  std::size_t accepted_entries() const {
    return sim_.cp()->accepted_entry_count_for_test(kQueueId, kProcessId);
  }
  bool faulted() const { return sim_.cp()->queue_faulted_for_test(kQueueId, kProcessId); }

  void step() {
    sim_.engine->schedule_event_now(sim_.cp()->doorbell_event());
    if (!sim_.engine->step())
      throw std::runtime_error("transient AQL test did not execute a doorbell event");
  }

private:
  std::function<void()>
  replacement_callback(std::shared_ptr<TransientAqlAddressSpace> replacement) {
    return [this, replacement = std::move(replacement)] {
      if (!sim_.soc->gpu_vm().replace_translated(address_space_, replacement, replacement))
        throw std::runtime_error("cannot replace transient AQL test address space");
    };
  }

  static constexpr uint32_t kProcessId = 1252;
  static constexpr uint32_t kQueueId = 1252;
  static constexpr uint64_t kReadPointerVa = 0x80;
  static constexpr uint64_t kWritePointerVa = 0x88;
  static constexpr uint64_t kDoorbellVa = 0x90;
  static constexpr uint64_t kRingVa = 0x100;
  static constexpr uint64_t kSignalVa = 0x200;
  static constexpr uint64_t kSignalValueVa = kSignalVa + 8;
  static constexpr uint64_t kKernelObjectVa = 0x300;

  Gfx1250Sim &sim_;
  std::shared_ptr<TransientAqlAddressSpace> backing_;
  amdgpu::AddressSpaceHandle address_space_;
  uint64_t registration_id_ = 0;
};

class HostSdmaQueueForTest {
public:
  explicit HostSdmaQueueForTest(Gfx1250Sim &sim, uint64_t initial_doorbell = 0,
                                uint64_t last_doorbell = 0)
      : sim_(sim), legacy_vm_(sim.soc->gpu_vm(), sim.soc->memory()), process_(kProcessId),
        doorbells_{initial_doorbell} {
    address_space_ = legacy_vm_.register_address_space(kProcessId, &process_.page_table_,
                                                       &process_.page_table_mutex_,
                                                       process_.page_table_generation());
    if (!address_space_)
      throw std::runtime_error("cannot register the SDMA test address space");
    if (!legacy_vm_.set_passthrough(address_space_, true))
      throw std::runtime_error("cannot enable SDMA test passthrough");

    queue_handle_ = sim_.soc->queue_registry().register_queue({
        .identity = {.address_space = address_space_,
                     .process_id = kProcessId,
                     .queue_id = kQueueId},
        .ring = {.base_address = reinterpret_cast<uint64_t>(ring_.data()),
                 .size_bytes = static_cast<uint32_t>(ring_.size() * sizeof(uint32_t)),
                 .consumer_pointer_address = reinterpret_cast<uint64_t>(&read_idx_),
                 .producer_pointer_address = reinterpret_cast<uint64_t>(&write_idx_)},
        .doorbell = {.mode = amdgpu::QueueDoorbellMode::HostPolled,
                     .offset = 0,
                     .host_base = doorbells_.data(),
                     .last_value = last_doorbell},
        .binding_factory = std::make_shared<amdgpu::SdmaQueueBindingFactory>(
            sim_.soc->sdma_queue_scheduler(), amdgpu::SdmaCallbackFactory{},
            [this](const amdgpu::SdmaQueueContext &) {
              return [this](const amdgpu::SdmaQueueProgress &progress) {
                const std::lock_guard lock(progress_mutex_);
                progress_cursor_ = progress.consumer_cursor;
                terminal_ = progress.terminal;
                progress_condition_.notify_all();
              };
            }),
        .initial_consumer_cursor = read_idx_,
        .type = amdgpu::QueueType::Sdma,
        .packet_format = amdgpu::QueuePacketFormat::Sdma,
    });
    if (!queue_handle_)
      throw std::runtime_error("cannot register the SDMA test queue");
  }

  ~HostSdmaQueueForTest() {
    (void)sim_.soc->queue_registry().unregister_queue(queue_handle_,
                                                      amdgpu::QueueCloseMode::ForceCancel);
    (void)legacy_vm_.unregister_address_space(address_space_);
  }

  uint32_t *ring() { return ring_.data(); }

  void submit(uint32_t dwords, SdmaSubmissionWait wait = SdmaSubmissionWait::RetirementOrTerminal) {
    const uint64_t write_index = static_cast<uint64_t>(dwords) * sizeof(uint32_t);
    std::atomic_ref<uint64_t>(write_idx_).store(write_index, std::memory_order_release);
    std::atomic_ref<uint64_t>(doorbells_[0]).store(write_index, std::memory_order_release);
    last_submission_status_ =
        sim_.soc->queue_registry().submit_producer(queue_handle_, write_index).status;
    if (last_submission_status_ == amdgpu::QueueSubmissionStatus::Accepted &&
        wait == SdmaSubmissionWait::RetirementOrTerminal) {
      std::unique_lock lock(progress_mutex_);
      const bool completed = progress_condition_.wait_for(lock, std::chrono::seconds(1), [&]() {
        return terminal_ || progress_cursor_ >= write_index;
      });
      if (!completed) {
        ADD_FAILURE() << "timed out waiting for the host SDMA queue to retire or terminate";
        return;
      }
      if (terminal_)
        last_submission_status_ = amdgpu::QueueSubmissionStatus::Faulted;
    }
  }

  uint64_t read_idx() const {
    return std::atomic_ref<const uint64_t>(read_idx_).load(std::memory_order_acquire);
  }

  amdgpu::QueueSubmissionStatus last_submission_status() const { return last_submission_status_; }

private:
  // These buffers are ordinary pointers in this process. Keep the legacy
  // routing VMID at zero while still requiring a real, nonzero GpuVm handle;
  // nonzero VMIDs are reserved for fixtures with an explicit KFD page table.
  static constexpr uint32_t kProcessId = 0;
  static constexpr uint32_t kQueueId = 1250;

  Gfx1250Sim &sim_;
  amdgpu::LegacyGpuVmAdapter legacy_vm_;
  KfdProcess process_;
  amdgpu::AddressSpaceHandle address_space_;
  amdgpu::QueueHandle queue_handle_;
  amdgpu::QueueSubmissionStatus last_submission_status_ = amdgpu::QueueSubmissionStatus::Accepted;
  mutable std::mutex progress_mutex_;
  std::condition_variable progress_condition_;
  uint64_t progress_cursor_ = 0;
  bool terminal_ = false;
  std::array<uint32_t, 64> ring_{};
  alignas(8) uint64_t read_idx_ = 0;
  alignas(8) uint64_t write_idx_ = 0;
  std::array<uint64_t, 1> doorbells_{};
};

class TranslatedSdmaQueueForTest {
public:
  explicit TranslatedSdmaQueueForTest(Gfx1250Sim &sim, amdgpu::InterruptSink interrupt_sink = {})
      : sim_(sim), legacy_vm_(sim.soc->gpu_vm(), sim.soc->memory()), process_(kProcessId),
        interrupt_sink_(std::move(interrupt_sink)) {
    address_space_ = legacy_vm_.register_address_space(kProcessId, &process_.page_table_,
                                                       &process_.page_table_mutex_,
                                                       process_.page_table_generation());
    if (!address_space_)
      throw std::runtime_error("cannot register the translated SDMA test address space");
    process_.map_pages(kRingVa, ring_.data(), ring_.size() * sizeof(ring_[0]));
    process_.map_pages(kQueueStateVa, queue_state_.data(),
                       queue_state_.size() * sizeof(queue_state_[0]));
    process_.map_pages(kSrcVa, src_.data(), src_.size());
    process_.map_pages(kDstVa, dst_.data(), dst_.size());
    process_.map_pages(kDst2Va, dst2_.data(), dst2_.size());
    process_.map_pages(kSignalVa, signal_.data(), signal_.size() * sizeof(signal_[0]));
    process_.map_pages(kPollVa, poll_.data(), poll_.size() * sizeof(poll_[0]));

    register_queue(kQueueStateVa);
  }

  void register_queue(uint64_t read_ptr_va) {
    queue_handle_ = sim_.soc->queue_registry().register_queue({
        .identity = {.address_space = address_space_,
                     .interrupt_sink = interrupt_sink_,
                     .process_id = kProcessId,
                     .queue_id = kQueueId},
        .ring = {.base_address = kRingVa,
                 .size_bytes = static_cast<uint32_t>(ring_.size() * sizeof(ring_[0])),
                 .consumer_pointer_address = read_ptr_va,
                 .producer_pointer_address = kQueueStateVa + sizeof(queue_state_[0])},
        .doorbell = {.mode = amdgpu::QueueDoorbellMode::HostPolled,
                     .offset = 0,
                     .host_base = doorbells_.data()},
        .binding_factory = std::make_shared<amdgpu::SdmaQueueBindingFactory>(
            sim_.soc->sdma_queue_scheduler(), amdgpu::SdmaCallbackFactory{},
            [this](const amdgpu::SdmaQueueContext &) {
              return [this](const amdgpu::SdmaQueueProgress &progress) {
                const std::lock_guard lock(progress_mutex_);
                progress_cursor_ = progress.consumer_cursor;
                terminal_ = progress.terminal;
                progress_condition_.notify_all();
              };
            }),
        .initial_consumer_cursor =
            std::atomic_ref<uint64_t>(queue_state_[0]).load(std::memory_order_acquire),
        .type = amdgpu::QueueType::Sdma,
        .packet_format = amdgpu::QueuePacketFormat::Sdma,
    });
    if (!queue_handle_)
      throw std::runtime_error("cannot register the translated SDMA test queue");
  }

  ~TranslatedSdmaQueueForTest() {
    (void)sim_.soc->queue_registry().unregister_queue(queue_handle_,
                                                      amdgpu::QueueCloseMode::ForceCancel);
    (void)legacy_vm_.unregister_address_space(address_space_);
  }

  uint32_t *ring() { return ring_.data(); }
  uint8_t *src() { return src_.data(); }
  uint8_t *dst() { return dst_.data(); }
  uint8_t *dst2() { return dst2_.data(); }
  int64_t &signal_value() { return signal_[0]; }
  /// @brief The signal word at @p index, for layouts other than value-at-zero.
  int64_t &signal_word(size_t index) { return signal_[index]; }
  uint64_t &poll_value() { return poll_[0]; }

  uint64_t src_va() const { return kSrcVa; }
  uint64_t dst_va() const { return kDstVa; }
  uint64_t dst2_va() const { return kDst2Va; }
  uint64_t signal_va() const { return kSignalVa; }
  uint64_t poll_va() const { return kPollVa; }

  void set_passthrough(bool passthrough) {
    if (!legacy_vm_.set_passthrough(address_space_, passthrough))
      throw std::runtime_error("cannot update SDMA test passthrough");
  }

  void set_fault_reporter(amdgpu::MemoryFaultReporter *reporter) {
    if (!legacy_vm_.set_fault_reporter(address_space_, reporter))
      throw std::runtime_error("cannot update SDMA test fault reporter");
  }

  template <typename T> void store(uint64_t address, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto access = sim_.soc->gpu_vm().snapshot(address_space_);
    if (!access || access->write(address, std::as_bytes(std::span(&value, 1))) !=
                       amdgpu::VmAccessOutcome::Complete) {
      throw std::runtime_error("cannot write SDMA test address space");
    }
  }

  template <typename T> T load(uint64_t address) const {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    auto access = sim_.soc->gpu_vm().snapshot(address_space_);
    if (!access || access->read(address, std::as_writable_bytes(std::span(&value, 1))) !=
                       amdgpu::VmAccessOutcome::Complete) {
      throw std::runtime_error("cannot read SDMA test address space");
    }
    return value;
  }

  /// @brief Leave the signal value mapped but drop the metadata behind it.
  void clip_signal_mapping_after_value() {
    process_.unmap_pages(kSignalVa, signal_.size() * sizeof(signal_[0]));
    process_.map_pages(kSignalVa, signal_.data(), 2 * sizeof(signal_[0]));
  }

  void clip_dst_mapping(size_t size) {
    process_.unmap_pages(kDstVa, dst_.size());
    process_.map_pages(kDstVa, dst_.data(), size);
  }

  void map_low_test_page() { process_.map_pages(0, low_page_.data(), low_page_.size()); }

  void unmap_src_tail_page() {
    process_.unmap_pages(kSrcVa + KfdProcess::kPageSize, KfdProcess::kPageSize);
  }

  void unmap_dst_tail_page() {
    process_.unmap_pages(kDstVa + KfdProcess::kPageSize, KfdProcess::kPageSize);
  }

  void remap_dst_tail_page() {
    process_.map_pages(kDstVa + KfdProcess::kPageSize, dst_.data() + KfdProcess::kPageSize,
                       KfdProcess::kPageSize);
  }

  void submit(uint32_t dwords, SdmaSubmissionWait wait = SdmaSubmissionWait::RetirementOrTerminal) {
    const uint64_t write_index = static_cast<uint64_t>(dwords) * sizeof(uint32_t);
    std::atomic_ref<uint64_t>(queue_state_[1]).store(write_index, std::memory_order_release);
    std::atomic_ref<uint64_t>(doorbells_[0]).store(write_index, std::memory_order_release);
    last_submission_status_ =
        sim_.soc->queue_registry().submit_producer(queue_handle_, write_index).status;
    if (last_submission_status_ == amdgpu::QueueSubmissionStatus::Accepted &&
        wait == SdmaSubmissionWait::RetirementOrTerminal) {
      std::unique_lock lock(progress_mutex_);
      const bool completed = progress_condition_.wait_for(lock, std::chrono::seconds(1), [&]() {
        return terminal_ || progress_cursor_ >= write_index;
      });
      if (!completed) {
        ADD_FAILURE() << "timed out waiting for the translated SDMA queue to retire or terminate";
        return;
      }
      if (terminal_)
        last_submission_status_ = amdgpu::QueueSubmissionStatus::Faulted;
    }
  }

  uint64_t read_idx() const {
    return std::atomic_ref<const uint64_t>(queue_state_[0]).load(std::memory_order_acquire);
  }

  amdgpu::QueueSubmissionStatus last_submission_status() const { return last_submission_status_; }

  /// @brief Repoint the queue's read pointer, for tests that make it unwritable.
  void set_read_ptr_va(uint64_t va) {
    (void)sim_.soc->queue_registry().unregister_queue(queue_handle_);
    queue_handle_ = {};
    register_queue(va);
  }

private:
  static constexpr uint32_t kProcessId = 1251;
  static constexpr uint32_t kQueueId = 1251;
  static constexpr uint64_t kRingVa = 0x1000'0000'0000ULL;
  static constexpr uint64_t kQueueStateVa = 0x1000'0000'1000ULL;
  static constexpr uint64_t kSrcVa = 0x1000'0000'2000ULL;
  static constexpr uint64_t kDstVa = 0x1000'0000'4000ULL;
  static constexpr uint64_t kDst2Va = 0x1000'0000'6000ULL;
  static constexpr uint64_t kSignalVa = 0x1000'0000'8000ULL;
  static constexpr uint64_t kPollVa = 0x1000'0000'9000ULL;

  Gfx1250Sim &sim_;
  amdgpu::LegacyGpuVmAdapter legacy_vm_;
  KfdProcess process_;
  amdgpu::AddressSpaceHandle address_space_;
  amdgpu::QueueHandle queue_handle_;
  amdgpu::InterruptSink interrupt_sink_;
  amdgpu::QueueSubmissionStatus last_submission_status_ = amdgpu::QueueSubmissionStatus::Accepted;
  mutable std::mutex progress_mutex_;
  std::condition_variable progress_condition_;
  uint64_t progress_cursor_ = 0;
  bool terminal_ = false;
  alignas(4096) std::array<uint32_t, 1024> ring_{};
  alignas(4096) std::array<uint64_t, 512> queue_state_{};
  alignas(4096) std::array<uint8_t, 8192> src_{};
  alignas(4096) std::array<uint8_t, 8192> dst_{};
  alignas(4096) std::array<uint8_t, 8192> dst2_{};
  alignas(4096) std::array<int64_t, 512> signal_{};
  alignas(4096) std::array<uint64_t, 512> poll_{};
  alignas(4096) std::array<uint8_t, KfdProcess::kPageSize> low_page_{};
  std::array<uint64_t, 1> doorbells_{};
};

void write_sdma_qword_va(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, uint64_t va) {
  packet[lo_dw] = static_cast<uint32_t>(va) & ~0x7u;
  packet[hi_dw] = static_cast<uint32_t>(va >> 32);
}

void write_sdma_qword_address(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, const void *addr) {
  write_sdma_qword_va(packet, lo_dw, hi_dw, reinterpret_cast<uintptr_t>(addr));
}

amdgpu::QueueHandle create_sdma_queue(Gfx1250Sim &sim, amdgpu::QueueRegistrationRequest info,
                                      uint32_t engine_id = 0) {
  info.binding_factory = amdgpu::make_sdma_queue_binding_factory(sim.soc->sdma_queue_scheduler());
  info.engine_id = engine_id;
  info.type = amdgpu::QueueType::Sdma;
  info.packet_format = amdgpu::QueuePacketFormat::Sdma;
  return sim.soc->queue_registry().register_queue(info);
}

TEST(Gfx1250SdmaTest, UnrungDoorbellSentinelDoesNotAdvanceAnEmptyQueue) {
  Gfx1250Sim sim;
  constexpr uint64_t kUnrungDoorbell = std::numeric_limits<uint64_t>::max();
  HostSdmaQueueForTest queue(sim, kUnrungDoorbell, kUnrungDoorbell);
  // Make accidental packet consumption terminate quickly instead of scanning
  // the all-ones producer range indefinitely.
  queue.ring()[0] = 0xFFu;

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  EXPECT_EQ(queue.read_idx(), 0u);
}

TEST(Gfx1250SdmaTest, LegacyQueueStartsFromPublishedReadCursorWhenNoMqdCursorExists) {
  constexpr uint32_t kProcessId = 1249;
  constexpr uint32_t kQueueId = 1249;
  constexpr uint64_t kReadPointerVa = 0x80;
  constexpr uint64_t kWritePointerVa = 0x88;
  constexpr uint64_t kRingVa = 0x100;
  constexpr uint64_t kSignalVa = 0x200;
  constexpr uint64_t kInitialReadPointer = sizeof(uint32_t);
  constexpr uint64_t kWritePointer = kInitialReadPointer + 4 * sizeof(uint32_t);
  constexpr uint32_t kFenceValue = 0xC0FFEE12u;

  Gfx1250Sim sim;
  auto backing = std::make_shared<TransientAqlAddressSpace>();
  backing->store(kRingVa, uint32_t{0xff});
  std::array<uint32_t, 4> packet{};
  packet[0] = kSdmaOpFence;
  write_sdma_qword_va(packet.data(), 1, 2, kSignalVa);
  packet[3] = kFenceValue;
  backing->store(kRingVa + kInitialReadPointer, packet);
  backing->store(kReadPointerVa, kInitialReadPointer);
  backing->store(kWritePointerVa, kWritePointer);
  backing->store(kSignalVa, uint64_t{0});
  const amdgpu::AddressSpaceHandle address_space =
      sim.soc->gpu_vm().register_translated(kProcessId, backing, backing);
  ASSERT_TRUE(address_space);

  std::array<uint64_t, 1> doorbell{kWritePointer};

  const amdgpu::QueueHandle queue = create_sdma_queue(
      sim,
      {.identity = {.address_space = address_space, .process_id = kProcessId, .queue_id = kQueueId},
       .ring = {.base_address = kRingVa,
                .size_bytes = 64,
                .consumer_pointer_address = kReadPointerVa,
                .producer_pointer_address = kWritePointerVa},
       .doorbell = {.mode = amdgpu::QueueDoorbellMode::HostPolled,
                    .host_base = doorbell.data(),
                    .last_value = 0},
       .binding_factory = {}});
  ASSERT_TRUE(queue);
  EXPECT_EQ(sim.soc->queue_registry().submit_producer(queue, kWritePointer).status,
            amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(wait_until([&]() { return backing->load_u64(kReadPointerVa) == kWritePointer; }));
  EXPECT_EQ(static_cast<uint32_t>(backing->load_u64(kSignalVa)), kFenceValue);

  EXPECT_TRUE(sim.soc->queue_registry().unregister_queue(queue));
  EXPECT_TRUE(sim.soc->gpu_vm().unregister_address_space(address_space));
}

TEST(Gfx1250SdmaTest, VfioQueueUsesDoorbellWithoutTranslatingMqdWritePointer) {
  constexpr uint32_t kProcessId = 1253;
  constexpr uint32_t kQueueId = 1253;
  constexpr uint64_t kReadPointerVa = 0x80;
  constexpr uint64_t kWritePointerVa = 0x88;
  constexpr uint64_t kDoorbellVa = 0x90;
  constexpr uint64_t kRingVa = 0x100;
  constexpr uint64_t kSignalVa = 0x200;
  constexpr uint32_t kFenceValue = 0xC001D00Du;
  constexpr uint64_t kWritePointer = 4 * sizeof(uint32_t);

  Gfx1250Sim sim;
  auto backing = std::make_shared<TransientAqlAddressSpace>();
  std::array<uint32_t, 4> packet{};
  packet[0] = kSdmaOpFence;
  write_sdma_qword_va(packet.data(), 1, 2, kSignalVa);
  packet[3] = kFenceValue;
  backing->store(kRingVa, packet);
  backing->store(kReadPointerVa, uint64_t{0});
  backing->store(kWritePointerVa, kWritePointer);
  backing->store(kSignalVa, uint64_t{0});
  backing->return_next_atomic_load_at(kWritePointerVa, amdgpu::VmAccessOutcome::Faulted);

  const amdgpu::AddressSpaceHandle address_space =
      sim.soc->gpu_vm().register_translated(kProcessId, backing, backing);
  ASSERT_TRUE(address_space);

  const amdgpu::QueueHandle queue = create_sdma_queue(
      sim,
      {.identity = {.address_space = address_space, .process_id = kProcessId, .queue_id = kQueueId},
       .ring = {.base_address = kRingVa,
                .size_bytes = 64,
                .consumer_pointer_address = kReadPointerVa,
                .producer_pointer_address = kWritePointerVa},
       .doorbell = {.mode = amdgpu::QueueDoorbellMode::VmPolled,
                    .address = kDoorbellVa,
                    .last_value = kWritePointer},
       .binding_factory = {}});
  ASSERT_TRUE(queue);
  EXPECT_EQ(sim.soc->queue_registry().submit_producer(queue, kWritePointer).status,
            amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(wait_until([&]() { return backing->load_u64(kReadPointerVa) == kWritePointer; }));
  EXPECT_EQ(static_cast<uint32_t>(backing->load_u64(kSignalVa)), kFenceValue);
  EXPECT_EQ(backing->atomic_load_attempts_at(kWritePointerVa), 0u);

  EXPECT_TRUE(sim.soc->queue_registry().unregister_queue(queue));
  EXPECT_TRUE(sim.soc->gpu_vm().unregister_address_space(address_space));
}

TEST(Gfx1250SdmaTest, VfioQueueStartsFromMqdReadCursorNotWritebackMemory) {
  constexpr uint32_t kProcessId = 1254;
  constexpr uint32_t kQueueId = 1254;
  constexpr uint64_t kReadPointerVa = 0x80;
  constexpr uint64_t kWritePointerVa = 0x88;
  constexpr uint64_t kRingVa = 0x100;
  constexpr uint64_t kSignalVa = 0x200;
  constexpr uint64_t kInitialReadPointer = sizeof(uint32_t);
  constexpr uint32_t kFenceValue = 0x1A17C0DEu;
  constexpr uint64_t kWritePointer = kInitialReadPointer + 4 * sizeof(uint32_t);

  Gfx1250Sim sim;
  auto backing = std::make_shared<TransientAqlAddressSpace>();
  std::array<uint32_t, 4> packet{};
  packet[0] = kSdmaOpFence;
  write_sdma_qword_va(packet.data(), 1, 2, kSignalVa);
  packet[3] = kFenceValue;
  backing->store(kRingVa + kInitialReadPointer, packet);
  // RPTR memory is a writeback destination, not the ring consumer's source of truth.
  backing->store(kReadPointerVa, uint64_t{0});
  backing->store(kWritePointerVa, kWritePointer);
  backing->store(kSignalVa, uint64_t{0});

  const amdgpu::AddressSpaceHandle address_space =
      sim.soc->gpu_vm().register_translated(kProcessId, backing, backing);
  ASSERT_TRUE(address_space);

  const amdgpu::QueueHandle queue = create_sdma_queue(
      sim,
      {.identity = {.address_space = address_space, .process_id = kProcessId, .queue_id = kQueueId},
       .ring = {.base_address = kRingVa,
                .size_bytes = 64,
                .consumer_pointer_address = kReadPointerVa,
                .producer_pointer_address = kWritePointerVa},
       .doorbell = {.mode = amdgpu::QueueDoorbellMode::VmPolled,
                    .address = kWritePointerVa,
                    .last_value = kWritePointer},
       .binding_factory = {},
       .initial_consumer_cursor = kInitialReadPointer});
  ASSERT_TRUE(queue);
  EXPECT_EQ(sim.soc->queue_registry().submit_producer(queue, kWritePointer).status,
            amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_TRUE(wait_until([&]() { return backing->load_u64(kReadPointerVa) == kWritePointer; }));
  EXPECT_EQ(static_cast<uint32_t>(backing->load_u64(kSignalVa)), kFenceValue);

  EXPECT_TRUE(sim.soc->queue_registry().unregister_queue(queue));
  EXPECT_TRUE(sim.soc->gpu_vm().unregister_address_space(address_space));
}

TEST(CommandProcessorInterruptRoutingTest, QueuesWithTheSameProcessIdKeepDistinctOwners) {
  Gfx1250Sim sim;
  constexpr uint32_t kProcessId = 71;
  KfdProcess process(kProcessId);
  amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
  const amdgpu::AddressSpaceHandle address_space =
      legacy_vm.register_address_space(kProcessId, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(address_space);
  ASSERT_TRUE(legacy_vm.set_passthrough(address_space, true));
  alignas(uint64_t) uint64_t first_status = 0;
  alignas(uint64_t) uint64_t second_status = 0;
  uint32_t first_calls = 0;
  uint32_t second_calls = 0;
  amdgpu::InterruptSubscription first_owner([&](uint32_t process_id, uint32_t event_id) {
    EXPECT_EQ(process_id, kProcessId);
    EXPECT_EQ(event_id, 17u);
    ++first_calls;
    first_status = 0;
  });
  amdgpu::InterruptSubscription second_owner([&](uint32_t process_id, uint32_t event_id) {
    EXPECT_EQ(process_id, kProcessId);
    EXPECT_EQ(event_id, 29u);
    ++second_calls;
    second_status = 0;
  });

  auto register_queue = [&](uint32_t queue_id, uint64_t &exception_status, uint32_t event_id,
                            const amdgpu::InterruptSink &interrupt_sink) {
    amdgpu::AqlQueueConfig queue{};
    queue.address_space = address_space;
    queue.interrupt_sink = interrupt_sink;
    queue.process_id = kProcessId;
    queue.queue_id = queue_id;
    queue.ring_base_va = 0x100;
    queue.ring_size = 4096;
    queue.read_ptr_va = 0x80;
    queue.write_ptr_va = 0x88;
    queue.exception_status_va = reinterpret_cast<uint64_t>(&exception_status);
    queue.exception_event_id = event_id;
    (void)sim.cp()->register_queue(std::move(queue));
  };
  register_queue(17, first_status, 17, first_owner.sink());
  register_queue(29, second_status, 29, second_owner.sink());

  EXPECT_TRUE(sim.cp()->signal_queue_exception(17, kProcessId, 0x11));
  EXPECT_EQ(first_calls, 1u);
  EXPECT_EQ(second_calls, 0u);
  EXPECT_TRUE(sim.cp()->signal_queue_exception(29, kProcessId, 0x22));
  EXPECT_EQ(first_calls, 1u);
  EXPECT_EQ(second_calls, 1u);

  first_owner.reset();
  EXPECT_TRUE(sim.cp()->signal_queue_exception(29, kProcessId, 0x33));
  EXPECT_EQ(first_calls, 1u);
  EXPECT_EQ(second_calls, 2u);

  sim.cp()->unregister_queue(17, kProcessId);
  sim.cp()->unregister_queue(29, kProcessId);
  EXPECT_TRUE(legacy_vm.unregister_address_space(address_space));
}

TEST(CommandProcessorAqlTest, UnavailablePacketReadDoesNotAdvancePastUnfetchedWork) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.fail_next_packet_read();

  queue.step();
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.packet_read_attempts(), 2u) << "the failed packet was skipped instead of retried";
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, UnavailableReadPointerPublicationRetriesWithoutRefetch) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.fail_next_read_pointer_store();

  queue.step();
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.accepted_entries(), 1u)
      << "a publication retry fetched the already-committed packet again";
  EXPECT_EQ(queue.read_pointer_store_attempts(), 2u);
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, UnavailableWritePointerReadRetriesWithoutLosingPacket) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.fail_next_write_pointer_load(amdgpu::VmAccessOutcome::Unavailable);

  queue.step();
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.write_pointer_load_attempts(), 2u);
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, ExplicitDoorbellBoundsAdmissionToNotifiedCursor) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim, amdgpu::QueueDoorbellMode::Explicit);
  queue.set_write_pointer(2);

  queue.notify(0);
  queue.step();
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.accepted_entries(), 1u);

  queue.notify(1);
  queue.step();
  EXPECT_EQ(queue.read_pointer(), 2u);
  EXPECT_EQ(queue.accepted_entries(), 2u);
}

TEST(SoCLifetimeTest, ReleasesDirectCommandProcessorQueuesBeforeGpuVm) {
  auto sim = std::make_unique<Gfx1250Sim>();
  auto backing = std::make_shared<TransientAqlAddressSpace>();
  constexpr uint32_t kProcessId = 1255;
  constexpr uint64_t kAqlRing = 0x100;
  constexpr uint64_t kAqlReadPointer = 0x180;
  constexpr uint64_t kAqlWritePointer = 0x188;
  constexpr uint64_t kPm4Ring = 0x200;
  constexpr uint64_t kPm4ReadPointer = 0x280;
  backing->store(kAqlReadPointer, uint64_t{0});
  backing->store(kAqlWritePointer, uint64_t{0});
  backing->store(kPm4ReadPointer, uint64_t{0});
  const amdgpu::AddressSpaceHandle address_space =
      sim->soc->gpu_vm().register_translated(kProcessId, backing, backing);
  ASSERT_TRUE(address_space);

  amdgpu::AqlQueueConfig aql{};
  aql.address_space = address_space;
  aql.process_id = kProcessId;
  aql.queue_id = 1;
  aql.ring_base_va = kAqlRing;
  aql.ring_size = amdgpu::kAqlPacketBytes;
  aql.read_ptr_va = kAqlReadPointer;
  aql.write_ptr_va = kAqlWritePointer;
  aql.doorbell_mode = amdgpu::QueueDoorbellMode::Explicit;
  ASSERT_NE(sim->cp()->register_queue(std::move(aql)), 0u);
  ASSERT_NE(sim->cp()->register_pm4_queue({.address_space = address_space,
                                           .ring_base = kPm4Ring,
                                           .ring_size_bytes = 64,
                                           .consumer_pointer_address = kPm4ReadPointer,
                                           .initial_consumer_cursor = std::nullopt,
                                           .packet_callbacks = {}}),
            0u);

  sim.reset();
  SUCCEED();
}

TEST(CommandProcessorAqlTest, TerminalWritePointerReadFaultsQueue) {
  for (const amdgpu::VmAccessOutcome outcome :
       {amdgpu::VmAccessOutcome::Faulted, amdgpu::VmAccessOutcome::Malformed}) {
    Gfx1250Sim sim;
    TransientAqlQueueForTest queue(sim);
    queue.fail_next_write_pointer_load(outcome);

    queue.step();
    EXPECT_EQ(queue.read_pointer(), 0u) << static_cast<int>(outcome);
    EXPECT_EQ(queue.accepted_entries(), 0u) << static_cast<int>(outcome);
    EXPECT_EQ(queue.write_pointer_load_attempts(), 1u) << static_cast<int>(outcome);
    EXPECT_TRUE(queue.faulted()) << static_cast<int>(outcome);
  }
}

TEST(CommandProcessorAqlTest, UnavailableDependencyReadRetriesFetchedPacket) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.set_barrier_dependency(0);
  queue.fail_next_dependency_value_load(amdgpu::VmAccessOutcome::Unavailable);

  queue.step();
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.dependency_value_load_attempts(), 2u);
  EXPECT_EQ(queue.dependency_handle_load_attempts(), 0u);
  EXPECT_EQ(queue.completion_handle_load_attempts(), 0u);
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, TerminalDependencyReadFaultsQueue) {
  for (const amdgpu::VmAccessOutcome outcome :
       {amdgpu::VmAccessOutcome::Faulted, amdgpu::VmAccessOutcome::Malformed}) {
    Gfx1250Sim sim;
    TransientAqlQueueForTest queue(sim);
    queue.set_barrier_dependency(0);
    queue.fail_next_dependency_value_load(outcome);

    queue.step();
    EXPECT_EQ(queue.read_pointer(), 0u) << static_cast<int>(outcome);
    EXPECT_EQ(queue.accepted_entries(), 0u) << static_cast<int>(outcome);
    EXPECT_EQ(queue.dependency_value_load_attempts(), 1u) << static_cast<int>(outcome);
    EXPECT_TRUE(queue.faulted()) << static_cast<int>(outcome);
  }
}

TEST(CommandProcessorAqlTest, UnavailableKernelDescriptorReadRetriesBeforeAdmission) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.set_kernel_dispatch();
  queue.fail_next_kernel_descriptor_read(amdgpu::VmAccessOutcome::Unavailable);

  queue.step();
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.kernel_descriptor_read_attempts(), 2u);
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, TerminalKernelDescriptorReadFaultsBeforeAdmission) {
  for (const amdgpu::VmAccessOutcome outcome :
       {amdgpu::VmAccessOutcome::Faulted, amdgpu::VmAccessOutcome::Malformed}) {
    Gfx1250Sim sim;
    TransientAqlQueueForTest queue(sim);
    queue.set_kernel_dispatch();
    queue.fail_next_kernel_descriptor_read(outcome);

    queue.step();
    EXPECT_EQ(queue.read_pointer(), 0u) << static_cast<int>(outcome);
    EXPECT_EQ(queue.accepted_entries(), 0u) << static_cast<int>(outcome);
    EXPECT_EQ(queue.kernel_descriptor_read_attempts(), 1u) << static_cast<int>(outcome);
    EXPECT_TRUE(queue.faulted()) << static_cast<int>(outcome);
  }
}

TEST(CommandProcessorAqlTest, QueuePointerSnapshotPinsRingFetchAndCursorPublication) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  const auto replacement = queue.make_invalid_replacement();
  queue.replace_after_next_write_pointer_load(replacement);

  queue.step();

  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.read_pointer_on(replacement), 0u);
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, RingSnapshotPinsDependencyReads) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.set_barrier_dependency(0);
  const auto replacement = queue.make_invalid_replacement();
  queue.set_replacement_dependency(replacement, 1);
  queue.replace_after_next_packet_read(replacement);

  queue.step();

  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.read_pointer_on(replacement), 0u);
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, RingSnapshotPinsKernelDescriptorReads) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.set_kernel_dispatch();
  const auto replacement = queue.make_invalid_replacement();
  queue.fail_replacement_kernel_descriptor_read(replacement, amdgpu::VmAccessOutcome::Faulted);
  queue.replace_after_next_packet_read(replacement);

  queue.step();

  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.read_pointer_on(replacement), 0u);
  EXPECT_FALSE(queue.faulted());
}

TEST(CommandProcessorAqlTest, CursorRetryKeepsSnapshotThatReadTheDependency) {
  Gfx1250Sim sim;
  TransientAqlQueueForTest queue(sim);
  queue.set_barrier_dependency(0);
  queue.fail_next_read_pointer_store();
  const auto replacement = queue.make_invalid_replacement();
  queue.replace_after_next_dependency_load(replacement);

  queue.step();

  EXPECT_EQ(queue.accepted_entries(), 1u);
  EXPECT_EQ(queue.read_pointer(), 1u);
  EXPECT_EQ(queue.read_pointer_store_attempts(), 2u);
  EXPECT_EQ(queue.read_pointer_on(replacement), 0u);
  EXPECT_FALSE(queue.faulted());
}

TEST(Gfx1250SdmaTest, StaleAddressSpaceCannotExecuteThroughLegacyBacking) {
  Gfx1250Sim sim;
  constexpr uint32_t kProcessId = 1252;
  KfdProcess process(kProcessId);
  amdgpu::LegacyGpuVmAdapter legacy_vm(sim.soc->gpu_vm(), sim.soc->memory());
  const amdgpu::AddressSpaceHandle stale =
      legacy_vm.register_address_space(kProcessId, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(stale);
  ASSERT_TRUE(legacy_vm.set_passthrough(stale, true));
  ASSERT_TRUE(legacy_vm.unregister_address_space(stale));

  constexpr uint32_t kQueueId = 1252;
  alignas(8) std::array<uint32_t, 16> ring{};
  alignas(8) uint64_t read_idx = 0;
  alignas(8) uint64_t write_idx = 7 * sizeof(uint32_t);
  std::array<uint64_t, 1> doorbells{write_idx};
  std::array<uint8_t, 32> source{};
  std::array<uint8_t, 32> destination{};
  for (size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<uint8_t>(index + 1);

  ring[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  ring[1] = static_cast<uint32_t>(source.size() - 1);
  write_sdma_qword_address(ring.data(), 3, 4, source.data());
  write_sdma_qword_address(ring.data(), 5, 6, destination.data());

  const amdgpu::QueueHandle queue = create_sdma_queue(
      sim,
      {.identity = {.address_space = stale, .process_id = kProcessId, .queue_id = kQueueId},
       .ring = {.base_address = reinterpret_cast<uint64_t>(ring.data()),
                .size_bytes = static_cast<uint32_t>(ring.size() * sizeof(uint32_t)),
                .consumer_pointer_address = reinterpret_cast<uint64_t>(&read_idx),
                .producer_pointer_address = reinterpret_cast<uint64_t>(&write_idx)},
       .doorbell = {.mode = amdgpu::QueueDoorbellMode::HostPolled, .host_base = doorbells.data()},
       .binding_factory = {}});
  EXPECT_FALSE(queue);
  EXPECT_EQ(read_idx, 0u);
  EXPECT_TRUE(std::ranges::all_of(destination, [](uint8_t value) { return value == 0; }))
      << "a stale nonzero handle fell back to the legacy passthrough store";
}

TEST(Gfx1250SdmaTest, PollMem64WaitsForFull64BitCondition) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.poll_value() = uint64_t{1} << 32;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28); // 64-bit equal poll.
  write_sdma_qword_va(packet, 1, 2, queue.poll_va());
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);

  std::atomic_ref<uint64_t>(queue.poll_value()).store(0, std::memory_order_release);
  EXPECT_TRUE(wait_until([&queue]() { return queue.read_idx() == 8u * sizeof(uint32_t); }))
      << "the SDMA worker did not retry the waiting packet";
}

TEST(Gfx1250SdmaTest, Fence64WritesFull64BitValue) {
  Gfx1250Sim sim;
  HostSdmaQueueForTest queue(sim);
  alignas(8) uint64_t value = 0;
  constexpr uint64_t kFenceValue = 0x12345678ABCDEF01ULL;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpFence | (kSdmaSubopFence64 << 8) | (3u << 16);
  write_sdma_qword_address(packet, 1, 2, &value);
  packet[3] = static_cast<uint32_t>(kFenceValue);
  packet[4] = static_cast<uint32_t>(kFenceValue >> 32);

  queue.submit(5);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 5u * sizeof(uint32_t));
  EXPECT_EQ(std::atomic_ref<uint64_t>(value).load(std::memory_order_acquire), kFenceValue);
}

// A wrong GCR packet size silently desyncs the SDMA ring read pointer and
// corrupts the following packet. Emit OP_GCR followed by a 32-bit FENCE and
// assert both the read pointer advance and that the FENCE decoded at the right
// boundary (its sentinel lands). gfx11/12 GCR is 5 dwords; gfx1250 is 6.
TEST(Gfx1250SdmaTest, GcrPacketSizeMatchesDialectAndKeepsRingInSync) {
  constexpr uint32_t kGcrLegacySize = 5;
  constexpr uint32_t kGcrGfx1250Size = 6;
  constexpr uint32_t kFenceSize = 4;
  constexpr uint32_t kFenceSentinel = 0xC0FFEE11u;
  // GL2 invalidate control bit position differs by dialect; setting it exercises
  // a realistic invalidate GCR but does not affect the decoded packet size.
  constexpr uint32_t kLegacyGl2InvControlDw = 2;
  constexpr uint32_t kLegacyGl2InvBit = 1u << 30;
  constexpr uint32_t kGfx1250Gl2InvControlDw = 3;
  constexpr uint32_t kGfx1250Gl2InvBit = 1u << 14;

  auto run_dialect = [kFenceSentinel](amdgpu::SdmaPacketDialect dialect, uint32_t gcr_size,
                                      uint32_t control_dw, uint32_t control_bit) {
    Gfx1250Sim sim;
    ASSERT_TRUE(sim.soc->sdma_queue_scheduler().set_packet_dialect(dialect));
    HostSdmaQueueForTest queue(sim);
    alignas(8) uint32_t fence_value = 0;

    auto *packet = queue.ring();
    packet[0] = kSdmaOpGcr;
    packet[control_dw] = control_bit;

    uint32_t *fence = packet + gcr_size;
    fence[0] = kSdmaOpFence; // 32-bit fence (sub_op 0).
    write_sdma_qword_address(fence, 1, 2, &fence_value);
    fence[3] = kFenceSentinel;

    queue.submit(gcr_size + kFenceSize);
    EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
    EXPECT_EQ(queue.read_idx(), (gcr_size + kFenceSize) * sizeof(uint32_t));
    EXPECT_EQ(std::atomic_ref<uint32_t>(fence_value).load(std::memory_order_acquire),
              kFenceSentinel);
  };

  run_dialect(amdgpu::SdmaPacketDialect::Gfx11Plus, kGcrLegacySize, kLegacyGl2InvControlDw,
              kLegacyGl2InvBit);
  run_dialect(amdgpu::SdmaPacketDialect::Gfx1250, kGcrGfx1250Size, kGfx1250Gl2InvControlDw,
              kGfx1250Gl2InvBit);
}

// SDMA writes go straight to backing while L2 may still hold a dirty line that
// overlaps the destination. Seed that state explicitly with writeback_line().
// The fix flushes the caches before the direct write, so the dirty line is
// published first and the SDMA data supersedes it. Regression for that ordering.
TEST(Gfx1250SdmaTest, ConstFillSupersedesOverlappingDirtyL2Line) {
  Gfx1250Sim sim;
  // The config-driven topology build wires the XCD's L2 into the CP, so the SDMA
  // cache maintenance operates on the same L2 instance we dirty below.
  auto *l2 = sim.xcd()->l2_cache();
  ASSERT_NE(l2, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kStaleWord = 0x11111111u;
  constexpr uint32_t kFillWord = 0x22222222u;

  // Seed a dirty L2 line overlapping the destination, without touching backing.
  uint8_t stale_line[amdgpu::L2Cache::LINE_SIZE];
  std::memset(stale_line, static_cast<int>(kStaleWord & 0xFF), sizeof(stale_line));
  l2->writeback_line(queue.dst_va(), stale_line, amdgpu::Mtype::RW, kProcessId);

  // CONST_FILL the destination line with a different byte pattern.
  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = amdgpu::L2Cache::LINE_SIZE - 1; // count-1 bytes.

  queue.submit(5);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);

  // Backing must reflect the SDMA fill, not the stale cached line.
  EXPECT_EQ(queue.load<uint32_t>(queue.dst_va()), kFillWord);
  EXPECT_NE(queue.load<uint32_t>(queue.dst_va()), kStaleWord);
}

TEST(Gfx1250SdmaTest, ConstFillWritesMappedPrefixAndAdvances) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr size_t kFillBytes = 128;
  constexpr size_t kMappedBytes = 64;
  constexpr uint8_t kInitialByte = 0xa5;
  constexpr uint32_t kFillWord = 0x44332211;
  queue.clip_dst_mapping(kMappedBytes);
  std::fill_n(queue.dst(), kFillBytes, kInitialByte);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = kFillBytes - 1;

  queue.submit(5);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 5u * sizeof(uint32_t));

  std::array<uint8_t, sizeof(kFillWord)> pattern{};
  std::memcpy(pattern.data(), &kFillWord, sizeof(kFillWord));
  for (size_t i = 0; i < kMappedBytes; ++i)
    EXPECT_EQ(queue.dst()[i], pattern[i % pattern.size()]);
  EXPECT_TRUE(std::all_of(queue.dst() + kMappedBytes, queue.dst() + kFillBytes,
                          [](uint8_t value) { return value == kInitialByte; }));
}

// A byte count that runs past the end of the address space is a malformed
// packet, not one waiting on a mapping. Retrying it re-runs the same packet on
// every doorbell and the queue never drains, and the page walk underneath adds
// the offset without rechecking, so a wrapped range resumes at address zero and
// modifies unrelated low memory while reporting that it completed. Each packet
// type is followed by a fence that must never run.
// A completion signal is decremented and then announced. Everything that can
// refuse therefore has to be settled first: once the value drops, the waiter may
// already have observed it, and faulting afterwards leaves a signal that fired
// with nothing behind it. Page-table entries carry sub-page extents by design,
// so the metadata record can straddle the end of its backing -- which reads back
// part fabricated, and a half-read event id names some other event.
TEST(Gfx1250SdmaTest, ClippedSignalMetadataHaltsWithoutDecrementing) {
  Gfx1250Sim sim;
  constexpr int64_t kSignalStart = 5;

  std::atomic<uint32_t> notified{0};
  amdgpu::InterruptSubscription interrupt_subscription(
      [&](uint32_t, uint32_t event_id) { notified.store(event_id, std::memory_order_release); });
  TranslatedSdmaQueueForTest queue(sim, interrupt_subscription.sink());

  // Back the signal value but stop the mapping before the mailbox and event id,
  // so the record the completion path must read is clipped.
  queue.clip_signal_mapping_after_value();
  // The packet addresses the value field, and the record's base is eight bytes
  // below it, so the value is the second word and the mailbox and event id are
  // the third and fourth -- the ones the clipped mapping drops.
  queue.signal_word(1) = kSignalStart;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpAtomic | (47u << 25); // SDMA_ATOMIC_ADD64
  write_sdma_qword_va(packet, 1, 2, queue.signal_va() + 8);
  packet[3] = static_cast<uint32_t>(-1);
  packet[4] = 0xFFFFFFFFu;

  queue.submit(5);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);

  EXPECT_EQ(queue.signal_word(1), kSignalStart)
      << "the signal was decremented before its metadata was known to be readable";
  EXPECT_EQ(notified.load(std::memory_order_acquire), 0u) << "an event was notified from a "
                                                             "clipped record";
}

TEST(Gfx1250SdmaTest, WrappingCopyHaltsInsteadOfRetrying) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);
  constexpr uint64_t kNearTop = std::numeric_limits<uint64_t>::max() - 15;
  constexpr uint32_t kCopyBytes = 128;
  queue.signal_value() = 5;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  packet[2] = 0;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, kNearTop);
  packet[7] = kSdmaOpFence;
  write_sdma_qword_va(packet, 8, 9, queue.signal_va());
  packet[10] = 0xDEADBEEFu;

  queue.submit(11);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.signal_value(), 5) << "a fence behind a wrapping copy ran";

  // Parking the read pointer looks identical whether the queue faulted or is
  // retrying, so replace the malformed packet with a valid one. A retrying
  // queue re-reads the ring at the same position and would run it; a faulted
  // queue is over.
  packet[0] = kSdmaOpFence;
  write_sdma_qword_va(packet, 1, 2, queue.signal_va());
  packet[3] = 0xDEADBEEFu;
  queue.submit(4);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.signal_value(), 5) << "the queue retried after a wrapping copy";
}

TEST(Gfx1250SdmaTest, WrappingConstFillHaltsInsteadOfRetrying) {
  class RecordingReporter : public amdgpu::MemoryFaultReporter {
  public:
    void report_memory_fault(uint32_t, uint64_t addr, amdgpu::MemoryFaultCause cause) override {
      addresses.push_back(addr);
      causes.push_back(cause);
    }
    std::vector<uint64_t> addresses;
    std::vector<amdgpu::MemoryFaultCause> causes;
  };

  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);
  constexpr uint64_t kNearTop = std::numeric_limits<uint64_t>::max() - 15;
  queue.signal_value() = 5;

  // The fill resolves and walks the range itself, so if it rejected a malformed
  // one on its own the queue would halt with nothing to explain it and a
  // workload waiting on the fence behind it would hang silently.
  RecordingReporter reporter;
  queue.set_fault_reporter(&reporter);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30);
  write_sdma_qword_va(packet, 1, 2, kNearTop);
  packet[3] = 0x44332211;
  packet[4] = 127;
  packet[5] = kSdmaOpFence;
  write_sdma_qword_va(packet, 6, 7, queue.signal_va());
  packet[8] = 0xDEADBEEFu;

  queue.submit(9);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.signal_value(), 5) << "a fence behind a wrapping fill ran";
  ASSERT_EQ(reporter.addresses.size(), 1u) << "the queue halted without reporting why";
  EXPECT_EQ(reporter.addresses.front(), kNearTop);
  EXPECT_EQ(reporter.causes.front(), amdgpu::MemoryFaultCause::NotPresent);

  // As above: swap in a valid packet, which only a retrying queue would run.
  packet[0] = kSdmaOpFence;
  write_sdma_qword_va(packet, 1, 2, queue.signal_va());
  packet[3] = 0xDEADBEEFu;
  queue.submit(4);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.signal_value(), 5) << "the queue retried after a wrapping fill";

  queue.set_fault_reporter(nullptr);
}

TEST(Gfx1250SdmaTest, WrappingWriteHaltsInsteadOfLandingInLowMemory) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);
  constexpr uint64_t kNearTop = std::numeric_limits<uint64_t>::max() - 15;
  // Inside the bytes a wrapped walk would resume over.
  constexpr uint64_t kWrapTarget = 0x10;
  constexpr uint32_t kSentinel = 0xFEEDFACEu;
  queue.map_low_test_page();
  queue.store(kWrapTarget, kSentinel);
  queue.signal_value() = 5;

  constexpr uint32_t kDwords = 8;
  auto *packet = queue.ring();
  packet[0] = kSdmaOpWrite;
  write_sdma_qword_va(packet, 1, 2, kNearTop);
  packet[3] = kDwords - 1;
  for (uint32_t i = 0; i < kDwords; ++i)
    packet[4 + i] = 0xA5A5A5A5u;
  packet[4 + kDwords] = kSdmaOpFence;
  write_sdma_qword_va(packet, 5 + kDwords, 6 + kDwords, queue.signal_va());
  packet[7 + kDwords] = 0xDEADBEEFu;

  queue.submit(8 + kDwords);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.load<uint32_t>(kWrapTarget), kSentinel) << "a wrapping write reached low memory";
  EXPECT_EQ(queue.signal_value(), 5) << "a fence behind a wrapping write ran";
}

TEST(Gfx1250SdmaTest, ConstFillUnmappedTailDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr size_t kFillBytes = 8192;
  queue.unmap_dst_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30);
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = 0x44332211;
  packet[4] = kFillBytes - 1;

  queue.submit(5, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
}

// A scalar L1 (K$) can retain a clean snapshot overlapping an SDMA destination.
// The pre-write maintenance invalidates that snapshot, and later K$ maintenance
// must not publish it over the direct SDMA result.
TEST(Gfx1250SdmaTest, ConstFillSupersedesOverlappingScalarL1Line) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  ASSERT_NE(cu, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kStaleWord = 0x11111111u;
  constexpr uint32_t kFillWord = 0x22222222u;

  // Populate a K$ line overlapping the SDMA destination via a write-through
  // scalar store. K$ retains a clean snapshot of the pre-fill value.
  cu->l1_scalar().store(queue.dst_va(), /*num_dwords=*/1, &kStaleWord, kProcessId);

  // CONST_FILL the destination line with a different pattern.
  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = amdgpu::L2Cache::LINE_SIZE - 1; // count-1 bytes.

  queue.submit(5);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);

  // The SDMA pre-write maintenance must have invalidated the old K$ snapshot,
  // so the first scalar reload observes the fill rather than kStaleWord.
  uint32_t scalar_value = 0;
  cu->l1_scalar().load(queue.dst_va(), /*num_dwords=*/1, &scalar_value, kProcessId);
  EXPECT_EQ(scalar_value, kFillWord);

  // Mimic later acquire/release maintenance. The clean K$ snapshot must not
  // resurrect stale data over the SDMA fill.
  cu->flush_l1(kProcessId);
  if (auto *l2 = sim.xcd()->l2_cache())
    l2->flush_all();

  // Backing must reflect the SDMA fill, not the stale scalar line.
  EXPECT_EQ(queue.load<uint32_t>(queue.dst_va()), kFillWord);
  EXPECT_NE(queue.load<uint32_t>(queue.dst_va()), kStaleWord);
}

// OP_TIMESTAMP is a direct backing-store write like COPY/FENCE/CONST_FILL, so it
// has the same clobber hazard: a dirty cached line overlapping the timestamp
// address must be published before the store, not written out over it by a later
// flush. Seed a dirty L2 line at the timestamp address, issue OP_TIMESTAMP, then
// force a flush; the stored timestamp must survive (stale word gone, value set).
TEST(Gfx1250SdmaTest, TimestampSupersedesOverlappingDirtyL2Line) {
  Gfx1250Sim sim;
  auto *l2 = sim.xcd()->l2_cache();
  ASSERT_NE(l2, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint64_t kStaleQword = 0x1111111111111111ULL;

  // Seed a dirty L2 line overlapping the timestamp destination, without touching
  // backing.
  uint8_t stale_line[amdgpu::L2Cache::LINE_SIZE];
  std::memset(stale_line, 0x11, sizeof(stale_line));
  l2->writeback_line(queue.dst_va(), stale_line, amdgpu::Mtype::RW, kProcessId);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpTimestamp;
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());

  queue.submit(3); // TIMESTAMP is 3 dwords.
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);

  // Force any still-resident dirty line out, mimicking a later flush.
  if (auto *xl2 = sim.xcd()->l2_cache())
    xl2->flush_all();

  // The timestamp value is nondeterministic, but it must not be the stale word
  // and must be a plausible nonzero nanosecond count.
  const uint64_t stored = queue.load<uint64_t>(queue.dst_va());
  EXPECT_NE(stored, kStaleQword);
  EXPECT_NE(stored, 0u);
}

// The GCR decoder now distinguishes GL2 writeback (publish dirty lines),
// invalidate/discard (drop without writeback), and no-op (no GL2 bits). This is
// the data-loss distinction the PR protects. Dirty an L2 line, then issue each
// GCR flavor and observe whether the dirty data reaches backing.
TEST(Gfx1250SdmaTest, GcrWritebackPublishesInvalidateDropsNoopKeeps) {
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kDirtyWord = 0x33333333u;
  constexpr uint32_t kBackingWord = 0x44444444u;
  // gfx1250 GCR control dword (DW3) bit positions.
  constexpr uint32_t kControlDw = 3;
  constexpr uint32_t kGl2InvBit = 1u << 14;
  constexpr uint32_t kGl2WbBit = 1u << 15;

  // Outcome of a GCR flavor: the value in backing (read directly through the
  // page table) and the value seen through L2 (which returns the resident dirty
  // line if still present, or re-fetches backing if the line was dropped).
  struct GcrOutcome {
    uint32_t backing = 0;
    uint32_t via_l2 = 0;
  };

  enum class GcrKind { WritebackOnly, InvalidateOnly, Noop };
  // Void return so a missing L2 is a fatal guard (ASSERT_*) before we deref it.
  auto run = [&](GcrKind kind, GcrOutcome &out) {
    Gfx1250Sim sim;
    auto *l2 = sim.xcd()->l2_cache();
    ASSERT_NE(l2, nullptr);
    TranslatedSdmaQueueForTest queue(sim);

    // Put a known value in backing, then a different dirty value in L2 on top.
    queue.store(queue.dst_va(), kBackingWord);
    uint8_t dirty_line[amdgpu::L2Cache::LINE_SIZE];
    std::memset(dirty_line, static_cast<int>(kDirtyWord & 0xFF), sizeof(dirty_line));
    l2->writeback_line(queue.dst_va(), dirty_line, amdgpu::Mtype::RW, kProcessId);

    auto *packet = queue.ring();
    packet[0] = kSdmaOpGcr;
    if (kind == GcrKind::WritebackOnly)
      packet[kControlDw] = kGl2WbBit;
    else if (kind == GcrKind::InvalidateOnly)
      packet[kControlDw] = kGl2InvBit;
    else
      packet[kControlDw] = 0; // no GL2 bits: no-op.

    queue.submit(6); // gfx1250 GCR is 6 dwords.
    EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);

    out.backing = queue.load<uint32_t>(queue.dst_va());
    // Read back through L2: a still-resident dirty line returns kDirtyWord; a
    // dropped line re-fetches from backing on the miss.
    uint32_t l2_word = 0;
    l2->read(queue.dst_va(), reinterpret_cast<uint8_t *>(&l2_word), sizeof(l2_word),
             amdgpu::Mtype::RW, kProcessId);
    out.via_l2 = l2_word;
  };

  // Writeback publishes the dirty line to backing.
  GcrOutcome wb;
  run(GcrKind::WritebackOnly, wb);
  EXPECT_EQ(wb.backing, kDirtyWord);
  // Invalidate/discard drops the dirty line without writeback; backing keeps its
  // original value and the line is no longer resident.
  GcrOutcome inv;
  run(GcrKind::InvalidateOnly, inv);
  EXPECT_EQ(inv.backing, kBackingWord);
  EXPECT_EQ(inv.via_l2, kBackingWord); // line dropped → L2 re-fetches backing.
  // No GL2 bits: no cache maintenance at all. Backing is untouched and the dirty
  // line stays resident in L2 (this is what distinguishes no-op from
  // invalidate-only: an incorrect invalidate would drop the line here too).
  GcrOutcome noop;
  run(GcrKind::Noop, noop);
  EXPECT_EQ(noop.backing, kBackingWord);
  EXPECT_EQ(noop.via_l2, kDirtyWord); // dirty line still resident in L2.
}

TEST(Gfx1250SdmaTest, CopyWaitSignalResolvesTranslatedAddresses) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  queue.poll_value() = 0;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x5a);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30) | (1u << 31);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, queue.poll_va());
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());
  packet[14] = 0x70;
  write_sdma_qword_va(packet, 15, 16, queue.signal_va());
  packet[17] = 1;
  packet[18] = 0;

  queue.submit(19);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 19u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 4);
}

// An endpoint that does not resolve YET is worth retrying, and
// ConstFillUnmappedTailDoesNotAdvance pins that. An endpoint that does not
// exist is not: the violation has already been reported to the process, and
// leaving the packet pending would wedge the queue behind work that can never
// land. This asserts the queue drains instead.
// The signal on a copy packet asserts that the destination now holds the copied
// bytes. A faulted copy deliberately leaves the destination alone, so raising
// the signal anyway hands a waiter a green light over stale data -- and it does
// so before the reported fault reaches the runtime, which is exactly the window
// where the wrong answer gets used.
// Suppressing the signal embedded in one packet is not enough: a plain
// COPY_LINEAR followed by an ordinary FENCE would still run the fence and
// publish completion for a copy that never landed. Hardware halts the engine on
// a VM fault and leaves it for the driver, so nothing queued behind the faulted
// packet may run.
// The read pointer is how the owner learns which packets are done. If it cannot
// be published the queue must stop: leaving a stale value visible means the next
// doorbell re-runs copies, fences and atomics that already executed.
TEST(Gfx1250SdmaTest, UnpublishableReadPointerHaltsInsteadOfReplayingPackets) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);

  constexpr uint32_t kCopyBytes = 64;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i + 1);
    queue.dst()[i] = 0;
  }

  // A read pointer whose page cannot be reached at all. PROT_NONE rather than
  // an unmapped hole: both are Inaccessible to the writability probe and both
  // arrive as MemoryFaultCause::NotPresent, so this is the same classification
  // reaching the queue the same way. Leaving an actual hole in the address
  // space for the remainder of the process instead crashes LeakSanitizer's
  // exit-time tracer inside its own thread-scanning code -- with a bare
  // SYS_munmap as readily as with munmap(3), so nothing in rocjitsu is on that
  // path. Reserving the page keeps the case testable under ASan.
  void *raw = mmap(nullptr, KfdProcess::kPageSize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  queue.set_read_ptr_va(reinterpret_cast<uint64_t>(raw));
  ASSERT_EQ(mprotect(raw, KfdProcess::kPageSize, PROT_NONE), 0);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  packet[2] = 0;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0) << "the copy itself must run";

  // Ring again: a queue that could not publish its progress must not replay.
  std::memset(queue.dst(), 0, kCopyBytes);
  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  for (uint32_t i = 0; i < kCopyBytes; ++i)
    ASSERT_EQ(queue.dst()[i], 0u) << "packet replayed after an unpublishable read pointer, byte "
                                  << i;

  ASSERT_EQ(mprotect(raw, KfdProcess::kPageSize, PROT_READ | PROT_WRITE), 0);
  munmap(raw, KfdProcess::kPageSize);
}

// The read pointer is how the owner learns which packets are done. If it cannot
// be published the queue must stop: leaving a stale value visible means the next
// doorbell re-runs copies, fences and atomics that already executed.
TEST(Gfx1250SdmaTest, UnwritableReadPointerHaltsInsteadOfReplayingPackets) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);

  constexpr uint32_t kCopyBytes = 64;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i + 1);
    queue.dst()[i] = 0;
  }

  // A read pointer the queue can read but not write.
  void *raw = mmap(nullptr, KfdProcess::kPageSize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  queue.set_read_ptr_va(reinterpret_cast<uint64_t>(raw));
  ASSERT_EQ(mprotect(raw, KfdProcess::kPageSize, PROT_READ), 0);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  packet[2] = 0;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0) << "the copy itself must run";

  // Ring again: a queue that could not publish its progress must not replay.
  std::memset(queue.dst(), 0, kCopyBytes);
  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  for (uint32_t i = 0; i < kCopyBytes; ++i)
    ASSERT_EQ(queue.dst()[i], 0u) << "packet replayed after an unpublishable read pointer, byte "
                                  << i;

  mprotect(raw, KfdProcess::kPageSize, PROT_READ | PROT_WRITE);
  munmap(raw, KfdProcess::kPageSize);
}

/// @brief A poll on a faulted address must halt rather than spin forever.
/// @details A poll that cannot read its address is normally right to retry --
/// waiting for a value to appear is the entire point of the packet. But a
/// faulted address never becomes readable, so the retry re-reports the same
/// violation on every doorbell and the queue never drains: an unattributable
/// hang plus an unbounded stream of faults.
TEST(Gfx1250SdmaTest, PollOnFaultedAddressHaltsInsteadOfSpinning) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);

  void *raw = mmap(nullptr, KfdProcess::kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  const uint64_t faulted_va = reinterpret_cast<uint64_t>(raw);

  queue.signal_value() = 5;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28); // 64-bit equal poll.
  write_sdma_qword_va(packet, 1, 2, faulted_va);
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  // A fence behind it that must never run.
  packet[7] = kSdmaOpFence;
  write_sdma_qword_va(packet, 8, 9, queue.signal_va());
  packet[10] = 0xDEADBEEFu;

  queue.submit(11);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.signal_value(), 5) << "a packet behind a faulted poll ran";

  // Parking the read pointer looks the same whether the queue faulted or is
  // merely waiting, so make the address satisfy the poll and ring again. A
  // waiting queue would proceed; a faulted one is over.
  ASSERT_EQ(mprotect(raw, KfdProcess::kPageSize, PROT_READ | PROT_WRITE), 0);
  const uint64_t read_idx_after_fault = queue.read_idx();
  queue.submit(11);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.signal_value(), 5) << "a faulted queue resumed once its address became readable";
  EXPECT_EQ(queue.read_idx(), read_idx_after_fault) << "the queue advanced after faulting";

  munmap(raw, KfdProcess::kPageSize);
}

TEST(Gfx1250SdmaTest, FaultedCopyHaltsTheQueueBeforeLaterPackets) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);

  void *raw = mmap(nullptr, KfdProcess::kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  const uint64_t faulted_va = reinterpret_cast<uint64_t>(raw);

  constexpr uint32_t kCopyBytes = 128;
  constexpr uint8_t kSentinel = 0xC7;
  std::memset(queue.dst(), kSentinel, kCopyBytes);
  queue.signal_value() = 5;

  auto *packet = queue.ring();
  // A plain copy from an address that does not exist.
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  packet[2] = 0;
  write_sdma_qword_va(packet, 3, 4, faulted_va);
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());
  // Followed by a separate fence that would announce it completed.
  packet[7] = kSdmaOpFence;
  write_sdma_qword_va(packet, 8, 9, queue.signal_va());
  packet[10] = 0xDEADBEEFu;

  queue.submit(11);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);

  for (uint32_t i = 0; i < kCopyBytes; ++i)
    ASSERT_EQ(queue.dst()[i], kSentinel) << "destination byte " << i;
  EXPECT_EQ(queue.signal_value(), 5)
      << "a packet behind a faulted copy must not run and publish completion";

  // The halt has to persist. One pass proves only that this servicing stopped;
  // ring the doorbell again and the queue must still refuse to run the fence.
  const uint64_t read_idx_after_fault = queue.read_idx();
  queue.submit(11);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.signal_value(), 5) << "the queue resumed after faulting";
  EXPECT_EQ(queue.read_idx(), read_idx_after_fault) << "the queue advanced after faulting";
}

TEST(Gfx1250SdmaTest, FaultedCopyDoesNotPublishItsCompletionSignal) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.set_passthrough(true);

  void *raw = mmap(nullptr, KfdProcess::kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  const uint64_t faulted_va = reinterpret_cast<uint64_t>(raw);

  constexpr uint32_t kCopyBytes = 128;
  constexpr uint8_t kSentinel = 0xC7;
  std::memset(queue.dst(), kSentinel, kCopyBytes);
  queue.poll_value() = 0;
  queue.signal_value() = 5;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30) | (1u << 31);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, queue.poll_va());
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, faulted_va);
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());
  packet[14] = 0x70;
  write_sdma_qword_va(packet, 15, 16, queue.signal_va());
  packet[17] = 1;
  packet[18] = 0;

  queue.submit(19);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.read_idx(), 19u * sizeof(uint32_t))
      << "a faulted endpoint must still retire the packet";
  EXPECT_EQ(queue.signal_value(), 5)
      << "a faulted copy must not publish completion over a destination it never wrote";
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    ASSERT_EQ(queue.dst()[i], kSentinel) << "destination byte " << i;
  }

  munmap(raw, KfdProcess::kPageSize);
}

TEST(Gfx1250SdmaTest, CopyFromFaultedAddressRetiresPacketInsteadOfWedgingQueue) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  // Passthrough is what makes an unresolved address resolvable as a host
  // address at all, so it is required to reach the faulting path.
  queue.set_passthrough(true);

  void *raw = mmap(nullptr, KfdProcess::kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw, MAP_FAILED);
  const uint64_t faulted_va = reinterpret_cast<uint64_t>(raw);

  constexpr uint32_t kCopyBytes = 128;
  // A sentinel, so the destination can prove it was left alone. Zero would be
  // indistinguishable from the fabricated bytes a faulted read leaves behind.
  constexpr uint8_t kSentinel = 0xC7;
  std::memset(queue.dst(), kSentinel, kCopyBytes);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  packet[2] = 0;
  write_sdma_qword_va(packet, 3, 4, faulted_va);
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Faulted);
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t))
      << "a faulted endpoint must retire the packet, not hold the queue for a retry";
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    ASSERT_EQ(queue.dst()[i], kSentinel)
        << "a faulted source must not overwrite the destination with fabricated bytes, byte " << i;
  }

  munmap(raw, KfdProcess::kPageSize);
}

TEST(Gfx1250SdmaTest, CompactCopyWaitPacketCopies) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  queue.poll_value() = 0;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xb7);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, queue.poll_va());
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());

  queue.submit(14);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 14u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CompactCopySignalPacketCopiesAndSignals) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xd3);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());
  packet[7] = 0x70;
  write_sdma_qword_va(packet, 8, 9, queue.signal_va());
  packet[10] = 1;
  packet[11] = 0;

  queue.submit(12);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 12u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 4);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedWaitAddressDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedWaitVa = 0x2000'0000'0000ULL;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xa5);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, kUnmappedWaitVa);
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());

  queue.submit(14, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedDstDoesNotAdvanceOrSignal) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedDstVa = 0x2000'0000'2000ULL;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x3c);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, kUnmappedDstVa);
  packet[7] = 0x70;
  write_sdma_qword_va(packet, 8, 9, queue.signal_va());
  packet[10] = 1;
  packet[11] = 0;

  queue.submit(12, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 5);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedSignalDoesNotAdvanceOrCopy) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedSignalVa = 0x2000'0000'4000ULL;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x69);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());
  packet[7] = 0x70;
  write_sdma_qword_va(packet, 8, 9, kUnmappedSignalVa);
  packet[10] = 1;
  packet[11] = 0;

  queue.submit(12, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 5);
}

TEST(Gfx1250SdmaTest, CopyLinearUnresolvedDstDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedDstVa = 0x2000'0000'3000ULL;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xc3);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, kUnmappedDstVa);

  queue.submit(7, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyLinearUnmappedTailDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  queue.unmap_dst_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
}

TEST(Gfx1250SdmaTest, CopyLinearResumesAfterTailMappingIsInstalled) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>((i * 31 + 11) & 0xff);
    queue.dst()[i] = 0;
  }
  queue.unmap_dst_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);

  queue.remap_dst_tail_page();
  EXPECT_TRUE(wait_until([&queue]() { return queue.read_idx() == 7u * sizeof(uint32_t); }))
      << "the SDMA worker did not resume after the mapping became available";
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyLinearUnmappedSourceTailDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  queue.unmap_src_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
}

TEST(Gfx1250SdmaTest, CopyLinearClippedDstAdvancesDeterministically) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint32_t kMappedBytes = 64;
  queue.clip_dst_mapping(kMappedBytes);
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x6d);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kMappedBytes), 0);
  EXPECT_TRUE(std::all_of(queue.dst() + kMappedBytes, queue.dst() + kCopyBytes,
                          [](uint8_t value) { return value == 0; }));
}

TEST(Gfx1250SdmaTest, CopyLinearTransfersMultipleScratchChunks) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, BroadcastCopyTransfersMultipleScratchChunks) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>((i * 29 + 7) & 0xff);
    queue.dst()[i] = 0;
    queue.dst2()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 27);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());
  write_sdma_qword_va(packet, 7, 8, queue.dst2_va());

  queue.submit(9);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 9u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(std::memcmp(queue.dst2(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyLinearNpdBitDoesNotDecodeAsBroadcast) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x4d);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 28);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, PollMem64ResolvesTranslatedAddress) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.poll_value() = 1;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28);
  write_sdma_qword_va(packet, 1, 2, queue.poll_va());
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);

  std::atomic_ref<uint64_t>(queue.poll_value()).store(0, std::memory_order_release);
  EXPECT_TRUE(wait_until([&queue]() { return queue.read_idx() == 8u * sizeof(uint32_t); }))
      << "the SDMA worker did not retry the translated poll";
}

TEST(Gfx1250SdmaTest, PollMem64UnresolvedAddressDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint64_t kUnmappedPollVa = 0x2000'0000'1000ULL;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28);
  write_sdma_qword_va(packet, 1, 2, kUnmappedPollVa);
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8, SdmaSubmissionWait::None);
  EXPECT_EQ(queue.last_submission_status(), amdgpu::QueueSubmissionStatus::Accepted);
  EXPECT_EQ(queue.read_idx(), 0u);
}

} // namespace
