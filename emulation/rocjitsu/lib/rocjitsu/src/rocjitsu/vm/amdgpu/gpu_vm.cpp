// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"

#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <deque>
#include <format>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace rocjitsu::amdgpu {

namespace {

std::atomic<uint64_t> next_vm_access_state_id{1};

uint64_t allocate_vm_access_state_id() {
  uint64_t id = next_vm_access_state_id.load(std::memory_order_relaxed);
  for (;;) {
    if (id == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("GPU VM cache identity exhausted");
    if (next_vm_access_state_id.compare_exchange_weak(id, id + 1, std::memory_order_relaxed))
      return id;
  }
}

} // namespace

class GpuVmRootLineage {};

class GpuVmAccessState {
public:
  class FaultReporter {
  public:
    explicit FaultReporter(std::function<void(uint64_t, VmAccessKind)> callback)
        : callback_(std::move(callback)) {}

    explicit operator bool() const { return static_cast<bool>(callback_); }

    void operator()(uint64_t address, VmAccessKind access) const {
      // A const std::function can invoke a mutable target. Every generation
      // shares this guard along with the target; a generation's shared access
      // lease does not exclude other readers or pinned generations. Recursive
      // acquisition allows same-thread callback reentry through independent
      // access states; it does not make a state's revocation lease recursive.
      std::lock_guard lock(mutex_);
      callback_(address, access);
    }

  private:
    const std::function<void(uint64_t, VmAccessKind)> callback_;
    mutable std::recursive_mutex mutex_;
  };

  GpuVmAccessState(std::shared_ptr<AddressSpaceTranslator> translator = {},
                   std::shared_ptr<PhysicalMemoryAccess> physical_memory = {},
                   std::shared_ptr<const FaultReporter> fault_reporter = {},
                   std::shared_ptr<std::atomic<uint64_t>> cache_write_sequence = {},
                   std::shared_ptr<const GpuVmRootLineage> root_lineage = {})
      : translator(std::move(translator)), physical_memory(std::move(physical_memory)),
        fault_reporter(std::move(fault_reporter)), cache_identity(allocate_vm_access_state_id()),
        cache_write_sequence(cache_write_sequence ? std::move(cache_write_sequence)
                                                  : std::make_shared<std::atomic<uint64_t>>(0)),
        root_lineage(root_lineage ? std::move(root_lineage)
                                  : std::make_shared<GpuVmRootLineage>()) {}

  // One shared lifetime per snapshot; copying a snapshot must not copy the
  // fault callback or separately contend on each backing object's reference count.
  const std::shared_ptr<AddressSpaceTranslator> translator;
  const std::shared_ptr<PhysicalMemoryAccess> physical_memory;
  // Generations share the callable too: pinning must not copy its target while
  // a fault callback is running under another generation's shared lease.
  const std::shared_ptr<const FaultReporter> fault_reporter;
  const uint64_t cache_identity;
  const std::shared_ptr<std::atomic<uint64_t>> cache_write_sequence;
  // Invalidation retains this identity; an explicit backing/root replacement
  // rotates it even if the caller reuses the same translator objects.
  const std::shared_ptr<const GpuVmRootLineage> root_lineage;
  const std::shared_ptr<VmTransferEffectGate> effect_gate =
      std::make_shared<VmTransferEffectGate>();
  mutable util::DistributedSharedMutex mutex;
  // Written under mutex; atomic so copied policy hits can check retirement without a lease.
  std::atomic<bool> valid{true};
};

class GpuVmBindingState {
public:
  std::atomic<uint32_t> references{0};
};

namespace {

std::atomic<uint64_t> next_vm_instance_id{1};

struct VmAccessBatchSnapshot {
  const GpuVm *vm = nullptr;
  uint64_t vm_instance = 0;
  AddressSpaceHandle handle;
  GpuVmAccess access;
  bool routed = false;
};

thread_local uint32_t vm_access_batch_depth = 0;
thread_local std::deque<VmAccessBatchSnapshot> vm_access_batch_snapshots;

constexpr uint64_t kGfx12PteValid = uint64_t{1} << 0;
constexpr uint64_t kGfx12PteSystem = uint64_t{1} << 1;
constexpr uint64_t kGfx12PteSnooped = uint64_t{1} << 2;
constexpr uint64_t kGfx12PteExecutable = uint64_t{1} << 4;
constexpr uint64_t kGfx12PteReadable = uint64_t{1} << 5;
constexpr uint64_t kGfx12PteWriteable = uint64_t{1} << 6;
constexpr uint64_t kGfx12PdePte = uint64_t{1} << 63;
constexpr uint64_t kGfx12PageBytes = 4096;
constexpr std::array<uint32_t, 5> kGfx12PageTableShifts = {48, 39, 30, 21, 12};

constexpr uint64_t gfx12_virtual_address_mask(Gfx12VmConfig config) {
  const auto bits = static_cast<uint8_t>(config.virtual_address_width);
  return (uint64_t{1} << bits) - 1;
}

constexpr uint64_t gfx12_physical_address_mask(Gfx12VmConfig config) {
  const auto bits = static_cast<uint8_t>(config.physical_address_width);
  return (uint64_t{1} << bits) - 1;
}

constexpr uint64_t gfx12_pte_address_mask(Gfx12VmConfig config) {
  constexpr uint64_t kPageOffsetMask = (uint64_t{1} << 12) - 1;
  return gfx12_physical_address_mask(config) & ~kPageOffsetMask;
}

constexpr uint64_t gfx12_pde_address_mask(Gfx12VmConfig config) {
  constexpr uint64_t kPdeFlagMask = (uint64_t{1} << 6) - 1;
  return gfx12_physical_address_mask(config) & ~kPdeFlagMask;
}

bool valid_gart_config(const GartConfig &config, Gfx12VmConfig vm_config) {
  const uint64_t physical_mask = gfx12_physical_address_mask(vm_config);
  const uint64_t page_table_address_mask = gfx12_pte_address_mask(vm_config);
  constexpr uint64_t kPageTableRootFlags = kGfx12PteValid | kGfx12PteSystem | kGfx12PteSnooped;
  const uint64_t page_table_address = config.page_table_base & page_table_address_mask;
  const bool page_table_present =
      page_table_address != 0 || (config.page_table_base & kGfx12PteValid) != 0;
  const uint64_t virtual_mask = gfx12_virtual_address_mask(vm_config);
  if (!page_table_present ||
      (config.page_table_base & ~(page_table_address_mask | kPageTableRootFlags)) != 0) {
    return false;
  }
  if (config.aperture_start > config.aperture_end ||
      (config.aperture_start & (kGfx12PageBytes - 1)) != 0 ||
      (config.aperture_end & (kGfx12PageBytes - 1)) != kGfx12PageBytes - 1 ||
      config.aperture_end > virtual_mask) {
    return false;
  }

  const uint64_t page_count = (config.aperture_end - config.aperture_start) / kGfx12PageBytes + 1;
  const uint64_t last_pte_offset = (page_count - 1) * sizeof(uint64_t);
  return last_pte_offset <= physical_mask - page_table_address;
}

Mtype gfx12_mtype(uint64_t entry) {
  switch ((entry >> 54) & 0x3) {
  case 1:
    return Mtype::CC;
  case 3:
    return Mtype::UC;
  case 0:
  case 2:
  default:
    return Mtype::RW;
  }
}

Gfx12PteMtype gfx12_pte_mtype(uint64_t entry) {
  return static_cast<Gfx12PteMtype>((entry >> 54) & 0x3);
}

template <typename Span>
VmTransferStep transfer_translated_span(const AddressSpaceTranslator &translator,
                                        PhysicalMemoryAccess &memory, uint64_t address, Span bytes,
                                        VmAccessKind access) {
  const VmTranslationResult translated =
      access == VmAccessKind::Execute ? translator.probe_translation(address, bytes.size(), access)
                                      : translator.translate(address, bytes.size(), access);
  if (!translated)
    return {.outcome = translated.outcome, .report_translation_fault = true};
  if (translated.translation.contiguous_bytes == 0)
    return {.outcome = VmAccessOutcome::Malformed};
  const auto chunk = static_cast<std::size_t>(
      std::min<uint64_t>(bytes.size(), translated.translation.contiguous_bytes));
  const VmAccessOutcome outcome = [&] {
    if constexpr (std::is_const_v<typename Span::element_type>)
      return memory.write(translated.translation.domain, translated.translation.address,
                          bytes.first(chunk));
    else
      return memory.read_for_access(translated.translation.domain, translated.translation.address,
                                    bytes.first(chunk), access);
  }();
  return {.completed_bytes = outcome == VmAccessOutcome::Complete ? chunk : 0, .outcome = outcome};
}

template <typename Span>
VmAccessOutcome access_translated(const AddressSpaceTranslator &translator,
                                  PhysicalMemoryAccess &memory, uint64_t address, Span bytes,
                                  std::size_t &completed_bytes, VmAccessKind access,
                                  const GpuVmAccessState::FaultReporter *fault_reporter = nullptr,
                                  std::optional<VmTransferToken> *rejected_deferred = nullptr,
                                  const std::shared_ptr<VmTransferEffectGate> &effect_gate = {}) {
  if (completed_bytes > bytes.size())
    return VmAccessOutcome::Malformed;
  while (completed_bytes < bytes.size()) {
    auto deferred = [&] {
      if constexpr (std::is_const_v<typename Span::element_type>)
        return translator.prepare_deferred_write(memory, address + completed_bytes,
                                                 bytes.subspan(completed_bytes), effect_gate);
      else
        return translator.prepare_deferred_read(
            memory, address + completed_bytes, bytes.size() - completed_bytes, access, effect_gate);
    }();
    if (deferred) {
      if (rejected_deferred)
        *rejected_deferred = std::move(*deferred);
      return VmAccessOutcome::Malformed;
    }
    const VmTransferStep step = [&] {
      if constexpr (std::is_const_v<typename Span::element_type>)
        return translator.write_step(memory, address + completed_bytes,
                                     bytes.subspan(completed_bytes));
      else
        return translator.read_step(memory, address + completed_bytes,
                                    bytes.subspan(completed_bytes), access);
    }();
    if (step.outcome != VmAccessOutcome::Complete) {
      if (step.report_translation_fault && fault_reporter && *fault_reporter &&
          step.outcome != VmAccessOutcome::Unavailable && step.outcome != VmAccessOutcome::Revoked)
        (*fault_reporter)(address + completed_bytes, access);
      return step.outcome;
    }
    if (step.completed_bytes == 0 || step.completed_bytes > bytes.size() - completed_bytes)
      return VmAccessOutcome::Malformed;
    completed_bytes += step.completed_bytes;
  }
  return VmAccessOutcome::Complete;
}

} // namespace

bool VmTransferEffectGate::enter() noexcept {
  uint64_t observed = state_.load(std::memory_order_acquire);
  while ((observed & kRevoked) == 0) {
    if (observed == kRevoked - 1)
      return false;
    if (state_.compare_exchange_weak(observed, observed + 1, std::memory_order_acquire,
                                     std::memory_order_acquire))
      return true;
  }
  return false;
}

VmTransferEffectGate::Permit::~Permit() {
  if (gate_ != nullptr)
    gate_->leave();
}

std::optional<VmTransferEffectGate::Permit> VmTransferEffectGate::try_acquire() noexcept {
  if (!enter())
    return std::nullopt;
  return Permit(this);
}

void VmTransferEffectGate::leave() noexcept {
  const uint64_t previous = state_.fetch_sub(1, std::memory_order_acq_rel);
  assert((previous & ~kRevoked) != 0);
  if ((previous & kRevoked) != 0 && (previous & ~kRevoked) == 1)
    state_.notify_all();
}

void VmTransferEffectGate::close() noexcept {
  state_.fetch_or(kRevoked, std::memory_order_acq_rel);
}

bool VmTransferEffectGate::drained() const noexcept {
  return (state_.load(std::memory_order_acquire) & ~kRevoked) == 0;
}

void VmTransferEffectGate::revoke() {
  close();
  uint64_t observed = state_.load(std::memory_order_acquire);
  while ((observed & ~kRevoked) != 0) {
    state_.wait(observed, std::memory_order_acquire);
    observed = state_.load(std::memory_order_acquire);
  }
}

VmTransferToken::VmTransferToken(std::unique_ptr<VmDeferredTransfer> operation, std::size_t bytes,
                                 VmDeferredWriteMaintenance write_maintenance, bool peer_origin)
    : operation_(std::move(operation)), peer_origin_(peer_origin), read_result_(bytes),
      bytes_(bytes), write_maintenance_(write_maintenance) {}

VmTransferToken::~VmTransferToken() { cancel(); }

VmTransferToken::VmTransferToken(VmTransferToken &&other) noexcept
    : operation_(std::move(other.operation_)), peer_origin_(other.peer_origin_),
      completed_read_(std::move(other.completed_read_)),
      read_result_(std::move(other.read_result_)), bytes_(other.bytes_),
      total_bytes_(other.total_bytes_), offset_(other.offset_), address_(other.address_),
      kind_(other.kind_), state_(std::move(other.state_)),
      write_maintenance_(
          std::exchange(other.write_maintenance_, VmDeferredWriteMaintenance::Caller)),
      submitted_(other.submitted_) {}

VmTransferToken &VmTransferToken::operator=(VmTransferToken &&other) noexcept {
  if (this == &other)
    return *this;
  cancel();
  operation_ = std::move(other.operation_);
  peer_origin_ = other.peer_origin_;
  completed_read_ = std::move(other.completed_read_);
  read_result_ = std::move(other.read_result_);
  bytes_ = other.bytes_;
  total_bytes_ = other.total_bytes_;
  offset_ = other.offset_;
  address_ = other.address_;
  kind_ = other.kind_;
  state_ = std::move(other.state_);
  write_maintenance_ = std::exchange(other.write_maintenance_, VmDeferredWriteMaintenance::Caller);
  submitted_ = other.submitted_;
  return *this;
}

void VmTransferToken::cancel() noexcept {
  if (operation_)
    operation_->cancel();
  operation_.reset();
  peer_origin_ = false;
  completed_read_.reset();
  read_result_.clear();
  bytes_ = 0;
  total_bytes_ = 0;
  offset_ = 0;
  address_ = 0;
  state_ = nullptr;
  write_maintenance_ = VmDeferredWriteMaintenance::Caller;
  submitted_ = false;
}

VmTransferStep AddressSpaceTranslator::read_step(PhysicalMemoryAccess &memory, uint64_t address,
                                                 std::span<std::byte> bytes,
                                                 VmAccessKind access) const {
  return transfer_translated_span(*this, memory, address, bytes, access);
}

VmTransferStep AddressSpaceTranslator::write_step(PhysicalMemoryAccess &memory, uint64_t address,
                                                  std::span<const std::byte> bytes) const {
  return transfer_translated_span(*this, memory, address, bytes, VmAccessKind::Write);
}

GpuVmAccessBatchGuard::GpuVmAccessBatchGuard() {
  if (vm_access_batch_depth++ == 0) {
    vm_access_batch_snapshots.clear();
  }
}

GpuVmAccessBatchGuard::~GpuVmAccessBatchGuard() {
  assert(vm_access_batch_depth != 0);
  if (--vm_access_batch_depth != 0)
    return;
  vm_access_batch_snapshots.clear();
}

bool GpuVmAccessBatchGuard::active() { return vm_access_batch_depth != 0; }

const GpuVmAccess *GpuVmAccessBatchGuard::find_snapshot(const GpuVm *vm,
                                                        AddressSpaceHandle handle) {
  if (vm_access_batch_depth == 0)
    return nullptr;
  const auto entry = std::ranges::find_if(vm_access_batch_snapshots, [&](const auto &candidate) {
    return candidate.vm == vm && candidate.vm_instance == vm->instance_id_ &&
           candidate.handle == handle && candidate.access.is_current();
  });
  return entry != vm_access_batch_snapshots.end() ? &entry->access : nullptr;
}

const GpuVmAccess *GpuVmAccessBatchGuard::find_snapshot_vmid(const GpuVm *vm, uint32_t vmid) {
  if (vm_access_batch_depth == 0)
    return nullptr;
  const auto entry = std::ranges::find_if(vm_access_batch_snapshots, [&](const auto &candidate) {
    return candidate.vm == vm && candidate.vm_instance == vm->instance_id_ && candidate.routed &&
           candidate.access.info().vmid == vmid && candidate.access.is_current();
  });
  return entry != vm_access_batch_snapshots.end() ? &entry->access : nullptr;
}

void GpuVmAccessBatchGuard::retain_snapshot(const GpuVm *vm, AddressSpaceHandle handle,
                                            const GpuVmAccess &access, bool routed) {
  if (vm_access_batch_depth != 0)
    vm_access_batch_snapshots.push_back({vm, vm->instance_id_, handle, access, routed});
}

GpuVmBindingLease::GpuVmBindingLease(GpuVmBindingLease &&other) noexcept
    : state_(std::move(other.state_)), handle_(other.handle_), info_(other.info_) {
  other.handle_ = {};
  other.info_ = {};
}

GpuVmBindingLease &GpuVmBindingLease::operator=(GpuVmBindingLease &&other) noexcept {
  if (this == &other)
    return *this;
  release();
  state_ = std::move(other.state_);
  handle_ = other.handle_;
  info_ = other.info_;
  other.handle_ = {};
  other.info_ = {};
  return *this;
}

GpuVmBindingLease::~GpuVmBindingLease() { release(); }

void GpuVmBindingLease::release() noexcept {
  if (state_ == nullptr)
    return;
  const uint32_t previous = state_->references.fetch_sub(1, std::memory_order_acq_rel);
  assert(previous != 0);
  (void)previous;
  state_.reset();
  handle_ = {};
  info_ = {};
}

GpuVm::GpuVm(Gfx12VmConfig gfx12_config)
    : instance_id_(next_vm_instance_id.fetch_add(1, std::memory_order_relaxed)),
      gfx12_config_(gfx12_config) {}

bool GpuVm::protect_controlled_cache_backing(AddressSpaceHandle handle, uint64_t address,
                                             uint64_t size) {
  if (!handle || size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return false;
  // Keep the binding alive until the policy is visible. Cache sinks never
  // acquire this registry lock when they consult the published snapshot.
  std::shared_lock registry_lock(mutex_);
  const Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->vmid == 0)
    return false;
  std::lock_guard policy_lock(controlled_backing_mutex_);
  auto replacement = std::make_shared<std::vector<ControlledCacheBacking>>();
  const auto current = controlled_backings_.load(std::memory_order_acquire);
  if (current) {
    replacement->reserve(current->size() + 1);
    for (const ControlledCacheBacking &entry : *current) {
      if (!entry.active->load(std::memory_order_acquire) || entry.binding.expired())
        continue;
      if (entry.handle == handle && entry.first == address && entry.last == address + size - 1)
        return true;
      replacement->push_back(entry);
    }
  }
  replacement->push_back({handle, binding->vmid, address, address + size - 1,
                          binding->binding_state, std::make_shared<std::atomic<bool>>(true)});
  controlled_backings_.store(std::move(replacement), std::memory_order_release);
  return true;
}

void GpuVm::retire_controlled_cache_backing(AddressSpaceHandle handle, uint64_t address,
                                            uint64_t size) noexcept {
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return;
  const auto current = controlled_backings_.load(std::memory_order_acquire);
  if (!current)
    return;
  for (const ControlledCacheBacking &entry : *current)
    if (entry.handle == handle && entry.first == address && entry.last == address + size - 1)
      entry.active->store(false, std::memory_order_release);
}

bool GpuVm::controlled_cache_backing_overlaps(uint32_t vmid, uint64_t address,
                                              uint64_t size) const noexcept {
  if (vmid == 0 || size == 0)
    return false;
  if (size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return true;
  const auto current = controlled_backings_.load(std::memory_order_acquire);
  if (!current)
    return false;
  const uint64_t last = address + size - 1;
  for (const ControlledCacheBacking &entry : *current)
    if (entry.vmid == vmid && address <= entry.last && entry.first <= last &&
        entry.active->load(std::memory_order_acquire) && !entry.binding.expired())
      return true;
  return false;
}

VmAccessOutcome read_translated(const AddressSpaceTranslator &translator,
                                PhysicalMemoryAccess &memory, uint64_t address,
                                std::span<std::byte> bytes) {
  MemoryIngressAdmission pt_admitted =
      bytes.empty() ? MemoryIngressAdmission{} : memory.admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return pt_admitted.outcome;
  std::size_t completed_bytes = 0;
  return access_translated(translator, memory, address, bytes, completed_bytes, VmAccessKind::Read);
}

VmAccessOutcome write_translated(const AddressSpaceTranslator &translator,
                                 PhysicalMemoryAccess &memory, uint64_t address,
                                 std::span<const std::byte> bytes) {
  MemoryIngressAdmission pt_admitted =
      bytes.empty() ? MemoryIngressAdmission{} : memory.admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return pt_admitted.outcome;
  std::size_t completed_bytes = 0;
  return access_translated(translator, memory, address, bytes, completed_bytes,
                           VmAccessKind::Write);
}

VmTranslationResult IdentityAddressSpaceTranslator::translate(uint64_t address, std::size_t size,
                                                              VmAccessKind access) const {
  (void)access;
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  const uint64_t page_offset = address & (kPageSize - 1);
  return {.outcome = VmAccessOutcome::Complete,
          .translation = {.domain = VmMemoryDomain::Local,
                          .address = address,
                          .contiguous_bytes = kPageSize - page_offset,
                          .mtype = Mtype::RW,
                          .permissions = {.readable = true, .writable = true, .executable = true}}};
}

bool GpuVmAccess::is_current() const {
  return access_state_ != nullptr && access_state_->valid.load(std::memory_order_acquire);
}

uint64_t GpuVmAccess::cache_identity() const {
  return access_state_ != nullptr ? access_state_->cache_identity : 0;
}

bool GpuVmAccess::supports_translated_local_cache() const {
  if (access_state_ == nullptr)
    return false;
  std::shared_lock lock(access_state_->mutex);
  return access_state_->valid && access_state_->physical_memory != nullptr &&
         access_state_->physical_memory->supports_translated_local_cache();
}

uint64_t GpuVmAccess::cache_write_sequence() const {
  return access_state_ != nullptr
             ? access_state_->cache_write_sequence->load(std::memory_order_acquire)
             : std::numeric_limits<uint64_t>::max();
}

bool GpuVmAccess::same_root_lineage(const GpuVmAccess &other) const {
  return access_state_ != nullptr && other.access_state_ != nullptr &&
         address_space_ == other.address_space_ &&
         access_state_->root_lineage == other.access_state_->root_lineage;
}

std::shared_ptr<VmTransferEffectGate> GpuVmAccess::retain_effect_gate() const {
  return access_state_ != nullptr ? access_state_->effect_gate : nullptr;
}

MemoryIngressAdmission GpuVmAccess::admit_pt_rewalk() const {
  if (access_state_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .permit = {}};
  if (!access_state_->valid.load(std::memory_order_acquire))
    return {.outcome = VmAccessOutcome::Revoked, .permit = {}};
  if (access_state_->physical_memory == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .permit = {}};
  return access_state_->physical_memory->admit_pt_rewalk();
}

MemoryIngressAdmission GpuVmAccess::admit_pt_rewalk(const MemoryPtEpoch &expected) const {
  if (access_state_ == nullptr || !access_state_->valid.load(std::memory_order_acquire))
    return {.outcome = VmAccessOutcome::Revoked, .permit = {}};
  if (access_state_->physical_memory == nullptr)
    return {.outcome = VmAccessOutcome::Revoked, .permit = {}};
  return access_state_->physical_memory->admit_pt_rewalk(expected);
}

VmAccessAdmission GpuVmAccess::admit_access(uint64_t address, std::size_t size, VmAccessKind access,
                                            bool report_fault) const {
  if (access_state_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable};
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return {.outcome = VmAccessOutcome::Revoked};
  if (access_state_->translator == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable};
  const VmAccessAdmission admission =
      access_state_->translator->admit_access(address, size, access);
  if (admission && (access == VmAccessKind::Write || access == VmAccessKind::Atomic)) {
    auto &sequence = *access_state_->cache_write_sequence;
    uint64_t previous = sequence.load(std::memory_order_relaxed);
    while (previous != std::numeric_limits<uint64_t>::max() &&
           !sequence.compare_exchange_weak(previous, previous + 1, std::memory_order_release,
                                           std::memory_order_relaxed)) {
    }
  }
  if (report_fault)
    report_terminal_fault(address, access, admission.outcome);
  return admission;
}

VmAccessAdmission GpuVmAccess::query_cache_write_grant(uint64_t address, std::size_t size) const {
  if (access_state_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable};
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return {.outcome = VmAccessOutcome::Revoked};
  if (access_state_->translator == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable};
  return access_state_->translator->query_cache_write_grant(address, size);
}

VmTranslationResult GpuVmAccess::query_translation(uint64_t address, std::size_t size,
                                                   VmAccessKind access) const {
  if (access_state_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return {.outcome = pt_admitted.outcome, .translation = {}};
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return {.outcome = VmAccessOutcome::Revoked, .translation = {}};
  if (access_state_->translator == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  return access_state_->translator->translate(address, size, access);
}

VmTranslationResult GpuVmAccess::translate(uint64_t address, std::size_t size,
                                           VmAccessKind access) const {
  if (access_state_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return {.outcome = pt_admitted.outcome, .translation = {}};
  std::shared_lock state_lock(access_state_->mutex);
  VmTranslationResult translated = !access_state_->valid
                                       ? VmTranslationResult{
                                             .outcome = VmAccessOutcome::Revoked,
                                             .translation = {},
                                         }
                                   : access_state_->translator == nullptr
                                       ? VmTranslationResult{
                                             .outcome = VmAccessOutcome::Unavailable,
                                             .translation = {},
                                         }
                                       : access_state_->translator->translate(address, size, access);
  report_terminal_fault(address, access, translated.outcome);
  return translated;
}

VmAccessOutcome GpuVmAccess::validate_deferred_system_span(uint64_t gpu_va, uint64_t iova,
                                                           std::size_t size, VmAccessKind access,
                                                           const VmTranslation &selected) const {
  if (access_state_ == nullptr || size == 0 ||
      (access != VmAccessKind::Read && access != VmAccessKind::Write) ||
      selected.domain != VmMemoryDomain::System || selected.address != iova ||
      selected.contiguous_bytes < size || !selected.permissions.allows(access))
    return VmAccessOutcome::Revoked;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || access_state_->translator == nullptr)
    return VmAccessOutcome::Revoked;
  const VmTranslationResult translated = access_state_->translator->translate(gpu_va, size, access);
  if (translated.outcome == VmAccessOutcome::Unavailable)
    return VmAccessOutcome::Unavailable;
  return translated && translated.translation.domain == VmMemoryDomain::System &&
                 translated.translation.address == iova &&
                 translated.translation.contiguous_bytes == selected.contiguous_bytes &&
                 translated.translation.mtype == selected.mtype &&
                 translated.translation.gfx12_pte_mtype == selected.gfx12_pte_mtype &&
                 translated.translation.permissions == selected.permissions
             ? VmAccessOutcome::Complete
             : VmAccessOutcome::Revoked;
}

VmDataRoute GpuVmAccess::classify_data_route(uint64_t address, std::size_t size,
                                             VmAccessKind access, bool report_fault) const {
  if (access_state_ == nullptr)
    return VmDataRoute::Faulted;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || access_state_->translator == nullptr)
    return VmDataRoute::Faulted;
  const VmDataRoute route = access_state_->translator->classify_data_route(address, size, access);
  if (report_fault && route == VmDataRoute::Faulted)
    report_terminal_fault(address, access, VmAccessOutcome::Faulted);
  return route;
}

bool GpuVmAccess::try_write_private_dwords(std::span<const VmRamDwordStore> stores,
                                           Mtype instruction_mtype, Mtype expected_mtype) const {
  if (stores.size() < 2 || stores.size() > VmRamDwordStore::kMaxBatch || !access_state_)
    return false;
  const uint64_t page = stores.front().address / IdentityAddressSpaceTranslator::kPageSize;
  for (const auto &[address, source] : stores)
    if (!source || (address & 3) || address / IdentityAddressSpaceTranslator::kPageSize != page)
      return false;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return false;
  std::shared_lock state_lock(access_state_->mutex);
  return access_state_->valid && access_state_->translator && access_state_->physical_memory &&
         access_state_->translator->try_write_private_dwords(
             *access_state_->physical_memory, stores, instruction_mtype, expected_mtype);
}

bool GpuVmAccess::try_read_uncached_ram(uint64_t address, std::span<std::byte> bytes) const {
  if (!access_state_)
    return false;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return false;
  std::shared_lock state_lock(access_state_->mutex);
  return access_state_->valid && access_state_->translator && access_state_->physical_memory &&
         access_state_->translator->try_read_uncached_ram(*access_state_->physical_memory, address,
                                                          bytes);
}

bool GpuVmAccess::revoked() const {
  return access_state_ != nullptr && !access_state_->valid.load(std::memory_order_acquire);
}

bool GpuVmAccess::try_read_contiguous(uint64_t address, std::span<std::byte> bytes) const {
  if (!access_state_)
    return false;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return false;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || !access_state_->translator || !access_state_->physical_memory)
    return false;
  // Bypass the fault-reporting translate() wrapper for this optional span.
  const auto translation =
      access_state_->translator->translate(address, bytes.size(), VmAccessKind::Read);
  return translation && translation.translation.contiguous_bytes >= bytes.size() &&
         access_state_->physical_memory->try_read_contiguous(
             translation.translation.domain, translation.translation.address, bytes);
}

bool GpuVmAccess::try_write_contiguous(uint64_t address, std::span<const std::byte> bytes) const {
  if (!access_state_)
    return false;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return false;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || !access_state_->translator || !access_state_->physical_memory)
    return false;
  // Bypass the fault-reporting translate() wrapper for this optional span.
  const auto translation =
      access_state_->translator->translate(address, bytes.size(), VmAccessKind::Write);
  return translation && translation.translation.contiguous_bytes >= bytes.size() &&
         access_state_->physical_memory->try_write_contiguous(
             translation.translation.domain, translation.translation.address, bytes);
}

std::optional<Mtype> GpuVmAccess::query_mtype(uint64_t address) const {
  if (access_state_ == nullptr)
    return std::nullopt;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return std::nullopt;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || access_state_->translator == nullptr)
    return std::nullopt;
  return access_state_->translator->query_mtype(address);
}

std::optional<Mtype> GpuVmAccess::query_mtype(uint64_t address, VmMtypeCache &cache) const {
  if (access_state_ == nullptr)
    return std::nullopt;
  // A hit uses copied policy only. Retirement invalidates the snapshot's state
  // before releasing any frontend storage; misses remain under its lease.
  // Compare generation ownership without acquiring another strong reference.
  const bool same_generation = cache.snapshots_ &&
                               !cache.access_state_.owner_before(access_state_) &&
                               !access_state_.owner_before(cache.access_state_);
  const std::size_t index = VmMtypeCache::snapshot_index(address);
  if (same_generation && access_state_->valid.load(std::memory_order_acquire)) {
    const auto &snapshot = (*cache.snapshots_)[index];
    if (snapshot.unchanged(address))
      return snapshot.mtype;
  }
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return std::nullopt;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || access_state_->translator == nullptr)
    return std::nullopt;
  if (!same_generation) {
    // Complete first-use allocation before publishing the generation. Reuse the
    // existing table on generation changes, dropping every old mutation token.
    if (!cache.snapshots_)
      cache.snapshots_ = std::make_unique<VmMtypeCache::Snapshots>();
    else
      *cache.snapshots_ = {};
    cache.access_state_ = access_state_;
  }
  auto &snapshot = (*cache.snapshots_)[index];
  snapshot = access_state_->translator->snapshot_mtype(address);
  return snapshot.mtype;
}

bool GpuVmAccess::cached_private_uc_hint(uint64_t address, const VmMtypeCache &cache) const {
  if (!access_state_ || !cache.snapshots_)
    return false;
  const bool same_generation = !cache.access_state_.owner_before(access_state_) &&
                               !access_state_.owner_before(cache.access_state_);
  const auto &snapshot = (*cache.snapshots_)[VmMtypeCache::snapshot_index(address)];
  return same_generation && access_state_->valid.load(std::memory_order_acquire) &&
         snapshot.may_batch_private_uc && snapshot.unchanged(address);
}

std::optional<Mtype> GpuVmAccess::cached_private_ram_mtype(uint64_t address,
                                                           const VmMtypeCache &cache) const {
  if (!access_state_ || !cache.snapshots_)
    return std::nullopt;
  const bool same_generation = !cache.access_state_.owner_before(access_state_) &&
                               !access_state_.owner_before(cache.access_state_);
  const auto &snapshot = (*cache.snapshots_)[VmMtypeCache::snapshot_index(address)];
  if (same_generation && access_state_->valid.load(std::memory_order_acquire) &&
      snapshot.may_batch_private_ram && snapshot.unchanged(address))
    return snapshot.mtype;
  return std::nullopt;
}

VmAccessOutcome GpuVmAccess::query_access(uint64_t address, std::size_t size,
                                          VmAccessKind access) const {
  return probe_impl(address, size, access, false);
}

VmAccessOutcome GpuVmAccess::probe(uint64_t address, std::size_t size, VmAccessKind access) const {
  return probe_impl(address, size, access, true);
}

VmAccessOutcome GpuVmAccess::probe_impl(uint64_t address, std::size_t size, VmAccessKind access,
                                        bool report_fault) const {
  if (access_state_ == nullptr)
    return VmAccessOutcome::Unavailable;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return pt_admitted.outcome;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return VmAccessOutcome::Revoked;
  if (access_state_->translator == nullptr)
    return VmAccessOutcome::Unavailable;
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address) {
    if (report_fault)
      report_terminal_fault(address, access, VmAccessOutcome::Malformed);
    return VmAccessOutcome::Malformed;
  }

  std::size_t completed_bytes = 0;
  while (completed_bytes < size) {
    const uint64_t current_address = address + completed_bytes;
    const VmTranslationResult translated = access_state_->translator->probe_translation(
        current_address, size - completed_bytes, access);
    if (!translated) {
      if (report_fault)
        report_terminal_fault(current_address, access, translated.outcome);
      return translated.outcome;
    }
    if (translated.translation.contiguous_bytes == 0) {
      if (report_fault)
        report_terminal_fault(current_address, access, VmAccessOutcome::Malformed);
      return VmAccessOutcome::Malformed;
    }
    completed_bytes += static_cast<std::size_t>(
        std::min<uint64_t>(size - completed_bytes, translated.translation.contiguous_bytes));
  }
  return VmAccessOutcome::Complete;
}

void GpuVmAccess::report_terminal_fault(uint64_t address, VmAccessKind access,
                                        VmAccessOutcome outcome) const {
  if (access_state_ && access_state_->fault_reporter && *access_state_->fault_reporter &&
      outcome != VmAccessOutcome::Complete && outcome != VmAccessOutcome::Unavailable &&
      outcome != VmAccessOutcome::Revoked) {
    (*access_state_->fault_reporter)(address, access);
  }
}

VmAccessOutcome GpuVmAccess::read(uint64_t address, std::span<std::byte> bytes,
                                  VmAccessKind access) const {
  std::size_t completed_bytes = 0;
  return read(address, bytes, completed_bytes, access);
}

VmAccessOutcome GpuVmAccess::read(uint64_t address, std::span<std::byte> bytes,
                                  std::size_t &completed_bytes, VmAccessKind access) const {
  if (access != VmAccessKind::Read && access != VmAccessKind::Execute)
    return VmAccessOutcome::Malformed;
  if (access_state_ == nullptr)
    return VmAccessOutcome::Unavailable;
  MemoryIngressAdmission pt_admitted =
      completed_bytes >= bytes.size() ? MemoryIngressAdmission{} : admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return pt_admitted.outcome;
  std::optional<VmTransferToken> rejected_deferred;
  VmAccessOutcome outcome;
  {
    std::shared_lock state_lock(access_state_->mutex);
    if (!access_state_->valid)
      return VmAccessOutcome::Revoked;
    if (access_state_->translator == nullptr || access_state_->physical_memory == nullptr)
      return VmAccessOutcome::Unavailable;
    outcome =
        access_translated(*access_state_->translator, *access_state_->physical_memory, address,
                          bytes, completed_bytes, access, access_state_->fault_reporter.get(),
                          &rejected_deferred, access_state_->effect_gate);
  }
  return outcome;
}

VmAccessOutcome GpuVmAccess::write(uint64_t address, std::span<const std::byte> bytes) const {
  std::size_t completed_bytes = 0;
  return write(address, bytes, completed_bytes);
}

VmAccessOutcome GpuVmAccess::write(uint64_t address, std::span<const std::byte> bytes,
                                   std::size_t &completed_bytes) const {
  if (access_state_ == nullptr)
    return VmAccessOutcome::Unavailable;
  MemoryIngressAdmission pt_admitted =
      completed_bytes >= bytes.size() ? MemoryIngressAdmission{} : admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return pt_admitted.outcome;
  std::optional<VmTransferToken> rejected_deferred;
  VmAccessOutcome outcome;
  {
    std::shared_lock state_lock(access_state_->mutex);
    if (!access_state_->valid)
      return VmAccessOutcome::Revoked;
    if (access_state_->translator == nullptr || access_state_->physical_memory == nullptr)
      return VmAccessOutcome::Unavailable;
    outcome = access_translated(*access_state_->translator, *access_state_->physical_memory,
                                address, bytes, completed_bytes, VmAccessKind::Write,
                                access_state_->fault_reporter.get(), &rejected_deferred,
                                access_state_->effect_gate);
  }
  return outcome;
}

template <typename Span>
VmAccessOutcome GpuVmAccess::transfer_tokenized(
    uint64_t address, Span bytes, std::size_t &completed_bytes, VmTransferToken &token,
    VmAccessKind access, bool allow_deferred, const VmTransferMaintenance &maintenance,
    const VmTransferIngress &ingress, const VmPrivateUcGttCopySource &copy_source) const {
  if (access == VmAccessKind::Execute || completed_bytes > bytes.size()) {
    token.cancel();
    return VmAccessOutcome::Malformed;
  }
  if (bytes.size() != 0 && bytes.size() - 1 > std::numeric_limits<uint64_t>::max() - address) {
    report_terminal_fault(address, access, VmAccessOutcome::Malformed);
    token.cancel();
    return VmAccessOutcome::Malformed;
  }
  if (token.pending() && completed_bytes == bytes.size()) {
    token.cancel();
    return VmAccessOutcome::Malformed;
  }
  if (access_state_ == nullptr) {
    token.cancel();
    return VmAccessOutcome::Unavailable;
  }

  while (completed_bytes < bytes.size()) {
    if (token.pending()) {
      if (!allow_deferred) {
        token.cancel();
        return VmAccessOutcome::Malformed;
      }
      if (token.state_ != access_state_ || token.address_ != address ||
          token.total_bytes_ != bytes.size() || token.offset_ != completed_bytes ||
          token.kind_ != access || token.bytes_ == 0 ||
          token.bytes_ > bytes.size() - completed_bytes) {
        token.cancel();
        return VmAccessOutcome::Malformed;
      }
      bool valid = false;
      {
        std::shared_lock state_lock(access_state_->mutex);
        valid = access_state_->valid;
      }
      // Only the owner event may execute the request. Neither submission nor
      // polling holds the VM revocation lease or caller-owned data pointers.
      if (valid && !token.submitted_) {
        token.submitted_ = true;
        token.operation_->submit();
      }
      const VmAccessOutcome outcome =
          valid ? token.operation_->poll(token.read_result_)
                : token.operation_->reconcile_revocation(token.read_result_);
      if (outcome == VmAccessOutcome::Unavailable)
        return outcome;
      if (outcome != VmAccessOutcome::Complete) {
        report_terminal_fault(address + completed_bytes, access, outcome);
        token.cancel();
        return outcome;
      }
      if constexpr (!std::is_const_v<typename Span::element_type>)
        std::ranges::copy(token.read_result_, bytes.begin() + completed_bytes);
      if constexpr (!std::is_const_v<typename Span::element_type>) {
        if (std::optional<CanonicalLineFreshness::ReadStamp> stamp =
                token.operation_->completed_read_stamp())
          token.completed_read_ =
              VmCompletedRead{.stamp = std::move(*stamp), .peer_origin = token.peer_origin_};
        else
          token.completed_read_.reset();
      }
      completed_bytes += token.bytes_;
      token.operation_.reset();
      token.peer_origin_ = false;
      token.write_maintenance_ = VmDeferredWriteMaintenance::Caller;
      token.read_result_.clear();
      token.bytes_ = 0;
      token.state_ = nullptr;
      if (!valid)
        return completed_bytes == bytes.size() ? VmAccessOutcome::Complete
                                               : VmAccessOutcome::Revoked;
      return completed_bytes == bytes.size() ? VmAccessOutcome::Complete
                                             : VmAccessOutcome::Unavailable;
    }

    MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
    if (pt_admitted.outcome != VmAccessOutcome::Complete)
      return pt_admitted.outcome;
    std::optional<VmTransferToken> prepared;
    VmTransferStep step;
    const auto prepare = [&] {
      if constexpr (std::is_const_v<typename Span::element_type>) {
        return access_state_->translator->prepare_deferred_write(
            *access_state_->physical_memory, address + completed_bytes,
            bytes.subspan(completed_bytes), access_state_->effect_gate, *this,
            pt_admitted.permit.pt_epoch());
      } else {
        return access_state_->translator->prepare_deferred_read(
            *access_state_->physical_memory, address + completed_bytes,
            bytes.size() - completed_bytes, access, access_state_->effect_gate, *this,
            pt_admitted.permit.pt_epoch());
      }
    };
    const auto local_step = [&] {
      if constexpr (std::is_const_v<typename Span::element_type>)
        return access_state_->translator->write_step(*access_state_->physical_memory,
                                                     address + completed_bytes,
                                                     bytes.subspan(completed_bytes));
      else
        return access_state_->translator->read_step(*access_state_->physical_memory,
                                                    address + completed_bytes,
                                                    bytes.subspan(completed_bytes), access);
    };

    if (maintenance || ingress || copy_source) {
      // Prepare without a cache lease first. A deferred token owns its backing
      // route; a local result must be checked again after ingress and cache
      // maintenance. Neither lease may span deferred submit or poll.
      {
        std::shared_lock state_lock(access_state_->mutex);
        if (!access_state_->valid)
          return VmAccessOutcome::Revoked;
        if (!access_state_->translator || !access_state_->physical_memory)
          return VmAccessOutcome::Unavailable;
        prepared = prepare();
      }
      {
        MemoryIngressAdmission admitted;
        if (!prepared && ingress) {
          admitted = ingress(address + completed_bytes, bytes.size() - completed_bytes, access);
          if (admitted.outcome != VmAccessOutcome::Complete)
            return admitted.outcome;
        }
        VmPrivateUcGttCopySourceAdmission source_admitted;
        if constexpr (!std::is_const_v<typename Span::element_type>) {
          if (!prepared && maintenance && copy_source) {
            source_admitted =
                copy_source(address + completed_bytes, bytes.size() - completed_bytes);
            if (source_admitted.outcome != VmAccessOutcome::Complete)
              return source_admitted.outcome;
          }
        }
        const bool source_owns_read_maintenance = static_cast<bool>(source_admitted.permit);
        const bool endpoint_owns_write_maintenance =
            access == VmAccessKind::Write && prepared && prepared->pending() &&
            prepared->write_maintenance_ == VmDeferredWriteMaintenance::ControlledLocalEndpoint;
        [[maybe_unused]] DeviceCacheMaintenanceLease cache_lease =
            maintenance && !endpoint_owns_write_maintenance && !source_owns_read_maintenance
                ? maintenance()
                : DeviceCacheMaintenanceLease{};
        if (!prepared) {
          std::shared_lock state_lock(access_state_->mutex);
          if (!access_state_->valid)
            return VmAccessOutcome::Revoked;
          if (!access_state_->translator || !access_state_->physical_memory)
            return VmAccessOutcome::Unavailable;
          prepared = prepare();
          // A route that became deferred after source admission has no local
          // read to authorize. Retry with the deferred caller-maintenance path.
          if (prepared && source_owns_read_maintenance)
            return VmAccessOutcome::Unavailable;
          if (!prepared)
            step = local_step();
        }
      }
      // Cache and VM locks are both gone before a deferred submit or poll.
    } else {
      std::shared_lock state_lock(access_state_->mutex);
      if (!access_state_->valid)
        return VmAccessOutcome::Revoked;
      if (!access_state_->translator || !access_state_->physical_memory)
        return VmAccessOutcome::Unavailable;
      prepared = prepare();
      if (!prepared)
        step = local_step();
    }
    // Deferred submit and poll run without a PT permit. The executing owner
    // acquires its own permit before its entered effect and validation rewalk.
    pt_admitted.permit.reset();
    if (prepared) {
      if (!allow_deferred)
        return VmAccessOutcome::Malformed;
      if (!prepared->pending() || prepared->bytes_ == 0 ||
          prepared->bytes_ > bytes.size() - completed_bytes)
        return VmAccessOutcome::Malformed;
      token = std::move(*prepared);
      token.state_ = access_state_;
      token.address_ = address;
      token.total_bytes_ = bytes.size();
      token.offset_ = completed_bytes;
      token.kind_ = access;
      continue;
    }
    if (step.outcome != VmAccessOutcome::Complete) {
      if (step.report_translation_fault && step.outcome != VmAccessOutcome::Unavailable &&
          step.outcome != VmAccessOutcome::Revoked)
        report_terminal_fault(address + completed_bytes, access, step.outcome);
      return step.outcome;
    }
    if (step.completed_bytes == 0 || step.completed_bytes > bytes.size() - completed_bytes)
      return VmAccessOutcome::Malformed;
    completed_bytes += step.completed_bytes;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome GpuVmAccess::read(uint64_t address, std::span<std::byte> bytes,
                                  std::size_t &completed_bytes, VmTransferToken &token,
                                  VmAccessKind access, bool allow_deferred) const {
  return read(address, bytes, completed_bytes, token, access, allow_deferred, {});
}

VmAccessOutcome GpuVmAccess::read(uint64_t address, std::span<std::byte> bytes,
                                  std::size_t &completed_bytes, VmTransferToken &token,
                                  VmAccessKind access, bool allow_deferred,
                                  const VmTransferMaintenance &maintenance,
                                  const VmTransferIngress &ingress,
                                  const VmPrivateUcGttCopySource &copy_source) const {
  if (access != VmAccessKind::Read) {
    token.cancel();
    return VmAccessOutcome::Malformed;
  }
  return transfer_tokenized(address, bytes, completed_bytes, token, access, allow_deferred,
                            maintenance, ingress, copy_source);
}

VmAccessOutcome GpuVmAccess::write(uint64_t address, std::span<const std::byte> bytes,
                                   std::size_t &completed_bytes, VmTransferToken &token,
                                   bool allow_deferred) const {
  return write(address, bytes, completed_bytes, token, allow_deferred, {});
}

VmAccessOutcome GpuVmAccess::write(uint64_t address, std::span<const std::byte> bytes,
                                   std::size_t &completed_bytes, VmTransferToken &token,
                                   bool allow_deferred, const VmTransferMaintenance &maintenance,
                                   const VmTransferIngress &ingress) const {
  return transfer_tokenized(address, bytes, completed_bytes, token, VmAccessKind::Write,
                            allow_deferred, maintenance, ingress, {});
}

AtomicLoadResult GpuVmAccess::atomic_load(uint64_t address, uint32_t width) const {
  if ((width != sizeof(uint16_t) && width != sizeof(uint32_t) && width != sizeof(uint64_t)) ||
      (address & (width - 1)) != 0) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return {.outcome = VmAccessOutcome::Malformed, .value = 0};
  }
  if (access_state_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .value = 0};
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return {.outcome = pt_admitted.outcome, .value = 0};
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return {.outcome = VmAccessOutcome::Revoked, .value = 0};
  if (access_state_->translator == nullptr || access_state_->physical_memory == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .value = 0};

  const VmTranslationResult translated =
      access_state_->translator->translate(address, width, VmAccessKind::Atomic);
  if (!translated) {
    report_terminal_fault(address, VmAccessKind::Atomic, translated.outcome);
    return {.outcome = translated.outcome, .value = 0};
  }
  if (translated.translation.contiguous_bytes < width) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return {.outcome = VmAccessOutcome::Malformed, .value = 0};
  }
  return access_state_->physical_memory->atomic_load(translated.translation.domain,
                                                     translated.translation.address, width);
}

VmAccessOutcome GpuVmAccess::atomic_store(uint64_t address, uint32_t width, uint64_t value) const {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || (address & (width - 1)) != 0) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return VmAccessOutcome::Malformed;
  }
  if (access_state_ == nullptr)
    return VmAccessOutcome::Unavailable;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return pt_admitted.outcome;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return VmAccessOutcome::Revoked;
  if (access_state_->translator == nullptr || access_state_->physical_memory == nullptr)
    return VmAccessOutcome::Unavailable;

  const VmTranslationResult translated =
      access_state_->translator->translate(address, width, VmAccessKind::Atomic);
  if (!translated) {
    report_terminal_fault(address, VmAccessKind::Atomic, translated.outcome);
    return translated.outcome;
  }
  if (translated.translation.contiguous_bytes < width) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return VmAccessOutcome::Malformed;
  }
  return access_state_->physical_memory->atomic_store(translated.translation.domain,
                                                      translated.translation.address, width, value);
}

AtomicCompareExchangeResult GpuVmAccess::compare_exchange(uint64_t address, uint32_t width,
                                                          uint64_t expected,
                                                          uint64_t desired) const {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || (address & (width - 1)) != 0) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return {.outcome = VmAccessOutcome::Malformed};
  }
  if (access_state_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable};
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return {.outcome = pt_admitted.outcome};
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return {.outcome = VmAccessOutcome::Revoked};
  if (access_state_->translator == nullptr || access_state_->physical_memory == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable};

  const VmTranslationResult translated =
      access_state_->translator->translate(address, width, VmAccessKind::Atomic);
  if (!translated) {
    report_terminal_fault(address, VmAccessKind::Atomic, translated.outcome);
    return {.outcome = translated.outcome};
  }
  if (translated.translation.contiguous_bytes < width) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return {.outcome = VmAccessOutcome::Malformed};
  }
  return access_state_->physical_memory->compare_exchange(
      translated.translation.domain, translated.translation.address, width, expected, desired);
}

VmAccessOutcome
GpuVmAccess::atomic_modify(uint64_t address, uint32_t width,
                           const PhysicalMemoryAccess::AtomicMutation &mutation) const {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || (address & (width - 1)) != 0 ||
      !mutation) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return VmAccessOutcome::Malformed;
  }
  if (access_state_ == nullptr)
    return VmAccessOutcome::Unavailable;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return pt_admitted.outcome;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return VmAccessOutcome::Revoked;
  if (access_state_->translator == nullptr || access_state_->physical_memory == nullptr)
    return VmAccessOutcome::Unavailable;

  const VmTranslationResult translated =
      access_state_->translator->translate(address, width, VmAccessKind::Atomic);
  if (!translated) {
    report_terminal_fault(address, VmAccessKind::Atomic, translated.outcome);
    return translated.outcome;
  }
  if (translated.translation.contiguous_bytes < width) {
    report_terminal_fault(address, VmAccessKind::Atomic, VmAccessOutcome::Malformed);
    return VmAccessOutcome::Malformed;
  }
  return access_state_->physical_memory->atomic_modify(
      translated.translation.domain, translated.translation.address, width, mutation);
}

VmAccessOutcome GpuVmAccess::atomic_modify_lanes(
    std::span<const uint64_t> lane_addresses, uint64_t lane_mask, uint32_t width,
    const PhysicalMemoryAccess::AtomicLaneMutation &mutation, uint32_t &next_lane) const {
  const uint32_t lane_count = static_cast<uint32_t>(lane_addresses.size());
  if (access_state_ == nullptr)
    return VmAccessOutcome::Unavailable;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid)
    return VmAccessOutcome::Revoked;
  if (access_state_->translator == nullptr || access_state_->physical_memory == nullptr)
    return VmAccessOutcome::Unavailable;

  // The last translated span: operands inside it translate by a constant offset
  // under the same lease, so the translator need not be consulted again.
  uint64_t span_begin = 0, span_bytes = 0, span_physical = 0;
  VmMemoryDomain span_domain = VmMemoryDomain::Compatibility;
  std::array<uint64_t, 64> physical{};
  if (lane_addresses.size() > physical.size())
    return VmAccessOutcome::Malformed;

  uint32_t lane = next_lane;
  while (lane < lane_count) {
    // Translate a run of lanes sharing one domain; apply it as a unit. A lane
    // that fails to translate ends the run, so earlier lanes still take effect
    // (and fault) before the failure is reported, as in lane-at-a-time order.
    uint64_t run_mask = 0;
    VmMemoryDomain run_domain = VmMemoryDomain::Compatibility;
    uint32_t run_start = lane;
    VmAccessOutcome failure = VmAccessOutcome::Complete;
    uint64_t failure_address = 0;
    for (; lane < lane_count; ++lane) {
      if (!(lane_mask & (1ULL << lane)))
        continue;
      const uint64_t address = lane_addresses[lane];
      if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) ||
          (address & (width - 1)) != 0 || !mutation) {
        failure = VmAccessOutcome::Malformed;
        failure_address = address;
        break;
      }
      if (address - span_begin >= span_bytes || span_bytes - (address - span_begin) < width) {
        const VmTranslationResult translated =
            access_state_->translator->translate(address, width, VmAccessKind::Atomic);
        if (!translated) {
          failure = translated.outcome;
          failure_address = address;
          break;
        }
        if (translated.translation.contiguous_bytes < width) {
          failure = VmAccessOutcome::Malformed;
          failure_address = address;
          break;
        }
        span_begin = address;
        span_bytes = translated.translation.contiguous_bytes;
        span_physical = translated.translation.address;
        span_domain = translated.translation.domain;
      }
      if (run_mask == 0)
        run_domain = span_domain;
      else if (span_domain != run_domain)
        break;
      physical[lane] = span_physical + (address - span_begin);
      run_mask |= 1ULL << lane;
    }

    if (run_mask != 0) {
      uint32_t applied = run_start;
      const VmAccessOutcome outcome = access_state_->physical_memory->atomic_modify_lanes(
          run_domain, std::span<const uint64_t>(physical.data(), lane_count), run_mask, width,
          mutation, applied);
      if (outcome != VmAccessOutcome::Complete) {
        next_lane = applied;
        return outcome;
      }
    }
    if (failure != VmAccessOutcome::Complete) {
      report_terminal_fault(failure_address, VmAccessKind::Atomic, failure);
      next_lane = lane;
      return failure;
    }
  }
  next_lane = lane_count;
  return VmAccessOutcome::Complete;
}

std::byte *GpuVmAccess::resolve_host_pointer(uint64_t address, std::size_t size) const {
  if (access_state_ == nullptr)
    return nullptr;
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return nullptr;
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || access_state_->translator == nullptr ||
      access_state_->physical_memory == nullptr)
    return nullptr;
  const VmTranslationResult translated =
      access_state_->translator->translate(address, size, VmAccessKind::Read);
  if (!translated || translated.translation.contiguous_bytes < size)
    return nullptr;
  return access_state_->physical_memory->resolve_host_pointer(translated.translation.domain,
                                                              translated.translation.address, size);
}

std::pair<uint64_t, uint64_t> GpuVmAccess::host_range(uint64_t address) const {
  if (access_state_ == nullptr)
    return {0, 0};
  MemoryIngressAdmission pt_admitted = admit_pt_rewalk();
  if (pt_admitted.outcome != VmAccessOutcome::Complete)
    return {0, 0};
  std::shared_lock state_lock(access_state_->mutex);
  if (!access_state_->valid || access_state_->translator == nullptr ||
      access_state_->physical_memory == nullptr)
    return {0, 0};
  const VmTranslationResult translated =
      access_state_->translator->translate(address, 1, VmAccessKind::Read);
  if (!translated)
    return {0, 0};
  return access_state_->physical_memory->host_range(translated.translation.domain,
                                                    translated.translation.address);
}

Gfx12PageTableTranslator::Gfx12PageTableTranslator(std::shared_ptr<PhysicalMemoryAccess> memory,
                                                   uint64_t page_table_base, Gfx12VmConfig config)
    : memory_(std::move(memory)), page_table_base_(page_table_base), config_(config) {}

namespace {

std::optional<VmTransferToken> prepare_gfx12_read(const AddressSpaceTranslator &translator,
                                                  PhysicalMemoryAccess &memory, uint64_t address,
                                                  std::size_t size, VmAccessKind access,
                                                  const std::shared_ptr<VmTransferEffectGate> &gate,
                                                  const GpuVmAccess *executing_access,
                                                  const std::optional<MemoryPtEpoch> &pt_epoch) {
  if (size == 0 || access == VmAccessKind::Execute)
    return std::nullopt;
  const VmTranslationResult translated = translator.translate(address, size, access);
  if (!translated || translated.translation.domain != VmMemoryDomain::System ||
      translated.translation.contiguous_bytes == 0)
    return std::nullopt;
  const auto chunk =
      static_cast<std::size_t>(std::min<uint64_t>(size, translated.translation.contiguous_bytes));
  return executing_access != nullptr
             ? memory.prepare_deferred_system_read(translated.translation.address, chunk, access,
                                                   gate, *executing_access, address,
                                                   translated.translation, pt_epoch)
             : memory.prepare_deferred_system_read(translated.translation.address, chunk, access,
                                                   gate);
}

std::optional<VmTransferToken> prepare_gfx12_write(
    const AddressSpaceTranslator &translator, PhysicalMemoryAccess &memory, uint64_t address,
    std::span<const std::byte> bytes, const std::shared_ptr<VmTransferEffectGate> &gate,
    const GpuVmAccess *executing_access, const std::optional<MemoryPtEpoch> &pt_epoch) {
  if (bytes.empty())
    return std::nullopt;
  const VmTranslationResult translated =
      translator.translate(address, bytes.size(), VmAccessKind::Write);
  if (!translated || translated.translation.domain != VmMemoryDomain::System ||
      translated.translation.contiguous_bytes == 0)
    return std::nullopt;
  const auto chunk = static_cast<std::size_t>(
      std::min<uint64_t>(bytes.size(), translated.translation.contiguous_bytes));
  return executing_access != nullptr
             ? memory.prepare_deferred_system_write(translated.translation.address,
                                                    bytes.first(chunk), gate, *executing_access,
                                                    address, translated.translation, pt_epoch)
             : memory.prepare_deferred_system_write(translated.translation.address,
                                                    bytes.first(chunk), gate);
}

} // namespace

std::optional<VmTransferToken> Gfx12PageTableTranslator::prepare_deferred_read(
    PhysicalMemoryAccess &memory, uint64_t address, std::size_t size, VmAccessKind access,
    const std::shared_ptr<VmTransferEffectGate> &gate) const {
  return prepare_gfx12_read(*this, memory, address, size, access, gate, nullptr, {});
}

std::optional<VmTransferToken> Gfx12PageTableTranslator::prepare_deferred_write(
    PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<VmTransferEffectGate> &gate) const {
  return prepare_gfx12_write(*this, memory, address, bytes, gate, nullptr, {});
}

std::optional<VmTransferToken> Gfx12PageTableTranslator::prepare_deferred_read(
    PhysicalMemoryAccess &memory, uint64_t address, std::size_t size, VmAccessKind access,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access) const {
  return prepare_gfx12_read(*this, memory, address, size, access, gate, &executing_access, {});
}

std::optional<VmTransferToken> Gfx12PageTableTranslator::prepare_deferred_read(
    PhysicalMemoryAccess &memory, uint64_t address, std::size_t size, VmAccessKind access,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access,
    const std::optional<MemoryPtEpoch> &pt_epoch) const {
  return prepare_gfx12_read(*this, memory, address, size, access, gate, &executing_access,
                            pt_epoch);
}

std::optional<VmTransferToken> Gfx12PageTableTranslator::prepare_deferred_write(
    PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access) const {
  return prepare_gfx12_write(*this, memory, address, bytes, gate, &executing_access, {});
}

std::optional<VmTransferToken> Gfx12PageTableTranslator::prepare_deferred_write(
    PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access,
    const std::optional<MemoryPtEpoch> &pt_epoch) const {
  return prepare_gfx12_write(*this, memory, address, bytes, gate, &executing_access, pt_epoch);
}

VmTranslationResult Gfx12PageTableTranslator::translate(uint64_t address, std::size_t size,
                                                        VmAccessKind access) const {
  if (memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  const uint64_t virtual_address_mask = gfx12_virtual_address_mask(config_);
  if (address > virtual_address_mask || size - 1 > virtual_address_mask - address)
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  const uint64_t pte_address_mask = gfx12_pte_address_mask(config_);
  const uint64_t pde_address_mask = gfx12_pde_address_mask(config_);
  uint64_t table = page_table_base_ & pte_address_mask;
  util::Logger::vm([&](auto &os) {
    os << std::format("gfx12 walk begin va={:#x} size={} access={} root_raw={:#x} root={:#x} "
                      "va_bits={} levels={} pa_bits={}",
                      address, size, static_cast<unsigned>(access), page_table_base_, table,
                      static_cast<unsigned>(config_.virtual_address_width),
                      config_.page_table_levels(),
                      static_cast<unsigned>(config_.physical_address_width));
  });
  if (table == 0)
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  VmMemoryDomain table_domain =
      (page_table_base_ & kGfx12PteSystem) != 0 ? VmMemoryDomain::System : VmMemoryDomain::Local;

  const std::size_t first_level = kGfx12PageTableShifts.size() - config_.page_table_levels();
  for (const uint32_t shift : std::span(kGfx12PageTableShifts).subspan(first_level)) {
    const uint64_t index = (address >> shift) & 0x1ff;
    const uint64_t entry_address = table + index * sizeof(uint64_t);
    std::array<std::byte, sizeof(uint64_t)> raw_entry{};
    const VmAccessOutcome read = memory_->read(table_domain, entry_address, raw_entry);
    if (read != VmAccessOutcome::Complete) {
      util::Logger::vm([&](auto &os) {
        os << std::format("gfx12 walk va={:#x} shift={} index={:#x} table_domain={} "
                          "entry_address={:#x} read_outcome={}",
                          address, shift, index, static_cast<unsigned>(table_domain), entry_address,
                          static_cast<unsigned>(read));
      });
      return {.outcome = read, .translation = {}};
    }

    const uint64_t entry = std::bit_cast<uint64_t>(raw_entry);
    util::Logger::vm([&](auto &os) {
      os << std::format("gfx12 walk va={:#x} shift={} index={:#x} table_domain={} "
                        "entry_address={:#x} entry={:#018x} valid={} system={} p={}",
                        address, shift, index, static_cast<unsigned>(table_domain), entry_address,
                        entry, (entry & kGfx12PteValid) != 0, (entry & kGfx12PteSystem) != 0,
                        (entry & kGfx12PdePte) != 0);
    });
    if ((entry & kGfx12PteValid) == 0)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};

    const VmMemoryDomain entry_domain =
        (entry & kGfx12PteSystem) != 0 ? VmMemoryDomain::System : VmMemoryDomain::Local;
    if (shift == 12 && (entry & kGfx12PdePte) == 0)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    if ((entry & kGfx12PdePte) != 0) {
      const uint64_t page_bytes = uint64_t{1} << shift;
      const uint64_t page_offset = address & (page_bytes - 1);
      const VmPermissions permissions{
          .readable = (entry & kGfx12PteReadable) != 0,
          .writable = (entry & kGfx12PteWriteable) != 0,
          .executable = (entry & kGfx12PteExecutable) != 0,
      };
      if (!permissions.allows(access))
        return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
      const uint64_t physical_address = (entry & pte_address_mask) + page_offset;
      util::Logger::vm([&](auto &os) {
        os << std::format("gfx12 walk complete va={:#x} shift={} domain={} backing={:#x} span={}",
                          address, shift, static_cast<unsigned>(entry_domain), physical_address,
                          page_bytes - page_offset);
      });
      return {
          .outcome = VmAccessOutcome::Complete,
          .translation =
              {
                  .domain = entry_domain,
                  .address = physical_address,
                  .contiguous_bytes = page_bytes - page_offset,
                  .mtype = gfx12_mtype(entry),
                  .gfx12_pte_mtype = gfx12_pte_mtype(entry),
                  .permissions = permissions,
              },
      };
    }

    table = entry & pde_address_mask;
    if (table == 0)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    table_domain = entry_domain;
  }
  return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
}

Gfx12GartTranslator::Gfx12GartTranslator(PhysicalMemoryAccess &memory, uint64_t page_table_base,
                                         uint64_t aperture_start, uint64_t aperture_end,
                                         Gfx12VmConfig config)
    : memory_(&memory), page_table_base_(page_table_base), aperture_start_(aperture_start),
      aperture_end_(aperture_end), config_(config) {}

Gfx12GartTranslator::Gfx12GartTranslator(std::shared_ptr<PhysicalMemoryAccess> memory,
                                         uint64_t page_table_base, uint64_t aperture_start,
                                         uint64_t aperture_end, Gfx12VmConfig config)
    : retained_memory_(std::move(memory)), memory_(retained_memory_.get()),
      page_table_base_(page_table_base), aperture_start_(aperture_start),
      aperture_end_(aperture_end), config_(config) {}

std::optional<VmTransferToken> Gfx12GartTranslator::prepare_deferred_read(
    PhysicalMemoryAccess &memory, uint64_t address, std::size_t size, VmAccessKind access,
    const std::shared_ptr<VmTransferEffectGate> &gate) const {
  return prepare_gfx12_read(*this, memory, address, size, access, gate, nullptr, {});
}

std::optional<VmTransferToken> Gfx12GartTranslator::prepare_deferred_write(
    PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<VmTransferEffectGate> &gate) const {
  return prepare_gfx12_write(*this, memory, address, bytes, gate, nullptr, {});
}

std::optional<VmTransferToken> Gfx12GartTranslator::prepare_deferred_read(
    PhysicalMemoryAccess &memory, uint64_t address, std::size_t size, VmAccessKind access,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access) const {
  return prepare_gfx12_read(*this, memory, address, size, access, gate, &executing_access, {});
}

std::optional<VmTransferToken> Gfx12GartTranslator::prepare_deferred_read(
    PhysicalMemoryAccess &memory, uint64_t address, std::size_t size, VmAccessKind access,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access,
    const std::optional<MemoryPtEpoch> &pt_epoch) const {
  return prepare_gfx12_read(*this, memory, address, size, access, gate, &executing_access,
                            pt_epoch);
}

std::optional<VmTransferToken> Gfx12GartTranslator::prepare_deferred_write(
    PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access) const {
  return prepare_gfx12_write(*this, memory, address, bytes, gate, &executing_access, {});
}

std::optional<VmTransferToken> Gfx12GartTranslator::prepare_deferred_write(
    PhysicalMemoryAccess &memory, uint64_t address, std::span<const std::byte> bytes,
    const std::shared_ptr<VmTransferEffectGate> &gate, const GpuVmAccess &executing_access,
    const std::optional<MemoryPtEpoch> &pt_epoch) const {
  return prepare_gfx12_write(*this, memory, address, bytes, gate, &executing_access, pt_epoch);
}

VmTranslationResult Gfx12GartTranslator::translate(uint64_t address, std::size_t size,
                                                   VmAccessKind access) const {
  if (memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  if (address < aperture_start_ || address > aperture_end_) {
    uint64_t contiguous_bytes = size;
    if (address < aperture_start_)
      contiguous_bytes = std::min<uint64_t>(contiguous_bytes, aperture_start_ - address);
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation =
            {
                .domain = VmMemoryDomain::Local,
                .address = address,
                .contiguous_bytes = contiguous_bytes,
                .mtype = Mtype::RW,
                .permissions = {.readable = true, .writable = true, .executable = true},
            },
    };
  }

  const uint64_t pte_address_mask = gfx12_pte_address_mask(config_);
  const uint64_t table = page_table_base_ & pte_address_mask;
  if (table == 0 && (page_table_base_ & kGfx12PteValid) == 0)
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  const uint64_t page = (address - aperture_start_) / kGfx12PageBytes;
  if (page > (std::numeric_limits<uint64_t>::max() - table) / sizeof(uint64_t))
    return {.outcome = VmAccessOutcome::Malformed, .translation = {}};

  std::array<std::byte, sizeof(uint64_t)> raw_entry{};
  const VmMemoryDomain table_domain =
      (page_table_base_ & kGfx12PteSystem) != 0 ? VmMemoryDomain::System : VmMemoryDomain::Local;
  const VmAccessOutcome read =
      memory_->read(table_domain, table + page * sizeof(uint64_t), raw_entry);
  if (read != VmAccessOutcome::Complete)
    return {.outcome = read, .translation = {}};
  const uint64_t entry = std::bit_cast<uint64_t>(raw_entry);
  constexpr uint64_t kRequiredFlags = kGfx12PteValid | kGfx12PteSystem | kGfx12PdePte;
  if ((entry & kRequiredFlags) != kRequiredFlags) {
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  }

  const VmPermissions permissions{
      .readable = (entry & kGfx12PteReadable) != 0,
      .writable = (entry & kGfx12PteWriteable) != 0,
      .executable = (entry & kGfx12PteExecutable) != 0,
  };
  if (!permissions.allows(access))
    return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
  const uint64_t page_offset = (address - aperture_start_) & (kGfx12PageBytes - 1);
  const uint64_t aperture_bytes = aperture_end_ - address + 1;
  return {
      .outcome = VmAccessOutcome::Complete,
      .translation =
          {
              .domain = VmMemoryDomain::System,
              .address = (entry & pte_address_mask) + page_offset,
              .contiguous_bytes = std::min(kGfx12PageBytes - page_offset, aperture_bytes),
              .mtype = gfx12_mtype(entry),
              .gfx12_pte_mtype = gfx12_pte_mtype(entry),
              .permissions = permissions,
          },
  };
}

AddressSpaceHandle GpuVm::allocate_locked(Binding binding) {
  if (binding.access_state == nullptr)
    binding.access_state = std::make_shared<GpuVmAccessState>();
  if (binding.binding_state == nullptr)
    binding.binding_state = std::make_shared<GpuVmBindingState>();
  uint32_t slot_index = 0;
  if (free_slots_.empty()) {
    slot_index = static_cast<uint32_t>(slots_.size());
    slots_.push_back({});
  } else {
    slot_index = free_slots_.back();
    free_slots_.pop_back();
  }
  Slot &slot = slots_[slot_index];
  slot.binding = std::move(binding);
  return {.slot = slot_index, .generation = slot.generation};
}

GpuVm::Binding *GpuVm::find_locked(AddressSpaceHandle handle) {
  if (!handle || handle.slot >= slots_.size())
    return nullptr;
  Slot &slot = slots_[handle.slot];
  return slot.generation == handle.generation && slot.binding ? &*slot.binding : nullptr;
}

const GpuVm::Binding *GpuVm::find_locked(AddressSpaceHandle handle) const {
  if (!handle || handle.slot >= slots_.size())
    return nullptr;
  const Slot &slot = slots_[handle.slot];
  return slot.generation == handle.generation && slot.binding ? &*slot.binding : nullptr;
}

void GpuVm::advance_access_state_locked(Binding &binding,
                                        std::shared_ptr<GpuVmAccessState> replacement) {
  if (binding.access_state != nullptr) {
    {
      std::unique_lock state_lock(binding.access_state->mutex);
      binding.access_state->valid = false;
    }
    binding.access_state->effect_gate->revoke();
  }
  binding.access_state = std::move(replacement);
}

void GpuVm::revoke_access_state_locked(Binding &binding) {
  if (binding.access_state != nullptr) {
    {
      std::unique_lock state_lock(binding.access_state->mutex);
      binding.access_state->valid = false;
    }
    binding.access_state->effect_gate->revoke();
  }
  for (const std::weak_ptr<GpuVmAccessState> &weak_state : binding.pinned_access_states) {
    if (const std::shared_ptr<GpuVmAccessState> state = weak_state.lock()) {
      {
        std::unique_lock state_lock(state->mutex);
        state->valid = false;
      }
      state->effect_gate->revoke();
    }
  }
  binding.pinned_access_states.clear();
}

AddressSpaceHandle
GpuVm::register_address_space(uint32_t vmid, std::shared_ptr<AddressSpaceTranslator> translator,
                              std::shared_ptr<PhysicalMemoryAccess> physical_memory,
                              std::function<void(uint64_t, VmAccessKind)> fault_reporter,
                              bool legacy_cache_compatible) {
  if (translator == nullptr || physical_memory == nullptr)
    return {};
  std::lock_guard lock(mutex_);
  if (vmid_handles_.contains(vmid))
    return {};
  AddressSpaceHandle handle = allocate_locked(
      {.vmid = vmid,
       .translation_epoch = 1,
       .queue_references = 0,
       .binding_state = {},
       .access_state = std::make_shared<GpuVmAccessState>(
           std::move(translator), std::move(physical_memory),
           std::make_shared<const GpuVmAccessState::FaultReporter>(std::move(fault_reporter))),
       .pinned_access_states = {},
       .legacy_cache_compatible = legacy_cache_compatible,
       .device_gart = false});
  vmid_handles_.emplace(vmid, handle);
  return handle;
}

AddressSpaceHandle GpuVm::register_unrouted_address_space(
    uint32_t vmid, std::shared_ptr<AddressSpaceTranslator> translator,
    std::shared_ptr<PhysicalMemoryAccess> physical_memory,
    std::function<void(uint64_t, VmAccessKind)> fault_reporter, bool legacy_cache_compatible) {
  if (translator == nullptr || physical_memory == nullptr)
    return {};
  std::lock_guard lock(mutex_);
  return allocate_locked(
      {.vmid = vmid,
       .translation_epoch = 1,
       .queue_references = 0,
       .binding_state = {},
       .access_state = std::make_shared<GpuVmAccessState>(
           std::move(translator), std::move(physical_memory),
           std::make_shared<const GpuVmAccessState::FaultReporter>(std::move(fault_reporter))),
       .pinned_access_states = {},
       .legacy_cache_compatible = legacy_cache_compatible,
       .device_gart = false});
}

AddressSpaceHandle GpuVm::register_translated(uint32_t vmid,
                                              std::shared_ptr<AddressSpaceTranslator> translator,
                                              std::shared_ptr<PhysicalMemoryAccess> memory) {
  return register_address_space(vmid, std::move(translator), std::move(memory));
}

AddressSpaceHandle
GpuVm::register_gfx12_address_space(uint32_t vmid, uint64_t page_table_base,
                                    std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (page_table_base == 0 || memory == nullptr)
    return {};
  auto translator =
      std::make_shared<Gfx12PageTableTranslator>(memory, page_table_base, gfx12_config_);
  return register_translated(vmid, std::move(translator), std::move(memory));
}

AddressSpaceHandle GpuVm::register_unrouted_gfx12_address_space(
    uint32_t vmid, uint64_t page_table_base, std::shared_ptr<PhysicalMemoryAccess> memory,
    std::function<void(uint64_t, VmAccessKind)> fault_reporter) {
  if (page_table_base == 0 || memory == nullptr)
    return {};
  auto translator =
      std::make_shared<Gfx12PageTableTranslator>(memory, page_table_base, gfx12_config_);
  return register_unrouted_address_space(vmid, std::move(translator), std::move(memory),
                                         std::move(fault_reporter));
}

bool GpuVm::replace_translated(AddressSpaceHandle handle,
                               std::shared_ptr<AddressSpaceTranslator> translator,
                               std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (translator == nullptr || memory == nullptr)
    return false;
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->access_state->translator == nullptr ||
      binding->legacy_cache_compatible || binding->device_gart)
    return false;
  advance_access_state_locked(
      *binding, std::make_shared<GpuVmAccessState>(std::move(translator), std::move(memory),
                                                   binding->access_state->fault_reporter));
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

bool GpuVm::replace_gfx12_address_space_root(AddressSpaceHandle handle, uint64_t page_table_base,
                                             std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (page_table_base == 0 || memory == nullptr)
    return false;
  auto translator =
      std::make_shared<Gfx12PageTableTranslator>(memory, page_table_base, gfx12_config_);
  return replace_translated(handle, std::move(translator), std::move(memory));
}

AddressSpaceHandle GpuVm::initialize_gart_address_space() {
  std::lock_guard lock(mutex_);
  if (find_locked(gart_address_space_) != nullptr)
    return gart_address_space_;

  gart_address_space_ = allocate_locked({.vmid = 0,
                                         .translation_epoch = 1,
                                         .queue_references = 0,
                                         .binding_state = {},
                                         .access_state = {},
                                         .pinned_access_states = {},
                                         .legacy_cache_compatible = false,
                                         .device_gart = true});
  return gart_address_space_;
}

bool GpuVm::publish_gart(const GartConfig &config, std::shared_ptr<PhysicalMemoryAccess> memory) {
  if (memory == nullptr || !valid_gart_config(config, gfx12_config_))
    return false;
  auto translator = std::make_shared<Gfx12GartTranslator>(
      memory, config.page_table_base, config.aperture_start, config.aperture_end, gfx12_config_);

  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(gart_address_space_);
  if (binding == nullptr || !binding->device_gart)
    return false;
  advance_access_state_locked(
      *binding, std::make_shared<GpuVmAccessState>(std::move(translator), std::move(memory),
                                                   binding->access_state->fault_reporter));
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

bool GpuVm::clear_gart_binding() {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(gart_address_space_);
  if (binding == nullptr || !binding->device_gart || binding->queue_references != 0 ||
      binding->binding_state->references.load(std::memory_order_acquire) != 0)
    return false;
  revoke_access_state_locked(*binding);
  binding->access_state = std::make_shared<GpuVmAccessState>();
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

AddressSpaceHandle GpuVm::gart_address_space() const {
  std::shared_lock lock(mutex_);
  return find_locked(gart_address_space_) != nullptr ? gart_address_space_ : AddressSpaceHandle{};
}

bool GpuVm::invalidate(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr)
    return false;
  auto replacement = std::make_shared<GpuVmAccessState>(
      binding->access_state->translator, binding->access_state->physical_memory,
      binding->access_state->fault_reporter, binding->access_state->cache_write_sequence,
      binding->access_state->root_lineage);
  revoke_access_state_locked(*binding);
  binding->access_state = std::move(replacement);
  ++binding->translation_epoch;
  if (binding->translation_epoch == 0)
    ++binding->translation_epoch;
  return true;
}

std::optional<AddressSpaceInfo> GpuVm::retain_queue_address_space(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->queue_references == UINT32_MAX)
    return std::nullopt;
  ++binding->queue_references;
  return AddressSpaceInfo{.vmid = binding->vmid,
                          .translation_epoch = binding->translation_epoch,
                          .queue_references = binding->queue_references,
                          .legacy_cache_compatible = binding->legacy_cache_compatible,
                          .ready = binding->access_state->translator != nullptr &&
                                   binding->access_state->physical_memory != nullptr};
}

bool GpuVm::retain_queue(AddressSpaceHandle handle) {
  return retain_queue_address_space(handle).has_value();
}

bool GpuVm::release_queue(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->queue_references == 0)
    return false;
  --binding->queue_references;
  return true;
}

std::optional<GpuVmBindingLease> GpuVm::retain_binding(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr ||
      binding->binding_state->references.load(std::memory_order_relaxed) == UINT32_MAX)
    return std::nullopt;
  binding->binding_state->references.fetch_add(1, std::memory_order_relaxed);
  AddressSpaceInfo info{.vmid = binding->vmid,
                        .translation_epoch = binding->translation_epoch,
                        .queue_references = binding->queue_references,
                        .legacy_cache_compatible = binding->legacy_cache_compatible,
                        .ready = binding->access_state->translator != nullptr &&
                                 binding->access_state->physical_memory != nullptr};
  return GpuVmBindingLease(binding->binding_state, handle, info);
}

bool GpuVm::unregister_address_space(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr || binding->queue_references != 0 ||
      binding->binding_state->references.load(std::memory_order_acquire) != 0 ||
      binding->device_gart)
    return false;
  revoke_access_state_locked(*binding);
  const uint32_t vmid = binding->vmid;
  const auto routed = vmid_handles_.find(vmid);
  if (routed != vmid_handles_.end() && routed->second == handle)
    vmid_handles_.erase(routed);
  Slot &slot = slots_[handle.slot];
  slot.binding.reset();
  ++slot.generation;
  if (slot.generation == 0)
    ++slot.generation;
  free_slots_.push_back(handle.slot);
  // The binding state just expired. Clear this handle's policy without an
  // allocation, then compact the immutable index on a best-effort basis. A
  // failed compaction cannot make the retired guard match a reused VMID.
  {
    std::lock_guard policy_lock(controlled_backing_mutex_);
    const auto current = controlled_backings_.load(std::memory_order_acquire);
    if (current) {
      for (const ControlledCacheBacking &entry : *current)
        if (entry.handle == handle)
          entry.active->store(false, std::memory_order_release);
      try {
        auto replacement = std::make_shared<std::vector<ControlledCacheBacking>>();
        replacement->reserve(current->size());
        for (const ControlledCacheBacking &entry : *current)
          if (entry.active->load(std::memory_order_acquire) && !entry.binding.expired())
            replacement->push_back(entry);
        controlled_backings_.store(std::move(replacement), std::memory_order_release);
      } catch (const std::bad_alloc &) {
      }
    }
  }
  return true;
}

std::optional<GpuVmAccess> GpuVm::snapshot(AddressSpaceHandle handle) const {
  std::shared_lock lock(mutex_);
  const Binding *binding = find_locked(handle);
  if (binding == nullptr)
    return std::nullopt;
  return GpuVmAccess(handle,
                     {.vmid = binding->vmid,
                      .translation_epoch = binding->translation_epoch,
                      .queue_references = binding->queue_references,
                      .legacy_cache_compatible = binding->legacy_cache_compatible,
                      .ready = binding->access_state->translator != nullptr &&
                               binding->access_state->physical_memory != nullptr},
                     binding->access_state);
}

GpuVm::CurrentNamespaceLease
GpuVm::try_acquire_current_namespace(AddressSpaceHandle handle,
                                     VmCacheNamespace expected) const noexcept {
  if (!expected || expected.address_space != handle)
    return {};
  try {
    std::shared_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock())
      return {};
    const Binding *binding = find_locked(handle);
    if (binding == nullptr || binding->translation_epoch != expected.translation_epoch ||
        binding->access_state == nullptr || !binding->access_state->valid)
      return {};
    return CurrentNamespaceLease(std::move(lock));
  } catch (...) {
    return {};
  }
}

std::optional<GpuVmAccess> GpuVm::snapshot_pinned(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  Binding *binding = find_locked(handle);
  if (binding == nullptr)
    return std::nullopt;
  std::erase_if(binding->pinned_access_states,
                [](const std::weak_ptr<GpuVmAccessState> &state) { return state.expired(); });
  auto access_state = std::make_shared<GpuVmAccessState>(
      binding->access_state->translator, binding->access_state->physical_memory,
      binding->access_state->fault_reporter, binding->access_state->cache_write_sequence,
      binding->access_state->root_lineage);
  binding->pinned_access_states.emplace_back(access_state);
  return GpuVmAccess(handle,
                     {.vmid = binding->vmid,
                      .translation_epoch = binding->translation_epoch,
                      .queue_references = binding->queue_references,
                      .legacy_cache_compatible = binding->legacy_cache_compatible,
                      .ready = binding->access_state->translator != nullptr &&
                               binding->access_state->physical_memory != nullptr},
                     std::move(access_state));
}

std::optional<GpuVmAccess> GpuVm::snapshot_vmid(uint32_t vmid) const {
  std::shared_lock lock(mutex_);
  const std::unordered_map<uint32_t, AddressSpaceHandle>::const_iterator found =
      vmid_handles_.find(vmid);
  if (found == vmid_handles_.end())
    return std::nullopt;
  const Binding *binding = find_locked(found->second);
  if (binding == nullptr)
    return std::nullopt;
  return GpuVmAccess(found->second,
                     {.vmid = binding->vmid,
                      .translation_epoch = binding->translation_epoch,
                      .queue_references = binding->queue_references,
                      .legacy_cache_compatible = binding->legacy_cache_compatible,
                      .ready = binding->access_state->translator != nullptr &&
                               binding->access_state->physical_memory != nullptr},
                     binding->access_state);
}

const GpuVmAccess *GpuVm::borrow_snapshot(AddressSpaceHandle handle) const {
  assert(GpuVmAccessBatchGuard::active());
  if (const GpuVmAccess *cached = GpuVmAccessBatchGuard::find_snapshot(this, handle))
    return cached;
  if (const std::optional<GpuVmAccess> access = snapshot(handle)) {
    GpuVmAccessBatchGuard::retain_snapshot(this, handle, *access, false);
    // A concurrent revocation makes this access retryable, not an absent binding.
    return &vm_access_batch_snapshots.back().access;
  }
  return nullptr;
}

const GpuVmAccess *GpuVm::borrow_snapshot_vmid(uint32_t vmid) const {
  assert(GpuVmAccessBatchGuard::active());
  if (const GpuVmAccess *cached = GpuVmAccessBatchGuard::find_snapshot_vmid(this, vmid))
    return cached;
  if (const std::optional<GpuVmAccess> access = snapshot_vmid(vmid)) {
    GpuVmAccessBatchGuard::retain_snapshot(this, access->cache_namespace().address_space, *access,
                                           true);
    // A concurrent revocation makes this access retryable, not an absent binding.
    return &vm_access_batch_snapshots.back().access;
  }
  return nullptr;
}

VmTranslationResult GpuVm::translate(AddressSpaceHandle handle, uint64_t address, std::size_t size,
                                     VmAccessKind access) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot
             ? access_snapshot->translate(address, size, access)
             : VmTranslationResult{.outcome = VmAccessOutcome::Faulted, .translation = {}};
}

VmAccessOutcome GpuVm::probe(AddressSpaceHandle handle, uint64_t address, std::size_t size,
                             VmAccessKind access) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->probe(address, size, access) : VmAccessOutcome::Faulted;
}

VmAccessOutcome GpuVm::read(AddressSpaceHandle handle, uint64_t address,
                            std::span<std::byte> bytes) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->read(address, bytes) : VmAccessOutcome::Faulted;
}

VmAccessOutcome GpuVm::write(AddressSpaceHandle handle, uint64_t address,
                             std::span<const std::byte> bytes) {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->write(address, bytes) : VmAccessOutcome::Faulted;
}

AtomicLoadResult GpuVm::atomic_load(AddressSpaceHandle handle, uint64_t address,
                                    uint32_t width) const {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->atomic_load(address, width)
                         : AtomicLoadResult{.outcome = VmAccessOutcome::Faulted, .value = 0};
}

VmAccessOutcome GpuVm::atomic_store(AddressSpaceHandle handle, uint64_t address, uint32_t width,
                                    uint64_t value) {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->atomic_store(address, width, value)
                         : VmAccessOutcome::Faulted;
}

AtomicCompareExchangeResult GpuVm::compare_exchange(AddressSpaceHandle handle, uint64_t address,
                                                    uint32_t width, uint64_t expected,
                                                    uint64_t desired) {
  const std::optional<GpuVmAccess> access_snapshot = snapshot(handle);
  return access_snapshot ? access_snapshot->compare_exchange(address, width, expected, desired)
                         : AtomicCompareExchangeResult{.outcome = VmAccessOutcome::Faulted};
}

std::optional<AddressSpaceInfo> GpuVm::lookup(AddressSpaceHandle handle) const {
  std::shared_lock lock(mutex_);
  const Binding *binding = find_locked(handle);
  if (binding == nullptr)
    return std::nullopt;
  return AddressSpaceInfo{.vmid = binding->vmid,
                          .translation_epoch = binding->translation_epoch,
                          .queue_references = binding->queue_references,
                          .legacy_cache_compatible = binding->legacy_cache_compatible,
                          .ready = binding->access_state->translator != nullptr &&
                                   binding->access_state->physical_memory != nullptr};
}

std::optional<AddressSpaceHandle> GpuVm::find_vmid(uint32_t vmid) const {
  std::shared_lock lock(mutex_);
  const std::unordered_map<uint32_t, AddressSpaceHandle>::const_iterator found =
      vmid_handles_.find(vmid);
  return found == vmid_handles_.end() ? std::nullopt
                                      : std::optional<AddressSpaceHandle>(found->second);
}

std::size_t GpuVm::active_address_spaces() const {
  std::shared_lock lock(mutex_);
  return static_cast<std::size_t>(
      std::ranges::count_if(slots_, [](const Slot &slot) { return slot.binding.has_value(); }));
}

uint64_t GpuVm::reset_epoch() const {
  std::shared_lock lock(mutex_);
  return reset_epoch_;
}

bool GpuVm::reset() {
  std::lock_guard lock(mutex_);
  const bool retained = std::ranges::any_of(slots_, [](const Slot &slot) {
    return slot.binding &&
           (slot.binding->queue_references != 0 ||
            slot.binding->binding_state->references.load(std::memory_order_acquire) != 0);
  });
  if (retained)
    return false;
  for (uint32_t index = 0; index < slots_.size(); ++index) {
    Slot &slot = slots_[index];
    if (!slot.binding)
      continue;
    revoke_access_state_locked(*slot.binding);
    slot.binding.reset();
    ++slot.generation;
    if (slot.generation == 0)
      ++slot.generation;
    free_slots_.push_back(index);
  }
  vmid_handles_.clear();
  controlled_backings_.store(std::shared_ptr<const std::vector<ControlledCacheBacking>>{},
                             std::memory_order_release);
  gart_address_space_ = {};
  ++reset_epoch_;
  if (reset_epoch_ == 0)
    ++reset_epoch_;
  return true;
}

} // namespace rocjitsu::amdgpu
