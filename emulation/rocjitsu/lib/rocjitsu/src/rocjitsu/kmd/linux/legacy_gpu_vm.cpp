// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

constexpr VmAccessOutcome vm_access_outcome(CopyOutcome outcome) {
  switch (outcome) {
  case CopyOutcome::Complete:
    return VmAccessOutcome::Complete;
  case CopyOutcome::Unavailable:
    return VmAccessOutcome::Unavailable;
  case CopyOutcome::Faulted:
    return VmAccessOutcome::Faulted;
  }
  return VmAccessOutcome::Malformed;
}

class FaultedDeferredTransfer final : public VmDeferredTransfer {
public:
  void submit() override {}
  [[nodiscard]] VmAccessOutcome poll(std::span<std::byte>) override {
    return VmAccessOutcome::Faulted;
  }
  [[nodiscard]] VmAccessOutcome reconcile_revocation(std::span<std::byte>) override {
    return VmAccessOutcome::Revoked;
  }
  void cancel() noexcept override {}
};

} // namespace

class LegacyGpuVmAdapter::Binding final : public AddressSpaceTranslator,
                                          public PhysicalMemoryAccess {
public:
  Binding(std::shared_ptr<GpuMemory> memory, uint32_t vmid,
          LegacyAddressSpaceRegistration registration, std::shared_ptr<void> frontend_lifetime)
      : memory_(std::move(memory)), address_space_(std::make_shared<LegacyAddressSpace>(*memory_)),
        vmid_(vmid), original_request_mutex_(registration.request_mutex),
        mutation_epoch_(std::move(registration.mutation_epoch)),
        raw_write_gate_(std::move(registration.raw_write_gate)),
        access_admission_(std::move(registration.access_admission)),
        data_route_(std::move(registration.data_route)),
        prepare_deferred_transfer_(std::move(registration.prepare_deferred_transfer)),
        frontend_lifetime_(std::move(frontend_lifetime)) {
    address_space_->register_process(vmid_, registration.page_table, registration.page_table_mutex,
                                     registration.page_table_generation,
                                     std::move(registration.request_mutex),
                                     std::move(registration.page_table_cache_state));
    address_space_->set_process_client_pid(vmid_, registration.client_pid);
    address_space_->set_process_mem_fd(vmid_, registration.client_mem_fd);
    address_space_->set_process_passthrough(vmid_, registration.passthrough);
    address_space_->set_process_fault_reporter(vmid_, registration.fault_reporter);
  }

  [[nodiscard]] bool try_write_private_dwords(PhysicalMemoryAccess &physical,
                                              std::span<const VmRamDwordStore> stores,
                                              Mtype instruction_mtype,
                                              Mtype expected_mtype) const override {
    if (&physical != static_cast<const PhysicalMemoryAccess *>(this) || stores.size() < 2 ||
        stores.size() > VmRamDwordStore::kMaxBatch)
      return false;
    std::array<VmRawBackingWriteGate::Permit, VmRamDwordStore::kMaxBatch> permits;
    std::size_t permit_count = 0;
    for (const auto &store : stores) {
      auto permit = try_begin_raw_write(store.address, sizeof(uint32_t));
      if (!permit)
        return false;
      permits[permit_count++] = std::move(*permit);
      const VmAccessAdmission admission =
          admit_access(store.address, sizeof(uint32_t), VmAccessKind::Write);
      if (!admission || !admission.direct_backing_write_allowed)
        return false;
    }
    return address_space_->try_write_private_dwords(stores, vmid_, instruction_mtype,
                                                    expected_mtype, original_request_mutex_);
  }

  [[nodiscard]] bool try_read_uncached_ram(PhysicalMemoryAccess &physical, uint64_t address,
                                           std::span<std::byte> bytes) const override {
    return &physical == static_cast<const PhysicalMemoryAccess *>(this) &&
           admit_access(address, bytes.size(), VmAccessKind::Read) &&
           address_space_->try_read_uncached_ram(address, bytes, vmid_, original_request_mutex_);
  }

  [[nodiscard]] bool raw_write_route_is_local(uint64_t address, std::size_t size) const {
    if (!data_route_)
      return true;
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return false;
    std::size_t checked = 0;
    while (checked < size) {
      const LegacyDataRoute route =
          data_route_(address + checked, size - checked, VmAccessKind::Write);
      if (route.route != VmDataRoute::Local || route.candidate != LegacyDeferredCandidate::None ||
          route.first_extent_bytes == 0)
        return false;
      checked += std::min(route.first_extent_bytes, size - checked);
    }
    return true;
  }

  [[nodiscard]] VmAccessAdmission admit_access(uint64_t address, std::size_t size,
                                               VmAccessKind access) const override {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = VmAccessOutcome::Malformed};
    // The untokenized compatibility paths, including cache fills and borrowed
    // host pointers, must never consume an owned candidate directly.
    if (data_route_) {
      if (access == VmAccessKind::Write) {
        if (!raw_write_route_is_local(address, size))
          return {.outcome = VmAccessOutcome::Faulted};
      } else {
        const LegacyDataRoute route = data_route_(address, size, access);
        if (route.route != VmDataRoute::Local || route.candidate != LegacyDeferredCandidate::None)
          return {.outcome = VmAccessOutcome::Faulted};
      }
      // Atomics also mutate backing without entering the deferred WRITE path.
      if (access == VmAccessKind::Atomic && !raw_write_route_is_local(address, size))
        return {.outcome = VmAccessOutcome::Faulted};
    }
    return access_admission_ ? access_admission_(address, size, access) : VmAccessAdmission{};
  }

  [[nodiscard]] VmAccessAdmission query_cache_write_grant(uint64_t address,
                                                          std::size_t size) const override {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = VmAccessOutcome::Malformed};
    // The frontend grant identifies a dirty cache tag even when its backing
    // write requires an owned deferred effect. Never use this query to write.
    return access_admission_ ? access_admission_(address, size, VmAccessKind::Write)
                             : admit_access(address, size, VmAccessKind::Write);
  }

  [[nodiscard]] std::optional<VmRawBackingWriteGate::Permit>
  try_begin_raw_write(uint64_t address, std::size_t size) const {
    // Preserve the ordinary malformed-access outcome; validation below will
    // reject this span before any backing effect can run.
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return VmRawBackingWriteGate::Permit{};
    if (raw_write_gate_)
      return raw_write_gate_->try_acquire(address, size);
    return VmRawBackingWriteGate::Permit{};
  }

  [[nodiscard]] VmDataRoute classify_data_route(uint64_t address, std::size_t size,
                                                VmAccessKind access) const override {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return VmDataRoute::Faulted;
    return data_route_ ? data_route_(address, size, access).route : VmDataRoute::Local;
  }

  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &physical, uint64_t address, std::size_t size,
                        VmAccessKind access,
                        const std::shared_ptr<VmTransferEffectGate> &) const override {
    if (&physical != static_cast<const PhysicalMemoryAccess *>(this) || !data_route_ ||
        access != VmAccessKind::Read || size == 0 ||
        size - 1 > std::numeric_limits<uint64_t>::max() - address ||
        data_route_(address, size, access).candidate != LegacyDeferredCandidate::Peer)
      return std::nullopt;
    // Untokenized callers must observe the deferred route and refuse it. They
    // have no executing snapshot to authorize an owned effect.
    return VmTransferToken(std::make_unique<FaultedDeferredTransfer>(), 1);
  }

  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &physical, uint64_t address, std::size_t size,
                        VmAccessKind access, const std::shared_ptr<VmTransferEffectGate> &gate,
                        const GpuVmAccess &executing_access) const override {
    if (&physical != static_cast<const PhysicalMemoryAccess *>(this) || !data_route_ ||
        access != VmAccessKind::Read || size == 0 ||
        size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return std::nullopt;
    const auto route = data_route_(address, size, access);
    if (route.candidate != LegacyDeferredCandidate::Peer)
      return std::nullopt;
    const size_t page_bytes = kLegacyPageSize - (address & (kLegacyPageSize - 1));
    const size_t chunk = std::min({size, page_bytes, route.first_extent_bytes});
    if (chunk == 0)
      return VmTransferToken(std::make_unique<FaultedDeferredTransfer>(), 1);
    auto operation = route.route == VmDataRoute::Deferred && prepare_deferred_transfer_
                         ? prepare_deferred_transfer_(route.candidate, address, chunk, {}, false,
                                                      gate, executing_access, address_space_)
                         : nullptr;
    if (!operation)
      operation = std::make_unique<FaultedDeferredTransfer>();
    return VmTransferToken(std::move(operation), chunk);
  }

  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &physical, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &) const override {
    if (&physical != static_cast<const PhysicalMemoryAccess *>(this) || !data_route_ ||
        bytes.empty() || bytes.size() - 1 > std::numeric_limits<uint64_t>::max() - address ||
        data_route_(address, bytes.size(), VmAccessKind::Write).candidate ==
            LegacyDeferredCandidate::None)
      return std::nullopt;
    return VmTransferToken(std::make_unique<FaultedDeferredTransfer>(), 1);
  }

  [[nodiscard]] std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &physical, uint64_t address,
                         std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &gate,
                         const GpuVmAccess &executing_access) const override {
    if (&physical != static_cast<const PhysicalMemoryAccess *>(this) || !data_route_ ||
        bytes.empty() || bytes.size() - 1 > std::numeric_limits<uint64_t>::max() - address)
      return std::nullopt;
    const auto route = data_route_(address, bytes.size(), VmAccessKind::Write);
    if (route.candidate == LegacyDeferredCandidate::None)
      return std::nullopt;
    const size_t page_bytes = kLegacyPageSize - (address & (kLegacyPageSize - 1));
    const size_t chunk = std::min({bytes.size(), page_bytes, route.first_extent_bytes});
    if (chunk == 0)
      return VmTransferToken(std::make_unique<FaultedDeferredTransfer>(), 1);
    auto operation =
        route.route == VmDataRoute::Deferred && prepare_deferred_transfer_
            ? prepare_deferred_transfer_(route.candidate, address, chunk, bytes.first(chunk), true,
                                         gate, executing_access, address_space_)
            : nullptr;
    const VmDeferredWriteMaintenance write_maintenance =
        route.route == VmDataRoute::Deferred &&
                route.candidate == LegacyDeferredCandidate::ControlledLocalWrite && operation
            ? VmDeferredWriteMaintenance::ControlledLocalEndpoint
            : VmDeferredWriteMaintenance::Caller;
    if (!operation)
      operation = std::make_unique<FaultedDeferredTransfer>();
    return VmTransferToken(std::move(operation), chunk, write_maintenance);
  }

  [[nodiscard]] VmTranslationResult translate(uint64_t address, std::size_t size,
                                              VmAccessKind access) const override {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = VmAccessOutcome::Malformed, .translation = {}};
    const VmAccessAdmission admission = admit_access(address, size, access);
    if (!admission)
      return {.outcome = admission.outcome, .translation = {}};
    const uint64_t page_bytes = kLegacyPageSize - (address & (kLegacyPageSize - 1));
    const auto policy = address_space_->translation_policy(address, vmid_);
    // A host extent ending inside the current GPU page is a hard boundary, not
    // another transfer unit. Reject a request that crosses it before copying a
    // prefix so the non-resumable read/write APIs retain their all-or-nothing
    // physical-request contract. Ordinary page boundaries remain resumable.
    if (policy.contiguous_bytes < std::min<uint64_t>(size, page_bytes))
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation =
            {
                .domain = VmMemoryDomain::Compatibility,
                .address = address,
                .contiguous_bytes = policy.contiguous_bytes,
                .mtype = policy.mtype,
                .permissions =
                    {
                        .readable = access == VmAccessKind::Read || access == VmAccessKind::Atomic,
                        .writable = access == VmAccessKind::Write || access == VmAccessKind::Atomic,
                        .executable = access == VmAccessKind::Execute,
                    },
            },
    };
  }

  [[nodiscard]] VmTransferStep read_step(PhysicalMemoryAccess &physical, uint64_t address,
                                         std::span<std::byte> bytes,
                                         VmAccessKind access) const override {
    if (&physical != static_cast<const PhysicalMemoryAccess *>(this) ||
        access != VmAccessKind::Read)
      return AddressSpaceTranslator::read_step(physical, address, bytes, access);
    if (bytes.empty() || bytes.size() - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = VmAccessOutcome::Malformed, .report_translation_fault = true};
    if (data_route_) {
      const auto route = data_route_(address, bytes.size(), access);
      if (route.candidate != LegacyDeferredCandidate::None)
        return {.outcome = VmAccessOutcome::Faulted, .report_translation_fault = true};
      if (route.first_extent_bytes != 0)
        bytes = bytes.first(std::min(bytes.size(), route.first_extent_bytes));
    }
    const VmAccessAdmission admission = admit_access(address, bytes.size(), access);
    if (!admission)
      return {.outcome = admission.outcome, .report_translation_fault = true};
    const auto step = address_space_->read_step(
        address, std::span<uint8_t>(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size()),
        vmid_);
    return {.completed_bytes = step.completed_bytes,
            .outcome = vm_access_outcome(step.outcome),
            .report_translation_fault = step.policy_fault};
  }

  [[nodiscard]] VmTransferStep write_step(PhysicalMemoryAccess &physical, uint64_t address,
                                          std::span<const std::byte> bytes) const override {
    if (&physical != static_cast<const PhysicalMemoryAccess *>(this))
      return AddressSpaceTranslator::write_step(physical, address, bytes);
    if (bytes.empty() || bytes.size() - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = VmAccessOutcome::Malformed, .report_translation_fault = true};
    if (data_route_) {
      const auto route = data_route_(address, bytes.size(), VmAccessKind::Write);
      if (route.candidate != LegacyDeferredCandidate::None)
        return {.outcome = VmAccessOutcome::Faulted, .report_translation_fault = true};
      if (route.first_extent_bytes != 0)
        bytes = bytes.first(std::min(bytes.size(), route.first_extent_bytes));
    }
    auto permit = try_begin_raw_write(address, bytes.size());
    if (!permit)
      return {.outcome = VmAccessOutcome::Unavailable};
    const VmAccessAdmission admission = admit_access(address, bytes.size(), VmAccessKind::Write);
    if (!admission || !admission.direct_backing_write_allowed)
      return {.outcome = admission ? VmAccessOutcome::Faulted : admission.outcome,
              .report_translation_fault = true};
    const auto step = address_space_->write_step(
        address,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()),
        vmid_);
    return {.completed_bytes = step.completed_bytes,
            .outcome = vm_access_outcome(step.outcome),
            .report_translation_fault = step.policy_fault};
  }

  [[nodiscard]] std::optional<Mtype> query_mtype(uint64_t address) const override {
    return address_space_->pte_mtype(address, vmid_);
  }

  [[nodiscard]] VmMtypeSnapshot snapshot_mtype(uint64_t address) const override {
    auto guard = address_space_->acquire_page_table_request(vmid_);
    if (!mutation_epoch_ || !guard.cacheable())
      return {.mtype = guard.owns_lock() ? address_space_->pte_mtype(address, guard)
                                         : query_mtype(address)};
    bool may_batch_private_uc = false, may_batch_private_ram = false;
    const auto mtype =
        address_space_->pte_mtype(address, guard, &may_batch_private_uc, &may_batch_private_ram);
    return {.mtype = mtype,
            .may_batch_private_uc = may_batch_private_uc,
            .may_batch_private_ram = may_batch_private_ram,
            .begin = address & ~(kLegacyPageSize - 1),
            .size = kLegacyPageSize,
            .mutation_epoch = mutation_epoch_,
            .captured_epoch = mutation_epoch_->load(std::memory_order_relaxed)};
  }

  [[nodiscard]] VmTranslationResult probe_translation(uint64_t address, std::size_t size,
                                                      VmAccessKind access) const override {
    const auto translated = translate(address, size, access);
    if (!translated)
      return translated;
    // GpuVmAccess walks translation spans. Probe only this span so a large
    // range does not repeatedly rescan every remaining host page.
    const auto span_size = std::min<uint64_t>(size, translated.translation.contiguous_bytes);
    const bool accessible =
        access == VmAccessKind::Execute ? address_space_->is_fetchable(address, vmid_)
        : access == VmAccessKind::Write || access == VmAccessKind::Atomic
            ? address_space_->has_writable_host_backing(address, vmid_, span_size)
            : address_space_->has_host_backing(address, vmid_, span_size);
    if (!accessible)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return translated;
  }

  [[nodiscard]] VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                                     std::span<std::byte> bytes) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    const VmAccessAdmission admission = admit_access(address, bytes.size(), VmAccessKind::Read);
    if (!admission)
      return admission.outcome;
    return vm_access_outcome(address_space_->read_block_strict(
        address, std::span<uint8_t>(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size()),
        vmid_));
  }

  [[nodiscard]] VmAccessOutcome read_for_access(VmMemoryDomain domain, uint64_t address,
                                                std::span<std::byte> bytes,
                                                VmAccessKind access) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    const VmAccessAdmission admission = admit_access(address, bytes.size(), access);
    if (!admission)
      return admission.outcome;
    if (access != VmAccessKind::Execute)
      return read(domain, address, bytes);
    return address_space_->read_block(
               address, std::span<uint8_t>(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size()),
               vmid_) == AccessOutcome::Complete
               ? VmAccessOutcome::Complete
               : VmAccessOutcome::Faulted;
  }

  [[nodiscard]] VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                                      std::span<const std::byte> bytes) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    auto permit = try_begin_raw_write(address, bytes.size());
    if (!permit)
      return VmAccessOutcome::Unavailable;
    const VmAccessAdmission admission = admit_access(address, bytes.size(), VmAccessKind::Write);
    if (!admission)
      return admission.outcome;
    if (!admission.direct_backing_write_allowed)
      return VmAccessOutcome::Faulted;
    return vm_access_outcome(address_space_->write_block_strict(
        address,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()),
        vmid_));
  }

  [[nodiscard]] bool try_read_contiguous(VmMemoryDomain domain, uint64_t address,
                                         std::span<std::byte> bytes) override {
    return domain == VmMemoryDomain::Compatibility &&
           admit_access(address, bytes.size(), VmAccessKind::Read) &&
           address_space_->try_copy_contiguous(address, bytes.data(), bytes.size(), vmid_, false);
  }

  [[nodiscard]] bool try_write_contiguous(VmMemoryDomain domain, uint64_t address,
                                          std::span<const std::byte> bytes) override {
    if (domain != VmMemoryDomain::Compatibility)
      return false;
    auto permit = try_begin_raw_write(address, bytes.size());
    if (!permit)
      return false;
    const VmAccessAdmission admission = admit_access(address, bytes.size(), VmAccessKind::Write);
    return admission && admission.direct_backing_write_allowed &&
           address_space_->try_copy_contiguous(address, const_cast<std::byte *>(bytes.data()),
                                               bytes.size(), vmid_, true);
  }

  [[nodiscard]] AtomicLoadResult atomic_load(VmMemoryDomain domain, uint64_t address,
                                             uint32_t width) override {
    if (domain != VmMemoryDomain::Compatibility)
      return {.outcome = VmAccessOutcome::Malformed, .value = 0};
    const VmAccessAdmission admission = admit_access(address, width, VmAccessKind::Atomic);
    if (!admission)
      return {.outcome = admission.outcome, .value = 0};
    uint64_t value = 0;
    const VmAccessOutcome outcome =
        vm_access_outcome(address_space_->atomic_load(address, width, value, vmid_));
    return {.outcome = outcome, .value = outcome == VmAccessOutcome::Complete ? value : 0};
  }

  [[nodiscard]] VmAccessOutcome atomic_store(VmMemoryDomain domain, uint64_t address,
                                             uint32_t width, uint64_t value) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    auto permit = try_begin_raw_write(address, width);
    if (!permit)
      return VmAccessOutcome::Unavailable;
    const VmAccessAdmission admission = admit_access(address, width, VmAccessKind::Atomic);
    if (!admission)
      return admission.outcome;
    if (!admission.direct_backing_write_allowed)
      return VmAccessOutcome::Faulted;
    return vm_access_outcome(address_space_->atomic_store_strict(address, width, value, vmid_));
  }

  [[nodiscard]] AtomicCompareExchangeResult compare_exchange(VmMemoryDomain domain,
                                                             uint64_t address, uint32_t width,
                                                             uint64_t expected,
                                                             uint64_t desired) override {
    if (domain != VmMemoryDomain::Compatibility)
      return {.outcome = VmAccessOutcome::Malformed};
    auto permit = try_begin_raw_write(address, width);
    if (!permit)
      return {.outcome = VmAccessOutcome::Unavailable};
    const VmAccessAdmission admission = admit_access(address, width, VmAccessKind::Atomic);
    if (!admission)
      return {.outcome = admission.outcome};
    if (!admission.direct_backing_write_allowed)
      return {.outcome = VmAccessOutcome::Faulted};
    uint64_t observed = 0;
    bool exchanged = false;
    const CopyOutcome outcome = address_space_->atomic_compare_exchange_strict(
        address, width, expected, desired, observed, exchanged, vmid_);
    return {.outcome = vm_access_outcome(outcome), .observed = observed, .exchanged = exchanged};
  }

  [[nodiscard]] VmAccessOutcome atomic_modify(VmMemoryDomain domain, uint64_t address,
                                              uint32_t width,
                                              const AtomicMutation &mutation) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    auto permit = try_begin_raw_write(address, width);
    if (!permit)
      return VmAccessOutcome::Unavailable;
    const VmAccessAdmission admission = admit_access(address, width, VmAccessKind::Atomic);
    if (!admission)
      return admission.outcome;
    if (!admission.direct_backing_write_allowed)
      return VmAccessOutcome::Faulted;
    const CopyOutcome outcome = address_space_->atomic_modify_strict(
        address, width,
        [&](uint8_t *bytes) {
          mutation(std::span<std::byte>(reinterpret_cast<std::byte *>(bytes), width));
        },
        vmid_);
    return vm_access_outcome(outcome);
  }

  [[nodiscard]] VmAccessOutcome atomic_modify_lanes(VmMemoryDomain domain,
                                                    std::span<const uint64_t> addresses,
                                                    uint64_t lane_mask, uint32_t width,
                                                    const AtomicLaneMutation &mutation,
                                                    uint32_t &next_lane) override {
    if (domain != VmMemoryDomain::Compatibility)
      return VmAccessOutcome::Malformed;
    return vm_access_outcome(address_space_->atomic_modify_strict_lanes(
        addresses, lane_mask, width,
        [&](uint32_t lane, uint8_t *bytes) {
          mutation(lane, std::span<std::byte>(reinterpret_cast<std::byte *>(bytes), width));
        },
        vmid_, next_lane));
  }

  [[nodiscard]] std::byte *resolve_host_pointer(VmMemoryDomain domain, uint64_t address,
                                                std::size_t size) const override {
    if (domain != VmMemoryDomain::Compatibility)
      return nullptr;
    // A borrowed pointer is writable even when it was requested as a read.
    if (!raw_write_route_is_local(address, size))
      return nullptr;
    if (!admit_access(address, size, VmAccessKind::Read))
      return nullptr;
    return reinterpret_cast<std::byte *>(address_space_->resolve_host_ptr(address, vmid_, size));
  }

  [[nodiscard]] std::pair<uint64_t, uint64_t> host_range(VmMemoryDomain domain,
                                                         uint64_t address) const override {
    if (domain != VmMemoryDomain::Compatibility)
      return {0, 0};
    const auto range = address_space_->find_host_range(address, vmid_);
    const auto *at_address = address_space_->resolve_host_ptr(address, vmid_, 1);
    const auto host_address = reinterpret_cast<uint64_t>(at_address);
    if (range.second == 0 || at_address == nullptr || host_address < range.first ||
        host_address - range.first >= range.second || address < host_address - range.first)
      return {0, 0};
    const uint64_t gpu_begin = address - (host_address - range.first);
    return raw_write_route_is_local(gpu_begin, range.second) &&
                   admit_access(gpu_begin, range.second, VmAccessKind::Read)
               ? range
               : std::pair<uint64_t, uint64_t>{0, 0};
  }

  void report_fault(uint64_t address) const { address_space_->report_vm_fault(vmid_, address); }
  void set_client_pid(pid_t client_pid) {
    address_space_->set_process_client_pid(vmid_, client_pid);
  }
  void set_client_mem_fd(int mem_fd) { address_space_->set_process_mem_fd(vmid_, mem_fd); }
  void set_passthrough(bool passthrough) {
    address_space_->set_process_passthrough(vmid_, passthrough);
  }
  void set_fault_reporter(MemoryFaultReporter *reporter) {
    address_space_->set_process_fault_reporter(vmid_, reporter);
  }
  LegacyAddressSpace &address_space() { return *address_space_; }
  const LegacyAddressSpace &address_space() const { return *address_space_; }
  std::shared_ptr<LegacyAddressSpace> address_space_snapshot() { return address_space_; }
  std::shared_ptr<const LegacyAddressSpace> address_space_snapshot() const {
    return address_space_;
  }

private:
  std::shared_ptr<GpuMemory> memory_;
  std::shared_ptr<LegacyAddressSpace> address_space_;
  uint32_t vmid_ = 0;
  // Synchronous word reads and stores must not destroy the final request-mutex owner
  // under the caller's VM-state lock, even after raw registry replacement.
  const std::shared_ptr<util::DistributedSharedMutex> original_request_mutex_;
  std::shared_ptr<const std::atomic<uint64_t>> mutation_epoch_;
  std::shared_ptr<VmRawBackingWriteGate> raw_write_gate_;
  std::function<VmAccessAdmission(uint64_t, std::size_t, VmAccessKind)> access_admission_;
  std::function<LegacyDataRoute(uint64_t, std::size_t, VmAccessKind)> data_route_;
  std::function<std::unique_ptr<VmDeferredTransfer>(
      LegacyDeferredCandidate, uint64_t, std::size_t, std::span<const std::byte>, bool,
      const std::shared_ptr<VmTransferEffectGate> &, const GpuVmAccess &,
      std::shared_ptr<const LegacyAddressSpace>)>
      prepare_deferred_transfer_;
  std::shared_ptr<void> frontend_lifetime_;
};

LegacyGpuVmAdapter::LegacyGpuVmAdapter(GpuVm &gpu_vm, GpuMemory *memory)
    : LegacyGpuVmAdapter(gpu_vm, std::shared_ptr<GpuMemory>(memory, [](GpuMemory *) {
                           // The topology owns this backing and outlives its KFD adapter.
                         })) {}

LegacyGpuVmAdapter::LegacyGpuVmAdapter(GpuVm &gpu_vm, std::shared_ptr<GpuMemory> memory)
    : gpu_vm_(&gpu_vm), memory_(std::move(memory)) {}

LegacyGpuVmAdapter::~LegacyGpuVmAdapter() {
  std::lock_guard lock(mutex_);
  for (const auto &[vmid, entry] : bindings_) {
    (void)vmid;
    (void)gpu_vm_->unregister_address_space(entry.handle);
    revoke_binding(entry.binding);
  }
}

bool LegacyGpuVmAdapter::set_memory(GpuMemory *memory) {
  return set_memory(std::shared_ptr<GpuMemory>(memory, [](GpuMemory *) {
    // The topology owns this backing and outlives its KFD adapter.
  }));
}

bool LegacyGpuVmAdapter::set_memory(std::shared_ptr<GpuMemory> memory) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  if (!bindings_.empty())
    return false;
  memory_ = memory;
  return true;
}

AddressSpaceHandle
LegacyGpuVmAdapter::register_address_space(uint32_t vmid,
                                           LegacyAddressSpaceRegistration registration,
                                           std::shared_ptr<void> frontend_lifetime) {
  if (registration.page_table == nullptr || registration.page_table_mutex == nullptr)
    return {};
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  if (memory_ == nullptr || bindings_.contains(vmid))
    return {};
  auto binding = std::make_shared<Binding>(memory_, vmid, std::move(registration),
                                           std::move(frontend_lifetime));
  const AddressSpaceHandle handle = gpu_vm_->register_address_space(
      vmid, binding, binding,
      [binding](uint64_t address, VmAccessKind) { binding->report_fault(address); }, true);
  if (!handle)
    return {};
  bindings_.emplace(vmid, Entry{.handle = handle, .binding = std::move(binding)});
  return handle;
}

AddressSpaceHandle LegacyGpuVmAdapter::register_address_space(
    uint32_t vmid, LegacyPageTable *page_table, util::DistributedSharedMutex *page_table_mutex,
    const uint64_t *page_table_generation,
    std::shared_ptr<util::DistributedSharedMutex> request_mutex,
    std::shared_ptr<void> frontend_lifetime) {
  return register_address_space(vmid,
                                {.page_table = page_table,
                                 .page_table_mutex = page_table_mutex,
                                 .page_table_generation = page_table_generation,
                                 .request_mutex = std::move(request_mutex),
                                 .access_admission = {}},
                                std::move(frontend_lifetime));
}

bool LegacyGpuVmAdapter::unregister_address_space(AddressSpaceHandle handle) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  const auto found = std::ranges::find_if(
      bindings_, [handle](const auto &entry) { return entry.second.handle == handle; });
  if (found == bindings_.end() || !gpu_vm_->unregister_address_space(handle))
    return false;
  revoke_binding(found->second.binding);
  bindings_.erase(found);
  return true;
}

bool LegacyGpuVmAdapter::unregister_vmid(uint32_t vmid) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  const auto found = bindings_.find(vmid);
  if (found == bindings_.end() || !gpu_vm_->unregister_address_space(found->second.handle))
    return false;
  revoke_binding(found->second.binding);
  bindings_.erase(found);
  return true;
}

bool LegacyGpuVmAdapter::set_client_pid(AddressSpaceHandle handle, pid_t client_pid) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_client_pid(client_pid);
  return true;
}

bool LegacyGpuVmAdapter::set_client_mem_fd(AddressSpaceHandle handle, int mem_fd) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_client_mem_fd(mem_fd);
  return true;
}

bool LegacyGpuVmAdapter::set_passthrough(AddressSpaceHandle handle, bool passthrough) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_passthrough(passthrough);
  return true;
}

bool LegacyGpuVmAdapter::set_fault_reporter(AddressSpaceHandle handle,
                                            MemoryFaultReporter *reporter) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  Entry *entry = find_locked(handle);
  if (entry == nullptr)
    return false;
  entry->binding->set_fault_reporter(reporter);
  return true;
}

LegacyAddressSpace *LegacyGpuVmAdapter::address_space(uint32_t vmid) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  const auto found = bindings_.find(vmid);
  return found == bindings_.end() ? nullptr : &found->second.binding->address_space();
}

const LegacyAddressSpace *LegacyGpuVmAdapter::address_space(uint32_t vmid) const {
  std::lock_guard lock(mutex_);
  const auto found = bindings_.find(vmid);
  return found == bindings_.end() || !gpu_vm_->lookup(found->second.handle).has_value()
             ? nullptr
             : &found->second.binding->address_space();
}

std::shared_ptr<LegacyAddressSpace> LegacyGpuVmAdapter::address_space_snapshot(uint32_t vmid) {
  std::lock_guard lock(mutex_);
  remove_stale_locked();
  const auto found = bindings_.find(vmid);
  return found == bindings_.end() ? nullptr : found->second.binding->address_space_snapshot();
}

std::shared_ptr<const LegacyAddressSpace>
LegacyGpuVmAdapter::address_space_snapshot(uint32_t vmid) const {
  std::lock_guard lock(mutex_);
  const auto found = bindings_.find(vmid);
  return found == bindings_.end() || !gpu_vm_->lookup(found->second.handle).has_value()
             ? nullptr
             : found->second.binding->address_space_snapshot();
}

void LegacyGpuVmAdapter::revoke_binding(const std::shared_ptr<Binding> &binding) {
  if (binding != nullptr)
    binding->set_fault_reporter(nullptr);
}

void LegacyGpuVmAdapter::remove_stale_locked() {
  for (auto entry = bindings_.begin(); entry != bindings_.end();) {
    if (gpu_vm_->lookup(entry->second.handle).has_value()) {
      ++entry;
      continue;
    }
    revoke_binding(entry->second.binding);
    entry = bindings_.erase(entry);
  }
}

LegacyGpuVmAdapter::Entry *LegacyGpuVmAdapter::find_locked(AddressSpaceHandle handle) {
  for (auto &[vmid, entry] : bindings_) {
    (void)vmid;
    if (entry.handle == handle)
      return &entry;
  }
  return nullptr;
}

const LegacyGpuVmAdapter::Entry *LegacyGpuVmAdapter::find_locked(AddressSpaceHandle handle) const {
  for (const auto &[vmid, entry] : bindings_) {
    (void)vmid;
    if (entry.handle == handle)
      return &entry;
  }
  return nullptr;
}

} // namespace rocjitsu::amdgpu
