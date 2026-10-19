// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file async_vm_test_support.h
/// @brief Share asynchronous VM fixtures and helpers among tests.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/canonical_line_freshness.h"
#include "simdojo/sim/simulation.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rocjitsu::test {

using amdgpu::AddressSpaceTranslator;
using amdgpu::AtomicLoadResult;
using amdgpu::PhysicalMemoryAccess;
using amdgpu::VmAccessKind;
using amdgpu::VmAccessOutcome;
using amdgpu::VmDataRoute;
using amdgpu::VmDeferredTransfer;
using amdgpu::VmMemoryDomain;
using amdgpu::VmTransferEffectGate;
using amdgpu::VmTransferStep;
using amdgpu::VmTransferToken;
using amdgpu::VmTranslationResult;

/// Owned storage shared by the request issuer and the Simdojo owner event.
class AsyncVmBacking {
public:
  explicit AsyncVmBacking(std::size_t size) : bytes(size) {}

  std::mutex mutex;
  std::vector<std::byte> bytes;
  std::atomic<uint32_t> posts{0};
  std::atomic<uint32_t> read_effects{0};
  std::atomic<uint32_t> write_effects{0};
  std::atomic<uint32_t> cancellations{0};
  std::function<void()> submit_probe;
  // Optional synthetic stamp for serialized cache-probe tests. Tests using
  // concurrent writers must provide their own writer authority.
  std::shared_ptr<CanonicalLineFreshness> read_freshness;
  uint64_t read_backing_id = 0;
};

struct AsyncVmRequest {
  enum class Phase : uint8_t { Queued, Entered, Done, Revoked };
  std::shared_ptr<AsyncVmBacking> backing;
  std::shared_ptr<VmTransferEffectGate> effect_gate;
  uint64_t address = 0;
  bool write = false;
  std::vector<std::byte> payload;
  std::vector<std::byte> response;
  std::atomic<Phase> phase{Phase::Queued};
  VmAccessOutcome outcome = VmAccessOutcome::Unavailable;
  std::optional<CanonicalLineFreshness::ReadStamp> read_stamp;
  std::atomic<bool> cancelled{false};
};

class AsyncVmOwner final : public simdojo::Component {
public:
  AsyncVmOwner() : Component("async-vm-owner") {
    event_.set_handler([this](simdojo::Tick, simdojo::Message *) { service(); });
  }

  void enqueue(std::shared_ptr<AsyncVmRequest> request) {
    {
      std::lock_guard lock(mutex_);
      queued_.push_back(std::move(request));
    }
    engine()->schedule_event_now(&event_);
  }

private:
  void service() {
    std::shared_ptr<AsyncVmRequest> request;
    {
      std::lock_guard lock(mutex_);
      if (queued_.empty())
        return;
      request = std::move(queued_.front());
      queued_.pop_front();
    }
    AsyncVmRequest::Phase expected = AsyncVmRequest::Phase::Queued;
    if (!request->phase.compare_exchange_strong(expected, AsyncVmRequest::Phase::Entered,
                                                std::memory_order_acq_rel))
      return;
    if (!request->effect_gate || !request->effect_gate->commit([&] {
          std::lock_guard backing_lock(request->backing->mutex);
          const std::size_t size =
              request->write ? request->payload.size() : request->response.size();
          if (request->address > request->backing->bytes.size() ||
              size > request->backing->bytes.size() - request->address) {
            request->outcome = VmAccessOutcome::Faulted;
            request->phase.store(AsyncVmRequest::Phase::Done, std::memory_order_release);
            return;
          }
          const auto at = request->backing->bytes.begin() + request->address;
          if (request->write) {
            std::ranges::copy(request->payload, at);
            ++request->backing->write_effects;
          } else {
            std::ranges::copy_n(at, size, request->response.begin());
            ++request->backing->read_effects;
            if (request->backing->read_freshness)
              request->read_stamp = request->backing->read_freshness->try_capture_read(
                  request->backing->read_backing_id, request->address, size);
          }
          request->outcome = VmAccessOutcome::Complete;
          request->phase.store(AsyncVmRequest::Phase::Done, std::memory_order_release);
        }))
      request->phase.store(AsyncVmRequest::Phase::Revoked, std::memory_order_release);
  }

  simdojo::Event event_{this, simdojo::EventType::TIMER_CALLBACK};
  std::mutex mutex_;
  std::deque<std::shared_ptr<AsyncVmRequest>> queued_;
};

class AsyncVmOperation final : public VmDeferredTransfer {
public:
  AsyncVmOperation(AsyncVmOwner &owner, std::shared_ptr<AsyncVmRequest> request)
      : owner_(owner), request_(std::move(request)) {}

  void submit() override {
    ++request_->backing->posts;
    if (request_->backing->submit_probe)
      request_->backing->submit_probe();
    owner_.enqueue(request_);
  }

  VmAccessOutcome poll(std::span<std::byte> result) override {
    const AsyncVmRequest::Phase phase = request_->phase.load(std::memory_order_acquire);
    if (phase == AsyncVmRequest::Phase::Queued || phase == AsyncVmRequest::Phase::Entered)
      return VmAccessOutcome::Unavailable;
    if (phase == AsyncVmRequest::Phase::Revoked)
      return VmAccessOutcome::Revoked;
    if (request_->outcome == VmAccessOutcome::Complete && !request_->write)
      std::ranges::copy(request_->response, result.begin());
    return request_->outcome;
  }

  VmAccessOutcome reconcile_revocation(std::span<std::byte> result) override {
    AsyncVmRequest::Phase phase = request_->phase.load(std::memory_order_acquire);
    if (phase == AsyncVmRequest::Phase::Queued) {
      AsyncVmRequest::Phase expected = AsyncVmRequest::Phase::Queued;
      if (request_->phase.compare_exchange_strong(expected, AsyncVmRequest::Phase::Revoked,
                                                  std::memory_order_acq_rel))
        return VmAccessOutcome::Revoked;
      phase = expected;
    }
    if (phase == AsyncVmRequest::Phase::Entered)
      return VmAccessOutcome::Unavailable;
    if (phase == AsyncVmRequest::Phase::Revoked)
      return VmAccessOutcome::Revoked;
    if (request_->outcome == VmAccessOutcome::Complete && !request_->write)
      std::ranges::copy(request_->response, result.begin());
    return request_->outcome;
  }

  [[nodiscard]] std::optional<CanonicalLineFreshness::ReadStamp>
  completed_read_stamp() const override {
    return request_->read_stamp;
  }

  void cancel() noexcept override {
    AsyncVmRequest::Phase expected = AsyncVmRequest::Phase::Queued;
    request_->phase.compare_exchange_strong(expected, AsyncVmRequest::Phase::Revoked,
                                            std::memory_order_acq_rel);
    if (!request_->cancelled.exchange(true, std::memory_order_acq_rel)) {
      ++request_->backing->cancellations;
    }
  }

private:
  AsyncVmOwner &owner_;
  std::shared_ptr<AsyncVmRequest> request_;
};

/// Fake translator that defers only its selected data range in four-byte steps.
/// Ordinary control addresses continue through the synchronous backing path.
class AsyncVmMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  AsyncVmMemory(AsyncVmOwner &owner, std::size_t size, uint64_t deferred_begin,
                uint64_t deferred_end)
      : backing(std::make_shared<AsyncVmBacking>(size)), owner_(owner),
        deferred_begin_(deferred_begin), deferred_end_(deferred_end) {}

  VmDataRoute classify_data_route(uint64_t address, std::size_t size,
                                  VmAccessKind access) const override {
    if (!overlaps_deferred(address, size))
      return VmDataRoute::Local;
    return access == VmAccessKind::Read || access == VmAccessKind::Write ? VmDataRoute::Deferred
                                                                         : VmDataRoute::Faulted;
  }

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0 || address >= backing->bytes.size())
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    std::size_t contiguous = std::min<std::size_t>(size, backing->bytes.size() - address);
    if (address < deferred_begin_)
      contiguous = std::min<std::size_t>(contiguous, deferred_begin_ - address);
    else if (is_deferred(address))
      contiguous = std::min<std::size_t>(contiguous, deferred_end_ - address);
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = contiguous,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  std::optional<VmTransferToken>
  prepare_deferred_read(PhysicalMemoryAccess &, uint64_t address, std::size_t size,
                        VmAccessKind access,
                        const std::shared_ptr<VmTransferEffectGate> &effect_gate) const override {
    if (access != VmAccessKind::Read || !is_deferred(address))
      return std::nullopt;
    const std::size_t chunk = std::min<std::size_t>({size, 4, deferred_end_ - address});
    auto request = std::make_shared<AsyncVmRequest>();
    request->backing = backing;
    request->effect_gate = effect_gate;
    request->address = address;
    request->response.resize(chunk);
    return VmTransferToken(std::make_unique<AsyncVmOperation>(owner_, std::move(request)), chunk);
  }

  std::optional<VmTransferToken>
  prepare_deferred_write(PhysicalMemoryAccess &, uint64_t address, std::span<const std::byte> bytes,
                         const std::shared_ptr<VmTransferEffectGate> &effect_gate) const override {
    if (!is_deferred(address))
      return std::nullopt;
    const std::size_t chunk = std::min<std::size_t>({bytes.size(), 4, deferred_end_ - address});
    auto request = std::make_shared<AsyncVmRequest>();
    request->backing = backing;
    request->effect_gate = effect_gate;
    request->address = address;
    request->write = true;
    request->payload.assign(bytes.begin(), bytes.begin() + chunk);
    return VmTransferToken(std::make_unique<AsyncVmOperation>(owner_, std::move(request)), chunk);
  }

  VmTransferStep read_step(PhysicalMemoryAccess &memory, uint64_t address,
                           std::span<std::byte> bytes, VmAccessKind access) const override {
    if (access == VmAccessKind::Execute && is_deferred(address))
      return {.outcome = VmAccessOutcome::Malformed};
    return AddressSpaceTranslator::read_step(memory, address, bytes, access);
  }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    if (overlaps_deferred(address, bytes.size()) || address > backing->bytes.size() ||
        bytes.size() > backing->bytes.size() - address)
      return VmAccessOutcome::Faulted;
    std::lock_guard lock(backing->mutex);
    std::ranges::copy_n(backing->bytes.begin() + address, bytes.size(), bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (overlaps_deferred(address, bytes.size()) || address > backing->bytes.size() ||
        bytes.size() > backing->bytes.size() - address)
      return VmAccessOutcome::Faulted;
    std::lock_guard lock(backing->mutex);
    std::ranges::copy(bytes, backing->bytes.begin() + address);
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    if ((width != 4 && width != 8) || overlaps_deferred(address, width) ||
        address > backing->bytes.size() || width > backing->bytes.size() - address)
      return {.outcome = VmAccessOutcome::Malformed};
    uint64_t value = 0;
    std::lock_guard lock(backing->mutex);
    std::memcpy(&value, backing->bytes.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if ((width != 4 && width != 8) || overlaps_deferred(address, width) ||
        address > backing->bytes.size() || width > backing->bytes.size() - address)
      return VmAccessOutcome::Malformed;
    std::lock_guard lock(backing->mutex);
    std::memcpy(backing->bytes.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  void store(uint64_t address, std::span<const std::byte> bytes) {
    std::lock_guard lock(backing->mutex);
    std::ranges::copy(bytes, backing->bytes.begin() + address);
  }

  [[nodiscard]] std::vector<std::byte> load(uint64_t address, std::size_t size) const {
    std::lock_guard lock(backing->mutex);
    return {backing->bytes.begin() + address, backing->bytes.begin() + address + size};
  }

  std::shared_ptr<AsyncVmBacking> backing;

private:
  [[nodiscard]] bool is_deferred(uint64_t address) const {
    return address >= deferred_begin_ && address < deferred_end_;
  }
  [[nodiscard]] bool overlaps_deferred(uint64_t address, std::size_t size) const {
    return address < deferred_end_ && size > deferred_begin_ - std::min(address, deferred_begin_);
  }

  AsyncVmOwner &owner_;
  uint64_t deferred_begin_;
  uint64_t deferred_end_;
};

class AsyncVmHarness {
public:
  AsyncVmHarness() : engine({.num_threads = 1}) {
    auto root = std::make_unique<simdojo::CompositeComponent>("root");
    owner = static_cast<AsyncVmOwner *>(root->add_child(std::make_unique<AsyncVmOwner>()));
    engine.topology().set_root(std::move(root));
    engine.create();
  }

  simdojo::SimulationEngine engine;
  AsyncVmOwner *owner = nullptr;
};

} // namespace rocjitsu::test
