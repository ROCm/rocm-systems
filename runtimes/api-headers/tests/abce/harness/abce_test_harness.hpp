/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// GPU-free test harness for ABCE.
//
// ABCE is pure logic over plain memory, so a "ring" is a byte array, the engine
// is a function that moves the read index, and a frame is bytes the decoder can
// walk. Nothing here touches a driver.

#ifndef ABCE_TEST_HARNESS_HPP_
#define ABCE_TEST_HARNESS_HPP_

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "abce_decode.h"
#include "abce_host.h"
#include "abce_ring_host.h"

#if defined(__SANITIZE_THREAD__)
extern "C" {
void __tsan_ignore_thread_begin();
void __tsan_ignore_thread_end();
void __tsan_acquire(void* addr);
void __tsan_release(void* addr);
}
#endif  // __SANITIZE_THREAD__

namespace abce_test {

// The fake engine stands in for hardware, which is not a participant in the C++
// memory model: it reads ring bytes the producers wrote and advances the read
// pointer without any ordering ThreadSanitizer can see. Accesses made inside
// this scope are invisible to it, so a CPU thread can play the engine without
// producing reports about the engine. Producer-versus-producer ordering and the
// commit serialisation stay fully checked.
class HardwareAccess {
 public:
  HardwareAccess() {
#if defined(__SANITIZE_THREAD__)
    __tsan_ignore_thread_begin();
#endif  // __SANITIZE_THREAD__
  }
  ~HardwareAccess() {
#if defined(__SANITIZE_THREAD__)
    __tsan_ignore_thread_end();
#endif  // __SANITIZE_THREAD__
  }
  HardwareAccess(const HardwareAccess&) = delete;
  HardwareAccess& operator=(const HardwareAccess&) = delete;
};

// Hardware-mediated ordering ThreadSanitizer cannot see: the engine finishes
// reading ring bytes, then advances the read pointer (release); a producer that
// observes the new read pointer (acquire) may then overwrite those bytes.
inline void HardwareRelease(void* addr) {
#if defined(__SANITIZE_THREAD__)
  __tsan_release(addr);
#else
  (void)addr;
#endif  // __SANITIZE_THREAD__
}
inline void HardwareAcquire(void* addr) {
#if defined(__SANITIZE_THREAD__)
  __tsan_acquire(addr);
#else
  (void)addr;
#endif  // __SANITIZE_THREAD__
}

//===----------------------------------------------------------------------===//
// ISA matrix
//===----------------------------------------------------------------------===//

struct IsaCase {
  const char* name;
  abce_isa_version_t isa;
  bool is_gfx125plus;
  bool swap_supported;
};

// One part per distinct packet-capability tier (OSS4 / OSS5 / OSS7), including
// both gfx9 swap-capable and swap-less parts.
inline const std::vector<IsaCase>& AllIsas() {
  static const std::vector<IsaCase> kIsas = {
      {"gfx90a", {9, 0, 10}, false, false}, {"gfx942", {9, 4, 2}, false, true},
      {"gfx950", {9, 5, 0}, false, true},   {"gfx1030", {10, 3, 0}, false, false},
      {"gfx1100", {11, 0, 0}, false, false}, {"gfx1201", {12, 0, 1}, false, false},
      {"gfx1250", {12, 5, 0}, true, true},
  };
  return kIsas;
}

inline IsaCase IsaByName(const std::string& name) {
  for (const IsaCase& isa_case : AllIsas())
    if (name == isa_case.name) return isa_case;
  return AllIsas().front();
}

//===----------------------------------------------------------------------===//
// Decoding
//===----------------------------------------------------------------------===//

struct DecodedPacket {
  std::string name;
  uint32_t dwords;
  size_t byte_offset;
};

struct WalkResult {
  std::vector<DecodedPacket> packets;
  // True only when the decoder lands exactly on the last byte and recognised
  // every packet. This is the sizing-vs-emission check: a mismatch means the
  // byte count the composer reserved is not the byte count it emitted.
  bool clean = false;
  size_t consumed_bytes = 0;

  std::vector<std::string> Names(bool skip_nops = true) const {
    std::vector<std::string> names;
    for (const DecodedPacket& packet : packets)
      if (!(skip_nops && packet.name == "NOP")) names.push_back(packet.name);
    return names;
  }
};

inline WalkResult Walk(const void* bytes, size_t num_bytes, abce_isa_version_t isa) {
  WalkResult result;
  const uint32_t* words = static_cast<const uint32_t*>(bytes);
  const uint32_t total_dwords = static_cast<uint32_t>(num_bytes / sizeof(uint32_t));
  uint32_t cursor_dword = 0;
  bool recognised = (num_bytes % sizeof(uint32_t)) == 0;
  while (cursor_dword < total_dwords) {
    const abce_decoded_packet_t decoded =
        abce_decode_sdma_packet(words + cursor_dword, total_dwords - cursor_dword, isa);
    const std::string name = decoded.name;
    if (name == "UNKNOWN" || name == "TRUNCATED") recognised = false;
    result.packets.push_back({name, decoded.dwords, cursor_dword * sizeof(uint32_t)});
    cursor_dword += decoded.dwords;
  }
  result.consumed_bytes = static_cast<size_t>(cursor_dword) * sizeof(uint32_t);
  result.clean = recognised && result.consumed_bytes == num_bytes;
  return result;
}

//===----------------------------------------------------------------------===//
// FakeRing: an SDMA ring over plain memory
//===----------------------------------------------------------------------===//

struct RingOptions {
  size_t size_bytes = 1u << 16;
  size_t min_submission_size = 0;
  bool align64 = false;
  // Seeds the write pointer, modelling an attach to a queue already in use.
  uint64_t initial_write_index = 0;
  // When true the "engine" consumes every doorbell at once, so a producer never
  // waits for space. Set false to drive back-pressure by hand.
  bool engine_consumes_instantly = true;
  // Queue-control backend: the directly mapped write/read/doorbell words, or the
  // abce_ring_queue_ops_t callbacks for APIs that keep them opaque (what the HSA
  // adapter uses). The callback form is also the one ThreadSanitizer can follow,
  // because it lets the harness annotate the engine's ordering.
  bool use_queue_ops = false;
};

class FakeRing {
 public:
  explicit FakeRing(const RingOptions& options = RingOptions())
      : options_(options),
        storage_((options.size_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t)),
        write_index_(options.initial_write_index),
        read_index_(options.initial_write_index),
        doorbell_(options.initial_write_index),
        published_index_(options.initial_write_index),
        last_captured_index_(options.initial_write_index) {
    abce_ring_config_t config;
    std::memset(&config, 0, sizeof(config));
    config.base = reinterpret_cast<char*>(storage_.data());
    config.size = options.size_bytes;
    if (options.use_queue_ops) {
      config.queue_ops.load_read_index = &FakeRing::LoadReadIndex;
      config.queue_ops.load_write_index = &FakeRing::LoadWriteIndex;
      config.queue_ops.publish_write_index = &FakeRing::PublishWriteIndex;
      config.queue_ops.user_data = this;
    } else {
      config.write_ptr = &write_index_;
      config.read_ptr = &read_index_;
      config.doorbell = &doorbell_;
    }
    config.min_submission_size = options.min_submission_size;
    config.align64 = options.align64;
    abce_ring_initialize(&config, &FakeRing::OnDoorbell, this, &ring_);
  }

  FakeRing(const FakeRing&) = delete;
  FakeRing& operator=(const FakeRing&) = delete;

  abce_ring_t* ring() { return &ring_; }
  char* base() { return reinterpret_cast<char*>(storage_.data()); }
  size_t size_bytes() const { return options_.size_bytes; }

  uint64_t write_index() const { return __atomic_load_n(&write_index_, __ATOMIC_ACQUIRE); }
  uint64_t doorbell() const { return __atomic_load_n(&doorbell_, __ATOMIC_ACQUIRE); }

  // Every doorbell value rung, in ring order. A well-behaved ring only ever
  // rings strictly increasing values.
  const std::vector<uint64_t>& doorbell_history() const { return doorbell_history_; }

  // Every byte published since construction, in order, NOP wrap padding
  // included. Equal to the concatenation of what the engine would have read.
  const std::vector<uint8_t>& stream() const { return stream_; }

  // Last index the ring published, as the engine would observe it. Safe to read
  // while producers run; write_index() reads the hardware word itself and is for
  // use once the producers are quiescent.
  uint64_t published_index() const { return published_index_.load(std::memory_order_acquire); }

  void EngineConsumeAll() { SetReadIndex(published_index()); }
  void SetReadIndex(uint64_t new_read_index) {
    HardwareRelease(const_cast<uint64_t*>(&read_index_));
    HardwareAccess engine;
    __atomic_store_n(&read_index_, new_read_index, __ATOMIC_RELEASE);
  }

 private:
  static uint64_t LoadReadIndex(void* user_data) {
    FakeRing* self = static_cast<FakeRing*>(user_data);
    const uint64_t value = __atomic_load_n(&self->read_index_, __ATOMIC_RELAXED);
    HardwareAcquire(const_cast<uint64_t*>(&self->read_index_));
    return value;
  }

  static uint64_t LoadWriteIndex(void* user_data) {
    return __atomic_load_n(&static_cast<FakeRing*>(user_data)->write_index_, __ATOMIC_RELAXED);
  }

  static void PublishWriteIndex(void* user_data, uint64_t new_write_index) {
    FakeRing* self = static_cast<FakeRing*>(user_data);
    __atomic_store_n(&self->write_index_, new_write_index, __ATOMIC_RELEASE);
    __atomic_store_n(&self->doorbell_, new_write_index, __ATOMIC_RELEASE);
  }

  static void OnDoorbell(void* user_data, uint64_t new_write_index) {
    static_cast<FakeRing*>(user_data)->Captured(new_write_index);
  }

  // Called by the ring after the write pointer and doorbell are stored and,
  // for one ring, in commit order, so no lock is needed here.
  void Captured(uint64_t new_write_index) {
    const char* ring_base = reinterpret_cast<const char*>(storage_.data());
    {
      HardwareAccess engine;
      for (uint64_t index = last_captured_index_; index < new_write_index; ++index)
        stream_.push_back(static_cast<uint8_t>(ring_base[index & (options_.size_bytes - 1)]));
    }
    last_captured_index_ = new_write_index;
    published_index_.store(new_write_index, std::memory_order_release);
    doorbell_history_.push_back(new_write_index);
    if (options_.engine_consumes_instantly) SetReadIndex(new_write_index);
  }

  RingOptions options_;
  std::vector<uint64_t> storage_;
  volatile uint64_t write_index_;
  volatile uint64_t read_index_;
  volatile uint64_t doorbell_;
  std::atomic<uint64_t> published_index_;
  uint64_t last_captured_index_;
  std::vector<uint64_t> doorbell_history_;
  std::vector<uint8_t> stream_;
  abce_ring_t ring_;
};

//===----------------------------------------------------------------------===//
// Mapper: an orchestrator wired to fake rings
//===----------------------------------------------------------------------===//

// Fake addresses. ABCE never dereferences a caller pointer, so tests can use
// addresses that map to nothing.
inline void* FakeAddress(uint64_t offset) {
  return reinterpret_cast<void*>(static_cast<uintptr_t>(0x7f0000000000ull + offset));
}

inline abce_copy_op_t MakeHostToDevice(uint64_t dst_offset, uint64_t src_offset, size_t size) {
  abce_copy_op_t op;
  abce_copy_op_initialize(&op);
  op.kind = ABCE_OP_KIND_LINEAR;
  op.dst = FakeAddress(dst_offset);
  op.src = FakeAddress(src_offset);
  op.size = size;
  op.src_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, ABCE_ANY_DEVICE);
  op.dst_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, 0);
  return op;
}

inline abce_copy_op_t MakeDeviceToDevice(uint64_t dst_offset, uint64_t src_offset, size_t size) {
  abce_copy_op_t op = MakeHostToDevice(dst_offset, src_offset, size);
  op.src_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, 0);
  return op;
}

struct MapperOptions {
  uint32_t num_engines = 1;
  RingOptions ring_options;
  // Override the ISA-derived platform capabilities (HDP flush, atomics, GCR).
  std::optional<abce_platform_caps_t> caps;
};

// The 64-bit words ABCE writes, kept 64-byte aligned like a real signal object.
struct alignas(64) SignalWords {
  uint64_t value = 0;
  uint64_t coordination = 0;
};

class Mapper {
 public:
  explicit Mapper(const IsaCase& isa_case, const MapperOptions& options = MapperOptions())
      : isa_case_(isa_case) {
    RequireOrAbort(abce_status_is_ok(abce_builder_initialize(isa_case.isa, nullptr, &builder_)));
    orchestrator_ = std::make_unique<abce_copy_orchestrator_t>();
    if (options.caps) {
      abce_copy_orchestrator_initialize_with_caps(&builder_, *options.caps, orchestrator_.get());
    } else {
      abce_copy_orchestrator_initialize(&builder_, orchestrator_.get());
    }
    for (uint32_t engine_slot = 0; engine_slot < options.num_engines; ++engine_slot) {
      rings_.push_back(std::make_unique<FakeRing>(options.ring_options));
      RequireOrAbort(abce_copy_orchestrator_register_engine(
          orchestrator_.get(), engine_slot, rings_.back()->ring(), nullptr));
    }
  }

  const IsaCase& isa_case() const { return isa_case_; }
  const abce_builder_t& builder() const { return builder_; }
  abce_copy_orchestrator_t* orchestrator() { return orchestrator_.get(); }
  FakeRing& ring(uint32_t engine_slot) { return *rings_.at(engine_slot); }
  SignalWords& signal() { return signal_; }

  // Metadata with the completion signal wired. Pass false to leave out the
  // coordination word, as a client that forgot it would.
  abce_copy_metadata_t NewMetadata(bool with_coordination_scratch = true) {
    abce_copy_metadata_t metadata;
    abce_copy_metadata_initialize(&metadata);
    metadata.out = abce_signal_ref_make(&signal_.value, /*completion_value=*/0, nullptr, 0,
                                        with_coordination_scratch ? &signal_.coordination : nullptr);
    return metadata;
  }

  abce_status_t Map(const std::vector<abce_copy_op_t>& ops, const abce_copy_metadata_t& metadata,
                    abce_plan_t* plan) {
    return abce_copy_orchestrator_map_copy(orchestrator_.get(), ops.data(),
                                           static_cast<uint32_t>(ops.size()), &metadata, plan);
  }

  abce_status_t Submit(abce_plan_t* plan) {
    return abce_copy_orchestrator_submit(orchestrator_.get(), plan);
  }

  // The packets one frame carries, copied out of the per-thread scratch.
  static std::vector<uint8_t> FrameBytes(const abce_plan_t& plan, uint32_t frame_idx) {
    const abce_plan_frame_t& frame = plan.frames[frame_idx];
    return std::vector<uint8_t>(frame.packets, frame.packets + frame.bytes);
  }

  WalkResult WalkFrame(const abce_plan_t& plan, uint32_t frame_idx) const {
    const std::vector<uint8_t> bytes = FrameBytes(plan, frame_idx);
    return Walk(bytes.data(), bytes.size(), isa_case_.isa);
  }

 private:
  // gtest assertions cannot be used in a constructor that returns a value, and
  // a failed setup would make every later result meaningless, so abort.
  static void RequireOrAbort(bool condition) {
    if (!condition) std::abort();
  }

  IsaCase isa_case_;
  abce_builder_t builder_;
  std::unique_ptr<abce_copy_orchestrator_t> orchestrator_;
  std::vector<std::unique_ptr<FakeRing>> rings_;
  SignalWords signal_;
};

}  // namespace abce_test

#endif  // ABCE_TEST_HARNESS_HPP_
