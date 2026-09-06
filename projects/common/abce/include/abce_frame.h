/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE)

#ifndef ABCE_FRAME_H_
#define ABCE_FRAME_H_

#include <cstddef>
#include <cstdint>

#include "abce_builder.h"

namespace abce {

/// Start gate for a gated fan-out, held in bit 62 of the coordination scratch
/// word (see SignalRef::coordination_scratch) whose low 32 bits are the fan-in
/// counter. The gate is deliberately not in the signal's value: a gate bit there
/// would make the signal transiently hold something that is not a legal signal
/// state, visible to any client or tool that reads it mid-flight. Bit 62 rather
/// than 63 so the word stays positive when read as a signed 64-bit value, which
/// keeps it printable by the same tools that dump signal values.
inline constexpr uint64_t kFanOutStartGate = uint64_t{1} << 62;

/// @brief Reference to a signal's device-visible state.
///
/// A thin adapter that carries the fields ABCE actually touches on a signal:
/// the value location, the completion value, the optional interrupt mailbox, and
/// the scratch word used to coordinate a fan-out.
struct SignalRef {
  void* value = nullptr;          ///< 64-bit signal value (device-visible).
  /// Value after the final hardware completion update. FENCE writes it directly;
  /// ATOMIC decrements from completion_value + 1, so a client using the atomic
  /// path arms the signal to that pending value before submitting.
  uint64_t completion_value = 0;
  void* event_mailbox = nullptr;  ///< non-null => emit interrupt after completion.
  uint32_t event_id = 0;
  /// Device-visible 64-bit scratch word, private to ABCE, holding the fan-in
  /// counter (low 32 bits) and the start gate (bit 62) of a gated fan-out. The
  /// SDMA engines poll it and atomically update it, so it must live in memory
  /// they can reach. Required only for a fan-out whose coordinator drains
  /// fan-in; single-engine plans and fan-outs that complete through their own
  /// body signals never touch it. HsaSignalRef() points it at the signal
  /// object's own reserved words, so it costs no separate allocation. The
  /// protocol drains the word back to zero (the coordinator subtracts the gate,
  /// each frame decrements the count), so a completed plan leaves no residue.
  void* coordination_scratch = nullptr;

  SignalRef() = default;
  SignalRef(void* value, uint64_t completion_value, void* event_mailbox = nullptr,
            uint32_t event_id = 0, void* coordination_scratch = nullptr)
      : value(value),
        completion_value(completion_value),
        event_mailbox(event_mailbox),
        event_id(event_id),
        coordination_scratch(coordination_scratch) {}
};

/// @brief Optional profiling timestamp surface holding two device-visible
/// 64-bit SDMA global-clock slots. The coordinator frame writes start
/// in its prologue (before the copy body) and end in its epilogue (after the body,
/// before completion is signalled, so a reader that observes completion also sees a
/// valid end). Null slots disable profiling; it is all-or-nothing. Passed on
/// CopyMetadata the same way as the completion SignalRef.
struct TimestampSurface {
  void* start_value = nullptr;
  void* end_value = nullptr;

  bool enabled() const { return start_value != nullptr && end_value != nullptr; }
  void* start() const { return start_value; }
  void* end() const { return end_value; }
};

/// @brief A dependency signal a frame must wait on before proceeding.
struct DepSignal {
  void* value = nullptr;  ///< 64-bit signal value location (device-visible).
  /// Value observed at record time. Used to omit an already-satisfied wait and,
  /// on non-gfx125plus, determine whether the high 32 bits also need a poll.
  uint64_t observed_value = 0;
  uint64_t reference_value = 0;  ///< value the signal must reach before proceeding.

  DepSignal() = default;
  DepSignal(void* value, uint64_t observed, uint64_t reference = 0)
      : value(value), observed_value(observed), reference_value(reference) {}
};

/// @brief Copy primitive.  Clients typically use kLinear / kMulticast / kFill;
/// kBroadcast is mainly produced internally (small 2-dst copies on non-gfx125plus)
/// but may be requested directly for an explicit 2-destination broadcast.
///   * kSwap     — bidirectional exchange of two buffers (dst<->src); fusible on
///                 gfx125plus, plain packet elsewhere.  Endpoint sizes may differ
///                 (size / size2); a distinct size2 is honored only on the fused path.
///   * kIndirect — gfx125plus-only gather/scatter copy whose src and/or dst address
///                 is read indirectly; always emitted via the wait/signal packet.
///   * kCopyRect — 2D/3D strided sub-window copy (PitchedPtr + offsets + range);
///                 not fusible (no fused rect packet).
enum class OpKind : uint8_t { kLinear, kMulticast, kBroadcast, kSwap, kIndirect, kCopyRect, kFill };

/// @brief Geometry for a kCopyRect op (caller-owned; borrowed by CopyOp/EngineOp
/// like the multicast `dsts` array). Mirrors ABCE::BuildCopyRectCommand.
struct CopyRectDesc {
  PitchedPtr dst{};
  Dim3 dst_offset{};
  PitchedPtr src{};
  Dim3 src_offset{};
  Dim3 range{};
};

union CoherencyControl {
  struct Flags {
    uint32_t emit_hdp_flush : 1;
    uint32_t emit_gcr : 1;
    uint32_t reserved : 30;
  } flags;
  uint32_t value;

  constexpr CoherencyControl() : flags{1, 1, 0} {}
};

/// @brief A whole batch's submission parameters (one completion signal).
struct CopyMetadata {
  const DepSignal* deps = nullptr;  ///< dependencies gating the whole batch.
  uint32_t num_deps = 0;

  SignalRef out{};  ///< the single completion signal (+ optional interrupt).

  /// Per-batch coherency packet triggers. A trigger is additionally gated by
  /// the corresponding platform capability / driver-ownership policy.
  CoherencyControl coherency{};

  /// Restrict the batch to this set of registered engines (bit i => engine i).
  /// 0 selects all registered engines.
  uint64_t engine_mask = 0;

  /// Cap the number of engines used (0 = no cap).  Applied after the mask.
  uint32_t max_engines = 0;

  /// ROCr-compatible multi-linear strategy. Automatic keeps batches whose
  /// entries are all at or below 256 KiB/copy back-to-back on one engine; larger
  /// batches use normal fan-out.
  LinearBatchMode linear_batch_mode = LinearBatchMode::kAutomatic;

  /// gfx125plus multicast selection. Automatic uses multicast through 256 KiB
  /// and fans out larger operations across engines.
  MulticastMode multicast_mode = MulticastMode::kAutomatic;

  /// Optional profiling timestamps.  When set, the coordinator frame records the
  /// global clock into @c timestamps.start in its prologue and @c timestamps.end
  /// in its epilogue (before the completion signal), just like a fused/single copy
  /// still carries a prologue + epilogue for this.  Passed like @c out.
  TimestampSurface timestamps{};

  /// Optional 32-bit slot for an ExecutionDescriptor (see abce_host.h): which
  /// hardware engines ran the batch, what kind of transfer it was, and the
  /// coordinator's engine class. Submit stamps it with a plain host store once
  /// the engines are final; hardware never touches it, so unlike @c timestamps
  /// it costs no packets and needs only 4-byte alignment. Null disables it.
  ///
  /// Held as an address rather than a field so ABCE commits to no particular
  /// signal layout: point it at whatever a reader of this signal will look at.
  void* execution_descriptor = nullptr;

  /// Prefer fused wait/signal copy packets when the device supports them. This
  /// enables direct completion for single-frame work and integrated fan-out
  /// synchronization for multi-frame work. Ignored on non-fused hardware.
  bool prefer_fused = true;
};

/// @brief One decomposed copy placed on a single engine.  Produced by the
/// orchestrator's MapCopy (lives on the stack there, never exposed in the public
/// Plan) and consumed by FrameComposer to size and emit packets.
struct EngineOp {
  EngineOp()
      : indirect_src(false),
        indirect_dst(false),
        has_engine_preference(false),
        fused(false) {}

  const void* src = nullptr;
  void* dst = nullptr;
  void* const* dsts = nullptr;
  const CopyRectDesc* rect = nullptr;
  size_t size = 0;
  size_t size2 = 0;  ///< secondary size (kSwap endpoint B); 0 => same as size.
  uint32_t engine = 0;
  uint32_t num_dsts = 0;
  uint32_t fill_value = 0;
  OpKind kind = OpKind::kLinear;
  uint8_t num_ranked_engines = 0;
  bool indirect_src : 1;
  bool indirect_dst : 1;
  bool has_engine_preference : 1;
  bool fused : 1;
  uint32_t ranked_engines[kMaxEngineChoices]{};
};

/// @brief Everything FrameComposer needs to size / emit one frame.
///
/// A frame is the group of packets written to a single engine's ring in one
/// reservation.  @c ops points at the whole batch's decomposed ops; the composer
/// filters to those matching @c engine (there are @c frame_num_ops of them).  The
/// coordinator frame (@c coordinator == true) additionally carries the prologue +
/// epilogue. For a fan-out (@c multi == true) the frames coordinate through
/// @c coordination_word: when ordering work requires it, non-coordinator frames
/// wait for the start gate (bit 62) to clear before their first body. Every
/// participating frame without a start gate, or every non-coordinator frame with
/// one, decrements the word after its last body, and the coordinator waits for
/// that count to drain.
struct FrameJob {
  const EngineOp* ops = nullptr;  ///< all ops in the batch (filtered by @c engine).
  uint32_t num_ops = 0;           ///< total ops in @c ops.
  const CopyMetadata* metadata = nullptr;

  uint32_t engine = 0;         ///< this frame's engine / ring index.
  uint32_t frame_num_ops = 0;  ///< ops on THIS frame (for the last-body signal).
  bool coordinator = false;    ///< carries the prologue + epilogue.
  bool multi = false;          ///< batch fans out across engines.
  /// Whether non-coordinator frames must wait for the coordinator to release
  /// the bit-62 start gate. gfx125+ fused fan-out without dependencies can
  /// start directly and has every frame signal the fan-in count.
  bool start_gate_required = false;
  /// Whether the coordinator must drain fan-in and perform completion work
  /// after its body. A dependency-free gfx125+ fused fan-out can instead let
  /// the final body SIGNAL reach the completion value directly.
  bool epilogue_required = true;

  /// The 64-bit word this plan's start gate and fan-in counter live in.
  ///
  /// It is the private coordination scratch (SignalRef::coordination_scratch)
  /// whenever the coordinator drains fan-in and then writes completion itself.
  /// It is the completion signal's own value in the two cases where the body
  /// signals *are* the completion transitions and nothing drains them: a
  /// single-frame direct fused completion, and a fan-out with no epilogue. Those
  /// cases only ever decrement toward the completion value, so they need no
  /// scratch and never leave the signal in an illegal state.
  void* coordination_word = nullptr;
};

/// @brief Composes SDMA builder packets into per-engine frames.
///
/// Stateless apart from the builder + a few device capability flags.  Owns the
/// size/emit pairs (PrologueBytes/EmitPrologue, BodyBytes/EmitBody, ...) so the
/// byte count and the packet stream can never drift apart. Configure() supplies
/// platform capabilities; packet-layout capabilities come from the builder.
class FrameComposer {
 public:
  explicit FrameComposer(const ABCE& builder)
      : builder_(builder), is_gfx125plus_(builder.IsGfx125plus()) {}

  void Configure(PlatformCaps caps) { caps_ = caps; }

  // ---- Sizing ----

  size_t FrameBytes(const FrameJob& job) const {
    const CopyMetadata& metadata = *job.metadata;
    size_t bytes = 0;
    if (job.coordinator)
      bytes += PrologueBytes(metadata, job.multi, job.start_gate_required);
    uint32_t body_idx = 0;
    for (uint32_t op_idx = 0; op_idx < job.num_ops; ++op_idx) {
      if (job.ops[op_idx].engine != job.engine) continue;
      const BodySync sync = BodySyncFor(job, body_idx);
      bytes += BodyBytes(job.ops[op_idx], sync.wait, sync.signal);
      body_idx++;
    }
    if (job.coordinator && job.epilogue_required)
      bytes += EpilogueBytes(metadata, job.multi);
    return bytes;
  }

  // ---- Emission ----

  size_t EmitFrame(char* buffer, const FrameJob& job) const {
    const CopyMetadata& metadata = *job.metadata;
    char* cursor = buffer;

    if (job.coordinator)
      cursor += EmitPrologue(cursor, metadata, job.multi, job.start_gate_required,
                             job.coordination_word);

    uint32_t body_idx = 0;
    for (uint32_t op_idx = 0; op_idx < job.num_ops; ++op_idx) {
      if (job.ops[op_idx].engine != job.engine) continue;
      cursor += EmitBody(cursor, job.ops[op_idx], BodySyncFor(job, body_idx));
      body_idx++;
    }

    if (job.coordinator && job.epilogue_required)
      cursor += EmitEpilogue(cursor, metadata, job.multi, job.coordination_word);
    return static_cast<size_t>(cursor - buffer);
  }

 private:
  /// Whether the body at @p body_idx of @p job carries the start-gate wait and
  /// the completion / fan-in signal, and which word each one targets. Derived
  /// once here so the sizing pass and the emit pass can never disagree about
  /// which bodies are synchronizing.
  struct BodySync {
    bool wait = false;
    bool signal = false;
    void* wait_addr = nullptr;
    void* signal_addr = nullptr;
  };

  static BodySync BodySyncFor(const FrameJob& job, uint32_t body_idx) {
    const bool wait_for_start = job.multi && job.start_gate_required && !job.coordinator;
    const bool signals_completion_directly = !job.multi && !job.epilogue_required;
    const bool contributes_to_fan_in =
        job.multi && (!job.coordinator || !job.start_gate_required);
    BodySync sync;
    sync.wait = wait_for_start && (body_idx == 0);
    sync.signal = (signals_completion_directly || contributes_to_fan_in) &&
                  (body_idx == job.frame_num_ops - 1);
    // Both roles read the same word for a given plan: FrameJob::coordination_word
    // already resolves to the scratch or to the signal value depending on whether
    // anything drains the count.
    if (sync.wait) sync.wait_addr = job.coordination_word;
    if (sync.signal) sync.signal_addr = job.coordination_word;
    return sync;
  }

  // ---- Per-command sizes (bytes) ----
  static constexpr uint32_t kPoll32 = sizeof(SDMA_PKT_POLL_REGMEM);
  static constexpr uint32_t kPoll64 = sizeof(SDMA_PKT_POLL_MEM_64B_GFX125PLUS);
  static constexpr uint32_t kFence32 = sizeof(SDMA_PKT_FENCE);
  static constexpr uint32_t kFence64 = sizeof(SDMA_PKT_FENCE_64B_GFX125PLUS);
  static constexpr uint32_t kAtomic = sizeof(SDMA_PKT_ATOMIC);
  static constexpr uint32_t kTimestamp = sizeof(SDMA_PKT_TIMESTAMP);
  static constexpr uint32_t kTrap = sizeof(SDMA_PKT_TRAP);
  static constexpr uint32_t kFlush = sizeof(SDMA_PKT_HDP_FLUSH);
  static constexpr uint32_t kCopyLinear = sizeof(SDMA_PKT_COPY_LINEAR);
  static constexpr uint32_t kFill = sizeof(SDMA_PKT_CONSTANT_FILL);

  uint32_t GcrBytes() const {
    if (!ShouldEmitGcr()) return 0;
    return is_gfx125plus_ ? sizeof(SDMA_PKT_GCR_GFX125PLUS) : sizeof(SDMA_PKT_GCR);
  }
  bool ShouldEmitGcr() const { return builder_.RequiresGcr() && !caps_.driver_manages_gcr; }
  uint32_t PollBytes() const { return is_gfx125plus_ ? kPoll64 : kPoll32; }

  static uint32_t MulticastPktBytes(uint32_t num_dsts) {
    return (5u + 2u * num_dsts) * sizeof(uint32_t);
  }

  /// Effective secondary size for a two-region op: the caller's @c size2 when
  /// available, otherwise @c size (symmetric).  Only kSwap's fused wait/signal
  /// path consumes a distinct value; the single-COUNT plain swap packet is
  /// symmetric-only.
  static size_t Size2(const EngineOp& op) { return op.size2 ? op.size2 : op.size; }

  size_t PrologueBytes(const CopyMetadata& metadata, bool multi,
                       bool start_gate_required) const {
    size_t bytes = DepWaitsBytes(metadata.deps, metadata.num_deps);
    if (metadata.timestamps.enabled()) bytes += kTimestamp;
    if (caps_.emit_hdp_flush && metadata.coherency.flags.emit_hdp_flush) bytes += kFlush;
    if (metadata.coherency.flags.emit_gcr) bytes += GcrBytes();
    if (multi && start_gate_required) bytes += kAtomic;
    return bytes;
  }

  size_t BodyBytes(const EngineOp& engine_op, bool wait_here, bool signal_here) const {
    if (engine_op.fused) return FusedBytes(engine_op, wait_here, signal_here);
    size_t bytes = 0;
    if (wait_here) bytes += PollBytes();
    bytes += PayloadBytes(engine_op);
    if (signal_here) bytes += kAtomic;
    return bytes;
  }

  size_t EpilogueBytes(const CopyMetadata& metadata, bool multi) const {
    size_t bytes = 0;
    if (multi) bytes += PollBytes();
    if (metadata.coherency.flags.emit_gcr) bytes += GcrBytes();
    if (metadata.timestamps.enabled()) bytes += kTimestamp;
    bytes += CompletionBytes(metadata.out);
    if (metadata.out.event_mailbox) bytes += kFence32 + kTrap;
    return bytes;
  }

  size_t PayloadBytes(const EngineOp& engine_op) const {
    switch (engine_op.kind) {
      case OpKind::kLinear:
        return static_cast<size_t>(builder_.NumCopyPackets(engine_op.size)) * kCopyLinear;
      case OpKind::kMulticast:
        return static_cast<size_t>(builder_.NumMulticastPackets(engine_op.size)) *
               MulticastPktBytes(engine_op.num_dsts);
      case OpKind::kBroadcast:
        return static_cast<size_t>(builder_.NumBroadcastPackets(engine_op.size)) *
               sizeof(SDMA_PKT_COPY_LINEAR_BROADCAST);
      case OpKind::kSwap:
        return static_cast<size_t>(builder_.NumSwapPackets(engine_op.size)) *
               sizeof(SDMA_PKT_COPY_LINEAR_SWAP);
      case OpKind::kCopyRect:
        return static_cast<size_t>(builder_.NumRectPackets(
                   &engine_op.rect->dst, &engine_op.rect->dst_offset, &engine_op.rect->src,
                   &engine_op.rect->src_offset, &engine_op.rect->range)) *
               sizeof(SDMA_PKT_COPY_LINEAR_RECT);
      case OpKind::kIndirect:
        return sizeof(SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS);
      case OpKind::kFill:
        return static_cast<size_t>(builder_.NumFillPackets(engine_op.size / sizeof(uint32_t))) *
               kFill;
    }
    return 0;
  }

  size_t FusedBytes(const EngineOp& engine_op, bool wait, bool signal) const {
    const uint32_t wait_dwords = wait ? 7u : 0u;
    const uint32_t signal_dwords = signal ? 5u : 0u;
    if (engine_op.kind == OpKind::kIndirect)
      return (1u + wait_dwords + 6u + signal_dwords) * sizeof(uint32_t);

    if (engine_op.kind == OpKind::kLinear || engine_op.kind == OpKind::kSwap) {
      uint32_t num_packets = 1u;
      if (engine_op.kind == OpKind::kSwap)
        num_packets = builder_.NumWaitSignalSwapPackets(engine_op.size, Size2(engine_op));
      else if (engine_op.kind == OpKind::kLinear)
        num_packets = builder_.NumCopyPackets(engine_op.size);
      const uint32_t total_dwords =
          num_packets * (1u + 6u) + wait_dwords + signal_dwords;
      return static_cast<size_t>(total_dwords) * sizeof(uint32_t);
    }
    if (engine_op.kind == OpKind::kMulticast) {
      const uint32_t packet_core_dwords = 1u + 4u + 2u * engine_op.num_dsts;
      const uint32_t total_dwords =
          builder_.NumMulticastPackets(engine_op.size) * packet_core_dwords +
          wait_dwords + signal_dwords;
      return static_cast<size_t>(total_dwords) * sizeof(uint32_t);
    }
    return 0;
  }

  size_t DepWaitsBytes(const DepSignal* deps, uint32_t num_deps) const {
    size_t bytes = 0;
    for (uint32_t i = 0; i < num_deps; ++i) {
      if (deps[i].observed_value == deps[i].reference_value) continue;
      bytes += is_gfx125plus_ ? kPoll64 : kPoll32;
      if (!is_gfx125plus_ &&
          (deps[i].observed_value >> 32) != (deps[i].reference_value >> 32))
        bytes += kPoll32;
    }
    return bytes;
  }

  uint32_t CompletionBytes(const SignalRef& target) const {
    if (caps_.device_atomic_support) return kAtomic;
    if (is_gfx125plus_) return kFence64;
    return (target.completion_value > UINT32_MAX) ? 2 * kFence32 : kFence32;
  }

  size_t EmitPrologue(char* out, const CopyMetadata& metadata, bool multi,
                      bool start_gate_required, void* coordination_word) const {
    char* cursor = out;
    cursor += EmitDepWaits(cursor, metadata.deps, metadata.num_deps);
    if (metadata.timestamps.enabled()) {
      builder_.BuildGetGlobalTimestampCommand(cursor, metadata.timestamps.start());
      cursor += kTimestamp;
    }
    if (caps_.emit_hdp_flush && metadata.coherency.flags.emit_hdp_flush) {
      builder_.BuildHdpFlushCommand(cursor);
      cursor += kFlush;
    }
    if (metadata.coherency.flags.emit_gcr && ShouldEmitGcr()) {
      builder_.BuildGCRCommand(cursor, /*invalidate=*/true);
      cursor += GcrBytes();
    }
    if (multi && start_gate_required) {
      builder_.BuildAtomicAddCommand(cursor, coordination_word, uint64_t{0} - kFanOutStartGate);
      cursor += kAtomic;
    }
    return static_cast<size_t>(cursor - out);
  }

  size_t EmitBody(char* out, const EngineOp& engine_op, const BodySync& sync) const {
    char* cursor = out;
    void* wait_addr = sync.wait_addr;
    void* signal_addr = sync.signal_addr;

    if (engine_op.fused) {
      if (engine_op.kind == OpKind::kIndirect) {
        builder_.BuildWaitSignalIndirectCopyCommand(cursor, engine_op.dst, engine_op.src,
                                                    engine_op.size, engine_op.indirect_src,
                                                    engine_op.indirect_dst, wait_addr, signal_addr,
                                                    /*wait_reference=*/0, kFanOutStartGate);
      } else if (engine_op.kind == OpKind::kLinear) {
        builder_.BuildWaitSignalCopyCommand(cursor, engine_op.dst, engine_op.src, engine_op.size,
                                            wait_addr, signal_addr, /*wait_reference=*/0,
                                            kFanOutStartGate,
                                            /*boundary_wait_signal=*/true);
      } else if (engine_op.kind == OpKind::kSwap) {
        builder_.BuildWaitSignalSwapCommand(cursor, engine_op.dst, const_cast<void*>(engine_op.src),
                                            engine_op.size, Size2(engine_op), wait_addr,
                                            signal_addr, /*wait_reference=*/0,
                                            kFanOutStartGate,
                                            /*boundary_wait_signal=*/true);
      } else if (engine_op.kind == OpKind::kMulticast) {
        builder_.BuildMulticastWaitSignalCopyCommand(cursor, engine_op.dsts, engine_op.num_dsts,
                                                     engine_op.src, engine_op.size, wait_addr,
                                                     signal_addr, /*wait_reference=*/0,
                                                     kFanOutStartGate,
                                                     /*boundary_wait_signal=*/true);
      }
      return FusedBytes(engine_op, sync.wait, sync.signal);
    }

    if (sync.wait) cursor += WriteStartGatePoll(cursor, wait_addr);
    switch (engine_op.kind) {
      case OpKind::kLinear:
        builder_.BuildCopyCommand(cursor, engine_op.dst, engine_op.src, engine_op.size);
        break;
      case OpKind::kMulticast:
        builder_.BuildMulticastCopyCommand(cursor, engine_op.dsts, engine_op.num_dsts,
                                           engine_op.src, engine_op.size);
        break;
      case OpKind::kBroadcast:
        builder_.BuildBroadcastCopyCommand(cursor, engine_op.dsts[0], engine_op.dsts[1],
                                           engine_op.src, engine_op.size);
        break;
      case OpKind::kSwap:
        builder_.BuildSwapCopyCommand(cursor, engine_op.dst, const_cast<void*>(engine_op.src),
                                      engine_op.size);
        break;
      case OpKind::kCopyRect: {
        char* rect_cursor = cursor;
        builder_.BuildCopyRectCommand(
            [&rect_cursor](size_t pkt_bytes) -> void* {
              char* slot = rect_cursor;
              rect_cursor += pkt_bytes;
              return slot;
            },
            &engine_op.rect->dst, &engine_op.rect->dst_offset, &engine_op.rect->src,
            &engine_op.rect->src_offset, &engine_op.rect->range);
        break;
      }
      case OpKind::kIndirect:
        break;
      case OpKind::kFill:
        builder_.BuildFillCommand(cursor, engine_op.dst, engine_op.fill_value,
                                  engine_op.size / sizeof(uint32_t));
        break;
    }
    cursor += PayloadBytes(engine_op);
    if (sync.signal) {
      builder_.BuildAtomicDecrementCommand(cursor, signal_addr);
      cursor += kAtomic;
    }
    return static_cast<size_t>(cursor - out);
  }

  size_t EmitEpilogue(char* out, const CopyMetadata& metadata, bool multi,
                      void* coordination_word) const {
    char* cursor = out;
    if (multi) cursor += WriteFanInPoll(cursor, coordination_word);
    if (metadata.coherency.flags.emit_gcr && ShouldEmitGcr()) {
      builder_.BuildGCRCommand(cursor, /*invalidate=*/false);
      cursor += GcrBytes();
    }
    if (metadata.timestamps.enabled()) {
      builder_.BuildGetGlobalTimestampCommand(cursor, metadata.timestamps.end());
      cursor += kTimestamp;
    }
    cursor += WriteCompletion(cursor, metadata.out);
    cursor += WriteMailbox(cursor, metadata.out);
    return static_cast<size_t>(cursor - out);
  }

  size_t EmitDepWaits(char* out, const DepSignal* deps, uint32_t num_deps) const {
    char* cursor = out;
    for (uint32_t i = 0; i < num_deps; ++i) {
      if (deps[i].observed_value == deps[i].reference_value) continue;
      if (is_gfx125plus_) {
        builder_.BuildPoll64bCommand(cursor, deps[i].value, deps[i].reference_value);
        cursor += kPoll64;
      } else {
        uint32_t* words = reinterpret_cast<uint32_t*>(deps[i].value);
        if ((deps[i].observed_value >> 32) != (deps[i].reference_value >> 32)) {
          builder_.BuildPollCommand(cursor, &words[1],
                                    static_cast<uint32_t>(deps[i].reference_value >> 32));
          cursor += kPoll32;
        }
        builder_.BuildPollCommand(cursor, &words[0],
                                  static_cast<uint32_t>(deps[i].reference_value));
        cursor += kPoll32;
      }
    }
    return static_cast<size_t>(cursor - out);
  }

  size_t WriteStartGatePoll(char* out, void* coordination_word) const {
    if (is_gfx125plus_) {
      builder_.BuildPoll64bCommand(out, coordination_word, /*reference=*/0, kFanOutStartGate);
      return kPoll64;
    }
    // Without a 64-bit poll, watch bit 30 of the high word — the same bit 62.
    uint32_t* coordination_words = reinterpret_cast<uint32_t*>(coordination_word);
    constexpr uint32_t kStartGateHighWord = uint32_t{1} << 30;
    builder_.BuildPollCommand(out, &coordination_words[1], /*reference=*/0, kStartGateHighWord);
    return kPoll32;
  }

  /// Wait for every participating frame to have decremented the fan-in counter.
  /// The counter is the low 32 bits of a word ABCE owns outright, so it drains to
  /// zero rather than to a value biased by the caller's completion value.
  size_t WriteFanInPoll(char* out, void* coordination_word) const {
    if (is_gfx125plus_) {
      builder_.BuildPoll64bCommand(out, coordination_word, /*reference=*/0, UINT32_MAX);
      return kPoll64;
    }
    builder_.BuildPollCommand(out, coordination_word, /*reference=*/0);
    return kPoll32;
  }

  size_t WriteCompletion(char* out, const SignalRef& target) const {
    if (caps_.device_atomic_support) {
      builder_.BuildAtomicDecrementCommand(out, target.value);
      return kAtomic;
    }
    if (is_gfx125plus_) {
      builder_.BuildFence64bCommand(out, target.value, target.completion_value);
      return kFence64;
    }
    char* cursor = out;
    uint32_t* words = reinterpret_cast<uint32_t*>(target.value);
    if (target.completion_value > UINT32_MAX) {
      builder_.BuildFenceCommand(cursor, &words[1],
                                 static_cast<uint32_t>(target.completion_value >> 32));
      cursor += kFence32;
    }
    builder_.BuildFenceCommand(cursor, &words[0], static_cast<uint32_t>(target.completion_value));
    cursor += kFence32;
    return static_cast<size_t>(cursor - out);
  }

  size_t WriteMailbox(char* out, const SignalRef& target) const {
    if (!target.event_mailbox) return 0;
    char* cursor = out;
    builder_.BuildFenceCommand(cursor, reinterpret_cast<uint32_t*>(target.event_mailbox),
                               target.event_id);
    cursor += kFence32;
    builder_.BuildTrapCommand(cursor, target.event_id);
    cursor += kTrap;
    return static_cast<size_t>(cursor - out);
  }

  const ABCE& builder_;
  PlatformCaps caps_{};
  bool is_gfx125plus_ = false;
};

}  // namespace abce

#endif  // ABCE_FRAME_H_
