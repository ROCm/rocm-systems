/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE)

#ifndef ABCE_BUILDER_H_
#define ABCE_BUILDER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "abce_types.h"
#include "sdma_packets.h"

// Host+device qualifier: builders are usable from a kernel (device-initiated
// SDMA) as well as the host. Guarded so any ABCE header may define it first.
#ifndef ABCE_HD
#if defined(__HIPCC__) || defined(__CUDACC__)
#define ABCE_HD __host__ __device__
#else
#define ABCE_HD
#endif
#endif  // ABCE_HD

// ABCE_ASSERT() pulls a host-only symbol; make it a no-op during device compilation.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define ABCE_ASSERT(x) ((void)0)
#else
#define ABCE_ASSERT(x) assert(x)
#endif

namespace abce {

ABCE_HD inline uint32_t ptrlow32(const void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

ABCE_HD inline uint32_t ptrhigh32(const void* p) {
  return static_cast<uint32_t>(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)) >> 32);
}

namespace detail {

template <typename T> ABCE_HD static inline T Min(const T& a, const T& b) {
  return (a < b) ? a : b;
}

template <typename T, typename... Arg>
ABCE_HD static inline T Min(const T& a, const T& b, Arg... args) {
  return Min(a, Min(b, args...));
}

template <typename T> ABCE_HD static inline T Max(const T& a, const T& b) {
  return (a > b) ? a : b;
}

}  // namespace detail

/// @brief SDMA (copy engine) packet writers for one device/queue.
///
/// Each builder writes packets into a caller-provided command buffer.  The
/// number of packets is computed internally from the transfer size and the
/// configured per-packet limit — callers never pass a packet count.  Public
/// NumXxxPackets() helpers let callers query the count for buffer sizing or
/// signal pre-arming.
///
/// ============================ ZEROED-BUFFER CONTRACT ========================
/// The fixed-layout builders (copy, broadcast, multicast, swap, fill, poll,
/// fence, atomic, timestamp, trap, gcr, rect) DO NOT zero the destination
/// buffer.  They set only the meaningful bitfields and rely on every other bit
/// of each packet DWORD already being 0.  The caller MUST provide a
/// zero-initialized buffer (e.g. the ring's reservation memset, or a NOP-padded
/// region where 0 == NOP).  Writing a builder over non-zeroed memory produces
/// malformed packets.
///
/// The fused wait/signal builders are the exception: they explicitly emit every
/// DWORD they write (compacted, variable length), so their output does not
/// depend on the zeroed-buffer contract.  They use small stack scratch structs
/// internally, which they zero themselves.
/// ===========================================================================
///
/// Packet format, GCR use, scope fields, and the default linear-copy limit are
/// derived from the gfx IP using the OSS4/OSS5/OSS7 capability table.
class ABCE {
 public:
  explicit ABCE(IsaVersion isa, BuilderConfig config = {})
      : isa_(isa),
        packet_caps_(DetectPacketCaps(isa)),
        scope_fields_(packet_caps_.scope_fields),
        is_gfx125plus_(packet_caps_.gfx125plus),
        max_copy_size_(config.max_linear_copy_size
                           ? config.max_linear_copy_size
                           : (config.use_copy_size_override ? packet_caps_.max_linear_copy_size
                                                            : SDMA_PKT_COPY_LINEAR::kMaxSize_)),
        use_extended_count_(config.max_linear_copy_size != 0 || config.use_copy_size_override),
        max_fill_size_(config.max_fill_size ? config.max_fill_size
                                            : SDMA_PKT_CONSTANT_FILL::kMaxSize_) {}

  // ---- Device/config queries ----

  ABCE_HD const PacketCaps& PacketCapabilities() const { return packet_caps_; }
  const IsaVersion& Isa() const { return isa_; }
  ABCE_HD bool IsGfx125plus() const { return is_gfx125plus_; }
  ABCE_HD size_t MaxCopySize() const { return max_copy_size_; }
  ABCE_HD bool RequiresGcr() const { return packet_caps_.use_gcr; }

  // ---- Packet-count queries (for buffer sizing / signal pre-arming) ----

  ABCE_HD uint32_t NumCopyPackets(size_t size) const {
    return static_cast<uint32_t>((size + max_copy_size_ - 1) / max_copy_size_);
  }

  ABCE_HD uint32_t NumBroadcastPackets(size_t size) const { return NumCopyPackets(size); }
  ABCE_HD uint32_t NumMulticastPackets(size_t size) const { return NumCopyPackets(size); }

  ABCE_HD uint32_t NumSwapPackets(size_t size) const {
    const size_t max = (scope_fields_ && is_gfx125plus_)
                           ? SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS::kMaxSize_
                           : SDMA_PKT_COPY_LINEAR_SWAP::kMaxSize_;
    return static_cast<uint32_t>((size + max - 1) / max);
  }

  ABCE_HD uint32_t NumWaitSignalSwapPackets(size_t size_a, size_t size_b) const {
    const size_t size_max = detail::Max(size_a, size_b);
    const size_t max = SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS::kMaxSize_;
    return static_cast<uint32_t>((size_max + max - 1) / max);
  }

  ABCE_HD uint32_t NumFillPackets(size_t count) const {
    const size_t size = count * sizeof(uint32_t);
    return static_cast<uint32_t>((size + max_fill_size_ - 1) / max_fill_size_);
  }

  /// Number of rect (2D/3D sub-window) copy packets.  Rect tiling is data
  /// dependent (pitch/alignment driven), so this replays the exact same loop as
  /// BuildCopyRectCommand with a counting append — the two can never diverge.
  /// May throw std::invalid_argument on out-of-range pitch/slice, same as emit.
  uint32_t NumRectPackets(const PitchedPtr* dst, const Dim3* dst_offset, const PitchedPtr* src,
                          const Dim3* src_offset, const Dim3* range) const {
    uint32_t num_packets = 0;
    alignas(16) char scratch[256];  // one throwaway packet's worth; reused per tile.
    BuildCopyRectCommand(
        [&num_packets, &scratch](size_t) -> void* {
          ++num_packets;
          return scratch;
        },
        dst, dst_offset, src, src_offset, range);
    return num_packets;
  }

  // ---- Builders (see ZEROED-BUFFER CONTRACT above) ----

  ABCE_HD void BuildFenceCommand(char* fence_command_addr, uint32_t* fence,
                                 uint32_t fence_value) const {
    ABCE_ASSERT(fence_command_addr != NULL);

    if (isa_.major >= 12) {
      auto* pkt = reinterpret_cast<SDMA_PKT_FENCE_GFX12*>(fence_command_addr);
      pkt->HEADER_UNION.op = SDMA_OP_FENCE;
      pkt->HEADER_UNION.mtype = 3;
      pkt->HEADER_UNION.sys = 1;
      if (scope_fields_) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

      pkt->ADDR_LO_UNION.addr_31_0 = ptrlow32(fence);
      pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence);
      pkt->DATA_UNION.data = fence_value;
    } else {
      auto* pkt = reinterpret_cast<SDMA_PKT_FENCE*>(fence_command_addr);
      pkt->HEADER_UNION.op = SDMA_OP_FENCE;
      if (isa_.major >= 10) pkt->HEADER_UNION.mtype = 3;

      pkt->ADDR_LO_UNION.addr_31_0 = ptrlow32(fence);
      pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence);
      pkt->DATA_UNION.data = fence_value;
    }
  }

  ABCE_HD void BuildCopyCommand(char* cmd_addr, void* dst, const void* src, size_t size) const {
    size_t cur_size = 0;
    while (cur_size < size) {
      const uint32_t copy_size =
          static_cast<uint32_t>(detail::Min(size - cur_size, max_copy_size_));

      void* cur_dst = static_cast<char*>(dst) + cur_size;
      const void* cur_src = static_cast<const char*>(src) + cur_size;

      auto* pkt = reinterpret_cast<SDMA_PKT_COPY_LINEAR*>(cmd_addr);
      pkt->HEADER_UNION.op = SDMA_OP_COPY;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
      if (scope_fields_) pkt->HEADER_UNION.npd = 1;

      if (use_extended_count_)
        pkt->COUNT_UNION.count_ext.count = copy_size - 1;
      else
        pkt->COUNT_UNION.count.count = copy_size - 1;

      if (scope_fields_) {
        pkt->PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
        pkt->PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;
      }

      pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
      pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);
      pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_dst);
      pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_dst);

      cmd_addr += sizeof(SDMA_PKT_COPY_LINEAR);
      cur_size += copy_size;
    }
  }

  ABCE_HD void BuildBroadcastCopyCommand(char* cmd_addr, void* dst1, void* dst2, const void* src,
                                         size_t size) const {
    [[maybe_unused]] constexpr size_t kMask = SDMA_PKT_COPY_LINEAR_BROADCAST::kDstAlignMask_;
    ABCE_ASSERT((reinterpret_cast<uintptr_t>(dst1) & kMask) ==
                (reinterpret_cast<uintptr_t>(dst2) & kMask));
    size_t cur_size = 0;
    while (cur_size < size) {
      const uint32_t copy_size =
          static_cast<uint32_t>(detail::Min(size - cur_size, max_copy_size_));

      void* cur_dst1 = static_cast<char*>(dst1) + cur_size;
      void* cur_dst2 = static_cast<char*>(dst2) + cur_size;
      const void* cur_src = static_cast<const char*>(src) + cur_size;

      auto* pkt = reinterpret_cast<SDMA_PKT_COPY_LINEAR_BROADCAST*>(cmd_addr);
      pkt->HEADER_UNION.op = SDMA_OP_COPY;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_BROADCAST;
      pkt->HEADER_UNION.broadcast = 1;

      if (use_extended_count_)
        pkt->COUNT_UNION.count_ext.count = copy_size - 1;
      else
        pkt->COUNT_UNION.count.count = copy_size - 1;

      pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
      pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);
      pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_dst1);
      pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_dst1);
      pkt->DST2_ADDR_LO_UNION.dst2_addr_31_0 = ptrlow32(cur_dst2);
      pkt->DST2_ADDR_HI_UNION.dst2_addr_63_32 = ptrhigh32(cur_dst2);

      cmd_addr += sizeof(SDMA_PKT_COPY_LINEAR_BROADCAST);
      cur_size += copy_size;
    }
  }

  ABCE_HD void BuildMulticastCopyCommand(char* cmd_addr, void* const* dsts, uint32_t num_dsts,
                                         const void* src, size_t size) const {
    const size_t pkt_bytes = (5 + 2 * static_cast<size_t>(num_dsts)) * sizeof(uint32_t);
    size_t cur_size = 0;
    while (cur_size < size) {
      const uint32_t copy_size =
          static_cast<uint32_t>(detail::Min(size - cur_size, max_copy_size_));

      auto* pkt = reinterpret_cast<SDMA_PKT_COPY_LINEAR_MULTICAST_GFX125PLUS*>(cmd_addr);
      pkt->HEADER_UNION.op = SDMA_OP_COPY;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;
      pkt->COUNT_UNION.count = copy_size - 1;
      pkt->PARAMETER_UNION.num_of_destination = num_dsts - 1;
      pkt->PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
      pkt->PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;

      const void* cur_src = static_cast<const char*>(src) + cur_size;
      pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
      pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);

      uint32_t* dst_dw = reinterpret_cast<uint32_t*>(cmd_addr) + 5;
      for (uint32_t d = 0; d < num_dsts; ++d) {
        const void* cur_dst = static_cast<const char*>(dsts[d]) + cur_size;
        dst_dw[d * 2] = ptrlow32(cur_dst);
        dst_dw[d * 2 + 1] = ptrhigh32(cur_dst);
      }

      cmd_addr += pkt_bytes;
      cur_size += copy_size;
    }
  }

  // ---- Fused wait/copy/signal builders (gfx125plus, variable-length) --------
  //
  // These packets are COMPACTED: header (1 DW) + optional wait (7 DW) + body
  // (varies per op kind) + optional signal (5 DW), packed contiguously.  The
  // engine reads each block immediately after the preceding one, keyed by the
  // header's wait/signal bits.  Absent blocks are not reserved.
  //
  // Each builder forms the packet in a zeroed scratch struct, then emits only
  // the present DW blocks.  They do not rely on the zeroed-buffer contract.
  //
  // @c boundary_wait_signal puts the WAIT on the first chunk and the SIGNAL on
  // the last, so a chunked transfer consumes exactly one wait and one signal.
  // This assumes the engine retires chunks of one packet stream in order, so the
  // final chunk's SIGNAL cannot outrun an earlier chunk's data.  ROCr's blit
  // path instead waits and signals on every chunk and pre-arms the output by
  // N-1; pass false to get that behavior.  Chunking only occurs above the
  // per-packet copy limit.
  // -------------------------------------------------------------------------

  ABCE_HD void BuildMulticastWaitSignalCopyCommand(char* cmd_addr, void* const* dsts,
                                                   uint32_t num_dsts, const void* src, size_t size,
                                                   void* wait_addr, void* signal_addr,
                                                   uint64_t wait_reference = 0,
                                                   uint64_t wait_mask = UINT64_MAX,
                                                   bool boundary_wait_signal = false) const {
    const bool do_wait = (wait_addr != nullptr);
    const bool do_signal = (signal_addr != nullptr);

    uint32_t wait_dws[7] = {};
    uint32_t signal_dws[5] = {};
    {
      SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS tmpl;
      memset(&tmpl, 0, sizeof(tmpl));
      tmpl.HEADER_UNION.op = SDMA_OP_COPY;
      tmpl.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;

      if (do_wait) {
        tmpl.WAIT_FUNCTION_UNION.wait_function = SDMA_FUNC_EQUAL;
        tmpl.WAIT_FUNCTION_UNION.wait_scope = SDMA_MEMORY_SCOPE_SYS;
        tmpl.WAIT_ADDR_LO_UNION.wait_addr_31_3 = ptrlow32(wait_addr) >> 3;
        tmpl.WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wait_addr);
        tmpl.WAIT_REFERENCE_LO_UNION.wait_reference_31_0 =
            static_cast<uint32_t>(wait_reference);
        tmpl.WAIT_REFERENCE_HI_UNION.wait_reference_63_32 =
            static_cast<uint32_t>(wait_reference >> 32);
        tmpl.WAIT_MASK_LO_UNION.wait_mask_31_0 = static_cast<uint32_t>(wait_mask);
        tmpl.WAIT_MASK_HI_UNION.wait_mask_63_32 = static_cast<uint32_t>(wait_mask >> 32);
        const uint32_t* s = reinterpret_cast<const uint32_t*>(&tmpl);
        memcpy(wait_dws, s + 1, sizeof(wait_dws));
      }
      if (do_signal) {
        tmpl.SIGNAL_OPERATION_UNION.signal_operation = SDMA_SIGNAL_OP_SUB64;
        tmpl.SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
        tmpl.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = ptrlow32(signal_addr) >> 3;
        tmpl.SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = ptrhigh32(signal_addr);
        tmpl.SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
        tmpl.SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;
        const uint32_t* s = reinterpret_cast<const uint32_t*>(&tmpl);
        memcpy(signal_dws, s + 14, sizeof(signal_dws));
      }
    }

    size_t cur_size = 0;
    while (cur_size < size) {
      const uint32_t copy_size =
          static_cast<uint32_t>(detail::Min(size - cur_size, max_copy_size_));
      const bool chunk_wait =
          do_wait && (!boundary_wait_signal || cur_size == 0);
      const bool chunk_signal =
          do_signal && (!boundary_wait_signal || cur_size + copy_size == size);

      uint32_t* out = reinterpret_cast<uint32_t*>(cmd_addr);
      uint32_t n = 0;

      SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS header{};
      header.HEADER_UNION.op = SDMA_OP_COPY;
      header.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;
      header.HEADER_UNION.wait = chunk_wait ? 1 : 0;
      header.HEADER_UNION.signal = chunk_signal ? 1 : 0;
      out[n++] = reinterpret_cast<const uint32_t*>(&header)[0];
      if (chunk_wait) {
        memcpy(out + n, wait_dws, sizeof(wait_dws));
        n += 7;
      }

      // Only DW8-11 are emitted from this scratch; DW8/DW9 are written through
      // bitfields, so clear them first (DW10/DW11 are whole-DWORD addresses).
      SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS chunk;
      chunk.COUNT_UNION.DW_8_DATA = 0;
      chunk.COPY_PARAMETER_UNION.DW_9_DATA = 0;
      chunk.COUNT_UNION.count = copy_size - 1;
      chunk.COPY_PARAMETER_UNION.num_of_destination = num_dsts - 1;
      chunk.COPY_PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
      chunk.COPY_PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;
      const char* cur_src = reinterpret_cast<const char*>(src) + cur_size;
      chunk.SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
      chunk.SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);
      const uint32_t* cs = reinterpret_cast<const uint32_t*>(&chunk);
      out[n++] = cs[8];
      out[n++] = cs[9];
      out[n++] = cs[10];
      out[n++] = cs[11];

      for (uint32_t j = 0; j < num_dsts; ++j) {
        const char* cur_dst = reinterpret_cast<const char*>(dsts[j]) + cur_size;
        out[n++] = ptrlow32(cur_dst);
        out[n++] = ptrhigh32(cur_dst);
      }
      if (chunk_signal) {
        memcpy(out + n, signal_dws, sizeof(signal_dws));
        n += 5;
      }

      cmd_addr += n * sizeof(uint32_t);
      cur_size += copy_size;
    }
  }

  ABCE_HD void BuildSwapCopyCommand(char* cmd_addr, void* addr_a, void* addr_b, size_t size) const {
    static_assert(sizeof(SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS) == sizeof(SDMA_PKT_COPY_LINEAR_SWAP),
                  "gfx125plus swap packet must match legacy swap packet size for shared stride");
    const bool use_gfx125plus = scope_fields_ && is_gfx125plus_;

    const size_t kAlign = use_gfx125plus ? SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS::kAlignment_
                                         : SDMA_PKT_COPY_LINEAR_SWAP::kAlignment_;
    ABCE_ASSERT((reinterpret_cast<uintptr_t>(addr_a) & (kAlign - 1)) == 0);
    ABCE_ASSERT((reinterpret_cast<uintptr_t>(addr_b) & (kAlign - 1)) == 0);

    const size_t max_copy_size = use_gfx125plus ? SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS::kMaxSize_
                                                : SDMA_PKT_COPY_LINEAR_SWAP::kMaxSize_;
    size_t cur_size = 0;
    while (cur_size < size) {
      const uint32_t copy_size = static_cast<uint32_t>(detail::Min(size - cur_size, max_copy_size));

      void* cur_addr_a = static_cast<char*>(addr_a) + cur_size;
      void* cur_addr_b = static_cast<char*>(addr_b) + cur_size;

      if (use_gfx125plus) {
        auto* pkt = reinterpret_cast<SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS*>(cmd_addr);
        pkt->HEADER_UNION.op = SDMA_OP_COPY;
        pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
        pkt->COUNT_UNION.count = copy_size - 1;
        pkt->PARAMETER_UNION.scope_a = SDMA_MEMORY_SCOPE_SYS;
        pkt->PARAMETER_UNION.scope_b = SDMA_MEMORY_SCOPE_SYS;
        pkt->ADDR_A_LO_UNION.DW_3_DATA = ptrlow32(cur_addr_a);
        pkt->ADDR_A_HI_UNION.addr_a_63_32 = ptrhigh32(cur_addr_a);
        pkt->ADDR_B_LO_UNION.DW_5_DATA = ptrlow32(cur_addr_b);
        pkt->ADDR_B_HI_UNION.addr_b_63_32 = ptrhigh32(cur_addr_b);
      } else {
        auto* pkt = reinterpret_cast<SDMA_PKT_COPY_LINEAR_SWAP*>(cmd_addr);
        pkt->HEADER_UNION.op = SDMA_OP_COPY;
        pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
        pkt->COUNT_UNION.count = copy_size - 1;
        pkt->ADDR_A_LO_UNION.DW_3_DATA = ptrlow32(cur_addr_a);
        pkt->ADDR_A_HI_UNION.addr_a_63_32 = ptrhigh32(cur_addr_a);
        pkt->ADDR_B_LO_UNION.DW_5_DATA = ptrlow32(cur_addr_b);
        pkt->ADDR_B_HI_UNION.addr_b_63_32 = ptrhigh32(cur_addr_b);
      }

      cmd_addr += sizeof(SDMA_PKT_COPY_LINEAR_SWAP);
      cur_size += copy_size;
    }
  }

  /// @brief Build linear sub-window (rect) copy packets.
  /// @p append is called per tile to obtain packet storage; it must return a
  /// pointer to at least @p size_t bytes of writable, ZEROED memory.  A lambda
  /// or function object is accepted and fully inlined (no type erasure).
  template <typename AppendFn>
  void BuildCopyRectCommand(AppendFn&& append, const PitchedPtr* dst, const Dim3* dst_offset,
                            const PitchedPtr* src, const Dim3* src_offset,
                            const Dim3* range) const {
    using detail::Min;
    auto maxAlignedElement = [](size_t width) {
      return detail::CountTrailingZeros64(width | 16ull);
    };

    const bool isGFX12Plus = (isa_.major >= 12);

    const uint32_t max_pitch = 1 << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::pitch_bits
                                                 : SDMA_PKT_COPY_LINEAR_RECT::pitch_bits);
    const uint64_t max_slice = 1ULL << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::slice_bits
                                                    : SDMA_PKT_COPY_LINEAR_RECT::slice_bits);
    const uint32_t max_x = 1 << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::rect_xy_bits
                                             : SDMA_PKT_COPY_LINEAR_RECT::rect_xy_bits);
    const uint32_t max_y = 1 << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::rect_xy_bits
                                             : SDMA_PKT_COPY_LINEAR_RECT::rect_xy_bits);
    const uint32_t max_z = 1 << (isGFX12Plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12::rect_z_bits
                                             : SDMA_PKT_COPY_LINEAR_RECT::rect_z_bits);

    auto max_ele = Min(maxAlignedElement(src->pitch), maxAlignedElement(dst->pitch));
    if (range->z != 1)
      max_ele = Min(max_ele, maxAlignedElement(src->slice), maxAlignedElement(dst->slice));

    auto min_ele = Min(max_ele, maxAlignedElement(range->x), maxAlignedElement(src_offset->x % 4),
                       maxAlignedElement(dst_offset->x % 4));

    if ((src->pitch >> min_ele) > max_pitch || (dst->pitch >> min_ele) > max_pitch)
      throw std::invalid_argument("Copy rect pitch out of limits.\n");
    if (range->z != 1) {
      if ((src->slice >> min_ele) > max_slice || (dst->slice >> min_ele) > max_slice)
        throw std::invalid_argument("Copy rect slice out of limits.\n");
    }

    for (uint32_t z = 0; z < range->z; z += max_z) {
      for (uint32_t y = 0; y < range->y; y += max_y) {
        uint32_t x = 0;
        while (x < range->x) {
          uint32_t width = range->x - x;

          auto aligned_ele = Min(maxAlignedElement((src_offset->x + x) % 4),
                                 maxAlignedElement((dst_offset->x + x) % 4), max_ele);

          int element = Min(maxAlignedElement(width), aligned_ele);
          int xcount = width >> element;

          if (xcount > static_cast<int>(max_x)) {
            element = aligned_ele;
            xcount = Min(width >> element, max_x);
          }

          uintptr_t sbase = (uintptr_t)src->base + src_offset->x + x +
                            (src_offset->y + y) * src->pitch + (src_offset->z + z) * src->slice;
          uintptr_t dbase = (uintptr_t)dst->base + dst_offset->x + x +
                            (dst_offset->y + y) * dst->pitch + (dst_offset->z + z) * dst->slice;
          uint32_t soff = (sbase % 4) >> element;
          uint32_t doff = (dbase % 4) >> element;
          sbase &= ~3ull;
          dbase &= ~3ull;

          x += xcount << element;

          if (isGFX12Plus) {
            auto* pkt = (SDMA_PKT_COPY_LINEAR_RECT_GFX12*)append(sizeof(SDMA_PKT_COPY_LINEAR_RECT));
            pkt->HEADER_UNION.op = SDMA_OP_COPY;
            pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_RECT;
            if (scope_fields_) pkt->HEADER_UNION.npd = 1;
            pkt->HEADER_UNION.element = element;
            pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = sbase;
            pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = sbase >> 32;
            pkt->SRC_PARAMETER_1_UNION.src_offset_x = soff;
            pkt->SRC_PARAMETER_2_UNION.src_pitch = (src->pitch >> element) - 1;
            pkt->SRC_PARAMETER_3_UNION.src_slice_pitch =
                (range->z == 1) ? 0 : (src->slice >> element) - 1;
            pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = dbase;
            pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = dbase >> 32;
            pkt->DST_PARAMETER_1_UNION.dst_offset_x = doff;
            pkt->DST_PARAMETER_2_UNION.dst_pitch = (dst->pitch >> element) - 1;
            pkt->DST_PARAMETER_3_UNION.dst_slice_pitch =
                (range->z == 1) ? 0 : (dst->slice >> element) - 1;
            pkt->RECT_PARAMETER_1_UNION.rect_x = xcount - 1;
            pkt->RECT_PARAMETER_1_UNION.rect_y = Min(range->y - y, max_y) - 1;
            pkt->RECT_PARAMETER_2_UNION.gfx12.rect_z = Min(range->z - z, max_z) - 1;
            if (scope_fields_) {
              pkt->RECT_PARAMETER_2_UNION.gfx125plus.dst_scope = SDMA_MEMORY_SCOPE_SYS;
              pkt->RECT_PARAMETER_2_UNION.gfx125plus.src_scope = SDMA_MEMORY_SCOPE_SYS;
            }
          } else {
            auto* pkt = (SDMA_PKT_COPY_LINEAR_RECT*)append(sizeof(SDMA_PKT_COPY_LINEAR_RECT));
            pkt->HEADER_UNION.op = SDMA_OP_COPY;
            pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_RECT;
            pkt->HEADER_UNION.element = element;
            pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = sbase;
            pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = sbase >> 32;
            pkt->SRC_PARAMETER_1_UNION.src_offset_x = soff;
            pkt->SRC_PARAMETER_2_UNION.src_pitch = (src->pitch >> element) - 1;
            pkt->SRC_PARAMETER_3_UNION.src_slice_pitch =
                (range->z == 1) ? 0 : (src->slice >> element) - 1;
            pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = dbase;
            pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = dbase >> 32;
            pkt->DST_PARAMETER_1_UNION.dst_offset_x = doff;
            pkt->DST_PARAMETER_2_UNION.dst_pitch = (dst->pitch >> element) - 1;
            pkt->DST_PARAMETER_3_UNION.dst_slice_pitch =
                (range->z == 1) ? 0 : (dst->slice >> element) - 1;
            pkt->RECT_PARAMETER_1_UNION.rect_x = xcount - 1;
            pkt->RECT_PARAMETER_1_UNION.rect_y = Min(range->y - y, max_y) - 1;
            pkt->RECT_PARAMETER_2_UNION.rect_z = Min(range->z - z, max_z) - 1;
          }
        }
      }
    }
  }

  ABCE_HD void BuildFillCommand(char* cmd_addr, void* ptr, uint32_t value, size_t count) const {
    using detail::Min;
    char* cur_ptr = reinterpret_cast<char*>(ptr);
    const uint32_t maxDwordCount = max_fill_size_ / sizeof(uint32_t);
    auto* pkt = reinterpret_cast<SDMA_PKT_CONSTANT_FILL*>(cmd_addr);

    while (count > 0) {
      const uint32_t fill_count = Min(count, size_t(maxDwordCount));

      pkt->HEADER_UNION.op = SDMA_OP_CONST_FILL;
      if (scope_fields_) {
        pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
        pkt->HEADER_UNION.npd = 1;
      }
      pkt->HEADER_UNION.fillsize = 2;

      pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_ptr);
      pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_ptr);
      pkt->DATA_UNION.src_data_31_0 = value;
      pkt->COUNT_UNION.count = (fill_count - 1) * sizeof(uint32_t);

      pkt++;
      cur_ptr += fill_count * sizeof(uint32_t);
      count -= fill_count;
    }
  }

  ABCE_HD void BuildPollCommand(char* cmd_addr, void* addr, uint32_t reference,
                                uint32_t mask = UINT32_MAX) const {
    auto* pkt = reinterpret_cast<SDMA_PKT_POLL_REGMEM*>(cmd_addr);
    pkt->HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
    pkt->HEADER_UNION.mem_poll = 1;
    pkt->HEADER_UNION.func = SDMA_FUNC_EQUAL;
    pkt->ADDR_LO_UNION.addr_31_0 = ptrlow32(addr);
    pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);
    pkt->VALUE_UNION.value = reference;
    pkt->MASK_UNION.mask = mask;
    pkt->DW5_UNION.interval = 0x04;
    pkt->DW5_UNION.retry_count = 0xfff;
    if (scope_fields_) pkt->DW5_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
  }

  ABCE_HD void BuildPoll64bCommand(char* cmd_addr, void* addr, uint64_t reference,
                                   uint64_t mask = UINT64_MAX) const {
    auto* pkt = reinterpret_cast<SDMA_PKT_POLL_MEM_64B_GFX125PLUS*>(cmd_addr);
    pkt->HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
    pkt->HEADER_UNION.sub_op = SDMA_SUBOP_POLL_MEM_64B;
    pkt->HEADER_UNION.func = SDMA_FUNC_EQUAL;
    pkt->ADDR_LO_UNION.addr_31_3 = ptrlow32(addr) >> 3;
    pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);
    pkt->REFERENCE_LO_UNION.reference_31_0 = static_cast<uint32_t>(reference);
    pkt->REFERENCE_HI_UNION.reference_63_32 = static_cast<uint32_t>(reference >> 32);
    pkt->MASK_LO_UNION.mask_31_0 = static_cast<uint32_t>(mask);
    pkt->MASK_HI_UNION.mask_63_32 = static_cast<uint32_t>(mask >> 32);
    pkt->HEADER_UNION.sys = 1;
    pkt->DW7_UNION.retry_count = 0;
    if (scope_fields_) pkt->DW7_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
  }

  ABCE_HD void BuildFence64bCommand(char* cmd_addr, void* fence_addr, uint64_t fence_value) const {
    auto* pkt = reinterpret_cast<SDMA_PKT_FENCE_64B_GFX125PLUS*>(cmd_addr);
    pkt->HEADER_UNION.op = SDMA_OP_FENCE;
    pkt->HEADER_UNION.sub_op = SDMA_SUBOP_FENCE_64B;
    pkt->HEADER_UNION.mtype = 3;
    pkt->HEADER_UNION.sys = 1;
    if (scope_fields_) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

    pkt->ADDR_LO_UNION.addr_31_3 = ptrlow32(fence_addr) >> 3;
    pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence_addr);
    pkt->DATA_LO_UNION.data_31_0 = static_cast<uint32_t>(fence_value);
    pkt->DATA_HI_UNION.data_63_32 = static_cast<uint32_t>(fence_value >> 32);
  }

  ABCE_HD void BuildWaitSignalCopyCommand(char* cmd_addr, void* dst, const void* src, size_t size,
                                          void* wait_addr, void* signal_addr,
                                          uint64_t wait_reference = 0,
                                          uint64_t wait_mask = UINT64_MAX,
                                          bool boundary_wait_signal = false) const {
    const bool do_wait = (wait_addr != nullptr);
    const bool do_signal = (signal_addr != nullptr);

    uint32_t wait_dws[7] = {};
    uint32_t signal_dws[5] = {};
    {
      SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS tmpl;
      memset(&tmpl, 0, sizeof(tmpl));
      tmpl.HEADER_UNION.op = SDMA_OP_COPY;
      tmpl.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
      tmpl.COPY_PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
      tmpl.COPY_PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;

      if (do_wait) {
        tmpl.WAIT_FUNCTION_UNION.wait_function = SDMA_FUNC_EQUAL;
        tmpl.WAIT_FUNCTION_UNION.wait_scope = SDMA_MEMORY_SCOPE_SYS;
        tmpl.WAIT_ADDR_LO_UNION.wait_addr_31_3 = ptrlow32(wait_addr) >> 3;
        tmpl.WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wait_addr);
        tmpl.WAIT_REFERENCE_LO_UNION.wait_reference_31_0 =
            static_cast<uint32_t>(wait_reference);
        tmpl.WAIT_REFERENCE_HI_UNION.wait_reference_63_32 =
            static_cast<uint32_t>(wait_reference >> 32);
        tmpl.WAIT_MASK_LO_UNION.wait_mask_31_0 = static_cast<uint32_t>(wait_mask);
        tmpl.WAIT_MASK_HI_UNION.wait_mask_63_32 = static_cast<uint32_t>(wait_mask >> 32);
        const uint32_t* s = reinterpret_cast<const uint32_t*>(&tmpl);
        memcpy(wait_dws, s + 1, sizeof(wait_dws));
      }
      if (do_signal) {
        tmpl.SIGNAL_OPERATION_UNION.signal_operation = SDMA_SIGNAL_OP_SUB64;
        tmpl.SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
        tmpl.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = ptrlow32(signal_addr) >> 3;
        tmpl.SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = ptrhigh32(signal_addr);
        tmpl.SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
        tmpl.SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;
        const uint32_t* s = reinterpret_cast<const uint32_t*>(&tmpl);
        memcpy(signal_dws, s + 14, sizeof(signal_dws));
      }
    }

    size_t cur_size = 0;
    while (cur_size < size) {
      const uint32_t copy_size =
          static_cast<uint32_t>(detail::Min(size - cur_size, max_copy_size_));
      const bool chunk_wait =
          do_wait && (!boundary_wait_signal || cur_size == 0);
      const bool chunk_signal =
          do_signal && (!boundary_wait_signal || cur_size + copy_size == size);

      uint32_t* out = reinterpret_cast<uint32_t*>(cmd_addr);
      uint32_t n = 0;

      SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS header{};
      header.HEADER_UNION.op = SDMA_OP_COPY;
      header.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
      header.HEADER_UNION.wait = chunk_wait ? 1 : 0;
      header.HEADER_UNION.signal = chunk_signal ? 1 : 0;
      out[n++] = reinterpret_cast<const uint32_t*>(&header)[0];
      if (chunk_wait) {
        memcpy(out + n, wait_dws, sizeof(wait_dws));
        n += 7;
      }

      const char* cur_src = reinterpret_cast<const char*>(src) + cur_size;
      char* cur_dst = reinterpret_cast<char*>(dst) + cur_size;
      SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS body;
      body.COPY_COUNT_UNION.DW_8_DATA = 0;
      body.COPY_COUNT_UNION.copy_count = copy_size - 1;
      body.COPY_PARAMETER_UNION.DW_9_DATA = 0;
      body.COPY_PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
      body.COPY_PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;
      body.SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(cur_src);
      body.SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(cur_src);
      body.DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(cur_dst);
      body.DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(cur_dst);
      const uint32_t* bd = reinterpret_cast<const uint32_t*>(&body);
      memcpy(out + n, bd + 8, 6 * sizeof(uint32_t));
      n += 6;

      if (chunk_signal) {
        memcpy(out + n, signal_dws, sizeof(signal_dws));
        n += 5;
      }

      cmd_addr += n * sizeof(uint32_t);
      cur_size += copy_size;
    }
  }

  ABCE_HD void BuildWaitSignalIndirectCopyCommand(char* cmd_addr, void* dst, const void* src,
                                                  size_t size, bool indirect_src, bool indirect_dst,
                                                  void* wait_addr, void* signal_addr,
                                                  uint64_t wait_reference = 0,
                                                  uint64_t wait_mask = UINT64_MAX) const {
    const bool do_wait = (wait_addr != nullptr);
    const bool do_signal = (signal_addr != nullptr);

    SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    tmpl.HEADER_UNION.op = SDMA_OP_COPY;
    tmpl.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_INDIRECT;
    tmpl.HEADER_UNION.indirect_src = indirect_src ? 1 : 0;
    tmpl.HEADER_UNION.indirect_dst = indirect_dst ? 1 : 0;
    tmpl.HEADER_UNION.wait = do_wait ? 1 : 0;
    tmpl.HEADER_UNION.signal = do_signal ? 1 : 0;

    if (do_wait) {
      tmpl.WAIT_FUNCTION_UNION.wait_function = SDMA_FUNC_EQUAL;
      tmpl.WAIT_FUNCTION_UNION.wait_scope = SDMA_MEMORY_SCOPE_SYS;
      tmpl.WAIT_ADDR_LO_UNION.wait_addr_31_3 = ptrlow32(wait_addr) >> 3;
      tmpl.WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wait_addr);
      tmpl.WAIT_REFERENCE_LO_UNION.wait_reference_31_0 =
          static_cast<uint32_t>(wait_reference);
      tmpl.WAIT_REFERENCE_HI_UNION.wait_reference_63_32 =
          static_cast<uint32_t>(wait_reference >> 32);
      tmpl.WAIT_MASK_LO_UNION.wait_mask_31_0 = static_cast<uint32_t>(wait_mask);
      tmpl.WAIT_MASK_HI_UNION.wait_mask_63_32 = static_cast<uint32_t>(wait_mask >> 32);
    }

    tmpl.COPY_COUNT_UNION.copy_count = static_cast<uint32_t>(size) - 1;
    tmpl.COPY_PARAMETER_UNION.copy_dst_scope = SDMA_MEMORY_SCOPE_SYS;
    tmpl.COPY_PARAMETER_UNION.copy_src_scope = SDMA_MEMORY_SCOPE_SYS;
    tmpl.COPY_PARAMETER_UNION.indirect_addr_scope = SDMA_MEMORY_SCOPE_SYS;
    tmpl.SRC_ADDR_LO_UNION.copy_src_addr_31_0 = ptrlow32(src);
    tmpl.SRC_ADDR_HI_UNION.copy_src_addr_63_32 = ptrhigh32(src);
    tmpl.DST_ADDR_LO_UNION.copy_dst_addr_31_0 = ptrlow32(dst);
    tmpl.DST_ADDR_HI_UNION.copy_dst_addr_63_32 = ptrhigh32(dst);

    if (do_signal) {
      tmpl.SIGNAL_OPERATION_UNION.signal_operation = SDMA_SIGNAL_OP_SUB64;
      tmpl.SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
      tmpl.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = ptrlow32(signal_addr) >> 3;
      tmpl.SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = ptrhigh32(signal_addr);
      tmpl.SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
      tmpl.SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;
    }

    // Emit only the present DW blocks contiguously: header, optional wait, the
    // 6-DW copy block (DW8-13), optional signal.
    const uint32_t* s = reinterpret_cast<const uint32_t*>(&tmpl);
    uint32_t* out = reinterpret_cast<uint32_t*>(cmd_addr);
    uint32_t n = 0;
    out[n++] = s[0];
    if (do_wait) {
      memcpy(out + n, s + 1, 7 * sizeof(uint32_t));
      n += 7;
    }
    memcpy(out + n, s + 8, 6 * sizeof(uint32_t));
    n += 6;
    if (do_signal) {
      memcpy(out + n, s + 14, 5 * sizeof(uint32_t));
      n += 5;
    }
  }

  ABCE_HD void BuildWaitSignalSwapCommand(char* cmd_addr, void* addr_a, void* addr_b, size_t size_a,
                                          size_t size_b, void* wait_addr, void* signal_addr,
                                          uint64_t wait_reference = 0,
                                          uint64_t wait_mask = UINT64_MAX,
                                          bool boundary_wait_signal = false) const {
    const bool do_wait = (wait_addr != nullptr);
    const bool do_signal = (signal_addr != nullptr);
    const size_t max_copy_size = SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS::kMaxSize_;

    uint32_t wait_dws[7] = {};
    uint32_t signal_dws[5] = {};
    {
      SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS tmpl;
      memset(&tmpl, 0, sizeof(tmpl));
      tmpl.HEADER_UNION.op = SDMA_OP_COPY;
      tmpl.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;

      if (do_wait) {
        tmpl.WAIT_FUNCTION_UNION.wait_function = SDMA_FUNC_EQUAL;
        tmpl.WAIT_FUNCTION_UNION.wait_scope = SDMA_MEMORY_SCOPE_SYS;
        tmpl.WAIT_ADDR_LO_UNION.wait_addr_31_3 = ptrlow32(wait_addr) >> 3;
        tmpl.WAIT_ADDR_HI_UNION.wait_addr_63_32 = ptrhigh32(wait_addr);
        tmpl.WAIT_REFERENCE_LO_UNION.wait_reference_31_0 =
            static_cast<uint32_t>(wait_reference);
        tmpl.WAIT_REFERENCE_HI_UNION.wait_reference_63_32 =
            static_cast<uint32_t>(wait_reference >> 32);
        tmpl.WAIT_MASK_LO_UNION.wait_mask_31_0 = static_cast<uint32_t>(wait_mask);
        tmpl.WAIT_MASK_HI_UNION.wait_mask_63_32 = static_cast<uint32_t>(wait_mask >> 32);
        const uint32_t* s = reinterpret_cast<const uint32_t*>(&tmpl);
        memcpy(wait_dws, s + 1, sizeof(wait_dws));
      }
      if (do_signal) {
        tmpl.SIGNAL_OPERATION_UNION.signal_operation = SDMA_SIGNAL_OP_SUB64;
        tmpl.SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
        tmpl.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = ptrlow32(signal_addr) >> 3;
        tmpl.SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = ptrhigh32(signal_addr);
        tmpl.SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
        const uint32_t* s = reinterpret_cast<const uint32_t*>(&tmpl);
        memcpy(signal_dws, s + 14, sizeof(signal_dws));
      }
    }

    size_t cur_a = 0, cur_b = 0;
    while (cur_a < size_a || cur_b < size_b) {
      const uint32_t chunk_a = static_cast<uint32_t>(detail::Min(size_a - cur_a, max_copy_size));
      const uint32_t chunk_b = static_cast<uint32_t>(detail::Min(size_b - cur_b, max_copy_size));
      // The single COUNT field is driven by the larger remaining side so it is
      // never 0 while the loop runs (an exhausted side has chunk == 0, which
      // would underflow to UINT32_MAX).  For a symmetric swap chunk_a == chunk_b.
      const uint32_t chunk = detail::Max(chunk_a, chunk_b);
      const bool chunk_wait =
          do_wait && (!boundary_wait_signal || (cur_a == 0 && cur_b == 0));
      const bool chunk_signal =
          do_signal &&
          (!boundary_wait_signal ||
           (cur_a + chunk_a == size_a && cur_b + chunk_b == size_b));

      const char* p_a = reinterpret_cast<const char*>(addr_a) + cur_a;
      const char* p_b = reinterpret_cast<const char*>(addr_b) + cur_b;

      uint32_t* out = reinterpret_cast<uint32_t*>(cmd_addr);
      uint32_t n = 0;

      SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS header{};
      header.HEADER_UNION.op = SDMA_OP_COPY;
      header.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
      header.HEADER_UNION.wait = chunk_wait ? 1 : 0;
      header.HEADER_UNION.signal = chunk_signal ? 1 : 0;
      out[n++] = reinterpret_cast<const uint32_t*>(&header)[0];
      if (chunk_wait) {
        memcpy(out + n, wait_dws, sizeof(wait_dws));
        n += 7;
      }

      SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS body;
      body.COUNT_UNION.DW_8_DATA = 0;
      body.COUNT_UNION.count = chunk - 1;
      body.COPY_PARAMETER_UNION.DW_9_DATA = 0;
      body.COPY_PARAMETER_UNION.scope_a = SDMA_MEMORY_SCOPE_SYS;
      body.COPY_PARAMETER_UNION.scope_b = SDMA_MEMORY_SCOPE_SYS;
      body.ADDR_A_LO_UNION.addr_a_31_0 = ptrlow32(p_a);
      body.ADDR_A_HI_UNION.addr_a_63_32 = ptrhigh32(p_a);
      body.ADDR_B_LO_UNION.addr_b_31_0 = ptrlow32(p_b);
      body.ADDR_B_HI_UNION.addr_b_63_32 = ptrhigh32(p_b);
      const uint32_t* bd = reinterpret_cast<const uint32_t*>(&body);
      memcpy(out + n, bd + 8, 6 * sizeof(uint32_t));
      n += 6;

      if (chunk_signal) {
        memcpy(out + n, signal_dws, sizeof(signal_dws));
        n += 5;
      }

      cmd_addr += n * sizeof(uint32_t);
      cur_a += chunk_a;
      cur_b += chunk_b;
    }
  }

  ABCE_HD void BuildAtomicAddCommand(char* cmd_addr, void* addr, uint64_t value) const {
    auto* pkt = reinterpret_cast<SDMA_PKT_ATOMIC*>(cmd_addr);
    pkt->HEADER_UNION.op = SDMA_OP_ATOMIC;
    pkt->HEADER_UNION.operation = SDMA_ATOMIC_ADD64;
    if (scope_fields_) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

    pkt->ADDR_LO_UNION.addr_31_0 = ptrlow32(addr);
    pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);
    pkt->SRC_DATA_LO_UNION.src_data_31_0 = static_cast<uint32_t>(value);
    pkt->SRC_DATA_HI_UNION.src_data_63_32 = static_cast<uint32_t>(value >> 32);
  }

  ABCE_HD void BuildAtomicDecrementCommand(char* cmd_addr, void* addr) const {
    BuildAtomicAddCommand(cmd_addr, addr, UINT64_MAX);
  }

  ABCE_HD void BuildGetGlobalTimestampCommand(char* cmd_addr, void* write_address) const {
    auto* pkt = reinterpret_cast<SDMA_PKT_TIMESTAMP*>(cmd_addr);
    pkt->HEADER_UNION.op = SDMA_OP_TIMESTAMP;
    pkt->HEADER_UNION.sub_op = SDMA_SUBOP_TIMESTAMP_GET_GLOBAL;
    if (scope_fields_) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

    pkt->ADDR_LO_UNION.addr_31_0 = ptrlow32(write_address);
    pkt->ADDR_HI_UNION.addr_63_32 = ptrhigh32(write_address);
  }

  ABCE_HD void BuildTrapCommand(char* cmd_addr, uint32_t event_id) const {
    auto* pkt = reinterpret_cast<SDMA_PKT_TRAP*>(cmd_addr);
    pkt->HEADER_UNION.op = SDMA_OP_TRAP;
    pkt->INT_CONTEXT_UNION.int_ctx = event_id;
  }

  ABCE_HD void BuildHdpFlushCommand(char* cmd_addr) const {
    ABCE_ASSERT(cmd_addr != NULL);
    memcpy(cmd_addr, &hdp_flush_cmd, sizeof(hdp_flush_cmd));
  }

  ABCE_HD void BuildGCRCommand(char* cmd_addr, bool invalidate) const {
    ABCE_ASSERT(cmd_addr != NULL);

    if (is_gfx125plus_) {
      auto* pkt = reinterpret_cast<SDMA_PKT_GCR_GFX125PLUS*>(cmd_addr);
      pkt->HEADER_UNION.op = SDMA_OP_GCR;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_USER_GCR;
      if (invalidate) {
        pkt->WORD3_UNION.GCR_CONTROL_GL2_SCOPE = 1;
        pkt->WORD3_UNION.GCR_CONTROL_GL2_INV = 1;
      } else {
        pkt->WORD3_UNION.GCR_CONTROL_GL2_SCOPE = 1;
        pkt->WORD3_UNION.GCR_CONTROL_GL2_WB = 1;
      }
      pkt->WORD3_UNION.GCR_CONTROL_GL2_RANGE = 0;
    } else {
      auto* pkt = reinterpret_cast<SDMA_PKT_GCR*>(cmd_addr);
      pkt->HEADER_UNION.op = SDMA_OP_GCR;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_USER_GCR;
      pkt->WORD2_UNION.GCR_CONTROL_GL2_WB = 1;
      pkt->WORD2_UNION.GCR_CONTROL_GLK_WB = 1;
      if (invalidate) {
        pkt->WORD2_UNION.GCR_CONTROL_GL2_INV = 1;
        pkt->WORD2_UNION.GCR_CONTROL_GL1_INV = 1;
        pkt->WORD2_UNION.GCR_CONTROL_GLV_INV = 1;
        pkt->WORD2_UNION.GCR_CONTROL_GLK_INV = 1;
      }
      pkt->WORD2_UNION.GCR_CONTROL_GL2_RANGE = 0;
    }
  }

 private:
  IsaVersion isa_;
  PacketCaps packet_caps_;
  bool scope_fields_;
  bool is_gfx125plus_;
  size_t max_copy_size_;
  bool use_extended_count_;
  size_t max_fill_size_;
};

}  // namespace abce

#endif  // ABCE_BUILDER_H_
