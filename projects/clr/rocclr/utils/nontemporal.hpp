/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef CLR_ROCCLR_UTILS_NONTEMPORAL_HPP_
#define CLR_ROCCLR_UTILS_NONTEMPORAL_HPP_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>
#include "os/os.hpp"

#if defined(ATI_ARCH_X86)
#if defined(__MINGW64__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#endif

namespace amd {

// Non-temporal store utilities for AQL packet submission.
//
// The ordering analysis below (WC buffers, SFENCE semantics, UC MMIO
// flushes) is specific to x86/x86-64.  On ARM or other CPU ISAs the memory
// model, cache attributes, and barrier instructions are different.  The
// non-x86 fallback paths in this file use plain memcpy / atomic fences
// which are correct but the performance rationale does not directly apply.
//
// --- CPU memory-mapping of device memory (x86) ---
//
// On discrete GPUs connected via PCIe, device memory mapped into the CPU
// address space is marked Write Combining (WC).  WC has weaker ordering
// than normal Write Back (WB) cached DRAM:
//
//   * Regular stores enter a write-combining buffer and may be coalesced
//     or reordered.  An explicit SFENCE is required to guarantee ordering
//     between groups of WC writes.
//   * Each SFENCE flushes the WC buffer and the CPU stalls until WriteACK
//     returns from the PCIe posted write (one bus round-trip).
//   * A write to UC (uncacheable) memory (e.g. the MMIO doorbell region)
//     also implicitly flushes the WC buffer, so no SFENCE is needed
//     between the last WC write and the doorbell.
//
// On A+A / XGMI topologies the GPU memory can also be mapped WB, giving
// the CPU the same strong ordering as host DRAM.
// In that case:
//
//   * Write ordering is guaranteed by x86 TSO — no SFENCE needed between
//     regular stores.
//   * CPU writes stay in the cache hierarchy; GPU reads may have to
//     probe-read them out of CPU caches (higher latency than reading from
//     local GPU memory).
//   * Non-temporal stores bypass the cache and go directly to the
//     destination, avoiding the probe-read penalty.  They still require
//     SFENCE for ordering relative to subsequent stores.
//
// On x86, certain instructions follow the WC memory model even when
// targeting WB memory:
//
//   * Streaming / non-temporal stores (MOVNTQ, MOVNTI, MOVNTPS, etc.)
//     do not necessarily close/flush the write-combining buffer
//     immediately; later stores may still coalesce in the WC buffer.
//     An SFENCE is required to guarantee the buffer is flushed and
//     the store begins its trek to the destination.
//
//   * MOVDIR64B / MOVDIRI close the WC buffer after execution but
//     still follow the weak WC memory ordering model (i.e. they can
//     be reordered with respect to other WC and NT stores).
//
// This is why SFENCE is necessary even on A+A / WB-mapped topologies:
// the NT stores used here always behave as WC regardless of the
// underlying memory type.
//
// These utilities use non-temporal stores so they are correct (and
// performant) regardless of whether the target is WC or WB mapped.
//
// --- Ordering points in the AQL submission flow ---
//
// When queues / kernargs are in device memory, the NT store sequence is:
//
//   1.  nontemporalMemcpy  — kernel args CPU → GPU (step #1)
//   2.  nontemporalCopyAQL — packet body (w/ invalid header) → ring buffer (step 6)
//       metadata_preloader writes metadata packet (steps 5/7/8)
//   3.  nontemporalStoreFence — ONE sfence that subsumes ordering points
//       1(a), 6(a), 7, and 9 from the AQL submission diagram:
//         * kernel-arg NT stores are visible          (#1a)
//         * dispatch-packet body is ordered before its header write (#6a)
//         * metadata body is ordered before header writes          (#7)
//         * metadata is valid before dispatch header is released   (#9)
//   4.  packet_store_release — atomic 32-bit write of valid header (step #10)
//   5.  doorbell ring — see note below                             (step #12)
//
//  +------------------+  +------------------+  +------------------+  +------------------+
//  | 1. Copy N bytes   |  | 2. Atomically    |  | 3. Create kernel |  | 4. Create kernel |
//  | of kernel args    |  | increment        |  | meta-data packet |  | dispatch packet  |
//  | from CPU -> GPU   |  | write_dispatch_id|  | in CPU w/invalid |  | in CPU w/invalid |
//  +--------+---------+  +--------+---------+  | header           |  | header           |
//           |                      |            +--------+---------+  +--------+---------+
//           |                      |                     |                     |
//           |                      |                     v                     v
//           |                      |            +------------------+  +------------------+
//           |                      |            | 5. Copy meta-data|  | 6. Copy dispatch |
//           |                      |            | packets w/invalid|  | packet w/invalid |
//           |                      |            | hdrs to GPU ring |  | hdr to GPU ring  |
//           |                      |            | buffer           |  | buffer           |
//           |                      |            +--------+---------+  +--------+---------+
//           |                      |                     |                     |
//           |                      |                     v                     v
//           |                      |       *============================*  *=============*
//           |                      |       H 7. Ensure meta-data packet H  H 6(a). Ensur-H
//           |                      |       H body write is ordered      H  H e dispatch  H
//           |                      |       H before header writes       H  H pkt body is H
//           |                      |       *=============+==============*  H ordered be- H
//           |                      |                     |                 H fore header H
//           |                      |                     v                 H write       H
//           |                      |            +------------------+       *======+======*
//           |                      |            | 8. Write valid   |              |
//           |                      |            | headers into     |              |
//           |                      |            | meta-data packet |              |
//           |                      |            | segments         |              |
//           |                      |            +--------+---------+              |
//           |                      |                     |                        |
//           |                      |                     |                        |
//           |                      |                     |                        |
//           |                      |                     |                        |
//           |                      |                     |                        |
//           |                      |                     |                        |
//           |                      |                     |                        v
//  *========+========*             |                     v                 *=============*
//  H 1(a). Ensure    H             |       *============================*  H 6(a). Ensur-H
//  H kernel args are H             |       H 9. Ensure meta-data packet H  H e dispatch  H
//  H visible before  H             |       H is valid before dispatch   H  H pkt body is H
//  H dispatch is     H             |       H pkt is released to HW      H  H ordered be- H
//  H legal           H             |       *=============+==============*  H fore header H
//  *========+========*             |                     |                 H write       H
//           |                      |                     |                 *======+======*
//           |                      |                     |                        |
//           +----------------------+---------------------+------------------------+
//                                  |
//                                  v
//                         +------------------+
//                         | 10. Write valid  |
//                         | header into      |
//                         | kernel dispatch  |
//                         | packet           |
//                         +--------+---------+
//                                  |
//                                  v
//                         *==================*
//                         H 11. Ensure valid H
//                         H packets visible  H
//                         H to the hardware  H
//                         *========+=========*
//                                  |
//                                  v
//                         +------------------+
//                         | 12. Ring the     |
//                         | queue's MMIO     |
//                         | doorbell         |
//                         +------------------+
//
//  Legend:  +--..--+  data operation    *==..==*  ordering / fence point
//                                      H      H
//
 //  CLR uses THREE sfences in the legacy (non-MOVDIR64B) submission path:
 //  (1) Inside metadata_preloader::SetPacket, between the metadata body
 //      writes and the metadata header writes (ordering point #7).  This
 //      ensures the CP's prefetcher never sees valid metadata headers
 //      before the body is flushed — required because the metadata ring
 //      buffer may be WC-mapped and the prefetcher reads it independently.
 //  (2) nontemporalStoreFence() before step #10 (packet_store_release),
 //      satisfying ordering points #1(a), #6(a), and #9.
 //  (3) sfence before the doorbell ring (step #11).  This is inside
 //      ROCr's AqlQueue::StoreRelease and ensures the valid dispatch
 //      header is flushed from the WC buffer before the UC doorbell write.
 //
 // SFENCE orders ALL preceding NT/WC stores, not just the most recent.
 // On PCIe each SFENCE causes the CPU to stall until WriteACK returns
 // from the PCIe posted write / XGMI coherence station.
 //
 // --- MOVDIR64B optimisation (runtime-detected) ---
 //
// When MOVDIR64B is available (hasMovdir64b()), the submission path
// switches to an optimised flow that reduces 3 sfences to 2:
 //
 // MOVDIR64B atomically writes 64 bytes and closes the WC buffer after
 // execution.  If used for both the metadata packet and the dispatch
 // packet, the body-to-header ordering within each 64-byte write is
// guaranteed atomically, eliminating sfences #1 and #2.  The explicit
// sfence before the dispatch MOVDIR64B ensures kernarg NT stores and
// metadata MOVDIR64B writes are globally ordered before the dispatch
// packet becomes visible (#1a / #9). ROCr still performs the final
// doorbell sfence in hsa_signal_store_screlease:
 //
 //   1. MOVDIR64B metadata segments (body + headers atomic per 64B, WC closed)
//   2. nontemporalStoreFence — explicit sfence (#1a / #7 / #9)
 //   3. MOVDIR64B dispatch packet   (body + header atomic, WC closed)
//   4. doorbell ring               (ROCr sfence + UC write)
 //
// This reduces the dispatch hot path from 3 sfences to 2.
 //
 // Note: the sfence between steps 1 and 3 also guarantees that
 // metadata MOVDIR64B writes (which follow WC ordering) are globally
 // visible before the dispatch MOVDIR64B.  This satisfies ordering
 // point #9 (metadata valid before dispatch is released to HW).
//
// --- Doorbell ring (step #11 / #12) ---
//
// CLR calls hsa_signal_store_screlease(doorbell_signal, index).
// ROCr's AqlQueue::StoreRelease (amd_aql_queue.cpp) does:
//
//     std::atomic_thread_fence(memory_order_release);   // (redundant)
//     _mm_sfence();
//     *(signal_.hardware_doorbell_ptr) = uint64_t(value);
//
// That SFENCE ensures the valid header written by packet_store_release
// (step #10) is flushed from the WC buffer before the UC doorbell store.
// CLR does NOT need its own SFENCE between step #10 and the doorbell.
// The UC (uncacheable) MMIO doorbell write implicitly flushes any remaining
// WC data, so no additional SFENCE is needed after the doorbell store.

// ================================================================================================
#if IS_LINUX
__attribute__((optimize("unroll-all-loops"), always_inline)) static inline void nontemporalMemcpy(
    void* __restrict dst, const void* __restrict src, size_t size) {
#if defined(ATI_ARCH_X86)
  // Drain unaligned head with scalar NT stores to reach 4-byte then 16-byte
  // dst alignment.  The streaming store intrinsics require aligned destinations;
  // source is always loaded with unaligned intrinsics since the caller may pass
  // an arbitrary struct pointer (e.g. AQL packet body at +4 offset).
  auto addr = reinterpret_cast<uintptr_t>(dst);
  if ((addr & 0x3) && size) {
    auto lead = std::min(size, static_cast<size_t>(4 - (addr & 0x3)));
    std::memcpy(dst, src, lead);
    dst = static_cast<char*>(dst) + lead;
    src = static_cast<const char*>(src) + lead;
    size -= lead;
    addr = reinterpret_cast<uintptr_t>(dst);
  }
  while ((addr & 0xF) && size >= sizeof(int)) {
    int val;
    std::memcpy(&val, src, sizeof(int));
    _mm_stream_si32(static_cast<int*>(dst), val);
    dst = static_cast<char*>(dst) + sizeof(int);
    src = static_cast<const char*>(src) + sizeof(int);
    size -= sizeof(int);
    addr += sizeof(int);
  }

#if defined(__AVX512F__)
  if ((reinterpret_cast<uintptr_t>(dst) & (sizeof(__m512i) - 1)) == 0) {
    for (auto i = 0u; i != size / sizeof(__m512i); ++i) {
      _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                          _mm512_loadu_si512(src));
      dst = static_cast<char*>(dst) + sizeof(__m512i);
      src = static_cast<const char*>(src) + sizeof(__m512i);
    }
    size = size % sizeof(__m512i);
  }
#endif

#if defined(__AVX__)
  if ((reinterpret_cast<uintptr_t>(dst) & (sizeof(__m256i) - 1)) == 0) {
    for (auto i = 0u; i != size / sizeof(__m256i); ++i) {
      _mm256_stream_si256(reinterpret_cast<__m256i*>(dst),
                          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src)));
      dst = static_cast<char*>(dst) + sizeof(__m256i);
      src = static_cast<const char*>(src) + sizeof(__m256i);
    }
    size = size % sizeof(__m256i);
  }
#endif

  for (auto i = 0u; i != size / sizeof(__m128i); ++i) {
    _mm_stream_si128(reinterpret_cast<__m128i*>(dst),
                     _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
    dst = static_cast<char*>(dst) + sizeof(__m128i);
    src = static_cast<const char*>(src) + sizeof(__m128i);
  }
  size = size % sizeof(__m128i);

  for (auto i = 0u; i != size / sizeof(long long); ++i) {
    long long val;
    std::memcpy(&val, src, sizeof(long long));
    _mm_stream_si64(static_cast<long long*>(dst), val);
    dst = static_cast<char*>(dst) + sizeof(long long);
    src = static_cast<const char*>(src) + sizeof(long long);
  }
  size = size % sizeof(long long);

  for (auto i = 0u; i != size / sizeof(int); ++i) {
    int val;
    std::memcpy(&val, src, sizeof(int));
    _mm_stream_si32(static_cast<int*>(dst), val);
    dst = static_cast<char*>(dst) + sizeof(int);
    src = static_cast<const char*>(src) + sizeof(int);
  }

  size = size % sizeof(int);
  std::memcpy(dst, src, size);
#else
  std::memcpy(dst, src, size);
#endif
}
#else
static inline void nontemporalMemcpy(void* __restrict dst, const void* __restrict src,
                                     size_t size) {
  std::memcpy(dst, src, size);
}
#endif

// Copy the body of an AQL packet (64 bytes) using non-temporal stores.
// The header (first 4 bytes) is zeroed (HSA_PACKET_TYPE_INVALID) so that the
// packet is invisible to the CP until the caller publishes the real header via
// packet_store_release.  Caller MUST issue nontemporalStoreFence() between
// this call and the header write (ordering point #6a / #9 in the diagram).
template <typename AqlPacket>
static inline void nontemporalCopyAQL(AqlPacket* __restrict dst,
                                      const AqlPacket* __restrict src) {
  static_assert(sizeof(AqlPacket) == 64, "AQL packets must be 64 bytes");
  // Skip the first 4 bytes (header + rest) -- they are written atomically later.
  // Zero the header in the destination to keep it HSA_PACKET_TYPE_INVALID.
  auto* d = reinterpret_cast<uint8_t*>(dst);
  auto* s = reinterpret_cast<const uint8_t*>(src);
  std::memset(d, 0, 4);
  nontemporalMemcpy(d + 4, s + 4, sizeof(AqlPacket) - 4);
}

// Drain the write-combining buffer and wait for all preceding NT stores to
// reach the destination coherence point.  On PCIe this stalls the CPU until
// WriteACK returns, so minimise the number of fences per submission.
static inline void nontemporalStoreFence() {
#if IS_LINUX && defined(ATI_ARCH_X86)
  _mm_sfence();
#else
  std::atomic_thread_fence(std::memory_order_release);
#endif
}

// ---- MOVDIR64B support ----------------------------------------------------------
//
// MOVDIR64B atomically writes 64 bytes and closes the WC buffer after
// execution.  When used for AQL packet submission it eliminates multiple
// sfences — see the "MOVDIR64B optimisation" section in the long comment
// above for the full ordering analysis.
//
// Detection lives in Os::hasMovdir64b() (CPUID leaf 7, sub-leaf 0, ECX
// bit 28).  The result is cached on first call.

static inline bool hasMovdir64b() {
#if IS_LINUX && defined(ATI_ARCH_X86) && defined(_LP64)
  return Os::hasMovdir64b();
#else
  return false;
#endif
}

// Atomically write 64 bytes from src to dst using MOVDIR64B.
//
// dst MUST be 64-byte aligned (hardware requirement; #GP otherwise).
// src has no alignment requirement but 64-byte alignment is optimal.
//
// The instruction reads 64 bytes from [src] (normal load, can be WB/WC/UC)
// and writes them as a single atomic 64-byte store to [dst].  The WC buffer
// is closed after execution, which guarantees the write is posted before
// any subsequent store begins filling a new WC buffer entry.
//
// Caller must ensure hasMovdir64b() == true before calling.
static inline void movdir64b_copy64(void* __restrict dst, const void* __restrict src) {
#if IS_LINUX && defined(ATI_ARCH_X86) && defined(_LP64)
  assert((reinterpret_cast<uintptr_t>(dst) & 63) == 0 && "MOVDIR64B dst must be 64-byte aligned");
  // AT&T syntax: movdir64b (%src_reg), %dst_reg
  __asm__ __volatile__("movdir64b %1, %0"
                       : : "r"(dst), "m"(*(const char(*)[64])src) : "memory");
#else
  (void)dst; (void)src;
#endif
}

// Write a full 64-byte AQL packet (body + valid header) atomically using
// MOVDIR64B.  Unlike the split nontemporalCopyAQL + sfence +
// packet_store_release sequence, this writes all 64 bytes including the
// valid header in a single atomic store.  The WC buffer is closed after
// execution, so no sfence is needed between this write and a subsequent
// UC doorbell store.
//
// src is the caller's CPU-side staging packet (e.g. dispatchPacket on the
// stack).  The valid header/rest are merged into a local aligned copy
// before the MOVDIR64B.
template <typename AqlPacket>
static inline void nontemporalWriteAQL(AqlPacket* __restrict dst,
                                       const AqlPacket* __restrict src,
                                       uint16_t header, uint16_t rest) {
  static_assert(sizeof(AqlPacket) == 64, "AQL packets must be 64 bytes");
  alignas(64) AqlPacket staging;
  std::memcpy(&staging, src, sizeof(AqlPacket));
  *reinterpret_cast<uint32_t*>(&staging) = header | (static_cast<uint32_t>(rest) << 16);
  movdir64b_copy64(dst, &staging);
}

// STL-compatible allocator that guarantees N-byte alignment.
// Use with std::vector to ensure the backing store is aligned for SIMD NT stores.
template <typename T, std::size_t Align>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

  T* allocate(std::size_t n) {
    void* p = ::operator new(n * sizeof(T), std::align_val_t{Align});
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t{Align});
  }

  template <typename U>
  struct rebind { using other = AlignedAllocator<U, Align>; };

  bool operator==(const AlignedAllocator&) const noexcept { return true; }
  bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <typename T>
using AlignedVector64 = std::vector<T, AlignedAllocator<T, 64>>;

}  // namespace amd

#endif  // CLR_ROCCLR_UTILS_NONTEMPORAL_HPP_
