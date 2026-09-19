/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — SDMA packet builders.

#ifndef ABCE_BUILDER_H_
#define ABCE_BUILDER_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "abce_config.h"
#include "abce_types.h"
#include "sdma_packets.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ABCE_ASSERT() pulls a host-only symbol; make it a no-op during device
// compilation.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define ABCE_ASSERT(x) ((void)0)
#else
#define ABCE_ASSERT(x) assert(x)
#endif

//===----------------------------------------------------------------------===//
// Address helpers
//===----------------------------------------------------------------------===//

static inline uint32_t abce_ptr_low32(const void* ptr) {
  return (uint32_t)(uintptr_t)ptr;
}

static inline uint32_t abce_ptr_high32(const void* ptr) {
  return (uint32_t)((uint64_t)(uintptr_t)ptr >> 32);
}

//===----------------------------------------------------------------------===//
// abce_builder_t
//===----------------------------------------------------------------------===//

// SDMA (copy engine) packet writers for one device/queue.
//
// Each builder writes packets into a caller-provided command buffer. The number
// of packets is computed internally from the transfer size and the configured
// per-packet limit — callers never pass a packet count. The
// abce_builder_num_*_packets() helpers let callers query the count for buffer
// sizing or signal pre-arming.
//
// ============================ ZEROED-BUFFER CONTRACT ========================
// The fixed-layout builders (copy, broadcast, multicast, swap, fill, poll,
// fence, atomic, timestamp, trap, gcr, rect) DO NOT zero the destination
// buffer. They set only the meaningful bitfields and rely on every other bit of
// each packet DWORD already being 0. The caller MUST provide a zero-initialized
// buffer (e.g. the ring's reservation memset, or a NOP-padded region where
// 0 == NOP). Writing a builder over non-zeroed memory produces malformed
// packets.
//
// The fused wait/signal builders are the exception: they explicitly emit every
// DWORD they write (compacted, variable length), so their output does not
// depend on the zeroed-buffer contract. They use small stack scratch structs
// internally, which they zero themselves.
// ===========================================================================
//
// Packet format, GCR use, scope fields, and the default linear-copy limit are
// derived from the gfx IP using the OSS4/OSS5/OSS7 capability table. Treat the
// fields as read-only after abce_builder_initialize().
typedef struct abce_builder_t {
  abce_isa_version_t isa;
  abce_packet_caps_t packet_caps;
  bool scope_fields;
  bool is_gfx125plus;
  bool use_extended_count;
  size_t max_copy_size;
  // Fill sizing and fill emission both step by this, so they cannot disagree.
  uint32_t max_fill_dwords;
} abce_builder_t;

// Initializes a builder for |isa|. |config| may be null for the defaults, which
// are what abce_builder_config_initialize() produces.
//
// Fails with ABCE_STATUS_INVALID_ARGUMENT, leaving |out_builder| untouched, when
// |config| asks for a limit the packets cannot carry: a linear-copy limit above
// what this IP's COUNT field encodes, or a fill limit that is not a whole number
// of dwords within SDMA_PKT_CONSTANT_FILL_MAX_SIZE.
static inline abce_status_t abce_builder_initialize(abce_isa_version_t isa,
                                                    const abce_builder_config_t* config,
                                                    abce_builder_t* out_builder) {
  abce_builder_config_t defaults;
  if (!config) {
    abce_builder_config_initialize(&defaults);
    config = &defaults;
  }
  const abce_packet_caps_t packet_caps = abce_detect_packet_caps(isa);
  if (config->max_linear_copy_size > packet_caps.max_linear_copy_size ||
      config->max_fill_size % sizeof(uint32_t) != 0 ||
      config->max_fill_size > SDMA_PKT_CONSTANT_FILL_MAX_SIZE)
    return ABCE_STATUS_INVALID_ARGUMENT;

  out_builder->isa = isa;
  out_builder->packet_caps = packet_caps;
  out_builder->scope_fields = packet_caps.scope_fields;
  out_builder->is_gfx125plus = packet_caps.gfx125plus;
  out_builder->max_copy_size =
      config->max_linear_copy_size
          ? config->max_linear_copy_size
          : (config->use_copy_size_override ? packet_caps.max_linear_copy_size
                                            : SDMA_PKT_COPY_LINEAR_MAX_SIZE);
  out_builder->use_extended_count =
      config->max_linear_copy_size != 0 || config->use_copy_size_override;
  const size_t max_fill_size =
      config->max_fill_size ? config->max_fill_size : SDMA_PKT_CONSTANT_FILL_MAX_SIZE;
  out_builder->max_fill_dwords = (uint32_t)(max_fill_size / sizeof(uint32_t));
  return ABCE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// Device/config queries
//===----------------------------------------------------------------------===//

static inline bool abce_builder_is_gfx125plus(const abce_builder_t* builder) {
  return builder->is_gfx125plus;
}

static inline size_t abce_builder_max_copy_size(const abce_builder_t* builder) {
  return builder->max_copy_size;
}

static inline bool abce_builder_requires_gcr(const abce_builder_t* builder) {
  return builder->packet_caps.use_gcr;
}

//===----------------------------------------------------------------------===//
// Packet-count queries (for buffer sizing / signal pre-arming)
//===----------------------------------------------------------------------===//

// The counts are exact for every size — nothing overflows and nothing narrows —
// but a frame sized from a huge one would not be: frame sizing multiplies a
// count by a packet size and sums over the frame. Every operation the host or a
// kernel submits is therefore held to ABCE_MAX_OP_PACKETS before anything is
// reserved or emitted. A frame near that size is one no ring can hold; only a
// small max_linear_copy_size override brings an addressable copy close to it.
#define ABCE_MAX_OP_PACKETS ((uint64_t)UINT32_MAX)

static inline uint64_t abce_builder_num_copy_packets(const abce_builder_t* builder, size_t size) {
  return abce_div_round_up_u64(size, builder->max_copy_size);
}

static inline uint64_t abce_builder_num_broadcast_packets(const abce_builder_t* builder,
                                                          size_t size) {
  return abce_builder_num_copy_packets(builder, size);
}

static inline uint64_t abce_builder_num_multicast_packets(const abce_builder_t* builder,
                                                          size_t size) {
  return abce_builder_num_copy_packets(builder, size);
}

static inline uint64_t abce_builder_num_swap_packets(const abce_builder_t* builder, size_t size) {
  const size_t max_size = (builder->scope_fields && builder->is_gfx125plus)
                              ? SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS_MAX_SIZE
                              : SDMA_PKT_COPY_LINEAR_SWAP_MAX_SIZE;
  return abce_div_round_up_u64(size, max_size);
}

static inline uint64_t abce_builder_num_wait_signal_swap_packets(const abce_builder_t* builder,
                                                                 size_t size_a, size_t size_b) {
  (void)builder;
  return abce_div_round_up_u64(ABCE_MAX(size_a, size_b),
                               SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS_MAX_SIZE);
}

// |count| is in dwords, the unit a fill is emitted in.
static inline uint64_t abce_builder_num_fill_packets(const abce_builder_t* builder, size_t count) {
  return abce_div_round_up_u64(count, builder->max_fill_dwords);
}

//===----------------------------------------------------------------------===//
// Fused wait/copy/signal parameters
//===----------------------------------------------------------------------===//

// Synchronization carried by the gfx125plus fused wait/signal packets.
//
// Bundled into a struct because C has no default arguments and three of the
// five fields are defaulted at nearly every call site; abce_wait_signal_none()
// gives the "no wait, no signal" form.
typedef struct abce_wait_signal_params_t {
  void* wait_addr;    // null => no wait block is emitted.
  void* signal_addr;  // null => no signal block is emitted.
  uint64_t wait_reference;
  uint64_t wait_mask;
  // Puts the WAIT on the first chunk and the SIGNAL on the last, so a chunked
  // transfer consumes exactly one wait and one signal. This assumes the engine
  // retires chunks of one packet stream in order, so the final chunk's SIGNAL
  // cannot outrun an earlier chunk's data. ROCr's blit path instead waits and
  // signals on every chunk and pre-arms the output by N-1; pass false to get
  // that behavior. Chunking only occurs above the per-packet copy limit.
  bool boundary_wait_signal;
} abce_wait_signal_params_t;

static inline abce_wait_signal_params_t abce_wait_signal_none(void) {
  abce_wait_signal_params_t params;
  params.wait_addr = NULL;
  params.signal_addr = NULL;
  params.wait_reference = 0;
  params.wait_mask = UINT64_MAX;
  params.boundary_wait_signal = false;
  return params;
}

//===----------------------------------------------------------------------===//
// Builders (see ZEROED-BUFFER CONTRACT above)
//===----------------------------------------------------------------------===//

static inline void abce_builder_build_fence(const abce_builder_t* builder, char* cmd_addr,
                                            uint32_t* fence, uint32_t fence_value) {
  ABCE_ASSERT(cmd_addr != NULL);

  if (builder->isa.major >= 12) {
    SDMA_PKT_FENCE_GFX12* pkt = (SDMA_PKT_FENCE_GFX12*)cmd_addr;
    pkt->HEADER_UNION.op = SDMA_OP_FENCE;
    pkt->HEADER_UNION.mtype = 3;
    pkt->HEADER_UNION.sys = 1;
    if (builder->scope_fields) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

    pkt->ADDR_LO_UNION.addr_31_0 = abce_ptr_low32(fence);
    pkt->ADDR_HI_UNION.addr_63_32 = abce_ptr_high32(fence);
    pkt->DATA_UNION.data = fence_value;
  } else {
    SDMA_PKT_FENCE* pkt = (SDMA_PKT_FENCE*)cmd_addr;
    pkt->HEADER_UNION.op = SDMA_OP_FENCE;
    if (builder->isa.major >= 10) pkt->HEADER_UNION.mtype = 3;

    pkt->ADDR_LO_UNION.addr_31_0 = abce_ptr_low32(fence);
    pkt->ADDR_HI_UNION.addr_63_32 = abce_ptr_high32(fence);
    pkt->DATA_UNION.data = fence_value;
  }
}

// One COPY_LINEAR packet of |copy_size| bytes, at most the builder's
// max_copy_size, into a zeroed |pkt|.
static inline void abce_builder_form_copy_packet(const abce_builder_t* builder,
                                                 SDMA_PKT_COPY_LINEAR* pkt, void* dst,
                                                 const void* src, uint32_t copy_size) {
  pkt->HEADER_UNION.op = SDMA_OP_COPY;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
  if (builder->scope_fields) pkt->HEADER_UNION.npd = 1;

  if (builder->use_extended_count)
    pkt->COUNT_UNION.count_ext.count = copy_size - 1;
  else
    pkt->COUNT_UNION.count.count = copy_size - 1;

  if (builder->scope_fields) {
    pkt->PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
    pkt->PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;
  }

  pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = abce_ptr_low32(src);
  pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = abce_ptr_high32(src);
  pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = abce_ptr_low32(dst);
  pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = abce_ptr_high32(dst);
}

static inline void abce_builder_build_copy(const abce_builder_t* builder, char* cmd_addr, void* dst,
                                           const void* src, size_t size) {
  size_t cur_size = 0;
  while (cur_size < size) {
    const size_t remaining = size - cur_size;
    const uint32_t copy_size = (uint32_t)ABCE_MIN(remaining, builder->max_copy_size);
    abce_builder_form_copy_packet(builder, (SDMA_PKT_COPY_LINEAR*)cmd_addr, (char*)dst + cur_size,
                                  (const char*)src + cur_size, copy_size);
    cmd_addr += sizeof(SDMA_PKT_COPY_LINEAR);
    cur_size += copy_size;
  }
}

static inline void abce_builder_build_broadcast_copy(const abce_builder_t* builder, char* cmd_addr,
                                                     void* dst1, void* dst2, const void* src,
                                                     size_t size) {
  ABCE_ASSERT(((uintptr_t)dst1 & SDMA_PKT_COPY_LINEAR_BROADCAST_DST_ALIGN_MASK) ==
              ((uintptr_t)dst2 & SDMA_PKT_COPY_LINEAR_BROADCAST_DST_ALIGN_MASK));
  size_t cur_size = 0;
  while (cur_size < size) {
    const size_t remaining = size - cur_size;
    const uint32_t copy_size = (uint32_t)ABCE_MIN(remaining, builder->max_copy_size);

    void* cur_dst1 = (char*)dst1 + cur_size;
    void* cur_dst2 = (char*)dst2 + cur_size;
    const void* cur_src = (const char*)src + cur_size;

    SDMA_PKT_COPY_LINEAR_BROADCAST* pkt = (SDMA_PKT_COPY_LINEAR_BROADCAST*)cmd_addr;
    pkt->HEADER_UNION.op = SDMA_OP_COPY;
    pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_BROADCAST;
    pkt->HEADER_UNION.broadcast = 1;

    if (builder->use_extended_count)
      pkt->COUNT_UNION.count_ext.count = copy_size - 1;
    else
      pkt->COUNT_UNION.count.count = copy_size - 1;

    pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = abce_ptr_low32(cur_src);
    pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = abce_ptr_high32(cur_src);
    pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = abce_ptr_low32(cur_dst1);
    pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = abce_ptr_high32(cur_dst1);
    pkt->DST2_ADDR_LO_UNION.dst2_addr_31_0 = abce_ptr_low32(cur_dst2);
    pkt->DST2_ADDR_HI_UNION.dst2_addr_63_32 = abce_ptr_high32(cur_dst2);

    cmd_addr += sizeof(SDMA_PKT_COPY_LINEAR_BROADCAST);
    cur_size += copy_size;
  }
}

static inline void abce_builder_build_multicast_copy(const abce_builder_t* builder, char* cmd_addr,
                                                     void* const* dsts, uint32_t num_dsts,
                                                     const void* src, size_t size) {
  const size_t pkt_bytes = (5 + 2 * (size_t)num_dsts) * sizeof(uint32_t);
  size_t cur_size = 0;
  while (cur_size < size) {
    const size_t remaining = size - cur_size;
    const uint32_t copy_size = (uint32_t)ABCE_MIN(remaining, builder->max_copy_size);

    SDMA_PKT_COPY_LINEAR_MULTICAST_GFX125PLUS* pkt =
        (SDMA_PKT_COPY_LINEAR_MULTICAST_GFX125PLUS*)cmd_addr;
    pkt->HEADER_UNION.op = SDMA_OP_COPY;
    pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;
    pkt->COUNT_UNION.count = copy_size - 1;
    pkt->PARAMETER_UNION.num_of_destination = num_dsts - 1;
    pkt->PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
    pkt->PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;

    const void* cur_src = (const char*)src + cur_size;
    pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = abce_ptr_low32(cur_src);
    pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = abce_ptr_high32(cur_src);

    uint32_t* dst_dw = (uint32_t*)cmd_addr + 5;
    for (uint32_t dst_idx = 0; dst_idx < num_dsts; ++dst_idx) {
      const void* cur_dst = (const char*)dsts[dst_idx] + cur_size;
      dst_dw[dst_idx * 2] = abce_ptr_low32(cur_dst);
      dst_dw[dst_idx * 2 + 1] = abce_ptr_high32(cur_dst);
    }

    cmd_addr += pkt_bytes;
    cur_size += copy_size;
  }
}

//===----------------------------------------------------------------------===//
// Fused wait/copy/signal builders (gfx125plus, variable-length)
//
// These packets are COMPACTED: header (1 DW) + optional wait (7 DW) + body
// (varies per op kind) + optional signal (5 DW), packed contiguously. The
// engine reads each block immediately after the preceding one, keyed by the
// header's wait/signal bits. Absent blocks are not reserved.
//
// Each builder forms the packet in a zeroed scratch struct, then emits only the
// present DW blocks. They do not rely on the zeroed-buffer contract.
//===----------------------------------------------------------------------===//

// The wait and signal DW blocks are identical across every fused packet kind
// (same offsets, same fields), so they are formed once here from whichever
// packet type the caller is building.
#define ABCE_FORM_WAIT_DWORDS(tmpl, params, wait_dws)                             \
  do {                                                                            \
    (tmpl).WAIT_FUNCTION_UNION.wait_function = SDMA_FUNC_EQUAL;                   \
    (tmpl).WAIT_FUNCTION_UNION.wait_scope = SDMA_MEMORY_SCOPE_SYS;                \
    (tmpl).WAIT_ADDR_LO_UNION.wait_addr_31_3 = abce_ptr_low32((params)->wait_addr) >> 3; \
    (tmpl).WAIT_ADDR_HI_UNION.wait_addr_63_32 = abce_ptr_high32((params)->wait_addr);    \
    (tmpl).WAIT_REFERENCE_LO_UNION.wait_reference_31_0 = (uint32_t)(params)->wait_reference; \
    (tmpl).WAIT_REFERENCE_HI_UNION.wait_reference_63_32 =                         \
        (uint32_t)((params)->wait_reference >> 32);                               \
    (tmpl).WAIT_MASK_LO_UNION.wait_mask_31_0 = (uint32_t)(params)->wait_mask;      \
    (tmpl).WAIT_MASK_HI_UNION.wait_mask_63_32 = (uint32_t)((params)->wait_mask >> 32); \
    memcpy((wait_dws), (const uint32_t*)&(tmpl) + 1, 7 * sizeof(uint32_t));       \
  } while (0)

#define ABCE_FORM_SIGNAL_DWORDS(tmpl, params, signal_dws)                             \
  do {                                                                                \
    (tmpl).SIGNAL_OPERATION_UNION.signal_operation = SDMA_SIGNAL_OP_SUB64;            \
    (tmpl).SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;               \
    (tmpl).SIGNAL_ADDR_LO_UNION.signal_addr_31_3 =                                    \
        abce_ptr_low32((params)->signal_addr) >> 3;                                   \
    (tmpl).SIGNAL_ADDR_HI_UNION.signal_addr_63_32 =                                   \
        abce_ptr_high32((params)->signal_addr);                                       \
    (tmpl).SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;                                 \
    (tmpl).SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;                                \
    memcpy((signal_dws), (const uint32_t*)&(tmpl) + 14, 5 * sizeof(uint32_t));        \
  } while (0)

static inline void abce_builder_build_multicast_wait_signal_copy(
    const abce_builder_t* builder, char* cmd_addr, void* const* dsts, uint32_t num_dsts,
    const void* src, size_t size, const abce_wait_signal_params_t* params) {
  const bool do_wait = (params->wait_addr != NULL);
  const bool do_signal = (params->signal_addr != NULL);

  uint32_t wait_dws[7] = {0};
  uint32_t signal_dws[5] = {0};
  {
    SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    tmpl.HEADER_UNION.op = SDMA_OP_COPY;
    tmpl.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;
    if (do_wait) ABCE_FORM_WAIT_DWORDS(tmpl, params, wait_dws);
    if (do_signal) ABCE_FORM_SIGNAL_DWORDS(tmpl, params, signal_dws);
  }

  size_t cur_size = 0;
  while (cur_size < size) {
    const size_t remaining = size - cur_size;
    const uint32_t copy_size = (uint32_t)ABCE_MIN(remaining, builder->max_copy_size);
    const bool chunk_wait = do_wait && (!params->boundary_wait_signal || cur_size == 0);
    const bool chunk_signal =
        do_signal && (!params->boundary_wait_signal || cur_size + copy_size == size);

    uint32_t* out = (uint32_t*)cmd_addr;
    uint32_t num_dwords = 0;

    SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS header;
    memset(&header, 0, sizeof(header));
    header.HEADER_UNION.op = SDMA_OP_COPY;
    header.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_MULTICAST;
    header.HEADER_UNION.wait = chunk_wait ? 1 : 0;
    header.HEADER_UNION.signal = chunk_signal ? 1 : 0;
    out[num_dwords++] = ((const uint32_t*)&header)[0];
    if (chunk_wait) {
      memcpy(out + num_dwords, wait_dws, sizeof(wait_dws));
      num_dwords += 7;
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
    const char* cur_src = (const char*)src + cur_size;
    chunk.SRC_ADDR_LO_UNION.src_addr_31_0 = abce_ptr_low32(cur_src);
    chunk.SRC_ADDR_HI_UNION.src_addr_63_32 = abce_ptr_high32(cur_src);
    const uint32_t* chunk_words = (const uint32_t*)&chunk;
    out[num_dwords++] = chunk_words[8];
    out[num_dwords++] = chunk_words[9];
    out[num_dwords++] = chunk_words[10];
    out[num_dwords++] = chunk_words[11];

    for (uint32_t dst_idx = 0; dst_idx < num_dsts; ++dst_idx) {
      const char* cur_dst = (const char*)dsts[dst_idx] + cur_size;
      out[num_dwords++] = abce_ptr_low32(cur_dst);
      out[num_dwords++] = abce_ptr_high32(cur_dst);
    }
    if (chunk_signal) {
      memcpy(out + num_dwords, signal_dws, sizeof(signal_dws));
      num_dwords += 5;
    }

    cmd_addr += num_dwords * sizeof(uint32_t);
    cur_size += copy_size;
  }
}

static inline void abce_builder_build_swap_copy(const abce_builder_t* builder, char* cmd_addr,
                                                void* addr_a, void* addr_b, size_t size) {
  ABCE_STATIC_ASSERT(sizeof(SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS) ==
                         sizeof(SDMA_PKT_COPY_LINEAR_SWAP),
                     "gfx125plus swap packet must match legacy swap packet size for shared stride");
  const bool use_gfx125plus = builder->scope_fields && builder->is_gfx125plus;

  const size_t alignment = use_gfx125plus ? SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS_ALIGNMENT
                                          : SDMA_PKT_COPY_LINEAR_SWAP_ALIGNMENT;
  (void)alignment;
  ABCE_ASSERT(((uintptr_t)addr_a & (alignment - 1)) == 0);
  ABCE_ASSERT(((uintptr_t)addr_b & (alignment - 1)) == 0);

  const size_t max_copy_size = use_gfx125plus ? SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS_MAX_SIZE
                                              : SDMA_PKT_COPY_LINEAR_SWAP_MAX_SIZE;
  size_t cur_size = 0;
  while (cur_size < size) {
    const size_t remaining = size - cur_size;
    const uint32_t copy_size = (uint32_t)ABCE_MIN(remaining, max_copy_size);

    void* cur_addr_a = (char*)addr_a + cur_size;
    void* cur_addr_b = (char*)addr_b + cur_size;

    if (use_gfx125plus) {
      SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS* pkt =
          (SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS*)cmd_addr;
      pkt->HEADER_UNION.op = SDMA_OP_COPY;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
      pkt->COUNT_UNION.count = copy_size - 1;
      pkt->PARAMETER_UNION.scope_a = SDMA_MEMORY_SCOPE_SYS;
      pkt->PARAMETER_UNION.scope_b = SDMA_MEMORY_SCOPE_SYS;
      pkt->ADDR_A_LO_UNION.DW_3_DATA = abce_ptr_low32(cur_addr_a);
      pkt->ADDR_A_HI_UNION.addr_a_63_32 = abce_ptr_high32(cur_addr_a);
      pkt->ADDR_B_LO_UNION.DW_5_DATA = abce_ptr_low32(cur_addr_b);
      pkt->ADDR_B_HI_UNION.addr_b_63_32 = abce_ptr_high32(cur_addr_b);
    } else {
      SDMA_PKT_COPY_LINEAR_SWAP* pkt = (SDMA_PKT_COPY_LINEAR_SWAP*)cmd_addr;
      pkt->HEADER_UNION.op = SDMA_OP_COPY;
      pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
      pkt->COUNT_UNION.count = copy_size - 1;
      pkt->ADDR_A_LO_UNION.DW_3_DATA = abce_ptr_low32(cur_addr_a);
      pkt->ADDR_A_HI_UNION.addr_a_63_32 = abce_ptr_high32(cur_addr_a);
      pkt->ADDR_B_LO_UNION.DW_5_DATA = abce_ptr_low32(cur_addr_b);
      pkt->ADDR_B_HI_UNION.addr_b_63_32 = abce_ptr_high32(cur_addr_b);
    }

    cmd_addr += sizeof(SDMA_PKT_COPY_LINEAR_SWAP);
    cur_size += copy_size;
  }
}

//===----------------------------------------------------------------------===//
// Rect (2D/3D sub-window) copy
//===----------------------------------------------------------------------===//

// Called per tile to obtain packet storage; must return a pointer to at least
// |packet_bytes| bytes of writable, ZEROED memory. This replaces the C++
// builder's AppendFn template parameter.
typedef void* (*abce_rect_append_fn_t)(void* user_data, size_t packet_bytes);

static inline int abce_max_aligned_element(size_t width) {
  return abce_count_trailing_zeros_u64((uint64_t)width | 16ull);
}

// Builds linear sub-window (rect) copy packets.
//
// Returns ABCE_STATUS_RECT_OUT_OF_RANGE when the pitch or slice exceeds what
// the packet can encode; the C++ builder threw std::invalid_argument here.
static inline abce_status_t abce_builder_build_copy_rect(
    const abce_builder_t* builder, abce_rect_append_fn_t append, void* user_data,
    const abce_pitched_ptr_t* dst, const abce_dim3_t* dst_offset, const abce_pitched_ptr_t* src,
    const abce_dim3_t* src_offset, const abce_dim3_t* range) {
  const bool is_gfx12plus = (builder->isa.major >= 12);

  const uint32_t max_pitch =
      1u << (is_gfx12plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12_PITCH_BITS
                          : SDMA_PKT_COPY_LINEAR_RECT_PITCH_BITS);
  const uint64_t max_slice =
      1ull << (is_gfx12plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12_SLICE_BITS
                            : SDMA_PKT_COPY_LINEAR_RECT_SLICE_BITS);
  const uint32_t max_x =
      1u << (is_gfx12plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12_RECT_XY_BITS
                          : SDMA_PKT_COPY_LINEAR_RECT_RECT_XY_BITS);
  const uint32_t max_y = max_x;
  const uint32_t max_z =
      1u << (is_gfx12plus ? SDMA_PKT_COPY_LINEAR_RECT_GFX12_RECT_Z_BITS
                          : SDMA_PKT_COPY_LINEAR_RECT_RECT_Z_BITS);

  const int src_pitch_ele = abce_max_aligned_element(src->pitch);
  const int dst_pitch_ele = abce_max_aligned_element(dst->pitch);
  int max_ele = ABCE_MIN(src_pitch_ele, dst_pitch_ele);
  if (range->z != 1) {
    const int src_slice_ele = abce_max_aligned_element(src->slice);
    const int dst_slice_ele = abce_max_aligned_element(dst->slice);
    max_ele = ABCE_MIN(max_ele, ABCE_MIN(src_slice_ele, dst_slice_ele));
  }

  const int range_x_ele = abce_max_aligned_element(range->x);
  const int src_off_ele = abce_max_aligned_element(src_offset->x % 4);
  const int dst_off_ele = abce_max_aligned_element(dst_offset->x % 4);
  const int min_ele = ABCE_MIN(ABCE_MIN(max_ele, range_x_ele), ABCE_MIN(src_off_ele, dst_off_ele));

  if ((src->pitch >> min_ele) > max_pitch || (dst->pitch >> min_ele) > max_pitch)
    return ABCE_STATUS_RECT_OUT_OF_RANGE;
  if (range->z != 1) {
    if ((src->slice >> min_ele) > max_slice || (dst->slice >> min_ele) > max_slice)
      return ABCE_STATUS_RECT_OUT_OF_RANGE;
  }

  for (uint32_t z = 0; z < range->z; z += max_z) {
    for (uint32_t y = 0; y < range->y; y += max_y) {
      uint32_t x = 0;
      while (x < range->x) {
        const uint32_t width = range->x - x;

        const int src_tile_ele = abce_max_aligned_element((src_offset->x + x) % 4);
        const int dst_tile_ele = abce_max_aligned_element((dst_offset->x + x) % 4);
        const int aligned_ele = ABCE_MIN(ABCE_MIN(src_tile_ele, dst_tile_ele), max_ele);

        const int width_ele = abce_max_aligned_element(width);
        int element = ABCE_MIN(width_ele, aligned_ele);
        uint32_t xcount = width >> element;

        if (xcount > max_x) {
          element = aligned_ele;
          const uint32_t shifted = width >> element;
          xcount = ABCE_MIN(shifted, max_x);
        }

        uintptr_t sbase = (uintptr_t)src->base + src_offset->x + x +
                          (src_offset->y + y) * src->pitch + (src_offset->z + z) * src->slice;
        uintptr_t dbase = (uintptr_t)dst->base + dst_offset->x + x +
                          (dst_offset->y + y) * dst->pitch + (dst_offset->z + z) * dst->slice;
        const uint32_t soff = (uint32_t)((sbase % 4) >> element);
        const uint32_t doff = (uint32_t)((dbase % 4) >> element);
        sbase &= ~3ull;
        dbase &= ~3ull;

        x += xcount << element;

        const uint32_t rect_y = ABCE_MIN(range->y - y, max_y) - 1;
        const uint32_t rect_z = ABCE_MIN(range->z - z, max_z) - 1;

        if (is_gfx12plus) {
          SDMA_PKT_COPY_LINEAR_RECT_GFX12* pkt =
              (SDMA_PKT_COPY_LINEAR_RECT_GFX12*)append(user_data, sizeof(SDMA_PKT_COPY_LINEAR_RECT));
          pkt->HEADER_UNION.op = SDMA_OP_COPY;
          pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_RECT;
          if (builder->scope_fields) pkt->HEADER_UNION.npd = 1;
          pkt->HEADER_UNION.element = element;
          pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = (uint32_t)sbase;
          pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = (uint32_t)(sbase >> 32);
          pkt->SRC_PARAMETER_1_UNION.src_offset_x = soff;
          pkt->SRC_PARAMETER_2_UNION.src_pitch = (uint32_t)(src->pitch >> element) - 1;
          pkt->SRC_PARAMETER_3_UNION.src_slice_pitch =
              (range->z == 1) ? 0 : (uint32_t)(src->slice >> element) - 1;
          pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = (uint32_t)dbase;
          pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = (uint32_t)(dbase >> 32);
          pkt->DST_PARAMETER_1_UNION.dst_offset_x = doff;
          pkt->DST_PARAMETER_2_UNION.dst_pitch = (uint32_t)(dst->pitch >> element) - 1;
          pkt->DST_PARAMETER_3_UNION.dst_slice_pitch =
              (range->z == 1) ? 0 : (uint32_t)(dst->slice >> element) - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_x = xcount - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_y = rect_y;
          pkt->RECT_PARAMETER_2_UNION.gfx12.rect_z = rect_z;
          if (builder->scope_fields) {
            pkt->RECT_PARAMETER_2_UNION.gfx125plus.dst_scope = SDMA_MEMORY_SCOPE_SYS;
            pkt->RECT_PARAMETER_2_UNION.gfx125plus.src_scope = SDMA_MEMORY_SCOPE_SYS;
          }
        } else {
          SDMA_PKT_COPY_LINEAR_RECT* pkt =
              (SDMA_PKT_COPY_LINEAR_RECT*)append(user_data, sizeof(SDMA_PKT_COPY_LINEAR_RECT));
          pkt->HEADER_UNION.op = SDMA_OP_COPY;
          pkt->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_RECT;
          pkt->HEADER_UNION.element = element;
          pkt->SRC_ADDR_LO_UNION.src_addr_31_0 = (uint32_t)sbase;
          pkt->SRC_ADDR_HI_UNION.src_addr_63_32 = (uint32_t)(sbase >> 32);
          pkt->SRC_PARAMETER_1_UNION.src_offset_x = soff;
          pkt->SRC_PARAMETER_2_UNION.src_pitch = (uint32_t)(src->pitch >> element) - 1;
          pkt->SRC_PARAMETER_3_UNION.src_slice_pitch =
              (range->z == 1) ? 0 : (uint32_t)(src->slice >> element) - 1;
          pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = (uint32_t)dbase;
          pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = (uint32_t)(dbase >> 32);
          pkt->DST_PARAMETER_1_UNION.dst_offset_x = doff;
          pkt->DST_PARAMETER_2_UNION.dst_pitch = (uint32_t)(dst->pitch >> element) - 1;
          pkt->DST_PARAMETER_3_UNION.dst_slice_pitch =
              (range->z == 1) ? 0 : (uint32_t)(dst->slice >> element) - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_x = xcount - 1;
          pkt->RECT_PARAMETER_1_UNION.rect_y = rect_y;
          pkt->RECT_PARAMETER_2_UNION.rect_z = rect_z;
        }
      }
    }
  }
  return ABCE_STATUS_OK;
}

// Counting append used by abce_builder_num_rect_packets(): hands back one
// throwaway packet's worth of scratch and tallies the calls.
typedef struct abce_rect_count_state_t {
  uint32_t num_packets;
  ABCE_ALIGNAS(16) char scratch[256];
} abce_rect_count_state_t;

static inline void* abce_rect_count_append(void* user_data, size_t packet_bytes) {
  abce_rect_count_state_t* state = (abce_rect_count_state_t*)user_data;
  (void)packet_bytes;
  ++state->num_packets;
  return state->scratch;
}

// Number of rect (2D/3D sub-window) copy packets.
//
// Rect tiling is data dependent (pitch/alignment driven), so this replays the
// exact same loop as abce_builder_build_copy_rect() with a counting append —
// the two can never diverge. Fails the same way the emit does.
static inline abce_status_t abce_builder_num_rect_packets(
    const abce_builder_t* builder, const abce_pitched_ptr_t* dst, const abce_dim3_t* dst_offset,
    const abce_pitched_ptr_t* src, const abce_dim3_t* src_offset, const abce_dim3_t* range,
    uint32_t* out_num_packets) {
  abce_rect_count_state_t state;
  state.num_packets = 0;
  const abce_status_t status = abce_builder_build_copy_rect(
      builder, abce_rect_count_append, &state, dst, dst_offset, src, src_offset, range);
  *out_num_packets = state.num_packets;
  return status;
}

//===----------------------------------------------------------------------===//
// Remaining fixed-layout builders
//===----------------------------------------------------------------------===//

static inline void abce_builder_build_fill(const abce_builder_t* builder, char* cmd_addr, void* ptr,
                                           uint32_t value, size_t count) {
  char* cur_ptr = (char*)ptr;
  SDMA_PKT_CONSTANT_FILL* pkt = (SDMA_PKT_CONSTANT_FILL*)cmd_addr;

  while (count > 0) {
    const uint32_t fill_count = (uint32_t)ABCE_MIN(count, (size_t)builder->max_fill_dwords);

    pkt->HEADER_UNION.op = SDMA_OP_CONST_FILL;
    if (builder->scope_fields) {
      pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
      pkt->HEADER_UNION.npd = 1;
    }
    pkt->HEADER_UNION.fillsize = 2;

    pkt->DST_ADDR_LO_UNION.dst_addr_31_0 = abce_ptr_low32(cur_ptr);
    pkt->DST_ADDR_HI_UNION.dst_addr_63_32 = abce_ptr_high32(cur_ptr);
    pkt->DATA_UNION.src_data_31_0 = value;
    pkt->COUNT_UNION.count = (fill_count - 1) * sizeof(uint32_t);

    pkt++;
    cur_ptr += fill_count * sizeof(uint32_t);
    count -= fill_count;
  }
}

static inline void abce_builder_build_poll(const abce_builder_t* builder, char* cmd_addr,
                                           void* addr, uint32_t reference, uint32_t mask) {
  SDMA_PKT_POLL_REGMEM* pkt = (SDMA_PKT_POLL_REGMEM*)cmd_addr;
  pkt->HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
  pkt->HEADER_UNION.mem_poll = 1;
  pkt->HEADER_UNION.func = SDMA_FUNC_EQUAL;
  pkt->ADDR_LO_UNION.addr_31_0 = abce_ptr_low32(addr);
  pkt->ADDR_HI_UNION.addr_63_32 = abce_ptr_high32(addr);
  pkt->VALUE_UNION.value = reference;
  pkt->MASK_UNION.mask = mask;
  pkt->DW5_UNION.interval = 0x04;
  pkt->DW5_UNION.retry_count = 0xfff;
  if (builder->scope_fields) pkt->DW5_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
}

static inline void abce_builder_build_poll_64b(const abce_builder_t* builder, char* cmd_addr,
                                               void* addr, uint64_t reference, uint64_t mask) {
  SDMA_PKT_POLL_MEM_64B_GFX125PLUS* pkt = (SDMA_PKT_POLL_MEM_64B_GFX125PLUS*)cmd_addr;
  pkt->HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_POLL_MEM_64B;
  pkt->HEADER_UNION.func = SDMA_FUNC_EQUAL;
  pkt->ADDR_LO_UNION.addr_31_3 = abce_ptr_low32(addr) >> 3;
  pkt->ADDR_HI_UNION.addr_63_32 = abce_ptr_high32(addr);
  pkt->REFERENCE_LO_UNION.reference_31_0 = (uint32_t)reference;
  pkt->REFERENCE_HI_UNION.reference_63_32 = (uint32_t)(reference >> 32);
  pkt->MASK_LO_UNION.mask_31_0 = (uint32_t)mask;
  pkt->MASK_HI_UNION.mask_63_32 = (uint32_t)(mask >> 32);
  pkt->HEADER_UNION.sys = 1;
  pkt->DW7_UNION.retry_count = 0;
  if (builder->scope_fields) pkt->DW7_UNION.scope = SDMA_MEMORY_SCOPE_SYS;
}

static inline void abce_builder_build_fence_64b(const abce_builder_t* builder, char* cmd_addr,
                                                void* fence_addr, uint64_t fence_value) {
  SDMA_PKT_FENCE_64B_GFX125PLUS* pkt = (SDMA_PKT_FENCE_64B_GFX125PLUS*)cmd_addr;
  pkt->HEADER_UNION.op = SDMA_OP_FENCE;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_FENCE_64B;
  pkt->HEADER_UNION.mtype = 3;
  pkt->HEADER_UNION.sys = 1;
  if (builder->scope_fields) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

  pkt->ADDR_LO_UNION.addr_31_3 = abce_ptr_low32(fence_addr) >> 3;
  pkt->ADDR_HI_UNION.addr_63_32 = abce_ptr_high32(fence_addr);
  pkt->DATA_LO_UNION.data_31_0 = (uint32_t)fence_value;
  pkt->DATA_HI_UNION.data_63_32 = (uint32_t)(fence_value >> 32);
}

static inline void abce_builder_build_wait_signal_copy(const abce_builder_t* builder,
                                                       char* cmd_addr, void* dst, const void* src,
                                                       size_t size,
                                                       const abce_wait_signal_params_t* params) {
  const bool do_wait = (params->wait_addr != NULL);
  const bool do_signal = (params->signal_addr != NULL);

  uint32_t wait_dws[7] = {0};
  uint32_t signal_dws[5] = {0};
  {
    SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    tmpl.HEADER_UNION.op = SDMA_OP_COPY;
    tmpl.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
    tmpl.COPY_PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
    tmpl.COPY_PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;
    if (do_wait) ABCE_FORM_WAIT_DWORDS(tmpl, params, wait_dws);
    if (do_signal) ABCE_FORM_SIGNAL_DWORDS(tmpl, params, signal_dws);
  }

  size_t cur_size = 0;
  while (cur_size < size) {
    const size_t remaining = size - cur_size;
    const uint32_t copy_size = (uint32_t)ABCE_MIN(remaining, builder->max_copy_size);
    const bool chunk_wait = do_wait && (!params->boundary_wait_signal || cur_size == 0);
    const bool chunk_signal =
        do_signal && (!params->boundary_wait_signal || cur_size + copy_size == size);

    uint32_t* out = (uint32_t*)cmd_addr;
    uint32_t num_dwords = 0;

    SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS header;
    memset(&header, 0, sizeof(header));
    header.HEADER_UNION.op = SDMA_OP_COPY;
    header.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
    header.HEADER_UNION.wait = chunk_wait ? 1 : 0;
    header.HEADER_UNION.signal = chunk_signal ? 1 : 0;
    out[num_dwords++] = ((const uint32_t*)&header)[0];
    if (chunk_wait) {
      memcpy(out + num_dwords, wait_dws, sizeof(wait_dws));
      num_dwords += 7;
    }

    const char* cur_src = (const char*)src + cur_size;
    char* cur_dst = (char*)dst + cur_size;
    SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS body;
    body.COPY_COUNT_UNION.DW_8_DATA = 0;
    body.COPY_COUNT_UNION.copy_count = copy_size - 1;
    body.COPY_PARAMETER_UNION.DW_9_DATA = 0;
    body.COPY_PARAMETER_UNION.dst_scope = SDMA_MEMORY_SCOPE_SYS;
    body.COPY_PARAMETER_UNION.src_scope = SDMA_MEMORY_SCOPE_SYS;
    body.SRC_ADDR_LO_UNION.src_addr_31_0 = abce_ptr_low32(cur_src);
    body.SRC_ADDR_HI_UNION.src_addr_63_32 = abce_ptr_high32(cur_src);
    body.DST_ADDR_LO_UNION.dst_addr_31_0 = abce_ptr_low32(cur_dst);
    body.DST_ADDR_HI_UNION.dst_addr_63_32 = abce_ptr_high32(cur_dst);
    memcpy(out + num_dwords, (const uint32_t*)&body + 8, 6 * sizeof(uint32_t));
    num_dwords += 6;

    if (chunk_signal) {
      memcpy(out + num_dwords, signal_dws, sizeof(signal_dws));
      num_dwords += 5;
    }

    cmd_addr += num_dwords * sizeof(uint32_t);
    cur_size += copy_size;
  }
}

static inline void abce_builder_build_wait_signal_indirect_copy(
    const abce_builder_t* builder, char* cmd_addr, void* dst, const void* src, size_t size,
    bool indirect_src, bool indirect_dst, const abce_wait_signal_params_t* params) {
  (void)builder;
  const bool do_wait = (params->wait_addr != NULL);
  const bool do_signal = (params->signal_addr != NULL);

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
    tmpl.WAIT_ADDR_LO_UNION.wait_addr_31_3 = abce_ptr_low32(params->wait_addr) >> 3;
    tmpl.WAIT_ADDR_HI_UNION.wait_addr_63_32 = abce_ptr_high32(params->wait_addr);
    tmpl.WAIT_REFERENCE_LO_UNION.wait_reference_31_0 = (uint32_t)params->wait_reference;
    tmpl.WAIT_REFERENCE_HI_UNION.wait_reference_63_32 = (uint32_t)(params->wait_reference >> 32);
    tmpl.WAIT_MASK_LO_UNION.wait_mask_31_0 = (uint32_t)params->wait_mask;
    tmpl.WAIT_MASK_HI_UNION.wait_mask_63_32 = (uint32_t)(params->wait_mask >> 32);
  }

  tmpl.COPY_COUNT_UNION.copy_count = (uint32_t)size - 1;
  tmpl.COPY_PARAMETER_UNION.copy_dst_scope = SDMA_MEMORY_SCOPE_SYS;
  tmpl.COPY_PARAMETER_UNION.copy_src_scope = SDMA_MEMORY_SCOPE_SYS;
  tmpl.COPY_PARAMETER_UNION.indirect_addr_scope = SDMA_MEMORY_SCOPE_SYS;
  tmpl.SRC_ADDR_LO_UNION.copy_src_addr_31_0 = abce_ptr_low32(src);
  tmpl.SRC_ADDR_HI_UNION.copy_src_addr_63_32 = abce_ptr_high32(src);
  tmpl.DST_ADDR_LO_UNION.copy_dst_addr_31_0 = abce_ptr_low32(dst);
  tmpl.DST_ADDR_HI_UNION.copy_dst_addr_63_32 = abce_ptr_high32(dst);

  if (do_signal) {
    tmpl.SIGNAL_OPERATION_UNION.signal_operation = SDMA_SIGNAL_OP_SUB64;
    tmpl.SIGNAL_OPERATION_UNION.signal_scope = SDMA_MEMORY_SCOPE_SYS;
    tmpl.SIGNAL_ADDR_LO_UNION.signal_addr_31_3 = abce_ptr_low32(params->signal_addr) >> 3;
    tmpl.SIGNAL_ADDR_HI_UNION.signal_addr_63_32 = abce_ptr_high32(params->signal_addr);
    tmpl.SIGNAL_DATA_LO_UNION.signal_data_31_0 = 1;
    tmpl.SIGNAL_DATA_HI_UNION.signal_data_63_32 = 0;
  }

  // Emit only the present DW blocks contiguously: header, optional wait, the
  // 6-DW copy block (DW8-13), optional signal.
  const uint32_t* tmpl_words = (const uint32_t*)&tmpl;
  uint32_t* out = (uint32_t*)cmd_addr;
  uint32_t num_dwords = 0;
  out[num_dwords++] = tmpl_words[0];
  if (do_wait) {
    memcpy(out + num_dwords, tmpl_words + 1, 7 * sizeof(uint32_t));
    num_dwords += 7;
  }
  memcpy(out + num_dwords, tmpl_words + 8, 6 * sizeof(uint32_t));
  num_dwords += 6;
  if (do_signal) {
    memcpy(out + num_dwords, tmpl_words + 14, 5 * sizeof(uint32_t));
    num_dwords += 5;
  }
}

static inline void abce_builder_build_wait_signal_swap(const abce_builder_t* builder,
                                                       char* cmd_addr, void* addr_a, void* addr_b,
                                                       size_t size_a, size_t size_b,
                                                       const abce_wait_signal_params_t* params) {
  (void)builder;
  const bool do_wait = (params->wait_addr != NULL);
  const bool do_signal = (params->signal_addr != NULL);
  const size_t max_copy_size = SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS_MAX_SIZE;

  uint32_t wait_dws[7] = {0};
  uint32_t signal_dws[5] = {0};
  {
    SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    tmpl.HEADER_UNION.op = SDMA_OP_COPY;
    tmpl.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
    if (do_wait) ABCE_FORM_WAIT_DWORDS(tmpl, params, wait_dws);
    if (do_signal) ABCE_FORM_SIGNAL_DWORDS(tmpl, params, signal_dws);
  }

  size_t cur_a = 0;
  size_t cur_b = 0;
  while (cur_a < size_a || cur_b < size_b) {
    const size_t remaining_a = size_a - cur_a;
    const size_t remaining_b = size_b - cur_b;
    const uint32_t chunk_a = (uint32_t)ABCE_MIN(remaining_a, max_copy_size);
    const uint32_t chunk_b = (uint32_t)ABCE_MIN(remaining_b, max_copy_size);
    // The single COUNT field is driven by the larger remaining side so it is
    // never 0 while the loop runs (an exhausted side has chunk == 0, which
    // would underflow to UINT32_MAX). For a symmetric swap chunk_a == chunk_b.
    const uint32_t chunk = ABCE_MAX(chunk_a, chunk_b);
    const bool chunk_wait =
        do_wait && (!params->boundary_wait_signal || (cur_a == 0 && cur_b == 0));
    const bool chunk_signal =
        do_signal && (!params->boundary_wait_signal ||
                      (cur_a + chunk_a == size_a && cur_b + chunk_b == size_b));

    const char* ptr_a = (const char*)addr_a + cur_a;
    const char* ptr_b = (const char*)addr_b + cur_b;

    uint32_t* out = (uint32_t*)cmd_addr;
    uint32_t num_dwords = 0;

    SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS header;
    memset(&header, 0, sizeof(header));
    header.HEADER_UNION.op = SDMA_OP_COPY;
    header.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_SWAP;
    header.HEADER_UNION.wait = chunk_wait ? 1 : 0;
    header.HEADER_UNION.signal = chunk_signal ? 1 : 0;
    out[num_dwords++] = ((const uint32_t*)&header)[0];
    if (chunk_wait) {
      memcpy(out + num_dwords, wait_dws, sizeof(wait_dws));
      num_dwords += 7;
    }

    SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS body;
    body.COUNT_UNION.DW_8_DATA = 0;
    body.COUNT_UNION.count = chunk - 1;
    body.COPY_PARAMETER_UNION.DW_9_DATA = 0;
    body.COPY_PARAMETER_UNION.scope_a = SDMA_MEMORY_SCOPE_SYS;
    body.COPY_PARAMETER_UNION.scope_b = SDMA_MEMORY_SCOPE_SYS;
    body.ADDR_A_LO_UNION.addr_a_31_0 = abce_ptr_low32(ptr_a);
    body.ADDR_A_HI_UNION.addr_a_63_32 = abce_ptr_high32(ptr_a);
    body.ADDR_B_LO_UNION.addr_b_31_0 = abce_ptr_low32(ptr_b);
    body.ADDR_B_HI_UNION.addr_b_63_32 = abce_ptr_high32(ptr_b);
    memcpy(out + num_dwords, (const uint32_t*)&body + 8, 6 * sizeof(uint32_t));
    num_dwords += 6;

    if (chunk_signal) {
      memcpy(out + num_dwords, signal_dws, sizeof(signal_dws));
      num_dwords += 5;
    }

    cmd_addr += num_dwords * sizeof(uint32_t);
    cur_a += chunk_a;
    cur_b += chunk_b;
  }
}

static inline void abce_builder_build_atomic_add(const abce_builder_t* builder, char* cmd_addr,
                                                 void* addr, uint64_t value) {
  SDMA_PKT_ATOMIC* pkt = (SDMA_PKT_ATOMIC*)cmd_addr;
  pkt->HEADER_UNION.op = SDMA_OP_ATOMIC;
  pkt->HEADER_UNION.operation = SDMA_ATOMIC_ADD64;
  if (builder->scope_fields) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

  pkt->ADDR_LO_UNION.addr_31_0 = abce_ptr_low32(addr);
  pkt->ADDR_HI_UNION.addr_63_32 = abce_ptr_high32(addr);
  pkt->SRC_DATA_LO_UNION.src_data_31_0 = (uint32_t)value;
  pkt->SRC_DATA_HI_UNION.src_data_63_32 = (uint32_t)(value >> 32);
}

static inline void abce_builder_build_atomic_decrement(const abce_builder_t* builder,
                                                       char* cmd_addr, void* addr) {
  abce_builder_build_atomic_add(builder, cmd_addr, addr, UINT64_MAX);
}

static inline void abce_builder_build_get_global_timestamp(const abce_builder_t* builder,
                                                           char* cmd_addr, void* write_address) {
  SDMA_PKT_TIMESTAMP* pkt = (SDMA_PKT_TIMESTAMP*)cmd_addr;
  pkt->HEADER_UNION.op = SDMA_OP_TIMESTAMP;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_TIMESTAMP_GET_GLOBAL;
  if (builder->scope_fields) pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

  pkt->ADDR_LO_UNION.addr_31_0 = abce_ptr_low32(write_address);
  pkt->ADDR_HI_UNION.addr_63_32 = abce_ptr_high32(write_address);
}

static inline void abce_builder_build_trap(const abce_builder_t* builder, char* cmd_addr,
                                           uint32_t event_id) {
  (void)builder;
  SDMA_PKT_TRAP* pkt = (SDMA_PKT_TRAP*)cmd_addr;
  pkt->HEADER_UNION.op = SDMA_OP_TRAP;
  pkt->INT_CONTEXT_UNION.int_ctx = event_id;
}

static inline void abce_builder_build_hdp_flush(const abce_builder_t* builder, char* cmd_addr) {
  (void)builder;
  ABCE_ASSERT(cmd_addr != NULL);
  memcpy(cmd_addr, &abce_sdma_hdp_flush_cmd, sizeof(abce_sdma_hdp_flush_cmd));
}

static inline void abce_builder_build_gcr(const abce_builder_t* builder, char* cmd_addr,
                                          bool invalidate) {
  ABCE_ASSERT(cmd_addr != NULL);

  if (builder->is_gfx125plus) {
    SDMA_PKT_GCR_GFX125PLUS* pkt = (SDMA_PKT_GCR_GFX125PLUS*)cmd_addr;
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
    SDMA_PKT_GCR* pkt = (SDMA_PKT_GCR*)cmd_addr;
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_BUILDER_H_
