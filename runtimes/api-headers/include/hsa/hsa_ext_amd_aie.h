/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// HSA AMD extension for AIE agents.

#ifndef HSA_RUNTIME_EXT_AMD_AIE_H_
#define HSA_RUNTIME_EXT_AMD_AIE_H_

#include "hsa.h"

/**
 * - 1.0 - initial version
 */
#define HSA_AMD_AIE_INTERFACE_VERSION_MAJOR 1
#define HSA_AMD_AIE_INTERFACE_VERSION_MINOR 0

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup aql Architected Queuing Language
 *  @{
 */

/**
 * @brief HSA AIE packet type.
 */
typedef enum {
  /**
   * The packet is ready to be processed by the packet processor.
   */
  HSA_AMD_AIE_PACKET_TYPE_READY = 0,

  /**
   * The packet has been processed in the past, but has not been reassigned to
   * the packet processor. A packet processor must not process a packet of this
   * type.
   */
  HSA_AMD_AIE_PACKET_TYPE_INVALID = 1,
} hsa_amd_aie_packet_type_t;

/**
 * @brief AMD AIE packet opcode.
 */
typedef enum {
  /**
   * AIE KMQ packet, for aie2 and aie2p AIE agents.
   */
  HSA_AMD_AIE_PACKET_OPCODE_KMQ = 0,

  /**
   * AIE UMQ packet, for aie4 AIE agents.
   */
  HSA_AMD_AIE_PACKET_OPCODE_UMQ = 1,
} hsa_amd_aie_packet_opcode_t;

/**
 * @brief AMD AIE agent kernel dispatch packet.
 *
 * One layout serves both queue types; ::opcode names the one the packet is for.
 *
 * On aie2 and aie2p (::HSA_AMD_AIE_PACKET_OPCODE_KMQ) the runtime reads the packet
 * and submits it through the kernel driver; the device never reads it.
 *
 * On aie4 (::HSA_AMD_AIE_PACKET_OPCODE_UMQ) the queue's ring is read in place by
 * the device's command processor, so the packet is laid out over the device's own
 * packet: the fields the application fills are where the device does not read,
 * and the reserved ones are where it does. ::reserved2 is filled in when the
 * packet is submitted, and ::kernel_object_low / ::kernel_object_high are
 * replaced with the address of the kernel's control code; the application must
 * not rely on either keeping the value it wrote.
 */
typedef struct hsa_amd_aie_kernel_dispatch_packet_s {
  union {
    struct {
      /**
       * Packet header. Used to configure multiple packet parameters such as the
       * packet type. The parameters are described by ::hsa_packet_header_t.
       */
      uint16_t header;

      /**
       * Packet opcode. The parameters are described by ::hsa_amd_aie_packet_opcode_t.
       */
      uint16_t opcode;
    };
    uint32_t full_header;
  };

  /**
   * Number of bytes in the packet after ::reserved2 and up to ::completion_signal. Must be 24.
   */
  uint16_t count;

  /**
   * Reserved. Must be 0.
   */
  uint8_t reserved0;

  /**
   * Reserved. Must be 0.
   */
  uint8_t reserved1;

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved3;

  /**
   * Opaque kernel object handle, as returned by ::HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT for a
   * kernel symbol of an executable loaded on the agent that owns the queue. The executable must
   * not be destroyed until the dispatch has completed.
   *
   * The kernel's code object selects the dispatch type: PDI plus instruction sequence, or full
   * ELF. Full ELF is supported only on aie2p and aie4 agents, and is the only type on aie4.
   *
   * On aie2p, the packets submitted by one doorbell ring fix the dispatch type for the batch,
   * and a packet of the other type in the same batch is rejected. A later batch on the same
   * queue may use the other type, at the cost of a hardware context rebuild.
   */
  uint32_t kernel_object_low;
  uint32_t kernel_object_high;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved4;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved5;

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved6;

  /**
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.
   */
  hsa_signal_t completion_signal;

  /**
   * Pointer to a buffer containing the kernel arguments. May be NULL only when num_kernargs is 0.
   *
   * The buffer must be allocated using ::hsa_memory_allocate, and must not be
   * modified once the kernel dispatch packet is enqueued until the dispatch has
   * completed execution.
   *
   * The buffer must contain exactly 2 * ::num_kernargs consecutive `uint64_t` entries:
   * - entries [0 .. ::num_kernargs - 1] are the argument addresses
   * - entries [::num_kernargs .. 2 * ::num_kernargs - 1] are the corresponding argument sizes in
   * bytes
   *
   * On aie2 and aie2p, a size of 0 for a kernarg means no cacheline is flushed for that argument;
   * the argument is still resolved and kept resident for the duration of the dispatch. aie4 is
   * cache-coherent and does not use the sizes.
   */
  void* kernarg_address;

  /**
   * Number of kernel arguments. Must be 0 if ::kernarg_address is NULL, and must be greater than 0
   * if ::kernarg_address is not NULL.
   */
  uint16_t num_kernargs;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved7;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved8;
} hsa_amd_aie_kernel_dispatch_packet_t;

/** @} */

#ifdef __cplusplus
}  // end extern "C" block
#endif

#endif  // header guard
