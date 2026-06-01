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
   * AIE KMQ packet.
   */
  HSA_AMD_AIE_PACKET_OPCODE_KMQ = 0,
} hsa_amd_aie_packet_opcode_t;

/**
 * @brief AMD AIE agent kernel dispatch packet.
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
   * Number of bytes in the packet after the ::completion_signal and up to ::kernarg_address. Must
   * be 24.
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
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.​
   */
  hsa_signal_t completion_signal;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved2;

  /**
   * Address of the instruction sequence.
   */
  uint32_t insts_addr_low;
  uint32_t insts_addr_high;

  /**
   * Number of kernel arguments. Must be 0 if ::kernarg_address is NULL, and must be greater than 0
   * if ::kernarg_address is not NULL.
   */
  uint16_t num_kernargs;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved3;

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
   */
  void* kernarg_address;

  /**
   * Size of the instruction sequence in bytes.
   */
  uint64_t insts_size;

  /**
   * PDI address.
   */
  void* pdi_addr;

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved4;
} hsa_amd_aie_kernel_dispatch_packet_t;

/** @} */

#ifdef __cplusplus
}  // end extern "C" block
#endif

#endif  // header guard
