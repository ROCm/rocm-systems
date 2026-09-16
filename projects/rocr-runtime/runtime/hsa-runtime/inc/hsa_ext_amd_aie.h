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
   *
   * A size of 0 for a kernarg means no cacheline is flushed for that argument; the argument is
   * still resolved and kept resident for the duration of the dispatch.
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
   * Byte offset into the instruction sequence at which the runtime writes the 64-bit device
   * address of ::pdi_addr, or 0 if the instruction sequence needs no such patch.
   *
   * This field selects between the two modes a dispatch can use:
   *
   * - **0 -- PDI plus instruction sequence.** ::insts_addr_low / ::insts_addr_high point at a
   *   standalone instruction sequence, ::pdi_addr at the PDI that configures the array for it,
   *   and the hardware patches the arguments in ::kernarg_address into the instruction sequence
   *   as it runs.
   *
   * - **Non-zero -- full ELF.** ::insts_addr_low / ::insts_addr_high point at the control code
   *   of a full-ELF kernel, which loads its own PDI rather than relying on the array having been
   *   configured out of band. Supported only on aie2p agents.
   *
   *   The application extracts the control code and the PDI from the ELF and allocates both from
   *   the agent's device memory pool. Neither has to sit at the start of its allocation, but the
   *   control code must be 16 KiB aligned.
   *
   *   The application patches its argument addresses into the control code before enqueuing, using
   *   the relocations in the ELF. The arguments are not passed to the hardware, so two dispatches
   *   with different arguments need two control-code buffers ::kernarg_address and ::num_kernargs
   *   must still list the argument buffers.
   *
   *   The PDI's device address resolution needs the offset of that patch site.
   *
   * The packets submitted by one doorbell ring fix the mode of queue for the batch, and a packet
   * of the other mode in the same batch is rejected. A later batch on the same queue may use the
   * other mode.
   *
   * Grouping dispatches by shape is the application's job; alternating shapes batch by batch works,
   * but pays a context rebuild each time it switches.
   */
  uint64_t pdi_patch_offset;
} hsa_amd_aie_kernel_dispatch_packet_t;

/** @} */

#ifdef __cplusplus
}  // end extern "C" block
#endif

#endif  // header guard
