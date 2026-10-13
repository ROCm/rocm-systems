/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Optional public-HSA adapter for the Accelerated Blit Copy Engine (ABCE).
// Include this header only in clients that submit to ROCr-created SDMA queues.

#ifndef ABCE_HSA_H_
#define ABCE_HSA_H_

#if __has_include(<hsa/hsa.h>)
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#else
#include <hsa.h>
#include <hsa_ext_amd.h>
#endif

#if __has_include(<hsa/amd_hsa_signal.h>)
#include <hsa/amd_hsa_signal.h>
#else
#include <amd_hsa_signal.h>
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "abce_config.h"
#include "abce_host.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Signal adapters
//===----------------------------------------------------------------------===//

// Completion target for an HSA signal. |completion_value| is the value the
// signal holds once hardware finishes; see abce_signal_ref_t for how each
// completion packet reaches it. Lives here rather than on abce_signal_ref_t so
// the frame composer stays free of the amd_signal_t layout.
//
// The fan-out coordination scratch is placed in the signal's own trailing
// reserved words: they are a naturally aligned 64-bit slot inside the same
// 64-byte signal object, so coordination shares a cache line with the value it
// guards and needs no separate allocation.
static inline abce_signal_ref_t abce_hsa_signal_ref(amd_signal_t* signal,
                                                    uint64_t completion_value) {
  ABCE_STATIC_ASSERT(sizeof(((amd_signal_t*)0)->reserved3) == sizeof(uint64_t),
                     "amd_signal_t::reserved3 is no longer a 64-bit slot");
  ABCE_STATIC_ASSERT(offsetof(amd_signal_t, reserved3) % ABCE_ALIGNOF(uint64_t) == 0,
                     "amd_signal_t::reserved3 is not 64-bit aligned");
  return abce_signal_ref_make((void*)&signal->value, completion_value,
                              (void*)(uintptr_t)signal->event_mailbox_ptr, signal->event_id,
                              &signal->reserved3[0]);
}

// Suggested placement for abce_copy_metadata_t::execution_descriptor: the
// signal's spare 32-bit reserved word, which shares the signal's cache line and
// so costs no extra allocation. Opt in by assigning this into the metadata; ABCE
// never writes it on its own, because which bytes of a signal may be repurposed
// is the caller's ABI decision, not ABCE's.
//
// amd_signal_t is a frozen, potentially cross-process format, so a client that
// cannot spend reserved1 should point the descriptor at its own wrapper struct
// instead -- exactly what ROCr does for SDMA timestamps, which live in
// SharedSignal rather than in amd_signal_t. Nothing about the descriptor depends
// on being inside the signal; it only has to be somewhere its reader agrees to
// look.
static inline uint32_t* abce_hsa_execution_descriptor_slot(amd_signal_t* signal) {
  return &signal->reserved1;
}

// Dependency on an HSA signal. |observed| is the value read at record time, used
// to elide an already-satisfied wait.
static inline abce_dep_signal_t abce_hsa_dep_signal(amd_signal_t* signal, uint64_t observed,
                                                    uint64_t reference) {
  return abce_dep_signal_make((void*)&signal->value, observed, reference);
}

//===----------------------------------------------------------------------===//
// abce_hsa_queue_api_t
//===----------------------------------------------------------------------===//

typedef hsa_status_t (*abce_hsa_queue_get_info_fn_t)(hsa_queue_t* queue,
                                                     hsa_queue_info_attribute_t attribute,
                                                     void* value);
typedef uint64_t (*abce_hsa_queue_load_read_index_fn_t)(const hsa_queue_t* queue);
typedef uint64_t (*abce_hsa_queue_load_write_index_fn_t)(const hsa_queue_t* queue);
typedef void (*abce_hsa_signal_store_release_fn_t)(hsa_signal_t signal, hsa_signal_value_t value);

// HSA entry points used by the queue ring. Keeping them injectable lets clients
// with a dynamically loaded ROCr runtime use the adapter without adding a static
// runtime dependency.
typedef struct abce_hsa_queue_api_t {
  abce_hsa_queue_get_info_fn_t queue_get_info;
  abce_hsa_queue_load_read_index_fn_t load_read_index;
  abce_hsa_queue_load_write_index_fn_t load_write_index;
  abce_hsa_signal_store_release_fn_t signal_store_release;
} abce_hsa_queue_api_t;

static inline bool abce_hsa_queue_api_complete(const abce_hsa_queue_api_t* api) {
  return api->queue_get_info != NULL && api->load_read_index != NULL &&
         api->load_write_index != NULL && api->signal_store_release != NULL;
}

static inline hsa_status_t abce_hsa_direct_queue_get_info(hsa_queue_t* queue,
                                                          hsa_queue_info_attribute_t attribute,
                                                          void* value) {
  return hsa_amd_queue_get_info(queue, attribute, value);
}

static inline uint64_t abce_hsa_direct_load_read_index(const hsa_queue_t* queue) {
  return hsa_queue_load_read_index_relaxed(queue);
}

static inline uint64_t abce_hsa_direct_load_write_index(const hsa_queue_t* queue) {
  return hsa_queue_load_write_index_relaxed(queue);
}

static inline void abce_hsa_direct_signal_store_release(hsa_signal_t signal,
                                                        hsa_signal_value_t value) {
  hsa_signal_store_screlease(signal, value);
}

// Entry-point table for clients linked directly to the HSA runtime.
static inline abce_hsa_queue_api_t abce_hsa_queue_api_direct(void) {
  abce_hsa_queue_api_t api;
  api.queue_get_info = abce_hsa_direct_queue_get_info;
  api.load_read_index = abce_hsa_direct_load_read_index;
  api.load_write_index = abce_hsa_direct_load_write_index;
  api.signal_store_release = abce_hsa_direct_signal_store_release;
  return api;
}

//===----------------------------------------------------------------------===//
// abce_hsa_queue_ring_t
//===----------------------------------------------------------------------===//

typedef struct abce_hsa_ring_options_t {
  size_t min_submission_size;
  bool align64;
  // The CPU's stores into the ring buffer pass through the GPU's HDP: a ring in
  // device memory (HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF) the CPU fills
  // through a PCIe BAR. False for a ring in host memory, and for one in device
  // memory behind a CPU-GPU xGMI link, which has no HDP.
  bool host_writes_through_hdp;
  abce_shared_ring_control_t* shared_control;
  abce_ring_commit_wait_fn_t commit_wait;
  void* commit_wait_user_data;
} abce_hsa_ring_options_t;

static inline void abce_hsa_ring_options_initialize(abce_hsa_ring_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

// Non-owning adapter from a public SDMA hsa_queue_t to an ABCE ring.
//
// The queue must remain alive until all submissions are complete and the adapter
// is no longer registered with an orchestrator. Host/device sharing is supported
// only when both views use the same abce_shared_ring_control_t. The adapter
// itself must remain at a stable address because the ring callbacks use it as
// their context.
typedef struct abce_hsa_queue_ring_t {
  hsa_queue_t* queue;
  abce_hsa_queue_api_t api;
  uint32_t engine_id;
  bool host_writes_through_hdp;
  abce_ring_t ring;
} abce_hsa_queue_ring_t;

static inline uint64_t abce_hsa_queue_ring_load_read_index(void* user_data) {
  abce_hsa_queue_ring_t* adapter = (abce_hsa_queue_ring_t*)user_data;
  return adapter->api.load_read_index(adapter->queue);
}

static inline uint64_t abce_hsa_queue_ring_load_write_index(void* user_data) {
  abce_hsa_queue_ring_t* adapter = (abce_hsa_queue_ring_t*)user_data;
  return adapter->api.load_write_index(adapter->queue);
}

static inline void abce_hsa_queue_ring_publish_write_index(void* user_data,
                                                           uint64_t new_write_index) {
  abce_hsa_queue_ring_t* adapter = (abce_hsa_queue_ring_t*)user_data;
  // SdmaQueue::StoreRelease stores the canonical write pointer and rings the
  // KFD doorbell behind release ordering only, which on x86 leaves the packet
  // stores in a VRAM ring unordered against the doorbell.
  abce_ring_host_store_fence();
  if (adapter->host_writes_through_hdp) {
    // HDP can still hold the packet stores when the doorbell, which does not
    // pass through it, reaches the engine: it then fetches the ring's old
    // contents. A read cannot complete ahead of the posted writes before it,
    // and on x86 the doorbell store cannot pass the read, so reading back the
    // frame's last dword lands the frame first. It is the readback CLR uses for
    // kernel arguments in device memory.
    const uint64_t last_dword_offset =
        abce_ring_wrap(new_write_index - sizeof(uint32_t), adapter->queue->size);
    (void)*(const volatile uint32_t*)((const char*)adapter->queue->base_address +
                                      last_dword_offset);
  }
  adapter->api.signal_store_release(adapter->queue->doorbell_signal,
                                    (hsa_signal_value_t)new_write_index);
}

// Initializes the adapter over |queue|. |options| may be null for the defaults.
static inline hsa_status_t abce_hsa_queue_ring_initialize(
    abce_hsa_queue_ring_t* out_adapter, hsa_queue_t* queue, const abce_hsa_queue_api_t* api,
    const abce_hsa_ring_options_t* options) {
  if (queue == NULL || api == NULL || !abce_hsa_queue_api_complete(api))
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  abce_hsa_ring_options_t default_options;
  if (!options) {
    abce_hsa_ring_options_initialize(&default_options);
    options = &default_options;
  }

  hsa_amd_queue_engine_t engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
  hsa_status_t status = api->queue_get_info(queue, HSA_AMD_QUEUE_INFO_ENGINE_TYPE, &engine_type);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (engine_type != HSA_AMD_QUEUE_ENGINE_SDMA) return HSA_STATUS_ERROR_INVALID_QUEUE;

  uint32_t resolved_engine_id = HSA_AMD_SDMA_ENGINE_ID_ANY;
  status = api->queue_get_info(queue, HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID, &resolved_engine_id);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (resolved_engine_id == HSA_AMD_SDMA_ENGINE_ID_ANY) return HSA_STATUS_ERROR_INVALID_QUEUE;

  const size_t ring_size = (size_t)queue->size;
  if (queue->base_address == NULL || !abce_is_power_of_two(ring_size))
    return HSA_STATUS_ERROR_INVALID_QUEUE;

  out_adapter->queue = queue;
  out_adapter->api = *api;
  out_adapter->engine_id = resolved_engine_id;
  out_adapter->host_writes_through_hdp = options->host_writes_through_hdp;

  abce_ring_config_t config;
  memset(&config, 0, sizeof(config));
  config.base = (char*)queue->base_address;
  config.size = ring_size;
  config.queue_ops.load_read_index = abce_hsa_queue_ring_load_read_index;
  config.queue_ops.load_write_index = abce_hsa_queue_ring_load_write_index;
  config.queue_ops.publish_write_index = abce_hsa_queue_ring_publish_write_index;
  config.queue_ops.user_data = out_adapter;
  config.shared_control = options->shared_control;
  config.min_submission_size = options->min_submission_size;
  config.align64 = options->align64;
  config.commit_wait = options->commit_wait;
  config.commit_wait_user_data = options->commit_wait_user_data;
  abce_ring_initialize(&config, NULL, NULL, &out_adapter->ring);
  return HSA_STATUS_SUCCESS;
}

static inline bool abce_hsa_queue_ring_initialized(const abce_hsa_queue_ring_t* adapter) {
  return adapter->queue != NULL;
}

static inline abce_ring_t* abce_hsa_queue_ring_ring(abce_hsa_queue_ring_t* adapter) {
  return abce_hsa_queue_ring_initialized(adapter) ? &adapter->ring : NULL;
}

static inline abce_status_t abce_hsa_queue_ring_drain(const abce_hsa_queue_ring_t* adapter) {
  return abce_hsa_queue_ring_initialized(adapter) ? abce_ring_drain(&adapter->ring)
                                                  : ABCE_STATUS_INVALID_ARGUMENT;
}

static inline bool abce_hsa_register_engine(abce_copy_orchestrator_t* orchestrator,
                                            uint32_t orchestrator_index,
                                            abce_hsa_queue_ring_t* queue_ring) {
  if (!abce_hsa_queue_ring_initialized(queue_ring)) return false;
  abce_engine_affinity_t affinity;
  abce_engine_affinity_initialize(&affinity);
  affinity.hw_engine_id = queue_ring->engine_id;
  return abce_copy_orchestrator_register_engine(orchestrator, orchestrator_index,
                                                abce_hsa_queue_ring_ring(queue_ring), &affinity);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_HSA_H_
