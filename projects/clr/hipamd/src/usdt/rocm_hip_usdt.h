/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * USDT (User Statically-Defined Tracing) probe shims for HIP.
 *
 * Provider: hip
 * Probes:
 *   hip:api_entry(uint32_t api_id, uint64_t correlation_id)
 *   hip:api_exit(uint32_t api_id, uint64_t correlation_id, int32_t hip_error)
 *
 * Disabled-build cost: zero (macros expand to (void)0).
 * Enabled-build, unattached cost: 1 nop per probe site.
 * Enabled-build, attached cost: ~1-3 us per fire (kernel uprobe trap + BPF program).
 */

#ifndef HIP_SRC_USDT_ROCM_HIP_USDT_H
#define HIP_SRC_USDT_ROCM_HIP_USDT_H

#include <cstdint>

#if defined(ROCM_ENABLE_USDT) && ROCM_ENABLE_USDT
#  include <sys/sdt.h>
#  define ROCM_HIP_USDT_API_ENTRY(api_id, corr) \
        DTRACE_PROBE2(hip, api_entry, (uint32_t)(api_id), (uint64_t)(corr))
#  define ROCM_HIP_USDT_API_EXIT(api_id, corr, err) \
        DTRACE_PROBE3(hip, api_exit, (uint32_t)(api_id), (uint64_t)(corr), (int32_t)(err))
#else
#  define ROCM_HIP_USDT_API_ENTRY(api_id, corr) ((void)0)
#  define ROCM_HIP_USDT_API_EXIT(api_id, corr, err) ((void)0)
#endif

#endif  // HIP_SRC_USDT_ROCM_HIP_USDT_H
