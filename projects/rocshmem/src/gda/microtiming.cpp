/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "microtiming.hpp"
#include "util.hpp"
#include <algorithm>
#include <cstdio>
#include <vector>

namespace rocshmem {

static void microtiming_set_enabled(int val) {
  microtiming_t* dev_ptr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&dev_ptr),
                                HIP_SYMBOL(g_microtiming)));
  CHECK_HIP(hipMemcpy(&dev_ptr->enabled, &val, sizeof(int), hipMemcpyHostToDevice));
}

__host__ void microtiming_enable() {
  microtiming_set_enabled(1);
}

__host__ void microtiming_disable() {
  microtiming_set_enabled(0);
}

__host__ void microtiming_reset() {
  microtiming_t* dev_ptr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&dev_ptr),
                                HIP_SYMBOL(g_microtiming)));
  microtiming_t zero{};
  CHECK_HIP(hipMemcpy(dev_ptr, &zero, sizeof(microtiming_t), hipMemcpyHostToDevice));
}

struct microtiming_stats {
  double min, max, p50, p99;
};

static microtiming_stats compute_stats(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return {
    v[0],
    v[n - 1],
    v[n / 2],
    v[(n * 99) / 100]
  };
}

/**
 * Query the wall clock period in nanoseconds.
 * s_memrealtime ticks at hipDeviceAttributeWallClockRate (in kHz).
 * ns_per_tick = 1e6 / rate_kHz.
 */
static double get_ns_per_tick() {
  int device;
  CHECK_HIP(hipGetDevice(&device));
  int wall_clk_rate_khz;
  CHECK_HIP(hipDeviceGetAttribute(&wall_clk_rate_khz,
            hipDeviceAttributeWallClockRate, device));
  return 1e6 / static_cast<double>(wall_clk_rate_khz);
}

__host__ void microtiming_print() {
  microtiming_t host_copy{};
  microtiming_t* dev_ptr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&dev_ptr),
                                HIP_SYMBOL(g_microtiming)));
  CHECK_HIP(hipMemcpy(&host_copy, dev_ptr, sizeof(microtiming_t), hipMemcpyDeviceToHost));

  int iters = host_copy.iter;
  if (iters > MICROTIMING_MAX_ITERS) iters = MICROTIMING_MAX_ITERS;

  if (iters == 0) {
    printf("=== Microtiming: no iterations recorded ===\n");
    return;
  }

  double ns_per_tick = get_ns_per_tick();

  // Compute per-section durations from 9 timestamps:
  //   T0: app call site          T1: before IPC check
  //   T2: before ActiveWFInfo    T3: before put_nbi
  //   T4: post_wqe_rma entry    T5: WQE written
  //   T6: doorbell rung         T7: after put_nbi returns
  //   T8: after putmem_nbi returns (app side)
  static constexpr int NUM_SECTIONS = 13;
  const char* section_names[NUM_SECTIONS] = {
    "dispatch",      // T1-T0: app -> putmem_nbi entry
    "ipc_check",     // T2-T1: IPC availability check
    "wf_info+qp",    // T3-T2: ActiveWFInfo + get_qp_index + offset
    "put_nbi_call",  // T4-T3: put_nbi wrapper -> post_wqe_rma entry
    "wqe+lock+cq",   // T5-T4: CQ poll + lock + WQE build + SQ write
    "doorbell",      // T6-T5: doorbell ring
    "put_nbi_ret",   // T7-T6: post_wqe_rma return -> put_nbi return
    "return_path",   // T8-T7: putmem_nbi return -> app
    "total",         // T8-T0: end-to-end
    "post_wqe_rma",  // T6-T4: post_wqe_rma entry to doorbell
    "entry_ovhd",    // T4-T0: app call to post_wqe_rma entry
    "ret_ovhd",      // T8-T6: doorbell to app return
    "lib_overhead",  // entry_ovhd + ret_ovhd
  };

  std::vector<double> sections[NUM_SECTIONS];
  for (int s = 0; s < NUM_SECTIONS; s++) {
    sections[s].reserve(iters);
  }

  for (int i = 0; i < iters; i++) {
    int base = i * MICROTIMING_STAMPS_PER_ITER;
    uint64_t t0 = host_copy.ts[base + 0];
    uint64_t t1 = host_copy.ts[base + 1];
    uint64_t t2 = host_copy.ts[base + 2];
    uint64_t t3 = host_copy.ts[base + 3];
    uint64_t t4 = host_copy.ts[base + 4];
    uint64_t t5 = host_copy.ts[base + 5];
    uint64_t t6 = host_copy.ts[base + 6];
    uint64_t t7 = host_copy.ts[base + 7];
    uint64_t t8 = host_copy.ts[base + 8];

    // Skip iterations with zero timestamps (incomplete recording)
    if (t0 == 0 || t8 == 0) continue;

    sections[0].push_back((t1 - t0) * ns_per_tick);
    sections[1].push_back((t2 - t1) * ns_per_tick);
    sections[2].push_back((t3 - t2) * ns_per_tick);
    sections[3].push_back((t4 - t3) * ns_per_tick);
    sections[4].push_back((t5 - t4) * ns_per_tick);
    sections[5].push_back((t6 - t5) * ns_per_tick);
    sections[6].push_back((t7 - t6) * ns_per_tick);
    sections[7].push_back((t8 - t7) * ns_per_tick);
    sections[8].push_back((t8 - t0) * ns_per_tick);
    sections[9].push_back((t6 - t4) * ns_per_tick);
    sections[10].push_back((t4 - t0) * ns_per_tick);
    sections[11].push_back((t8 - t6) * ns_per_tick);
    sections[12].push_back(((t4 - t0) + (t8 - t6)) * ns_per_tick);
  }

  int valid = static_cast<int>(sections[0].size());
  printf("=== Microtiming: %d/%d valid iterations (wall_clk=%.0f ns/tick) ===\n",
         valid, iters, ns_per_tick);

  if (valid == 0) return;

  printf("%-14s  %10s  %10s  %10s  %10s\n",
         "section", "min(ns)", "P50(ns)", "P99(ns)", "max(ns)");

  for (int s = 0; s < NUM_SECTIONS; s++) {
    auto st = compute_stats(sections[s]);
    printf("%-14s  %10.0f  %10.0f  %10.0f  %10.0f\n",
           section_names[s], st.min, st.p50, st.p99, st.max);
  }

  if (host_copy.quiet_start && host_copy.quiet_end) {
    printf("%-14s  %10s  %10s  %10s  %10.0f\n",
           "quiet", "-", "-", "-",
           (host_copy.quiet_end - host_copy.quiet_start) * ns_per_tick);
  }

  if (host_copy.e2e_start && host_copy.e2e_end) {
    double e2e_ns = (host_copy.e2e_end - host_copy.e2e_start) * ns_per_tick;
    printf("\nEnd-to-end (s_memrealtime): %.0f ns = %.2f us "
           "(covers all timed iterations + quiet)\n", e2e_ns, e2e_ns / 1000.0);
    if (valid > 0) {
      printf("Per-message (s_memrealtime): %.2f us (%d msgs)\n",
             e2e_ns / 1000.0 / valid, valid);
    }
  }
  printf("===\n");
}

}  // namespace rocshmem
