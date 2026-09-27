/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

// Enqueue-time half of the scheduling hint. The instruction mix and the
// COMGR record are filled in once, when the code object is loaded.
// VirtualGPU::submitKernelInternal already has the launch and the metadata
// (VGPR/SGPR, group and private segments, argument signature); it would call
// predict() with those fields. Keep the formulas aligned with
// kernel_sched_hint/model.py.

#ifndef EXPERIMENTAL_KERNEL_SCHED_HINT_HPP_
#define EXPERIMENTAL_KERNEL_SCHED_HINT_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace kernel_sched_hint {

constexpr double kLaunchOverheadS = 5e-6;
constexpr double kDefaultThroughputScale = 1.0 / 0.35;
constexpr double kLowEfficiency = 0.10;
constexpr double kLdsLatencyCycles = 30.0;
constexpr double kSfuLatencyCycles = 16.0;
constexpr double kMixedRatio = 0.75;

struct Work {
  double valu_f32_flops = 0;
  double valu_f64_flops = 0;
  double valu_f16_flops = 0;
  double mfma_f32_flops = 0;
  double mfma_f64_flops = 0;
  double mfma_f16_flops = 0;
  double global_bytes = 0;
  double lds_bytes = 0;
  double scratch_bytes = 0;
  double atomic_ops = 0;
  double mem_ops = 0;
  double lds_ops = 0;
  double waitcnts = 0;
  double branches = 0;
  double sfu_ops = 0;
  double valu_issues = 0;
  int vgprs = 0;
  int sgprs = 0;
  int wavefront_size = 64;
};

struct Device {
  int cu_count = 0;
  int simd_per_cu = 0;
  double clock_hz = 0;
  double fp32_vector_flops = 0;
  double fp64_vector_flops = 0;
  double fp16_vector_flops = 0;
  double fp32_matrix_flops = 0;
  double fp64_matrix_flops = 0;
  double fp16_matrix_flops = 0;
  double hbm_bytes_per_s = 0;
  double hbm_latency_s = 0;
  int lds_bytes_per_cu_per_clock = 0;
  int sgprs_per_simd = 0;
  int lds_bytes_per_cu = 0;
  // gfx9 occupancy: waves per EU at this VGPR count. gfx942/gfx950/gfx90a.
  int waves_per_eu_for_vgprs(int vgprs) const {
    const int count = vgprs > 0 ? vgprs : 64;
    const int limits[] = {64, 80, 96, 128, 160, 256};
    const int waves[] = {8, 6, 5, 4, 3, 2};
    for (int i = 0; i < 6; ++i) {
      if (count <= limits[i]) return waves[i];
    }
    return 1;
  }
};

struct Terms {
  double t_valu = 0;
  double t_matrix = 0;
  double t_hbm = 0;
  double t_lds = 0;
  double t_issue = 0;
  double t_latency = 0;
  double t_launch = 0;
  double t_throughput = 0;
  double flops = 0;
  double bytes = 0;
};

inline Terms evaluate(const Work& work, const Device& dev, double waves, double passes,
                      int lds_bytes_per_workgroup = 0, int workgroup_size = 0,
                      double global_byte_cap = -1.0) {
  const int assumed = work.vgprs > 0 ? work.vgprs : 64;
  int wpe = dev.waves_per_eu_for_vgprs(assumed);
  if (dev.sgprs_per_simd > 0 && work.sgprs > 0) {
    wpe = std::min(wpe, std::max(1, dev.sgprs_per_simd / work.sgprs));
  }
  double resident = static_cast<double>(dev.cu_count * dev.simd_per_cu * wpe);
  if (lds_bytes_per_workgroup > 0 && workgroup_size > 0 && dev.lds_bytes_per_cu > 0) {
    const int wg_per_cu = std::max(1, dev.lds_bytes_per_cu / lds_bytes_per_workgroup);
    const int wf = std::max(1, work.wavefront_size);
    const int waves_per_wg = std::max(1, (workgroup_size + wf - 1) / wf);
    resident = std::min(resident, static_cast<double>(dev.cu_count * wg_per_cu * waves_per_wg));
  }
  if (resident < 1.0) resident = 1.0;
  const double hide = std::min(1.0, waves / resident);
  const double factor = passes * waves;

  Terms terms;
  terms.t_valu = factor * (work.valu_f32_flops / dev.fp32_vector_flops +
                           work.valu_f64_flops / dev.fp64_vector_flops +
                           work.valu_f16_flops / dev.fp16_vector_flops);
  terms.t_matrix = factor * (work.mfma_f32_flops / dev.fp32_matrix_flops +
                             work.mfma_f64_flops / dev.fp64_matrix_flops +
                             work.mfma_f16_flops / dev.fp16_matrix_flops);
  double global_bytes_total = factor * work.global_bytes;
  if (global_byte_cap >= 0.0) global_bytes_total = std::min(global_bytes_total, global_byte_cap);
  const double scratch_bytes_total = factor * work.scratch_bytes;
  terms.t_hbm = (global_bytes_total + scratch_bytes_total) / dev.hbm_bytes_per_s;
  const double lds_bw =
      static_cast<double>(dev.lds_bytes_per_cu_per_clock) * dev.cu_count * dev.clock_hz;
  terms.t_lds = factor * work.lds_bytes / lds_bw;
  const double issue_slots = static_cast<double>(dev.cu_count) * dev.simd_per_cu * dev.clock_hz;
  terms.t_issue = factor * work.valu_issues / issue_slots;
  terms.t_throughput = std::max(
      terms.t_valu,
      std::max(terms.t_matrix, std::max(terms.t_hbm, std::max(terms.t_lds, terms.t_issue))));

  const bool has_global =
      (work.mem_ops + work.atomic_ops) > 0 || (work.global_bytes + work.scratch_bytes) > 0;
  const bool has_lds = work.lds_ops > 0 || work.lds_bytes > 0;
  double chain = 0;
  if (work.waitcnts > 0) {
    if (has_global || !has_lds)
      chain += work.waitcnts * dev.hbm_latency_s;
    else
      chain += work.waitcnts * (kLdsLatencyCycles / dev.clock_hz);
  } else if (has_global) {
    chain += dev.hbm_latency_s;
  } else if (has_lds) {
    chain += kLdsLatencyCycles / dev.clock_hz;
  }
  if (work.sfu_ops > 0) chain += work.sfu_ops * (kSfuLatencyCycles / dev.clock_hz);
  terms.t_latency = chain * passes * (1.0 - hide);
  terms.t_launch = kLaunchOverheadS;
  terms.flops = factor * (work.valu_f32_flops + work.valu_f64_flops + work.valu_f16_flops +
                          work.mfma_f32_flops + work.mfma_f64_flops + work.mfma_f16_flops);
  terms.bytes = global_bytes_total + scratch_bytes_total;
  return terms;
}

inline const char* resource_class(const char* resource) {
  if (std::string(resource) == "valu" || std::string(resource) == "matrix" ||
      std::string(resource) == "issue")
    return "alu";
  if (std::string(resource) == "hbm" || std::string(resource) == "lds") return "memory";
  if (std::string(resource) == "latency") return "latency";
  return "unknown";
}

struct Prediction {
  std::string bound;
  std::string resource;
  double roofline_s = 0;
  double duration_s = 0;
};

inline Prediction predict(const Work& work, const Device& dev, double waves, double passes = 1.0,
                          double scale = kDefaultThroughputScale, int lds_bytes_per_workgroup = 0,
                          int workgroup_size = 0, double global_byte_cap = -1.0) {
  const Terms terms =
      evaluate(work, dev, waves, passes, lds_bytes_per_workgroup, workgroup_size, global_byte_cap);
  const double comps[] = {terms.t_valu, terms.t_matrix, terms.t_hbm,
                          terms.t_lds,  terms.t_issue,  terms.t_latency};
  const char* names[] = {"valu", "matrix", "hbm", "lds", "issue", "latency"};
  int top = 0;
  int second = 1;
  for (int i = 1; i < 6; ++i) {
    if (comps[i] > comps[top]) {
      second = top;
      top = i;
    } else if (i != top && comps[i] > comps[second]) {
      second = i;
    }
  }
  Prediction out;
  if (comps[top] <= 0) {
    out.bound = "unknown";
    out.resource = "none";
  } else {
    out.resource = names[top];
    out.bound = resource_class(names[top]);
    if (comps[second] > 0 && resource_class(names[second]) != out.bound &&
        comps[second] / comps[top] >= kMixedRatio) {
      out.bound = "mixed";
    }
  }
  out.roofline_s = terms.t_throughput + terms.t_latency + terms.t_launch;
  out.duration_s = terms.t_throughput * scale + terms.t_latency + terms.t_launch;
  return out;
}

inline Device mi300x() {
  Device dev;
  dev.cu_count = 304;
  dev.simd_per_cu = 4;
  dev.clock_hz = 2.1e9;
  dev.fp32_vector_flops = 163.4e12;
  dev.fp64_vector_flops = 81.7e12;
  dev.fp16_vector_flops = 163.4e12;
  dev.fp32_matrix_flops = 163.4e12;
  dev.fp64_matrix_flops = 163.4e12;
  dev.fp16_matrix_flops = 1307.4e12;
  dev.hbm_bytes_per_s = 5.3e12;
  dev.hbm_latency_s = 4e-7;
  dev.lds_bytes_per_cu_per_clock = 128;
  dev.sgprs_per_simd = 800;
  dev.lds_bytes_per_cu = 64 * 1024;
  return dev;
}

}  // namespace kernel_sched_hint

#endif  // EXPERIMENTAL_KERNEL_SCHED_HINT_HPP_
