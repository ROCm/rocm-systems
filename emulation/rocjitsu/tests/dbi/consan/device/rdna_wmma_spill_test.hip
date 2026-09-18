// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_device_test_support.h"
#include <string_view>

namespace {

// Keep an accumulator live at an LDS probe immediately after WMMA. Full
// register pressure forces the probe to spill; memory waits alone cannot make
// the pending WMMA result safe to save and later restore. The explicit wave
// scheduling mode matches the hipBLASLt kernel that exposed the corruption.
__global__ __launch_bounds__(32) void rdna_wmma_spill_for_consan(float *output) {
#if defined(__gfx1201__)
  __shared__ uint4 staged[32];
  const uint32_t lane = threadIdx.x;
  staged[lane] = make_uint4(1u, 2u, 3u, 4u);
  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&staged[lane]) & 0xffffu);
  register float accumulator0 asm("v0") = 0.0f;
  register float accumulator1 asm("v1") = 0.0f;
  register float accumulator2 asm("v2") = 0.0f;
  register float accumulator3 asm("v3") = 0.0f;
  register float accumulator4 asm("v4") = 0.0f;
  register float accumulator5 asm("v5") = 0.0f;
  register float accumulator6 asm("v6") = 0.0f;
  register float accumulator7 asm("v7") = 0.0f;
  asm volatile(
      ".irp "
      "r,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,"
      "73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,"
      "125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,167,169,"
      "171,173,175,177,179,181,183,185,187,189,191,193,195,197,199,201,203,205,207,209,211,213,215,"
      "217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255\n\t"
      "v_mov_b32 v\\r, 0\n\t"
      ".endr\n\t"
      "v_mov_b32 v8, 0x3c003c00\n\t"
      "v_mov_b32 v9, v8\n\t"
      "v_mov_b32 v10, v8\n\t"
      "v_mov_b32 v11, v8\n\t"
      "s_wait_dscnt 0\n\t"
      "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2\n\t"
      "v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[8:11], v[0:7]\n\t"
      "ds_load_b128 v[20:23], %8\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
      "s_wait_alu 0\n\t"
      ".irp "
      "r,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,"
      "57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,"
      "109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,"
      "149,151,153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,"
      "189,191,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,"
      "229,231,233,235,237,239,241,243,245,247,249,251,253,255\n\t"
      "v_mov_b32 v\\r, v\\r\n\t"
      ".endr\n\t"
      "s_wait_dscnt 0\n\t"
      "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 0\n\t"
      : "+v"(accumulator0), "+v"(accumulator1), "+v"(accumulator2), "+v"(accumulator3),
        "+v"(accumulator4), "+v"(accumulator5), "+v"(accumulator6), "+v"(accumulator7)
      : "v"(address)
      : "v8", "v9", "v10", "v11", "v20", "v22", "v13", "v15", "v17", "v19", "v21", "v23", "v25",
        "v27", "v29", "v31", "v33", "v35", "v37", "v39", "v41", "v43", "v45", "v47", "v49", "v51",
        "v53", "v55", "v57", "v59", "v61", "v63", "v65", "v67", "v69", "v71", "v73", "v75", "v77",
        "v79", "v81", "v83", "v85", "v87", "v89", "v91", "v93", "v95", "v97", "v99", "v101", "v103",
        "v105", "v107", "v109", "v111", "v113", "v115", "v117", "v119", "v121", "v123", "v125",
        "v127", "v129", "v131", "v133", "v135", "v137", "v139", "v141", "v143", "v145", "v147",
        "v149", "v151", "v153", "v155", "v157", "v159", "v161", "v163", "v165", "v167", "v169",
        "v171", "v173", "v175", "v177", "v179", "v181", "v183", "v185", "v187", "v189", "v191",
        "v193", "v195", "v197", "v199", "v201", "v203", "v205", "v207", "v209", "v211", "v213",
        "v215", "v217", "v219", "v221", "v223", "v225", "v227", "v229", "v231", "v233", "v235",
        "v237", "v239", "v241", "v243", "v245", "v247", "v249", "v251", "v253", "v255", "memory");
  output[lane * 8 + 0] = accumulator0;
  output[lane * 8 + 1] = accumulator1;
  output[lane * 8 + 2] = accumulator2;
  output[lane * 8 + 3] = accumulator3;
  output[lane * 8 + 4] = accumulator4;
  output[lane * 8 + 5] = accumulator5;
  output[lane * 8 + 6] = accumulator6;
  output[lane * 8 + 7] = accumulator7;
#endif
}

void check_wmma_spill() {
  hipDeviceProp_t properties{};
  HIP_ASSERT(hipGetDeviceProperties(&properties, 0));
  if (std::string_view(properties.gcnArchName).substr(0, 7) != "gfx1201")
    GTEST_SKIP() << "Requires physical gfx1201 WMMA scheduling";
  float *output = nullptr;
  HIP_ASSERT(hipMalloc(&output, 32 * 8 * sizeof(float)));
  rdna_wmma_spill_for_consan<<<1, 32>>>(output);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  float result[32 * 8]{};
  HIP_ASSERT(hipMemcpy(result, output, sizeof(result), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(output));
  for (uint32_t i = 0; i < 32 * 8; ++i)
    EXPECT_EQ(result[i], 16.0f) << "WMMA spill element=" << i;
}

TEST(ConSanDeviceRdnaWmmaSpillTest, Correct) { check_wmma_spill(); }

} // namespace
