// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _GFX9_PRIMITIVES_H_
#define _GFX9_PRIMITIVES_H_

#include <stdint.h>
#include <cstdint>

#define COPY_DATA_SEL_REG 0  ///< Mem-mapped register
#define COPY_DATA_SEL_SRC_SYS_PERF_COUNTER 4
#define COPY_DATA_SEL_COUNT_1DW 0  ///< Copy 1 word (32 bits)

// Counter Select Register value lambdas
#define select_value(reg_name)                                                \
  [](const counter_des_t& counter_des) {                                      \
    uint32_t select = SET_REG_FIELD_BITS(reg_name, PERF_SEL, counter_des.id); \
    return select;                                                            \
  }
#define select_value_t2(reg_name)                                                       \
  [](const counter_des_t& counter_des) {                                                \
    uint32_t select = SET_REG_FIELD_BITS(reg_name, PERFCOUNTER_SELECT, counter_des.id); \
    return select;                                                                      \
  }
#define select_value_t3(reg_name)                                              \
  [](const counter_des_t& counter_des) {                                       \
    uint32_t select = SET_REG_FIELD_BITS(reg_name, CNTR_SEL0, counter_des.id); \
    return select;                                                             \
  }
#define mc_select_value(reg_name)                                                           \
  [](const counter_des_t& counter_des) {                                                    \
    uint32_t select = SET_REG_FIELD_BITS(reg_name, PERF_SEL, counter_des.id) |              \
                      SET_REG_FIELD_BITS(reg_name, PERF_MODE, PERFMON_COUNTER_MODE_ACCUM) | \
                      SET_REG_FIELD_BITS(reg_name, ENABLE, 1);                              \
    return select;                                                                          \
  }

namespace gfxip {
namespace gfx9 {

class gfx9_cntx_prim {
 public:
  static const uint32_t GFXIP_LEVEL = 9;
  static const uint32_t NUMBER_OF_BLOCKS = LastCounterBlockId + 1;
  static constexpr Register GRBM_GFX_INDEX_ADDR = REG_32B_ADDR(GC, 0, mmGRBM_GFX_INDEX);
  static constexpr Register COMPUTE_PERFCOUNT_ENABLE_ADDR =
      REG_32B_ADDR(GC, 0, mmCOMPUTE_PERFCOUNT_ENABLE);
  static constexpr Register RLC_PERFMON_CLK_CNTL_ADDR = REG_32B_ADDR(GC, 0, mmRLC_PERFMON_CLK_CNTL);
  static constexpr Register CP_PERFMON_CNTL_ADDR = REG_32B_ADDR(GC, 0, mmCP_PERFMON_CNTL);

  static const uint32_t MC_PERFCOUNTER_RSLT_CNTL__ENABLE_ANY_MASK_PRM = 0x01000000L;
  static const uint32_t MC_PERFCOUNTER_RSLT_CNTL__CLEAR_ALL_MASK_PRM = 0x02000000L;

  static constexpr Register SPI_SQG_EVENT_CTL_ADDR{};
  static constexpr Register SQ_PERFCOUNTER_CTRL_ADDR = REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL);
  static constexpr Register SQ_PERFCOUNTER_CTRL2_ADDR{};
  static constexpr Register COMPUTE_THREAD_TRACE_ENABLE_ADDR{};
  static constexpr Register SQ_PERFCOUNTER_MASK_ADDR = REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_MASK);
  static constexpr Register SQ_THREAD_TRACE_MASK_ADDR = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_MASK);
  static constexpr Register SQ_THREAD_TRACE_PERF_MASK_ADDR =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_PERF_MASK);
  static constexpr Register SQ_THREAD_TRACE_TOKEN_MASK_ADDR =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_TOKEN_MASK);
  static constexpr Register SQ_THREAD_TRACE_TOKEN_MASK2_ADDR =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_TOKEN_MASK2);
  static constexpr Register SQ_THREAD_TRACE_MODE_ADDR = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_MODE);
  static constexpr Register SQ_THREAD_TRACE_BUF0_BASE_LO_ADDR{};
  static constexpr Register SQ_THREAD_TRACE_BUF0_BASE_HI_ADDR{};
  static constexpr Register SQ_THREAD_TRACE_BUF0_SIZE_ADDR{};
  static constexpr Register SQ_THREAD_TRACE_BASE_ADDR = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_BASE);
  static constexpr Register SQ_THREAD_TRACE_BASE2_ADDR =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_BASE2);
  static constexpr Register SQ_THREAD_TRACE_SIZE_ADDR = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_SIZE);
  static constexpr Register SQ_THREAD_TRACE_CTRL_ADDR = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_CTRL);
  static constexpr Register SQ_THREAD_TRACE_HIWATER_ADDR =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_HIWATER);
  static const uint32_t SQ_THREAD_TRACE_HIWATER_VAL = 0x6;
  static constexpr Register SQ_THREAD_TRACE_STATUS_ADDR =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_STATUS);
  static constexpr Register SQ_THREAD_TRACE_CNTR_ADDR = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_CNTR);
  static constexpr Register SQ_THREAD_TRACE_WPTR_ADDR = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_WPTR);
  static constexpr Register SQ_THREAD_TRACE_STATUS_OFFSET = []() {
    Register reg = REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_STATUS);
    reg.offset -= UCONFIG_SPACE_START;
    return reg;
  }();
  static const uint32_t TT_BUFF_ALIGN_SHIFT = 12;

  static const uint32_t SDMA_COUNTER_BLOCK_NUM_INSTANCES = SdmaCounterBlockMaxInstances;
  static const uint32_t UMC_COUNTER_BLOCK_NUM_INSTANCES = UmcCounterBlockMaxInstances;

  static constexpr Register RLC_SPM_PERFMON_CNTL__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_PERFMON_CNTL);
  static constexpr Register RLC_SPM_MC_CNTL__ADDR = REG_32B_ADDR(GC, 0, mmRLC_SPM_MC_CNTL);
  static constexpr Register RLC_SPM_PERFMON_RING_BASE_LO__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_PERFMON_RING_BASE_LO);
  static constexpr Register RLC_SPM_PERFMON_RING_BASE_HI__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_PERFMON_RING_BASE_HI);
  static constexpr Register RLC_SPM_PERFMON_RING_SIZE__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_PERFMON_RING_SIZE);
  static constexpr Register RLC_SPM_PERFMON_SEGMENT_SIZE__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_PERFMON_SEGMENT_SIZE);
#if defined(regRLC_SPM_PERFMON_SEGMENT_SIZE_CORE1)
  static constexpr Register RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1__ADDR =
      REG_32B_ADDR(GC, 0, regRLC_SPM_PERFMON_SEGMENT_SIZE_CORE1);
#else
  static constexpr Register RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1__ADDR = Register(0xDCAF);
#endif
  static constexpr Register RLC_SPM_GLOBAL_MUXSEL_ADDR__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_GLOBAL_MUXSEL_ADDR);
  static constexpr Register RLC_SPM_GLOBAL_MUXSEL_DATA__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_GLOBAL_MUXSEL_DATA);
  static constexpr Register RLC_SPM_SE_MUXSEL_ADDR__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_SE_MUXSEL_ADDR);
  static constexpr Register RLC_SPM_SE_MUXSEL_DATA__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_SE_MUXSEL_DATA);
  static constexpr Register RLC_SPM_PERFMON_SAMPLE_DELAY_MAX__ADDR =
      REG_32B_ADDR(GC, 0, mmRLC_SPM_PERFMON_SAMPLE_DELAY_MAX);
  static const uint32_t RLC_SPM_COUNTERS_PER_LINE = 16;
  static const uint32_t RLC_SPM_TIMESTAMP_SIZE16 = 4;

  static constexpr Register SQ_THREAD_TRACE_USERDATA_0 =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_USERDATA_0);
  static constexpr Register SQ_THREAD_TRACE_USERDATA_1 =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_USERDATA_1);
  static constexpr Register SQ_THREAD_TRACE_USERDATA_2 =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_USERDATA_2);
  static constexpr Register SQ_THREAD_TRACE_USERDATA_3 =
      REG_32B_ADDR(GC, 0, mmSQ_THREAD_TRACE_USERDATA_3);

  static Register sqtt_perfcounter_addr(uint32_t index) {
    static const Register SQTT_PERFCOUNTERS_SELECT[16] = {
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER0_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER1_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER2_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER3_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER4_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER5_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER6_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER7_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER8_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER9_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER10_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER11_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER12_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER13_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER14_SELECT),
        REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER15_SELECT)};
    return SQTT_PERFCOUNTERS_SELECT[index & 0xF];
  }

  union mux_info_t {
    uint16_t data;
    struct {
      uint16_t counter : 6;
      uint16_t block : 5;
      uint16_t instance : 5;
    } gfx;
  };

  static const uint32_t SQ_BLOCK_ID = SqCounterBlockId;
  static const uint32_t SQ_BLOCK_SPM_ID = 9;

  static const uint32_t COPY_DATA_SEL_REG_PRM = COPY_DATA_SEL_REG;
  static const uint32_t COPY_DATA_SEL_SRC_SYS_PERF_COUNTER_PRM = COPY_DATA_SEL_SRC_SYS_PERF_COUNTER;
  static const uint32_t COPY_DATA_SEL_COUNT_1DW_PRM = COPY_DATA_SEL_COUNT_1DW;

  static uint32_t Low32(const uint64_t& v) { return (uint32_t)v; }
  static uint32_t High32(const uint64_t& v) { return (uint32_t)(v >> 32); }

  // SPM delay functions for global instance
  static uint32_t get_spm_global_delay(const counter_des_t& counter_des,
                                       const uint32_t& instance_index) {
    const auto* block_info = counter_des.block_info;
    return block_info->delay_info.val[instance_index];
  }

  // SPM delay functions for se instance
  static uint32_t get_spm_se_delay(const counter_des_t& counter_des, const uint32_t& se_index,
                                   const uint32_t& instance_index) {
    const auto* block_info = counter_des.block_info;
    int delay_index = se_index * block_info->instance_count + instance_index;
    return block_info->delay_info.val[delay_index];
  }

  // GRBM broadcasting mode
  static uint32_t grbm_broadcast_value() {
    uint32_t grbm_gfx_index = SET_REG_FIELD_BITS(GRBM_GFX_INDEX, INSTANCE_BROADCAST_WRITES, 1) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SE_BROADCAST_WRITES, 1) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SH_BROADCAST_WRITES, 1);
    return grbm_gfx_index;
  }

  // GRBM SE indexing
  static uint32_t grbm_inst_index_value(const uint32_t& instance_index) {
    uint32_t grbm_gfx_index = SET_REG_FIELD_BITS(GRBM_GFX_INDEX, INSTANCE_INDEX, instance_index) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SE_BROADCAST_WRITES, 1) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SH_BROADCAST_WRITES, 1);
    return grbm_gfx_index;
  }

  // GRBM SE indexing
  static uint32_t grbm_se_index_value(const uint32_t& se_index) {
    uint32_t grbm_gfx_index = SET_REG_FIELD_BITS(GRBM_GFX_INDEX, INSTANCE_BROADCAST_WRITES, 1) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SE_INDEX, se_index) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SH_BROADCAST_WRITES, 1);
    return grbm_gfx_index;
  }

  // GRBM SE/BlockInstance indexing
  static uint32_t grbm_inst_se_index_value(const uint32_t& instance_index,
                                           const uint32_t& se_index) {
    uint32_t grbm_gfx_index = SET_REG_FIELD_BITS(GRBM_GFX_INDEX, INSTANCE_INDEX, instance_index) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SE_INDEX, se_index) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SH_BROADCAST_WRITES, 1);
    return grbm_gfx_index;
  }

  // GRBM SE/SH indexing
  static uint32_t grbm_se_sh_index_value(const uint32_t& se_index, const uint32_t& sh_index) {
    uint32_t grbm_gfx_index = SET_REG_FIELD_BITS(GRBM_GFX_INDEX, INSTANCE_BROADCAST_WRITES, 1) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SE_INDEX, se_index) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SH_INDEX, sh_index);
    return grbm_gfx_index;
  }

  // GRBM SH/SE/BlockInstance indexing
  static uint32_t grbm_inst_se_sh_index_value(const uint32_t& instance_index,
                                              const uint32_t& se_index, const uint32_t& sh_index) {
    uint32_t grbm_gfx_index = SET_REG_FIELD_BITS(GRBM_GFX_INDEX, INSTANCE_INDEX, instance_index) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SE_INDEX, se_index) |
                              SET_REG_FIELD_BITS(GRBM_GFX_INDEX, SH_INDEX, sh_index);
    return grbm_gfx_index;
  }

  // GRBM SE/SH/WGP indexing
  static uint32_t grbm_se_sh_wgp_index_value(const uint32_t&, const uint32_t&, const uint32_t&) { return 0; }
  // GRBM SE/SH/WGP/BlockInstance indexing
  static uint32_t grbm_inst_se_sh_wgp_index_value(const uint32_t&, const uint32_t&, const uint32_t&, const uint32_t&) { return 0; }

  // CP_PERFMON_CNTL value to reset counters
  static uint32_t cp_perfmon_cntl_reset_value() {
    uint32_t cp_perfmon_cntl{0};
    return cp_perfmon_cntl;
  }

  // CP_PERFMON_CNTL value to start counters
  static uint32_t cp_perfmon_cntl_start_value() {
    uint32_t cp_perfmon_cntl = SET_REG_FIELD_BITS(CP_PERFMON_CNTL, PERFMON_STATE, 1);
    return cp_perfmon_cntl;
  }

  // CP_PERFMON_CNTL value to stop/freeze counters
  static uint32_t cp_perfmon_cntl_stop_value() {
    uint32_t cp_perfmon_cntl = SET_REG_FIELD_BITS(CP_PERFMON_CNTL, PERFMON_STATE, 2);
    return cp_perfmon_cntl;
  }

  // CP_PERFMON_CNTL value to stop/freeze counters
  static uint32_t cp_perfmon_cntl_read_value() {
    uint32_t cp_perfmon_cntl = SET_REG_FIELD_BITS(CP_PERFMON_CNTL, PERFMON_STATE, 1) |
                               SET_REG_FIELD_BITS(CP_PERFMON_CNTL, PERFMON_SAMPLE_ENABLE, 1);
    return cp_perfmon_cntl;
  }

  // Compute Perfcount Enable register value to enable counting
  static uint32_t cp_perfcount_enable_value() {
    uint32_t compute_perfcount_enable =
        SET_REG_FIELD_BITS(COMPUTE_PERFCOUNT_ENABLE, PERFCOUNT_ENABLE, 1);
    return compute_perfcount_enable;
  }

  // Compute Perfcount Disable register value to enable counting
  static uint32_t cp_perfcount_disable_value() {
    uint32_t compute_perfcount_enable =
        SET_REG_FIELD_BITS(COMPUTE_PERFCOUNT_ENABLE, PERFCOUNT_ENABLE, 0);
    return compute_perfcount_enable;
  }

  // SQ Block primitives

  // SQ Counter Select Register value
  static uint32_t sq_select_value(const counter_des_t& counter_des) {
    uint32_t sq_perfcounter0_select =
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, SIMD_MASK, 0xF) |
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, SQC_BANK_MASK, 0xF) |
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, SQC_CLIENT_MASK, 0xF) |
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, PERF_SEL, counter_des.id);
    return sq_perfcounter0_select;
  }
  static uint32_t sq_spm_select_value(const counter_des_t& counter_des) {
    uint32_t sq_perfcounter0_select =
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, SIMD_MASK, 0xF) |
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, SQC_BANK_MASK, 0xF) |
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, SQC_CLIENT_MASK, 0xF) |
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, PERF_SEL, counter_des.id) |
        SET_REG_FIELD_BITS(SQ_PERFCOUNTER0_SELECT, SPM_MODE, 3);  // PERFMON_SPM_MODE_32BIT_CLAMP
    return sq_perfcounter0_select;
  }

  // SQ Counter Mask Register value
  static uint32_t sq_mask_value(const counter_des_t&) {
    uint32_t sq_perfcounter_mask = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_MASK, SH0_MASK, 0xFFFF) |
                                   SET_REG_FIELD_BITS(SQ_PERFCOUNTER_MASK, SH1_MASK, 0xFFFF);
    return sq_perfcounter_mask;
  }

  // SQ Counter Control Register value
  static uint32_t sq_control_value(const counter_des_t& counter_des) {
    const uint32_t block_id = counter_des.block_des.id;
    uint32_t sq_perfcounter_ctrl{0};
    if (block_id == SqCounterBlockId) {
      sq_perfcounter_ctrl = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, GS_EN, 0x1) |
                            SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, VS_EN, 0x1) |
                            SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, PS_EN, 0x1) |
                            SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, HS_EN, 0x1) |
                            SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, CS_EN, 0x1);
    } else if (block_id == SqGsCounterBlockId) {
      sq_perfcounter_ctrl = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, GS_EN, 0x1);
    } else if (block_id == SqVsCounterBlockId) {
      sq_perfcounter_ctrl = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, VS_EN, 0x1);
    } else if (block_id == SqPsCounterBlockId) {
      sq_perfcounter_ctrl = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, PS_EN, 0x1);
    } else if (block_id == SqHsCounterBlockId) {
      sq_perfcounter_ctrl = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, HS_EN, 0x1);
    } else if (block_id == SqCsCounterBlockId) {
      sq_perfcounter_ctrl = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, CS_EN, 0x1);
    }
#if defined(SQ_PERFCOUNTER_CTRL__VMID_MASK__SHIFT)
    sq_perfcounter_ctrl |= SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, VMID_MASK, 0xFFFF);
#else
    sq_perfcounter_ctrl |= 0xFFFF0000;
#endif
    return sq_perfcounter_ctrl;
  }

  // SQ validate counter attributes
  static void validate_counters(uint32_t counters_vec_attr) {
#if SQ_CONFLICT_CHECK == 1
    const uint32_t mask = CounterBlockSqAttr | CounterBlockTcAttr;
    const bool conflict = ((counters_vec_attr & mask) == mask);
    if (conflict) abort();
#endif
  }

  // SQ Counter Control enable perfomance counter in graphics pipeline stages
  static uint32_t sq_control_enable_value() {
    uint32_t sq_perfcounter_ctrl = SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, PS_EN, 0x1) |
                                   SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, VS_EN, 0x1) |
                                   SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, GS_EN, 0x1) |
                                   SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, ES_EN, 0x1) |
                                   SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, HS_EN, 0x1) |
                                   SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, LS_EN, 0x1) |
                                   SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, CS_EN, 0x1);
#if defined(SQ_PERFCOUNTER_CTRL__VMID_MASK__SHIFT)
    sq_perfcounter_ctrl |= SET_REG_FIELD_BITS(SQ_PERFCOUNTER_CTRL, VMID_MASK, 0xFFFF);
#else
    sq_perfcounter_ctrl |= 0xFFFF0000;
#endif
    return sq_perfcounter_ctrl;
  }
  static uint32_t sq_control2_enable_value() { return 0; }
  static uint32_t sq_control2_disable_value() { return 0; }

  // MC Block primitives

  // MC Channel value
  static uint32_t mc_config_value(const counter_des_t& counter_des) { return counter_des.index; }

  // MC registers values
  static auto constexpr mc_select_value_MC_VM_L2_PERFCOUNTER0_CFG =
      mc_select_value(MC_VM_L2_PERFCOUNTER0_CFG);
  static auto constexpr mc_select_value_ATC_L2_PERFCOUNTER0_CFG =
      mc_select_value(ATC_L2_PERFCOUNTER0_CFG);
  static auto constexpr mc_select_value_ATC_PERFCOUNTER0_CFG =
      mc_select_value(ATC_PERFCOUNTER0_CFG);
  static auto constexpr mc_select_value_GCEA_PERFCOUNTER0_CFG =
      mc_select_value(GCEA_PERFCOUNTER0_CFG);
  static auto constexpr mc_select_value_RPB_PERFCOUNTER0_CFG =
      mc_select_value(RPB_PERFCOUNTER0_CFG);

  static uint32_t mc_reset_value() { return MC_PERFCOUNTER_RSLT_CNTL__CLEAR_ALL_MASK_PRM; }
  static uint32_t mc_start_value() { return MC_PERFCOUNTER_RSLT_CNTL__ENABLE_ANY_MASK_PRM; }

  static auto constexpr select_value_CB_PERFCOUNTER0_SELECT = select_value(CB_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_DB_PERFCOUNTER0_SELECT = select_value(DB_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_GRBM_PERFCOUNTER0_SELECT =
      select_value(GRBM_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_GRBM_SE0_PERFCOUNTER_SELECT =
      select_value(GRBM_SE0_PERFCOUNTER_SELECT);
  static auto constexpr select_value_PA_SU_PERFCOUNTER0_SELECT =
      select_value(PA_SU_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_PA_SC_PERFCOUNTER0_SELECT =
      select_value(PA_SC_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_SPI_PERFCOUNTER0_SELECT =
      select_value(SPI_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_TA_PERFCOUNTER0_SELECT = select_value(TA_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_TCA_PERFCOUNTER0_SELECT =
      select_value(TCA_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_TCC_PERFCOUNTER0_SELECT =
      select_value(TCC_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_TD_PERFCOUNTER0_SELECT = select_value(TD_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_TCP_PERFCOUNTER0_SELECT =
      select_value(TCP_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_VGT_PERFCOUNTER0_SELECT =
      select_value(VGT_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_IA_PERFCOUNTER0_SELECT = select_value(IA_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_WD_PERFCOUNTER0_SELECT = select_value(WD_PERFCOUNTER0_SELECT);

  // static auto constexpr select_value_SX_PERFCOUNTER0_SELECT =
  // select_value_t2(SX_PERFCOUNTER0_SELECT); static auto constexpr
  // select_value_GDS_PERFCOUNTER0_SELECT = select_value_t2(GDS_PERFCOUNTER0_SELECT);

  static auto constexpr select_value_SX_PERFCOUNTER0_SELECT = [](const counter_des_t& counter_des) {
    return (uint32_t)0;
  };
  static auto constexpr select_value_GDS_PERFCOUNTER0_SELECT =
      [](const counter_des_t& counter_des) { return (uint32_t)0; };

  static auto constexpr select_value_CPC_PERFCOUNTER0_SELECT =
      select_value_t3(CPC_PERFCOUNTER0_SELECT);
  static auto constexpr select_value_CPF_PERFCOUNTER0_SELECT =
      select_value_t3(CPF_PERFCOUNTER0_SELECT);

  static uint32_t spm_select_value(const counter_des_t& counter_des) {
    uint32_t tcc_perfcounter0_select =
        SET_REG_FIELD_BITS(TCC_PERFCOUNTER0_SELECT, PERF_SEL, counter_des.id) |
        SET_REG_FIELD_BITS(TCC_PERFCOUNTER0_SELECT, CNTR_MODE, 3);  // PERFMON_SPM_MODE_32BIT_CLAMP
    return tcc_perfcounter0_select;
  }
  static uint32_t spm_even_select_value(const counter_des_t& counter_des) {
    uint32_t tcc_perfcounter0_select =
        SET_REG_FIELD_BITS(TCC_PERFCOUNTER0_SELECT, PERF_SEL, counter_des.id) |
        SET_REG_FIELD_BITS(TCC_PERFCOUNTER0_SELECT, CNTR_MODE, 1);  // PERFMON_SPM_MODE_32BIT_CLAMP
    return tcc_perfcounter0_select;
  }
  static uint32_t spm_odd_select_value(const counter_des_t& counter_des) {
    uint32_t tcc_perfcounter0_select =
        SET_REG_FIELD_BITS(TCC_PERFCOUNTER0_SELECT, PERF_SEL1, counter_des.id) |
        SET_REG_FIELD_BITS(TCC_PERFCOUNTER0_SELECT, CNTR_MODE, 1);  // PERFMON_SPM_MODE_32BIT_CLAMP
    return tcc_perfcounter0_select;
  }
  static mux_info_t spm_mux_ram_value(const counter_des_t& counter_des) {
    mux_info_t mxinfo{0};
    mxinfo.gfx.counter = counter_des.index;
    mxinfo.gfx.block = counter_des.block_info->spm_block_id;
    mxinfo.gfx.instance = counter_des.block_des.index;
    return mxinfo;
  }
  static mux_info_t spm_mux_ram_value(uint16_t counter, uint16_t block, uint16_t instance) {
    mux_info_t mxinfo{0};
    mxinfo.gfx.counter = counter;
    mxinfo.gfx.block = block;
    mxinfo.gfx.instance = instance;
    return mxinfo;
  }
  static uint32_t spm_mux_ram_idx_incr(uint32_t idx) {
    uint32_t incr_idx = ++idx;
    if (!(incr_idx % RLC_SPM_COUNTERS_PER_LINE)) incr_idx += RLC_SPM_COUNTERS_PER_LINE;
    return incr_idx;
  }

  // SDMA primitives
  static uint32_t sdma_disable_clear_value() { return 0; }

  static uint32_t sdma_enable_value() { return 0; }

  static uint32_t sdma_select_value(const counter_des_t& counter_des) { return 0; }

  static uint32_t sdma_stop_value(const counter_des_t& counter_des) { return 0; }

  // SPM trace routines
  static uint32_t rlc_spm_mc_cntl_value() {
    uint32_t rlc_spm_mc_cntl = SET_REG_FIELD_BITS(RLC_SPM_MC_CNTL, RLC_SPM_VMID, 15);
    return rlc_spm_mc_cntl;
  }
  static uint32_t cp_perfmon_cntl_spm_start_value() {
    uint32_t cp_perfmon_cntl = SET_REG_FIELD_BITS(CP_PERFMON_CNTL, SPM_PERFMON_STATE, 1);
    return cp_perfmon_cntl;
  }
  static uint32_t cp_perfmon_cntl_spm_stop_value() {
    uint32_t cp_perfmon_cntl = SET_REG_FIELD_BITS(CP_PERFMON_CNTL, SPM_PERFMON_STATE, 2);
    return cp_perfmon_cntl;
  }

  static uint32_t rlc_spm_muxsel_data(const uint32_t& value, const counter_des_t& counter_des,
                                      const uint32_t& block, const uint32_t& hi) {
    return 0;
  }

  static uint32_t rlc_spm_perfmon_cntl_value(const uint32_t& sampling_rate) {
    uint32_t rlc_spm_perfmon_cntl =
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_CNTL, PERFMON_SAMPLE_INTERVAL, sampling_rate);
    return rlc_spm_perfmon_cntl;
  }
  static uint32_t rlc_spm_perfmon_segment_size_value(const uint32_t& global_count,
                                                     const uint32_t& se_count) {
    const uint32_t global_nlines = global_count;
    const uint32_t se_nlines = se_count;
    const uint32_t segment_size = (global_nlines + (4 * se_nlines));
    uint32_t rlc_spm_perfmon_segment_size =
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE, GLOBAL_NUM_LINE, global_nlines) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE, SE0_NUM_LINE, se_nlines) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE, SE1_NUM_LINE, se_nlines) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE, SE2_NUM_LINE, se_nlines) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE, PERFMON_SEGMENT_SIZE, segment_size);
    return rlc_spm_perfmon_segment_size;
  }

  static uint32_t rlc_spm_perfmon_segment_size_core1_value(const uint32_t& se_count) {
    const uint32_t se_nlines = se_count;
    const uint32_t segment_size = 4 * se_nlines;
    uint32_t rlc_spm_perfmon_segment_size_core1{0};
#if defined(RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1__PERFMON_SEGMENT_SIZE_CORE1__SHIFT)
    rlc_spm_perfmon_segment_size_core1 =
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1, PERFMON_SEGMENT_SIZE_CORE1,
                           segment_size) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1, SE4_NUM_LINE, se_nlines) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1, SE5_NUM_LINE, se_nlines) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1, SE6_NUM_LINE, se_nlines) |
        SET_REG_FIELD_BITS(RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1, SE7_NUM_LINE, se_nlines);
#endif
    return rlc_spm_perfmon_segment_size_core1;
  }

  // Enable Thread Trace for all VM Id's
  // Enable all of the SIMD's of the compute unit
  // Enable Compute Unit (CU) at index Zero to be used for fine-grained data
  // Enable Shader Array (SH) at index Zero to be used for fine-grained data
  //
  // @note: Not enabling REG_STALL_EN, SPI_STALL_EN and SQ_STALL_EN bits. They
  // are useful if we wish to program buffer throttling.
  //
  static uint32_t sqtt_mask_value(uint32_t targetCu, uint32_t simd, uint32_t vmIdMask) {
    uint32_t sq_thread_trace_mask = SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, SH_SEL, 0x0) |
                                    SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, SIMD_EN, simd) |
                                    SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, CU_SEL, targetCu) |
                                    SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, SQ_STALL_EN, 0x1) |
                                    SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, SPI_STALL_EN, 0x1) |
                                    SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, REG_STALL_EN, 0x1) |
                                    SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, VM_ID_MASK, vmIdMask);
    return sq_thread_trace_mask;
  }

  // Mask of compute units to get thread trace data from
  static uint32_t sqtt_perf_mask_value() {
    uint32_t sq_thread_trace_perf_mask =
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_PERF_MASK, SH0_MASK, 0xFFFF) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_PERF_MASK, SH1_MASK, 0xFFFF);
    return sq_thread_trace_perf_mask;
  }

  // Indicate the different TT messages/tokens that should be enabled/logged
  // Indicate the different TT tokens that specify register operations to be logged

  static const uint32_t SQTT_TOKEN_MISC = 1 << 0;
  static const uint32_t SQTT_TOKEN_TIME = 1 << 1;
  static const uint32_t SQTT_TOKEN_REG = 1 << 2;
  static const uint32_t SQTT_TOKEN_WAVE_START = 1 << 3;
  static const uint32_t SQTT_TOKEN_REG_CS = 1 << 5;
  static const uint32_t SQTT_TOKEN_WAVE_END = 1 << 6;
  static const uint32_t SQTT_TOKEN_INST = 1 << 10;
  static const uint32_t SQTT_TOKEN_INST_PC = 1 << 11;
  static const uint32_t SQTT_TOKEN_USERDATA = 1 << 12;
  static const uint32_t SQTT_TOKEN_ISSUE = 1 << 13;
  static const uint32_t SQTT_TOKEN_REG_CS_PRIV = 1 << 15;

  static uint32_t sqtt_token_mask_on_value() {
    uint32_t sq_thread_trace_token_mask;
    uint32_t sq_thread_trace_token_mask_token_mask =
        SQTT_TOKEN_MISC | SQTT_TOKEN_TIME | SQTT_TOKEN_REG | SQTT_TOKEN_WAVE_START |
        SQTT_TOKEN_WAVE_END | SQTT_TOKEN_INST | SQTT_TOKEN_INST_PC | SQTT_TOKEN_USERDATA |
        SQTT_TOKEN_ISSUE | SQTT_TOKEN_REG_CS | SQTT_TOKEN_REG_CS_PRIV;

    sq_thread_trace_token_mask = SET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK, REG_MASK, 0xF) |
                                 SET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK, TOKEN_MASK,
                                                    sq_thread_trace_token_mask_token_mask);
    return sq_thread_trace_token_mask;
  }

  static uint32_t sqtt_token_mask_off_value() {
    uint32_t sq_thread_trace_token_mask =
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK, REG_MASK, 0x0) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK, TOKEN_MASK, 0xF);
    return sq_thread_trace_token_mask;
  }

  static uint32_t sqtt_token_mask_occupancy_value() {
    uint32_t sq_thread_trace_token_mask;
    uint32_t sq_thread_trace_token_mask_token_mask =
        SQTT_TOKEN_MISC | SQTT_TOKEN_TIME | SQTT_TOKEN_REG | SQTT_TOKEN_WAVE_START |
        SQTT_TOKEN_WAVE_END | SQTT_TOKEN_REG_CS_PRIV | SQTT_TOKEN_REG_CS | SQTT_TOKEN_USERDATA;

    sq_thread_trace_token_mask = SET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK, REG_MASK, 0xF) |
                                 SET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK, TOKEN_MASK,
                                                    sq_thread_trace_token_mask_token_mask);
    return sq_thread_trace_token_mask;
  }

  // Indicate the different TT tokens that specify instruction operations to be logged
  // Disabling specifically instruction operations updating Program Counter (PC).
  // @note: The field is defined in the spec incorrectly as a 16-bit value
  static uint32_t sqtt_token_mask2_value() {
    uint32_t sq_thread_trace_token_mask2 =
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK2, INST_MASK, 0xFFFFFFFF);
    return sq_thread_trace_token_mask2;
  }

  // Check if stalling is supported
  static bool sqtt_stalling_enabled(const uint32_t& mask_val, const uint32_t& token_mask_val) {
    return GET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, SQ_STALL_EN, mask_val) ||
           GET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, SPI_STALL_EN, mask_val) ||
           GET_REG_FIELD_BITS(SQ_THREAD_TRACE_MASK, REG_STALL_EN, mask_val) ||
           GET_REG_FIELD_BITS(SQ_THREAD_TRACE_TOKEN_MASK, REG_DROP_ON_STALL, token_mask_val);
  }

  // Indicates various attributes of a thread trace session.
  //
  // MASK_CS: Which shader types should be enabled for data collection
  //      Enable CS Shader types.
  //
  // WRAP: How trace buffer should be used as a ring buffer or as a linear
  //      buffer - Disable WRAP mode i.e use it as a linear buffer
  //
  // MODE: Enables a thread trace session
  //
  // CAPTURE_MODE: When thread trace data is collected immediately after MODE
  //      is enabled or wait until a Thread Trace Start event is received
  //
  // AUTOFLUSH_EN: Flush thread trace data to buffer often automatically
  //
  // Thread trace mode OFF value
  static uint32_t sqtt_mode_off_value() {
    uint32_t sq_thread_trace_mode =
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, WRAP, 0) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, CAPTURE_MODE, 0) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, MASK_CS, 1) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, AUTOFLUSH_EN, 1) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, MODE, SQ_THREAD_TRACE_MODE_OFF);
    return sq_thread_trace_mode;
  }
  // Thread trace mode ON value
  static uint32_t sqtt_mode_on_value() {
    uint32_t sq_thread_trace_mode =
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, WRAP, 0) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, CAPTURE_MODE, 0) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, MASK_CS, 1) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, AUTOFLUSH_EN, 1) |
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_MODE, MODE, SQ_THREAD_TRACE_MODE_ON);
    return sq_thread_trace_mode;
  }

  // Base address of buffer to use for thread trace
  static uint32_t sqtt_base_value_lo(const uint64_t& base_addr) {
    uint32_t sq_thread_trace_base =
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_BASE, ADDR, Low32(base_addr >> TT_BUFF_ALIGN_SHIFT));
    return sq_thread_trace_base;
  }
  static uint32_t sqtt_base_value_hi(const uint64_t& base_addr) {
    uint32_t sq_thread_trace_base = SET_REG_FIELD_BITS(SQ_THREAD_TRACE_BASE2, ADDR_HI,
                                                       High32(base_addr >> TT_BUFF_ALIGN_SHIFT));
    return sq_thread_trace_base;
  }

  // Indicates the size of buffer to use per Shader Engine instance.
  // The size is specified in terms of 4KB blocks
  static uint32_t sqtt_buffer_size_value(uint32_t size_val, uint32_t base_hi) {
    uint32_t sq_thread_trace_size =
        SET_REG_FIELD_BITS(SQ_THREAD_TRACE_SIZE, SIZE, (size_val >> TT_BUFF_ALIGN_SHIFT));
    return sq_thread_trace_size;
  }

  static uint32_t sqtt_buffer0_size_value(uint32_t size_val) { return 0; }

  static uint32_t spi_sqg_event_ctl(bool enableSqgEvents) { return 0; }

  static uint32_t sqtt_zero_size_value() { return 0; }

  // Thread trace ctrl register value
  static uint32_t sqtt_ctrl_value(bool on) {
    uint32_t sq_thread_trace_ctrl = SET_REG_FIELD_BITS(SQ_THREAD_TRACE_CTRL, RESET_BUFFER, 1);
    return sq_thread_trace_ctrl;
  }

  // SPM primitives
  static uint16_t spm_timestamp_muxsel() { return 0xF0F0; }

  enum ESQTT_STATUS_MASK {
    // Mask to check if memory error was received
    TT_CONTROL_UTC_ERR_MASK = 0x10000000,
    // Mask to check if SQTT buffer is wrapped
    TT_CONTROL_FULL_MASK = 0x80000000,
    TT_WRITE_PTR_MASK = 0x3FFFFFFF
  };

  static uint32_t sqtt_busy_mask() {
    const uint32_t BUSY_BIT = 30;
    return 1u << BUSY_BIT;
  }

  static uint32_t sqtt_pending_mask() {
    const uint32_t NUM_PIPES = 8;
    return (1u << NUM_PIPES) - 1;
  }
};

}  // namespace gfx9
}  // namespace gfxip

#endif  // _GFX9_PRIMITIVES_H_
