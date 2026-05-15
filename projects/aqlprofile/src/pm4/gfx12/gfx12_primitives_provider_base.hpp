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

// Shared GFX12 primitives provider implementation parameterised on the variant
// primitives class (gfxip::gfx12::gfx1200::Primitives or
// gfxip::gfx12::gfx1250::Primitives).  All methods are inline so no
// separate .cpp compilation unit is required.

#ifndef SRC_PM4_GFX12_PRIMITIVES_PROVIDER_BASE_HPP_
#define SRC_PM4_GFX12_PRIMITIVES_PROVIDER_BASE_HPP_

#include "pm4/primitives_provider.hpp"

namespace pm4_builder {

template <typename Prim>
class Gfx12PrimitivesProviderBase : public PrimitivesProvider {
 public:
  // ---- Config constants -------------------------------------------------------

  uint32_t GetGfxipLevel() const override { return Prim::GFXIP_LEVEL; }
  uint32_t GetNumberOfBlocks() const override { return Prim::NUMBER_OF_BLOCKS; }
  uint32_t GetSdmaCounterBlockNumInstances() const override {
    return Prim::SDMA_COUNTER_BLOCK_NUM_INSTANCES;
  }
  uint32_t GetRlcSpmCountersPerLine() const override { return Prim::RLC_SPM_COUNTERS_PER_LINE; }
  uint32_t GetRlcSpmTimestampSize16() const override { return Prim::RLC_SPM_TIMESTAMP_SIZE16; }
  uint32_t GetSqBlockId() const override { return Prim::SQ_BLOCK_ID; }
  uint32_t GetSqBlockSpmId() const override { return Prim::SQ_BLOCK_SPM_ID; }
  size_t   GetTtBuffAlignShift() const override { return Prim::TT_BUFF_ALIGN_SHIFT; }
  uint32_t GetSqThreadTraceHiwaterVal() const override {
    return Prim::SQ_THREAD_TRACE_HIWATER_VAL;
  }
  size_t GetTtControlUtcErrMask() const override { return Prim::TT_CONTROL_UTC_ERR_MASK; }
  size_t GetTtControlFullMask() const override { return Prim::TT_CONTROL_FULL_MASK; }
  size_t GetTtLockdownFailMask() const override { return Prim::TT_LOCKDOWN_FAIL; }
  size_t GetTtWritePtrMask() const override { return Prim::TT_WRITE_PTR_MASK; }
  uint32_t GetCopyDataSelCountOneDw() const override { return Prim::COPY_DATA_SEL_COUNT_1DW_PRM; }

  // ---- Capability flags -------------------------------------------------------

  bool SupportsGusCounters() const override { return false; }
  bool HasRlcSpmCore1() const override { return false; }

  // ---- Register addresses -----------------------------------------------------

  Register GetGrbmGfxIndexAddr() const override { return Prim::GRBM_GFX_INDEX_ADDR; }
  Register GetGrbmaGfxIndexAddr() const override { return Prim::GRBMA_GFX_INDEX_ADDR; }
  Register GetCpPerfmonCntlAddr() const override { return Prim::CP_PERFMON_CNTL_ADDR; }
  Register GetRlcPerfmonClkCntlAddr() const override { return Prim::RLC_PERFMON_CLK_CNTL_ADDR; }
  Register GetComputePerfcountEnableAddr() const override {
    return Prim::COMPUTE_PERFCOUNT_ENABLE_ADDR;
  }
  Register GetAidPerfmonCntlAddr() const override { return Prim::AID_PERFMON_CNTL_ADDR; }
  Register GetSqPerfcounterCtrlAddr() const override { return Prim::SQ_PERFCOUNTER_CTRL_ADDR; }
  Register GetSqPerfcounterCtrl2Addr() const override { return Prim::SQ_PERFCOUNTER_CTRL2_ADDR; }
  Register GetSqPerfcounterMaskAddr() const override { return Prim::SQ_PERFCOUNTER_MASK_ADDR; }
  Register GetSpiSqgEventCtlAddr() const override { return Prim::SPI_SQG_EVENT_CTL_ADDR; }
  Register GetComputeThreadTraceEnableAddr() const override {
    return Prim::COMPUTE_THREAD_TRACE_ENABLE_ADDR;
  }
  Register GetGusRsltCntlAddr() const override {
    // GFX12 does not have a GUS result control register.
    return Register{};
  }
  Register GetRlcSpmPerfmonCntlAddr() const override { return Prim::RLC_SPM_PERFMON_CNTL__ADDR; }
  Register GetRlcSpmMcCntlAddr() const override { return Prim::RLC_SPM_MC_CNTL__ADDR; }
  Register GetRlcSpmPerfmonRingBaseLo() const override {
    return Prim::RLC_SPM_PERFMON_RING_BASE_LO__ADDR;
  }
  Register GetRlcSpmPerfmonRingBaseHi() const override {
    return Prim::RLC_SPM_PERFMON_RING_BASE_HI__ADDR;
  }
  Register GetRlcSpmPerfmonRingSize() const override {
    return Prim::RLC_SPM_PERFMON_RING_SIZE__ADDR;
  }
  Register GetRlcSpmPerfmonSegmentSize() const override {
    return Prim::RLC_SPM_PERFMON_SEGMENT_SIZE__ADDR;
  }
  Register GetRlcSpmPerfmonSegmentSizeCore1() const override {
    return Prim::RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1__ADDR;
  }
  Register GetRlcSpmGlobalMuxselAddr() const override {
    return Prim::RLC_SPM_GLOBAL_MUXSEL_ADDR__ADDR;
  }
  Register GetRlcSpmGlobalMuxselData() const override {
    return Prim::RLC_SPM_GLOBAL_MUXSEL_DATA__ADDR;
  }
  Register GetRlcSpmSeMuxselAddr() const override { return Prim::RLC_SPM_SE_MUXSEL_ADDR__ADDR; }
  Register GetRlcSpmSeMuxselData() const override { return Prim::RLC_SPM_SE_MUXSEL_DATA__ADDR; }
  Register GetRlcSpmPerfmonSampleDelayMax() const override {
    return Prim::RLC_SPM_PERFMON_SAMPLE_DELAY_MAX__ADDR;
  }
  Register GetSqThreadTraceMaskAddr() const override { return Prim::SQ_THREAD_TRACE_MASK_ADDR; }
  Register GetSqThreadTracePerfMaskAddr() const override {
    return Prim::SQ_THREAD_TRACE_PERF_MASK_ADDR;
  }
  Register GetSqThreadTraceTokenMaskAddr() const override {
    return Prim::SQ_THREAD_TRACE_TOKEN_MASK_ADDR;
  }
  Register GetSqThreadTraceTokenMask2Addr() const override {
    return Prim::SQ_THREAD_TRACE_TOKEN_MASK2_ADDR;
  }
  Register GetSqThreadTraceModeAddr() const override { return Prim::SQ_THREAD_TRACE_MODE_ADDR; }
  Register GetSqThreadTraceHiwaterAddr() const override {
    return Prim::SQ_THREAD_TRACE_HIWATER_ADDR;
  }
  Register GetSqThreadTraceBaseAddr() const override { return Prim::SQ_THREAD_TRACE_BASE_ADDR; }
  Register GetSqThreadTraceBuf0BaseLo() const override {
    return Prim::SQ_THREAD_TRACE_BUF0_BASE_LO_ADDR;
  }
  Register GetSqThreadTraceBuf0BaseHi() const override {
    return Prim::SQ_THREAD_TRACE_BUF0_BASE_HI_ADDR;
  }
  Register GetSqThreadTraceBuf0Size() const override {
    return Prim::SQ_THREAD_TRACE_BUF0_SIZE_ADDR;
  }
  Register GetSqThreadTraceBuf1BaseLo() const override {
    return Prim::SQ_THREAD_TRACE_BUF1_BASE_LO_ADDR;
  }
  Register GetSqThreadTraceBuf1BaseHi() const override {
    return Prim::SQ_THREAD_TRACE_BUF1_BASE_HI_ADDR;
  }
  Register GetSqThreadTraceBuf1Size() const override {
    return Prim::SQ_THREAD_TRACE_BUF1_SIZE_ADDR;
  }
  Register GetSqThreadTraceBase2Addr() const override { return Prim::SQ_THREAD_TRACE_BASE2_ADDR; }
  Register GetSqThreadTraceSizeAddr() const override { return Prim::SQ_THREAD_TRACE_SIZE_ADDR; }
  Register GetSqThreadTraceCtrlAddr() const override { return Prim::SQ_THREAD_TRACE_CTRL_ADDR; }
  Register GetSqThreadTraceStatusAddr() const override {
    return Prim::SQ_THREAD_TRACE_STATUS_ADDR;
  }
  Register GetSqThreadTraceStatusOffset() const override {
    return Prim::SQ_THREAD_TRACE_STATUS_OFFSET;
  }
  Register GetSqThreadTraceCntrAddr() const override { return Prim::SQ_THREAD_TRACE_CNTR_ADDR; }
  Register GetSqThreadTraceWptrAddr() const override { return Prim::SQ_THREAD_TRACE_WPTR_ADDR; }
  Register GetSqThreadTraceStatus2Addr() const override {
    return Prim::SQ_THREAD_TRACE_STATUS2_ADDR;
  }
  Register GetSqThreadTraceUserdata2() const override { return Prim::SQ_THREAD_TRACE_USERDATA_2; }
  Register GetSqThreadTraceUserdata3() const override { return Prim::SQ_THREAD_TRACE_USERDATA_3; }
  Register SqttPerfcounterAddr(uint32_t index) const override {
    return Prim::sqtt_perfcounter_addr(index);
  }

  // ---- GRBM index value functions ---------------------------------------------

  uint32_t GrbmBroadcastValue() const override { return Prim::grbm_broadcast_value(); }
  uint32_t GrbmInstIndexValue(uint32_t instance_index) const override {
    return Prim::grbm_inst_index_value(instance_index);
  }
  uint32_t GrbmSeIndexValue(uint32_t se_index) const override {
    return Prim::grbm_se_index_value(se_index);
  }
  uint32_t GrbmInstSeIndexValue(uint32_t instance_index, uint32_t se_index) const override {
    return Prim::grbm_inst_se_index_value(instance_index, se_index);
  }
  uint32_t GrbmSeShIndexValue(uint32_t se_index, uint32_t sh_index) const override {
    return Prim::grbm_se_sh_index_value(se_index, sh_index);
  }
  uint32_t GrbmInstSeShIndexValue(uint32_t inst, uint32_t se, uint32_t sh) const override {
    return Prim::grbm_inst_se_sh_index_value(inst, se, sh);
  }
  uint32_t GrbmSeShWgpIndexValue(uint32_t se, uint32_t sh, uint32_t wgp) const override {
    return Prim::grbm_se_sh_wgp_index_value(se, sh, wgp);
  }
  uint32_t GrbmInstSeShWgpIndexValue(uint32_t inst, uint32_t se, uint32_t sh,
                                     uint32_t wgp) const override {
    return Prim::grbm_inst_se_sh_wgp_index_value(inst, se, sh, wgp);
  }

  // ---- CP perfmon control values ----------------------------------------------

  uint32_t CpPerfmonCntlResetValue() const override {
    return Prim::cp_perfmon_cntl_reset_value();
  }
  uint32_t CpPerfmonCntlStartValue() const override {
    return Prim::cp_perfmon_cntl_start_value();
  }
  uint32_t CpPerfmonCntlStopValue() const override { return Prim::cp_perfmon_cntl_stop_value(); }
  uint32_t CpPerfmonCntlReadValue() const override { return Prim::cp_perfmon_cntl_read_value(); }
  uint32_t CpPerfmonCntlSpmStopValue() const override {
    return Prim::cp_perfmon_cntl_spm_stop_value();
  }
  uint32_t CpPerfmonCntlSpmStartValue() const override {
    return Prim::cp_perfmon_cntl_spm_start_value();
  }
  uint32_t CpPerfcountEnableValue() const override { return Prim::cp_perfcount_enable_value(); }
  uint32_t CpPerfcountDisableValue() const override { return Prim::cp_perfcount_disable_value(); }

  // ---- MC / AID block values --------------------------------------------------

  uint32_t McResetValue() const override { return Prim::mc_reset_value(); }
  uint32_t McStartValue() const override { return Prim::mc_start_value(); }
  uint32_t McConfigValue(const counter_des_t& c) const override {
    return Prim::mc_config_value(c);
  }

  // ---- SDMA block values ------------------------------------------------------

  uint32_t SdmaSelectValue(const counter_des_t& c) const override {
    return Prim::sdma_select_value(c);
  }
  uint32_t SdmaDisableClearValue() const override { return Prim::sdma_disable_clear_value(); }
  uint32_t SdmaEnableValue() const override { return Prim::sdma_enable_value(); }
  uint32_t SdmaStopValue(const counter_des_t& c) const override {
    return Prim::sdma_stop_value(c);
  }

  // ---- SQ block values --------------------------------------------------------

  uint32_t SqControlEnableValue() const override { return Prim::sq_control_enable_value(); }
  uint32_t SqControl2EnableValue() const override { return Prim::sq_control2_enable_value(); }
  uint32_t SqMaskValue(const counter_des_t& c) const override { return Prim::sq_mask_value(c); }
  uint32_t SqControlValue(const counter_des_t& c) const override {
    return Prim::sq_control_value(c);
  }
  uint32_t SqSpmSelectValue(const counter_des_t& c) const override {
    return Prim::sq_spm_select_value(c);
  }
  void ValidateCounters(uint32_t attr) const override { Prim::validate_counters(attr); }

  // ---- GUS block values (no GUS on GFX12; all return 0) ----------------------

  uint32_t GusDisableClearValue() const override { return 0; }
  uint32_t GusStartValue() const override { return 0; }
  uint32_t GusStopValue() const override { return 0; }
  uint32_t GusSelectValue(const counter_des_t&) const override { return 0; }

  // ---- SPM-specific -----------------------------------------------------------

  uint32_t RlcSpmPerfmonCntlValue(uint32_t sampling_rate) const override {
    return Prim::rlc_spm_perfmon_cntl_value(sampling_rate);
  }
  uint32_t RlcSpmPerfmonSegmentSizeValue(uint32_t global_count,
                                         uint32_t se_count) const override {
    return Prim::rlc_spm_perfmon_segment_size_value(global_count, se_count);
  }
  uint32_t RlcSpmPerfmonSegmentSizeCore1Value(uint32_t se_count) const override {
    return Prim::rlc_spm_perfmon_segment_size_core1_value(se_count);
  }
  uint16_t SpmTimestampMuxsel() const override { return Prim::spm_timestamp_muxsel(); }
  uint16_t SpmMuxRamValue(const counter_des_t& c) const override {
    return Prim::spm_mux_ram_value(c).data;
  }
  uint16_t SpmMuxRamValue(uint16_t counter, uint16_t block, uint16_t instance) const override {
    return Prim::spm_mux_ram_value(counter, block, instance).data;
  }
  size_t SpmMuxRamIdxIncr(size_t idx) const override {
    return Prim::spm_mux_ram_idx_incr(static_cast<uint32_t>(idx));
  }
  uint32_t SpmEvenSelectValue(const counter_des_t& c) const override {
    return Prim::spm_even_select_value(c);
  }
  uint32_t SpmOddSelectValue(const counter_des_t& c) const override {
    return Prim::spm_odd_select_value(c);
  }
  uint32_t GetSpmGlobalDelay(const counter_des_t& c, uint32_t instance_index) const override {
    return Prim::get_spm_global_delay(c, instance_index);
  }
  uint32_t GetSpmSeDelay(const counter_des_t& c, uint32_t se_index,
                         uint32_t instance_index) const override {
    return Prim::get_spm_se_delay(c, se_index, instance_index);
  }

  // ---- SQTT-specific ----------------------------------------------------------

  uint32_t SqttMaskValue(uint32_t target_cu, uint32_t simd_sel,
                         uint32_t vm_id_mask) const override {
    return Prim::sqtt_mask_value(target_cu, simd_sel, vm_id_mask);
  }
  uint32_t SqttTokenMaskOnValue(bool with_xcc) const override {
    return Prim::sqtt_token_mask_on_value(with_xcc);
  }
  uint32_t SqttTokenMaskOffValue() const override { return Prim::sqtt_token_mask_off_value(); }
  uint32_t SqttTokenMaskOccupancyValue() const override {
    return Prim::sqtt_token_mask_occupancy_value();
  }
  uint32_t SqttTokenMask2Value() const override { return Prim::sqtt_token_mask2_value(); }
  uint32_t SqttModeOffValue() const override { return Prim::sqtt_mode_off_value(); }
  uint32_t SqttModeOnValue(bool double_buffer) const override {
    return Prim::sqtt_mode_on_value(double_buffer);
  }
  uint32_t SqttCtrlValue(bool enable, bool double_buffer) const override {
    return Prim::sqtt_ctrl_value(enable, double_buffer);
  }
  uint32_t SqttZeroSizeValue() const override { return Prim::sqtt_zero_size_value(); }
  uint32_t SqttBusyMask() const override { return Prim::sqtt_busy_mask(); }
  uint32_t SqttPendingMask() const override { return Prim::sqtt_pending_mask(); }
  bool SqttStallingEnabled(uint32_t mask, uint32_t token_mask) const override {
    return Prim::sqtt_stalling_enabled(mask, token_mask);
  }
  uint32_t SqttBaseValueLo(uint64_t base_addr) const override {
    return Prim::sqtt_base_value_lo(base_addr);
  }
  uint32_t SqttBaseValueHi(uint64_t base_addr) const override {
    return Prim::sqtt_base_value_hi(base_addr);
  }
  uint32_t SqttBufferSizeValue(uint64_t size, uint32_t base_hi) const override {
    // GFX12 uses separate BUF0/BUF1 size registers; the legacy combined register is unused.
    return Prim::sqtt_buffer_size_value(static_cast<uint32_t>(size), base_hi);
  }
  uint32_t SqttBuffer0SizeValue(uint64_t size) const override {
    return Prim::sqtt_buffer0_size_value(size);
  }
  uint32_t SpiSqgEventCtlValue(bool enable) const override {
    return Prim::spi_sqg_event_ctl(enable);
  }
};

}  // namespace pm4_builder

#endif  // SRC_PM4_GFX12_PRIMITIVES_PROVIDER_BASE_HPP_
