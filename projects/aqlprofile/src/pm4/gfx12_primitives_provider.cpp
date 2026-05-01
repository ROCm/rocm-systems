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

#include "pm4/gfx12_primitives_provider.hpp"
#include "def/gfx12_def.h"

namespace pm4_builder {

// ---- Config constants -------------------------------------------------------

uint32_t Gfx12PrimitivesProvider::GetGfxipLevel() const {
  return gfx12_cntx_prim::GFXIP_LEVEL;
}

uint32_t Gfx12PrimitivesProvider::GetNumberOfBlocks() const {
  return gfx12_cntx_prim::NUMBER_OF_BLOCKS;
}

uint32_t Gfx12PrimitivesProvider::GetSdmaCounterBlockNumInstances() const {
  return gfx12_cntx_prim::SDMA_COUNTER_BLOCK_NUM_INSTANCES;
}

uint32_t Gfx12PrimitivesProvider::GetRlcSpmCountersPerLine() const {
  return gfx12_cntx_prim::RLC_SPM_COUNTERS_PER_LINE;
}

uint32_t Gfx12PrimitivesProvider::GetRlcSpmTimestampSize16() const {
  return gfx12_cntx_prim::RLC_SPM_TIMESTAMP_SIZE16;
}

uint32_t Gfx12PrimitivesProvider::GetSqBlockId() const {
  return gfx12_cntx_prim::SQ_BLOCK_ID;
}

uint32_t Gfx12PrimitivesProvider::GetSqBlockSpmId() const {
  return gfx12_cntx_prim::SQ_BLOCK_SPM_ID;
}

size_t Gfx12PrimitivesProvider::GetTtBuffAlignShift() const {
  return gfx12_cntx_prim::TT_BUFF_ALIGN_SHIFT;
}

uint32_t Gfx12PrimitivesProvider::GetSqThreadTraceHiwaterVal() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_HIWATER_VAL;
}

size_t Gfx12PrimitivesProvider::GetTtControlUtcErrMask() const {
  return gfx12_cntx_prim::TT_CONTROL_UTC_ERR_MASK;
}

size_t Gfx12PrimitivesProvider::GetTtControlFullMask() const {
  return gfx12_cntx_prim::TT_CONTROL_FULL_MASK;
}

size_t Gfx12PrimitivesProvider::GetTtLockdownFailMask() const {
  return gfx12_cntx_prim::TT_LOCKDOWN_FAIL;
}

size_t Gfx12PrimitivesProvider::GetTtWritePtrMask() const {
  return gfx12_cntx_prim::TT_WRITE_PTR_MASK;
}

uint32_t Gfx12PrimitivesProvider::GetCopyDataSelCountOneDw() const {
  return gfx12_cntx_prim::COPY_DATA_SEL_COUNT_1DW_PRM;
}

// ---- Capability flags -------------------------------------------------------

bool Gfx12PrimitivesProvider::SupportsGusCounters() const { return false; }

bool Gfx12PrimitivesProvider::HasRlcSpmCore1() const { return false; }

// ---- Register addresses -----------------------------------------------------

Register Gfx12PrimitivesProvider::GetGrbmGfxIndexAddr() const {
  return gfx12_cntx_prim::GRBM_GFX_INDEX_ADDR;
}

Register Gfx12PrimitivesProvider::GetGrbmaGfxIndexAddr() const {
  return gfx12_cntx_prim::GRBMA_GFX_INDEX_ADDR;
}

Register Gfx12PrimitivesProvider::GetCpPerfmonCntlAddr() const {
  return gfx12_cntx_prim::CP_PERFMON_CNTL_ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcPerfmonClkCntlAddr() const {
  return gfx12_cntx_prim::RLC_PERFMON_CLK_CNTL_ADDR;
}

Register Gfx12PrimitivesProvider::GetComputePerfcountEnableAddr() const {
  return gfx12_cntx_prim::COMPUTE_PERFCOUNT_ENABLE_ADDR;
}

Register Gfx12PrimitivesProvider::GetAidPerfmonCntlAddr() const {
  return gfx12_cntx_prim::AID_PERFMON_CNTL_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqPerfcounterCtrlAddr() const {
  return gfx12_cntx_prim::SQ_PERFCOUNTER_CTRL_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqPerfcounterCtrl2Addr() const {
  return gfx12_cntx_prim::SQ_PERFCOUNTER_CTRL2_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqPerfcounterMaskAddr() const {
  return gfx12_cntx_prim::SQ_PERFCOUNTER_MASK_ADDR;
}

Register Gfx12PrimitivesProvider::GetSpiSqgEventCtlAddr() const {
  return gfx12_cntx_prim::SPI_SQG_EVENT_CTL_ADDR;
}

Register Gfx12PrimitivesProvider::GetComputeThreadTraceEnableAddr() const {
  return gfx12_cntx_prim::COMPUTE_THREAD_TRACE_ENABLE_ADDR;
}

Register Gfx12PrimitivesProvider::GetGusRsltCntlAddr() const {
  // GFX12 does not have a GUS result control register.
  return Register{};
}

Register Gfx12PrimitivesProvider::GetRlcSpmPerfmonCntlAddr() const {
  return gfx12_cntx_prim::RLC_SPM_PERFMON_CNTL__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmMcCntlAddr() const {
  return gfx12_cntx_prim::RLC_SPM_MC_CNTL__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmPerfmonRingBaseLo() const {
  return gfx12_cntx_prim::RLC_SPM_PERFMON_RING_BASE_LO__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmPerfmonRingBaseHi() const {
  return gfx12_cntx_prim::RLC_SPM_PERFMON_RING_BASE_HI__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmPerfmonRingSize() const {
  return gfx12_cntx_prim::RLC_SPM_PERFMON_RING_SIZE__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmPerfmonSegmentSize() const {
  return gfx12_cntx_prim::RLC_SPM_PERFMON_SEGMENT_SIZE__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmPerfmonSegmentSizeCore1() const {
  return gfx12_cntx_prim::RLC_SPM_PERFMON_SEGMENT_SIZE_CORE1__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmGlobalMuxselAddr() const {
  return gfx12_cntx_prim::RLC_SPM_GLOBAL_MUXSEL_ADDR__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmGlobalMuxselData() const {
  return gfx12_cntx_prim::RLC_SPM_GLOBAL_MUXSEL_DATA__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmSeMuxselAddr() const {
  return gfx12_cntx_prim::RLC_SPM_SE_MUXSEL_ADDR__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmSeMuxselData() const {
  return gfx12_cntx_prim::RLC_SPM_SE_MUXSEL_DATA__ADDR;
}

Register Gfx12PrimitivesProvider::GetRlcSpmPerfmonSampleDelayMax() const {
  return gfx12_cntx_prim::RLC_SPM_PERFMON_SAMPLE_DELAY_MAX__ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceMaskAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_MASK_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTracePerfMaskAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_PERF_MASK_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceTokenMaskAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_TOKEN_MASK_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceTokenMask2Addr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_TOKEN_MASK2_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceModeAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_MODE_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceHiwaterAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_HIWATER_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBaseAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BASE_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBuf0BaseLo() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BUF0_BASE_LO_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBuf0BaseHi() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BUF0_BASE_HI_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBuf0Size() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BUF0_SIZE_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBuf1BaseLo() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BUF1_BASE_LO_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBuf1BaseHi() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BUF1_BASE_HI_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBuf1Size() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BUF1_SIZE_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceBase2Addr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_BASE2_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceSizeAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_SIZE_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceCtrlAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_CTRL_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceStatusAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_STATUS_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceStatusOffset() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_STATUS_OFFSET;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceCntrAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_CNTR_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceWptrAddr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_WPTR_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceStatus2Addr() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_STATUS2_ADDR;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceUserdata2() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_USERDATA_2;
}

Register Gfx12PrimitivesProvider::GetSqThreadTraceUserdata3() const {
  return gfx12_cntx_prim::SQ_THREAD_TRACE_USERDATA_3;
}

Register Gfx12PrimitivesProvider::SqttPerfcounterAddr(uint32_t index) const {
  return gfx12_cntx_prim::sqtt_perfcounter_addr(index);
}

// ---- GRBM index value functions ---------------------------------------------

uint32_t Gfx12PrimitivesProvider::GrbmBroadcastValue() const {
  return gfx12_cntx_prim::grbm_broadcast_value();
}

uint32_t Gfx12PrimitivesProvider::GrbmInstIndexValue(uint32_t instance_index) const {
  return gfx12_cntx_prim::grbm_inst_index_value(instance_index);
}

uint32_t Gfx12PrimitivesProvider::GrbmSeIndexValue(uint32_t se_index) const {
  return gfx12_cntx_prim::grbm_se_index_value(se_index);
}

uint32_t Gfx12PrimitivesProvider::GrbmInstSeIndexValue(uint32_t instance_index,
                                                       uint32_t se_index) const {
  return gfx12_cntx_prim::grbm_inst_se_index_value(instance_index, se_index);
}

uint32_t Gfx12PrimitivesProvider::GrbmSeShIndexValue(uint32_t se_index,
                                                      uint32_t sh_index) const {
  return gfx12_cntx_prim::grbm_se_sh_index_value(se_index, sh_index);
}

uint32_t Gfx12PrimitivesProvider::GrbmInstSeShIndexValue(uint32_t inst, uint32_t se,
                                                          uint32_t sh) const {
  return gfx12_cntx_prim::grbm_inst_se_sh_index_value(inst, se, sh);
}

uint32_t Gfx12PrimitivesProvider::GrbmSeShWgpIndexValue(uint32_t se, uint32_t sh,
                                                         uint32_t wgp) const {
  return gfx12_cntx_prim::grbm_se_sh_wgp_index_value(se, sh, wgp);
}

uint32_t Gfx12PrimitivesProvider::GrbmInstSeShWgpIndexValue(uint32_t inst, uint32_t se,
                                                             uint32_t sh, uint32_t wgp) const {
  return gfx12_cntx_prim::grbm_inst_se_sh_wgp_index_value(inst, se, sh, wgp);
}

// ---- CP perfmon control values ----------------------------------------------

uint32_t Gfx12PrimitivesProvider::CpPerfmonCntlResetValue() const {
  return gfx12_cntx_prim::cp_perfmon_cntl_reset_value();
}

uint32_t Gfx12PrimitivesProvider::CpPerfmonCntlStartValue() const {
  return gfx12_cntx_prim::cp_perfmon_cntl_start_value();
}

uint32_t Gfx12PrimitivesProvider::CpPerfmonCntlStopValue() const {
  return gfx12_cntx_prim::cp_perfmon_cntl_stop_value();
}

uint32_t Gfx12PrimitivesProvider::CpPerfmonCntlReadValue() const {
  return gfx12_cntx_prim::cp_perfmon_cntl_read_value();
}

uint32_t Gfx12PrimitivesProvider::CpPerfmonCntlSpmStopValue() const {
  return gfx12_cntx_prim::cp_perfmon_cntl_spm_stop_value();
}

uint32_t Gfx12PrimitivesProvider::CpPerfmonCntlSpmStartValue() const {
  return gfx12_cntx_prim::cp_perfmon_cntl_spm_start_value();
}

uint32_t Gfx12PrimitivesProvider::CpPerfcountEnableValue() const {
  return gfx12_cntx_prim::cp_perfcount_enable_value();
}

uint32_t Gfx12PrimitivesProvider::CpPerfcountDisableValue() const {
  return gfx12_cntx_prim::cp_perfcount_disable_value();
}

// ---- MC / AID block values --------------------------------------------------

uint32_t Gfx12PrimitivesProvider::McResetValue() const {
  return gfx12_cntx_prim::mc_reset_value();
}

uint32_t Gfx12PrimitivesProvider::McStartValue() const {
  return gfx12_cntx_prim::mc_start_value();
}

uint32_t Gfx12PrimitivesProvider::McConfigValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::mc_config_value(c);
}

// ---- SDMA block values ------------------------------------------------------

uint32_t Gfx12PrimitivesProvider::SdmaSelectValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::sdma_select_value(c);
}

uint32_t Gfx12PrimitivesProvider::SdmaDisableClearValue() const {
  return gfx12_cntx_prim::sdma_disable_clear_value();
}

uint32_t Gfx12PrimitivesProvider::SdmaEnableValue() const {
  return gfx12_cntx_prim::sdma_enable_value();
}

uint32_t Gfx12PrimitivesProvider::SdmaStopValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::sdma_stop_value(c);
}

// ---- SQ block values --------------------------------------------------------

uint32_t Gfx12PrimitivesProvider::SqControlEnableValue() const {
  return gfx12_cntx_prim::sq_control_enable_value();
}

uint32_t Gfx12PrimitivesProvider::SqControl2EnableValue() const {
  return gfx12_cntx_prim::sq_control2_enable_value();
}

uint32_t Gfx12PrimitivesProvider::SqMaskValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::sq_mask_value(c);
}

uint32_t Gfx12PrimitivesProvider::SqControlValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::sq_control_value(c);
}

uint32_t Gfx12PrimitivesProvider::SqSpmSelectValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::sq_spm_select_value(c);
}

void Gfx12PrimitivesProvider::ValidateCounters(uint32_t attr) const {
  gfx12_cntx_prim::validate_counters(attr);
}

// ---- GUS block values (no GUS on GFX12) ------------------------------------

uint32_t Gfx12PrimitivesProvider::GusDisableClearValue() const { return 0; }

uint32_t Gfx12PrimitivesProvider::GusStartValue() const { return 0; }

uint32_t Gfx12PrimitivesProvider::GusStopValue() const { return 0; }

uint32_t Gfx12PrimitivesProvider::GusSelectValue(const counter_des_t&) const { return 0; }

// ---- SPM-specific -----------------------------------------------------------

uint32_t Gfx12PrimitivesProvider::RlcSpmPerfmonCntlValue(uint32_t sampling_rate) const {
  return gfx12_cntx_prim::rlc_spm_perfmon_cntl_value(sampling_rate);
}

uint32_t Gfx12PrimitivesProvider::RlcSpmPerfmonSegmentSizeValue(uint32_t global_count,
                                                                uint32_t se_count) const {
  return gfx12_cntx_prim::rlc_spm_perfmon_segment_size_value(global_count, se_count);
}

uint32_t Gfx12PrimitivesProvider::RlcSpmPerfmonSegmentSizeCore1Value(uint32_t se_count) const {
  return gfx12_cntx_prim::rlc_spm_perfmon_segment_size_core1_value(se_count);
}

uint16_t Gfx12PrimitivesProvider::SpmTimestampMuxsel() const {
  return gfx12_cntx_prim::spm_timestamp_muxsel();
}

uint16_t Gfx12PrimitivesProvider::SpmMuxRamValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::spm_mux_ram_value(c).data;
}

uint16_t Gfx12PrimitivesProvider::SpmMuxRamValue(uint16_t counter, uint16_t block,
                                                  uint16_t instance) const {
  return gfx12_cntx_prim::spm_mux_ram_value(counter, block, instance).data;
}

size_t Gfx12PrimitivesProvider::SpmMuxRamIdxIncr(size_t idx) const {
  return gfx12_cntx_prim::spm_mux_ram_idx_incr(static_cast<uint32_t>(idx));
}

uint32_t Gfx12PrimitivesProvider::SpmEvenSelectValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::spm_even_select_value(c);
}

uint32_t Gfx12PrimitivesProvider::SpmOddSelectValue(const counter_des_t& c) const {
  return gfx12_cntx_prim::spm_odd_select_value(c);
}

uint32_t Gfx12PrimitivesProvider::GetSpmGlobalDelay(const counter_des_t& c,
                                                     uint32_t instance_index) const {
  return gfx12_cntx_prim::get_spm_global_delay(c, instance_index);
}

uint32_t Gfx12PrimitivesProvider::GetSpmSeDelay(const counter_des_t& c, uint32_t se_index,
                                                 uint32_t instance_index) const {
  return gfx12_cntx_prim::get_spm_se_delay(c, se_index, instance_index);
}

// ---- SQTT-specific ----------------------------------------------------------

uint32_t Gfx12PrimitivesProvider::SqttMaskValue(uint32_t target_cu, uint32_t simd_sel,
                                                 uint32_t vm_id_mask) const {
  return gfx12_cntx_prim::sqtt_mask_value(target_cu, simd_sel, vm_id_mask);
}

uint32_t Gfx12PrimitivesProvider::SqttTokenMaskOnValue(bool with_xcc) const {
  return gfx12_cntx_prim::sqtt_token_mask_on_value(with_xcc);
}

uint32_t Gfx12PrimitivesProvider::SqttTokenMaskOffValue() const {
  return gfx12_cntx_prim::sqtt_token_mask_off_value();
}

uint32_t Gfx12PrimitivesProvider::SqttTokenMaskOccupancyValue() const {
  return gfx12_cntx_prim::sqtt_token_mask_occupancy_value();
}

uint32_t Gfx12PrimitivesProvider::SqttTokenMask2Value() const {
  return gfx12_cntx_prim::sqtt_token_mask2_value();
}

uint32_t Gfx12PrimitivesProvider::SqttModeOffValue() const {
  return gfx12_cntx_prim::sqtt_mode_off_value();
}

uint32_t Gfx12PrimitivesProvider::SqttModeOnValue(bool double_buffer) const {
  return gfx12_cntx_prim::sqtt_mode_on_value(double_buffer);
}

uint32_t Gfx12PrimitivesProvider::SqttCtrlValue(bool enable, bool double_buffer) const {
  return gfx12_cntx_prim::sqtt_ctrl_value(enable, double_buffer);
}

uint32_t Gfx12PrimitivesProvider::SqttZeroSizeValue() const {
  return gfx12_cntx_prim::sqtt_zero_size_value();
}

uint32_t Gfx12PrimitivesProvider::SqttBusyMask() const {
  return gfx12_cntx_prim::sqtt_busy_mask();
}

uint32_t Gfx12PrimitivesProvider::SqttPendingMask() const {
  return gfx12_cntx_prim::sqtt_pending_mask();
}

bool Gfx12PrimitivesProvider::SqttStallingEnabled(uint32_t mask, uint32_t token_mask) const {
  return gfx12_cntx_prim::sqtt_stalling_enabled(mask, token_mask);
}

uint32_t Gfx12PrimitivesProvider::SqttBaseValueLo(uint64_t base_addr) const {
  return gfx12_cntx_prim::sqtt_base_value_lo(base_addr);
}

uint32_t Gfx12PrimitivesProvider::SqttBaseValueHi(uint64_t base_addr) const {
  return gfx12_cntx_prim::sqtt_base_value_hi(base_addr);
}

uint32_t Gfx12PrimitivesProvider::SqttBufferSizeValue(uint64_t size, uint32_t base_hi) const {
  // GFX12 uses separate BUF0/BUF1 size registers; the legacy combined register is unused.
  return gfx12_cntx_prim::sqtt_buffer_size_value(static_cast<uint32_t>(size), base_hi);
}

uint32_t Gfx12PrimitivesProvider::SqttBuffer0SizeValue(uint64_t size) const {
  return gfx12_cntx_prim::sqtt_buffer0_size_value(size);
}

uint32_t Gfx12PrimitivesProvider::SpiSqgEventCtlValue(bool enable) const {
  return gfx12_cntx_prim::spi_sqg_event_ctl(enable);
}

}  // namespace pm4_builder
