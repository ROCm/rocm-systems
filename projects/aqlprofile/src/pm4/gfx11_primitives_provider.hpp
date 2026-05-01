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

#ifndef SRC_PM4_GFX11_PRIMITIVES_PROVIDER_HPP_
#define SRC_PM4_GFX11_PRIMITIVES_PROVIDER_HPP_

#include "pm4/primitives_provider.hpp"

namespace pm4_builder {

class Gfx11PrimitivesProvider : public PrimitivesProvider {
 public:
  // ---- Config constants -------------------------------------------------------

  uint32_t GetGfxipLevel() const override;
  uint32_t GetNumberOfBlocks() const override;
  uint32_t GetSdmaCounterBlockNumInstances() const override;
  uint32_t GetRlcSpmCountersPerLine() const override;
  uint32_t GetRlcSpmTimestampSize16() const override;
  uint32_t GetSqBlockId() const override;
  uint32_t GetSqBlockSpmId() const override;
  size_t   GetTtBuffAlignShift() const override;
  uint32_t GetSqThreadTraceHiwaterVal() const override;
  size_t   GetTtControlUtcErrMask() const override;
  size_t   GetTtControlFullMask() const override;
  size_t   GetTtLockdownFailMask() const override;
  size_t   GetTtWritePtrMask() const override;
  uint32_t GetCopyDataSelCountOneDw() const override;

  // ---- Capability flags -------------------------------------------------------

  bool SupportsGusCounters() const override;
  bool HasRlcSpmCore1() const override;

  // ---- Register addresses -----------------------------------------------------

  Register GetGrbmGfxIndexAddr() const override;
  Register GetGrbmaGfxIndexAddr() const override;
  Register GetCpPerfmonCntlAddr() const override;
  Register GetRlcPerfmonClkCntlAddr() const override;
  Register GetComputePerfcountEnableAddr() const override;
  Register GetAidPerfmonCntlAddr() const override;
  Register GetSqPerfcounterCtrlAddr() const override;
  Register GetSqPerfcounterCtrl2Addr() const override;
  Register GetSqPerfcounterMaskAddr() const override;
  Register GetSpiSqgEventCtlAddr() const override;
  Register GetComputeThreadTraceEnableAddr() const override;
  Register GetGusRsltCntlAddr() const override;
  Register GetRlcSpmPerfmonCntlAddr() const override;
  Register GetRlcSpmMcCntlAddr() const override;
  Register GetRlcSpmPerfmonRingBaseLo() const override;
  Register GetRlcSpmPerfmonRingBaseHi() const override;
  Register GetRlcSpmPerfmonRingSize() const override;
  Register GetRlcSpmPerfmonSegmentSize() const override;
  Register GetRlcSpmPerfmonSegmentSizeCore1() const override;
  Register GetRlcSpmGlobalMuxselAddr() const override;
  Register GetRlcSpmGlobalMuxselData() const override;
  Register GetRlcSpmSeMuxselAddr() const override;
  Register GetRlcSpmSeMuxselData() const override;
  Register GetRlcSpmPerfmonSampleDelayMax() const override;
  Register GetSqThreadTraceMaskAddr() const override;
  Register GetSqThreadTracePerfMaskAddr() const override;
  Register GetSqThreadTraceTokenMaskAddr() const override;
  Register GetSqThreadTraceTokenMask2Addr() const override;
  Register GetSqThreadTraceModeAddr() const override;
  Register GetSqThreadTraceHiwaterAddr() const override;
  Register GetSqThreadTraceBaseAddr() const override;
  Register GetSqThreadTraceBuf0BaseLo() const override;
  Register GetSqThreadTraceBuf0BaseHi() const override;
  Register GetSqThreadTraceBuf0Size() const override;
  Register GetSqThreadTraceBuf1BaseLo() const override;
  Register GetSqThreadTraceBuf1BaseHi() const override;
  Register GetSqThreadTraceBuf1Size() const override;
  Register GetSqThreadTraceBase2Addr() const override;
  Register GetSqThreadTraceSizeAddr() const override;
  Register GetSqThreadTraceCtrlAddr() const override;
  Register GetSqThreadTraceStatusAddr() const override;
  Register GetSqThreadTraceStatusOffset() const override;
  Register GetSqThreadTraceCntrAddr() const override;
  Register GetSqThreadTraceWptrAddr() const override;
  Register GetSqThreadTraceStatus2Addr() const override;
  Register GetSqThreadTraceUserdata2() const override;
  Register GetSqThreadTraceUserdata3() const override;
  Register SqttPerfcounterAddr(uint32_t index) const override;

  // ---- GRBM index value functions ---------------------------------------------

  uint32_t GrbmBroadcastValue() const override;
  uint32_t GrbmInstIndexValue(uint32_t instance_index) const override;
  uint32_t GrbmSeIndexValue(uint32_t se_index) const override;
  uint32_t GrbmInstSeIndexValue(uint32_t instance_index, uint32_t se_index) const override;
  uint32_t GrbmSeShIndexValue(uint32_t se_index, uint32_t sh_index) const override;
  uint32_t GrbmInstSeShIndexValue(uint32_t inst, uint32_t se, uint32_t sh) const override;
  uint32_t GrbmSeShWgpIndexValue(uint32_t se, uint32_t sh, uint32_t wgp) const override;
  uint32_t GrbmInstSeShWgpIndexValue(uint32_t inst, uint32_t se, uint32_t sh,
                                     uint32_t wgp) const override;

  // ---- CP perfmon control values ----------------------------------------------

  uint32_t CpPerfmonCntlResetValue() const override;
  uint32_t CpPerfmonCntlStartValue() const override;
  uint32_t CpPerfmonCntlStopValue() const override;
  uint32_t CpPerfmonCntlReadValue() const override;
  uint32_t CpPerfmonCntlSpmStopValue() const override;
  uint32_t CpPerfmonCntlSpmStartValue() const override;
  uint32_t CpPerfcountEnableValue() const override;
  uint32_t CpPerfcountDisableValue() const override;

  // ---- MC / AID block values --------------------------------------------------

  uint32_t McResetValue() const override;
  uint32_t McStartValue() const override;
  uint32_t McConfigValue(const counter_des_t& c) const override;

  // ---- SDMA block values ------------------------------------------------------

  uint32_t SdmaSelectValue(const counter_des_t& c) const override;
  uint32_t SdmaDisableClearValue() const override;
  uint32_t SdmaEnableValue() const override;
  uint32_t SdmaStopValue(const counter_des_t& c) const override;

  // ---- SQ block values --------------------------------------------------------

  uint32_t SqControlEnableValue() const override;
  uint32_t SqControl2EnableValue() const override;
  uint32_t SqMaskValue(const counter_des_t& c) const override;
  uint32_t SqControlValue(const counter_des_t& c) const override;
  uint32_t SqSpmSelectValue(const counter_des_t& c) const override;
  void     ValidateCounters(uint32_t attr) const override;

  // ---- GUS block values -------------------------------------------------------

  uint32_t GusDisableClearValue() const override;
  uint32_t GusStartValue() const override;
  uint32_t GusStopValue() const override;
  uint32_t GusSelectValue(const counter_des_t& c) const override;

  // ---- SPM-specific -----------------------------------------------------------

  uint32_t RlcSpmPerfmonCntlValue(uint32_t sampling_rate) const override;
  uint32_t RlcSpmPerfmonSegmentSizeValue(uint32_t global_count, uint32_t se_count) const override;
  uint32_t RlcSpmPerfmonSegmentSizeCore1Value(uint32_t se_count) const override;
  uint16_t SpmTimestampMuxsel() const override;
  uint16_t SpmMuxRamValue(const counter_des_t& c) const override;
  uint16_t SpmMuxRamValue(uint16_t counter, uint16_t block, uint16_t instance) const override;
  size_t   SpmMuxRamIdxIncr(size_t idx) const override;
  uint32_t SpmEvenSelectValue(const counter_des_t& c) const override;
  uint32_t SpmOddSelectValue(const counter_des_t& c) const override;
  uint32_t GetSpmGlobalDelay(const counter_des_t& c, uint32_t instance_index) const override;
  uint32_t GetSpmSeDelay(const counter_des_t& c, uint32_t se_index,
                         uint32_t instance_index) const override;

  // ---- SQTT-specific ----------------------------------------------------------

  uint32_t SqttMaskValue(uint32_t target_cu, uint32_t simd_sel,
                         uint32_t vm_id_mask) const override;
  uint32_t SqttTokenMaskOnValue(bool with_xcc) const override;
  uint32_t SqttTokenMaskOffValue() const override;
  uint32_t SqttTokenMaskOccupancyValue() const override;
  uint32_t SqttTokenMask2Value() const override;
  uint32_t SqttModeOffValue() const override;
  uint32_t SqttModeOnValue(bool double_buffer) const override;
  uint32_t SqttCtrlValue(bool enable, bool double_buffer) const override;
  uint32_t SqttZeroSizeValue() const override;
  uint32_t SqttBusyMask() const override;
  uint32_t SqttPendingMask() const override;
  bool     SqttStallingEnabled(uint32_t mask, uint32_t token_mask) const override;
  uint32_t SqttBaseValueLo(uint64_t base_addr) const override;
  uint32_t SqttBaseValueHi(uint64_t base_addr) const override;
  uint32_t SqttBufferSizeValue(uint64_t size, uint32_t base_hi) const override;
  uint32_t SqttBuffer0SizeValue(uint64_t size) const override;
  uint32_t SpiSqgEventCtlValue(bool enable) const override;
};

}  // namespace pm4_builder

#endif  // SRC_PM4_GFX11_PRIMITIVES_PROVIDER_HPP_
