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

#ifndef SRC_PM4_PRIMITIVES_PROVIDER_HPP_
#define SRC_PM4_PRIMITIVES_PROVIDER_HPP_

#include <stdint.h>
#include <cstddef>

#include "def/gpu_block_info.h"
#include "pm4/cmd_config.h"

namespace pm4_builder {

/// Abstract interface replacing the Primitives template parameter used by GpuPmcBuilder,
/// GpuSpmBuilder, and GpuSqttBuilder.  Each method corresponds to one or more static
/// members/functions that previously lived in a gfx*_cntx_prim traits class and were
/// accessed via template inheritance.  Concrete implementations live in
/// gfx{9,10,11,12}_primitives_provider.{hpp,cpp}.
class PrimitivesProvider {
 public:
  virtual ~PrimitivesProvider() = default;

  // ---- Config constants -------------------------------------------------------

  virtual uint32_t GetGfxipLevel() const = 0;
  virtual uint32_t GetNumberOfBlocks() const = 0;
  /// Number of SDMA counter block instances (used for per-instance tracking arrays).
  virtual uint32_t GetSdmaCounterBlockNumInstances() const = 0;
  virtual uint32_t GetRlcSpmCountersPerLine() const = 0;
  virtual uint32_t GetRlcSpmTimestampSize16() const = 0;
  virtual uint32_t GetSqBlockId() const = 0;
  virtual uint32_t GetSqBlockSpmId() const = 0;
  virtual size_t   GetTtBuffAlignShift() const = 0;
  virtual uint32_t GetSqThreadTraceHiwaterVal() const = 0;
  virtual size_t   GetTtControlUtcErrMask() const = 0;
  virtual size_t   GetTtControlFullMask() const = 0;
  virtual size_t   GetTtLockdownFailMask() const = 0;
  virtual size_t   GetTtWritePtrMask() const = 0;
  virtual uint32_t GetCopyDataSelCountOneDw() const = 0;

  // ---- Capability flags -------------------------------------------------------
  // Replaces #if defined(_GFX10_PRIMITIVES_H_) || defined(_GFX11_PRIMITIVES_H_) guards.

  virtual bool SupportsGusCounters() const = 0;
  virtual bool HasRlcSpmCore1() const = 0;

  // ---- Register addresses -----------------------------------------------------

  virtual Register GetGrbmGfxIndexAddr() const = 0;
  virtual Register GetGrbmaGfxIndexAddr() const = 0;
  virtual Register GetCpPerfmonCntlAddr() const = 0;
  virtual Register GetRlcPerfmonClkCntlAddr() const = 0;
  virtual Register GetComputePerfcountEnableAddr() const = 0;
  virtual Register GetAidPerfmonCntlAddr() const = 0;
  virtual Register GetSqPerfcounterCtrlAddr() const = 0;
  virtual Register GetSqPerfcounterCtrl2Addr() const = 0;
  virtual Register GetSqPerfcounterMaskAddr() const = 0;
  virtual Register GetSpiSqgEventCtlAddr() const = 0;
  virtual Register GetComputeThreadTraceEnableAddr() const = 0;
  /// GUS result control register; returns null register on architectures without GUS.
  virtual Register GetGusRsltCntlAddr() const = 0;
  virtual Register GetRlcSpmPerfmonCntlAddr() const = 0;
  virtual Register GetRlcSpmMcCntlAddr() const = 0;
  virtual Register GetRlcSpmPerfmonRingBaseLo() const = 0;
  virtual Register GetRlcSpmPerfmonRingBaseHi() const = 0;
  virtual Register GetRlcSpmPerfmonRingSize() const = 0;
  virtual Register GetRlcSpmPerfmonSegmentSize() const = 0;
  virtual Register GetRlcSpmPerfmonSegmentSizeCore1() const = 0;
  virtual Register GetRlcSpmGlobalMuxselAddr() const = 0;
  virtual Register GetRlcSpmGlobalMuxselData() const = 0;
  virtual Register GetRlcSpmSeMuxselAddr() const = 0;
  virtual Register GetRlcSpmSeMuxselData() const = 0;
  virtual Register GetRlcSpmPerfmonSampleDelayMax() const = 0;
  virtual Register GetSqThreadTraceMaskAddr() const = 0;
  virtual Register GetSqThreadTracePerfMaskAddr() const = 0;
  virtual Register GetSqThreadTraceTokenMaskAddr() const = 0;
  virtual Register GetSqThreadTraceTokenMask2Addr() const = 0;
  virtual Register GetSqThreadTraceModeAddr() const = 0;
  virtual Register GetSqThreadTraceHiwaterAddr() const = 0;
  virtual Register GetSqThreadTraceBaseAddr() const = 0;
  virtual Register GetSqThreadTraceBuf0BaseLo() const = 0;
  virtual Register GetSqThreadTraceBuf0BaseHi() const = 0;
  virtual Register GetSqThreadTraceBuf0Size() const = 0;
  virtual Register GetSqThreadTraceBuf1BaseLo() const = 0;
  virtual Register GetSqThreadTraceBuf1BaseHi() const = 0;
  virtual Register GetSqThreadTraceBuf1Size() const = 0;
  virtual Register GetSqThreadTraceBase2Addr() const = 0;
  virtual Register GetSqThreadTraceSizeAddr() const = 0;
  virtual Register GetSqThreadTraceCtrlAddr() const = 0;
  virtual Register GetSqThreadTraceStatusAddr() const = 0;
  virtual Register GetSqThreadTraceStatusOffset() const = 0;
  virtual Register GetSqThreadTraceCntrAddr() const = 0;
  virtual Register GetSqThreadTraceWptrAddr() const = 0;
  virtual Register GetSqThreadTraceStatus2Addr() const = 0;
  virtual Register GetSqThreadTraceUserdata2() const = 0;
  virtual Register GetSqThreadTraceUserdata3() const = 0;
  virtual Register SqttPerfcounterAddr(uint32_t index) const = 0;

  // ---- GRBM index value functions ---------------------------------------------

  virtual uint32_t GrbmBroadcastValue() const = 0;
  virtual uint32_t GrbmInstIndexValue(uint32_t instance_index) const = 0;
  virtual uint32_t GrbmSeIndexValue(uint32_t se_index) const = 0;
  virtual uint32_t GrbmInstSeIndexValue(uint32_t instance_index, uint32_t se_index) const = 0;
  virtual uint32_t GrbmSeShIndexValue(uint32_t se_index, uint32_t sh_index) const = 0;
  virtual uint32_t GrbmInstSeShIndexValue(uint32_t inst, uint32_t se, uint32_t sh) const = 0;
  virtual uint32_t GrbmSeShWgpIndexValue(uint32_t se, uint32_t sh, uint32_t wgp) const = 0;
  virtual uint32_t GrbmInstSeShWgpIndexValue(uint32_t inst, uint32_t se, uint32_t sh,
                                             uint32_t wgp) const = 0;

  // ---- CP perfmon control values ----------------------------------------------

  virtual uint32_t CpPerfmonCntlResetValue() const = 0;
  virtual uint32_t CpPerfmonCntlStartValue() const = 0;
  virtual uint32_t CpPerfmonCntlStopValue() const = 0;
  virtual uint32_t CpPerfmonCntlReadValue() const = 0;
  virtual uint32_t CpPerfmonCntlSpmStopValue() const = 0;
  virtual uint32_t CpPerfmonCntlSpmStartValue() const = 0;
  virtual uint32_t CpPerfcountEnableValue() const = 0;
  virtual uint32_t CpPerfcountDisableValue() const = 0;

  // ---- MC / AID block values --------------------------------------------------

  virtual uint32_t McResetValue() const = 0;
  virtual uint32_t McStartValue() const = 0;
  virtual uint32_t McConfigValue(const counter_des_t& c) const = 0;

  // ---- SDMA block values ------------------------------------------------------

  virtual uint32_t SdmaSelectValue(const counter_des_t& c) const = 0;
  virtual uint32_t SdmaDisableClearValue() const = 0;
  virtual uint32_t SdmaEnableValue() const = 0;
  virtual uint32_t SdmaStopValue(const counter_des_t& c) const = 0;

  // ---- SQ block values --------------------------------------------------------

  virtual uint32_t SqControlEnableValue() const = 0;
  virtual uint32_t SqControl2EnableValue() const = 0;
  virtual uint32_t SqMaskValue(const counter_des_t& c) const = 0;
  virtual uint32_t SqControlValue(const counter_des_t& c) const = 0;
  virtual uint32_t SqSpmSelectValue(const counter_des_t& c) const = 0;
  virtual void     ValidateCounters(uint32_t attr) const = 0;

  // ---- GUS block values (GFX10/11 only; no-ops on other arches) ---------------

  virtual uint32_t GusDisableClearValue() const = 0;
  virtual uint32_t GusStartValue() const = 0;
  virtual uint32_t GusStopValue() const = 0;
  virtual uint32_t GusSelectValue(const counter_des_t& c) const = 0;

  // ---- SPM-specific -----------------------------------------------------------

  virtual uint32_t RlcSpmPerfmonCntlValue(uint32_t sampling_rate) const = 0;
  virtual uint32_t RlcSpmPerfmonSegmentSizeValue(uint32_t global_count,
                                                 uint32_t se_count) const = 0;
  virtual uint32_t RlcSpmPerfmonSegmentSizeCore1Value(uint32_t se_count) const = 0;
  /// Mux select values returned as uint16_t (replaces mux_info_t union).
  virtual uint16_t SpmTimestampMuxsel() const = 0;
  virtual uint16_t SpmMuxRamValue(const counter_des_t& c) const = 0;
  virtual uint16_t SpmMuxRamValue(uint16_t counter, uint16_t block, uint16_t instance) const = 0;
  virtual size_t   SpmMuxRamIdxIncr(size_t idx) const = 0;
  virtual uint32_t SpmEvenSelectValue(const counter_des_t& c) const = 0;
  virtual uint32_t SpmOddSelectValue(const counter_des_t& c) const = 0;
  virtual uint32_t GetSpmGlobalDelay(const counter_des_t& c, uint32_t instance_index) const = 0;
  virtual uint32_t GetSpmSeDelay(const counter_des_t& c, uint32_t se_index,
                                 uint32_t instance_index) const = 0;

  // ---- SQTT-specific ----------------------------------------------------------

  virtual uint32_t SqttMaskValue(uint32_t target_cu, uint32_t simd_sel,
                                 uint32_t vm_id_mask) const = 0;
  virtual uint32_t SqttTokenMaskOnValue(bool with_xcc) const = 0;
  virtual uint32_t SqttTokenMaskOffValue() const = 0;
  virtual uint32_t SqttTokenMaskOccupancyValue() const = 0;
  virtual uint32_t SqttTokenMask2Value() const = 0;
  virtual uint32_t SqttModeOffValue() const = 0;
  virtual uint32_t SqttModeOnValue(bool double_buffer) const = 0;
  virtual uint32_t SqttCtrlValue(bool enable, bool double_buffer) const = 0;
  virtual uint32_t SqttZeroSizeValue() const = 0;
  virtual uint32_t SqttBusyMask() const = 0;
  virtual uint32_t SqttPendingMask() const = 0;
  virtual bool     SqttStallingEnabled(uint32_t mask, uint32_t token_mask) const = 0;
  virtual uint32_t SqttBaseValueLo(uint64_t base_addr) const = 0;
  virtual uint32_t SqttBaseValueHi(uint64_t base_addr) const = 0;
  virtual uint32_t SqttBufferSizeValue(uint64_t size, uint32_t base_hi) const = 0;
  virtual uint32_t SqttBuffer0SizeValue(uint64_t size) const = 0;
  virtual uint32_t SpiSqgEventCtlValue(bool enable) const = 0;
};

}  // namespace pm4_builder

#endif  // SRC_PM4_PRIMITIVES_PROVIDER_HPP_
