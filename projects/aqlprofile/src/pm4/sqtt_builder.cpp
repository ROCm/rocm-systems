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

#include "pm4/sqtt_builder.h"

#include "pm4/cmd_builder.h"

namespace pm4_builder {

// XCC_Packet_Lock

XCC_Packet_Lock::XCC_Packet_Lock(CmdBuilder& _builder, CmdBuffer* cmd_buffer, uint32_t xcc_number,
                                 uint32_t xcc_mask)
    : builder_(_builder) {
  this->xcc_number = xcc_number;
  this->cmd_buffer = cmd_buffer;
  this->xcc_mask = xcc_mask;
  this->xcc_initial_cmd_size = (uint32_t)cmd_buffer->DwSize();

  if (xcc_number > 1) builder_.BuildPredExecPacket(this->cmd_buffer, this->xcc_mask, 0);
}

XCC_Packet_Lock::~XCC_Packet_Lock() {
  if (xcc_number < 2) return;

  CmdBuffer pred_exec;
  builder_.BuildPredExecPacket(&pred_exec, 0, 0);

  auto xcc_buf_size = cmd_buffer->DwSize() - pred_exec.DwSize() - xcc_initial_cmd_size;

  // update first PRED_EXEC packet to its correct value
  pred_exec.Clear();
  builder_.BuildPredExecPacket(&pred_exec, xcc_mask, xcc_buf_size);
  const uint32_t* data = (const uint32_t*)pred_exec.Data();

  for (size_t i = 0; i < pred_exec.DwSize(); ++i)
    cmd_buffer->Assign(xcc_initial_cmd_size + i, data[i]);
}

// GpuSqttBuilder

void GpuSqttBuilder::DebugTrace(uint32_t value) {
  CmdBuffer cmd_buffer;
  uint32_t header[2] = {0, value};
  APPEND_COMMAND_WRAPPER((&cmd_buffer), header);
}

void GpuSqttBuilder::SetGRBMToBroadcast(CmdBuffer* cmd_buffer) {
  auto broadcast = prim_->GrbmBroadcastValue();
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(), broadcast);
}

void GpuSqttBuilder::Select_GRBM_SE_SH0(CmdBuffer* cmd_buffer, int se_index) {
  auto sh0 = prim_->GrbmSeShIndexValue(se_index, 0);
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetGrbmGfxIndexAddr(), sh0);
}

void GpuSqttBuilder::StartPerfMon(CmdBuffer* cmd_buffer, TraceConfig* config) {
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcPerfmonClkCntlAddr(), 1);

  builder_->BuildWriteShRegPacket(cmd_buffer, prim_->GetComputePerfcountEnableAddr(),
                                  prim_->CpPerfcountEnableValue());
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
                                       prim_->CpPerfmonCntlResetValue());

  for (int perf = 0; perf < (int)config->perfcounters.size() && perf < 8; perf++) {
    size_t mask = config->perfcounters[perf].second << SQTT_PERFCOUNTER_SIMD_MASK;
    builder_->BuildWriteConfigRegPacket(cmd_buffer, prim_->SqttPerfcounterAddr(perf),
                                        config->perfcounters[perf].first | mask);
  }
  for (int perf = config->perfcounters.size(); perf < 8; perf++)
  {
    builder_->BuildWriteConfigRegPacket(cmd_buffer, prim_->SqttPerfcounterAddr(perf), 0);
  }
  // Trace only masked
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTracePerfMaskAddr(),
                                        config->perfMASK);
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqPerfcounterMaskAddr(),
                                       config->perfMASK);
  builder_->BuildWriteConfigRegPacket(cmd_buffer, prim_->GetSqPerfcounterCtrlAddr(),
                                      config->perfCTRL);
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
                                       prim_->CpPerfmonCntlStartValue());
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
}

void GpuSqttBuilder::StopPerfMon(CmdBuffer* cmd_buffer) {
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
                                       prim_->CpPerfmonCntlStopValue());
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetCpPerfmonCntlAddr(),
                                       prim_->CpPerfmonCntlResetValue());
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetRlcPerfmonClkCntlAddr(), 0);
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
}

void GpuSqttBuilder::Begin(CmdBuffer* cmd_buffer, TraceConfig* config) {
  // Iterate through the list of SE's and program the register
  // for carrying address of thread trace buffer which is aligned
  // to 4KB per thread trace specification
  const uint64_t se_number_xcc = se_number_total / GetXCCNumber();
  uint64_t base_addr = reinterpret_cast<uint64_t>(config->data_buffer_ptr);
  const size_t tt_buff_align_shift = prim_->GetTtBuffAlignShift();
  if (prim_->GetGfxipLevel() == 10 || prim_->GetGfxipLevel() == 11)
    config->capacity_per_disabled_se = 1 << tt_buff_align_shift;

  const uint64_t base_step = GetBaseStep(config);

  // Old v1 API calls this with buffer == 0 first
  if (config->data_buffer_size > 0)
  {
    // Max 16GB for gfx{9, 10, 12} and 512MB for gfx11. Min of 32 page per SE.
    if (base_step >= (1ull<<34) || (prim_->GetGfxipLevel() == 11 && base_step >= (1ull<<29)))
      throw std::runtime_error("SQTT Buffer size too high");
    else if (base_step < (1ull<<17))
      throw std::runtime_error("SQTT Buffer size too low");
  }
  config->capacity_per_se = base_step;

  const bool legacy_mode =
      config->deprecated_mask && config->deprecated_tokenMask && config->deprecated_tokenMask2;

  for (uint64_t se_index = 0; se_index < se_number_total; se_index++) {
    bool bMaskedIn = ((1 << se_index) & config->se_mask) != 0;
    config->target_cu_per_se[se_index] = bMaskedIn ? config->targetCu : -1;
  }

  if (prim_->GetGfxipLevel() == 9) {
    // Program Grbm to broadcast messages to all shader engines
    SetGRBMToBroadcast(cmd_buffer);

    // Issue a CSPartialFlush cmd including cache flush
    if (config->concurrent == 0) builder_->BuildWriteWaitIdlePacket(cmd_buffer);
    // Program the thread trace mask - specifies SH, CU, SIMD and
    // VM Id masks to apply. Enabling SQ/SPI/REG_STALL_EN bits
    const uint32_t mask_value =
        (legacy_mode)
            ? config->deprecated_mask
            : prim_->SqttMaskValue(config->targetCu, config->simd_sel, config->vmIdMask);
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceMaskAddr(),
                                          mask_value);

    if (config->perfcounters.size()) StartPerfMon(cmd_buffer, config);

    // Program the thread trace token mask
    uint32_t token_mask_value = (config->occupancy_mode)
                                    ? prim_->SqttTokenMaskOccupancyValue()
                                    : prim_->SqttTokenMaskOnValue(false);
    if (config->perfcounters.size()) token_mask_value |= SQTT_PERFCOUNTER_TOKEN;
    if (legacy_mode) token_mask_value = config->deprecated_tokenMask;

    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceTokenMaskAddr(),
                                         token_mask_value);
    // Program the thread trace token mask2 to specify the list of instruction
    // tokens to record. Disabling INST_PC instruction tokens

    // Program the thread trace mode register, mode OFF
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceModeAddr(),
                                         prim_->SqttModeOffValue());
    // Program the HiWaterMark register to support stalling
    if (prim_->SqttStallingEnabled(mask_value, token_mask_value)) {
      builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceHiwaterAddr(),
                                           prim_->GetSqThreadTraceHiwaterVal());
    }
    for (uint64_t se_index = 0; se_index < se_number_total; se_index++) {
      config->se_base_addresses[se_index] = base_addr;
      if (config->target_cu_per_se.at(se_index) < 0) {
        base_addr += config->capacity_per_disabled_se;
        continue;
      }

      uint32_t token_mask2_value = prim_->SqttTokenMask2Value();
      if (legacy_mode)
        token_mask2_value = config->deprecated_tokenMask2;
      else if (((1 << se_index) & config->se_mask) == 0)
        token_mask2_value = 0;

      uint64_t xcc_index = se_index / se_number_xcc;
      uint64_t se_index_xcc = se_index % se_number_xcc;

      XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), xcc_index);

        // Program Grbm to direct writes to one SE
        Select_GRBM_SE_SH0(cmd_buffer, se_index_xcc);
        builder_->BuildPrimeL2(cmd_buffer, base_addr);
        // Program tokenmask2
        builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceTokenMask2Addr(),
                                             token_mask2_value);
        // Set SQTT STATUS to 0
        builder_->BuildWritePConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceStatusAddr(), 0);
        // Program base address of buffer to use for thread trace
        builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceBaseAddr(),
                                             prim_->SqttBaseValueLo(base_addr));
        builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceBase2Addr(),
                                             prim_->SqttBaseValueHi(base_addr));
        // Program the size of thread trace buffer
        builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceSizeAddr(),
                                             prim_->SqttBufferSizeValue(base_step, 0));
        // Program the thread trace ctrl register
        builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceCtrlAddr(),
                                             prim_->SqttCtrlValue(true, false));
        // Issue a CSPartialFlush cmd including cache flush
        builder_->BuildWriteWaitIdlePacket(cmd_buffer);
        // Program the thread trace mode register, mode ON
        builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceModeAddr(),
                                         prim_->SqttModeOnValue(!config->buffer_data.empty()));

        // If we are in double buffer mode
        if (!config->buffer_data.empty())
        {
          builder_->BuildWriteWaitIdlePacket(cmd_buffer);
          uint64_t buf2_addr = reinterpret_cast<uint64_t>(config->buffer_data.at(se_index).at(0));

          builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceBaseAddr(),
                                               prim_->SqttBaseValueLo(buf2_addr));
          builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceBase2Addr(),
                                               prim_->SqttBaseValueHi(buf2_addr));
        }
        base_addr += base_step;
    }
    // Reset the GRBM to broadcast mode
    SetGRBMToBroadcast(cmd_buffer);
  } else {
    SetGRBMToBroadcast(cmd_buffer);
    builder_->BuildWritePConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceStatusAddr(), 0);

    if (prim_->GetGfxipLevel() == 12) {
      WriteConfigPacket(cmd_buffer, prim_->GetSpiSqgEventCtlAddr(),
                        prim_->SpiSqgEventCtlValue(true));
    }

    for (int xcc=0; xcc < (int)xcc_number_; xcc++)
    {
      if (!isXccEnabled(xcc, se_number_xcc, config)) continue;

      XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), xcc);

      for (size_t local_se = 0; local_se < se_number_xcc; local_se++)
      {
        size_t global_se = local_se + se_number_xcc * xcc;

        config->se_base_addresses[global_se] = base_addr;
        bool bMaskedIn = config->target_cu_per_se.at(global_se) >= 0;

        const unsigned baddr_lo = Low32(base_addr >> tt_buff_align_shift);
        const unsigned baddr_hi = High32(base_addr >> tt_buff_align_shift);
        const uint64_t sqtt_size = bMaskedIn ? base_step : config->capacity_per_disabled_se;
        if (sqtt_size == 0) continue;

        uint32_t ctrl_val = prim_->SqttCtrlValue(true, !config->buffer_data.empty());

        Select_GRBM_SE_SH0(cmd_buffer, local_se);
        builder_->BuildPrimeL2(cmd_buffer, base_addr);

        if (prim_->GetGfxipLevel() == 12) {
          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceBuf0Size(),
                            prim_->SqttBuffer0SizeValue(sqtt_size));

          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceBuf0BaseLo(), baddr_lo);

          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceBuf0BaseHi(), baddr_hi);
          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceWptrAddr(), 0);
        } else {
          const uint32_t sqtt_reg_size = prim_->SqttBufferSizeValue(sqtt_size, baddr_hi);
          // Program size of buffer to use for thread trace
          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceSizeAddr(), sqtt_reg_size);
          // Program base address of buffer to use for thread trace
          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceBaseAddr(), baddr_lo);
        }

        // Program the thread trace mask
        const uint32_t mask_value =
            prim_->SqttMaskValue(config->targetCu, config->simd_sel, config->vmIdMask);
        WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceMaskAddr(), mask_value);

        uint32_t token_mask = (config->occupancy_mode)
                                  ? prim_->SqttTokenMaskOccupancyValue()
                                  : prim_->SqttTokenMaskOnValue(xcc_number_ > 1);
        if (((1 << global_se) & config->se_mask) == 0)
          token_mask = prim_->SqttTokenMaskOffValue();

        WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceTokenMaskAddr(), token_mask);
        // Program the thread trace ctrl register
        WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceCtrlAddr(), ctrl_val);
        // If we are in double buffer mode
        if (!config->buffer_data.empty())
        {
          if (prim_->GetGfxipLevel() != 12) throw std::runtime_error("Not supported");

          uint64_t buf1_addr = reinterpret_cast<uint64_t>(config->buffer_data.at(global_se).at(0));
          unsigned buff1_lo = Low32(buf1_addr >> tt_buff_align_shift);
          unsigned buff1_hi = High32(buf1_addr >> tt_buff_align_shift) & 0x3FFFu;

          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceBuf1Size(),
                            prim_->SqttBuffer0SizeValue(sqtt_size));
          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceBuf1BaseLo(), buff1_lo);
          builder_->BuildWriteWaitIdlePacket(cmd_buffer);
          WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceBuf1BaseHi(), buff1_hi);
        }
        base_addr += sqtt_size;
      }
      for (uint64_t local_se = 0; local_se < se_number_xcc; local_se++)
      {
        if (config->target_cu_per_se.at(local_se + se_number_xcc * xcc) < 0) continue;  // Ignore masked SEs

        Select_GRBM_SE_SH0(cmd_buffer, local_se);
        builder_->BuildWriteShRegPacket(cmd_buffer, prim_->GetComputeThreadTraceEnableAddr(), 1);
      }
    }
    // Reset the GRBM to broadcast mode
    SetGRBMToBroadcast(cmd_buffer);
  }
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);

  rocprof_trace_decoder_instrument_enable_t header{};
  header.char1 = '\0';
  header.char2 = 'R';
  header.char3 = 'O';
  header.char4 = 'C';
  auto userdata_channel = prim_->GetSqThreadTraceUserdata2();

  builder_->BuildWriteUConfigRegPacket(cmd_buffer, userdata_channel, header.u32All);
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, userdata_channel, 524801);

  rocprof_trace_decoder_packet_header_t packet{};
  packet.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO;

  if (config->enable_rt_timestamp)
  {
    packet.type = ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ;
    packet.data20 = this->timestamp_freq / 1000;
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, userdata_channel, packet.u32All);
  }
  if (prim_->GetGfxipLevel() == 9 && config->perfcounters.size())
  {
    packet.type = ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL;
    packet.data20 = (1 + cu_per_se) * ((config->perfcounters.size() + 3) & ~3) * config->perfPeriod;
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, userdata_channel, packet.u32All);
  }
  if (prim_->GetGfxipLevel() == 9 && config->enable_rt_timestamp)
  {
    for (size_t xcc = 0; xcc < GetXCCNumber(); xcc++)
    {
      if (!isXccEnabled(xcc, se_number_xcc, config)) continue;

      XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), xcc);
      auto& control = reinterpret_cast<TraceControl*>(config->control_buffer_ptr)[xcc];
      InsertTimestampMarker(cmd_buffer, &control.gpu_clock_cnt_start);
    }
  }
}

void GpuSqttBuilder::End(CmdBuffer* cmd_buffer, TraceConfig* config) {
  SetGRBMToBroadcast(cmd_buffer);
  // Issue a CSPartialFlush cmd including cache flush
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
  const uint32_t se_number_xcc = se_number_total / std::max(1u, GetXCCNumber());

  if (prim_->GetGfxipLevel() == 9)
  {
    if (config->enable_rt_timestamp)
    {
      for (size_t xcc = 0; xcc < GetXCCNumber(); xcc++)
      {
        if (!isXccEnabled(xcc, se_number_xcc, config)) continue;

        XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), xcc);
        auto& control = reinterpret_cast<TraceControl*>(config->control_buffer_ptr)[xcc];
        InsertTimestampMarker(cmd_buffer, &control.gpu_clock_cnt_end);
      }
      builder_->BuildWriteWaitIdlePacket(cmd_buffer);
    }

    // Program the thread trace mode register to disable thread trace
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceModeAddr(),
                                         prim_->SqttModeOffValue());
    // Issue a CSPartialFlush cmd including cache flush
    builder_->BuildWriteWaitIdlePacket(cmd_buffer);

    if (config->perfcounters.size()) StopPerfMon(cmd_buffer);

    // Iterate through the list of SE's and read the Status, Counter and
    // Write Pointer registers of Thread Trace subsystem
    for (size_t se_index = 0; se_index < se_number_total; se_index++) {
      if (config->target_cu_per_se.at(se_index) < 0) continue;

      size_t xcc_index = se_index / se_number_xcc;
      size_t se_index_xcc = se_index % se_number_xcc;

      XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), xcc_index);

      // Program Grbm to direct writes to one SE
      Select_GRBM_SE_SH0(cmd_buffer, se_index_xcc);

      // Issue WaitRegMem command to wait until SQTT event has completed
      const uint32_t mask_val = prim_->SqttBusyMask();
      auto status_offset = prim_->GetSqThreadTraceStatusOffset();
      builder_->BuildWaitRegMemCommand(cmd_buffer, false, status_offset, false, mask_val, 1);

      ReadValues(cmd_buffer, config, se_index);
    }
    // Reset the GRBM to broadcast mode
    SetGRBMToBroadcast(cmd_buffer);
    // Initialize cache flush request object
    builder_->BuildCacheFlushPacket(cmd_buffer, size_t(config->control_buffer_ptr),
                                    config->control_buffer_size);
    // Program zero size of thread trace buffer
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceSizeAddr(),
                                         prim_->SqttZeroSizeValue());
    // Program the thread trace ctrl register
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceCtrlAddr(),
                                         prim_->SqttCtrlValue(true, false));
    // Issue a CSPartialFlush cmd including cache flush
    builder_->BuildWriteWaitIdlePacket(cmd_buffer);
  } else {
    SetGRBMToBroadcast(cmd_buffer);
    builder_->BuildWriteShRegPacket(cmd_buffer, prim_->GetComputeThreadTraceEnableAddr(), 0);

    if (prim_->GetGfxipLevel() >= 12) builder_->BuildThreadTraceEventFinish(cmd_buffer);

    {
      // Wait for FINISH_PENDING
      const uint32_t mask_val = prim_->SqttPendingMask();
      auto status_offset = prim_->GetSqThreadTraceStatusAddr();
      builder_->BuildWaitRegMemCommand(cmd_buffer, false, status_offset, true, mask_val, 0);
    }

    // Program the thread trace ctrl register to set mode to 0
    const uint32_t ctrl_val = prim_->SqttCtrlValue(false, false);
    WriteConfigPacket(cmd_buffer, prim_->GetSqThreadTraceCtrlAddr(), ctrl_val);

    {
      // Wait until SQTT_BUSY is 0
      const uint32_t mask_val = prim_->SqttBusyMask();
      auto status_offset = prim_->GetSqThreadTraceStatusAddr();
      builder_->BuildWaitRegMemCommand(cmd_buffer, false, status_offset, true, mask_val, 0);
    }

    for (int xcc=0; xcc<(int)xcc_number_; xcc++)
    {
      if (!isXccEnabled(xcc, se_number_xcc, config)) continue;

      XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), xcc);
      for (uint64_t index = 0; index < se_number_xcc; index++)
      {
        Select_GRBM_SE_SH0(cmd_buffer, index);
        ReadValues(cmd_buffer, config, index + xcc*se_number_xcc);
      }
    }

    // Reset the GRBM to broadcast mode
    SetGRBMToBroadcast(cmd_buffer);
  }

  if (prim_->GetGfxipLevel() != 10)
    builder_->BuildCacheFlushPacket(cmd_buffer, size_t(config->control_buffer_ptr),
                                    config->control_buffer_size);
  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
}

void GpuSqttBuilder::ReadValues(CmdBuffer* cmd_buffer, const TraceConfig* config, size_t se_index) {
  // Retrieve the values from various status registers
  auto& control = reinterpret_cast<TraceControl*>(config->control_buffer_ptr)[se_index];
  const uint32_t copy_data_sel = prim_->GetCopyDataSelCountOneDw();

  builder_->BuildCopyRegDataPacket(cmd_buffer, prim_->GetSqThreadTraceStatusAddr(),
                                   &control.status, copy_data_sel, true);

  builder_->BuildCopyRegDataPacket(cmd_buffer, prim_->GetSqThreadTraceCntrAddr(), &control.cntr,
                                   copy_data_sel, true);

  builder_->BuildCopyRegDataPacket(cmd_buffer, prim_->GetSqThreadTraceWptrAddr(), &control.wptr,
                                   copy_data_sel, true);

  if (prim_->GetGfxipLevel() >= 12)
    builder_->BuildCopyRegDataPacket(cmd_buffer, prim_->GetSqThreadTraceStatus2Addr(),
                                     &control.status2, copy_data_sel, true);
}

uint64_t GpuSqttBuilder::PopCount(uint64_t se_mask) const {
  uint64_t num_enabled = 0;
  while (se_mask) {
    num_enabled += se_mask & 1;
    se_mask >>= 1;
  }
  return std::max<uint64_t>(num_enabled, 1u);
}

bool GpuSqttBuilder::isXccEnabled(int xcc, uint64_t se_number_xcc, TraceConfig* config) {
  for (size_t index = 0; index < se_number_xcc; index++)
    if (config->target_cu_per_se.at(xcc*se_number_xcc + index) >= 0)
      return true;

  return false;
}

uint64_t GpuSqttBuilder::GetBaseStep(TraceConfig* config) const {
  // Get number of selected shader engines
  uint64_t num_enabled = PopCount(config->se_mask);
  int64_t size_disabled = (64 - num_enabled) * config->capacity_per_disabled_se;

  // Make sure num divides buffersize
  int64_t buffer_per_se = (config->data_buffer_size - size_disabled) / num_enabled;
  const size_t tt_buff_align_shift = prim_->GetTtBuffAlignShift();
  return uint64_t(buffer_per_se) & ~((1 << tt_buff_align_shift) - 1);
}

hsa_status_t GpuSqttBuilder::InsertCodeobjMarker(CmdBuffer* cmd_buffer, uint32_t data,
                                                  unsigned channel) {
  rocprof_trace_decoder_packet_header_t header{};
  header.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
  header.type = channel;
  auto userdata_channel = prim_->GetSqThreadTraceUserdata2();

  SetGRBMToBroadcast(cmd_buffer);
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, userdata_channel, header.u32All);
  builder_->BuildWriteUConfigRegPacket(cmd_buffer, userdata_channel, data);
  return HSA_STATUS_SUCCESS;
}

void GpuSqttBuilder::InsertTimestampMarker(CmdBuffer* cmd_buffer, uint64_t* addr) {
  rocprof_trace_decoder_packet_header_t header{};
  header.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP;
  header.type = 0;
  header.data20 = 0;

  SetGRBMToBroadcast(cmd_buffer);
  builder_->BuildGPUClockPacket(cmd_buffer, addr, prim_->GetSqThreadTraceUserdata3(),
                                header.u32All);
}

void GpuSqttBuilder::WriteConfigPacket(CmdBuffer* cmdbuf, const Register& reg, uint32_t value) {
  if (prim_->GetGfxipLevel() == 11)
    builder_->BuildWriteUConfigRegPacket(cmdbuf, reg, value);
  else
    builder_->BuildWritePConfigRegPacket(cmdbuf, reg, value);
}

void GpuSqttBuilder::GetStatusPacket(CmdBuffer* cmd_buffer, TraceConfig* config,
                                     TraceControl& control, int se_id) {
  int se_per_xcc = se_number_total / GetXCCNumber();
  XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), se_id / se_per_xcc);
  Select_GRBM_SE_SH0(cmd_buffer, se_id % se_per_xcc);

  auto status_addr = (prim_->GetGfxipLevel() >= 12) ? prim_->GetSqThreadTraceStatus2Addr()
                                                     : prim_->GetSqThreadTraceStatusAddr();
  builder_->BuildCopyRegDataPacket(cmd_buffer, status_addr, &control.status_double_buffer,
                                   prim_->GetCopyDataSelCountOneDw(), false);

  builder_->BuildWriteWaitIdlePacket(cmd_buffer);
  builder_->BuildCacheFlushPacket(cmd_buffer, size_t(&control), sizeof(TraceControl));
  SetGRBMToBroadcast(cmd_buffer);
}

void GpuSqttBuilder::Swapbuffer(CmdBuffer* cmd_buffer, TraceConfig* config, void* addr,
                                 void* prev, int se_id, bool buf1) {
  int se_per_xcc = se_number_total / GetXCCNumber();
  uint64_t base_addr = reinterpret_cast<uint64_t>(addr);
  const size_t tt_buff_align_shift = prim_->GetTtBuffAlignShift();

  XCC_Packet_Lock lock(*builder_, cmd_buffer, GetXCCNumber(), se_id / se_per_xcc);
  Select_GRBM_SE_SH0(cmd_buffer, se_id % se_per_xcc);

  if (prim_->GetGfxipLevel() == 9)
  {
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceBaseAddr(),
                                         prim_->SqttBaseValueLo(base_addr));
    builder_->BuildWriteUConfigRegPacket(cmd_buffer, prim_->GetSqThreadTraceBase2Addr(),
                                         prim_->SqttBaseValueHi(base_addr));
  }
  else
  {
    unsigned buff1_lo = Low32(base_addr >> tt_buff_align_shift);
    unsigned buff1_hi = High32(base_addr >> tt_buff_align_shift) & 0x3FFFu;

    auto reg_lo = buf1 ? prim_->GetSqThreadTraceBuf1BaseLo() : prim_->GetSqThreadTraceBuf0BaseLo();
    auto reg_hi = buf1 ? prim_->GetSqThreadTraceBuf1BaseHi() : prim_->GetSqThreadTraceBuf0BaseHi();

    WriteConfigPacket(cmd_buffer, reg_lo, buff1_lo);
    builder_->BuildWriteWaitIdlePacket(cmd_buffer);
    WriteConfigPacket(cmd_buffer, reg_hi, buff1_hi);
  }
  builder_->BuildCacheFlushPacket(cmd_buffer, size_t(prev), config->data_buffer_size);

  SetGRBMToBroadcast(cmd_buffer);
}

}  // namespace pm4_builder
