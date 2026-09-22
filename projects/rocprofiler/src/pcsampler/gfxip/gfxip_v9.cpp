/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <cassert>
#include <cstring>
#include <fstream>
#include <cassert>
#include <optional>

#include <sys/types.h>
#include <unistd.h>
#include <linux/types.h>
#include <linux/ioctl.h>

#include <hsa/hsa.h>

#include "rocprofiler.h"
#include "src/core/hsa/hsa_support.h"
#include "src/pcsampler/session/pc_sampler.h"
#include "gfxip.h"
#include "vega10_enum.h"
#include "gc/gc_9_0_offset.h"
#include "gc/gc_9_0_sh_mask.h"
#include "osssys/osssys_4_2_0_offset.h"
#include "osssys/osssys_4_2_0_sh_mask.h"

namespace rocprofiler::pc_sampler::gfxip {

namespace {

uint32_t read_sq_register(const device_t& dev, uint32_t simd, uint32_t wave_id,
                          uint32_t register_address) {
  uint32_t data = REG_SET_FIELD(0, SQ_IND_INDEX, WAVE_ID, wave_id);
  data = REG_SET_FIELD(data, SQ_IND_INDEX, SIMD_ID, simd);
  data = REG_SET_FIELD(data, SQ_IND_INDEX, INDEX, register_address);
  dev.pci_memory_[REG_OFFSET(GC, 0, mmSQ_IND_INDEX)] = data;
  return dev.pci_memory_[REG_OFFSET(GC, 0, mmSQ_IND_DATA)];
}

uint32_t debugfs_ioctl_read_sq_register(const device_t& dev,
                                        const struct amdgpu_debugfs_regs2_iocdata& ioc,
                                        const uint32_t simd, const uint32_t wave_id,
                                        const uint32_t register_address) {
  uint32_t data = REG_SET_FIELD(0, SQ_IND_INDEX, WAVE_ID, wave_id);
  data = REG_SET_FIELD(data, SQ_IND_INDEX, SIMD_ID, simd);
  data = REG_SET_FIELD(data, SQ_IND_INDEX, INDEX, register_address);
  debugfs_ioctl_write_register(dev, ioc, REG_OFFSET(GC, 0, mmSQ_IND_INDEX), data);
  return debugfs_ioctl_read_register(dev, ioc, REG_OFFSET(GC, 0, mmSQ_IND_DATA));
}

void fill_record(const device_t& dev, rocprofiler_record_pc_sample_t* record, uint32_t se,
                 uint64_t pc, hsa_kernel_dispatch_packet_t* pkt) {
  /*
   * XXX: Use of the reserved2 field in the HSA dispatch packet to uniquely
   * identify kernel dispatches for PC sampling is an internal implementation
   * detail which is subject to change.  See the comment associated with
   * rocprofiler::rocprofiler::kernel_dispatch_counter_.
   */
  record->pc_sample.dispatch_id = rocprofiler_kernel_dispatch_id_t{pkt->reserved2};

  /*
   * TODO: Fill this with gpu_clock_counter via AMDKFD_IOC_GET_CLOCK_COUNTERS,
   * or just call hsaKmtGetClockCounters, which provides the same info.  The ioctl
   * wants the device's kfd gpu_id (*not* the HSA node_id), whereas the hsaKmt
   * function takes an HSA node_id and translates it.
   *
   * The caller should fix up the field by adding a CPU TSC delta for each
   * waveslot read, though the CPU TSC needs translation into the GPU's clock
   * domain.  ROCr has code to translate timestamps from the GPU's clock domain
   * to the system clock domain, but not the other way around, so this would
   * need to be written.
   *
   * Future sampling methods may fill this in automatically from the GPU's
   * real-time counter.
   */
  // record->pc_sample.cycle = 0;
  rocprofiler_get_timestamp(&record->pc_sample.timestamp);

  record->pc_sample.pc = pc;
  record->pc_sample.se = se;
  const auto& hdl = dev.agent_info_.getHandle();

  /*
   * XXX FIXME: For consistency, this is the same method as used by
   * rocprofiler::queue::AsyncSignalHandler in queue.cpp to fill
   * rocprofiler_record_profiler_t::gpu_id, but it should be changed; see the
   * comment in rocprofiler::hsa_support::Initialize about using KFD's gpu_id for
   * more information.
   */
  record->pc_sample.gpu_id = rocprofiler_agent_id_t{
      HSASupport_Singleton::GetInstance().GetHSAAgentInfo(hdl).GetDeviceInfo().getGPUId()};
}

}  // namespace

void read_pc_samples_v9(const device_t& dev, PCSampler* sampler) {
  assert(sampler);

  uint32_t saved_grbm_gfx_index = dev.pci_memory_[REG_OFFSET(GC, 0, mmGRBM_GFX_INDEX)];
  uint32_t data;

  for (uint32_t se = 0; se < dev.agent_info_.GetDeviceInfo().getShaderEngineCount(); ++se)
    for (uint32_t sh = 0; sh < dev.agent_info_.GetDeviceInfo().getShaderArraysPerSE(); ++sh)
      for (uint32_t cu = 0; cu < dev.agent_info_.GetDeviceInfo().getCUCountPerSH(); ++cu) {
        // Select the SE, SH, and CU.
        data = REG_SET_FIELD(0, GRBM_GFX_INDEX, INSTANCE_INDEX, cu);
        data = REG_SET_FIELD(data, GRBM_GFX_INDEX, SH_INDEX, sh);
        data = REG_SET_FIELD(data, GRBM_GFX_INDEX, SE_INDEX, se);
        dev.pci_memory_[REG_OFFSET(GC, 0, mmGRBM_GFX_INDEX)] = data;

        // Iterate over all the waves in the compute unit.
        for (uint32_t simd = 0; simd < dev.agent_info_.GetDeviceInfo().getSimdCountPerCU(); ++simd)
          for (uint32_t wave_id = 0; wave_id < dev.agent_info_.GetDeviceInfo().getWaveSlotsPerSimd(); ++wave_id) {
            // FatalHalt the wave
            data = REG_SET_FIELD(0, SQ_CMD, CMD, SQ_IND_CMD_CMD_SETFATALHALT);
            data = REG_SET_FIELD(data, SQ_CMD, MODE, SQ_IND_CMD_MODE_SINGLE);
            data = REG_SET_FIELD(data, SQ_CMD, DATA, 1);
            data = REG_SET_FIELD(data, SQ_CMD, WAVE_ID, wave_id);
            data = REG_SET_FIELD(data, SQ_CMD, SIMD_ID, simd);
            dev.pci_memory_[REG_OFFSET(GC, 0, mmSQ_CMD)] = data;

            // Skip this slot if the wave is not valid.
            uint32_t status = read_sq_register(dev, simd, wave_id, ixSQ_WAVE_STATUS);
            if (!REG_GET_FIELD(status, SQ_WAVE_STATUS, VALID)) continue;

            uint32_t hw_id = read_sq_register(dev, simd, wave_id, ixSQ_WAVE_HW_ID);
            uint32_t vm_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, VM_ID);

            rocprofiler_record_pc_sample_t record;

            // If the wave's PASID matches the process', read and report the PC
            // and dispatch packet for the wave.
            std::optional<uint64_t> pc;
            if (dev.pci_memory_[REG_OFFSET(OSSSYS, 0, mmIH_VMID_0_LUT) + vm_id] == pasid()) {
              pc = (uint64_t)read_sq_register(dev, simd, wave_id, ixSQ_WAVE_PC_HI) << 32 |
                  read_sq_register(dev, simd, wave_id, ixSQ_WAVE_PC_LO);

              // The dispatch index into the queue
              uint32_t disp_idx = read_sq_register(dev, simd, wave_id, ixSQ_WAVE_TTMP6);

              // Set up reading CP_HQD_PQ_BASE and CP_HQD_PQ_BASE_HI
              uint32_t pipe_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, PIPE_ID);
              uint32_t queue_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, QUEUE_ID);
              uint32_t me_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, ME_ID);
              data = REG_SET_FIELD(0, GRBM_GFX_CNTL, PIPEID, pipe_id);
              data = REG_SET_FIELD(data, GRBM_GFX_CNTL, QUEUEID, queue_id);
              data = REG_SET_FIELD(data, GRBM_GFX_CNTL, MEID, me_id);
              data = REG_SET_FIELD(data, GRBM_GFX_CNTL, VMID, vm_id);
              dev.pci_memory_[REG_OFFSET(GC, 0, mmGRBM_GFX_CNTL)] = data;

              uint32_t pq_base_lo = dev.pci_memory_[REG_OFFSET(GC, 0, mmCP_HQD_PQ_BASE)];
              uint32_t pq_base_hi = dev.pci_memory_[REG_OFFSET(GC, 0, mmCP_HQD_PQ_BASE_HI)] & 0xff;
              uint64_t pq_base = (uint64_t)pq_base_hi << 40 | (uint64_t)pq_base_lo << 8;
              uint32_t cp_hqd_pq_control_queue_size =
                  dev.pci_memory_[REG_OFFSET(GC, 0, mmCP_HQD_PQ_CONTROL)] & 0x3f;
              uint32_t queue_size = 1 << (cp_hqd_pq_control_queue_size + 1);

              auto pkt = (hsa_kernel_dispatch_packet_t*)(pq_base +
                                                         disp_idx % queue_size *
                                                             sizeof(hsa_kernel_dispatch_packet_t));
              fill_record(dev, &record, se, *pc, pkt);
            }

            // Resume the wave.
            data = REG_SET_FIELD(0, SQ_CMD, CMD, SQ_IND_CMD_CMD_SETFATALHALT);
            data = REG_SET_FIELD(data, SQ_CMD, MODE, SQ_IND_CMD_MODE_SINGLE);
            data = REG_SET_FIELD(data, SQ_CMD, DATA, 0);
            data = REG_SET_FIELD(data, SQ_CMD, WAVE_ID, wave_id);
            data = REG_SET_FIELD(data, SQ_CMD, SIMD_ID, simd);
            dev.pci_memory_[REG_OFFSET(GC, 0, mmSQ_CMD)] = data;

            if (pc && record.pc_sample.dispatch_id.value != 0) {
              sampler->AddRecord(record);
            }
          }
      }

  // Restore the GRBM_GFX_INDEX register.
  dev.pci_memory_[REG_OFFSET(GC, 0, mmGRBM_GFX_INDEX)] = saved_grbm_gfx_index;
}

void read_pc_samples_v9_ioctl(const device_t& dev, PCSampler* sampler) {
  assert(sampler);

  struct amdgpu_debugfs_regs2_iocdata ioc {};
  ioc.use_grbm = 1;

  uint32_t data;

  for (uint32_t se = 0; se < dev.agent_info_.GetDeviceInfo().getShaderEngineCount(); ++se)
    for (uint32_t sh = 0; sh < dev.agent_info_.GetDeviceInfo().getShaderArraysPerSE(); ++sh)
      for (uint32_t cu = 0; cu < dev.agent_info_.GetDeviceInfo().getCUCountPerSH(); ++cu) {
        ioc.grbm.se = se;
        ioc.grbm.sh = sh;
        ioc.grbm.instance = cu;

        // Iterate over all the waves in the compute unit.
        for (uint32_t simd = 0; simd < dev.agent_info_.GetDeviceInfo().getSimdCountPerCU(); ++simd)
          for (uint32_t wave_id = 0; wave_id < dev.agent_info_.GetDeviceInfo().getWaveSlotsPerSimd(); ++wave_id) {
            // FatalHalt the wave
            data = REG_SET_FIELD(0, SQ_CMD, CMD, SQ_IND_CMD_CMD_SETFATALHALT);
            data = REG_SET_FIELD(data, SQ_CMD, MODE, SQ_IND_CMD_MODE_SINGLE);
            data = REG_SET_FIELD(data, SQ_CMD, DATA, 1);
            data = REG_SET_FIELD(data, SQ_CMD, WAVE_ID, wave_id);
            data = REG_SET_FIELD(data, SQ_CMD, SIMD_ID, simd);
            debugfs_ioctl_write_register(dev, ioc, REG_OFFSET(GC, 0, mmSQ_CMD), data);

            // Skip this slot if the wave is not valid.
            debugfs_ioctl_set_state(dev, ioc);
            uint32_t status =
                debugfs_ioctl_read_sq_register(dev, ioc, simd, wave_id, ixSQ_WAVE_STATUS);
            if (!REG_GET_FIELD(status, SQ_WAVE_STATUS, VALID)) continue;

            debugfs_ioctl_set_state(dev, ioc);
            uint32_t hw_id =
                debugfs_ioctl_read_sq_register(dev, ioc, simd, wave_id, ixSQ_WAVE_HW_ID);
            uint32_t vm_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, VM_ID);

            rocprofiler_record_pc_sample_t record;

            // If the wave's PASID matches the process', read and report the PC
            // and dispatch packet for the wave.
            std::optional<uint64_t> pc;
            if (debugfs_ioctl_read_register(
                    dev, ioc, REG_OFFSET(OSSSYS, 0, mmIH_VMID_0_LUT) + vm_id) == pasid()) {
              pc =
                  (uint64_t)debugfs_ioctl_read_sq_register(dev, ioc, simd, wave_id, ixSQ_WAVE_PC_HI)
                      << 32 |
                  debugfs_ioctl_read_sq_register(dev, ioc, simd, wave_id, ixSQ_WAVE_PC_LO);

              // The dispatch index into the queue
              uint32_t disp_idx =
                  debugfs_ioctl_read_sq_register(dev, ioc, simd, wave_id, ixSQ_WAVE_TTMP6);

              // Set up reading CP_HQD_PQ_BASE and CP_HQD_PQ_BASE_HI
              uint32_t pipe_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, PIPE_ID);
              uint32_t queue_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, QUEUE_ID);
              uint32_t me_id = REG_GET_FIELD(hw_id, SQ_WAVE_HW_ID, ME_ID);
              data = REG_SET_FIELD(0, GRBM_GFX_CNTL, PIPEID, pipe_id);
              data = REG_SET_FIELD(data, GRBM_GFX_CNTL, QUEUEID, queue_id);
              data = REG_SET_FIELD(data, GRBM_GFX_CNTL, MEID, me_id);
              data = REG_SET_FIELD(data, GRBM_GFX_CNTL, VMID, vm_id);
              debugfs_ioctl_write_register(dev, ioc, REG_OFFSET(GC, 0, mmGRBM_GFX_CNTL), data);

              uint32_t pq_base_lo =
                  debugfs_ioctl_read_register(dev, ioc, REG_OFFSET(GC, 0, mmCP_HQD_PQ_BASE));
              uint32_t pq_base_hi =
                  debugfs_ioctl_read_register(dev, ioc, REG_OFFSET(GC, 0, mmCP_HQD_PQ_BASE_HI)) &
                  0xff;
              uint64_t pq_base = (uint64_t)pq_base_hi << 40 | (uint64_t)pq_base_lo << 8;
              uint32_t cp_hqd_pq_control_queue_size =
                  debugfs_ioctl_read_register(dev, ioc, REG_OFFSET(GC, 0, mmCP_HQD_PQ_CONTROL)) &
                  0x3f;
              uint32_t queue_size = 1 << (cp_hqd_pq_control_queue_size + 1);

              auto pkt = (hsa_kernel_dispatch_packet_t*)(pq_base +
                                                         disp_idx % queue_size *
                                                             sizeof(hsa_kernel_dispatch_packet_t));
              fill_record(dev, &record, se, *pc, pkt);
            }

            // Resume the wave.
            data = REG_SET_FIELD(0, SQ_CMD, CMD, SQ_IND_CMD_CMD_SETFATALHALT);
            data = REG_SET_FIELD(data, SQ_CMD, MODE, SQ_IND_CMD_MODE_SINGLE);
            data = REG_SET_FIELD(data, SQ_CMD, DATA, 0);
            data = REG_SET_FIELD(data, SQ_CMD, WAVE_ID, wave_id);
            data = REG_SET_FIELD(data, SQ_CMD, SIMD_ID, simd);
            debugfs_ioctl_write_register(dev, ioc, REG_OFFSET(GC, 0, mmSQ_CMD), data);

            if (pc && record.pc_sample.dispatch_id.value != 0) {
              sampler->AddRecord(record);
            }
          }
      }
}

}  // namespace rocprofiler::pc_sampler::gfxip
