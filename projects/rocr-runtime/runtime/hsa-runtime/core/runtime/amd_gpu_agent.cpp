////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2023, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_gpu_agent.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <climits>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <iomanip>
#include <cmath>

#include "core/inc/amd_aql_queue.h"
#include "core/inc/amd_blit_kernel.h"
#include "core/inc/amd_blit_sdma.h"
#include "core/inc/amd_gpu_pm4.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/default_signal.h"
#include "core/inc/interrupt_signal.h"
#include "core/inc/isa.h"
#include "core/inc/runtime.h"
#include "core/util/os.h"
#include "inc/hsa_ext_image.h"
#include "inc/hsa_ven_amd_aqlprofile.h"
#include "inc/hsa_ven_amd_pc_sampling.h"

#include "core/inc/amd_trap_handler_v1.h"
#include "core/inc/amd_blit_shaders.h"
#include "core/inc/hsa_api_trace_int.h"
// Generated header
#include "amd_trap_handler_v2.h"
#include "amd_blit_shaders_v2.h"

#if defined(__linux__)
// libdrm headers
#include <xf86drm.h>
#include <amdgpu.h>
#endif


// Size of scratch (private) segment pre-allocated per thread, in bytes.
#define DEFAULT_SCRATCH_BYTES_PER_THREAD 2048
#define MAX_WAVE_SCRATCH 8387584  // See COMPUTE_TMPRING_SIZE.WAVESIZE
#define MAX_NUM_DOORBELLS 0x400

namespace rocr {

namespace AMD {
const uint64_t CP_DMA_DATA_TRANSFER_CNT_MAX = (1 << 26);

GpuAgent::GpuAgent(HSAuint32 node, const HsaNodeProperties& node_props, bool xnack_mode,
                   uint32_t index, core::DriverType driver_type)
    : GpuAgentInt(node, driver_type),
      properties_(node_props),
      current_coherency_type_(HSA_AMD_COHERENCY_TYPE_COHERENT),
      scratch_used_large_(0),
      queues_(),
      trap_code_buf_(NULL),
      trap_code_buf_size_(0),
      doorbell_queue_map_(NULL),
      memory_bus_width_(0),
      memory_max_frequency_(0),
      enum_index_(index),
      ape1_base_(0),
      pending_copy_req_ref_(0),
      pending_copy_stat_check_ref_(0),
      sdma_blit_used_mask_(0),
      scratch_limit_async_threshold_(0),
      scratch_cache_(
          [this](void* base, size_t size, bool large) { ReleaseScratch(base, size, large); }),
      trap_handler_tma_region_(NULL),
      rec_sdma_eng_override_(false),
      pcs_hosttrap_data_(),
      pcs_stochastic_data_(),
      xgmi_cpu_gpu_(false),
      large_bar_enabled_(false){
  const bool is_apu_node = (properties_.NumCPUCores > 0);
  profile_ = (is_apu_node) ? HSA_PROFILE_FULL : HSA_PROFILE_BASE;

  if (node_props.Capability.ui32.DoorbellType != 2)
    throw AMD::hsa_exception(HSA_STATUS_ERROR, "Agent creation failed.\nThe GPU node uses a deprecated doorbell type\n");

  hsa_status_t err = driver().GetClockCounters(node_id(), &t0_);
  t1_ = t0_;
  historical_clock_ratio_ = 0.0;
  assert(err == HSA_STATUS_SUCCESS && "hsaGetClockCounters error");

  const core::Isa *isa_base;

  if (node_props.OverrideEngineId.Value != 0) {
     isa_base = core::IsaRegistry::GetIsa(
         core::Isa::Version(node_props.OverrideEngineId.ui32.Major,
                            node_props.OverrideEngineId.ui32.Minor,
                            node_props.OverrideEngineId.ui32.Stepping));
  } else {
     isa_base = core::IsaRegistry::GetIsa(
         core::Isa::Version(node_props.EngineId.ui32.Major,
                            node_props.EngineId.ui32.Minor,
                            node_props.EngineId.ui32.Stepping));
  }

  if (!isa_base) {
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ISA, "Agent creation failed.\nThe GPU node has an unrecognized id.\n");
  }

  rocr::core::IsaFeature sramecc = rocr::core::IsaFeature::Unsupported;
  if (isa_base->IsSrameccSupported()) {
    switch (core::Runtime::runtime_singleton_->flag().sramecc_enable()) {
      case Flag::SRAMECC_DISABLED:
        sramecc = core::IsaFeature::Disabled;
        break;
      case Flag::SRAMECC_ENABLED:
        sramecc = core::IsaFeature::Enabled;
        break;
      case Flag::SRAMECC_DEFAULT:
        sramecc = node_props.Capability.ui32.SRAM_EDCSupport == 1 ? core::IsaFeature::Enabled
                                                                  : core::IsaFeature::Disabled;
        break;
    }
  }

  rocr::core::IsaFeature xnack = rocr::core::IsaFeature::Unsupported;
  if (isa_base->IsXnackSupported()) {
    // TODO: This needs to be obtained form KFD once HMM implemented.
    xnack = xnack_mode ? core::IsaFeature::Enabled
                      : core::IsaFeature::Disabled;
  }

  if (node_props.OverrideEngineId.Value != 0) {
    isa_ = (core::Isa*)core::IsaRegistry::GetIsa(
          core::Isa::Version(node_props.OverrideEngineId.ui32.Major, node_props.OverrideEngineId.ui32.Minor,
                             node_props.OverrideEngineId.ui32.Stepping), sramecc, xnack);
  } else {
  // Set instruction set architecture via node property, only on GPU device.
    isa_ = (core::Isa*)core::IsaRegistry::GetIsa(
          core::Isa::Version(node_props.EngineId.ui32.Major, node_props.EngineId.ui32.Minor,
                             node_props.EngineId.ui32.Stepping), sramecc, xnack);
  }

  assert(isa_ != nullptr && "ISA registry inconsistency.");

  supported_isas_.push_back(isa_);
  if (!isa_->GetIsaGeneric().empty()) {
    supported_isas_.push_back(core::IsaRegistry::GetIsa(isa_->GetIsaGeneric()));
  }

  current_coherency_type((profile_ == HSA_PROFILE_FULL)
                             ? HSA_AMD_COHERENCY_TYPE_COHERENT
                             : HSA_AMD_COHERENCY_TYPE_NONCOHERENT);

  max_queues_ = core::Runtime::runtime_singleton_->flag().max_queues();
#if !defined(HSA_LARGE_MODEL) || !defined(__linux__)
  if (max_queues_ == 0) {
    max_queues_ = 10;
  }
  max_queues_ = std::min(10U, max_queues_);
#else
  if (max_queues_ == 0) {
    max_queues_ = 128;
  }
  max_queues_ = std::min(128U, max_queues_);
#endif

  // Initialize libdrm device handle
  InitLibDrm();

#if !defined(__linux__)
  wallclock_frequency_ = 0;
#else
  bool model_enabled;
  hsa_status_t status = driver().IsModelEnabled(&model_enabled);
  assert(status == HSA_STATUS_SUCCESS && "IsModelEnabled failed");
  if (model_enabled) {
    wallclock_frequency_ = 0;
  } else {
    // Get wallclock freq
    err = driver().GetWallclockFrequency(node_id(), &wallclock_frequency_);
    if (err != HSA_STATUS_SUCCESS) {
      throw AMD::hsa_exception(err, "Agent creation failed.\nGetWallclockFrequency error.\n");
    }
  }
#endif

  auto& first_cpu = core::Runtime::runtime_singleton_->cpu_agents()[0];
  auto link_info = core::Runtime::runtime_singleton_->GetLinkInfo(first_cpu->node_id(), node_id());
  xgmi_cpu_gpu_ = (link_info.info.link_type == HSA_AMD_LINK_INFO_TYPE_XGMI);

  if (link_info.num_hop >= 1) {
    large_bar_enabled_ = true;
  }

  // Populate region list.
  InitRegionList();

  // Populate cache list.
  InitCacheList();

  // Initialize thresholds for async-scratch handling
  InitAsyncScratchThresholds();
}

GpuAgent::~GpuAgent() {
  std::for_each(regions_.begin(), regions_.end(), DeleteObject());
  regions_.clear();
}

void GpuAgent::AssembleShader(const char* func_name, AssembleTarget assemble_target,
                              void*& code_buf, size_t& code_buf_size) const {
  // Select precompiled shader implementation from name/target.
  struct ASICShader {
    const void* code;
    size_t size;
    int num_sgprs;
    int num_vgprs;
  };

  struct CompiledShader {
    ASICShader compute_7;
    ASICShader compute_8;
    ASICShader compute_9;
    ASICShader compute_90a;
    ASICShader compute_942;
    ASICShader compute_1010;
    ASICShader compute_10;
    ASICShader compute_11;
    ASICShader compute_12;
  };

  std::map<std::string, CompiledShader> compiled_shaders = {
      {"TrapHandler",
       {
           {NULL, 0, 0, 0},                                                 // gfx7
           {kCodeTrapHandler8, sizeof(kCodeTrapHandler8), 2, 4},            // gfx8
           {kCodeTrapHandler9, sizeof(kCodeTrapHandler9), 2, 4},            // gfx9
           {kCodeTrapHandler90a, sizeof(kCodeTrapHandler90a), 2, 4},        // gfx90a
           {NULL, 0, 0, 0},                                                 // gfx942
           {kCodeTrapHandler1010, sizeof(kCodeTrapHandler1010), 2, 4},      // gfx1010
           {kCodeTrapHandler10, sizeof(kCodeTrapHandler10), 2, 4},          // gfx10
           {NULL, 0, 0, 0},                                                 // gfx11
           // GFX12_TODO: Using one for GFX10 for now.
           //             If NULL is used (like GFX11), get an assert.
           {kCodeTrapHandler10, sizeof(kCodeTrapHandler10), 2, 4},          // gfx12
       }},
      {"TrapHandlerKfdExceptions",
       {
           {NULL, 0, 0, 0},                                                 // gfx7
           {kCodeTrapHandler8, sizeof(kCodeTrapHandler8), 2, 4},            // gfx8
           {kCodeTrapHandlerV2_9, sizeof(kCodeTrapHandlerV2_9), 2, 4},      // gfx9
           {kCodeTrapHandlerV2_9, sizeof(kCodeTrapHandlerV2_9), 2, 4},      // gfx90a
           {kCodeTrapHandlerV2_942, sizeof(kCodeTrapHandlerV2_942), 2, 4},  // gfx942
           {kCodeTrapHandlerV2_1010, sizeof(kCodeTrapHandlerV2_1010), 2, 4},// gfx1010
           {kCodeTrapHandlerV2_10, sizeof(kCodeTrapHandlerV2_10), 2, 4},    // gfx10
           {kCodeTrapHandlerV2_11, sizeof(kCodeTrapHandlerV2_11), 2, 4},    // gfx11
           {kCodeTrapHandlerV2_12, sizeof(kCodeTrapHandlerV2_12), 2, 4},    // gfx12
       }},
      {"CopyAligned",
       {
           {kCodeCopyAligned7, sizeof(kCodeCopyAligned7), 32, 12},          // gfx7
           {kCodeCopyAligned8, sizeof(kCodeCopyAligned8), 32, 12},          // gfx8
           {kCodeCopyAligned9, sizeof(kCodeCopyAligned9), 32, 12},          // gfx9
           {kCodeCopyAligned9, sizeof(kCodeCopyAligned9), 32, 12},          // gfx90a
           {kCodeCopyAligned9, sizeof(kCodeCopyAligned9), 32, 12},          // gfx942
           {kCodeCopyAligned10, sizeof(kCodeCopyAligned10), 32, 12},        // gfx1010
           {kCodeCopyAligned10, sizeof(kCodeCopyAligned10), 32, 12},        // gfx10
           {kCodeCopyAligned11, sizeof(kCodeCopyAligned11), 32, 12},        // gfx11
           {kCodeCopyAligned12, sizeof(kCodeCopyAligned12), 32, 12},        // gfx12
       }},
      {"CopyMisaligned",
       {
           {kCodeCopyMisaligned7, sizeof(kCodeCopyMisaligned7), 23, 10},    // gfx7
           {kCodeCopyMisaligned8, sizeof(kCodeCopyMisaligned8), 23, 10},    // gfx8
           {kCodeCopyMisaligned9, sizeof(kCodeCopyMisaligned9), 23, 10},    // gfx9
           {kCodeCopyMisaligned9, sizeof(kCodeCopyMisaligned9), 23, 10},    // gfx90a
           {kCodeCopyMisaligned9, sizeof(kCodeCopyMisaligned9), 23, 10},    // gfx942
           {kCodeCopyMisaligned10, sizeof(kCodeCopyMisaligned10), 23, 10},  // gfx1010
           {kCodeCopyMisaligned10, sizeof(kCodeCopyMisaligned10), 23, 10},  // gfx10
           {kCodeCopyMisaligned11, sizeof(kCodeCopyMisaligned11), 23, 10},  // gfx11
           {kCodeCopyMisaligned12, sizeof(kCodeCopyMisaligned12), 23, 10},  // gfx12
       }},
      {"Fill",
       {
           {kCodeFill7, sizeof(kCodeFill7), 19, 8},                         // gfx7
           {kCodeFill8, sizeof(kCodeFill8), 19, 8},                         // gfx8
           {kCodeFill9, sizeof(kCodeFill9), 19, 8},                         // gfx9
           {kCodeFill9, sizeof(kCodeFill9), 19, 8},                         // gfx90a
           {kCodeFill9, sizeof(kCodeFill9), 19, 8},                         // gfx942
           {kCodeFill10, sizeof(kCodeFill10), 19, 8},                       // gfx1010
           {kCodeFill10, sizeof(kCodeFill10), 19, 8},                       // gfx10
           {kCodeFill11, sizeof(kCodeFill11), 19, 8},                       // gfx11
           {kCodeFill12, sizeof(kCodeFill12), 19, 8},                       // gfx12
       }}};

  auto compiled_shader_it = compiled_shaders.find(func_name);
  assert(compiled_shader_it != compiled_shaders.end() &&
         "Precompiled shader unavailable");

  ASICShader* asic_shader = NULL;

  switch (isa_->GetMajorVersion()) {
    case 7:
      asic_shader = &compiled_shader_it->second.compute_7;
      break;
    case 8:
      asic_shader = &compiled_shader_it->second.compute_8;
      break;
    case 9:
      if((isa_->GetMinorVersion() == 0) && (isa_->GetStepping() == 10)) {
        asic_shader = &compiled_shader_it->second.compute_90a;
      } else if(isa_->GetMinorVersion() == 4 || isa_->GetMinorVersion() == 5) {
        asic_shader = &compiled_shader_it->second.compute_942;
      } else {
        asic_shader = &compiled_shader_it->second.compute_9;
      }
      break;
    case 10:
      if(isa_->GetMinorVersion() == 1)
        asic_shader = &compiled_shader_it->second.compute_1010;
      else
        asic_shader = &compiled_shader_it->second.compute_10;
      break;
    case 11:
        asic_shader = &compiled_shader_it->second.compute_11;
      break;
    case 12:
        asic_shader = &compiled_shader_it->second.compute_12;
      break;
    default:
      assert(false && "Precompiled shader unavailable for target");
  }

  // Allocate a GPU-visible buffer for the shader.
  size_t header_size =
      (assemble_target == AssembleTarget::AQL ? sizeof(amd_kernel_code_t) : 0);
  code_buf_size = AlignUp(header_size + asic_shader->size, 0x1000);

  code_buf = system_allocator()(code_buf_size, 0x1000,
    core::MemoryRegion::AllocateExecutable | core::MemoryRegion::AllocateExecutableBlitKernelObject);
  assert(code_buf != NULL && "Code buffer allocation failed");

  memset(code_buf, 0, code_buf_size);

  // Populate optional code object header.
  if (assemble_target == AssembleTarget::AQL) {
    amd_kernel_code_t* header = reinterpret_cast<amd_kernel_code_t*>(code_buf);

    int gran_sgprs = std::max(0, (int(asic_shader->num_sgprs) - 1) / 8);
    int gran_vgprs = std::max(0, (int(asic_shader->num_vgprs) - 1) / 4);

    header->kernel_code_entry_byte_offset = sizeof(amd_kernel_code_t);
    AMD_HSA_BITS_SET(header->kernel_code_properties,
                     AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                     1);
    AMD_HSA_BITS_SET(header->compute_pgm_rsrc1,
                     AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT,
                     gran_sgprs);
    AMD_HSA_BITS_SET(header->compute_pgm_rsrc1,
                     AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT,
                     gran_vgprs);
    AMD_HSA_BITS_SET(header->compute_pgm_rsrc1,
                     AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64, 3);
    AMD_HSA_BITS_SET(header->compute_pgm_rsrc1,
                     AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE, 1);
    AMD_HSA_BITS_SET(header->compute_pgm_rsrc2,
                     AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT, 2);
    AMD_HSA_BITS_SET(header->compute_pgm_rsrc2,
                     AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, 1);

    // gfx90a, gfx942, gfx950
    if ((isa_->GetMajorVersion() == 9) &&
        (((isa_->GetMinorVersion() == 0) && (isa_->GetStepping() == 10)) ||
        (isa_->GetMinorVersion() == 4 || isa_->GetMinorVersion() == 5))) {
      // Program COMPUTE_PGM_RSRC3.ACCUM_OFFSET for 0 ACC VGPRs on gfx90a.
      // FIXME: Assemble code objects from source at build time
      int gran_accvgprs = ((gran_vgprs + 1) * 8) / 4 - 1;
      header->max_scratch_backing_memory_byte_size = uint64_t(gran_accvgprs) << 32;
    }
  }

  // Copy shader code into the GPU-visible buffer.
  memcpy((void*)(uintptr_t(code_buf) + header_size), asic_shader->code,
         asic_shader->size);
}

void GpuAgent::ReleaseShader(void* code_buf, size_t code_buf_size) const {
  system_deallocator()(code_buf);
}

void GpuAgent::InitRegionList() {
  const bool is_apu_node = (properties_.NumCPUCores > 0);

  std::vector<HsaMemoryProperties> mem_props(properties_.NumMemoryBanks);
  if (HSA_STATUS_SUCCESS == driver().GetMemoryProperties(node_id(), mem_props)) {
    for (uint32_t mem_idx = 0; mem_idx < properties_.NumMemoryBanks;
         ++mem_idx) {
      // Ignore the one(s) with unknown size.
      if (mem_props[mem_idx].SizeInBytes == 0) {
        continue;
      }

      switch (mem_props[mem_idx].HeapType) {
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
          if (!is_apu_node) {
            mem_props[mem_idx].VirtualBaseAddress = 0;
          }

          memory_bus_width_ = mem_props[mem_idx].Width;
          memory_max_frequency_ = mem_props[mem_idx].MemoryClockMax;
        case HSA_HEAPTYPE_GPU_LDS:
        case HSA_HEAPTYPE_GPU_SCRATCH: {
          MemoryRegion* region =
              new MemoryRegion(false, false, false, false, true, this, mem_props[mem_idx]);

          regions_.push_back(region);

          if (region->IsLocalMemory()) {
            // Extended Fine-Grain memory
            if (!(isa_->GetMajorVersion() == 12 && isa_->GetMinorVersion() == 0))
              regions_.push_back(
                  new MemoryRegion(false, false, false, true, true, this, mem_props[mem_idx]));

            // Expose VRAM as uncached/fine grain over PCIe (if enabled) or XGMI.
            bool user_visible = (properties_.HiveID != 0) ||
                core::Runtime::runtime_singleton_->flag().fine_grain_pcie();

            regions_.push_back(new MemoryRegion(true, false, false, false, user_visible, this,
                                                mem_props[mem_idx]));
          }
          break;
        }
        case HSA_HEAPTYPE_SYSTEM:
          if (is_apu_node) {
            memory_bus_width_ = mem_props[mem_idx].Width;
            memory_max_frequency_ = mem_props[mem_idx].MemoryClockMax;
          }
          break;
        case HSA_HEAPTYPE_MMIO_REMAP:
          // Remap offsets defined in kfd_ioctl.h
          HDP_flush_.HDP_MEM_FLUSH_CNTL = (uint32_t*)mem_props[mem_idx].VirtualBaseAddress;
          HDP_flush_.HDP_REG_FLUSH_CNTL = HDP_flush_.HDP_MEM_FLUSH_CNTL + 1;
          break;
        default:
          continue;
      }
    }
  }
}

void GpuAgent::InitScratchPool() {
  scratch_per_thread_ =
      core::Runtime::runtime_singleton_->flag().scratch_mem_size();
  if (scratch_per_thread_ == 0)
    scratch_per_thread_ = DEFAULT_SCRATCH_BYTES_PER_THREAD;

  // Scratch length is: waves/CU * threads/wave * queues * #CUs *
  // scratch/thread
  const uint32_t num_cu =
      properties_.NumFComputeCores / properties_.NumSIMDPerCU;
  queue_scratch_len_ = AlignUp(32 * 64 * num_cu * scratch_per_thread_, 65536);
  size_t max_scratch_len = queue_scratch_len_ * max_queues_;

#if defined(HSA_LARGE_MODEL) && defined(__linux__)
  // For 64-bit linux use max queues unless otherwise specified
  if ((max_scratch_len == 0) || (max_scratch_len > MaxScratchDevice())) {
    max_scratch_len = MaxScratchDevice();  // 4GB per XCC aperture max
  }
#endif

  void* scratch_base = nullptr;
  hsa_status_t err = driver().AllocateScratchMemory(node_id(), max_scratch_len, &scratch_base);
  assert(err == HSA_STATUS_SUCCESS && "AllocateScratchMemory failed");
  assert(IsMultipleOf(scratch_base, 0x1000) &&
         "Scratch base is not page aligned!");

  scratch_pool_. ~SmallHeap();
  if (HSA_STATUS_SUCCESS == err) {
    new (&scratch_pool_) SmallHeap(scratch_base, max_scratch_len);
  } else {
    new (&scratch_pool_) SmallHeap();
  }
}

void GpuAgent::InitAsyncScratchThresholds() {
  if (!AsyncScratchReclaimEnabled()) return;

  scratch_limit_async_threshold_ =
      core::Runtime::runtime_singleton_->flag().scratch_single_limit_async();

  if (!scratch_limit_async_threshold_) {
    // User did not set env var HSA_SCRATCH_SINGLE_LIMIT_ASYNC
    scratch_limit_async_threshold_ =
      core::Runtime::runtime_singleton_->flag().DEFAULT_SCRATCH_SINGLE_LIMIT_ASYNC_PER_XCC *
      (uint64_t)(properties().NumXcc);
  }
}

void GpuAgent::ReserveScratch()
{
  size_t reserved_sz = core::Runtime::runtime_singleton_->flag().scratch_single_limit();
  if (reserved_sz > MaxScratchDevice()) {
    fprintf(stdout, "User specified scratch limit exceeds device limits (requested:%lu max:%lu)!\n",
                    reserved_sz, MaxScratchDevice());
    reserved_sz = MaxScratchDevice();
  }

  size_t available;
  hsa_status_t err = driver().AvailableMemory(node_id(), &available);
  assert(err == HSA_STATUS_SUCCESS && "AvailableMemory failed");
  ScopedAcquire<KernelMutex> lock(&scratch_lock_);
  if (!scratch_cache_.reserved_bytes() && reserved_sz && available > 8 * reserved_sz) {
    HSAuint64 alt_va;
    void* reserved_base = scratch_pool_.alloc(reserved_sz);
    assert(reserved_base && "Could not allocate reserved memory");

    if (driver().MakeMemoryResident(reserved_base, reserved_sz, &alt_va) == HSA_STATUS_SUCCESS)
      scratch_cache_.reserve(reserved_sz, reserved_base);
    else
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES, "Reserve scratch memory failed.");
  }
}

void GpuAgent::InitCacheList() {
  // Get GPU cache information.
  // Similar to getting CPU cache but here we use FComputeIdLo.
  cache_props_.resize(properties_.NumCaches);
  if (HSA_STATUS_SUCCESS !=
      driver().GetCacheProperties(node_id(), properties_.FComputeIdLo, cache_props_)) {
    cache_props_.clear();
  } else {
    // Only store GPU D-cache.
    for (size_t cache_id = 0; cache_id < cache_props_.size(); ++cache_id) {
      const HsaCacheType type = cache_props_[cache_id].CacheType;
      if (type.ui32.HSACU != 1 || type.ui32.Instruction == 1) {
        cache_props_.erase(cache_props_.begin() + cache_id);
        --cache_id;
      }
    }
  }

  // Update cache objects
  caches_.clear();
  caches_.resize(cache_props_.size());
  char name[64];
  GetInfo(HSA_AGENT_INFO_NAME, name);
  std::string deviceName = name;
  for (size_t i = 0; i < caches_.size(); i++)
    caches_[i].reset(new core::Cache(deviceName + " L" + std::to_string(cache_props_[i].CacheLevel),
                                     cache_props_[i].CacheLevel, cache_props_[i].CacheSize));
}

void GpuAgent::InitLibDrm() {
  hsa_status_t status;

  HsaAMDGPUDeviceHandle device_handle;
  status = driver().GetDeviceHandle(node_id(), &device_handle);
  if (status != HSA_STATUS_SUCCESS)
    throw AMD::hsa_exception(status,
                             "Agent creation failed.\nlibdrm get device handle failed.\n");

  ldrm_dev_ = (amdgpu_device_handle)device_handle;
}

hsa_status_t GpuAgent::IterateRegion(
    hsa_status_t (*callback)(hsa_region_t region, void* data),
    void* data) const {
  return VisitRegion(true, callback, data);
}

hsa_status_t GpuAgent::IterateCache(hsa_status_t (*callback)(hsa_cache_t cache, void* data),
                                    void* data) const {
  AMD::callback_t<decltype(callback)> call(callback);
  for (size_t i = 0; i < caches_.size(); i++) {
    hsa_status_t stat = call(core::Cache::Convert(caches_[i].get()), data);
    if (stat != HSA_STATUS_SUCCESS) return stat;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::IterateSupportedIsas(
                    hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                                          void* data) const {
  AMD::callback_t<decltype(callback)> call(callback);
  for (const auto& isa : supported_isas()) {
    hsa_status_t stat = call(core::Isa::Handle(isa), data);
    if (stat != HSA_STATUS_SUCCESS) return stat;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::VisitRegion(bool include_peer,
                                   hsa_status_t (*callback)(hsa_region_t region,
                                                            void* data),
                                   void* data) const {
  if (include_peer) {
    // Only expose system, local, and LDS memory of the blit agent.
    const auto& gpu_ids = core::Runtime::runtime_singleton_->gpu_ids();
    for (auto& gpu_id : gpu_ids) {
      if (this->node_id() == gpu_id) {
        hsa_status_t stat = VisitRegion(regions_, callback, data);
        if (stat != HSA_STATUS_SUCCESS) {
          return stat;
        }
      }
    }

    // Also expose system regions accessible by this agent.
    hsa_status_t stat =
        VisitRegion(core::Runtime::runtime_singleton_->system_regions_fine(),
                    callback, data);
    if (stat != HSA_STATUS_SUCCESS) {
      return stat;
    }

    return VisitRegion(
        core::Runtime::runtime_singleton_->system_regions_coarse(), callback,
        data);
  }

  // Only expose system, local, and LDS memory of this agent.
  return VisitRegion(regions_, callback, data);
}

hsa_status_t GpuAgent::VisitRegion(
    const std::vector<const core::MemoryRegion*>& regions,
    hsa_status_t (*callback)(hsa_region_t region, void* data),
    void* data) const {
  AMD::callback_t<decltype(callback)> call(callback);
  for (const core::MemoryRegion* region : regions) {
    if (!region->user_visible()) continue;

    const AMD::MemoryRegion* amd_region =
        reinterpret_cast<const AMD::MemoryRegion*>(region);

    // Only expose system, local, and LDS memory.
    if (amd_region->IsSystem() || amd_region->IsLocalMemory() ||
        amd_region->IsLDS()) {
      hsa_region_t region_handle = core::MemoryRegion::Convert(region);
      hsa_status_t status = call(region_handle, data);
      if (status != HSA_STATUS_SUCCESS) {
        return status;
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

core::Queue* GpuAgent::CreateInterceptibleQueue(void (*callback)(hsa_status_t status,
                                                                 hsa_queue_t* source, void* data),
                                                void* data, const uint32_t in_size) {
  // Disabled intercept of internal queues pending tools updates.
  core::Queue* queue = nullptr;
  uint32_t size = std::max(in_size, minAqlSize_);
  size = std::min(size, maxAqlSize_);

  QueueCreate(size, HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_CREATE_SYSTEM_MEM, callback, data, 0, 0,
              &queue);
  if (queue != nullptr)
    core::Runtime::runtime_singleton_->InternalQueueCreateNotify(core::Queue::Convert(queue),
                                                                 this->public_handle());
  return queue;
}

core::Blit* GpuAgent::CreateBlitSdma(bool use_xgmi, int rec_eng) {
  AMD::BlitSdmaBase* sdma;
  size_t copy_size_override = 0;
  const size_t copy_size_overrides[2] = {0x3fffff, 0x3fffffff};

  switch (isa_->GetMajorVersion()) {
    case 9:
      sdma = new BlitSdmaV4();
      copy_size_override = (isa_->GetMinorVersion() == 0 && isa_->GetStepping() == 10) ?
                            copy_size_overrides[1] : copy_size_overrides[0];
      break;
    case 10:
      sdma = new BlitSdmaV5();
      copy_size_override = isa_->GetMinorVersion() < 3 ? copy_size_overrides[0] :
                                                         copy_size_overrides[1];
      break;
    case 11:
    case 12:
      sdma = new BlitSdmaV5();
      copy_size_override = copy_size_overrides[1];
      break;
    default:
      assert(false && "Unexpected device major version.");
      return nullptr;
  }

  Flag::SDMA_OVERRIDE copy_size_override_setting =
    core::Runtime::runtime_singleton_->flag().enable_sdma_copy_size_override();
  if (copy_size_override_setting == Flag::SDMA_DISABLE) copy_size_override = 0;

  rec_eng = uses_rec_sdma_eng_id_mask_ || !use_xgmi ? rec_eng : -1;

  if (sdma->Initialize(*this, use_xgmi, copy_size_override, rec_eng) != HSA_STATUS_SUCCESS) {
    sdma->Destroy(*this);
    delete sdma;
    sdma = nullptr;
  }

  return sdma;
}

core::Blit* GpuAgent::CreateBlitKernel(core::Queue* queue) {
  AMD::BlitKernel* kernl = new AMD::BlitKernel(queue);

  if (kernl->Initialize(*this) != HSA_STATUS_SUCCESS) {
    kernl->Destroy(*this);
    delete kernl;
    kernl = NULL;
  }

  return kernl;
}

void GpuAgent::InitDma() {
  // Setup lazy init pointers on queues and blits.
  auto queue_lambda = [this](HSA_QUEUE_PRIORITY priority = HSA_QUEUE_PRIORITY_NORMAL) {
    auto queue = CreateInterceptibleQueue();
    if (queue == nullptr)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                               "Internal queue creation failed.");

    if (priority != HSA_QUEUE_PRIORITY_NORMAL)
      if (queue->SetPriority(priority) != HSA_STATUS_SUCCESS)
        throw AMD::hsa_exception(HSA_STATUS_ERROR,
                                "Failed to increase queue priority for PC Sampling");
    return queue;
  };

  // Dedicated compute queue for host-to-device blits.
  queues_[QueueBlitOnly].reset(queue_lambda);
  // Share utility queue with device-to-host blits.
  queues_[QueueUtility].reset(queue_lambda);

  // Dedicated compute queue for PC Sampling CP-DMA commands. We need a dedicated queue that runs at
  // highest priority because we do not want the CP-DMA commands to be delayed/blocked due to
  // other dispatches/barriers that could be in the other AQL queues.
  queues_[QueuePCSampling].reset([queue_lambda]() { return queue_lambda(HSA_QUEUE_PRIORITY_MAXIMUM); });

  // Decide which engine to use for blits.
  auto blit_lambda = [this](bool use_xgmi, lazy_ptr<core::Queue>& queue, bool isHostToDev, uint32_t rec_eng) {
    Flag::SDMA_OVERRIDE sdma_override = core::Runtime::runtime_singleton_->flag().enable_sdma();

    // User SDMA queues are unstable on gfx8 and unsupported on gfx1013.
    bool use_sdma =
        ((isa_->GetMajorVersion() != 8) && (isa_->GetVersion() != std::make_tuple(10, 1, 3)));
    if (sdma_override != Flag::SDMA_DEFAULT) use_sdma = (sdma_override == Flag::SDMA_ENABLE);

    if (use_sdma && (HSA_PROFILE_BASE == profile_)) {
      // On gfx90a ensure that HostToDevice queue is created first and so is placed on SDMA0.
      if ((!use_xgmi) && (!isHostToDev) && (isa_->GetMajorVersion() == 9) &&
          (isa_->GetMinorVersion() == 0) && (isa_->GetStepping() == 10)) {
        GetBlitObject(BlitHostToDev);
        *blits_[BlitHostToDev];
      }

      // gfx94x is more efficient with reverse order of SDMA0/1 for host<->device copies
      if (!use_xgmi && isa_->GetMajorVersion() == 9 && isa_->GetMinorVersion() >= 4)
        rec_eng = (rec_eng + 1) % properties_.NumSdmaEngines;

      // Check support for targeted SDMA engines
      auto kfd_version = core::Runtime::runtime_singleton_->KfdVersion().version;
      if (!(kfd_version.KernelInterfaceMajorVersion > 1 ||
            (kfd_version.KernelInterfaceMajorVersion == 1 &&
             kfd_version.KernelInterfaceMinorVersion >= 17)))
        rec_eng = -1;

      // Observing strange behavior when fixing host<->device engines
      // on GFX9 devices older than GFX90a, so bypass engine fix.
      if (!use_xgmi && isa_->GetMajorVersion() == 9 && isa_->GetMinorVersion() == 0
          && isa_->GetStepping() < 10)
        rec_eng = -1;

      // devices without dedicated xGMI SDMA engines should not target specific
      // SDMA engines for queue creation as resources are limited
      if (!properties_.NumSdmaXgmiEngines)
        rec_eng = -1;

      auto ret = CreateBlitSdma(use_xgmi, rec_eng);
      if (ret != nullptr) return ret;
    }

    // pending_copy_stat_check_ref_ will prevent unnecessary compute queue creation
    // since there is no graceful way to handle lazy loading when the caller needs to know
    // the status of available SDMA HW resources without a fallback.
    // Call to isSDMA should be used as a proxy error check if !blit_copy_fallback.
    auto ret = pending_copy_stat_check_ref_ ? new AMD::BlitKernel(NULL) :
                                              CreateBlitKernel((*queue).get());
    if (ret == nullptr)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES, "Blit creation failed.");
    return ret;
  };

  // Determine and instantiate the number of blit objects to
  // engage. The total number is sum of three plus number of
  // sdma-xgmi engines
  uint32_t blit_cnt_ = DefaultBlitCount + properties_.NumSdmaXgmiEngines;
  blits_.resize(blit_cnt_);

  // Initialize blit objects used for D2D, H2D, D2H, and
  // P2P copy operations.
  // -- Blit at index BlitDevToDev(0) deals with copies within
  //    local framebuffer and always engages a Blit Kernel
  // -- Blit at index BlitHostToDev(1) deals with copies from
  //    Host to Device (H2D) and could engage either a Blit
  //    Kernel or sDMA
  // -- Blit at index BlitDevToHost(2) deals with copies from
  //    Device to Host (D2H) and Peer to Peer (P2P) over PCIe.
  //    It could engage either a Blit Kernel or sDMA
  // -- Blit at index DefaultBlitCount(3) and beyond deal
  //    exclusively P2P over xGMI links
  blits_[BlitDevToDev].reset([this]() {
    auto ret = CreateBlitKernel((*queues_[QueueUtility]).get());
    if (ret == nullptr)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES, "Blit creation failed.");
    return ret;
  });
  blits_[BlitHostToDev].reset(
      [blit_lambda, this]() { return blit_lambda(false, queues_[QueueBlitOnly], true, 0); });
  blits_[BlitDevToHost].reset(
      [blit_lambda, this]() { return blit_lambda(false, queues_[QueueUtility], false, 1); });

  // XGMI engines.
  for (uint32_t idx = DefaultBlitCount; idx < blit_cnt_; idx++) {
    const int eng = idx - 1;
    blits_[idx].reset(
        [blit_lambda, this, eng]() { return blit_lambda(true, queues_[QueueUtility], false, eng); });
  }

  // GWS queues.
  InitGWS();
}

void GpuAgent::InitGWS() {
  gws_queue_.queue_.reset([this]() {
    if (properties_.NumGws == 0) return (core::Queue*)nullptr;
    const uint32_t defaultGWSQueueSize = 0x4000; // 16KB
    std::unique_ptr<core::Queue> queue(CreateInterceptibleQueue(defaultGWSQueueSize));
    if (queue == nullptr)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                               "Internal queue creation failed.");

    auto err = static_cast<AqlQueue*>(queue.get())->EnableGWS(1);
    if (err != HSA_STATUS_SUCCESS) throw AMD::hsa_exception(err, "GWS allocation failed.");

    gws_queue_.ref_ct_ = 0;
    return queue.release();
  });
}

void GpuAgent::GWSRelease() {
  ScopedAcquire<KernelMutex> lock(&gws_queue_.lock_);
  gws_queue_.ref_ct_--;
  if (gws_queue_.ref_ct_ != 0) return;
  InitGWS();
}

void GpuAgent::PreloadBlits() {
  for (auto& blit : blits_) {
    blit.touch();
  }
}

void GpuAgent::ReleaseResources() {
  if (this->Enabled()) {
    this->Disable();
    for (auto& blit : blits_) {
      if (!blit.empty()) {
        hsa_status_t status = blit->Destroy(*this);
        assert(status == HSA_STATUS_SUCCESS);
      }
    }

    if (ape1_base_ != 0) {
      _aligned_free(reinterpret_cast<void*>(ape1_base_));
    }

    scratch_cache_.trim(true);
    scratch_cache_.free_reserve();

    if (scratch_pool_.base() != NULL) {
      driver().FreeMemory(scratch_pool_.base(), scratch_pool_.size());
    }

    for (int i = 0; i < QueueCount; i++)
      queues_[i].reset();

    system_deallocator()(doorbell_queue_map_);

    if (trap_code_buf_ != NULL)
      system_deallocator()(trap_code_buf_);
  }
}

hsa_status_t GpuAgent::PostToolsInit() {
  // Defer memory allocation until agents have been discovered.
  InitAllocators();
  InitScratchPool();
  BindTrapHandler();
  InitDma();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::DmaCopy(void* dst, const void* src, size_t size) {
  return blits_[BlitDevToDev]->SubmitLinearCopyCommand(dst, src, size);
}

void GpuAgent::SetCopyRequestRefCount(bool set) {
  ScopedAcquire<KernelMutex> lock(&blit_lock_);
  while (pending_copy_stat_check_ref_) {
    blit_lock_.Release();
    os::YieldThread();
    blit_lock_.Acquire();
  }
  if (!set && pending_copy_req_ref_) pending_copy_req_ref_--;
  else pending_copy_req_ref_++;
}

void GpuAgent::SetCopyStatusCheckRefCount(bool set) {
  ScopedAcquire<KernelMutex> lock(&blit_lock_);
  while (pending_copy_req_ref_) {
    blit_lock_.Release();
    os::YieldThread();
    blit_lock_.Acquire();
  }
  if (!set && pending_copy_stat_check_ref_) pending_copy_stat_check_ref_--;
  else pending_copy_stat_check_ref_++;
}

// Assign direct peer gang factor to GPU
void GpuAgent::RegisterGangPeer(core::Agent& peer, unsigned int max_bandwidth_factor) {
  gang_peers_info_[peer.public_handle().handle] = max_bandwidth_factor;
}

// Assign direct peer recommended SDMA engine IDs to GPU
void GpuAgent::RegisterRecSdmaEngIdMaskPeer(core::Agent& peer, uint32_t rec_sdma_eng_id_mask) {
  auto kfd_version = core::Runtime::runtime_singleton_->KfdVersion().version;
  bool rec_eng_enabled = core::Runtime::runtime_singleton_->flag().enable_sdma_recommended_eng() !=
                         Flag::SDMA_DISABLE;

  // Assume all recommended masks with single recommended engine (IsPowerOfTwo)
  // will only support targeting that engine and will not gang.
  // Also assume support is uniform for every device in the system.
  uses_rec_sdma_eng_id_mask_ = (kfd_version.KernelInterfaceMajorVersion > 1 ||
                                 (kfd_version.KernelInterfaceMajorVersion == 1 &&
                                  kfd_version.KernelInterfaceMinorVersion >= 17)) &&
                               isa_->GetMajorVersion() == 9 && isa_->GetMinorVersion() >= 4 &&
                               IsPowerOfTwo(rec_sdma_eng_id_mask) && rec_eng_enabled;

  rec_sdma_eng_id_peers_info_[peer.public_handle().handle] = uses_rec_sdma_eng_id_mask_ ?
                                                             rec_sdma_eng_id_mask : 0;
}

// Destroy gang signal
static bool GangCopyCompleteHandler(hsa_signal_value_t, void *arg ) {
  core::Signal *gang_signal = reinterpret_cast<core::Signal*>(arg);
  if (gang_signal->IsValid()) {
    gang_signal->DestroySignal();
    if (!gang_signal->IsValid()) {
      return false;
    }
  }
  return true;
}

hsa_status_t GpuAgent::DmaCopy(void* dst, core::Agent& dst_agent,
                               const void* src, core::Agent& src_agent,
                               size_t size,
                               std::vector<core::Signal*>& dep_signals,
                               core::Signal& out_signal) {
  // Recommended SDMA engine copies only have gang factor 1
  uint32_t rec_sdma_eng = ffs(rec_sdma_eng_id_peers_info_[dst_agent.public_handle().handle]);

  if (rec_sdma_eng)
    return DmaCopyOnEngine(dst, dst_agent, src, src_agent, size,
                           dep_signals, out_signal, rec_sdma_eng, false);

  if (profiling_enabled()) {
    // Track the agent so we could translate the resulting timestamp to system
    // domain correctly.
    out_signal.async_copy_agent(core::Agent::Convert(this->public_handle()));
  }

  // Calculate the number of gang items
  unsigned int gang_factor = 1;
  if (core::Runtime::runtime_singleton_->flag().enable_sdma_gang() != Flag::SDMA_DISABLE &&
      size >= 4096 && dst_agent.device_type() == core::Agent::kAmdGpuDevice)
    gang_factor = gang_peers_info_[dst_agent.public_handle().handle];
  // Use non-D2D (auxillary) SDMA engines in the event of xGMI D2D support
  // when xGMI SDMA context is not available.
  bool has_aux_gang = gang_factor > 1 &&
                      gang_factor >= properties_.NumSdmaEngines &&
                      !!!properties_.NumSdmaXgmiEngines;
  if (gang_factor > 1) {
    gang_factor = has_aux_gang ?
                      std::min(gang_factor, properties_.NumSdmaEngines) :
                      std::min(gang_factor, properties_.NumSdmaXgmiEngines);
  }

  ScopedAcquire<KernelMutex> lock(&sdma_gang_lock_);
  // Manage internal gang signals
  std::vector<core::Signal*> gang_signals;
  if (gang_factor > 1) {
    for (int i = 0; i < gang_factor - 1; i++) {
      core::Signal *gang_signal;

      // Initial value is 2 where 1 is for gang-leader to ack and
      // 1 for non-leader gang item to decrement
      gang_signal = new core::DefaultSignal(2);

      // Fall back to non-gang copy
      if (!gang_signal->IsValid()) {
        for (int j = 0; j < gang_signals.size(); j++) gang_signals[j]->DestroySignal();
        gang_factor = 1;
        break;
      }

      core::Runtime::runtime_singleton_->SetAsyncSignalHandler(
                                         core::Signal::Convert(gang_signal),
                                         HSA_SIGNAL_CONDITION_EQ, 0, GangCopyCompleteHandler,
                                         reinterpret_cast<void*>(gang_signal));
      gang_signals.push_back(gang_signal);
    }
  }

  // Bind the Blit object that will drive this copy operation
  size_t offset = 0, remainder_size = size;
  int gang_sig_count = 0;
  for (int i = 0; i < gang_factor; i++) {
    // Set leader and gang status to blit
    SetCopyRequestRefCount(true);
    MAKE_SCOPE_GUARD([&]() { SetCopyRequestRefCount(false); });
    lazy_ptr<core::Blit>& blit = gang_factor > 1 ?
                                 (has_aux_gang ? blits_[i + 1] : blits_[i + DefaultBlitCount]) :
                                 GetBlitObject(dst_agent, src_agent, size);
    blit->GangLeader(gang_factor > 1 && !i);

    hsa_status_t stat;
    size_t chunk = std::min(remainder_size, (size + gang_factor - 1)/gang_factor);
    if (!blit->GangLeader() && !gang_signals.empty()) {
      stat = blit->SubmitLinearCopyCommand(reinterpret_cast<uint8_t*>(dst) + offset,
                                           reinterpret_cast<const uint8_t*>(src) + offset,
                                           chunk, dep_signals,
                                           *gang_signals[gang_sig_count], gang_signals);
      gang_sig_count++;
    } else {
      stat = blit->SubmitLinearCopyCommand(reinterpret_cast<uint8_t*>(dst) + offset,
                                           reinterpret_cast<const uint8_t*>(src) + offset,
                                           chunk, dep_signals,
                                           out_signal, gang_signals);
    }

    if (stat)
      return stat;

    offset += chunk;
    remainder_size -= chunk;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::DmaCopyOnEngine(void* dst, core::Agent& dst_agent,
                               const void* src, core::Agent& src_agent,
                               size_t size,
                               std::vector<core::Signal*>& dep_signals,
                               core::Signal& out_signal,
                               int engine_offset,
                               bool force_copy_on_sdma) {
  // At this point it is guaranteed that one of
  // the two devices is a GPU, potentially both
  assert(((src_agent.device_type() == core::Agent::kAmdGpuDevice) ||
          (dst_agent.device_type() == core::Agent::kAmdGpuDevice)) &&
         ("Both devices are CPU agents which is not expected"));

  if (engine_offset > properties_.NumSdmaEngines + properties_.NumSdmaXgmiEngines) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // check if dst and src are the same gpu or over xGMI.
  bool is_same_gpu = (src_agent.public_handle().handle == dst_agent.public_handle().handle) &&
                     (dst_agent.public_handle().handle == public_handle_.handle);

  bool is_p2p = !is_same_gpu && src_agent.device_type() == core::Agent::kAmdGpuDevice &&
                                dst_agent.device_type() == core::Agent::kAmdGpuDevice;

  if ((is_p2p &&
      core::Runtime::runtime_singleton_->flag().enable_peer_sdma() == Flag::SDMA_DISABLE) ||
      core::Runtime::runtime_singleton_->flag().enable_sdma() == Flag::SDMA_DISABLE) {
    // Note  that VDI/HIP will call DmaCopy instead of DmaCopyOnEngine for P2P copies, but
    // we still want to handle force Blit Kernels in this function in case other libraries
    // decide to use DmaCopyOnEngine for P2P copies

    engine_offset = BlitDevToDev;
  } else {
    bool is_xgmi = is_p2p && dst_agent.HiveId() && src_agent.HiveId() == dst_agent.HiveId() &&
                         properties_.NumSdmaXgmiEngines;

    // Due to a RAS issue, GFX90a can only support H2D copies on SDMA0
    bool is_h2d_blit = (src_agent.device_type() == core::Agent::kAmdCpuDevice &&
      dst_agent.device_type() == core::Agent::kAmdGpuDevice);
    bool limit_h2d_blit = isa_->GetVersion() == core::Isa::Version(9, 0, 10);

    // Ensure engine selection is within proper range based on transfer type
    if ((is_xgmi && !rec_sdma_eng_override_ && engine_offset <= properties_.NumSdmaEngines) ||
        (!is_xgmi && engine_offset > (properties_.NumSdmaEngines +
                                      properties_.NumSdmaXgmiEngines)) ||
          (!is_h2d_blit && !is_same_gpu && limit_h2d_blit &&
            engine_offset == BlitHostToDev)) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    engine_offset = is_same_gpu && !force_copy_on_sdma ? BlitDevToDev : engine_offset;
  }

  SetCopyRequestRefCount(true);
  MAKE_SCOPE_GUARD([&]() { SetCopyRequestRefCount(false); });
  lazy_ptr<core::Blit>& blit = GetBlitObject(engine_offset);

  if (profiling_enabled()) {
    // Track the agent so we could translate the resulting timestamp to system
    // domain correctly.
    out_signal.async_copy_agent(core::Agent::Convert(this->public_handle()));
  }

  std::vector<core::Signal*> gang_signals(0);

  hsa_status_t stat = blit->SubmitLinearCopyCommand(dst, src, size, dep_signals, out_signal,
                                                    gang_signals);

  return stat;
}

bool GpuAgent::DmaEngineIsFree(uint32_t engine_offset) {
  SetCopyStatusCheckRefCount(true);
  MAKE_SCOPE_GUARD([&]() { SetCopyStatusCheckRefCount(false); });
  bool is_free = !!!(sdma_blit_used_mask_ & (1 << engine_offset)) ||
                    (blits_[engine_offset]->isSDMA() &&
                     !!!blits_[engine_offset]->PendingBytes());
  return is_free;
}

hsa_status_t GpuAgent::DmaCopyStatus(core::Agent& dst_agent, core::Agent& src_agent,
                                     uint32_t *engine_ids_mask) {
  assert(((src_agent.device_type() == core::Agent::kAmdGpuDevice) ||
          (dst_agent.device_type() == core::Agent::kAmdGpuDevice)) &&
         ("Both devices are CPU agents which is not expected"));

  *engine_ids_mask = 0;
  if (src_agent.device_type() == core::Agent::kAmdGpuDevice &&
                   dst_agent.device_type() == core::Agent::kAmdGpuDevice &&
                     dst_agent.HiveId() && src_agent.HiveId() == dst_agent.HiveId() &&
                       properties_.NumSdmaXgmiEngines) {
    //Find a free xGMI SDMA engine
    if (rec_sdma_eng_override_) {
      for (int i = 0; i < (properties_.NumSdmaEngines + properties_.NumSdmaXgmiEngines); i++) {
        if (DmaEngineIsFree(BlitHostToDev + i)) {
          *engine_ids_mask |= (HSA_AMD_SDMA_ENGINE_0 << i);
        }
      }
    } else {
      for (int i = 0; i < properties_.NumSdmaXgmiEngines; i++) {
        if (DmaEngineIsFree(DefaultBlitCount + i)) {
          *engine_ids_mask |= (HSA_AMD_SDMA_ENGINE_2 << i);
        }
      }
    }
  } else {
    bool is_h2d_blit = (src_agent.device_type() == core::Agent::kAmdCpuDevice &&
      dst_agent.device_type() == core::Agent::kAmdGpuDevice);
    // Due to a RAS issue, GFX90a can only support H2D copies on SDMA0
    bool limit_h2d_blit = isa_->GetVersion() == core::Isa::Version(9, 0, 10);

    // Check if H2D is free
    if (DmaEngineIsFree(BlitHostToDev)) {
      if (is_h2d_blit || !limit_h2d_blit) {
        *engine_ids_mask |= HSA_AMD_SDMA_ENGINE_0;
      }
    }

    // Check is D2H is free
    if (DmaEngineIsFree(BlitDevToHost)) {
      *engine_ids_mask |= properties_.NumSdmaEngines > 1 ?
                          HSA_AMD_SDMA_ENGINE_1 :
                          HSA_AMD_SDMA_ENGINE_0;
    }
    // Find a free xGMI SDMA engine for H2D/D2H though it may be lower bandwidth
    for (int i = 0; i < properties_.NumSdmaXgmiEngines; i++) {
      if (DmaEngineIsFree(DefaultBlitCount + i)) {
         *engine_ids_mask |= (HSA_AMD_SDMA_ENGINE_2 << i);
      }
    }
  }

  return !!(*engine_ids_mask) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

hsa_status_t GpuAgent::DmaPreferredEngine(core::Agent& dst_agent, core::Agent& src_agent,
                                          uint32_t *recommended_ids_mask) {
  assert(((src_agent.device_type() == core::Agent::kAmdGpuDevice) ||
          (dst_agent.device_type() == core::Agent::kAmdGpuDevice)) &&
         ("Both devices are CPU agents which is not expected"));

  *recommended_ids_mask = rec_sdma_eng_id_peers_info_[dst_agent.public_handle().handle];

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::DmaCopyRect(const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset,
                                   const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset,
                                   const hsa_dim3_t* range, hsa_amd_copy_direction_t dir,
                                   std::vector<core::Signal*>& dep_signals,
                                   core::Signal& out_signal) {
  if (isa_->GetMajorVersion() < 9) return HSA_STATUS_ERROR_INVALID_AGENT;

  SetCopyRequestRefCount(true);
  MAKE_SCOPE_GUARD([&]() { SetCopyRequestRefCount(false); });
  lazy_ptr<core::Blit>& blit = GetBlitObject((dir == hsaHostToDevice) ? BlitHostToDev :
                                                                        BlitDevToHost);

  if (!blit->isSDMA()) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (profiling_enabled()) {
    // Track the agent so we could translate the resulting timestamp to system
    // domain correctly.
    out_signal.async_copy_agent(core::Agent::Convert(this->public_handle()));
  }

  BlitSdmaBase* sdmaBlit = static_cast<BlitSdmaBase*>((*blit).get());
  hsa_status_t stat = sdmaBlit->SubmitCopyRectCommand(dst, dst_offset, src, src_offset, range,
                                                      dep_signals, out_signal);

  return stat;
}

hsa_status_t GpuAgent::DmaFill(void* ptr, uint32_t value, size_t count) {
  return blits_[BlitDevToDev]->SubmitLinearFillCommand(ptr, value, count);
}

hsa_status_t GpuAgent::EnableDmaProfiling(bool enable) {
  for (auto& blit : blits_) {
    if (!blit.empty()) {
      const hsa_status_t stat = blit->EnableProfiling(enable);
      if (stat != HSA_STATUS_SUCCESS) {
        return stat;
      }
    }
  }

  if (enable) CheckClockTicks();

  return HSA_STATUS_SUCCESS;
}

void GpuAgent::GetInfoMemoryProperties(uint8_t value[8]) const {
  auto setFlag = [&](uint32_t bit) {
    assert(bit < 8 * 8 && "Flag value exceeds input parameter size");

    uint index = bit / 8;
    uint subBit = bit % 8;
    ((uint8_t*)value)[index] |= 1 << subBit;
  };

  // Fill the HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU flag
  if (properties_.Integrated)
      setFlag(HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU);
}

hsa_status_t GpuAgent::GetInfo(hsa_agent_info_t attribute, void* value) const {
  // agent, and vendor name size limit
  const size_t attribute_u = static_cast<size_t>(attribute);
  // agent, and vendor name length limit excluding terminating nul character.
  constexpr size_t hsa_name_size = 63;

  const bool isa_has_image_support =
      (isa_->GetMajorVersion() == 9 &&
      (isa_->GetMinorVersion() == 4 || isa_->GetMinorVersion() == 5)) ? false : true;

  switch (attribute_u) {
    case HSA_AGENT_INFO_NAME: {
      std::string name = isa_->GetProcessorName();
      assert(name.size() <= hsa_name_size);
      std::memset(value, 0, hsa_name_size);
      char* temp = reinterpret_cast<char*>(value);
      std::strcpy(temp, name.c_str());
      break;
    }
    case HSA_AGENT_INFO_VENDOR_NAME:
      std::memset(value, 0, hsa_name_size);
      std::memcpy(value, "AMD", sizeof("AMD"));
      break;
    case HSA_AGENT_INFO_FEATURE:
      *((hsa_agent_feature_t*)value) = HSA_AGENT_FEATURE_KERNEL_DISPATCH;
      break;
    case HSA_AGENT_INFO_MACHINE_MODEL:
#if defined(HSA_LARGE_MODEL)
      *((hsa_machine_model_t*)value) = HSA_MACHINE_MODEL_LARGE;
#else
      *((hsa_machine_model_t*)value) = HSA_MACHINE_MODEL_SMALL;
#endif
      break;
    case HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:
    case HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:
      *((hsa_default_float_rounding_mode_t*)value) =
          HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;
      break;
    case HSA_AGENT_INFO_FAST_F16_OPERATION:
      if (isa_->GetMajorVersion() >= 8) {
        *((bool*)value) = true;
      } else {
        *((bool*)value) = false;
      }
      break;
    case HSA_AGENT_INFO_PROFILE:
      *((hsa_profile_t*)value) = profile_;
      break;
    case HSA_AGENT_INFO_WAVEFRONT_SIZE:
      *((uint32_t*)value) = properties_.WaveFrontSize;
      break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_DIM: {
      // TODO: must be per-device
      const uint16_t group_size[3] = {1024, 1024, 1024};
      std::memcpy(value, group_size, sizeof(group_size));
    } break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_SIZE:
      // TODO: must be per-device
      *((uint32_t*)value) = 1024;
      break;
    case HSA_AGENT_INFO_GRID_MAX_DIM: {
      const hsa_dim3_t grid_size = {INT32_MAX, UINT16_MAX, UINT16_MAX};
      std::memcpy(value, &grid_size, sizeof(hsa_dim3_t));
    } break;
    case HSA_AGENT_INFO_GRID_MAX_SIZE:
      *((uint32_t*)value) = UINT32_MAX;
      break;
    case HSA_AGENT_INFO_FBARRIER_MAX_SIZE:
      // TODO: to confirm
      *((uint32_t*)value) = 32;
      break;
    case HSA_AGENT_INFO_QUEUES_MAX:
      *((uint32_t*)value) = max_queues_;
      break;
    case HSA_AGENT_INFO_QUEUE_MIN_SIZE:
      *((uint32_t*)value) = minAqlSize_;
      break;
    case HSA_AGENT_INFO_QUEUE_MAX_SIZE:
      *((uint32_t*)value) = maxAqlSize_;
      break;
    case HSA_AGENT_INFO_QUEUE_TYPE:
      *((hsa_queue_type32_t*)value) = HSA_QUEUE_TYPE_MULTI;
      break;
    case HSA_AGENT_INFO_NODE:
      // TODO: associate with OS NUMA support (numactl / GetNumaProcessorNode).
      *((uint32_t*)value) = node_id();
      break;
    case HSA_AGENT_INFO_DEVICE:
      *((hsa_device_type_t*)value) = HSA_DEVICE_TYPE_GPU;
      break;
    case HSA_AGENT_INFO_CACHE_SIZE: {
      std::memset(value, 0, sizeof(uint32_t) * 4);
      assert(cache_props_.size() > 0 && "GPU cache info missing.");
      const size_t num_cache = cache_props_.size();
      for (size_t i = 0; i < num_cache; ++i) {
        const uint32_t line_level = cache_props_[i].CacheLevel;
          /*
           * L1 Cache is per CU.
           * For L2 Cache and above, we report total for the partition so we sum
           * all the node entries.
           */
        if (line_level >= 2)
          reinterpret_cast<uint32_t*>(value)[line_level - 1] += cache_props_[i].CacheSize * 1024;
        else if (reinterpret_cast<uint32_t*>(value)[line_level - 1] == 0)
          reinterpret_cast<uint32_t*>(value)[line_level - 1] = cache_props_[i].CacheSize * 1024;
      }
    } break;
    case HSA_AGENT_INFO_ISA:
      *((hsa_isa_t*)value) = core::Isa::Handle(isa_);
      break;
    case HSA_AGENT_INFO_EXTENSIONS: {
      memset(value, 0, sizeof(uint8_t) * 128);

      auto setFlag = [&](uint32_t bit) {
        assert(bit < 128 * 8 && "Extension value exceeds extension bitmask");
        uint index = bit / 8;
        uint subBit = bit % 8;
        ((uint8_t*)value)[index] |= 1 << subBit;
      };

      if (core::hsa_internal_api_table().finalizer_api.hsa_ext_program_finalize_fn != NULL) {
        setFlag(HSA_EXTENSION_FINALIZER);
      }

      if (core::hsa_internal_api_table().image_api.hsa_ext_image_create_fn != NULL) {
        setFlag(HSA_EXTENSION_IMAGES);
      }

      if (core::hsa_internal_api_table().pcs_api.hsa_ven_amd_pcs_iterate_configuration_fn != NULL) {
        setFlag(HSA_EXTENSION_AMD_PC_SAMPLING);
      }

      if (os::LibHandle lib = os::LoadLib(kAqlProfileLib)) {
        os::CloseLib(lib);
        setFlag(HSA_EXTENSION_AMD_AQLPROFILE);
      }

      setFlag(HSA_EXTENSION_AMD_PROFILER);

      break;
    }
    case HSA_AGENT_INFO_VERSION_MAJOR:
      *((uint16_t*)value) = 1;
      break;
    case HSA_AGENT_INFO_VERSION_MINOR:
      *((uint16_t*)value) = 1;
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS:
      if (!isa_has_image_support)
        *((uint32_t*)value) = 0;
      else
        return hsa_amd_image_get_info_max_dim(public_handle(), attribute, value);
      break;
    case HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES:
      // TODO: hardcode based on OCL constants.
      *((uint32_t*)value) = isa_has_image_support ? 128 : 0;
      break;
    case HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES:
      *((uint32_t*)value) = isa_has_image_support ? 64 : 0;
      break;
    case HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS:
      *((uint32_t*)value) = isa_has_image_support ? 16 : 0;
      break;
    case HSA_AMD_AGENT_INFO_CHIP_ID:
      *((uint32_t*)value) = properties_.DeviceId;
      break;
    case HSA_AMD_AGENT_INFO_CACHELINE_SIZE:
      for (auto& cache : cache_props_) {
        if ((cache.CacheLevel == 2) && (cache.CacheLineSize != 0)) {
          *((uint32_t*)value) = cache.CacheLineSize;
          return HSA_STATUS_SUCCESS;
        }
      }
      // Fallback for when KFD is returning zero.
      *((uint32_t*)value) = 64;
      break;
    case HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT:
      *((uint32_t*)value) =
          (properties_.NumFComputeCores / properties_.NumSIMDPerCU);
      break;
    case HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY:
      *((uint32_t*)value) = properties_.MaxEngineClockMhzFCompute;
      break;
    case HSA_AMD_AGENT_INFO_DRIVER_NODE_ID:
      *((uint32_t*)value) = node_id();
      break;
    case HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS:
      *((uint32_t*)value) = static_cast<uint32_t>(
          1 << properties_.Capability.ui32.WatchPointsTotalBits);
      break;
    case HSA_AMD_AGENT_INFO_BDFID:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.LocationId);
      break;
    case HSA_AMD_AGENT_INFO_MEMORY_WIDTH:
      *((uint32_t*)value) = memory_bus_width_;
      break;
    case HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY:
      *((uint32_t*)value) = memory_max_frequency_;
      break;

    // The code copies HsaNodeProperties.MarketingName a Unicode string
    // which is encoded in UTF-16 as a 7-bit ASCII string
    case HSA_AMD_AGENT_INFO_PRODUCT_NAME: {
      std::memset(value, 0, HSA_PUBLIC_NAME_SIZE);
      char* temp = reinterpret_cast<char*>(value);
      for (uint32_t idx = 0;
           properties_.MarketingName[idx] != 0 && idx < HSA_PUBLIC_NAME_SIZE - 1; idx++) {
        temp[idx] = (uint8_t)properties_.MarketingName[idx];
      }
      break;
    }
    case HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU:
      *((uint32_t*)value) = static_cast<uint32_t>(
          properties_.NumSIMDPerCU * properties_.MaxWavesPerSIMD);
      break;
    case HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU:
      *((uint32_t*)value) = properties_.NumSIMDPerCU;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES:
      *((uint32_t*)value) = properties_.NumShaderBanks;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE:
      *((uint32_t*)value) = properties_.NumArrays;
      break;
    case HSA_AMD_AGENT_INFO_HDP_FLUSH:
      *((hsa_amd_hdp_flush_t*)value) = HDP_flush_;
      break;
    case HSA_AMD_AGENT_INFO_DOMAIN:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.Domain);
      break;
    case HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES:
      *((bool*)value) = properties_.NumGws != 0;
      break;
    case HSA_AMD_AGENT_INFO_UUID: {
      uint64_t uuid_value = static_cast<uint64_t>(properties_.UniqueID);

      // Either device does not support UUID e.g. a Gfx8 device,
      // or runtime is using an older thunk library that does not
      // support UUID's
      if (uuid_value == 0) {
        char uuid_tmp[] = "GPU-XX";
        snprintf((char*)value, sizeof(uuid_tmp), "%s", uuid_tmp);
        break;
      }

      // Device supports UUID, build UUID string to return
      std::stringstream ss;
      ss << "GPU-" << std::setfill('0') << std::setw(sizeof(uint64_t) * 2) << std::hex
         << uuid_value;
      snprintf((char*)value, (ss.str().length() + 1), "%s", (char*)ss.str().c_str());
      break;
    }
    case HSA_AMD_AGENT_INFO_ASIC_REVISION:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.Capability.ui32.ASICRevision);
      break;
    case HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS:
      assert(regions_.size() != 0 && "No device local memory found!");
      *((bool*)value) = properties_.Capability.ui32.CoherentHostAccess == 1;
      break;
    case HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT:
      if (core::Runtime::runtime_singleton_->flag().coop_cu_count() &&
          !(core::Runtime::runtime_singleton_->flag().cu_mask(enum_index_).empty())) {
        debug_warning("Cooperative launch and CU masking are currently incompatible!");
        *((uint32_t*)value) = 0;
        break;
      }

      if (core::Runtime::runtime_singleton_->flag().coop_cu_count() &&
          (isa_->GetMajorVersion() == 9) && (isa_->GetMinorVersion() == 0) &&
          (isa_->GetStepping() == 10)) {
        uint32_t count = 0;
        hsa_status_t err = GetInfo((hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &count);
        assert(err == HSA_STATUS_SUCCESS && "CU count query failed.");
        *((uint32_t*)value) = (count & 0xFFFFFFF8) - 8;  // value = floor(count/8)*8-8
        break;
      }
      return GetInfo((hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, value);
    case HSA_AMD_AGENT_INFO_MEMORY_AVAIL: {
      HSAuint64 availableBytes;
      hsa_status_t status;

      status = driver().AvailableMemory(node_id(), &availableBytes);

      if (status != HSA_STATUS_SUCCESS) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

      for (auto r : regions()) availableBytes += ((AMD::MemoryRegion*)r)->GetCacheSize();

      availableBytes += scratch_cache_.free_bytes() - scratch_cache_.reserved_bytes();

      *((uint64_t*)value) = availableBytes;
      break;
    }
    case HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY:
      *((uint64_t*)value) = wallclock_frequency_;
      break;
    case HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.FamilyID);
      break;
    case HSA_AMD_AGENT_INFO_UCODE_VERSION:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.EngineId.ui32.uCode);
      break;
    case HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.uCodeEngineVersions.uCodeSDMA);
      break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_ENG:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.NumSdmaEngines);
      break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.NumSdmaXgmiEngines);
      break;
    case HSA_AMD_AGENT_INFO_IOMMU_SUPPORT:
      if (properties_.Capability.ui32.HSAMMUPresent)
        *((hsa_amd_iommu_version_t*)value) = HSA_IOMMU_SUPPORT_V2;
      else
        *((hsa_amd_iommu_version_t*)value) = HSA_IOMMU_SUPPORT_NONE;
      break;
    case HSA_AMD_AGENT_INFO_NUM_XCC:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.NumXcc);
      break;
    case HSA_AMD_AGENT_INFO_DRIVER_UID:
      *((uint32_t*)value) = KfdGpuID();
      break;
    case HSA_AMD_AGENT_INFO_NEAREST_CPU:
      *((hsa_agent_t*)value) = GetNearestCpuAgent()->public_handle();
      break;
    case HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES:
      memset(value, 0, sizeof(uint8_t) * 8);
      GetInfoMemoryProperties((uint8_t*)value);
      break;
    case HSA_AMD_AGENT_INFO_AQL_EXTENSIONS:
      memset(value, 0, sizeof(uint8_t) * 8);
      /* Not yet implemented */
      break;
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX:
      *((uint64_t*)value) = MaxScratchDevice();
      break;
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT:
      *((uint64_t*)value) = scratch_limit_async_threshold_;
      break;
    case HSA_AMD_AGENT_INFO_CLOCK_COUNTERS: {
      HsaClockCounters hsakmt_counters = {};
      hsa_amd_clock_counters_t* counters = static_cast<hsa_amd_clock_counters_t*>(value);

      hsa_status_t err = driver().GetClockCounters(node_id(), &hsakmt_counters);
      if (err == HSA_STATUS_SUCCESS) {
        counters->cpu_clock_counter = hsakmt_counters.CPUClockCounter;
        counters->gpu_clock_counter = hsakmt_counters.GPUClockCounter;
        counters->system_clock_counter = hsakmt_counters.SystemClockCounter;
        counters->system_clock_frequency = hsakmt_counters.SystemClockFrequencyHz;
        break;
      }
      return HSA_STATUS_ERROR;
    }
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      break;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                                   core::HsaEventCallback event_callback, void* data,
                                   uint32_t private_segment_size, uint32_t group_segment_size,
                                   core::Queue** queue) {
  // Handle GWS queues.
  if (queue_type == HSA_QUEUE_TYPE_COOPERATIVE) {
    ScopedAcquire<KernelMutex> lock(&gws_queue_.lock_);
    auto ret = (*gws_queue_.queue_).get();
    if (ret != nullptr) {
      gws_queue_.ref_ct_++;
      *queue = ret;
      return HSA_STATUS_SUCCESS;
    }
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  // AQL queues must be a power of two in length.
  if (!IsPowerOfTwo(size)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Enforce max size
  if (size > maxAqlSize_) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  // Enforce min size
  if (size < minAqlSize_) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Allocate scratch memory
  ScratchInfo scratch = {0};
  if (private_segment_size == UINT_MAX) {
    private_segment_size = (profile_ == HSA_PROFILE_BASE) ? 0 : scratch_per_thread_;
  }

  if (private_segment_size > 262128) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  // Asynchronous reclaim flag bit is set by CP FW on queue-connect, we will update this when
  // we get the first scratch request.
  scratch.async_reclaim = false;

  scratch.main_lanes_per_wave = 64;
  scratch.main_size_per_thread = AlignUp(private_segment_size, 1024 / scratch.main_lanes_per_wave);
  if (scratch.main_size_per_thread > 262128) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  scratch.main_size_per_thread = private_segment_size;

  const uint32_t num_cu = properties_.NumFComputeCores / properties_.NumSIMDPerCU;
  scratch.main_size = scratch.main_size_per_thread * properties_.MaxSlotsScratchCU *
      scratch.main_lanes_per_wave * num_cu;
  scratch.main_queue_base = nullptr;
  scratch.main_queue_process_offset = 0;

  MAKE_NAMED_SCOPE_GUARD(scratchGuard, [&]() { ReleaseQueueMainScratch(scratch); });

  if (scratch.main_size != 0) {
    AcquireQueueMainScratch(scratch);
    if (scratch.main_queue_base == nullptr) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
  }

  // Ensure utility queue has been created.
  // Deferring longer risks exhausting queue count before ISA upload and invalidation capability is
  // ensured.
  queues_[QueueUtility].touch();

  bool dev_mem_queue_descriptor = (flags & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR) != 0;

  // Create an HW AQL queue
  core::SharedQueue* shared_queue = nullptr;

  if (dev_mem_queue_descriptor) {
    shared_queue = static_cast<core::SharedQueue*>(
        finegrain_allocator()(sizeof(core::SharedQueue), core::MemoryRegion::AllocateUncached));
  } else {
    shared_queue =
        static_cast<core::SharedQueue*>(core::Runtime::runtime_singleton_->system_allocator()(
            sizeof(core::SharedQueue), MemoryRegion::GetPageSize(),
            isMES() ? (MemoryRegion::AllocateGTTAccess | MemoryRegion::AllocateNonPaged) : 0,
            node_id()));
  }

  if (!shared_queue) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto aql_queue = new AqlQueue(shared_queue, this, size, node_id(), scratch, event_callback, data,
                                flags);
  *queue = aql_queue;
  aql_queues_.push_back(aql_queue);

  if (doorbell_queue_map_) {
    // Calculate index of the queue doorbell within the doorbell aperture.
    auto doorbell_addr = uintptr_t(aql_queue->signal_.hardware_doorbell_ptr);
    auto doorbell_idx = (doorbell_addr >> 3) & (MAX_NUM_DOORBELLS - 1);
    doorbell_queue_map_[doorbell_idx] = &aql_queue->amd_queue_;
  }

  scratchGuard.Dismiss();
  return HSA_STATUS_SUCCESS;
}

void GpuAgent::AcquireQueueMainScratch(ScratchInfo& scratch) {
  assert(scratch.main_queue_base == nullptr &&
         "AcquireQueueMainScratch called while holding scratch.");
  bool need_queue_scratch_base = (isa_->GetMajorVersion() > 8);

  if (scratch.main_size == 0) {
    scratch.main_size = queue_scratch_len_;
    scratch.main_size_per_thread = scratch_per_thread_;
  }
  scratch.retry = false;

  // Fail scratch allocation if per wave limits are exceeded.
  uint64_t size_per_wave = AlignUp(scratch.main_size_per_thread * properties_.WaveFrontSize, 1024);
  if (size_per_wave > MAX_WAVE_SCRATCH) return;

  /*
  Determine size class needed.

  Scratch allocations come in two flavors based on how it is retired.  Small allocations may be
  kept bound to a queue and reused by firmware.  This memory can not be reclaimed by the runtime
  on demand so must be kept small to avoid egregious OOM conditions.  Other allocations, aka large,
  may be used by firmware only for one dispatch and are then surrendered to the runtime.  This has
  significant latency so we don't want to make all scratch allocations large (ie single use).

  Note that the designation "large" is for contrast with "small", which must really be small
  amounts of memory, and does not always imply a large quantity of memory is needed.  Other
  properties of the allocation may require single use and so qualify the allocation or use as
  "large".

  Here we decide on the boundaries for small scratch allocations.  Both the largest small single
  allocation and the maximum amount of memory bound by small allocations are limited.  Additionally
  some legacy devices do not support large scratch.

  For small scratch we must allocate enough memory for every physical scratch slot.
  For large scratch compute the minimum memory needed to run the dispatch without limiting
  occupancy.
  Limit total bound small scratch allocations to 1/8th of scratch pool and 1/4 of that for a single
  allocation.
  */
  bool large;

  ScopedAcquire<KernelMutex> lock(&scratch_lock_);
  const size_t small_limit = scratch_pool_.size() >> 3;
  bool use_reclaim = true;

  large = (scratch.main_size > scratch.use_once_limit) ||
          (!AsyncScratchReclaimEnabled() &&
            ((scratch_pool_.size() - scratch_pool_.remaining() - scratch_cache_.free_bytes() +
             scratch.main_size) > small_limit));

  if ((isa_->GetMajorVersion() < 8) ||
      core::Runtime::runtime_singleton_->flag().no_scratch_reclaim()) {
    large = false;
    use_reclaim = false;
  }

  // If large is selected then the scratch will not be retained.
  // In that case allocate the minimum necessary for the dispatch since we don't need all slots.
  if (large) scratch.main_size = scratch.dispatch_size;

  // Ensure mapping will be in whole pages.
  scratch.main_size = AlignUp(scratch.main_size, 4096);

  /*
  Sequence of attempts is:
    check cache
    attempt a new allocation
    trim unused blocks from cache
    attempt a new allocation
    check cache for sufficient used block, steal and wait (not implemented)
    trim used blocks from cache, evaluate retry
    reduce occupancy
  */

  // Lambda called in place.
  // Used to allow exit from nested loops.
  [&]() {
    // Check scratch cache
    scratch.large = large;
    if (scratch_cache_.allocMain(scratch)) return;

    // Attempt new allocation.
    for (int i = 0; i < 3; i++) {
      if (large)
        scratch.main_queue_base = scratch_pool_.alloc_high(scratch.main_size);
      else
        scratch.main_queue_base = scratch_pool_.alloc(scratch.main_size);

      scratch.large = large | (scratch.main_queue_base > scratch_pool_.high_split());
      assert(((!scratch.large) | use_reclaim) && "Large scratch used with reclaim disabled.");

      if (scratch.main_queue_base != nullptr) {
        HSAuint64 alternate_va;
        if ((profile_ == HSA_PROFILE_FULL) ||
            (driver().MakeMemoryResident(scratch.main_queue_base, scratch.main_size,
                                         &alternate_va) == HSA_STATUS_SUCCESS)) {
          if (scratch.large) scratch_used_large_ += scratch.main_size;
          scratch_cache_.insertMain(scratch);
          return;
        }
      }

      // Scratch request failed allocation or mapping.
      scratch_pool_.free(scratch.main_queue_base);
      scratch.main_queue_base = nullptr;

      // Release cached scratch and retry.
      // First iteration trims unused blocks, second trims all. 3rd uses reserved memory
      switch (i) {
        case 0:
          scratch_cache_.trim(false);
          break;
        case 1:
          scratch_cache_.trim(true);
          break;
        case 2:
          if (scratch_cache_.use_reserved(scratch)) return;
      }
    }

    // Retry if large may yield needed space.
    if (scratch_used_large_ != 0) {
      if (AddScratchNotifier(scratch.queue_retry, 0x8000000000000000ull)) scratch.retry = true;
      return;
    }

    // Fail scratch allocation if reducing occupancy is disabled.
    if (scratch.cooperative || (!use_reclaim) ||
        core::Runtime::runtime_singleton_->flag().no_scratch_thread_limiter())
      return;

    // Attempt to trim the maximum number of concurrent waves to allow scratch to fit.
    if (core::Runtime::runtime_singleton_->flag().enable_queue_fault_message())
      debug_print("Failed to map requested scratch (%ld) - reducing queue occupancy.\n",
                  scratch.main_size);
    const uint64_t num_cus = properties_.NumFComputeCores / properties_.NumSIMDPerCU;
    const uint64_t se_per_xcc = properties_.NumShaderBanks / properties_.NumXcc;

    const uint64_t total_waves = scratch.main_size / size_per_wave;
    uint64_t waves_per_cu = AlignUp(total_waves / num_cus, scratch.main_waves_per_group);

    while (waves_per_cu != 0) {
      size_t size = waves_per_cu * num_cus * size_per_wave;
      void* base = scratch_pool_.alloc_high(size);
      HSAuint64 alternate_va;
      if ((base != nullptr) &&
          ((profile_ == HSA_PROFILE_FULL) ||
           (driver().MakeMemoryResident(base, size, &alternate_va) == HSA_STATUS_SUCCESS))) {
        // Scratch allocated and either full profile or map succeeded.
        scratch.main_queue_base = base;
        scratch.main_size = size;
        scratch.large = true;
        scratch_used_large_ += scratch.main_size;
        scratch_cache_.insertMain(scratch);
        if (core::Runtime::runtime_singleton_->flag().enable_queue_fault_message())
          debug_print("  %ld scratch mapped, %.2f%% occupancy.\n", scratch.main_size,
                      float(waves_per_cu * num_cus) / scratch.dispatch_slots * 100.0f);
        return;
      }
      scratch_pool_.free(base);

      // Wave count must be divisible by #SEs in an XCC. If occupancy must be reduced
      // such that waves_per_cu < waves_per_group, continue reducing by #SEs per XCC
      // (only allowed if waves_per_group is a multiple #SEs per XCC).
      waves_per_cu -= (waves_per_cu <= scratch.main_waves_per_group &&
                       se_per_xcc < scratch.main_waves_per_group &&
                       scratch.main_waves_per_group % se_per_xcc == 0)
                       ? se_per_xcc
                       : scratch.main_waves_per_group;
    }

    // Failed to allocate minimal scratch
    assert(scratch.main_queue_base == nullptr && "bad scratch data");
    if (core::Runtime::runtime_singleton_->flag().enable_queue_fault_message())
      debug_print("  Could not allocate scratch for one wave per CU.\n");
    return;
  }();

  scratch.main_queue_process_offset = need_queue_scratch_base
      ? uintptr_t(scratch.main_queue_base)
      : uintptr_t(scratch.main_queue_base) - uintptr_t(scratch_pool_.base());
}

/* Should be called with scratch_lock_ */
void GpuAgent::ReleaseQueueMainScratch(ScratchInfo& scratch) {
  assert(scratch.main_queue_base);

  scratch_cache_.freeMain(scratch);
  scratch.main_queue_base = nullptr;
}

void GpuAgent::AcquireQueueAltScratch(ScratchInfo& scratch) {
  assert(scratch.async_reclaim && "Acquire Alt Scratch when FW does not support it");
  assert(scratch.alt_queue_base == nullptr &&
         "AcquireQueueAltScratch called while holding alt scratch.");

  // Fail scratch allocation if per wave limits are exceeded.
  uint64_t size_per_wave = AlignUp(scratch.alt_size_per_thread * properties_.WaveFrontSize, 1024);
  if (size_per_wave > MAX_WAVE_SCRATCH) return;

  ScopedAcquire<KernelMutex> lock(&scratch_lock_);

  // Ensure mapping will be in whole pages.
  scratch.alt_size = AlignUp(scratch.alt_size, 4096);

  /*
  Sequence of attempts is:
    check cache
    attempt a new allocation
    trim unused blocks from cache
    attempt a new allocation
    check cache for sufficient used block, steal and wait (not implemented)
    trim used blocks from cache, evaluate retry
  */

  // Lambda called in place.
  // Used to allow exit from nested loops.
  [&]() {
    // Check scratch cache
    if (scratch_cache_.allocAlt(scratch)) return;

    // Attempt new allocation.
    for (int i = 0; i < 2; i++) {
      scratch.alt_queue_base = scratch_pool_.alloc(scratch.alt_size);
      if (scratch.alt_queue_base != nullptr) {
        HSAuint64 alternate_va;
        if ((profile_ == HSA_PROFILE_FULL) ||
            (driver().MakeMemoryResident(scratch.alt_queue_base, scratch.alt_size, &alternate_va) ==
             HSA_STATUS_SUCCESS)) {
          scratch_cache_.insertAlt(scratch);
          return;
        }
      }

      // Scratch request failed allocation or mapping.
      scratch_pool_.free(scratch.alt_queue_base);
      scratch.alt_queue_base = nullptr;

      // Release cached scratch and retry.
      // First iteration trims unused blocks, second trims all. 3rd uses reserved memory
      switch (i) {
        case 0:
          scratch_cache_.trim(false);
          break;
        case 1:
          scratch_cache_.trim(true);
          break;
      }
    }

    if (core::Runtime::runtime_singleton_->flag().enable_queue_fault_message())
      debug_print("  Could not allocate alt scratch.\n");
    return;
  }();

  scratch.alt_queue_process_offset = uintptr_t(scratch.alt_queue_base);
}

/* Should be called with scratch_lock_ */
void GpuAgent::ReleaseQueueAltScratch(ScratchInfo& scratch) {
  assert(scratch.alt_queue_base);

  scratch_cache_.freeAlt(scratch);
  scratch.alt_queue_base = nullptr;
}

void GpuAgent::ReleaseScratch(void* base, size_t size, bool large) {
  if (profile_ == HSA_PROFILE_BASE) {
    if (HSA_STATUS_SUCCESS != driver().MakeMemoryUnresident(base)) {
      assert(false && "Unmap scratch subrange failed!");
    }
  }
  scratch_pool_.free(base);

  if (large) scratch_used_large_ -= size;

  // Notify waiters that additional scratch may be available.
  for (auto notifier : scratch_notifiers_) {
    HSA::hsa_signal_or_relaxed(notifier.first, notifier.second);
  }
  ClearScratchNotifiers();
}

// Go through all the AQL queues and try to release scratch memory
void GpuAgent::AsyncReclaimScratchQueues() {
  for (auto iter : aql_queues_) {
    auto aqlQueue = static_cast<AqlQueue*>(iter);
    aqlQueue->AsyncReclaimMainScratch();
    aqlQueue->AsyncReclaimAltScratch();
  }
}

hsa_status_t GpuAgent::SetAsyncScratchThresholds(size_t use_once_limit) {
  if (use_once_limit > MaxScratchDevice()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  scratch_limit_async_threshold_ = use_once_limit;

  for (auto iter : aql_queues_) {
    auto aqlQueue = static_cast<AqlQueue*>(iter);
    aqlQueue->CheckScratchLimits();
  }
  return HSA_STATUS_SUCCESS;
}

void GpuAgent::TranslateTime(core::Signal* signal, hsa_amd_profiling_dispatch_time_t& time) {
  uint64_t start, end;
  signal->GetRawTs(false, start, end);
  // Order is important, we want to translate the end time first to ensure that packet duration is
  // not impacted by clock measurement latency jitter.
  time.end = TranslateTime(end);
  time.start = TranslateTime(start);

  if ((start == 0) || (end == 0) || (start < t0_.GPUClockCounter) || (end < t0_.GPUClockCounter))
    debug_print("Signal %p time stamps may be invalid.\n", &signal->signal_);
}

void GpuAgent::TranslateTime(core::Signal* signal, hsa_amd_profiling_async_copy_time_t& time) {
  uint64_t start, end;
  signal->GetRawTs(true, start, end);
  // Order is important, we want to translate the end time first to ensure that packet duration is
  // not impacted by clock measurement latency jitter.
  time.end = TranslateTime(end);
  time.start = TranslateTime(start);

  if ((start == 0) || (end == 0) || (start < t0_.GPUClockCounter) || (end < t0_.GPUClockCounter))
    debug_print("Signal %p time stamps may be invalid.\n", &signal->signal_);
}

/*
Times during program execution are interpolated to adjust for relative clock drift.
Interval timing may appear as ticks well before process start, leading to large errors due to
frequency adjustment (ie the profiling with NTP problem).  This is fixed by using a fixed frequency
for early times.
Intervals larger than t0_ will be frequency adjusted.  This admits a numerical error of not more
than twice the frequency stability (~10^-5).
*/
uint64_t GpuAgent::TranslateTime(uint64_t tick) {
  // Only allow short (error bounded) extrapolation for times during program execution.
  // Limit errors due to relative frequency drift to ~0.5us.  Sync clocks at 16Hz.
  const int64_t max_extrapolation = core::Runtime::runtime_singleton_->sys_clock_freq() >> 4;

  ScopedAcquire<KernelMutex> lock(&t1_lock_);
  // Limit errors due to correlated pair certainty to ~0.5us.
  // extrapolated time < (0.5us / half clock read certainty) * delay between clock measures
  // clock read certainty is <4us.
  if (((t1_.GPUClockCounter - t0_.GPUClockCounter) >> 2) + t1_.GPUClockCounter < tick) SyncClocks();

  // Good for ~300 yrs
  // uint64_t sysdelta = t1_.SystemClockCounter - t0_.SystemClockCounter;
  // uint64_t gpudelta = t1_.GPUClockCounter - t0_.GPUClockCounter;
  // int64_t offtick = int64_t(tick - t1_.GPUClockCounter);
  //__int128 num = __int128(sysdelta)*__int128(offtick) +
  //__int128(gpudelta)*__int128(t1_.SystemClockCounter);
  //__int128 sysLarge = num / __int128(gpudelta);
  // return sysLarge;

  // Good for ~3.5 months.
  uint64_t system_tick = 0;
  int64_t elapsed = 0;
  double ratio;

  // Valid ticks only need at most one SyncClocks.
  for (int i = 0; i < 2; i++) {
    ratio = double(t1_.SystemClockCounter - t0_.SystemClockCounter) /
        double(t1_.GPUClockCounter - t0_.GPUClockCounter);
    elapsed = int64_t(ratio * double(int64_t(tick - t1_.GPUClockCounter)));

    // Skip clock sync if under the extrapolation limit.
    if (elapsed < max_extrapolation) break;
    SyncClocks();
  }

  system_tick = uint64_t(elapsed) + t1_.SystemClockCounter;

  // tick predates HSA startup - extrapolate with fixed clock ratio
  if (tick < t0_.GPUClockCounter) {
    if (historical_clock_ratio_ == 0.0) historical_clock_ratio_ = ratio;
    system_tick = uint64_t(historical_clock_ratio_ * double(int64_t(tick - t0_.GPUClockCounter))) +
        t0_.SystemClockCounter;
  }

  return system_tick;
}

/* This function is deprecated */
bool GpuAgent::current_coherency_type(hsa_amd_coherency_type_t type) {
  current_coherency_type_ = type;
  return true;
}

uint16_t GpuAgent::GetMicrocodeVersion() const {
  return (properties_.EngineId.ui32.uCode);
}

uint16_t GpuAgent::GetSdmaMicrocodeVersion() const {
  return (properties_.uCodeEngineVersions.uCodeSDMA);
}

void GpuAgent::SyncClocks() {
  hsa_status_t err = driver().GetClockCounters(node_id(), &t1_);
  assert(err == HSA_STATUS_SUCCESS && "hsaGetClockCounters error");
}

hsa_status_t GpuAgent::UpdateTrapHandlerWithPCS(pcs_sampling_data_t* pcs_hosttrap_buffers, pcs_sampling_data_t* pcs_stochastic_buffers) {
  // Assemble the trap handler source code.
  void* tma_addr = nullptr;
  uint64_t tma_size = 0;

  assert(core::Runtime::runtime_singleton_->KfdVersion().supports_exception_debugging);

  AssembleShader("TrapHandlerKfdExceptions", AssembleTarget::ISA, trap_code_buf_,
                 trap_code_buf_size_);

  /* pcs_hosttrap_buffers and pcs_stochastic_buffers are NULL until PC sampling is enabled */
  if (pcs_hosttrap_buffers || pcs_stochastic_buffers) {
    // ON non-large BAR systems, we cannot access device memory so we create a host copy
    // and then do a DmaCopy to device memory
    void* tma_region_host = (uint64_t*)system_allocator()(2 * sizeof(uint64_t), 0x1000, 0);
    if (tma_region_host == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    MAKE_SCOPE_GUARD([&]() { system_deallocator()(tma_region_host); });

    ((uint64_t*)tma_region_host)[0] = (uint64_t)pcs_hosttrap_buffers;
    ((uint64_t*)tma_region_host)[1] = (uint64_t)pcs_stochastic_buffers;

    if (!trap_handler_tma_region_) {
      trap_handler_tma_region_ = (uint64_t*)finegrain_allocator()(2 * sizeof(uint64_t), 0);
      if (trap_handler_tma_region_ == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

      // NearestCpuAgent owns pool returned system_allocator()
      auto cpuAgent = GetNearestCpuAgent()->public_handle();

      hsa_status_t ret =
          AMD::hsa_amd_agents_allow_access(1, &cpuAgent, NULL, trap_handler_tma_region_);
      assert(ret == HSA_STATUS_SUCCESS);
    }

    /* On non-large BAR systems, we may not be able to access device memory, so do a DmaCopy */
    if (DmaCopy(trap_handler_tma_region_, tma_region_host, 2 * sizeof(uint64_t)) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;

    tma_size = 2 * sizeof(uint64_t);
    tma_addr = trap_handler_tma_region_;
  } else if (trap_handler_tma_region_) {
    finegrain_deallocator()(trap_handler_tma_region_);
    trap_handler_tma_region_ = NULL;
  }

  // Bind the trap handler to this node.
  return driver().SetTrapHandler(node_id(), trap_code_buf_, trap_code_buf_size_, tma_addr,
                                 tma_size);
}

void GpuAgent::BindTrapHandler() {
  if (isa_->GetMajorVersion() == 7) {
    // No trap handler support on Gfx7, soft error.
    return;
  }

  // Assemble the trap handler source code.
  void* tma_addr = nullptr;
  uint64_t tma_size = 0;

  if (core::Runtime::runtime_singleton_->KfdVersion().supports_exception_debugging) {
    AssembleShader("TrapHandlerKfdExceptions", AssembleTarget::ISA, trap_code_buf_,
                   trap_code_buf_size_);
  } else {
    if (isa_->GetMajorVersion() >= 11 ||
       (isa_->GetMajorVersion() == 9 &&
        (isa_->GetMinorVersion() == 4 || isa_->GetMinorVersion() == 5))) {
      // No trap handler support without exception handling, soft error.
      return;
    }

    AssembleShader("TrapHandler", AssembleTarget::ISA, trap_code_buf_, trap_code_buf_size_);

    // Make an empty map from doorbell index to queue.
    // The trap handler uses this to retrieve a wave's amd_queue_v2_t*.
    auto doorbell_queue_map_size = MAX_NUM_DOORBELLS * sizeof(amd_queue_v2_t*);

    doorbell_queue_map_ = (amd_queue_v2_t**)system_allocator()(doorbell_queue_map_size, 0x1000, 0);
    assert(doorbell_queue_map_ != NULL && "Doorbell queue map allocation failed");

    memset(doorbell_queue_map_, 0, doorbell_queue_map_size);

    tma_addr = doorbell_queue_map_;
    tma_size = doorbell_queue_map_size;
  }

  // Bind the trap handler to this node.
  hsa_status_t err =
      driver().SetTrapHandler(node_id(), trap_code_buf_, trap_code_buf_size_, tma_addr, tma_size);
  assert(err == HSA_STATUS_SUCCESS && "SetTrapHandler() failed");
}

void GpuAgent::InvalidateCodeCaches(void *ptr, size_t size) {
  // Check for microcode cache invalidation support.
  // This is deprecated in later microcode builds.
  if (isa_->GetMajorVersion() == 7) {
    if (properties_.EngineId.ui32.uCode < 420) {
      // Microcode is handling code cache invalidation.
      return;
    }
  } else if (isa_->GetMajorVersion() == 8 && isa_->GetMinorVersion() == 0) {
    if (properties_.EngineId.ui32.uCode < 685) {
      // Microcode is handling code cache invalidation.
      return;
    }
  } else if (isa_->GetMajorVersion() > 12) {
    assert(false && "Code cache invalidation not implemented for this agent");
  }

  // Invalidate caches which may hold lines of code object allocation.
  uint32_t cache_inv[8] = {0};
  uint32_t cache_inv_size_dw;

  if (isa_->GetMajorVersion() < 10) {
      cache_inv[1] = PM4_ACQUIRE_MEM_DW1_COHER_CNTL(
          PM4_ACQUIRE_MEM_COHER_CNTL_SH_ICACHE_ACTION_ENA |
          PM4_ACQUIRE_MEM_COHER_CNTL_SH_KCACHE_ACTION_ENA |
          PM4_ACQUIRE_MEM_COHER_CNTL_TC_ACTION_ENA |
          PM4_ACQUIRE_MEM_COHER_CNTL_TC_WB_ACTION_ENA);

      cache_inv_size_dw = 7;
  } else {
      cache_inv[7] = PM4_ACQUIRE_MEM_DW7_GCR_CNTL(
          PM4_ACQUIRE_MEM_GCR_CNTL_GLI_INV(1) |
          PM4_ACQUIRE_MEM_GCR_CNTL_GLK_INV |
          PM4_ACQUIRE_MEM_GCR_CNTL_GLV_INV |
          PM4_ACQUIRE_MEM_GCR_CNTL_GL1_INV |
          PM4_ACQUIRE_MEM_GCR_CNTL_GL2_INV);

      cache_inv_size_dw = 8;
  }

  cache_inv[0] = PM4_HDR(PM4_HDR_IT_OPCODE_ACQUIRE_MEM, cache_inv_size_dw,
             isa_->GetMajorVersion());

  if (ptr) {
    size_t size_granule = (size + 0xFF) >> 8;
    cache_inv[2] = PM4_ACQUIRE_MEM_DW2_COHER_SIZE(size_granule);
    cache_inv[3] = PM4_ACQUIRE_MEM_DW3_COHER_SIZE_HI(size_granule >> 32);
    cache_inv[4] = PM4_ACQUIRE_MEM_DW4_COHER_BASE((uint64_t)ptr);
    cache_inv[5] = PM4_ACQUIRE_MEM_DW4_COHER_BASE_HI((uint64_t)ptr);
  } else {
    cache_inv[2] = PM4_ACQUIRE_MEM_DW2_COHER_SIZE(0xFFFFFFFF);
    cache_inv[3] = PM4_ACQUIRE_MEM_DW3_COHER_SIZE_HI(0xFF);
  }

  // Submit the command to the utility queue and wait for it to complete.
  queues_[QueueUtility]->ExecutePM4(cache_inv, cache_inv_size_dw * sizeof(uint32_t));
}

lazy_ptr<core::Blit>& GpuAgent::GetBlitObject(uint32_t engine_offset) {
  sdma_blit_used_mask_ |= 1 << engine_offset;
  return blits_[engine_offset];
}

lazy_ptr<core::Blit>& GpuAgent::GetXgmiBlit(const core::Agent& dst_agent) {
  // Determine if destination is a member xgmi peers list
  uint32_t xgmi_engine_cnt = properties_.NumSdmaXgmiEngines;
  assert((xgmi_engine_cnt > 0) && ("Illegal condition, should not happen"));

  ScopedAcquire<KernelMutex> lock(&xgmi_peer_list_lock_);

  for (uint32_t idx = 0; idx < xgmi_peer_list_.size(); idx++) {
    uint64_t dst_handle = dst_agent.public_handle().handle;
    uint64_t peer_handle = xgmi_peer_list_[idx]->public_handle().handle;
    if (peer_handle == dst_handle) {
      return blits_[(idx % xgmi_engine_cnt) + DefaultBlitCount];
    }
  }

  // Add agent to the xGMI neighbours list
  xgmi_peer_list_.push_back(&dst_agent);
  return GetBlitObject(((xgmi_peer_list_.size() - 1) % xgmi_engine_cnt) + DefaultBlitCount);
}

lazy_ptr<core::Blit>& GpuAgent::GetPcieBlit(const core::Agent& dst_agent,
                                            const core::Agent& src_agent) {
  bool is_h2d = (src_agent.device_type() == core::Agent::kAmdCpuDevice &&
                 dst_agent.device_type() == core::Agent::kAmdGpuDevice);

  lazy_ptr<core::Blit>& blit = GetBlitObject(is_h2d ? BlitHostToDev : BlitDevToHost);
  return blit;
}

lazy_ptr<core::Blit>& GpuAgent::GetBlitObject(const core::Agent& dst_agent,
                                              const core::Agent& src_agent,
                                              const size_t size) {
  // At this point it is guaranteed that one of
  // the two devices is a GPU, potentially both
  assert(((src_agent.device_type() == core::Agent::kAmdGpuDevice) ||
          (dst_agent.device_type() == core::Agent::kAmdGpuDevice)) &&
         ("Both devices are CPU agents which is not expected"));

  // Determine if Src and Dst devices are same and are the copying device
  // Such a copy is in the device local memory, which can only be saturated by a blit kernel.
  if ((src_agent.public_handle().handle) == (dst_agent.public_handle().handle) &&
      (dst_agent.public_handle().handle == public_handle_.handle)) {
    // If the copy is very small then cache flush overheads can dominate.
    // Choose a (potentially) SDMA enabled engine to avoid cache flushing.
    if (size < core::Runtime::runtime_singleton_->flag().force_sdma_size()) {
      return GetBlitObject(BlitDevToHost);
    }
    return blits_[BlitDevToDev];
  }

  if (core::Runtime::runtime_singleton_->flag().enable_peer_sdma() == Flag::SDMA_DISABLE
      && src_agent.device_type() == core::Agent::kAmdGpuDevice
      && dst_agent.device_type() == core::Agent::kAmdGpuDevice) {
      return blits_[BlitDevToDev];
  }

  // Acquire Hive Id of Src and Dst devices - ignore hive id for CPU devices.
  // CPU-GPU connections should always use the host (aka pcie) facing SDMA engines, even if the
  // connection is XGMI.
  uint64_t src_hive_id =
      (src_agent.device_type() == core::Agent::kAmdGpuDevice) ? src_agent.HiveId() : 0;
  uint64_t dst_hive_id =
      (dst_agent.device_type() == core::Agent::kAmdGpuDevice) ? dst_agent.HiveId() : 0;

  // Bind to a PCIe facing Blit object if the two
  // devices have different Hive Ids. This can occur
  // for following scenarios:
  //
  //  Neither device claims membership in a Hive
  //   srcId = 0 <-> dstId = 0;
  //
  //  Src device claims membership in a Hive
  //   srcId = 0x1926 <-> dstId = 0;
  //
  //  Dst device claims membership in a Hive
  //   srcId = 0 <-> dstId = 0x1123;
  //
  //  Both device claims membership in a Hive
  //  and the  Hives are different
  //   srcId = 0x1926 <-> dstId = 0x1123;
  //
  if ((dst_hive_id != src_hive_id) || (dst_hive_id == 0)) {
    return GetPcieBlit(dst_agent, src_agent);
  }

  // Accommodates platforms where devices have xGMI
  // links but without sdmaXgmiEngines e.g. Vega 20
  if (properties_.NumSdmaXgmiEngines == 0) {
    return GetPcieBlit(dst_agent, src_agent);
  }

  return GetXgmiBlit(dst_agent);
}

void GpuAgent::Trim() {
  Agent::Trim();
  AsyncReclaimScratchQueues();
  ScopedAcquire<KernelMutex> lock(&scratch_lock_);
  scratch_cache_.trim(false);
}

void GpuAgent::InitAllocators() {
  for (auto pool : GetNearestCpuAgent()->regions()) {
    if (pool->kernarg()) {
      system_allocator_ = [pool](size_t size, size_t alignment,
                                 MemoryRegion::AllocateFlags alloc_flags) -> void* {
        assert(alignment <= 4096);
        void* ptr = nullptr;
        return (HSA_STATUS_SUCCESS ==
                core::Runtime::runtime_singleton_->AllocateMemory(pool, size, alloc_flags, &ptr))
            ? ptr
            : nullptr;
      };

      system_deallocator_ = [](void* ptr) { core::Runtime::runtime_singleton_->FreeMemory(ptr); };
    }
  }
  assert(system_allocator_ && "Nearest NUMA node did not have a kernarg pool.");

  // Setup this GPU's fine-grain and coarse-grain allocators.
  for (auto region : regions()) {
    const AMD::MemoryRegion* amd_region = static_cast<const AMD::MemoryRegion*>(region);

    auto region_allocator = [region](size_t size,
                                     MemoryRegion::AllocateFlags alloc_flags) -> void* {
      void* ptr = nullptr;
       return (HSA_STATUS_SUCCESS ==
               core::Runtime::runtime_singleton_->AllocateMemory(region, size, alloc_flags, &ptr))
           ? ptr
           : nullptr;
    };

    auto region_deallocator = [](void* ptr) { core::Runtime::runtime_singleton_->FreeMemory(ptr); };

    if (amd_region->IsLocalMemory() && amd_region->fine_grain()) {
      finegrain_allocator_ = region_allocator;
      finegrain_deallocator_ = region_deallocator;
    } else if (amd_region->IsLocalMemory() &&
               !(amd_region->fine_grain() || amd_region->extended_scope_fine_grain())) {
      coarsegrain_allocator_ = region_allocator;
      coarsegrain_deallocator_ = region_deallocator;
    }
  }
  assert(finegrain_allocator_ && "GPU agent does not have a fine-grain allocator");
  assert(coarsegrain_allocator_ && "GPU agent does not have a coarse-grain allocator");
}

core::Agent* GpuAgent::GetNearestCpuAgent() const {
  core::Agent* nearCpu = nullptr;
  uint32_t dist = -1u;
  for (auto cpu : core::Runtime::runtime_singleton_->cpu_agents()) {
    const core::Runtime::LinkInfo link_info =
        core::Runtime::runtime_singleton_->GetLinkInfo(node_id(), cpu->node_id());
    if (link_info.info.numa_distance < dist) {
      dist = link_info.info.numa_distance;
      nearCpu = cpu;
    }
  }
  return nearCpu;
}

hsa_status_t ConvertHsaKmtPcSamplingInfoToHsa(HsaPcSamplingInfo* hsaKmtPcSampling,
                                              hsa_ven_amd_pcs_configuration_t* hsaPcSampling) {
  assert(hsaKmtPcSampling && "Invalid hsaKmtPcSampling");
  assert(hsaPcSampling && "Invalid hsaPcSampling");

  switch (hsaKmtPcSampling->method) {
    case HSA_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1:
      hsaPcSampling->method = HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1;
      break;
    case HSA_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1:
      hsaPcSampling->method = HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1;
      break;
    default:
      // Sampling method not supported do not return this method to the user
      return HSA_STATUS_ERROR;
  }
  switch (hsaKmtPcSampling->units) {
    case HSA_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS:
      hsaPcSampling->units = HSA_VEN_AMD_PCS_INTERVAL_UNITS_MICRO_SECONDS;
      break;
    case HSA_PC_SAMPLING_UNIT_INTERVAL_CYCLES:
      hsaPcSampling->units = HSA_VEN_AMD_PCS_INTERVAL_UNITS_CLOCK_CYCLES;
      break;
    case HSA_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS:
      hsaPcSampling->units = HSA_VEN_AMD_PCS_INTERVAL_UNITS_INSTRUCTIONS;
      break;
    default:
      // Sampling unit not supported do not return this method to the user
      return HSA_STATUS_ERROR;
  }

  hsaPcSampling->min_interval = hsaKmtPcSampling->value_min;
  hsaPcSampling->max_interval = hsaKmtPcSampling->value_max;
  hsaPcSampling->flags = hsaKmtPcSampling->flags;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::PcSamplingIterateConfig(hsa_ven_amd_pcs_iterate_configuration_callback_t cb,
                                               void* cb_data) {
   uint32_t size = 0;

  if (!core::Runtime::runtime_singleton_->KfdVersion().supports_exception_debugging)
    return HSA_STATUS_ERROR;

  // First query to get size of list needed
  HSAKMT_STATUS ret = HSAKMT_CALL(hsaKmtPcSamplingQueryCapabilities(node_id(), NULL, 0, &size));
  if (ret != HSAKMT_STATUS_SUCCESS || size == 0) return HSA_STATUS_ERROR;

  std::vector<HsaPcSamplingInfo> sampleInfoList(size);
  ret = HSAKMT_CALL(hsaKmtPcSamplingQueryCapabilities(node_id(), sampleInfoList.data(), sampleInfoList.size(),
                                          &size));

  if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  for (uint32_t i = 0; i < size; i++) {
    hsa_ven_amd_pcs_configuration_t hsaPcSampling;
    if (ConvertHsaKmtPcSamplingInfoToHsa(&sampleInfoList[i], &hsaPcSampling) == HSA_STATUS_SUCCESS
        && cb(&hsaPcSampling, cb_data) == HSA_STATUS_INFO_BREAK)
          return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::PcSamplingCreate(pcs::PcsRuntime::PcSamplingSession& session) {
  hsa_status_t ret;
  HsaPcSamplingInfo sampleInfo = {};
  HsaPcSamplingTraceId thunkId;

  // IOCTL id does not exist at the moment, so passing 0 is OK,
  // since it will be overridden later in this function.
  ret = PcSamplingCreateFromId(0, session);
  if (ret != HSA_STATUS_SUCCESS) return ret;

  // Obtain the sampling information from the session.
  session.GetHsaKmtSamplingInfo(&sampleInfo);

  // Pass the sampling information to the kernel driver to create PC
  // sampling session.
  HSAKMT_STATUS retkmt = HSAKMT_CALL(hsaKmtPcSamplingCreate(node_id(), &sampleInfo, &thunkId));
  if (retkmt != HSAKMT_STATUS_SUCCESS) {
    return (retkmt == HSAKMT_STATUS_KERNEL_ALREADY_OPENED) ? (hsa_status_t)HSA_STATUS_ERROR_RESOURCE_BUSY
            : HSA_STATUS_ERROR;
  }

  debug_print("Created PC sampling session with thunkId:%d\n", thunkId);

  session.SetThunkId(thunkId);

  return ret;
}

hsa_status_t GpuAgent::PcSamplingCreateFromId(HsaPcSamplingTraceId ioctlId,
                                              pcs::PcsRuntime::PcSamplingSession& session) {
  // Determine the sampling method from the session
  hsa_ven_amd_pcs_method_kind_t sampling_method = session.method();

  pcs_data_t* pcs_data = nullptr;

  if (sampling_method == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1) {
    pcs_data = &pcs_hosttrap_data_;
  } else if (sampling_method == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1) {
    pcs_data = &pcs_stochastic_data_;
  } else {
    // Unsupported sampling method
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Ensure only one session is active at a time for the given method
  if (pcs_data->session)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;  // TODO: For now, we can only have
                                               // 1 pc sampling session at a
                                               // time. As a final solution, we
                                               // want to be able to support
                                               // multiple sessions at a time.
                                               // But this makes the
                                               // session.HandleSampleData more
                                               // complicated if multiple
                                               // sessions have different buffer
                                               // sizes.

  // This is current amd_aql_queue->pm4_ib_size_b_
  pcs_data->cmd_data_sz = 0x1000;  // 4KB
  pcs_data->cmd_data = (uint32_t*)malloc(pcs_data->cmd_data_sz);
  if (!pcs_data->cmd_data) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  if (HSA::hsa_signal_create(1, 0, NULL, &pcs_data->exec_pm4_signal) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  pcs_data->old_val = (uint64_t*)system_allocator()(sizeof(uint64_t), 0x1000, 0);
  if (!pcs_data->old_val) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  if (AMD::hsa_amd_agents_allow_access(1, &public_handle_, NULL, pcs_data->old_val))
    return HSA_STATUS_ERROR;

  // Local copy of pc sampling data - we cannot access device memory directly on non-large BAR
  // systems
  pcs_sampling_data_t* device_datahost =
      (pcs_sampling_data_t*)system_allocator()(sizeof(pcs_sampling_data_t), 0x1000, 0);
  if (!device_datahost) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  MAKE_SCOPE_GUARD([&]() { system_deallocator()(device_datahost); });

  memset(device_datahost, 0, sizeof(*device_datahost));

  if (AMD::hsa_amd_agents_allow_access(1, &public_handle_, NULL, device_datahost) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  MAKE_NAMED_SCOPE_GUARD(freeResources, [&]() {
    if (pcs_data->device_data) {
      if (pcs_data->device_data->done_sig0.handle)
        HSA::hsa_signal_destroy(pcs_data->device_data->done_sig0);
      if (pcs_data->device_data->done_sig1.handle)
        HSA::hsa_signal_destroy(pcs_data->device_data->done_sig1);

      finegrain_deallocator()(pcs_data->device_data);
    }
    if (pcs_data->host_buffer) system_deallocator()(pcs_data->host_buffer);
  });

  // Force creating of PC Sampling queue to trigger exception early in case we exceed max availble
  // CP queues on this agent
  queues_[QueuePCSampling].touch();

  /*
   * When calling queue->ExecutePM4() Indirect Buffer size which is 0x1000 bytes (1024 DW).
   * The maximum indirect buffer size we need occurs when we enqueue the
   * WAIT_REG_MEM, DMA_COPY(s), WRITE_DATA ops:
   * For WAIT_REG_MEM = 7 DW
   * For each DMA_COPY = 7 DW
   * For WRITE_DATA_CMD = 6 DW
   *
   * So maximum number of DMA_COPY ops is:
   * (MAX_IB_SIZE - sizeof(WAIT_REG_MEM) - sizeof(WRITE_DATA_CMD)) / sizeof(DMA_COPY)
   * (1024 - 7 - 6) / 7 = 144
   *
   * Each DMA_COPY op can transfer (1 << 26) bytes, which is 9 GB. trap_buffer_size is a 32-bit
   * number, so the buffer must be < 4 GB. So we are not limited by Indirect Buffer size.
   * Set current limit to 256 MB to limit device VRAM usage
   */
  const size_t max_trap_buffer_size =
      core::Runtime::runtime_singleton_->flag().pc_sampling_max_device_buffer_size();

  /*
   * We use a double-buffer mechanism where there are 2 trap-buffers and 1 host-buffer
   * Warning: This currently assumes that client latency is smaller than time to fill 1
   * trap-buffer If latency is bigger, we have to increate host-buffer
   *
   * host-buffer must be >= client-buffer so that we can copy full size of client-buffer each
   * time. To avoid having to deal with wrap-arounds, host-buffer must be a multiple of
   * trap-buffers
   *
   * if client-buffer size is greater than 2x max_trap_buffer_size:
   *    We are limited by max_trap_buffer_size.
   *    trap-buffer = max-trap-buffer-size
   *    host-buffer = 2*smallest size greater than client-buffer but multiple of 1 trap-buffer
   * else:
   *    We reduce the trap-buffers so that:
   *    trap-buffer = half of user-buffer
   *    host-buffer = 2*user-buffer
   *
   * TODO: We are currently using a temporary host-buffer so that we can increase host-buffer to
   * factor in client latency. Using a direct-copy to the client buffer would be more efficient.
   * Revisit this once we have empirical data of latency vs how long it takes to fill 1
   * trap-buffer.
   */

  size_t trap_buffer_size = 0;
  if (session.buffer_size() > 2 * max_trap_buffer_size) {
    trap_buffer_size = max_trap_buffer_size;
    pcs_data->host_buffer_size = 2 * AlignUp(session.buffer_size(), trap_buffer_size);
    } else {
      trap_buffer_size = session.buffer_size() / 2;
      pcs_data->host_buffer_size = 2 * session.buffer_size();
    }

    pcs_data->host_buffer = (uint8_t*)system_allocator()(pcs_data->host_buffer_size, 0x1000, 0);
    if (!pcs_data->host_buffer) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    if (AMD::hsa_amd_agents_allow_access(1, &public_handle_, NULL, pcs_data->host_buffer) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;

    device_datahost->buf_size = trap_buffer_size / session.sample_size();

    if (HSA::hsa_signal_create(1, 0, NULL, &device_datahost->done_sig0) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    if (HSA::hsa_signal_create(1, 0, NULL, &device_datahost->done_sig1) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    // TODO: Once we have things working and can measure
    // latency after 2nd level trap handler decrements signals and set watermark accordingly
    device_datahost->buf_watermark0 = 0.8 * device_datahost->buf_size;
    device_datahost->buf_watermark1 = 0.8 * device_datahost->buf_size;

    // Allocate device memory for 2nd level trap handler TMA
    size_t deviceAllocSize = sizeof(pcs_sampling_data_t) + (2 * trap_buffer_size);
    pcs_data->device_data = (pcs_sampling_data_t*)finegrain_allocator()(deviceAllocSize, 0);
    if (pcs_data->device_data == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    // This cpuAgent is the owner of the system_allocator() pool
    auto cpuAgent = GetNearestCpuAgent()->public_handle();
    if (AMD::hsa_amd_agents_allow_access(1, &cpuAgent, NULL, pcs_data->device_data) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;

    if (DmaCopy(pcs_data->device_data, device_datahost, sizeof(*device_datahost)) !=
        HSA_STATUS_SUCCESS) {
      debug_print("Failed to dmaCopy!\n");
      return HSA_STATUS_ERROR;
    }

    uint8_t* device_buf_ptr =
	reinterpret_cast<uint8_t*>(pcs_data->device_data) + sizeof(pcs_sampling_data_t);
    size_t count_in_bytes = deviceAllocSize - sizeof(pcs_sampling_data_t);
    size_t count_in_dwords = count_in_bytes / sizeof(uint32_t);

    if (DmaFill(device_buf_ptr, 0, count_in_dwords) !=
	 HSA_STATUS_SUCCESS) {
      debug_print("Failed to dmaFill!\n");
      return HSA_STATUS_ERROR;
    }

    pcs_data->lost_sample_count = 0;
    pcs_data->host_buffer_wrap_pos = 0;
    pcs_data->host_write_ptr = pcs_data->host_buffer;
    pcs_data->host_read_ptr = pcs_data->host_write_ptr;

    pcs_data->session = &session;

    if (UpdateTrapHandlerWithPCS(
            sampling_method == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1 ? pcs_data->device_data : nullptr,
            sampling_method == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1
                ? pcs_data->device_data
                : nullptr) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;

    session.SetThunkId(ioctlId);

    freeResources.Dismiss();

    return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::PcSamplingDestroy(pcs::PcsRuntime::PcSamplingSession& session) {
  if (PcSamplingStop(session) != HSA_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  HSAKMT_STATUS retKmt = HSAKMT_CALL(hsaKmtPcSamplingDestroy(node_id(), session.ThunkId()));
  hsa_ven_amd_pcs_method_kind_t sampling_method = session.method();

  pcs_data_t* pcs_data = nullptr;

  if (sampling_method == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1) {
    pcs_data = &pcs_hosttrap_data_;
  } else if (sampling_method == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1) {
    pcs_data = &pcs_stochastic_data_;
  } else {
    // Unsupported sampling method
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Mark session as inactive
  pcs_data->session = nullptr;

  free(pcs_data->cmd_data);
  system_deallocator()(pcs_data->old_val);
  HSA::hsa_signal_destroy(pcs_data->exec_pm4_signal);
  HSA::hsa_signal_destroy(pcs_data->device_data->done_sig0);
  HSA::hsa_signal_destroy(pcs_data->device_data->done_sig1);
  finegrain_deallocator()(pcs_data->device_data);
  system_deallocator()(pcs_data->host_buffer);

  pcs_data->device_data = NULL;
  pcs_data->host_buffer = NULL;
  pcs_data->session = NULL;

  // Update the trap handler to clear any associated device data
  UpdateTrapHandlerWithPCS(nullptr, nullptr);

  return (retKmt == HSAKMT_STATUS_SUCCESS) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t GpuAgent::PcSamplingStart(pcs::PcsRuntime::PcSamplingSession& session) {
  if (session.isActive()) return HSA_STATUS_SUCCESS;


  auto method = session.method();

  pcs_data_t* pcs_data = nullptr;
  const char* thread_name = nullptr;
  if (method == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1) {
    pcs_data = &pcs_hosttrap_data_;
    thread_name = "PcSamplingHostTrapThread";
  } else if (method == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1) {
    pcs_data = &pcs_stochastic_data_;
    thread_name = "PcSamplingStochasticThread";
  } else {
    // Unsupported sampling method
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Check if a session is already active
  if (pcs_data->session && pcs_data->session->isActive()) {
    debug_warning("Already have a PC sampling session in progress!");
    return (hsa_status_t)HSA_STATUS_ERROR_RESOURCE_BUSY;
  }

  // Assign the new session and mark it as active
  pcs_data->session = &session;
  pcs_data->session->start();

  // Creating thread data
  struct ThreadData {
    GpuAgent* agent;
    pcs_data_t* pcs_data;
    const char* thread_name;
  };

  auto* thread_data = new ThreadData{this, pcs_data, thread_name};

  // This thread will handle all PC Sampling sessions on this agent
  pcs_data->thread = os::CreateThread(
      [](void* arg) -> void {
        auto* thread_data = static_cast<ThreadData*>(arg);
        try {
          GpuAgent* agent = thread_data->agent;
          pcs_data_t* pcs_data = thread_data->pcs_data;
          const char* thread_name = thread_data->thread_name;

          agent->PcSamplingThread(*pcs_data, thread_name);
        } catch (...) {
	   fprintf(stdout, "Exception caught in PcSamplingThread. Exiting the thread!");
        }

        delete thread_data;
      },
      thread_data);

  if (!pcs_data->thread) {
    // if thread creation failed
    delete thread_data;
    throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                             "Failed to start PC Sampling thread.");
  }

  // Start the sampling session in the kernel driver
  if (HSAKMT_CALL(hsaKmtPcSamplingStart(node_id(), session.ThunkId())) == HSAKMT_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;

  debug_print("Failed to start PC sampling session with thunkId:%d\n", session.ThunkId());
  // Clean up if starting the session failed
  pcs_data->session->stop();
  os::WaitForThread(pcs_data->thread);
  os::CloseThread(pcs_data->thread);
  pcs_data->thread = nullptr;
  pcs_data->session = nullptr;

  return HSA_STATUS_ERROR;
}

hsa_status_t GpuAgent::PcSamplingStop(pcs::PcsRuntime::PcSamplingSession& session) {
  if (!session.isActive()) return HSA_STATUS_SUCCESS;

  // Stop the session
  session.stop();

  // Stop PC sampling in the kernel driver
  HSAKMT_STATUS retKmt = HSAKMT_CALL(hsaKmtPcSamplingStop(node_id(), session.ThunkId()));
  if (retKmt != HSAKMT_STATUS_SUCCESS)
    throw AMD::hsa_exception(HSA_STATUS_ERROR, "Failed to stop PC Sampling session.");

  // Determine the sampling method and corresponding data
  pcs_data_t* pcs_data = nullptr;
  auto method = session.method();

  if (method == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1) {
    pcs_data = &pcs_hosttrap_data_;
  } else if (method == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1) {
    pcs_data = &pcs_stochastic_data_;
  } else {
    // Unsupported sampling method
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  // Wake up pcs_hosttrap_thread_ if it is waiting for data
  HSA::hsa_signal_store_screlease(pcs_data->device_data->done_sig0, -1);
  HSA::hsa_signal_store_screlease(pcs_data->device_data->done_sig1, -1);

  // Wait for the thread to finish and clean up
  os::WaitForThread(pcs_data->thread);
  os::CloseThread(pcs_data->thread);
  pcs_data->thread = nullptr;
  pcs_data->session = nullptr;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuAgent::PcSamplingFlushDeviceBuffers(
    pcs::PcsRuntime::PcSamplingSession& session) {
  pcs_data_t* pcs_data = nullptr;

  if (session.method() == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1) {
    pcs_data = &pcs_hosttrap_data_;
  } else if (session.method() == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1) {
    pcs_data = &pcs_stochastic_data_;
  } else {
    // No sampling session active
    return HSA_STATUS_SUCCESS;
  }

  /*
   * Device-buffer to Host-buffer to User-Buffer copy logic
   *
   * Device-buffer = buffer written by 2nd level trap handler
   * Host-buffer = buffer inside ROCr
   * User-buffer = Session buffer size specified in PCSamplingSessionCreate
   *
   * Conditions for the buffer sizes:
   * Host buffer is at least 2 times bigger than device buffer and Host buffer
   * is also at least 2 times bigger than User-Buffer.
   *
   * Key:
   * Device-Buffer[==--][----] : Device-Buffer#1 has size 4*N, and is half-full
   *                             Device-Buffer#2 has size 4*N and is empty
   *
   * Host-Buffer[=---------] : Host Buffer has size 10*N and is filled with N.
   *
   * N will vary based on the User-buffer size, this example is to show the
   * relative sizes between each copy.
   *
   * 1. Initial state
   *    - User has created a new session with buffer size = 7*N
   *
   *    Device-Buffer[---][---]
   *    Host-Buffer[--------------] wptr=0 rptr=0 wrap_pos=0
   *    User-Buffer[-------]
   *
   *    -- Device Buffer has size 3*N
   *    -- Host-Buffer has size 14*N (2x User-Buffer)
   *    -- User-Buffer has size 7*N
   *
   * 2. Device Buffer#1 hits watermark
   *    State at beginning:
   *    Device-Buffer[===][---]
   *    Host-Buffer[--------------]
   *    User-Buffer[-------]
   *
   *    -- Copy 3*N from Device-Buffer#1 to Host-Buffer
   *    -- In the meantime, 2nd level trap handler is writing to Device-Buffer#2
   *    -- We do not have enough data to fill User-Buffer
   *
   *    State at end:
   *    Device-Buffer[---][=--]
   *    Host-Buffer[===-----------] wptr=3 rptr=0, wrap_pos=0
   *    User-Buffer[-------]
   *
   * 3. Device Buffer#2 hits watermark
   *    State at beginning:
   *    Device-Buffer[---][===]
   *    Host-Buffer[===-----------]
   *    User-Buffer[-------]
   *
   *    -- Copy 3*N from Device-Buffer#2 to Host-Buffer
   *    -- In the meantime, 2nd level trap handler is writing to Device-Buffer#1
   *    -- We do not have enough data to fill User-Buffer
   *
   *    State at end:
   *    Device-Buffer[=--][---]
   *    Host-Buffer[======--------] wptr=6 rptr=0 wrap_pos=0
   *    User-Buffer[-------]
   *
   * 4. Device Buffer#1 hits watermark
   *    State at beginning:
   *    Device-Buffer[---][===]
   *    Host-Buffer[======--------]
   *    User-Buffer[-------]
   *
   *    -- Copy 3*N from Device-Buffer#2 to Host-Buffer
   *    -- In the meantime, 2nd level trap handler is writing to Device-Buffer#1
   *
   *    Device-Buffer[=--][---]
   *    Host-Buffer[=========-----]
   *    User-Buffer[-------]
   *
   *    -- We have enough data to fill User-Buffer. Callback user data-ready to
   *    -- copy 7*N to user.
   *
   *    Device-Buffer[=--][---]
   *    Host-Buffer[-------==-----]
   *    User-Buffer[=======]
   *
   *    -- User processes User-Buffer
   *
   *    Device-Buffer[=--][---]
   *    Host-Buffer[-------==-----] wptr=9 rptr=7 wrap_pos=0
   *    User-Buffer[-------]
   *
   * 6. Device Buffer#1 hits watermark
   *    State at end:
   *    Device-Buffer[---][=--]
   *    Host-Buffer[-------=====--] wptr=12 rptr=7 wrap_pos=0
   *    User-Buffer[-------]
   *
   * 7. Device Buffer#2 hits watermark
   *    State at beginning:
   *    Device-Buffer[---][===]
   *    Host-Buffer[-------=====--] wptr=12 rptr=7 wrap_pos=0
   *    User-Buffer[-------]
   *
   *    -- We do not have enough space after wptr. The CP-DMA copy
   *    -- can only copy a contiguous range, so copy to the
   *    -- beginning of Host-Buffer and set wrap_pos
   *
   *    Device-Buffer[=--][---]
   *    Host-Buffer[===----=====--] wptr=3 rptr=7 wrap_pos=12
   *    User-Buffer[-------]
   *
   *    -- We have enough data to fill User-Buffer. Callback user data-ready to
   *    -- copy 7*N to user. We copy the tail end (index 7-12) of Host-Buffer
   *    -- before copying the beginning of Host-Buffer (index 0-2).
   *
   *    Device-Buffer[=--][---]
   *    Host-Buffer[--=-----------] wptr=3 rptr=2 wrap_pos=0
   *    User-Buffer[=======]
   *
   *     -- User processes User-Buffer
   *
   * 8. Device Buffer#1 hits watermark
   *    State at end:
   *    Device-Buffer[---][=--]
   *    Host-Buffer[--====--------] wptr=6 rptr=2 wrap_pos=0
   *    User-Buffer[-------]
   */

  uint32_t next_buffer;

  uint64_t reset_write_val;
  uint32_t to_copy = 0, copy_bytes;

  const uint32_t atomic_ex_cmd_sz = 9;
  const uint32_t wait_reg_mem_cmd_sz = 7;
  const uint32_t acquire_mem_cmd_sz = 8;
  const uint32_t dma_data_cmd_sz = 7;
  const uint32_t copy_data_cmd_sz = 6;
  const uint32_t write_data_cmd_sz = 5;
  const uint32_t pred_exec_cmd_sz = 2;

  uint64_t buf_write_val;
  uint64_t buf_written_val[2];
  size_t buf_offset;
  uint8_t* buffer[2];
  size_t buf_size;

  uint32_t& which_buffer = pcs_data->which_buffer;
  uint32_t* cmd_data = pcs_data->cmd_data;
  size_t cmd_data_sz = pcs_data->cmd_data_sz;
  uint64_t* old_val = pcs_data->old_val;
  hsa_signal_t& exec_pm4_signal = pcs_data->exec_pm4_signal;

  uint8_t* host_buffer_begin = pcs_data->host_buffer;
  size_t& host_buffer_size = pcs_data->host_buffer_size;
  uint8_t*& host_write_ptr = pcs_data->host_write_ptr;
  uint8_t* host_buffer_end = host_buffer_begin + host_buffer_size;

  buf_write_val = reinterpret_cast<uint64_t>(&pcs_data->device_data->buf_write_val);
  buf_written_val[0] = reinterpret_cast<uint64_t>(&pcs_data->device_data->buf_written_val0);
  buf_written_val[1] = reinterpret_cast<uint64_t>(&pcs_data->device_data->buf_written_val1);
  buf_size = pcs_data->device_data->buf_size;

  buf_offset =
      offsetof(pcs_sampling_data_t, reserved1) + sizeof(((pcs_sampling_data_t*)0)->reserved1);

  buffer[0] = reinterpret_cast<uint8_t*>(pcs_data->device_data) + buf_offset;
  buffer[1] = buffer[0] + buf_size * session.sample_size();

  next_buffer = (which_buffer + 1) % 2;
  reset_write_val = (uint64_t)next_buffer << 63;

  unsigned int i = 0;
  if (properties_.NumXcc > 1) i+= pred_exec_cmd_sz;
  memset(cmd_data, 0, cmd_data_sz);

  /*
   * ATOMIC_MEM, perform atomic_exchange
   * We use a double-buffer mechanism so that trap handlers calls are writing to one buffer while
   * hsa-runtime is copying data from the other buffer.
   *
   * 1. Atomically swap buffers on the device. Future trap handler calls will put their data into
   *    next_buffer.
   * 2. Return a 64-bit packed value to ROCr; the upper bit is the old buffer and can be ignored.
   *    The lower 63 bits are how many trap handler entrances happened before the atomic swap
   *    i.e., what value to wait for in buf_written_val to know all previous trap entries were
   *    done.
   */

  cmd_data[i++] = PM4_HDR(PM4_HDR_IT_OPCODE_ATOMIC_MEM, atomic_ex_cmd_sz, isa_->GetMajorVersion());
  cmd_data[i++] = PM4_ATOMIC_MEM_DW1_ATOMIC(PM4_ATOMIC_MEM_GL2_OP_ATOMIC_SWAP_RTN_64);
  cmd_data[i++] = PM4_ATOMIC_MEM_DW2_ADDR_LO(buf_write_val);
  cmd_data[i++] = PM4_ATOMIC_MEM_DW3_ADDR_HI((buf_write_val) >> 32);
  cmd_data[i++] = PM4_ATOMIC_MEM_DW4_SRC_DATA_LO((uint64_t)reset_write_val);
  cmd_data[i++] = PM4_ATOMIC_MEM_DW5_SRC_DATA_HI(((uint64_t)reset_write_val) >> 32);
  i += 3;
  /* copy data */
  cmd_data[i++] = PM4_HDR(PM4_HDR_IT_OPCODE_COPY_DATA, copy_data_cmd_sz, isa_->GetMajorVersion());
  cmd_data[i++] =
      PM4_COPY_DATA_DW1(PM4_COPY_DATA_SRC_SEL_ATOMIC_RETURN_DATA | PM4_COPY_DATA_DST_SEL_TC_12 |
                        PM4_COPY_DATA_COUNT_SEL | PM4_COPY_DATA_WR_CONFIRM);
  i += 2;
  cmd_data[i++] = PM4_COPY_DATA_DW4_DST_ADDR_LO((uint64_t)old_val);
  cmd_data[i++] = PM4_COPY_DATA_DW5_DST_ADDR_HI(((uint64_t)old_val) >> 32);

  if (properties_.NumXcc > 1) {
    cmd_data[0] =
      PM4_HDR(PM4_HDR_IT_OPCODE_PRED_EXEC, pred_exec_cmd_sz, isa_->GetMajorVersion());
    cmd_data[1] =
      PM4_PRED_EXEC_DW2_EXEC_COUNT(i - pred_exec_cmd_sz) | PM4_PRED_EXEC_DW2_VIRTUALXCCID_SELECT(0x1);
  }

  HSA::hsa_signal_store_screlease(exec_pm4_signal, 1);

  queues_[QueuePCSampling]->ExecutePM4(
      cmd_data, i * sizeof(uint32_t), HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_SYSTEM, &exec_pm4_signal);
  do {
    hsa_signal_value_t val = HSA::hsa_signal_wait_scacquire(
        exec_pm4_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    if (val == -1) return HSA_STATUS_SUCCESS;
    if (val == 0) break;
  } while (true);

  *old_val &= (ULLONG_MAX >> 1);
  /* If the number of entries in old_val is larger than buf_size, then there was a buffer overflow
   * and the 2nd level trap handler code will skip recording samples, causing lost samples
   */
  if (*old_val > buf_size) {
    pcs_data->lost_sample_count = *old_val - buf_size;
    *old_val = buf_size;
  }

  to_copy = *old_val * session.sample_size();

  /* Make sure there is enough space after host_write_ptr */
  if (host_write_ptr + to_copy >= host_buffer_end) {
    // Need to wrap around
    pcs_data->host_buffer_wrap_pos = host_write_ptr;
    host_write_ptr = host_buffer_begin;
  }

  i = 0;
  if (properties_.NumXcc > 1) i+= pred_exec_cmd_sz;
  memset(cmd_data, 0, cmd_data_sz);

  /*
   * Do the WAIT_REG_MEM, DMA_DATA(s) and WRITE_DATA
   *
   * 1. Wait for all trap handlers have finished writing values to this buffer by waiting for
   *    buf_written_val to equal to old_val.
   * 2. Copy the values out of buffer to the host buffers.
   * 3. Reset buf_written_val so that we start writing to beginning of this buffer on the next
   *    buffer swap.
   */

  /* WAIT_REG_MEM, wait on buf_written_val */
  cmd_data[i++] =
      PM4_HDR(PM4_HDR_IT_OPCODE_WAIT_REG_MEM, wait_reg_mem_cmd_sz, isa_->GetMajorVersion());
  cmd_data[i++] = PM4_WAIT_REG_MEM_DW1(PM4_WAIT_REG_MEM_FUNCTION_EQUAL_TO_REFERENCE |
                                       PM4_WAIT_REG_MEM_MEM_SPACE_MEMORY_SPACE |
                                       PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM);
  cmd_data[i++] = PM4_WAIT_REG_MEM_DW2_MEM_POLL_ADDR_LO(buf_written_val[which_buffer]);
  cmd_data[i++] = PM4_WAIT_REG_MEM_DW3_MEM_POLL_ADDR_HI((buf_written_val[which_buffer]) >> 32);
  cmd_data[i++] = PM4_WAIT_REG_MEM_DW4_REFERENCE(*old_val);
  cmd_data[i++] = 0xFFFFFFFF;
  cmd_data[i++] = PM4_WAIT_REG_MEM_DW6(PM4_WAIT_REG_MEM_POLL_INTERVAL(4) |
                                       PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE);

  // For GFX1200 and GFX1201 only - add an ACQUIRE_MEM packet to flush L2 cache before DMA.
  // This ensures that any data written by the trap handler is visible to the DMA engine.
  if ((isa_->GetMajorVersion() == 12) && (isa_->GetMinorVersion() == 0)) {
    cmd_data[i++] =
        PM4_HDR(PM4_HDR_IT_OPCODE_ACQUIRE_MEM, acquire_mem_cmd_sz, isa_->GetMajorVersion());
    cmd_data[i++] = 0;                                // DW1: COHER_CNTL
    cmd_data[i++] = 0;                                // DW2: COHER_SIZE
    cmd_data[i++] = 0;                                // DW3: COHER_SIZE_HI
    cmd_data[i++] = 0;                                // DW4: COHER_BASE_LO
    cmd_data[i++] = 0;                                // DW5: COHER_BASE_HI
    cmd_data[i++] = 4;                                // DW6: POLL_INTERVAL
    cmd_data[i++] = PM4_ACQUIRE_MEM_GCR_CNTL_GL2_WB;  // DW7: GCR_CNTL (GL2_WB=1, RANGE=ALL)
  }

  uint8_t* buffer_temp = buffer[which_buffer];

  for (copy_bytes = std::min(to_copy, (uint32_t)CP_DMA_DATA_TRANSFER_CNT_MAX); 0 < to_copy;
       to_copy -= copy_bytes) {

    /* DMA_DATA PACKETS, copy buffer using CPDMA */
    cmd_data[i++] = PM4_HDR(PM4_HDR_IT_OPCODE_DMA_DATA, dma_data_cmd_sz, isa_->GetMajorVersion());
    cmd_data[i++] = PM4_DMA_DATA_DW1(PM4_DMA_DATA_DST_SEL_DST_ADDR_USING_L2 |
                                     PM4_DMA_DATA_SRC_SEL_SRC_ADDR_USING_L2);
    cmd_data[i++] = PM4_DMA_DATA_DW2_SRC_ADDR_LO((uint64_t)buffer_temp);
    cmd_data[i++] = PM4_DMA_DATA_DW3_SRC_ADDR_HI(((uint64_t)buffer_temp) >> 32);
    cmd_data[i++] = PM4_DMA_DATA_DW4_DST_ADDR_LO((uint64_t)host_write_ptr);
    cmd_data[i++] = PM4_DMA_DATA_DW5_DST_ADDR_HI(((uint64_t)host_write_ptr) >> 32);
    if (copy_bytes >= to_copy) {
      copy_bytes = to_copy;
      cmd_data[i++] =
          PM4_DMA_DATA_DW6(PM4_DMA_DATA_BYTE_COUNT(copy_bytes) | PM4_DMA_DATA_DIS_WC_LAST);
    } else {
      cmd_data[i++] = PM4_DMA_DATA_DW6(PM4_DMA_DATA_BYTE_COUNT(copy_bytes) | PM4_DMA_DATA_DIS_WC);
    }
    buffer_temp += copy_bytes;
    host_write_ptr += copy_bytes;
  }

  /* WRITE_DATA, Reset buf_written_val */
  cmd_data[i++] = PM4_HDR(PM4_HDR_IT_OPCODE_WRITE_DATA, write_data_cmd_sz, isa_->GetMajorVersion());
  cmd_data[i++] = PM4_WRITE_DATA_DW1(PM4_WRITE_DATA_DST_SEL_TC_L2 |
                                     PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  cmd_data[i++] = PM4_WRITE_DATA_DW2_DST_MEM_ADDR_LO(buf_written_val[which_buffer]);
  cmd_data[i++] = PM4_WRITE_DATA_DW3_DST_MEM_ADDR_HI((buf_written_val[which_buffer]) >> 32);
  cmd_data[i++] = PM4_WRITE_DATA_DW4_DATA(0);

  if (properties_.NumXcc > 1) {
    cmd_data[0] =
      PM4_HDR(PM4_HDR_IT_OPCODE_PRED_EXEC, pred_exec_cmd_sz, isa_->GetMajorVersion());
    cmd_data[1] =
      PM4_PRED_EXEC_DW2_EXEC_COUNT(i - pred_exec_cmd_sz) | PM4_PRED_EXEC_DW2_VIRTUALXCCID_SELECT(0x1);
  }

  HSA::hsa_signal_store_screlease(exec_pm4_signal, 1);
  queues_[QueuePCSampling]->ExecutePM4(cmd_data, i * sizeof(uint32_t), HSA_FENCE_SCOPE_NONE,
                                       HSA_FENCE_SCOPE_SYSTEM, &exec_pm4_signal);
  do {
    hsa_signal_value_t val = HSA::hsa_signal_wait_scacquire(
        exec_pm4_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    if (val == -1) return HSA_STATUS_SUCCESS;
    if (val == 0) break;
  } while (true);

  // save the position of next buffer
  which_buffer = next_buffer;

  return HSA_STATUS_SUCCESS;
}

void GpuAgent::PcSamplingThread(pcs_data_t& pcs_data, const char* thread_name) {
  // TODO: Implement lost sample count
  // TODO: Implement latency

  try {
    pcs::PcsRuntime::PcSamplingSession& session = *pcs_data.session;
    uint32_t& which_buffer = pcs_data.which_buffer;

    uint8_t* host_buffer_begin = pcs_data.host_buffer;
    uint8_t* host_buffer_end = pcs_data.host_buffer + pcs_data.host_buffer_size;

    hsa_signal_t done_sig[] = {pcs_data.device_data->done_sig0, pcs_data.device_data->done_sig1};

    while (pcs_data.session->isActive()) {
      // Wait for the signal to process the buffer
      do {
        hsa_signal_value_t val = HSA::hsa_signal_wait_scacquire(
            done_sig[which_buffer], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        if (val == -1) goto thread_exit;
        if (val == 0) break;
      } while (true);
      HSA::hsa_signal_store_screlease(done_sig[which_buffer], 1);

      // Lock buffer to ensure thread-safe access
      std::lock_guard<std::mutex> lock(pcs_data.host_buffer_mutex);
      // Flush device buffers
      if (PcSamplingFlushDeviceBuffers(session) != HSA_STATUS_SUCCESS)
	    goto thread_exit;

      size_t bytes_before_wrap;
      size_t bytes_after_wrap;

      assert(pcs_data.host_read_ptr >= host_buffer_begin && pcs_data.host_read_ptr < host_buffer_end);
      assert(pcs_data.host_write_ptr >= host_buffer_begin && pcs_data.host_write_ptr < host_buffer_end);
      assert(pcs_data.host_buffer_wrap_pos ? (pcs_data.host_read_ptr > pcs_data.host_write_ptr)
                                           : (pcs_data.host_read_ptr <= pcs_data.host_write_ptr));

      if (pcs_data.host_buffer_wrap_pos) {
        assert(pcs_data.host_buffer_wrap_pos <= host_buffer_end &&
               pcs_data.host_buffer_wrap_pos > host_buffer_begin);
        assert(pcs_data.host_read_ptr <= pcs_data.host_buffer_wrap_pos);

        // Wrapped around
        bytes_before_wrap = pcs_data.host_buffer_wrap_pos - pcs_data.host_read_ptr;
        bytes_after_wrap = pcs_data.host_write_ptr - host_buffer_begin;

        while (bytes_before_wrap >= session.buffer_size()) {
          session.HandleSampleData(pcs_data.host_read_ptr, session.buffer_size(), nullptr, 0,
                                   pcs_data.lost_sample_count);
          pcs_data.host_read_ptr += session.buffer_size();
          bytes_before_wrap = pcs_data.host_buffer_wrap_pos - pcs_data.host_read_ptr;
          pcs_data.lost_sample_count = 0;
        }

        if (bytes_before_wrap + bytes_after_wrap >= session.buffer_size()) {
          session.HandleSampleData(pcs_data.host_read_ptr, bytes_before_wrap, host_buffer_begin,
                                   (session.buffer_size() - bytes_before_wrap), 0);
          pcs_data.host_read_ptr = host_buffer_begin + (session.buffer_size() - bytes_before_wrap);
          bytes_before_wrap = 0;
          pcs_data.host_buffer_wrap_pos = 0;
          bytes_after_wrap = pcs_data.host_write_ptr - pcs_data.host_read_ptr;
          pcs_data.lost_sample_count = 0;
        }

        while (bytes_after_wrap >= session.buffer_size()) {
          session.HandleSampleData(pcs_data.host_read_ptr, session.buffer_size(), nullptr, 0,
                                   pcs_data.lost_sample_count);
          pcs_data.host_read_ptr += session.buffer_size();
          bytes_before_wrap = 0;
          bytes_after_wrap = pcs_data.host_write_ptr - pcs_data.host_read_ptr;
          pcs_data.lost_sample_count = 0;
        }
      } else {
        // Handle non-wrapped buffer
        bytes_before_wrap = pcs_data.host_write_ptr - pcs_data.host_read_ptr;

        while (bytes_before_wrap >= session.buffer_size()) {
          assert(pcs_data.host_read_ptr >= host_buffer_begin &&
                 pcs_data.host_read_ptr + session.buffer_size() <= host_buffer_end);
          session.HandleSampleData(pcs_data.host_read_ptr, session.buffer_size(), nullptr, 0,
                                   pcs_data.lost_sample_count);
          pcs_data.host_read_ptr += session.buffer_size();
          bytes_before_wrap = pcs_data.host_write_ptr - pcs_data.host_read_ptr;
          pcs_data.lost_sample_count = 0;
        }
      }
    }
thread_exit:
  debug_print("%s::Exiting\n", thread_name);
} catch (const std::exception& e) {
  debug_print("Exception in %s: %s\n", thread_name, e.what());
} catch (...) {
  debug_print("Unknown exception in %s\n", thread_name);
}
}

hsa_status_t GpuAgent::PcSamplingFlush(pcs::PcsRuntime::PcSamplingSession& session) {
  pcs_data_t* pcs_data = nullptr;

  if (session.method() == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1) {
    pcs_data = &pcs_hosttrap_data_;
  } else if (session.method() == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1) {
    pcs_data = &pcs_stochastic_data_;
  } else {
    return HSA_STATUS_SUCCESS;  // Unsupported sampling method
  }

  uint8_t* host_buffer_begin = pcs_data->host_buffer;
  uint8_t* host_buffer_end = pcs_data->host_buffer + pcs_data->host_buffer_size;

  size_t bytes_before_wrap;
  size_t bytes_after_wrap;

  std::lock_guard<std::mutex> lock(pcs_data->host_buffer_mutex);
  // Flush device buffers
  if (PcSamplingFlushDeviceBuffers(session) != HSA_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  assert(pcs_data->host_read_ptr >= host_buffer_begin && pcs_data->host_read_ptr < host_buffer_end);
  assert(pcs_data->host_write_ptr >= host_buffer_begin &&
         pcs_data->host_write_ptr < host_buffer_end);
  assert(pcs_data->host_buffer_wrap_pos ? (pcs_data->host_read_ptr > pcs_data->host_write_ptr)
                                        : (pcs_data->host_read_ptr <= pcs_data->host_write_ptr));

  if (pcs_data->host_buffer_wrap_pos) {
    assert(pcs_data->host_buffer_wrap_pos <= host_buffer_end &&
           pcs_data->host_buffer_wrap_pos > host_buffer_begin);
    assert(pcs_data->host_read_ptr <= pcs_data->host_buffer_wrap_pos);

    // Handle wrapped-around buffer
    bytes_before_wrap = pcs_data->host_buffer_wrap_pos - pcs_data->host_read_ptr;
    bytes_after_wrap = pcs_data->host_write_ptr - host_buffer_begin;

    while (bytes_before_wrap > 0) {
      size_t bytes_to_copy = std::min(bytes_before_wrap, session.buffer_size());

      session.HandleSampleData(pcs_data->host_read_ptr, bytes_to_copy, nullptr, 0,
                               pcs_data->lost_sample_count);
      pcs_data->host_read_ptr += bytes_to_copy;
      bytes_before_wrap = pcs_data->host_buffer_wrap_pos - pcs_data->host_read_ptr;
      pcs_data->lost_sample_count = 0;
    }

    assert(pcs_data->host_read_ptr == pcs_data->host_buffer_wrap_pos);
    pcs_data->host_buffer_wrap_pos = 0;
    pcs_data->host_read_ptr = host_buffer_begin;

    while (bytes_after_wrap > 0) {
      size_t bytes_to_copy = std::min(bytes_after_wrap, session.buffer_size());

      session.HandleSampleData(pcs_data->host_read_ptr, bytes_to_copy, nullptr, 0,
                               pcs_data->lost_sample_count);
      pcs_data->host_read_ptr += bytes_to_copy;
      bytes_after_wrap = pcs_data->host_write_ptr - pcs_data->host_read_ptr;
      pcs_data->lost_sample_count = 0;
    }
  } else {
    bytes_before_wrap = pcs_data->host_write_ptr - pcs_data->host_read_ptr;

    while (bytes_before_wrap > 0) {
      size_t bytes_to_copy = std::min(bytes_before_wrap, session.buffer_size());
      assert(pcs_data->host_read_ptr >= host_buffer_begin &&
             pcs_data->host_read_ptr + bytes_to_copy <= host_buffer_end);

      session.HandleSampleData(pcs_data->host_read_ptr, bytes_to_copy, nullptr, 0,
                               pcs_data->lost_sample_count);
      pcs_data->host_read_ptr += bytes_to_copy;
      bytes_before_wrap = pcs_data->host_write_ptr - pcs_data->host_read_ptr;
      pcs_data->lost_sample_count = 0;
    }
  }
  return HSA_STATUS_SUCCESS;
}

}  // namespace amd
}  // namespace rocr
