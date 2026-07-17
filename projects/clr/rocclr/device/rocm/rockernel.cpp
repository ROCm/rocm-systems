/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rockernel.hpp"

#include <algorithm>

namespace amd::roc {

bool Kernel::init() { return GetAttrCodePropMetadata(); }

bool Kernel::postLoad() {
  // Set the kernel symbol name and size/alignment based on the kernel metadata
  // NOTE: kernel name is used to get the kernel code handle in V2,
  //       but kernel symbol name is used in V3
  if (codeObjectVer() == 2) {
    symbolName_ = name();
  }
  kernargSegmentAlignment_ = amd::alignUp(std::max(kernargSegmentAlignment_, 128u),
                                          device().info().globalMemCacheLineSize_);

  // Set the workgroup information for the kernel
  workGroupInfo_.availableLDSSize_ = device().info().localMemSizePerCU_;
  assert(workGroupInfo_.availableLDSSize_ > 0);

  // Get the kernel code handle
  hsa_status_t hsaStatus;
  hsa_executable_symbol_t symbol;
  hsa_agent_t agent = program()->rocDevice().getBackendDevice();
  hsaStatus = Hsa::executable_get_symbol_by_name(program()->hsaExecutable(), symbolName().c_str(),
                                                &agent, &symbol);
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             "Cannot Get Symbol : %s, failed with hsa_status: %d \n", symbolName().c_str(),
                      hsaStatus);
    return false;
  }

  hsaStatus = Hsa::executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                             &kernelCodeHandle_);
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             " Cannot Get Symbol Info: %s, failed with hsa_status: %d \n ",
                      symbolName().c_str(), hsaStatus);
    return false;
  }

  // Resolve the metadata kernel descriptor for prefetching at load time,
  // so dispatch doesn't need to call loaderQueryHostAddress per packet.
  const void* host_address = nullptr;
  Device::loaderQueryHostAddress(reinterpret_cast<void*>(kernelCodeHandle_), &host_address);
  if (host_address != nullptr) {
    constexpr size_t KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET = 16;
    auto* descriptor = reinterpret_cast<const hsa_amd_metadata_kernel_descriptor_t*>(
        reinterpret_cast<const uint8_t*>(host_address) + KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET);
    metadataKernelDescriptor_ = descriptor;
    metadata_preload_length_ = descriptor->kernarg_preload.length;
    metadata_preload_offset_ = descriptor->kernarg_preload.offset;
  }

  hsaStatus = Hsa::executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK, &kernelHasDynamicCallStack_);
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             " Cannot Get Dynamic callstack info, failed with hsa_status: %d \n ",
                      hsaStatus);
    return false;
  }

  uint32_t hsaPrivateSegmentSize = 0;
  hsaStatus = Hsa::executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &hsaPrivateSegmentSize);
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             " Cannot Get Private Segment Size for %s, failed with hsa_status: %d \n ",
             symbolName().c_str(), hsaStatus);
    return false;
  }

  uint32_t hsaGroupSegmentSize = 0;
  hsaStatus = Hsa::executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &hsaGroupSegmentSize);
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             " Cannot Get Group Segment Size for %s, failed with hsa_status: %d \n ",
             symbolName().c_str(), hsaStatus);
    return false;
  }
  const bool hotSwapSource = program()->isHotSwapSource();
  if (hsaGroupSegmentSize > workGroupInfo_.availableLDSSize_ && !hotSwapSource) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             " Group Segment Size %u exceeds available LDS size %zu for %s \n ",
             hsaGroupSegmentSize, workGroupInfo_.availableLDSSize_, symbolName().c_str());
    return false;
  }
  if (!RuntimeHandle().empty()) {
    hsa_executable_symbol_t kernelSymbol;
    uint32_t variable_size = 0;
    uint64_t variable_address = 0;

    // Only kernels that could be enqueued by another kernel has the RuntimeHandle metadata. The
    // RuntimeHandle metadata is a string that represents a variable from which the library code can
    // retrieve the kernel code object handle of such a kernel. The address of the variable and the
    // kernel code object handle are known only after the hsa executable is loaded. The below code
    // copies the kernel code object handle to the address of the variable.
    hsaStatus = Hsa::executable_get_symbol_by_name(program()->hsaExecutable(),
                                                  RuntimeHandle().c_str(), &agent, &kernelSymbol);
    if (hsaStatus != HSA_STATUS_SUCCESS) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
               "Cannot get Kernel Symbol by name: %s, failed with hsa_status: %d \n",
                        RuntimeHandle().c_str(), hsaStatus);
      return false;
    }

    hsaStatus = Hsa::executable_symbol_get_info(
        kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &variable_size);
    if (hsaStatus != HSA_STATUS_SUCCESS) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
          "[ROC][Kernel] Cannot get Kernel Symbol Info, failed with hsa_status: %d \n", hsaStatus);
      return false;
    }
    if (variable_size != sizeof(struct RuntimeHandle)) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
              "[ROC][Kernel] Runtime handle symbol %s has unexpected size %u, expected %zu \n",
              RuntimeHandle().c_str(), variable_size, sizeof(struct RuntimeHandle));
      return false;
    }

    hsaStatus = Hsa::executable_symbol_get_info(
        kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &variable_address);
    if (hsaStatus != HSA_STATUS_SUCCESS) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
               "[ROC][Kernel] Cannot get Kernel Address, failed with hsa_status: %d \n",
                        hsaStatus);
      return false;
    }

    const struct RuntimeHandle runtime_handle = {
        kernelCodeHandle_, hsaPrivateSegmentSize, hsaGroupSegmentSize};
    hsaStatus = Hsa::memory_copy(reinterpret_cast<void*>(variable_address), &runtime_handle,
                                 sizeof(runtime_handle));

    if (hsaStatus != HSA_STATUS_SUCCESS) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
               "[ROC][Kernel] HSA Memory copy failed, failed with hsa_status: %d \n",
                        hsaStatus);
      return false;
    }
  }

  SetWorkitemPrivateSegmentByteSize(hsaPrivateSegmentSize);
  SetWorkgroupGroupSegmentByteSize(hsaGroupSegmentSize);

  // This can be set in code object and the value might be different than what HSA reports
  // For example on Navi GPUs someone using -mwavefrontsize64
  // We set the value to HSA if the value is uninitialized
  uint32_t wavefront_size = workGroupInfo_.wavefrontPerSIMD_;
  if (wavefront_size == 0 &&
      Hsa::agent_get_info(program()->rocDevice().getBackendDevice(), HSA_AGENT_INFO_WAVEFRONT_SIZE,
                          &wavefront_size) != HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             "[ROC][Kernel] Cannot get Wavefront Size, failed with hsa_status: %d \n",
                      hsaStatus);
    return false;
  }
  assert(wavefront_size > 0);

  workGroupInfo_.availableVGPRs_ = device().info().availableVGPRs_;
  workGroupInfo_.availableSGPRs_ = device().info().availableSGPRs_;
  workGroupInfo_.privateMemSize_ = hsaPrivateSegmentSize;
  // Keep the source packet's fixed LDS size intact for ROCr's lazy
  // source-to-target delta calculation, but do not let pre-translation
  // metadata inflate CLR's dynamic LDS capacity.
  const uint64_t accountedGroupSegmentSize =
      hotSwapSource ? std::min<uint64_t>(hsaGroupSegmentSize, workGroupInfo_.availableLDSSize_)
                    : hsaGroupSegmentSize;
  workGroupInfo_.localMemSize_ = accountedGroupSegmentSize;
  workGroupInfo_.usedLDSSize_ = accountedGroupSegmentSize;
  workGroupInfo_.preferredSizeMultiple_ = wavefront_size;
  workGroupInfo_.usedStackSize_ = kernelHasDynamicCallStack_;
  workGroupInfo_.wavefrontPerSIMD_ =
      program()->rocDevice().info().maxWorkItemSizes_[0] / wavefront_size;
  workGroupInfo_.constMemSize_ = 0;
  workGroupInfo_.maxDynamicSharedSizeBytes_ =
      workGroupInfo_.availableLDSSize_ - accountedGroupSegmentSize;
  if (workGroupInfo_.size_ == 0) {
    return false;
  }

  // handle the printf metadata if any
  std::vector<std::string> printfStr;
  if (!GetPrintfStr(&printfStr)) {
    return false;
  }

  if (!printfStr.empty()) {
    InitPrintf(printfStr);
  }
  // Add kernel to the map of all kernels on the device
  program()->rocDevice().AddKernel(*this);
  return true;
}

}  // namespace amd::roc
