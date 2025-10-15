/* Copyright (c) 2008 - 2025 Advanced Micro Devices, Inc.

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

#include "rocprogram.hpp"

#include "utils/options.hpp"
#include "rockernel.hpp"

#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>

namespace amd::roc {

static inline const char* hsa_strerror(hsa_status_t status) {
  const char* str = nullptr;
  if (Hsa::status_string(status, &str) == HSA_STATUS_SUCCESS) {
    return str;
  }
  return "Unknown error";
}

Program::~Program() {
  // Destroy the executable.
  if (hsaExecutable_.handle != 0) {
    Hsa::executable_destroy(hsaExecutable_);
  }
  if (hsaCodeObjectReader_.handle != 0) {
    Hsa::code_object_reader_destroy(hsaCodeObjectReader_);
  }
  releaseClBinary();
}

Program::Program(roc::NullDevice& device, amd::Program& owner) : device::Program(device, owner) {
  hsaExecutable_.handle = 0;
  hsaCodeObjectReader_.handle = 0;
}

bool Program::initClBinary(char* binaryIn, size_t size) {
  // Save the original binary that isn't owned by ClBinary
  clBinary()->saveOrigBinary(binaryIn, size);

  char* bin = binaryIn;
  size_t sz = size;

  int encryptCode;

  char* decryptedBin;
  size_t decryptedSize;
  if (!clBinary()->decryptElf(binaryIn, size, &decryptedBin, &decryptedSize, &encryptCode)) {
    buildLog_ += "Decrypting ELF Failed\n";
    return false;
  }
  if (decryptedBin != nullptr) {
    // It is decrypted binary.
    bin = decryptedBin;
    sz = decryptedSize;
  }

  // Both 32-bit and 64-bit are allowed!
  if (!amd::Elf::isElfMagic(bin)) {
    // Invalid binary.
    delete[] decryptedBin;
    buildLog_ += "Elf Magic failed\n";
    return false;
  }

  clBinary()->setFlags(encryptCode);

  return clBinary()->setBinary(bin, sz, (decryptedBin != nullptr));
}


bool Program::defineGlobalVar(const char* name, void* dptr) {
  if (!device().isOnline()) {
    return false;
  }

  hsa_agent_t hsa_device = rocDevice().getBackendDevice();

  hsa_status_t status =
      Hsa::executable_agent_global_variable_define(hsaExecutable_, hsa_device, name, dptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Could not define global variable : ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
  }

  return (status == HSA_STATUS_SUCCESS);
}

bool Program::createGlobalVarObj(amd::Memory** amd_mem_obj, void** device_pptr, size_t* bytes,
                                 const char* global_name) const {
  if (!device().isOnline()) {
    return false;
  }

  hsa_status_t status = HSA_STATUS_SUCCESS;
  const roc::Device* roc_device = nullptr;
  hsa_agent_t hsa_device;
  hsa_symbol_kind_t sym_type;
  hsa_executable_symbol_t global_symbol;

  if (amd_mem_obj == nullptr) {
    buildLog_ += "amd_mem_obj is null";
    buildLog_ += "\n";
    return false;
  }

  hsa_device = rocDevice().getBackendDevice();

  /* Find HSA Symbol by name */
  status =
      Hsa::executable_get_symbol_by_name(hsaExecutable_, global_name, &hsa_device, &global_symbol);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Failed to find the Symbol by Name: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  /* Find HSA Symbol Type */
  status =
      Hsa::executable_symbol_get_info(global_symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &sym_type);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Failed to find the Symbol Type : ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  /* Make sure symbol type is VARIABLE */
  if (sym_type != HSA_SYMBOL_KIND_VARIABLE) {
    buildLog_ += "Error: Symbol is not of type VARIABLE : ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the size of the variable */
  status = Hsa::executable_symbol_get_info(global_symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE,
                                          bytes);

  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Failed to retrieve the Symbol Size : ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Handle size 0 symbols
  if (*bytes != 0) {
    // Find HSA Symbol Address
    status = Hsa::executable_symbol_get_info(
        global_symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, device_pptr);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to find the Symbol Address : ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    roc_device = &(rocDevice());
    *amd_mem_obj = new (roc_device->context())
        amd::Buffer(roc_device->context(), ROCCLR_MEM_INTERNAL_MEMORY, *bytes, *device_pptr);

    if (*amd_mem_obj == nullptr) {
      buildLog_ += "[OCL] Failed to create a mem object!";
      buildLog_ += "\n";
      return false;
    }

    if (!((*amd_mem_obj)->create(nullptr))) {
      buildLog_ += "[OCL] failed to create a svm hidden buffer!";
      buildLog_ += "\n";
      (*amd_mem_obj)->release();
      return false;
    }
  }
  return true;
}

HSAILProgram::HSAILProgram(roc::NullDevice& device, amd::Program& owner)
    : roc::Program(device, owner) {}

HSAILProgram::~HSAILProgram() {}

bool HSAILProgram::saveBinaryAndSetType(type_t type) { return true; }

bool HSAILProgram::setKernels(void* binary, size_t binSize, amd::Os::FileDesc fdesc, size_t foffset,
                              std::string uri) {
  return true;
}


LightningProgram::LightningProgram(roc::NullDevice& device, amd::Program& owner)
    : roc::Program(device, owner) {
  isLC_ = true;
  isHIP_ = (owner.language() == amd::Program::HIP);
}

bool LightningProgram::createBinary(amd::option::Options* options) {
#if defined(USE_COMGR_LIBRARY)
  if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt, type())) {
    LogError("Failed to create ELF binary image!");
    return false;
  }
#endif  // defined(USE_COMGR_LIBRARY)
  return true;
}

bool LightningProgram::saveBinaryAndSetType(type_t type, void* rawBinary, size_t size) {
#if defined(USE_COMGR_LIBRARY)
  // Write binary to memory
  if (type == TYPE_EXECUTABLE) {  // handle code object binary
    assert(rawBinary != nullptr && size != 0 && "must pass in the binary");
  } else {  // handle LLVM binary
    if (llvmBinary_.empty()) {
      buildLog_ += "ERROR: Tried to save empty LLVM binary \n";
      return false;
    }
    rawBinary = (void*)llvmBinary_.data();
    size = llvmBinary_.size();
  }
  clBinary()->saveBIFBinary((char*)rawBinary, size);

  // Set the type of binary
  setType(type);
#endif  // defined(USE_COMGR_LIBRARY)
  return true;
}

bool LightningProgram::createKernels(void* binary, size_t binSize, bool useUniformWorkGroupSize,
                                     bool internalKernel) {
  // Find the size of global variables from the binary
  if (!FindGlobalVarSize(binary, binSize)) {
    buildLog_ += "Error: Cannot Find Global Var Sizes\n";
    return false;
  }

  for (const auto& kernelMeta : kernelMetadataMap_) {
    const std::string kernelName = kernelMeta.first;
    Kernel* aKernel = new roc::Kernel(kernelName, this);
    if (!aKernel->init()) {
      return false;
    }
    if (codeObjectVer() < 5) {
      aKernel->setUniformWorkGroupSize(useUniformWorkGroupSize);
    }
    aKernel->setInternalKernelFlag(internalKernel);
    addKernel(aKernel);
  }
  return true;
}

bool LightningProgram::setKernels(void* binary, size_t binSize, amd::Os::FileDesc fdesc,
                                  size_t foffset, std::string uri) {
#if defined(USE_COMGR_LIBRARY)
  // Stop compilation if it is an offline device - HSA runtime does not
  // support ISA compiled offline
  if (!device().isOnline()) {
    return true;
  }

  hsa_agent_t agent = rocDevice().getBackendDevice();
  hsa_status_t status;

  status = Hsa::executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     nullptr, &hsaExecutable_);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Executable for AMD HSA Code Object isn't created: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Load the code object, either with file descriptor and offset
  // or binary image and binary size with URI
  // or binary image and binary size
  status = Hsa::code_object_reader_create_from_memory(binary, binSize, &hsaCodeObjectReader_);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object Reader create failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  status = Hsa::executable_load_agent_code_object(hsaExecutable_, agent, hsaCodeObjectReader_,
                                                 nullptr, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object loading failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Freeze the executable.
  status = Hsa::executable_freeze(hsaExecutable_, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Freezing the executable failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  for (auto& kit : kernels()) {
    Kernel* kernel = static_cast<Kernel*>(kit.second);
    if (!kernel->postLoad()) {
      return false;
    }
  }
#endif  // defined(USE_COMGR_LIBRARY)
  return true;
}

}  // namespace amd::roc

