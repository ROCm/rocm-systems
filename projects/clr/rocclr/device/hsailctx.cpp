/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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

#if defined(WITH_COMPILER_LIB)
#include "os/os.hpp"
#include "utils/flags.hpp"
#include "hsailctx.hpp"

namespace amd {
std::once_flag Hsail::initialized;
HsailEntryPoints Hsail::cep_;
bool Hsail::is_ready_ = false;

bool Hsail::LoadLib() {
#if defined(HSAIL_DYN_DLL)
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "Loading HSAIL library.");
  static constexpr const char* HsailLibName =
      LP64_SWITCH(WINDOWS_SWITCH("amdhsail32.dll", "libamdhsail32.so"),
                  WINDOWS_SWITCH("amdhsail64.dll", "libamdhsail64.so"));
  cep_.handle = Os::loadLibrary(HsailLibName);
  if (nullptr == cep_.handle) {
    return false;
  }
#endif
  GET_HSAIL_SYMBOL(aclCompilerInit)
  GET_HSAIL_SYMBOL(aclCompilerFini)
  GET_HSAIL_SYMBOL(aclCompilerVersion)
  GET_HSAIL_SYMBOL(aclVersionSize)
  GET_HSAIL_SYMBOL(aclGetErrorString)
  GET_HSAIL_SYMBOL(aclGetArchInfo)
  GET_HSAIL_SYMBOL(aclGetDeviceInfo)
  GET_HSAIL_SYMBOL(aclGetTargetInfo)
  GET_HSAIL_SYMBOL(aclGetTargetInfoFromChipID)
  GET_HSAIL_SYMBOL(aclGetArchitecture)
  GET_HSAIL_SYMBOL(aclGetChipOptions)
  GET_HSAIL_SYMBOL(aclGetFamily)
  GET_HSAIL_SYMBOL(aclGetChip)
  GET_HSAIL_SYMBOL(aclBinaryInit)
  GET_HSAIL_SYMBOL(aclBinaryFini)
  GET_HSAIL_SYMBOL(aclReadFromFile)
  GET_HSAIL_SYMBOL(aclReadFromMem)
  GET_HSAIL_SYMBOL(aclWriteToFile)
  GET_HSAIL_SYMBOL(aclWriteToMem)
  GET_HSAIL_SYMBOL(aclCreateFromBinary)
  GET_HSAIL_SYMBOL(aclBinaryVersion)
  GET_HSAIL_SYMBOL(aclInsertSection)
  GET_HSAIL_SYMBOL(aclInsertSymbol)
  GET_HSAIL_SYMBOL(aclExtractSection)
  GET_HSAIL_SYMBOL(aclExtractSymbol)
  GET_HSAIL_SYMBOL(aclRemoveSection)
  GET_HSAIL_SYMBOL(aclRemoveSymbol)
  GET_HSAIL_SYMBOL(aclQueryInfo)
  GET_HSAIL_SYMBOL(aclDbgAddArgument)
  GET_HSAIL_SYMBOL(aclDbgRemoveArgument)
  GET_HSAIL_SYMBOL(aclCompile)
  GET_HSAIL_SYMBOL(aclLink)
  GET_HSAIL_SYMBOL(aclGetCompilerLog)
  GET_HSAIL_SYMBOL(aclRetrieveType)
  GET_HSAIL_SYMBOL(aclSetType)
  GET_HSAIL_SYMBOL(aclConvertType)
  GET_HSAIL_SYMBOL(aclDisassemble)
  GET_HSAIL_SYMBOL(aclGetDeviceBinary)
  GET_HSAIL_SYMBOL(aclValidateBinaryImage)
  GET_HSAIL_SYMBOL(aclJITObjectImageCreate)
  GET_HSAIL_SYMBOL(aclJITObjectImageCopy)
  GET_HSAIL_SYMBOL(aclJITObjectImageDestroy)
  GET_HSAIL_SYMBOL(aclJITObjectImageFinalize)
  GET_HSAIL_SYMBOL(aclJITObjectImageSize)
  GET_HSAIL_SYMBOL(aclJITObjectImageData)
  GET_HSAIL_SYMBOL(aclJITObjectImageGetGlobalsSize)
  GET_HSAIL_SYMBOL(aclJITObjectImageIterateSymbols)
  GET_HSAIL_SYMBOL(aclDumpBinary)
  GET_HSAIL_SYMBOL(aclGetKstatsSI)
  GET_HSAIL_SYMBOL(aclInsertKernelStatistics)
  GET_HSAIL_SYMBOL(aclFreeMem)
  is_ready_ = true;
  return true;
}

}  // namespace amd
#endif
