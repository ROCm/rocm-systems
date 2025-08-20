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

#pragma once

#include <mutex>
#if defined(WITH_COMPILER_LIB)
#include "top.hpp"
#include "acl.h"

#ifndef ACL_API_ENTRY
#if defined(_WIN32) || defined(__CYGWIN__)
#define ACL_API_ENTRY __stdcall
#else
#define ACL_API_ENTRY
#endif
#endif

namespace amd {
typedef aclCompiler*(ACL_API_ENTRY* t_aclCompilerInit)(aclCompilerOptions* opts,
                                                       acl_error* error_code);
typedef acl_error(ACL_API_ENTRY* t_aclCompilerFini)(aclCompiler* cl);
typedef aclCLVersion(ACL_API_ENTRY* t_aclCompilerVersion)(aclCompiler* cl, acl_error* error_code);
typedef uint32_t(ACL_API_ENTRY* t_aclVersionSize)(aclCLVersion num, acl_error* error_code);
typedef const char*(ACL_API_ENTRY* t_aclGetErrorString)(acl_error error_code);
typedef acl_error(ACL_API_ENTRY* t_aclGetArchInfo)(const char** arch_names, size_t* arch_size);
typedef acl_error(ACL_API_ENTRY* t_aclGetDeviceInfo)(const char* arch, const char** names,
                                                     size_t* device_size);
typedef aclTargetInfo(ACL_API_ENTRY* t_aclGetTargetInfo)(const char* arch, const char* device,
                                                         acl_error* error_code);
typedef aclTargetInfo(ACL_API_ENTRY* t_aclGetTargetInfoFromChipID)(const char* arch,
                                                                   const uint32_t chip_id,
                                                                   acl_error* error_code);
typedef const char*(ACL_API_ENTRY* t_aclGetArchitecture)(const aclTargetInfo& target);
typedef const uint64_t(ACL_API_ENTRY* t_aclGetChipOptions)(const aclTargetInfo& target);
typedef const char*(ACL_API_ENTRY* t_aclGetFamily)(const aclTargetInfo& target);
typedef const char*(ACL_API_ENTRY* t_aclGetChip)(const aclTargetInfo& target);
typedef aclBinary*(ACL_API_ENTRY* t_aclBinaryInit)(size_t struct_version,
                                                   const aclTargetInfo* target,
                                                   const aclBinaryOptions* options,
                                                   acl_error* error_code);
typedef acl_error(ACL_API_ENTRY* t_aclBinaryFini)(aclBinary* bin);
typedef aclBinary*(ACL_API_ENTRY* t_aclReadFromFile)(const char* str, acl_error* error_code);
typedef aclBinary*(ACL_API_ENTRY* t_aclReadFromMem)(const void* mem, size_t size,
                                                    acl_error* error_code);
typedef acl_error(ACL_API_ENTRY* t_aclWriteToFile)(aclBinary* bin, const char* str);
typedef acl_error(ACL_API_ENTRY* t_aclWriteToMem)(aclBinary* bin, void** mem, size_t* size);
typedef aclBinary*(ACL_API_ENTRY* t_aclCreateFromBinary)(const aclBinary* binary,
                                                         aclBIFVersion version);
typedef aclBIFVersion(ACL_API_ENTRY* t_aclBinaryVersion)(const aclBinary* binary);
typedef acl_error(ACL_API_ENTRY* t_aclInsertSection)(aclCompiler* cl, aclBinary* binary,
                                                     const void* data, size_t data_size,
                                                     aclSections id);
typedef acl_error(ACL_API_ENTRY* t_aclInsertSymbol)(aclCompiler* cl, aclBinary* binary,
                                                    const void* data, size_t data_size,
                                                    aclSections id, const char* symbol);
typedef const void*(ACL_API_ENTRY* t_aclExtractSection)(aclCompiler* cl, const aclBinary* binary,
                                                        size_t* size, aclSections id,
                                                        acl_error* error_code);
typedef const void*(ACL_API_ENTRY* t_aclExtractSymbol)(aclCompiler* cl, const aclBinary* binary,
                                                       size_t* size, aclSections id,
                                                       const char* symbol, acl_error* error_code);
typedef acl_error(ACL_API_ENTRY* t_aclRemoveSection)(aclCompiler* cl, aclBinary* binary,
                                                     aclSections id);
typedef acl_error(ACL_API_ENTRY* t_aclRemoveSymbol)(aclCompiler* cl, aclBinary* binary,
                                                    aclSections id, const char* symbol);
typedef acl_error(ACL_API_ENTRY* t_aclQueryInfo)(aclCompiler* cl, const aclBinary* binary,
                                                 aclQueryType query, const char* kernel,
                                                 void* data_ptr, size_t* ptr_size);
typedef acl_error(ACL_API_ENTRY* t_aclDbgAddArgument)(aclCompiler* cl, aclBinary* binary,
                                                      const char* kernel, const char* name,
                                                      bool byVal);
typedef acl_error(ACL_API_ENTRY* t_aclDbgRemoveArgument)(aclCompiler* cl, aclBinary* binary,
                                                         const char* kernel, const char* name);
typedef acl_error(ACL_API_ENTRY* t_aclCompile)(aclCompiler* cl, aclBinary* bin, const char* options,
                                               aclType from, aclType to,
                                               aclLogFunction compile_callback);
typedef acl_error(ACL_API_ENTRY* t_aclLink)(aclCompiler* cl, aclBinary* src_bin,
                                            unsigned int num_libs, aclBinary** libs,
                                            aclType link_mode, const char* options,
                                            aclLogFunction link_callback);
typedef const char*(ACL_API_ENTRY* t_aclGetCompilerLog)(aclCompiler* cl);
typedef const void*(ACL_API_ENTRY* t_aclRetrieveType)(aclCompiler* cl, const aclBinary* bin,
                                                      const char* name, size_t* data_size,
                                                      aclType type, acl_error* error_code);
typedef acl_error(ACL_API_ENTRY* t_aclSetType)(aclCompiler* cl, aclBinary* bin, const char* name,
                                               aclType type, const void* data, size_t size);
typedef acl_error(ACL_API_ENTRY* t_aclConvertType)(aclCompiler* cl, aclBinary* bin,
                                                   const char* name, aclType type);
typedef acl_error(ACL_API_ENTRY* t_aclDisassemble)(aclCompiler* cl, aclBinary* bin,
                                                   const char* kernel,
                                                   aclLogFunction disasm_callback);
typedef const void*(ACL_API_ENTRY* t_aclGetDeviceBinary)(aclCompiler* cl, const aclBinary* bin,
                                                         const char* kernel, size_t* size,
                                                         acl_error* error_code);
typedef bool(ACL_API_ENTRY* t_aclValidateBinaryImage)(const void* binary, size_t length,
                                                      unsigned type);
typedef aclJITObjectImage(ACL_API_ENTRY* t_aclJITObjectImageCreate)(aclCompiler* cl,
                                                                    const void* buffer,
                                                                    size_t length, aclBinary* bin,
                                                                    acl_error* error_code);
typedef aclJITObjectImage(ACL_API_ENTRY* t_aclJITObjectImageCopy)(aclCompiler* cl,
                                                                  const void* buffer, size_t length,
                                                                  acl_error* error_code);
typedef acl_error(ACL_API_ENTRY* t_aclJITObjectImageDestroy)(aclCompiler* cl,
                                                             aclJITObjectImage buffer);
typedef acl_error(ACL_API_ENTRY* t_aclJITObjectImageFinalize)(aclCompiler* cl,
                                                              aclJITObjectImage image);
typedef size_t(ACL_API_ENTRY* t_aclJITObjectImageSize)(aclCompiler* cl, aclJITObjectImage image,
                                                       acl_error* error_code);
typedef const char*(ACL_API_ENTRY* t_aclJITObjectImageData)(aclCompiler* cl,
                                                            aclJITObjectImage image,
                                                            acl_error* error_code);
typedef size_t(ACL_API_ENTRY* t_aclJITObjectImageGetGlobalsSize)(aclCompiler* cl,
                                                                 aclJITObjectImage image,
                                                                 acl_error* error_code);
typedef acl_error(ACL_API_ENTRY* t_aclJITObjectImageIterateSymbols)(aclCompiler* cl,
                                                                    aclJITObjectImage image,
                                                                    aclJITSymbolCallback callback,
                                                                    void* data);
typedef void(ACL_API_ENTRY* t_aclDumpBinary)(const aclBinary* bin);
typedef void(ACL_API_ENTRY* t_aclGetKstatsSI)(const void* shader, aclKernelStats& kstats);
typedef acl_error(ACL_API_ENTRY* t_aclInsertKernelStatistics)(aclCompiler* cl, aclBinary* bin);
typedef acl_error(ACL_API_ENTRY* t_aclFreeMem)(aclBinary* bin, void* mem);

struct HsailEntryPoints {
  void* handle;
  t_aclCompilerInit aclCompilerInit;
  t_aclCompilerFini aclCompilerFini;
  t_aclCompilerVersion aclCompilerVersion;
  t_aclVersionSize aclVersionSize;
  t_aclGetErrorString aclGetErrorString;
  t_aclGetArchInfo aclGetArchInfo;
  t_aclGetDeviceInfo aclGetDeviceInfo;
  t_aclGetTargetInfo aclGetTargetInfo;
  t_aclGetTargetInfoFromChipID aclGetTargetInfoFromChipID;
  t_aclGetArchitecture aclGetArchitecture;
  t_aclGetChipOptions aclGetChipOptions;
  t_aclGetFamily aclGetFamily;
  t_aclGetChip aclGetChip;
  t_aclBinaryInit aclBinaryInit;
  t_aclBinaryFini aclBinaryFini;
  t_aclReadFromFile aclReadFromFile;
  t_aclReadFromMem aclReadFromMem;
  t_aclWriteToFile aclWriteToFile;
  t_aclWriteToMem aclWriteToMem;
  t_aclCreateFromBinary aclCreateFromBinary;
  t_aclBinaryVersion aclBinaryVersion;
  t_aclInsertSection aclInsertSection;
  t_aclInsertSymbol aclInsertSymbol;
  t_aclExtractSection aclExtractSection;
  t_aclExtractSymbol aclExtractSymbol;
  t_aclRemoveSection aclRemoveSection;
  t_aclRemoveSymbol aclRemoveSymbol;
  t_aclQueryInfo aclQueryInfo;
  t_aclDbgAddArgument aclDbgAddArgument;
  t_aclDbgRemoveArgument aclDbgRemoveArgument;
  t_aclCompile aclCompile;
  t_aclLink aclLink;
  t_aclGetCompilerLog aclGetCompilerLog;
  t_aclRetrieveType aclRetrieveType;
  t_aclSetType aclSetType;
  t_aclConvertType aclConvertType;
  t_aclDisassemble aclDisassemble;
  t_aclGetDeviceBinary aclGetDeviceBinary;
  t_aclValidateBinaryImage aclValidateBinaryImage;
  t_aclJITObjectImageCreate aclJITObjectImageCreate;
  t_aclJITObjectImageCopy aclJITObjectImageCopy;
  t_aclJITObjectImageDestroy aclJITObjectImageDestroy;
  t_aclJITObjectImageFinalize aclJITObjectImageFinalize;
  t_aclJITObjectImageSize aclJITObjectImageSize;
  t_aclJITObjectImageData aclJITObjectImageData;
  t_aclJITObjectImageGetGlobalsSize aclJITObjectImageGetGlobalsSize;
  t_aclJITObjectImageIterateSymbols aclJITObjectImageIterateSymbols;
  t_aclDumpBinary aclDumpBinary;
  t_aclGetKstatsSI aclGetKstatsSI;
  t_aclInsertKernelStatistics aclInsertKernelStatistics;
  t_aclFreeMem aclFreeMem;
};

#ifdef HSAIL_DYN_DLL
#define HSAIL_DYN(NAME) cep_.NAME
#define GET_HSAIL_SYMBOL(NAME)                                                                     \
  cep_.NAME = reinterpret_cast<t_##NAME>(Os::getSymbol(cep_.handle, #NAME));                       \
  if (nullptr == cep_.NAME) {                                                                      \
    return false;                                                                                  \
  }
#else
#define HSAIL_DYN(NAME) NAME
#define GET_HSAIL_SYMBOL(NAME)
#endif

class Hsail : public amd::AllStatic {
 public:
  static std::once_flag initialized;

  static bool LoadLib();

  static bool IsReady() { return is_ready_; }

  static aclCompiler* CompilerInit(aclCompilerOptions* opts, acl_error* error_code) {
    return HSAIL_DYN(aclCompilerInit)(opts, error_code);
  }
  static acl_error CompilerFini(aclCompiler* cl) { return HSAIL_DYN(aclCompilerFini)(cl); }
  static aclCLVersion CompilerVersion(aclCompiler* cl, acl_error* error_code) {
    return HSAIL_DYN(aclCompilerVersion)(cl, error_code);
  }
  static uint32_t VersionSize(aclCLVersion num, acl_error* error_code) {
    return HSAIL_DYN(aclVersionSize)(num, error_code);
  }
  static const char* GetErrorString(acl_error error_code) {
    return HSAIL_DYN(aclGetErrorString)(error_code);
  }
  static acl_error GetArchInfo(const char** arch_names, size_t* arch_size) {
    return HSAIL_DYN(aclGetArchInfo)(arch_names, arch_size);
  }
  static acl_error GetDeviceInfo(const char* arch, const char** names, size_t* device_size) {
    return HSAIL_DYN(aclGetDeviceInfo)(arch, names, device_size);
  }
  static aclTargetInfo GetTargetInfo(const char* arch, const char* device, acl_error* error_code) {
    return HSAIL_DYN(aclGetTargetInfo)(arch, device, error_code);
  }
  static aclTargetInfo GetTargetInfoFromChipID(const char* arch, const uint32_t chip_id,
                                               acl_error* error_code) {
    return HSAIL_DYN(aclGetTargetInfoFromChipID)(arch, chip_id, error_code);
  }
  static const char* GetArchitecture(const aclTargetInfo& target) {
    return HSAIL_DYN(aclGetArchitecture)(target);
  }
  static uint64_t GetChipOptions(const aclTargetInfo& target) {
    return HSAIL_DYN(aclGetChipOptions)(target);
  }
  static const char* GetFamily(const aclTargetInfo& target) {
    return HSAIL_DYN(aclGetFamily)(target);
  }
  static const char* GetChip(const aclTargetInfo& target) { return HSAIL_DYN(aclGetChip)(target); }
  static aclBinary* BinaryInit(size_t struct_version, const aclTargetInfo* target,
                               const aclBinaryOptions* options, acl_error* error_code) {
    return HSAIL_DYN(aclBinaryInit)(struct_version, target, options, error_code);
  }
  static acl_error BinaryFini(aclBinary* bin) { return HSAIL_DYN(aclBinaryFini)(bin); }
  static aclBinary* ReadFromFile(const char* str, acl_error* error_code) {
    return HSAIL_DYN(aclReadFromFile)(str, error_code);
  }
  static aclBinary* ReadFromMem(const void* mem, size_t size, acl_error* error_code) {
    return HSAIL_DYN(aclReadFromMem)(mem, size, error_code);
  }
  static acl_error WriteToFile(aclBinary* bin, const char* str) {
    return HSAIL_DYN(aclWriteToFile)(bin, str);
  }
  static acl_error WriteToMem(aclBinary* bin, void** mem, size_t* size) {
    return HSAIL_DYN(aclWriteToMem)(bin, mem, size);
  }
  static aclBinary* CreateFromBinary(const aclBinary* binary, aclBIFVersion version) {
    return HSAIL_DYN(aclCreateFromBinary)(binary, version);
  }
  static aclBIFVersion BinaryVersion(const aclBinary* binary) {
    return HSAIL_DYN(aclBinaryVersion)(binary);
  }
  static acl_error InsertSection(aclCompiler* cl, aclBinary* binary, const void* data,
                                 size_t data_size, aclSections id) {
    return HSAIL_DYN(aclInsertSection)(cl, binary, data, data_size, id);
  }
  static const acl_error InsertSymbol(aclCompiler* cl, aclBinary* binary, const void* data,
                                      size_t data_size, aclSections id, const char* symbol) {
    return HSAIL_DYN(aclInsertSymbol)(cl, binary, data, data_size, id, symbol);
  }
  static const void* ExtractSection(aclCompiler* cl, const aclBinary* binary, size_t* size,
                                    aclSections id, acl_error* error_code) {
    return HSAIL_DYN(aclExtractSection)(cl, binary, size, id, error_code);
  }
  static const void* ExtractSymbol(aclCompiler* cl, const aclBinary* binary, size_t* size,
                                   aclSections id, const char* symbol, acl_error* error_code) {
    return HSAIL_DYN(aclExtractSymbol)(cl, binary, size, id, symbol, error_code);
  }
  static acl_error RemoveSection(aclCompiler* cl, aclBinary* binary, aclSections id) {
    return HSAIL_DYN(aclRemoveSection)(cl, binary, id);
  }
  static acl_error RemoveSymbol(aclCompiler* cl, aclBinary* binary, aclSections id,
                                const char* symbol) {
    return HSAIL_DYN(aclRemoveSymbol)(cl, binary, id, symbol);
  }
  static acl_error QueryInfo(aclCompiler* cl, const aclBinary* binary, aclQueryType query,
                             const char* kernel, void* data_ptr, size_t* ptr_size) {
    return HSAIL_DYN(aclQueryInfo)(cl, binary, query, kernel, data_ptr, ptr_size);
  }
  static acl_error DbgAddArgument(aclCompiler* cl, aclBinary* binary, const char* kernel,
                                  const char* name, bool byVal) {
    return HSAIL_DYN(aclDbgAddArgument)(cl, binary, kernel, name, byVal);
  }
  static acl_error DbgRemoveArgument(aclCompiler* cl, aclBinary* binary, const char* kernel,
                                     const char* name) {
    return HSAIL_DYN(aclDbgRemoveArgument)(cl, binary, kernel, name);
  }
  static acl_error Compile(aclCompiler* cl, aclBinary* bin, const char* options, aclType from,
                           aclType to, aclLogFunction compile_callback) {
    return HSAIL_DYN(aclCompile)(cl, bin, options, from, to, compile_callback);
  }
  static acl_error Link(aclCompiler* cl, aclBinary* src_bin, unsigned int num_libs,
                        aclBinary** libs, aclType link_mode, const char* options,
                        aclLogFunction link_callback) {
    return HSAIL_DYN(aclLink)(cl, src_bin, num_libs, libs, link_mode, options, link_callback);
  }
  static const char* GetCompilerLog(aclCompiler* cl) { return HSAIL_DYN(aclGetCompilerLog)(cl); }
  static const void* RetrieveType(aclCompiler* cl, const aclBinary* bin, const char* name,
                                  size_t* data_size, aclType type, acl_error* error_code) {
    return HSAIL_DYN(aclRetrieveType)(cl, bin, name, data_size, type, error_code);
  }
  static acl_error SetType(aclCompiler* cl, aclBinary* bin, const char* name, aclType type,
                           const void* data, size_t size) {
    return HSAIL_DYN(aclSetType)(cl, bin, name, type, data, size);
  }
  static acl_error ConvertType(aclCompiler* cl, aclBinary* bin, const char* name, aclType type) {
    return HSAIL_DYN(aclConvertType)(cl, bin, name, type);
  }
  static acl_error Disassemble(aclCompiler* cl, aclBinary* bin, const char* kernel,
                               aclLogFunction disasm_callback) {
    return HSAIL_DYN(aclDisassemble)(cl, bin, kernel, disasm_callback);
  }
  static const void* GetDeviceBinary(aclCompiler* cl, const aclBinary* bin, const char* kernel,
                                     size_t* size, acl_error* error_code) {
    return HSAIL_DYN(aclGetDeviceBinary)(cl, bin, kernel, size, error_code);
  }
  static const bool ValidateBinaryImage(const void* binary, size_t length, unsigned type) {
#if defined(HSAIL_DYN_DLL)
    if (cep_.aclValidateBinaryImage == nullptr) {
      return false;
    }
#endif  // defined(HSAIL_DYN_DLL)
    return HSAIL_DYN(aclValidateBinaryImage)(binary, length, type);
  }
  static aclJITObjectImage JITObjectImageCreate(aclCompiler* cl, const void* buffer, size_t length,
                                                aclBinary* bin, acl_error* error_code) {
    return HSAIL_DYN(aclJITObjectImageCreate)(cl, buffer, length, bin, error_code);
  }
  static aclJITObjectImage JITObjectImageCopy(aclCompiler* cl, const void* buffer, size_t length,
                                              acl_error* error_code) {
    return HSAIL_DYN(aclJITObjectImageCopy)(cl, buffer, length, error_code);
  }
  static acl_error JITObjectImageDestroy(aclCompiler* cl, aclJITObjectImage buffer) {
    return HSAIL_DYN(aclJITObjectImageDestroy)(cl, buffer);
  }
  static acl_error JITObjectImageFinalize(aclCompiler* cl, aclJITObjectImage image) {
    return HSAIL_DYN(aclJITObjectImageFinalize)(cl, image);
  }
  static size_t JITObjectImageSize(aclCompiler* cl, aclJITObjectImage image,
                                   acl_error* error_code) {
    return HSAIL_DYN(aclJITObjectImageSize)(cl, image, error_code);
  }
  static const char* JITObjectImageData(aclCompiler* cl, aclJITObjectImage image,
                                        acl_error* error_code) {
    return HSAIL_DYN(aclJITObjectImageData)(cl, image, error_code);
  }
  static size_t JITObjectImageGetGlobalsSize(aclCompiler* cl, aclJITObjectImage image,
                                             acl_error* error_code) {
    return HSAIL_DYN(aclJITObjectImageGetGlobalsSize)(cl, image, error_code);
  }
  static acl_error JITObjectImageIterateSymbols(aclCompiler* cl, aclJITObjectImage image,
                                                aclJITSymbolCallback callback, void* data) {
    return HSAIL_DYN(aclJITObjectImageIterateSymbols)(cl, image, callback, data);
  }
  static void DumpBinary(const aclBinary* bin) { HSAIL_DYN(aclDumpBinary)(bin); }
  static void GetKstatsSI(const void* shader, aclKernelStats& kstats) {
    return HSAIL_DYN(aclGetKstatsSI)(shader, kstats);
  }
  static acl_error InsertKernelStatistics(aclCompiler* cl, aclBinary* bin) {
    return HSAIL_DYN(aclInsertKernelStatistics)(cl, bin);
  }
  static acl_error FreeMem(aclBinary* bin, void* mem) { return HSAIL_DYN(aclFreeMem)(bin, mem); }

 private:
  static HsailEntryPoints cep_;
  static bool is_ready_;
};

}  // namespace amd
#endif
