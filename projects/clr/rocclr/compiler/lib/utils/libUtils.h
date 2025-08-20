/* Copyright (c) 2011 - 2021 Advanced Micro Devices, Inc.

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

#ifndef _CL_LIB_UTILS_0_8_H_
#define _CL_LIB_UTILS_0_8_H_
#include "acl.h"
#include <string>
#include <sstream>
#include <iterator>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include "library.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/options.hpp"
using namespace bif;

// Utility function to set a flag in option structure
// of the aclDevCaps.
void setFlag(aclDevCaps* elf, compDeviceCaps option);

// Utility function to flip a flag in option structure
// of the aclDevCaps.
void flipFlag(aclDevCaps* elf, compDeviceCaps option);

// Utility function to clear a flag in option structure
// of the aclDevCaps.
void clearFlag(aclDevCaps* elf, compDeviceCaps option);

// Utility function to check that a flag in option structure
// of the aclDevCaps is set.
bool checkFlag(aclDevCaps* elf, compDeviceCaps option);

// Utility function to initialize and elf device capabilities
void initElfDeviceCaps(aclBinary* elf);

// Append the string to the aclCompiler log string.
void appendLogToCL(aclCompiler* cl, const std::string& logStr);

const char* getDeviceName(const aclTargetInfo& target);

// Select the correct library from the target information.
amd::LibrarySelector getLibraryType(const aclTargetInfo* target);

// get family_enum from the target information.
unsigned getFamilyEnum(const aclTargetInfo* target);

// get chip_enum from the target information.
unsigned getChipEnum(const aclTargetInfo* target);

// get isa type name (compute capability) from the target information.
const std::string& getIsaTypeName(const aclTargetInfo* target);

// get isa type  (compute capability) from the target information.
int getIsaType(const aclTargetInfo* target);

// get Feature String for target.
std::string getFeatureString(const aclTargetInfo& target, amd::option::Options* OptionsObj);

// Create a copy of an ELF and duplicate all sections/symbols
aclBinary* createELFCopy(aclBinary* src);

// Create a BIF2.1 elf from a BIF 2.0 elf
aclBinary* convertBIF20ToBIF21(aclBinary* src);

// Create a BIF3.0 elf from a BIF 2.0 elf
aclBinary* convertBIF20ToBIF30(aclBinary* src);

// Create a BIF3.1 elf from a BIF 2.0 elf
aclBinary* convertBIF20ToBIF31(aclBinary* src);

// Create a BIF2.0 elf from a BIF 2.1 elf
aclBinary* convertBIF21ToBIF20(aclBinary* src);

// Create a BIF3.0 elf from a BIF 2.1 elf
aclBinary* convertBIF21ToBIF30(aclBinary* src);

// Create a BIF3.1 elf from a BIF 2.1 elf
aclBinary* convertBIF21ToBIF31(aclBinary* src);

// Create a BIF2.0 elf from a BIF 3.0 elf
aclBinary* convertBIF30ToBIF20(aclBinary* src);

// Create a BIF2.1 elf from a BIF 3.0 elf
aclBinary* convertBIF30ToBIF21(aclBinary* src);

// Create a BIF3.1 elf from a BIF 3.0 elf
aclBinary* convertBIF30ToBIF31(aclBinary* src);

// Create a BIF2.0 elf from a BIF 3.1 elf
aclBinary* convertBIF31ToBIF20(aclBinary* src);

// Create a BIF2.1 elf from a BIF 3.1 elf
aclBinary* convertBIF31ToBIF21(aclBinary* src);

// Create a BIF3.0 elf from a BIF 3.1 elf
aclBinary* convertBIF31ToBIF30(aclBinary* src);

// get a pointer to the aclBIF irrespective of the
// binary version.
aclBIF* aclutGetBIF(aclBinary*);

// Get a pointer to the aclOptions irrespective of
// the binary version.
aclOptions* aclutGetOptions(aclBinary*);

// Get a pointer to the aclBinaryOptions struct
// irrespective of the binary version.
aclBinaryOptions* aclutGetBinOpts(aclBinary*);

// Get a pointer to the target info struct
// irrespective of the binary version.
aclTargetInfo* aclutGetTargetInfo(aclBinary*);

// Get a pointer to the device caps
// irrespective of the binary version.
aclDevCaps* aclutGetCaps(aclBinary*);

// Copy two binary option structures irrespective
// of the binary version and uses defaults when
// things don't match up.
void aclutCopyBinOpts(aclBinaryOptions* dst, const aclBinaryOptions* src, bool is64bit);

// Retrieve kernel statistics from binary
// and insert to elf as symbol
acl_error aclutInsertKernelStatistics(aclCompiler*, aclBinary*);

// Returns target chip name.
std::string aclutGetCodegenName(const aclTargetInfo& tgtInfo);

// Helper function that returns the
// allocation function from the binary.
AllocFunc aclutAlloc(const aclBinary* bin);

// Helper function that returns the
// de-allocation function from the binary.
FreeFunc aclutFree(const aclBinary* bin);


// Helper function that returns the
// allocation function from the compiler.
AllocFunc aclutAlloc(const aclCompiler* bin);

// Helper function that returns the
// de-allocation function from the compiler.
FreeFunc aclutFree(const aclCompiler* bin);

// Helper function that returns the
// allocation function from the compiler options.
AllocFunc aclutAlloc(const aclCompilerOptions* bin);

// Helper function that returns the
// de-allocation function from the compiler options.
FreeFunc aclutFree(const aclCompilerOptions* bin);

inline std::vector<std::string> splitSpaceSeparatedString(char* str) {
  std::string s(str);
  std::stringstream ss(s);
  std::istream_iterator<std::string> beg(ss), end;
  std::vector<std::string> vec(beg, end);
  return vec;
}

// Helper function that returns OpenCL mangled kernel name.
inline std::string aclutOpenclMangledKernelName(const std::string& kernel_name) {
  const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclKernel);
  assert(sym && "symbol not found");
  return std::string("&") + sym->str[PRE] + kernel_name + sym->str[POST];
}

// Helper function that returns OpenCL mangled kernel metadata symbol name.
inline std::string aclutOpenclMangledKernelMetadataName(const std::string& kernel_name) {
  const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclMeta);
  assert(sym && "symbol not found");
  return sym->str[PRE] + aclutOpenclMangledKernelName(kernel_name) + sym->str[POST];
}

#ifdef WITH_TARGET_HSAIL
// Helper function that updates metadata for all the kernels in binary;
// the updated attribute is the number of hidden kernel arguments.
inline acl_error aclutUpdateMetadataWithHiddenKernargsNum(aclCompiler* cl, aclBinary* bin,
                                                          uint32_t num) {
  if (num == MAX_HIDDEN_KERNARGS_NUM) {
    return ACL_SUCCESS;
  }
  const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclMeta);
  assert(sym && "symbol not found");
  aclSections secID = sym->sections[0];
  size_t kernelNamesSize = 0;
  acl_error error_code = aclQueryInfo(cl, bin, RT_KERNEL_NAMES, NULL, NULL, &kernelNamesSize);
  if (error_code != ACL_SUCCESS) {
    return error_code;
  }
  char* kernelNames = new char[kernelNamesSize];
  error_code = aclQueryInfo(cl, bin, RT_KERNEL_NAMES, NULL, kernelNames, &kernelNamesSize);
  if (error_code != ACL_SUCCESS) {
    delete[] kernelNames;
    return error_code;
  }
  std::vector<std::string> vKernels = splitSpaceSeparatedString(kernelNames);
  delete[] kernelNames;
  size_t roSize = 0;
  for (auto it = vKernels.begin(); it != vKernels.end(); ++it) {
    std::string symbol = aclutOpenclMangledKernelMetadataName(*it);
    void* roSec =
        const_cast<void*>(aclExtractSymbol(cl, bin, &roSize, secID, symbol.c_str(), &error_code));
    if (error_code != ACL_SUCCESS) {
      return error_code;
    }
    if (!roSec || roSize == 0) {
      error_code = ACL_ELF_ERROR;
      return error_code;
    }
    aclMetadata* md = reinterpret_cast<aclMetadata*>(roSec);
    md->numHiddenKernelArgs = num;
    error_code = aclRemoveSymbol(cl, bin, secID, symbol.c_str());
    if (error_code != ACL_SUCCESS) {
      return error_code;
    }
    error_code = aclInsertSymbol(cl, bin, md, roSize, secID, symbol.c_str());
    if (error_code != ACL_SUCCESS) {
      return error_code;
    }
  }
  return error_code;
}
#endif

struct _target_mappings_rec;
typedef _target_mappings_rec TargetMapping;

// Returns the TargetMapping for the specific target device.
const TargetMapping& getTargetMapping(const aclTargetInfo& target);

inline bool is64BitTarget(const aclTargetInfo& target) {
  return (target.arch_id == aclX64 || target.arch_id == aclAMDIL64 || target.arch_id == aclHSAIL64);
}

inline bool isCpuTarget(const aclTargetInfo& target) {
  return (target.arch_id == aclX64 || target.arch_id == aclX86);
}

inline bool isGpuTarget(const aclTargetInfo& target) {
  return (target.arch_id == aclAMDIL || target.arch_id == aclAMDIL64 ||
          target.arch_id == aclHSAIL || target.arch_id == aclHSAIL64);
}

inline bool isAMDILTarget(const aclTargetInfo& target) {
  return (target.arch_id == aclAMDIL || target.arch_id == aclAMDIL64);
}

inline bool isHSAILTarget(const aclTargetInfo& target) {
  return (target.arch_id == aclHSAIL || target.arch_id == aclHSAIL64);
}

const std::string& getLegacyLibName();

inline bool isValidTarget(const aclTargetInfo& target) {
  return (target.arch_id && target.chip_id);
}

bool isChipSupported(const aclTargetInfo& target);

enum scId {
  SC_AMDIL = 0,
  SC_HSAIL = 0,
  SC_LAST,
};

// Helper function that allocates an aligned memory.
inline void* alignedMalloc(size_t size, size_t alignment) {
#if defined(_WIN32)
  return ::_aligned_malloc(size, alignment);
#else
  void* ptr = NULL;
  if (0 == ::posix_memalign(&ptr, alignment, size)) {
    return ptr;
  }
  return NULL;
#endif
}

// Helper function that frees an aligned memory.
inline void alignedFree(void* ptr) {
#if defined(_WIN32)
  ::_aligned_free(ptr);
#else
  free(ptr);
#endif
}

#if defined(_WIN32)
inline void convertLongAbsFilePathIfNeeded(std::string& filename) {
  if (filename.empty()) {
    return;
  }
  std::wstring ws(filename.begin(), filename.end());
  wchar_t abs_path[_MAX_ENV];
  _wfullpath(abs_path, ws.c_str(), _MAX_ENV);
  std::wstring ws_abs = std::wstring(abs_path);
  if (ws_abs.size() >= _MAX_PATH) {
    std::string s(ws_abs.begin(), ws_abs.end());
    filename = "\\\\?\\" + s;
  }
}
#endif

inline char* readFile(std::string source_filename, size_t& size) {
#if defined(_WIN32)
  convertLongAbsFilePathIfNeeded(source_filename);
#endif
  FILE* fp = ::fopen(source_filename.c_str(), "rb");
  unsigned int length;
  size_t offset = 0;
  char* ptr;
  if (!fp) {
    return NULL;
  }
  // obtain file size
  ::fseek(fp, 0, SEEK_END);
  length = ::ftell(fp);
  ::rewind(fp);
  ptr = reinterpret_cast<char*>(::malloc(offset + length + 1));
  if (length != fread(&ptr[offset], 1, length, fp)) {
    ::free(ptr);
    ::fclose(fp);
    return NULL;
  }
  ptr[offset + length] = '\0';
  size = offset + length;
  ::fclose(fp);
  return ptr;
}

inline bool writeFile(std::string source_filename, const char* source, size_t size) {
#if defined(_WIN32)
  convertLongAbsFilePathIfNeeded(source_filename);
#endif
  FILE* fp = ::fopen(source_filename.c_str(), "wb");
  if (!fp) {
    return EXIT_FAILURE;
  }
  if (!::fwrite(source, size, 1, fp)) {
    ::fclose(fp);
    return EXIT_FAILURE;
  }
  ::fclose(fp);
  return EXIT_SUCCESS;
}

#if !defined(BCMAG)
#define BCMAG "BC"
#define SBCMAG 2
#endif
// Helper predicate returns true if p starts with bit code signature.
// TODO: Move it into Compiler Lib back in new 1_0 API
inline static bool isBcMagic(const char* p) {
  if (p == NULL || strncmp(p, BCMAG, SBCMAG) != 0) {
    return false;
  }
  return true;
}

void dump(aclBinary* bin);

#endif  // _CL_LIB_UTILS_0_8_H_
