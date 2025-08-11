////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AMD_HSA_CODE_UTIL_HPP_
#define AMD_HSA_CODE_UTIL_HPP_

#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#ifdef _WIN32
#include <malloc.h>
#else // _WIN32
#include <cstdlib>
#endif // _WIN32
#include "inc/amd_hsa_kernel_code.h"
#include "inc/amd_hsa_elf.h"
#include "inc/hsa.h"
#include "inc/hsa_ext_finalize.h"

#define hsa_error(e) static_cast<hsa_status_t>(e)

#define release_assert(e)                                                      \
  if (!(e)) {                                                                  \
    std::cerr << __FILE__ << ":";                                              \
    std::cerr << __LINE__ << ":";                                              \
    std::cerr << " Assertion `" << #e << "' failed." << std::endl;             \
    std::abort();                                                              \
  }                                                                            \

namespace rocr {
namespace amd {
namespace hsa {

std::string HsaSymbolKindToString(hsa_symbol_kind_t kind);
std::string HsaSymbolLinkageToString(hsa_symbol_linkage_t linkage);
std::string HsaVariableAllocationToString(hsa_variable_allocation_t allocation);
std::string HsaVariableSegmentToString(hsa_variable_segment_t segment);
std::string HsaProfileToString(hsa_profile_t profile);
std::string HsaMachineModelToString(hsa_machine_model_t model);
std::string HsaFloatRoundingModeToString(hsa_default_float_rounding_mode_t mode);
std::string AmdMachineKindToString(amd_machine_kind16_t machine);
std::string AmdFloatRoundModeToString(amd_float_round_mode_t round_mode);
std::string AmdFloatDenormModeToString(amd_float_denorm_mode_t denorm_mode);
std::string AmdSystemVgprWorkitemIdToString(amd_system_vgpr_workitem_id_t system_vgpr_workitem_id);
std::string AmdElementByteSizeToString(amd_element_byte_size_t element_byte_size);
std::string AmdExceptionKindToString(amd_exception_kind16_t exceptions);
std::string AmdPowerTwoToString(amd_powertwo8_t p);
amdgpu_hsa_elf_segment_t AmdHsaElfSectionSegment(amdgpu_hsa_elf_section_t sec);
bool IsAmdHsaElfSectionROData(amdgpu_hsa_elf_section_t sec);
std::string AmdHsaElfSegmentToString(amdgpu_hsa_elf_segment_t seg);
std::string AmdPTLoadToString(uint64_t type);

void PrintAmdKernelCode(std::ostream& out, const amd_kernel_code_t *akc);
void PrintAmdComputePgmRsrcOne(std::ostream& out, amd_compute_pgm_rsrc_one32_t compute_pgm_rsrc1);
void PrintAmdComputePgmRsrcTwo(std::ostream& out, amd_compute_pgm_rsrc_two32_t compute_pgm_rsrc2);
void PrintAmdKernelCodeProperties(std::ostream& out, amd_kernel_code_properties32_t kernel_code_properties);
void PrintAmdControlDirectives(std::ostream& out, const amd_control_directives_t &control_directives);

namespace code_options {
  // Space between options (not at the beginning).
  std::ostream& space(std::ostream& out);

  // Control directive option without value.
  struct control_directive {
    const char *name;
    control_directive(const char* name_) : name(name_) { }
  };
  std::ostream& operator<<(std::ostream& out, const control_directive& d);

  // Exceptions mask string.
  struct exceptions_mask {
    uint16_t mask;
    exceptions_mask(uint16_t mask_) : mask(mask_) { }
  };
  std::ostream& operator<<(std::ostream& out, const exceptions_mask& e);

  // Control directives options.
  struct control_directives {
    const hsa_ext_control_directives_t& d;
    control_directives(const hsa_ext_control_directives_t& d_) : d(d_) { }
  };
  std::ostream& operator<<(std::ostream& out, const control_directives& cd);
}

const char* hsaerr2str(hsa_status_t status);
bool ReadFileIntoBuffer(const std::string& filename, std::vector<char>& buffer);

// Create new empty temporary file that will be deleted when closed.
int OpenTempFile(const char* prefix);
void CloseTempFile(int fd);

// Helper comment types for isa disassembler
enum DumpIsaCommentType  {
  COMMENT_AMD_KERNEL_CODE_T_BEGIN = 1,
  COMMENT_AMD_KERNEL_CODE_T_END,
  COMMENT_KERNEL_ISA_BEGIN,
};

// Callbacks to create helper comments for isa disassembler
const char * CommentTopCallBack(void *ctx, int type);
const char * CommentRightCallBack(void *ctx, int type);

// Parse disassembler instruction line to find offset
uint32_t ParseInstructionOffset(const std::string& instruction);

// Trim whitespaces from start of string
void ltrim(std::string &str);


// Helper function that allocates an aligned memory.
inline void*
alignedMalloc(size_t size, size_t alignment)
{
#if defined(_WIN32)
  return ::_aligned_malloc(size, alignment);
#else
  void * ptr = NULL;
  alignment = (std::max)(alignment, sizeof(void*));
  if (0 == ::posix_memalign(&ptr, alignment, size)) {
    return ptr;
  }
  return NULL;
#endif
}

// Helper function that frees an aligned memory.
inline void
alignedFree(void *ptr)
{
#if defined(_WIN32)
  ::_aligned_free(ptr);
#else
  free(ptr);
#endif
}

inline uint64_t alignUp(uint64_t num, uint64_t align)
{
  assert(align);
  assert((align & (align - 1)) == 0);
  return (num + align - 1) & ~(align - 1);
}

inline uint32_t alignUp(uint32_t num, uint32_t align)
{
  assert(align);
  assert((align & (align - 1)) == 0);
  return (num + align - 1) & ~(align - 1);
}

std::string DumpFileName(const std::string& dir, const char* prefix, const char* ext, unsigned n, unsigned i = 0);

}   //  namespace hsa
}   //  namespace amd
}   //  namespace rocr

#endif // AMD_HSA_CODE_UTIL_HPP_
