/* Copyright (c) 2012 - 2021 Advanced Micro Devices, Inc.

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

#ifndef _ACL_ENUMS_0_8_H_
#define _ACL_ENUMS_0_8_H_

typedef enum _acl_error_enum_0_8 {
  ACL_SUCCESS = 0,
  ACL_ERROR = 1,
  ACL_INVALID_ARG = 2,
  ACL_OUT_OF_MEM = 3,
  ACL_SYS_ERROR = 4,
  ACL_UNSUPPORTED = 5,
  ACL_ELF_ERROR = 6,
  ACL_INVALID_FILE = 7,
  ACL_INVALID_COMPILER = 8,
  ACL_INVALID_TARGET = 9,
  ACL_INVALID_BINARY = 10,
  ACL_INVALID_OPTION = 11,
  ACL_INVALID_TYPE = 12,
  ACL_INVALID_SECTION = 13,
  ACL_INVALID_SYMBOL = 14,
  ACL_INVALID_QUERY = 15,
  ACL_FRONTEND_FAILURE = 16,
  ACL_INVALID_BITCODE = 17,
  ACL_LINKER_ERROR = 18,
  ACL_OPTIMIZER_ERROR = 19,
  ACL_CODEGEN_ERROR = 20,
  ACL_ISAGEN_ERROR = 21,
  ACL_INVALID_SOURCE = 22,
  ACL_LIBRARY_ERROR = 23,
  ACL_INVALID_SPIR = 24,
  ACL_LWVERIFY_FAIL = 25,
  ACL_HWVERIFY_FAIL = 26,
  ACL_SPIRV_LOAD_FAIL = 27,
  ACL_SPIRV_SAVE_FAIL = 28,
  ACL_LAST_ERROR = 29
} acl_error_0_8;

typedef enum _comp_device_caps_enum_0_8 {
  capError = 0,
  capFMA = 1,
  capImageSupport = 2,
  capSaveSOURCE = 3,  // input source
  capSaveLLVMIR = 4,  // output LLVMIR from frontend
  capSaveCG = 5,      // output from LLVM-BE
  capSaveEXE = 6,     // output executable
  capSaveAMDIL = 7,   // Save per-kernel AMDIL
  capSaveHSAIL = 8,   // Save per-kernel HSAIL
  capEncrypted = 9,
  capSaveDISASM = 10,
  capSaveAS = 11,
  capSaveSPIR = 12,
  capDumpLast = 13
} compDeviceCaps_0_8;

typedef enum _comp_opt_settings_enum_0_8 {
  optO0 = 0,  // No optimization setting.
  optO1 = 1,
  optO2 = 2,
  optO3 = 3,
  optO4 = 4,
  optOs = 5,
  optError = 6,  // Invalid optimization set
  optLast = 7
} compOptSettings_0_8;

#define FLAG_SHIFT_VALUE 5
#define FLAG_MASK_VALUE ((1 << capDumpLast) - 1)
#define FLAG_BITLOC(A) (1 << ((A) & FLAG_MASK_VALUE))
#define FLAG_ARRAY_SIZE 4

//! An enumeration that defines the possible valid device types that
// can be compiled for.
typedef enum _acl_dev_type_enum_0_8 {
  aclError = 0,    // aclDevType of 0 is an error.
  aclX86 = 1,      // Targeting a 32bit X86 CPU device.
  aclAMDIL = 2,    // Targeting an AMDIL GPU device.
  aclHSAIL = 3,    // Targeting an HSAIL GPU device.
  aclX64 = 4,      // Targeting a 64bit X86 CPU device.
  aclHSAIL64 = 5,  // Targeting a 64bit HSAIL GPU device.
  aclAMDIL64 = 6,  // Targeting a 64bit AMDIL GPU device
  aclLast = 7
} aclDevType_0_8;

//! Enum that represents the versions of the compiler
typedef enum _acl_cl_version_enum_0_8 {
  ACL_VERSION_ERROR = 0,
  ACL_VERSION_0_7 = 1,
  ACL_VERSION_0_8 = 2,
  ACL_VERSION_0_8_1 = 3,
  ACL_VERSION_0_9 = 4,
  ACL_VERSION_1_0 = 5,
  ACL_VERSION_LAST = 6
} aclCLVersion_0_8;

//! Enum of the various aclTypes that are supported
typedef enum _acl_type_enum_0_8 {
  ACL_TYPE_DEFAULT = 0,
  ACL_TYPE_OPENCL = 1,
  ACL_TYPE_LLVMIR_TEXT = 2,
  ACL_TYPE_LLVMIR_BINARY = 3,
  ACL_TYPE_SPIR_TEXT = 4,
  ACL_TYPE_SPIR_BINARY = 5,
  ACL_TYPE_AMDIL_TEXT = 6,
  ACL_TYPE_AMDIL_BINARY = 7,
  ACL_TYPE_HSAIL_TEXT = 8,
  ACL_TYPE_HSAIL_BINARY = 9,
  ACL_TYPE_X86_TEXT = 10,
  ACL_TYPE_X86_BINARY = 11,
  ACL_TYPE_CG = 12,
  ACL_TYPE_SOURCE = 13,
  ACL_TYPE_ISA = 14,
  ACL_TYPE_HEADER = 15,
  ACL_TYPE_RSLLVMIR_BINARY = 16,
  ACL_TYPE_SPIRV_BINARY = 17,
  ACL_TYPE_ASM_TEXT = 18,
  ACL_TYPE_LAST = 19
} aclType_0_8;

//! Enum of the various loader types that are supported.
typedef enum _acl_loader_type_enum_0_8 {
  ACL_LOADER_COMPLIB = 0,
  ACL_LOADER_FRONTEND = 1,
  ACL_LOADER_LINKER = 2,
  ACL_LOADER_OPTIMIZER = 3,
  ACL_LOADER_CODEGEN = 4,
  ACL_LOADER_BACKEND = 5,
  ACL_LOADER_SC = 6,
  ACL_LOADER_LAST = 7
} aclLoaderType_0_8;

// Enumeration for the various acl versions
typedef enum _bif_version_enum_0_8 {
  aclBIFVersionError = 0,                 // Error
  aclBIFVersion20 = 1,                    // Version 2.0 of the OpenCL BIF
  aclBIFVersion21 = 2,                    // Version 2.1 of the OpenCL BIF
  aclBIFVersion30 = 3,                    // Version 3.0 of the OpenCL BIF
  aclBIFVersion31 = 4,                    // Version 3.1 of the OpenCL BIF
  aclBIFVersionLatest = aclBIFVersion31,  // Most recent version of the BIF
  aclBIFVersionCAL = 5,
  aclBIFVersionLast = 6
} aclBIFVersion_0_8;

// Enumeration for the various platform types
typedef enum _bif_platform_enum_0_8 {
  aclPlatformCAL = 0,  // For BIF 2.0 backward compatibility
  aclPlatformCPU = 1,  // For BIF 2.0 backward compatibility
  aclPlatformCompLib = 2,
  aclPlatformLast = 3
} aclPlatform_0_8;

// Enumeration for the various bif sections
typedef enum _bif_sections_enum_0_8 {
  aclLLVMIR = 0,
  aclSOURCE = 1,
  aclILTEXT = 2,  // For BIF 2.0 backward compatibility
  aclASTEXT = 3,  // For BIF 2.0 backward compatibility
  aclCAL = 4,     // For BIF 2.0 backward compatibility
  aclDLL = 5,     // For BIF 2.0 backward compatibility
  aclSTRTAB = 6,
  aclSYMTAB = 7,
  aclRODATA = 8,
  aclSHSTRTAB = 9,
  aclNOTES = 10,
  aclCOMMENT = 11,
  aclILDEBUG = 12,  // For BIF 2.0 backward compatibility
  aclDEBUG_INFO = 13,
  aclDEBUG_ABBREV = 14,
  aclDEBUG_LINE = 15,
  aclDEBUG_PUBNAMES = 16,
  aclDEBUG_PUBTYPES = 17,
  aclDEBUG_LOC = 18,
  aclDEBUG_ARANGES = 19,
  aclDEBUG_RANGES = 20,
  aclDEBUG_MACINFO = 21,
  aclDEBUG_STR = 22,
  aclDEBUG_FRAME = 23,
  aclJITBINARY = 24,  // For BIF 2.0 backward compatibility
  aclCODEGEN = 25,
  aclTEXT = 26,
  aclINTERNAL = 27,
  aclSPIR = 28,
  aclHEADER = 29,
  aclBRIG = 30,
  aclBRIGxxx1 = 31,
  aclBRIGxxx2 = 32,
  aclBRIGxxx3 = 33,
  aclHSADEBUG = 34,
  aclKSTATS = 35,  // For storing kernel statistics
  aclSPIRV = 36,
  aclLAST = 37
} aclSections_0_8;

//! An enumeration that defines what are valid queries for aclQueryInfo.
typedef enum _rt_query_types_enum_0_8 {
  RT_ABI_VERSION = 0,
  RT_DEVICE_NAME = 1,
  RT_MEM_SIZES = 2,
  RT_GPU_FUNC_CAPS = 3,
  RT_GPU_FUNC_ID = 4,
  RT_GPU_DEFAULT_ID = 5,
  RT_WORK_GROUP_SIZE = 6,
  RT_WORK_REGION_SIZE = 7,
  RT_ARGUMENT_ARRAY = 8,
  RT_GPU_PRINTF_ARRAY = 9,
  RT_CPU_BARRIER_NAMES = 10,
  RT_DEVICE_ENQUEUE = 11,
  RT_KERNEL_INDEX = 12,
  RT_KERNEL_NAME = 13,
  RT_KERNEL_NAMES = 14,
  RT_CONTAINS_LLVMIR = 15,
  RT_CONTAINS_OPTIONS = 16,
  RT_CONTAINS_BRIG = 17,
  RT_CONTAINS_HSAIL = 18,
  RT_CONTAINS_ISA = 19,
  RT_CONTAINS_LOADER_MAP = 20,
  RT_CONTAINS_SPIR = 21,
  RT_NUM_KERNEL_HIDDEN_ARGS = 22,
  RT_CONTAINS_SPIRV = 23,
  RT_WAVES_PER_SIMD_HINT = 24,
  RT_WORK_GROUP_SIZE_HINT = 25,
  RT_VEC_TYPE_HINT = 26,
  RT_LAST_TYPE = 27
} aclQueryType_0_8;

//! An enumeration for the various GPU capabilities
typedef enum _rt_gpu_caps_enum_0_8 {
  RT_COMPILER_WRITE = 1 << 0,
  RT_DATA_SECTION = 1 << 1,
  RT_WGS = 1 << 2,
  RT_LIMIT_WGS = 1 << 3,
  RT_PACKED_REGS = 1 << 4,
  RT_64BIT_ABI = 1 << 5,
  RT_PRINTF = 1 << 6,
  RT_ARENA_UAV = 1 << 7,
  RT_LRP_MEM = 1 << 8,  // Local/Region/Private Memory
  RT_INDEX_TEMPS = 1 << 9,
  RT_WRS = 1 << 10,
  RT_GWS = 1 << 11,
  RT_SWGWS = 1 << 12,
  RT_GPU_CAPS_MASK = 0xFFF
} aclGPUCaps_0_8;

//! An enumeration for the various CPU capabilities.
typedef enum _rt_cpu_caps_enum_0_8 {
  RT_KERNEL_BARRIER = 1 << 0,
  RT_PROGRAM_BARRIER = 1 << 1,
  RT_CPU_CAPS_MASK = 0x3
} aclCPUCaps_0_8;

//! An enumeration that maps Resource type to index values
typedef enum _rt_gpu_resource_enum_0_8 {
  RT_RES_UAV = 0,  // UAV resources
  RT_RES_PRI = 1,  // Private resources
  RT_RES_LDS = 2,  // LDS resources
  RT_RES_GDS = 3,  // GDS resources
  RT_RES_CON = 4,  // Constant resources
  RT_RES_LAST = 5
} aclGPUResource_0_8;

//! An enumeration that maps memory types to index values
typedef enum _rt_gpu_mem_sizes_enum_0_8 {
  RT_MEM_HW_LOCAL = 0,
  RT_MEM_SW_LOCAL = 1,
  RT_MEM_HW_PRIVATE = 2,
  RT_MEM_SW_PRIVATE = 3,
  RT_MEM_HW_REGION = 4,
  RT_MEM_SW_REGION = 5,
  RT_MEM_LAST = 6
} aclGPUMemSizes_0_8;

// Enumerations for the various argument types.
typedef enum _acl_arg_type_enum_0_8 {
  ARG_TYPE_ERROR = 0,
  ARG_TYPE_SAMPLER = 1,
  ARG_TYPE_IMAGE = 2,
  ARG_TYPE_COUNTER = 3,
  ARG_TYPE_VALUE = 4,
  ARG_TYPE_POINTER = 5,
  ARG_TYPE_SEMAPHORE = 6,
  ARG_TYPE_QUEUE = 7,  // enum for device enqueue
  ARG_TYPE_LAST = 8
} aclArgType_0_8;

// Enumerations of the valid data types for pass by value and
// pass by pointer kernel arguments.
typedef enum _acl_data_type_enum_0_8 {
  DATATYPE_ERROR = 0,
  DATATYPE_i1 = 1,
  DATATYPE_i8 = 2,
  DATATYPE_i16 = 3,
  DATATYPE_i32 = 4,
  DATATYPE_i64 = 5,
  DATATYPE_u8 = 6,
  DATATYPE_u16 = 7,
  DATATYPE_u32 = 8,
  DATATYPE_u64 = 9,
  DATATYPE_f16 = 10,
  DATATYPE_f32 = 11,
  DATATYPE_f64 = 12,
  DATATYPE_f80 = 13,
  DATATYPE_f128 = 14,
  DATATYPE_struct = 15,
  DATATYPE_union = 16,
  DATATYPE_event = 17,
  DATATYPE_opaque = 18,
  DATATYPE_unknown = 19,
  DATATYPE_LAST = 20
} aclArgDataType_0_8;

// Enumerations of the valid memory types for pass by pointer
// kernel arguments
typedef enum _acl_memory_type_enum_0_8 {
  PTR_MT_ERROR = 0,          // Error
  PTR_MT_GLOBAL = 1,         // global buffer
  PTR_MT_SCRATCH_EMU = 2,    // SW emulated private memory
  PTR_MT_LDS_EMU = 3,        // SW emulated local memory
  PTR_MT_UAV = 4,            // uniformed access vector memory
  PTR_MT_CONSTANT_EMU = 5,   // SW emulated constant memory
  PTR_MT_GDS_EMU = 6,        // SW emulated region memory
  PTR_MT_LDS = 7,            // HW local memory
  PTR_MT_SCRATCH = 8,        // HW private memory
  PTR_MT_CONSTANT = 9,       // HW constant memory
  PTR_MT_GDS = 10,           // HW region memory
  PTR_MT_UAV_SCRATCH = 11,   // SI and later HW private memory
  PTR_MT_UAV_CONSTANT = 12,  // SI and later HW constant memory
  PTR_MT_LAST = 13
} aclMemoryType_0_8;

// Enumeration that specifies the various access types for a pointer/image.
typedef enum _acl_access_type_enum_0_8 {
  ACCESS_TYPE_ERROR = 0,
  ACCESS_TYPE_RO = 1,
  ACCESS_TYPE_WO = 2,
  ACCESS_TYPE_RW = 3,
  ACCESS_TYPE_LAST = 4
} aclAccessType_0_8;

// Enumeration that specifies the binary types.
typedef enum _acl_binary_image_type_enum_0_8 {
  BINARY_TYPE_ELF = 1,
  BINARY_TYPE_LLVM = 2,
  BINARY_TYPE_SPIRV = 4,
} aclBinaryImageType_0_8;

#endif  // _ACL_ENUMS_0_8_H_
