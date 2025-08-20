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

#ifndef _ACL_STRUCTS_0_8_H_
#define _ACL_STRUCTS_0_8_H_
#define ACL_STRUCT_HEADER size_t struct_size
//! A structure that holds information on the various types of arguments
// The format in memory of this structure is
// -------------
// | aclArgData |
// -------------
// |->argStr    |
// -------------
// |->typeStr   |
// -------------
typedef struct _acl_md_arg_type_0_8 {
  ACL_STRUCT_HEADER;
  size_t argNameSize;
  size_t typeStrSize;
  const char* argStr;
  const char* typeStr;
  union {
    struct {  // Struct for sampler arguments
      unsigned ID;
      unsigned isKernelDefined;
      unsigned value;
    } sampler;
    struct {  // Struct for image arguments
      unsigned resID;
      unsigned cbNum;
      unsigned cbOffset;
      aclAccessType type;
      bool is2D;
      bool is1D;
      bool isArray;
      bool isBuffer;
    } image;
    struct {  // struct for atomic counter arguments
      unsigned is32bit;
      unsigned resID;
      unsigned cbNum;
      unsigned cbOffset;
    } counter;
    struct {  // struct for semaphore arguments
      unsigned resID;
      unsigned cbNum;
      unsigned cbOffset;
    } sema;
    struct {  // struct for pass by value arguments
      unsigned numElements;
      unsigned cbNum;
      unsigned cbOffset;
      aclArgDataType data;
    } value;
    struct {  // struct for pass by pointer arguments
      unsigned numElements;
      unsigned cbNum;
      unsigned cbOffset;
      unsigned bufNum;
      unsigned align;
      aclArgDataType data;
      aclMemoryType memory;
      aclAccessType type;
      bool isVolatile;
      bool isRestrict;
      bool isPipe;
    } pointer;
    struct {  // Struct for queue arguments
      unsigned numElements;
      unsigned cbNum;
      unsigned cbOffset;
      aclArgDataType data;
      aclMemoryType memory;
    } queue;
  } arg;
  aclArgType type;
  bool isConst;
} aclArgData_0_8;

//! A structure that holds information for printf
// The format in memory of this structure is
// --------------
// | aclPrintfFmt|
// --------------
// |->argSizes   |
// --------------
// |->fmrStr     |
// --------------

typedef struct _acl_md_printf_fmt_0_8 {
  ACL_STRUCT_HEADER;
  unsigned ID;
  size_t numSizes;
  size_t fmtStrSize;
  uint32_t* argSizes;
  const char* fmtStr;
} aclPrintfFmt_0_8;

//! A structure that holds the metadata in the RODATA section.
typedef struct _acl_metadata_0_8 {
  ACL_STRUCT_HEADER;  // This holds the size of the structure itself for versioning.
  size_t data_size;   // This holds the size of all the memory allocated for this structure.
  uint32_t major, minor, revision;  // RT_ABI_VERSION
  uint32_t gpuCaps;                 // RT_GPU_FUNC_CAPS
  uint32_t funcID;                  // RT_GPU_FUNC_ID
  uint32_t gpuRes[5];               // RT_GPU_DEFAULT_ID
  size_t wgs[3];                    // RT_WORK_GROUP_SIZE
  uint32_t wrs[3];                  // RT_WORK_REGION_SIZE
  size_t kernelNameSize;
  size_t deviceNameSize;
  size_t mem[6];  // RT_MEM_SIZES
  size_t numArgs;
  size_t numPrintf;

  aclArgData_0_8* args;          // RT_ARGUMENT_ARRAY
  aclPrintfFmt_0_8* printf;      // RT_GPU_PRINTF_ARRAY
  const char* kernelName;        // RT_KERNEL_NAME
  const char* deviceName;        // RT_DEVICE_NAME
  bool enqueue_kernel;           // RT_DEVICE_ENQUEUE
  uint32_t kernel_index;         // RT_KERNEL_INDEX
  uint32_t numHiddenKernelArgs;  // RT_NUM_KERNEL_HIDDEN_ARGS
  size_t wavesPerSimdHint;       // RT_WAVES_PER_SIMD_HINT
  size_t wsh[3];                 // RT_WORK_GROUP_SIZE_HINT
  size_t vecTypeHintSize;
  const char* vth;  // RT_VEC_TYPE_HINT
} aclMetadata_0_8;

//! An structure that holds information on the capabilities of the bif device.
typedef struct _acl_device_caps_rec_0_8 {
  ACL_STRUCT_HEADER;
  uint32_t flags[4];
  uint32_t encryptCode;
} aclDevCaps_0_8;

//! Structure that holds information on the target that the source is
// being compiled for.
typedef struct _acl_target_info_rec_0_8 {
  ACL_STRUCT_HEADER;
  aclDevType arch_id;  // An identifier for the architecture.
  uint32_t chip_id;    // A identifier for the chip.
} aclTargetInfo_0_8;

// Structure for the version 0.8 of the structure.
typedef struct _acl_binary_opts_rec_0_8 {
  ACL_STRUCT_HEADER;
  uint32_t elfclass;
  uint32_t bitness;
  const char* temp_file;
  uint32_t kernelArgAlign;
} aclBinaryOptions_0_8;

// Structure for the version 0.8.1 of the structure.
// This versions addes in alloc/dealloc functions.
typedef struct _acl_binary_opts_rec_0_8_1 {
  ACL_STRUCT_HEADER;
  uint32_t elfclass;
  uint32_t bitness;
  const char* temp_file;
  uint32_t kernelArgAlign;
  AllocFunc_0_8 alloc;
  FreeFunc_0_8 dealloc;
} aclBinaryOptions_0_8_1;

//! Structure that holds the OpenCL binary information.
typedef struct _acl_bif_rec_0_8 {
  ACL_STRUCT_HEADER;
  aclTargetInfo_0_8 target;      // Information about the target device.
  aclBIF* bin;                   // Pointer to the acl.
  aclOptions* options;           // Pointer to acl options.
  aclBinaryOptions_0_8 binOpts;  // Pointer to the binary options.
  aclDevCaps_0_8 caps;           // Capabilities of the BIF.
} aclBinary_0_8;

//! Version of the aclBinary that uses the 0_8_1 version of the aclBinaryOptions.
typedef struct _acl_bif_rec_0_8_1 {
  ACL_STRUCT_HEADER;
  aclTargetInfo_0_8 target;        // Information about the target device.
  aclBIF* bin;                     // Pointer to the acl.
  aclOptions* options;             // Pointer to acl options.
  aclBinaryOptions_0_8_1 binOpts;  // Pointer to the binary options.
  aclDevCaps_0_8 caps;             // Capabilities of the BIF.
} aclBinary_0_8_1;

#define ACL_LOADER_COMMON                                                                          \
  ACL_STRUCT_HEADER;                                                                               \
  bool isBuiltin;                                                                                  \
  const char* libName;                                                                             \
  void* handle;                                                                                    \
  LoaderInit init;                                                                                 \
  LoaderFini fini;

// Struct that maps to the common structure between all loaders.
typedef struct _acl_common_loader_rec_0_8 {
  ACL_LOADER_COMMON;
} aclCommonLoader_0_8;

typedef struct _acl_cl_loader_rec_0_8 {
  ACL_LOADER_COMMON;
  Compile compile;
  Link link;
  CompLog getLog;
  RetrieveType_0_8 retrieveType;
  SetType_0_8 setType;
  ConvertType_0_8 convertType;
  Disassemble disassemble;
  GetDevBinary_0_8 devBinary;
  InsertSec insSec;
  ExtractSec extSec;
  RemoveSec remSec;
  InsertSym insSym;
  ExtractSym extSym;
  RemoveSym remSym;
  QueryInfo getInfo;
  AddDbgArg addDbg;
  RemoveDbgArg removeDbg;
  SetupLoaderObject setupLoaderObject;
  JITObjectImageCreate jitOICreate;
  JITObjectImageCopy jitOICopy;
  JITObjectImageDestroy jitOIDestroy;
  JITObjectImageSize jitOISize;
  JITObjectImageData jitOIData;
  JITObjectImageFinalize jitOIFinalize;
  JITObjectImageGetGlobalsSize jitOIGlobalSize;
  JITObjectImageIterateSymbols jitOIIterateSymbols;
  JITObjectImageDisassembleKernel jitOIDisassembleKernel;
} aclCLLoader_0_8;

//! Structure that holds the required functions
// that sc exports for the SCDLL infrastructure.
typedef struct _acl_sc_loader_rec_0_8 {
  ACL_LOADER_COMMON;
  uint32_t /*SC_UINT32*/ sc_interface_version;
  void /**SC_EXPORT_FUNCTIONS**/* scef;
  // Any version specific fields go here.
} aclSCLoader_0_8;

typedef struct _acl_fe_loader_rec_0_8 {
  ACL_LOADER_COMMON;
  FEToIR toIR;        // Used for Source to aclModule containing LLVMIR
  FEToIR toModule;    // Used to convert raw SPIR/LLVM-IR to aclModule
  SourceToISA toISA;  // Used for Source to ISA
} aclFELoader_0_8;

typedef struct _acl_opt_loader_rec_0_8 {
  ACL_LOADER_COMMON;
  IRPhase optimize;  // Used for IR to IR transformation
} aclOptLoader_0_8;

typedef struct _acl_link_loader_rec_0_8 {
  ACL_LOADER_COMMON;
  LinkPhase link;    // Used for Linking in IR modules
  IRPhase toLLVMIR;  // Used for converting SPIR to LLVMIR
  IRPhase toSPIR;    // Used for converting LLVMIR to SPIR
} aclLinkLoader_0_8;

typedef struct _acl_cg_loader_rec_0_8 {
  ACL_LOADER_COMMON;
  CGPhase codegen;  // Used for converting from LLVMIR to target ASM.
} aclCGLoader_0_8;

typedef struct _acl_be_loader_rec_0_8 {
  ACL_LOADER_COMMON;
  SourceToISA finalize;   // Used for converting from target source to target ISA.
  SourceToISA assemble;   // Used for converting from target text to target binary.
  DisasmISA disassemble;  // Used for converting from target binary to target ISA.
} aclBELoader_0_8;

typedef struct _acl_compiler_opts_rec_0_8 {
  ACL_STRUCT_HEADER;  // Size of the structure for version checking.
  const char* clLib;
  const char* feLib;
  const char* optLib;
  const char* linkLib;
  const char* cgLib;
  const char* beLib;
  const char* scLib;
} aclCompilerOptions_0_8;

typedef struct _acl_compiler_opts_rec_0_8_1 {
  ACL_STRUCT_HEADER;  // Size of the structure for version checking.
  const char* clLib;
  const char* feLib;
  const char* optLib;
  const char* linkLib;
  const char* cgLib;
  const char* beLib;
  const char* scLib;  // Name or path to the shader compiler shared library
  AllocFunc alloc;
  FreeFunc dealloc;
} aclCompilerOptions_0_8_1;

//! Structure that holds the OpenCL compiler and various loaders.
typedef struct _acl_compiler_rec_0_8 {
  ACL_STRUCT_HEADER;         // Size of structure for version checking.
  aclCLLoader clAPI;         // Pointer to the compiler API.
  aclFELoader feAPI;         // Pointer to the FE Loader API.
  aclOptLoader optAPI;       // Pointer to the Opt Loader API.
  aclLinkLoader linkAPI;     // Pointer to the Link Loader API.
  aclCGLoader cgAPI;         // Pointer to the CG Loader API.
  aclBELoader beAPI;         // Pointer to the BE Loader API.
  aclSCLoader scAPI;         // Pointer to the SC Loader API.
  aclCompilerOptions* opts;  // The options structure for the compiler.
  void* llvm_shutdown;       // Pointer to the llvm shutdown object.
  char* buildLog;            // Pointer to the current build log.
  unsigned logSize;          // Size of the current build log.
  aclLoaderData* apiData;    // pointer to data store for the compiler API loader.
} aclCompilerHandle_0_8;

//! Structure that holds the OpenCL compiler and various loaders.
typedef struct _acl_compiler_rec_0_8_1 {
  ACL_STRUCT_HEADER;
  aclCLLoader clAPI;      // Pointer to the compiler API.
  aclFELoader feAPI;      // Pointer to the FE Loader API.
  aclOptLoader optAPI;    // Pointer to the Opt Loader API.
  aclLinkLoader linkAPI;  // Pointer to the Link Loader API.
  aclCGLoader cgAPI;      // Pointer to the CG Loader API.
  aclBELoader beAPI;      // Pointer to the BE Loader API.
  aclSCLoader scAPI;      // Pointer to the SC Loader API.
  AllocFunc alloc;
  FreeFunc dealloc;
  aclCompilerOptions* opts;  // The options structure for the compiler.
  void* llvm_shutdown;       // Pointer to the llvm shutdown object.
  char* buildLog;            // Pointer to the current build log.
  unsigned logSize;          // Size of the current build log.
  aclLoaderData* apiData;    // pointer to data store for the compiler API loader.
} aclCompilerHandle_0_8_1;

//! Structure to hold kernel statistics obtained from kernel
typedef struct _acl_kernel_stats_0_8_1 {
  unsigned int scratchRegs;
  unsigned int scratchSize;
  unsigned int availablevgprs;
  unsigned int availablesgprs;
  unsigned int usedvgprs;
  unsigned int usedsgprs;
  unsigned int availableldssize;
  unsigned int usedldssize;
  unsigned int availablestacksize;
  unsigned int usedstacksize;
  unsigned int wavefrontsize;
  unsigned int wavefrontpersimd;
  unsigned int threadsperworkgroup;
  unsigned int reqdworkgroup_x;
  unsigned int reqdworkgroup_y;
  unsigned int reqdworkgroup_z;
} aclKernelStats;

#endif  // _ACL_STRUCTS_0_8_H_
