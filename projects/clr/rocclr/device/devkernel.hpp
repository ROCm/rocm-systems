/* Copyright (c) 2008 - 2022 Advanced Micro Devices, Inc.

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

#if defined(WITH_COMPILER_LIB)
#include "aclTypes.h"
#endif
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"

namespace amd {
class Device;
class KernelSignature;
class NDRange;

struct KernelParameterDescriptor {
  enum Desc {
    Value = 0,
    MemoryObject = 1,
    ReferenceObject = 2,
    ValueObject = 3,
    ImageObject = 4,
    SamplerObject = 5,
    QueueObject = 6,
    HiddenNone = 7,
    HiddenGlobalOffsetX = 8,
    HiddenGlobalOffsetY = 9,
    HiddenGlobalOffsetZ = 10,
    HiddenPrintfBuffer = 11,
    HiddenDefaultQueue = 12,
    HiddenCompletionAction = 13,
    HiddenMultiGridSync = 14,
    HiddenHeap = 15,
    HiddenHostcallBuffer = 16,
    HiddenBlockCountX = 17,
    HiddenBlockCountY = 18,
    HiddenBlockCountZ = 19,
    HiddenGroupSizeX = 20,
    HiddenGroupSizeY = 21,
    HiddenGroupSizeZ = 22,
    HiddenRemainderX = 23,
    HiddenRemainderY = 24,
    HiddenRemainderZ = 25,
    HiddenGridDims = 26,
    HiddenPrivateBase = 27,
    HiddenSharedBase = 28,
    HiddenQueuePtr = 29,
    HiddenDynamicLdsSize = 30,
    HiddenLast = 31,
    MaxSize = 32,
  };
  clk_value_type_t type_;  //!< The parameter's type
  size_t offset_;          //!< Its offset in the parameter's stack
  size_t size_;            //!< Its size in bytes
  union InfoData {
    struct {
      uint32_t oclObject_ : 6;            //!< OCL object type
      uint32_t readOnly_ : 1;             //!< OCL object is read only, applied to memory only
      uint32_t rawPointer_ : 1;           //!< Arguments have a raw GPU VA
      uint32_t defined_ : 1;              //!< The argument was defined by the app
      uint32_t hidden_ : 1;               //!< It's a hidden argument
      uint32_t shared_ : 1;               //!< Dynamic shared memory
      uint32_t isReadOnlyByCompiler : 1;  //!< Compiler determine it is read only
      uint32_t arrayIndex_ : 20;          //!< Index in the objects array or LDS alignment
    };
    uint32_t allValues_;
    InfoData() : allValues_(0) {}
  } info_;

  cl_kernel_arg_address_qualifier addressQualifier_ =
      CL_KERNEL_ARG_ADDRESS_PRIVATE;  //!< Argument's address qualifier
  cl_kernel_arg_access_qualifier accessQualifier_ =
      CL_KERNEL_ARG_ACCESS_NONE;                //!< Argument's access qualifier
  cl_kernel_arg_type_qualifier typeQualifier_;  //!< Argument's type qualifier

  std::string name_;      //!< The parameter's name in the source
  std::string typeName_;  //!< Argument's type name
  uint32_t alignment_;    //!< Argument's alignment
};
}  // namespace amd

#if defined(USE_COMGR_LIBRARY)
//! Runtime handle structure for device enqueue
struct RuntimeHandle {
  uint64_t kernel_handle;         //!< Pointer to amd_kernel_code_s or kernel_descriptor_t
  uint32_t private_segment_size;  //!< From PRIVATE_SEGMENT_FIXED_SIZE
  uint32_t group_segment_size;    //!< From GROUP_SEGMENT_FIXED_SIZE
};

#include "amd_comgr/amd_comgr.h"

//  for Code Object V3
enum class ArgField : uint8_t {
  Name = 0,
  TypeName = 1,
  Size = 2,
  Align = 3,
  ValueKind = 4,
  PointeeAlign = 5,
  AddrSpaceQual = 6,
  AccQual = 7,
  ActualAccQual = 8,
  IsConst = 9,
  IsRestrict = 10,
  IsVolatile = 11,
  IsPipe = 12,
  Offset = 13,
  MaxSize = 14
};

enum class AttrField : uint8_t {
  ReqdWorkGroupSize = 0,
  WorkGroupSizeHint = 1,
  VecTypeHint = 2,
  RuntimeHandle = 3,
  MaxSize = 4,
};

enum class CodePropField : uint8_t {
  KernargSegmentSize = 0,
  GroupSegmentFixedSize = 1,
  PrivateSegmentFixedSize = 2,
  KernargSegmentAlign = 3,
  WavefrontSize = 4,
  NumSGPRs = 5,
  NumVGPRs = 6,
  MaxFlatWorkGroupSize = 7,
  IsDynamicCallStack = 8,
  IsXNACKEnabled = 9,
  NumSpilledSGPRs = 10,
  NumSpilledVGPRs = 11,
  MaxSize = 12,
};

//  for Code Object V3
enum class KernelField : uint8_t {
  SymbolName = 0,
  ReqdWorkGroupSize = 1,
  WorkGroupSizeHint = 2,
  VecTypeHint = 3,
  DeviceEnqueueSymbol = 4,
  KernargSegmentSize = 5,
  GroupSegmentFixedSize = 6,
  PrivateSegmentFixedSize = 7,
  KernargSegmentAlign = 8,
  WavefrontSize = 9,
  NumSGPRs = 10,
  NumVGPRs = 11,
  MaxFlatWorkGroupSize = 12,
  NumSpilledSGPRs = 13,
  NumSpilledVGPRs = 14,
  Kind = 15,
  WgpMode = 16,
  UniformWrokGroupSize = 17,
  MaxSize = 18
};

#endif  // defined(USE_COMGR_LIBRARY)

namespace amd {
namespace hsa {
namespace loader {
class Symbol;
}  // namespace loader
namespace code {
namespace Kernel {
class Metadata;
}  // namespace Kernel
}  // namespace code
}  // namespace hsa
}  // namespace amd

namespace amd::device {

class Program;

//! Printf info structure
struct PrintfInfo {
  std::string fmtString_;        //!< formated string for printf
  std::vector<uint> arguments_;  //!< passed arguments to the printf() call
};

//! \class DeviceKernel, which will contain the common fields for any device
class Kernel : public amd::HeapObject {
 public:
  typedef std::vector<amd::KernelParameterDescriptor> parameters_t;

  //! \struct The device kernel workgroup info structure
  struct WorkGroupInfo : public amd::EmbeddedObject {
    size_t size_;                   //!< kernel workgroup size
    size_t compileSize_[3];         //!< kernel compiled workgroup size
    uint64_t localMemSize_;         //!< amount of used local memory
    size_t preferredSizeMultiple_;  //!< preferred multiple for launch
    uint64_t privateMemSize_;       //!< amount of used private memory
    size_t scratchRegs_;            //!< amount of used scratch registers
    size_t wavefrontPerSIMD_;       //!< number of wavefronts per SIMD
    size_t wavefrontSize_;          //!< number of threads per wavefront
    size_t availableGPRs_;          //!< GPRs available to the program
    size_t usedGPRs_;               //!< GPRs used by the program
    size_t availableSGPRs_;         //!< SGPRs available to the program
    size_t usedSGPRs_;              //!< SGPRs used by the program
    size_t availableVGPRs_;         //!< VGPRs addressable to the program per thread in DWORDs
    size_t usedVGPRs_;              //!< VGPRs used by the program per thread in DWORDs
    size_t availableLDSSize_;       //!< available LDS size
    size_t usedLDSSize_;            //!< used LDS size
    size_t availableStackSize_;     //!< available stack size
    size_t usedStackSize_;          //!< used stack size
    size_t compileSizeHint_[3];     //!< kernel compiled workgroup size hint
    size_t wavesPerSimdHint_;       //!< waves per simd hit
    size_t constMemSize_;           //!< size of user-allocated constant memory
    size_t maxDynamicSharedSizeBytes_;
    std::string compileVecTypeHint_;  //!< kernel compiled vector type hint

    int maxOccupancyPerCu_;      //!< Max occupancy per compute unit in threads
    bool isWGPMode_;             //!< kernel compiled in WGP/cumode
    bool uniformWorkGroupSize_;  //!< uniform work group size option
  };

  //! Default constructor
  Kernel(const amd::Device& dev, const std::string& name, const Program& prog);

  //! Default destructor
  virtual ~Kernel();

  //! Returns the kernel info structure
  const WorkGroupInfo* workGroupInfo() const { return &workGroupInfo_; }
  //! Returns the kernel info structure for filling in
  WorkGroupInfo* workGroupInfo() { return &workGroupInfo_; }

  //! Returns the kernel signature
  const amd::KernelSignature& signature() const { return *signature_; }

  //! Returns the kernel name
  const std::string& name() const { return name_; }

  //! Initializes the kernel parameters for the abstraction layer
  bool createSignature(const parameters_t& params, uint32_t numParameters, uint32_t version);

  void setUniformWorkGroupSize(bool u) { workGroupInfo_.uniformWorkGroupSize_ = u; }

  bool getUniformWorkGroupSize() const { return workGroupInfo_.uniformWorkGroupSize_; }

  void setReqdWorkGroupSize(size_t x, size_t y, size_t z) {
    workGroupInfo_.compileSize_[0] = x;
    workGroupInfo_.compileSize_[1] = y;
    workGroupInfo_.compileSize_[2] = z;
  }

  size_t getReqdWorkGroupSize(int dim) { return workGroupInfo_.compileSize_[dim]; }

  void setWorkGroupSizeHint(size_t x, size_t y, size_t z) {
    workGroupInfo_.compileSizeHint_[0] = x;
    workGroupInfo_.compileSizeHint_[1] = y;
    workGroupInfo_.compileSizeHint_[2] = z;
  }

  size_t getWorkGroupSizeHint(int dim) const { return workGroupInfo_.compileSizeHint_[dim]; }

  //! Returns GPU device object, associated with this kernel
  const amd::Device& device() const { return dev_; }

  void setVecTypeHint(const std::string& hint) { workGroupInfo_.compileVecTypeHint_ = hint; }

  void setLocalMemSize(size_t size) { workGroupInfo_.localMemSize_ = size; }

  void setPreferredSizeMultiple(size_t size) { workGroupInfo_.preferredSizeMultiple_ = size; }

  const std::string& RuntimeHandle() const { return runtimeHandle_; }
  void setRuntimeHandle(const std::string& handle) { runtimeHandle_ = handle; }

  //! Return the build log
  const std::string& buildLog() const { return buildLog_; }

#if defined(WITH_COMPILER_LIB)
  static std::string openclMangledName(const std::string& name);
#endif

  const std::unordered_map<size_t, size_t>& patch() const { return patchReferences_; }

  //! Returns TRUE if kernel uses dynamic parallelism
  bool dynamicParallelism() const { return (flags_.dynamicParallelism_) ? true : false; }

  //! set dynamic parallelism flag
  void setDynamicParallelFlag(bool flag) { flags_.dynamicParallelism_ = flag; }

  //! Returns TRUE if kernel is internal kernel
  bool isInternalKernel() const { return (flags_.internalKernel_) ? true : false; }

  //! set internal kernel flag
  void setInternalKernelFlag(bool flag) { flags_.internalKernel_ = flag; }

  //! Return TRUE if kernel uses images
  bool imageEnable() const { return (flags_.imageEna_) ? true : false; }

  //! Return TRUE if kernel wirtes images
  bool imageWrite() const { return (flags_.imageWriteEna_) ? true : false; }

  //! Returns TRUE if it's a HSA kernel
  bool hsa() const { return (flags_.hsa_) ? true : false; }

  //! Return printf info array
  const std::vector<PrintfInfo>& printfInfo() const { return printf_; }

  //! Finds local workgroup size
  void FindLocalWorkSize(size_t workDim,                   //!< Work dimension
                         const amd::NDRange& gblWorkSize,  //!< Global work size
                         amd::NDRange& lclWorkSize         //!< Calculated local work size
  ) const;

  const uint64_t KernelCodeHandle() const { return kernelCodeHandle_; }

  const uint32_t WorkgroupGroupSegmentByteSize() const { return workgroupGroupSegmentByteSize_; }
  void SetWorkgroupGroupSegmentByteSize(uint32_t size) { workgroupGroupSegmentByteSize_ = size; }

  const uint32_t WorkitemPrivateSegmentByteSize() const { return workitemPrivateSegmentByteSize_; }
  void SetWorkitemPrivateSegmentByteSize(uint32_t size) { workitemPrivateSegmentByteSize_ = size; }

  const bool KernalHasDynamicCallStack() const { return kernelHasDynamicCallStack_; }

  const uint32_t KernargSegmentByteSize() const { return kernargSegmentByteSize_; }
  void SetKernargSegmentByteSize(uint32_t size) { kernargSegmentByteSize_ = size; }

  const uint32_t KernargSegmentAlignment() const { return kernargSegmentAlignment_; }
  void SetKernargSegmentAlignment(uint32_t align) { kernargSegmentAlignment_ = align; }

  void SetSymbolName(const std::string& name) { symbolName_ = name; }

  void SetKernelKind(const std::string& kind) {
    kind_ = (kind == "init") ? Init : ((kind == "fini") ? Fini : Normal);
  }

  void SetWGPMode(bool wgpMode) { workGroupInfo_.isWGPMode_ = wgpMode; }

  bool isInitKernel() const { return kind_ == Init; }

  bool isFiniKernel() const { return kind_ == Fini; }

 protected:
  //! Initializes the abstraction layer kernel parameters
#if defined(USE_COMGR_LIBRARY)
  void InitParameters(const amd_comgr_metadata_node_t kernelMD);

  //! Retrieve kernel attribute and code properties metadata
  bool GetAttrCodePropMetadata();

  //! Retrieve the printf string metadata
  bool GetPrintfStr(std::vector<std::string>* printfStr);

  //! Returns the kernel symbol name
  const std::string& symbolName() const { return symbolName_; }

  //! Returns the kernel code object version
  const uint32_t codeObjectVer() const { return prog().codeObjectVer(); }
  //! Initializes HSAIL Printf metadata and info for LC
  void InitPrintf(const std::vector<std::string>& printfInfoStrings);
#endif
#if defined(WITH_COMPILER_LIB)
  void InitParameters(const aclArgData* aclArg,  //!< List of ACL arguments
                      uint32_t argBufferSize);
  //! Initializes HSAIL Printf metadata and info
  void InitPrintf(const aclPrintfFmt* aclPrintf);
#endif
  //! Returns program associated with this kernel
  const Program& prog() const { return prog_; }

  const amd::Device& dev_;           //!< GPU device object
  std::string name_;                 //!< kernel name
  const Program& prog_;              //!< Reference to the parent program
  std::string symbolName_;           //!< kernel symbol name
  WorkGroupInfo workGroupInfo_;      //!< device kernel info structure
  amd::KernelSignature* signature_;  //!< kernel signature
  std::string buildLog_;             //!< build log
  std::vector<PrintfInfo> printf_;   //!< Format strings for GPU printf support
  std::string runtimeHandle_;        //!< Runtime handle for context loader

  uint64_t kernelCodeHandle_ = 0;  //!< Kernel code handle (aka amd_kernel_code_t)
  uint32_t workgroupGroupSegmentByteSize_ = 0;
  uint32_t workitemPrivateSegmentByteSize_ = 0;
  uint32_t kernargSegmentByteSize_ = 0;  //!< Size of kernel argument buffer
  uint32_t kernargSegmentAlignment_ = 0;
  bool kernelHasDynamicCallStack_ = 0;

  union Flags {
    struct {
      uint imageEna_ : 1;            //!< Kernel uses images
      uint imageWriteEna_ : 1;       //!< Kernel uses image writes
      uint dynamicParallelism_ : 1;  //!< Dynamic parallelism enabled
      uint internalKernel_ : 1;      //!< True: internal kernel
      uint hsa_ : 1;                 //!< HSA kernel
    };
    uint value_;
    Flags() : value_(0) {}
  } flags_;


 private:
  //! Disable default copy constructor
  Kernel(const Kernel&);

  //! Disable operator=
  Kernel& operator=(const Kernel&);

  std::unordered_map<size_t, size_t> patchReferences_;  //!< Patch table for references

  enum KernelKind { Normal = 0, Init = 1, Fini = 2 };

  KernelKind kind_{Normal};  //!< Kernel kind, is normal unless specified otherwise
};

#if defined(USE_COMGR_LIBRARY)
amd_comgr_status_t getMetaBuf(const amd_comgr_metadata_node_t meta, std::string* str);
#endif  // defined(USE_COMGR_LIBRARY)
}  // namespace amd::device
