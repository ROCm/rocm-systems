/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#if defined(USE_COMGR_LIBRARY)
#include "amd_comgr/amd_comgr.h"
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

namespace amd {

class Device;
class Program;

namespace option {
class Options;
}  // namespace option
}  // namespace amd

namespace amd::device {
class ClBinary;
class Kernel;

struct SymbolInfo {
  int sym_type;
  std::vector<std::string>* var_names;
};

struct SymbolLoweredName {
  const char* name_expression;
  std::string* loweredName;
};

//! A program object for a specific device.
class Program : public amd::HeapObject {
 public:
  typedef std::pair<const void* /* binary_image */, size_t /* binary size */> binary_t;
  typedef std::pair<amd::Os::FileDesc /* file_desc */, size_t /* file_offset */> finfo_t;
  typedef std::unordered_map<std::string, Kernel*> kernels_t;
  // type of the program
  typedef enum {
    TYPE_NONE = 0,     // uncompiled
    TYPE_COMPILED,     // compiled
    TYPE_LIBRARY,      // linked library
    TYPE_EXECUTABLE,   // linked executable
    TYPE_INTERMEDIATE  // intermediate
  } type_t;

  //! type of the input file
  typedef enum {
    FILE_TYPE_DEFAULT = 0,
    FILE_TYPE_OPENCL = 1,
    FILE_TYPE_LLVMIR_TEXT = 2,
    FILE_TYPE_LLVMIR_BINARY = 3,
    FILE_TYPE_SPIR_TEXT = 4,
    FILE_TYPE_SPIR_BINARY = 5,
    FILE_TYPE_AMDIL_TEXT = 6,
    FILE_TYPE_AMDIL_BINARY = 7,
    FILE_TYPE_HSAIL_TEXT = 8,
    FILE_TYPE_HSAIL_BINARY = 9,
    FILE_TYPE_X86_TEXT = 10,
    FILE_TYPE_X86_BINARY = 11,
    FILE_TYPE_CG = 12,
    FILE_TYPE_SOURCE = 13,
    FILE_TYPE_ISA = 14,
    FILE_TYPE_HEADER = 15,
    FILE_TYPE_RSLLVMIR_BINARY = 16,
    FILE_TYPE_SPIRV_BINARY = 17,
    FILE_TYPE_ASM_TEXT = 18,
    FILE_TYPE_LAST = 19
  } file_type_t;

 private:
  //! The device target for this binary.
  amd::SharedReference<amd::Device> device_;
  amd::Program& owner_;  //!< owner of this program

  kernels_t kernels_;  //!< The kernel entry points this binary.
  type_t type_;        //!< type of this program

  std::vector<const Kernel*> initKernels_;  //!< Init kernels
  std::vector<const Kernel*> finiKernels_;  //!< Fini kernels

  bool runInitFiniKernel(const std::vector<const Kernel*>& kernels) const;

#if defined(WITH_COMPILER_LIB)
  static amd::Monitor buildLock_;  //!< Global build lock for HSAIL which isn't thread-safe
#endif

 protected:
  union {
    struct {
      uint32_t isNull_ : 1;           //!< Null program no memory allocations
      uint32_t internal_ : 1;         //!< Internal blit program
      uint32_t isLC_ : 1;             //!< LC was used for the program compilation
      uint32_t hasGlobalStores_ : 1;  //!< Program has writable program scope variables
      uint32_t isHIP_ : 1;            //!< Determine if the program is for HIP
      uint32_t coLoaded_ : 1;         //!< Has the code objected been loaded
      uint32_t trapHandler_ : 1;      //!< It is a trap handler for debugger
    };
    uint32_t flags_;  //!< Program flags
  };

  ClBinary* clBinary_;                    //!< The CL program binary file
  std::string llvmBinary_;                //!< LLVM IR binary code
  amd::Elf::ElfSections elfSectionType_;  //!< LLVM IR binary code is in SPIR format
  std::string compileOptions_;            //!< compile/build options.
  std::string linkOptions_;               //!< link options.
                             //!< the option arg passed in to clCompileProgram(), clLinkProgram(),
                             //! or clBuildProgram(), whichever is called last
#if defined(WITH_COMPILER_LIB)
  aclBinaryOptions binOpts_;  //!< Binary options to create aclBinary
  aclBinary* binaryElf_;      //!< Binary for the new compiler library
#endif

  std::string lastBuildOptionsArg_;
  mutable std::string buildLog_;  //!< build log.
  int32_t buildStatus_;           //!< build status.
  int32_t buildError_;            //!< build error

#if defined(WITH_COMPILER_LIB)
  aclTargetInfo info_;  //!< The info target for this binary.
#endif
  size_t globalVariableTotalSize_;
  amd::option::Options* programOptions_;


#if defined(USE_COMGR_LIBRARY)
  amd_comgr_metadata_node_t metadata_ = {};                             //!< COMgr metadata
  uint32_t codeObjectVer_;                                              //!< version of code object
  std::map<std::string, amd_comgr_metadata_node_t> kernelMetadataMap_;  //!< Map of kernel metadata
#endif
  //! Sanitizer lock - lock when launching init/fini kernels
  static amd::Monitor initFiniLock_;

 public:
  //! Construct a section.
  Program(amd::Device& device, amd::Program& owner);

  //! Destroy this binary image.
  virtual ~Program();

  //! Destroy all the kernels
  void clear();

  amd::Program* owner() const { return &owner_; }

  //! Return the compiler options passed to build this program
  amd::option::Options* getCompilerOptions() const { return programOptions_; }

  //! Compile the device program.
  int32_t compile(const std::string& sourceCode, const std::vector<const std::string*>& headers,
                  const char** headerIncludeNames, const char* origOptions,
                  amd::option::Options* options);

  //! Link the device program.
  int32_t link(const std::vector<Program*>& inputPrograms, const char* origLinkOptions,
               amd::option::Options* linkOptions);

  //! Build the device program.
  int32_t build(const std::string& sourceCode, const char* origOptions,
                amd::option::Options* options);

  //! Load the device program.
  bool load();

  //! Return the device object, associated with this program.
  const amd::Device& device() const { return device_(); }

  //! Return the compiler options used to build the program.
  const std::string& compileOptions() const { return compileOptions_; }

  const std::string& linkOptions() const { return linkOptions_; }

  //! Return the option arg passed in to clCompileProgram(), clLinkProgram(),
  //! or clBuildProgram(), whichever is called last
  const std::string lastBuildOptionsArg() const { return lastBuildOptionsArg_; }

  //! Return the build log.
  const std::string& buildLog() const { return buildLog_; }

  //! Return the build status.
  cl_build_status buildStatus() const { return buildStatus_; }

  //! Return the build error.
  int32_t buildError() const { return buildError_; }

  //! Return the symbols vector.
  const kernels_t& kernels() const { return kernels_; }
  kernels_t& kernels() { return kernels_; }

  //! Add kernel to the map and init fini kernel vector.
  void addKernel(Kernel* k);

  //! Return the binary image.
  inline const binary_t binary() const;
  inline binary_t binary();
  inline finfo_t BinaryFd() const;
  inline std::string BinaryURI() const;

  //! Returns the CL program binary file
  ClBinary* clBinary() { return clBinary_; }
  const ClBinary* clBinary() const { return clBinary_; }

  bool setBinary(const char* binaryIn, size_t size, const device::Program* same_dev_prog = nullptr,
                 amd::Os::FileDesc fdesc = amd::Os::FDescInit(), size_t foffset = 0,
                 std::string uri = std::string());

  type_t type() const { return type_; }

  void setGlobalVariableTotalSize(size_t size) { globalVariableTotalSize_ = size; }

  size_t globalVariableTotalSize() const { return globalVariableTotalSize_; }

#if defined(WITH_COMPILER_LIB)
  //! Returns the aclBinary associated with the program
  aclBinary* binaryElf() const { return static_cast<aclBinary*>(binaryElf_); }
#endif

  //! Returns TRUE if the program just compiled
  bool isNull() const { return isNull_; }

  //! Returns TRUE if the program used internally by runtime
  bool isInternal() const { return internal_; }

  //! Returns TRUE if Lightning compiler was used for this program
  bool isLC() const { return isLC_; }

  //! Global variables are a part of the code segment
  bool hasGlobalStores() const { return hasGlobalStores_; }

  //! Return TRUE if the program has been loaded
  bool isCodeObjectLoaded() const { return coLoaded_; }

  //! Returns TRUE if the program is a trap handler for debugger support
  bool isTrapHandler() const { return trapHandler_; }

#if defined(USE_COMGR_LIBRARY)
  amd_comgr_metadata_node_t metadata() const { return metadata_; }

  //! Get the kernel metadata
  const bool getKernelMetadata(const std::string name, amd_comgr_metadata_node_t* meta) const {
    auto it = kernelMetadataMap_.find(name);
    if (it != kernelMetadataMap_.end()) {
      *meta = it->second;
      return true;
    }
    return false;
  }

  const uint32_t codeObjectVer() const { return codeObjectVer_; }
#endif

  //! Check if program is HIP based
  const bool isHIP() const { return (isHIP_ == 1); }

  //! Get mangled name of a name expresion
  const bool getLoweredNames(std::vector<std::string>* mangledNames) const;

  //! Get demangled names
  bool getDemangledName(const std::string& mangledNames, std::string& demangledNames) const;

  bool getGlobalFuncFromCodeObj(std::vector<std::string>* func_names) const;
  bool getGlobalVarFromCodeObj(std::vector<std::string>* var_names) const;
  bool getUndefinedVarFromCodeObj(std::vector<std::string>* var_names) const;

  virtual bool createGlobalVarObj(amd::Memory** amd_mem_obj, void** dptr, size_t* bytes,
                                  const char* globalName) const {
    ShouldNotReachHere();
    return false;
  }

  //! Run kernels marked with "init" kind metadata
  bool runInitKernels();

  //! Run kernels marked with "fini" kind metadata
  bool runFiniKernels();

 protected:
  //! pre-compile setup
  bool initBuild(amd::option::Options* options);

  //! post-compile cleanup
  bool finiBuild(bool isBuildGood);

  /*! \brief Compiles GPU CL program to LLVM binary (compiler frontend)
   *
   *  \return True if we successefully compiled a GPU program
   */
  virtual bool compileImpl(const std::string& sourceCode,  //!< the program's source code
                           const std::vector<const std::string*>& headers,
                           const char** headerIncludeNames,
                           amd::option::Options* options  //!< compile options's object
  );

  //! Link the device program.
  virtual bool linkImpl(amd::option::Options* options);

  //! Link the device programs.
  virtual bool linkImpl(const std::vector<Program*>& inputPrograms, amd::option::Options* options,
                        bool createLibrary);

  virtual bool createBinary(amd::option::Options* options) = 0;

  //! Initialize Binary (used only for clCreateProgramWithBinary()).
  bool initClBinary(const char* binaryIn, size_t size,
                    amd::Os::FileDesc fdesc = amd::Os::FDescInit(), size_t foffset = 0,
                    std::string uri = std::string());

  //! Initialize Binary
  virtual bool initClBinary();

  virtual bool saveBinaryAndSetType(type_t type) = 0;

  //! Release the Binary
  void releaseClBinary();

#if defined(WITH_COMPILER_LIB)
  //! return target info
  virtual const aclTargetInfo& info() = 0;
#endif
  virtual bool createKernels(void* binary, size_t binSize, bool useUniformWorkGroupSize,
                             bool internalKernel) {
    return true;
  }

  virtual bool setKernels(void* binary, size_t binSize,
                          amd::Os::FileDesc fdesc = amd::Os::FDescInit(), size_t foffset = 0,
                          std::string uri = std::string()) {
    return true;
  }

  //! Returns all the options to be appended while passing to the compiler library
  std::vector<std::string> ProcessOptions(amd::option::Options* options);

  //! Returns all the options to be appended while passing to the compiler library,
  //! flattened into one string.
  std::string ProcessOptionsFlattened(amd::option::Options* options);

  //! At linking time, get the set of compile options to be used from
  //! the set of input program, warn if they have inconsisten compile options.
  bool getCompileOptionsAtLinking(const std::vector<Program*>& inputPrograms,
                                  const amd::option::Options* linkOptions);

  void setType(type_t newType) { type_ = newType; }

  /* \brief Returns the next stage to compile from, based on sections in binary,
   *  also returns completeStages in a vector, which contains at least ACL_TYPE_DEFAULT,
   *  sets needOptionsCheck to true if options check is needed to decide whether or not to recompile
   */
  file_type_t getCompilationStagesFromBinary(std::vector<file_type_t>& completeStages,
                                             bool& needOptionsCheck);

  /* \brief Returns the next stage to compile from, based on sections and options in binary
   */
  file_type_t getNextCompilationStageFromBinary(amd::option::Options* options);

  //! Finds the total size of all global variables in the program
  bool FindGlobalVarSize(void* binary, size_t binSize);

  bool isElf(const char* bin) const { return amd::Elf::isElfMagic(bin); }

  virtual bool defineGlobalVar(const char* name, void* dptr) {
    ShouldNotReachHere();
    return false;
  }

#if defined(USE_COMGR_LIBRARY)
  bool getSymbolsFromCodeObj(std::vector<std::string>* var_names,
                             amd_comgr_symbol_type_t sym_type) const;
#endif
  bool getUndefinedVarInfo(std::string var_name, void** var_addr, size_t* var_size);
  bool defineUndefinedVars();

 private:
  //! Compile the device program with LC path
  bool compileImplLC(const std::string& sourceCode, const std::vector<const std::string*>& headers,
                     const char** headerIncludeNames, amd::option::Options* options);

  //! Compile the device program with HSAIL path
  bool compileImplHSAIL(const std::string& sourceCode,
                        const std::vector<const std::string*>& headers,
                        const char** headerIncludeNames, amd::option::Options* options);

  //! Link the device programs with LC path
  bool linkImplLC(const std::vector<Program*>& inputPrograms, amd::option::Options* options,
                  bool createLibrary);

  //! Link the device programs with HSAIL path
  bool linkImplHSAIL(const std::vector<Program*>& inputPrograms, amd::option::Options* options,
                     bool createLibrary);

  //! Link the device program with LC path
  bool linkImplLC(amd::option::Options* options);

  //! Link the device program with HSAIL path
  bool linkImplHSAIL(amd::option::Options* options);

  //! Load the device program with LC path
  bool loadLC();

  //! Load the device program with HSAIL path
  bool loadHSAIL();

#if defined(USE_COMGR_LIBRARY)
  //! Dump the log data object to the build log, if a log data object is present
  void extractBuildLog(amd_comgr_data_set_t dataSet);
  //! Dump the code object data
  amd_comgr_status_t extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
                                           const amd_comgr_data_kind_t dataKind,
                                           const std::string& outFileName,
                                           char* outBinary[] = nullptr, size_t* outSize = nullptr);

  //! Create code object and add it into the data set
  amd_comgr_status_t addCodeObjData(const char* source, const size_t size,
                                    const amd_comgr_data_kind_t type, const char* name,
                                    amd_comgr_data_set_t* dataSet);

  //! Create action for the specified language, target and options
  amd_comgr_status_t createAction(const amd_comgr_language_t oclver,
                                  const std::vector<std::string>& options,
                                  amd_comgr_action_info_t* action, bool* hasAction);

  //! Create the bitcode of the linked input dataset
  bool linkLLVMBitcode(const amd_comgr_data_set_t inputs, const std::vector<std::string>& options,
                       amd::option::Options* amdOptions, amd_comgr_data_set_t* output,
                       char* binaryData[] = nullptr, size_t* binarySize = nullptr);

  //! Create the bitcode of the compiled input dataset
  bool compileToLLVMBitcode(const amd_comgr_data_set_t compileInputs,
                            const std::vector<std::string>& options,
                            amd::option::Options* amdOptions, char* binaryData[],
                            size_t* binarySize, const bool link_dev_libs = true);

  //! Compile and create the excutable of the input dataset
  bool compileAndLinkExecutable(const amd_comgr_data_set_t inputs,
                                const std::vector<std::string>& options,
                                amd::option::Options* amdOptions, char* executable[],
                                size_t* executableSize, file_type_t continueCompileFrom);

  //! Create the map for the kernel name and its metadata for fast access
  bool createKernelMetadataMap(void* binary, size_t binSize);
#endif

  bool trySubstObjFile(const char* SubstCfgFile, const std::string& sourceCode,
                       const amd::option::Options* options);

  //! Disable default copy constructor
  Program(const Program&);

  //! Disable operator=
  Program& operator=(const Program&);
};

#if defined(USE_COMGR_LIBRARY)

class ComgrBinaryData {
 public:
  ComgrBinaryData() : binaryData_({0}), created_(false) {}
  ~ComgrBinaryData();
  bool create(amd_comgr_data_kind_t kind, void* binary, size_t binSize);
  amd_comgr_data_t& data();

 private:
  amd_comgr_data_t binaryData_;
  bool created_;
};

#endif

}  // namespace amd::device
