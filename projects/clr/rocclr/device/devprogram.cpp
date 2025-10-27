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

#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "platform/ndrange.hpp"
#include "devprogram.hpp"
#include "devkernel.hpp"
#include "utils/macros.hpp"
#include "utils/options.hpp"
#include "comgrctx.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cstdio>

#if defined(ATI_OS_LINUX)
#include <dlfcn.h>
#include <libgen.h>
#endif  // defined(ATI_OS_LINUX)

namespace amd::device {

inline static std::vector<std::string> splitSpaceSeparatedString(const char* str) {
  std::string s(str);
  std::stringstream ss(s);
  std::istream_iterator<std::string> beg(ss), end;
  std::vector<std::string> vec(beg, end);
  return vec;
}

// ================================================================================================
Program::Program(amd::Device& device, amd::Program& owner)
    : device_(device),
      owner_(owner),
      type_(TYPE_NONE),
      initKernels_(),
      finiKernels_(),
      flags_(0),
      clBinary_(nullptr),
      llvmBinary_(),
      elfSectionType_(amd::Elf::LLVMIR),
      compileOptions_(),
      linkOptions_(),
      lastBuildOptionsArg_(),
      buildStatus_(CL_BUILD_NONE),
      buildError_(CL_SUCCESS),
      globalVariableTotalSize_(0),
      programOptions_(nullptr) {}

// ================================================================================================
Program::~Program() {
  clear();
  for (auto const& kernelMeta : kernelMetadataMap_) {
    amd::Comgr::destroy_metadata(kernelMeta.second);
  }
  amd::Comgr::destroy_metadata(metadata_);
}

// ================================================================================================
void Program::clear() {
  initKernels_.clear();
  finiKernels_.clear();
  // Destroy all device kernels
  for (const auto& it : kernels_) {
    delete it.second;
  }
  kernels_.clear();
}

// ================================================================================================

// If buildLog is not null, and dataSet contains a log object, extract the
// first log data object from dataSet and process it with
// extractByteCodeBinary.
void Program::extractBuildLog(amd_comgr_data_set_t dataSet) {
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  size_t count;
  status = amd::Comgr::action_data_count(dataSet, AMD_COMGR_DATA_KIND_LOG, &count);

  if (status == AMD_COMGR_STATUS_SUCCESS && count > 0) {
    char* logData = nullptr;
    size_t logSize;
    status = extractByteCodeBinary(dataSet, AMD_COMGR_DATA_KIND_LOG, "", &logData, &logSize);
    buildLog_ += logData;
    delete[] logData;
  }
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Warning: extracting build log failed.\n";
  }
}

//  Extract the byte code binary from the data set.  The binary will be saved to an output
//  file if the file name is provided. If buffer pointer, outBinary, is provided, the
//  binary will be passed back to the caller.
//
amd_comgr_status_t Program::extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
                                                  const amd_comgr_data_kind_t dataKind,
                                                  const std::string& outFileName, char* outBinary[],
                                                  size_t* outSize) {
  amd_comgr_data_t binaryData;

  amd_comgr_status_t status = amd::Comgr::action_data_get_data(inDataSet, dataKind, 0, &binaryData);

  size_t binarySize = 0;
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_data(binaryData, &binarySize, NULL);
  }

  size_t bufSize = (dataKind == AMD_COMGR_DATA_KIND_LOG) ? binarySize + 1 : binarySize;

  char* binary = new char[bufSize];
  if (binary == nullptr) {
    amd::Comgr::release_data(binaryData);
    return AMD_COMGR_STATUS_ERROR;
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_data(binaryData, &binarySize, binary);
  }

  if (dataKind == AMD_COMGR_DATA_KIND_LOG) {
    binary[binarySize] = '\0';
  }

  amd::Comgr::release_data(binaryData);

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    delete[] binary;
    return status;
  }

  // save the binary to the file as output file name is specified
  if (!outFileName.empty()) {
    std::ofstream f(outFileName.c_str(), std::ios::trunc | std::ios::binary);
    if (f.is_open()) {
      f.write(binary, binarySize);
      f.close();
    } else {
      buildLog_ += "Warning: opening the file to dump the code failed.\n";
    }
  }

  if (outBinary != nullptr) {
    // Pass the dump binary and its size back to the caller
    *outBinary = binary;
    *outSize = binarySize;
  } else {
    delete[] binary;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t Program::addCodeObjData(const char* source, const size_t size,
                                           const amd_comgr_data_kind_t type, const char* name,
                                           amd_comgr_data_set_t* dataSet) {
  amd_comgr_data_t data;
  amd_comgr_status_t status;

  status = amd::Comgr::create_data(type, &data);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  status = amd::Comgr::set_data(data, size, source);

  if ((name != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd::Comgr::set_data_name(data, name);
  }

  if ((dataSet != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd::Comgr::data_set_add(*dataSet, data);
  }

  amd::Comgr::release_data(data);

  return status;
}

static amd_comgr_language_t getCOMGRLanguage(bool isHIP, const amd::option::Options& amdOptions) {
  if (isHIP) {
    return AMD_COMGR_LANGUAGE_HIP;
  } else {
    const char* clStd = amdOptions.oVariables->CLStd;
    uint clcStd = (clStd[2] - '0') * 100 + (clStd[4] - '0') * 10;

    switch (clcStd) {
      case 100:
      case 110:
      case 120:
        return AMD_COMGR_LANGUAGE_OPENCL_1_2;
      case 200:
        return AMD_COMGR_LANGUAGE_OPENCL_2_0;
      default:
        break;
    }
  }

  DevLogPrintfError("Cannot set Language version for %s \n", amdOptions.oVariables->CLStd);
  return AMD_COMGR_LANGUAGE_NONE;
}


amd_comgr_status_t Program::createAction(const amd_comgr_language_t oclver,
                                         const std::vector<std::string>& options,
                                         amd_comgr_action_info_t* action, bool* hasAction) {
  *hasAction = false;
  amd_comgr_status_t status = amd::Comgr::create_action_info(action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    *hasAction = true;
    if (oclver != AMD_COMGR_LANGUAGE_NONE) {
      status = amd::Comgr::action_info_set_language(*action, oclver);
    }
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::action_info_set_isa_name(*action, device().isa().isaName().c_str());
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::vector<const char*> optionsArgv;
    optionsArgv.reserve(options.size());
    for (auto& option : options) {
      optionsArgv.push_back(option.c_str());
    }
    status =
        amd::Comgr::action_info_set_option_list(*action, optionsArgv.data(), optionsArgv.size());
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::action_info_set_logging(*action, true);
  }

  return status;
}

bool Program::linkLLVMBitcode(const amd_comgr_data_set_t inputs,
                              const std::vector<std::string>& options,
                              amd::option::Options* amdOptions, amd_comgr_data_set_t* output,
                              char* binaryData[], size_t* binarySize) {
  amd_comgr_language_t langver = getCOMGRLanguage(isHIP(), *amdOptions);
  if (langver == AMD_COMGR_LANGUAGE_NONE) {
    return false;
  }

  //  Create the action for linking
  amd_comgr_action_info_t action;
  bool hasAction = false;

  amd_comgr_status_t status = createAction(langver, options, &action, &hasAction);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, action, inputs, *output);
    extractBuildLog(*output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string dumpFileName;
    if (amdOptions->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
      dumpFileName = amdOptions->getDumpFileName("_linked.bc");
    }
    status = extractByteCodeBinary(*output, AMD_COMGR_DATA_KIND_BC, dumpFileName, binaryData,
                                   binarySize);
  }

  if (hasAction) {
    amd::Comgr::destroy_action_info(action);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Program::compileToLLVMBitcode(const amd_comgr_data_set_t compileInputs,
                                   const std::vector<std::string>& options,
                                   amd::option::Options* amdOptions, char* binaryData[],
                                   size_t* binarySize, const bool link_dev_libs) {
  amd_comgr_language_t langver = getCOMGRLanguage(isHIP(), *amdOptions);
  if (langver == AMD_COMGR_LANGUAGE_NONE) {
    return false;
  }

  //  Create the output data set
  amd_comgr_action_info_t action{};
  amd_comgr_data_set_t output{};
  amd_comgr_data_set_t dataSetPCH{};
  amd_comgr_data_set_t input = compileInputs;

  bool hasAction = false;
  bool hasOutput = false;
  bool hasDataSetPCH = false;

  amd_comgr_status_t status = createAction(langver, options, &action, &hasAction);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&output);
  }

  //  Adding Precompiled Headers
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasOutput = true;
    status = amd::Comgr::create_data_set(&dataSetPCH);
  }

  // Preprocess the source
  // FIXME: This must happen before the precompiled headers are added, as they
  // do not embed the source text of the header, and so reference paths in the
  // filesystem which do not exist at runtime.
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasDataSetPCH = true;

    if (amdOptions->isDumpFlagSet(amd::option::DUMP_I)) {
      amd_comgr_data_set_t dataSetPreprocessor;
      bool hasDataSetPreprocessor = false;

      status = amd::Comgr::create_data_set(&dataSetPreprocessor);

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        hasDataSetPreprocessor = true;
        status = amd::Comgr::do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR, action, input,
                                       dataSetPreprocessor);
        extractBuildLog(dataSetPreprocessor);
      }

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        std::string outFileName = amdOptions->getDumpFileName(".i");
        status =
            extractByteCodeBinary(dataSetPreprocessor, AMD_COMGR_DATA_KIND_SOURCE, outFileName);
      }

      if (hasDataSetPreprocessor) {
        amd::Comgr::destroy_data_set(dataSetPreprocessor);
      }
    }
  }

  if (!isHIP()) {
    if (status == AMD_COMGR_STATUS_SUCCESS) {
      status = amd::Comgr::do_action(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS, action, input,
                                     dataSetPCH);
      extractBuildLog(dataSetPCH);
    }

    // Set input for the next stage
    input = dataSetPCH;
  }

  //  Compiling the source codes with precompiled headers or directly compileInputs
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    if (link_dev_libs) {
      status = amd::Comgr::do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action,
                                     input, output);
    } else {
      status = amd::Comgr::do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, action, input, output);
    }
    extractBuildLog(output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string outFileName;
    if (amdOptions->isDumpFlagSet(amd::option::DUMP_BC_OPTIMIZED)) {
      outFileName = amdOptions->getDumpFileName("_optimized.bc");
    }
    status =
        extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, outFileName, binaryData, binarySize);
  }

  if (hasAction) {
    amd::Comgr::destroy_action_info(action);
  }

  if (hasDataSetPCH) {
    amd::Comgr::destroy_data_set(dataSetPCH);
  }

  if (hasOutput) {
    amd::Comgr::destroy_data_set(output);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

//  Create an executable from an input data set.  To generate the executable,
//  the input data set is converted to relocatable code, then executable binary.
//  If assembly code is required, the input data set is converted to assembly.
bool Program::compileAndLinkExecutable(const amd_comgr_data_set_t inputs,
                                       const std::vector<std::string>& options,
                                       amd::option::Options* amdOptions, char* executable[],
                                       size_t* executableSize, file_type_t continueCompileFrom) {
  // create the linked output
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t relocatableData;
  bool hasAction = false;
  bool hasOutput = false;
  bool hasRelocatableData = false;

  amd_comgr_status_t status = createAction(AMD_COMGR_LANGUAGE_NONE, options, &action, &hasAction);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasOutput = true;

    if ((amdOptions->isDumpFlagSet(amd::option::DUMP_ISA)) ||
        (isHIP() && amdOptions->origOptionStr.find("-save-temps") != std::string::npos)) {
      //  create the assembly data set
      amd_comgr_data_set_t assemblyData;
      bool hasAssemblyData = false;

      status = amd::Comgr::create_data_set(&assemblyData);
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        hasAssemblyData = true;
        status = amd::Comgr::do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY, action, inputs,
                                       assemblyData);
        extractBuildLog(assemblyData);
      }

      // dump the ISA
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        std::string dumpIsaName = amdOptions->getDumpFileName(".s");
        status = extractByteCodeBinary(assemblyData, AMD_COMGR_DATA_KIND_SOURCE, dumpIsaName);
      }

      if (hasAssemblyData) {
        amd::Comgr::destroy_data_set(assemblyData);
      }
    }
  }

  //  Create the relocatable data set
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&relocatableData);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasRelocatableData = true;
    amd_comgr_action_kind_t kind = (continueCompileFrom == FILE_TYPE_ASM_TEXT)
                                       ? AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE
                                       : AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE;
    status = amd::Comgr::do_action(kind, action, inputs, relocatableData);
    extractBuildLog(relocatableData);
  }

  // Create executable from the relocatable data set
  amd::Comgr::action_info_set_option_list(action, nullptr, 0);
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action,
                                   relocatableData, output);
    extractBuildLog(output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    // Extract the executable binary
    std::string outFileName;
    if (amdOptions->isDumpFlagSet(amd::option::DUMP_O)) {
      outFileName = amdOptions->getDumpFileName(".so");
    }
    status = extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_EXECUTABLE, outFileName, executable,
                                   executableSize);
  }

  if (hasAction) {
    amd::Comgr::destroy_action_info(action);
  }

  if (hasRelocatableData) {
    amd::Comgr::destroy_data_set(relocatableData);
  }

  if (hasOutput) {
    amd::Comgr::destroy_data_set(output);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

static std::size_t getOCLSourceHash(const std::string& sourceCode) {
  return std::hash<std::string>()(sourceCode);
}

static std::size_t getOCLOptionsHash(const amd::option::Options& options) {
  std::string opts;
  for (const std::string& S : options.clangOptions) opts.append(S);
  return std::hash<std::string>()(opts);
}

bool Program::compileImpl(const std::string& sourceCode,
                          const std::vector<const std::string*>& headers,
                          const char** headerIncludeNames, amd::option::Options* options) {
  const char* xLang = options->oVariables->XLang;
  if (xLang != nullptr) {
    if (strcmp(xLang, "asm") == 0) {
      clBinary()->elfOut()->addSection(amd::Elf::SOURCE, sourceCode.data(), sourceCode.size());
      return true;
    } else if (!strcmp(xLang, "cl")) {
      buildLog_ += "Unsupported language: \"" + std::string(xLang) + "\".\n";
      return false;
    }
  }

  // add CL source to input data set
  amd_comgr_data_set_t inputs;

  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create output buffer for LLVM bitcode.\n";
    return false;
  }

  if (addCodeObjData(sourceCode.c_str(), sourceCode.length(), AMD_COMGR_DATA_KIND_SOURCE,
                     "CompileSource", &inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data from source.\n";
    amd::Comgr::destroy_data_set(inputs);
    return false;
  }

  std::vector<std::string> driverOptions;
  // Set the -O#
  std::ostringstream optLevel;
  optLevel << "-O" << options->oVariables->OptLevel;
  driverOptions.push_back(optLevel.str());

  if (!isHIP()) {
    driverOptions.insert(driverOptions.end(), options->clangOptions.begin(),
                         options->clangOptions.end());
    // TODO: Can this be fixed at the source? options->llvmOptions is a flat
    // string, but should really be a vector of strings.
    std::vector<std::string> splitLlvmOptions =
        splitSpaceSeparatedString(options->llvmOptions.c_str());
    driverOptions.insert(driverOptions.end(), splitLlvmOptions.begin(), splitLlvmOptions.end());
  }

  std::vector<std::string> processedOptions = ProcessOptions(options);
  driverOptions.insert(driverOptions.end(), processedOptions.begin(), processedOptions.end());

  // Set whole program mode
  driverOptions.push_back("-mllvm");
  driverOptions.push_back("-amdgpu-prelink");

  if (!device().settings().enableWgpMode_) {
    driverOptions.push_back("-mcumode");
  }

  if (device().settings().lcWavefrontSize64_) {
    driverOptions.push_back("-mwavefrontsize64");
  }
  driverOptions.push_back("-mcode-object-version=" +
                          std::to_string(options->oVariables->LCCodeObjectVersion));

  // Iterate through each source code and dump it into tmp
  std::fstream f;
  std::vector<std::string> headerFileNames(headers.size());

  if (!headers.empty()) {
    for (size_t i = 0; i < headers.size(); ++i) {
      std::string headerIncludeName(headerIncludeNames[i]);
      // replace / in path with current os's file separator
      if (amd::Os::fileSeparator() != '/') {
        for (auto& it : headerIncludeName) {
          if (it == '/') it = amd::Os::fileSeparator();
        }
      }
      if (addCodeObjData(headers[i]->c_str(), headers[i]->length(), AMD_COMGR_DATA_KIND_INCLUDE,
                         headerIncludeName.c_str(), &inputs) != AMD_COMGR_STATUS_SUCCESS) {
        buildLog_ += "Error: COMGR fails to add headers into inputs.\n";
        amd::Comgr::destroy_data_set(inputs);
        return false;
      }
    }
  }

  if (!isHIP() && options->isDumpFlagSet(amd::option::DUMP_CL)) {
    std::ostringstream driverOptionsOStrStr;
    std::copy(driverOptions.begin(), driverOptions.end(),
              std::ostream_iterator<std::string>(driverOptionsOStrStr, " "));

    std::ofstream f(options->getDumpFileName(".cl").c_str(), std::ios::trunc);
    if (f.is_open()) {
      auto srcHash = getOCLSourceHash(sourceCode);
      auto optHash = getOCLOptionsHash(*options);

      f << "/* Compiler options:\n"
           "-c -emit-llvm -target amdgcn-amd-amdhsa -x cl "
        << driverOptionsOStrStr.str() << " -include opencl-c.h "
        << "\nHash to override:"
        << "\n  Source: 0x" << std::setbase(16) << srcHash << "\n  Source + clang options: 0x"
        << (srcHash ^ optHash) << "\n*/\n\n"
        << sourceCode;
      f.close();
    } else {
      buildLog_ += "Warning: opening the file to dump the OpenCL source failed.\n";
    }
  }

  // Append Options provided by user to driver options
  if (isHIP()) {
    if (options->origOptionStr.size()) {
      std::istringstream userOptions{options->origOptionStr};
      std::copy(std::istream_iterator<std::string>(userOptions),
                std::istream_iterator<std::string>(), std::back_inserter(driverOptions));
    }
  }

  // Compile source to IR
  char* binaryData = nullptr;
  size_t binarySize = 0;
  bool ret = compileToLLVMBitcode(inputs, driverOptions, options, &binaryData, &binarySize);
  if (ret) {
    llvmBinary_.assign(binaryData, binarySize);
    // Destroy the original LLVM binary, received after compilation
    delete[] binaryData;

    elfSectionType_ = amd::Elf::LLVMIR;

    if (clBinary()->saveSOURCE()) {
      clBinary()->elfOut()->addSection(amd::Elf::SOURCE, sourceCode.data(), sourceCode.size());
    }
    if (clBinary()->saveLLVMIR()) {
      clBinary()->elfOut()->addSection(amd::Elf::LLVMIR, llvmBinary_.data(), llvmBinary_.size());
      compileOptions_.clear();
    }
  } else {
    buildLog_ += "Error: Failed to compile source (from CL or HIP source to LLVM IR).\n";
  }

  amd::Comgr::destroy_data_set(inputs);
  return ret;
}

// ================================================================================================
bool Program::linkImpl(const std::vector<Program*>& inputPrograms, amd::option::Options* options,
                       bool createLibrary) {
  amd_comgr_data_set_t inputs;

  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data set.\n";
    return false;
  }

  size_t idx = 0;
  for (auto program : inputPrograms) {
    bool result = true;
    if (program->llvmBinary_.empty()) {
      result = (program->clBinary() != nullptr);
      if (result) {
        // We are using CL binary directly.
        // Setup elfIn() and try to load llvmIR from binary
        // This elfIn() will be released at the end of build by finiBuild().
        result = program->clBinary()->setElfIn();
      }

      if (result) {
        result =
            program->clBinary()->loadLlvmBinary(program->llvmBinary_, program->elfSectionType_);
      }
    }

    if (result) {
      result = (program->elfSectionType_ == amd::Elf::LLVMIR);
    }

    if (result) {
      std::string llvmName = "LLVM Binary " + std::to_string(idx);
      result = (addCodeObjData(program->llvmBinary_.data(), program->llvmBinary_.size(),
                               AMD_COMGR_DATA_KIND_BC, llvmName.c_str(),
                               &inputs) == AMD_COMGR_STATUS_SUCCESS);
    }

    if (!result) {
      amd::Comgr::destroy_data_set(inputs);
      buildLog_ += "Error: Linking bitcode failed: failing to generate LLVM binary.\n";
      return false;
    }

    idx++;

    // release elfIn() for the program
    program->clBinary()->resetElfIn();
  }

  // create the linked output
  amd_comgr_data_set_t output;
  if (amd::Comgr::create_data_set(&output) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create output buffer for LLVM bitcode.\n";
    amd::Comgr::destroy_data_set(inputs);
    return false;
  }

  // NOTE: The options parameter is also used to identy cached code object.
  //       This parameter should not contain any dyanamically generated filename.
  char* binaryData = nullptr;
  size_t binarySize = 0;
  std::vector<std::string> linkOptions;
  bool ret = linkLLVMBitcode(inputs, linkOptions, options, &output, &binaryData, &binarySize);

  amd::Comgr::destroy_data_set(output);
  amd::Comgr::destroy_data_set(inputs);

  if (!ret) {
    buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
    return false;
  }

  llvmBinary_.assign(binaryData, binarySize);

  // Destroy llvm binary, received after compilation
  delete[] binaryData;

  elfSectionType_ = amd::Elf::LLVMIR;

  if (clBinary()->saveLLVMIR()) {
    clBinary()->elfOut()->addSection(amd::Elf::LLVMIR, llvmBinary_.data(), llvmBinary_.size());
  }

  // skip the rest if we are building an opencl library
  if (createLibrary) {
    setType(TYPE_LIBRARY);
    if (!createBinary(options)) {
      buildLog_ += "Internal error: creating OpenCL binary failed\n";
      return false;
    }
    return true;
  }

  return linkImpl(options);
}

// ================================================================================================
static void dumpCodeObject(const std::string& image) {
  char fname[30];
  static std::atomic<int> index;
  sprintf(fname, "_code_object%04d.o", index++);
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "Code object saved in %s\n", fname);
  std::ofstream ofs;
  ofs.open(fname, std::ios::binary);
  ofs << image;
  ofs.close();
}

// ================================================================================================
bool Program::linkImpl(amd::option::Options* options) {
  file_type_t continueCompileFrom = FILE_TYPE_LLVMIR_BINARY;

  internal_ = (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ? true : false;

  amd_comgr_data_set_t inputs;
  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data set for linking.\n";
    return false;
  }

  bool bLinkLLVMBitcode = true;
  if (llvmBinary_.empty()) {
    continueCompileFrom = getNextCompilationStageFromBinary(options);
  }

  switch (continueCompileFrom) {
    case FILE_TYPE_CG:
    case FILE_TYPE_LLVMIR_BINARY: {
      break;
    }
    case FILE_TYPE_ASM_TEXT: {
      char* section;
      size_t sz;
      clBinary()->elfOut()->getSection(amd::Elf::SOURCE, &section, &sz);

      if (addCodeObjData(section, sz, AMD_COMGR_DATA_KIND_BC, "Assembly Text", &inputs) !=
          AMD_COMGR_STATUS_SUCCESS) {
        buildLog_ += "Error: COMGR fails to create assembly input.\n";
        amd::Comgr::destroy_data_set(inputs);
        return false;
      }

      bLinkLLVMBitcode = false;
      break;
    }
    case FILE_TYPE_ISA: {
      amd::Comgr::destroy_data_set(inputs);
      binary_t isaBinary = binary();
      if (GPU_DUMP_CODE_OBJECT) {
        dumpCodeObject(std::string{(const char*)isaBinary.first, isaBinary.second});
      }

      if (!createKernels(const_cast<void*>(isaBinary.first), isaBinary.second,
                         options->oVariables->UniformWorkGroupSize, internal_)) {
        buildLog_ += "Error: Cannot create kernels.\n";
        return false;
      }
      return true;
    }
    default:
      buildLog_ += "Error while Codegen phase: the binary is incomplete \n";
      amd::Comgr::destroy_data_set(inputs);
      return false;
  }

  // call LinkLLVMBitcode
  if (bLinkLLVMBitcode) {
    // open the bitcode libraries
    std::vector<std::string> linkOptions;

    if (options->oVariables->FP32RoundDivideSqrt) {
      linkOptions.push_back("correctly_rounded_sqrt");
    }
    if (options->oVariables->FiniteMathOnly || options->oVariables->FastRelaxedMath) {
      linkOptions.push_back("finite_only");
    }
    if (options->oVariables->UnsafeMathOpt || options->oVariables->FastRelaxedMath) {
      linkOptions.push_back("unsafe_math");
    }
    if (device().settings().lcWavefrontSize64_) {
      linkOptions.push_back("wavefrontsize64");
    }
    linkOptions.push_back("code_object_v" +
                          std::to_string(options->oVariables->LCCodeObjectVersion));

    amd_comgr_status_t status = addCodeObjData(llvmBinary_.data(), llvmBinary_.size(),
                                               AMD_COMGR_DATA_KIND_BC, "LLVM Binary", &inputs);

    amd_comgr_data_set_t linked_bc;
    bool hasLinkedBC = false;

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      status = amd::Comgr::create_data_set(&linked_bc);
    }

    bool ret = (status == AMD_COMGR_STATUS_SUCCESS);
    if (ret) {
      hasLinkedBC = true;
      ret = linkLLVMBitcode(inputs, linkOptions, options, &linked_bc);
    }

    amd::Comgr::destroy_data_set(inputs);

    if (!ret) {
      if (hasLinkedBC) {
        amd::Comgr::destroy_data_set(linked_bc);
      }
      buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
      return false;
    }

    inputs = linked_bc;
  }

  std::vector<std::string> codegenOptions;

  // TODO: Can this be fixed at the source? options->llvmOptions is a flat
  // string, but should really be a vector of strings.
  std::vector<std::string> splitLlvmOptions =
      splitSpaceSeparatedString(options->llvmOptions.c_str());
  codegenOptions.insert(codegenOptions.end(), splitLlvmOptions.begin(), splitLlvmOptions.end());

  // Set the -O#
  std::ostringstream optLevel;
  optLevel << "-O" << options->oVariables->OptLevel;
  codegenOptions.push_back(optLevel.str());

  // Pass clang options
  if (continueCompileFrom != FILE_TYPE_ASM_TEXT) {
    std::copy_if(options->clangOptions.begin(), options->clangOptions.end(),
                 std::back_inserter(codegenOptions),
                 [](const std::string& opt) { return opt.rfind("-I", 0) != 0; });
  } else {
    codegenOptions.insert(codegenOptions.end(), options->clangOptions.begin(),
                          options->clangOptions.end());
  }

  // Set whole program mode
  codegenOptions.push_back("-mllvm");
  codegenOptions.push_back("-amdgpu-internalize-symbols");

  if (!device().settings().enableWgpMode_) {
    codegenOptions.push_back("-mcumode");
  }

  if (device().settings().lcWavefrontSize64_) {
    codegenOptions.push_back("-mwavefrontsize64");
  }
  codegenOptions.push_back("-mcode-object-version=" +
                           std::to_string(options->oVariables->LCCodeObjectVersion));

  // NOTE: The params is also used to identy cached code object. This parameter
  //       should not contain any dyanamically generated filename.
  char* executable = nullptr;
  size_t executableSize = 0;
  bool ret = compileAndLinkExecutable(inputs, codegenOptions, options, &executable, &executableSize,
                                      continueCompileFrom);
  amd::Comgr::destroy_data_set(inputs);

  if (!ret) {
    if (continueCompileFrom == FILE_TYPE_ASM_TEXT) {
      buildLog_ += "Error: Creating the executable from ISA assembly text failed.\n";
    } else {
      buildLog_ += "Error: Creating the executable from LLVM IRs failed.\n";
    }
    return false;
  }

  // Save the binary and type
  clBinary()->saveBIFBinary(executable, executableSize);

  // Destroy original memory with executable after compilation
  delete[] executable;

  if (!createKernels(const_cast<void*>(clBinary()->data().first), clBinary()->data().second,
                     options->oVariables->UniformWorkGroupSize, internal_)) {
    buildLog_ += "Error: Cannot create kernels.\n";
    return false;
  }

  setType(TYPE_EXECUTABLE);

  return true;
}

// ================================================================================================
bool Program::initClBinary() {
  if (clBinary_ == nullptr) {
    clBinary_ = new ClBinary(device());
    if (clBinary_ == nullptr) {
      return false;
    }
  }
  return true;
}

// ================================================================================================
void Program::releaseClBinary() {
  delete clBinary_;
  clBinary_ = nullptr;
}

// ================================================================================================
bool Program::initBuild(amd::option::Options* options) {
  compileOptions_ = options->origOptionStr;
  programOptions_ = options;

  if (options->oVariables->DumpFlags > 0) {
    static std::atomic<uint> build_num{0};
    options->setBuildNo(build_num++);
  }
  buildLog_.clear();
  if (!initClBinary()) {
    DevLogError("Init CL Binary failed \n");
    return false;
  }

  if (!amd::IS_HIP) {
    std::string targetID = device().isa().targetId();
#if defined(_WIN32)
    // Replace special charaters that are not supported by Windows FS.
    std::replace(targetID.begin(), targetID.end(), ':', '@');
#endif
    options->setPerBuildInfo(targetID.c_str(), clBinary()->getEncryptCode(), true);
  }

  // Elf Binary setup
  std::string outFileName;
  bool tempFile = false;

  // true means hsail required
  clBinary()->init(options);
  if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
    outFileName = options->getDumpFileName(".bin");
  } else {
    // elf lib needs a writable temp file
    outFileName = amd::Os::getTempFileName();
    tempFile = true;
  }

  if (!clBinary()->setElfOut(LP64_SWITCH(ELFCLASS32, ELFCLASS64),
                             (outFileName.size() > 0) ? outFileName.c_str() : nullptr, tempFile)) {
    LogError("Setup elf out for gpu failed");
    return false;
  }

  return true;
}

// ================================================================================================
bool Program::finiBuild(bool isBuildGood) {
  clBinary()->resetElfOut();
  clBinary()->resetElfIn();

  if (!isBuildGood) {
    // Prevent the encrypted binary form leaking out
    clBinary()->setBinary(nullptr, 0);
  }

  return true;
}

// ================================================================================================
int32_t Program::compile(const std::string& sourceCode,
                         const std::vector<const std::string*>& headers,
                         const char** headerIncludeNames, const char* origOptions,
                         amd::option::Options* options) {
  uint64_t start_time = 0;
  if (options->oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  lastBuildOptionsArg_ = origOptions ? origOptions : "";
  if (options) {
    compileOptions_ = options->origOptionStr;
  }

  buildStatus_ = CL_BUILD_IN_PROGRESS;
  if (!initBuild(options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation init failed.";
    }
  }

  if (options->oVariables->FP32RoundDivideSqrt &&
      !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
    buildStatus_ = CL_BUILD_ERROR;
    buildLog_ +=
        "Error: -cl-fp32-correctly-rounded-divide-sqrt "
        "specified without device support";
  }

  // Compile the source code if any
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !sourceCode.empty() &&
      !compileImpl(sourceCode, headers, headerIncludeNames, options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation failed.";
    }
  }

  setType(TYPE_COMPILED);

  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !createBinary(options)) {
    buildLog_ += "Internal Error: creating OpenCL binary failed!\n";
  }

  if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation fini failed.";
    }
  }

  if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
    buildStatus_ = CL_BUILD_SUCCESS;
  } else {
    buildError_ = CL_COMPILE_PROGRAM_FAILURE;
  }

  if (options->oVariables->EnableBuildTiming) {
    std::stringstream tmp_ss;
    tmp_ss << "\nTotal Compile Time: " << (amd::Os::timeNanos() - start_time) / 1000ULL << " us\n";
    buildLog_ += tmp_ss.str();
  }

  if (options->oVariables->BuildLog && !buildLog_.empty()) {
    if (strcmp(options->oVariables->BuildLog, "stderr") == 0) {
      fprintf(stderr, "%s\n", options->optionsLog().c_str());
      fprintf(stderr, "%s\n", buildLog_.c_str());
    } else if (strcmp(options->oVariables->BuildLog, "stdout") == 0) {
      printf("%s\n", options->optionsLog().c_str());
      printf("%s\n", buildLog_.c_str());
    } else {
      std::fstream f;
      std::stringstream tmp_ss;
      std::string logs = options->optionsLog() + buildLog_;
      tmp_ss << options->oVariables->BuildLog << "." << options->getBuildNo();
      f.open(tmp_ss.str().c_str(), (std::fstream::out | std::fstream::binary));
      f.write(logs.data(), logs.size());
      f.close();
    }
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
int32_t Program::link(const std::vector<Program*>& inputPrograms, const char* origLinkOptions,
                      amd::option::Options* linkOptions) {
  lastBuildOptionsArg_ = origLinkOptions ? origLinkOptions : "";
  if (linkOptions) {
    linkOptions_ = linkOptions->origOptionStr;
  }

  buildStatus_ = CL_BUILD_IN_PROGRESS;

  amd::option::Options options;
  if (!getCompileOptionsAtLinking(inputPrograms, linkOptions)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Get compile options failed.";
    }
  } else {
    if (!amd::option::parseAllOptions(compileOptions_, options, false)) {
      buildStatus_ = CL_BUILD_ERROR;
      buildLog_ += options.optionsLog();
      LogError("Parsing compile options failed.");
    }
  }

  uint64_t start_time = 0;
  if (options.oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  // initBuild() will clear buildLog_, so store it in a temporary variable
  std::string tmpBuildLog = buildLog_;

  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !initBuild(&options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Compilation init failed.";
    }
  }

  buildLog_ += tmpBuildLog;

  if (options.oVariables->FP32RoundDivideSqrt &&
      !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
    buildStatus_ = CL_BUILD_ERROR;
    buildLog_ +=
        "Error: -cl-fp32-correctly-rounded-divide-sqrt "
        "specified without device support";
  }

  bool createLibrary = linkOptions ? linkOptions->oVariables->clCreateLibrary : false;
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !linkImpl(inputPrograms, &options, createLibrary)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Link failed.\n";
      buildLog_ += "Make sure the system setup is correct.";
    }
  }

  if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation fini failed.";
    }
  }

  if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
    buildStatus_ = CL_BUILD_SUCCESS;
  } else {
    buildError_ = CL_LINK_PROGRAM_FAILURE;
  }

  if (options.oVariables->EnableBuildTiming) {
    std::stringstream tmp_ss;
    tmp_ss << "\nTotal Link Time: " << (amd::Os::timeNanos() - start_time) / 1000ULL << " us\n";
    buildLog_ += tmp_ss.str();
  }

  if (options.oVariables->BuildLog && !buildLog_.empty()) {
    if (strcmp(options.oVariables->BuildLog, "stderr") == 0) {
      fprintf(stderr, "%s\n", options.optionsLog().c_str());
      fprintf(stderr, "%s\n", buildLog_.c_str());
    } else if (strcmp(options.oVariables->BuildLog, "stdout") == 0) {
      printf("%s\n", options.optionsLog().c_str());
      printf("%s\n", buildLog_.c_str());
    } else {
      std::fstream f;
      std::stringstream tmp_ss;
      std::string logs = options.optionsLog() + buildLog_;
      tmp_ss << options.oVariables->BuildLog << "." << options.getBuildNo();
      f.open(tmp_ss.str().c_str(), (std::fstream::out | std::fstream::binary));
      f.write(logs.data(), logs.size());
      f.close();
    }
  }

  if (!buildLog_.empty()) {
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
static std::pair<std::string, size_t> getSubstBinFileName(const char* SubstCfgFile, size_t srcHash,
                                                          size_t optHash) {
  using namespace std;
  const size_t srcAndOptHash = srcHash ^ optHash;
  ifstream cfgFile(SubstCfgFile);
  if (cfgFile.good()) {
    string line;
    while (getline(cfgFile, line)) {
      istringstream ss(line);
      size_t hash;
      ss >> setbase(16) >> hash;
      if (ss.fail() || !isspace(ss.peek())) continue;

      if (hash == srcAndOptHash || hash == srcHash) {
        ss >> ws;
        string objFileName;
        getline(ss, objFileName);  // get the rest of line with spaces
        return make_pair(objFileName, hash);
      }
    }
  } else
    return make_pair(string(), (size_t)1);
  return make_pair(string(), (size_t)0);
}

bool Program::trySubstObjFile(const char* SubstCfgFile, const std::string& sourceCode,
                              const amd::option::Options* options) {
  std::string buffer;
  std::ostringstream str(buffer);

  size_t srcHash = getOCLSourceHash(sourceCode);
  size_t optHash = getOCLOptionsHash(*options);
  auto substRes = getSubstBinFileName(SubstCfgFile, srcHash, optHash);
  if (substRes.first.empty()) {
    switch (substRes.second) {
      default:
        break;
      case 1:
        str << "Subst failure: cannot open config file " << SubstCfgFile << std::endl;
        break;
    }
    buildLog_ += str.str();
    return false;
  }

  uint8_t* binary = nullptr;
  size_t binSize = 0;
  std::ifstream binFile(substRes.first, std::ios::binary | std::ios::ate);
  if (binFile.good()) {
    binSize = binFile.tellg();
    binFile.seekg(0, std::ios::beg);
    binary = new (std::nothrow) uint8_t[binSize];
    if (binary && !binFile.read(reinterpret_cast<char*>(binary), binSize)) {
      delete[] binary;
      binary = nullptr;
    }
  }

  if (!binary) {
    buildStatus_ = CL_BUILD_ERROR;
    buildError_ = CL_BUILD_PROGRAM_FAILURE;
    str << "Subst failure: cannot read binary file " << substRes.first << '\n';
  } else {
    if (setKernels(binary, binSize)) {
      buildStatus_ = CL_BUILD_SUCCESS;
      buildError_ = 0;
      str << "Substituted program hash 0x" << std::setbase(16) << substRes.second << " with "
          << substRes.first << '\n';
    }
  }
  buildLog_ += str.str();
  return true;
}

int32_t Program::build(const std::string& sourceCode, const char* origOptions,
                       amd::option::Options* options) {
  if (AMD_OCL_SUBST_OBJFILE != NULL &&
      trySubstObjFile(AMD_OCL_SUBST_OBJFILE, sourceCode, options)) {
    return buildError();
  }

  uint64_t start_time = 0;
  if (options->oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  lastBuildOptionsArg_ = origOptions ? origOptions : "";
  if (options) {
    compileOptions_ = options->origOptionStr;
  }

  buildStatus_ = CL_BUILD_IN_PROGRESS;
  if (!initBuild(options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation init failed.";
    }
  }

  if (options->oVariables->FP32RoundDivideSqrt &&
      !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
    buildStatus_ = CL_BUILD_ERROR;
    buildLog_ +=
        "Error: -cl-fp32-correctly-rounded-divide-sqrt "
        "specified without device support";
  }

  std::vector<const std::string*> headers;
  std::vector<const char*> headerIncludeNames;
  const std::vector<std::string>& tmpHeaderNames = owner()->headerNames();
  const std::vector<std::string>& tmpHeaders = owner()->headers();
  for (size_t i = 0; i < tmpHeaders.size(); ++i) {
    headers.push_back(&tmpHeaders[i]);
    headerIncludeNames.push_back(tmpHeaderNames[i].c_str());
  }
  // Compile the source code if any
  bool compileStatus = true;
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !sourceCode.empty()) {
    if (!headerIncludeNames.empty()) {
      compileStatus = compileImpl(sourceCode, headers, &headerIncludeNames[0], options);
    } else {
      compileStatus = compileImpl(sourceCode, headers, nullptr, options);
    }
  }
  if (!compileStatus) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation failed.";
    }
  }
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !linkImpl(options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Link failed.\n";
      buildLog_ += "Make sure the system setup is correct.";
    }
  }

  if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation fini failed.";
    }
  }

  if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
    buildStatus_ = CL_BUILD_SUCCESS;
  } else {
    buildError_ = CL_BUILD_PROGRAM_FAILURE;
  }

  if (options->oVariables->EnableBuildTiming) {
    std::stringstream tmp_ss;
    tmp_ss << "\nTotal Build Time: " << (amd::Os::timeNanos() - start_time) / 1000ULL << " us\n";
    buildLog_ += tmp_ss.str();
  }

  if (options->oVariables->BuildLog && !buildLog_.empty()) {
    if (strcmp(options->oVariables->BuildLog, "stderr") == 0) {
      fprintf(stderr, "%s\n", options->optionsLog().c_str());
      fprintf(stderr, "%s\n", buildLog_.c_str());
    } else if (strcmp(options->oVariables->BuildLog, "stdout") == 0) {
      printf("%s\n", options->optionsLog().c_str());
      printf("%s\n", buildLog_.c_str());
    } else {
      std::fstream f;
      std::stringstream tmp_ss;
      std::string logs = options->optionsLog() + buildLog_;
      tmp_ss << options->oVariables->BuildLog << "." << options->getBuildNo();
      f.open(tmp_ss.str().c_str(), (std::fstream::out | std::fstream::binary));
      f.write(logs.data(), logs.size());
      f.close();
    }
  }

  if (!buildLog_.empty()) {
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
bool Program::load() {
  coLoaded_ = setKernels(const_cast<void*>(binary().first), binary().second, BinaryFd().first,
                    BinaryFd().second, BinaryURI());
  return coLoaded_;
}

// ================================================================================================
std::vector<std::string> Program::ProcessOptions(amd::option::Options* options) {
  std::vector<std::string> optionsVec;

  if (!isHIP()) {
    int major, minor;
    ::sscanf(device().info().version_, "OpenCL %d.%d ", &major, &minor);
    std::stringstream ss;
    ss << "-D__OPENCL_VERSION__=" << (major * 100 + minor * 10);
    optionsVec.push_back(ss.str());
  }

  if (!isHIP()) {
    if (device().info().imageSupport_ && options->oVariables->ImageSupport) {
      optionsVec.push_back("-D__IMAGE_SUPPORT__=1");
    }

    uint clcStd =
        (options->oVariables->CLStd[2] - '0') * 100 + (options->oVariables->CLStd[4] - '0') * 10;

    if (clcStd >= 200) {
      std::stringstream opts;
      // Add only for CL2.0 and later
      opts << "-D"
           << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=" << device().info().maxGlobalVariableSize_;
      optionsVec.push_back(opts.str());
    } else {
      options->oVariables->UniformWorkGroupSize = true;
    }

    // Tokenize the extensions string into a vector of strings
    std::istringstream istrstr(device().info().extensions_);
    std::istream_iterator<std::string> sit(istrstr), end;
    std::vector<std::string> extensions(sit, end);

    if (!extensions.empty()) {
      std::ostringstream clext;

      clext << "-cl-ext=+";
      std::copy(extensions.begin(), extensions.end() - 1,
                std::ostream_iterator<std::string>(clext, ",+"));
      clext << extensions.back();

      optionsVec.push_back("-Xclang");
      optionsVec.push_back(clext.str());
    }
  }

  return optionsVec;
}

std::string Program::ProcessOptionsFlattened(amd::option::Options* options) {
  std::vector<std::string> processOptions = ProcessOptions(options);
  std::ostringstream processOptionsOStrStr;
  processOptionsOStrStr << " ";
  std::copy(processOptions.begin(), processOptions.end(),
            std::ostream_iterator<std::string>(processOptionsOStrStr, " "));
  return processOptionsOStrStr.str();
}

// ================================================================================================
bool Program::getCompileOptionsAtLinking(const std::vector<Program*>& inputPrograms,
                                         const amd::option::Options* linkOptions) {
  amd::option::Options compileOptions;
  auto it = inputPrograms.cbegin();
  const auto itEnd = inputPrograms.cend();
  for (size_t i = 0; it != itEnd; ++it, ++i) {
    Program* program = *it;

    amd::option::Options compileOptions2;
    amd::option::Options* thisCompileOptions = i == 0 ? &compileOptions : &compileOptions2;
    if (!amd::option::parseAllOptions(program->compileOptions_, *thisCompileOptions, false)) {
      buildLog_ += thisCompileOptions->optionsLog();
      LogError("Parsing compile options failed.");
      return false;
    }

    if (i == 0) compileOptions_ = program->compileOptions_;

    // if we are linking a program executable, and if "program" is a
    // compiled module or a library created with "-enable-link-options",
    // we can overwrite "program"'s compile options with linking options
    if (!linkOptions_.empty() && !linkOptions->oVariables->clCreateLibrary) {
      bool linkOptsCanOverwrite = false;
      if (program->type() != TYPE_LIBRARY) {
        linkOptsCanOverwrite = true;
      } else {
        amd::option::Options thisLinkOptions;
        if (!amd::option::parseLinkOptions(program->linkOptions_, thisLinkOptions)) {
          buildLog_ += thisLinkOptions.optionsLog();
          LogError("Parsing link options failed.");
          return false;
        }
        if (thisLinkOptions.oVariables->clEnableLinkOptions) linkOptsCanOverwrite = true;
      }
      if (linkOptsCanOverwrite) {
        if (!thisCompileOptions->setOptionVariablesAs(*linkOptions)) {
          buildLog_ += thisCompileOptions->optionsLog();
          LogError("Setting link options failed.");
          return false;
        }
      }
      if (i == 0) compileOptions_ += " " + linkOptions_;
    }
    // warn if input modules have inconsistent compile options
    if (i > 0) {
      if (!compileOptions.equals(*thisCompileOptions, true /*ignore clc options*/)) {
        buildLog_ +=
            "Warning: Input OpenCL binaries has inconsistent"
            " compile options. Using compile options from"
            " the first input binary!\n";
      }
    }
  }
  return true;
}

// ================================================================================================
bool Program::initClBinary(const char* binaryIn, size_t size, amd::Os::FileDesc fdesc,
                           size_t foffset, std::string uri) {
  if (!initClBinary()) {
    DevLogError("Init CL Binary failed \n");
    return false;
  }

  // Save the original binary that isn't owned by ClBinary
  clBinary()->saveOrigBinary(binaryIn, size);

  const char* bin = binaryIn;
  size_t sz = size;

  // unencrypted
  int encryptCode = 0;
  char* decryptedBin = nullptr;

  size_t decryptedSize;
  if (!clBinary()->decryptElf(binaryIn, size, &decryptedBin, &decryptedSize, &encryptCode)) {
    DevLogError("Cannot Decrypt Elf \n");
    return false;
  }
  if (decryptedBin != nullptr) {
    // It is decrypted binary.
    bin = decryptedBin;
    sz = decryptedSize;
  }

  if (!isElf(bin)) {
    // Invalid binary.
    delete[] decryptedBin;
    DevLogError("Bin is not ELF \n");
    return false;
  }

  clBinary()->setFlags(encryptCode);

  return clBinary()->setBinary(bin, sz, (decryptedBin != nullptr), fdesc, foffset, uri);
}

// ================================================================================================
void Program::addKernel(Kernel* k) {
  kernels_[k->name()] = k;
  if (k->isInitKernel()) {
    initKernels_.push_back(k);
  } else if (k->isFiniKernel()) {
    finiKernels_.push_back(k);
  }
}

// ================================================================================================
bool Program::setBinary(const char* binaryIn, size_t size, const device::Program* same_dev_prog,
                        amd::Os::FileDesc fdesc, size_t foffset, std::string uri) {
  if (!initClBinary(binaryIn, size, fdesc, foffset, uri)) {
    DevLogError("Init CL Binary failed \n");
    return false;
  }

  if (!clBinary()->setElfIn()) {
    LogError("Setting input OCL binary failed");
    return false;
  }
  uint16_t type;
  if (!clBinary()->elfIn()->getType(type)) {
    LogError("Bad OCL Binary: error loading ELF type!");
    return false;
  }
  switch (type) {
    case ET_NONE: {
      setType(TYPE_NONE);
      break;
    }
    case ET_REL: {
      if (clBinary()->isSPIR() || clBinary()->isSPIRV()) {
        setType(TYPE_INTERMEDIATE);
      } else {
        setType(TYPE_COMPILED);
      }
      break;
    }
    case ET_DYN: {
      char* sect = nullptr;
      size_t sz = 0;
      if (clBinary()->elfIn()->isHsaCo()) {
        setType(TYPE_EXECUTABLE);
      } else {
        setType(TYPE_LIBRARY);
      }
      break;
    }
    case ET_EXEC: {
      setType(TYPE_EXECUTABLE);
      break;
    }
    default:
      LogError("Bad OCL Binary: bad ELF type!");
      return false;
  }

  if (same_dev_prog != nullptr) {
    compileOptions_ = same_dev_prog->compileOptions();
    linkOptions_ = same_dev_prog->linkOptions();
  } else if (!amd::IS_HIP) {
    compileOptions_.clear();
    linkOptions_.clear();
  }

  clBinary()->resetElfIn();
  return true;
}

// ================================================================================================
Program::file_type_t Program::getCompilationStagesFromBinary(
    std::vector<Program::file_type_t>& completeStages, bool& needOptionsCheck) {
  Program::file_type_t from = FILE_TYPE_DEFAULT;
  completeStages.clear();
  needOptionsCheck = true;
  //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
  // Checking llvmir in .llvmir section
  bool containsLlvmirText = (type() == TYPE_COMPILED);
  bool containsShaderIsa = (type() == TYPE_EXECUTABLE);
  bool containsOpts = !(compileOptions_.empty() && linkOptions_.empty());

  if (containsLlvmirText && containsOpts) {
    completeStages.push_back(from);
    from = FILE_TYPE_LLVMIR_BINARY;
  }
  if (containsShaderIsa) {
    completeStages.push_back(from);
    from = FILE_TYPE_ISA;
  }
  std::string sCurOptions = compileOptions_ + linkOptions_;
  amd::option::Options curOptions;
  if (!amd::option::parseAllOptions(sCurOptions, curOptions, false)) {
    buildLog_ += curOptions.optionsLog();
    LogError("Parsing compile options failed.");
    return FILE_TYPE_DEFAULT;
  }
  switch (from) {
    case FILE_TYPE_CG:
    case FILE_TYPE_ISA:
      // do not check options, if LLVMIR is absent or might be absent or options are absent
      if (!curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
        needOptionsCheck = false;
      }
      break;
      // recompilation might be needed
    case FILE_TYPE_LLVMIR_BINARY:
    case FILE_TYPE_DEFAULT:
    default:
      break;
  }
  return from;
}

// ================================================================================================
Program::file_type_t Program::getNextCompilationStageFromBinary(amd::option::Options* options) {
  Program::file_type_t continueCompileFrom = FILE_TYPE_DEFAULT;
  binary_t binary = this->binary();
  finfo_t finfo = this->BinaryFd();
  std::string uri = this->BinaryURI();
  // If the binary already exists
  if ((binary.first != nullptr) && (binary.second > 0)) {
    // save the current options
    std::string sCurCompileOptions = compileOptions_;
    std::string sCurLinkOptions = linkOptions_;
    std::string sCurOptions = compileOptions_ + linkOptions_;

    // Saving binary in the interface class,
    // which also load compile & link options from binary
    setBinary(static_cast<const char*>(binary.first), binary.second, nullptr, finfo.first,
              finfo.second, uri);

    // Calculate the next stage to compile from, based on sections in binaryElf_;
    // No any validity checks here
    std::vector<file_type_t> completeStages;
    bool needOptionsCheck = true;
    continueCompileFrom = getCompilationStagesFromBinary(completeStages, needOptionsCheck);
    if (!options || !needOptionsCheck) {
      return continueCompileFrom;
    }
    bool recompile = false;
    //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
    switch (continueCompileFrom) {
      case FILE_TYPE_HSAIL_BINARY:
      case FILE_TYPE_CG:
      case FILE_TYPE_ISA: {
        // Compare options loaded from binary with current ones, recompile if differ;
        // If compile options are absent in binary, do not compare and recompile
        if (compileOptions_.empty()) break;

        compileOptions_ = sCurCompileOptions;
        linkOptions_ = sCurLinkOptions;

        amd::option::Options curOptions;
        if (!amd::option::parseAllOptions(sCurOptions, curOptions, false)) {
          buildLog_ += curOptions.optionsLog();
          LogError("Parsing compile options failed.");
          return FILE_TYPE_DEFAULT;
        }
        break;
      }
      default:
        break;
    }
    if (recompile) {
      while (!completeStages.empty()) {
        continueCompileFrom = completeStages.back();
        if (continueCompileFrom == FILE_TYPE_SPIRV_BINARY ||
            continueCompileFrom == FILE_TYPE_LLVMIR_BINARY ||
            continueCompileFrom == FILE_TYPE_SPIR_BINARY ||
            continueCompileFrom == FILE_TYPE_DEFAULT) {
          break;
        }
        completeStages.pop_back();
      }
    }
  } else {
    const char* xLang = options->oVariables->XLang;
    if (xLang != nullptr && strcmp(xLang, "asm") == 0) {
      continueCompileFrom = FILE_TYPE_ASM_TEXT;
    }
  }
  return continueCompileFrom;
}

// ================================================================================================
bool ComgrBinaryData::create(amd_comgr_data_kind_t kind, void* binary, size_t binSize) {
  amd_comgr_status_t status = amd::Comgr::create_data(kind, &binaryData_);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  created_ = true;

  status = amd::Comgr::set_data(binaryData_, binSize, reinterpret_cast<const char*>(binary));
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }

  return true;
}

amd_comgr_data_t& ComgrBinaryData::data() {
  assert(created_);
  return binaryData_;
}

ComgrBinaryData::~ComgrBinaryData() {
  if (created_) {
    amd::Comgr::release_data(binaryData_);
  }
}

bool Program::createKernelMetadataMap(void* binary, size_t binSize) {
  ComgrBinaryData binaryData;
  if (!binaryData.create(AMD_COMGR_DATA_KIND_EXECUTABLE, binary, binSize)) {
    buildLog_ += "Error: COMGR failed to create code object data object.\n";
    return false;
  }

  amd_comgr_status_t status;
  if (device().isOnline()) {
    size_t requiredSize = 0;
    status = amd::Comgr::get_data_isa_name(binaryData.data(), &requiredSize, nullptr);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "Error: COMGR failed to get code object ISA name.\n";
      return false;
    }

    std::vector<char> binaryIsaName(requiredSize);
    status = amd::Comgr::get_data_isa_name(binaryData.data(), &requiredSize, binaryIsaName.data());
    if ((status != AMD_COMGR_STATUS_SUCCESS) || (requiredSize != binaryIsaName.size())) {
      buildLog_ += "Error: COMGR failed to get code object ISA name.\n";
      return false;
    }

    const amd::Isa* binaryIsa = amd::Isa::findIsa(binaryIsaName.data());
    if (!binaryIsa) {
      buildLog_ +=
          "Error: Could not find the program ISA " + std::string(binaryIsaName.data()) + "\n";
      return false;
    }

    if (!amd::Isa::isCompatible(*binaryIsa, device().isa())) {
      buildLog_ += "Error: The program ISA " + std::string(binaryIsaName.data());
      buildLog_ += " is not compatible with the device ISA " + device().isa().isaName() + "\n";
      return false;
    }
  }

  status = amd::Comgr::get_data_metadata(binaryData.data(), &metadata_);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR failed to get the metadata.\n";
    return false;
  }

  amd_comgr_metadata_node_t kernelsMD;
  bool hasKernelMD = false;
  size_t size = 0;

  status = amd::Comgr::metadata_lookup(metadata_, "Kernels", &kernelsMD);
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "Using Code Object V2.");
    hasKernelMD = true;
    codeObjectVer_ = 2;
  } else {
    amd_comgr_metadata_node_t versionMD, versionNode;
    char major_version, minor_version;

    status = amd::Comgr::metadata_lookup(metadata_, "amdhsa.version", &versionMD);

    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "No amdhsa.version metadata found.");
      return false;
    }

    status = amd::Comgr::index_list_metadata(versionMD, 0, &versionNode);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata major version node.");
      amd::Comgr::destroy_metadata(versionMD);
      return false;
    }

    size = 1;
    status = amd::Comgr::get_metadata_string(versionNode, &size, &major_version);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata major version.");
      amd::Comgr::destroy_metadata(versionNode);
      amd::Comgr::destroy_metadata(versionMD);
      return false;
    }
    amd::Comgr::destroy_metadata(versionNode);

    status = amd::Comgr::index_list_metadata(versionMD, 1, &versionNode);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata minor version node.");
      amd::Comgr::destroy_metadata(versionMD);
      return false;
    }

    size = 1;
    status = amd::Comgr::get_metadata_string(versionNode, &size, &minor_version);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata minor version.");
      amd::Comgr::destroy_metadata(versionNode);
      amd::Comgr::destroy_metadata(versionMD);
      return false;
    }
    amd::Comgr::destroy_metadata(versionNode);

    amd::Comgr::destroy_metadata(versionMD);

    if (major_version == '1') {
      if (minor_version == '0') {
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "Using Code Object V3.");
        codeObjectVer_ = 3;
      } else if (minor_version == '1') {
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "Using Code Object V4.");
        codeObjectVer_ = 4;
      } else if (minor_version == '2') {
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "Using Code Object V5.");
        codeObjectVer_ = 5;
      } else {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "Unknown code object metadata minor version [%s.%s].", major_version,
                minor_version);
      }
    } else {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Unknown code object metadata major version [%s.%s].",
              major_version, minor_version);
    }

    status = amd::Comgr::metadata_lookup(metadata_, "amdhsa.kernels", &kernelsMD);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      hasKernelMD = true;
    }
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_metadata_list_size(kernelsMD, &size);
  } else if (amd::IS_HIP) {
    // Assume an empty binary. HIP may have binaries with just global variables
    return true;
  }

  for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
    amd_comgr_metadata_node_t nameMeta;
    bool hasNameMeta = false;
    bool hasKernelNode = false;

    amd_comgr_metadata_node_t kernelNode;

    std::string kernelName;
    status = amd::Comgr::index_list_metadata(kernelsMD, i, &kernelNode);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      hasKernelNode = true;
      status = amd::Comgr::metadata_lookup(kernelNode, (codeObjectVer() == 2) ? "Name" : ".name",
                                           &nameMeta);
    }

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      hasNameMeta = true;
      status = getMetaBuf(nameMeta, &kernelName);
    }

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      kernelMetadataMap_[kernelName] = kernelNode;
    } else {
      if (hasKernelNode) {
        amd::Comgr::destroy_metadata(kernelNode);
      }
      for (auto const& kernelMeta : kernelMetadataMap_) {
        amd::Comgr::destroy_metadata(kernelMeta.second);
      }
      kernelMetadataMap_.clear();
    }

    if (hasNameMeta) {
      amd::Comgr::destroy_metadata(nameMeta);
    }
  }

  if (hasKernelMD) {
    amd::Comgr::destroy_metadata(kernelsMD);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Program::FindGlobalVarSize(void* binary, size_t binSize) {
  // HIP doesn't need information about global variable size.
  // Hence runtime can skip expensive Elf object creation for parsing
  if (!amd::IS_HIP) {
    size_t progvarsTotalSize = 0;
    size_t dynamicSize = 0;
    size_t progvarsWriteSize = 0;

    amd::Elf elfIn(ELFCLASSNONE, reinterpret_cast<const char*>(binary), binSize, nullptr,
                   amd::Elf::ELF_C_READ);

    if (!elfIn.isSuccessful()) {
      buildLog_ += "Creating input amd::Elf object failed\n";
      return false;
    }

    auto numpHdrs = elfIn.getSegmentNum();
    for (unsigned int i = 0; i < numpHdrs; ++i) {
      amd::ELFIO::segment* seg = nullptr;
      if (!elfIn.getSegment(i, seg)) {
        continue;
      }

      // Accumulate the size of R & !X loadable segments
      if (seg->get_type() == PT_LOAD && !(seg->get_flags() & PF_X)) {
        if (seg->get_flags() & PF_R) {
          progvarsTotalSize += seg->get_memory_size();
        }
        if (seg->get_flags() & PF_W) {
          progvarsWriteSize += seg->get_memory_size();
        }
      } else if (seg->get_type() == PT_DYNAMIC) {
        dynamicSize += seg->get_memory_size();
      }
    }

    progvarsTotalSize -= dynamicSize;
    setGlobalVariableTotalSize(progvarsTotalSize);

    if (progvarsWriteSize != dynamicSize) {
      hasGlobalStores_ = true;
    }
  }

  if (!createKernelMetadataMap(binary, binSize)) {
    buildLog_ += "Error: create kernel metadata map using COMgr\n";
    return false;
  }
  return true;
}

amd_comgr_status_t getSymbolFromModule(amd_comgr_symbol_t symbol, void* userData) {
  size_t nlen = 0;
  size_t* userDataInfo = nullptr;
  amd_comgr_status_t status;
  amd_comgr_symbol_type_t type;
  std::vector<std::string>* var_names = nullptr;

  /* Unpack the user data */
  SymbolInfo* sym_info = reinterpret_cast<SymbolInfo*>(userData);

  if (!sym_info) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  /* Retrieve the symbol info */
  status = amd::Comgr::symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, &nlen);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  /* Retrieve the symbol name */
  char* name = new char[nlen + 1];
  status = amd::Comgr::symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME, name);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  /* Retrieve the symbol type*/
  status = amd::Comgr::symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_TYPE, &type);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  /* If symbol type is object(Variable) add it to vector */
  if ((std::strcmp(name, "") != 0) && (type == sym_info->sym_type)) {
    sym_info->var_names->push_back(std::string(name));
  }

  delete[] name;
  return status;
}

bool Program::getSymbolsFromCodeObj(std::vector<std::string>* var_names,
                                    amd_comgr_symbol_type_t sym_type) const {
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  amd_comgr_data_t dataObject;
  SymbolInfo sym_info;
  bool ret_val = true;

  do {
    /* Create comgr data */
    status = amd::Comgr::create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &dataObject);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot create comgr data \n";
      ret_val = false;
      break;
    }

    /* Set the binary as a dataObject */
    status = amd::Comgr::set_data(dataObject, static_cast<size_t>(clBinary_->data().second),
                                  reinterpret_cast<const char*>(clBinary_->data().first));
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot set comgr data \n";
      ret_val = false;
      break;
    }

    /* Pack the user data */
    sym_info.sym_type = sym_type;
    sym_info.var_names = var_names;

    /* Iterate through list of symbols */
    status = amd::Comgr::iterate_symbols(dataObject, getSymbolFromModule, &sym_info);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot iterate comgr symbols \n";
      ret_val = false;
      break;
    }
    amd::Comgr::release_data(dataObject);
  } while (0);

  return ret_val;
}

const bool Program::getLoweredNames(std::vector<std::string>* mangledNames) const {
  /* Iterate thru kernel names first */
  for (auto const& kernelMeta : kernelMetadataMap_) {
    mangledNames->emplace_back(kernelMeta.first);
  }

  /* Itrate thru global vars */
  if (!getSymbolsFromCodeObj(mangledNames, AMD_COMGR_SYMBOL_TYPE_OBJECT)) {
    DevLogError("Cannot get Symbols from Code Obj \n");
    return false;
  }

  return true;
}

bool Program::getDemangledName(const std::string& mangledName, std::string& demangledName) const {
  amd_comgr_data_t mangled_data;
  amd_comgr_data_t demangled_data;

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data))
    return false;

  if (AMD_COMGR_STATUS_SUCCESS !=
      amd::Comgr::set_data(mangled_data, mangledName.size(), mangledName.c_str())) {
    amd::Comgr::release_data(mangled_data);
    return false;
  }

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::demangle_symbol_name(mangled_data, &demangled_data)) {
    amd::Comgr::release_data(mangled_data);
    return false;
  }

  size_t demangled_size = 0;
  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::get_data(demangled_data, &demangled_size, NULL)) {
    amd::Comgr::release_data(mangled_data);
    amd::Comgr::release_data(demangled_data);
    return false;
  }

  demangledName.resize(demangled_size);

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::get_data(demangled_data, &demangled_size,
                                                       const_cast<char*>(demangledName.data()))) {
    amd::Comgr::release_data(mangled_data);
    amd::Comgr::release_data(demangled_data);
    return false;
  }

  amd::Comgr::release_data(mangled_data);
  amd::Comgr::release_data(demangled_data);
  return true;
}

bool Program::getGlobalFuncFromCodeObj(std::vector<std::string>* func_names) const {
  return getSymbolsFromCodeObj(func_names, AMD_COMGR_SYMBOL_TYPE_FUNC);
}

bool Program::getGlobalVarFromCodeObj(std::vector<std::string>* var_names) const {
  return getSymbolsFromCodeObj(var_names, AMD_COMGR_SYMBOL_TYPE_OBJECT);
}

// Init Fini Launch Lock
amd::Monitor Program::initFiniLock_(true);

bool Program::runInitFiniKernel(const std::vector<const Kernel*>& kernels) const {
  amd::HostQueue* queue = nullptr;

  for (const auto& kernel : kernels) {
    amd::ScopedLock sl(initFiniLock_);

    if (queue == nullptr) {
      queue = new amd::HostQueue(device_().context(), device_(), 0);
      if (queue == nullptr) {
        LogError("Unable to create queue");
        return false;
      }
      queue->create();
    }

    LogPrintfInfo("%s is marked init/fini", kernel->name().c_str());

    size_t globalWorkOffset[3] = {0};
    size_t globalWorkSize[3] = {1, 1, 1};
    size_t localWorkSize[3] = {1, 1, 1};
    amd::NDRangeContainer ndrange(3, globalWorkOffset, globalWorkSize, localWorkSize);
    amd::Command::EventWaitList waitList;

    auto symbol = owner_.findSymbol(kernel->name().c_str());
    amd::Kernel* k = new amd::Kernel(owner_, *symbol, kernel->name().c_str());
    if (!k) {
      queue->release();
      LogError("Unable to create kernel");
      return false;
    }

    amd::NDRangeKernelCommand* kernelCommand =
        new amd::NDRangeKernelCommand(*queue, waitList, *k, ndrange);
    if (!kernelCommand) {
      LogError("Unale to allocate memory to launch kernel");
      k->release();
      queue->release();
      return false;
    }
    if (CL_SUCCESS != kernelCommand->captureAndValidate()) {
      LogError("Kernel Capture and Validate failed");
      kernelCommand->release();
      k->release();
      queue->release();
      return false;
    }
    kernelCommand->enqueue();
    queue->finish();
    k->release();
    kernelCommand->release();
  }

  if (queue != nullptr) {
    queue->release();
  }
  return true;
}

bool Program::runInitKernels() { return runInitFiniKernel(initKernels_); }

bool Program::runFiniKernels() { return runInitFiniKernel(finiKernels_); }
} /* namespace amd::device*/
