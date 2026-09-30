/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/program.hpp"
#include "platform/ndrange.hpp"
#include "devprogram.hpp"
#include "devkernel.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "utils/macros.hpp"
#include "utils/options.hpp"
#include "os/os.hpp"
#include "top.hpp"
#include "elf.hpp"
#include "elfio/elf_types.hpp"
#include "elfio/elfio_segment.hpp"
#include "comgrctx.hpp"
#include "amd_comgr/amd_comgr.h"
#include "CL/cl.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace amd::device {

inline static std::vector<std::string> SplitSpaceSeparatedString(const char* str) {
  const std::string s(str);
  std::stringstream ss(s);
  const std::istream_iterator<std::string> beg(ss), end;
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
  for (auto const& kernel_meta : kernelMetadataMap_) {
    amd::Comgr::destroy_metadata(kernel_meta.second);
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

// If buildLog is not null, and data_set contains a log object, extract the
// first log data object from data_set and process it with
// extractByteCodeBinary.
void Program::extractBuildLog(amd_comgr_data_set_t data_set) {
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  size_t count = 0;
  status = amd::Comgr::action_data_count(data_set, AMD_COMGR_DATA_KIND_LOG, &count);

  if (status == AMD_COMGR_STATUS_SUCCESS && count > 0) {
    char* log_data = nullptr;
    size_t log_size = 0;
    status = extractByteCodeBinary(data_set, AMD_COMGR_DATA_KIND_LOG, "", &log_data, &log_size);
    buildLog_ += log_data;
    delete[] log_data;
  }
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Warning: extracting build log failed.\n";
  }
}

//  Extract the byte code binary from the data set.  The binary will be saved to an output
//  file if the file name is provided. If buffer pointer, outBinary, is provided, the
//  binary will be passed back to the caller.
//
amd_comgr_status_t Program::extractByteCodeBinary(const amd_comgr_data_set_t in_data_set,
                                                  const amd_comgr_data_kind_t data_kind,
                                                  const std::string& out_file_name,
                                                  char* out_binary[], size_t* out_size) {
  amd_comgr_data_t binary_data;

  amd_comgr_status_t status =
      amd::Comgr::action_data_get_data(in_data_set, data_kind, 0, &binary_data);

  size_t binary_size = 0;
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_data(binary_data, &binary_size, nullptr);
  }

  const size_t buf_size = (data_kind == AMD_COMGR_DATA_KIND_LOG) ? binary_size + 1 : binary_size;

  char* binary = new char[buf_size];
  if (binary == nullptr) {
    amd::Comgr::release_data(binary_data);
    return AMD_COMGR_STATUS_ERROR;
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_data(binary_data, &binary_size, binary);
  }

  if (data_kind == AMD_COMGR_DATA_KIND_LOG) {
    binary[binary_size] = '\0';
  }

  amd::Comgr::release_data(binary_data);

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    delete[] binary;
    return status;
  }

  // save the binary to the file as output file name is specified
  if (!out_file_name.empty()) {
    std::ofstream f(out_file_name.c_str(), std::ios::trunc | std::ios::binary);
    if (f.is_open()) {
      f.write(binary, static_cast<std::streamsize>(binary_size));
      f.close();
    } else {
      buildLog_ += "Warning: opening the file to dump the code failed.\n";
    }
  }

  if (out_binary != nullptr) {
    // Pass the dump binary and its size back to the caller
    *out_binary = binary;
    *out_size = binary_size;
  } else {
    delete[] binary;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t Program::addCodeObjData(const char* source, const size_t size,
                                           const amd_comgr_data_kind_t type, const char* name,
                                           amd_comgr_data_set_t* data_set) {
  amd_comgr_data_t data;
  amd_comgr_status_t status = amd::Comgr::create_data(type, &data);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  status = amd::Comgr::set_data(data, size, source);

  if ((name != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd::Comgr::set_data_name(data, name);
  }

  if ((data_set != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd::Comgr::data_set_add(*data_set, data);
  }

  amd::Comgr::release_data(data);

  return status;
}

static amd_comgr_language_t GetCOMGRLanguage(bool is_hip, const amd::option::Options& amd_options) {
  if (is_hip) {
    return AMD_COMGR_LANGUAGE_HIP;
  } else {
    const char* cl_std = amd_options.oVariables->CLStd;
    const uint clc_std = (cl_std[2] - '0') * 100 + (cl_std[4] - '0') * 10;

    switch (clc_std) {
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

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COMGR, "Cannot set Language version for %s \n",
          amd_options.oVariables->CLStd);
  return AMD_COMGR_LANGUAGE_NONE;
}


amd_comgr_status_t Program::createAction(const amd_comgr_language_t oclver,
                                         const std::vector<std::string>& options,
                                         amd_comgr_action_info_t* action, bool* has_action) {
  *has_action = false;
  amd_comgr_status_t status = amd::Comgr::create_action_info(action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    *has_action = true;
    if (oclver != AMD_COMGR_LANGUAGE_NONE) {
      status = amd::Comgr::action_info_set_language(*action, oclver);
    }
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::action_info_set_isa_name(*action, device().isa().isaName().c_str());
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::vector<const char*> options_argv;
    options_argv.reserve(options.size());
    for (auto& option : options) {
      options_argv.push_back(option.c_str());
    }
    status =
        amd::Comgr::action_info_set_option_list(*action, options_argv.data(), options_argv.size());
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::action_info_set_logging(*action, true);
  }

  return status;
}

bool Program::linkLLVMBitcode(const amd_comgr_data_set_t inputs,
                              const std::vector<std::string>& options,
                              amd::option::Options* amd_options, amd_comgr_data_set_t* output,
                              char* binary_data[], size_t* binary_size) {
  const amd_comgr_language_t langver = GetCOMGRLanguage(isHIP(), *amd_options);
  if (langver == AMD_COMGR_LANGUAGE_NONE) {
    return false;
  }

  //  Create the action for linking
  amd_comgr_action_info_t action;
  bool has_action = false;

  amd_comgr_status_t status = createAction(langver, options, &action, &has_action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, action, inputs, *output);
    extractBuildLog(*output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string dump_file_name;
    if (amd_options->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
      dump_file_name = amd_options->getDumpFileName("_linked.bc");
    }
    status = extractByteCodeBinary(*output, AMD_COMGR_DATA_KIND_BC, dump_file_name, binary_data,
                                   binary_size);
  }

  if (has_action) {
    amd::Comgr::destroy_action_info(action);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Program::compileToLLVMBitcode(const amd_comgr_data_set_t compile_inputs,
                                   const std::vector<std::string>& options,
                                   amd::option::Options* amd_options, char* binary_data[],
                                   size_t* binary_size, const bool link_dev_libs) {
  const amd_comgr_language_t langver = GetCOMGRLanguage(isHIP(), *amd_options);
  if (langver == AMD_COMGR_LANGUAGE_NONE) {
    return false;
  }

  //  Create the output data set
  amd_comgr_action_info_t action{};
  amd_comgr_data_set_t output{};
  const amd_comgr_data_set_t input = compile_inputs;

  bool has_action = false;
  bool has_output = false;

  amd_comgr_status_t status = createAction(langver, options, &action, &has_action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    has_output = true;
  }

  // Preprocess the source
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    if (amd_options->isDumpFlagSet(amd::option::DUMP_I)) {
      amd_comgr_data_set_t data_set_preprocessor;
      bool has_data_set_preprocessor = false;

      status = amd::Comgr::create_data_set(&data_set_preprocessor);

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        has_data_set_preprocessor = true;
        status = amd::Comgr::do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR, action, input,
                                       data_set_preprocessor);
        extractBuildLog(data_set_preprocessor);
      }

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        const std::string out_file_name = amd_options->getDumpFileName(".i");
        status =
            extractByteCodeBinary(data_set_preprocessor, AMD_COMGR_DATA_KIND_SOURCE, out_file_name);
      }

      if (has_data_set_preprocessor) {
        amd::Comgr::destroy_data_set(data_set_preprocessor);
      }
    }
  }

  //  Compiling the source codes
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
    std::string out_file_name;
    if (amd_options->isDumpFlagSet(amd::option::DUMP_BC_OPTIMIZED)) {
      out_file_name = amd_options->getDumpFileName("_optimized.bc");
    }
    status = extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, out_file_name, binary_data,
                                   binary_size);
  }

  if (has_action) {
    amd::Comgr::destroy_action_info(action);
  }

  if (has_output) {
    amd::Comgr::destroy_data_set(output);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

//  Create an executable from an input data set.  To generate the executable,
//  the input data set is converted to relocatable code, then executable binary.
//  If assembly code is required, the input data set is converted to assembly.
bool Program::compileAndLinkExecutable(const amd_comgr_data_set_t inputs,
                                       const std::vector<std::string>& options,
                                       amd::option::Options* amd_options, char* executable[],
                                       size_t* executable_size, file_type_t continue_compile_from) {
  // create the linked output
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t relocatable_data;
  bool has_action = false;
  bool has_output = false;
  bool has_relocatable_data = false;

  amd_comgr_status_t status = createAction(AMD_COMGR_LANGUAGE_NONE, options, &action, &has_action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    has_output = true;

    if ((amd_options->isDumpFlagSet(amd::option::DUMP_ISA)) ||
        (isHIP() && amd_options->origOptionStr.find("-save-temps") != std::string::npos)) {
      //  create the assembly data set
      amd_comgr_data_set_t assembly_data;
      bool has_assembly_data = false;

      status = amd::Comgr::create_data_set(&assembly_data);
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        has_assembly_data = true;
        status = amd::Comgr::do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY, action, inputs,
                                       assembly_data);
        extractBuildLog(assembly_data);
      }

      // dump the ISA
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        const std::string dump_isa_name = amd_options->getDumpFileName(".s");
        status = extractByteCodeBinary(assembly_data, AMD_COMGR_DATA_KIND_SOURCE, dump_isa_name);
      }

      if (has_assembly_data) {
        amd::Comgr::destroy_data_set(assembly_data);
      }
    }
  }

  //  Create the relocatable data set
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&relocatable_data);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    has_relocatable_data = true;
    const amd_comgr_action_kind_t kind = (continue_compile_from == FILE_TYPE_ASM_TEXT)
                                             ? AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE
                                             : AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE;
    status = amd::Comgr::do_action(kind, action, inputs, relocatable_data);
    extractBuildLog(relocatable_data);
  }

  // Create executable from the relocatable data set
  amd::Comgr::action_info_set_option_list(action, nullptr, 0);
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action,
                                   relocatable_data, output);
    extractBuildLog(output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    // Extract the executable binary
    std::string out_file_name;
    if (amd_options->isDumpFlagSet(amd::option::DUMP_O)) {
      out_file_name = amd_options->getDumpFileName(".so");
    }
    status = extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_EXECUTABLE, out_file_name,
                                   executable, executable_size);
  }

  if (has_action) {
    amd::Comgr::destroy_action_info(action);
  }

  if (has_relocatable_data) {
    amd::Comgr::destroy_data_set(relocatable_data);
  }

  if (has_output) {
    amd::Comgr::destroy_data_set(output);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

static std::size_t GetOCLSourceHash(const std::string& source_code) {
  return std::hash<std::string>()(source_code);
}

static std::size_t GetOCLOptionsHash(const amd::option::Options& options) {
  std::string opts;
  for (const std::string& s : options.clangOptions) opts.append(s);
  return std::hash<std::string>()(opts);
}

bool Program::compileImpl(const std::string& source_code,
                          const std::vector<const std::string*>& headers,
                          const char** header_include_names, amd::option::Options* options) {
  const char* x_lang = options->oVariables->XLang;
  if (x_lang != nullptr) {
    if (strcmp(x_lang, "asm") == 0) {
      clBinary()->elfOut()->addSection(amd::Elf::SOURCE, source_code.data(), source_code.size());
      return true;
    } else if (!strcmp(x_lang, "cl")) {
      buildLog_ += "Unsupported language: \"" + std::string(x_lang) + "\".\n";
      return false;
    }
  }

  // add CL source to input data set
  amd_comgr_data_set_t inputs;

  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create output buffer for LLVM bitcode.\n";
    return false;
  }

  if (addCodeObjData(source_code.c_str(), source_code.length(), AMD_COMGR_DATA_KIND_SOURCE,
                     "CompileSource", &inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data from source.\n";
    amd::Comgr::destroy_data_set(inputs);
    return false;
  }

  std::vector<std::string> driver_options;
  // Set the -O#
  std::ostringstream opt_level;
  // OptLevel holds the ASCII digit ('0'..'5', 'g', 's'), so print it as a character.
  opt_level << "-O" << static_cast<char>(options->oVariables->OptLevel);
  driver_options.push_back(opt_level.str());

  if (!isHIP()) {
    driver_options.insert(driver_options.end(), options->clangOptions.begin(),
                          options->clangOptions.end());
    // TODO: Can this be fixed at the source? options->llvmOptions is a flat
    // string, but should really be a vector of strings.
    std::vector<std::string> split_llvm_options =
        SplitSpaceSeparatedString(options->llvmOptions.c_str());
    driver_options.insert(driver_options.end(), split_llvm_options.begin(),
                          split_llvm_options.end());
  }

  std::vector<std::string> processed_options = ProcessOptions(options);
  driver_options.insert(driver_options.end(), processed_options.begin(), processed_options.end());

  // Set whole program mode
  driver_options.push_back("-mllvm");
  driver_options.push_back("-amdgpu-prelink");

  if (!device().settings().enableWgpMode_) {
    driver_options.push_back("-mcumode");
  }

  if (device().settings().lcWavefrontSize64_) {
    driver_options.push_back("-mwavefrontsize64");
  }
  driver_options.push_back("-mcode-object-version=" +
                           std::to_string(options->oVariables->LCCodeObjectVersion));

  // Iterate through each source code and dump it into tmp
  if (!headers.empty()) {
    for (size_t i = 0; i < headers.size(); ++i) {
      std::string header_include_name(header_include_names[i]);
      // replace / in path with current os's file separator
      if (amd::Os::fileSeparator() != '/') {
        for (auto& it : header_include_name) {
          if (it == '/') it = amd::Os::fileSeparator();
        }
      }
      if (addCodeObjData(headers[i]->c_str(), headers[i]->length(), AMD_COMGR_DATA_KIND_INCLUDE,
                         header_include_name.c_str(), &inputs) != AMD_COMGR_STATUS_SUCCESS) {
        buildLog_ += "Error: COMGR fails to add headers into inputs.\n";
        amd::Comgr::destroy_data_set(inputs);
        return false;
      }
    }
  }

  if (!isHIP() && options->isDumpFlagSet(amd::option::DUMP_CL)) {
    std::ostringstream driver_options_str;
    std::copy(driver_options.begin(), driver_options.end(),
              std::ostream_iterator<std::string>(driver_options_str, " "));

    std::ofstream f(options->getDumpFileName(".cl").c_str(), std::ios::trunc);
    if (f.is_open()) {
      auto src_hash = GetOCLSourceHash(source_code);
      auto opt_hash = GetOCLOptionsHash(*options);

      f << "/* Compiler options:\n"
           "-c -emit-llvm -target amdgcn-amd-amdhsa -x cl "
        << driver_options_str.str() << " -include opencl-c.h "
        << "\nHash to override:"
        << "\n  Source: 0x" << std::setbase(16) << src_hash << "\n  Source + clang options: 0x"
        << (src_hash ^ opt_hash) << "\n*/\n\n"
        << source_code;
      f.close();
    } else {
      buildLog_ += "Warning: opening the file to dump the OpenCL source failed.\n";
    }
  }

  // Append Options provided by user to driver options
  if (isHIP()) {
    if (options->origOptionStr.size()) {
      std::istringstream user_options{options->origOptionStr};
      std::copy(std::istream_iterator<std::string>(user_options),
                std::istream_iterator<std::string>(), std::back_inserter(driver_options));
    }
  }

  // Compile source to IR
  char* binary_data = nullptr;
  size_t binary_size = 0;
  const bool ret =
      compileToLLVMBitcode(inputs, driver_options, options, &binary_data, &binary_size);
  if (ret) {
    llvmBinary_.assign(binary_data, binary_size);
    // Destroy the original LLVM binary, received after compilation
    delete[] binary_data;

    elfSectionType_ = amd::Elf::LLVMIR;

    if (clBinary()->saveSOURCE()) {
      clBinary()->elfOut()->addSection(amd::Elf::SOURCE, source_code.data(), source_code.size());
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
bool Program::linkImpl(const std::vector<Program*>& input_programs, amd::option::Options* options,
                       bool create_library) {
  amd_comgr_data_set_t inputs;

  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data set.\n";
    return false;
  }

  size_t idx = 0;
  for (auto program : input_programs) {
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
      const std::string llvm_name = "LLVM Binary " + std::to_string(idx);
      result = (addCodeObjData(program->llvmBinary_.data(), program->llvmBinary_.size(),
                               AMD_COMGR_DATA_KIND_BC, llvm_name.c_str(),
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
  char* binary_data = nullptr;
  size_t binary_size = 0;
  const std::vector<std::string> link_options;
  const bool ret =
      linkLLVMBitcode(inputs, link_options, options, &output, &binary_data, &binary_size);

  amd::Comgr::destroy_data_set(output);
  amd::Comgr::destroy_data_set(inputs);

  if (!ret) {
    buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
    return false;
  }

  llvmBinary_.assign(binary_data, binary_size);

  // Destroy llvm binary, received after compilation
  delete[] binary_data;

  elfSectionType_ = amd::Elf::LLVMIR;

  if (clBinary()->saveLLVMIR()) {
    clBinary()->elfOut()->addSection(amd::Elf::LLVMIR, llvmBinary_.data(), llvmBinary_.size());
  }

  // skip the rest if we are building an opencl library
  if (create_library) {
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
static void DumpCodeObject(const std::string& image) {
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
  file_type_t continue_compile_from = FILE_TYPE_LLVMIR_BINARY;

  internal_ = (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ? true : false;

  amd_comgr_data_set_t inputs;
  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data set for linking.\n";
    return false;
  }

  bool link_llvm_bitcode = true;
  if (llvmBinary_.empty()) {
    continue_compile_from = getNextCompilationStageFromBinary(options);
  }

  switch (continue_compile_from) {
    case FILE_TYPE_CG:
    case FILE_TYPE_LLVMIR_BINARY: {
      break;
    }
    case FILE_TYPE_ASM_TEXT: {
      char* section = nullptr;
      size_t sz = 0;
      clBinary()->elfOut()->getSection(amd::Elf::SOURCE, &section, &sz);

      if (addCodeObjData(section, sz, AMD_COMGR_DATA_KIND_BC, "Assembly Text", &inputs) !=
          AMD_COMGR_STATUS_SUCCESS) {
        buildLog_ += "Error: COMGR fails to create assembly input.\n";
        amd::Comgr::destroy_data_set(inputs);
        return false;
      }

      link_llvm_bitcode = false;
      break;
    }
    case FILE_TYPE_ISA: {
      amd::Comgr::destroy_data_set(inputs);
      const binary_t isa_binary = binary();
      if (GPU_DUMP_CODE_OBJECT) {
        DumpCodeObject(std::string{(const char*)isa_binary.first, isa_binary.second});
      }

      if (!createKernels(const_cast<void*>(isa_binary.first), isa_binary.second,
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
  if (link_llvm_bitcode) {
    // open the bitcode libraries
    std::vector<std::string> link_options;

    if (options->oVariables->FP32RoundDivideSqrt) {
      link_options.push_back("correctly_rounded_sqrt");
    }
    if (options->oVariables->FiniteMathOnly || options->oVariables->FastRelaxedMath) {
      link_options.push_back("finite_only");
    }
    if (options->oVariables->UnsafeMathOpt || options->oVariables->FastRelaxedMath) {
      link_options.push_back("unsafe_math");
    }
    if (device().settings().lcWavefrontSize64_) {
      link_options.push_back("wavefrontsize64");
    }
    link_options.push_back("code_object_v" +
                           std::to_string(options->oVariables->LCCodeObjectVersion));

    amd_comgr_status_t status = addCodeObjData(llvmBinary_.data(), llvmBinary_.size(),
                                               AMD_COMGR_DATA_KIND_BC, "LLVM Binary", &inputs);

    amd_comgr_data_set_t linked_bc;
    bool has_linked_bc = false;

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      status = amd::Comgr::create_data_set(&linked_bc);
    }

    bool ret = (status == AMD_COMGR_STATUS_SUCCESS);
    if (ret) {
      has_linked_bc = true;
      ret = linkLLVMBitcode(inputs, link_options, options, &linked_bc);
    }

    amd::Comgr::destroy_data_set(inputs);

    if (!ret) {
      if (has_linked_bc) {
        amd::Comgr::destroy_data_set(linked_bc);
      }
      buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
      return false;
    }

    inputs = linked_bc;
  }

  std::vector<std::string> codegen_options;

  // TODO: Can this be fixed at the source? options->llvmOptions is a flat
  // string, but should really be a vector of strings.
  std::vector<std::string> split_llvm_options =
      SplitSpaceSeparatedString(options->llvmOptions.c_str());
  codegen_options.insert(codegen_options.end(), split_llvm_options.begin(),
                         split_llvm_options.end());

  // Set the -O#
  std::ostringstream opt_level;
  // OptLevel holds the ASCII digit ('0'..'5', 'g', 's'), so print it as a character.
  opt_level << "-O" << static_cast<char>(options->oVariables->OptLevel);
  codegen_options.push_back(opt_level.str());

  // Pass clang options
  if (continue_compile_from != FILE_TYPE_ASM_TEXT) {
    std::copy_if(options->clangOptions.begin(), options->clangOptions.end(),
                 std::back_inserter(codegen_options),
                 [](const std::string& opt) { return opt.rfind("-I", 0) != 0; });
  } else {
    codegen_options.insert(codegen_options.end(), options->clangOptions.begin(),
                           options->clangOptions.end());
  }

  // Set whole program mode
  codegen_options.push_back("-mllvm");
  codegen_options.push_back("-amdgpu-internalize-symbols");

  if (!device().settings().enableWgpMode_) {
    codegen_options.push_back("-mcumode");
  }

  if (device().settings().lcWavefrontSize64_) {
    codegen_options.push_back("-mwavefrontsize64");
  }
  codegen_options.push_back("-mcode-object-version=" +
                            std::to_string(options->oVariables->LCCodeObjectVersion));

  // NOTE: The params is also used to identy cached code object. This parameter
  //       should not contain any dyanamically generated filename.
  char* executable = nullptr;
  size_t executable_size = 0;
  const bool ret = compileAndLinkExecutable(inputs, codegen_options, options, &executable,
                                            &executable_size, continue_compile_from);
  amd::Comgr::destroy_data_set(inputs);

  if (!ret) {
    if (continue_compile_from == FILE_TYPE_ASM_TEXT) {
      buildLog_ += "Error: Creating the executable from ISA assembly text failed.\n";
    } else {
      buildLog_ += "Error: Creating the executable from LLVM IRs failed.\n";
    }
    return false;
  }

  // Save the binary and type
  clBinary()->saveBIFBinary(executable, executable_size);

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
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN, "Init CL Binary failed \n");
    return false;
  }

  if (!amd::IS_HIP) {
    const std::string target_id = [this]() {
      std::string id = device().isa().targetId();
#if defined(_WIN32)
      // Replace special charaters that are not supported by Windows FS.
      std::replace(id.begin(), id.end(), ':', '@');
#endif
      return id;
    }();
    options->setPerBuildInfo(target_id.c_str(), clBinary()->getEncryptCode(), true);
  }

  // Elf Binary setup
  std::string out_file_name;
  bool temp_file = false;

  // true means hsail required
  clBinary()->init(options);
  if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
    out_file_name = options->getDumpFileName(".bin");
  } else {
    // elf lib needs a writable temp file
    out_file_name = amd::Os::getTempFileName();
    temp_file = true;
  }

  if (!clBinary()->setElfOut(LP64_SWITCH(ELFCLASS32, ELFCLASS64),
                             (out_file_name.size() > 0) ? out_file_name.c_str() : nullptr,
                             temp_file)) {
    LogError("Setup elf out for gpu failed");
    return false;
  }

  return true;
}

// ================================================================================================
bool Program::finiBuild(bool is_build_good) {
  clBinary()->resetElfOut();
  clBinary()->resetElfIn();

  if (!is_build_good) {
    // Prevent the encrypted binary form leaking out
    clBinary()->setBinary(nullptr, 0);
  }

  return true;
}

// ================================================================================================
int32_t Program::compile(const std::string& source_code,
                         const std::vector<const std::string*>& headers,
                         const char** header_include_names, const char* orig_options,
                         amd::option::Options* options) {
  uint64_t start_time = 0;
  if (options->oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  lastBuildOptionsArg_ = orig_options ? orig_options : "";
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
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !source_code.empty() &&
      !compileImpl(source_code, headers, header_include_names, options)) {
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
      f.write(logs.data(), static_cast<std::streamsize>(logs.size()));
      f.close();
    }
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
int32_t Program::link(const std::vector<Program*>& input_programs, const char* orig_link_options,
                      amd::option::Options* link_options) {
  lastBuildOptionsArg_ = orig_link_options ? orig_link_options : "";
  if (link_options) {
    linkOptions_ = link_options->origOptionStr;
  }

  buildStatus_ = CL_BUILD_IN_PROGRESS;

  amd::option::Options options;
  if (!getCompileOptionsAtLinking(input_programs, link_options)) {
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
  const std::string tmp_build_log = buildLog_;

  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !initBuild(&options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Compilation init failed.";
    }
  }

  buildLog_ += tmp_build_log;

  if (options.oVariables->FP32RoundDivideSqrt &&
      !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
    buildStatus_ = CL_BUILD_ERROR;
    buildLog_ +=
        "Error: -cl-fp32-correctly-rounded-divide-sqrt "
        "specified without device support";
  }

  const bool create_library = link_options ? link_options->oVariables->clCreateLibrary : false;
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) &&
      !linkImpl(input_programs, &options, create_library)) {
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
      f.write(logs.data(), static_cast<std::streamsize>(logs.size()));
      f.close();
    }
  }

  if (!buildLog_.empty()) {
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
static std::pair<std::string, size_t> GetSubstBinFileName(const char* subst_cfg_file,
                                                          size_t src_hash, size_t opt_hash) {
  using namespace std;
  const size_t src_and_opt_hash = src_hash ^ opt_hash;
  ifstream cfg_file(subst_cfg_file);
  if (cfg_file.good()) {
    string line;
    while (getline(cfg_file, line)) {
      istringstream ss(line);
      size_t hash = 0;
      ss >> setbase(16) >> hash;
      if (ss.fail() || !isspace(ss.peek())) continue;

      if (hash == src_and_opt_hash || hash == src_hash) {
        ss >> ws;
        string obj_file_name;
        getline(ss, obj_file_name);  // get the rest of line with spaces
        return make_pair(obj_file_name, hash);
      }
    }
  } else
    return make_pair(string(), (size_t)1);
  return make_pair(string(), (size_t)0);
}

bool Program::trySubstObjFile(const char* subst_cfg_file, const std::string& source_code,
                              const amd::option::Options* options) {
  const std::string buffer;
  std::ostringstream str(buffer);

  const size_t src_hash = GetOCLSourceHash(source_code);
  const size_t opt_hash = GetOCLOptionsHash(*options);
  auto subst_res = GetSubstBinFileName(subst_cfg_file, src_hash, opt_hash);
  if (subst_res.first.empty()) {
    switch (subst_res.second) {
      default:
        break;
      case 1:
        str << "Subst failure: cannot open config file " << subst_cfg_file << std::endl;
        break;
    }
    buildLog_ += str.str();
    return false;
  }

  uint8_t* binary = nullptr;
  size_t bin_size = 0;
  std::ifstream bin_file(subst_res.first, std::ios::binary | std::ios::ate);
  if (bin_file.good()) {
    bin_size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    binary = new (std::nothrow) uint8_t[bin_size];
    if (binary &&
        !bin_file.read(reinterpret_cast<char*>(binary), static_cast<std::streamsize>(bin_size))) {
      delete[] binary;
      binary = nullptr;
    }
  }

  if (!binary) {
    buildStatus_ = CL_BUILD_ERROR;
    buildError_ = CL_BUILD_PROGRAM_FAILURE;
    str << "Subst failure: cannot read binary file " << subst_res.first << '\n';
  } else {
    if (setKernels(binary, bin_size)) {
      buildStatus_ = CL_BUILD_SUCCESS;
      buildError_ = 0;
      str << "Substituted program hash 0x" << std::setbase(16) << subst_res.second << " with "
          << subst_res.first << '\n';
    }
  }
  buildLog_ += str.str();
  return true;
}

int32_t Program::build(const std::string& source_code, const char* orig_options,
                       amd::option::Options* options) {
  if (AMD_OCL_SUBST_OBJFILE != nullptr &&
      trySubstObjFile(AMD_OCL_SUBST_OBJFILE, source_code, options)) {
    return buildError();
  }

  uint64_t start_time = 0;
  if (options->oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  lastBuildOptionsArg_ = orig_options ? orig_options : "";
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
  std::vector<const char*> header_include_names;
  const std::vector<std::string>& tmp_header_names = owner()->headerNames();
  const std::vector<std::string>& tmp_headers = owner()->headers();
  headers.reserve(tmp_headers.size());
  header_include_names.reserve(tmp_header_names.size());
  for (size_t i = 0; i < tmp_headers.size(); ++i) {
    headers.push_back(&tmp_headers[i]);
    header_include_names.push_back(tmp_header_names[i].c_str());
  }
  // Compile the source code if any
  bool compile_status = true;
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !source_code.empty()) {
    if (!header_include_names.empty()) {
      compile_status = compileImpl(source_code, headers, &header_include_names[0], options);
    } else {
      compile_status = compileImpl(source_code, headers, nullptr, options);
    }
  }
  if (!compile_status) {
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
      f.write(logs.data(), static_cast<std::streamsize>(logs.size()));
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
  std::vector<std::string> options_vec;

  if (!isHIP()) {
    // version_ has the form "OpenCL <major>.<minor> <platform specific info>". On a parse
    // failure the extraction leaves major/minor at 0, matching the previous sscanf behavior.
    int major = 0;
    int minor = 0;
    std::istringstream version(device().info().version_);
    std::string prefix;
    char dot = '\0';
    version >> prefix >> major >> dot >> minor;

    std::stringstream ss;
    ss << "-D__OPENCL_VERSION__=" << (major * 100 + minor * 10);
    options_vec.push_back(ss.str());
  }

  if (!isHIP()) {
    if (device().info().imageSupport_ && options->oVariables->ImageSupport) {
      options_vec.push_back("-D__IMAGE_SUPPORT__=1");
    }

    const uint clc_std =
        (options->oVariables->CLStd[2] - '0') * 100 + (options->oVariables->CLStd[4] - '0') * 10;

    if (clc_std >= 200) {
      std::stringstream opts;
      // Add only for CL2.0 and later
      opts << "-D"
           << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=" << device().info().maxGlobalVariableSize_;
      options_vec.push_back(opts.str());
    } else {
      options->oVariables->UniformWorkGroupSize = true;
    }

    // Tokenize the extensions string into a vector of strings
    std::istringstream istrstr(device().info().extensions_);
    const std::istream_iterator<std::string> sit(istrstr), end;
    std::vector<std::string> extensions(sit, end);

    if (!extensions.empty()) {
      std::ostringstream clext;

      clext << "-cl-ext=+";
      std::copy(extensions.begin(), extensions.end() - 1,
                std::ostream_iterator<std::string>(clext, ",+"));
      clext << extensions.back();

      options_vec.push_back("-Xclang");
      options_vec.push_back(clext.str());
    }

    // ROCM-24914 - Convert incompatible pointer types error to warning for some Adobe apps.
    // This change was made upstream but the kernels used by these apps are still affected.
    // Refer: https://github.com/llvm/llvm-project/pull/157364
    std::string app_name = {};
    std::string app_path_and_name = {};
    amd::Os::getAppPathAndFileName(app_name, app_path_and_name);
    if ((app_name == "Indigo Benchmark.exe") || (app_name == "Adobe Premiere Pro.exe") ||
        (app_name == "AfterFX.exe")) {
      options_vec.push_back("-Xclang");
      options_vec.push_back("-Wno-error=incompatible-pointer-types");
    }
  }

  return options_vec;
}

std::string Program::ProcessOptionsFlattened(amd::option::Options* options) {
  std::vector<std::string> process_options = ProcessOptions(options);
  std::ostringstream process_options_str;
  process_options_str << " ";
  std::copy(process_options.begin(), process_options.end(),
            std::ostream_iterator<std::string>(process_options_str, " "));
  return process_options_str.str();
}

// ================================================================================================
bool Program::getCompileOptionsAtLinking(const std::vector<Program*>& input_programs,
                                         const amd::option::Options* link_options) {
  amd::option::Options compile_options;
  auto it = input_programs.cbegin();
  const auto it_end = input_programs.cend();
  for (size_t i = 0; it != it_end; ++it, ++i) {
    Program* program = *it;

    amd::option::Options compile_options2;
    amd::option::Options* this_compile_options = i == 0 ? &compile_options : &compile_options2;
    if (!amd::option::parseAllOptions(program->compileOptions_, *this_compile_options, false)) {
      buildLog_ += this_compile_options->optionsLog();
      LogError("Parsing compile options failed.");
      return false;
    }

    if (i == 0) compileOptions_ = program->compileOptions_;

    // if we are linking a program executable, and if "program" is a
    // compiled module or a library created with "-enable-link-options",
    // we can overwrite "program"'s compile options with linking options
    if (!linkOptions_.empty() && !link_options->oVariables->clCreateLibrary) {
      bool link_opts_can_overwrite = false;
      if (program->type() != TYPE_LIBRARY) {
        link_opts_can_overwrite = true;
      } else {
        amd::option::Options this_link_options;
        if (!amd::option::parseLinkOptions(program->linkOptions_, this_link_options)) {
          buildLog_ += this_link_options.optionsLog();
          LogError("Parsing link options failed.");
          return false;
        }
        if (this_link_options.oVariables->clEnableLinkOptions) link_opts_can_overwrite = true;
      }
      if (link_opts_can_overwrite) {
        if (!this_compile_options->setOptionVariablesAs(*link_options)) {
          buildLog_ += this_compile_options->optionsLog();
          LogError("Setting link options failed.");
          return false;
        }
      }
      if (i == 0) compileOptions_ += " " + linkOptions_;
    }
    // warn if input modules have inconsistent compile options
    if (i > 0) {
      if (!compile_options.equals(*this_compile_options, true /*ignore clc options*/)) {
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
bool Program::initClBinary(const char* binary_in, size_t size, amd::Os::FileDesc fdesc,
                           size_t foffset, std::string uri) {
  if (!initClBinary()) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN, "Init CL Binary failed \n");
    return false;
  }

  // Save the original binary that isn't owned by ClBinary
  clBinary()->saveOrigBinary(binary_in, size);

  const char* bin = binary_in;
  size_t sz = size;

  // unencrypted
  int encrypt_code = 0;
  char* decrypted_bin = nullptr;

  size_t decrypted_size = 0;
  if (!clBinary()->decryptElf(binary_in, size, &decrypted_bin, &decrypted_size, &encrypt_code)) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN, "Bin is not ELF \n");
    return false;
  }
  if (decrypted_bin != nullptr) {
    // It is decrypted binary.
    bin = decrypted_bin;
    sz = decrypted_size;
  }

  if (!isElf(bin)) {
    // Invalid binary.
    delete[] decrypted_bin;
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN, "Bin is not ELF \n");
    return false;
  }

  clBinary()->setFlags(encrypt_code);

  return clBinary()->setBinary(bin, sz, (decrypted_bin != nullptr), fdesc, foffset, uri);
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
bool Program::setBinary(const char* binary_in, size_t size, const device::Program* same_dev_prog,
                        amd::Os::FileDesc fdesc, size_t foffset, std::string uri) {
  if (!initClBinary(binary_in, size, fdesc, foffset, uri)) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN, "Init CL Binary failed \n");
    return false;
  }

  // Do not rely on amd::Elf to do validations here since it makes copies
  // and depending on the binary size, that might be costly.
  auto [binary, bin_size] = clBinary()->data();
  if (bin_size < sizeof(amd::Elf64_Ehdr)) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
            "The elf size is way too small to validate: %d \n", bin_size);
    return false;
  }

  // So we do the validation by explicitly getting the ELF header. Copy it into an
  // aligned local, since the binary buffer has no alignment guarantee.
  amd::Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, binary, sizeof(ehdr));
  const uint16_t type = ehdr.e_type;

  switch (type) {
    case ET_NONE: {
      setType(TYPE_NONE);
      break;
    }
    case ET_REL: {
      // isSPIR()/isSPIRV() inspect ELF sections, which requires elfIn_ to be
      // set up. Do it only for this (relocatable) case so the common
      // executable path avoids the amd::Elf copy cost.
      if (!clBinary()->setElfIn()) {
        // setElfIn() already logs on the amd::Elf failure path.
        return false;
      }
      if (clBinary()->isSPIR() || clBinary()->isSPIRV()) {
        setType(TYPE_INTERMEDIATE);
      } else {
        setType(TYPE_COMPILED);
      }
      clBinary()->resetElfIn();
      break;
    }
    case ET_DYN: {
      if (ehdr.e_machine == EM_AMDGPU) {
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

  return true;
}

// ================================================================================================
Program::file_type_t Program::getCompilationStagesFromBinary(
    std::vector<Program::file_type_t>& complete_stages, bool& need_options_check) {
  Program::file_type_t from = FILE_TYPE_DEFAULT;
  complete_stages.clear();
  need_options_check = true;
  //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
  // Checking llvmir in .llvmir section
  const bool contains_llvmir_text = (type() == TYPE_COMPILED);
  const bool contains_shader_isa = (type() == TYPE_EXECUTABLE);
  const bool contains_opts = !(compileOptions_.empty() && linkOptions_.empty());

  if (contains_llvmir_text && contains_opts) {
    complete_stages.push_back(from);
    from = FILE_TYPE_LLVMIR_BINARY;
  }
  if (contains_shader_isa) {
    complete_stages.push_back(from);
    from = FILE_TYPE_ISA;
  }
  std::string cur_options_str = compileOptions_ + linkOptions_;
  amd::option::Options cur_options;
  if (!amd::option::parseAllOptions(cur_options_str, cur_options, false)) {
    buildLog_ += cur_options.optionsLog();
    LogError("Parsing compile options failed.");
    return FILE_TYPE_DEFAULT;
  }
  switch (from) {
    case FILE_TYPE_CG:
    case FILE_TYPE_ISA:
      // do not check options, if LLVMIR is absent or might be absent or options are absent
      if (!cur_options.oVariables->BinLLVMIR || !contains_llvmir_text || !contains_opts) {
        need_options_check = false;
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
  Program::file_type_t continue_compile_from = FILE_TYPE_DEFAULT;
  const binary_t binary = this->binary();
  const finfo_t finfo = this->BinaryFd();
  const std::string uri = this->BinaryURI();
  // If the binary already exists
  if ((binary.first != nullptr) && (binary.second > 0)) {
    // save the current options
    const std::string cur_compile_options_str = compileOptions_;
    const std::string cur_link_options_str = linkOptions_;
    std::string cur_options_str = compileOptions_ + linkOptions_;

    // Saving binary in the interface class,
    // which also load compile & link options from binary
    setBinary(static_cast<const char*>(binary.first), binary.second, nullptr, finfo.first,
              finfo.second, uri);

    // Calculate the next stage to compile from, based on sections in binaryElf_;
    // No any validity checks here
    std::vector<file_type_t> complete_stages;
    bool need_options_check = true;
    continue_compile_from = getCompilationStagesFromBinary(complete_stages, need_options_check);
    if (!options || !need_options_check) {
      return continue_compile_from;
    }
    const bool recompile = false;
    //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
    switch (continue_compile_from) {
      case FILE_TYPE_CG:
      case FILE_TYPE_ISA: {
        // Compare options loaded from binary with current ones, recompile if differ;
        // If compile options are absent in binary, do not compare and recompile
        if (compileOptions_.empty()) break;

        compileOptions_ = cur_compile_options_str;
        linkOptions_ = cur_link_options_str;

        amd::option::Options cur_options;
        if (!amd::option::parseAllOptions(cur_options_str, cur_options, false)) {
          buildLog_ += cur_options.optionsLog();
          LogError("Parsing compile options failed.");
          return FILE_TYPE_DEFAULT;
        }
        break;
      }
      default:
        break;
    }
    if (recompile) {
      while (!complete_stages.empty()) {
        continue_compile_from = complete_stages.back();
        if (continue_compile_from == FILE_TYPE_SPIRV_BINARY ||
            continue_compile_from == FILE_TYPE_LLVMIR_BINARY ||
            continue_compile_from == FILE_TYPE_SPIR_BINARY ||
            continue_compile_from == FILE_TYPE_DEFAULT) {
          break;
        }
        complete_stages.pop_back();
      }
    }
  } else {
    const char* x_lang = options->oVariables->XLang;
    if (x_lang != nullptr && strcmp(x_lang, "asm") == 0) {
      continue_compile_from = FILE_TYPE_ASM_TEXT;
    }
  }
  return continue_compile_from;
}

// ================================================================================================
bool ComgrBinaryData::create(amd_comgr_data_kind_t kind, void* binary, size_t bin_size) {
  amd_comgr_status_t status = amd::Comgr::create_data(kind, &binaryData_);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  created_ = true;

  status = amd::Comgr::set_data(binaryData_, bin_size, reinterpret_cast<const char*>(binary));
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

bool Program::createKernelMetadataMap(void* binary, size_t bin_size) {
  ComgrBinaryData binary_data;
  if (!binary_data.create(AMD_COMGR_DATA_KIND_EXECUTABLE, binary, bin_size)) {
    buildLog_ += "Error: COMGR failed to create code object data object.\n";
    return false;
  }

  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  if (device().isOnline()) {
    size_t required_size = 0;
    status = amd::Comgr::get_data_isa_name(binary_data.data(), &required_size, nullptr);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "Error: COMGR failed to get code object ISA name.\n";
      return false;
    }

    std::vector<char> binary_isa_name(required_size);
    status =
        amd::Comgr::get_data_isa_name(binary_data.data(), &required_size, binary_isa_name.data());
    if ((status != AMD_COMGR_STATUS_SUCCESS) || (required_size != binary_isa_name.size())) {
      buildLog_ += "Error: COMGR failed to get code object ISA name.\n";
      return false;
    }

    const amd::Isa* binary_isa = amd::Isa::findIsa(binary_isa_name.data());
    if (!binary_isa) {
      buildLog_ +=
          "Error: Could not find the program ISA " + std::string(binary_isa_name.data()) + "\n";
      return false;
    }

    if (!amd::Isa::isCompatible(*binary_isa, device().isa())) {
      buildLog_ += "Error: The program ISA " + std::string(binary_isa_name.data());
      buildLog_ += " is not compatible with the device ISA " + device().isa().isaName() + "\n";
      return false;
    }
  }

  status = amd::Comgr::get_data_metadata(binary_data.data(), &metadata_);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR failed to get the metadata.\n";
    return false;
  }

  amd_comgr_metadata_node_t kernels_md;
  bool has_kernel_md = false;
  size_t size = 0;

  status = amd::Comgr::metadata_lookup(metadata_, "Kernels", &kernels_md);
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "Using Code Object V2.");
    has_kernel_md = true;
    codeObjectVer_ = 2;
  } else {
    amd_comgr_metadata_node_t version_md, version_node;
    char major_version = '\0', minor_version = '\0';

    status = amd::Comgr::metadata_lookup(metadata_, "amdhsa.version", &version_md);

    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "No amdhsa.version metadata found.");
      return false;
    }

    status = amd::Comgr::index_list_metadata(version_md, 0, &version_node);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata major version node.");
      amd::Comgr::destroy_metadata(version_md);
      return false;
    }

    size = 1;
    status = amd::Comgr::get_metadata_string(version_node, &size, &major_version);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata major version.");
      amd::Comgr::destroy_metadata(version_node);
      amd::Comgr::destroy_metadata(version_md);
      return false;
    }
    amd::Comgr::destroy_metadata(version_node);

    status = amd::Comgr::index_list_metadata(version_md, 1, &version_node);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata minor version node.");
      amd::Comgr::destroy_metadata(version_md);
      return false;
    }

    size = 1;
    status = amd::Comgr::get_metadata_string(version_node, &size, &minor_version);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Cannot get code object metadata minor version.");
      amd::Comgr::destroy_metadata(version_node);
      amd::Comgr::destroy_metadata(version_md);
      return false;
    }
    amd::Comgr::destroy_metadata(version_node);

    amd::Comgr::destroy_metadata(version_md);

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

    status = amd::Comgr::metadata_lookup(metadata_, "amdhsa.kernels", &kernels_md);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      has_kernel_md = true;
    }
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_metadata_list_size(kernels_md, &size);
  } else if (amd::IS_HIP) {
    // Assume an empty binary. HIP may have binaries with just global variables
    return true;
  }

  for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
    amd_comgr_metadata_node_t name_meta;
    bool has_name_meta = false;
    bool has_kernel_node = false;

    amd_comgr_metadata_node_t kernel_node;

    std::string kernel_name;
    status = amd::Comgr::index_list_metadata(kernels_md, i, &kernel_node);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      has_kernel_node = true;
      status = amd::Comgr::metadata_lookup(kernel_node, (codeObjectVer() == 2) ? "Name" : ".name",
                                           &name_meta);
    }

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      has_name_meta = true;
      status = getMetaBuf(name_meta, &kernel_name);
    }

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      kernelMetadataMap_[std::move(kernel_name)] = kernel_node;
    } else {
      if (has_kernel_node) {
        amd::Comgr::destroy_metadata(kernel_node);
      }
      for (auto const& kernel_meta : kernelMetadataMap_) {
        amd::Comgr::destroy_metadata(kernel_meta.second);
      }
      kernelMetadataMap_.clear();
    }

    if (has_name_meta) {
      amd::Comgr::destroy_metadata(name_meta);
    }
  }

  if (has_kernel_md) {
    amd::Comgr::destroy_metadata(kernels_md);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Program::FindGlobalVarSize(void* binary, size_t bin_size) {
  // HIP doesn't need information about global variable size.
  // Hence runtime can skip expensive Elf object creation for parsing
  if (!amd::IS_HIP) {
    size_t progvars_total_size = 0;
    size_t dynamic_size = 0;
    size_t progvars_write_size = 0;

    const amd::Elf elf_in(ELFCLASSNONE, reinterpret_cast<const char*>(binary), bin_size, nullptr,
                          amd::Elf::ELF_C_READ);

    if (!elf_in.isSuccessful()) {
      buildLog_ += "Creating input amd::Elf object failed\n";
      return false;
    }

    auto num_phdrs = elf_in.getSegmentNum();
    for (unsigned int i = 0; i < num_phdrs; ++i) {
      amd::ELFIO::segment* seg = nullptr;
      if (!elf_in.getSegment(i, seg)) {
        continue;
      }

      // Accumulate the size of R & !X loadable segments
      if (seg->get_type() == PT_LOAD && !(seg->get_flags() & PF_X)) {
        if (seg->get_flags() & PF_R) {
          progvars_total_size += seg->get_memory_size();
        }
        if (seg->get_flags() & PF_W) {
          progvars_write_size += seg->get_memory_size();
        }
      } else if (seg->get_type() == PT_DYNAMIC) {
        dynamic_size += seg->get_memory_size();
      }
    }

    progvars_total_size -= dynamic_size;
    setGlobalVariableTotalSize(progvars_total_size);

    if (progvars_write_size != dynamic_size) {
      hasGlobalStores_ = true;
    }
  }

  if (!createKernelMetadataMap(binary, bin_size)) {
    buildLog_ += "Error: create kernel metadata map using COMgr\n";
    return false;
  }
  return true;
}

amd_comgr_status_t GetSymbolFromModule(amd_comgr_symbol_t symbol, void* user_data) {
  size_t nlen = 0;
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  amd_comgr_symbol_type_t type = AMD_COMGR_SYMBOL_TYPE_UNKNOWN;

  /* Unpack the user data */
  SymbolInfo* sym_info = reinterpret_cast<SymbolInfo*>(user_data);

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
  amd_comgr_data_t data_object;
  SymbolInfo sym_info;
  bool ret_val = true;

  do {
    /* Create comgr data */
    status = amd::Comgr::create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data_object);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot create comgr data \n";
      ret_val = false;
      break;
    }

    /* Set the binary as a data_object */
    status = amd::Comgr::set_data(data_object, static_cast<size_t>(clBinary_->data().second),
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
    status = amd::Comgr::iterate_symbols(data_object, GetSymbolFromModule, &sym_info);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot iterate comgr symbols \n";
      ret_val = false;
      break;
    }
    amd::Comgr::release_data(data_object);
  } while (0);

  return ret_val;
}

const bool Program::getLoweredNames(std::vector<std::string>* mangled_names) const {
  /* Iterate thru kernel names first */
  mangled_names->reserve(mangled_names->size() + kernelMetadataMap_.size());
  for (auto const& kernel_meta : kernelMetadataMap_) {
    mangled_names->emplace_back(kernel_meta.first);
  }

  /* Itrate thru global vars */
  if (!getSymbolsFromCodeObj(mangled_names, AMD_COMGR_SYMBOL_TYPE_OBJECT)) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_COMGR, "Cannot get Symbols from Code Obj \n");
    return false;
  }

  return true;
}

bool Program::getDemangledName(const std::string& mangled_name, std::string& demangled_name) const {
  amd_comgr_data_t mangled_data;
  amd_comgr_data_t demangled_data;

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data))
    return false;

  if (AMD_COMGR_STATUS_SUCCESS !=
      amd::Comgr::set_data(mangled_data, mangled_name.size(), mangled_name.c_str())) {
    amd::Comgr::release_data(mangled_data);
    return false;
  }

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::demangle_symbol_name(mangled_data, &demangled_data)) {
    amd::Comgr::release_data(mangled_data);
    return false;
  }

  size_t demangled_size = 0;
  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::get_data(demangled_data, &demangled_size, nullptr)) {
    amd::Comgr::release_data(mangled_data);
    amd::Comgr::release_data(demangled_data);
    return false;
  }

  demangled_name.resize(demangled_size);

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::get_data(demangled_data, &demangled_size,
                                                       const_cast<char*>(demangled_name.data()))) {
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
std::recursive_mutex Program::initFiniLock_;

bool Program::runInitFiniKernel(const std::vector<const Kernel*>& kernels) const {
  amd::HostQueue* queue = nullptr;

  for (const auto& kernel : kernels) {
    const std::scoped_lock sl(initFiniLock_);

    if (queue == nullptr) {
      queue = new amd::HostQueue(device_().context(), device_(), 0);
      if (queue == nullptr) {
        LogError("Unable to create queue");
        return false;
      }
      queue->create();
    }

    LogPrintfInfo("%s is marked init/fini", kernel->name().c_str());

    size_t global_work_offset[3] = {0};
    size_t global_work_size[3] = {1, 1, 1};
    size_t local_work_size[3] = {1, 1, 1};
    const amd::NDRangeContainer ndrange(3, global_work_offset, global_work_size, local_work_size);
    const amd::Command::EventWaitList wait_list;

    auto symbol = owner_.findSymbol(kernel->name().c_str());
    amd::Kernel* k = new amd::Kernel(owner_, *symbol, kernel->name().c_str());
    if (!k) {
      queue->release();
      LogError("Unable to create kernel");
      return false;
    }

    amd::NDRangeKernelCommand* kernel_command =
        new amd::NDRangeKernelCommand(*queue, wait_list, *k, ndrange);
    if (!kernel_command) {
      LogError("Unale to allocate memory to launch kernel");
      k->release();
      queue->release();
      return false;
    }
    if (CL_SUCCESS != kernel_command->captureOpenCLArgsAndValidate()) {
      LogError("Kernel Capture and Validate failed");
      kernel_command->release();
      k->release();
      queue->release();
      return false;
    }
    kernel_command->enqueue();
    queue->finish();
    k->release();
    kernel_command->release();
  }

  if (queue != nullptr) {
    queue->release();
  }
  return true;
}

bool Program::runInitKernels() { return runInitFiniKernel(initKernels_); }

bool Program::runFiniKernels() { return runInitFiniKernel(finiKernels_); }
} /* namespace amd::device*/
