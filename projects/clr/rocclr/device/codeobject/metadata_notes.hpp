/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device/codeobject/msgpack.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

//! Code object formats: the NT_AMDGPU_METADATA notes. No HIP or OpenCL knowledge.
namespace amd::codeobject {

/**
 * Reads the NT_AMDGPU_METADATA notes of one code object, in note order. A linked code object can
 * hold one note per input object (e.g. rocBLAS's Tensile libraries, one per kernel). They are
 * merged like comgr's mergeNoteRecords: the amdhsa.kernels lists are concatenated and
 * amdhsa.printf is taken from the first note that has it. Stricter than comgr, which drops a
 * conflicting note and loads the rest: a note whose amdhsa.version or printf list differs from
 * an earlier one fails the parse, since its printf IDs would not match. Unknown keys are skipped.
 *
 * Each kernel map is handed to the caller in place, so kernels are built during the one walk.
 */
class MetadataNotes {
 public:
  //! Called with the reader at one kernel's map; it must consume the map. @returns False if the
  //! map is malformed.
  using KernelFn = std::function<bool(MsgPackReader&)>;

  //! @param printfStrings Receives the merged amdhsa.printf list.
  //! @param kernelFn      Called for each entry of each amdhsa.kernels list.
  MetadataNotes(std::vector<std::string>& printfStrings, KernelFn kernelFn)
      : printfStrings_(printfStrings), kernelFn_(std::move(kernelFn)) {}

  //! Reads one note. @returns False if it is malformed or conflicts with the notes before it;
  //! error() says why.
  bool parse(const char* note, uint32_t noteSize);

  //! @returns True once a note had amdhsa.version.
  bool hasVersion() const { return hasVersion_; }
  //! @returns The amdhsa.version [major, minor] of the notes.
  const std::pair<uint64_t, uint64_t>& version() const { return version_; }
  //! @returns True once a note had an amdhsa.kernels list (possibly empty).
  bool hasKernels() const { return hasKernels_; }
  //! @returns The reason the last parse() failed.
  const std::string& error() const { return error_; }

 private:
  //! Reads amdhsa.version; the reader is at its value.
  bool parseVersion(MsgPackReader& reader);
  //! Reads amdhsa.printf; the reader is at its value.
  bool parsePrintf(MsgPackReader& reader);

  std::vector<std::string>& printfStrings_;  //!< Merged amdhsa.printf strings (the caller's)
  KernelFn kernelFn_;                        //!< Builds one kernel from its map
  std::pair<uint64_t, uint64_t> version_;    //!< amdhsa.version [major, minor]
  bool hasVersion_ = false;                  //!< A note had amdhsa.version
  bool hasPrintf_ = false;                   //!< A note had amdhsa.printf
  bool hasKernels_ = false;                  //!< A note had amdhsa.kernels
  std::string error_;                        //!< Why the last parse() failed
};

}  // namespace amd::codeobject
