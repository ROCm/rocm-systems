//===-- ElfParserFuzz.cpp - Fuzz Test for ELF Parser ----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Fuzz test F-002: Malformed ELF files to parser, must never crash.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/CodeObjectParser.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <cstddef>

using namespace aegisbit;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
  if (Size == 0) {
    return 0;
  }

  // Create ArrayRef from fuzz input
  llvm::ArrayRef<uint8_t> Bytes(Data, Size);

  // First, check if it looks like an AMDGPU code object
  // This exercises the validation path
  bool IsAMDGPU = CodeObjectParser::isAMDGPUCodeObject(Bytes);
  (void)IsAMDGPU;

  // Try to parse as a code object
  // The parser should handle any malformed input gracefully
  auto ParseResult = CodeObjectParser::parse(Bytes);

  if (!ParseResult) {
    // Parse failed - expected for malformed input
    llvm::consumeError(ParseResult.takeError());
    return 0;
  }

  // Successfully parsed - exercise the parsed object
  const ParsedCodeObject& CodeObj = *ParseResult;

  // Access parsed fields - should not crash
  (void)CodeObj.EMachine;
  (void)CodeObj.EFlags;
  (void)CodeObj.GPUArch;
  (void)CodeObj.TextSectionIndex;
  (void)CodeObj.RodataSectionIndex;
  (void)CodeObj.NoteSectionIndex;

  // Access kernel info
  for (const auto& Kernel : CodeObj.Kernels) {
    (void)Kernel.Name;
    (void)Kernel.CodeOffset;
    (void)Kernel.CodeSize;
    (void)Kernel.DescriptorOffset;
  }

  // Access symbol entries
  for (const auto& Sym : CodeObj.Symbols) {
    (void)Sym.Name;
    (void)Sym.Value;
    (void)Sym.Size;
    (void)Sym.Type;
  }

  // Access raw bytes (verify they're within bounds)
  if (!CodeObj.TextSection.empty()) {
    (void)CodeObj.TextSection[0];
    (void)CodeObj.TextSection.back();
  }
  if (!CodeObj.RodataSection.empty()) {
    (void)CodeObj.RodataSection[0];
    (void)CodeObj.RodataSection.back();
  }
  if (!CodeObj.NoteSection.empty()) {
    (void)CodeObj.NoteSection[0];
    (void)CodeObj.NoteSection.back();
  }

  return 0;
}
