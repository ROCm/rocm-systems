//===-- aegisbit/CodeObjectHandler.h - Unified Interface --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unified high-level interface for AMD GPU code object handling.
/// Combines parsing, modification, and building into a single interface.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_CODE_OBJECT_HANDLER_H
#define AEGISBIT_CODE_OBJECT_HANDLER_H

#include "aegisbit/CodeObjectParser.h"
#include "aegisbit/NoteMetadataHandler.h"
#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aegisbit {

/// Unified handler for AMD GPU code objects.
///
/// Provides a high-level interface for the complete instrumentation pipeline:
/// - Load code object from bytes
/// - Access and modify sections
/// - Apply instrumentation patches
/// - Build modified code object
///
/// Example usage:
/// \code
///   auto Handler = CodeObjectHandler::loadFromBytes(OriginalBytes);
///   ArrayRef<uint8_t> Text = Handler->getTextSection();
///   // ... perform instrumentation ...
///   Handler->applyPatch(PatchResult, AddtlVGPRs, AddtlSGPRs, AddtlKernarg);
///   auto NewBytes = Handler->build();
/// \endcode
class CodeObjectHandler {
public:
  /// Load code object from raw bytes.
  /// \param Bytes Raw ELF file contents
  /// \return Handler instance or error
  static llvm::Expected<CodeObjectHandler> loadFromBytes(
      llvm::ArrayRef<uint8_t> Bytes);

  /// Load code object from file.
  /// \param Path Path to ELF file
  /// \return Handler instance or error
  static llvm::Expected<CodeObjectHandler> loadFromFile(llvm::StringRef Path);

  /// Get the GPU architecture string.
  /// \return Architecture string (e.g., "gfx942")
  std::string getGPUArch() const;

  /// Get the number of kernels in this code object.
  size_t getKernelCount() const;

  /// Get kernel names.
  std::vector<std::string> getKernelNames() const;

  /// Get kernel info by name.
  /// \param Name Kernel name
  /// \return Kernel info or nullptr if not found
  const KernelInfo* getKernel(llvm::StringRef Name) const;

  /// Get occupied code ranges in .text (all STT_FUNC symbols).
  /// Returns vector of {offset, offset+size} pairs relative to .text start.
  std::vector<std::pair<uint64_t, uint64_t>> getTextFunctionRanges() const;

  /// Get the .text section (machine code).
  llvm::ArrayRef<uint8_t> getTextSection() const;

  /// Get the .rodata section (kernel descriptors).
  llvm::ArrayRef<uint8_t> getRodataSection() const;

  /// Get the .note section (metadata).
  llvm::ArrayRef<uint8_t> getNoteSection() const;

  /// Set a new .text section.
  /// \param Text New text section bytes
  void setTextSection(llvm::ArrayRef<uint8_t> Text);

  /// Apply an instrumentation patch.
  ///
  /// This updates:
  /// - .text section with patched code
  /// - Kernel descriptors with new register counts
  /// - .note metadata (if applicable)
  ///
  /// \param PatchedCode New code bytes for .text section
  /// \param KernelName Name of kernel being patched
  /// \param AdditionalVGPRs Additional VGPRs needed for instrumentation
  /// \param AdditionalSGPRs Additional SGPRs needed for instrumentation
  /// \param AdditionalKernargSize Additional bytes for kernarg (trace buffer args)
  /// \param AdditionalScratchSize Additional bytes for scratch memory (VGPR spill)
  /// \return Error on failure
  llvm::Error applyPatch(llvm::ArrayRef<uint8_t> PatchedCode,
                          llvm::StringRef KernelName,
                          uint32_t AdditionalVGPRs,
                          uint32_t AdditionalSGPRs,
                          uint32_t AdditionalKernargSize,
                          uint32_t AdditionalScratchSize = 0,
                          int64_t CodeEntryOffsetAdjust = 0);

  /// Build the final modified code object.
  /// \return Complete ELF file bytes or error
  llvm::Expected<std::vector<uint8_t>> build();

private:
  /// Private constructor - use loadFromBytes() or loadFromFile()
  CodeObjectHandler() = default;

  /// Parsed code object
  ParsedCodeObject Parsed;

  /// Note content (parsed separately for modification)
  std::optional<NoteContent> Note;

  /// Modified text section
  std::vector<uint8_t> ModifiedText;

  /// Modified rodata section
  std::vector<uint8_t> ModifiedRodata;

  /// Track whether sections have been modified
  bool TextModified = false;
  bool RodataModified = false;
};

} // namespace aegisbit

#endif // AEGISBIT_CODE_OBJECT_HANDLER_H
