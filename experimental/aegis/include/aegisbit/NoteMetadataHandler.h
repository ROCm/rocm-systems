//===-- aegisbit/NoteMetadataHandler.h - Note Section Handling ---*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Handler for AMD GPU .note section metadata.
/// Parses and serializes YAML metadata containing kernel properties.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_NOTE_METADATA_HANDLER_H
#define AEGISBIT_NOTE_METADATA_HANDLER_H

#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace aegisbit {

/// Kernel metadata from .note section
struct KernelMetadata {
  std::string Name;         ///< Kernel name
  uint32_t SGPRCount = 0;   ///< SGPR count
  uint32_t VGPRCount = 0;   ///< VGPR count
  uint32_t LDSSize = 0;     ///< LDS size in bytes
  uint32_t ScratchSize = 0; ///< Scratch memory size per work-item
  uint32_t KernargSize = 0; ///< Kernel argument size
};

/// Parsed note section content
struct NoteContent {
  std::string GPUArch;                    ///< GPU architecture (e.g., "gfx942")
  std::vector<KernelMetadata> Kernels;    ///< Kernel metadata entries
  std::string RawYAML;                    ///< Raw serialized content (msgpack or YAML)
  bool IsMsgPack = false;                 ///< true if RawYAML holds msgpack binary
};

/// Handler for AMD GPU .note section metadata.
///
/// The .note section contains NT_AMDGPU_METADATA notes with YAML content
/// describing kernel properties like register counts and memory sizes.
///
/// Format:
/// \code
/// amdhsa.kernels:
///   - .name: kernel_name
///     .sgpr_count: 32
///     .vgpr_count: 128
///     .kernarg_segment_size: 32
///     .group_segment_fixed_size: 0
///     .private_segment_fixed_size: 0
/// \endcode
class NoteMetadataHandler {
public:
  /// Parse .note section content.
  /// \param NoteSection Raw .note section bytes
  /// \return Parsed note content or error
  static llvm::Expected<NoteContent> parse(llvm::ArrayRef<uint8_t> NoteSection);

  /// Update kernel metadata in note content.
  /// \param Content Note content to modify
  /// \param KernelName Kernel to update
  /// \param Descriptor New kernel descriptor values
  /// \return Error if kernel not found
  static llvm::Error updateKernelMetadata(NoteContent& Content,
                                           llvm::StringRef KernelName,
                                           const KernelDescriptor& Descriptor);

  /// Serialize note content back to .note section bytes.
  /// \param Content Note content to serialize
  /// \return Serialized .note section bytes
  static std::vector<uint8_t> serialize(const NoteContent& Content);

  /// Note type for AMDGPU metadata
  static constexpr uint32_t NT_AMDGPU_METADATA = 32;

  /// AMDGPU note name
  static constexpr const char* AMDGPU_NOTE_NAME = "AMDGPU";

private:
  /// Extract msgpack binary content from note section (modern ROCm)
  static llvm::Expected<std::string> extractMsgPack(llvm::ArrayRef<uint8_t> NoteSection);

  /// Parse kernel metadata from msgpack binary format (modern ROCm)
  static llvm::Error parseMsgPack(const std::string& MsgPackData, NoteContent& Content);

  /// Parse kernel metadata from YAML (legacy, pre-msgpack code objects)
  static llvm::Error parseYAML(const std::string& YAML, NoteContent& Content);

  /// Update msgpack blob with new kernel values (modern ROCm)
  static llvm::Error updateMsgPack(std::string& MsgPackData,
                                    llvm::StringRef KernelName,
                                    const KernelDescriptor& Descriptor);

  /// Update YAML content with new kernel values (legacy, pre-msgpack code objects)
  static std::string updateYAML(const std::string& YAML,
                                 llvm::StringRef KernelName,
                                 const KernelDescriptor& Descriptor);

  /// Build note section bytes from raw content (msgpack or YAML)
  static std::vector<uint8_t> buildNoteSection(const std::string& Data);

  /// Align value to 4-byte boundary
  static size_t align4(size_t Value) {
    return (Value + 3) & ~3;
  }
};

} // namespace aegisbit

#endif // AEGISBIT_NOTE_METADATA_HANDLER_H
