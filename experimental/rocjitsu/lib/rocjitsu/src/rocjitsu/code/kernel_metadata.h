// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Kernel argument metadata extracted from AMDGPU code objects.

#ifndef ROCJITSU_CODE_KERNEL_METADATA_H_
#define ROCJITSU_CODE_KERNEL_METADATA_H_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rocjitsu {

/// All kernel arguments for a single kernel, parsed from the AMDGPU code
/// object's .note metadata. Arguments are stored in declaration order.
/// Provides O(1) lookup by index and by value_kind (when the kind is unique).
class KernelArgs {
public:
  using ParseResult = std::variant<KernelArgs, std::string>;

  /// Parse kernel argument metadata from an AMDGPU device ELF image.
  /// Selects the kernel matching @p kernel_name (the ".name" field in metadata).
  /// On success, returns a KernelArgs. On failure, returns an error string.
  static ParseResult parse(const uint8_t *elf_data, size_t elf_size,
                           const std::string &kernel_name);

  /// Number of arguments.
  size_t size() const { return args_.size(); }

  /// Get the byte offset of the argument at the given index.
  uint32_t getOffsetFromIndex(size_t index) const { return args_[index].offset; }

  /// Get the byte size of the argument at the given index.
  uint32_t getSizeFromIndex(size_t index) const { return args_[index].size; }

  /// Get the value_kind string of the argument at the given index.
  /// Empty string if not present in the metadata.
  std::string getValueKindFromIndex(size_t index) const {
    return args_[index].value_kind;
  }

  /// Get the name of the argument at the given index.
  /// Empty string if not present in the metadata.
  std::string getNameFromIndex(size_t index) const {
    return args_[index].name;
  }

  /// Get the address space of the argument at the given index.
  /// Empty string if not present in the metadata.
  std::string getAddressSpaceFromIndex(size_t index) const {
    return args_[index].address_space;
  }

  /// Get the index of the unique argument with the given value_kind.
  /// Returns nullopt if zero or more than one argument has this kind.
  std::optional<size_t> getIndexFromValueKind(const std::string &kind) const {
    auto it = kind_index_.find(kind);
    if (it == kind_index_.end() || it->second.size() != 1) {
      return std::nullopt;
    }
    return it->second[0];
  }

  /// Count how many arguments have the given value_kind.
  size_t countValueKind(const std::string &kind) const {
    auto it = kind_index_.find(kind);
    return it != kind_index_.end() ? it->second.size() : 0;
  }

private:
  struct Arg {
    std::string value_kind;
    uint32_t offset = 0;
    uint32_t size = 0;
    std::string name;
    std::string address_space;
  };

  void add(Arg arg) {
    kind_index_[arg.value_kind].push_back(args_.size());
    args_.push_back(std::move(arg));
  }

  std::vector<Arg> args_;
  std::unordered_map<std::string, std::vector<size_t>> kind_index_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_KERNEL_METADATA_H_
