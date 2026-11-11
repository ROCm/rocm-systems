// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/result.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::amdhsa {

/// @brief One bounded MessagePack value from an AMDHSA metadata note.
/// @details Strings borrow the input image. Offsets retain the existing integer
/// encoding so resource edits can be validated before any bytes are changed.
class MetadataValue {
public:
  enum class Kind { Integer, String, Array, Map, Boolean, Nil };
  Kind kind = Kind::Nil;
  uint64_t integer = 0;
  std::string_view string;
  std::vector<MetadataValue> array;
  std::map<std::string_view, MetadataValue> map;
  size_t offset = 0;
  unsigned integer_bytes = 0;

  [[nodiscard]] const MetadataValue *field(std::string_view name) const;
  [[nodiscard]] std::optional<uint32_t> integer_field(std::string_view name) const;
  [[nodiscard]] bool text_field_is(std::string_view name, std::string_view expected) const;
};

/// @brief Parsed AMDHSA note whose container nodes own their storage.
/// @details Payload and string views borrow the image, which must remain alive
/// and at the same address while this document is inspected. Consumers retain
/// their own target, argument, launch, and analysis admission rules.
class MetadataDocument {
public:
  [[nodiscard]] static FailureOr<MetadataDocument> parse(std::span<const uint8_t> image,
                                                         std::string *error = nullptr);
  [[nodiscard]] const MetadataValue &root() const { return root_; }
  [[nodiscard]] std::span<const uint8_t> payload() const { return payload_; }

private:
  std::span<const uint8_t> payload_;
  MetadataValue root_;
};

/// @brief Update one kernel's numeric resources without resizing its note.
/// @details Only the named kernel in supported v1.2 metadata is edited. Missing
/// fields, ambiguous entries, descriptor overlap, or wider integer encodings
/// fail before any writes. Kernarg and private-segment sizes come from the new
/// descriptor. Absent register counts retain their original fields.
[[nodiscard]] Result
update_resource_metadata(std::span<uint8_t> image, const KernelDescriptorInfo &kernel,
                         const kernel_descriptor_t &descriptor, std::optional<uint32_t> sgprs,
                         std::optional<uint32_t> vgprs, std::string *error = nullptr);

} // namespace rocjitsu::amdhsa
