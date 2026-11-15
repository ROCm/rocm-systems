// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file json.h
/// @brief Minimal ordered JSON value for rj_isa_diff case, contract and result files.
///
/// @details The tree has no in-tree general JSON library (configuration uses
/// schema-driven FlatBuffers). Case files must stay hand-editable so that a reviewer
/// can write or audit a manual contract, so this keeps object member order and
/// prints indented text. Numbers keep an exact unsigned/signed 64-bit integer
/// representation when they have no fraction or exponent.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::isa_diff {

class Json {
public:
  enum class Kind { Null, Bool, Int, Uint, Double, String, Array, Object };
  using Member = std::pair<std::string, Json>;

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool v) : kind_(Kind::Bool), bool_(v) {}
  Json(int v) : kind_(Kind::Int), int_(v) {}
  Json(int64_t v) : kind_(Kind::Int), int_(v) {}
  Json(long long v) : kind_(Kind::Int), int_(static_cast<int64_t>(v)) {}
  Json(uint32_t v) : kind_(Kind::Uint), uint_(v) {}
  Json(uint64_t v) : kind_(Kind::Uint), uint_(v) {}
  Json(unsigned long long v) : kind_(Kind::Uint), uint_(static_cast<uint64_t>(v)) {}
  Json(double v) : kind_(Kind::Double), double_(v) {}
  Json(const char *v) : kind_(Kind::String), string_(v) {}
  Json(std::string v) : kind_(Kind::String), string_(std::move(v)) {}
  Json(std::string_view v) : kind_(Kind::String), string_(v) {}

  static Json array() {
    Json j;
    j.kind_ = Kind::Array;
    return j;
  }
  static Json object() {
    Json j;
    j.kind_ = Kind::Object;
    return j;
  }

  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] bool is_null() const { return kind_ == Kind::Null; }
  [[nodiscard]] bool is_object() const { return kind_ == Kind::Object; }
  [[nodiscard]] bool is_array() const { return kind_ == Kind::Array; }
  [[nodiscard]] bool is_string() const { return kind_ == Kind::String; }
  [[nodiscard]] bool is_number() const {
    return kind_ == Kind::Int || kind_ == Kind::Uint || kind_ == Kind::Double;
  }

  [[nodiscard]] bool as_bool(bool fallback = false) const {
    return kind_ == Kind::Bool ? bool_ : fallback;
  }
  [[nodiscard]] int64_t as_int(int64_t fallback = 0) const;
  [[nodiscard]] uint64_t as_uint(uint64_t fallback = 0) const;
  [[nodiscard]] double as_double(double fallback = 0) const;
  [[nodiscard]] const std::string &as_string() const;

  /// Object member lookup; returns a shared null value when absent.
  [[nodiscard]] const Json &operator[](std::string_view key) const;
  /// Object member insertion or replacement, preserving first-insertion order.
  Json &set(std::string_view key, Json value);
  [[nodiscard]] bool has(std::string_view key) const;
  [[nodiscard]] const std::vector<Member> &members() const { return members_; }

  /// Array access.
  Json &push(Json value);
  [[nodiscard]] const std::vector<Json> &items() const { return items_; }
  [[nodiscard]] size_t size() const {
    return kind_ == Kind::Array ? items_.size() : members_.size();
  }

  /// Serialize. indent < 0 prints a single line.
  [[nodiscard]] std::string dump(int indent = 2) const;

  /// Parse; on failure returns nullopt and sets *error (when non-null).
  static std::optional<Json> parse(std::string_view text, std::string *error = nullptr);

private:
  void dump_to(std::string &out, int indent, int depth) const;

  Kind kind_ = Kind::Null;
  bool bool_ = false;
  int64_t int_ = 0;
  uint64_t uint_ = 0;
  double double_ = 0;
  std::string string_;
  std::vector<Json> items_;
  std::vector<Member> members_;
};

/// @brief "0x%08x" formatting used for every register value in files.
[[nodiscard]] std::string hex32(uint32_t value);
/// @brief "0x%x" formatting for addresses.
[[nodiscard]] std::string hex64(uint64_t value);
/// @brief Parse "0x..." hex or a decimal number from a JSON string or number.
[[nodiscard]] std::optional<uint64_t> json_u64(const Json &value);

/// @brief Read a whole file; nullopt on error.
[[nodiscard]] std::optional<std::string> read_text_file(const std::string &path);
/// @brief Write a whole file; false on error.
[[nodiscard]] bool write_text_file(const std::string &path, std::string_view text);

} // namespace rocjitsu::isa_diff
