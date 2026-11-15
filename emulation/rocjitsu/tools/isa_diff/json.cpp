// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "json.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace rocjitsu::isa_diff {
namespace {

const Json &null_json() {
  static const Json null;
  return null;
}

const std::string &empty_string() {
  static const std::string empty;
  return empty;
}

void escape_to(std::string &out, std::string_view text) {
  out += '"';
  for (const char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
        out += buf;
      } else {
        out += c;
      }
    }
  }
  out += '"';
}

class Parser {
public:
  explicit Parser(std::string_view text) : text_(text) {}

  std::optional<Json> parse_document(std::string *error) {
    auto value = parse_value();
    skip_ws();
    if (value && pos_ != text_.size())
      fail("trailing characters");
    if (!error_.empty()) {
      if (error)
        *error = error_ + " at offset " + std::to_string(pos_);
      return std::nullopt;
    }
    return value;
  }

private:
  void fail(std::string message) {
    if (error_.empty())
      error_ = std::move(message);
  }

  void skip_ws() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' ||
                                   text_[pos_] == '\r' || text_[pos_] == '\t'))
      ++pos_;
  }

  bool consume(std::string_view token) {
    if (text_.substr(pos_, token.size()) != token)
      return false;
    pos_ += token.size();
    return true;
  }

  std::optional<Json> parse_value() {
    if (++depth_ > 256) {
      fail("nesting too deep");
      return std::nullopt;
    }
    skip_ws();
    std::optional<Json> out;
    if (pos_ >= text_.size()) {
      fail("unexpected end of input");
    } else if (text_[pos_] == '{') {
      out = parse_object();
    } else if (text_[pos_] == '[') {
      out = parse_array();
    } else if (text_[pos_] == '"') {
      if (auto s = parse_string())
        out = Json(std::move(*s));
    } else if (consume("true")) {
      out = Json(true);
    } else if (consume("false")) {
      out = Json(false);
    } else if (consume("null")) {
      out = Json(nullptr);
    } else {
      out = parse_number();
    }
    --depth_;
    return error_.empty() ? out : std::nullopt;
  }

  std::optional<Json> parse_object() {
    ++pos_;
    Json obj = Json::object();
    skip_ws();
    if (consume("}"))
      return obj;
    while (true) {
      skip_ws();
      auto key = parse_string();
      if (!key)
        return std::nullopt;
      skip_ws();
      if (!consume(":")) {
        fail("expected ':'");
        return std::nullopt;
      }
      auto value = parse_value();
      if (!value)
        return std::nullopt;
      obj.set(*key, std::move(*value));
      skip_ws();
      if (consume("}"))
        return obj;
      if (!consume(",")) {
        fail("expected ',' or '}'");
        return std::nullopt;
      }
    }
  }

  std::optional<Json> parse_array() {
    ++pos_;
    Json arr = Json::array();
    skip_ws();
    if (consume("]"))
      return arr;
    while (true) {
      auto value = parse_value();
      if (!value)
        return std::nullopt;
      arr.push(std::move(*value));
      skip_ws();
      if (consume("]"))
        return arr;
      if (!consume(",")) {
        fail("expected ',' or ']'");
        return std::nullopt;
      }
    }
  }

  std::optional<std::string> parse_string() {
    if (pos_ >= text_.size() || text_[pos_] != '"') {
      fail("expected string");
      return std::nullopt;
    }
    ++pos_;
    std::string out;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"')
        return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= text_.size())
        break;
      const char e = text_[pos_++];
      switch (e) {
      case '"':
      case '\\':
      case '/':
        out += e;
        break;
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case 'u': {
        if (pos_ + 4 > text_.size()) {
          fail("truncated \\u escape");
          return std::nullopt;
        }
        unsigned code = 0;
        auto [ptr, ec] = std::from_chars(text_.data() + pos_, text_.data() + pos_ + 4, code, 16);
        if (ec != std::errc{} || ptr != text_.data() + pos_ + 4) {
          fail("bad \\u escape");
          return std::nullopt;
        }
        pos_ += 4;
        if (code < 0x80) {
          out += static_cast<char>(code);
        } else if (code < 0x800) {
          out += static_cast<char>(0xc0 | (code >> 6));
          out += static_cast<char>(0x80 | (code & 0x3f));
        } else {
          out += static_cast<char>(0xe0 | (code >> 12));
          out += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
          out += static_cast<char>(0x80 | (code & 0x3f));
        }
        break;
      }
      default:
        fail("bad escape");
        return std::nullopt;
      }
    }
    fail("unterminated string");
    return std::nullopt;
  }

  std::optional<Json> parse_number() {
    const size_t start = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+'))
      ++pos_;
    bool is_float = false;
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c >= '0' && c <= '9') {
        ++pos_;
      } else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
        is_float = true;
        ++pos_;
      } else {
        break;
      }
    }
    const std::string_view token = text_.substr(start, pos_ - start);
    if (token.empty() || token == "-" || token == "+") {
      fail("expected value");
      return std::nullopt;
    }
    const char *first = token.data();
    const char *last = token.data() + token.size();
    if (!is_float) {
      if (token.front() == '-') {
        int64_t v = 0;
        auto [ptr, ec] = std::from_chars(first, last, v);
        if (ec == std::errc{} && ptr == last)
          return Json(v);
      } else {
        if (*first == '+')
          ++first;
        uint64_t v = 0;
        auto [ptr, ec] = std::from_chars(first, last, v);
        if (ec == std::errc{} && ptr == last)
          return Json(v);
      }
    }
    std::string copy(token);
    char *end = nullptr;
    errno = 0;
    const double d = std::strtod(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size() || errno == ERANGE) {
      fail("bad number");
      return std::nullopt;
    }
    return Json(d);
  }

  std::string_view text_;
  size_t pos_ = 0;
  int depth_ = 0;
  std::string error_;
};

} // namespace

int64_t Json::as_int(int64_t fallback) const {
  switch (kind_) {
  case Kind::Int:
    return int_;
  case Kind::Uint:
    return static_cast<int64_t>(uint_);
  case Kind::Double:
    return static_cast<int64_t>(double_);
  default:
    return fallback;
  }
}

uint64_t Json::as_uint(uint64_t fallback) const {
  switch (kind_) {
  case Kind::Int:
    return static_cast<uint64_t>(int_);
  case Kind::Uint:
    return uint_;
  case Kind::Double:
    return static_cast<uint64_t>(double_);
  default:
    return fallback;
  }
}

double Json::as_double(double fallback) const {
  switch (kind_) {
  case Kind::Int:
    return static_cast<double>(int_);
  case Kind::Uint:
    return static_cast<double>(uint_);
  case Kind::Double:
    return double_;
  default:
    return fallback;
  }
}

const std::string &Json::as_string() const {
  return kind_ == Kind::String ? string_ : empty_string();
}

const Json &Json::operator[](std::string_view key) const {
  if (kind_ != Kind::Object)
    return null_json();
  for (const auto &[name, value] : members_)
    if (name == key)
      return value;
  return null_json();
}

bool Json::has(std::string_view key) const {
  if (kind_ != Kind::Object)
    return false;
  for (const auto &member : members_)
    if (member.first == key)
      return true;
  return false;
}

Json &Json::set(std::string_view key, Json value) {
  if (kind_ != Kind::Object) {
    *this = object();
  }
  for (auto &[name, existing] : members_) {
    if (name == key) {
      existing = std::move(value);
      return existing;
    }
  }
  members_.emplace_back(std::string(key), std::move(value));
  return members_.back().second;
}

Json &Json::push(Json value) {
  if (kind_ != Kind::Array)
    *this = array();
  items_.push_back(std::move(value));
  return items_.back();
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_to(out, indent, 0);
  if (indent >= 0)
    out += '\n';
  return out;
}

void Json::dump_to(std::string &out, int indent, int depth) const {
  auto newline = [&](int d) {
    if (indent < 0)
      return;
    out += '\n';
    out.append(static_cast<size_t>(indent * d), ' ');
  };
  switch (kind_) {
  case Kind::Null:
    out += "null";
    return;
  case Kind::Bool:
    out += bool_ ? "true" : "false";
    return;
  case Kind::Int:
    out += std::to_string(int_);
    return;
  case Kind::Uint:
    out += std::to_string(uint_);
    return;
  case Kind::Double: {
    if (!std::isfinite(double_)) {
      out += "null";
      return;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", double_);
    out += buf;
    return;
  }
  case Kind::String:
    escape_to(out, string_);
    return;
  case Kind::Array: {
    if (items_.empty()) {
      out += "[]";
      return;
    }
    // Arrays of scalars print on one line to keep register dumps compact.
    bool scalar = true;
    for (const auto &item : items_)
      if (item.kind_ == Kind::Array || item.kind_ == Kind::Object)
        scalar = false;
    out += '[';
    for (size_t i = 0; i < items_.size(); ++i) {
      if (i)
        out += scalar ? ", " : ",";
      if (!scalar)
        newline(depth + 1);
      items_[i].dump_to(out, indent, depth + 1);
    }
    if (!scalar)
      newline(depth);
    out += ']';
    return;
  }
  case Kind::Object: {
    if (members_.empty()) {
      out += "{}";
      return;
    }
    out += '{';
    for (size_t i = 0; i < members_.size(); ++i) {
      if (i)
        out += ',';
      newline(depth + 1);
      escape_to(out, members_[i].first);
      out += indent < 0 ? ":" : ": ";
      members_[i].second.dump_to(out, indent, depth + 1);
    }
    newline(depth);
    out += '}';
    return;
  }
  }
}

std::optional<Json> Json::parse(std::string_view text, std::string *error) {
  Parser parser(text);
  return parser.parse_document(error);
}

std::string hex32(uint32_t value) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%08x", value);
  return buf;
}

std::string hex64(uint64_t value) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(value));
  return buf;
}

std::optional<uint64_t> json_u64(const Json &value) {
  if (value.kind() == Json::Kind::Uint || value.kind() == Json::Kind::Int)
    return value.as_uint();
  if (!value.is_string())
    return std::nullopt;
  std::string_view text = value.as_string();
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  uint64_t out = 0;
  auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out, base);
  if (ec != std::errc{} || ptr != text.data() + text.size() || text.empty())
    return std::nullopt;
  return out;
}

std::optional<std::string> read_text_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return std::nullopt;
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool write_text_file(const std::string &path, std::string_view text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    return false;
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  return static_cast<bool>(out);
}

} // namespace rocjitsu::isa_diff
