////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "hotswap_rules.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

// Minimal JSON parser — avoids nlohmann/json dependency.
// Supports the subset needed for hotswap rules files:
//   objects, arrays, strings, integers, booleans.
// Not a general-purpose JSON parser; error messages are terse.

namespace rocr {
namespace hotswap {

namespace {

// ── Minimal JSON value type ──────────────────────────────────────────────────

enum class JsonType { Null, Bool, Int, String, Array, Object };

struct JsonValue;
using JsonArray = std::vector<JsonValue>;
using JsonObject = std::vector<std::pair<std::string, JsonValue>>;

struct JsonValue {
  JsonType type = JsonType::Null;
  bool bool_val = false;
  int64_t int_val = 0;
  std::string str_val;
  JsonArray arr_val;
  JsonObject obj_val;

  bool is_null() const { return type == JsonType::Null; }
  bool is_string() const { return type == JsonType::String; }
  bool is_int() const { return type == JsonType::Int; }
  bool is_bool() const { return type == JsonType::Bool; }
  bool is_array() const { return type == JsonType::Array; }
  bool is_object() const { return type == JsonType::Object; }

  const JsonValue* get(const std::string& key) const {
    if (type != JsonType::Object) return nullptr;
    for (auto& kv : obj_val) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }

  std::string get_string(const std::string& key,
                         const std::string& def = "") const {
    auto* v = get(key);
    return (v && v->is_string()) ? v->str_val : def;
  }

  int64_t get_int(const std::string& key, int64_t def = 0) const {
    auto* v = get(key);
    return (v && v->is_int()) ? v->int_val : def;
  }

  bool get_bool(const std::string& key, bool def = false) const {
    auto* v = get(key);
    return (v && v->is_bool()) ? v->bool_val : def;
  }
};

// ── JSON tokenizer + recursive descent parser ────────────────────────────────

class JsonParser {
public:
  explicit JsonParser(const std::string& input)
      : src_(input), pos_(0) {}

  bool Parse(JsonValue& out, std::string& err) {
    SkipWhitespace();
    if (!ParseValue(out)) {
      err = error_;
      return false;
    }
    SkipWhitespace();
    if (pos_ < src_.size()) {
      err = "trailing content after JSON value";
      return false;
    }
    return true;
  }

private:
  const std::string& src_;
  size_t pos_;
  std::string error_;

  char Peek() const { return pos_ < src_.size() ? src_[pos_] : '\0'; }
  char Next() { return pos_ < src_.size() ? src_[pos_++] : '\0'; }

  void SkipWhitespace() {
    while (pos_ < src_.size()) {
      char c = src_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        ++pos_;
      else
        break;
    }
  }

  bool Error(const std::string& msg) {
    std::ostringstream oss;
    oss << "JSON parse error at offset " << pos_ << ": " << msg;
    error_ = oss.str();
    return false;
  }

  bool ParseValue(JsonValue& out) {
    SkipWhitespace();
    char c = Peek();
    if (c == '"') return ParseString(out);
    if (c == '{') return ParseObject(out);
    if (c == '[') return ParseArray(out);
    if (c == 't' || c == 'f') return ParseBool(out);
    if (c == 'n') return ParseNull(out);
    if (c == '-' || (c >= '0' && c <= '9')) return ParseInt(out);
    return Error("unexpected character");
  }

  bool ParseString(JsonValue& out) {
    if (Next() != '"') return Error("expected '\"'");
    std::string s;
    while (true) {
      if (pos_ >= src_.size()) return Error("unterminated string");
      char c = Next();
      if (c == '"') break;
      if (c == '\\') {
        if (pos_ >= src_.size()) return Error("unterminated escape");
        char esc = Next();
        switch (esc) {
          case '"': s += '"'; break;
          case '\\': s += '\\'; break;
          case '/': s += '/'; break;
          case 'n': s += '\n'; break;
          case 't': s += '\t'; break;
          case 'r': s += '\r'; break;
          case 'b': s += '\b'; break;
          case 'f': s += '\f'; break;
          default: return Error("unknown escape sequence");
        }
      } else {
        s += c;
      }
    }
    out.type = JsonType::String;
    out.str_val = std::move(s);
    return true;
  }

  bool ParseInt(JsonValue& out) {
    size_t start = pos_;
    if (Peek() == '-') ++pos_;
    if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9')
      return Error("expected digit");

    // Handle 0x hex prefix
    if (src_[pos_] == '0' && pos_ + 1 < src_.size() &&
        (src_[pos_ + 1] == 'x' || src_[pos_ + 1] == 'X')) {
      pos_ += 2;
      while (pos_ < src_.size()) {
        char c = src_[pos_];
        if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
            (c >= 'A' && c <= 'F'))
          ++pos_;
        else
          break;
      }
      out.type = JsonType::Int;
      out.int_val = std::strtoll(src_.c_str() + start, nullptr, 0);
      return true;
    }

    while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9')
      ++pos_;
    out.type = JsonType::Int;
    out.int_val = std::strtoll(src_.c_str() + start, nullptr, 10);
    return true;
  }

  bool ParseBool(JsonValue& out) {
    if (src_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      out.type = JsonType::Bool;
      out.bool_val = true;
      return true;
    }
    if (src_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      out.type = JsonType::Bool;
      out.bool_val = false;
      return true;
    }
    return Error("expected 'true' or 'false'");
  }

  bool ParseNull(JsonValue& out) {
    if (src_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      out.type = JsonType::Null;
      return true;
    }
    return Error("expected 'null'");
  }

  bool ParseArray(JsonValue& out) {
    if (Next() != '[') return Error("expected '['");
    out.type = JsonType::Array;
    SkipWhitespace();
    if (Peek() == ']') { Next(); return true; }
    while (true) {
      JsonValue elem;
      if (!ParseValue(elem)) return false;
      out.arr_val.push_back(std::move(elem));
      SkipWhitespace();
      if (Peek() == ']') { Next(); return true; }
      if (Peek() != ',') return Error("expected ',' or ']'");
      Next();
      SkipWhitespace();
    }
  }

  bool ParseObject(JsonValue& out) {
    if (Next() != '{') return Error("expected '{'");
    out.type = JsonType::Object;
    SkipWhitespace();
    if (Peek() == '}') { Next(); return true; }
    while (true) {
      SkipWhitespace();
      JsonValue key;
      if (!ParseString(key)) return Error("expected string key");
      SkipWhitespace();
      if (Next() != ':') return Error("expected ':'");
      JsonValue val;
      if (!ParseValue(val)) return false;
      out.obj_val.emplace_back(std::move(key.str_val), std::move(val));
      SkipWhitespace();
      if (Peek() == '}') { Next(); return true; }
      if (Peek() != ',') return Error("expected ',' or '}'");
      Next();
    }
  }
};

// ── Hex string → byte vector ────────────────────────────────────────────────

static bool ParseHexBytes(const std::string& hex, std::vector<uint8_t>& out) {
  if (hex.size() % 2 != 0) return false;
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    char hi = hex[i], lo = hex[i + 1];
    auto hexval = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return -1;
    };
    int h = hexval(hi), l = hexval(lo);
    if (h < 0 || l < 0) return false;
    out.push_back(static_cast<uint8_t>((h << 4) | l));
  }
  return true;
}

// ── Parse offset strings ("0x1a4" or decimal) ───────────────────────────────

static bool ParseOffset(const std::string& s, int64_t& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  errno = 0;
  out = std::strtoll(s.c_str(), &end, 0);
  return (errno == 0 && end == s.c_str() + s.size());
}

// ── Parse a single rule from a JSON object ───────────────────────────────────

static bool ParseRule(const JsonValue& obj, RewriteRule& rule,
                      std::string& err) {
  rule.name = obj.get_string("name", "unnamed");

  // Match criteria
  auto* match = obj.get("match");
  if (match && match->is_object()) {
    rule.match_mnemonic = match->get_string("mnemonic");
    rule.match_kernel = match->get_string("kernel");

    auto* offset_val = match->get("offset");
    if (offset_val) {
      if (offset_val->is_string()) {
        if (!ParseOffset(offset_val->str_val, rule.match_offset)) {
          err = "rule '" + rule.name + "': invalid offset format";
          return false;
        }
      } else if (offset_val->is_int()) {
        rule.match_offset = offset_val->int_val;
      }
    }

    auto* operands_val = match->get("operands");
    if (operands_val && operands_val->is_array()) {
      for (auto& op : operands_val->arr_val) {
        OperandMatch om;
        if (op.is_object()) {
          auto* imm = op.get("imm");
          auto* reg = op.get("reg_class");
          if (imm && imm->is_int()) {
            om.kind = OperandMatch::Kind::Immediate;
            om.imm_value = imm->int_val;
          } else if (reg && reg->is_string()) {
            om.kind = OperandMatch::Kind::RegClass;
            om.reg_class = reg->str_val;
          }
          // else: wildcard
        }
        rule.operands.push_back(std::move(om));
      }
    }
  } else if (!match) {
    // match criteria can also be at top level for simple offset-only rules
    rule.match_mnemonic = obj.get_string("mnemonic");
  }

  // Replace action — determine which one is specified
  auto* replace = obj.get("replace");
  auto* replace_asm = obj.get("replace_asm");
  auto* replace_bytes = obj.get("replace_bytes");

  if (replace && replace->is_object()) {
    rule.action = ReplaceAction::MnemonicSwap;
    rule.replace_mnemonic = replace->get_string("mnemonic");
    rule.preserve_operands = replace->get_bool("preserve_operands", true);
    if (rule.replace_mnemonic.empty()) {
      err = "rule '" + rule.name + "': replace.mnemonic is empty";
      return false;
    }
  } else if (replace_asm) {
    rule.action = ReplaceAction::AsmReplace;
    if (replace_asm->is_string()) {
      rule.replace_asm.push_back(replace_asm->str_val);
    } else if (replace_asm->is_array()) {
      for (auto& line : replace_asm->arr_val) {
        if (line.is_string()) {
          rule.replace_asm.push_back(line.str_val);
        }
      }
    }
    if (rule.replace_asm.empty()) {
      err = "rule '" + rule.name + "': replace_asm is empty";
      return false;
    }
  } else if (replace_bytes) {
    rule.action = ReplaceAction::ByteReplace;
    if (replace_bytes->is_string()) {
      if (!ParseHexBytes(replace_bytes->str_val, rule.replace_bytes)) {
        err = "rule '" + rule.name + "': invalid hex in replace_bytes";
        return false;
      }
    }
    if (rule.replace_bytes.empty()) {
      err = "rule '" + rule.name + "': replace_bytes is empty";
      return false;
    }
  } else {
    err = "rule '" + rule.name +
          "': must specify replace, replace_asm, or replace_bytes";
    return false;
  }

  // Optional resource adjustments
  rule.extra_vgprs = static_cast<int32_t>(obj.get_int("extra_vgprs", 0));
  rule.extra_sgprs = static_cast<int32_t>(obj.get_int("extra_sgprs", 0));

  return true;
}

// ── Cached rules singleton ───────────────────────────────────────────────────

static std::once_flag g_rules_init_flag;
static std::unique_ptr<RulesFile> g_cached_rules;

static void InitCachedRules() {
  const char* path = std::getenv("HSA_HOTSWAP_RULES");
  if (!path || !*path) return;

  std::string err_msg;
  auto rules = std::make_unique<RulesFile>(ParseRulesFile(path, err_msg));
  if (rules->version == 0) {
    std::cerr << "hotswap: failed to parse rules file '" << path
              << "': " << err_msg << "\n";
    return;
  }
  g_cached_rules = std::move(rules);
}

} // anonymous namespace

// ── Public API ───────────────────────────────────────────────────────────────

RulesFile ParseRulesString(const std::string& json, std::string& err_msg) {
  RulesFile result;

  JsonValue root;
  JsonParser parser(json);
  if (!parser.Parse(root, err_msg)) return result;

  if (!root.is_object()) {
    err_msg = "root must be a JSON object";
    return result;
  }

  result.version = static_cast<uint32_t>(root.get_int("version", 0));
  if (result.version == 0) {
    err_msg = "missing or zero 'version' field";
    return result;
  }

  result.target = root.get_string("target");

  auto* rules_arr = root.get("rules");
  if (!rules_arr || !rules_arr->is_array()) {
    err_msg = "missing or non-array 'rules' field";
    result.version = 0;
    return result;
  }

  for (auto& rule_val : rules_arr->arr_val) {
    if (!rule_val.is_object()) {
      err_msg = "rule entry must be a JSON object";
      result.version = 0;
      return result;
    }
    RewriteRule rule;
    if (!ParseRule(rule_val, rule, err_msg)) {
      result.version = 0;
      return result;
    }
    result.rules.push_back(std::move(rule));
  }

  return result;
}

RulesFile ParseRulesFile(const std::string& path, std::string& err_msg) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    err_msg = "cannot open file: " + path;
    return RulesFile{};
  }

  std::ostringstream oss;
  oss << ifs.rdbuf();
  return ParseRulesString(oss.str(), err_msg);
}

const RulesFile* GetCachedRules() {
  std::call_once(g_rules_init_flag, InitCachedRules);
  return g_cached_rules.get();
}

} // namespace hotswap
} // namespace rocr
