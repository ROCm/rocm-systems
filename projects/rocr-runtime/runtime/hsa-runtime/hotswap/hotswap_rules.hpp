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

#ifndef ROCR_HOTSWAP_RULES_HPP
#define ROCR_HOTSWAP_RULES_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace rocr {
namespace hotswap {

/// Operand match criterion for a rewrite rule.
struct OperandMatch {
  enum class Kind {
    Wildcard,   // Match any operand
    Immediate,  // Match specific immediate value
    RegClass,   // Match register class name (e.g. "VGPR_32")
  };

  Kind kind = Kind::Wildcard;
  int64_t imm_value = 0;       // For Kind::Immediate
  std::string reg_class;       // For Kind::RegClass
};

/// How a matched instruction should be replaced.
enum class ReplaceAction {
  MnemonicSwap,   // Replace mnemonic, preserve operands (same-size)
  AsmReplace,     // Replace with assembly string(s) (may need trampoline)
  ByteReplace,    // Replace with raw hex bytes
};

/// A single rewrite rule parsed from the JSON rules file.
struct RewriteRule {
  std::string name;

  // Match criteria (all specified criteria must match)
  std::string match_mnemonic;          // Empty = don't match on mnemonic
  std::vector<OperandMatch> operands;  // Empty = don't match on operands
  std::string match_kernel;            // Empty = match all kernels
  int64_t match_offset = -1;           // -1 = don't match on offset

  // Replace action
  ReplaceAction action = ReplaceAction::MnemonicSwap;
  std::string replace_mnemonic;        // For MnemonicSwap
  bool preserve_operands = true;       // For MnemonicSwap
  std::vector<std::string> replace_asm; // For AsmReplace
  std::vector<uint8_t> replace_bytes;  // For ByteReplace

  // Optional resource adjustments
  int32_t extra_vgprs = 0;
  int32_t extra_sgprs = 0;
};

/// Parsed rules file containing target-specific rewrite rules.
struct RulesFile {
  uint32_t version = 0;
  std::string target;                  // e.g. "amdgcn-amd-amdhsa--gfx1201"
  std::vector<RewriteRule> rules;
};

/// Parse a JSON rules file from a file path.
/// Returns empty RulesFile with version=0 on parse error, with error
/// description written to err_msg.
RulesFile ParseRulesFile(const std::string& path, std::string& err_msg);

/// Parse a JSON rules string.
RulesFile ParseRulesString(const std::string& json, std::string& err_msg);

/// Get the cached rules for the current process. Loads from
/// HSA_HOTSWAP_RULES on first call. Returns nullptr if no rules are set
/// or if parsing failed.
const RulesFile* GetCachedRules();

} // namespace hotswap
} // namespace rocr

#endif // ROCR_HOTSWAP_RULES_HPP
