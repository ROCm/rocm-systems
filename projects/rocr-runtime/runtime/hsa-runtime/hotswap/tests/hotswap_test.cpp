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

/// Unit tests for the hotswap ISA rewrite engine.
/// Build with the standalone hotswap CMakeLists.txt.

#include "hotswap_rules.hpp"
#include "trampoline.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

using namespace rocr::hotswap;

// ── Rule parsing tests ───────────────────────────────────────────────────────

static void TestParseValidRules() {
  std::string json = R"({
    "version": 1,
    "target": "amdgcn-amd-amdhsa--gfx1201",
    "rules": [
      {
        "name": "test_mnemonic_swap",
        "match": { "mnemonic": "v_mac_f32_e32" },
        "replace": { "mnemonic": "v_fmac_f32_e32", "preserve_operands": true }
      },
      {
        "name": "test_asm_replace",
        "match": { "mnemonic": "s_sleep", "operands": [{ "imm": 10 }] },
        "replace_asm": "s_nop 0"
      },
      {
        "name": "test_byte_replace",
        "match": { "kernel": "my_kernel", "offset": "0x1a4" },
        "replace_bytes": "BF800000"
      },
      {
        "name": "test_multi_asm",
        "match": { "mnemonic": "v_dot2_f32_f16" },
        "replace_asm": ["s_nop 0", "s_nop 0"],
        "extra_vgprs": 2
      }
    ]
  })";

  std::string err;
  RulesFile rf = ParseRulesString(json, err);
  assert(rf.version == 1);
  assert(rf.target == "amdgcn-amd-amdhsa--gfx1201");
  assert(rf.rules.size() == 4);

  // Rule 0: mnemonic swap
  assert(rf.rules[0].name == "test_mnemonic_swap");
  assert(rf.rules[0].match_mnemonic == "v_mac_f32_e32");
  assert(rf.rules[0].action == ReplaceAction::MnemonicSwap);
  assert(rf.rules[0].replace_mnemonic == "v_fmac_f32_e32");
  assert(rf.rules[0].preserve_operands == true);

  // Rule 1: asm replace with operand match
  assert(rf.rules[1].name == "test_asm_replace");
  assert(rf.rules[1].match_mnemonic == "s_sleep");
  assert(rf.rules[1].operands.size() == 1);
  assert(rf.rules[1].operands[0].kind == OperandMatch::Kind::Immediate);
  assert(rf.rules[1].operands[0].imm_value == 10);
  assert(rf.rules[1].action == ReplaceAction::AsmReplace);
  assert(rf.rules[1].replace_asm.size() == 1);
  assert(rf.rules[1].replace_asm[0] == "s_nop 0");

  // Rule 2: byte replace with kernel + offset match
  assert(rf.rules[2].name == "test_byte_replace");
  assert(rf.rules[2].match_kernel == "my_kernel");
  assert(rf.rules[2].match_offset == 0x1a4);
  assert(rf.rules[2].action == ReplaceAction::ByteReplace);
  assert(rf.rules[2].replace_bytes.size() == 4);
  assert(rf.rules[2].replace_bytes[0] == 0xBF);
  assert(rf.rules[2].replace_bytes[1] == 0x80);
  assert(rf.rules[2].replace_bytes[2] == 0x00);
  assert(rf.rules[2].replace_bytes[3] == 0x00);

  // Rule 3: multi-line asm with extra vgprs
  assert(rf.rules[3].name == "test_multi_asm");
  assert(rf.rules[3].action == ReplaceAction::AsmReplace);
  assert(rf.rules[3].replace_asm.size() == 2);
  assert(rf.rules[3].extra_vgprs == 2);

  std::cout << "TestParseValidRules: PASSED\n";
}

static void TestParseInvalidRules() {
  // Missing version
  {
    std::string json = R"({"rules": []})";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 0);
    assert(!err.empty());
  }

  // Missing replace action
  {
    std::string json = R"({
      "version": 1,
      "rules": [{ "name": "bad", "match": { "mnemonic": "foo" } }]
    })";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 0);
  }

  // Invalid hex in replace_bytes
  {
    std::string json = R"({
      "version": 1,
      "rules": [{
        "name": "bad_hex",
        "match": { "offset": 0 },
        "replace_bytes": "ZZZZ"
      }]
    })";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 0);
  }

  // Invalid JSON
  {
    std::string json = R"({not valid json)";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 0);
  }

  std::cout << "TestParseInvalidRules: PASSED\n";
}

// ── Trampoline encoding tests ────────────────────────────────────────────────

static void TestSBranchEncoding() {
  // Forward branch: from offset 0x100 to offset 0x200
  // byte_delta = 0x200 - 0x100 - 4 = 0xFC
  // dword_offset = 0xFC / 4 = 0x3F = 63
  {
    uint8_t bytes[4];
    bool ok = EncodeSBranch(0x100, 0x200, bytes);
    assert(ok);
    uint32_t encoded;
    std::memcpy(&encoded, bytes, 4);
    assert((encoded & 0xFFFF0000u) == 0xBF820000u);
    int16_t offset = static_cast<int16_t>(encoded & 0xFFFF);
    assert(offset == 63);
  }

  // Backward branch: from offset 0x200 to offset 0x100
  // byte_delta = 0x100 - 0x200 - 4 = -0x104
  // dword_offset = -0x104 / 4 = -65
  {
    uint8_t bytes[4];
    bool ok = EncodeSBranch(0x200, 0x100, bytes);
    assert(ok);
    uint32_t encoded;
    std::memcpy(&encoded, bytes, 4);
    int16_t offset = static_cast<int16_t>(encoded & 0xFFFF);
    assert(offset == -65);
  }

  // Self-branch (branch to next instruction): from 0 to 4
  // byte_delta = 4 - 0 - 4 = 0
  // dword_offset = 0
  {
    uint8_t bytes[4];
    bool ok = EncodeSBranch(0, 4, bytes);
    assert(ok);
    uint32_t encoded;
    std::memcpy(&encoded, bytes, 4);
    int16_t offset = static_cast<int16_t>(encoded & 0xFFFF);
    assert(offset == 0);
  }

  std::cout << "TestSBranchEncoding: PASSED\n";
}

static void TestSNopEncoding() {
  uint8_t bytes[4];
  EncodeSNop(bytes);
  uint32_t encoded;
  std::memcpy(&encoded, bytes, 4);
  assert(encoded == 0xBF800000u);

  std::cout << "TestSNopEncoding: PASSED\n";
}

// ── JSON parser edge cases ───────────────────────────────────────────────────

static void TestJsonEdgeCases() {
  // Hex offset as string
  {
    std::string json = R"({
      "version": 1,
      "rules": [{
        "name": "hex_offset",
        "match": { "offset": "0xFF" },
        "replace_bytes": "00000000"
      }]
    })";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 1);
    assert(rf.rules[0].match_offset == 0xFF);
  }

  // Hex offset as integer
  {
    std::string json = R"({
      "version": 1,
      "rules": [{
        "name": "int_offset",
        "match": { "offset": 256 },
        "replace_bytes": "00000000"
      }]
    })";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 1);
    assert(rf.rules[0].match_offset == 256);
  }

  // Empty rules array (valid)
  {
    std::string json = R"({ "version": 1, "rules": [] })";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 1);
    assert(rf.rules.empty());
  }

  // Wildcard operands
  {
    std::string json = R"({
      "version": 1,
      "rules": [{
        "name": "wildcard_ops",
        "match": { "mnemonic": "v_add_f32", "operands": [{}, {"imm": 42}] },
        "replace_bytes": "00000000"
      }]
    })";
    std::string err;
    RulesFile rf = ParseRulesString(json, err);
    assert(rf.version == 1);
    assert(rf.rules[0].operands.size() == 2);
    assert(rf.rules[0].operands[0].kind == OperandMatch::Kind::Wildcard);
    assert(rf.rules[0].operands[1].kind == OperandMatch::Kind::Immediate);
    assert(rf.rules[0].operands[1].imm_value == 42);
  }

  std::cout << "TestJsonEdgeCases: PASSED\n";
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
  TestParseValidRules();
  TestParseInvalidRules();
  TestSBranchEncoding();
  TestSNopEncoding();
  TestJsonEdgeCases();

  std::cout << "\nAll hotswap tests passed.\n";
  return 0;
}
