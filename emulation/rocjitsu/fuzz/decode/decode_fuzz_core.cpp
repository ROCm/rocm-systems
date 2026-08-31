// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_fuzz_core.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace rocjitsu::decode_fuzz {
namespace {

[[noreturn]] void fail_invariant(std::string_view message) {
  std::cerr << "rj_decode_fuzz invariant failed: " << message << '\n';
  std::abort();
}

std::string json_escape(std::string_view value) {
  std::ostringstream out;
  for (const unsigned char c : value) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<unsigned int>(c) << std::dec;
      } else {
        out << c;
      }
    }
  }
  return out.str();
}

void emit_string_array(std::ostringstream &out, const std::vector<std::string> &values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      out << ',';
    out << '"' << json_escape(values[i]) << '"';
  }
  out << ']';
}

} // namespace

std::unique_ptr<Decoder> create_decoder(std::string_view target) {
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder)
    throw std::invalid_argument("unsupported decoder target: " + std::string(target));
  return decoder;
}

DecodeRecord decode_window(Decoder &decoder, std::span<const uint8_t> bytes) {
  if (bytes.size() != kWindowSize)
    throw std::invalid_argument("decoder input must be exactly 16 bytes");

  std::array<uint32_t, kWindowSize / sizeof(uint32_t)> words{};
  for (size_t word = 0; word < words.size(); ++word) {
    for (size_t byte = 0; byte < sizeof(uint32_t); ++byte)
      words[word] |= static_cast<uint32_t>(bytes[word * sizeof(uint32_t) + byte]) << (byte * 8);
  }

  util::StringDiagnostic rejection;
  DecodeResult decoded = decoder.decode(words.data(), rejection.emitter());
  if (decoded.failed()) {
    DecodeRecord record;
    record.rejection = rejection.message();
    return record;
  }
  std::unique_ptr<Instruction> inst = std::move(decoded).value();

  const int size = inst->size();
  if (size != 4 && size != 8 && size != 12 && size != 16)
    fail_invariant("instruction size is outside {4, 8, 12, 16}");
  if (!inst->raw_encoding())
    fail_invariant("instruction has no raw encoding");
  // AMDGPU decoders retain the base word, but do not uniformly expose extension
  // words as contiguous storage. Keep this invariant to the common contract.
  if (inst->raw_encoding()[0] != words[0]) {
    std::ostringstream message;
    message << "raw encoding differs at word 0 for " << inst->mnemonic() << " (size " << size
            << "): got 0x" << std::hex << inst->raw_encoding()[0] << ", expected 0x" << words[0];
    fail_invariant(message.str());
  }
  if (inst->mnemonic().empty())
    fail_invariant("instruction mnemonic is empty");
  for (int i = 0; i < inst->num_dst_operands(); ++i) {
    if (!inst->dst_operand(i))
      fail_invariant("destination operand is null");
  }
  for (int i = 0; i < inst->num_src_operands(); ++i) {
    if (!inst->src_operand(i))
      fail_invariant("source operand is null");
  }

  const std::string disassembly = inst->disassemble();
  if (disassembly.empty())
    fail_invariant("instruction disassembly is empty");
  const std::string_view mnemonic = inst->mnemonic();
  const size_t dual_separator = mnemonic.find(" :: ");
  bool mnemonic_matches =
      dual_separator == std::string_view::npos
          ? disassembly.starts_with(mnemonic)
          : disassembly.starts_with(mnemonic.substr(0, dual_separator)) &&
                disassembly.find(mnemonic.substr(dual_separator)) != std::string::npos;
  if (!mnemonic_matches && dual_separator == std::string_view::npos) {
    const std::string_view rendered =
        std::string_view(disassembly).substr(0, disassembly.find(' '));
    const auto semantic_base =
        mnemonic.ends_with("_e32") ? mnemonic.substr(0, mnemonic.size() - 4) : mnemonic;
    auto rendered_base = rendered;
    if (rendered_base.ends_with("_e64_dpp"))
      rendered_base.remove_suffix(8);
    else if (rendered_base.ends_with("_dpp"))
      rendered_base.remove_suffix(4);
    mnemonic_matches = rendered_base == semantic_base;
  }
  if (!mnemonic_matches) {
    std::ostringstream message;
    message << "disassembly mnemonic does not match semantic mnemonic '" << mnemonic << "': '"
            << disassembly << '\'';
    fail_invariant(message.str());
  }

  DecodeRecord record;
  record.valid = true;
  record.size = size;
  record.encoding_id = inst->encoding_id();
  record.opcode = inst->opcode();
  record.mnemonic = inst->mnemonic();
  record.disassembly = disassembly;
  for (int i = 0; i < inst->num_dst_operands(); ++i) {
    const Operand *operand = inst->dst_operand(i);
    if (!operand)
      fail_invariant("destination operand is null");
    record.destinations.emplace_back(operand->name());
  }
  for (int i = 0; i < inst->num_src_operands(); ++i) {
    const Operand *operand = inst->src_operand(i);
    if (!operand)
      fail_invariant("source operand is null");
    record.sources.emplace_back(operand->name());
  }
  return record;
}

std::string record_json(const DecodeRecord &record, std::span<const uint8_t> bytes,
                        std::string_view input_name) {
  std::ostringstream out;
  out << '{';
  if (!input_name.empty())
    out << "\"input\":\"" << json_escape(input_name) << "\",";
  out << "\"bytes\":\"";
  for (const uint8_t byte : bytes)
    out << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(byte);
  out << std::dec << "\",\"status\":\"" << (record.valid ? "valid" : "invalid") << '"';
  if (record.valid) {
    out << ",\"size\":" << record.size << ",\"encoding_id\":" << record.encoding_id
        << ",\"opcode\":" << record.opcode << ",\"mnemonic\":\"" << json_escape(record.mnemonic)
        << "\",\"disassembly\":\"" << json_escape(record.disassembly) << "\",\"destinations\":";
    emit_string_array(out, record.destinations);
    out << ",\"sources\":";
    emit_string_array(out, record.sources);
  } else {
    out << ",\"rejection\":\"" << json_escape(record.rejection) << '"';
  }
  out << '}';
  return out.str();
}

} // namespace rocjitsu::decode_fuzz
