// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_inventory_diagnostics.h"

#include <optional>
#include <string_view>

namespace rocjitsu {
namespace {

void append_decoded_operand(std::string &summary, std::string_view name,
                            std::optional<int64_t> value) {
  if (!value)
    return;
  if (!summary.empty())
    summary += ',';
  summary += name;
  summary += '=';
  summary += std::to_string(*value);
}

} // namespace

std::string consan_fixed_hex(uint64_t value, unsigned digits) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(digits, '0');
  for (unsigned i = 0; i < digits; ++i) {
    result[digits - i - 1] = kHex[value & 0xfu];
    value >>= 4u;
  }
  return result;
}

std::string consan_atomic_semantic_role(const ConSanAtomicSite &site) {
  if (site.raw_th.value_or(0) != 0 || site.returns_old_value.value_or(false))
    return "atomic-acquire-rmw";
  return "atomic-release-rmw";
}

std::string consan_barrier_decoded_operands(const ConSanBarrierSite &site, uint32_t encoding) {
  std::string result = "encoding=0x" + consan_fixed_hex(encoding, 8);
  result += ",barrier_id=";
  result += site.barrier_id ? std::to_string(*site.barrier_id) : "-";
  result += ",operand_source=";
  result += consan_barrier_operand_source_name(site.operand_source);
  result += ",scope=";
  result += consan_barrier_scope_name(site.scope);
  result += ",raw_selector=";
  result += site.raw_operand_selector ? std::to_string(*site.raw_operand_selector) : "-";
  result += ",literal_width_bits=";
  result += site.literal_width_bits ? std::to_string(*site.literal_width_bits) : "-";
  result += ",literal_value=";
  result += site.literal_value ? std::to_string(*site.literal_value) : "-";
  result += ",raw_simm16=";
  result += site.raw_simm16 ? std::to_string(*site.raw_simm16) : "-";
  return result;
}

std::string consan_atomic_decoded_operands(const ConSanAtomicSite &site) {
  std::string result;
  append_decoded_operand(result, "dst_vgpr", site.dst_vgpr);
  append_decoded_operand(result, "addr_vgpr", site.addr_vgpr);
  append_decoded_operand(result, "data_vgpr", site.data_vgpr);
  append_decoded_operand(result, "saddr_sgpr", site.saddr_sgpr);
  if (site.raw_scale_offset)
    append_decoded_operand(result, "raw_scale_offset", *site.raw_scale_offset ? 1 : 0);
  append_decoded_operand(result, "raw_ioffset", site.raw_ioffset);
  append_decoded_operand(result, "raw_scope", site.raw_scope);
  append_decoded_operand(result, "raw_th", site.raw_th);
  append_decoded_operand(result, "raw_addr", site.raw_addr);
  append_decoded_operand(result, "raw_data0", site.raw_data0);
  append_decoded_operand(result, "raw_data1", site.raw_data1);
  append_decoded_operand(result, "raw_vdata", site.raw_vdata);
  append_decoded_operand(result, "raw_rsrc", site.raw_rsrc);
  append_decoded_operand(result, "raw_soffset", site.raw_soffset);
  if (site.returns_old_value)
    append_decoded_operand(result, "returns_old", *site.returns_old_value ? 1 : 0);
  return result.empty() ? "-" : result;
}

std::string consan_lds_decoded_operands(const ConSanAccessOperandFacts &operands) {
  std::string result;
  append_decoded_operand(result, "dst_vgpr", operands.destination_vgpr);
  append_decoded_operand(result, "dst_accvgpr", operands.destination_accvgpr);
  append_decoded_operand(result, "addr_vgpr", operands.address_vgpr);
  append_decoded_operand(result, "data_vgpr", operands.data_vgpr);
  append_decoded_operand(result, "second_data_vgpr", operands.second_data_vgpr);
  return result.empty() ? "-" : result;
}

std::string consan_ordinary_memory_decoded_operands(const ConSanOrdinaryMemorySite &site) {
  std::string result;
  append_decoded_operand(result, "dst_vgpr", site.destination_vgpr);
  append_decoded_operand(result, "addr_vgpr", site.address_vgpr);
  append_decoded_operand(result, "addr_sgpr", site.address_sgpr);
  append_decoded_operand(result, "value_vgpr", site.value_vgpr);
  append_decoded_operand(result, "raw_saddr", site.raw_saddr);
  append_decoded_operand(result, "raw_nv", site.raw_nv);
  if (site.raw_scale_offset)
    append_decoded_operand(result, "raw_scale_offset", *site.raw_scale_offset ? 1 : 0);
  append_decoded_operand(result, "raw_sve", site.raw_sve);
  append_decoded_operand(result, "raw_vaddr", site.raw_vaddr);
  append_decoded_operand(result, "raw_vsrc", site.raw_vsrc);
  append_decoded_operand(result, "raw_vdst", site.raw_vdst);
  append_decoded_operand(result, "raw_ioffset", site.raw_ioffset);
  append_decoded_operand(result, "raw_scope", site.raw_scope);
  append_decoded_operand(result, "raw_th", site.raw_th);
  return result.empty() ? "-" : result;
}

} // namespace rocjitsu
