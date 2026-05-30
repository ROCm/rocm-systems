/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_BITSET_H_INCLUDED
#define PARAM_PARSER_BITSET_H_INCLUDED

#include "param/parser_common.h"
#include "param/utils.h"

#include <type_traits>

namespace nccl {
namespace param {
namespace parser {

template <typename EnumT, size_t N>
struct bitsetCtx {
  ncclOptionSet<EnumT, N> options;
  char delimiter;
};

template <typename EnumT, size_t N>
ncclResult_t bitsetResolve(const void* ctx, const char* input,
                           std::underlying_type_t<EnumT>& out) {
  using ResultT = std::underlying_type_t<EnumT>;
  if (input == nullptr) return ncclInvalidArgument;
  auto& bc = *static_cast<const bitsetCtx<EnumT, N>*>(ctx);
  ResultT res = 0;
  std::string str(input);
  size_t start = 0;
  while (start <= str.size()) {
    size_t end = str.find(bc.delimiter, start);
    if (end == std::string::npos) end = str.size();
    std::string token = nccl::param::utils::trim(str.substr(start, end - start));
    if (!token.empty()) {
      bool found = false;
      for (const auto& opt : bc.options) {
        if (nccl::param::utils::iequals(opt.name, token)) {
          res |= static_cast<ResultT>(opt.value); found = true; break;
        }
      }
      if (!found) return ncclInvalidArgument;
    }
    if (end == str.size()) break;
    start = end + 1;
  }
  out = res;
  return ncclSuccess;
}

template <typename EnumT, size_t N>
bool bitsetValidate(const void*, const std::underlying_type_t<EnumT>&) { return true; }

template <typename EnumT, size_t N>
std::string bitsetToString(const void* ctx, const std::underlying_type_t<EnumT>& value) {
  using ResultT = std::underlying_type_t<EnumT>;
  auto& bc = *static_cast<const bitsetCtx<EnumT, N>*>(ctx);
  std::string strResult;
  ResultT remaining = value;

  // First check for exact matches (like "ALL")
  for (const auto& opt : bc.options) {
    if (static_cast<ResultT>(opt.value) == value) {
      return opt.name;
    }
  }

  // Otherwise, decompose into individual bits
  auto isSingleBit = [](ResultT v) -> bool {
    using U = std::make_unsigned_t<ResultT>;
    U uv = static_cast<U>(v);
    return uv != 0 && (uv & (uv - 1)) == 0;
  };
  for (const auto& opt : bc.options) {
    ResultT optVal = static_cast<ResultT>(opt.value);
    if (!isSingleBit(optVal)) continue;  // skip composite aliases like MOST
    if (optVal != 0 && (remaining & optVal) == optVal) {
      if (!strResult.empty()) strResult += ",";
      strResult += opt.name;
      remaining &= ~optVal;
    }
  }
  return strResult.empty() ? "NONE" : strResult;
}

} // namespace parser
} // namespace param
} // namespace nccl

// ncclParamBitsetOf: Create a parser for bitmask types
// EnumT is the enum type used in options, ResultT is derived from its underlying type
template <typename EnumT, size_t N>
ncclParamParser<std::underlying_type_t<EnumT>> ncclParamBitsetOf(ncclOptionSet<EnumT, N> options,
                                                                 char delimiter = ',') {
  using namespace nccl::param::parser;
  auto ctx = std::make_shared<bitsetCtx<EnumT, N>>(
    bitsetCtx<EnumT, N>{std::move(options), delimiter});
  std::string d = "Comma-separated list of:";
  for (const auto& opt : ctx->options) {
    d += "\n        ";
    d += opt.name;
    if (opt.desc != nullptr) { d += " - "; d += opt.desc; }
  }
  return {bitsetResolve<EnumT, N>, bitsetValidate<EnumT, N>, bitsetToString<EnumT, N>,
          std::move(ctx), std::move(d)};
}

#endif /* PARAM_PARSER_BITSET_H_INCLUDED */
