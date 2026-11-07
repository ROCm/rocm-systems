// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/export.h"

#include "addc/common.hpp"
#include "addc/mca/registers.hpp"
#include "addc/product.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace addc::mca
{

struct CpuLocation
{
    std::optional<int32_t> ccd;
    std::optional<int32_t> core;
    std::optional<int32_t> thread;
};

struct HbmLocation
{
    std::optional<uint32_t> col;
    std::optional<uint32_t> bank;
    std::optional<uint32_t> row;
    std::optional<uint32_t> pc;
    std::optional<uint32_t> sid;
};

struct ErrorLocation
{
    std::optional<uint32_t> socket;
    std::optional<std::string> die;
    std::optional<CpuLocation> cpu;
    std::optional<HbmLocation> hbm;
};

struct DecodedMca
{
    std::optional<std::string> bank;
    std::optional<std::string> error_type;
    std::string severity;
    std::optional<ErrorLocation> location;
    std::optional<std::string> instance;
    std::optional<std::string> description;
    std::optional<std::string> syndrome;
    std::optional<std::string> address;
    std::vector<int> afid = {kAfidSentinel};
    McaRegisters regs;
};

[[nodiscard]] nlohmann::ordered_json to_json(const DecodedMca& decoded);

// Everything captured for one MCA bank that decode_event() needs: the SMCA bank
// registers plus an optional firmware-translated physical address. transaddr is
// a PPR-only input, not a semantic decode input; sources without it leave it 0.
// cpuid_eax is the CPUID EAX of the processor that logged the bank, when the
// source carries one; it lets a product tell silicon variants apart (for
// example SP7 and SP8 on Venice). Sources without it leave it empty.
struct BankCapture
{
    McaRegisters regs;
    uint64_t transaddr = 0; ///< translated physical address, for PPR
    std::optional<uint32_t> cpuid_eax = std::nullopt; ///< CPUID EAX, when known
};

// One decoded MCA bank with the common-schema pieces derived by Base. The
// `analysis` field is intentionally absent because its placement is report and
// facade policy rather than part of MCA decoding.
struct McaResult
{
    DecodedMca decoded;
    nlohmann::ordered_json event_report;
    nlohmann::ordered_json validation;
    nlohmann::ordered_json ppr;
};

// Transitional name used by upper-tier code while Wave 6 lands bottom-up.
using McaEvent = McaResult;

/// Decode the tier-independent Base fields without selecting a product.
///
/// When `product` is given, severity follows that product's definition (see
/// `decode_severity(regs, product)`).
[[nodiscard]] McaResult decode_common(
    const BankCapture& bank, uint32_t afid = kAfidSentinel,
    std::optional<product::ProductId> product = std::nullopt);

/// Severity with poison evaluated first (Instinct GPUs and unspecified product).
[[nodiscard]] std::string decode_severity(McaRegisters regs);

/// Product-aware severity. Embedded and EPYC products (embedded_2005, genoa,
/// turin, venice) evaluate poison last, so a poisoned record that also has
/// pcc, or uc and tcc, stays Fatal. Every other product uses the poison-first
/// definition above.
[[nodiscard]] std::string decode_severity(McaRegisters regs,
                                          product::ProductId product);

/// Complete Base MCA operation selected explicitly by the tier composition.
using McaDecodeFunction = std::optional<McaResult> (*)(
    std::string_view, const BankCapture&);

[[nodiscard]] std::optional<McaResult> try_decode_project_event(
    std::string_view project, const BankCapture& bank);

[[nodiscard]] DecodedMca decode(
    std::string_view project, uint64_t status, uint64_t ipid, uint64_t synd = 0,
    uint64_t addr = 0, uint64_t misc0 = 0, uint64_t misc1 = 0);

[[nodiscard]] std::string decode_to_json(
    std::string_view project, uint64_t status, uint64_t ipid, uint64_t synd = 0,
    uint64_t addr = 0, uint64_t misc0 = 0, uint64_t misc1 = 0);

} // namespace addc::mca
