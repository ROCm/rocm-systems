// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/mca/decoder.hpp"

#include "addc/detail/format.hpp"
#ifdef ADDC_HAS_MI300A
#include "addc/mca/mi300a.hpp"
#endif
#if defined(ADDC_HAS_MI300X) || defined(ADDC_HAS_MI325X) || \
    defined(ADDC_HAS_MI350)
#include "addc/mca/mi300x.hpp"
#endif
#ifdef ADDC_HAS_MI450
#include "addc/mca/mi450.hpp"
#endif
#include "addc/product.hpp"
#include "addc/schema_version.hpp"
#ifdef ADDC_HAS_VENICE
#include "addc/mca/venice.hpp"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>

namespace addc::mca
{

std::string decode_severity(McaRegisters regs)
{
    const auto& s = regs.status;
    if (s.poison)
    {
        return "Non-Fatal";
    }
    if (s.pcc)
    {
        return "Fatal";
    }
    if (!s.pcc && s.uc && s.tcc)
    {
        return "Fatal";
    }
    if (!s.pcc && s.uc && !s.tcc)
    {
        return "Non-Fatal";
    }
    if (!s.pcc && !s.uc && !s.tcc && s.deferred)
    {
        return "Non-Fatal";
    }
    if (!s.pcc && !s.uc && !s.tcc && !s.deferred)
    {
        return "Corrected";
    }
    return "Unknown";
}

McaResult decode_common(const BankCapture& bank, uint32_t afid)
{
    const McaRegisters& regs = bank.regs;
    McaResult event;
    event.decoded = DecodedMca{
        .bank = std::nullopt,
        .error_type = std::nullopt,
        .severity = addc::mca::decode_severity(regs),
        .location = std::nullopt,
        .instance = std::nullopt,
        .description = std::nullopt,
        .syndrome = std::nullopt,
        .address = std::nullopt,
        .afid = addc::afid_list(static_cast<int>(afid)),
        .regs = regs,
    };
    event.event_report = to_json(event.decoded);
    event.validation = nlohmann::ordered_json{
        {"struct", true},        {"bank", false},
        {"corrupted", false},    {"possible_non_mca_fatal", false},
        {"mtl_raw_data", false},
    };
    event.ppr = nullptr;
    return event;
}

std::optional<McaResult> try_decode_project_event(
    std::string_view project, const BankCapture& bank)
{
    const auto* definition = product::find_compiled_product(project);
    if (definition == nullptr || !definition->capabilities.mca)
    {
        return std::nullopt;
    }

    switch (definition->identity.id)
    {
#ifdef ADDC_HAS_VENICE
        case product::ProductId::venice:
            return base::venice::decode(bank);
#endif
#ifdef ADDC_HAS_MI300A
        case product::ProductId::mi300a:
            return base::mi300a::decode(bank);
#endif
#ifdef ADDC_HAS_MI300X
        case product::ProductId::mi300x:
            return base::mi300x::decode(bank);
#endif
#ifdef ADDC_HAS_MI325X
        case product::ProductId::mi325x:
            return base::mi300x::decode(bank);
#endif
#ifdef ADDC_HAS_MI350
        case product::ProductId::mi350:
            return base::mi300x::decode(bank);
#endif
#ifdef ADDC_HAS_MI450
        case product::ProductId::mi450:
            return base::mi450::decode(bank);
#endif
        default:
            return decode_common(bank);
    }
}

namespace
{

nlohmann::ordered_json cpuLocationToJson(const CpuLocation& cpu)
{
    nlohmann::ordered_json j;
    j["ccd"] = cpu.ccd.has_value() ? nlohmann::ordered_json(*cpu.ccd)
                                   : nlohmann::ordered_json(nullptr);
    j["core"] = cpu.core.has_value() ? nlohmann::ordered_json(*cpu.core)
                                     : nlohmann::ordered_json(nullptr);
    j["thread"] = cpu.thread.has_value() ? nlohmann::ordered_json(*cpu.thread)
                                         : nlohmann::ordered_json(nullptr);
    return j;
}

nlohmann::ordered_json hbmLocationToJson(const HbmLocation& hbm)
{
    nlohmann::ordered_json j;
    j["col"] = hbm.col.has_value() ? nlohmann::ordered_json(*hbm.col)
                                   : nlohmann::ordered_json(nullptr);
    j["bank"] = hbm.bank.has_value() ? nlohmann::ordered_json(*hbm.bank)
                                     : nlohmann::ordered_json(nullptr);
    j["row"] = hbm.row.has_value() ? nlohmann::ordered_json(*hbm.row)
                                   : nlohmann::ordered_json(nullptr);
    j["pc"] = hbm.pc.has_value() ? nlohmann::ordered_json(*hbm.pc)
                                 : nlohmann::ordered_json(nullptr);
    j["sid"] = hbm.sid.has_value() ? nlohmann::ordered_json(*hbm.sid)
                                   : nlohmann::ordered_json(nullptr);
    return j;
}

nlohmann::ordered_json errorLocationToJson(
    const ErrorLocation& loc, const std::optional<std::string>& instance)
{
    nlohmann::ordered_json j;
    j["instance"] = instance ? nlohmann::ordered_json(*instance)
                             : nlohmann::ordered_json(nullptr);
    j["socket"] = loc.socket.has_value() ? nlohmann::ordered_json(*loc.socket)
                                         : nlohmann::ordered_json(nullptr);
    j["die"] = loc.die.has_value() ? nlohmann::ordered_json(*loc.die)
                                   : nlohmann::ordered_json(nullptr);
    j["cpu"] = loc.cpu.has_value() ? cpuLocationToJson(*loc.cpu)
                                   : nlohmann::ordered_json(nullptr);
    j["hbm"] = loc.hbm.has_value() ? hbmLocationToJson(*loc.hbm)
                                   : nlohmann::ordered_json(nullptr);
    return j;
}

} // namespace

nlohmann::ordered_json to_json(const DecodedMca& d)
{
    nlohmann::ordered_json j;
    if (d.bank)
    {
        std::string bank_lower = *d.bank;
        std::transform(bank_lower.begin(), bank_lower.end(), bank_lower.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        j["bank"] = std::move(bank_lower);
    }
    else
    {
        j["bank"] = nullptr;
    }
    j["error_location"] =
        d.location.has_value()
            ? errorLocationToJson(*d.location, d.instance)
            : [&] {
                  nlohmann::ordered_json loc;
                  loc["instance"] = d.instance
                                        ? nlohmann::ordered_json(*d.instance)
                                        : nlohmann::ordered_json(nullptr);
                  loc["socket"] = nullptr;
                  loc["die"] = nullptr;
                  loc["cpu"] = nullptr;
                  loc["hbm"] = nullptr;
                  return loc;
              }();
    j["error_type"] = d.error_type ? nlohmann::ordered_json(*d.error_type)
                                   : nlohmann::ordered_json(nullptr);
    j["error_description"] = d.description
                                 ? nlohmann::ordered_json(*d.description)
                                 : nlohmann::ordered_json(nullptr);
    j["error_severity"] = d.severity;
    j["error_syndrome"] =
        d.syndrome ? nlohmann::ordered_json(
                         nlohmann::ordered_json::array({*d.syndrome}))
                   : nlohmann::ordered_json(nullptr);
    j["error_address"] = d.address ? nlohmann::ordered_json(*d.address)
                                   : nlohmann::ordered_json(nullptr);
    j["signature_id"] =
        addc::format("0x{:016x}{:016x}{:016x}", d.regs.status.raw,
                     d.regs.ipid.raw, d.regs.synd.raw);
    {
        const auto& s = d.regs.status;
        nlohmann::ordered_json misc;
        misc["uc"] = s.uc;
        misc["pcc"] = s.pcc;
        misc["tcc"] = s.tcc;
        misc["deferred"] = s.deferred;
        misc["addr_v"] = s.addr_v;
        misc["cecc"] = s.cecc;
        misc["uecc"] = s.uecc;
        misc["poison"] = s.poison;
        misc["transparent"] = s.transparent;
        misc["scrub"] = s.scrub;
        j["misc_status"] = std::move(misc);
    }
    return j;
}

DecodedMca decode(std::string_view project, uint64_t status, uint64_t ipid,
                  uint64_t synd, uint64_t addr, uint64_t misc0, uint64_t misc1)
{
    const auto decoded = try_decode_project_event(
        project,
        BankCapture{.regs = McaRegisters{status, ipid, synd, addr, misc0,
                                         misc1}});
    if (!decoded)
    {
        throw std::invalid_argument(addc::format(
            "unsupported_by_build: MCA project '{}' is not compiled", project));
    }
    return decoded->decoded;
}

std::string decode_to_json(std::string_view project, uint64_t status,
                           uint64_t ipid, uint64_t synd, uint64_t addr,
                           uint64_t misc0, uint64_t misc1)
{
    const auto decoded = decode(project, status, ipid, synd, addr, misc0,
                                misc1);

    nlohmann::ordered_json da;
    da["status"] = addc::format("0x{:016x}", status);
    da["ipid"] = addc::format("0x{:016x}", ipid);
    da["synd"] = addc::format("0x{:016x}", synd);
    da["addr"] = addc::format("0x{:016x}", addr);
    da["misc0"] = addc::format("0x{:016x}", misc0);
    da["misc1"] = addc::format("0x{:016x}", misc1);

    nlohmann::ordered_json event;
    event["metadata"] = nlohmann::ordered_json{
        {"event_class", "mca"},    {"section_index", nullptr},
        {"context_type", nullptr}, {"timestamp", nullptr},
        {"addc_version", nullptr},
    };
    event["platform"] = nullptr;
    event["event_report"] = to_json(decoded);
    event["data_array"] = std::move(da);
    event["validation"] = nullptr;
    event["analysis"] = nlohmann::ordered_json{{"afid", decoded.afid}};

    nlohmann::ordered_json j;
    j["schema_version"] = addc::kSchemaVersion;
    j["tool_version"] = ADDC_VERSION;
    j["source"] = nlohmann::ordered_json{
        {"type", "mca"},         {"filename", nullptr},
        {"timestamp", nullptr},  {"platform_id", nullptr},
        {"creator_id", nullptr},
    };
    j["events"] = nlohmann::ordered_json::array({std::move(event)});

    return j.dump();
}

} // namespace addc::mca
