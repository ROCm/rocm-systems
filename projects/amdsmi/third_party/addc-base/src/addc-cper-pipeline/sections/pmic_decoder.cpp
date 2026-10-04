// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/pmic_decoder.hpp"

#include "addc/common.hpp"
#include "addc/detail/format.hpp"
#include "addc/pipeline/section_data.hpp"

namespace addc::pipeline
{

std::vector<PipelineEvent> decode_pmic(
    const SectionDescriptor& descriptor,
    const AmdEpycPmicSection& s, const DecodeContext& context,
    [[maybe_unused]] std::string* error_out)
{

    PipelineEvent ev;
    ev.metadata["section_index"] = context.section_index;
    ev.metadata["context_structure_index"] = 0;
    ev.metadata["context_type"] = "PMIC";
    ev.metadata["event_class"] = "pmic";
    ev.metadata["event_index"] = 0;
    ev.metadata["timestamp"] = context.timestamp;
    if (!descriptor.section_severity.empty())
    {
        ev.metadata["severity"] = descriptor.section_severity;
    }
    ev.metadata["addc_version"] = {{"name", s.platform_name},
                                   {"value", s.addc_version}};

    ev.platform["fru"] = s.fru;
    ev.platform["fru_id"] = s.fru_id;
    ev.platform["cpu"] = nullptr;
    ev.platform["gpu"] = nullptr;

    ev.event_report["section_version"] =
        addc::format("0x{:08x}", s.section_version);
    ev.event_report["section_length"] =
        addc::format("0x{:08x}", s.section_length);

    nlohmann::ordered_json id_j;
    id_j["socket_id"] = addc::format("0x{:02x}", s.socket_id);
    id_j["channel_id"] = addc::format("0x{:02x}", s.channel_id);
    id_j["dimm_slot"] = addc::format("0x{:02x}", s.dimm_slot);
    id_j["pmic_index"] = addc::format("0x{:02x}", s.pmic_index);
    ev.event_report["error_location"] = std::move(id_j);

    ev.data_array = nullptr;

    // ── data_array
    // ──────────────────────────────────────────────────────────── Bit-decode
    // is based on the JEDEC JESD301-1 PMIC5020 register map (public). Producing
    // this at base level keeps the JSON self-contained so BMC and downstream
    // tools can consume decoded fields without re-implementing the spec.
    nlohmann::ordered_json da;
    da["section_version"] = addc::format("0x{:08x}", s.section_version);
    da["section_length"] = addc::format("0x{:08x}", s.section_length);

    nlohmann::ordered_json da_valid;
    da_valid["raw"] = addc::format("0x{:08x}", s.valid_bits);
    da_valid["identification_valid"] = static_cast<int>(s.valid_bits & 0x1U);
    da_valid["fault_status_valid"] =
        static_cast<int>((s.valid_bits >> 1U) & 0x1U);
    da_valid["error_logs_valid"] =
        static_cast<int>((s.valid_bits >> 2U) & 0x1U);
    da["valid"] = std::move(da_valid);

    nlohmann::ordered_json da_id;
    da_id["socket_id"] = addc::format("0x{:02x}", s.socket_id);
    da_id["channel_id"] = addc::format("0x{:02x}", s.channel_id);
    da_id["dimm_slot"] = addc::format("0x{:02x}", s.dimm_slot);
    da_id["pmic_index"] = addc::format("0x{:02x}", s.pmic_index);
    da["identification"] = std::move(da_id);

    nlohmann::ordered_json da_fs;
    da_fs["Log_0x04"] = addc::format("0x{:02x}", s.log_04);
    da_fs["Log_0x05"] = addc::format("0x{:02x}", s.log_05);
    da_fs["Log_0x06"] = addc::format("0x{:02x}", s.log_06);
    da_fs["Reg_0x08"] = addc::format("0x{:02x}", s.reg_08);
    da_fs["Reg_0x09"] = addc::format("0x{:02x}", s.reg_09);
    da_fs["Reg_0x0A"] = addc::format("0x{:02x}", s.reg_0A);
    da_fs["Reg_0x0B"] = addc::format("0x{:02x}", s.reg_0B);
    da_fs["Reg_0x33"] = addc::format("0x{:02x}", s.reg_33);

    nlohmann::ordered_json decoded;
    // R04 — Global Error Count / Log
    decoded["global_error_count"] = static_cast<int>((s.log_04 >> 7) & 0x1U);
    decoded["err_log_buck_ov_uv"] = static_cast<int>((s.log_04 >> 6) & 0x1U);
    decoded["err_log_vin_bulk_ov"] = static_cast<int>((s.log_04 >> 5) & 0x1U);
    decoded["err_log_critical_temp"] = static_cast<int>((s.log_04 >> 4) & 0x1U);
    decoded["global_error_count_en"] = static_cast<int>(s.log_04 & 0xFU);
    // R05 — PMIC Power On Status  (bit[1]=SWA, [2]=SWB, [3]=SWC, [4]=SWD,
    // [7:5]=error_log)
    decoded["swa_power_not_good"] = static_cast<int>((s.log_05 >> 1) & 0x1U);
    decoded["swb_power_not_good"] = static_cast<int>((s.log_05 >> 2) & 0x1U);
    decoded["swc_power_not_good"] = static_cast<int>((s.log_05 >> 3) & 0x1U);
    decoded["swd_power_not_good"] = static_cast<int>((s.log_05 >> 4) & 0x1U);
    decoded["pmic_error_log"] = static_cast<int>((s.log_05 >> 5) & 0x7U);
    // R06 — SWx UV/OV Lockout Log  (bit[0..2]=SWA..SWC_UVLO,
    // [3..6]=SWA..SWD_OV, [7]=SWD_UVLO)
    decoded["swa_uvlo"] = static_cast<int>((s.log_06 >> 0) & 0x1U);
    decoded["swb_uvlo"] = static_cast<int>((s.log_06 >> 1) & 0x1U);
    decoded["swc_uvlo"] = static_cast<int>((s.log_06 >> 2) & 0x1U);
    decoded["swd_uvlo"] = static_cast<int>((s.log_06 >> 7) & 0x1U);
    decoded["swa_ov"] = static_cast<int>((s.log_06 >> 3) & 0x1U);
    decoded["swb_ov"] = static_cast<int>((s.log_06 >> 4) & 0x1U);
    decoded["swc_ov"] = static_cast<int>((s.log_06 >> 5) & 0x1U);
    decoded["swd_ov"] = static_cast<int>((s.log_06 >> 6) & 0x1U);
    // R08 — Real-time Fault Status
    decoded["vin_bulk_pg_not_good"] = static_cast<int>((s.reg_08 >> 7) & 0x1U);
    decoded["crit_temp_shutdown"] = static_cast<int>((s.reg_08 >> 6) & 0x1U);
    decoded["swa_pg_not_good"] = static_cast<int>((s.reg_08 >> 5) & 0x1U);
    decoded["swb_pg_not_good"] = static_cast<int>((s.reg_08 >> 4) & 0x1U);
    decoded["swc_pg_not_good"] = static_cast<int>((s.reg_08 >> 3) & 0x1U);
    decoded["swd_pg_not_good"] = static_cast<int>((s.reg_08 >> 2) & 0x1U);
    decoded["vin_mgmt_ov"] = static_cast<int>((s.reg_08 >> 1) & 0x1U);
    decoded["vin_bulk_ov"] = static_cast<int>((s.reg_08 >> 0) & 0x1U);
    // R09 — Real-time Warning Status
    decoded["high_temp_warning"] = static_cast<int>((s.reg_09 >> 7) & 0x1U);
    decoded["vbias_power_not_good"] = static_cast<int>((s.reg_09 >> 6) & 0x1U);
    decoded["vout_1v8_power_not_good"] =
        static_cast<int>((s.reg_09 >> 5) & 0x1U);
    decoded["vin_mgmt_switchover"] = static_cast<int>((s.reg_09 >> 4) & 0x1U);
    decoded["swa_high_current"] = static_cast<int>((s.reg_09 >> 3) & 0x1U);
    decoded["swb_high_current"] = static_cast<int>((s.reg_09 >> 2) & 0x1U);
    decoded["swc_high_current"] = static_cast<int>((s.reg_09 >> 1) & 0x1U);
    decoded["swd_high_current"] = static_cast<int>((s.reg_09 >> 0) & 0x1U);
    // R0A — SWx OV / Comm Error Status
    decoded["swa_ov_rt"] = static_cast<int>((s.reg_0A >> 1) & 0x1U);
    decoded["swb_ov_rt"] = static_cast<int>((s.reg_0A >> 2) & 0x1U);
    decoded["swc_ov_rt"] = static_cast<int>((s.reg_0A >> 3) & 0x1U);
    decoded["swd_ov_rt"] = static_cast<int>((s.reg_0A >> 4) & 0x1U);
    decoded["pec_error"] = static_cast<int>((s.reg_0A >> 5) & 0x1U);
    decoded["parity_error"] = static_cast<int>((s.reg_0A >> 6) & 0x1U);
    decoded["ibi_global"] = static_cast<int>((s.reg_0A >> 7) & 0x1U);
    // R0B — SWx Current Limiter / UVLO Status
    decoded["swa_curr_limiter"] = static_cast<int>((s.reg_0B >> 7) & 0x1U);
    decoded["swb_curr_limiter"] = static_cast<int>((s.reg_0B >> 6) & 0x1U);
    decoded["swc_curr_limiter"] = static_cast<int>((s.reg_0B >> 5) & 0x1U);
    decoded["swd_curr_limiter"] = static_cast<int>((s.reg_0B >> 4) & 0x1U);
    decoded["swa_uvlo_rt"] = static_cast<int>((s.reg_0B >> 3) & 0x1U);
    decoded["swb_uvlo_rt"] = static_cast<int>((s.reg_0B >> 2) & 0x1U);
    decoded["swc_uvlo_rt"] = static_cast<int>((s.reg_0B >> 1) & 0x1U);
    decoded["swd_uvlo_rt"] = static_cast<int>((s.reg_0B >> 0) & 0x1U);
    // R33 — Temperature / Summary Status
    decoded["temperature_code"] = static_cast<int>(s.reg_33 & 0x7U);
    {
        constexpr std::string_view kTempTable[] = {
            "< 85\u00B0C",    "85-95\u00B0C",   "95-105\u00B0C",
            "105-115\u00B0C", "115-125\u00B0C", "125-135\u00B0C",
            "> 135\u00B0C",   "Reserved"};
        decoded["temperature"] = std::string{kTempTable[s.reg_33 & 0x7U]};
    }
    decoded["vbias_vin_bulk_uvlo"] = static_cast<int>((s.reg_33 >> 3) & 0x1U);
    decoded["vin_mgmt_pg_switchover"] =
        static_cast<int>((s.reg_33 >> 4) & 0x1U);
    decoded["vout_1v0_not_good"] = static_cast<int>((s.reg_33 >> 5) & 0x1U);
    da_fs["decoded"] = std::move(decoded);
    da["fault_status"] = std::move(da_fs);

    nlohmann::ordered_json da_pl = nlohmann::ordered_json::object();
    for (const auto& [k, v] : s.persistent_logs)
    {
        da_pl[k] = v;
    }
    da["persistent_logs"] = std::move(da_pl);
    ev.data_array = std::move(da);

    ev.validation["identification_valid"] =
        static_cast<bool>(s.valid_bits & 0x1U);
    ev.validation["fault_status_valid"] =
        static_cast<bool>((s.valid_bits >> 1U) & 0x1U);
    ev.validation["error_logs_valid"] =
        static_cast<bool>((s.valid_bits >> 2U) & 0x1U);
    ev.validation["corrupted"] = false;

    ev.analysis["afid"] = addc::afid_list(addc::kAfidSentinel);

    return {std::move(ev)};
}

} // namespace addc::pipeline
