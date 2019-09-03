/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>

#include <assert.h>
#include <sys/stat.h>
#include <stdint.h>
#include <string>
#include <map>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>

#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_kfd.h"

extern "C" {
#include "shared_mutex.h"  // NOLINT
};

namespace amd {
namespace smi {

// Device sysfs file names
static const char *kDevPerfLevelFName = "power_dpm_force_performance_level";
static const char *kDevDevIDFName = "device";
static const char *kDevVendorIDFName = "vendor";
static const char *kDevSubSysDevIDFName = "subsystem_device";
static const char *kDevSubSysVendorIDFName = "subsystem_vendor";
static const char *kDevOverDriveLevelFName = "pp_sclk_od";
static const char *kDevGPUSClkFName = "pp_dpm_sclk";
static const char *kDevGPUMClkFName = "pp_dpm_mclk";
static const char *kDevDCEFClkFName = "pp_dpm_dcefclk";
static const char *kDevFClkFName = "pp_dpm_fclk";
static const char *kDevSOCClkFName = "pp_dpm_socclk";
static const char *kDevPCIEClkFName = "pp_dpm_pcie";
static const char *kDevPowerProfileModeFName = "pp_power_profile_mode";
static const char *kDevPowerODVoltageFName = "pp_od_clk_voltage";
static const char *kDevUsageFName = "gpu_busy_percent";
static const char *kDevVBiosVerFName = "vbios_version";
static const char *kDevPCIEThruPutFName = "pcie_bw";
static const char *kDevErrCntSDMAFName = "ras/sdma_err_count";
static const char *kDevErrCntUMCFName = "ras/umc_err_count";
static const char *kDevErrCntGFXFName = "ras/gfx_err_count";
static const char *kDevErrCntFeaturesFName = "ras/features";
static const char *kDevMemPageBadFName = "ras/gpu_vram_bad_pages";
static const char *kDevMemTotGTTFName = "mem_info_gtt_total";
static const char *kDevMemTotVisVRAMFName = "mem_info_vis_vram_total";
static const char *kDevMemTotVRAMFName = "mem_info_vram_total";
static const char *kDevMemUsedGTTFName = "mem_info_gtt_used";
static const char *kDevMemUsedVisVRAMFName = "mem_info_vis_vram_used";
static const char *kDevMemUsedVRAMFName = "mem_info_vram_used";
static const char *kDevPCIEReplayCountFName = "pcie_replay_count";
static const char *kDevUniqueIdFName = "unique_id";
static const char *kDevDFCountersAvailableFName = "df_cntr_avail";
static const char *kDevMemBusyPercentFName = "mem_busy_percent";
static const char *kDevXGMIErrorFName = "xgmi_error";
static const char *kDevSerialNumberFName = "serial_number";

// Firmware version files
static const char *kDevFwVersionAsdFName = "fw_version/asd_fw_version";
static const char *kDevFwVersionCeFName = "fw_version/ce_fw_version";
static const char *kDevFwVersionDmcuFName = "fw_version/dmcu_fw_version";
static const char *kDevFwVersionMcFName = "fw_version/mc_fw_version";
static const char *kDevFwVersionMeFName = "fw_version/me_fw_version";
static const char *kDevFwVersionMecFName = "fw_version/mec_fw_version";
static const char *kDevFwVersionMec2FName = "fw_version/mec2_fw_version";
static const char *kDevFwVersionPfpFName = "fw_version/pfp_fw_version";
static const char *kDevFwVersionRlcFName = "fw_version/rlc_fw_version";
static const char *kDevFwVersionRlcSrlcFName = "fw_version/rlc_srlc_fw_version";
static const char *kDevFwVersionRlcSrlgFName = "fw_version/rlc_srlg_fw_version";
static const char *kDevFwVersionRlcSrlsFName = "fw_version/rlc_srls_fw_version";
static const char *kDevFwVersionSdmaFName = "fw_version/sdma_fw_version";
static const char *kDevFwVersionSdma2FName = "fw_version/sdma2_fw_version";
static const char *kDevFwVersionSmcFName = "fw_version/smc_fw_version";
static const char *kDevFwVersionSosFName = "fw_version/sos_fw_version";
static const char *kDevFwVersionTaRasFName = "fw_version/ta_ras_fw_version";
static const char *kDevFwVersionTaXgmiFName = "fw_version/ta_xgmi_fw_version";
static const char *kDevFwVersionUvdFName = "fw_version/uvd_fw_version";
static const char *kDevFwVersionVceFName = "fw_version/vce_fw_version";
static const char *kDevFwVersionVcnFName = "fw_version/vcn_fw_version";

static const char *kDevKFDNodePropCachesCntSName = "caches_count";
static const char *kDevKFDNodePropIoLinksCntSName = "io_links_count";
static const char *kDevKFDNodePropCPUCoreIdBaseSName = "cpu_core_id_base";
static const char *kDevKFDNodePropSimdIdBaseSName = "simd_id_base";
static const char *kDevKFDNodePropMaxWavePerSimdSName = "max_waves_per_simd";
static const char *kDevKFDNodePropLdsSzSName = "lds_size_in_kb";
static const char *kDevKFDNodePropGdsSzSName = "gds_size_in_kb";
static const char *kDevKFDNodePropNumGWSSName = "num_gws";
static const char *kDevKFDNodePropWaveFrontSizeSName = "wave_front_size";
static const char *kDevKFDNodePropArrCntSName = "array_count";
static const char *kDevKFDNodePropSimdArrPerEngSName = "simd_arrays_per_engine";
static const char *kDevKFDNodePropCuPerSimdArrSName = "cu_per_simd_array";
static const char *kDevKFDNodePropSimdPerCUSName = "simd_per_cu";
static const char *kDevKFDNodePropMaxSlotsScratchCuSName =
                                                       "max_slots_scratch_cu";
static const char *kDevKFDNodePropVendorIdSName = "vendor_id";
static const char *kDevKFDNodePropDeviceIdSName = "device_id";
static const char *kDevKFDNodePropLocationIdSName = "location_id";
static const char *kDevKFDNodePropDrmRenderMinorSName = "drm_render_minor";
static const char *kDevKFDNodePropHiveIdSName = "hive_id";
static const char *kDevKFDNodePropNumSdmaEnginesSName = "num_sdma_engines";
static const char *kDevKFDNodePropNumSdmaXgmiEngsSName =
                                                      "num_sdma_xgmi_engines";
static const char *kDevKFDNodePropMaxEngClkFCompSName =
                                                    "max_engine_clk_fcompute";
static const char *kDevKFDNodePropLocMemSzSName = "local_mem_size";
static const char *kDevKFDNodePropFwVerSName = "fw_version";
static const char *kDevKFDNodePropCapabilitySName = "capability";
static const char *kDevKFDNodePropDbgPropSName = "debug_prop";
static const char *kDevKFDNodePropSdmaFwVerSName = "sdma_fw_version";
static const char *kDevKFDNodePropMaxEngClkCCompSName =
                                                    "max_engine_clk_ccompute";
static const char *kDevKFDNodePropDomainSName = "domain";

static const std::map<DevKFDNodePropTypes, const char *> kDevKFDPropNameMap = {
    {kDevKFDNodePropCachesCnt, kDevKFDNodePropCachesCntSName},
    {kDevKFDNodePropIoLinksCnt, kDevKFDNodePropIoLinksCntSName},
    {kDevKFDNodePropCPUCoreIdBase, kDevKFDNodePropCPUCoreIdBaseSName},
    {kDevKFDNodePropSimdIdBase, kDevKFDNodePropSimdIdBaseSName},
    {kDevKFDNodePropMaxWavePerSimd, kDevKFDNodePropMaxWavePerSimdSName},
    {kDevKFDNodePropLdsSz, kDevKFDNodePropLdsSzSName},
    {kDevKFDNodePropGdsSz, kDevKFDNodePropGdsSzSName},
    {kDevKFDNodePropNumGWS, kDevKFDNodePropNumGWSSName},
    {kDevKFDNodePropWaveFrontSize, kDevKFDNodePropWaveFrontSizeSName},
    {kDevKFDNodePropArrCnt, kDevKFDNodePropArrCntSName},
    {kDevKFDNodePropSimdArrPerEng, kDevKFDNodePropSimdArrPerEngSName},
    {kDevKFDNodePropCuPerSimdArr, kDevKFDNodePropCuPerSimdArrSName},
    {kDevKFDNodePropSimdPerCU, kDevKFDNodePropSimdPerCUSName},
    {kDevKFDNodePropMaxSlotsScratchCu, kDevKFDNodePropMaxSlotsScratchCuSName},
    {kDevKFDNodePropVendorId, kDevKFDNodePropVendorIdSName},
    {kDevKFDNodePropDeviceId, kDevKFDNodePropDeviceIdSName},
    {kDevKFDNodePropLocationId, kDevKFDNodePropLocationIdSName},
    {kDevKFDNodePropDrmRenderMinor, kDevKFDNodePropDrmRenderMinorSName},
    {kDevKFDNodePropHiveId, kDevKFDNodePropHiveIdSName},
    {kDevKFDNodePropNumSdmaEngines, kDevKFDNodePropNumSdmaEnginesSName},
    {kDevKFDNodePropNumSdmaXgmiEngs, kDevKFDNodePropNumSdmaXgmiEngsSName},
    {kDevKFDNodePropMaxEngClkFComp, kDevKFDNodePropMaxEngClkFCompSName},
    {kDevKFDNodePropLocMemSz, kDevKFDNodePropLocMemSzSName},
    {kDevKFDNodePropFwVer, kDevKFDNodePropFwVerSName},
    {kDevKFDNodePropCapability, kDevKFDNodePropCapabilitySName},
    {kDevKFDNodePropDbgProp, kDevKFDNodePropDbgPropSName},
    {kDevKFDNodePropSdmaFwVer, kDevKFDNodePropSdmaFwVerSName},
    {kDevKFDNodePropMaxEngClkCComp, kDevKFDNodePropMaxEngClkCCompSName},
    {kDevKFDNodePropDomain, kDevKFDNodePropDomainSName},
};

// Strings that are found within sysfs files
static const char *kDevPerfLevelAutoStr = "auto";
static const char *kDevPerfLevelLowStr = "low";
static const char *kDevPerfLevelHighStr = "high";
static const char *kDevPerfLevelManualStr = "manual";
static const char *kDevPerfLevelStandardStr = "profile_standard";
static const char *kDevPerfLevelMinMClkStr = "profile_min_mclk";
static const char *kDevPerfLevelMinSClkStr = "profile_min_sclk";
static const char *kDevPerfLevelPeakStr = "profile_peak";
static const char *kDevPerfLevelUnknownStr = "unknown";

static const std::map<DevInfoTypes, const char *> kDevAttribNameMap = {
    {kDevPerfLevel, kDevPerfLevelFName},
    {kDevOverDriveLevel, kDevOverDriveLevelFName},
    {kDevDevID, kDevDevIDFName},
    {kDevVendorID, kDevVendorIDFName},
    {kDevSubSysDevID, kDevSubSysDevIDFName},
    {kDevSubSysVendorID, kDevSubSysVendorIDFName},
    {kDevGPUMClk, kDevGPUMClkFName},
    {kDevGPUSClk, kDevGPUSClkFName},
    {kDevDCEFClk, kDevDCEFClkFName},
    {kDevFClk, kDevFClkFName},
    {kDevSOCClk, kDevSOCClkFName},
    {kDevPCIEClk, kDevPCIEClkFName},
    {kDevPowerProfileMode, kDevPowerProfileModeFName},
    {kDevUsage, kDevUsageFName},
    {kDevPowerODVoltage, kDevPowerODVoltageFName},
    {kDevVBiosVer, kDevVBiosVerFName},
    {kDevPCIEThruPut, kDevPCIEThruPutFName},
    {kDevErrCntSDMA, kDevErrCntSDMAFName},
    {kDevErrCntUMC, kDevErrCntUMCFName},
    {kDevErrCntGFX, kDevErrCntGFXFName},
    {kDevErrCntFeatures, kDevErrCntFeaturesFName},
    {kDevMemTotGTT, kDevMemTotGTTFName},
    {kDevMemTotVisVRAM, kDevMemTotVisVRAMFName},
    {kDevMemBusyPercent, kDevMemBusyPercentFName},
    {kDevMemTotVRAM, kDevMemTotVRAMFName},
    {kDevMemUsedGTT, kDevMemUsedGTTFName},
    {kDevMemUsedVisVRAM, kDevMemUsedVisVRAMFName},
    {kDevMemUsedVRAM, kDevMemUsedVRAMFName},
    {kDevPCIEReplayCount, kDevPCIEReplayCountFName},
    {kDevUniqueId, kDevUniqueIdFName},
    {kDevDFCountersAvailable, kDevDFCountersAvailableFName},
    {kDevXGMIError, kDevXGMIErrorFName},
    {kDevFwVersionAsd, kDevFwVersionAsdFName},
    {kDevFwVersionCe, kDevFwVersionCeFName},
    {kDevFwVersionDmcu, kDevFwVersionDmcuFName},
    {kDevFwVersionMc, kDevFwVersionMcFName},
    {kDevFwVersionMe, kDevFwVersionMeFName},
    {kDevFwVersionMec, kDevFwVersionMecFName},
    {kDevFwVersionMec2, kDevFwVersionMec2FName},
    {kDevFwVersionPfp, kDevFwVersionPfpFName},
    {kDevFwVersionRlc, kDevFwVersionRlcFName},
    {kDevFwVersionRlcSrlc, kDevFwVersionRlcSrlcFName},
    {kDevFwVersionRlcSrlg, kDevFwVersionRlcSrlgFName},
    {kDevFwVersionRlcSrls, kDevFwVersionRlcSrlsFName},
    {kDevFwVersionSdma, kDevFwVersionSdmaFName},
    {kDevFwVersionSdma2, kDevFwVersionSdma2FName},
    {kDevFwVersionSmc, kDevFwVersionSmcFName},
    {kDevFwVersionSos, kDevFwVersionSosFName},
    {kDevFwVersionTaRas, kDevFwVersionTaRasFName},
    {kDevFwVersionTaXgmi, kDevFwVersionTaXgmiFName},
    {kDevFwVersionUvd, kDevFwVersionUvdFName},
    {kDevFwVersionVce, kDevFwVersionVceFName},
    {kDevFwVersionVcn, kDevFwVersionVcnFName},
    {kDevSerialNumber, kDevSerialNumberFName},
    {kDevMemPageBad, kDevMemPageBadFName},
};

static const std::map<rsmi_dev_perf_level, const char *> kDevPerfLvlMap = {
    {RSMI_DEV_PERF_LEVEL_AUTO, kDevPerfLevelAutoStr},
    {RSMI_DEV_PERF_LEVEL_LOW, kDevPerfLevelLowStr},
    {RSMI_DEV_PERF_LEVEL_HIGH, kDevPerfLevelHighStr},
    {RSMI_DEV_PERF_LEVEL_MANUAL, kDevPerfLevelManualStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_STD, kDevPerfLevelStandardStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK, kDevPerfLevelMinMClkStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK, kDevPerfLevelMinSClkStr},
    {RSMI_DEV_PERF_LEVEL_STABLE_PEAK, kDevPerfLevelPeakStr},

    {RSMI_DEV_PERF_LEVEL_UNKNOWN, kDevPerfLevelUnknownStr},
};

static  std::map<DevInfoTypes, uint8_t> kDevInfoVarTypeToRSMIVariant = {
    // rsmi_memory_type_t
    {kDevMemTotGTT, RSMI_MEM_TYPE_GTT},
    {kDevMemTotVisVRAM, RSMI_MEM_TYPE_VIS_VRAM},
    {kDevMemTotVRAM, RSMI_MEM_TYPE_VRAM},
    {kDevMemUsedGTT, RSMI_MEM_TYPE_GTT},
    {kDevMemUsedVisVRAM, RSMI_MEM_TYPE_VIS_VRAM},
    {kDevMemUsedVRAM, RSMI_MEM_TYPE_VRAM},

    // rsmi_clk_type_t
    {kDevGPUSClk, RSMI_CLK_TYPE_SYS},
    {kDevGPUMClk, RSMI_CLK_TYPE_MEM},
    {kDevFClk, RSMI_CLK_TYPE_DF},
    {kDevDCEFClk, RSMI_CLK_TYPE_DCEF},
    {kDevSOCClk, RSMI_CLK_TYPE_SOC},

    // rsmi_fw_block_t
    {kDevFwVersionAsd, RSMI_FW_BLOCK_ASD},
    {kDevFwVersionCe, RSMI_FW_BLOCK_CE},
    {kDevFwVersionDmcu, RSMI_FW_BLOCK_DMCU},
    {kDevFwVersionMc, RSMI_FW_BLOCK_MC},
    {kDevFwVersionMe, RSMI_FW_BLOCK_ME},
    {kDevFwVersionMec, RSMI_FW_BLOCK_MEC},
    {kDevFwVersionMec2, RSMI_FW_BLOCK_MEC2},
    {kDevFwVersionPfp, RSMI_FW_BLOCK_PFP},
    {kDevFwVersionRlc, RSMI_FW_BLOCK_RLC},
    {kDevFwVersionRlcSrlc, RSMI_FW_BLOCK_RLC_SRLC},
    {kDevFwVersionRlcSrlg, RSMI_FW_BLOCK_RLC_SRLG},
    {kDevFwVersionRlcSrls, RSMI_FW_BLOCK_RLC_SRLS},
    {kDevFwVersionSdma, RSMI_FW_BLOCK_SDMA},
    {kDevFwVersionSdma2, RSMI_FW_BLOCK_SDMA2},
    {kDevFwVersionSmc, RSMI_FW_BLOCK_SMC},
    {kDevFwVersionSos, RSMI_FW_BLOCK_SOS},
    {kDevFwVersionTaRas, RSMI_FW_BLOCK_TA_RAS},
    {kDevFwVersionTaXgmi, RSMI_FW_BLOCK_TA_XGMI},
    {kDevFwVersionUvd, RSMI_FW_BLOCK_UVD},
    {kDevFwVersionVce, RSMI_FW_BLOCK_VCE},
    {kDevFwVersionVcn, RSMI_FW_BLOCK_VCN},

    // rsmi_gpu_block_t
    {kDevErrCntUMC, RSMI_GPU_BLOCK_UMC},
    {kDevErrCntSDMA, RSMI_GPU_BLOCK_SDMA},
    {kDevErrCntGFX, RSMI_GPU_BLOCK_GFX},

    // rsmi_event_group_t
    {kDevDFCountersAvailable, RSMI_EVNT_GRP_XGMI}
};

static const std::map<const char *, dev_depends_t> kDevFuncDependsMap = {
    // Functions with only mandatory dependencies
  {"rsmi_dev_id_get",                    {{kDevDevIDFName}, {}}},
  {"rsmi_dev_vendor_id_get",             {{kDevVendorIDFName}, {}}},

  {"rsmi_dev_name_get",                  {{kDevVendorIDFName,
                                                        kDevDevIDFName}, {}}},
  {"rsmi_dev_brand_get",                 {{kDevVendorIDFName}, {}}},
  {"rsmi_dev_vendor_name_get",           {{kDevVendorIDFName}, {}}},
  {"rsmi_dev_serial_number_get",         {{kDevSerialNumberFName}, {}}},
  {"rsmi_dev_subsystem_id_get",          {{kDevSubSysDevIDFName}, {}}},
  {"rsmi_dev_subsystem_name_get",        {{kDevSubSysVendorIDFName,
                                           kDevVendorIDFName,
                                           kDevDevIDFName}, {}}},
  {"rsmi_dev_drm_render_minor_get",      {{}, {}}},
  {"rsmi_dev_subsystem_vendor_id_get",   {{kDevSubSysVendorIDFName}, {}}},
  {"rsmi_dev_unique_id_get",             {{kDevUniqueIdFName}, {}}},
  {"rsmi_dev_pci_bandwidth_get",         {{kDevPCIEClkFName}, {}}},
  {"rsmi_dev_pci_id_get",                {{}, {}}},
  {"rsmi_dev_pci_throughput_get",        {{kDevPCIEThruPutFName}, {}}},
  {"rsmi_dev_pci_replay_counter_get",    {{kDevPCIEReplayCountFName}, {}}},
  {"rsmi_dev_pci_bandwidth_set",         {{kDevPerfLevelFName,
                                           kDevPCIEClkFName}, {}}},
  {"rsmi_dev_power_profile_set",         {{kDevPerfLevelFName,
                                           kDevPowerProfileModeFName}, {}}},
  {"rsmi_dev_memory_busy_percent_get",   {{kDevMemBusyPercentFName}, {}}},
  {"rsmi_dev_busy_percent_get",          {{kDevUsageFName}, {}}},
  {"rsmi_dev_memory_reserved_pages_get", {{kDevMemPageBadFName}, {}}},
  {"rsmi_dev_overdrive_level_get",       {{kDevOverDriveLevelFName}, {}}},
  {"rsmi_dev_power_profile_presets_get", {{kDevPowerProfileModeFName}, {}}},
  {"rsmi_dev_perf_level_set",            {{kDevPerfLevelFName}, {}}},
  {"rsmi_dev_perf_level_get",            {{kDevPerfLevelFName}, {}}},
  {"rsmi_dev_overdrive_level_set",       {{kDevOverDriveLevelFName}, {}}},
  {"rsmi_dev_vbios_version_get",         {{kDevVBiosVerFName}, {}}},
  {"rsmi_dev_od_volt_info_get",          {{kDevPowerODVoltageFName}, {}}},
  {"rsmi_dev_od_volt_curve_regions_get", {{kDevPowerODVoltageFName}, {}}},
  {"rsmi_dev_ecc_enabled_get",           {{kDevErrCntFeaturesFName}, {}}},
  {"rsmi_dev_ecc_status_get",            {{kDevErrCntFeaturesFName}, {}}},
  {"rsmi_dev_counter_group_supported",   {{}, {}}},
  {"rsmi_dev_counter_create",            {{}, {}}},
  {"rsmi_dev_xgmi_error_status",         {{kDevXGMIErrorFName}, {}}},
  {"rsmi_dev_xgmi_error_reset",          {{kDevXGMIErrorFName}, {}}},
  {"rsmi_dev_memory_reserved_pages_get", {{kDevMemPageBadFName}, {}}},

  // These functions with variants, but no sensors/units. (May or may not
  // have mandatory dependencies.)
  {"rsmi_dev_memory_total_get",     { .mandatory_depends = {},
                                      .variants = {
                                        kDevMemTotGTT, kDevMemTotVisVRAM,
                                        kDevMemTotVRAM,
                                       }
                                    }
  },
  {"rsmi_dev_memory_usage_get",     { .mandatory_depends = {},
                                      .variants = {
                                        kDevMemUsedGTT,
                                        kDevMemUsedVisVRAM,
                                        kDevMemUsedVRAM,
                                      }
                                    }
  },
  {"rsmi_dev_gpu_clk_freq_get",     { .mandatory_depends = {},
                                      .variants = {
                                        kDevGPUSClk,
                                        kDevGPUMClk,
                                        kDevFClk,
                                        kDevDCEFClk,
                                        kDevSOCClk,
                                       }
                                    }
  },
  {"rsmi_dev_gpu_clk_freq_set",     { .mandatory_depends =
                                        {kDevPerfLevelFName},
                                      .variants = {
                                        kDevGPUSClk,
                                        kDevGPUMClk,
                                        kDevFClk,
                                        kDevDCEFClk,
                                        kDevSOCClk,
                                      }
                                    }
  },
  {"rsmi_dev_firmware_version_get", { .mandatory_depends = {},
                                      .variants = {
                                        kDevFwVersionAsd,
                                        kDevFwVersionCe,
                                        kDevFwVersionDmcu,
                                        kDevFwVersionMc,
                                        kDevFwVersionMe,
                                        kDevFwVersionMec,
                                        kDevFwVersionMec2,
                                        kDevFwVersionPfp,
                                        kDevFwVersionRlc,
                                        kDevFwVersionRlcSrlc,
                                        kDevFwVersionRlcSrlg,
                                        kDevFwVersionRlcSrls,
                                        kDevFwVersionSdma,
                                        kDevFwVersionSdma2,
                                        kDevFwVersionSmc,
                                        kDevFwVersionSos,
                                        kDevFwVersionTaRas,
                                        kDevFwVersionTaXgmi,
                                        kDevFwVersionUvd,
                                        kDevFwVersionVce,
                                        kDevFwVersionVcn,
                                      }
                                    }
  },
  {"rsmi_dev_ecc_count_get",        { .mandatory_depends = {},
                                      .variants = {
                                        kDevErrCntUMC,
                                        kDevErrCntSDMA,
                                        kDevErrCntGFX,
                                      }
                                    }
  },
  {"rsmi_counter_available_counters_get", { .mandatory_depends = {},
                                            .variants = {
                                              kDevDFCountersAvailable,
                                            }
                                          }
  },
};

#define RET_IF_NONZERO(X) { \
  if (X) return X; \
}

Device::Device(std::string p, RocmSMI_env_vars const *e) : path_(p), env_(e) {
  monitor_ = nullptr;

  // Get the device name
  size_t i = path_.rfind('/', path_.length());
  std::string dev = path_.substr(i + 1, path_.length() - i);

  std::string m_name("/rocm_smi_");
  m_name += dev;
  m_name += '_';
  m_name += std::to_string(geteuid());

  mutex_ = shared_mutex_init(m_name.c_str(), 0777);

  if (mutex_.ptr == nullptr) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                                       "Failed to create shared mem. mutex.");
  }
}

Device:: ~Device() {
  shared_mutex_close(mutex_);
}

template <typename T>
int Device::openSysfsFileStream(DevInfoTypes type, T *fs, const char *str) {
  auto sysfs_path = path_;

  if (env_->path_DRM_root_override && type == env_->enum_override) {
    sysfs_path = env_->path_DRM_root_override;

    if (str) {
      sysfs_path += ".write";
    }
  }

  sysfs_path += "/device/";
  sysfs_path += kDevAttribNameMap.at(type);

  DBG_FILE_ERROR(sysfs_path, str);
  bool reg_file;

  int ret = isRegularFile(sysfs_path, &reg_file);

  if (ret != 0) {
    return ret;
  }
  if (!reg_file) {
    return ENOENT;
  }

  fs->open(sysfs_path);

  if (!fs->is_open()) {
      return errno;
  }

  return 0;
}

int Device::readDevInfoStr(DevInfoTypes type, std::string *retStr) {
  std::ifstream fs;
  int ret = 0;

  assert(retStr != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    return ret;
  }

  fs >> *retStr;
  fs.close();

  return 0;
}

int Device::writeDevInfoStr(DevInfoTypes type, std::string valStr) {
  auto tempPath = path_;
  std::ofstream fs;
  int ret;

  ret = openSysfsFileStream(type, &fs, valStr.c_str());
  if (ret != 0) {
    return ret;
  }

  try {
    fs << valStr;
  } catch (...) {
    std::cout << "Write to file threw exception" << std::endl;
  }
  fs.close();

  return 0;
}

rsmi_dev_perf_level Device::perfLvlStrToEnum(std::string s) {
  rsmi_dev_perf_level pl;

  for (pl = RSMI_DEV_PERF_LEVEL_FIRST; pl <= RSMI_DEV_PERF_LEVEL_LAST; ) {
    if (s == kDevPerfLvlMap.at(pl)) {
      return pl;
    }
    pl = static_cast<rsmi_dev_perf_level>(static_cast<uint32_t>(pl) + 1);
  }
  return RSMI_DEV_PERF_LEVEL_UNKNOWN;
}

int Device::writeDevInfo(DevInfoTypes type, uint64_t val) {
  switch (type) {
    // The caller is responsible for making sure "val" is within a valid range
    case kDevOverDriveLevel:  // integer between 0 and 20
    case kDevPowerProfileMode:
      return writeDevInfoStr(type, std::to_string(val));
      break;

    case kDevPerfLevel:  // string: "auto", "low", "high", "manual", ...
      return writeDevInfoStr(type,
                                 kDevPerfLvlMap.at((rsmi_dev_perf_level)val));
      break;

    default:
      break;
  }

  return -1;
}

int Device::writeDevInfo(DevInfoTypes type, std::string val) {
  switch (type) {
    case kDevGPUMClk:
    case kDevDCEFClk:
    case kDevFClk:
    case kDevGPUSClk:
    case kDevPCIEClk:
    case kDevPowerODVoltage:
    case kDevSOCClk:
      return writeDevInfoStr(type, val);

    default:
      break;
  }

  return -1;
}

int Device::readDevInfoLine(DevInfoTypes type, std::string *line) {
  int ret;
  std::ifstream fs;

  assert(line != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    return ret;
  }

  std::getline(fs, *line);

  return 0;
}

int Device::readDevInfoMultiLineStr(DevInfoTypes type,
                                           std::vector<std::string> *retVec) {
  std::string line;
  int ret;
  std::ifstream fs;

  assert(retVec != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    return ret;
  }

  while (std::getline(fs, line)) {
    retVec->push_back(line);
  }

  if (retVec->size() == 0) {
    return 0;
  }
  // Remove any *trailing* empty (whitespace) lines
  while (retVec->back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
    retVec->pop_back();
  }
  return 0;
}

int Device::readDevInfo(DevInfoTypes type, uint64_t *val) {
  assert(val != nullptr);

  std::string tempStr;
  int ret;
  switch (type) {
    case kDevDevID:
    case kDevSubSysDevID:
    case kDevSubSysVendorID:
    case kDevVendorID:
    case kDevErrCntFeatures:
      ret = readDevInfoStr(type, &tempStr);
      RET_IF_NONZERO(ret);
      *val = std::stoi(tempStr, 0, 16);
      break;

    case kDevUsage:
    case kDevOverDriveLevel:
    case kDevMemTotGTT:
    case kDevMemTotVisVRAM:
    case kDevMemTotVRAM:
    case kDevMemUsedGTT:
    case kDevMemUsedVisVRAM:
    case kDevMemUsedVRAM:
    case kDevPCIEReplayCount:
    case kDevDFCountersAvailable:
    case kDevMemBusyPercent:
    case kDevXGMIError:
      ret = readDevInfoStr(type, &tempStr);
      RET_IF_NONZERO(ret);
      *val = std::stoul(tempStr, 0);
      break;

    case kDevUniqueId:
    case kDevFwVersionAsd:
    case kDevFwVersionCe:
    case kDevFwVersionDmcu:
    case kDevFwVersionMc:
    case kDevFwVersionMe:
    case kDevFwVersionMec:
    case kDevFwVersionMec2:
    case kDevFwVersionPfp:
    case kDevFwVersionRlc:
    case kDevFwVersionRlcSrlc:
    case kDevFwVersionRlcSrlg:
    case kDevFwVersionRlcSrls:
    case kDevFwVersionSdma:
    case kDevFwVersionSdma2:
    case kDevFwVersionSmc:
    case kDevFwVersionSos:
    case kDevFwVersionTaRas:
    case kDevFwVersionTaXgmi:
    case kDevFwVersionUvd:
    case kDevFwVersionVce:
    case kDevFwVersionVcn:
      ret = readDevInfoStr(type, &tempStr);
      RET_IF_NONZERO(ret);
      *val = std::stoul(tempStr, 0, 16);
      break;

    default:
      return -1;
  }
  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::vector<std::string> *val) {
  assert(val != nullptr);

  switch (type) {
    case kDevGPUMClk:
    case kDevGPUSClk:
    case kDevDCEFClk:
    case kDevFClk:
    case kDevPCIEClk:
    case kDevSOCClk:
    case kDevPowerProfileMode:
    case kDevPowerODVoltage:
    case kDevErrCntSDMA:
    case kDevErrCntUMC:
    case kDevErrCntGFX:
    case kDevMemPageBad:
      return readDevInfoMultiLineStr(type, val);
      break;

    default:
      return -1;
  }

  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::string *val) {
  assert(val != nullptr);

  switch (type) {
    case kDevPerfLevel:
    case kDevUsage:
    case kDevOverDriveLevel:
    case kDevDevID:
    case kDevSubSysDevID:
    case kDevSubSysVendorID:
    case kDevVendorID:
    case kDevVBiosVer:
    case kDevPCIEThruPut:
    case kDevSerialNumber:
      return readDevInfoStr(type, val);
      break;

    default:
      return -1;
  }
  return 0;
}

int Device::populateKFDNodeProperties(bool force_update) {
  int ret;

  std::vector<std::string> propVec;

  if (kfdNodePropMap_.size() > 0 && !force_update) {
    return 0;
  }

  ret = ReadKFDDeviceProperties(index_, &propVec);

  if (ret) {
    return ret;
  }

  std::string key_str;
  // std::string val_str;
  uint64_t val_int;  // Assume all properties are unsigned integers for now
  std::istringstream fs;

  for (uint32_t i = 0; i < propVec.size(); ++i) {
    fs.str(propVec[i]);
    fs >> key_str;
    fs >> val_int;

    kfdNodePropMap_[key_str] = val_int;

    fs.str("");
    fs.clear();
  }

  return 0;
}

int Device::getKFDNodeProperty(DevKFDNodePropTypes prop, uint64_t *val) {
  assert(val != nullptr);
  assert(kDevKFDPropNameMap.find(prop) != kDevKFDPropNameMap.end());

  const char *prop_name = kDevKFDPropNameMap.at(prop);
  if (kfdNodePropMap_.find(prop_name) == kfdNodePropMap_.end()) {
    return EINVAL;
  }

  *val = kfdNodePropMap_.at(prop_name);
  return 0;
}

void Device::DumpSupportedFunctions(void) {
  SupportedFuncMapIt func_iter = supported_funcs_.begin();

  std::cout << "*** Supported Functions ***" << std::endl;

  while (func_iter != supported_funcs_.end()) {
    std::cout << func_iter->first << std::endl;

    std::cout << "\tSupported Variants(Monitors): ";
    if (func_iter->second) {
      VariantMapIt var_iter = func_iter->second->begin();

      // We should have at least 1 supported variant or the function should
      // not be listed as supported.
      assert(var_iter != func_iter->second->end());

      while (var_iter != func_iter->second->end()) {
        std::cout << static_cast<uint32_t>(var_iter->first);

        if (var_iter->second) {
          std::cout << "(";

          SubVariantIt mon_iter = var_iter->second->begin();

          // We should have at least 1 supported monitor or the function should
          // not be listed as supported.
          assert(mon_iter != var_iter->second->end());
          while (mon_iter != var_iter->second->end()) {
            std::cout << static_cast<uint32_t>(*mon_iter) << ", ";
            mon_iter++;
          }
          std::cout << ")";
        }
        std::cout << ", ";
        var_iter++;
      }
      std::cout << std::endl;
    } else {
      std::cout << "Not Applicable" << std::endl;
    }
    func_iter++;
  }
}

void Device::fillSupportedFuncs(void) {
  if (supported_funcs_.size() != 0) {
    return;
  }

  std::map<const char *, dev_depends_t>::const_iterator it =
                                                   kDevFuncDependsMap.begin();
  std::string dev_rt = path_ + "/device";
  bool mand_depends_met;
  std::shared_ptr<VariantMap> supported_variants;

  while (it != kDevFuncDependsMap.end()) {
    // First, see if all the mandatory dependencies are there
    std::vector<const char *>::const_iterator dep =
                                         it->second.mandatory_depends.begin();

    mand_depends_met = true;
    for (; dep != it->second.mandatory_depends.end(); dep++) {
      std::string dep_path = dev_rt + "/" + *dep;
      if (!FileExists(dep_path.c_str())) {
        mand_depends_met = false;
        break;
      }
    }

    if (!mand_depends_met) {
      it++;
      continue;
    }
    // Then, see if the variants are supported.
    std::vector<DevInfoTypes>::const_iterator var =
                                                  it->second.variants.begin();

    if (it->second.variants.size() == 0) {
      supported_funcs_[it->first] = nullptr;
      it++;
      continue;
    }
    supported_variants = std::make_shared<VariantMap>();

    for (; var != it->second.variants.end(); var++) {
      std::string variant_path = dev_rt + "/" + kDevAttribNameMap.at(*var);
      if (!FileExists(variant_path.c_str())) {
        continue;
      }
      // At this point we assume no monitors, so map to nullptr
      (*supported_variants)[kDevInfoVarTypeToRSMIVariant.at(*var)] = nullptr;
    }

    if ((*supported_variants).size() > 0) {
      supported_funcs_[it->first] = supported_variants;
    }

    it++;
  }
  monitor()->fillSupportedFuncs(&supported_funcs_);
  // DumpSupportedFunctions();
}

#undef RET_IF_NONZERO
}  // namespace smi
}  // namespace amd
