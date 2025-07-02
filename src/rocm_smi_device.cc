/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2023, Advanced Micro Devices, Inc.
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
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "shared_mutex.h"  // NOLINT

namespace amd {
namespace smi {

// Debug root file path
static const char *kPathDebugRootFName = "/sys/kernel/debug/dri/";

// Device debugfs file names
static const char *kDevGpuResetFName = "amdgpu_gpu_recover";

// Device sysfs file names
static const char *kDevPerfLevelFName = "power_dpm_force_performance_level";
static const char *kDevDevProdNameFName = "product_name";
static const char *kDevDevProdNumFName = "product_number";
static const char *kDevDevIDFName = "device";
static const char* kDevXGMIPhysicalIDFName = "xgmi_physical_id";
static const char *kDevDevRevIDFName = "revision";
static const char *kDevVendorIDFName = "vendor";
static const char *kDevSubSysDevIDFName = "subsystem_device";
static const char *kDevSubSysVendorIDFName = "subsystem_vendor";
static const char *kDevOverDriveLevelFName = "pp_sclk_od";
static const char *kDevMemOverDriveLevelFName = "pp_mclk_od";
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
static const char *kDevErrCntMMHUBFName = "ras/mmhub_err_count";
static const char *kDevErrCntPCIEBIFFName = "ras/pcie_bif_err_count";
static const char *kDevErrCntHDPFName = "ras/hdp_err_count";
static const char *kDevErrCntXGMIWAFLFName = "ras/xgmi_wafl_err_count";
static const char *kDevErrCntFeaturesFName = "ras/features";
static const char *kDevMemPageBadFName = "ras/gpu_vram_bad_pages";
static const char *kDevMemTotGTTFName = "mem_info_gtt_total";
static const char *kDevMemTotVisVRAMFName = "mem_info_vis_vram_total";
static const char *kDevMemTotVRAMFName = "mem_info_vram_total";
static const char *kDevMemUsedGTTFName = "mem_info_gtt_used";
static const char *kDevMemUsedVisVRAMFName = "mem_info_vis_vram_used";
static const char *kDevMemUsedVRAMFName = "mem_info_vram_used";
static const char *kDevVramVendorFName = "mem_info_vram_vendor";
static const char *kDevPCIEReplayCountFName = "pcie_replay_count";
static const char *kDevUniqueIdFName = "unique_id";
static const char *kDevDFCountersAvailableFName = "df_cntr_avail";
static const char *kDevMemBusyPercentFName = "mem_busy_percent";
static const char *kDevXGMIErrorFName = "xgmi_error";
static const char *kDevSerialNumberFName = "serial_number";
static const char *kDevNumaNodeFName = "numa_node";
static const char *kDevGpuMetricsFName = "gpu_metrics";
static const char *kDevAvailableComputePartitionFName =
                  "available_compute_partition";
static const char *kDevComputePartitionFName = "current_compute_partition";
static const char *kDevMemoryPartitionFName = "current_memory_partition";

// Firmware version files
static const char *kDevFwVersionAsdFName = "fw_version/asd_fw_version";
static const char *kDevFwVersionCeFName = "fw_version/ce_fw_version";
static const char *kDevFwVersionDmcuFName = "fw_version/dmcu_fw_version";
static const char *kDevFwVersionMcFName = "fw_version/mc_fw_version";
static const char *kDevFwVersionMeFName = "fw_version/me_fw_version";
static const char *kDevFwVersionMecFName = "fw_version/mec_fw_version";
static const char *kDevFwVersionMec2FName = "fw_version/mec2_fw_version";
static const char *kDevFwVersionMesFName = "fw_version/mes_fw_version";
static const char *kDevFwVersionMesKiqFName = "fw_version/mes_kiq_fw_version";
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
static const char *kDevPerfLevelDeterminismStr = "perf_determinism";
static const char *kDevPerfLevelUnknownStr = "unknown";

static const std::map<DevInfoTypes, const char *> kDevAttribNameMap = {
    {kDevPerfLevel, kDevPerfLevelFName},
    {kDevOverDriveLevel, kDevOverDriveLevelFName},
    {kDevMemOverDriveLevel, kDevMemOverDriveLevelFName},
    {kDevDevProdName, kDevDevProdNameFName},
    {kDevDevProdNum, kDevDevProdNumFName},
    {kDevDevID, kDevDevIDFName},
    {kDevXGMIPhysicalID, kDevXGMIPhysicalIDFName},
    {kDevDevRevID, kDevDevRevIDFName},
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
    {kDevErrCntMMHUB, kDevErrCntMMHUBFName},
    {kDevErrCntPCIEBIF, kDevErrCntPCIEBIFFName},
    {kDevErrCntHDP, kDevErrCntHDPFName},
    {kDevErrCntXGMIWAFL, kDevErrCntXGMIWAFLFName},
    {kDevErrCntFeatures, kDevErrCntFeaturesFName},
    {kDevMemTotGTT, kDevMemTotGTTFName},
    {kDevMemTotVisVRAM, kDevMemTotVisVRAMFName},
    {kDevMemBusyPercent, kDevMemBusyPercentFName},
    {kDevMemTotVRAM, kDevMemTotVRAMFName},
    {kDevMemUsedGTT, kDevMemUsedGTTFName},
    {kDevMemUsedVisVRAM, kDevMemUsedVisVRAMFName},
    {kDevMemUsedVRAM, kDevMemUsedVRAMFName},
    {kDevVramVendor, kDevVramVendorFName},
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
    {kDevFwVersionMes, kDevFwVersionMesFName},
    {kDevFwVersionMesKiq, kDevFwVersionMesKiqFName},
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
    {kDevNumaNode, kDevNumaNodeFName},
    {kDevGpuMetrics, kDevGpuMetricsFName},
    {kDevGpuReset, kDevGpuResetFName},
    {kDevAvailableComputePartition, kDevAvailableComputePartitionFName},
    {kDevComputePartition, kDevComputePartitionFName},
    {kDevMemoryPartition, kDevMemoryPartitionFName},
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
    {RSMI_DEV_PERF_LEVEL_DETERMINISM, kDevPerfLevelDeterminismStr},

    {RSMI_DEV_PERF_LEVEL_UNKNOWN, kDevPerfLevelUnknownStr},
};

static const std::map<DevInfoTypes, uint8_t> kDevInfoVarTypeToRSMIVariant = {
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
    {kDevPCIEClk, RSMI_CLK_TYPE_PCIE},

    // rsmi_fw_block_t
    {kDevFwVersionAsd, RSMI_FW_BLOCK_ASD},
    {kDevFwVersionCe, RSMI_FW_BLOCK_CE},
    {kDevFwVersionDmcu, RSMI_FW_BLOCK_DMCU},
    {kDevFwVersionMc, RSMI_FW_BLOCK_MC},
    {kDevFwVersionMe, RSMI_FW_BLOCK_ME},
    {kDevFwVersionMec, RSMI_FW_BLOCK_MEC},
    {kDevFwVersionMec2, RSMI_FW_BLOCK_MEC2},
    {kDevFwVersionMes, RSMI_FW_BLOCK_MES},
    {kDevFwVersionMesKiq, RSMI_FW_BLOCK_MES_KIQ},
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
    {kDevErrCntMMHUB, RSMI_GPU_BLOCK_MMHUB},
    {kDevErrCntPCIEBIF, RSMI_GPU_BLOCK_PCIE_BIF},
    {kDevErrCntHDP, RSMI_GPU_BLOCK_HDP},
    {kDevErrCntXGMIWAFL, RSMI_GPU_BLOCK_XGMI_WAFL},

    // rsmi_event_group_t
    {kDevDFCountersAvailable, RSMI_EVNT_GRP_XGMI}
};

const std::map<DevInfoTypes, const char*>
Device::devInfoTypesStrings = {
  {kDevPerfLevel, "kDevPerfLevel"},
  {kDevOverDriveLevel, "kDevOverDriveLevel"},
  {kDevMemOverDriveLevel, "kDevMemOverDriveLevel"},
  {kDevDevID, "kDevDevID"},
  {kDevXGMIPhysicalID, "kDevXGMIPhysicalID"},
  {kDevDevRevID, "kDevDevRevID"},
  {kDevDevProdName, "kDevDevProdName"},
  {kDevDevProdNum, "kDevDevProdNum"},
  {kDevVendorID, "kDevVendorID"},
  {kDevSubSysDevID, "kDevSubSysDevID"},
  {kDevSubSysVendorID, "kDevSubSysVendorID"},
  {kDevGPUMClk, "kDevGPUMClk"},
  {kDevGPUSClk, "kDevGPUSClk"},
  {kDevDCEFClk, "kDevDCEFClk"},
  {kDevFClk, "kDevFClk"},
  {kDevSOCClk, "kDevSOCClk"},
  {kDevPCIEClk, "kDevPCIEClk"},
  {kDevPowerProfileMode, "kDevPowerProfileMode"},
  {kDevUsage, "kDevUsage"},
  {kDevPowerODVoltage, "kDevPowerODVoltage"},
  {kDevVBiosVer, "kDevVBiosVer"},
  {kDevPCIEThruPut, "kDevPCIEThruPut"},
  {kDevErrCntSDMA, "kDevErrCntSDMA"},
  {kDevErrCntUMC, "kDevErrCntUMC"},
  {kDevErrCntGFX, "kDevErrCntGFX"},
  {kDevErrCntMMHUB, "kDevErrCntMMHUB"},
  {kDevErrCntPCIEBIF, "kDevErrCntPCIEBIF"},
  {kDevErrCntHDP, "kDevErrCntHDP"},
  {kDevErrCntXGMIWAFL, "kDevErrCntXGMIWAFL"},
  {kDevErrCntFeatures, "kDevErrCntFeatures"},
  {kDevMemTotGTT, "kDevMemTotGTT"},
  {kDevMemTotVisVRAM, "kDevMemTotVisVRAM"},
  {kDevMemTotVRAM, "kDevMemTotVRAM"},
  {kDevMemUsedGTT, "kDevMemUsedGTT"},
  {kDevMemUsedVisVRAM, "kDevMemUsedVisVRAM"},
  {kDevMemUsedVRAM, "kDevMemUsedVRAM"},
  {kDevVramVendor, "kDevVramVendor"},
  {kDevPCIEReplayCount, "kDevPCIEReplayCount"},
  {kDevUniqueId, "kDevUniqueId"},
  {kDevDFCountersAvailable, "kDevDFCountersAvailable"},
  {kDevMemBusyPercent, "kDevMemBusyPercent"},
  {kDevXGMIError, "kDevXGMIError"},
  {kDevFwVersionAsd, "kDevFwVersionAsd"},
  {kDevFwVersionCe, "kDevFwVersionCe"},
  {kDevFwVersionDmcu, "kDevFwVersionDmcu"},
  {kDevFwVersionMc, "kDevFwVersionMc"},
  {kDevFwVersionMe, "kDevFwVersionMe"},
  {kDevFwVersionMec, "kDevFwVersionMec"},
  {kDevFwVersionMec2, "kDevFwVersionMec2"},
  {kDevFwVersionMes, "kDevFwVersionMes"},
  {kDevFwVersionMesKiq, "kDevFwVersionMesKiq"},
  {kDevFwVersionPfp, "kDevFwVersionPfp"},
  {kDevFwVersionRlc, "kDevFwVersionRlc"},
  {kDevFwVersionRlcSrlc, "kDevFwVersionRlcSrlc"},
  {kDevFwVersionRlcSrlg, "kDevFwVersionRlcSrlg"},
  {kDevFwVersionRlcSrls, "kDevFwVersionRlcSrls"},
  {kDevFwVersionSdma, "kDevFwVersionSdma"},
  {kDevFwVersionSdma2, "kDevFwVersionSdma2"},
  {kDevFwVersionSmc, "kDevFwVersionSmc"},
  {kDevFwVersionSos, "kDevFwVersionSos"},
  {kDevFwVersionTaRas, "kDevFwVersionTaRas"},
  {kDevFwVersionTaXgmi, "kDevFwVersionTaXgmi"},
  {kDevFwVersionUvd, "kDevFwVersionUvd"},
  {kDevFwVersionVce, "kDevFwVersionVce"},
  {kDevFwVersionVcn, "kDevFwVersionVcn"},
  {kDevSerialNumber, "kDevSerialNumber"},
  {kDevMemPageBad, "kDevMemPageBad"},
  {kDevNumaNode, "kDevNumaNode"},
  {kDevGpuMetrics, "kDevGpuMetrics"},
  {kDevGpuReset, "kDevGpuReset"},
  {kDevAvailableComputePartition, "kDevAvailableComputePartition"},
  {kDevComputePartition, "kDevComputePartition"},
  {kDevMemoryPartition, "kDevMemoryPartition"},
};

static const std::map<const char *, dev_depends_t> kDevFuncDependsMap = {
    // Functions with only mandatory dependencies
  {"rsmi_dev_vram_vendor_get",           {{kDevVramVendorFName}, {}}},
  {"rsmi_dev_id_get",                    {{kDevDevIDFName}, {}}},
  {"rsmi_dev_xgmi_physical_id_get",      {{kDevXGMIPhysicalIDFName}, {}}},
  {"rsmi_dev_revision_get",              {{kDevDevRevIDFName}, {}}},
  {"rsmi_dev_vendor_id_get",             {{kDevVendorIDFName}, {}}},
  {"rsmi_dev_name_get",                  {{kDevVendorIDFName,
                                           kDevDevIDFName}, {}}},
  {"rsmi_dev_sku_get",                   {{kDevDevProdNumFName}, {}}},
  {"rsmi_dev_brand_get",                 {{kDevVendorIDFName,
                                           kDevVBiosVerFName}, {}}},
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
  {"rsmi_dev_mem_overdrive_level_get",   {{kDevMemOverDriveLevelFName}, {}}},
  {"rsmi_dev_power_profile_presets_get", {{kDevPowerProfileModeFName}, {}}},
  {"rsmi_dev_perf_level_set",            {{kDevPerfLevelFName}, {}}},
  {"rsmi_dev_perf_level_set_v1",         {{kDevPerfLevelFName}, {}}},
  {"rsmi_dev_perf_level_get",            {{kDevPerfLevelFName}, {}}},
  {"rsmi_perf_determinism_mode_set",     {{kDevPerfLevelFName,
                                           kDevPowerODVoltageFName}, {}}},
  {"rsmi_dev_overdrive_level_set",       {{kDevOverDriveLevelFName}, {}}},
  {"rsmi_dev_vbios_version_get",         {{kDevVBiosVerFName}, {}}},
  {"rsmi_dev_od_volt_info_get",          {{kDevPowerODVoltageFName}, {}}},
  {"rsmi_dev_od_volt_info_set",          {{kDevPowerODVoltageFName,
                                           kDevPerfLevelFName},  {}}},
  {"rsmi_dev_od_volt_curve_regions_get", {{kDevPowerODVoltageFName}, {}}},
  {"rsmi_dev_ecc_enabled_get",           {{kDevErrCntFeaturesFName}, {}}},
  {"rsmi_dev_ecc_status_get",            {{kDevErrCntFeaturesFName}, {}}},
  {"rsmi_dev_counter_group_supported",   {{}, {}}},
  {"rsmi_dev_counter_create",            {{}, {}}},
  {"rsmi_dev_xgmi_error_status",         {{kDevXGMIErrorFName}, {}}},
  {"rsmi_dev_xgmi_error_reset",          {{kDevXGMIErrorFName}, {}}},
  {"rsmi_dev_memory_reserved_pages_get", {{kDevMemPageBadFName}, {}}},
  {"rsmi_topo_numa_affinity_get",        {{kDevNumaNodeFName}, {}}},
  {"rsmi_dev_gpu_metrics_info_get",      {{kDevGpuMetricsFName}, {}}},
  {"rsmi_dev_gpu_reset",                 {{kDevGpuResetFName}, {}}},
  {"rsmi_dev_energy_count_get",          {{kDevGpuMetricsFName}, {}}},
  {"rsmi_dev_current_socket_power_get",  {{kDevGpuMetricsFName}, {}}},

  {"rsmi_dev_compute_partition_get",     {{kDevComputePartitionFName}, {}}},
  {"rsmi_dev_compute_partition_set",     {{kDevComputePartitionFName}, {}}},
  {"rsmi_dev_memory_partition_get",      {{kDevMemoryPartitionFName}, {}}},
  {"rsmi_dev_memory_partition_set",      {{kDevMemoryPartitionFName}, {}}},

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
                                        kDevFwVersionMes,
                                        kDevFwVersionMesKiq,
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
                                        kDevErrCntMMHUB,
                                        kDevErrCntPCIEBIF,
                                        kDevErrCntHDP,
                                        kDevErrCntXGMIWAFL,
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

Device::Device(std::string p, RocmSMI_env_vars const *e) :
            monitor_(nullptr), path_(p), env_(e), evt_notif_anon_fd_(-1),
                                                   m_gpu_metrics_header{0, 0, 0} {
#ifndef DEBUG
    env_ = nullptr;
#endif

  // Get the device name
  size_t i = path_.rfind('/', path_.length());
  std::string dev = path_.substr(i + 1, path_.length() - i);

  std::string m_name("/rocm_smi_");
  m_name += dev;

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
int Device::openDebugFileStream(DevInfoTypes type, T *fs, const char *str) {
  std::string debugfs_path;

  debugfs_path = kPathDebugRootFName;
  debugfs_path += std::to_string(index());
  debugfs_path += "/";
  debugfs_path += kDevAttribNameMap.at(type);

  DBG_FILE_ERROR(debugfs_path, str);
  bool reg_file;
  int ret = isRegularFile(debugfs_path, &reg_file);

  if (ret != 0) {
    return ret;
  }
  if (!reg_file) {
    return ENOENT;
  }

  fs->open(debugfs_path);
  if (!fs->is_open()) {
      return errno;
  }
  return 0;
}

template <typename T>
int Device::openSysfsFileStream(DevInfoTypes type, T *fs, const char *str) {
  auto sysfs_path = path_;
  std::ostringstream ss;

#ifdef DEBUG
  if (env_->path_DRM_root_override
      && (env_->enum_overrides.find(type) != env_->enum_overrides.end())) {
    sysfs_path = env_->path_DRM_root_override;
  }
#endif

  sysfs_path += "/device/";
  sysfs_path += kDevAttribNameMap.at(type);

  DBG_FILE_ERROR(sysfs_path, str);
  bool reg_file;

  int ret = isRegularFile(sysfs_path, &reg_file);
  if (ret != 0) {
    ss << __PRETTY_FUNCTION__ << " | Issue: File did not exist - SYSFS file ("
       << sysfs_path
       << ") for DevInfoInfoType (" << devInfoTypesStrings.at(type)
       << "), returning " << std::to_string(ret);
    LOG_ERROR(ss);
    return ret;
  }
  if (!reg_file) {
    ss << __PRETTY_FUNCTION__
       << " | Issue: File is not a regular file - SYSFS file ("
       << sysfs_path << ") for "
       << "DevInfoInfoType (" << devInfoTypesStrings.at(type) << "),"
       << " returning ENOENT (" << std::strerror(ENOENT) << ")";
    LOG_ERROR(ss);
    return ENOENT;
  }

  fs->open(sysfs_path);

  if (!fs->is_open()) {
    ss << __PRETTY_FUNCTION__
       << " | Issue: Could not open - SYSFS file (" << sysfs_path << ") for "
       << "DevInfoInfoType (" << devInfoTypesStrings.at(type) << "), "
       << ", returning " << std::to_string(errno) << " ("
       << std::strerror(errno) << ")";
    LOG_ERROR(ss);
    return errno;
  }

  ss << __PRETTY_FUNCTION__ << " | Successfully opened SYSFS file ("
     << sysfs_path
     << ") for DevInfoInfoType (" << devInfoTypesStrings.at(type)
     << ")";
  LOG_INFO(ss);
  return 0;
}

int Device::readDebugInfoStr(DevInfoTypes type, std::string *retStr) {
  std::ifstream fs;
  std::string line;
  int ret = 0;
  std::ostringstream ss;

  assert(retStr != nullptr);

  ret = openDebugFileStream(type, &fs);
  if (ret != 0) {
    ss << "Could not read debugInfoStr for DevInfoType ("
     << devInfoTypesStrings.at(type)<< "), returning "
     << std::to_string(ret);
    LOG_ERROR(ss);
    return ret;
  }

  if (!(fs.peek() == std::ifstream::traits_type::eof())) {
    getline(fs, line);
    *retStr = line;
  }

  fs.close();

  ss << "Successfully read debugInfoStr for DevInfoType ("
     << devInfoTypesStrings.at(type)<< "), retString= " << *retStr;
  LOG_INFO(ss);

  return 0;
}

int Device::readDevInfoStr(DevInfoTypes type, std::string *retStr) {
  std::ifstream fs;
  int ret = 0;
  std::ostringstream ss;

  assert(retStr != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    ss << "Could not read device info string for DevInfoType ("
     << devInfoTypesStrings.at(type) << "), returning "
     << std::to_string(ret);
    LOG_ERROR(ss);
    return ret;
  }

  fs >> *retStr;
  fs.close();
  ss << __PRETTY_FUNCTION__
     << "Successfully read device info string for DevInfoType (" <<
            devInfoTypesStrings.at(type) << "): " + *retStr
     << " | "
     << (fs.is_open() ? " File stream is opened" : " File stream is closed")
     << " | " << (fs.bad() ? "[ERROR] Bad read operation" :
     "[GOOD] No bad bit read, successful read operation")
     << " | " << (fs.fail() ? "[ERROR] Failed read - format error" :
     "[GOOD] No fail - Successful read operation")
     << " | " << (fs.eof() ? "[ERROR] Failed read - EOF error" :
     "[GOOD] No eof - Successful read operation")
     << " | " << (fs.good() ? "[GOOD] read good - Successful read operation" :
     "[ERROR] Failed read - good error");
  LOG_INFO(ss);
  return 0;
}

int Device::writeDevInfoStr(DevInfoTypes type, std::string valStr,
                           bool returnWriteErr) {
  // returnWriteErr = false, backwards compatability (old calls)
  // returnWriteErr = true, improvement - allows us to detect errors
  // when writing to file
  // (such as EBUSY)
  auto sysfs_path = path_;
  sysfs_path += "/device/";
  sysfs_path += kDevAttribNameMap.at(type);
  std::ofstream fs;
  int ret;
  std::ostringstream ss;

  fs.flush();
  fs.rdbuf()->pubsetbuf(0, 0);
  ret = openSysfsFileStream(type, &fs, valStr.c_str());
  if (ret != 0) {
    fs.close();
    ss << __PRETTY_FUNCTION__ << " | Issue: Could not open fileStream; "
       << "Could not write device info string (" << valStr
       << ") for DevInfoType (" << devInfoTypesStrings.at(type)
       << "), returning " << std::to_string(ret);
    LOG_ERROR(ss);
    return ret;
  }

  // We'll catch any exceptions in rocm_smi.cc code.
  if (fs << valStr) {
    fs.flush();
    fs.close();
    ss << "Successfully wrote device info string (" << valStr
       << ") for DevInfoType (" << devInfoTypesStrings.at(type)
       << "), returning RSMI_STATUS_SUCCESS";
    LOG_INFO(ss);
    ret = RSMI_STATUS_SUCCESS;
  } else {
    if (returnWriteErr) {
      ret = errno;
    } else {
      ret = RSMI_STATUS_NOT_SUPPORTED;
    }
    fs.flush();
    fs.close();
    ss << __PRETTY_FUNCTION__ << " | Issue: Could not write to file; "
       << "Could not write device info string (" << valStr
       << ") for DevInfoType (" << devInfoTypesStrings.at(type)
       << "), returning " << getRSMIStatusString(ErrnoToRsmiStatus(ret));
    ss << " | "
       << (fs.is_open() ? "[ERROR] File stream open" :
          "[GOOD] File stream closed")
       << " | " << (fs.bad() ? "[ERROR] Bad write operation" :
                    "[GOOD] No bad bit write, successful write operation")
       << " | " << (fs.fail() ? "[ERROR] Failed write - format error" :
                    "[GOOD] No fail - Successful write operation")
       << " | " << (fs.eof() ? "[ERROR] Failed write - EOF error" :
                    "[GOOD] No eof - Successful write operation")
       << " | " << (fs.good() ?
                   "[GOOD] Write good - Successful write operation" :
                   "[ERROR] Failed write - good error");
    LOG_ERROR(ss);
  }

  return ret;
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
    case kDevPowerODVoltage:
    case kDevPowerProfileMode:
      return writeDevInfoStr(type, std::to_string(val));
      break;

    case kDevPerfLevel:  // string: "auto", "low", "high", "manual", ...
      return writeDevInfoStr(type,
                                 kDevPerfLvlMap.at((rsmi_dev_perf_level)val));
      break;

    default:
      return EINVAL;
  }

  return -1;
}

int Device::writeDevInfo(DevInfoTypes type, std::string val) {
  auto sysfs_path = path_;
  sysfs_path += "/device/";
  sysfs_path += kDevAttribNameMap.at(type);
  switch (type) {
    case kDevGPUMClk:
    case kDevDCEFClk:
    case kDevFClk:
    case kDevGPUSClk:
    case kDevPCIEClk:
    case kDevPowerODVoltage:
    case kDevSOCClk:
      return writeDevInfoStr(type, val);
    case kDevComputePartition:
    case kDevMemoryPartition:
      return writeDevInfoStr(type, val, true);

    default:
      return EINVAL;
  }

  return -1;
}

int Device::readDevInfoLine(DevInfoTypes type, std::string *line) {
  int ret;
  std::ifstream fs;
  std::ostringstream ss;

  assert(line != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    ss << "Could not read DevInfoLine for DevInfoType ("
       << devInfoTypesStrings.at(type) << ")";
    LOG_ERROR(ss);
    return ret;
  }

  std::getline(fs, *line);
  ss << "Successfully read DevInfoLine for DevInfoType ("
     << devInfoTypesStrings.at(type) << "), returning *line = "
     << *line;
  LOG_INFO(ss);

  return 0;
}

int Device::readDevInfoBinary(DevInfoTypes type, std::size_t b_size,
                                void *p_binary_data) {
  auto sysfs_path = path_;
  std::ostringstream ss;

  FILE *ptr;
  sysfs_path += "/device/";
  sysfs_path += kDevAttribNameMap.at(type);

  ptr = fopen(sysfs_path.c_str(), "rb");
  if (!ptr) {
    ss << "Could not read DevInfoBinary for DevInfoType ("
       << devInfoTypesStrings.at(type) << ")"
       << " - SYSFS (" << sysfs_path << ")"
       << ", returning " << std::to_string(errno) << " ("
       << std::strerror(errno) << ")";
    LOG_ERROR(ss);
    return errno;
  }

  size_t num = fread(p_binary_data, b_size, 1, ptr);
  fclose(ptr);
  if ((num*b_size) != b_size) {
    ss << "Could not read DevInfoBinary for DevInfoType ("
       << devInfoTypesStrings.at(type) << ") - SYSFS ("
       << sysfs_path << "), binary size error; "
       << "[buff: "
       << p_binary_data
       << " size: "
       << b_size
       << " read: "
       << num
       << "]"
       << ", returning ENOENT (" << std::strerror(ENOENT) << ")";
    LOG_ERROR(ss);
    return ENOENT;
  }
  ss << "Successfully read DevInfoBinary for DevInfoType ("
     << devInfoTypesStrings.at(type) << ") - SYSFS ("
     << sysfs_path << "), returning binaryData = " << p_binary_data
     << "; byte_size = " << std::dec << static_cast<int>(b_size);

  std::string metricDescription = "AMD SMI GPU METRICS (16-byte width), "
                                  + sysfs_path;
  logHexDump(metricDescription.c_str(), p_binary_data, b_size, 16);
  LOG_INFO(ss);
  return 0;
}

int Device::readDevInfoMultiLineStr(DevInfoTypes type,
                                           std::vector<std::string> *retVec) {
  std::string line;
  int ret;
  std::ifstream fs;
  std::string allLines;
  std::ostringstream ss;

  assert(retVec != nullptr);

  ret = openSysfsFileStream(type, &fs);
  if (ret != 0) {
    return ret;
  }

  while (std::getline(fs, line)) {
    retVec->push_back(line);
  }

  if (retVec->empty()) {
    ss << "Read devInfoMultiLineStr for DevInfoType ("
       << devInfoTypesStrings.at(type) << ")"
       << ", but contained no string lines";
    LOG_ERROR(ss);
    return ENXIO;
  }
  // Remove any *trailing* empty (whitespace) lines
  while (!retVec->empty() &&
        retVec->back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
    retVec->pop_back();
  }

  // allow logging output of multiline strings
  for (const auto& l: *retVec) {
    allLines += "\n" + l;
  }

  if (!allLines.empty()) {
    ss << "Successfully read devInfoMultiLineStr for DevInfoType ("
       << devInfoTypesStrings.at(type) << ") "
       << ", returning lines read = " << allLines;
    LOG_INFO(ss);
  } else {
    ss << "Read devInfoMultiLineStr for DevInfoType ("
       << devInfoTypesStrings.at(type) << ")"
       << ", but lines were empty";
    LOG_INFO(ss);
    return ENXIO;
  }
  return 0;
}

int Device::readDevInfo(DevInfoTypes type, uint64_t *val) {
  assert(val != nullptr);

  std::string tempStr;
  int ret;
  int tmp_val;
  std::ostringstream ss;

  switch (type) {
    case kDevDevID:
    case kDevDevRevID:
    case kDevSubSysDevID:
    case kDevSubSysVendorID:
    case kDevVendorID:
    case kDevErrCntFeatures:
    case kDevXGMIPhysicalID:
      ret = readDevInfoStr(type, &tempStr);
      RET_IF_NONZERO(ret);

      if (tempStr.empty()) {
        return EINVAL;
      }
      tmp_val = std::stoi(tempStr, nullptr, 16);
      if (tmp_val < 0) {
        return EINVAL;
      }
      *val = static_cast<uint64_t>(tmp_val);
      break;

    case kDevUsage:
    case kDevOverDriveLevel:
    case kDevMemOverDriveLevel:
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
      if (tempStr.empty()) {
        return EINVAL;
      }
      *val = std::stoul(tempStr, nullptr);
      break;

    case kDevUniqueId:
    case kDevFwVersionAsd:
    case kDevFwVersionCe:
    case kDevFwVersionDmcu:
    case kDevFwVersionMc:
    case kDevFwVersionMe:
    case kDevFwVersionMec:
    case kDevFwVersionMec2:
    case kDevFwVersionMes:
    case kDevFwVersionMesKiq:
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
      if (tempStr.empty()) {
        return EINVAL;
      }
      *val = std::stoul(tempStr, nullptr, 16);
      break;

    case kDevGpuReset:
      ret = readDebugInfoStr(type, &tempStr);
      RET_IF_NONZERO(ret);
      break;

    default:
      return EINVAL;
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
    case kDevErrCntMMHUB:
    case kDevErrCntPCIEBIF:
    case kDevErrCntHDP:
    case kDevErrCntXGMIWAFL:
    case kDevMemPageBad:
      return readDevInfoMultiLineStr(type, val);
      break;

    default:
      return EINVAL;
  }

  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::size_t b_size,
                                        void *p_binary_data) {
  assert(p_binary_data != nullptr);

  switch (type) {
     case kDevGpuMetrics:
      return readDevInfoBinary(type, b_size, p_binary_data);
      break;

    default:
      return EINVAL;
  }

  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::string *val) {
  assert(val != nullptr);

  switch (type) {
    case kDevPerfLevel:
    case kDevUsage:
    case kDevOverDriveLevel:
    case kDevMemOverDriveLevel:
    case kDevDevProdName:
    case kDevDevProdNum:
    case kDevDevID:
    case kDevDevRevID:
    case kDevSubSysDevID:
    case kDevSubSysVendorID:
    case kDevVendorID:
    case kDevVramVendor:
    case kDevVBiosVer:
    case kDevPCIEThruPut:
    case kDevSerialNumber:
    case kDevAvailableComputePartition:
    case kDevComputePartition:
    case kDevMemoryPartition:
    case kDevNumaNode:
    case kDevXGMIPhysicalID:
      return readDevInfoStr(type, val);
      break;

    default:
      return EINVAL;
  }
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
  if (!supported_funcs_.empty()) {
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
      std::string debugfs_path;
      debugfs_path = kPathDebugRootFName;
      debugfs_path += std::to_string(index());
      debugfs_path += "/";
      debugfs_path += *dep;
      if (!FileExists(dep_path.c_str()) && !FileExists(debugfs_path.c_str())) {
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

    if (it->second.variants.empty()) {
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

    if (!(*supported_variants).empty()) {
      supported_funcs_[it->first] = supported_variants;
    }

    it++;
  }
  if (monitor() != nullptr) {
    monitor()->fillSupportedFuncs(&supported_funcs_);
  }
  // DumpSupportedFunctions();
}

static bool subvariant_match(const std::shared_ptr<SubVariant> *sv,
                                                             uint64_t sub_v) {
  assert(sv != nullptr);

  SubVariantIt it = (*sv)->begin();
  for (; it != (*sv)->end(); it++) {
    if ((*it & MONITOR_IND_BIT_MASK) == sub_v) {
      return true;
    }
  }
  return false;
}

bool Device::DeviceAPISupported(std::string name, uint64_t variant,
                                                       uint64_t sub_variant) {
  SupportedFuncMapIt func_it;
  VariantMapIt var_it;

  fillSupportedFuncs();
  func_it = supported_funcs_.find(name);

  if (func_it == supported_funcs_.end()) {
    return false;
  }

  if (variant != RSMI_DEFAULT_VARIANT) {
    // if variant is != RSMI_DEFAULT_VARIANT, we should not have a nullptr
    assert(func_it->second != nullptr);
    var_it = func_it->second->find(variant);

    if (var_it == func_it->second->end()) {
      return false;
    }

    if (sub_variant == RSMI_DEFAULT_VARIANT) {
      return true;
    }
    // sub_variant != RSMI_DEFAULT_VARIANT
    // if variant is != RSMI_DEFAULT_VARIANT, we should not have a nullptr
    assert(var_it->second != nullptr);

    return subvariant_match(&(var_it->second), sub_variant);
  }
  // variant == RSMI_DEFAULT_VARIANT
  if (func_it->second != nullptr) {
    var_it = func_it->second->find(variant);
  }
  if (sub_variant == RSMI_DEFAULT_VARIANT) {
    return true;
  }
  // sub_variant != RSMI_DEFAULT_VARIANT
  if (func_it->second == nullptr) {
    return false;
  }
  return subvariant_match(&(var_it->second), sub_variant);
}

rsmi_status_t Device::restartAMDGpuDriver(void) {
  REQUIRE_ROOT_ACCESS
  bool restartSuccessful = true;
  bool success = false;
  std::string out;
  bool wasGdmServiceActive = false;

  // sudo systemctl is-active gdm
  // we do not care about the success of checking if gdm is active
  std::tie(success, out) = executeCommand("systemctl is-active gdm");
  (out == "active") ? (restartSuccessful &= success) :
                         (restartSuccessful = true);

  // if gdm is active -> sudo systemctl stop gdm
  // TODO: are are there other display manager's we need to take into account?
  // see https://en.wikipedia.org/wiki/GNOME_Display_Manager
  if (success && (out == "active")) {
    wasGdmServiceActive = true;
    std::tie(success, out) = executeCommand("systemctl stop gdm&", false);
    restartSuccessful &= success;
  }

  // sudo modprobe -r amdgpu
  // sudo modprobe amdgpu
  std::tie(success, out) =
    executeCommand("modprobe -r amdgpu && modprobe amdgpu&", false);
  restartSuccessful &= success;

  // if gdm was active -> sudo systemctl start gdm
  if (wasGdmServiceActive) {
    std::tie(success, out) = executeCommand("systemctl start gdm&", false);
    restartSuccessful &= success;
  }

  return (restartSuccessful ? RSMI_STATUS_SUCCESS :
          RSMI_STATUS_AMDGPU_RESTART_ERR);
}

template <typename T> rsmi_status_t storeParameter(uint32_t dv_ind);

// Stores parameters depending on which rsmi type is provided.
// Uses template specialization, to restrict types to identify
// calls needed to complete the function.
// typename - restricted to
// rsmi_compute_partition_type_t or rsmi_memory_partition_type_t
// dv_ind - device index
// tempFileName - base file name
template <>
rsmi_status_t storeParameter<rsmi_compute_partition_type_t>(uint32_t dv_ind) {
  rsmi_status_t returnStatus = RSMI_STATUS_SUCCESS;
  bool doesFileExist;
  std::tie(doesFileExist, std::ignore) = readTmpFile(dv_ind, "boot",
                                                     "compute_partition");
  // if temporary file exists -> we do not need to store anything new
  // if not, read & store the state value
  if (doesFileExist) {
    return returnStatus;
  }
  const uint32_t kLen = 128;
  char data[kLen];
  rsmi_status_t ret = rsmi_dev_compute_partition_get(dv_ind, data, kLen);
  rsmi_status_t storeRet;

  if (ret == RSMI_STATUS_SUCCESS) {
    storeRet = storeTmpFile(dv_ind, "compute_partition", "boot", data);
  } else if (ret == RSMI_STATUS_NOT_SUPPORTED) {
    // not supported is ok
    storeRet = storeTmpFile(dv_ind, "compute_partition", "boot", "UNKNOWN");
  } else {
    storeRet = storeTmpFile(dv_ind, "compute_partition", "boot", "UNKNOWN");
    returnStatus = ret;
  }

  if (storeRet != RSMI_STATUS_SUCCESS) {
    // file storage err takes precedence over other errors
    returnStatus = storeRet;
  }
  return returnStatus;
}

// Stores parameters depending on which rsmi type is provided.
// Uses template specialization, to restrict types to identify
// calls needed to complete the function.
// typename - restricted to
// rsmi_compute_partition_type_t or rsmi_memory_partition_type_t
// dv_ind - device index
// tempFileName - base file name
template <>
rsmi_status_t storeParameter<rsmi_memory_partition_type_t>(uint32_t dv_ind) {
  rsmi_status_t returnStatus = RSMI_STATUS_SUCCESS;
  uint32_t kDatalength = 128;
  char data[kDatalength];
  bool doesFileExist;
  std::tie(doesFileExist, std::ignore) = readTmpFile(dv_ind, "boot",
                                                     "memory_partition");
  // if temporary file exists -> we do not need to store anything new
  // if not, read & store the state value
  if (doesFileExist) {
    return returnStatus;
  }
  rsmi_status_t ret = rsmi_dev_memory_partition_get(dv_ind, data, kDatalength);
  rsmi_status_t storeRet;

  if (ret == RSMI_STATUS_SUCCESS) {
    storeRet = storeTmpFile(dv_ind, "memory_partition", "boot", data);
  } else if (ret == RSMI_STATUS_NOT_SUPPORTED) {
    // not supported is ok
    storeRet = storeTmpFile(dv_ind, "memory_partition", "boot", "UNKNOWN");
  } else {
    storeRet = storeTmpFile(dv_ind, "memory_partition", "boot", "UNKNOWN");
    returnStatus = ret;
  }

  if (storeRet != RSMI_STATUS_SUCCESS) {
    // file storage err takes precedence over other errors
    returnStatus = storeRet;
  }
  return returnStatus;
}

rsmi_status_t Device::storeDevicePartitions(uint32_t dv_ind) {
  rsmi_status_t returnStatus = RSMI_STATUS_SUCCESS;
  returnStatus = storeParameter<rsmi_compute_partition_type_t>(dv_ind);
  rsmi_status_t ret = storeParameter<rsmi_memory_partition_type_t>(dv_ind);
  if (returnStatus == RSMI_STATUS_SUCCESS) {  // only record earliest error
    returnStatus = ret;
  }
  return returnStatus;
}

// Reads a device's boot partition state, depending on which rsmi type is
// provided and device index.
// Uses template specialization, to restrict types to identify
// calls needed to complete the function.
// typename - restricted to rsmi_compute_partition_type_t
// or rsmi_compute_partition_type_t
// dv_ind - device index
template <>
std::string Device::readBootPartitionState<rsmi_compute_partition_type_t>(
    uint32_t dv_ind) {
  std::string boot_state;
  std::tie(std::ignore, boot_state) = readTmpFile(dv_ind, "boot",
                                                  "compute_partition");
  return boot_state;
}

// Reads a device's boot partition state, depending on which rsmi type is
// provided and device index.
// Uses template specialization, to restrict types to identify
// calls needed to complete the function.
// typename - restricted to rsmi_compute_partition_type_t
// or rsmi_compute_partition_type_t
// dv_ind - device index
template <>
std::string Device::readBootPartitionState<rsmi_memory_partition_type_t>(
    uint32_t dv_ind) {
  std::string boot_state;
  std::tie(std::ignore, boot_state) = readTmpFile(dv_ind, "boot",
                                                  "memory_partition");
  return boot_state;
}


#undef RET_IF_NONZERO
}  // namespace smi
}  // namespace amd
