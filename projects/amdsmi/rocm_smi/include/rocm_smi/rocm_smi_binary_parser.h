/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef ROCM_SMI_ROCM_SMI_BINARY_PARSER_H_
#define ROCM_SMI_ROCM_SMI_BINARY_PARSER_H_

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi.h"

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

/**
 *  A binary parser to read the pm table and register table
 *
 */
namespace amd::smi {

// definition for register table
enum amdgpu_sysfs_reg_offset {
    AMDGPU_SYS_REG_STATE_XGMI   = 0x0000,
    AMDGPU_SYS_REG_STATE_WAFL   = 0x1000,
    AMDGPU_SYS_REG_STATE_PCIE   = 0x2000,
    AMDGPU_SYS_REG_STATE_USR    = 0x3000,
    AMDGPU_SYS_REG_STATE_USR_1  = 0x4000,
    AMDGPU_SYS_REG_STATE_END    = 0x5000,
};

#define FIELD_FLAG_NUM_INSTANCE   0x01
#define FIELD_FLAG_NUM_SMN        0x02
#define FIELD_FLAG_INSTANCE_START 0x04
#define FIELD_FLAG_SMN_START      0x08

#define FIELD_TYPE_U8           0x01
#define FIELD_TYPE_U16          0x02
#define FIELD_TYPE_U32          0x04
#define FIELD_TYPE_U64          0x08

struct metric_field {
    uint8_t field_type;
    int field_arr_size;
    const char *field_name;
    uint8_t field_flag;
};

struct metric_field xgmi_regs[] = {
    { FIELD_TYPE_U16, 1, "structure_size", 0 },
    { FIELD_TYPE_U8,  1, "format_revision", 0 },
    { FIELD_TYPE_U8,  1, "content_revision", 0 },
    { FIELD_TYPE_U8,  1, "state_type", 0 },
    { FIELD_TYPE_U8,  1, "num_instances", FIELD_FLAG_NUM_INSTANCE },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U16, 1, "instance", FIELD_FLAG_INSTANCE_START },
    { FIELD_TYPE_U16, 1, "state", 0 },
    { FIELD_TYPE_U16, 1, "num_smn_regs", FIELD_FLAG_NUM_SMN },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U64, 1, "addr", FIELD_FLAG_SMN_START },
    { FIELD_TYPE_U32, 1, "value", 0 },
    { FIELD_TYPE_U32, 1, "pad", 0 },
    { 0, 0, NULL, 0 },
};

// check me!
struct metric_field wafl_regs[] = {
    { FIELD_TYPE_U16, 1, "structure_size", 0 },
    { FIELD_TYPE_U8,  1, "format_revision", 0 },
    { FIELD_TYPE_U8,  1, "content_revision", 0 },
    { FIELD_TYPE_U8,  1, "state_type", 0 },
    { FIELD_TYPE_U8,  1, "num_instances", FIELD_FLAG_NUM_INSTANCE },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U16, 1, "instance", FIELD_FLAG_INSTANCE_START },
    { FIELD_TYPE_U16, 1, "state", 0 },
    { FIELD_TYPE_U16, 1, "num_smn_regs", FIELD_FLAG_NUM_SMN },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U64, 1, "addr", FIELD_FLAG_SMN_START },
    { FIELD_TYPE_U32, 1, "value", 0 },
    { FIELD_TYPE_U32, 1, "pad", 0 },
    { 0, 0, NULL, 0 },
};

struct metric_field pcie_regs[] = {
    { FIELD_TYPE_U16, 1, "structure_size", 0 },
    { FIELD_TYPE_U8,  1, "format_revision", 0 },
    { FIELD_TYPE_U8,  1, "content_revision", 0 },
    { FIELD_TYPE_U8,  1, "state_type", 0 },
    { FIELD_TYPE_U8,  1, "num_instances", FIELD_FLAG_NUM_INSTANCE },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U16, 1, "instance", FIELD_FLAG_INSTANCE_START },
    { FIELD_TYPE_U16, 1, "state", 0 },
    { FIELD_TYPE_U16, 1, "num_smn_regs", FIELD_FLAG_NUM_SMN },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U16, 1, "device_status", 0 },
    { FIELD_TYPE_U16, 1, "link_status", 0 },
    { FIELD_TYPE_U32, 1, "sub_bus_number_latency", 0 },
    { FIELD_TYPE_U32, 1, "pcie_corr_err_status", 0 },
    { FIELD_TYPE_U32, 1, "pcie_uncorr_err_status", 0 },

    { FIELD_TYPE_U64, 1, "addr", FIELD_FLAG_SMN_START },
    { FIELD_TYPE_U32, 1, "value", 0 },
    { FIELD_TYPE_U32, 1, "pad", 0 },
    { 0, 0, NULL, 0 },
};

struct metric_field usr_regs[] = {
    { FIELD_TYPE_U16, 1, "structure_size", 0 },
    { FIELD_TYPE_U8,  1, "format_revision", 0 },
    { FIELD_TYPE_U8,  1, "content_revision", 0 },
    { FIELD_TYPE_U8,  1, "state_type", 0 },
    { FIELD_TYPE_U8,  1, "num_instances", FIELD_FLAG_NUM_INSTANCE },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U16, 1, "instance", FIELD_FLAG_INSTANCE_START },
    { FIELD_TYPE_U16, 1, "state", 0 },
    { FIELD_TYPE_U16, 1, "num_smn_regs", FIELD_FLAG_NUM_SMN },
    { FIELD_TYPE_U16, 1, "pad", 0 },

    { FIELD_TYPE_U64, 1, "addr", FIELD_FLAG_SMN_START },
    { FIELD_TYPE_U32, 1, "value", 0 },
    { FIELD_TYPE_U32, 1, "pad", 0 },
    { 0, 0, NULL, 0 },
};

// definition for pm metrics table
#define FIELD_FLAG_ACCUMULATOR  0x01

struct metric_field smu_13_0_6_v8[] = {
    { FIELD_TYPE_U16, 1, "structure_size", 0 },
    { FIELD_TYPE_U16, 1, "pad", 0 },
    { FIELD_TYPE_U32, 1, "mp1_ip_discovery_version", 0 },
    { FIELD_TYPE_U32, 1, "pmfw_version", 0 },
    { FIELD_TYPE_U32, 1, "pmmetrics_version", 0 },

    { FIELD_TYPE_U32, 1, "AccumulationCounter", 0 },

    { FIELD_TYPE_U32,  1, "MaxSocketTemperature", 0 },
    { FIELD_TYPE_U32,  1, "MaxVrTemperature", 0 },
    { FIELD_TYPE_U32,  1, "MaxHbmTemperature", 0 },
    { FIELD_TYPE_U64,  1, "MaxSocketTemperatureAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "MaxVrTemperatureAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "MaxHbmTemperatureAcc", FIELD_FLAG_ACCUMULATOR },

    { FIELD_TYPE_U32,  1, "SocketPowerLimit", 0 },
    { FIELD_TYPE_U32,  1, "MaxSocketPowerLimit", 0 },
    { FIELD_TYPE_U32,  1, "SocketPower", 0 },

    { FIELD_TYPE_U64,  1, "Timestamp", 0 },
    { FIELD_TYPE_U64,  1, "SocketEnergyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "CcdEnergyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "XcdEnergyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "AidEnergyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "HbmEnergyAcc", FIELD_FLAG_ACCUMULATOR },

    { FIELD_TYPE_U32,  1, "CclkFrequencyLimit", 0 },
    { FIELD_TYPE_U32,  1, "GfxclkFrequencyLimit", 0 },
    { FIELD_TYPE_U32,  1, "FclkFrequency", 0 },
    { FIELD_TYPE_U32,  1, "UclkFrequency", 0 },
    { FIELD_TYPE_U32,  4, "SocclkFrequency", 0 },
    { FIELD_TYPE_U32,  4, "VclkFrequency", 0 },
    { FIELD_TYPE_U32,  4, "DclkFrequency", 0 },
    { FIELD_TYPE_U32,  4, "LclkFrequency", 0 },
    { FIELD_TYPE_U64,  8, "GfxclkFrequencyAcc", FIELD_FLAG_ACCUMULATOR},
    { FIELD_TYPE_U64,  96, "CclkFrequencyAcc", FIELD_FLAG_ACCUMULATOR },

    { FIELD_TYPE_U32,  1, "MaxCclkFrequency", 0 },
    { FIELD_TYPE_U32,  1, "MinCclkFrequency", 0 },
    { FIELD_TYPE_U32,  1, "MaxGfxclkFrequency", 0 },
    { FIELD_TYPE_U32,  1, "MinGfxclkFrequency", 0 },
    { FIELD_TYPE_U32,  4, "FclkFrequencyTable", 0 },
    { FIELD_TYPE_U32,  4, "UclkFrequencyTable", 0 },
    { FIELD_TYPE_U32,  4, "SocclkFrequencyTable", 0 },
    { FIELD_TYPE_U32,  4, "VclkFrequencyTable", 0 },
    { FIELD_TYPE_U32,  4, "DclkFrequencyTable", 0 },
    { FIELD_TYPE_U32,  4, "LclkFrequencyTable", 0 },
    { FIELD_TYPE_U32,  1, "MaxLclkDpmRange", 0 },
    { FIELD_TYPE_U32,  1, "MinLclkDpmRange", 0 },

    { FIELD_TYPE_U32,  1, "XgmiWidth", 0 },
    { FIELD_TYPE_U32,  1, "XgmiBitrate", 0 },
    { FIELD_TYPE_U64,  8, "XgmiReadBandwidthAcc", 0 },
    { FIELD_TYPE_U64,  8, "XgmiWriteBandwidthAcc", 0 },

    { FIELD_TYPE_U32,  1, "SocketC0Residency", 0 },
    { FIELD_TYPE_U32,  1, "SocketGfxBusy", 0 },
    { FIELD_TYPE_U32,  1, "DramBandwidthUtilization", 0 },
    { FIELD_TYPE_U64,  1, "SocketC0ResidencyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "SocketGfxBusyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  1, "DramBandwidthAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U32,  1, "MaxDramBandwidth", 0 },
    { FIELD_TYPE_U64,  1, "DramBandwidthUtilizationAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  4, "PcieBandwidthAcc", FIELD_FLAG_ACCUMULATOR },

    { FIELD_TYPE_U32,  1, "ProchotResidencyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U32,  1, "PptResidencyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U32,  1, "SocketThmResidencyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U32,  1, "VrThmResidencyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U32,  1, "HbmThmResidencyAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U32,  1, "GfxLockXCDMak", 0 },

    { FIELD_TYPE_U32,  8, "GfxclkFrequency", 0 },

    { FIELD_TYPE_U64,  4, "PublicSerialNumber_AID", 0 },
    { FIELD_TYPE_U64,  8, "PublicSerialNumber_XCD", 0 },
    { FIELD_TYPE_U64,  12, "PublicSerialNumber_CCD", 0 },

    { FIELD_TYPE_U64,  8, "XgmiReadDataSizeAcc", FIELD_FLAG_ACCUMULATOR },
    { FIELD_TYPE_U64,  8, "XgmiWriteDataSizeAcc", FIELD_FLAG_ACCUMULATOR },
    { 0, 0, NULL, 0 },
};

}  // namespace amd::smi

#endif // ROCM_SMI_ROCM_SMI_BINARY_PARSER_H_

