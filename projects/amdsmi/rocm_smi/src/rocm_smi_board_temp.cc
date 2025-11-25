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

#include "rocm_smi/rocm_smi_board_temp.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi_logger.h"
#include <fstream>
#include <cstring>
#include <cerrno>
#include <iomanip>
#include <sstream>
#include <map>

using amd::smi::getRSMIStatusString;

namespace amd::smi {

// Static mapping tables for temperature type conversions
static const std::map<int, rsmi_temperature_type_t> vr_temp_map = {
    {AMDGPU_VDDCR_VDD0_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD0},
    {AMDGPU_VDDCR_VDD1_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD1},
    {AMDGPU_VDDCR_VDD2_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD2},
    {AMDGPU_VDDCR_VDD3_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD3},
    {AMDGPU_VDDCR_SOC_A_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOC_A},
    {AMDGPU_VDDCR_SOC_C_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOC_C},
    {AMDGPU_VDDCR_SOCIO_A_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOCIO_A},
    {AMDGPU_VDDCR_SOCIO_C_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOCIO_C},
    {AMDGPU_VDD_085_HBM_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDD_085_HBM},
    {AMDGPU_VDDCR_11_HBM_B_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_11_HBM_B},
    {AMDGPU_VDDCR_11_HBM_D_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_11_HBM_D},
    {AMDGPU_VDD_USR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDD_USR},
    {AMDGPU_VDDIO_11_E32_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDIO_11_E32}
};

static const std::map<int, rsmi_temperature_type_t> node_temp_map = {
    {AMDGPU_RETIMER_X_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_RETIMER_X},
    {AMDGPU_OAM_X_IBC_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC},
    {AMDGPU_OAM_X_IBC_2_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC_2},
    {AMDGPU_OAM_X_VDD18_VR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR},
    {AMDGPU_OAM_X_04_HBM_B_VR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR},
    {AMDGPU_OAM_X_04_HBM_D_VR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR}
};

static const std::map<int, rsmi_temperature_type_t> system_temp_map = {
    {AMDGPU_UBB_FPGA_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA},
    {AMDGPU_UBB_FRONT_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FRONT},
    {AMDGPU_UBB_BACK_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_BACK},
    {AMDGPU_UBB_OAM7_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM7},
    {AMDGPU_UBB_IBC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_IBC},
    {AMDGPU_UBB_UFPGA_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_UFPGA},
    {AMDGPU_UBB_OAM1_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM1},
    {AMDGPU_OAM_0_1_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_0_1_HSC},
    {AMDGPU_OAM_2_3_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_2_3_HSC},
    {AMDGPU_OAM_4_5_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_4_5_HSC},
    {AMDGPU_OAM_6_7_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_6_7_HSC},
    {AMDGPU_UBB_FPGA_0V72_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA_0V72_VR},
    {AMDGPU_UBB_FPGA_3V3_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA_3V3_VR},
    {AMDGPU_RETIMER_0_1_2_3_1V2_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR},
    {AMDGPU_RETIMER_4_5_6_7_1V2_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR},
    {AMDGPU_RETIMER_0_1_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR},
    {AMDGPU_RETIMER_4_5_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR},
    {AMDGPU_RETIMER_2_3_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR},
    {AMDGPU_RETIMER_6_7_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR},
    {AMDGPU_OAM_0_1_2_3_3V3_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR},
    {AMDGPU_OAM_4_5_6_7_3V3_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR},
    {AMDGPU_IBC_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_IBC_HSC},
    {AMDGPU_IBC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_IBC}
};

// Helper function to create hex dump string
static std::string createHexDump(const void* data, size_t size, const std::string& description) {
    std::ostringstream ss;
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    
    ss << "=== " << description << " (size: " << size << " bytes) ===" << std::endl;
    
    for (size_t i = 0; i < size; i += 16) {
        // Print offset
        ss << std::hex << std::setfill('0') << std::setw(8) << i << ": ";
        
        // Print hex bytes
        for (size_t j = 0; j < 16; ++j) {
            if (i + j < size) {
                ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i + j]) << " ";
            } else {
                ss << "   ";
            }
        }
        
        ss << " | ";
        
        // Print ASCII representation
        for (size_t j = 0; j < 16 && i + j < size; ++j) {
            unsigned char c = bytes[i + j];
            ss << (std::isprint(c) ? static_cast<char>(c) : '.');
        }
        
        ss << std::endl;
    }
    
    ss << "=== End " << description << " ===" << std::endl;
    return ss.str();
}


rsmi_status_t read_gpuboard_temp_metrics(const char* filename, amdgpu_gpuboard_temp_metrics_v1_0& metrics) {
    if (!filename) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
           << " | Fail | filename is null | Returning = "
           << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
        LOG_INFO(ss);
        return RSMI_STATUS_INVALID_ARGS;
    }

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
       << " | filename: " << filename;
    LOG_INFO(ss);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::ostringstream ess;
        ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Fail | Could not open file: " << filename
            << " | errno: " << errno << " (" << std::strerror(errno) << ")"
            << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
        LOG_INFO(ess);
        return ErrnoToRsmiStatus(errno);
    }

    // Clear the metrics structure
    std::memset(&metrics, 0, sizeof(metrics));

    // Read the entire structure
    file.read(reinterpret_cast<char*>(&metrics), sizeof(metrics));
    
    if (file.bad()) {
        std::ostringstream ess;
        ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Fail | File read error | errno: " << errno << " (" << std::strerror(errno) << ")"
            << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
        LOG_INFO(ess);
        return ErrnoToRsmiStatus(errno);
    }
    
    // Always create hex dump for debugging, using the number of bytes actually read
    std::string hexDump = createHexDump(&metrics, file.gcount(), "GPU Board Temperature Metrics");
    LOG_DEBUG(hexDump);

    if (file.gcount() != sizeof(metrics)) {
        std::ostringstream ess;
        ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Fail | Insufficient data read"
            << " | Expected: " << sizeof(metrics) << " bytes"
            << " | Actual: " << file.gcount() << " bytes"
            << " | Returning = " << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
        LOG_INFO(ess);
        return RSMI_STATUS_INSUFFICIENT_SIZE;
    }

    std::ostringstream oss;
    oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Success | File: " << filename
        << " | Bytes read: " << sizeof(metrics)
        << " | Header format: " << static_cast<unsigned>(metrics.common_header.format_revision)
        << " | Header content: " << static_cast<unsigned>(metrics.common_header.content_revision)
        << " | Node ID: " << metrics.node_id
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
    LOG_INFO(oss);

    return RSMI_STATUS_SUCCESS;
}

rsmi_status_t read_baseboard_temp_metrics(const char* filename, amdgpu_baseboard_temp_metrics_v1_0& metrics) {
    if (!filename) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
           << " | Fail | filename is null | Returning = "
           << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
        LOG_INFO(ss);
        return RSMI_STATUS_INVALID_ARGS;
    }

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
       << " | filename: " << filename;
    LOG_INFO(ss);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::ostringstream ess;
        ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Fail | Could not open file: " << filename
            << " | errno: " << errno << " (" << std::strerror(errno) << ")"
            << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
        LOG_INFO(ess);
        return ErrnoToRsmiStatus(errno);
    }

    // Clear the metrics structure
    std::memset(&metrics, 0, sizeof(metrics));

    // Read the entire structure
    file.read(reinterpret_cast<char*>(&metrics), sizeof(metrics));
    
    if (file.bad()) {
        std::ostringstream ess;
        ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Fail | File read error | errno: " << errno << " (" << std::strerror(errno) << ")"
            << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
        LOG_INFO(ess);
        return ErrnoToRsmiStatus(errno);
    }
    
    // Always create hex dump for debugging, using the number of bytes actually read
    std::string hexDump = createHexDump(&metrics, file.gcount(), "Baseboard Temperature Metrics");
    LOG_DEBUG(hexDump);

    if (file.gcount() != sizeof(metrics)) {
        std::ostringstream ess;
        ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Fail | Insufficient data read"
            << " | Expected: " << sizeof(metrics) << " bytes"
            << " | Actual: " << file.gcount() << " bytes"
            << " | Returning = " << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
        LOG_INFO(ess);
        return RSMI_STATUS_INSUFFICIENT_SIZE;
    }

    std::ostringstream oss;
    oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Success | File: " << filename
        << " | Bytes read: " << sizeof(metrics)
        << " | Header format: " << static_cast<unsigned>(metrics.common_header.format_revision)
        << " | Header content: " << static_cast<unsigned>(metrics.common_header.content_revision)
        << " | Node ID: " << metrics.node_id
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
    LOG_INFO(oss);

    return RSMI_STATUS_SUCCESS;
}


// Decode encoded temperature value: bits 24-31 = sensor id, bits 0-23 = signed temperature (Celsius)
static int32_t decode_temperature_value(uint32_t encoded, uint8_t* sensor_id = nullptr) {
    if (sensor_id) {
        *sensor_id = static_cast<uint8_t>((encoded >> 24) & 0xFF);
    }
    // Extract signed 24-bit temperature value
    int32_t temp = static_cast<int32_t>(encoded & 0xFFFFFF);
    // Sign-extend if negative
    if (temp & 0x800000) {
        temp |= ~0xFFFFFF;
    }

    temp *= 1000; // Convert Celsius to milli-Celsius
    return temp;
}

rsmi_status_t get_gpuboard_temp_value(const amdgpu_gpuboard_temp_metrics_v1_0& metrics,
                                      rsmi_temperature_type_t temperature_type,
                                      int64_t* value) {
    if (!value) {
        return RSMI_STATUS_INVALID_ARGS;
    }

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
       << " | Node ID: " << metrics.node_id
       << " | Temperature type: " << static_cast<int>(temperature_type);
    LOG_INFO(ss);

    *value = 0;  // Initialize to 0
    const uint32_t INVALID_VALUE = std::numeric_limits<uint32_t>::max();

    // Check VR (Voltage Regulator) temperatures first
    for (int i = 0; i < AMDGPU_VR_MAX_TEMP_ENTRIES; ++i) {
        if (metrics.vr_temp[i] != INVALID_VALUE) {
            auto it = vr_temp_map.find(i);
            if (it != vr_temp_map.end() && it->second == temperature_type) {
                *value = decode_temperature_value(metrics.vr_temp[i]);
                
                std::ostringstream oss;
                oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
                    << " | Success | VR temp found at index: " << i
                    << " | Raw value: " << *value
                    << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
                LOG_INFO(oss);
                return RSMI_STATUS_SUCCESS;
            }
        }
    }

    // Check node temperatures if not found in VR
    for (int i = 0; i < AMDGPU_NODE_MAX_TEMP_ENTRIES; ++i) {
        if (metrics.node_temp[i] != INVALID_VALUE) {  // Max int indicates invalid temperature reading
            auto it = node_temp_map.find(i);
            if (it != node_temp_map.end() && it->second == temperature_type) {
                *value = decode_temperature_value(metrics.node_temp[i]);

                std::ostringstream oss;
                oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
                    << " | Success | Node temp found at index: " << i
                    << " | Raw value: " << *value
                    << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
                LOG_INFO(oss);
                return RSMI_STATUS_SUCCESS;
            }
        }
    }

    // Temperature type not found in metrics
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Temperature type not found in GPU board metrics"
        << " | Temperature type: " << static_cast<int>(temperature_type)
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED) << " |";
    LOG_ERROR(ess);
    return RSMI_STATUS_NOT_SUPPORTED;
}

rsmi_status_t get_baseboard_temp_value(const amdgpu_baseboard_temp_metrics_v1_0& metrics,
                                       rsmi_temperature_type_t temperature_type,
                                       int64_t* value) {
    if (!value) {
        return RSMI_STATUS_INVALID_ARGS;
    }

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
       << " | Node ID: " << metrics.node_id
       << " | Temperature type: " << static_cast<int>(temperature_type);
    LOG_INFO(ss);

    *value = 0;  // Initialize to 0
    const uint32_t INVALID_VALUE = std::numeric_limits<uint32_t>::max();

    // Check system temperatures
    for (int i = 0; i < AMDGPU_SYSTEM_MAX_TEMP_ENTRIES; ++i) {
        if (metrics.system_temp[i] != INVALID_VALUE) {  // Max int indicates invalid temperature reading
            auto it = system_temp_map.find(i);
            if (it != system_temp_map.end() && it->second == temperature_type) {
                *value = decode_temperature_value(metrics.system_temp[i]);
                
                std::ostringstream oss;
                oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
                    << " | Success | System temp found at index: " << i
                    << " | Raw value: " << *value
                    << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
                LOG_INFO(oss);
                return RSMI_STATUS_SUCCESS;
            }
        }
    }

    // Temperature type not found in metrics
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Temperature type not found in baseboard metrics"
        << " | Temperature type: " << static_cast<int>(temperature_type)
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED) << " |";
    LOG_ERROR(ess);
    return RSMI_STATUS_NOT_SUPPORTED;
}


}  // end namespace