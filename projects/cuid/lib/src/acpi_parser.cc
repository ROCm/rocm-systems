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

#include "src/acpi_parser.h"
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include <errno.h>

// MADT entry type constants
constexpr uint8_t MADT_TYPE_LOCAL_APIC = 0;
constexpr uint8_t MADT_TYPE_LOCAL_X2APIC = 9;

// MADT flags
constexpr uint32_t MADT_FLAG_ENABLED = 0x00000001;

// ACPI table path
constexpr const char* ACPI_TABLES_PATH = "/sys/firmware/acpi/tables/";

amdcuid_status_t AcpiParser::read_acpi_table(const char* table_name, std::vector<uint8_t>& data) {
    std::string path = std::string(ACPI_TABLES_PATH) + table_name;
    
    // Check if file exists and get size
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        if (errno == ENOENT) {
            return AMDCUID_STATUS_ACPI_ERROR;
        } else if (errno == EACCES) {
            return AMDCUID_STATUS_PERMISSION_DENIED;
        }
        return AMDCUID_STATUS_FILE_ERROR;
    }
    
    // Read table data
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (errno == EACCES) {
            return AMDCUID_STATUS_PERMISSION_DENIED;
        }
        return AMDCUID_STATUS_FILE_ERROR;
    }
    
    data.resize(st.st_size);
    file.read(reinterpret_cast<char*>(data.data()), st.st_size);
    
    if (!file) {
        return AMDCUID_STATUS_FILE_ERROR;
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

bool AcpiParser::validate_checksum(const uint8_t* data, size_t length) {
    if (!data || length < sizeof(AcpiTableHeader)) {
        return false;
    }
    
    uint8_t sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }
    
    return sum == 0;
}

bool AcpiParser::parse_madt_entry(const uint8_t* entry, AcpiCpuInfo& cpu_info) {
    const MadtEntryHeader* header = reinterpret_cast<const MadtEntryHeader*>(entry);
    
    if (header->type == MADT_TYPE_LOCAL_APIC) {
        if (header->length < sizeof(MadtLocalApic)) {
            return false;
        }
        
        const MadtLocalApic* apic = reinterpret_cast<const MadtLocalApic*>(entry);
        
        cpu_info.apic_id = apic->apic_id;
        cpu_info.processor_uid = apic->acpi_processor_uid;
        cpu_info.enabled = (apic->flags & MADT_FLAG_ENABLED) != 0;
        cpu_info.is_x2apic = false;
        
        return true;
    }
    else if (header->type == MADT_TYPE_LOCAL_X2APIC) {
        if (header->length < sizeof(MadtLocalX2Apic)) {
            return false;
        }
        
        const MadtLocalX2Apic* x2apic = reinterpret_cast<const MadtLocalX2Apic*>(entry);
        
        cpu_info.apic_id = x2apic->x2apic_id;
        cpu_info.processor_uid = x2apic->acpi_processor_uid;
        cpu_info.enabled = (x2apic->flags & MADT_FLAG_ENABLED) != 0;
        cpu_info.is_x2apic = true;
        
        return true;
    }
    
    return false;
}

amdcuid_status_t AcpiParser::parse_madt(std::vector<AcpiCpuInfo>& cpu_info) {
    cpu_info.clear();
    
    // Read MADT/APIC table
    std::vector<uint8_t> table_data;
    amdcuid_status_t status = read_acpi_table("APIC", table_data);
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }
    
    // Validate minimum size
    if (table_data.size() < sizeof(MadtHeader)) {
        return AMDCUID_STATUS_INVALID_FORMAT;
    }
    
    const MadtHeader* madt = reinterpret_cast<const MadtHeader*>(table_data.data());
    
    // Validate signature
    if (std::memcmp(madt->header.signature, "APIC", 4) != 0) {
        return AMDCUID_STATUS_INVALID_FORMAT;
    }
    
    // Validate table length
    if (madt->header.length != table_data.size()) {
        return AMDCUID_STATUS_INVALID_FORMAT;
    }
    
    // Validate checksum
    if (!validate_checksum(table_data.data(), madt->header.length)) {
        return AMDCUID_STATUS_INVALID_FORMAT;
    }
    
    // Parse interrupt controller structures
    const uint8_t* entry = table_data.data() + sizeof(MadtHeader);
    const uint8_t* end = table_data.data() + madt->header.length;
    
    while (entry < end) {
        const MadtEntryHeader* header = reinterpret_cast<const MadtEntryHeader*>(entry);
        
        // Validate entry doesn't overflow table
        if (entry + header->length > end || header->length < sizeof(MadtEntryHeader)) {
            return AMDCUID_STATUS_INVALID_FORMAT;
        }
        
        // Parse Local APIC or x2APIC entries
        AcpiCpuInfo info;
        if (parse_madt_entry(entry, info)) {
            cpu_info.push_back(info);
        }
        
        entry += header->length;
    }
    
    // Should have found at least one CPU
    if (cpu_info.empty()) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AcpiParser::get_cpu_count(uint32_t& count) {
    std::vector<AcpiCpuInfo> cpu_info;
    amdcuid_status_t status = parse_madt(cpu_info);
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        count = 0;
        return status;
    }
    
    // Count only enabled CPUs
    count = 0;
    for (const auto& info : cpu_info) {
        if (info.enabled) {
            count++;
        }
    }
    
    return AMDCUID_STATUS_SUCCESS;
}
