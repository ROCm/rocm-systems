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
#include <iomanip>
#include <functional>
#include <optional>
#include <iostream>

#include <amd_smi/amdsmi.h>
#include <amd_smi/impl/amd_smi_system.h>
#include <amd_smi/impl/amd_smi_utils.h>
#include <amd_smi/impl/amd_smi_cper.h>

namespace {
struct RAII {
    RAII(std::function<void()> init, std::function<void()> finish) 
    : _finish(finish) {
        init();
    }
    ~RAII() {
        _finish();
    }
    std::function<void()> _finish;
};

std::string severityAsString(amdsmi_cper_sev_t sev) {
    switch(sev) {
        case AMDSMI_CPER_SEV_FATAL:
            return "FATAL";
        case AMDSMI_CPER_SEV_NON_FATAL_CORRECTED:
            return "NON FATAL CORRECTED";
        case AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED:
            return "NON FATAL UNCORRECTED";
        default:
            return "UNKNOWN";
    }
}
}//namespace

int main(int argc, char *argv[]) {

    if(argc != 2 || !argv[1]) {
        std::cout << "Missing path to a CPER file\n";
        return -1;
    }
    std::vector<uint8_t> cper_data(1024*8);
    uint64_t buf_size = cper_data.size();
    std::vector<amdsmi_cper_hdr_t> amdsmi_cper_hdrs(100);
    uint64_t entry_count = amdsmi_cper_hdrs.size();
    uint64_t cursor = 0;
    uint64_t product_serial = 0;
    amdsmi_cper_hdr_t *cper_hdrs = amdsmi_cper_hdrs.data();
    uint32_t severity_mask = 1 << AMDSMI_CPER_SEV_FATAL |
                             1 << AMDSMI_CPER_SEV_NON_FATAL_CORRECTED |
                             1 << AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED;
    amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
        argv[1], //const char *amdgpu_ring_cper_file, 
        severity_mask,//uint32_t severity_mask,
        reinterpret_cast<char*>(cper_data.data()), //char *cper_data, 
        &buf_size, //uint64_t *buf_size, 
        &cper_hdrs, //amdsmi_cper_hdr_t **cper_hdrs,
        &entry_count, //uint64_t *entry_count, 
        &cursor, 
        product_serial);
    std::cout << "amdsmi_get_gpu_cper_entries_by_path returned " << status << "\n";
    std::cout << "Number of CPER entries: " << entry_count << "\n";
    for(uint64_t i = 0; i < entry_count; ++i) { 
        amdsmi_cper_hdr_t &hdr = cper_hdrs[i];
        std::cout << "CPER Entry " << i << ":\n";
        std::cout << "  Timestampe: " 
                  << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(hdr.timestamp.month) << "/"
                  << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(hdr.timestamp.day) << "/"
                  << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(hdr.timestamp.year) << " "
                  << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(hdr.timestamp.hours) << ":"
                  << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(hdr.timestamp.minutes) << ":"
                  << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(hdr.timestamp.seconds) 
                  << "\n";
        std::cout << "  Signature: " << std::string(hdr.signature, 4) << "\n";
        std::cout << "  Revision: " << hdr.revision << "\n";
        std::cout << "  Error Severity: 0x" << std::hex << static_cast<uint32_t>(hdr.error_severity) << "\n";
        std::cout << "     " << severityAsString(hdr.error_severity) << "\n";
        std::cout << "  Record Length: " << hdr.record_length << "\n";
        std::cout << "  Creator ID: " << std::string(hdr.creator_id, 16) << "\n";
        std::cout << "  Record ID: " << std::string(hdr.record_id, 8) << "\n";
    }
    return status;
}
