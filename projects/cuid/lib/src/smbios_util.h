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

#ifndef SMBIOS_UTIL_H
#define SMBIOS_UTIL_H

#include "include/amd_cuid.h"
#include <iostream>

class SmbiosUtil {
public:
    static amdcuid_status_t get_system_uuid(uint8_t* uuid);
    static amdcuid_status_t get_uuid_from_smbios_table(uint8_t* uuid);
    static amdcuid_status_t get_system_serial(std::string &serial);
    static amdcuid_status_t get_board_info(std::string &vendor, std::string &name, std::string &version);
    static amdcuid_status_t get_bios_info(std::string &vendor, std::string &version, std::string &date);
    static amdcuid_status_t get_product_info(std::string &name, std::string &family);
private:
    static constexpr const char* DMI_PATH = "/sys/class/dmi/id/";
    static constexpr const char* DMI_TABLES_PATH = "/sys/firmware/dmi/tables/";
};

#endif // SMBIOS_UTIL_H