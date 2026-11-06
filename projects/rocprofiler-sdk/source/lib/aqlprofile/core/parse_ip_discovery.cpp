// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <regex>
#include <iomanip>
#include <cassert>

#include "lib/aqlprofile/core/ip_discovery.h"
#include "lib/common/environment.hpp"

#define PCI_BUS_NUM(x)  (((x) >> 8) & 0xff)
#define PCI_SLOT(devfn) (((devfn) >> 3) & 0x1f)
#define PCI_FUNC(devfn) ((devfn) &0x07)

namespace fs = std::filesystem;

namespace
{
constexpr auto ip_discovery_path_env = "AQLPROFILE_CUSTOM_IP_DISCOVERY_PATH";

/**
 * @brief Reads a single integer (decimal or hexadecimal) from a sysfs file.
 *
 * This helper function reads a file containing a single numeric value and parses it
 * as either a decimal or hexadecimal integer, based on the provided flag.
 *
 * @param fname The path to the sysfs file containing the numeric value.
 *
 * @return An `std::optional<int>` containing the parsed integer if successful, or `std::nullopt`
 *         if the file does not exist, cannot be opened, or contains invalid data.
 */
std::optional<int>
read_sysfs_single_int(const fs::path& path)
{
    std::ifstream file(path);
    if(!file.is_open()) return std::nullopt;  // Failed to open file

    int value;
    file >> value;
    if(file.fail()) return std::nullopt;  // Failed to parse data

    file.close();
    return value;
}

/**
 * @brief Reads base address segments from a sysfs file.
 *
 * This helper function reads a file containing hexadecimal values representing
 * base address segments and parses them into a `base_addr_segments_t` structure.
 *
 * @param fname The path to the sysfs file containing base address segments.
 *
 * @return An `std::optional<base_addr_segments_t>` containing the parsed base address segments
 *         if successful, or `std::nullopt` if the file does not exist, cannot be opened,
 *         or contains invalid data.
 */
struct parsed_base_addr_segments
{
    base_addr_segments_t values{};
    size_t               count = 0;
};

std::optional<parsed_base_addr_segments>
read_sysfs_base_addr_segments(const fs::path& path)
{
    std::ifstream file(path);
    if(!file.is_open()) return std::nullopt;  // Failed to open file

    parsed_base_addr_segments segments{};
    std::string               databuf;
    while(std::getline(file, databuf) && segments.count < segments.values.size())
    {
        std::stringstream ss(databuf);
        ss >> std::hex >> segments.values[segments.count];
        if(ss.fail()) return std::nullopt;  // Failed to parse data
        ++segments.count;
    }

    return segments;
}

/**
 * @brief Parses IP instances for a given die and IP name from the sysfs directory structure.
 *
 * This function reads attributes such as base address segments, version information,
 * and instance number for each IP instance and stores them in the discovery table.
 *
 * @param die_num The die number associated with the IP instances.
 * @param diepath The sysfs path to the die directory.
 * @param ipname The name of the IP to be parsed.
 *
 *  @return The discovery table where parsed IP instance data will be stored.
 */
discovery_table_t
parse_ip_instances(int die_num, const fs::path& diepath, const std::string& ipname)
{
    // /sys/bus/pci/devices/{domain_bdf_str}/ip_discovery/die{die_num}/{ipname}
    const fs::path dir_path = fs::path(diepath) / ipname;
    if(!fs::exists(dir_path) || !fs::is_directory(dir_path))
    {
        throw std::runtime_error("sysfs path does not exist or is not a directory: " +
                                 dir_path.string());
    }

    discovery_table_t instances{};

    // sub-folders in "/sys/bus/pci/devices/{domain_bdf_str}/ip_discovery/die{die_num}/{ipname}"
    for(const auto& dir_entry : fs::directory_iterator(dir_path))
    {
        if(!std::isdigit(dir_entry.path().filename().string()[0])) continue;

        discovery_table_entry_t table_entry{};
        table_entry.die = die_num;
        table_entry.id  = std::stoi(dir_entry.path().filename());

        // "/sys/bus/pci/devices/{domain_bdf_str}/ip_discovery/die{die_num}/{ipname}/{instance_num}"
        fs::path instance_path = dir_path / dir_entry.path().filename();

        // base_addr list
        if(auto segments = read_sysfs_base_addr_segments(instance_path / "base_addr"))
        {
            table_entry.segments      = segments->values;
            table_entry.segment_count = segments->count;
        }
        else
            throw std::runtime_error("Failed to read IP base_addr segments for ipname=" + ipname +
                                     " die=" + std::to_string(die_num));

        // major
        if(auto major = read_sysfs_single_int(instance_path / "major"))
            table_entry.major = *major;
        else
            throw std::runtime_error("Failed to read IP major version for ipname=" + ipname +
                                     " die=" + std::to_string(die_num));

        // minor
        if(auto minor = read_sysfs_single_int(instance_path / "minor"))
            table_entry.minor = *minor;
        else
            throw std::runtime_error("Failed to read IP minor version for ipname=" + ipname +
                                     " die=" + std::to_string(die_num));

        // revision
        if(auto revision = read_sysfs_single_int(instance_path / "revision"))
            table_entry.revision = *revision;
        else
            throw std::runtime_error("Failed to read IP revision for ipname=" + ipname +
                                     " die=" + std::to_string(die_num));

        // instance
        if(auto instance = read_sysfs_single_int(instance_path / "num_instance"))
            table_entry.instance = *instance;
        else
            throw std::runtime_error("Failed to read IP instance for ipname=" + ipname +
                                     " die=" + std::to_string(die_num));

        // convert name to lowercase
        table_entry.ipname = ipname;
        std::transform(table_entry.ipname.begin(),
                       table_entry.ipname.end(),
                       table_entry.ipname.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        instances.emplace_back(table_entry);
    }

    return instances;
}

/**
 * @brief Generates a PCI domain BDF (Bus:Device.Function) string.
 *
 * This function converts the given PCI domain and BDF (Bus:Device.Function) values
 * into a standardized string format: "Domain:Bus:Device.Function".
 *
 * @param domain The PCI domain number (32-bit unsigned integer).
 * @param bdf The PCI Bus/Device/Function (BDF) value (32-bit unsigned integer).
 *
 * @return A string representing the PCI domain and BDF in the format "Domain:Bus:Device.Function".
 *         Example: "0000:47:00.0".
 *
 * @details
 * - The domain is represented as a 4-digit hexadecimal value.
 * - The bus is represented as a 2-digit hexadecimal value.
 * - The device is represented as a 2-digit hexadecimal value.
 * - The function is represented as a single decimal digit.
 */
std::string
get_domain_bdf_str(uint32_t domain, uint32_t bdf)
{
    uint8_t pci_bus   = PCI_BUS_NUM(bdf);
    uint8_t pci_devfn = bdf & 0xFF;
    uint8_t pci_dev   = PCI_SLOT(pci_devfn);
    uint8_t pci_func  = 0;  // PCI_FUNC(pci_devfn); // Future ToDo: Use the macro PCI_FUNC() to
                            // support multiple functions. For now, it's always zero.

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(4) << domain << ":" << std::setw(2)
       << static_cast<int>(pci_bus) << ":" << std::setw(2) << static_cast<int>(pci_dev) << "."
       << static_cast<int>(pci_func);
    return ss.str();
}

}  // namespace

bool
use_custom_ip_discovery_path()
{
    return rocprofiler::common::get_env_optional(ip_discovery_path_env).has_value();
}

/**
 * @brief Parses IP discovery information for a given PCI domain and BDF (Bus:Device.Function).
 *
 * This function discovers IP instances for all dies associated with a given PCI device.
 * It reads the sysfs directory structure to extract information about IP instances
 * and populates the provided discovery table.
 *
 * @param domain The PCI domain number (32-bit unsigned integer).
 * @param bdf The PCI Bus/Device/Function (BDF) value (32-bit unsigned integer).
 * @return table The discovery table where parsed IP instance data will be stored.
 *
 * @throws std::runtime_error If the sysfs directory does not exist, is not a directory,
 *                            or if no IP instances are found.
 *
 * @details
 * - Uses `AQLPROFILE_CUSTOM_IP_DISCOVERY_PATH` as the PCI device directory when it is set.
 * - Otherwise, constructs the sysfs path for the PCI device using the domain and BDF values.
 * - Iterates over the dies in the selected directory.
 * - For each die, iterates over the IP directories and calls `parse_ip_instances` to parse
 *   individual IP instance data.
 * - If no IP instances are found, throws an exception.
 */
discovery_table_t
parse_ip_discovery(uint32_t domain, uint32_t bdf)
{
    // /sys/bus/pci/devices/{domain_bdf_str}/ip_discovery/die
    const auto ip_discover_override = rocprofiler::common::get_env_optional(ip_discovery_path_env);
    if(ip_discover_override && ip_discover_override->empty())
        throw std::runtime_error(std::string{ip_discovery_path_env} + " is set but empty");

    const auto sysfs_path =
        fs::path("/sys/bus/pci/devices") / get_domain_bdf_str(domain, bdf) / "ip_discovery/die";

    const fs::path die_path =
        ip_discover_override ? fs::path{*ip_discover_override} / "ip_discovery/die" : sysfs_path;

    if(!fs::exists(die_path) || !fs::is_directory(die_path))
    {
        throw std::runtime_error("IP discovery path does not exist or is not a directory: " +
                                 die_path.string());
    }

    discovery_table_t table{};

    // iterate over every die
    // subfolders in "/sys/bus/pci/devices/{domain_bdf_str}/ip_discovery/die"
    for(const auto& die_entry : fs::directory_iterator(die_path))
    {
        if(!die_entry.is_directory()) continue;

        // "/sys/bus/pci/devices/{domain_bdf_str}/ip_discovery/die/{die_num}"
        const fs::path die_entry_path = die_entry.path();
        int            die_num        = std::stoi(die_entry_path.filename());

        // subfolders in "/sys/bus/pci/devices/{domain_bdf_str}/ip_discovery/die/{die_num}"
        for(const auto& ip_entry : fs::directory_iterator(die_entry_path))
        {
            if(!ip_entry.is_directory()) continue;
            const std::string filename = ip_entry.path().filename().string();
            if(std::isalpha(filename[0]))
            {
                const auto instances = parse_ip_instances(die_num, die_entry.path(), filename);
                table.insert(table.end(), instances.begin(), instances.end());
            }
        }
    }

    if(table.empty())
    {
        throw std::runtime_error("No IP instances found");
    }

    return table;
}
