// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/output/node_info.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/sha256.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/format.h>
#include <sys/utsname.h>

#include <fstream>

namespace rocprofiler
{
namespace tool
{
namespace
{
using utsname_t = struct utsname;

std::string
sha256_hex(const std::string& input)
{
    auto sha = common::sha256{};
    sha.update(input);
    return sha.hexdigest();
}

// --- Machine ID Utility ---
std::string
read_file_first_line(const std::string& path)
{
    if(auto file = std::ifstream{path}; file.is_open())
    {
        auto line = std::string{};
        std::getline(file, line);
        return line;
    }
    return std::string{};
}

std::string
get_mac_address(std::string_view iface)
{
    if(auto mac = read_file_first_line(fmt::format("/sys/class/net/{}/address", iface));
       !mac.empty())
        return mac;

    return {};
}

std::string
read_file(const std::string& filePath)
{
    auto file = std::ifstream{filePath, std::ios::in | std::ios::binary};
    if(file.is_open())
    {
        auto buffer = std::stringstream{};
        buffer << file.rdbuf();
        return buffer.str();
    }
    return std::string{};
}

std::string
get_mac_address(const std::vector<std::string>& interfaces = {"eth0", "enp0s3", "wlan0", "eno1"})
{
    namespace fs = ::rocprofiler::common::filesystem;

    auto remove_duplicates = [](auto _data) {
        std::sort(_data.begin(), _data.end());
        _data.erase(std::unique(_data.begin(), _data.end()), _data.end());
        return _data;
    };

    for(std::string_view iface : interfaces)
    {
        if(auto mac = get_mac_address(iface); !mac.empty()) return mac;
    }

    for(const auto& itr : fs::directory_iterator{fs::path{"/sys/class/net"}})
    {
        if(auto path = fs::path{itr}; fs::exists(path / "address"))
        {
            if(auto mac = get_mac_address(path.filename().string()); !mac.empty())
            {
                // some network interfaces have generic addresses like 00:00:00:00:00:00 or
                // ee:ee:ee:ee:ee:ee and we want to ignore these
                if(remove_duplicates(sdk::parse::tokenize(mac, ":")).size() > 1) return mac;
            }
        }
    }

    return std::string{};
}

std::string
get_machine_id()
{
    // not all Linux distributions have /etc/machine-id so we need to fallback on various
    // alternatives to try to uniquely identify the system
    if(std::string id = read_file_first_line("/etc/machine-id"); !id.empty()) return id;

    if(std::string id = read_file_first_line("/var/lib/dbus/machine-id"); !id.empty()) return id;

    //
    // for all values beyond this point, encrypt the id with sha256 since this is potentially
    // sensitive information. prefix is used for salt separation
    //
    if(std::string id = read_file_first_line("/sys/class/dmi/id/product_uuid"); !id.empty())
        return sha256_hex(fmt::format("product_uuid:{}", id));

    if(std::string id = read_file_first_line("/sys/class/dmi/id/board_serial"); !id.empty())
        return sha256_hex(fmt::format("board_serial:{}", id));

    if(std::string id = read_file("/proc/cpuinfo") + read_file("/proc/version") +
                        read_file("/proc/devices") + read_file("/proc/filesystems");
       !id.empty())
        return sha256_hex(fmt::format("procinfo:{}", id));

    if(std::string id = get_mac_address(); !id.empty())
        return sha256_hex(fmt::format("mac_address:{}", id));

    return std::string{};
}
}  // namespace

node_info&
read_node_info(node_info& _info)
{
    _info.machine_id = get_machine_id();

    auto _sys_info = utsname_t{};
    if(uname(&_sys_info) == 0)
    {
        auto _assign = [](auto& _dst, const char* _src) {
            if(_src) _dst = std::string{_src};
        };

        _assign(_info.system_name, _sys_info.sysname);
        _assign(_info.hostname, _sys_info.nodename);
        _assign(_info.release, _sys_info.release);
        _assign(_info.version, _sys_info.version);
        _assign(_info.hardware_name, _sys_info.machine);
        _assign(_info.domain_name, _sys_info.domainname);
    }
    else
    {
        ROCP_WARNING << "error retrieving uname info";
    }

    return _info;
}

node_info
read_node_info()
{
    auto _val = node_info{};
    return read_node_info(_val);
}
}  // namespace tool
}  // namespace rocprofiler
