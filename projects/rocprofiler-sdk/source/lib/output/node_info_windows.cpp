// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Windows counterpart of node_info.cpp. The Linux version fills node_info from uname() and
// sysfs; neither exists here, so the same fields come from the computer name APIs and the
// registry keys the OS version is published under.

#include "lib/common/logging.hpp"
#include "lib/output/node_info.hpp"

#include <fmt/format.h>

#include <windows.h>

#include <array>
#include <string>

namespace rocprofiler
{
namespace tool
{
namespace
{
std::string
read_registry_string(const char* subkey, const char* value)
{
    auto _buff = std::array<char, 512>{};
    _buff.fill('\0');
    auto _size = static_cast<DWORD>(_buff.size() - 1);

    // RRF_SUBKEY_WOW6464KEY: the OS version and machine identity live in the 64-bit view, and
    // a 32-bit build would otherwise be redirected to the WOW6432Node copy.
    if(::RegGetValueA(HKEY_LOCAL_MACHINE,
                      subkey,
                      value,
                      RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
                      nullptr,
                      _buff.data(),
                      &_size) != ERROR_SUCCESS)
        return std::string{};

    return std::string{_buff.data()};
}

uint32_t
read_registry_dword(const char* subkey, const char* value)
{
    auto _val  = DWORD{0};
    auto _size = static_cast<DWORD>(sizeof(_val));

    if(::RegGetValueA(HKEY_LOCAL_MACHINE,
                      subkey,
                      value,
                      RRF_RT_REG_DWORD | RRF_SUBKEY_WOW6464KEY,
                      nullptr,
                      &_val,
                      &_size) != ERROR_SUCCESS)
        return 0;

    return static_cast<uint32_t>(_val);
}

std::string
get_computer_name(COMPUTER_NAME_FORMAT format)
{
    auto _buff = std::array<char, 512>{};
    _buff.fill('\0');
    auto _size = static_cast<DWORD>(_buff.size());

    if(::GetComputerNameExA(format, _buff.data(), &_size) == 0) return std::string{};

    return std::string{_buff.data()};
}

std::string
get_machine_id()
{
    // MachineGuid is written once at install time and is the direct counterpart of
    // /etc/machine-id.
    return read_registry_string("SOFTWARE\\Microsoft\\Cryptography", "MachineGuid");
}

std::string
get_hardware_name()
{
    auto _info = SYSTEM_INFO{};
    ::GetNativeSystemInfo(&_info);

    switch(_info.wProcessorArchitecture)
    {
        case PROCESSOR_ARCHITECTURE_AMD64: return "x86_64";
        case PROCESSOR_ARCHITECTURE_ARM64: return "aarch64";
        case PROCESSOR_ARCHITECTURE_INTEL: return "i686";
        default: break;
    }

    return fmt::format("unknown-{}", _info.wProcessorArchitecture);
}
}  // namespace

node_info&
read_node_info(node_info& _info)
{
    static constexpr auto* current_version = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";

    _info.machine_id  = get_machine_id();
    _info.system_name = "Windows";
    _info.hostname    = get_computer_name(ComputerNameDnsHostname);
    _info.domain_name = get_computer_name(ComputerNameDnsDomain);

    // uname reports the kernel revision in release and the product string in version; the
    // closest equivalents are the build number (with its update revision) and the edition.
    auto _build = read_registry_string(current_version, "CurrentBuildNumber");
    auto _ubr   = read_registry_dword(current_version, "UBR");
    if(!_build.empty()) _info.release = fmt::format("{}.{}", _build, _ubr);

    auto _product = read_registry_string(current_version, "ProductName");
    auto _display = read_registry_string(current_version, "DisplayVersion");
    if(!_product.empty())
        _info.version = _display.empty() ? _product : fmt::format("{} {}", _product, _display);

    _info.hardware_name = get_hardware_name();

    ROCP_WARNING_IF(_info.machine_id.empty()) << "machine id unavailable: could not read "
                                                 "HKLM\\SOFTWARE\\Microsoft\\Cryptography";

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
