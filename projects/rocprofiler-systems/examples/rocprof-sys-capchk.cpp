// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <linux/capability.h>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using cap_value_t = int;

struct cap_info
{
    const char* name  = nullptr;
    cap_value_t value = -1;
};

struct cap_status
{
    unsigned long long inherited = 0;
    unsigned long long permitted = 0;
    unsigned long long effective = 0;
    unsigned long long bounding  = 0;
    unsigned long long ambient   = 0;
};

#define CAP_INFO_ENTRY(VAL)                                                              \
    cap_info { #VAL, VAL }

namespace
{
std::initializer_list<cap_info> known_capabilities = {
#if defined(CAP_CHOWN)
    CAP_INFO_ENTRY(CAP_CHOWN),
#endif

#if defined(CAP_DAC_OVERRIDE)
    CAP_INFO_ENTRY(CAP_DAC_OVERRIDE),
#endif

#if defined(CAP_DAC_READ_SEARCH)
    CAP_INFO_ENTRY(CAP_DAC_READ_SEARCH),
#endif

#if defined(CAP_FOWNER)
    CAP_INFO_ENTRY(CAP_FOWNER),
#endif

#if defined(CAP_FSETID)
    CAP_INFO_ENTRY(CAP_FSETID),
#endif

#if defined(CAP_KILL)
    CAP_INFO_ENTRY(CAP_KILL),
#endif

#if defined(CAP_SETGID)
    CAP_INFO_ENTRY(CAP_SETGID),
#endif

#if defined(CAP_SETUID)
    CAP_INFO_ENTRY(CAP_SETUID),
#endif

#if defined(CAP_SETPCAP)
    CAP_INFO_ENTRY(CAP_SETPCAP),
#endif

#if defined(CAP_LINUX_IMMUTABLE)
    CAP_INFO_ENTRY(CAP_LINUX_IMMUTABLE),
#endif

#if defined(CAP_NET_BIND_SERVICE)
    CAP_INFO_ENTRY(CAP_NET_BIND_SERVICE),
#endif

#if defined(CAP_NET_BROADCAST)
    CAP_INFO_ENTRY(CAP_NET_BROADCAST),
#endif

#if defined(CAP_NET_ADMIN)
    CAP_INFO_ENTRY(CAP_NET_ADMIN),
#endif

#if defined(CAP_NET_RAW)
    CAP_INFO_ENTRY(CAP_NET_RAW),
#endif

#if defined(CAP_IPC_LOCK)
    CAP_INFO_ENTRY(CAP_IPC_LOCK),
#endif

#if defined(CAP_IPC_OWNER)
    CAP_INFO_ENTRY(CAP_IPC_OWNER),
#endif

#if defined(CAP_SYS_MODULE)
    CAP_INFO_ENTRY(CAP_SYS_MODULE),
#endif

#if defined(CAP_SYS_RAWIO)
    CAP_INFO_ENTRY(CAP_SYS_RAWIO),
#endif

#if defined(CAP_SYS_CHROOT)
    CAP_INFO_ENTRY(CAP_SYS_CHROOT),
#endif

#if defined(CAP_SYS_PTRACE)
    CAP_INFO_ENTRY(CAP_SYS_PTRACE),
#endif

#if defined(CAP_SYS_PACCT)
    CAP_INFO_ENTRY(CAP_SYS_PACCT),
#endif

#if defined(CAP_SYS_ADMIN)
    CAP_INFO_ENTRY(CAP_SYS_ADMIN),
#endif

#if defined(CAP_SYS_BOOT)
    CAP_INFO_ENTRY(CAP_SYS_BOOT),
#endif

#if defined(CAP_SYS_NICE)
    CAP_INFO_ENTRY(CAP_SYS_NICE),
#endif

#if defined(CAP_SYS_RESOURCE)
    CAP_INFO_ENTRY(CAP_SYS_RESOURCE),
#endif

#if defined(CAP_SYS_TIME)
    CAP_INFO_ENTRY(CAP_SYS_TIME),
#endif

#if defined(CAP_SYS_TTY_CONFIG)
    CAP_INFO_ENTRY(CAP_SYS_TTY_CONFIG),
#endif

#if defined(CAP_MKNOD)
    CAP_INFO_ENTRY(CAP_MKNOD),
#endif

#if defined(CAP_LEASE)
    CAP_INFO_ENTRY(CAP_LEASE),
#endif

#if defined(CAP_AUDIT_WRITE)
    CAP_INFO_ENTRY(CAP_AUDIT_WRITE),
#endif

#if defined(CAP_AUDIT_CONTROL)
    CAP_INFO_ENTRY(CAP_AUDIT_CONTROL),
#endif

#if defined(CAP_SETFCAP)
    CAP_INFO_ENTRY(CAP_SETFCAP),
#endif

#if defined(CAP_MAC_OVERRIDE)
    CAP_INFO_ENTRY(CAP_MAC_OVERRIDE),
#endif

#if defined(CAP_MAC_ADMIN)
    CAP_INFO_ENTRY(CAP_MAC_ADMIN),
#endif

#if defined(CAP_SYSLOG)
    CAP_INFO_ENTRY(CAP_SYSLOG),
#endif

#if defined(CAP_WAKE_ALARM)
    CAP_INFO_ENTRY(CAP_WAKE_ALARM),
#endif

#if defined(CAP_BLOCK_SUSPEND)
    CAP_INFO_ENTRY(CAP_BLOCK_SUSPEND),
#endif

#if defined(CAP_AUDIT_READ)
    CAP_INFO_ENTRY(CAP_AUDIT_READ),
#endif

#if defined(CAP_PERFMON)
    CAP_INFO_ENTRY(CAP_PERFMON),
#endif

#if defined(CAP_BPF)
    CAP_INFO_ENTRY(CAP_BPF),
#endif

#if defined(CAP_CHECKPOINT_RESTORE)
    CAP_INFO_ENTRY(CAP_CHECKPOINT_RESTORE),
#endif

#if defined(CAP_LAST_CAP)
    CAP_INFO_ENTRY(CAP_LAST_CAP),
#endif
};

auto cap_max_bits_v = []() {
    unsigned _value = 0;
    for(const auto& itr : known_capabilities)
        _value = std::max<unsigned>(_value, itr.value + 1);
    return _value;
}();

std::string
to_lower(std::string&& _s)
{
    for(auto& citr : _s)
        citr = tolower(citr);
    return _s;
}

std::string
to_upper(std::string _s)
{
    for(auto& citr : _s)
        citr = toupper(citr);
    return _s;
}

cap_status
cap_read(pid_t _pid)
{
    auto fname = std::string{ "/proc/" } + std::to_string(_pid) + "/status";
    auto ifs   = std::ifstream{ fname };
    if(!ifs) return cap_status{};

    auto _lines = std::vector<std::string>{};

    while(ifs && ifs.good())
    {
        auto _line = std::string{};
        std::getline(ifs, _line);
        if(ifs && ifs.good() && !_line.empty()) _lines.emplace_back(std::move(_line));
    }

    auto _data = cap_status{};
    for(const auto& itr : _lines)
    {
        auto iss    = std::istringstream{ itr };
        auto _key   = std::string{};
        auto _value = std::string{};
        iss >> _key;

        if(_key.find("Cap") == 0) iss >> _value;

        if(!_value.empty())
        {
            auto _key_matches = [&_key](std::string_view _cap_id_str) {
                return (_key.find(_cap_id_str) == 0);
            };
            if(_key_matches("CapInh"))
                _data.inherited = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapPrm"))
                _data.permitted = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapEff"))
                _data.effective = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapBnd"))
                _data.bounding = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapAmb"))
                _data.ambient = std::stoull(_value, nullptr, 16);
        }
    }

    return _data;
}

std::string
cap_name(cap_value_t _v)
{
    for(const auto& itr : known_capabilities)
        if(itr.value == _v) return to_lower(std::string{ itr.name });

    return std::string{};
}

std::vector<cap_value_t>
decode(unsigned long long value)
{
    auto _data = std::vector<cap_value_t>{};
    for(unsigned cap = 0; (cap < 64) && ((value >> cap) != 0U); ++cap)
    {
        auto _mask = value & (1ULL << cap);
        if(_mask != 0U)
        {
            if(cap >= 0 && cap < cap_max_bits_v) _data.emplace_back(cap);
        }
    }

    return _data;
}

std::vector<cap_value_t>
decode(const char* arg)
{
    return decode(std::strtoull(arg, nullptr, 16));
}
/*
template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16)
{
    std::stringstream _ss;
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _v;
    return _ss.str();
}

void run(std::string&& arg)
{
    if(arg.find("0x") == 0)
        arg = arg.substr(2);

    arg.insert(0, "0x");
    // arg = std::string{ "0x" } + arg;

    std::cout << arg << "=";
    auto _decoded = decode(arg.c_str());
    auto _msg     = std::stringstream{};
    for(auto&& itr : _decoded)
    {
        _msg << "," << cap_name(itr);
    }
    auto _msg_v = _msg.str();
    if(!_msg_v.empty())
    {
        std::cout << _msg_v.substr(1);
    }
    std::cout << "\n";
}

void
run(std::string_view _label, unsigned long long arg)
{
    std::cout << " " << std::setw(12) << _label << " ";
    run(as_hex(arg));
}
*/
}  // namespace

int
main(int argc, char** argv)
{
    const auto* _usage = R"usage(
usage: rocprof-sys-capchk <capability-name> <capability-set> <pid>

    Description:
        Simple tool for checking the effective capabilities of a running process

    Arguments:
        capability-name (string):
            case-insensitive string matching CAP_* fields defined in `man 7 capabilities`

        capability-set (string; optional):
            Choices:
                - effective (default)
                - permitted
                - inherited
                - bounding
                - ambient
            See `man 7 capabilities` for more info

        pid (numeric process-identifier; optional):
            target process identifier for capability query. If not specified, queries the
            capabilities of the current process (this exe).

    Exit value:
        0 if the process has the specified capability in the specified set
        1 if the process does not have the capability
        2 if the capability name is not supported
        3 if the capability set name is not supported

    Examples:
        $ rocprof-sys-capchk CAP_SYS_ADMIN
            Check if this exe (self) has CAP_SYS_ADMIN capability in the (default) effective capability set

        $ rocprof-sys-capchk sys_admin bounding 423032
            Check if process 423032 has CAP_SYS_ADMIN capability in the bounding capability set
    )usage";

    std::string capability_name = {};
    std::string capability_mode = "effective";
    pid_t       target_pid      = getpid();

    for(int i = 1; i < argc; ++i)
    {
        auto arg = std::string_view{ argv[i] };
        if(arg == "-h" || arg == "--help" || arg == "-?")
        {
            std::cout << _usage << "\n";
            return EXIT_SUCCESS;
        }
    }

    if(argc > 1) capability_name = to_lower(std::string{ argv[1] });
    if(argc > 2) capability_mode = to_lower(std::string{ argv[2] });
    if(argc > 3)
    {
        auto _pid_s = to_lower(std::string{ argv[3] });
        if(_pid_s != "self") target_pid = std::stoul(argv[3]);
    }

    if(capability_name.find("cap_") != 0) capability_name.insert(0, "cap_");

    capability_name = to_upper(capability_name);

    const cap_info* _info = nullptr;
    for(const auto& itr : known_capabilities)
    {
        if(capability_name == std::string_view{ itr.name })
        {
            _info = &itr;
            break;
        }
    }

    if(!_info)
    {
        fprintf(stderr, "Error! invalid capability: %s\n", capability_name.c_str());
        return EXIT_FAILURE + 1;
    }

    auto  _status  = cap_read(target_pid);
    auto* _dataset = &_status.effective;

    /*
    std::cout << "pid=" << target_pid << ":\n";
    run("inherited", _status.inherited);
    run("permitted", _status.permitted);
    run("effective", _status.effective);
    run("bounding", _status.bounding);
    run("ambient", _status.ambient);
    */

    if(capability_mode == "effective")
        _dataset = &_status.effective;
    else if(capability_mode == "permitted")
        _dataset = &_status.permitted;
    else if(capability_mode == "inherited")
        _dataset = &_status.inherited;
    else if(capability_mode == "bounding")
        _dataset = &_status.bounding;
    else if(capability_mode == "ambient")
        _dataset = &_status.ambient;
    else
    {
        fprintf(stderr, "Error! invalid capability set: %s\n", capability_mode.c_str());
        return EXIT_FAILURE + 2;
    }

    auto _ec = EXIT_FAILURE;
    for(auto&& itr : decode(*_dataset))
    {
        if(itr == _info->value)
        {
            _ec = EXIT_SUCCESS;
            break;
        }
    }

    std::cout << ((_ec == EXIT_SUCCESS) ? "Found" : "Missing") << " capability "
              << capability_name << " in " << capability_mode
              << " capability set. Exit code: " << _ec << ".\n";
    return _ec;
}
