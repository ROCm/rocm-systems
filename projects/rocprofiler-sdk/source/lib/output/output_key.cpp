// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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
//

#include "output_key.hpp"
#include "format_path.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/utility.hpp"

#if !defined(_WIN32)
#    include <linux/limits.h>
#else
#    include <windows.h>
//
#    include <tlhelp32.h>
#endif
#include <array>
#include <fstream>

namespace rocprofiler
{
namespace tool
{
namespace
{
namespace fs = common::filesystem;

template <typename Tp>
auto
as_pointer(Tp&& _val)
{
    return new Tp{_val};
}

std::string*
get_local_datetime(const std::string& dt_format, std::time_t*& dt_curr);

std::time_t* launch_time  = nullptr;
const auto*  launch_clock = as_pointer(std::chrono::system_clock::now());
const auto*  launch_datetime =
    get_local_datetime(common::get_env("ROCPROF_TIME_FORMAT", "%F_%H.%M"), launch_time);
const auto* launch_date =
    get_local_datetime(common::get_env("ROCPROF_DATE_FORMAT", "%F"), launch_time);

std::string*
get_local_datetime(const std::string& dt_format, std::time_t*& _dt_curr)
{
    constexpr auto strsize = 512;

    if(!_dt_curr) _dt_curr = new std::time_t{std::time_t{std::time(nullptr)}};

    char mbstr[strsize] = {};
    memset(mbstr, '\0', sizeof(mbstr) * sizeof(char));

    struct tm tm_struct;
#if !defined(_WIN32)
    auto* _tm = localtime_r(_dt_curr, &tm_struct);
#else
    // note the reversed argument order relative to localtime_r
    auto* _tm = (::localtime_s(&tm_struct, _dt_curr) == 0) ? &tm_struct : nullptr;
#endif
    if(_tm != nullptr && std::strftime(mbstr, sizeof(mbstr) - 1, dt_format.c_str(), _tm) != 0)
        return new std::string{mbstr};

    return nullptr;
}

bool
not_is_space(int ch)
{
    return std::isspace(ch) == 0;
}

std::string
ltrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    return s;
}

std::string
rtrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
    return s;
}

std::string
trim(std::string s, bool (*f)(int) = not_is_space)
{
    ltrim(s, f);
    rtrim(s, f);
    return s;
}

#if !defined(_WIN32)
std::string
get_hostname()
{
    auto _hostname_buff = std::array<char, PATH_MAX>{};
    _hostname_buff.fill('\0');
    if(gethostname(_hostname_buff.data(), _hostname_buff.size() - 1) != 0)
    {
        auto _err = errno;
        ROCP_WARNING << "Hostname unknown. gethostname failed with error code " << _err << ": "
                     << strerror(_err);
        return std::string{"UNKNOWN_HOSTNAME"};
    }

    return std::string{_hostname_buff.data()};
}

pid_t
get_parent_process_id()
{
    return getppid();
}

pid_t
get_process_group_id()
{
    return getpgid(getpid());
}

pid_t
get_session_id()
{
    return getsid(getpid());
}

std::vector<pid_t>
get_siblings(pid_t _id = getppid())
{
    auto _data = std::vector<pid_t>{};

    auto _ifs = std::ifstream{"/proc/" + std::to_string(_id) + "/task/" + std::to_string(_id) +
                              "/children"};
    while(_ifs)
    {
        pid_t _n = 0;
        _ifs >> _n;
        if(!_ifs || _n <= 0) break;
        _data.emplace_back(_n);
    }
    return _data;
}
#else
std::string
get_hostname()
{
    auto _hostname_buff = std::array<char, MAX_COMPUTERNAME_LENGTH + 1>{};
    _hostname_buff.fill('\0');
    auto _size = static_cast<DWORD>(_hostname_buff.size());

    if(::GetComputerNameExA(ComputerNameDnsHostname, _hostname_buff.data(), &_size) == 0)
    {
        ROCP_WARNING << "Hostname unknown. GetComputerNameExA failed with error code "
                     << ::GetLastError();
        return std::string{"UNKNOWN_HOSTNAME"};
    }

    return std::string{_hostname_buff.data()};
}

// Windows does not track a parent link in the process itself, so the only way to recover it
// is to find this process in a system-wide snapshot. The same snapshot answers the sibling
// question, so both go through it.
std::vector<pid_t>
snapshot_processes(pid_t& parent_of_self)
{
    parent_of_self = 0;

    auto _snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if(_snapshot == INVALID_HANDLE_VALUE) return {};

    auto _self    = static_cast<DWORD>(common::get_pid());
    auto _entries = std::vector<std::pair<pid_t, pid_t> >{};
    auto _entry   = PROCESSENTRY32W{};
    _entry.dwSize = sizeof(_entry);

    if(::Process32FirstW(_snapshot, &_entry) != 0)
    {
        do
        {
            if(_entry.th32ProcessID == _self)
                parent_of_self = static_cast<pid_t>(_entry.th32ParentProcessID);
            _entries.emplace_back(static_cast<pid_t>(_entry.th32ProcessID),
                                  static_cast<pid_t>(_entry.th32ParentProcessID));
        } while(::Process32NextW(_snapshot, &_entry) != 0);
    }

    ::CloseHandle(_snapshot);

    auto _siblings = std::vector<pid_t>{};
    if(parent_of_self > 0)
        for(const auto& itr : _entries)
            if(itr.second == parent_of_self) _siblings.emplace_back(itr.first);

    return _siblings;
}

pid_t
get_parent_process_id()
{
    auto _ppid = pid_t{0};
    snapshot_processes(_ppid);
    return _ppid;
}

// Windows has no process group. Report 0 rather than substituting the job object, which is
// not the same concept and is usually absent.
pid_t
get_process_group_id()
{
    return 0;
}

// The Terminal Services session is the closest analogue of a POSIX session id: every process
// launched from the same logon shares it.
pid_t
get_session_id()
{
    auto _sid = DWORD{0};
    if(::ProcessIdToSessionId(::GetCurrentProcessId(), &_sid) == 0) return 0;
    return static_cast<pid_t>(_sid);
}

std::vector<pid_t>
get_siblings()
{
    auto _ppid = pid_t{0};
    return snapshot_processes(_ppid);
}
#endif

auto
get_num_siblings()
{
    return get_siblings().size();
}
}  // namespace

output_key::output_key(std::string _key, std::string _val, std::string _desc)
: key{std::move(_key)}
, value{std::move(_val)}
, description{std::move(_desc)}
{}

std::vector<output_key>
output_keys(std::string _tag)
{
    using strpair_t = std::pair<std::string, std::string>;

    auto _cmdline = common::read_command_line(common::get_pid());

    if(_tag.empty() && !_cmdline.empty()) _tag = fs::path{_cmdline.front()}.filename().string();

    std::string        _argv_string = {};    // entire argv cmd
    std::string        _args_string = {};    // cmdline args
    std::string        _argt_string = _tag;  // prefix + cmdline args
    const std::string& _tag0_string = _tag;  // only the basic prefix
    auto               _options     = std::vector<output_key>{};

    auto _replace = [](auto& _v, const strpair_t& pitr) {
        auto pos = std::string::npos;
        while((pos = _v.find(pitr.first)) != std::string::npos)
            _v.replace(pos, pitr.first.length(), pitr.second);
    };

    if(_cmdline.size() > 1 && _cmdline.at(1) == "--") _cmdline.erase(_cmdline.begin() + 1);

    for(auto& itr : _cmdline)
    {
        itr = trim(itr);
        _replace(itr, {"/", "_"});
        while(!itr.empty() && itr.at(0) == '.')
            itr = itr.substr(1);
        while(!itr.empty() && itr.at(0) == '_')
            itr = itr.substr(1);
    }

    if(!_cmdline.empty())
    {
        for(size_t i = 0; i < _cmdline.size(); ++i)
        {
            const auto  _l = std::string{(i == 0) ? "" : "_"};
            const auto& _v = _cmdline.at(i);
            _argv_string += _l + _v;
            if(i > 0)
            {
                _argt_string += (i > 1) ? (_l + _v) : _v;
                _args_string += (i > 1) ? (_l + _v) : _v;
            }
        }
    }

    auto _mpi_size = get_mpi_size();
    auto _mpi_rank = get_mpi_rank();

    auto _dmp_size      = fmt::format("{}", (_mpi_size) > 0 ? _mpi_size : 1);
    auto _dmp_rank      = fmt::format("{}", (_mpi_rank) > 0 ? _mpi_rank : 0);
    auto _proc_id       = fmt::format("{}", common::get_pid());
    auto _parent_id     = fmt::format("{}", get_parent_process_id());
    auto _pgroup_id     = fmt::format("{}", get_process_group_id());
    auto _session_id    = fmt::format("{}", get_session_id());
    auto _proc_size     = fmt::format("{}", get_num_siblings());
    auto _pwd_string    = common::get_env<std::string>("PWD", ".");
    auto _slurm_job_id  = common::get_env<std::string>("SLURM_JOB_ID", "0");
    auto _slurm_proc_id = common::get_env("SLURM_PROCID", _dmp_rank);

    auto _uniq_id = _proc_id;
    if(common::get_env<int32_t>("SLURM_PROCID", -1) >= 0)
    {
        _uniq_id = _slurm_proc_id;
    }
    else if(_mpi_size > 0 || _mpi_rank >= 0)
    {
        _uniq_id = _dmp_rank;
    }

    for(auto&& itr : std::initializer_list<output_key>{
            {"argv", _argv_string, "Entire command-line condensed into a single string"},
            {"argt",
             _argt_string,
             "Similar to `%argv%` except basename of first command line argument"},
            {"args", _args_string, "All command line arguments condensed into a single string"},
            {"tag", _tag0_string, "Basename of first command line argument"}})
    {
        _options.emplace_back(fmt::format("%{}%", itr.key), itr.value, itr.description);
        _options.emplace_back(fmt::format("{}{}{}", '{', itr.key, '}'), itr.value, itr.description);
    }

    if(!_cmdline.empty())
    {
        for(size_t i = 0; i < _cmdline.size(); ++i)
        {
            const auto& _v = _cmdline.at(i);
            auto itr = output_key{fmt::format("arg{}", i), _v, fmt::format("Argument #{}", i)};
            _options.emplace_back(fmt::format("%{}%", itr.key), itr.value, itr.description);
            _options.emplace_back(
                fmt::format("{}{}{}", '{', itr.key, '}'), itr.value, itr.description);
        }
    }

    auto _launch_time = (launch_datetime) ? *launch_datetime : std::string{".UNKNOWN_LAUNCH_TIME."};
    auto _launch_date = (launch_date) ? *launch_date : std::string{".UNKNOWN_LAUNCH_DATE."};
    auto _hostname    = get_hostname();

    for(auto&& itr : std::initializer_list<output_key>{
            {"hostname", _hostname, "Network hostname"},
            {"pid", _proc_id, "Process identifier"},
            {"ppid", _parent_id, "Parent process identifier"},
            {"pgid", _pgroup_id, "Process group identifier"},
            {"psid", _session_id, "Process session identifier"},
            {"psize", _proc_size, "Number of sibling process"},
            {"job", _slurm_job_id, "SLURM_JOB_ID env variable"},
            {"rank", _slurm_proc_id, "MPI/UPC++ rank"},
            {"size", _dmp_size, "MPI/UPC++ size"},
            {"nid", _uniq_id, "%rank% if possible, otherwise %pid%"},
            {"cwd", fs::current_path().string(), "Current working path"},
            {"launch_date", _launch_date, "Date according to date format ROCPROF_DATE_FORMAT"},
            {"launch_time", _launch_time, "Date and/or time according to ROCPROF_TIME_FORMAT"},
        })
    {
        _options.emplace_back(fmt::format("%{}%", itr.key), itr.value, itr.description);
        _options.emplace_back(fmt::format("{}{}{}", '{', itr.key, '}'), itr.value, itr.description);
    }

    for(auto&& itr : std::initializer_list<output_key>{
            {"%h", _hostname, "Shorthand for %hostname%"},
            {"%p", _proc_id, "Shorthand for %pid%"},
            {"%j", _slurm_job_id, "Shorthand for %job%"},
            {"%r", _slurm_proc_id, "Shorthand for %rank%"},
            {"%s", _dmp_size, "Shorthand for %size"},
        })
    {
        _options.emplace_back(itr);
    }

    return _options;
}
}  // namespace tool
}  // namespace rocprofiler
