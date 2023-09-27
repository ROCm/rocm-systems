// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "lib/common/config.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/utility.hpp"

#include <fmt/core.h>

#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace
{
std::time_t* launch_time = new std::time_t{std::time(nullptr)};

std::string
get_local_datetime(const char* dt_format, std::time_t* dt_curr)
{
    char mbstr[512];
    if(!dt_curr) dt_curr = launch_time;

    if(std::strftime(mbstr, sizeof(mbstr), dt_format, std::localtime(dt_curr)) != 0)
        return std::string{mbstr};
    return std::string{};
}

inline bool
not_is_space(int ch)
{
    return std::isspace(ch) == 0;
}

inline std::string
ltrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    return s;
}

inline std::string
rtrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
    return s;
}

inline std::string
trim(std::string s, bool (*f)(int) = not_is_space)
{
    ltrim(s, f);
    rtrim(s, f);
    return s;
}

inline std::vector<pid_t>
get_siblings(pid_t _id = getppid())
{
    auto _data = std::vector<pid_t>{};

    std::ifstream _ifs{"/proc/" + std::to_string(_id) + "/task/" + std::to_string(_id) +
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

inline auto
get_num_siblings(pid_t _id = getppid())
{
    return get_siblings(_id).size();
}
}  // namespace

int
get_mpi_size()
{
    static int _v = get_env<int>("OMPI_COMM_WORLD_SIZE",
                                 get_env<int>("MV2_COMM_WORLD_SIZE", get_env<int>("MPI_SIZE", 0)));
    return _v;
}

int
get_mpi_rank()
{
    static int _v = get_env<int>("OMPI_COMM_WORLD_RANK",
                                 get_env<int>("MV2_COMM_WORLD_RANK", get_env<int>("MPI_RANK", -1)));
    return _v;
}

std::vector<output_key>
output_keys(std::string _tag)
{
    using strpair_t = std::pair<std::string, std::string>;

    auto _cmdline = read_command_line(getpid());

    if(_tag.empty() && !_cmdline.empty()) _tag = ::basename(_cmdline.front().c_str());

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
            const auto _l = std::string{(i == 0) ? "" : "_"};
            auto       _v = _cmdline.at(i);
            _argv_string += _l + _v;
            if(i > 0)
            {
                _argt_string += (i > 1) ? (_l + _v) : _v;
                _args_string += (i > 1) ? (_l + _v) : _v;
            }
        }
    }

    auto* _launch_time = launch_time;
    auto  _time_format = get_env<std::string>("ROCP_TIME_FORMAT", "%F_%H.%M");

    auto _mpi_size = get_env<int>("OMPI_COMM_WORLD_SIZE", get_env<int>("MV2_COMM_WORLD_SIZE", 0));
    auto _mpi_rank = get_env<int>("OMPI_COMM_WORLD_RANK", get_env<int>("MV2_COMM_WORLD_RANK", -1));

    auto _dmp_size      = fmt::format("{}", (_mpi_size) > 0 ? _mpi_size : 1);
    auto _dmp_rank      = fmt::format("{}", (_mpi_rank) > 0 ? _mpi_rank : 0);
    auto _proc_id       = fmt::format("{}", getpid());
    auto _parent_id     = fmt::format("{}", getppid());
    auto _pgroup_id     = fmt::format("{}", getpgid(getpid()));
    auto _session_id    = fmt::format("{}", getsid(getpid()));
    auto _proc_size     = fmt::format("{}", get_num_siblings());
    auto _pwd_string    = get_env<std::string>("PWD", ".");
    auto _slurm_job_id  = get_env<std::string>("SLURM_JOB_ID", "0");
    auto _slurm_proc_id = get_env("SLURM_PROCID", _dmp_rank);
    auto _launch_string = get_local_datetime(_time_format.c_str(), _launch_time);

    auto _uniq_id = _proc_id;
    if(get_env<int32_t>("SLURM_PROCID", -1) >= 0)
    {
        _uniq_id = _slurm_proc_id;
    }
    else if(_mpi_size > 0 || _mpi_rank >= 0)
    {
        _uniq_id = _dmp_rank;
    }

    for(auto&& itr : std::initializer_list<output_key>{
            {"%argv%", _argv_string, "Entire command-line condensed into a single string"},
            {"%argt%",
             _argt_string,
             "Similar to `%argv%` except basename of first command line argument"},
            {"%args%", _args_string, "All command line arguments condensed into a single string"},
            {"%tag%", _tag0_string, "Basename of first command line argument"}})
    {
        _options.emplace_back(itr);
    }

    if(!_cmdline.empty())
    {
        for(size_t i = 0; i < _cmdline.size(); ++i)
        {
            auto _v = _cmdline.at(i);
            _options.emplace_back(fmt::format("%arg{}%", i), _v, fmt::format("Argument #{}", i));
        }
    }

    for(auto&& itr : std::initializer_list<output_key>{
            {"%pid%", _proc_id, "Process identifier"},
            {"%ppid%", _parent_id, "Parent process identifier"},
            {"%pgid%", _pgroup_id, "Process group identifier"},
            {"%psid%", _session_id, "Process session identifier"},
            {"%psize%", _proc_size, "Number of sibling process"},
            {"%job%", _slurm_job_id, "SLURM_JOB_ID env variable"},
            {"%rank%", _slurm_proc_id, "MPI/UPC++ rank"},
            {"%size%", _dmp_size, "MPI/UPC++ size"},
            {"%nid%", _uniq_id, "%rank% if possible, otherwise %pid%"},
            {"%launch_time%", _launch_string, "Data and/or time of run according to time format"},
        })
    {
        _options.emplace_back(itr);
    }

    for(auto&& itr : std::initializer_list<output_key>{
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

std::string
format(std::string _fpath, const std::string& _tag)
{
    if(_fpath.find('%') == std::string::npos && _fpath.find('$') == std::string::npos)
        return _fpath;

    auto _replace = [](auto& _v, const output_key& pitr) {
        auto pos = std::string::npos;
        while((pos = _v.find(pitr.key)) != std::string::npos)
            _v.replace(pos, pitr.key.length(), pitr.value);
    };

    for(auto&& itr : output_keys(_tag))
        _replace(_fpath, itr);

    // environment and configuration variables
    try
    {
        for(const auto& _expr : {std::string{"(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)"},
                                 std::string{"(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)"}})
        {
            std::regex  _re{_expr};
            std::string _cbeg   = (_expr.find("(.*)%") == 0) ? "%" : "$";
            std::string _cend   = (_expr.find("(.*)%") == 0) ? "}%" : "}";
            bool        _is_env = (_expr.find("(env|ENV)") != std::string::npos);
            _cbeg += (_is_env) ? "env{" : "cfg{";
            while(std::regex_search(_fpath, _re))
            {
                auto        _var = std::regex_replace(_fpath, _re, "$3");
                std::string _val = {};
                if(_is_env)
                {
                    _val = get_env<std::string>(_var, "");
                }
                auto _beg = std::regex_replace(_fpath, _re, "$1");
                auto _end = std::regex_replace(_fpath, _re, "$4");
                _fpath    = fmt::format("{}{}{}", _beg, _val, _end);
            }
        }
    } catch(std::exception& _e)
    {
        LOG(WARNING) << "[rocprofiler] " << __FUNCTION__ << " threw an exception :: " << _e.what()
                     << "\n";
    }

    // remove %arg<N>% where N >= argc
    try
    {
        std::regex _re{"(.*)%(arg[0-9]+)%([-/_]*)(.*)"};
        while(std::regex_search(_fpath, _re))
            _fpath = std::regex_replace(_fpath, _re, "$1$4");
    } catch(std::exception& _e)
    {
        LOG(WARNING) << "[rocprofiler] " << __FUNCTION__ << " threw an exception :: " << _e.what()
                     << "\n";
    }

    return _fpath;
}

std::string
compose_filename(const config& _cfg)
{
    auto _output_path = _cfg.output_path;
    auto _output_file = _cfg.output_file;
    auto _output_ext  = _cfg.output_ext;

    if(_output_path.empty()) _output_path = ".";
    if(_cfg.mpi_size > 0)
    {
        if(_cfg.mpi_rank >= 0)
        {
            _output_file = fmt::format("{}.{}", _output_file, _cfg.mpi_rank);
        }
        else
        {
            _output_file = fmt::format("{}.{}", _output_file, getpid());
        }
    }
    if(!_output_ext.empty())
    {
        if(_output_ext.find('.') == std::string::npos) _output_ext.insert(0, ".");
        if(_output_file.length() < _output_ext.length() ||
           _output_file.find(_output_ext) != _output_file.length() - _output_ext.length())
            _output_file += _output_ext;
    }

    // join <OUTPUT_PATH>/<OUTPUT_FILE> and replace any keys with values
    auto _prefix = format(std::filesystem::path{_output_path} / _output_file);

    // return on empty
    if(_prefix.empty()) return std::string{};

    // get the absolute path
    auto _fname = std::filesystem::absolute(std::filesystem::path{_prefix});

    // create the directory if necessary
    auto _fname_path = _fname.parent_path();
    if(!std::filesystem::exists(_fname_path))
        std::filesystem::create_directories(_fname.parent_path());

    return _fname.string();
}

std::string
format_name(std::string_view _name, const config& _cfg)
{
    if(_cfg.demangle && _cfg.truncate)
    {
        return truncate_name(cxx_demangle(_name));
    }

    if(_cfg.demangle)
    {
        return cxx_demangle(_name);
    }

    if(_cfg.truncate)
    {
        return truncate_name(_name);
    }

    return std::string{_name};
}

void
initialize()
{
    (void) get_config<config_context::global>();
}

output_key::output_key(std::string _key, std::string _val, std::string _desc)
: key{std::move(_key)}
, value{std::move(_val)}
, description{std::move(_desc)}
{}
}  // namespace common
}  // namespace rocprofiler
