// MIT License
//
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

#include "config.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>

#include <linux/limits.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace
{
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
    get_local_datetime(get_env("ROCP_TIME_FORMAT", "%F_%H.%M"), launch_time);
const auto env_regexes =
    new std::array<std::regex, 3>{std::regex{"(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)"},
                                  std::regex{"(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)"},
                                  std::regex{"(.*)%q\\{([A-Z0-9_]+)\\}(.*)"}};
// env regex examples:
//  - %env{USER}%       Consistent with other output key formats (start+end with %)
//  - $ENV{USER}        Similar to CMake
//  - %q{USER}          Compatibility with NVIDIA
//

std::string*
get_local_datetime(const std::string& dt_format, std::time_t*& _dt_curr)
{
    constexpr auto strsize = 512;

    if(!_dt_curr) _dt_curr = new std::time_t{std::time_t{std::time(nullptr)}};

    char mbstr[strsize] = {};
    memset(mbstr, '\0', sizeof(mbstr) * sizeof(char));

    if(std::strftime(mbstr, sizeof(mbstr) - 1, dt_format.c_str(), std::localtime(_dt_curr)) != 0)
        return new std::string{mbstr};

    return nullptr;
}

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

// replace unsuported specail chars with space
void
handle_special_chars(std::string& str)
{
    // Iterate over the string and replace any special characters with a space.
    auto pos = std::string::npos;
    while((pos = str.find_first_of("!@#$%&(),*+-./;<>?@{}^`~|")) != std::string::npos)
        str.at(pos) = ' ';
}

bool
has_counter_format(std::string const& str)
{
    return std::find_if(str.begin(), str.end(), [](unsigned char ch) {
               return (isalnum(ch) != 0 || ch == '_');
           }) != str.end();
}

// validate kernel names
std::unordered_set<uint32_t>
get_kernel_filter_range(const std::string& kernel_filter)
{
    if(kernel_filter.empty()) return {};

    auto delim     = rocprofiler::sdk::parse::tokenize(kernel_filter, "[], ");
    auto range_set = std::unordered_set<uint32_t>{};
    for(const auto& itr : delim)
    {
        if(itr.find('-') != std::string::npos)
        {
            auto drange = rocprofiler::sdk::parse::tokenize(itr, "- ");

            ROCP_FATAL_IF(drange.size() != 2)
                << "bad range format for '" << itr << "'. Expected [A-B] where A and B are numbers";

            uint32_t start_range = std::stoul(drange.front());
            uint32_t end_range   = std::stoul(drange.back());
            for(auto i = start_range; i <= end_range; i++)
                range_set.emplace(i);
        }
        else
        {
            ROCP_FATAL_IF(itr.find_first_not_of("0123456789") != std::string::npos)
                << "expected integer for " << itr << ". Non-integer value detected";
            range_set.emplace(std::stoul(itr));
        }
    }
    return range_set;
}

std::set<std::string>
parse_counters(std::string line)
{
    auto counters = std::set<std::string>{};

    if(line.empty()) return counters;

    // strip the comment
    if(auto pos = std::string::npos; (pos = line.find('#')) != std::string::npos)
        line = line.substr(0, pos);

    // trim line for any white spaces after comment strip
    trim(line);

    // check to see if comment stripping + trim resulted in empty line
    if(line.empty()) return counters;

    constexpr auto pmc_qualifier = std::string_view{"pmc:"};
    auto           pos           = std::string::npos;

    // should we handle an "pmc:" not being present? Seems like it should be a fatal error
    if((pos = line.find(pmc_qualifier)) != std::string::npos)
    {
        // strip out pmc qualifier
        line = line.substr(pos + pmc_qualifier.length());

        handle_special_chars(line);

        auto input_ss = std::stringstream{line};
        while(true)
        {
            auto counter = std::string{};
            input_ss >> counter;
            if(counter.empty())
                break;
            else if(counter != pmc_qualifier && has_counter_format(counter))
                counters.emplace(counter);
        }
    }

    return counters;
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

config::config()
: kernel_filter_range{get_kernel_filter_range(
      get_env("ROCPROF_KERNEL_FILTER_RANGE", std::string{}))}
, counters{parse_counters(get_env("ROCPROF_COUNTERS", std::string{}))}
{
    auto to_upper = [](std::string val) {
        for(auto& vitr : val)
            vitr = toupper(vitr);
        return val;
    };

    auto output_format = get_env("ROCPROF_OUTPUT_FORMAT", "CSV");
    auto entries       = std::set<std::string>{};
    for(const auto& itr : sdk::parse::tokenize(output_format, " \t,;:"))
        entries.emplace(to_upper(itr));

    csv_output     = entries.count("CSV") > 0 || entries.empty();
    json_output    = entries.count("JSON") > 0;
    pftrace_output = entries.count("PFTRACE") > 0;
    otf2_output    = entries.count("OTF2") > 0;

    const auto supported_formats = std::set<std::string_view>{"CSV", "JSON", "PFTRACE", "OTF2"};
    for(const auto& itr : entries)
    {
        LOG_IF(FATAL, supported_formats.count(itr) == 0)
            << "Unsupported output format type: " << itr;
    }
    if(kernel_filter_include.empty()) kernel_filter_include = std::string(".*");

    const auto supported_perfetto_backends = std::set<std::string_view>{"inprocess", "system"};
    LOG_IF(FATAL, supported_perfetto_backends.count(perfetto_backend) == 0)
        << "Unsupported perfetto backend type: " << perfetto_backend;

    if(stats_summary_unit == "sec")
        stats_summary_unit_value = common::units::sec;
    else if(stats_summary_unit == "msec")
        stats_summary_unit_value = common::units::msec;
    else if(stats_summary_unit == "usec")
        stats_summary_unit_value = common::units::usec;
    else if(stats_summary_unit == "nsec")
        stats_summary_unit_value = common::units::nsec;
    else
    {
        ROCP_FATAL << "Unsupported summary units value: " << stats_summary_unit;
    }

    if(auto _summary_grps = get_env("ROCPROF_STATS_SUMMARY_GROUPS", ""); !_summary_grps.empty())
    {
        stats_summary_groups =
            sdk::parse::tokenize(_summary_grps, std::vector<std::string_view>{"##@@##"});

        // remove any empty strings (just in case these slipped through)
        stats_summary_groups.erase(std::remove_if(stats_summary_groups.begin(),
                                                  stats_summary_groups.end(),
                                                  [](const auto& itr) { return itr.empty(); }),
                                   stats_summary_groups.end());
    }

    // enable summary output if any of these are enabled
    summary_output = (stats_summary || stats_summary_per_domain || !stats_summary_groups.empty());
}

std::vector<output_key>
output_keys(std::string _tag)
{
    using strpair_t = std::pair<std::string, std::string>;

    auto _cmdline = common::read_command_line(getpid());

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

    auto _mpi_size = get_mpi_size();
    auto _mpi_rank = get_mpi_rank();

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

    auto _launch_time = (launch_datetime) ? *launch_datetime : std::string{".UNKNOWN_LAUNCH_TIME."};
    auto _hostname    = get_hostname();

    for(auto&& itr : std::initializer_list<output_key>{
            {"%hostname%", _hostname, "Network hostname"},
            {"%pid%", _proc_id, "Process identifier"},
            {"%ppid%", _parent_id, "Parent process identifier"},
            {"%pgid%", _pgroup_id, "Process group identifier"},
            {"%psid%", _session_id, "Process session identifier"},
            {"%psize%", _proc_size, "Number of sibling process"},
            {"%job%", _slurm_job_id, "SLURM_JOB_ID env variable"},
            {"%rank%", _slurm_proc_id, "MPI/UPC++ rank"},
            {"%size%", _dmp_size, "MPI/UPC++ size"},
            {"%nid%", _uniq_id, "%rank% if possible, otherwise %pid%"},
            {"%launch_time%", _launch_time, "Data and/or time of run according to time format"},
        })
    {
        _options.emplace_back(itr);
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

namespace
{
std::string
format_impl(std::string _fpath, const std::vector<output_key>& _keys)
{
    if(_fpath.find('%') == std::string::npos && _fpath.find('$') == std::string::npos)
        return _fpath;

    auto _replace = [](auto& _v, const output_key& pitr) {
        auto pos = std::string::npos;
        while((pos = _v.find(pitr.key)) != std::string::npos)
            _v.replace(pos, pitr.key.length(), pitr.value);
    };

    for(auto&& itr : _keys)
        _replace(_fpath, itr);

    // environment and configuration variables
    try
    {
        auto strip_leading_and_replace =
            [](std::string_view inp_v, std::initializer_list<char> keys, const char* val) {
                auto inp = std::string{inp_v};
                for(auto key : keys)
                {
                    auto pos = std::string::npos;
                    while((pos = inp.find(key)) == 0)
                        inp = inp.substr(pos + 1);

                    while((pos = inp.find(key)) != std::string::npos)
                        inp = inp.replace(pos, 1, val);
                }
                return inp;
            };

        for(const auto& _re : *env_regexes)
        {
            while(std::regex_search(_fpath, _re))
            {
                auto        _var = std::regex_replace(_fpath, _re, "$3");
                std::string _val = get_env<std::string>(_var, "");
                _val             = strip_leading_and_replace(_val, {'\t', ' ', '/'}, "_");
                auto _beg        = std::regex_replace(_fpath, _re, "$1");
                auto _end        = std::regex_replace(_fpath, _re, "$4");
                _fpath           = fmt::format("{}{}{}", _beg, _val, _end);
            }
        }
    } catch(std::exception& _e)
    {
        ROCP_WARNING << "[rocprofiler] " << __FUNCTION__ << " threw an exception :: " << _e.what()
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
        ROCP_WARNING << "[rocprofiler] " << __FUNCTION__ << " threw an exception :: " << _e.what()
                     << "\n";
    }

    return _fpath;
}

std::string
format(std::string _fpath, const std::vector<output_key>& _keys)
{
    if(_fpath.find('%') == std::string::npos && _fpath.find('$') == std::string::npos)
        return _fpath;

    auto _ref = _fpath;
    _fpath    = format_impl(std::move(_fpath), _keys);

    return (_fpath == _ref) ? _fpath : format(std::move(_fpath), _keys);
}
}  // namespace

std::string
format(std::string _fpath, const std::string& _tag)
{
    return format(std::move(_fpath), output_keys(_tag));
}

std::string
format_name(std::string_view _name, const config& _cfg)
{
    if(!_cfg.demangle && !_cfg.truncate) return std::string{_name};

    // truncating requires demangling first so always demangle
    auto _demangled_name =
        common::cxx_demangle(std::regex_replace(_name.data(), std::regex{"(\\.kd)$"}, ""));

    if(_cfg.truncate) return common::truncate_name(_demangled_name);

    return _demangled_name;
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
}  // namespace tool
}  // namespace rocprofiler
