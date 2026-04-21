// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "argparse.hpp"
#include "argparse/core_flags.hpp"
#include "argparse/interpreter.hpp"
#include "common/environment.hpp"
#include "common/path.hpp"
#include "config.hpp"
#include "exception.hpp"
#include "gpu.hpp"
#include "state.hpp"
#include <cstdint>

#include <timemory/settings/types.hpp>
#include <timemory/utility/filepath.hpp>

#include "logger/debug.hpp"

#include <spdlog/fmt/ranges.h>

#include <cstdint>

namespace rocprofsys
{
namespace argparse
{
namespace
{
namespace filepath = ::tim::filepath;
namespace path     = rocprofsys::common::path;
using rocprofsys::common::remove_env;

auto
get_clock_id_choices()
{
    auto clock_name = [](std::string _v) {
        constexpr auto _clock_prefix = std::string_view{ "clock_" };
        for(auto& itr : _v)
            itr = tolower(itr);
        auto _pos = _v.find(_clock_prefix);
        if(_pos == 0) _v = _v.substr(_pos + _clock_prefix.length());
        if(_v == "process_cputime_id") _v = "cputime";
        return _v;
    };

#define ROCPROFSYS_CLOCK_IDENTIFIER(VAL)                                                 \
    std::make_tuple(clock_name(#VAL), VAL, std::string_view{ #VAL })

    auto _choices = strvec_t{};
    auto _aliases = std::map<std::string, strvec_t>{};
    for(auto itr : { ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_REALTIME),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_PROCESS_CPUTIME_ID),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC_RAW),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_REALTIME_COARSE),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC_COARSE),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_BOOTTIME) })
    {
        auto _choice = std::to_string(std::get<1>(itr));
        _choices.emplace_back(_choice);
        _aliases[_choice] = { std::get<0>(itr), std::string{ std::get<2>(itr) } };
    }

#undef ROCPROFSYS_CLOCK_IDENTIFIER

    return std::make_pair(_choices, _aliases);
}

using rocprofsys::common::update_mode;

template <typename Tp>
void
update_env(parser_data& _data, std::string_view _env_var, Tp&& _env_val,
           update_mode _mode = update_mode::REPLACE, std::string_view _join_delim = ":")
{
    rocprofsys::common::update_env(_data.env.current, _env_var,
                                   std::forward<Tp>(_env_val), _mode, _join_delim,
                                   _data.env.updated, _data.env.initial);
}

void
apply_papi_choice_filter(const std::shared_ptr<tim::vsettings>& setting)
{
    if(setting->get_name() != "papi_events") return;
    auto choices = setting->get_choices();
    choices.erase(
        std::remove_if(choices.begin(), choices.end(),
                       [](const auto& choice) {
                           return std::regex_search(
                                      choice, std::regex{ "[A-Za-z0-9]:([A-Za-z_]+)" }) ||
                                  std::regex_search(choice, std::regex{ "io:::" });
                       }),
        choices.end());
    choices.emplace_back("... run `rocprof-sys-avail -H -c CPU` for full list ...");
    setting->set_choices(choices);
}

void
sort_settings_by_name_length(std::vector<std::shared_ptr<tim::vsettings>>& settings)
{
    std::sort(settings.begin(), settings.end(), [](const auto& lhs, const auto& rhs) {
        const auto& lhs_name = lhs->get_name();
        const auto& rhs_name = rhs->get_name();
        if(lhs_name.length() > 4 && rhs_name.length() > 4 &&
           lhs_name.substr(0, 4) == rhs_name.substr(0, 4))
            return lhs_name < rhs_name;
        return lhs_name.length() < rhs_name.length();
    });
}

}  // namespace

bool
default_setting_filter(vsetting_t* _v, const parser_data& _data)
{
    return (_data.reg.processed_settings.count(_v) == 0 &&
            _data.reg.processed_environs.count(_v->get_name()) == 0 &&
            _data.reg.processed_environs.count(_v->get_env_name()) == 0);
}

bool
default_environ_filter(std::string_view _v, const parser_data& _data)
{
    return (_data.reg.processed_environs.count(_v.data()) == 0);
}

bool
default_grouping_filter(std::string_view _v, const parser_data& _data)
{
    return (_data.reg.processed_groups.count(_v.data()) == 0);
}

parser_data&
init_parser(parser_data& _data)
{
    tim::settings::suppress_config()  = true;
    tim::settings::suppress_parsing() = true;

    set_state(State::Init);
    config::configure_settings(false);

    _data.env.dl_libpath =
        path::realpath(path::get_internal_libpath("librocprof-sys-dl.so").c_str());
    _data.env.omni_libpath =
        path::realpath(path::get_internal_libpath("librocprof-sys.so").c_str());

    auto _libexecpath = path::realpath(path::get_internal_script_path());
    update_env(_data, "ROCPROFSYS_SCRIPT_PATH", _libexecpath, update_mode::REPLACE);

    auto _rootpath = path::realpath(path::get_rocprofsys_root());
    update_env(_data, "ROCPROFSYS_ROOT", _rootpath, update_mode::REPLACE);

    return _data;
}

parser_data&
add_ld_preload(parser_data& _data)
{
    update_env(_data, "LD_PRELOAD", _data.env.dl_libpath, update_mode::APPEND);
    return _data;
}

parser_data&
add_ld_library_path(parser_data& _data)
{
    auto _libdir = filepath::dirname(_data.env.dl_libpath);
    if(filepath::exists(_libdir))
        update_env(_data, "LD_LIBRARY_PATH", _libdir, update_mode::APPEND);
    return _data;
}

parser_data&
add_torch_library_path(parser_data& _data, bool verbose)
{
    if(_data.out.command.empty()) return _data;
    rocprofsys::common::add_torch_library_path(
        _data.env.current, _data.out.command.front(), verbose, _data.env.updated);
    return _data;
}

parser_data&
add_core_arguments(parser_t& _parser, parser_data& _data)
{
    register_group(_parser, _data, debug_group());

    add_group_arguments(_parser, "debugging", _data);
    add_group_arguments(_parser, "mode", _data, true);

    register_group(_parser, _data, general_group());

    strset_t _backend_choices = { "all",        "kokkosp", "mpip", "ompt",
                                  "rcclp",      "amd-smi", "rocm", "mutex-locks",
                                  "spin-locks", "rw-locks" };

#if(!defined(ROCPROFSYS_USE_MPI) || ROCPROFSYS_USE_MPI == 0) &&                          \
    (!defined(ROCPROFSYS_USE_MPI_HEADERS) || ROCPROFSYS_USE_MPI_HEADERS == 0)
    _backend_choices.erase("mpip");
#endif

#if !defined(ROCPROFSYS_USE_OMPT) || ROCPROFSYS_USE_OMPT == 0
    _backend_choices.erase("ompt");
#endif

    if(gpu::device_count() == 0)
    {
        // remove GPU-specific backends
        _backend_choices.erase("rcclp");
        _backend_choices.erase("amd-smi");
        _backend_choices.erase("rocm");

        update_env(_data, "ROCPROFSYS_USE_AMD_SMI", false);
    }

    _parser.start_group("BACKEND OPTIONS",
                        "These options control region information captured "
                        "w/o sampling or instrumentation");

    if(_data.reg.environ_filter("include", _data))
    {
        _parser.add_argument({ "-I", "--include" }, "Include data from these backends")
            .min_count(1)
            .max_count(_backend_choices.size())
            .dtype("[backend...]")
            .choices(_backend_choices)
            .action([&](parser_t& p) {
                auto _v      = p.get<strset_t>("include");
                auto _update = [&](const auto& _opt, bool _cond) {
                    if(_cond || _v.count("all") > 0) update_env(_data, _opt, true);
                };
                _update("ROCPROFSYS_USE_KOKKOSP", _v.count("kokkosp") > 0);
                _update("ROCPROFSYS_USE_MPIP", _v.count("mpip") > 0);
                _update("ROCPROFSYS_USE_OMPT", _v.count("ompt") > 0);
                _update("ROCPROFSYS_USE_RCCLP", _v.count("rcclp") > 0);
                _update("ROCPROFSYS_USE_AMD_SMI", _v.count("amd-smi") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_LOCKS", _v.count("mutex-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_RW_LOCKS", _v.count("rw-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS", _v.count("spin-locks") > 0);

                if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                    update_env(_data, "KOKKOS_TOOLS_LIBS", _data.env.omni_libpath,
                               update_mode::PREPEND);
            });

        _data.reg.processed_environs.emplace("include");
    }

    if(_data.reg.environ_filter("exclude", _data))
    {
        _parser.add_argument({ "-E", "--exclude" }, "Exclude data from these backends")
            .min_count(1)
            .max_count(_backend_choices.size())
            .dtype("[backend...]")
            .choices(_backend_choices)
            .action([&](parser_t& p) {
                auto _v      = p.get<strset_t>("exclude");
                auto _update = [&](const auto& _opt, bool _cond) {
                    if(_cond || _v.count("all") > 0) update_env(_data, _opt, false);
                };
                _update("ROCPROFSYS_USE_KOKKOSP", _v.count("kokkosp") > 0);
                _update("ROCPROFSYS_USE_MPIP", _v.count("mpip") > 0);
                _update("ROCPROFSYS_USE_OMPT", _v.count("ompt") > 0);
                _update("ROCPROFSYS_USE_RCCLP", _v.count("rcclp") > 0);
                _update("ROCPROFSYS_USE_AMD_SMI", _v.count("amd-smi") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_LOCKS", _v.count("mutex-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_RW_LOCKS", _v.count("rw-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS", _v.count("spin-locks") > 0);

                if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                    remove_env(_data.env.current, "KOKKOS_TOOLS_LIBS", _data.env.initial);
            });

        _data.reg.processed_environs.emplace("exclude");
    }

    add_group_arguments(_parser, "backend", _data);
    add_group_arguments(_parser, "parallelism", _data, true);

    if(_data.reg.environ_filter("launcher", _data))
    {
        _parser
            .add_argument(
                { "-l", "--launcher" },
                "When running MPI jobs, typically the associated '--' for this "
                "executable should be right before the target executable, e.g. `mpirun "
                "-n 2 <THIS_EXE> -- <TARGET_EXE> <TARGET_EXE_ARGS...>`. This options "
                "enables prefixing the entire command (i.e. before `mpirun`, `srun`, "
                "etc.). Pass the name of the target executable (or a regex for matching "
                "to the name of the target) as the argument to this option and this "
                "executable will insert itself a second time in the appropriate "
                "location, e.g. `<THIS_EXE> --launcher sleep -- mpirun -n 2 sleep 10` is "
                "equivalent to `mpirun -n 2 <THIS_EXE> -- sleep 10`")
            .count(1)
            .dtype("target-exe")
            .action([&](parser_t& p) {
                _data.out.launcher = p.get<std::string>("launcher");
            });

        _data.reg.processed_environs.emplace("launcher");
    }

    register_group(_parser, _data, tracing_group());

    if(_data.reg.environ_filter("trace_clock_id", _data))
    {
        auto _clock_id_choices = get_clock_id_choices();
        _parser
            .add_argument(
                { "--trace-clock-id" },
                "Set the default clock ID for for trace delay/duration. Note: "
                "\"cputime\" is "
                "the *process* CPU time and might need to be scaled based on the number "
                "of "
                "threads, i.e. 4 seconds of CPU-time for an application with 4 fully "
                "active "
                "threads would equate to ~1 second of realtime. If this proves to be "
                "difficult to handle in practice, please file a feature request for "
                "rocprof-sys to auto-scale based on the number of threads.")
            .count(1)
            .dtype("clock-id")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID",
                           p.get<double>("trace-clock-id"));
            })
            .choices(_clock_id_choices.first)
            .choice_aliases(_clock_id_choices.second);

        _data.reg.processed_environs.emplace("trace_clock_id");
        _data.reg.processed_environs.emplace("trace_period_clock_id");
    }

    register_group(_parser, _data, profile_group());
    register_group(_parser, _data, process_sampling_group());
    register_group(_parser, _data, general_sampling_group());
    register_group(_parser, _data, sampling_timer_group());

    _parser.start_group(
        "ADVANCED SAMPLING OPTIONS",
        "These options determine the heuristic for deciding when to take a sample");

    add_group_arguments(_parser, "sampling", _data);

    register_group(_parser, _data, hw_counter_group());

    add_group_arguments(_parser, "category", _data, true);
    add_group_arguments(_parser, "io", _data, true);
    add_group_arguments(_parser, "perfetto", _data, true);
    add_group_arguments(_parser, "timemory", _data, true);
    add_group_arguments(_parser, "rocm", _data, true);

    register_group(_parser, _data, misc_group());

    _parser.end_group();

    return _data;
}

parser_data&
add_group_arguments(parser_t& _parser, const std::string& _group_name, parser_data& _data,
                    bool _add_group)
{
    if(!_data.reg.grouping_filter(_group_name, _data)) return _data;

    auto _get_name = [](const std::shared_ptr<tim::vsettings>& itr) {
        auto _name = itr->get_name();
        auto _pos  = std::string::npos;
        while((_pos = _name.find('_')) != std::string::npos)
            _name = _name.replace(_pos, 1, "-");
        return _name;
    };

    auto _add_option = [&_parser, &_data](const std::string&                     _name,
                                          const std::shared_ptr<tim::vsettings>& itr) {
        if(!_data.reg.setting_filter(itr.get(), _data)) return false;

        if(_name.empty())
            throw exception<std::runtime_error>("Error! empty name for " +
                                                itr->get_name());

        _data.reg.processed_settings.emplace(itr.get());

        auto _opt_name = std::string{ "--" } + _name;
        itr->set_command_line({ _opt_name });
        auto* _arg = static_cast<parser_t::argument*>(itr->add_argument(_parser));
        if(_arg)
        {
            _arg->action([&_data, itr, _name](parser_t& p) {
                auto _value = fmt::format("{}", fmt::join(p.get<strvec_t>(_name), " "));
                if(_value.empty()) _value = p.get<std::string>(_name);
                if(_value.empty()) _value = fmt::format("{}", p.get<bool>(_name));
                if(_value.empty())
                    throw exception<std::runtime_error>("Error! no value for " + _name);
                update_env(_data, itr->get_env_name(), _value);
            });
        }
        else
        {
            LOG_WARNING("Option {} ({}) is not enabled", _name, itr->get_env_name());
            _parser.add_argument({ _opt_name }, itr->get_description())
                .action([&](parser_t& p) {
                    auto _value =
                        fmt::format("{}", fmt::join(p.get<strvec_t>(_name), " "));
                    if(_value.empty())
                        throw exception<std::runtime_error>("Error! no value for " +
                                                            _name);
                    update_env(_data, itr->get_env_name(), _value);
                });
        }
        return true;
    };

    auto _settings = std::vector<std::shared_ptr<tim::vsettings>>{};
    for(auto& itr : *rocprofsys::settings::instance())
    {
        if(itr.second->get_categories().count("rocprofsys") == 0) continue;
        if(itr.second->get_categories().count("deprecated") > 0) continue;
        if(itr.second->get_hidden()) continue;
        if(!_data.reg.setting_filter(itr.second.get(), _data)) continue;
        if(!_data.reg.environ_filter(itr.second->get_name(), _data)) continue;
        if(itr.second->get_categories().count(_group_name) == 0) continue;

        itr.second->set_enabled(true);
        _settings.emplace_back(itr.second);
        apply_papi_choice_filter(itr.second);
    }

    sort_settings_by_name_length(_settings);

    if(_add_group)
    {
        auto _group_label = _group_name;
        for(auto& c : _group_label)
            c = toupper(c);
        _parser.start_group(_group_label);
    }

    for(const auto& itr : _settings)
    {
        _add_option(_get_name(itr), itr);
    }

    if(_add_group) _parser.end_group();

    return _data;
}

parser_data&
add_extended_arguments(parser_t& _parser, parser_data& _data)
{
    auto _category_count_map = std::unordered_map<std::string, std::uint32_t>{};
    auto _settings           = std::vector<std::shared_ptr<tim::vsettings>>{};
    for(auto& itr : *rocprofsys::settings::instance())
    {
        if(itr.second->get_categories().count("rocprofsys") == 0) continue;
        if(itr.second->get_categories().count("deprecated") > 0) continue;
        if(itr.second->get_hidden()) continue;
        if(!_data.reg.setting_filter(itr.second.get(), _data)) continue;
        if(!_data.reg.environ_filter(itr.second->get_name(), _data)) continue;

        itr.second->set_enabled(true);
        _settings.emplace_back(itr.second);
        apply_papi_choice_filter(itr.second);

        for(const auto& citr : itr.second->get_categories())
        {
            if(std::regex_search(citr, std::regex{ "rocprofsys|timemory|^("
                                                   "native|custom|advanced|analysis)$" }))
                continue;
            _category_count_map[citr] += 1;
        }
    }

    auto _category_count_vec = strvec_t{};
    for(const auto& itr : _category_count_map)
        _category_count_vec.emplace_back(itr.first);

    std::sort(_category_count_vec.begin(), _category_count_vec.end(),
              [&_category_count_map](const auto& _lhs, const auto& _rhs) {
                  auto _lhs_v = _category_count_map.at(_lhs);
                  auto _rhs_v = _category_count_map.at(_rhs);
                  if(_lhs_v == _rhs_v) return _lhs < _rhs;
                  return _lhs_v > _rhs_v;
              });

    auto _groups =
        std::unordered_map<std::string, std::vector<std::shared_ptr<tim::vsettings>>>{};
    for(const auto& citr : _category_count_vec)
    {
        _groups[citr] = {};
        for(const auto& itr : _settings)
        {
            if(itr->get_categories().count(citr) > 0) _groups[citr].emplace_back(itr);
        }
        _settings.erase(std::remove_if(_settings.begin(), _settings.end(),
                                       [&citr](const auto& itr) {
                                           return itr->get_categories().count(citr) > 0;
                                       }),
                        _settings.end());
    }

    for(const auto& citr : _category_count_vec)
    {
        auto _group = _groups.at(citr);
        if(_group.empty()) continue;

        add_group_arguments(_parser, citr, _data, true);
    }

    return _data;
}
}  // namespace argparse
}  // namespace rocprofsys
