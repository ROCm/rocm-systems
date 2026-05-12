// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-causal.hpp"
#include "common/common_utils.hpp"

#include <timemory/log/macros.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
#include <unistd.h>

namespace utils = rocprofsys::common_utils;

namespace
{
std::vector<std::string>
to_string_vec(const std::vector<char*>& argv)
{
    std::vector<std::string> out;
    out.reserve(argv.size());
    for(const auto* arg : argv)
        if(arg != nullptr) out.emplace_back(arg);
    return out;
}
}  // namespace

int
main(int argc, char** argv)
{
    auto _base_env   = get_initial_environment();
    auto _causal_env = std::vector<std::map<std::string_view, std::string>>{};

    bool _has_double_hyphen = false;
    for(int arg_idx = 1; arg_idx < argc; ++arg_idx)
    {
        auto _arg = std::string_view{ argv[arg_idx] };
        if(_arg == "--" || _arg == "-?" || _arg == "-h" || _arg == "--help" ||
           _arg == "--version")
            _has_double_hyphen = true;
    }

    std::vector<char*> _argv = {};
    if(_has_double_hyphen)
    {
        _argv = parse_args(argc, argv, _base_env, _causal_env);
    }
    else
    {
        _argv.reserve(argc);
        for(int arg_idx = 1; arg_idx < argc; ++arg_idx)
            _argv.emplace_back(argv[arg_idx]);
        _causal_env.resize(1);
    }

    prepare_command_for_run(argv[0], _argv);
    prepare_environment_for_run(_base_env);

    if(get_verbose() >= 3)
    {
        TIMEMORY_PRINTF_INFO(stderr, "causal environments to be executed:\n");
        size_t _n = 0;
        for(auto& citr : _causal_env)
        {
            auto _env = _base_env;
            for(const auto& eitr : citr)
                update_env(_env, eitr.first, eitr.second);
            auto _prefix = std::to_string(_n++) + ":  ";
            utils::print_environment(_env, get_updated_envs(), true, _prefix);
        }
    }

    if(!_argv.empty())
    {
        if(_causal_env.size() == 1)
        {
            auto _env = _base_env;
            for(const auto& eitr : _causal_env.front())
                update_env(_env, eitr.first, eitr.second);
            auto _verbose = get_verbose();
            if(_verbose >= 0)
                utils::print_environment(_env, get_updated_envs(), _verbose >= 1, "0: ");
            if(_verbose >= 1) utils::print_command(to_string_vec(_argv), "0: ");
            _argv.emplace_back(nullptr);
            auto envp_ptrs = utils::to_c_argv(_env);
            return execvpe(_argv.front(), _argv.data(), envp_ptrs.data());
        }

        forward_signals({ SIGINT, SIGTERM, SIGQUIT });
        size_t _ncount = 0;
        size_t _width  = std::log10(_causal_env.size()) + 1;
        for(auto& citr : _causal_env)
        {
            auto _n        = _ncount++;
            auto _main_pid = getpid();
            auto _pid      = fork();

            if(get_verbose() >= 3)
            {
                TIMEMORY_PRINTF_INFO(stderr, "process %i returned %i from fork...\n",
                                     getpid(), _pid);
            }

            if(_pid == 0)
            {
                auto _prefix = std::stringstream{};
                _prefix << std::setw(_width) << std::right << _n << "/"
                        << std::setw(_width) << std::left << _causal_env.size() << ": ["
                        << _main_pid << " -> " << getpid() << "] ";

                auto _env = _base_env;
                for(const auto& eitr : citr)
                    update_env(_env, eitr.first, eitr.second);
                auto _verbose = get_verbose();
                if(_verbose >= 0)
                    utils::print_environment(_env, get_updated_envs(), _verbose >= 1,
                                             _prefix.str());
                if(_verbose >= 1)
                    utils::print_command(to_string_vec(_argv), _prefix.str());
                _argv.emplace_back(nullptr);
                auto envp_ptrs = utils::to_c_argv(_env);
                return execvpe(_argv.front(), _argv.data(), envp_ptrs.data());
            }
            else
            {
                add_child_pid(_pid);
                auto _status = wait_pid(_pid);
                auto _ret    = diagnose_status(_pid, _status);
                remove_child_pid(_pid);
                if(_ret != 0) return _ret;
            }
        }
    }
}
