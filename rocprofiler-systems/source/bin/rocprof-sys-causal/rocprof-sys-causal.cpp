// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "rocprof-sys-causal.hpp"

#include <timemory/log/macros.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
#include <unistd.h>

int
main(int argc, char** argv)
{
    auto _base_env   = get_initial_environment();
    auto _causal_env = std::vector<std::map<std::string_view, std::string>>{};

    bool _has_double_hyphen = false;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string_view{ argv[i] };
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
        for(int i = 1; i < argc; ++i)
            _argv.emplace_back(argv[i]);
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
            print_updated_environment(_env, _prefix);
        }
    }

    if(!_argv.empty())
    {
        if(_causal_env.size() == 1)
        {
            auto _env = _base_env;
            for(const auto& eitr : _causal_env.front())
                update_env(_env, eitr.first, eitr.second);
            print_updated_environment(_env, "0: ");
            print_command(_argv, "0: ");
            _argv.emplace_back(nullptr);
            _env.emplace_back(nullptr);
            return execvpe(_argv.front(), _argv.data(), _env.data());
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
                print_updated_environment(_env, _prefix.str());
                print_command(_argv, _prefix.str());
                _argv.emplace_back(nullptr);
                _env.emplace_back(nullptr);
                return execvpe(_argv.front(), _argv.data(), _env.data());
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
