// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-sample.hpp"
#include "common/output.hpp"

#include <string_view>
#include <unistd.h>

namespace output = rocprofsys::common::output;

int
main(int argc, char** argv)
{
    auto _env = get_initial_environment();

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
        _argv = parse_args(argc, argv, _env);
    }
    else
    {
        _argv.reserve(argc);
        for(int i = 1; i < argc; ++i)
            _argv.emplace_back(argv[i]);
    }

    add_torch_library_path(_env, _argv);

    auto _verbose = get_verbose_level();
    if(_verbose >= 0) output::print_environment(_env, get_updated_envs(), _verbose >= 1);

    if(!_argv.empty())
    {
        if(_verbose >= 1) output::print_command(_argv);
        _argv.emplace_back(nullptr);
        _env.emplace_back(nullptr);

        return execvpe(_argv.front(), _argv.data(), _env.data());
    }
}
