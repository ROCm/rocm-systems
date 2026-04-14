// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-sample.hpp"
#include "common/output.hpp"

#include "common/common_utils.hpp"

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
           _arg == "--version" || _arg == "--export-config" ||
           _arg.find("--export-config=") == 0 || _arg == "--list-presets" ||
           _arg == "--explain" || _arg.find("--explain=") == 0)
            _has_double_hyphen = true;
    }

    std::vector<char*> _argv = {};
    if(_has_double_hyphen)
    {
        try
        {
            _argv = parse_args(argc, argv, _env);
        } catch(const rocprofsys::common_utils::cli_done& e)
        {
            return e.exit_code;
        }
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
