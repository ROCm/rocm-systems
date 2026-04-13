// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-run.hpp"

#include "common/common_utils.hpp"
#include "common/environment.hpp"
#include "common/output.hpp"
#include "core/mproc.hpp"

#include <timemory/environment.hpp>
#include <timemory/log/color.hpp>
#include <timemory/log/macros.hpp>

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <unistd.h>

namespace output = rocprofsys::common::output;

namespace
{
auto* _getenv_at_load = getenv("TIMEMORY_LIBRARY_CTOR");
auto  _setenv_at_load = setenv("TIMEMORY_LIBRARY_CTOR", "0", 0);
}  // namespace

int
main(int argc, char** argv)
{
    if(!_getenv_at_load)
        unsetenv("TIMEMORY_LIBRARY_CTOR");
    else
        setenv("TIMEMORY_LIBRARY_CTOR", _getenv_at_load, 1);

    auto _print_usage = [argv]() {
        std::cerr << tim::log::color::fatal() << "Usage: " << argv[0]
                  << " <OPTIONS> -- <COMMAND> <ARGS>" << tim::log::color::end()
                  << std::endl;
    };

    if(argc == 1)
    {
        _print_usage();
        return EXIT_FAILURE;
    }

    auto _parse_data = parser_data_t{};
    auto _fork_exec  = false;
    try
    {
        parse_args(argc, argv, _parse_data, _fork_exec);
    } catch(const rocprofsys::common_utils::cli_done& e)
    {
        return e.exit_code;
    }
    prepare_command_for_run(argv[0], _parse_data);
    prepare_environment_for_run(_parse_data);

    auto& _argv = _parse_data.command;
    auto& _envp = _parse_data.current;
    if(!_argv.empty())
    {
        auto _verbose = get_verbose(_parse_data);
        if(_verbose >= 0)
            output::print_environment(_parse_data.current, _parse_data.updated,
                                      _verbose >= 1, "ROCPROFSYS: ");
        if(_verbose >= 1) output::print_command(_parse_data.command, "ROCPROFSYS: ");
        _argv.emplace_back(nullptr);
        _envp.emplace_back(nullptr);

        if(_fork_exec)
        {
            auto _pid = fork();

            if(_pid == 0)
            {
                return execvpe(_argv.front(), _argv.data(), _envp.data());
            }
            else
            {
                auto _status = rocprofsys::mproc::wait_pid(_pid);
                auto _ec     = rocprofsys::mproc::diagnose_status(_pid, _status);
                if(_ec != 0 && _parse_data.verbose >= 0)
                {
                    TIMEMORY_PRINTF_FATAL(
                        stderr, "process %i exiting with non-zero exit code: %i\n", _pid,
                        _ec);
                }
                else if(_parse_data.verbose >= 2)
                {
                    TIMEMORY_PRINTF_FATAL(
                        stderr,
                        "rocprof-sys run in process %i completed. exit code: %i\n", _pid,
                        _ec);
                }
                return _ec;
            }
        }
        else
        {
            return execvpe(_argv.front(), _argv.data(), _envp.data());
        }
    }

    _print_usage();
    return EXIT_FAILURE;
}
