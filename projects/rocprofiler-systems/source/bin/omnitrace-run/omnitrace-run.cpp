// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "omnitrace-run.hpp"

#include <timemory/log/color.hpp>
#include <timemory/log/macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
#include <unistd.h>

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
                  << " <OPTIONS> -- <COMMAND> <ARGS>" << std::endl;
    };

    if(argc == 1)
    {
        _print_usage();
        return EXIT_FAILURE;
    }

    auto _parse_data = parser_data_t{};
    parse_args(argc, argv, _parse_data);
    prepare_command_for_run(argv[0], _parse_data);
    prepare_environment_for_run(_parse_data);

    auto& _argv = _parse_data.command;
    auto& _envp = _parse_data.current;
    if(!_argv.empty())
    {
        print_updated_environment(_parse_data, "OMNITRACE: ");
        print_command(_parse_data, "OMNITRACE: ");
        _argv.emplace_back(nullptr);
        _envp.emplace_back(nullptr);
        return execvpe(_argv.front(), _argv.data(), _envp.data());
    }

    _print_usage();
    return EXIT_FAILURE;
}
