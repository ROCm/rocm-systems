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

#pragma once

#include <string>
#include <string_view>
#include <vector>

enum update_mode : int
{
    UPD_REPLACE = 0x1,
    UPD_PREPEND = 0x2,
    UPD_APPEND  = 0x3,
    UPD_WEAK    = 0x4,
};

std::string
get_realpath(const std::string& _fpath);

void
print_command(const std::vector<char*>& _argv);

void
print_updated_environment(std::vector<char*> _env);

std::vector<char*>
get_initial_environment();

std::string
get_internal_libpath(const std::string& _lib);

std::string
get_internal_script_path(void);

template <typename Tp>
void
update_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val,
           update_mode&& _mode = UPD_REPLACE, std::string_view _join_delim = ":");

void
remove_env(std::vector<char*>& _environ, std::string_view _env_var);

std::vector<char*>
parse_args(int argc, char** argv, std::vector<char*>& envp);
