// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/argparse.hpp"

#include <csignal>
#include <sched.h>
#include <string_view>

using parser_data_t = rocprofsys::argparse::parser_data;

int
get_verbose(parser_data_t&);

void
prepare_command_for_run(char*, parser_data_t&);

void
prepare_environment_for_run(parser_data_t&);

parser_data_t&
parse_args(int argc, char** argv, parser_data_t&, bool&);

parser_data_t&
parse_command(int argc, char** argv, parser_data_t&);
