// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "argparse.hpp"
#include "flag_descriptor.hpp"

namespace rocprofsys
{
namespace argparse
{

void
register_flag(parser_t& parser, parser_data& data, const flag_descriptor& descriptor);

void
register_group(parser_t& parser, parser_data& data, const flag_group& group);

}  // namespace argparse
}  // namespace rocprofsys
