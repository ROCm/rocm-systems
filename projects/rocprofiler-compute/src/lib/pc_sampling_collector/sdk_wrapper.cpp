// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "sdk_wrapper.h"

#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"

using namespace rocprofiler_compute_tool;

sdk_wrapper_t::ptr sdk_wrapper_t::create()
{
    return std::make_shared<sdk_wrapper_impl_t>();
}

std::string_view sdk_wrapper_impl_t::source_frame_separator() const
{
    return rocprofiler::sdk::codeobj::disassembly::Instruction::separator;
}
