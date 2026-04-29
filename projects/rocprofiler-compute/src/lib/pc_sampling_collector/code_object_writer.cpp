// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_writer.h"
using namespace rocm_compute;

void code_object_writer_json_t::start_code_obj(size_t obj_id) {}

void code_object_writer_json_t::end_code_obj_desc(size_t obj_id) {}

void code_object_writer_json_t::start_symbol(const symbol_t& symbol) {}

void code_object_writer_json_t::end_symbol() {}

void code_object_writer_json_t::write_instruction(const instruction_t& inst) {}

std::string code_object_writer_json_t::get_result()
{
    return {};
}
