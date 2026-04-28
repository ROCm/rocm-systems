// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_translator.h"

#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"

using namespace rocm_compute;

code_object_translator_impl_t::code_object_translator_impl_t()
    : m_translator(std::make_unique<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>())
{
}

void code_object_translator_impl_t::add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size)
{
}

void code_object_translator_impl_t::add_code_object(const void* data,
                                               size_t      memory_size,
                                               size_t      id,
                                               uint64_t    load_addr,
                                               uint64_t    mem_size)
{
}
