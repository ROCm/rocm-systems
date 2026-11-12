// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "code_object_translator.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

class TestCodeObjectTranslator : public ::testing::Test
{
protected:
    static constexpr uint64_t kLoadAddress        = 0x10000000;
    static constexpr uint64_t kTextVirtualAddress = 0x1000;
    static constexpr uint64_t kTextFileOffset     = 0x200;
    static constexpr size_t   kInstructionSize    = sizeof(uint32_t);

    static std::vector<char> make_code_object(const std::vector<uint32_t>& instructions);

    rocprofiler_compute_tool::code_object_translator_impl_t m_translator;
};
