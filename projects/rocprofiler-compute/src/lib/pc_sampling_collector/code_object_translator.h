// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <memory>
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"

namespace rocm_compute
{
class code_object_translator_t
{
public:
    virtual ~code_object_translator_t() = default;
    virtual void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size) = 0;
    virtual void add_code_object(uint64_t data,
                                 size_t   memory_size,
                                 size_t   id,
                                 uint64_t load_addr,
                                 uint64_t mem_size) = 0;
    virtual std::vector<size_t> get_code_object_ids() const = 0;

};

class code_object_translator_impl_t : public code_object_translator_t
{
public:
    code_object_translator_impl_t();
    void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size) override;
    void add_code_object(uint64_t memory_base,
                         size_t   memory_size,
                         size_t   id,
                         uint64_t load_base,
                         uint64_t load_size) override;
    std::vector<size_t> get_code_object_ids() const override;

private:
    std::unique_ptr<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate> m_translator;
    std::vector<size_t> m_code_object_ids;
};

}  // namespace pc_sampling_collector
