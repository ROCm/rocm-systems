// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <memory>

namespace rocprofiler::sdk::codeobj::disassembly
{
class CodeobjAddressTranslate;
}

namespace rocm_compute
{
class code_object_translator_t
{
public:
    virtual ~code_object_translator_t() = default;
    virtual void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size) = 0;
    virtual void add_code_object(const void* data,
                                 size_t      memory_size,
                                 size_t      id,
                                 uint64_t    load_addr,
                                 uint64_t    mem_size) = 0;

};

class code_object_translator_impl_t : public code_object_translator_t
{
public:
    code_object_translator_impl_t();
    void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size) override;
    void add_code_object(const void* data,
                         size_t      memory_size,
                         size_t      id,
                         uint64_t    load_addr,
                         uint64_t    mem_size) override;
private:
    std::unique_ptr<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate> m_translator;
};

}  // namespace pc_sampling_collector
