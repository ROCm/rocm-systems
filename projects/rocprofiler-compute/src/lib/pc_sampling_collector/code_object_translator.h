// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rocprofiler::sdk::codeobj::disassembly
{
class CodeobjAddressTranslate;
}

namespace rocprofiler_compute_tool
{
struct symbol_t
{
    std::string name{};
    uint64_t    code_object_offset = 0;
    uint64_t    virtual_address    = 0;
    uint64_t    size               = 0;
};

struct instruction_t
{
    std::string name{};
    std::string comment{};
    uint64_t    virtual_address = 0;
    uint64_t    code_obj_offset = 0;
    size_t      size{0};
};

class code_object_translator_t
{
public:
    virtual ~code_object_translator_t() = default;
    virtual void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size) = 0;
    virtual void                       add_code_object(uint64_t data,
                                                       size_t   memory_size,
                                                       size_t   id,
                                                       uint64_t load_addr,
                                                       uint64_t mem_size)                   = 0;
    virtual const std::vector<size_t>& get_code_object_ids() const                          = 0;
    virtual std::vector<symbol_t>      get_symbols(size_t object_id) const                  = 0;
    virtual instruction_t get_instruction(size_t object_id, uint64_t virtual_address) const = 0;
};

class code_object_translator_impl_t : public code_object_translator_t
{
public:
    code_object_translator_impl_t();
    ~code_object_translator_impl_t() override;
    void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size) override;
    void add_code_object(uint64_t memory_base,
                         size_t   memory_size,
                         size_t   id,
                         uint64_t load_base,
                         uint64_t load_size) override;

    const std::vector<size_t>& get_code_object_ids() const override;
    std::vector<symbol_t>      get_symbols(size_t object_id) const override;
    instruction_t get_instruction(size_t object_id, uint64_t virtual_address) const override;

private:
    std::unique_ptr<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate> m_translator;
    std::vector<size_t>                                                              m_obj_ids;
    std::map<size_t, uint64_t> m_obj_id_to_load_addr;
};

}  // namespace rocprofiler_compute_tool
