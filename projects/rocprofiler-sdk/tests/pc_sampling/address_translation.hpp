// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace client
{
namespace address_translation
{
using Instruction             = rocprofiler::sdk::codeobj::disassembly::Instruction;
using CodeobjAddressTranslate = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;
using marker_id_t             = rocprofiler::sdk::codeobj::disassembly::marker_id_t;

/**
 * @brief Pair (code_object_id, pc_addr) uniquely identifies an instruction.
 */
struct inst_id_t
{
    marker_id_t code_object_id = 0;
    uint64_t    pc_addr        = 0;

    bool operator==(const inst_id_t& b) const
    {
        return this->pc_addr == b.pc_addr && this->code_object_id == b.code_object_id;
    };

    bool operator<(const inst_id_t& b) const
    {
        if(this->code_object_id == b.code_object_id) return this->pc_addr < b.pc_addr;
        return this->code_object_id < b.code_object_id;
    };
};

class KernelObject
{
private:
    using process_inst_fn = std::function<void(const Instruction&)>;

public:
    KernelObject() = default;
    KernelObject(uint64_t    code_object_id,
                 std::string kernel_name,
                 uint64_t    begin_address,
                 uint64_t    end_address);

    // write lock required
    void add_instruction(std::unique_ptr<Instruction> instruction)
    {
        auto lock = std::unique_lock{mut};

        instructions_.push_back(std::move(instruction));
    }

    // read lock required
    void iterate_instrunctions(process_inst_fn fn) const
    {
        auto lock = std::shared_lock{mut};

        for(const auto& inst : this->instructions_)
            fn(*inst);
    }

    uint64_t    code_object_id() const { return code_object_id_; };
    std::string kernel_name() const { return kernel_name_; };
    uint64_t    begin_address() const { return begin_address_; };
    uint64_t    end_address() const { return end_address_; };

private:
    mutable std::shared_mutex                 mut             = {};
    uint64_t                                  code_object_id_ = 0;
    std::string                               kernel_name_    = {};
    uint64_t                                  begin_address_  = 0;
    uint64_t                                  end_address_    = 0;
    std::vector<std::unique_ptr<Instruction>> instructions_   = {};
};

class KernelObjectMap
{
private:
    using process_kernel_fn = std::function<void(const KernelObject*)>;

public:
    KernelObjectMap() = default;

    // write lock required
    void add_kernel(uint64_t    code_object_id,
                    std::string name,
                    uint64_t    begin_address,
                    uint64_t    end_address)
    {
        auto lock = std::unique_lock{mut};

        auto key = form_key(code_object_id, name, begin_address);
        auto it  = kernel_object_map.find(key);
        assert(it == kernel_object_map.end());
        kernel_object_map.insert(
            {key,
             std::make_unique<KernelObject>(code_object_id, name, begin_address, end_address)});
    }

#if 0
    // read lock required
    KernelObject* get_kernel(uint64_t code_object_id, std::string name)
    {
        auto lock = std::shared_lock{mut};

        auto key = form_key(code_object_id, name);
        auto it = kernel_object_map.find(key);
        if(it == kernel_object_map.end())
        {
            return nullptr;
        }

        return it->second.get();
    }
#endif

    // read lock required
    void iterate_kernel_objects(process_kernel_fn fn) const
    {
        auto lock = std::shared_lock{mut};

        for(auto& [_, kernel_obj] : kernel_object_map)
            fn(kernel_obj.get());
    }

private:
    std::unordered_map<std::string, std::unique_ptr<KernelObject>> kernel_object_map = {};
    mutable std::shared_mutex                                      mut               = {};

    std::string form_key(uint64_t code_object_id, std::string kernel_name, uint64_t begin_address)
    {
        return std::to_string(code_object_id) + "_" + kernel_name + "_" +
               std::to_string(begin_address);
    }
};

class SampleInstruction
{
private:
    using proces_sample_inst_fn = std::function<void(const SampleInstruction&)>;

public:
    SampleInstruction() = default;
    SampleInstruction(std::unique_ptr<Instruction> inst)
    : inst_(std::move(inst))
    {}

    // write lock required
    void add_sample(uint64_t exec_mask)
    {
        auto lock = std::unique_lock{mut};

        if(exec_mask_counts_.find(exec_mask) == exec_mask_counts_.end())
        {
            exec_mask_counts_[exec_mask] = 0;
        }
        exec_mask_counts_[exec_mask]++;
        sample_count_++;
    }

    // read lock required
    void process(proces_sample_inst_fn fn) const
    {
        auto lock = std::shared_lock{mut};

        fn(*this);
    }

    Instruction* inst() const { return inst_.get(); };
    // In case an instruction is samples with different exec masks,
    // keep track of how many time each exec_mask was observed.
    const std::map<uint64_t, uint64_t>& exec_mask_counts() const { return exec_mask_counts_; }
    // How many time this instruction is samples
    uint64_t sample_count() const { return sample_count_; };

private:
    mutable std::shared_mutex mut = {};

    // FIXME: prevent direct access of the following fields.
    // The following fields should be accessible only from within `process` function.
    std::unique_ptr<Instruction> inst_ = {};
    // In case an instruction is samples with different exec masks,
    // keep track of how many time each exec_mask was observed.
    std::map<uint64_t, uint64_t> exec_mask_counts_ = {};
    // How many time this instruction is samples
    uint64_t sample_count_ = 0;
};

class FlatProfile
{
public:
    FlatProfile() = default;

    // write lock required
    void add_sample(std::unique_ptr<Instruction> instruction, uint64_t exec_mask)
    {
        // counting valid decoded samples
        valid_decoded_samples_num++;
        auto lock = std::unique_lock{mut};

        inst_id_t inst_id = {.code_object_id = instruction->codeobj_id,
                             .pc_addr        = instruction->ld_addr};
        auto      itr     = samples.find(inst_id);
        if(itr == samples.end())
        {
            // Add new instruction
            samples.insert({inst_id, std::make_unique<SampleInstruction>(std::move(instruction))});
            itr = samples.find(inst_id);
        }

        auto* sample_instruction = itr->second.get();
        sample_instruction->add_sample(exec_mask);
    }

    // read lock required
    const SampleInstruction* get_sample_instruction(const Instruction& inst) const
    {
        auto lock = std::shared_lock{mut};

        // TODO: Avoid creating a new instance of `inst_id_t` whenever querying
        // sampled instructions.
        inst_id_t inst_id = {.code_object_id = inst.codeobj_id, .pc_addr = inst.ld_addr};
        auto      itr     = samples.find(inst_id);
        if(itr == samples.end()) return nullptr;
        return itr->second.get();
        return nullptr;
    }

    void add_invalid_sample()
    {
        // counting invalid samples
        invalid_decoded_samples_num++;
    }

    /**
     * @brief Verify that more valid decoded samples is generated.
     */
    bool more_valid_decoded_samples_expected() const
    {
        return valid_decoded_samples_num > invalid_decoded_samples_num;
    }

    uint64_t get_valid_decoded_samples_num() const { return valid_decoded_samples_num; }

    uint64_t get_invalid_samples_num() const { return invalid_decoded_samples_num; }

private:
    // TODO: optimize to use unordered_map
    std::map<inst_id_t, std::unique_ptr<SampleInstruction>> samples                     = {};
    std::atomic<uint64_t>                                   valid_decoded_samples_num   = {};
    std::atomic<uint64_t>                                   invalid_decoded_samples_num = {};
    mutable std::shared_mutex                               mut                         = {};
};

std::mutex&
get_global_mutex();

CodeobjAddressTranslate&
get_address_translator();

KernelObjectMap&
get_kernel_object_map();

FlatProfile&
get_flat_profile();

void
dump_flat_profile();

void
init();

void
fini();

void
disable_strict_checks_if_needed(rocprofiler_agent_t agent);

}  // namespace address_translation
}  // namespace client
