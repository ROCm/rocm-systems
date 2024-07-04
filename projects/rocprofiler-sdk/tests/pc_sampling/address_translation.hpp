// MIT License
//
// Copyright (c) 2024 ROCm Developer Tools
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

#include <rocprofiler-sdk/amd_detail/rocprofiler-sdk-codeobj/code_printing.hpp>

#include <algorithm>
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
using Instruction             = rocprofiler::codeobj::disassembly::Instruction;
using CodeobjAddressTranslate = rocprofiler::codeobj::disassembly::CodeobjAddressTranslate;

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
    mutable std::shared_mutex                 mut;
    uint64_t                                  code_object_id_;
    std::string                               kernel_name_;
    uint64_t                                  begin_address_;
    uint64_t                                  end_address_;
    std::vector<std::unique_ptr<Instruction>> instructions_;
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
    std::unordered_map<std::string, std::unique_ptr<KernelObject>> kernel_object_map;
    mutable std::shared_mutex                                      mut;

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
    mutable std::shared_mutex mut;

    // FIXME: prevent direct access of the following fields.
    // The following fields should be accessible only from within `process` function.
    std::unique_ptr<Instruction> inst_;
    // In case an instruction is samples with different exec masks,
    // keep track of how many time each exec_mask was observed.
    std::map<uint64_t, uint64_t> exec_mask_counts_;
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
        auto lock = std::unique_lock{mut};

        auto inst_id = get_instruction_id(*instruction);
        auto itr     = samples.find(inst_id);
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

        auto inst_id = get_instruction_id(inst);
        auto itr     = samples.find(inst_id);
        if(itr == samples.end()) return nullptr;
        return itr->second.get();
    }

private:
    // For the sake of this test, we use `ld_addr` as the instruction identifier.
    // TODO: To cover code object loading/unloading and relocations,
    // use `(code_object_id + ld_addr)` as the unique identifier.
    // This assumes the decoder chage to return code_object_id as part
    // of the `LoadedCodeobjDecoder::get(uint64_t ld_addr)` method.
    using instrution_id_t = uint64_t;
    instrution_id_t get_instruction_id(const Instruction& instruction) const
    {
        // Ensure the decoder determined the `ld_addr`.
        assert(instruction.ld_addr > 0);
        return instruction.ld_addr;
    }

    std::unordered_map<instrution_id_t, std::unique_ptr<SampleInstruction>> samples;
    mutable std::shared_mutex                                               mut;
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
}  // namespace address_translation
}  // namespace client
