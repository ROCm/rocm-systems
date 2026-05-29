// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/aqlprofile/hsa_includes.h"
#include "lib/aqlprofile/def/gpu_block_info.h"
#include "lib/aqlprofile/core/aql_profile.hpp"
#include "lib/aqlprofile/core/pm4_factory.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <array>

struct EventDimension
{
    static void        init() {}  // legacy call. replaced by below
    static const auto& get_dimension_table()
    {
        static auto event_dimension_table = []() {
            auto _data = std::unordered_map<std::string_view, size_t>{};
            for(const auto* itr : {"XCD", "AID", "SE", "SA", "WGP", "INSTANCE"})
                _data.emplace(std::string_view{itr}, _data.size());
            return _data;
        }();
        return event_dimension_table;
    }

    EventDimension(std::string_view _name, size_t _extent)
    : id{get_dimension_table().at(_name)}
    , name{_name}
    , extent{_extent}
    {}
    ~EventDimension()                           = default;
    EventDimension(const EventDimension& other) = default;
    EventDimension& operator=(const EventDimension& other) = default;
    EventDimension(EventDimension&& other) noexcept        = default;
    EventDimension& operator=(EventDimension&& other) noexcept = default;

    uint64_t         id     = 0;
    uint64_t         extent = 0;
    std::string_view name   = {};
};

class EventKey
{
public:
    uint64_t agent = 0;
    uint64_t block = 0;

    bool operator==(const EventKey& other) const
    {
        return agent == other.agent && block == other.block;
    }
    bool operator!=(const EventKey& other) const { return !(*this == other); }
};

template <>
struct std::hash<EventKey>
{
    uint64_t operator()(const EventKey& ev) const
    {
        return ev.agent | (ev.block << 56) | (ev.block >> 8);
    }
};

class EventAttribDimension
{
public:
    static constexpr size_t event_id_bit = 24;

    template <typename AgentType>
    EventAttribDimension(AgentType agent, hsa_ven_amd_aqlprofile_block_name_t block_name)
    : key({agent.handle, (uint64_t) block_name})
    {
        EventDimension::init();

        aql_profile::Pm4Factory* pm4_factory = aql_profile::Pm4Factory::Create(agent);
        this->block_info                     = pm4_factory->GetBlockInfo(block_name);

        bIsGFX12 = pm4_factory->IsGFX12();
        bIsGFX11 = pm4_factory->IsGFX11();
        bIsGFX9  = pm4_factory->IsGFX9();

        num_xccs = pm4_factory->GetXccNumber();
        if(num_xccs > 1 && HasAttr(CounterBlockUmcAttr))
        {  // For MI300 AID only
            num_xccs = 1;
            num_aid  = 4;
        }
        if(num_xccs > 1 && HasAttr(CounterBlockGrbmaAttr))
        {
            num_aid  = (num_xccs + pm4_factory->GetXccPerAid() - 1) / pm4_factory->GetXccPerAid();
            num_xccs = 1;
        }
        shader_engine = HasAttr(CounterBlockSeAttr);
        shader_array  = HasAttr(CounterBlockSaAttr);

        if(bIsGFX9)
            compute_unit = HasAttr(CounterBlockTcAttr) && shader_engine;
        else if(HasAttr(CounterBlockWgpAttr))
            workgroup_processor = true;
        else if(bIsGFX11 || bIsGFX12)
            workgroup_processor = HasAttr(CounterBlockSqAttr);

        se_num  = pm4_factory->GetShaderEnginesNumber();
        sarrays = pm4_factory->GetShaderArraysNumber() * se_num;

        cu_num  = (pm4_factory->GetComputeUnitNumber() + sarrays - 1) / sarrays;
        wgp_num = (pm4_factory->GetComputeUnitNumber() / 2 + sarrays - 1) / sarrays;

        if(HasAttr(CounterBlockUmcAttr))
            block_instance_count = block_info->instance_count / num_aid;
        else if(compute_unit)
            block_instance_count = std::min<size_t>(block_info->instance_count, cu_num + 1);
        else
            block_instance_count = block_info->instance_count;

        if(num_xccs > 1) dimensions.push_back({"XCD", num_xccs});
        if(num_aid > 1) dimensions.push_back({"AID", num_aid});

        if(workgroup_processor)
        {
            dimensions.push_back({"WGP", wgp_num});
            if(bIsGFX11)
                dimensions.push_back({"INSTANCE", block_instance_count});
            else if(block_instance_count > 1)
                dimensions.push_back({"INSTANCE", block_instance_count});
        }
        else
            dimensions.push_back({"INSTANCE", block_instance_count});

        if(shader_engine)
            dimensions.push_back(
                {"SE", pm4_factory->GetShaderEnginesNumber() / (num_xccs > 0 ? num_xccs : 1)});
        if(shader_array) dimensions.push_back({"SA", pm4_factory->GetShaderArraysNumber()});
    }

    size_t get_num_xccs() const { return num_xccs; };
    size_t get_total_elements() const
    {
        size_t acc = 1;
        for(auto& d : dimensions)
            acc *= d.extent;
        return acc;
    }
    uint64_t       get_num() const { return dimensions.size(); };
    EventDimension get_dim(uint64_t index) const { return dimensions.at(index); };

    hsa_status_t get_coordinates(uint8_t* coordinates, int64_t cumulative_id) const
    {
        const int end = static_cast<int>(get_num()) - 1;
        for(int i = end; i >= 0; i--)
        {
            coordinates[i] = static_cast<uint8_t>(cumulative_id % dimensions.at(i).extent);
            cumulative_id /= dimensions.at(i).extent;
        }
        if(cumulative_id != 0) return HSA_STATUS_ERROR_INVALID_INDEX;
        return HSA_STATUS_SUCCESS;
    }

    size_t get_num_instances() const { return block_instance_count; }

private:
    bool HasAttr(CounterBlockAttr attr) const { return (block_info->attr & attr) != 0; }

    EventKey                       key        = {};
    const GpuBlockInfo*            block_info = nullptr;
    hsa_ven_amd_aqlprofile_event_t event      = {};

    bool bIsGFX12 = false;
    bool bIsGFX11 = false;
    bool bIsGFX9  = false;

    bool shader_engine       = false;
    bool shader_array        = false;
    bool compute_unit        = false;
    bool workgroup_processor = false;

    size_t num_xccs             = 1;
    size_t num_aid              = 1;
    size_t se_num               = 1;
    size_t sarrays              = 1;
    size_t cu_num               = 1;
    size_t wgp_num              = 1;
    size_t block_instance_count = 1;

    std::vector<EventDimension> dimensions = {};

public:
    template <typename AgentType>
    static const EventAttribDimension& get(AgentType                           agent,
                                           hsa_ven_amd_aqlprofile_block_name_t block_name)
    {
        thread_local auto event_cache = std::shared_ptr<EventAttribDimension>{nullptr};
        thread_local auto event_map =
            std::unordered_map<EventKey, std::shared_ptr<EventAttribDimension>>{};

        auto key = EventKey{agent.handle, static_cast<uint64_t>(block_name)};

        if(!event_cache || event_cache->key != key)
        {
            if(auto it = event_map.find(key); it != event_map.end())
                event_cache = it->second;
            else
                event_cache =
                    event_map
                        .emplace(key, std::make_shared<EventAttribDimension>(agent, block_name))
                        .first->second;
        }

        return *event_cache;
    }
};
