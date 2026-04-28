// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rocprofsys
{

enum class agent_type : uint8_t
{
    CPU,  ///< Agent type is a CPU
    GPU,  ///< Agent type is a GPU
    NIC,  ///< Agent type is a NIC
};

inline const char*
to_string(agent_type type)
{
    switch(type)
    {
        case agent_type::GPU: return "GPU";
        case agent_type::CPU: return "CPU";
        case agent_type::NIC: return "NIC";
        default: throw std::runtime_error("Invalid agent type.");
    }
}

struct agent
{
    agent_type  type;
    uint64_t    handle;
    uint64_t    device_id;
    uint32_t    node_id;
    int32_t     logical_node_id;
    int32_t     logical_node_type_id;
    std::string name;
    std::string model_name;
    std::string vendor_name;
    std::string product_name;

    size_t device_type_index{
        0
    };  // Per-type ID (GPU, CPU) of the agent as they are stored in the agent_manager
    size_t base_id{ 0 };  // Database entry index of the agent

    std::string
        agent_info;  // JSON formatted serialization of the available agent information
};

}  // namespace rocprofsys
