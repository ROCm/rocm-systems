// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.1 allows NIC on agent.type and pmc_info.target_arch. The PMC row is unchanged.

#include "writers/common/info_registration_writer.hpp"
#include "writers/info_registration_writer.hpp"

#include "data_storage/schema_version.hpp"

namespace profiler_hub
{

template <>
class info_registration_writer<data_storage::schema_3_0_1_tag>
: public info_registration_common<
      info_registration_writer<data_storage::schema_3_0_1_tag>>
{
public:
    using info_registration_common<info_registration_writer<
        data_storage::schema_3_0_1_tag>>::info_registration_common;

    void reject_agent_type(const writer_types::agent_info_t& agent_info) const
    {
        if(!agent_info.unique_id.agent_type.has_value()) return;
        reject_unknown_agent_type(*agent_info.unique_id.agent_type);
    }

    void reject_pmc_fields(const writer_types::pmc_info_t& pmc_info) const
    {
        if(pmc_info.target_arch.has_value())
        {
            reject_unknown_target_arch(*pmc_info.target_arch);
        }
        reject_spm(pmc_info);
        reject_pmc_location(pmc_info);
    }

    void write_pmc_info(const writer_types::pmc_info_t& pmc_info,
                        primary_key_t                   primary_key,
                        primary_key_t                   process_pk,
                        std::optional<primary_key_t>    agent_pk)
    {
        write_pmc_info_3_0_0(pmc_info, primary_key, process_pk, agent_pk);
    }
};

}  // namespace profiler_hub
