// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sample_writer.h"

#include "nlohmann/json.hpp"

#include <unistd.h>

#include <filesystem>
#include <fstream>

TEST_F(test_pc_sample_writer_t, ProvidedBegin_TopKeyIsSingleElementArray)
{
    m_writer.begin();

    const auto json = nlohmann::json::parse(m_writer.get_result());
    ASSERT_TRUE(json.contains("rocprofiler-sdk-tool"));
    ASSERT_TRUE(json["rocprofiler-sdk-tool"].is_array());
    EXPECT_EQ(json["rocprofiler-sdk-tool"].size(), 1u);
}

TEST_F(test_pc_sample_writer_t, ProvidedStochasticRecord_SerializesUnderStochasticBuffer)
{
    const auto record = make_stochastic_record();

    m_writer.begin();
    m_writer.append_stochastic(record, 5);

    const auto  json       = nlohmann::json::parse(m_writer.get_result());
    const auto& root       = json["rocprofiler-sdk-tool"][0];
    const auto& stochastic = root["buffer_records"]["pc_sample_stochastic"];
    ASSERT_EQ(stochastic.size(), 1u);

    const auto& entry = stochastic[0];

    // inst_index is a SIBLING of "record", not inside it.
    ASSERT_TRUE(entry.contains("inst_index"));
    EXPECT_EQ(entry["inst_index"], 5);

    const auto& rec = entry["record"];
    ASSERT_FALSE(rec.contains("inst_index"));

    EXPECT_EQ(rec["flags"]["has_mem_cnt"], record.flags.has_mem_cnt);

    EXPECT_EQ(rec["hw_id"]["chiplet"], record.hw_id.chiplet);
    EXPECT_EQ(rec["hw_id"]["wave_id"], record.hw_id.wave_id);
    EXPECT_EQ(rec["hw_id"]["simd_id"], record.hw_id.simd_id);
    EXPECT_EQ(rec["hw_id"]["pipe_id"], record.hw_id.pipe_id);
    EXPECT_EQ(rec["hw_id"]["cu_or_wgp_id"], record.hw_id.cu_or_wgp_id);
    EXPECT_EQ(rec["hw_id"]["shader_array_id"], record.hw_id.shader_array_id);
    EXPECT_EQ(rec["hw_id"]["shader_engine_id"], record.hw_id.shader_engine_id);
    EXPECT_EQ(rec["hw_id"]["workgroup_id"], record.hw_id.workgroup_id);
    EXPECT_EQ(rec["hw_id"]["vm_id"], record.hw_id.vm_id);
    EXPECT_EQ(rec["hw_id"]["queue_id"], record.hw_id.queue_id);
    EXPECT_EQ(rec["hw_id"]["microengine_id"], record.hw_id.microengine_id);

    EXPECT_EQ(rec["pc"]["code_object_id"], record.pc.code_object_id);
    EXPECT_EQ(rec["pc"]["code_object_offset"], record.pc.code_object_offset);

    EXPECT_EQ(rec["exec_mask"], record.exec_mask);
    EXPECT_EQ(rec["timestamp"], record.timestamp);
    EXPECT_EQ(rec["dispatch_id"], record.dispatch_id);

    EXPECT_EQ(rec["corr_id"]["internal"], record.corr_id.internal);
    EXPECT_EQ(rec["corr_id"]["external"], record.corr_id.external);

    EXPECT_EQ(rec["wrkgrp_id"]["x"], record.wrkgrp_id.x);
    EXPECT_EQ(rec["wrkgrp_id"]["y"], record.wrkgrp_id.y);
    EXPECT_EQ(rec["wrkgrp_id"]["z"], record.wrkgrp_id.z);

    EXPECT_EQ(rec["wave_in_grp"], record.wave_in_grp);
    EXPECT_EQ(rec["wave_issued"], record.wave_issued);
    EXPECT_EQ(rec["wave_cnt"], record.wave_cnt);

    // inst_type is serialized as the SDK enum-name string for the raw value.
    ASSERT_TRUE(rec["inst_type"].is_string());
    EXPECT_EQ(rec["inst_type"],
              rocprofiler_get_pc_sampling_instruction_type_name(
                  static_cast<rocprofiler_pc_sampling_instruction_type_t>(record.inst_type)));

    // snapshot.stall_reason is serialized as the SDK enum-name string.
    const auto& snap = rec["snapshot"];
    ASSERT_TRUE(snap["stall_reason"].is_string());
    EXPECT_EQ(snap["stall_reason"],
              rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
                  static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                      record.snapshot.stall_reason)));
    EXPECT_EQ(snap["dual_issue_valu"], record.snapshot.dual_issue_valu);

    EXPECT_EQ(snap["arb_state_issue_valu"], record.snapshot.arb_state_issue_valu);
    EXPECT_EQ(snap["arb_state_issue_matrix"], record.snapshot.arb_state_issue_matrix);
    EXPECT_EQ(snap["arb_state_issue_lds"], record.snapshot.arb_state_issue_lds);
    EXPECT_EQ(snap["arb_state_issue_lds_direct"], record.snapshot.arb_state_issue_lds_direct);
    EXPECT_EQ(snap["arb_state_issue_scalar"], record.snapshot.arb_state_issue_scalar);
    EXPECT_EQ(snap["arb_state_issue_vmem_tex"], record.snapshot.arb_state_issue_vmem_tex);
    EXPECT_EQ(snap["arb_state_issue_flat"], record.snapshot.arb_state_issue_flat);
    EXPECT_EQ(snap["arb_state_issue_exp"], record.snapshot.arb_state_issue_exp);
    EXPECT_EQ(snap["arb_state_issue_misc"], record.snapshot.arb_state_issue_misc);
    EXPECT_EQ(snap["arb_state_issue_brmsg"], record.snapshot.arb_state_issue_brmsg);

    EXPECT_EQ(snap["arb_state_stall_valu"], record.snapshot.arb_state_stall_valu);
    EXPECT_EQ(snap["arb_state_stall_matrix"], record.snapshot.arb_state_stall_matrix);
    EXPECT_EQ(snap["arb_state_stall_lds"], record.snapshot.arb_state_stall_lds);
    EXPECT_EQ(snap["arb_state_stall_lds_direct"], record.snapshot.arb_state_stall_lds_direct);
    EXPECT_EQ(snap["arb_state_stall_scalar"], record.snapshot.arb_state_stall_scalar);
    EXPECT_EQ(snap["arb_state_stall_vmem_tex"], record.snapshot.arb_state_stall_vmem_tex);
    EXPECT_EQ(snap["arb_state_stall_flat"], record.snapshot.arb_state_stall_flat);
    EXPECT_EQ(snap["arb_state_stall_exp"], record.snapshot.arb_state_stall_exp);
    EXPECT_EQ(snap["arb_state_stall_misc"], record.snapshot.arb_state_stall_misc);
    EXPECT_EQ(snap["arb_state_stall_brmsg"], record.snapshot.arb_state_stall_brmsg);
}

TEST_F(test_pc_sample_writer_t, ProvidedHostTrapRecord_SerializesWithoutStochasticOnlyFields)
{
    const auto record = make_host_trap_record();

    m_writer.begin();
    m_writer.append_host_trap(record, 8);

    const auto  json      = nlohmann::json::parse(m_writer.get_result());
    const auto& root      = json["rocprofiler-sdk-tool"][0];
    const auto& host_trap = root["buffer_records"]["pc_sample_host_trap"];
    ASSERT_EQ(host_trap.size(), 1u);

    const auto& entry = host_trap[0];
    ASSERT_TRUE(entry.contains("inst_index"));
    EXPECT_EQ(entry["inst_index"], 8);

    const auto& rec = entry["record"];
    EXPECT_FALSE(rec.contains("snapshot"));
    EXPECT_FALSE(rec.contains("wave_cnt"));
    EXPECT_FALSE(rec.contains("inst_type"));

    ASSERT_TRUE(rec.contains("hw_id"));
    ASSERT_TRUE(rec.contains("pc"));
    EXPECT_EQ(rec["exec_mask"], record.exec_mask);
    EXPECT_EQ(rec["timestamp"], record.timestamp);
    EXPECT_EQ(rec["dispatch_id"], record.dispatch_id);
    EXPECT_EQ(rec["corr_id"]["internal"], record.corr_id.internal);
    EXPECT_EQ(rec["corr_id"]["external"], record.corr_id.external);
    EXPECT_EQ(rec["wrkgrp_id"]["x"], record.wrkgrp_id.x);
    EXPECT_EQ(rec["wave_in_grp"], record.wave_in_grp);
}

TEST_F(test_pc_sample_writer_t, ProvidedStringTable_SerializesInstructionsAndComments)
{
    rocprofiler_compute_tool::pc_string_table_t string_table;
    string_table.insert("s_nop", "a.cpp:1");
    string_table.insert("v_add", "b.cpp:2");

    m_writer.begin();
    m_writer.set_strings(string_table);

    const auto  json    = nlohmann::json::parse(m_writer.get_result());
    const auto& strings = json["rocprofiler-sdk-tool"][0]["strings"];

    ASSERT_TRUE(strings["pc_sample_instructions"].is_array());
    ASSERT_EQ(strings["pc_sample_instructions"].size(), 2u);
    EXPECT_EQ(strings["pc_sample_instructions"][0], "s_nop");
    EXPECT_EQ(strings["pc_sample_instructions"][1], "v_add");

    ASSERT_TRUE(strings["pc_sample_comments"].is_array());
    ASSERT_EQ(strings["pc_sample_comments"].size(), 2u);
    EXPECT_EQ(strings["pc_sample_comments"][0], "a.cpp:1");
    EXPECT_EQ(strings["pc_sample_comments"][1], "b.cpp:2");
}

TEST_F(test_pc_sample_writer_t, ProvidedKernelSymbols_SerializesThemInOrder)
{
    const std::vector<rocprofiler_compute_tool::kernel_symbol_entry_t> syms{
        {2, "foo"},
        {3, "bar"},
    };

    m_writer.begin();
    m_writer.set_kernel_symbols(syms);

    const auto  json           = nlohmann::json::parse(m_writer.get_result());
    const auto& kernel_symbols = json["rocprofiler-sdk-tool"][0]["kernel_symbols"];

    ASSERT_EQ(kernel_symbols.size(), 2u);
    EXPECT_EQ(kernel_symbols[0]["code_object_id"], 2);
    EXPECT_EQ(kernel_symbols[0]["formatted_kernel_name"], "foo");
    EXPECT_EQ(kernel_symbols[1]["code_object_id"], 3);
    EXPECT_EQ(kernel_symbols[1]["formatted_kernel_name"], "bar");
}

TEST_F(test_pc_sample_writer_t, ProvidedNoApiTraces_AllApiCategoriesAreEmptyArrays)
{
    m_writer.begin();

    const auto  json           = nlohmann::json::parse(m_writer.get_result());
    const auto& buffer_records = json["rocprofiler-sdk-tool"][0]["buffer_records"];

    for (const char* category : {"hip_api",
                                 "hsa_api",
                                 "memory_copy",
                                 "marker_api",
                                 "rccl_api",
                                 "memory_allocation",
                                 "scratch_memory",
                                 "kfd",
                                 "rocdecode_api",
                                 "rocjpeg_api"})
    {
        ASSERT_TRUE(buffer_records.contains(category)) << category;
        ASSERT_TRUE(buffer_records[category].is_array()) << category;
        EXPECT_EQ(buffer_records[category].size(), 0u) << category;
    }
}

TEST_F(test_pc_sample_writer_t, ProvidedNoRecords_ReturnsValidJsonWithEmptySampleArrays)
{
    m_writer.begin();

    const auto& result = m_writer.get_result();
    EXPECT_FALSE(result.empty());
    EXPECT_TRUE(nlohmann::json::accept(result));

    const auto  json           = nlohmann::json::parse(result);
    const auto& buffer_records = json["rocprofiler-sdk-tool"][0]["buffer_records"];

    ASSERT_TRUE(buffer_records["pc_sample_stochastic"].is_array());
    EXPECT_EQ(buffer_records["pc_sample_stochastic"].size(), 0u);

    ASSERT_TRUE(buffer_records["pc_sample_host_trap"].is_array());
    EXPECT_EQ(buffer_records["pc_sample_host_trap"].size(), 0u);
}

TEST_F(test_pc_sample_writer_t, ProvidedEmptyOutputFilePath_Throws)
{
    m_writer.begin();
    EXPECT_THROW(m_writer.flush(""), std::runtime_error);
}

TEST_F(test_pc_sample_writer_t, ProvidedUnopenableOutputFilePath_Throws)
{
    m_writer.begin();
    // A path whose "parent" is an existing regular file cannot be opened/created.
    const auto tmp = std::filesystem::temp_directory_path() /
                     ("rpc_pcw_" + std::to_string(::getpid()) + ".json");
    {
        std::ofstream ofs(tmp);
        ofs << "x";
    }
    const auto unopenable = tmp / "child.json";  // tmp is a file, not a dir
    EXPECT_THROW(m_writer.flush(unopenable), std::runtime_error);
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
}

// Cross-language contract: pin the exact key path the Python analyze side
// (analysis_db.calc_pc_sampling_data) reads, so a rename here fails loudly.
TEST_F(test_pc_sample_writer_t, SerializesContractKeyPathConsumedByAnalyze)
{
    const auto stochastic = make_stochastic_record();
    const auto host_trap  = make_host_trap_record();

    rocprofiler_compute_tool::pc_string_table_t string_table;
    const size_t                                idx = string_table.insert("v_add", "kernel.cpp:7");

    m_writer.begin();
    m_writer.append_stochastic(stochastic, idx);
    m_writer.append_host_trap(host_trap, idx);
    m_writer.set_strings(string_table);
    m_writer.set_kernel_symbols({{42, "my_kernel(int)"}});
    m_writer.set_metadata(1234);

    const auto json = nlohmann::json::parse(m_writer.get_result());

    ASSERT_TRUE(json["rocprofiler-sdk-tool"].is_array());
    const auto& root = json["rocprofiler-sdk-tool"][0];

    ASSERT_TRUE(root["buffer_records"]["pc_sample_stochastic"].is_array());
    ASSERT_TRUE(root["buffer_records"]["pc_sample_host_trap"].is_array());
    ASSERT_TRUE(root["strings"]["pc_sample_instructions"].is_array());
    ASSERT_TRUE(root["strings"]["pc_sample_comments"].is_array());
    ASSERT_TRUE(root["kernel_symbols"].is_array());

    // inst_index is a sibling of "record", referencing the string tables.
    const auto& s_entry = root["buffer_records"]["pc_sample_stochastic"][0];
    ASSERT_TRUE(s_entry.contains("record"));
    ASSERT_TRUE(s_entry.contains("inst_index"));
    EXPECT_EQ(s_entry["inst_index"], idx);
    EXPECT_EQ(root["strings"]["pc_sample_instructions"][idx], "v_add");
    EXPECT_EQ(root["strings"]["pc_sample_comments"][idx], "kernel.cpp:7");

    const auto& k = root["kernel_symbols"][0];
    EXPECT_EQ(k["code_object_id"], 42u);
    EXPECT_EQ(k["formatted_kernel_name"], "my_kernel(int)");
}

TEST_F(test_pc_sample_writer_t, ProvidedAgents_SerializesThemInSdkShape)
{
    const auto agent = make_agent_record();

    m_writer.begin();
    m_writer.set_agents({agent});

    const auto  json   = nlohmann::json::parse(m_writer.get_result());
    const auto& agents = json["rocprofiler-sdk-tool"][0]["agents"];

    ASSERT_TRUE(agents.is_array());
    ASSERT_EQ(agents.size(), 1u);

    const auto& a = agents[0];
    EXPECT_EQ(a["size"], agent.size);
    EXPECT_EQ(a["id"]["handle"], agent.id_handle);
    EXPECT_EQ(a["type"], agent.type);
    EXPECT_EQ(a["node_id"], agent.node_id);
    EXPECT_EQ(a["logical_node_id"], agent.logical_node_id);
    EXPECT_EQ(a["cu_count"], agent.cu_count);
    EXPECT_EQ(a["gpu_id"], agent.gpu_id);
    EXPECT_EQ(a["wave_front_size"], agent.wave_front_size);
    EXPECT_EQ(a["simd_count"], agent.simd_count);
}

TEST_F(test_pc_sample_writer_t, ProvidedKernelDispatches_SerializesThemInSdkShape)
{
    const auto dispatch = make_kernel_dispatch_record();

    m_writer.begin();
    m_writer.set_kernel_dispatches({dispatch});

    const auto  json = nlohmann::json::parse(m_writer.get_result());
    const auto& kd   = json["rocprofiler-sdk-tool"][0]["buffer_records"]["kernel_dispatch"];

    ASSERT_TRUE(kd.is_array());
    ASSERT_EQ(kd.size(), 1u);

    const auto& d = kd[0];
    EXPECT_EQ(d["size"], dispatch.size);
    EXPECT_EQ(d["kind"], dispatch.kind);
    EXPECT_EQ(d["operation"], dispatch.operation);
    EXPECT_EQ(d["thread_id"], dispatch.thread_id);
    EXPECT_EQ(d["correlation_id"]["internal"], dispatch.corr_internal);
    EXPECT_EQ(d["correlation_id"]["external"], dispatch.corr_external);
    EXPECT_EQ(d["start_timestamp"], dispatch.start_timestamp);
    EXPECT_EQ(d["end_timestamp"], dispatch.end_timestamp);

    const auto& di = d["dispatch_info"];
    EXPECT_EQ(di["size"], dispatch.dispatch_info_size);
    EXPECT_EQ(di["agent_id"]["handle"], dispatch.agent_id_handle);
    EXPECT_EQ(di["queue_id"]["handle"], dispatch.queue_id_handle);
    EXPECT_EQ(di["kernel_id"], dispatch.kernel_id);
    EXPECT_EQ(di["dispatch_id"], dispatch.dispatch_id);
    EXPECT_EQ(di["private_segment_size"], dispatch.private_segment_size);
    EXPECT_EQ(di["group_segment_size"], dispatch.group_segment_size);
    EXPECT_EQ(di["workgroup_size"]["x"], dispatch.workgroup_size.x);
    EXPECT_EQ(di["workgroup_size"]["y"], dispatch.workgroup_size.y);
    EXPECT_EQ(di["workgroup_size"]["z"], dispatch.workgroup_size.z);
    EXPECT_EQ(di["grid_size"]["x"], dispatch.grid_size.x);
    EXPECT_EQ(di["grid_size"]["y"], dispatch.grid_size.y);
    EXPECT_EQ(di["grid_size"]["z"], dispatch.grid_size.z);
}

TEST_F(test_pc_sample_writer_t, ProvidedKernelSymbols_SerializesKernelId)
{
    m_writer.begin();
    m_writer.set_kernel_symbols({{7, "foo", 99}});

    const auto  json = nlohmann::json::parse(m_writer.get_result());
    const auto& k    = json["rocprofiler-sdk-tool"][0]["kernel_symbols"][0];
    EXPECT_EQ(k["code_object_id"], 7u);
    EXPECT_EQ(k["formatted_kernel_name"], "foo");
    EXPECT_EQ(k["kernel_id"], 99u);
}
