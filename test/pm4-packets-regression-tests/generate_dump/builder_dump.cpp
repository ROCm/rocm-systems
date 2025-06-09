#include <iostream>
#include <stdexcept>
#define LITTLEENDIAN_CPU 1
#include <bits/stdc++.h>
#include <cstddef>
#include <cstdint>
#include <fstream>

// #define DEBUG_TRACE

#include "core/pm4_factory.h"
#include "builder_dump.h"

// Below static member vairables need to be defined as we are including pm4_factory.h
bool aql_profile::Pm4Factory::concurrent_create_mode_ = false;
bool aql_profile::Pm4Factory::spm_kfd_mode_ = false;
aql_profile::Pm4Factory::instances_t* aql_profile::Pm4Factory::instances_ = NULL;

const AgentInfo *GetAgentInfoByName(string gfx_name) {
  HsaRsrcFactory* hsa_rsrc_factory_instance = HsaRsrcFactory::Create();
  uint32_t gpu_agent_count = hsa_rsrc_factory_instance->GetCountOfGpuAgents();
  for (uint32_t idx = 0; idx < gpu_agent_count; idx++) {
    const AgentInfo *agent_info;
    hsa_rsrc_factory_instance->GetGpuAgentInfo(idx, &agent_info);
    if (gfx_name == agent_info->name) {
      return agent_info;
    }
  }
  return nullptr;
}

std::vector<std::string> GetAllAgentNames() {
  std::vector<std::string> names;
  HsaRsrcFactory* hsa_rsrc_factory_instance = HsaRsrcFactory::Create();
  uint32_t gpu_agent_count = hsa_rsrc_factory_instance->GetCountOfGpuAgents();
  for (uint32_t idx = 0; idx < gpu_agent_count; idx++) {
    const AgentInfo *agent_info;
    hsa_rsrc_factory_instance->GetGpuAgentInfo(idx, &agent_info);
    names.emplace_back(agent_info->name); 
  }
  return names;
}

void generate_individual_pmc_dump(
    const AgentInfo *agent_info, hsa_ven_amd_aqlprofile_block_name_t block_name,
    std::string output_file_name) {
      
  auto profile = create_profile(agent_info, block_name);
  aql_profile::gpu_id_t gpu_id = get_gpu_id(agent_info->name);
  if (gpu_id == aql_profile::INVAL_GPU_ID) {
    throw std::runtime_error("Invalid GPU ID");
  }
  auto pm4_factory = get_pm4_factory(agent_info, gpu_id);
  pm4_builder::counters_vector test_counters_vec = CountersVec(profile, pm4_factory);
  pmc_builder_dump::GenerateDump(output_file_name,
                                      pm4_factory->GetPmcBuilder(),
                                      test_counters_vec);
  destroy_profile(profile);
}

void generate_spm_dump(
    const AgentInfo *agent_info, hsa_ven_amd_aqlprofile_block_name_t block_name,
    std::string output_file_name) {
      
  auto profile = create_profile(agent_info, block_name);
  aql_profile::gpu_id_t gpu_id = get_gpu_id(agent_info->name);
  if (gpu_id == aql_profile::INVAL_GPU_ID) {
    throw std::runtime_error("Invalid GPU ID");
  }
  auto pm4_factory = get_pm4_factory(agent_info, gpu_id);
  pm4_builder::counters_vector test_counters_vec = CountersVec(profile, pm4_factory);
  spm_builder_dump::GenerateDump(output_file_name,
                                      pm4_factory->GetSpmBuilder(),
                                      test_counters_vec);
  destroy_profile(profile);
}

void generate_sqtt_dump(
    const AgentInfo *agent_info, 
    std::string output_file_name) {
      
  auto profile = create_profile(agent_info, HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ);
  aql_profile::gpu_id_t gpu_id = get_gpu_id(agent_info->name);
  if (gpu_id == aql_profile::INVAL_GPU_ID) {
    throw std::runtime_error("Invalid GPU ID");
  }

  pm4_builder::TraceConfig config{};
  config.control_buffer_ptr = nullptr;
  config.control_buffer_size = 0;
  config.data_buffer_ptr = nullptr;
  config.data_buffer_size = 0;

  auto pm4_factory = get_pm4_factory(agent_info, gpu_id);
  sqtt_builder_dump::GenerateDump(output_file_name,
                                      pm4_factory->GetSqttBuilder(),
                                      &config
                                      );
  destroy_profile(profile);
}

typedef std::vector<std::pair<hsa_ven_amd_aqlprofile_block_name_t, std::string>> output_file_map_t;

output_file_map_t gfx9_pmc_map(){
    output_file_map_t
      pmc_output_file_map = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, "PMC_CPC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, "PMC_CPF_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, "PMC_GDS_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, "PMC_GRBM_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, "PMC_GRBMSE_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, "PMC_SPI_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, "PMC_SQ_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS, "PMC_SQCS_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, "PMC_SX_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, "PMC_TA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA, "PMC_TCA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, "PMC_TCC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, "PMC_TCP_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, "PMC_TD_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, "PMC_MCVM_L2_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, "PMC_ATC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, "PMC_ATC_L2_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, "PMC_GCEA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, "PMC_RPB_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, "PMC_SDMA_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC, "PMC_UMC_dump.txt"},
      };
    return pmc_output_file_map;
}

output_file_map_t gfx10_pmc_map(){
    output_file_map_t
      pmc_output_file_map = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, "PMC_CPC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, "PMC_CPF_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, "PMC_GDS_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, "PMC_GRBM_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, "PMC_GRBMSE_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, "PMC_SPI_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, "PMC_SQ_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS, "PMC_SQCS_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, "PMC_SX_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, "PMC_TA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, "PMC_GCEA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, "PMC_GL1A_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C, "PMC_GL1C_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A, "PMC_GL2A_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C, "PMC_GL2C_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR, "PMC_GCR_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS, "PMC_GUS_dump.txt"},
      };
    return pmc_output_file_map;
}

output_file_map_t gfx11_pmc_map(){
    output_file_map_t
      pmc_output_file_map = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, "PMC_CPC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, "PMC_CPF_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, "PMC_GDS_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, "PMC_GRBM_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, "PMC_GRBMSE_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, "PMC_SPI_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, "PMC_SQ_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS, "PMC_SQCS_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, "PMC_SX_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, "PMC_TA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, "PMC_TCP_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, "PMC_GCEA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, "PMC_GL1A_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C, "PMC_GL1C_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A, "PMC_GL2A_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C, "PMC_GL2C_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR, "PMC_GCR_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS, "PMC_GUS_dump.txt"},
      };
    return pmc_output_file_map;
}

output_file_map_t gfx12_pmc_map(){
    output_file_map_t
      pmc_output_file_map = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, "PMC_CPC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, "PMC_CPF_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, "PMC_GRBM_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, "PMC_SPI_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, "PMC_SQ_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, "PMC_SX_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, "PMC_TA_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, "PMC_GCEA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, "PMC_GL1A_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C, "PMC_GL1C_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A, "PMC_GL2A_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C, "PMC_GL2C_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR, "PMC_GCR_dump.txt"},
      };
    return pmc_output_file_map;
}

output_file_map_t gfx9_spm_pmc_map(){
    output_file_map_t
      pmc_output_file_map = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, "SPM_CPC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, "SPM_CPF_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, "SPM_GDS_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, "SPM_GRBM_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, "SPM_GRBMSE_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, "SPM_SPI_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, "SPM_SQ_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS, "SPM_SQCS_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, "SPM_SX_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, "SPM_TA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA, "SPM_TCA_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, "SPM_TCC_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, "SPM_TCP_dump.txt"},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, "SPM_TD_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, "SPM_MCVM_L2_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, "SPM_ATC_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, "SPM_ATC_L2_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, "SPM_GCEA_dump.txt"},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, "SPM_RPB_dump.txt"},
      };
    return pmc_output_file_map;
}

int main(int argc, char *argv[]) {

    if (argc < 4) {
    throw std::runtime_error("Expecting 3 args but received " +
                             std::to_string(argc - 1));
    return 1;
    }

    string mode = argv[1];
    string output_dir_path = argv[2];
    string gfxip_name = argv[3];

    const AgentInfo *test_param_agent_info;

    test_param_agent_info = GetAgentInfoByName(gfxip_name);
    if (test_param_agent_info == nullptr) {
      auto agent_names = GetAllAgentNames();
      std::string names = std::accumulate(std::next(agent_names.begin()), agent_names.end(), agent_names[0],
        [](std::string a, const std::string& b) {
            return a + ", " + b;
        });
      throw std::runtime_error("Agent not found: " + gfxip_name + ". Supported agents on this machine are: " + names);
      return 1;
    }

    if (mode == "pmc") {
      output_file_map_t pmc_output_file_map;
      
        if(gfxip_name.substr(0,4) == "gfx9"){
          pmc_output_file_map = gfx9_pmc_map();
        }
        else if(gfxip_name.substr(0,5) == "gfx10"){
          pmc_output_file_map = gfx10_pmc_map();
        }
        else if(gfxip_name.substr(0,5) == "gfx11") {
          pmc_output_file_map = gfx11_pmc_map();
        }
        else if(gfxip_name.substr(0,5) == "gfx12"){
          pmc_output_file_map = gfx12_pmc_map();
        }
        else
        {
          throw std::runtime_error("gfxip not found: " + gfxip_name);
          return 1;
        }


      for (const auto &[block_name, output_file_name] : pmc_output_file_map) {
        std::cout << std::string(20, '-') << "PMC DEBUG_TRACE for "
                  << output_file_name << std::string(20, '-') << std::endl;
        generate_individual_pmc_dump(test_param_agent_info, block_name,
                                    output_dir_path + "/" + output_file_name);
        std::cout << std::string(100, '-') << "\n\n";
      }
    } else if (mode == "spm") {
      auto pmc_output_file_map = gfx9_spm_pmc_map();
      for (const auto &[block_name, output_file_name] : pmc_output_file_map) {
        std::cout << std::string(20, '-') << "SPM DEBUG_TRACE for "
                  << output_file_name << std::string(20, '-') << std::endl;
        generate_spm_dump(test_param_agent_info, block_name,
                          output_dir_path + "/" + output_file_name);
        std::cout << std::string(100, '-') << "\n\n";
      }
    } else if (mode == "sqtt") {
      generate_sqtt_dump(test_param_agent_info, output_dir_path +
      "/SQTT_dump.txt");
    } else {
      throw std::runtime_error("Incorrect mode specified: " + mode +
                              ". Supported modes are pmc/spm/sqtt");
      return 1;
    }

    return 0;
};
