#ifndef PMC_BUILDER_DUMP_H
#define PMC_BUILDER_DUMP_H

#include "pm4/cmd_builder.h"
#include "pm4/pmc_builder.h"

static constexpr bool CMD_BUFFER_DUMP_ENABLED = true;

using namespace std;

static inline pm4_builder::counters_vector
CountersVec(const hsa_ven_amd_aqlprofile_profile_t *profile,
            const aql_profile::Pm4Factory *pm4_factory) {
  pm4_builder::counters_vector vec;
  std::map<block_des_t, uint32_t, lt_block_des> index_map;
  for (const hsa_ven_amd_aqlprofile_event_t *p = profile->events;
       p < profile->events + profile->event_count; ++p) {
    const GpuBlockInfo *block_info = pm4_factory->GetBlockInfo(p);
    const block_des_t block_des = {pm4_factory->GetBlockInfo(p)->id,
                                   p->block_index};
    // Counting counter register index per block
    const auto ret = index_map.insert({block_des, 0});
    uint32_t &reg_index = ret.first->second;

    if (reg_index >= block_info->counter_count) {
      // throw event_exception("Event is out of block counter registers number
      // limit, ", *p);
      throw "Event is out of block counter registers number limit";
    }

    vec.push_back({p->counter_id, reg_index, block_des, block_info});

    ++reg_index;
  }

  if (pm4_factory->IsGFX10() && (vec.get_attr() & CounterBlockSqAttr) != 0 &&
      (vec.get_attr() & CounterBlockGRBMAttr) == 0) {
    hsa_ven_amd_aqlprofile_event_t grbm_event{
        .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM,
        .block_index = 0,
        .counter_id = 0};
    const GpuBlockInfo *block_info = pm4_factory->GetBlockInfo(&grbm_event);
    if (block_info == nullptr)
      return vec;
    const block_des_t block_des = {block_info->id, 0};
    const auto ret = index_map.insert({block_des, 0});
    uint32_t &reg_index = ret.first->second;
    vec.push_back({0, reg_index, block_des, block_info});
    reg_index++;
  }
  return vec;
}

aql_profile::Pm4Factory *get_pm4_factory(const AgentInfo *agent_info,
                                         aql_profile::gpu_id_t gpu_id) {
  aql_profile::Pm4Factory *pm4_factory =
      aql_profile::Pm4Factory::Create(agent_info, gpu_id, false);
  return pm4_factory;
}

hsa_ven_amd_aqlprofile_profile_t *create_profile(const AgentInfo *agent_info, hsa_ven_amd_aqlprofile_block_name_t block_name) {
  
  int num_events = 1;
  hsa_ven_amd_aqlprofile_event_t* events =
      new hsa_ven_amd_aqlprofile_event_t[num_events];

  for (int i = 0; i < num_events; ++i) {
    events[i].block_name = block_name;
    events[i].block_index = 0;
    events[i].counter_id = 0;
  }

  // Preparing the profile structure to get the packets
  hsa_ven_amd_aqlprofile_profile_t *profile =
      new hsa_ven_amd_aqlprofile_profile_t{
          agent_info->dev_id,
          HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,
          events,
          num_events,
          NULL,
          0,
          0,
          0};

  return profile;
}

void destroy_profile(hsa_ven_amd_aqlprofile_profile_t *profile) {
  delete[] profile->events;
  delete profile;
}

aql_profile::gpu_id_t get_gpu_id(std::string_view gfx_ip) {
  std::vector<std::pair<std::string, aql_profile::gpu_id_t>> gfxip_map = {
    {"gfx908", aql_profile::MI100_GPU_ID},
    {"gfx90a", aql_profile::MI200_GPU_ID},
    {"gfx900", aql_profile::GFX9_GPU_ID},
    {"gfx902", aql_profile::GFX9_GPU_ID},
    {"gfx906", aql_profile::GFX9_GPU_ID},
    {"gfx94", aql_profile::MI300_GPU_ID},
    {"gfx95", aql_profile::MI350_GPU_ID},
    {"gfx10", aql_profile::GFX10_GPU_ID},
    {"gfx11", aql_profile::GFX11_GPU_ID},
    {"gfx12", aql_profile::GFX12_GPU_ID},
  };

  for (const auto& [name, id] : gfxip_map) {
    if (gfx_ip.rfind(name, 0) == 0) {
      return id;
    }
  }

  return aql_profile::INVAL_GPU_ID;
}

void dump_cmd_buffer(ofstream &file_handle,
                            pm4_builder::CmdBuffer &cmdBuf) {
  uint32_t numElems = cmdBuf.Size() / sizeof(uint32_t);
  const uint32_t *start_ptr = static_cast<const uint32_t *>(cmdBuf.Data());
  if(CMD_BUFFER_DUMP_ENABLED){
    for (uint32_t i = 0; i < numElems; i++) file_handle << ", " << start_ptr[i];
    file_handle << endl;
  }
  cmdBuf.Clear();
}

inline void file_dump(std::ofstream& file_handle, const std::string& text) {
    if (CMD_BUFFER_DUMP_ENABLED)
        file_handle << text;
}

class pmc_builder_dump {
public:
  static void GenerateDump(string file_name,
                           pm4_builder::PmcBuilder *pmc_builder,
                           const pm4_builder::counters_vector &counters_vec) {
    ofstream file_handle;
    if (CMD_BUFFER_DUMP_ENABLED) file_handle.open(file_name, ios::out);
    string delim = ", ";

    pm4_builder::CmdBuffer cmd_buf;
    void *data_buffer = 0;

    file_dump(file_handle, "pmc_builder::Enable(cmd_buf)");
    pmc_builder->Enable(&cmd_buf);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "pmc_builder::Disable(cmd_buf)");
    pmc_builder->Disable(&cmd_buf);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "pmc_builder::WaitIdle(cmd_buf)");
    pmc_builder->WaitIdle(&cmd_buf);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "pmc_builder::Start(cmd_buf)");
    pmc_builder->Start(&cmd_buf, counters_vec);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "pmc_builder::Stop(cmd_buf)");
    pmc_builder->Stop(&cmd_buf, counters_vec);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "pmc_builder::Read(cmd_buf)");
    pmc_builder->Read(&cmd_buf, counters_vec, data_buffer);
    dump_cmd_buffer(file_handle, cmd_buf);

  }


};

class spm_builder_dump {
public:
  static void GenerateDump(string file_name,
                           pm4_builder::SpmBuilder *spm_builder,
                           const pm4_builder::counters_vector &counters_vec) {
    ofstream file_handle;
    if (CMD_BUFFER_DUMP_ENABLED) file_handle.open(file_name, ios::out);
    string delim = ", ";

    pm4_builder::CmdBuffer cmd_buf;
    pm4_builder::SpmConfig config;
    config.data_buffer_ptr = nullptr;
    config.data_buffer_size = 0;
    config.spm_kfd_mode = false;
    config.mi100 = false;

    file_dump(file_handle, "spm_builder::Begin(cmd_buf,SpmConfig{.spm_kfd_mode=false,.mi100=false},counters_vec)");
    spm_builder->Begin(&cmd_buf, &config, counters_vec);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "spm_builder::Stop(cmd_buf, SpmConfig{.spm_kfd_mode=false,.mi100=false})");
    spm_builder->End(&cmd_buf, &config);
    dump_cmd_buffer(file_handle, cmd_buf);

    config.spm_kfd_mode = true;
    config.mi100 = true;

    file_dump(file_handle, "spm_builder::Begin(cmd_buf,SpmConfig{.spm_kfd_mode=true,.mi100=true},counters_vec)");
    spm_builder->Begin(&cmd_buf, &config, counters_vec);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "spm_builder::Stop(cmd_buf,SpmConfig{.spm_kfd_mode=true,.mi100=true})");
    spm_builder->End(&cmd_buf, &config);
    dump_cmd_buffer(file_handle, cmd_buf);

  }

};


class sqtt_builder_dump {
public:
  static void GenerateDump(string file_name,
                           pm4_builder::SqttBuilder *sqtt_builder,
                           pm4_builder::TraceConfig* config) {
    ofstream file_handle;
    if (CMD_BUFFER_DUMP_ENABLED) file_handle.open(file_name, ios::out);
    string delim = ", ";

    pm4_builder::CmdBuffer cmd_buf;

    file_dump(file_handle, "sqtt_builder->Begin(&cmd_buf, config)");
    sqtt_builder->Begin(&cmd_buf, config);
    dump_cmd_buffer(file_handle, cmd_buf);

    file_dump(file_handle, "sqtt_builder->End(&cmd_buf, config)");
    sqtt_builder->End(&cmd_buf, config);
    dump_cmd_buffer(file_handle, cmd_buf);

  }

};

#endif // PMC_BUILDER_DUMP_H