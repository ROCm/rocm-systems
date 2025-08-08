#ifndef HSA_TEST_H
#define HSA_TEST_H

#include <map>
#include <string>
#include <vector>

#include "hsa.h"
#include "hsa_ext_amd.h"
#include "hsa_ext_finalize.h"

#include "HSAILTool.h"

class HsaTest {
 public:
  HsaTest(const char* test_name);
  virtual ~HsaTest();

  void Init();
  void Cleanup();

  virtual void Run() = 0;

 protected:
  static hsa_status_t IterateAgents(hsa_agent_t agent, void* data);
  static hsa_status_t IteratePools(hsa_amd_memory_pool_t pool, void* data);

  typedef struct AgentProps {
    AgentProps(hsa_agent_t);

    char name[64];
    char vendor_name[64];
    hsa_agent_feature_t feature;
    hsa_machine_model_t machine_model;
    hsa_profile_t profile;
    hsa_default_float_rounding_mode_t default_float_rounding_mode;
    hsa_default_float_rounding_mode_t base_profile_float_rounding_mode;
    bool fast_f16_operation;
    uint32_t wavefront_size;
    uint16_t workgroup_max_dim[3];
    uint32_t workgroup_max_size;
    hsa_dim3_t grid_max_dim;
    uint32_t grid_max_size;
    uint32_t fbarrier_max_size;
    uint32_t queue_max;
    uint32_t queue_min_size;
    uint32_t queue_max_size;
    hsa_queue_type_t queue_type;
    uint32_t node;
    hsa_device_type_t device_type;
    uint32_t cache_size[4];
    hsa_isa_t isa;
    uint8_t extensions[128];
    uint16_t version_major;
    uint16_t version_minor;
  } AgentProps;

  typedef struct PoolProps {
    PoolProps(hsa_amd_memory_pool_t pool);

    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_global_flag_t global_flag;
    size_t size;
    bool alloc_allowed;
    size_t alloc_granule;
    size_t alloc_alignment;
    bool all_accessible;
  } PoolProps;

  class Kernel {
   public:
    Kernel(hsa_agent_t agent, std::string hsail_file);

    virtual ~Kernel();

    uint64_t GetCodeHandle(const char* kernel_name);
    hsa_status_t GetScratchSize(uint32_t* size);

   protected:
    virtual void Initialize();

    virtual void Cleanup();

    bool CreateProgramFromHsailFile();

    bool CreateCodeObjectAndExecutable();

    HSAIL_ASM::Tool tool_;

    hsa_agent_t agent_;
    hsa_profile_t profile_;

    hsa_ext_program_t program_;
    hsa_code_object_t code_object_;
    hsa_executable_t executable_;
    hsa_executable_symbol_t kernel_symbol_;

    std::string hsail_file_;
  };

  virtual void GetGpuPeer(hsa_agent_t master,
                          std::vector<hsa_agent_t>& gpu_peers);
  virtual void* AllocateSystemMemory(bool fine_grain, size_t size);
  virtual void* AllocateLocalMemory(hsa_agent_t agent, size_t size);
  virtual void FreeMemory(void* ptr);

  virtual void LaunchPacket(hsa_queue_t& queue, hsa_packet_type_t type,
                            void* packet);

  virtual void PrintAgentInfo(AgentProps& prop);
  virtual void PrintPeers(hsa_agent_t agent);
  virtual void PrintPoolInfo(PoolProps& prop);

  std::string test_name_;

  std::vector<hsa_agent_t> cpus_;
  std::vector<hsa_agent_t> gpus_;

  std::map<uint64_t, hsa_amd_memory_pool_t> global_fine_;
  std::map<uint64_t, hsa_amd_memory_pool_t> global_coarse_;
  std::map<uint64_t, hsa_amd_memory_pool_t> group_;
};

#endif  // HSA_TEST_H
