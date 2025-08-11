#ifndef HSA_RSRC_FACTORY_H_
#define HSA_RSRC_FACTORY_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>

#include "hsatimer.h"
#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "HSAILTool.h"


#define HSA_ARGUMENT_ALIGN_BYTES 16
#define HSA_QUEUE_ALIGN_BYTES 64
#define HSA_PACKET_ALIGN_BYTES 64

#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
    const char *emsg = 0; \
    hsa_status_string(status, &emsg); \
    printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>"); \
    exit(1); \
}

#define check_build(msg, status) \
if (status != STATUS_SUCCESS) { \
    printf("%s\n", msg); \
    exit(1); \
}

// Define required BRIG data structures.
typedef uint32_t BrigCodeOffset32_t;
typedef uint32_t BrigDataOffset32_t;
typedef uint16_t BrigKinds16_t;
typedef uint8_t BrigLinkage8_t;
typedef uint8_t BrigExecutableModifier8_t;
typedef BrigDataOffset32_t BrigDataOffsetString32_t;

/*
enum BrigKinds {
  BRIG_KIND_NONE = 0x0000,
  BRIG_KIND_DIRECTIVE_BEGIN = 0x1000,
  BRIG_KIND_DIRECTIVE_KERNEL = 0x1008,
};

typedef struct BrigBase BrigBase;
struct BrigBase {
  uint16_t byteCount;
  BrigKinds16_t kind;
};

typedef struct BrigExecutableModifier BrigExecutableModifier;
struct BrigExecutableModifier {
  BrigExecutableModifier8_t allBits;
};

typedef struct BrigDirectiveExecutable BrigDirectiveExecutable;
struct BrigDirectiveExecutable {
  uint16_t byteCount;
  BrigKinds16_t kind;
  BrigDataOffsetString32_t name;
  uint16_t outArgCount;
  uint16_t inArgCount;
  BrigCodeOffset32_t firstInArg;
  BrigCodeOffset32_t firstCodeBlockEntry;
  BrigCodeOffset32_t nextModuleEntry;
  uint32_t codeBlockEntryCount;
  BrigExecutableModifier modifier;
  BrigLinkage8_t linkage;
  uint16_t reserved;
};

typedef struct BrigData BrigData;
struct BrigData {
  uint32_t byteCount;
  uint8_t bytes[1];
};
*/

// Provide access to command line arguments passed in by user
extern uint32_t hsa_cmdline_arg_cnt;
extern char **hsa_cmdline_arg_list;

// Encapsulates information about a Hsa Agent such as its
// handle, name, max queue size, max wavefront size, etc.
typedef struct {

  // Handle of Agent
  hsa_agent_t dev_id;
  
  // Agent type - Cpu = 0, Gpu = 1 or Dsp = 2
  uint32_t dev_type;

  // Name of Agent whose length is less than 64
  char name[64];

  // Max size of Wavefront size
  uint32_t max_wave_size;

  // Max size of Queue buffer
  uint32_t max_queue_size;

  // Hsail profile supported by agent
  hsa_profile_t profile;

  // Memory region supporting kernel parameters
  hsa_region_t coarse_region;

  // Memory region supporting kernel arguments
  hsa_region_t kernarg_region;

} AgentInfo;

class HsaRsrcFactory {

 public:

  // Constructor of the class. Will initialize the Hsa Runtime and
  // query the system topology to get the list of Cpu and Gpu devices
  HsaRsrcFactory( );

  // Destructor of the class
  ~HsaRsrcFactory( );

  // Get the count of Hsa Gpu Agents available on the platform
  //
  // @return uint32_t Number of Gpu agents on platform
  //
  uint32_t GetCountOfGpuAgents( );

  // Get the count of Hsa Cpu Agents available on the platform
  //
  // @return uint32_t Number of Cpu agents on platform
  //
  uint32_t GetCountOfCpuAgents( );

  // Get the AgentInfo handle of a Gpu device
  //
  // @param idx Gpu Agent at specified index
  //
  // @param agent_info Output parameter updated with AgentInfo
  //
  // @return bool true if successful, false otherwise
  //
  bool GetGpuAgentInfo(uint32_t idx, AgentInfo **agent_info);

  // Get the AgentInfo handle of a Cpu device
  //
  // @param idx Cpu Agent at specified index
  //
  // @param agent_info Output parameter updated with AgentInfo
  //
  // @return bool true if successful, false otherwise
  //
  bool GetCpuAgentInfo(uint32_t idx, AgentInfo **agent_info);

  // Create a Queue object and return its handle. The queue object is expected
  // to support user requested number of Aql dispatch packets.
  //
  // @param agent_info Gpu Agent on which to create a queue object
  //
  // @param num_Pkts Number of packets to be held by queue
  //
  // @param queue Output parameter updated with handle of queue object
  //
  // @return bool true if successful, false otherwise
  //
  bool CreateQueue(AgentInfo *agent_info,
                   uint32_t num_pkts, hsa_queue_t **queue);

  // Create a Signal object and return its handle.
  //
  // @param value Initial value of signal object
  //
  // @param signal Output parameter updated with handle of signal object
  //
  // @return bool true if successful, false otherwise
  //
  bool CreateSignal(uint32_t value, hsa_signal_t *signal);

  // Allocate memory for use by a kernel of specified size in specified
  // agent's memory region. Currently supports Global segment whose Kernarg
  // flag set.
  //
  // @param agent_info Agent from whose memory region to allocate
  //
  // @param size Size of memory in terms of bytes
  //
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  //
  uint8_t* AllocateLocalMemory(AgentInfo *agent_info, size_t size);
  uint8_t* AllocateMemory(AgentInfo *agent_info, size_t size);

  bool TransferData(uint8_t *dest_buff, uint8_t *src_buff,
                    uint32_t length, bool host_to_dev);

  // Allocate memory tp pass kernel parameters.
  //
  // @param agent_info Agent from whose memory region to allocate
  //
  // @param size Size of memory in terms of bytes
  //
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  //
  uint8_t* AllocateSysMemory(AgentInfo *agent_info, size_t size);

  // Loads an Assembled Brig file and Finalizes it into Device Isa
  //
  // @param agent_info Gpu device for which to finalize
  //
  // @param brig_path File path of the Assembled Brig file
  //
  // @param kernel_name Name of the kernel to finalize
  //
  // @param code_desc Handle of finalized Code Descriptor that could
  // be used to submit for execution
  //
  // @return bool true if successful, false otherwise
  //
  bool LoadAndFinalize(AgentInfo *agent_info,
                       const char *brig_path, char *kernel_name,
                       hsa_executable_symbol_t *code_desc);

  // Add an instance of AgentInfo representing a Hsa Gpu agent
  void AddAgentInfo(AgentInfo *agent_info, bool gpu);

  // Returns the file path where brig files is located
  static char* GetBrigPath( );

  // Returns the number of compute units present on platform
  static uint32_t GetNumOfCUs( );

  // Returns the maximum number of waves that can be launched
  // per compute unit. The actual number that can be launched
  // is affected by resource availability
  static uint32_t GetNumOfWavesPerCU( );

  // Returns the number of work-items that can execute per wave
  static uint32_t GetNumOfWorkItemsPerWave( );
  
  // Returns the number of times kernel loop body should execute.
  static uint32_t GetKernelLoopCount();
  
  // Returns boolean flag to indicate if debug info should be printed
  static uint32_t GetPrintDebugInfo();

 private:
 
  // Number of queues to create
  uint32_t num_queues_;

  // Used to maintain a list of Hsa Queue handles
  std::vector<hsa_queue_t *> queue_list_;
 
  // Number of Signals to create
  uint32_t num_signals_;
 
  // Used to maintain a list of Hsa Signal handles
  std::vector<hsa_signal_t *> signal_list_;
 
  // Number of agents reported by platform
  uint32_t num_agents_;
 
  // Used to maintain a list of Hsa Gpu Agent Info
  std::vector<AgentInfo *> gpu_list_;
 
  // Used to maintain a list of Hsa Cpu Agent Info
  std::vector<AgentInfo *> cpu_list_;

  // Records the file path where Brig file is located.
  // Value is available only after an instance has been built.
  static char* brig_path_;
  static char* brig_path_key_;

  // Records the number of Compute units present on system.
  // Value is available only after an instance has been built.
  static uint32_t num_cus_;
  static char* num_cus_key_;

  // Records the number of waves that can be launched per Compute unit
  // Value is available only after an instance has been built.
  static uint32_t num_waves_;
  static char* num_waves_key_;

  // Records the number of work-items that can be packed into a wave
  // Value is available only after an instance has been built.
  static uint32_t num_workitems_;
  static char* num_workitems_key_;

  // Records the number of times kernel loop body should run. Value
  // is available only after an instance has been built.
  static uint32_t kernel_loop_count_;
  static char* kernel_loop_count_key_;

  // Records the number of times kernel loop body should run. Value
  // is available only after an instance has been built.
  static bool print_debug_info_;
  static char* print_debug_key_;

  // Print the various fields of Hsa Gpu Agents
  bool PrintGpuAgents( );
  
  // Process command line arguments. The method will capture
  // various user command line parameters for tests to use
  static void ProcessCmdline( );
  
  // Prints the help banner on user arg keys
  static void PrintHelpMsg( );

  // Maps an index for the user argument
  static uint32_t GetArgIndex(char *arg_value);

  HSAIL_ASM::Tool tool;
};

#endif  //  HSA_RSRC_FACTORY_H_
