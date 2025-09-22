#ifndef __HSA_BASE__
#define __HSA_BASE__


#include <vector>
#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "hsa_ext_amd.h"
#include "hsatimer.h"
#include "utilities.h"
#include "common.hpp"
#include "HSAILTool.h"

class HSA_UTIL{
    public:
	    HSA_UTIL();
	    ~HSA_UTIL();

	public:
	    void GetHsailNameAndKernelName(char *hail_file_name_full, char *hail_file_name_base, char *kernel_name);
	    bool HsaInit();
        void Close();
	double GetKernelTime();
	double GetSetupTime();
	void* AllocateLocalMemory(size_t size) ;
	void* AllocateSysMemory(size_t size);
	bool TransferData(void *dest, void *src, uint length, bool host_to_dev) ;
	
	double Run(int dim, int group_x, int group_y, int group_z, int s_size, int grid_x, int grid_y, int grid_z, void* kernel_args, int kernel_args_size);

	public:
		hsa_status_t err;
		uint32_t queue_size;
		hsa_agent_t device;
		MemRegion mem_region;
              //hsa_region_t kernarg_region;
		// Memory region supporting kernel parameters
             // hsa_region_t coarse_region;
		// Hsail profile supported by agent
              hsa_profile_t profile;

		char hail_file_name_full[128];
		char hail_file_name_base[128];
		char hsa_kernel_name[128];

		hsa_queue_t* command_queue;
		HSAIL_ASM::Tool tool;
		hsa_ext_module_t module;
		hsa_ext_program_t hsa_program;
		hsa_executable_t hsaExecutable;
	  hsa_executable_symbol_t kernelSymbol;
		hsa_code_object_t code_object;
		uint64_t codeHandle;
		hsa_signal_t hsa_signal;
		hsa_kernel_dispatch_packet_t dispatch_packet; 	
		hsa_region_t hsa_kernarg_region;

		PerfTimer base_timer;
		int base_kernel_time_idx;
		int base_setup_time_idx;
};


#endif

