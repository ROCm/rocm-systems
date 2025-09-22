#include "utilities.h"

/*
 * Prints no more than 256 elements of the given array.
 * Prints full array if length is less than 256.
 * Prints Array name followed by elements.
 */
template<typename T> 
void PrintArray(
    string header, 
    const T * data, 
    const int width,
    const int height) 
{
    cout<<"\n"<<header<<"\n";
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            cout<<data[i*width+j]<<" ";
        }
        cout<<"\n";
    }
    cout<<"\n";
}

template<typename T>
int IsPowerOf2(T val)
{
    long long _val = val;
    if((_val & (-_val))-_val == 0 && _val != 0)
        return 0;
    else
        return -1;
}


template<typename T>
T RoundToPowerOf2(T val)
{
    int bytes = sizeof(T);

    val--;
    for(int i = 0; i < bytes; i++)
        val |= val >> (1<<i);  
    val++;

    return val;
}

template<typename T> 
int FillRandom(
         T * arrayPtr, 
         const int width,
         const int height,
         const T rangeMin,
         const T rangeMax,
         unsigned int seed)
{
    if(!arrayPtr)
    {
        printf("Cannot fill array. NULL pointer.");
        return -1;
    }

    if(!seed)
        seed = (unsigned int)time(NULL);

    srand(seed);
    double range = double(rangeMax - rangeMin) + 1.0; 

    /* random initialisation of input */
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            int index = i*width + j;
            arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0)); 
        }

    return 0;
}

#if 0
//get a memory region that can be used for global memory allocations.
hsa_status_t get_global_region(hsa_region_t region, void* data) 
{
	hsa_region_segment_t segment;
	hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
	if (HSA_REGION_SEGMENT_GLOBAL == segment) 
	{
		hsa_region_t* ret = (hsa_region_t*) data;
		*ret = region;
	}
	return HSA_STATUS_SUCCESS;
}


/*
 * Finds the specified symbols offset in the specified brig_module.
 * If the symbol is found the function returns HSA_STATUS_SUCCESS, 
 * otherwise it returns HSA_STATUS_ERROR.
 */
hsa_status_t find_symbol_offset(hsa_ext_brig_module_t* brig_module, 
		char* symbol_name,
		hsa_ext_brig_code_section_offset32_t* offset) 
{

	/*  
	 * Get the data section 
	 */
	hsa_ext_brig_section_header_t* data_section_header = 
		brig_module->section[HSA_EXT_BRIG_SECTION_DATA];
	/*  
	 * Get the code section
	 */
	hsa_ext_brig_section_header_t* code_section_header =
		brig_module->section[HSA_EXT_BRIG_SECTION_CODE];

	/*  
	 * First entry into the BRIG code section
	 */
	BrigCodeOffset32_t code_offset = code_section_header->header_byte_count;
	BrigBase* code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
	while (code_offset != code_section_header->byte_count) 
	{
		if (code_entry->kind == BRIG_KIND_DIRECTIVE_KERNEL) 
		{
			/* 
			 * Now find the data in the data section
			 */
			BrigDirectiveExecutable* directive_kernel = (BrigDirectiveExecutable*) (code_entry);
			BrigDataOffsetString32_t data_name_offset = directive_kernel->name;
			BrigData* data_entry = (BrigData*)((char*) data_section_header + data_name_offset);
			if (!strncmp(symbol_name, (char*) data_entry->bytes, strlen(symbol_name))) 
			{
				*offset = code_offset;
				return HSA_STATUS_SUCCESS;
			}
		}
		code_offset += code_entry->byteCount;
		code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
	}   
	return HSA_STATUS_ERROR;
}
#endif

/*
 * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
 * and sets the value of data to the agent handle if it is.
 */
hsa_status_t find_gpu(hsa_agent_t agent, void *data) 
{
	if (data == NULL) 
	{
		return HSA_STATUS_ERROR_INVALID_ARGUMENT;
	}   
	hsa_device_type_t device_type;
	hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
	if (stat != HSA_STATUS_SUCCESS) 
	{
		return stat;
	}   
	if (device_type == HSA_DEVICE_TYPE_GPU) 
	{
		*((hsa_agent_t *)data) = agent;
	}   
	return HSA_STATUS_SUCCESS;
}


/*
 * Determines if a memory region can be used for kernarg
 * allocations.
 */
hsa_status_t get_memory_region(hsa_region_t region, void* data) 
{
	hsa_region_global_flag_t flags;
	hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);

	MemRegion *my_mem_region = (MemRegion *)data;
	
	if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
             my_mem_region->coarse_region = region;
       }
	
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) 
	{
		my_mem_region->kernarg_region= region;
	}   
	
	return HSA_STATUS_SUCCESS;
}

