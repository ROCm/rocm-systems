#ifndef __HSA_UTILITY__
#define __HSA_UTILITY__

#include <vector>
#include <thread>

#include "hsa.h"
#include "hsa_ext_finalize.h"

#include <string.h>
#include<iostream>
using namespace std;



#define HSA_ARGUMENT_ALIGN_BYTES 16

#if defined(_MSC_VER)
  #define ALIGNED_(x) __declspec(align(x))

#pragma warning(disable: 4800)
#pragma warning(disable: 4305) // truncation from 'double' to 'const float'
#pragma warning(disable: 4267) // conversion from 'size_t' to 'int', possible loss of data

typedef unsigned int uint;

#else
  #if defined(__GNUC__)
    #define ALIGNED_(x) __attribute__ ((aligned(x)))
  #endif // __GNUC__
#endif // _MSC_VER

#define SDK_FAILURE 1
#define SDK_SUCCESS 0

/*
#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
	printf("%s failed.\n", #msg); \
	exit(1); \
} else { \
	printf("%s succeeded.\n", #msg); \
}
*/
#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
	printf("%s failed.\n", #msg); \
	exit(1); \
} else { \
	; \
}

/*
 * Define required BRIG data structures.
 */

typedef uint32_t BrigCodeOffset32_t;

typedef uint32_t BrigDataOffset32_t;

typedef uint16_t BrigKinds16_t;

typedef uint8_t BrigLinkage8_t;

typedef uint8_t BrigExecutableModifier8_t;

typedef BrigDataOffset32_t BrigDataOffsetString32_t;

typedef struct {
  // memory region accessed by GPU only
  hsa_region_t coarse_region;

  // system memory access by gpu and cpu
  hsa_region_t kernarg_region;

} MemRegion;


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

struct float2
{
    float s0;
    float s1;


    float2 operator * (float2 &fl)
    {
        float2 temp;
        temp.s0 = (this->s0) * fl.s0;
        temp.s1 = (this->s1) * fl.s1;
        return temp;
    }

    float2 operator * (float scalar)
    {
        float2 temp;
        temp.s0 = (this->s0) * scalar;
        temp.s1 = (this->s1) * scalar;
        return temp;
    }

    float2 operator + (float2 &fl)
    {
        float2 temp;
        temp.s0 = (this->s0) + fl.s0;
        temp.s1 = (this->s1) + fl.s1;
        return temp;
    }
    
    float2 operator - (float2 fl)
    {
        float2 temp;
        temp.s0 = (this->s0) - fl.s0;
        temp.s1 = (this->s1) - fl.s1;
        return temp;
    }
};


struct uint2
{
    uint s0;
    uint s1;


    uint2 operator * (uint2 &fl)
    {
        uint2 temp;
        temp.s0 = (this->s0) * fl.s0;
        temp.s1 = (this->s1) * fl.s1;
        return temp;
    }

    uint2 operator * (float scalar)
    {
        uint2 temp;
        temp.s0 = (this->s0) * scalar;
        temp.s1 = (this->s1) * scalar;
        return temp;
    }

    uint2 operator + (uint2 &fl)
    {
        uint2 temp;
        temp.s0 = (this->s0) + fl.s0;
        temp.s1 = (this->s1) + fl.s1;
        return temp;
    }
    
    uint2 operator - (uint2 fl)
    {
        uint2 temp;
        temp.s0 = (this->s0) - fl.s0;
        temp.s1 = (this->s1) - fl.s1;
        return temp;
    }
};


/*
 * Prints no more than 256 elements of the given array.
 * Prints full array if length is less than 256.
 * Prints Array name followed by elements.
 */
template<typename T> void PrintArray(string header, const T * data, const int width, const int height);

template<typename T> int IsPowerOf2(T val);

template<typename T> T RoundToPowerOf2(T val);

template<typename T> int FillRandom(T * arrayPtr, const int width, const int height, const T rangeMin, const T rangeMax, unsigned int seed=123);

//get a memory region that can be used for global memory allocations.
hsa_status_t get_global_region(hsa_region_t region, void* data); 

/*
 * Finds the specified symbols offset in the specified brig_module.
 * If the symbol is found the function returns HSA_STATUS_SUCCESS, 
 * otherwise it returns HSA_STATUS_ERROR.
 */
 
//hsa_status_t find_symbol_offset(hsa_ext_brig_module_t* brig_module, char* symbol_name, hsa_ext_brig_code_section_offset32_t* offset);

/*
 * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
 * and sets the value of data to the agent handle if it is.
 */
hsa_status_t find_gpu(hsa_agent_t agent, void *data);

/*
 * Determines if a memory region can be used for kernarg
 * allocations.
 */
hsa_status_t get_memory_region(hsa_region_t region, void* data);

#endif
