/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/// @cond
#pragma once
#include <algorithm>
#include <cstring>
#include <future>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef NIC_EXEC_ENABLED
#    include <arpa/inet.h>
#    include <fcntl.h>
#    include <filesystem>
#    include <fstream>
#    include <infiniband/verbs.h>
#    include <stdbool.h>
#    include <stdint.h>
#    include <stdio.h>
#    include <string.h>
#    include <unistd.h>
#endif

#if defined(__NVCC__)
#    include <cuda_runtime.h>
#else
#    include <hip/hip_ext.h>
#    include <hip/hip_runtime.h>
#    include <hsa/hsa.h>
#    include <hsa/hsa_ext_amd.h>
#endif
/// @endcond

namespace TransferBench
{
using std::map;
using std::pair;
using std::set;
using std::vector;

constexpr char VERSION[] = "1.64";

/**
 * Enumeration of supported Executor types
 *
 * @note The Executor is the device used to perform a Transfer
 */
enum ExeType
{
    EXE_CPU         = 0,  ///<  CPU executor              (subExecutor = CPU thread)
    EXE_GPU_GFX     = 1,  ///<  GPU kernel-based executor (subExecutor = threadblock/CU)
    EXE_GPU_DMA     = 2,  ///<  GPU SDMA executor         (subExecutor = not supported)
    EXE_NIC         = 3,  ///<  NIC RDMA executor         (subExecutor = queue pair)
    EXE_NIC_NEAREST = 4   ///<  NIC RDMA nearest executor (subExecutor = queue pair)
};
char const ExeTypeStr[6] = "CGDIN";
inline bool
IsCpuExeType(ExeType e)
{
    return e == EXE_CPU;
}
inline bool
IsGpuExeType(ExeType e)
{
    return e == EXE_GPU_GFX || e == EXE_GPU_DMA;
}
inline bool
IsNicExeType(ExeType e)
{
    return e == EXE_NIC || e == EXE_NIC_NEAREST;
}

/**
 * A ExeDevice defines a specific Executor
 */
struct ExeDevice
{
    ExeType exeType;   ///< Executor type
    int32_t exeIndex;  ///< Executor index

    bool operator<(ExeDevice const& other) const
    {
        return (exeType < other.exeType) ||
               (exeType == other.exeType && exeIndex < other.exeIndex);
    }
};

/**
 * Enumeration of supported memory types
 *
 * @note These are possible types of memory to be used as sources/destinations
 */
enum MemType
{
    MEM_CPU          = 0,  ///< Coarse-grained pinned CPU memory
    MEM_GPU          = 1,  ///< Coarse-grained global GPU memory
    MEM_CPU_FINE     = 2,  ///< Fine-grained pinned CPU memory
    MEM_GPU_FINE     = 3,  ///< Fine-grained global GPU memory
    MEM_CPU_UNPINNED = 4,  ///< Unpinned CPU memory
    MEM_NULL         = 5,  ///< NULL memory - used for empty
    MEM_MANAGED      = 6,  ///< Managed memory
    MEM_CPU_CLOSEST  = 7,  ///< Coarse-grained pinned CPU memory indexed by closest GPU
};
char const MemTypeStr[9] = "CGBFUNMP";
inline bool
IsCpuMemType(MemType m)
{
    return (m == MEM_CPU || m == MEM_CPU_FINE || m == MEM_CPU_UNPINNED ||
            m == MEM_CPU_CLOSEST);
}
inline bool
IsGpuMemType(MemType m)
{
    return (m == MEM_GPU || m == MEM_GPU_FINE || m == MEM_MANAGED);
}

/**
 * A MemDevice indicates a memory type on a specific device
 */
struct MemDevice
{
    MemType memType;   ///< Memory type
    int32_t memIndex;  ///< Device index

    bool operator<(MemDevice const& other) const
    {
        return (memType < other.memType) ||
               (memType == other.memType && memIndex < other.memIndex);
    }
};

/**
 * A Transfer adds together data from zero or more sources then writes the sum to zero or
 * more desintations
 */
struct Transfer
{
    size_t            numBytes    = 0;   ///< Number of bytes to Transfer
    vector<MemDevice> srcs        = {};  ///< List of source memory devices
    vector<MemDevice> dsts        = {};  ///< List of destination memory devices
    ExeDevice         exeDevice   = {};  ///< Executor to use
    int32_t           exeSubIndex = -1;  ///< Executor subindex
    int numSubExecs = 0;  ///< Number of subExecutors to use for this Transfer
};

/**
 * General options
 */
struct GeneralOptions
{
    int numIterations = 10;  ///< Number of timed iterations to perform. If negative, run
                             ///< for -numIterations seconds instead
    int numSubIterations   = 1;  ///< Number of sub-iterations per iteration
    int numWarmups         = 3;  ///< Number of un-timed warmup iterations to perform
    int recordPerIteration = 0;  ///< Record per-iteration timing information
    int useInteractive     = 0;  ///< Pause for user-input before starting transfer loop
};

/**
 * Data options
 */
struct DataOptions
{
    int alwaysValidate = 0;  ///< Validate after each iteration instead of once at end
    int blockBytes = 256;    ///< Each subexecutor works on a multiple of this many bytes
    int byteOffset = 0;      ///< Byte-offset for memory allocations
    vector<float> fillPattern  = {};  ///< Pattern of floats used to fill source data
    vector<int>   fillCompress = {};  ///< Customized data patterns (overrides fillPattern
                                      ///< if non-empty)
    int validateDirect = 0;  ///< Validate GPU results directly instead of copying to host
    int validateSource = 0;  ///< Validate src GPU memory immediately after preparation
};

/**
 * GFX Executor options
 */
struct GfxOptions
{
    int blockOrder = 0;   ///< Determines how threadblocks are ordered (0=sequential,
                          ///< 1=interleaved, 2=random)
    int blockSize = 256;  ///< Size of each threadblock (must be multiple of 64)
    vector<uint32_t>    cuMask       = {};  ///< Bit-vector representing the CU mask
    vector<vector<int>> prefXccTable = {};  ///< 2D table with preferred XCD to use for a
                                            ///< specific [src][dst] GPU device
    int temporalMode =
        0;  ///< Non-temporal load/store mode 0=none, 1=load, 2=store, 3=both
    int unrollFactor   = 4;  ///< GFX-kernel unroll factor
    int useHipEvents   = 1;  ///< Use HIP events for timing GFX Executor
    int useMultiStream = 0;  ///< Use multiple streams for GFX
    int useSingleTeam  = 0;  ///< Team all subExecutors across the data array
    int waveOrder      = 0;  ///< GFX-kernel wavefront ordering
    int wordSize = 4;  ///< GFX-kernel packed data size (4=dwordx4, 2=dwordx2, 1=dwordx1)
};

/**
 * DMA Executor options
 */
struct DmaOptions
{
    int useHipEvents = 1;  ///< Use HIP events for timing DMA Executor
    int useHsaCopy   = 0;  ///< Use HSA copy instead of HIP copy to perform DMA
};

/**
 * NIC Executor options
 */
struct NicOptions
{
    vector<int> closestNics = {};     ///< Overrides the auto-detected closest NIC per GPU
    int         ibGidIndex  = -1;     ///< GID Index for RoCE NICs (-1 is auto)
    uint8_t     ibPort      = 1;      ///< NIC port number to be used
    int         ipAddressFamily = 4;  ///< 4=IPv4, 6=IPv6 (used for auto GID detection)
    int maxRecvWorkReq  = 16;   ///< Maximum number of recv work requests per queue pair
    int maxSendWorkReq  = 16;   ///< Maximum number of send work requests per queue pair
    int queueSize       = 100;  ///< Completion queue size
    int roceVersion     = 2;    ///< RoCE version (used for auto GID detection)
    int useRelaxedOrder = 1;    ///< Use relaxed ordering
    int useNuma         = 0;    ///< Switch to closest numa thread for execution
};

/**
 * Configuration options for performing Transfers
 */
struct ConfigOptions
{
    GeneralOptions general;  ///< General options
    DataOptions    data;     ///< Data options

    GfxOptions gfx;  ///< GFX executor options
    DmaOptions dma;  ///< DMA executor options
    NicOptions nic;  ///< NIC executor options
};

/**
 * Enumeration of possible error types
 */
enum ErrType
{
    ERR_NONE  = 0,  ///< No errors
    ERR_WARN  = 1,  ///< Warning - results may not be accurate
    ERR_FATAL = 2,  ///< Fatal error - results are invalid
};

/**
 * Enumeration of GID priority
 *
 * @note These are the GID types ordered in priority from lowest (0) to highest
 */
enum GidPriority
{
    UNKNOWN           = -1,  ///< Default
    ROCEV1_LINK_LOCAL = 0,   ///< RoCEv1 Link-local
    ROCEV2_LINK_LOCAL = 1,   ///< RoCEv2 Link-local fe80::/10
    ROCEV1_IPV6       = 2,   ///< RoCEv1 IPv6
    ROCEV2_IPV6       = 3,   ///< RoCEv2 IPv6
    ROCEV1_IPV4       = 4,   ///< RoCEv1 IPv4-mapped IPv6
    ROCEV2_IPV4       = 5,   ///< RoCEv2 IPv4-mapped IPv6 ::ffff:192.168.x.x
};

const char* GidPriorityStr[] = {
    "RoCEv1 Link-local", "RoCEv2 Link-local",       "RoCEv1 IPv6",
    "RoCEv2 IPv6",       "RoCEv1 IPv4-mapped IPv6", "RoCEv2 IPv4-mapped IPv6"
};

/**
 * ErrResult consists of error type and error message
 */
struct ErrResult
{
    ErrType     errType;  ///< Error type
    std::string errMsg;   ///< Error details

    ErrResult() = default;
#if defined(__NVCC__)
    ErrResult(cudaError_t err);
#else
    ErrResult(hipError_t err);
    ErrResult(hsa_status_t err);
#endif
    ErrResult(ErrType err);
    ErrResult(ErrType errType, const char* format, ...);
};

/**
 * Results for a single Executor
 */
struct ExeResult
{
    size_t numBytes;  ///< Total bytes transferred by this Executor
    double
        avgDurationMsec;  ///< Averaged duration for all the Transfers for this Executor
    double avgBandwidthGbPerSec;  ///< Average bandwidth for this Executor
    double sumBandwidthGbPerSec;  ///< Naive sum of individual Transfer average bandwidths
    vector<int> transferIdx;      ///< Indicies of Transfers this Executor executed
};

/**
 * Results for a single Transfer
 */
struct TransferResult
{
    size_t numBytes;         ///< Number of bytes transferred by this Transfer
    double avgDurationMsec;  ///< Duration for this Transfer, averaged over all timed
                             ///< iterations
    double
        avgBandwidthGbPerSec;  ///< Bandwidth for this Transfer based on averaged duration

    // Only filled in if recordPerIteration = 1
    vector<double> perIterMsec;  ///< Duration for each individual iteration
    vector<set<pair<int, int>>>
        perIterCUs;  ///< GFX-Executor only. XCC:CU used per iteration

    ExeDevice exeDevice;     ///< Tracks which executor performed this Transfer (e.g. for
                             ///< EXE_NIC_NEAREST)
    ExeDevice exeDstDevice;  ///< Tracks actual destination executor (only valid for
                             ///< EXE_NIC/EXE_NIC_NEAREST)
};

/**
 * TestResults contain timing results for a set of Transfers as a group as well as per
 * Executor and per Transfer timing information
 */
struct TestResults
{
    int    numTimedIterations;     ///< Number of iterations executed
    size_t totalBytesTransferred;  ///< Total bytes transferred per iteration
    double avgTotalDurationMsec;   ///< Wall-time (msec) to finish all Transfers (averaged
                                   ///< across all timed iterations)
    double avgTotalBandwidthGbPerSec;  ///< Bandwidth based on all Transfers and average
                                       ///< wall time
    double overheadMsec;  ///< Difference between total wall time and slowest executor

    map<ExeDevice, ExeResult> exeResults;  ///< Per Executor results
    vector<TransferResult>    tfrResults;  ///< Per Transfer results
    vector<ErrResult>         errResults;  ///< List of any errors/warnings that occurred
};

/**
 * Run a set of Transfers
 *
 * @param[in]  config     Configuration options
 * @param[in]  transfers  Set of Transfers to execute
 * @param[out] results    Timing results
 * @returns true if and only if Transfers were run successfully without any fatal errors
 */
bool
RunTransfers(ConfigOptions const& config, vector<Transfer> const& transfers,
             TestResults& results);

/**
 * Enumeration of implementation attributes
 */
enum IntAttribute
{
    ATR_GFX_MAX_BLOCKSIZE,  ///< Maximum blocksize for GFX executor
    ATR_GFX_MAX_UNROLL,     ///< Maximum unroll factor for GFX executor
};

enum StrAttribute
{
    ATR_SRC_PREP_DESCRIPTION  ///< Description of how source memory is prepared
};

/**
 * Query attributes (integer)
 *
 * @note This allows querying of implementation information such as limits
 *
 * @param[in] attribute   Attribute to query
 * @returns Value of the attribute
 */
int
GetIntAttribute(IntAttribute attribute);

/**
 * Query attributes (string)
 *
 * @note This allows query of implementation details such as limits
 *
 * @param[in] attrtibute Attribute to query
 * @returns Value of the attribute
 */
std::string
GetStrAttribute(StrAttribute attribute);

/**
 * Returns information about number of available available Executors
 *
 * @param[in] exeType    Executor type to query
 * @returns Number of detected Executors of exeType
 */
int
GetNumExecutors(ExeType exeType);

/**
 * Returns the number of possible Executor subindices
 *
 * @note For CPU, this is 0
 * @note For GFX, this refers to the number of XCDs
 * @note For DMA, this refers to the number of DMA engines
 *
 * @param[in] exeDevice The specific Executor to query
 * @returns Number of detected executor subindices
 */
int
GetNumExecutorSubIndices(ExeDevice exeDevice);

/**
 * Returns number of subExecutors for a given ExeDevice
 *
 * @param[in] exeDevice   The specific Executor to query
 * @returns Number of detected subExecutors for the given ExePair
 */
int
GetNumSubExecutors(ExeDevice exeDevice);

/**
 * Returns the index of the NUMA node closest to the given GPU
 *
 * @param[in] gpuIndex Index of the GPU to query
 * @returns NUMA node index closest to GPU gpuIndex, or -1 if unable to detect
 */
int
GetClosestCpuNumaToGpu(int gpuIndex);

/**
 * Returns the index of the NUMA node closest to the given NIC
 *
 * @param[in] nicIndex Index of the NIC to query
 * @returns NUMA node index closest to the NIC nicIndex, or -1 if unable to detect
 */
int
GetClosestCpuNumaToNic(int nicIndex);

/**
 * Returns the index of the NIC closest to the given GPU
 *
 * @param[in] gpuIndex Index of the GPU to query
 * @note This function is applicable when the IBV/RDMA executor is available
 * @returns IB Verbs capable NIC index closest to GPU gpuIndex, or -1 if unable to detect
 */
int
GetClosestNicToGpu(int gpuIndex);

/**
 * Helper function to parse a line containing Transfers into a vector of Transfers
 *
 * @param[in]  str       String containing description of Transfers
 * @param[out] transfers List of Transfers described by 'str'
 * @returns Information about any error that may have occured
 */
ErrResult
ParseTransfers(std::string str, std::vector<Transfer>& transfers);
};  // namespace TransferBench
//==========================================================================================
// End of TransferBench API
//==========================================================================================

// Redefinitions for CUDA compatibility
//==========================================================================================
#if defined(__NVCC__)

// ROCm specific
#    define wall_clock64 clock64
#    define gcnArchName  name

// Datatypes
#    define hipDeviceProp_t cudaDeviceProp
#    define hipError_t      cudaError_t
#    define hipEvent_t      cudaEvent_t
#    define hipStream_t     cudaStream_t

// Enumerations
#    define hipDeviceAttributeClockRate           cudaDevAttrClockRate
#    define hipDeviceAttributeMultiprocessorCount cudaDevAttrMultiProcessorCount
#    define hipErrorPeerAccessAlreadyEnabled      cudaErrorPeerAccessAlreadyEnabled
#    define hipFuncCachePreferShared              cudaFuncCachePreferShared
#    define hipMemcpyDefault                      cudaMemcpyDefault
#    define hipMemcpyDeviceToHost                 cudaMemcpyDeviceToHost
#    define hipMemcpyHostToDevice                 cudaMemcpyHostToDevice
#    define hipSuccess                            cudaSuccess

// Functions
#    define hipDeviceCanAccessPeer    cudaDeviceCanAccessPeer
#    define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#    define hipDeviceGetAttribute     cudaDeviceGetAttribute
#    define hipDeviceGetPCIBusId      cudaDeviceGetPCIBusId
#    define hipDeviceSetCacheConfig   cudaDeviceSetCacheConfig
#    define hipDeviceSynchronize      cudaDeviceSynchronize
#    define hipEventCreate            cudaEventCreate
#    define hipEventDestroy           cudaEventDestroy
#    define hipEventElapsedTime       cudaEventElapsedTime
#    define hipEventRecord            cudaEventRecord
#    define hipFree                   cudaFree
#    define hipGetDeviceCount         cudaGetDeviceCount
#    define hipGetDeviceProperties    cudaGetDeviceProperties
#    define hipGetErrorString         cudaGetErrorString
#    define hipHostFree               cudaFreeHost
#    define hipHostMalloc             cudaMallocHost
#    define hipMalloc                 cudaMalloc
#    define hipMallocManaged          cudaMallocManaged
#    define hipMemcpy                 cudaMemcpy
#    define hipMemcpyAsync            cudaMemcpyAsync
#    define hipMemset                 cudaMemset
#    define hipMemsetAsync            cudaMemsetAsync
#    define hipSetDevice              cudaSetDevice
#    define hipStreamCreate           cudaStreamCreate
#    define hipStreamDestroy          cudaStreamDestroy
#    define hipStreamSynchronize      cudaStreamSynchronize

// Define float2 addition operator for NVIDIA platform
__device__ inline float2&
operator+=(float2& a, const float2& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

// Define float4 addition operator for NVIDIA platform
__device__ inline float4&
operator+=(float4& a, const float4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
#endif

// Helper macro functions
//==========================================================================================

// Macro for collecting CU/SM GFX kernel is running on
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) ||              \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) ||              \
    defined(__gfx1201__)
#    define GetHwId(hwId) hwId = 0
#elif defined(__NVCC__)
#    define GetHwId(hwId) asm("mov.u32 %0, %smid;" : "=r"(hwId))
#else
#    define GetHwId(hwId)                                                                \
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hwId));
#endif

// Macro for collecting XCC GFX kernel is running on
#if defined(__gfx942__) || defined(__gfx950__)
#    define GetXccId(val)                                                                \
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(val));
#else
#    define GetXccId(val) val = 0
#endif

// Error check macro (NOTE: This will return even for ERR_WARN)
#define ERR_CHECK(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        ErrResult err = (cmd);                                                           \
        if(err.errType != ERR_NONE) return err;                                          \
    } while(0)

// Appends warn/fatal errors to a list, return false if fatal
#define ERR_APPEND(cmd, list)                                                            \
    do                                                                                   \
    {                                                                                    \
        ErrResult err = (cmd);                                                           \
        if(err.errType != ERR_NONE) list.push_back(err);                                 \
        if(err.errType == ERR_FATAL) return false;                                       \
    } while(0)

// Helper macros for calling RDMA functions and reporting errors
#ifdef VERBS_DEBUG
#    define IBV_CALL(__func__, ...)                                                      \
        do                                                                               \
        {                                                                                \
            int error = __func__(__VA_ARGS__);                                           \
            if(error != 0)                                                               \
            {                                                                            \
                return { ERR_FATAL,                                                      \
                         "Encountered IbVerbs error (%d) at line (%d) "                  \
                         "and function (%s)",                                            \
                         (error), __LINE__, #__func__ };                                 \
            }                                                                            \
        } while(0)

#    define IBV_PTR_CALL(__ptr__, __func__, ...)                                         \
        do                                                                               \
        {                                                                                \
            __ptr__ = __func__(__VA_ARGS__);                                             \
            if(__ptr__ == nullptr)                                                       \
            {                                                                            \
                return { ERR_FATAL,                                                      \
                         "Encountered IbVerbs nullptr error at line (%d) "               \
                         "and function (%s)",                                            \
                         __LINE__, #__func__ };                                          \
            }                                                                            \
        } while(0)
#else
#    define IBV_CALL(__func__, ...)                                                      \
        do                                                                               \
        {                                                                                \
            int error = __func__(__VA_ARGS__);                                           \
            if(error != 0)                                                               \
            {                                                                            \
                return { ERR_FATAL, "Encountered IbVerbs error (%d) in func (%s) ",      \
                         error, #__func__ };                                             \
            }                                                                            \
        } while(0)

#    define IBV_PTR_CALL(__ptr__, __func__, ...)                                         \
        do                                                                               \
        {                                                                                \
            __ptr__ = __func__(__VA_ARGS__);                                             \
            if(__ptr__ == nullptr)                                                       \
            {                                                                            \
                return { ERR_FATAL, "Encountered IbVerbs nullptr error in func (%s) ",   \
                         #__func__ };                                                    \
            }                                                                            \
        } while(0)
#endif

namespace TransferBench
{

/// @cond
// Helper functions ('hidden' in anonymous namespace)
//========================================================================================
namespace
{

// Constants
//========================================================================================

int constexpr MAX_BLOCKSIZE  = 1024;                // Max threadblock size
int constexpr MAX_WAVEGROUPS = MAX_BLOCKSIZE / 64;  // Max wavegroups/warps
int constexpr MAX_UNROLL     = 8;                   // Max unroll factor
int constexpr MAX_SRCS       = 8;                   // Max srcs per Transfer
int constexpr MAX_DSTS       = 8;                   // Max dsts per Transfer
int constexpr MEMSET_CHAR    = 75;                  // Value to memset (char)
float constexpr MEMSET_VAL   = 13323083.0f;         // Value to memset (double)

// Parsing-related functions
//========================================================================================

static ErrResult
CharToMemType(char const c, MemType& memType)
{
    char const* val = strchr(MemTypeStr, toupper(c));
    if(val)
    {
        memType = (MemType) (val - MemTypeStr);
        return ERR_NONE;
    }
    return { ERR_FATAL, "Unexpected memory type (%c)", c };
}

static ErrResult
CharToExeType(char const c, ExeType& exeType)
{
    char const* val = strchr(ExeTypeStr, toupper(c));
    if(val)
    {
        exeType = (ExeType) (val - ExeTypeStr);
        return ERR_NONE;
    }
    return { ERR_FATAL, "Unexpected executor type (%c)", c };
}

static ErrResult
ParseMemType(std::string const& token, std::vector<MemDevice>& memDevices)
{
    char    memTypeChar;
    int     offset = 0, memIndex, inc;
    MemType memType;
    bool    found = false;

    memDevices.clear();
    while(sscanf(token.c_str() + offset, " %c %d%n", &memTypeChar, &memIndex, &inc) == 2)
    {
        offset += inc;

        ErrResult err = CharToMemType(memTypeChar, memType);
        if(err.errType != ERR_NONE) return err;

        if(memType != MEM_NULL) memDevices.push_back({ memType, memIndex });
        found = true;
    }
    if(found) return ERR_NONE;
    return {
        ERR_FATAL,
        "Unable to parse memory type token %s.  Expected one of %s followed by an index",
        token.c_str(), MemTypeStr
    };
}

static ErrResult
ParseExeType(std::string const& token, ExeDevice& exeDevice, int& exeSubIndex)
{
    char exeTypeChar;
    exeSubIndex = -1;

    int numTokensParsed = sscanf(token.c_str(), " %c%d.%d", &exeTypeChar,
                                 &exeDevice.exeIndex, &exeSubIndex);
    if(numTokensParsed < 2)
    {
        return { ERR_FATAL,
                 "Unable to parse valid executor token (%s)."
                 "Expected one of %s followed by an index",
                 token.c_str(), ExeTypeStr };
    }
    return CharToExeType(exeTypeChar, exeDevice.exeType);
}

// Memory-related functions
//========================================================================================
// Enable peer access between two GPUs
static ErrResult
EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
    int canAccess;
    ERR_CHECK(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
    if(!canAccess)
        return { ERR_FATAL,
                 "Peer access is unavailable between GPU devices %d to %d."
                 "For AMD hardware, check IOMMU configuration",
                 peerDeviceId, deviceId };

    ERR_CHECK(hipSetDevice(deviceId));
    hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
    if(error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
    {
        return { ERR_FATAL, "Unable to enable peer to peer access from %d to %d (%s)",
                 deviceId, peerDeviceId, hipGetErrorString(error) };
    }
    return ERR_NONE;
}

// Allocate memory
static ErrResult
AllocateMemory(MemDevice memDevice, size_t numBytes, void** memPtr)
{
    if(numBytes == 0)
    {
        return { ERR_FATAL, "Unable to allocate 0 bytes" };
    }
    *memPtr = nullptr;

    MemType const& memType = memDevice.memType;

    if(IsCpuMemType(memType))
    {
        // Allocate host-pinned memory
        if(memType == MEM_CPU_FINE)
        {
#if defined(__NVCC__)
            return { ERR_FATAL,
                     "Fine-grained CPU memory not supported on NVIDIA platform" };
#else
            ERR_CHECK(hipHostMalloc((void**) memPtr, numBytes, hipHostMallocCoherent));
#endif
        }
        else if(memType == MEM_CPU || memType == MEM_CPU_CLOSEST)
        {
#if defined(__NVCC__)
            ERR_CHECK(hipHostMalloc((void**) memPtr, numBytes, 0));
#else
            ERR_CHECK(hipHostMalloc((void**) memPtr, numBytes, hipHostMallocNonCoherent));
#endif
        }
        else if(memType == MEM_CPU_UNPINNED)
        {
            *memPtr = malloc(numBytes);
            if(*memPtr == nullptr)
            {
                return { ERR_FATAL, "Failed to allocate %lu bytes of unpinned memory",
                         numBytes };
            }
        }

        // Clear the memory
        memset(*memPtr, 0, numBytes);
    }
    else if(IsGpuMemType(memType))
    {
        // Switch to the appropriate GPU
        ERR_CHECK(hipSetDevice(memDevice.memIndex));

        if(memType == MEM_GPU)
        {
            // Allocate GPU memory on appropriate device
            ERR_CHECK(hipMalloc((void**) memPtr, numBytes));
        }
        else if(memType == MEM_GPU_FINE)
        {
#if defined(__NVCC__)
            return { ERR_FATAL,
                     "Fine-grained GPU memory not supported on NVIDIA platform" };
#else
            int flag = hipDeviceMallocUncached;
            ERR_CHECK(hipExtMallocWithFlags((void**) memPtr, numBytes, flag));
#endif
        }
        else if(memType == MEM_MANAGED)
        {
            ERR_CHECK(hipMallocManaged((void**) memPtr, numBytes));
        }

        // Clear the memory
        ERR_CHECK(hipMemset(*memPtr, 0, numBytes));
        ERR_CHECK(hipDeviceSynchronize());
    }
    else
    {
        return { ERR_FATAL, "Unsupported memory type (%d)", memType };
    }
    return ERR_NONE;
}

// Deallocate memory
static ErrResult
DeallocateMemory(MemType memType, void* memPtr, size_t const bytes)
{
    // Avoid deallocating nullptr
    if(memPtr == nullptr)
        return { ERR_FATAL, "Attempted to free null pointer for %lu bytes", bytes };

    switch(memType)
    {
        case MEM_CPU:
        case MEM_CPU_FINE:
        case MEM_CPU_CLOSEST:
        {
            ERR_CHECK(hipHostFree(memPtr));
            break;
        }
        case MEM_CPU_UNPINNED:
        {
            free(memPtr);
            break;
        }
        case MEM_GPU:
        case MEM_GPU_FINE:
        case MEM_MANAGED:
        {
            ERR_CHECK(hipFree(memPtr));
            break;
        }
        default:
            return { ERR_FATAL, "Attempting to deallocate unrecognized memory type (%d)",
                     memType };
    }
    return ERR_NONE;
}

// HSA-related functions
//========================================================================================

#if !defined(__NVCC__)
// Get the hsa_agent_t associated with a ExeDevice
static ErrResult
GetHsaAgent(ExeDevice const& exeDevice, hsa_agent_t& agent)
{
    static bool                     isInitialized = false;
    static std::vector<hsa_agent_t> cpuAgents;
    static std::vector<hsa_agent_t> gpuAgents;

    int const& exeIndex = exeDevice.exeIndex;
    int const  numCpus  = GetNumExecutors(EXE_CPU);
    int const  numGpus  = GetNumExecutors(EXE_GPU_GFX);

    // Initialize results on first use
    if(!isInitialized)
    {
        hsa_amd_pointer_info_t info;
        info.size = sizeof(info);

        ErrResult err;
        int32_t*  tempBuffer;

        // Index CPU agents
        cpuAgents.clear();
        for(int i = 0; i < numCpus; i++)
        {
            ERR_CHECK(AllocateMemory({ MEM_CPU, i }, 1024, (void**) &tempBuffer));
            ERR_CHECK(hsa_amd_pointer_info(tempBuffer, &info, NULL, NULL, NULL));
            cpuAgents.push_back(info.agentOwner);
            ERR_CHECK(DeallocateMemory(MEM_CPU, tempBuffer, 1024));
        }

        // Index GPU agents
        gpuAgents.clear();
        for(int i = 0; i < numGpus; i++)
        {
            ERR_CHECK(AllocateMemory({ MEM_GPU, i }, 1024, (void**) &tempBuffer));
            ERR_CHECK(hsa_amd_pointer_info(tempBuffer, &info, NULL, NULL, NULL));
            gpuAgents.push_back(info.agentOwner);
            ERR_CHECK(DeallocateMemory(MEM_GPU, tempBuffer, 1024));
        }
        isInitialized = true;
    }

    switch(exeDevice.exeType)
    {
        case EXE_CPU:
            if(exeIndex < 0 || exeIndex >= numCpus)
                return { ERR_FATAL, "CPU index must be between 0 and %d inclusively",
                         numCpus - 1 };
            agent = cpuAgents[exeDevice.exeIndex];
            break;
        case EXE_GPU_GFX:
        case EXE_GPU_DMA:
            if(exeIndex < 0 || exeIndex >= numGpus)

                return { ERR_FATAL, "GPU index must be between 0 and %d inclusively",
                         numGpus - 1 };
            agent = gpuAgents[exeIndex];
            break;
        default:
            return { ERR_FATAL,
                     "Attempting to get HSA agent of unknown or unsupported executor "
                     "type (%d)",
                     exeDevice.exeType };
    }
    return ERR_NONE;
}

// Get the hsa_agent_t associated with a MemDevice
static ErrResult
GetHsaAgent(MemDevice const& memDevice, hsa_agent_t& agent)
{
    if(memDevice.memType == MEM_CPU_CLOSEST)
        return GetHsaAgent({ EXE_CPU, GetClosestCpuNumaToGpu(memDevice.memIndex) },
                           agent);
    if(IsCpuMemType(memDevice.memType))
        return GetHsaAgent({ EXE_CPU, memDevice.memIndex }, agent);
    if(IsGpuMemType(memDevice.memType))
        return GetHsaAgent({ EXE_GPU_GFX, memDevice.memIndex }, agent);
    return { ERR_FATAL, "Unable to get HSA agent for memDevice (%d,%d)",
             memDevice.memType, memDevice.memIndex };
}
#endif

// Setup validation-related functions
//========================================================================================

static ErrResult
GetActualExecutor(ConfigOptions const& cfg, ExeDevice const& origExeDevice,
                  ExeDevice& actualExeDevice)
{
    // By default, nothing needs to change
    actualExeDevice = origExeDevice;

    // When using NIC_NEAREST, remap to the closest NIC to the GPU
    if(origExeDevice.exeType == EXE_NIC_NEAREST)
    {
        actualExeDevice.exeType = EXE_NIC;

        if(cfg.nic.closestNics.size() > 0)
        {
            if(origExeDevice.exeIndex < 0 ||
               static_cast<size_t>(origExeDevice.exeIndex) >= cfg.nic.closestNics.size())
                return { ERR_FATAL, "NIC index is out of range (%d)",
                         origExeDevice.exeIndex };

            actualExeDevice.exeIndex = cfg.nic.closestNics[origExeDevice.exeIndex];
        }
        else
        {
            actualExeDevice.exeIndex = GetClosestNicToGpu(origExeDevice.exeIndex);
        }
    }
    return ERR_NONE;
}

// Validate that MemDevice exists
static ErrResult
CheckMemDevice(MemDevice const& memDevice)
{
    if(memDevice.memType == MEM_NULL) return ERR_NONE;

    if(IsCpuMemType(memDevice.memType) && memDevice.memType != MEM_CPU_CLOSEST)
    {
        int numCpus = GetNumExecutors(EXE_CPU);
        if(memDevice.memIndex < 0 || memDevice.memIndex >= numCpus)
            return { ERR_FATAL, "CPU index must be between 0 and %d (instead of %d)",
                     numCpus - 1, memDevice.memIndex };
        return ERR_NONE;
    }

    if(IsGpuMemType(memDevice.memType) || memDevice.memType == MEM_CPU_CLOSEST)
    {
        int numGpus = GetNumExecutors(EXE_GPU_GFX);
        if(memDevice.memIndex < 0 || memDevice.memIndex >= numGpus)
            return { ERR_FATAL, "GPU index must be between 0 and %d (instead of %d)",
                     numGpus - 1, memDevice.memIndex };
        if(memDevice.memType == MEM_CPU_CLOSEST)
        {
            if(GetClosestCpuNumaToGpu(memDevice.memIndex) == -1)
            {
                return { ERR_FATAL, "Unable to determine closest NUMA node for GPU %d",
                         memDevice.memIndex };
            }
        }
        return ERR_NONE;
    }
    return { ERR_FATAL, "Unsupported memory type (%d)", memDevice.memType };
}

// Validate configuration options - return trues if and only if an fatal error is detected
static bool
ConfigOptionsHaveErrors(ConfigOptions const& cfg, std::vector<ErrResult>& errors)
{
    // Check general options
    if(cfg.general.numWarmups < 0)
        errors.push_back(
            { ERR_FATAL, "[general.numWarmups] must be a non-negative number" });

    // Check data options
    if(cfg.data.blockBytes == 0 || cfg.data.blockBytes % 4)
        errors.push_back({ ERR_FATAL,
                           "[data.blockBytes] must be positive multiple of %lu",
                           sizeof(float) });
    if(cfg.data.byteOffset < 0 || cfg.data.byteOffset % sizeof(float))
        errors.push_back({ ERR_FATAL,
                           "[data.byteOffset] must be positive multiple of %lu",
                           sizeof(float) });
    if(cfg.data.fillCompress.size() > 0 && cfg.data.fillPattern.size() > 0)
        errors.push_back({ ERR_WARN, "[data.fillCompress] will override "
                                     "[data.fillPattern] when both are specified" });
    if(cfg.data.fillCompress.size() > 0)
    {
        int sum = 0;
        for(int bin : cfg.data.fillCompress)
            sum += bin;
        if(sum != 100)
        {
            errors.push_back(
                { ERR_FATAL, "[data.fillCompress] values must add up to 100" });
        }
    }
    if(cfg.data.fillCompress.size() > 5)
    {
        errors.push_back(
            { ERR_FATAL, "[data.fillCompress] may only have up to 5 values" });
    }

    // Check GFX options
    if(cfg.gfx.blockOrder < 0 || cfg.gfx.blockOrder > 2)
        errors.push_back({ ERR_FATAL, "[gfx.blockOrder] must be 0 for sequential, 1 for "
                                      "interleaved, or 2 for random" });

    if(cfg.gfx.useMultiStream && cfg.gfx.blockOrder > 0)
        errors.push_back(
            { ERR_WARN,
              "[gfx.blockOrder] will be ignored when running in multi-stream mode" });

    int gfxMaxBlockSize = GetIntAttribute(ATR_GFX_MAX_BLOCKSIZE);
    if(cfg.gfx.blockSize < 0 || cfg.gfx.blockSize % 64 ||
       cfg.gfx.blockSize > gfxMaxBlockSize)
        errors.push_back(
            { ERR_FATAL,
              "[gfx.blockSize] must be positive multiple of 64 less than or equal to %d",
              gfxMaxBlockSize });

    if(cfg.gfx.temporalMode < 0 || cfg.gfx.temporalMode > 3)
        errors.push_back(
            { ERR_FATAL,
              "[gfx.temporalMode] must be non-negative and less than or equal to 3" });

#if defined(__NVCC__)
    if(cfg.gfx.temporalMode > 0)
        errors.push_back(
            { ERR_FATAL, "[gfx.temporalMode] is not supported on NVIDIA hardware" });
#endif

    int gfxMaxUnroll = GetIntAttribute(ATR_GFX_MAX_UNROLL);
    if(cfg.gfx.unrollFactor < 0 || cfg.gfx.unrollFactor > gfxMaxUnroll)
        errors.push_back(
            { ERR_FATAL,
              "[gfx.unrollFactor] must be non-negative and less than or equal to %d",
              gfxMaxUnroll });
    if(cfg.gfx.waveOrder < 0 || cfg.gfx.waveOrder >= 6)
        errors.push_back(
            { ERR_FATAL, "[gfx.waveOrder] must be non-negative and less than 6" });

    if(!(cfg.gfx.wordSize == 1 || cfg.gfx.wordSize == 2 || cfg.gfx.wordSize == 4))
        errors.push_back({ ERR_FATAL, "[gfx.wordSize] must be either 1, 2 or 4" });

    int                        numGpus = GetNumExecutors(EXE_GPU_GFX);
    int                        numXccs = GetNumExecutorSubIndices({ EXE_GPU_GFX, 0 });
    vector<vector<int>> const& table   = cfg.gfx.prefXccTable;

    if(!table.empty())
    {
        if(static_cast<int>(table.size()) != numGpus)
        {
            errors.push_back({ ERR_FATAL, "[gfx.prefXccTable] must be have size %dx%d",
                               numGpus, numGpus });
        }
        else
        {
            for(size_t i = 0; i < table.size(); i++)
            {
                if(static_cast<int>(table[i].size()) != numGpus)
                {
                    errors.push_back({ ERR_FATAL,
                                       "[gfx.prefXccTable] must be have size %dx%d",
                                       numGpus, numGpus });
                    break;
                }
                else
                {
                    for(auto x : table[i])
                    {
                        if(x < 0 || x >= numXccs)
                        {
                            errors.push_back({ ERR_FATAL,
                                               "[gfx.prefXccTable] must contain values "
                                               "between 0 and %d",
                                               numXccs - 1 });
                            break;
                        }
                    }
                }
            }
        }
    }

    // Check NIC options
#ifdef NIC_EXEC_ENABLED
    int numNics = GetNumExecutors(EXE_NIC);
    for(auto const& nic : cfg.nic.closestNics)
        if(nic < 0 || nic >= numNics)
            errors.push_back({ ERR_FATAL,
                               "NIC index (%d) in user-specified closest NIC list must "
                               "be between 0 and %d",
                               nic, numNics - 1 });

    size_t closetNicsSize = cfg.nic.closestNics.size();
    if(closetNicsSize > 0 && closetNicsSize < numGpus)
        errors.push_back({ ERR_FATAL,
                           "User-specified closest NIC list must match GPU count of %d",
                           numGpus });
#endif

        // NVIDIA specific
#if defined(__NVCC__)
    if(cfg.data.validateDirect)
        errors.push_back(
            { ERR_FATAL, "[data.validateDirect] is not supported on NVIDIA hardware" });
#else
    // AMD specific
    // Check for largeBar enablement on GPUs
    for(int i = 0; i < numGpus; i++)
    {
        int        isLargeBar = 0;
        hipError_t err =
            hipDeviceGetAttribute(&isLargeBar, hipDeviceAttributeIsLargeBar, i);
        if(err != hipSuccess)
        {
            errors.push_back(
                { ERR_FATAL, "Unable to query if GPU %d has largeBAR enabled", i });
        }
        else if(!isLargeBar)
        {
            errors.push_back({ ERR_WARN,
                               "Large BAR is not enabled for GPU %d in BIOS. "
                               "Large BAR is required to enable multi-gpu data access",
                               i });
        }
    }
#endif

    // Check for fatal errors
    for(auto const& err : errors)
        if(err.errType == ERR_FATAL) return true;
    return false;
}

// Validate Transfers to execute - returns true if and only if fatal error detected
static bool
TransfersHaveErrors(ConfigOptions const& cfg, std::vector<Transfer> const& transfers,
                    std::vector<ErrResult>& errors)
{
    int numCpus = GetNumExecutors(EXE_CPU);
    int numGpus = GetNumExecutors(EXE_GPU_GFX);
#ifdef NIC_EXEC_ENABLED
    int numNics = GetNumExecutors(EXE_NIC);
#endif
    (void) numCpus;  // May be unused
    (void) numGpus;  // May be unused

    std::set<ExeDevice>      executors;
    std::map<ExeDevice, int> transferCount;
    std::map<ExeDevice, int> useSubIndexCount;
    std::map<ExeDevice, int> totalSubExecs;

    // Per-Transfer checks
    for(size_t i = 0; i < transfers.size(); i++)
    {
        Transfer const& t = transfers[i];

        if(t.numBytes == 0)
            errors.push_back(
                { ERR_FATAL, "Transfer %d: Cannot perform 0-byte transfers", i });

        if(t.exeDevice.exeType == EXE_GPU_GFX || t.exeDevice.exeType == EXE_CPU)
        {
            size_t const N              = t.numBytes / sizeof(float);
            int const    targetMultiple = cfg.data.blockBytes / sizeof(float);
            int const    maxSubExecToUse =
                std::min((size_t) (N + targetMultiple - 1) / targetMultiple,
                         (size_t) t.numSubExecs);

            if(maxSubExecToUse < t.numSubExecs)
                errors.push_back({ ERR_WARN,
                                   "Transfer %d data size is too small - will only use "
                                   "%d of %d subexecutors",
                                   i, maxSubExecToUse, t.numSubExecs });
        }

        // Check sources and destinations
        if(t.srcs.empty() && t.dsts.empty())
            errors.push_back(
                { ERR_FATAL, "Transfer %d: Must have at least one source or destination",
                  i });

        for(size_t j = 0; j < t.srcs.size(); j++)
        {
            ErrResult err = CheckMemDevice(t.srcs[j]);
            if(err.errType != ERR_NONE)
                errors.push_back(
                    { ERR_FATAL, "Transfer %d: SRC %lu: %s", i, j, err.errMsg.c_str() });
        }
        for(size_t j = 0; j < t.dsts.size(); j++)
        {
            ErrResult err = CheckMemDevice(t.dsts[j]);
            if(err.errType != ERR_NONE)
                errors.push_back(
                    { ERR_FATAL, "Transfer %d: DST %lu: %s", i, j, err.errMsg.c_str() });
        }

        // Check executor
        executors.insert(t.exeDevice);
        transferCount[t.exeDevice]++;
        switch(t.exeDevice.exeType)
        {
            case EXE_CPU:
                if(t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numCpus)
                    errors.push_back({ ERR_FATAL,
                                       "Transfer %d: CPU index must be between 0 and %d "
                                       "(instead of %d)",
                                       i, numCpus - 1, t.exeDevice.exeIndex });
                break;
            case EXE_GPU_GFX:
                if(t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numGpus)
                {
                    errors.push_back({ ERR_FATAL,
                                       "Transfer %d: GFX index must be between 0 and %d "
                                       "(instead of %d)",
                                       i, numGpus - 1, t.exeDevice.exeIndex });
                }
                else
                {
                    if(t.exeSubIndex != -1)
                    {
#if defined(__NVCC__)
                        errors.push_back({ ERR_FATAL,
                                           "Transfer %d: GFX executor subindex not "
                                           "supported on NVIDIA hardware",
                                           i });
#else
                        useSubIndexCount[t.exeDevice]++;
                        int numSubIndices = GetNumExecutorSubIndices(t.exeDevice);
                        if(t.exeSubIndex >= numSubIndices)
                            errors.push_back({ ERR_FATAL,
                                               "Transfer %d: GFX subIndex (XCC) must be "
                                               "between 0 and %d",
                                               i, numSubIndices - 1 });
#endif
                    }
                }
                break;
            case EXE_GPU_DMA:
                if(t.srcs.size() != 1 || t.dsts.size() != 1)
                {
                    errors.push_back({ ERR_FATAL,
                                       "Transfer %d: DMA executor must have exactly 1 "
                                       "source and 1 destination",
                                       i });
                }

                if(t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numGpus)
                {
                    errors.push_back({ ERR_FATAL,
                                       "Transfer %d: DMA index must be between 0 and %d "
                                       "(instead of %d)",
                                       i, numGpus - 1, t.exeDevice.exeIndex });
                    // Cannot proceed with any further checks
                    continue;
                }

                if(t.exeSubIndex != -1)
                {
#if defined(__NVCC__)
                    errors.push_back({ ERR_FATAL,
                                       "Transfer %d: DMA executor subindex not supported "
                                       "on NVIDIA hardware",
                                       i });
#else
                    useSubIndexCount[t.exeDevice]++;
                    int numSubIndices = GetNumExecutorSubIndices(t.exeDevice);
                    if(t.exeSubIndex >= numSubIndices)
                        errors.push_back({ ERR_FATAL,
                                           "Transfer %d: DMA subIndex (engine) must be "
                                           "between 0 and %d",
                                           i, numSubIndices - 1 });

                    // Check that engine Id exists between agents
                    hsa_agent_t srcAgent, dstAgent;
                    ErrResult   err;
                    err = GetHsaAgent(t.srcs[0], srcAgent);
                    if(err.errType != ERR_NONE)
                    {
                        errors.push_back(err);
                        if(err.errType == ERR_FATAL) break;
                    }
                    err = GetHsaAgent(t.dsts[0], dstAgent);
                    if(err.errType != ERR_NONE)
                    {
                        errors.push_back(err);
                        if(err.errType == ERR_FATAL) break;
                    }

                    // Skip check of engine Id mask for self copies
                    if(srcAgent.handle != dstAgent.handle)
                    {
                        uint32_t engineIdMask = 0;
                        err = hsa_amd_memory_copy_engine_status(dstAgent, srcAgent,
                                                                &engineIdMask);
                        if(err.errType != ERR_NONE)
                        {
                            errors.push_back(err);
                            if(err.errType == ERR_FATAL) break;
                        }
                        hsa_amd_sdma_engine_id_t sdmaEngineId =
                            (hsa_amd_sdma_engine_id_t) (1U << t.exeSubIndex);
                        if(!(sdmaEngineId & engineIdMask))
                        {
                            errors.push_back({ ERR_FATAL,
                                               "Transfer %d: DMA %d.%d does not exist or "
                                               "cannot copy between src/dst",
                                               i, t.exeDevice.exeIndex, t.exeSubIndex });
                        }
                    }
#endif
                }

                if(!IsGpuMemType(t.srcs[0].memType) && !IsGpuMemType(t.dsts[0].memType))
                {
                    errors.push_back({ ERR_WARN,
                                       "Transfer %d: No GPU memory for source or "
                                       "destination.  Copy might not execute on DMA %d",
                                       i, t.exeDevice.exeIndex });
                }
                else
                {
                    // Currently HIP will use src agent if source memory is GPU, otherwise
                    // dst agent
                    if(IsGpuMemType(t.srcs[0].memType))
                    {
                        if(t.srcs[0].memIndex != t.exeDevice.exeIndex)
                        {
                            errors.push_back(
                                { ERR_WARN,
                                  "Transfer %d: DMA executor will automatically switch "
                                  "to using the source memory device (%d) not (%d)",
                                  i, t.srcs[0].memIndex, t.exeDevice.exeIndex });
                        }
                    }
                    else if(t.dsts[0].memIndex != t.exeDevice.exeIndex)
                    {
                        errors.push_back(
                            { ERR_WARN,
                              "Transfer %d: DMA executor will automatically switch to "
                              "using the destination memory device (%d) not (%d)",
                              i, t.dsts[0].memIndex, t.exeDevice.exeIndex });
                    }
                }
                break;
            case EXE_NIC:
#ifdef NIC_EXEC_ENABLED
            {
                int srcIndex = t.exeDevice.exeIndex;
                int dstIndex = t.exeSubIndex;
                if(srcIndex < 0 || srcIndex >= numNics)
                    errors.push_back({ ERR_FATAL,
                                       "Transfer %d: src NIC executor indexes an "
                                       "out-of-range NIC (%d)",
                                       i, srcIndex });
                if(dstIndex < 0 || dstIndex >= numNics)
                    errors.push_back({ ERR_FATAL,
                                       "Transfer %d: dst NIC executor indexes an "
                                       "out-of-range NIC (%d)",
                                       i, dstIndex });
            }
#else
                errors.push_back(
                    { ERR_FATAL,
                      "Transfer %d: NIC executor is requested but is not available", i });
#endif
            break;
            case EXE_NIC_NEAREST:
#ifdef NIC_EXEC_ENABLED
            {
                ExeDevice srcExeDevice;
                ErrResult errSrc = GetActualExecutor(cfg, t.exeDevice, srcExeDevice);
                if(errSrc.errType != ERR_NONE) errors.push_back(errSrc);
                ExeDevice dstExeDevice;
                ErrResult errDst = GetActualExecutor(
                    cfg, { t.exeDevice.exeType, t.exeSubIndex }, dstExeDevice);
                if(errDst.errType != ERR_NONE) errors.push_back(errDst);
            }
#else
                errors.push_back(
                    { ERR_FATAL,
                      "Transfer %d: NIC executor is requested but is not available", i });
#endif
            break;
        }

        // Check subexecutors
        if(t.numSubExecs <= 0)
            errors.push_back(
                { ERR_FATAL, "Transfer %d: # of subexecutors must be positive", i });
        else
            totalSubExecs[t.exeDevice] += t.numSubExecs;
    }

    int gpuMaxHwQueues = 4;
    if(getenv("GPU_MAX_HW_QUEUES")) gpuMaxHwQueues = atoi(getenv("GPU_MAX_HW_QUEUES"));

    // Aggregate checks
    for(auto const& exeDevice : executors)
    {
        switch(exeDevice.exeType)
        {
            case EXE_CPU:
            {
                // Check total number of subexecutors requested
                int numCpuSubExec = GetNumSubExecutors(exeDevice);
                if(totalSubExecs[exeDevice] > numCpuSubExec)
                    errors.push_back(
                        { ERR_WARN,
                          "CPU %d requests %d total cores however only %d available. "
                          "Serialization will occur",
                          exeDevice.exeIndex, totalSubExecs[exeDevice], numCpuSubExec });
                break;
            }
            case EXE_GPU_GFX:
            {
                // Check total number of subexecutors requested
                int numGpuSubExec = GetNumSubExecutors(exeDevice);
                if(totalSubExecs[exeDevice] > numGpuSubExec)
                    errors.push_back(
                        { ERR_WARN,
                          "GPU %d requests %d total CUs however only %d available. "
                          "Serialization will occur",
                          exeDevice.exeIndex, totalSubExecs[exeDevice], numGpuSubExec });
                // Check that if executor subindices are used, all Transfers specify
                // executor subindices
                if(useSubIndexCount[exeDevice] > 0 &&
                   useSubIndexCount[exeDevice] != transferCount[exeDevice])
                {
                    errors.push_back({ ERR_FATAL,
                                       "GPU %d specifies XCC on only %d of %d Transfers. "
                                       "Must either specific none or all",
                                       exeDevice.exeIndex, useSubIndexCount[exeDevice],
                                       transferCount[exeDevice] });
                }

                if(cfg.gfx.useMultiStream && transferCount[exeDevice] > gpuMaxHwQueues)
                {
                    errors.push_back({ ERR_WARN,
                                       "GPU %d attempting %d parallel transfers, however "
                                       "GPU_MAX_HW_QUEUES only set to %d",
                                       exeDevice.exeIndex, transferCount[exeDevice],
                                       gpuMaxHwQueues });
                }
                break;
            }
            case EXE_GPU_DMA:
            {
                // Check that if executor subindices are used, all Transfers specify
                // executor subindices
                if(useSubIndexCount[exeDevice] > 0 &&
                   useSubIndexCount[exeDevice] != transferCount[exeDevice])
                {
                    errors.push_back(
                        { ERR_FATAL,
                          "DMA %d specifies engine on only %d of %d Transfers. "
                          "Must either specific none or all",
                          exeDevice.exeIndex, useSubIndexCount[exeDevice],
                          transferCount[exeDevice] });
                }
                if(transferCount[exeDevice] > gpuMaxHwQueues)
                {
                    errors.push_back({ ERR_WARN,
                                       "DMA %d attempting %d parallel transfers, however "
                                       "GPU_MAX_HW_QUEUES only set to %d",
                                       exeDevice.exeIndex, transferCount[exeDevice],
                                       gpuMaxHwQueues });
                }

                char* enableSdma = getenv("HSA_ENABLE_SDMA");
                if(enableSdma && !strcmp(enableSdma, "0"))
                    errors.push_back(
                        { ERR_WARN,
                          "DMA functionality disabled due to environment variable "
                          "HSA_ENABLE_SDMA=0. "
                          "DMA %d copies will fallback to blit (GFX) kernels",
                          exeDevice.exeIndex });
                break;
            }
            default: break;
        }
    }

    // Check for fatal errors
    for(auto const& err : errors)
        if(err.errType == ERR_FATAL) return true;
    return false;
}

// Internal data structures
//========================================================================================

// Parameters for each SubExecutor
struct SubExecParam
{
    // Inputs
    size_t  N;               ///< Number of floats this subExecutor works on
    int     numSrcs;         ///< Number of source arrays
    int     numDsts;         ///< Number of destination arrays
    float*  src[MAX_SRCS];   ///< Source array pointers
    float*  dst[MAX_DSTS];   ///< Destination array pointers
    int32_t preferredXccId;  ///< XCC ID to execute on (GFX only)

    // Prepared
    int teamSize;  ///< Index of this sub executor amongst team
    int teamIdx;   ///< Size of team this sub executor is part of

    // Outputs
    long long startCycle;  ///< Start timestamp for in-kernel timing (GPU-GFX executor)
    long long stopCycle;   ///< Stop  timestamp for in-kernel timing (GPU-GFX executor)
    uint32_t  hwId;        ///< Hardware ID
    uint32_t  xccId;       ///< XCC ID
};

// Internal resources allocated per Transfer
struct TransferResources
{
    int                  transferIdx;      ///< The associated Transfer
    size_t               numBytes;         ///< Number of bytes to Transfer
    vector<float*>       srcMem;           ///< Source memory
    vector<float*>       dstMem;           ///< Destination memory
    vector<SubExecParam> subExecParamCpu;  ///< Defines subarrays for each subexecutor
    vector<int>          subExecIdx;       ///< Indices into subExecParamGpu
    int                  numaNode;         ///< NUMA node to use for this Transfer

    // For GFX executor
    SubExecParam* subExecParamGpuPtr;

    // For targeted-SDMA
#if !defined(__NVCC__)
    hsa_agent_t              dstAgent;      ///< DMA destination memory agent
    hsa_agent_t              srcAgent;      ///< DMA source memory agent
    hsa_signal_t             signal;        ///< HSA signal for completion
    hsa_amd_sdma_engine_id_t sdmaEngineId;  ///< DMA engine ID
#endif

// For IBV executor
#ifdef NIC_EXEC_ENABLED
    int                 srcNicIndex;    ///< SRC NIC index
    int                 dstNicIndex;    ///< DST NIC index
    ibv_context*        srcContext;     ///< Device context for SRC NIC
    ibv_context*        dstContext;     ///< Device context for DST NIC
    ibv_pd*             srcProtect;     ///< Protection domain for SRC NIC
    ibv_pd*             dstProtect;     ///< Protection domain for DST NIC
    ibv_cq*             srcCompQueue;   ///< Completion queue for SRC NIC
    ibv_cq*             dstCompQueue;   ///< Completion queue for DST NIC
    ibv_port_attr       srcPortAttr;    ///< Port attributes for SRC NIC
    ibv_port_attr       dstPortAttr;    ///< Port attributes for DST NIC
    ibv_gid             srcGid;         ///< GID handle for SRC NIC
    ibv_gid             dstGid;         ///< GID handle for DST NIC
    vector<ibv_qp*>     srcQueuePairs;  ///< Queue pairs for SRC NIC
    vector<ibv_qp*>     dstQueuePairs;  ///< Queue pairs for DST NIC
    ibv_mr*             srcMemRegion;   ///< Memory region for SRC
    ibv_mr*             dstMemRegion;   ///< Memory region for DST
    uint8_t             qpCount;  ///< Number of QPs to be used for transferring data
    vector<ibv_sge>     sgePerQueuePair;   ///< Scatter-gather elements per queue pair
    vector<ibv_send_wr> sendWorkRequests;  ///< Send work requests per queue pair
#endif

    // Counters
    double totalDurationMsec;    ///< Total duration for all iterations for this Transfer
    vector<double> perIterMsec;  ///< Duration for each individual iteration
    vector<set<pair<int, int>>>
        perIterCUs;  ///< GFX-Executor only. XCC:CU used per iteration
};

// Internal resources allocated per Executor
struct ExeInfo
{
    size_t totalBytes;         ///< Total bytes this executor transfers
    double totalDurationMsec;  ///< Total duration for all iterations for this Executor
    int    totalSubExecs;      ///< Total number of subExecutors to use
    bool   useSubIndices;      ///< Use subexecutor indicies
    int    numSubIndices;      ///< Number of subindices this ExeDevice has
    vector<SubExecParam> subExecParamCpu;  ///< Subexecutor parameters for this executor
    vector<TransferResources> resources;   ///< Per-Transfer resources

    // For GPU-Executors
    SubExecParam*       subExecParamGpu;  ///< GPU copy of subExecutor parameters
    vector<hipStream_t> streams;          ///< HIP streams to launch on
    vector<hipEvent_t>  startEvents;      ///< HIP start timing event
    vector<hipEvent_t>  stopEvents;       ///< HIP stop timing event
    int                 wallClockRate;    ///< (GFX-only) Device wall clock rate
};

// Structure to track PCIe topology
struct PCIeNode
{
    std::string        address;      ///< PCIe address for this PCIe node
    std::string        description;  ///< Description for this PCIe node
    std::set<PCIeNode> children;     ///< Children PCIe nodes

    // Default constructor
    PCIeNode()
    : address("")
    , description("")
    {}

    // Constructor
    PCIeNode(std::string const& addr)
    : address(addr)
    {}

    // Constructor
    PCIeNode(std::string const& addr, std::string const& desc)
    : address(addr)
    , description(desc)
    {}

    // Comparison operator for std::set
    bool operator<(PCIeNode const& other) const { return address < other.address; }
};

#ifdef NIC_EXEC_ENABLED
// Structure to track information about IBV devices
struct IbvDevice
{
    ibv_device* devicePtr;
    std::string name;
    std::string busId;
    bool        hasActivePort;
    int         numaNode;
    int         gidIndex;
    std::string gidDescriptor;
    bool        isRoce;
};
#endif

#ifdef NIC_EXEC_ENABLED
// Function to collect information about IBV devices
//========================================================================================
static bool
IsConfiguredGid(union ibv_gid const& gid)
{
    const struct in6_addr* a = (struct in6_addr*) gid.raw;
    int trailer              = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
    if(((a->s6_addr32[0] | trailer) == 0UL) ||
       ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL)))
    {
        return false;
    }
    return true;
}

static bool
LinkLocalGid(union ibv_gid const& gid)
{
    const struct in6_addr* a = (struct in6_addr*) gid.raw;
    if(a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL)
    {
        return true;
    }
    return false;
}

static ErrResult
GetRoceVersionNumber(struct ibv_context* const& context, int const& portNum,
                     int const& gidIndex, int& version)
{
    char const* deviceName             = ibv_get_device_name(context->device);
    char        gidRoceVerStr[16]      = {};
    char        roceTypePath[PATH_MAX] = {};
    sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
            deviceName, portNum, gidIndex);

    int fd = open(roceTypePath, O_RDONLY);
    if(fd == -1)
        return { ERR_FATAL, "Failed while opening RoCE file path (%s)", roceTypePath };

    int ret = read(fd, gidRoceVerStr, 15);
    close(fd);

    if(ret == -1) return { ERR_FATAL, "Failed while reading RoCE version" };

    if(strlen(gidRoceVerStr))
    {
        if(strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 ||
           strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0)
        {
            version = 1;
        }
        else if(strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0)
        {
            version = 2;
        }
    }
    return ERR_NONE;
}

static bool
IsIPv4MappedIPv6(const union ibv_gid& gid)
{
    // look for ::ffff:x.x.x.x format
    // From Broadcom documentation
    // https://techdocs.broadcom.com/us/en/storage-and-ethernet-connectivity/ethernet-nic-controllers/bcm957xxx/adapters/frequently-asked-questions1.html
    // "The IPv4 address is really an IPv4 address mapped into the IPv6 address space.
    // This can be identified by 80 “0” bits, followed by 16 “1” bits (“FFFF” in
    // hexadecimal) followed by the original 32-bit IPv4 address."
    return (gid.global.subnet_prefix == 0 && gid.raw[8] == 0 && gid.raw[9] == 0 &&
            gid.raw[10] == 0xff && gid.raw[11] == 0xff);
}

static ErrResult
GetGidIndex(struct ibv_context* context, int const& gidTblLen, int const& portNum,
            std::pair<int, std::string>& gidInfo)
{
    if(gidInfo.first >= 0) return ERR_NONE;  // honor user choice
    union ibv_gid gid;

    GidPriority highestPriority = GidPriority::UNKNOWN;
    int         gidIndex        = -1;

    for(int i = 0; i < gidTblLen; ++i)
    {
        IBV_CALL(ibv_query_gid, context, portNum, i, &gid);
        if(!IsConfiguredGid(gid)) continue;
        int gidCurrRoceVersion;
        if(GetRoceVersionNumber(context, portNum, i, gidCurrRoceVersion).errType !=
           ERR_NONE)
            continue;
        GidPriority currPriority;
        if(IsIPv4MappedIPv6(gid))
        {
            currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_IPV4
                                                     : GidPriority::ROCEV1_IPV4;
        }
        else if(!LinkLocalGid(gid))
        {
            currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_IPV6
                                                     : GidPriority::ROCEV1_IPV6;
        }
        else
        {
            currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_LINK_LOCAL
                                                     : GidPriority::ROCEV1_LINK_LOCAL;
        }
        if(currPriority > highestPriority)
        {
            highestPriority = currPriority;
            gidIndex        = i;
        }
    }

    if(highestPriority == GidPriority::UNKNOWN)
    {
        gidInfo.first = -1;
        return { ERR_FATAL, "Failed to auto-detect a valid GID index. Try setting it "
                            "manually through IB_GID_INDEX" };
    }
    gidInfo.first  = gidIndex;
    gidInfo.second = GidPriorityStr[highestPriority];
    return ERR_NONE;
}

static vector<IbvDevice>&
GetIbvDeviceList()
{
    static bool              isInitialized = false;
    static vector<IbvDevice> ibvDeviceList = {};

    // Build list on first use
    if(!isInitialized)
    {
        // Query the number of IBV devices
        int          numIbvDevices = 0;
        ibv_device** deviceList    = ibv_get_device_list(&numIbvDevices);

        if(deviceList && numIbvDevices > 0)
        {
            // Loop over each device to collect information
            for(int i = 0; i < numIbvDevices; i++)
            {
                IbvDevice ibvDevice;
                ibvDevice.devicePtr     = deviceList[i];
                ibvDevice.name          = deviceList[i]->name;
                ibvDevice.hasActivePort = false;
                {
                    struct ibv_context* context = ibv_open_device(ibvDevice.devicePtr);
                    if(context)
                    {
                        struct ibv_device_attr deviceAttr;
                        if(!ibv_query_device(context, &deviceAttr))
                        {
                            int activePort;
                            ibvDevice.gidIndex = -1;
                            for(int port = 1; port <= deviceAttr.phys_port_cnt; ++port)
                            {
                                struct ibv_port_attr portAttr;
                                if(ibv_query_port(context, port, &portAttr)) continue;
                                if(portAttr.state == IBV_PORT_ACTIVE)
                                {
                                    activePort              = port;
                                    ibvDevice.hasActivePort = true;
                                    if(portAttr.link_layer == IBV_LINK_LAYER_ETHERNET)
                                    {
                                        ibvDevice.isRoce = true;
                                        std::pair<int, std::string> gidInfo(-1, "");
                                        auto                        res =
                                            GetGidIndex(context, portAttr.gid_tbl_len,
                                                        activePort, gidInfo);
                                        if(res.errType == ERR_NONE)
                                        {
                                            ibvDevice.gidIndex      = gidInfo.first;
                                            ibvDevice.gidDescriptor = gidInfo.second;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        ibv_close_device(context);
                    }
                }
                ibvDevice.busId = "";
                {
                    std::string device_path(ibvDevice.devicePtr->dev_path);
                    if(std::filesystem::exists(device_path))
                    {
                        std::string pciPath =
                            std::filesystem::canonical(device_path + "/device").string();
                        std::size_t pos = pciPath.find_last_of('/');
                        if(pos != std::string::npos)
                        {
                            ibvDevice.busId = pciPath.substr(pos + 1);
                        }
                    }
                }

                // Get nearest numa node for this device
                ibvDevice.numaNode = -1;
                std::filesystem::path devicePath =
                    "/sys/bus/pci/devices/" + ibvDevice.busId + "/numa_node";
                std::string canonicalPath =
                    std::filesystem::canonical(devicePath).string();

                if(std::filesystem::exists(canonicalPath))
                {
                    std::ifstream file(canonicalPath);
                    if(file.is_open())
                    {
                        std::string numaNodeStr;
                        std::getline(file, numaNodeStr);
                        int numaNodeVal;
                        if(sscanf(numaNodeStr.c_str(), "%d", &numaNodeVal) == 1)
                            ibvDevice.numaNode = numaNodeVal;
                        file.close();
                    }
                }
                ibvDeviceList.push_back(ibvDevice);
            }
        }
        ibv_free_device_list(deviceList);
        isInitialized = true;
    }
    return ibvDeviceList;
}
#endif  // NIC_EXEC_ENABLED

#ifdef NIC_EXEC_ENABLED
// PCIe-related functions
//========================================================================================

// Prints off PCIe tree
static void
PrintPCIeTree(PCIeNode const& node, std::string const& prefix = "", bool isLast = true)
{
    if(!node.address.empty())
    {
        printf("%s%s%s", prefix.c_str(), (isLast ? "└── " : "├── "),
               node.address.c_str());
        if(!node.description.empty())
        {
            printf("(%s)", node.description.c_str());
        }
        printf("\n");
    }
    auto const& children = node.children;
    for(auto it = children.begin(); it != children.end(); ++it)
    {
        PrintPCIeTree(*it, prefix + (isLast ? "    " : "│   "),
                      std::next(it) == children.end());
    }
}

// Inserts nodes along pcieAddress down a tree starting from root
static ErrResult
InsertPCIePathToTree(std::string const& pcieAddress, std::string const& description,
                     PCIeNode& root)
{
    std::filesystem::path devicePath    = "/sys/bus/pci/devices/" + pcieAddress;
    std::string           canonicalPath = std::filesystem::canonical(devicePath).string();

    if(!std::filesystem::exists(devicePath))
    {
        return { ERR_FATAL, "Device path %s does not exist", devicePath.c_str() };
    }

    std::istringstream iss(canonicalPath);
    std::string        token;

    PCIeNode* currNode = &root;
    while(std::getline(iss, token, '/'))
    {
        auto it  = (currNode->children.insert(PCIeNode(token))).first;
        currNode = const_cast<PCIeNode*>(&(*it));
    }
    currNode->description = description;

    return ERR_NONE;
}

// Returns root node for PCIe tree.  Constructed on first use
static PCIeNode*
GetPCIeTreeRoot()
{
    static bool     isInitialized = false;
    static PCIeNode pcieRoot;

    // Build PCIe tree on first use
    if(!isInitialized)
    {
        // Add NICs to the tree
        int         numNics       = GetNumExecutors(EXE_NIC);
        auto const& ibvDeviceList = GetIbvDeviceList();
        for(IbvDevice const& ibvDevice : ibvDeviceList)
        {
            if(!ibvDevice.hasActivePort || ibvDevice.busId == "") continue;
            InsertPCIePathToTree(ibvDevice.busId, ibvDevice.name, pcieRoot);
        }

        // Add GPUs to the tree
        int numGpus = GetNumExecutors(EXE_GPU_GFX);
        for(int i = 0; i < numGpus; ++i)
        {
            char hipPciBusId[64];
            if(hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), i) == hipSuccess)
            {
                InsertPCIePathToTree(hipPciBusId, "GPU " + std::to_string(i), pcieRoot);
            }
        }
#    ifdef VERBS_DEBUG
        PrintPCIeTree(pcieRoot);
#    endif
        isInitialized = true;
    }
    return &pcieRoot;
}

// Finds the lowest common ancestor in PCIe tree between two nodes
static PCIeNode const*
GetLcaBetweenNodes(PCIeNode const* root, std::string const& node1Address,
                   std::string const& node2Address)
{
    if(!root || root->address == node1Address || root->address == node2Address)
        return root;

    PCIeNode const* lcaFound1 = nullptr;
    PCIeNode const* lcaFound2 = nullptr;

    // Recursively iterate over children
    for(auto const& child : root->children)
    {
        PCIeNode const* lca = GetLcaBetweenNodes(&child, node1Address, node2Address);
        if(!lca) continue;
        if(!lcaFound1)
        {
            // First time found
            lcaFound1 = lca;
        }
        else
        {
            // Second time found
            lcaFound2 = lca;
            break;
        }
    }

    // If two children were found, then current node is the lowest common ancestor
    return (lcaFound1 && lcaFound2) ? root : lcaFound1;
}

// Gets the depth of an node in the PCIe tree
static int
GetLcaDepth(std::string const& targetBusID, PCIeNode const* const& node, int depth = 0)
{
    if(!node) return -1;
    if(targetBusID == node->address) return depth;

    for(auto const& child : node->children)
    {
        int distance = GetLcaDepth(targetBusID, &child, depth + 1);
        if(distance != -1) return distance;
    }
    return -1;
}

// Function to extract the bus number from a PCIe address (domain:bus:device.function)
static int
ExtractBusNumber(std::string const& pcieAddress)
{
    int  domain, bus, device, function;
    char delimiter;

    std::istringstream iss(pcieAddress);
    iss >> std::hex >> domain >> delimiter >> bus >> delimiter >> device >> delimiter >>
        function;
    if(iss.fail())
    {
#    ifdef VERBS_DEBUG
        printf("Invalid PCIe address format: %s\n", pcieAddress.c_str());
#    endif
        return -1;
    }
    return bus;
}

// Function to compute the distance between two bus IDs
static int
GetBusIdDistance(std::string const& pcieAddress1, std::string const& pcieAddress2)
{
    int bus1 = ExtractBusNumber(pcieAddress1);
    int bus2 = ExtractBusNumber(pcieAddress2);
    return (bus1 < 0 || bus2 < 0) ? -1 : std::abs(bus1 - bus2);
}

// Given a target busID and a set of candidate devices, returns a set of indices
// that is "closest" to the target
static std::set<int>
GetNearestDevicesInTree(std::string const&              targetBusId,
                        std::vector<std::string> const& candidateBusIdList)
{
    int           maxDepth    = -1;
    int           minDistance = std::numeric_limits<int>::max();
    std::set<int> matches     = {};

    // Loop over the candidates to find the ones with the lowest common ancestor (LCA)
    for(int i = 0; i < candidateBusIdList.size(); i++)
    {
        std::string const& candidateBusId = candidateBusIdList[i];
        if(candidateBusId == "") continue;
        PCIeNode const* lca =
            GetLcaBetweenNodes(GetPCIeTreeRoot(), targetBusId, candidateBusId);
        if(!lca) continue;

        int depth        = GetLcaDepth(lca->address, GetPCIeTreeRoot());
        int currDistance = GetBusIdDistance(targetBusId, candidateBusId);

        // When more than one LCA match is found, choose the one with smallest busId
        // difference NOTE: currDistance could be -1, which signals problem with parsing,
        // however still
        //       remains a valid "closest" candidate, so is included
        if(depth > maxDepth ||
           (depth == maxDepth && depth >= 0 && currDistance < minDistance))
        {
            maxDepth = depth;
            matches.clear();
            matches.insert(i);
            minDistance = currDistance;
        }
        else if(depth == maxDepth && depth >= 0 && currDistance == minDistance)
        {
            matches.insert(i);
        }
    }
    return matches;
}
#endif  // NIC_EXEC_ENABLED

#ifdef NIC_EXEC_ENABLED
// IB Verbs-related functions
//========================================================================================

// Create a queue pair
static ErrResult
CreateQueuePair(ConfigOptions const& cfg, struct ibv_pd* pd, struct ibv_cq* cq,
                struct ibv_qp*& qp)
{
    // Set queue pair attributes
    struct ibv_qp_init_attr attr = {};
    attr.qp_type                 = IBV_QPT_RC;  // Set type to reliable connection
    attr.send_cq                 = cq;          // Send completion queue
    attr.recv_cq                 = cq;          // Recv completion queue
    attr.cap.max_send_wr         = cfg.nic.maxSendWorkReq;  // Max send work requests
    attr.cap.max_recv_wr         = cfg.nic.maxRecvWorkReq;  // Max recv work requests
    attr.cap.max_send_sge        = 1;  // Max send scatter-gather entries
    attr.cap.max_recv_sge        = 1;  // Max recv scatter-gather entries

    qp = ibv_create_qp(pd, &attr);
    if(qp == NULL) return { ERR_FATAL, "Error while creating QP" };

    return ERR_NONE;
}

// Initialize a queue pair
static ErrResult
InitQueuePair(struct ibv_qp* qp, uint8_t port, unsigned flags)
{
    struct ibv_qp_attr attr = {};            // Clear all attributes
    attr.qp_state           = IBV_QPS_INIT;  // Set the QP state to INIT
    attr.pkey_index         = 0;             // Set the partition key index to 0
    attr.port_num           = port;          // Set the port number to the defined IB_PORT
    attr.qp_access_flags    = flags;  // Set the QP access flags to the provided flags

    int ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE |             // Modify the QP state
                                IBV_QP_PKEY_INDEX |    // Modify the partition key index
                                IBV_QP_PORT |          // Modify the port number
                                IBV_QP_ACCESS_FLAGS);  // Modify the access flags

    if(ret != 0)
        return { ERR_FATAL, "Error during QP Init. IB Verbs Error code: %d", ret };

    return ERR_NONE;
}

// Transition QueuePair to Ready to Receive State
static ErrResult
TransitionQpToRtr(ibv_qp* qp, uint16_t const& dlid, uint32_t const& dqpn,
                  ibv_gid const& gid, uint8_t const& gidIndex, uint8_t const& port,
                  bool const& isRoCE, ibv_mtu const& mtu)
{
    // Prepare QP attributes
    struct ibv_qp_attr attr = {};
    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = mtu;
    attr.rq_psn             = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer      = 12;
    if(isRoCE)
    {
        attr.ah_attr.is_global                     = 1;
        attr.ah_attr.grh.dgid.global.subnet_prefix = gid.global.subnet_prefix;
        attr.ah_attr.grh.dgid.global.interface_id  = gid.global.interface_id;
        attr.ah_attr.grh.flow_label                = 0;
        attr.ah_attr.grh.sgid_index                = gidIndex;
        attr.ah_attr.grh.hop_limit                 = 255;
    }
    else
    {
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid      = dlid;
    }
    attr.ah_attr.sl            = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num      = port;
    attr.dest_qp_num           = dqpn;

    // Modify the QP
    int ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                IBV_QP_MIN_RNR_TIMER);
    if(ret != 0)
        return { ERR_FATAL, "Error during QP RTR. IB Verbs Error code: %d", ret };

    return ERR_NONE;
}

// Transition QueuePair to Ready to Send state
static ErrResult
TransitionQpToRts(struct ibv_qp* qp)
{
    struct ibv_qp_attr attr = {};
    attr.qp_state           = IBV_QPS_RTS;
    attr.sq_psn             = 0;
    attr.timeout            = 14;
    attr.retry_cnt          = 7;
    attr.rnr_retry          = 7;
    attr.max_rd_atomic      = 1;

    int ret =
        ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if(ret != 0)
        return { ERR_FATAL, "Error during QP RTS. IB Verbs Error code: %d", ret };

    return ERR_NONE;
}

static ErrResult
PrepareNicTransferResources(ConfigOptions const& cfg, ExeDevice const& srcExeDevice,
                            Transfer const& t, TransferResources& rss)

{
    // Switch to the closest NUMA node to this NIC (not supported without NUMA)
    int numaNode = GetIbvDeviceList()[srcExeDevice.exeIndex].numaNode;
    (void) numaNode;  // Suppress unused variable warning

    int const port = cfg.nic.ibPort;

    // Figure out destination NIC (Accounts for possible remap due to use of
    // EXE_NIC_NEAREST)
    ExeDevice dstExeDevice;
    ERR_CHECK(
        GetActualExecutor(cfg, { t.exeDevice.exeType, t.exeSubIndex }, dstExeDevice));

    rss.srcNicIndex = srcExeDevice.exeIndex;
    rss.dstNicIndex = dstExeDevice.exeIndex;
    rss.qpCount     = t.numSubExecs;

    // Check for valid NICs and active ports
    int numNics = GetNumExecutors(EXE_NIC);
    if(rss.srcNicIndex < 0 || rss.srcNicIndex >= numNics)
        return { ERR_FATAL, "SRC NIC index is out of range (%d)", rss.srcNicIndex };
    if(rss.dstNicIndex < 0 || rss.dstNicIndex >= numNics)
        return { ERR_FATAL, "DST NIC index is out of range (%d)", rss.dstNicIndex };
    if(!GetIbvDeviceList()[rss.srcNicIndex].hasActivePort)
        return { ERR_FATAL, "SRC NIC %d is not active\n", rss.srcNicIndex };
    if(!GetIbvDeviceList()[rss.dstNicIndex].hasActivePort)
        return { ERR_FATAL, "DST NIC %d is not active\n", rss.dstNicIndex };

    // Queue pair flags
    unsigned int rdmaAccessFlags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                                    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);

    unsigned int rdmaMemRegFlags = rdmaAccessFlags;
    if(cfg.nic.useRelaxedOrder) rdmaMemRegFlags |= IBV_ACCESS_RELAXED_ORDERING;

    // Open NIC contexts
    IBV_PTR_CALL(rss.srcContext, ibv_open_device,
                 GetIbvDeviceList()[rss.srcNicIndex].devicePtr);
    IBV_PTR_CALL(rss.dstContext, ibv_open_device,
                 GetIbvDeviceList()[rss.dstNicIndex].devicePtr);

    // Open protection domains
    IBV_PTR_CALL(rss.srcProtect, ibv_alloc_pd, rss.srcContext);
    IBV_PTR_CALL(rss.dstProtect, ibv_alloc_pd, rss.dstContext);

    // Register memory region
    IBV_PTR_CALL(rss.srcMemRegion, ibv_reg_mr, rss.srcProtect, rss.srcMem[0],
                 rss.numBytes, rdmaMemRegFlags);
    IBV_PTR_CALL(rss.dstMemRegion, ibv_reg_mr, rss.dstProtect, rss.dstMem[0],
                 rss.numBytes, rdmaMemRegFlags);

    // Create completion queues
    IBV_PTR_CALL(rss.srcCompQueue, ibv_create_cq, rss.srcContext, cfg.nic.queueSize, NULL,
                 NULL, 0);
    IBV_PTR_CALL(rss.dstCompQueue, ibv_create_cq, rss.dstContext, cfg.nic.queueSize, NULL,
                 NULL, 0);

    // Get port attributes
    IBV_CALL(ibv_query_port, rss.srcContext, port, &rss.srcPortAttr);
    IBV_CALL(ibv_query_port, rss.dstContext, port, &rss.dstPortAttr);

    if(rss.srcPortAttr.link_layer != rss.dstPortAttr.link_layer)
        return { ERR_FATAL,
                 "SRC NIC (%d) and DST NIC (%d) do not have the same link layer",
                 rss.srcNicIndex, rss.dstNicIndex };

    // Prepare GID index
    int srcGidIndex = cfg.nic.ibGidIndex;
    int dstGidIndex = cfg.nic.ibGidIndex;

    // Check for RDMA over Converged Ethernet (RoCE) and update GID index appropriately
    bool isRoCE = (rss.srcPortAttr.link_layer == IBV_LINK_LAYER_ETHERNET);
    if(isRoCE)
    {
        // Try to auto-detect the GID index
        std::pair<int, std::string> srcGidInfo(srcGidIndex, "");
        std::pair<int, std::string> dstGidInfo(dstGidIndex, "");
        ERR_CHECK(GetGidIndex(rss.srcContext, rss.srcPortAttr.gid_tbl_len, cfg.nic.ibPort,
                              srcGidInfo));
        ERR_CHECK(GetGidIndex(rss.dstContext, rss.dstPortAttr.gid_tbl_len, cfg.nic.ibPort,
                              dstGidInfo));
        srcGidIndex = srcGidInfo.first;
        dstGidIndex = dstGidInfo.first;
        IBV_CALL(ibv_query_gid, rss.srcContext, port, srcGidIndex, &rss.srcGid);
        IBV_CALL(ibv_query_gid, rss.dstContext, port, dstGidIndex, &rss.dstGid);
    }

    // Prepare queue pairs and send elements
    rss.srcQueuePairs.resize(rss.qpCount);
    rss.dstQueuePairs.resize(rss.qpCount);
    rss.sgePerQueuePair.resize(rss.qpCount);
    rss.sendWorkRequests.resize(rss.qpCount);

    for(int i = 0; i < rss.qpCount; ++i)
    {
        // Create scatter-gather element for the portion of memory assigned to this queue
        // pair
        ibv_sge sg             = {};
        sg.addr                = (uint64_t) rss.subExecParamCpu[i].src[0];
        sg.length              = rss.subExecParamCpu[i].N * sizeof(float);
        sg.lkey                = rss.srcMemRegion->lkey;
        rss.sgePerQueuePair[i] = sg;

        // Create send work request
        ibv_send_wr wr          = {};
        wr.wr_id                = i;
        wr.sg_list              = &rss.sgePerQueuePair[i];
        wr.num_sge              = 1;
        wr.opcode               = IBV_WR_RDMA_WRITE;
        wr.send_flags           = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr  = (uint64_t) rss.subExecParamCpu[i].dst[0];
        wr.wr.rdma.rkey         = rss.dstMemRegion->rkey;
        rss.sendWorkRequests[i] = wr;

        // Create SRC/DST queue pairs
        ERR_CHECK(
            CreateQueuePair(cfg, rss.srcProtect, rss.srcCompQueue, rss.srcQueuePairs[i]));
        ERR_CHECK(
            CreateQueuePair(cfg, rss.dstProtect, rss.dstCompQueue, rss.dstQueuePairs[i]));

        // Initialize SRC/DST queue pairs
        ERR_CHECK(InitQueuePair(rss.srcQueuePairs[i], port, rdmaAccessFlags));
        ERR_CHECK(InitQueuePair(rss.dstQueuePairs[i], port, rdmaAccessFlags));

        // Transition the SRC queue pair to ready to receive
        ERR_CHECK(TransitionQpToRtr(rss.srcQueuePairs[i], rss.dstPortAttr.lid,
                                    rss.dstQueuePairs[i]->qp_num, rss.dstGid, dstGidIndex,
                                    port, isRoCE, rss.srcPortAttr.active_mtu));

        // Transition the SRC queue pair to ready to send
        ERR_CHECK(TransitionQpToRts(rss.srcQueuePairs[i]));

        // Transition the DST queue pair to ready to receive
        ERR_CHECK(TransitionQpToRtr(rss.dstQueuePairs[i], rss.srcPortAttr.lid,
                                    rss.srcQueuePairs[i]->qp_num, rss.srcGid, srcGidIndex,
                                    port, isRoCE, rss.dstPortAttr.active_mtu));

        // Transition the DST queue pair to ready to send
        ERR_CHECK(TransitionQpToRts(rss.dstQueuePairs[i]));
    }

    return ERR_NONE;
}

static ErrResult
TeardownNicTransferResources(TransferResources& rss)
{
    // Deregister memory regions
    IBV_CALL(ibv_dereg_mr, rss.srcMemRegion);
    IBV_CALL(ibv_dereg_mr, rss.dstMemRegion);

    // Destroy queue pairs
    for(auto srcQueuePair : rss.srcQueuePairs)
        IBV_CALL(ibv_destroy_qp, srcQueuePair);
    rss.srcQueuePairs.clear();
    for(auto dstQueuePair : rss.dstQueuePairs)
        IBV_CALL(ibv_destroy_qp, dstQueuePair);
    rss.dstQueuePairs.clear();

    // Destroy completion queues
    IBV_CALL(ibv_destroy_cq, rss.srcCompQueue);
    IBV_CALL(ibv_destroy_cq, rss.dstCompQueue);

    // Deallocate protection domains
    IBV_CALL(ibv_dealloc_pd, rss.srcProtect);
    IBV_CALL(ibv_dealloc_pd, rss.dstProtect);

    // Destroy context
    IBV_CALL(ibv_close_device, rss.srcContext);
    IBV_CALL(ibv_close_device, rss.dstContext);

    return ERR_NONE;
}
#endif  // NIC_EXEC_ENABLED

// Data validation-related functions
//========================================================================================

// Pseudo-random formula for each element in array
static __host__ float
PrepSrcValue(int srcBufferIdx, size_t idx)
{
    return (((idx % 383) * 517) % 383 + 31) * (srcBufferIdx + 1);
}

// Fills a pre-sized buffer with the pattern, based on which src index buffer
// Note: Can also generate expected dst buffer
static void
PrepareReference(ConfigOptions const& cfg, std::vector<float>& cpuBuffer, int bufferIdx)
{
    size_t N = cpuBuffer.size();

    if(!cfg.data.fillCompress.empty())
    {
        // 0 -> Random
        // 1 ->  1B0 - The upper  1 byte  of each aligned 2 bytes is 0
        // 2 ->  2B0 - The upper  2 bytes of each aligned 4 bytes are 0
        // 3 ->  4B0 - The upper  4 bytes of each aligned 8 bytes are 0
        // 4 -> 32B0 - The upper 32 bytes of each 64-byte line are 0

        // Fill buffer with random floats
        std::mt19937 gen;
        gen.seed(bufferIdx * 425);
        std::uniform_real_distribution<float> dist(-100000.0f, +100000.0f);
        for(size_t i = 0; i < N; i++)
        {
            cpuBuffer[i] = dist(gen);
        }

        // Figure out distribution for lines based on the percentages given
        size_t                           numLines = N / 16;
        size_t                           leftover = numLines;
        std::vector<size_t>              lineCounts(5, 0);
        std::set<std::pair<double, int>> remainder;

        // Assign rounded down values first
        std::vector<int> percentages = cfg.data.fillCompress;
        while(percentages.size() < 5)
            percentages.push_back(0);
        for(size_t i = 0; i < percentages.size(); i++)
        {
            lineCounts[i] = (size_t) (numLines * (percentages[i] / 100.0));
            leftover -= lineCounts[i];
            remainder.insert(
                std::make_pair(numLines * (percentages[i] / 100.0) - lineCounts[i], i));
        }

        // Assign leftovers based on largest remainder
        while(leftover != 0)
        {
            auto last = *remainder.rbegin();
            lineCounts[last.second]++;
            remainder.erase(last);
            leftover--;
        }

        // Randomly decide which lines get assigned to which types
        std::vector<int> lineTypes(numLines, 0);
        size_t           offset = lineCounts[0];
        for(int i = 1; i < 5; i++)
        {
            for(size_t j = 0; j < lineCounts[i]; j++)
                lineTypes[offset++] = i;
        }
        std::shuffle(lineTypes.begin(), lineTypes.end(), gen);

        // Apply zero-ing
        int dumpLines = getenv("DUMP_LINES") ? atoi(getenv("DUMP_LINES")) : 0;

        if(dumpLines)
        {
            printf("Input pattern 64B line statistics for bufferIdx %d:\n", bufferIdx);
            printf("Total lines: %lu\n", numLines);
            printf("- 0: Random : %8lu (%8.3f%%)\n", lineCounts[0],
                   100.0 * lineCounts[0] / (1.0 * numLines));
            printf("- 1: 1B0    : %8lu (%8.3f%%)\n", lineCounts[1],
                   100.0 * lineCounts[1] / (1.0 * numLines));
            printf("- 2: 2B0    : %8lu (%8.3f%%)\n", lineCounts[2],
                   100.0 * lineCounts[2] / (1.0 * numLines));
            printf("- 3: 4B0    : %8lu (%8.3f%%)\n", lineCounts[3],
                   100.0 * lineCounts[3] / (1.0 * numLines));
            printf("- 4: 32B0   : %8lu (%8.3f%%)\n", lineCounts[4],
                   100.0 * lineCounts[4] / (1.0 * numLines));
        }

        for(size_t line = 0; line < numLines; line++)
        {
            unsigned char* linePtr = (unsigned char*) &cpuBuffer[line * 16];

            switch(lineTypes[line])
            {
                case 1:  // 1B0
                    for(int i = 0; i < 32; i++)
                        linePtr[2 * i + 1] = 0;
                    break;
                case 2:  // 2B0
                    for(int i = 0; i < 16; i++)
                    {
                        linePtr[4 * i + 2] = 0;
                        linePtr[4 * i + 3] = 0;
                    }
                    break;
                case 3:  // 4B0
                    for(int i = 0; i < 8; i++)
                    {
                        linePtr[8 * i + 4] = 0;
                        linePtr[8 * i + 5] = 0;
                        linePtr[8 * i + 6] = 0;
                        linePtr[8 * i + 7] = 0;
                    }
                    break;
                case 4:  // 32B0
                    for(int i = 32; i < 64; i++)
                        linePtr[i] = 0;
                    break;
            }

            if(static_cast<int>(line) < dumpLines)
            {
                printf("Line %02zu [%d]: ", line, lineTypes[line]);
                for(int j = 63; j >= 0; j--)
                {
                    printf("%02x ", linePtr[j]);
                    if(j % 16 == 0) printf(" ");
                }
                printf("\n");
            }
        }
    }
    else
    {
        // Use fill pattern if specified
        size_t patternLen = cfg.data.fillPattern.size();
        if(patternLen > 0)
        {
            size_t copies       = N / patternLen;
            size_t leftOver     = N % patternLen;
            float* cpuBufferPtr = cpuBuffer.data();
            for(size_t i = 0; i < copies; i++)
            {
                memcpy(cpuBufferPtr, cfg.data.fillPattern.data(),
                       patternLen * sizeof(float));
                cpuBufferPtr += patternLen;
            }
            if(leftOver)
                memcpy(cpuBufferPtr, cfg.data.fillPattern.data(),
                       leftOver * sizeof(float));
        }
        else
        {
            // Fall back to pseudo-random
            for(size_t i = 0; i < N; ++i)
                cpuBuffer[i] = PrepSrcValue(bufferIdx, i);
        }
    }
}

// Checks that destination buffers match expected values
static ErrResult
ValidateAllTransfers(ConfigOptions const& cfg, vector<Transfer> const& transfers,
                     vector<TransferResources*> const& transferResources,
                     vector<vector<float>> const&      dstReference,
                     vector<float>&                    outputBuffer)
{
    float* output;
    size_t initOffset = cfg.data.byteOffset / sizeof(float);

    for(auto rss : transferResources)
    {
        int             transferIdx = rss->transferIdx;
        Transfer const& t           = transfers[transferIdx];
        size_t          N           = t.numBytes / sizeof(float);

        float const* expected = dstReference[t.srcs.size()].data();
        for(size_t dstIdx = 0; dstIdx < rss->dstMem.size(); dstIdx++)
        {
            if(IsCpuMemType(t.dsts[dstIdx].memType) || cfg.data.validateDirect)
            {
                output = (rss->dstMem[dstIdx]) + initOffset;
            }
            else
            {
                ERR_CHECK(hipMemcpy(outputBuffer.data(),
                                    (rss->dstMem[dstIdx]) + initOffset, t.numBytes,
                                    hipMemcpyDefault));
                ERR_CHECK(hipDeviceSynchronize());
                output = outputBuffer.data();
            }

            if(memcmp(output, expected, t.numBytes))
            {
                // Difference found - find first error
                for(size_t i = 0; i < N; i++)
                {
                    if(output[i] != expected[i])
                    {
                        return { ERR_FATAL,
                                 "Transfer %d: Unexpected mismatch at index %lu of "
                                 "destination %d: Expected %10.5f Actual: %10.5f",
                                 transferIdx,
                                 i,
                                 dstIdx,
                                 expected[i],
                                 output[i] };
                    }
                }
                return { ERR_FATAL,
                         "Transfer %d: Unexpected output mismatch for destination %d",
                         transferIdx, dstIdx };
            }
        }
    }
    return ERR_NONE;
}

// Preparation-related functions
//========================================================================================

// Prepares input parameters for each subexecutor
// Determines how sub-executors will split up the work
// Initializes counters
static ErrResult
PrepareSubExecParams(ConfigOptions const& cfg, Transfer const& transfer,
                     TransferResources& rss)
{
    // Each subExecutor needs to know src/dst pointers and how many elements to transfer
    // Figure out the sub-array each subExecutor works on for this Transfer
    // - Partition N as evenly as possible, but try to keep subarray sizes as multiples of
    // data.blockBytes
    //   except the very last one, for alignment reasons
    size_t const N              = transfer.numBytes / sizeof(float);
    int const    initOffset     = cfg.data.byteOffset / sizeof(float);
    int const    targetMultiple = cfg.data.blockBytes / sizeof(float);

    // In some cases, there may not be enough data for all subExectors
    int const maxSubExecToUse =
        std::min((size_t) (N + targetMultiple - 1) / targetMultiple,
                 (size_t) transfer.numSubExecs);

    vector<SubExecParam>& subExecParam = rss.subExecParamCpu;
    subExecParam.clear();
    subExecParam.resize(transfer.numSubExecs);

    size_t assigned = 0;
    for(int i = 0; i < transfer.numSubExecs; ++i)
    {
        SubExecParam& p = subExecParam[i];
        p.numSrcs       = rss.srcMem.size();
        p.numDsts       = rss.dstMem.size();
        p.startCycle    = 0;
        p.stopCycle     = 0;
        p.hwId          = 0;
        p.xccId         = 0;

        // In single team mode, subexecutors stripe across the entire array
        if(cfg.gfx.useSingleTeam && transfer.exeDevice.exeType == EXE_GPU_GFX)
        {
            p.N        = N;
            p.teamSize = transfer.numSubExecs;
            p.teamIdx  = i;
            for(int iSrc = 0; iSrc < p.numSrcs; ++iSrc)
                p.src[iSrc] = rss.srcMem[iSrc] + initOffset;
            for(int iDst = 0; iDst < p.numDsts; ++iDst)
                p.dst[iDst] = rss.dstMem[iDst] + initOffset;
        }
        else
        {
            // Otherwise, each subexecutor works on separate subarrays
            int const    subExecLeft = std::max(0, maxSubExecToUse - i);
            size_t const leftover    = N - assigned;
            size_t const roundedN    = (leftover + targetMultiple - 1) / targetMultiple;

            p.N        = subExecLeft
                             ? std::min(leftover, ((roundedN / subExecLeft) * targetMultiple))
                             : 0;
            p.teamSize = 1;
            p.teamIdx  = 0;
            for(int iSrc = 0; iSrc < p.numSrcs; ++iSrc)
                p.src[iSrc] = rss.srcMem[iSrc] + initOffset + assigned;
            for(int iDst = 0; iDst < p.numDsts; ++iDst)
                p.dst[iDst] = rss.dstMem[iDst] + initOffset + assigned;
            assigned += p.N;
        }

        p.preferredXccId = transfer.exeSubIndex;
        // Override if XCC table has been specified
        vector<vector<int>> const& table = cfg.gfx.prefXccTable;
        if(transfer.exeDevice.exeType == EXE_GPU_GFX && transfer.exeSubIndex == -1 &&
           !table.empty() && transfer.dsts.size() == 1 &&
           IsGpuMemType(transfer.dsts[0].memType))
        {
            if(static_cast<int>(table.size()) <= transfer.exeDevice.exeIndex ||
               static_cast<int>(table[transfer.exeDevice.exeIndex].size()) <=
                   transfer.dsts[0].memIndex)
            {
                return { ERR_FATAL, "[gfx.xccPrefTable] is too small" };
            }
            p.preferredXccId =
                table[transfer.exeDevice.exeIndex][transfer.dsts[0].memIndex];
            if(p.preferredXccId < 0 ||
               p.preferredXccId >= GetNumExecutorSubIndices(transfer.exeDevice))
            {
                return { ERR_FATAL,
                         "[gfx.xccPrefTable] defines out-of-bound XCC index %d",
                         p.preferredXccId };
            }
        }
    }

    // Clear counters
    rss.totalDurationMsec = 0.0;

    return ERR_NONE;
}

// Prepare each executor
// Allocates memory for src/dst, prepares subexecutors, executor-specific data structures
static ErrResult
PrepareExecutor(ConfigOptions const& cfg, vector<Transfer> const& transfers,
                ExeDevice const& exeDevice, ExeInfo& exeInfo)
{
    exeInfo.totalDurationMsec = 0.0;

    // Loop over each transfer this executor is involved in
    for(auto& rss : exeInfo.resources)
    {
        Transfer const& t = transfers[rss.transferIdx];
        rss.numBytes      = t.numBytes;

        // Allocate source memory
        rss.srcMem.resize(t.srcs.size());
        for(size_t iSrc = 0; iSrc < t.srcs.size(); ++iSrc)
        {
            MemDevice const& srcMemDevice = t.srcs[iSrc];

            // Ensure executing GPU can access source memory
            if(IsGpuExeType(exeDevice.exeType) && IsGpuMemType(srcMemDevice.memType) &&
               srcMemDevice.memIndex != exeDevice.exeIndex)
            {
                ERR_CHECK(EnablePeerAccess(exeDevice.exeIndex, srcMemDevice.memIndex));
            }
            ERR_CHECK(AllocateMemory(srcMemDevice, t.numBytes + cfg.data.byteOffset,
                                     (void**) &rss.srcMem[iSrc]));
        }

        // Allocate destination memory
        rss.dstMem.resize(t.dsts.size());
        for(size_t iDst = 0; iDst < t.dsts.size(); ++iDst)
        {
            MemDevice const& dstMemDevice = t.dsts[iDst];

            // Ensure executing GPU can access destination memory
            if(IsGpuExeType(exeDevice.exeType) && IsGpuMemType(dstMemDevice.memType) &&
               dstMemDevice.memIndex != exeDevice.exeIndex)
            {
                ERR_CHECK(EnablePeerAccess(exeDevice.exeIndex, dstMemDevice.memIndex));
            }
            ERR_CHECK(AllocateMemory(dstMemDevice, t.numBytes + cfg.data.byteOffset,
                                     (void**) &rss.dstMem[iDst]));
        }

        if(exeDevice.exeType == EXE_GPU_DMA &&
           (t.exeSubIndex != -1 || cfg.dma.useHsaCopy))
        {
#if !defined(__NVCC__)
            // Collect HSA agent information
            hsa_amd_pointer_info_t info;
            info.size = sizeof(info);
            ERR_CHECK(hsa_amd_pointer_info(rss.dstMem[0], &info, NULL, NULL, NULL));
            rss.dstAgent = info.agentOwner;

            ERR_CHECK(hsa_amd_pointer_info(rss.srcMem[0], &info, NULL, NULL, NULL));
            rss.srcAgent = info.agentOwner;

            // Create HSA completion signal
            ERR_CHECK(hsa_signal_create(1, 0, NULL, &rss.signal));

            if(t.exeSubIndex != -1)
                rss.sdmaEngineId = (hsa_amd_sdma_engine_id_t) (1U << t.exeSubIndex);
#endif
        }

        // Prepare subexecutor parameters
        ERR_CHECK(PrepareSubExecParams(cfg, t, rss));
    }

    // Prepare additional requirements for GPU-based executors
    if(exeDevice.exeType == EXE_GPU_GFX || exeDevice.exeType == EXE_GPU_DMA)
    {
        ERR_CHECK(hipSetDevice(exeDevice.exeIndex));

        // Determine how many streams to use
        int const numStreamsToUse =
            (exeDevice.exeType == EXE_GPU_DMA ||
             (exeDevice.exeType == EXE_GPU_GFX && cfg.gfx.useMultiStream))
                ? exeInfo.resources.size()
                : 1;
        exeInfo.streams.resize(numStreamsToUse);

        // Create streams
        for(int i = 0; i < numStreamsToUse; ++i)
        {
            if(cfg.gfx.cuMask.size())
            {
#if !defined(__NVCC__)
                ERR_CHECK(hipExtStreamCreateWithCUMask(
                    &exeInfo.streams[i], cfg.gfx.cuMask.size(), cfg.gfx.cuMask.data()));
#else
                return { ERR_FATAL, "CU Masking in not supported on NVIDIA hardware" };
#endif
            }
            else
            {
                ERR_CHECK(hipStreamCreate(&exeInfo.streams[i]));
            }
        }

        if(cfg.gfx.useHipEvents || cfg.dma.useHipEvents)
        {
            exeInfo.startEvents.resize(numStreamsToUse);
            exeInfo.stopEvents.resize(numStreamsToUse);
            for(int i = 0; i < numStreamsToUse; ++i)
            {
                ERR_CHECK(hipEventCreate(&exeInfo.startEvents[i]));
                ERR_CHECK(hipEventCreate(&exeInfo.stopEvents[i]));
            }
        }
    }

    // Prepare for GPU GFX executor
    if(exeDevice.exeType == EXE_GPU_GFX)
    {
        // Allocate one contiguous chunk of GPU memory for threadblock parameters
        // This allows support for executing one transfer per stream, or all transfers in
        // a single stream
#if !defined(__NVCC__)
        MemType memType =
            MEM_GPU;  // AMD hardware can directly access GPU memory from host
#else
        MemType memType =
            MEM_MANAGED;  // NVIDIA hardware requires managed memory to access from host
#endif
        ERR_CHECK(AllocateMemory({ memType, exeDevice.exeIndex },
                                 exeInfo.totalSubExecs * sizeof(SubExecParam),
                                 (void**) &exeInfo.subExecParamGpu));

        // Create subexecutor parameter array for entire executor
        exeInfo.subExecParamCpu.clear();
        exeInfo.numSubIndices = GetNumExecutorSubIndices(exeDevice);
#if defined(__NVCC__)
        exeInfo.wallClockRate = 1000000;
#else
        ERR_CHECK(hipDeviceGetAttribute(
            &exeInfo.wallClockRate, hipDeviceAttributeWallClockRate, exeDevice.exeIndex));
#endif
        int transferOffset = 0;
        if(cfg.gfx.useMultiStream || cfg.gfx.blockOrder == 0)
        {
            // Threadblocks are ordered sequentially one transfer at a time
            for(auto& rss : exeInfo.resources)
            {
                Transfer const& t = transfers[rss.transferIdx];
                (void) t;  // Suppress unused variable warning
                rss.subExecParamGpuPtr = exeInfo.subExecParamGpu + transferOffset;
                for(auto p : rss.subExecParamCpu)
                {
                    rss.subExecIdx.push_back(exeInfo.subExecParamCpu.size());
                    exeInfo.subExecParamCpu.push_back(p);
                    transferOffset++;
                }
            }
        }
        else if(cfg.gfx.blockOrder == 1)
        {
            // Interleave threadblocks of different Transfers
            for(int subExecIdx = 0; exeInfo.subExecParamCpu.size() <
                                    static_cast<size_t>(exeInfo.totalSubExecs);
                ++subExecIdx)
            {
                for(auto& rss : exeInfo.resources)
                {
                    Transfer const& t = transfers[rss.transferIdx];
                    if(subExecIdx < t.numSubExecs)
                    {
                        rss.subExecIdx.push_back(exeInfo.subExecParamCpu.size());
                        exeInfo.subExecParamCpu.push_back(
                            rss.subExecParamCpu[subExecIdx]);
                    }
                }
            }
        }
        else if(cfg.gfx.blockOrder == 2)
        {
            // Build randomized threadblock list
            std::vector<std::pair<int, int>> indices;
            for(size_t i = 0; i < exeInfo.resources.size(); i++)
            {
                auto const&     rss = exeInfo.resources[i];
                Transfer const& t   = transfers[rss.transferIdx];
                for(int j = 0; j < t.numSubExecs; j++)
                    indices.push_back(std::make_pair(static_cast<int>(i), j));
            }

            std::random_device rd;
            std::mt19937       gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);

            // Build randomized threadblock list
            for(auto p : indices)
            {
                auto& rss = exeInfo.resources[p.first];
                rss.subExecIdx.push_back(exeInfo.subExecParamCpu.size());
                exeInfo.subExecParamCpu.push_back(rss.subExecParamCpu[p.second]);
            }
        }

        // Copy sub executor parameters to GPU
        ERR_CHECK(hipSetDevice(exeDevice.exeIndex));
        ERR_CHECK(hipMemcpy(exeInfo.subExecParamGpu, exeInfo.subExecParamCpu.data(),
                            exeInfo.totalSubExecs * sizeof(SubExecParam),
                            hipMemcpyHostToDevice));
        ERR_CHECK(hipDeviceSynchronize());
    }

    // Prepare for NIC-based executors
    if(IsNicExeType(exeDevice.exeType))
    {
#ifdef NIC_EXEC_ENABLED
        for(auto& rss : exeInfo.resources)
        {
            Transfer const& t = transfers[rss.transferIdx];
            ERR_CHECK(PrepareNicTransferResources(cfg, exeDevice, t, rss));
        }
#else
        return { ERR_FATAL, "RDMA executor is not supported" };
#endif
    }
    return ERR_NONE;
}

// Teardown-related functions
//========================================================================================

// Clean up all resources
static ErrResult
TeardownExecutor(ConfigOptions const& cfg, ExeDevice const& exeDevice,
                 vector<Transfer> const& transfers, ExeInfo& exeInfo)
{
    // Loop over each transfer this executor is involved in
    for(auto& rss : exeInfo.resources)
    {
        Transfer const& t = transfers[rss.transferIdx];

        // Deallocate source memory
        for(size_t iSrc = 0; iSrc < t.srcs.size(); ++iSrc)
        {
            ERR_CHECK(DeallocateMemory(t.srcs[iSrc].memType, rss.srcMem[iSrc],
                                       t.numBytes + cfg.data.byteOffset));
        }

        // Deallocate destination memory
        for(size_t iDst = 0; iDst < t.dsts.size(); ++iDst)
        {
            ERR_CHECK(DeallocateMemory(t.dsts[iDst].memType, rss.dstMem[iDst],
                                       t.numBytes + cfg.data.byteOffset));
        }

        // Destroy HSA signal for DMA executor
#if !defined(__NVCC__)
        if(exeDevice.exeType == EXE_GPU_DMA &&
           (t.exeSubIndex != -1 || cfg.dma.useHsaCopy))
        {
            ERR_CHECK(hsa_signal_destroy(rss.signal));
        }
#endif

        // Destroy NIC related resources
#ifdef NIC_EXEC_ENABLED
        if(IsNicExeType(exeDevice.exeType))
        {
            ERR_CHECK(TeardownNicTransferResources(rss));
        }
#endif
    }

    // Teardown additional requirements for GPU-based executors
    if(exeDevice.exeType == EXE_GPU_GFX || exeDevice.exeType == EXE_GPU_DMA)
    {
        for(auto stream : exeInfo.streams)
            ERR_CHECK(hipStreamDestroy(stream));
        if(cfg.gfx.useHipEvents || cfg.dma.useHipEvents)
        {
            for(auto event : exeInfo.startEvents)
                ERR_CHECK(hipEventDestroy(event));
            for(auto event : exeInfo.stopEvents)
                ERR_CHECK(hipEventDestroy(event));
        }
    }

    if(exeDevice.exeType == EXE_GPU_GFX)
    {
#if !defined(__NVCC__)
        MemType memType = MEM_GPU;
#else
        MemType memType = MEM_MANAGED;
#endif
        ERR_CHECK(DeallocateMemory(memType, exeInfo.subExecParamGpu,
                                   exeInfo.totalSubExecs * sizeof(SubExecParam)));
    }

    return ERR_NONE;
}

// CPU Executor-related functions
//========================================================================================

// Kernel for CPU execution (run by a single subexecutor)
static void
CpuReduceKernel(SubExecParam const& p, int numSubIterations)
{
    if(p.N == 0) return;

    int subIteration = 0;
    do
    {
        int const& numSrcs = p.numSrcs;
        int const& numDsts = p.numDsts;

        if(numSrcs == 0)
        {
            for(int i = 0; i < numDsts; ++i)
            {
                memset(p.dst[i], MEMSET_CHAR, p.N * sizeof(float));
                // for (int j = 0; j < p.N; j++) p.dst[i][j] = MEMSET_VAL;
            }
        }
        else if(numSrcs == 1)
        {
            float const* __restrict__ src = p.src[0];
            if(numDsts == 0)
            {
                float sum = 0.0;
                for(size_t j = 0; j < p.N; j++)
                    sum += p.src[0][j];

                // Add a dummy check to ensure the read is not optimized out
                if(sum != sum)
                {
                    printf("[ERROR] Nan detected\n");
                }
            }
            else
            {
                for(int i = 0; i < numDsts; ++i)
                    memcpy(p.dst[i], src, p.N * sizeof(float));
            }
        }
        else
        {
            float sum = 0.0f;
            for(size_t j = 0; j < p.N; j++)
            {
                sum = p.src[0][j];
                for(int i = 1; i < numSrcs; i++)
                    sum += p.src[i][j];
                for(int i = 0; i < numDsts; i++)
                    p.dst[i][j] = sum;
            }
        }
    } while(++subIteration != numSubIterations);
}

// Execution of a single CPU Transfers
static void
ExecuteCpuTransfer(int const iteration, ConfigOptions const& cfg, int const exeIndex,
                   TransferResources& rss)
{
    (void) exeIndex;  // May be unused
    auto                cpuStart = std::chrono::high_resolution_clock::now();
    vector<std::thread> childThreads;

    for(auto const& subExecParam : rss.subExecParamCpu)
        childThreads.emplace_back(std::thread(CpuReduceKernel, std::cref(subExecParam),
                                              cfg.general.numSubIterations));

    for(auto& subExecThread : childThreads)
        subExecThread.join();
    childThreads.clear();

    auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec =
        (std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() *
         1000.0) /
        cfg.general.numSubIterations;

    if(iteration >= 0)
    {
        rss.totalDurationMsec += deltaMsec;
        if(cfg.general.recordPerIteration) rss.perIterMsec.push_back(deltaMsec);
    }
}

// Execution of a single CPU executor
static ErrResult
RunCpuExecutor(int const iteration, ConfigOptions const& cfg, int const exeIndex,
               ExeInfo& exeInfo)
{
    // NUMA node affinity not supported
    auto cpuStart = std::chrono::high_resolution_clock::now();

    vector<std::thread> asyncTransfers;
    for(auto& rss : exeInfo.resources)
    {
        asyncTransfers.emplace_back(std::thread(ExecuteCpuTransfer, iteration,
                                                std::cref(cfg), exeIndex, std::ref(rss)));
    }
    for(auto& asyncTransfer : asyncTransfers)
        asyncTransfer.join();

    auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec =
        std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() *
        1000.0 / cfg.general.numSubIterations;

    if(iteration >= 0) exeInfo.totalDurationMsec += deltaMsec;
    return ERR_NONE;
}

#ifdef NIC_EXEC_ENABLED
// Execution of a single NIC Transfer
static ErrResult
ExecuteNicTransfer(int const iteration, ConfigOptions const& cfg, int const exeIndex,
                   TransferResources& rss)
{
    // Loop over each of the queue pairs and post the send
    ibv_send_wr* badWorkReq;
    for(int qpIndex = 0; qpIndex < rss.qpCount; qpIndex++)
    {
        int error = ibv_post_send(rss.srcQueuePairs[qpIndex],
                                  &rss.sendWorkRequests[qpIndex], &badWorkReq);
        if(error)
            return {
                ERR_FATAL,
                "Transfer %d: Error when calling ibv_post_send for QP %d Error code %d\n",
                rss.transferIdx, qpIndex, error
            };
    }
    return ERR_NONE;
}

// Execution of a single NIC executor
static ErrResult
RunNicExecutor(int const iteration, ConfigOptions const& cfg, int const exeIndex,
               ExeInfo& exeInfo)
{
    // Switch to the closest NUMA node to this NIC (not supported without NUMA)
    if(cfg.nic.useNuma)
    {
        int numaNode = GetIbvDeviceList()[exeIndex].numaNode;
        (void) numaNode;  // Suppress unused variable warning
    }

    auto                transferCount = exeInfo.resources.size();
    std::vector<double> totalTimeMsec(transferCount, 0.0);

    int  subIterations = 0;
    auto cpuStart      = std::chrono::high_resolution_clock::now();
    std::vector<std::chrono::high_resolution_clock::time_point> transferTimers(
        transferCount);

    do
    {
        std::vector<uint8_t> receivedQPs(transferCount, 0);
        // post the sends
        for(auto i = 0; i < transferCount; i++)
        {
            transferTimers[i] = std::chrono::high_resolution_clock::now();
            ERR_CHECK(ExecuteNicTransfer(iteration, cfg, exeIndex, exeInfo.resources[i]));
        }
        // poll for completions
        size_t completedTransfers = 0;
        while(completedTransfers < transferCount)
        {
            for(auto i = 0; i < transferCount; i++)
            {
                if(receivedQPs[i] < exeInfo.resources[i].qpCount)
                {
                    auto& rss = exeInfo.resources[i];
                    // Poll the completion queue until all queue pairs are complete
                    // The order of completion doesn't matter because this completion
                    // queue is dedicated to this Transfer
                    ibv_wc wc;
                    int    nc = ibv_poll_cq(rss.srcCompQueue, 1, &wc);
                    if(nc > 0)
                    {
                        receivedQPs[i]++;
                        if(wc.status != IBV_WC_SUCCESS)
                        {
                            return { ERR_FATAL,
                                     "Transfer %d: Received unsuccessful work completion",
                                     rss.transferIdx };
                        }
                    }
                    else if(nc < 0)
                    {
                        return { ERR_FATAL,
                                 "Transfer %d: Received negative work completion",
                                 rss.transferIdx };
                    }
                    if(receivedQPs[i] == rss.qpCount)
                    {
                        auto cpuDelta =
                            std::chrono::high_resolution_clock::now() - transferTimers[i];
                        double deltaMsec =
                            std::chrono::duration_cast<std::chrono::duration<double>>(
                                cpuDelta)
                                .count() *
                            1000.0;
                        if(iteration >= 0)
                        {
                            totalTimeMsec[i] += deltaMsec;
                        }
                        completedTransfers++;
                    }
                }
            }
        }
    } while(++subIterations < cfg.general.numSubIterations);

    auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec =
        std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() *
        1000.0 / cfg.general.numSubIterations;

    if(iteration >= 0)
    {
        exeInfo.totalDurationMsec += deltaMsec;
        for(int i = 0; i < transferCount; i++)
        {
            auto&  rss              = exeInfo.resources[i];
            double transferTimeMsec = totalTimeMsec[i] / cfg.general.numSubIterations;
            rss.totalDurationMsec += transferTimeMsec;
            if(cfg.general.recordPerIteration)
                rss.perIterMsec.push_back(transferTimeMsec);
        }
    }
    return ERR_NONE;
}
#endif
// GFX Executor-related functions
//========================================================================================

// Converts register value to a CU/SM index
static uint32_t
GetId(uint32_t hwId)
{
#if defined(__NVCC_)
    return hwId;
#else
    // Based on instinct-mi200-cdna2-instruction-set-architecture.pdf
    int const shId = (hwId >> 12) & 1;
    int const cuId = (hwId >> 8) & 15;
    int const seId = (hwId >> 13) & 3;
    return (shId << 5) + (cuId << 2) + seId;
#endif
}

// Device level timestamp function
__device__ int64_t
GetTimestamp()
{
#if defined(__NVCC__)
    int64_t result;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
    return result;
#else
    return wall_clock64();
#endif
}

// Helper function for memset
template <typename T>
__device__ __forceinline__ T
MemsetVal();
template <>
__device__ __forceinline__ float
MemsetVal()
{
    return MEMSET_VAL;
};
template <>
__device__ __forceinline__ float2
MemsetVal()
{
    return make_float2(MEMSET_VAL, MEMSET_VAL);
};
template <>
__device__ __forceinline__ float4
MemsetVal()
{
    return make_float4(MEMSET_VAL, MEMSET_VAL, MEMSET_VAL, MEMSET_VAL);
}

// Helper function for temporal/non-temporal reads / writes
#define TEMPORAL_NONE  0
#define TEMPORAL_LOAD  1
#define TEMPORAL_STORE 2
#define TEMPORAL_BOTH  3

template <int TEMPORAL_MODE>
__device__ __forceinline__ void
Load(float const* src, float& dst)
{
    if(TEMPORAL_MODE & TEMPORAL_LOAD)
    {
#if !defined(__NVCC__)
        dst = __builtin_nontemporal_load(src);

#endif
    }
    else
    {
        dst = *src;
    }
}

template <int TEMPORAL_MODE>
__device__ __forceinline__ void
Load(float2 const* src, float2& dst)
{
    if(TEMPORAL_MODE & TEMPORAL_LOAD)
    {
#if !defined(__NVCC__)
        dst.x = __builtin_nontemporal_load(&(src->x));
        dst.y = __builtin_nontemporal_load(&(src->y));
#endif
    }
    else
    {
        dst = *src;
    }
}

template <int TEMPORAL_MODE>
__device__ __forceinline__ void
Load(float4 const* src, float4& dst)
{
    if(TEMPORAL_MODE & TEMPORAL_LOAD)
    {
#if !defined(__NVCC__)
        dst.x = __builtin_nontemporal_load(&(src->x));
        dst.y = __builtin_nontemporal_load(&(src->y));
        dst.z = __builtin_nontemporal_load(&(src->z));
        dst.w = __builtin_nontemporal_load(&(src->w));
#endif
    }
    else
    {
        dst = *src;
    }
}

template <int TEMPORAL_MODE>
__device__ __forceinline__ void
Store(float const& src, float* dst)
{
    if(TEMPORAL_MODE & TEMPORAL_STORE)
    {
#if !defined(__NVCC__)
        __builtin_nontemporal_store(src, dst);
#endif
    }
    else
    {
        *dst = src;
    }
}

template <int TEMPORAL_MODE>
__device__ __forceinline__ void
Store(float2 const& src, float2* dst)
{
    if(TEMPORAL_MODE & TEMPORAL_STORE)
    {
#if !defined(__NVCC__)
        __builtin_nontemporal_store(src.x, &(dst->x));
        __builtin_nontemporal_store(src.y, &(dst->y));
#endif
    }
    else
    {
        *dst = src;
    }
}

template <int TEMPORAL_MODE>
__device__ __forceinline__ void
Store(float4 const& src, float4* dst)
{
    if(TEMPORAL_MODE & TEMPORAL_STORE)
    {
#if !defined(__NVCC__)
        __builtin_nontemporal_store(src.x, &(dst->x));
        __builtin_nontemporal_store(src.y, &(dst->y));
        __builtin_nontemporal_store(src.z, &(dst->z));
        __builtin_nontemporal_store(src.w, &(dst->w));
#endif
    }
    else
    {
        *dst = src;
    }
}

// Kernel for GFX execution
template <typename PACKED_FLOAT, int BLOCKSIZE, int UNROLL, int TEMPORAL_MODE>
__global__ void
__launch_bounds__(BLOCKSIZE)
    GpuReduceKernel(SubExecParam* params, int waveOrder, int numSubIterations)
{
    int64_t startCycle;
    if(threadIdx.x == 0) startCycle = GetTimestamp();

    SubExecParam& p = params[blockIdx.y];

    // Filter by XCC
#if !defined(__NVCC__)
    int32_t xccId;
    GetXccId(xccId);
    if(p.preferredXccId != -1 && xccId != p.preferredXccId) return;
#endif

    // Collect data information
    int32_t const numSrcs = p.numSrcs;
    int32_t const numDsts = p.numDsts;
    PACKED_FLOAT const* __restrict__ srcFloatPacked[MAX_SRCS];
    PACKED_FLOAT* __restrict__ dstFloatPacked[MAX_DSTS];
    for(int i = 0; i < numSrcs; i++)
        srcFloatPacked[i] = (PACKED_FLOAT const*) p.src[i];
    for(int i = 0; i < numDsts; i++)
        dstFloatPacked[i] = (PACKED_FLOAT*) p.dst[i];

    // Operate on wavefront granularity
    int32_t const nTeams =
        p.teamSize;  // Number of threadblocks working together on this subarray
    int32_t const teamIdx = p.teamIdx;  // Index of this threadblock within the team
    int32_t const nWaves =
        BLOCKSIZE / warpSize;  // Number of wavefronts within this threadblock
    int32_t const waveIdx =
        threadIdx.x / warpSize;  // Index of this wavefront within the threadblock
    int32_t const tIdx = threadIdx.x % warpSize;  // Thread index within wavefront

    size_t const numPackedFloat = p.N / (sizeof(PACKED_FLOAT) / sizeof(float));

    int32_t teamStride, waveStride, unrlStride, teamStride2, waveStride2;
    switch(waveOrder)
    {
        case 0: /* U,W,C */
            unrlStride  = 1;
            waveStride  = UNROLL;
            teamStride  = UNROLL * nWaves;
            teamStride2 = nWaves;
            waveStride2 = 1;
            break;
        case 1: /* U,C,W */
            unrlStride  = 1;
            teamStride  = UNROLL;
            waveStride  = UNROLL * nTeams;
            teamStride2 = 1;
            waveStride2 = nTeams;
            break;
        case 2: /* W,U,C */
            waveStride  = 1;
            unrlStride  = nWaves;
            teamStride  = nWaves * UNROLL;
            teamStride2 = nWaves;
            waveStride2 = 1;
            break;
        case 3: /* W,C,U */
            waveStride  = 1;
            teamStride  = nWaves;
            unrlStride  = nWaves * nTeams;
            teamStride2 = nWaves;
            waveStride2 = 1;
            break;
        case 4: /* C,U,W */
            teamStride  = 1;
            unrlStride  = nTeams;
            waveStride  = nTeams * UNROLL;
            teamStride2 = 1;
            waveStride2 = nTeams;
            break;
        case 5: /* C,W,U */
            teamStride  = 1;
            waveStride  = nTeams;
            unrlStride  = nTeams * nWaves;
            teamStride2 = 1;
            waveStride2 = nTeams;
            break;
    }

    int subIterations = 0;
    while(1)
    {
        // First loop: Each wavefront in the team works on UNROLL PACKED_FLOAT per thread
        size_t const loop1Stride = nTeams * nWaves * UNROLL * warpSize;
        size_t const loop1Limit  = numPackedFloat / loop1Stride * loop1Stride;
        {
            PACKED_FLOAT val[UNROLL];
            PACKED_FLOAT tmp[UNROLL];
            if(numSrcs == 0)
            {
#pragma unroll
                for(int u = 0; u < UNROLL; u++)
                    val[u] = MemsetVal<PACKED_FLOAT>();
            }

            for(size_t idx =
                    (teamIdx * teamStride + waveIdx * waveStride) * warpSize + tIdx;
                idx < loop1Limit; idx += loop1Stride)
            {
                // Read sources into memory and accumulate in registers
                if(numSrcs)
                {
#pragma unroll
                    for(int u = 0; u < UNROLL; u++)
                        Load<TEMPORAL_MODE>(
                            &srcFloatPacked[0][idx + u * unrlStride * warpSize], val[u]);

                    for(int s = 1; s < numSrcs; s++)
                    {
#pragma unroll
                        for(int u = 0; u < UNROLL; u++)
                            Load<TEMPORAL_MODE>(
                                &srcFloatPacked[s][idx + u * unrlStride * warpSize],
                                tmp[u]);
#pragma unroll
                        for(int u = 0; u < UNROLL; u++)
                            val[u] += tmp[u];
                    }
                }

                // Write accumulation to all outputs
                for(int d = 0; d < numDsts; d++)
                {
#pragma unroll
                    for(int u = 0; u < UNROLL; u++)
                        Store<TEMPORAL_MODE>(
                            val[u], &dstFloatPacked[d][idx + u * unrlStride * warpSize]);
                }
            }
        }

        // Second loop: Deal with remaining PACKED_FLOAT
        {
            if(loop1Limit < numPackedFloat)
            {
                PACKED_FLOAT val, tmp;
                if(numSrcs == 0) val = MemsetVal<PACKED_FLOAT>();

                size_t const loop2Stride = nTeams * nWaves * warpSize;
                for(size_t idx =
                        loop1Limit +
                        (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx;
                    idx < numPackedFloat; idx += loop2Stride)
                {
                    if(numSrcs)
                    {
                        Load<TEMPORAL_MODE>(&srcFloatPacked[0][idx], val);
                        for(int s = 1; s < numSrcs; s++)
                        {
                            Load<TEMPORAL_MODE>(&srcFloatPacked[s][idx], tmp);
                            val += tmp;
                        }
                    }
                    for(int d = 0; d < numDsts; d++)
                        Store<TEMPORAL_MODE>(val, &dstFloatPacked[d][idx]);
                }
            }
        }

        // Third loop; Deal with remaining floats
        {
            if(numPackedFloat * (sizeof(PACKED_FLOAT) / sizeof(float)) < p.N)
            {
                float val, tmp;
                if(numSrcs == 0) val = MemsetVal<float>();

                size_t const loop3Stride = nTeams * nWaves * warpSize;
                for(size_t idx =
                        numPackedFloat * (sizeof(PACKED_FLOAT) / sizeof(float)) +
                        (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx;
                    idx < p.N; idx += loop3Stride)
                {
                    if(numSrcs)
                    {
                        Load<TEMPORAL_MODE>(&p.src[0][idx], val);
                        for(int s = 1; s < numSrcs; s++)
                        {
                            Load<TEMPORAL_MODE>(&p.src[s][idx], tmp);
                            val += tmp;
                        }
                    }

                    for(int d = 0; d < numDsts; d++)
                        Store<TEMPORAL_MODE>(val, &p.dst[d][idx]);
                }
            }
        }

        if(++subIterations == numSubIterations) break;
    }

    // Wait for all threads to finish
    __syncthreads();
    if(threadIdx.x == 0)
    {
        __threadfence_system();
        p.stopCycle  = GetTimestamp();
        p.startCycle = startCycle;
        GetHwId(p.hwId);
        GetXccId(p.xccId);
    }
}

#define GPU_KERNEL_TEMPORAL_DECL(BLOCKSIZE, UNROLL, DWORD)                               \
    {                                                                                    \
        GpuReduceKernel<DWORD, BLOCKSIZE, UNROLL, TEMPORAL_NONE>,                        \
            GpuReduceKernel<DWORD, BLOCKSIZE, UNROLL, TEMPORAL_LOAD>,                    \
            GpuReduceKernel<DWORD, BLOCKSIZE, UNROLL, TEMPORAL_STORE>,                   \
            GpuReduceKernel<DWORD, BLOCKSIZE, UNROLL, TEMPORAL_BOTH>                     \
    }

#define GPU_KERNEL_DWORD_DECL(BLOCKSIZE, UNROLL)                                         \
    {                                                                                    \
        GPU_KERNEL_TEMPORAL_DECL(BLOCKSIZE, UNROLL, float),                              \
            GPU_KERNEL_TEMPORAL_DECL(BLOCKSIZE, UNROLL, float2),                         \
            GPU_KERNEL_TEMPORAL_DECL(BLOCKSIZE, UNROLL, float4)                          \
    }

#define GPU_KERNEL_UNROLL_DECL(BLOCKSIZE)                                                \
    {                                                                                    \
        GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 1), GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 2),        \
            GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 3), GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 4),    \
            GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 5), GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 6),    \
            GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 7), GPU_KERNEL_DWORD_DECL(BLOCKSIZE, 8)     \
    }

// Table of all GPU Reduction kernel functions (templated blocksize / unroll / dword size)
typedef void (*GpuKernelFuncPtr)(SubExecParam*, int, int);
GpuKernelFuncPtr GpuKernelTable[MAX_WAVEGROUPS][MAX_UNROLL][3][4] = {
    GPU_KERNEL_UNROLL_DECL(64),  GPU_KERNEL_UNROLL_DECL(128),
    GPU_KERNEL_UNROLL_DECL(192), GPU_KERNEL_UNROLL_DECL(256),
    GPU_KERNEL_UNROLL_DECL(320), GPU_KERNEL_UNROLL_DECL(384),
    GPU_KERNEL_UNROLL_DECL(448), GPU_KERNEL_UNROLL_DECL(512),
    GPU_KERNEL_UNROLL_DECL(576), GPU_KERNEL_UNROLL_DECL(640),
    GPU_KERNEL_UNROLL_DECL(704), GPU_KERNEL_UNROLL_DECL(768),
    GPU_KERNEL_UNROLL_DECL(832), GPU_KERNEL_UNROLL_DECL(896),
    GPU_KERNEL_UNROLL_DECL(960), GPU_KERNEL_UNROLL_DECL(1024),
};
#undef GPU_KERNEL_UNROLL_DECL
#undef GPU_KERNEL_DWORD_DECL
#undef GPU_KERNEL_TEMPORAL_DECL

// Execute a single GPU Transfer (when using 1 stream per Transfer)
static ErrResult
ExecuteGpuTransfer(int const iteration, hipStream_t const stream,
                   hipEvent_t const startEvent, hipEvent_t const stopEvent,
                   int const xccDim, ConfigOptions const& cfg, TransferResources& rss)
{
    auto cpuStart = std::chrono::high_resolution_clock::now();

    int        numSubExecs = rss.subExecParamCpu.size();
    dim3 const gridSize(xccDim, numSubExecs, 1);
    dim3 const blockSize(cfg.gfx.blockSize, 1);

    int  wordSizeIdx = cfg.gfx.wordSize == 1 ? 0 : cfg.gfx.wordSize == 2 ? 1 : 2;
    auto gpuKernel = GpuKernelTable[cfg.gfx.blockSize / 64 - 1][cfg.gfx.unrollFactor - 1]
                                   [wordSizeIdx][cfg.gfx.temporalMode];

#if defined(__NVCC__)
    if(startEvent != NULL) ERR_CHECK(hipEventRecord(startEvent, stream));
    gpuKernel<<<gridSize, blockSize, 0, stream>>>(
        rss.subExecParamGpuPtr, cfg.gfx.waveOrder, cfg.general.numSubIterations);
    if(stopEvent != NULL) ERR_CHECK(hipEventRecord(stopEvent, stream));
#else
    hipExtLaunchKernelGGL(gpuKernel, gridSize, blockSize, 0, stream, startEvent,
                          stopEvent, 0, rss.subExecParamGpuPtr, cfg.gfx.waveOrder,
                          cfg.general.numSubIterations);
#endif

    ERR_CHECK(hipStreamSynchronize(stream));

    auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec =
        std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() *
        1000.0 / cfg.general.numSubIterations;

    if(iteration >= 0)
    {
        double deltaMsec = cpuDeltaMsec;
        if(startEvent != NULL)
        {
            float gpuDeltaMsec;
            ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));
            deltaMsec = gpuDeltaMsec / cfg.general.numSubIterations;
        }
        rss.totalDurationMsec += deltaMsec;
        if(cfg.general.recordPerIteration)
        {
            rss.perIterMsec.push_back(deltaMsec);
            std::set<std::pair<int, int>> CUs;
            for(int i = 0; i < numSubExecs; i++)
            {
                CUs.insert(std::make_pair(rss.subExecParamGpuPtr[i].xccId,
                                          GetId(rss.subExecParamGpuPtr[i].hwId)));
            }
            rss.perIterCUs.push_back(CUs);
        }
    }
    return ERR_NONE;
}

// Execute a single GPU executor
static ErrResult
RunGpuExecutor(int const iteration, ConfigOptions const& cfg, int const exeIndex,
               ExeInfo& exeInfo)
{
    auto cpuStart = std::chrono::high_resolution_clock::now();
    ERR_CHECK(hipSetDevice(exeIndex));

    int xccDim = exeInfo.useSubIndices ? exeInfo.numSubIndices : 1;

    if(cfg.gfx.useMultiStream)
    {
        // Launch each Transfer separately in its own stream
        vector<std::future<ErrResult>> asyncTransfers;
        for(size_t i = 0; i < exeInfo.streams.size(); i++)
        {
            asyncTransfers.emplace_back(std::async(
                std::launch::async, ExecuteGpuTransfer, iteration, exeInfo.streams[i],
                cfg.gfx.useHipEvents ? exeInfo.startEvents[i] : NULL,
                cfg.gfx.useHipEvents ? exeInfo.stopEvents[i] : NULL, xccDim,
                std::cref(cfg), std::ref(exeInfo.resources[i])));
        }
        for(auto& asyncTransfer : asyncTransfers)
            ERR_CHECK(asyncTransfer.get());
    }
    else
    {
        // Combine all the Transfers into a single kernel launch
        int         numSubExecs = exeInfo.totalSubExecs;
        dim3 const  gridSize(xccDim, numSubExecs, 1);
        dim3 const  blockSize(cfg.gfx.blockSize, 1);
        hipStream_t stream = exeInfo.streams[0];

        int  wordSizeIdx = cfg.gfx.wordSize == 1 ? 0 : cfg.gfx.wordSize == 2 ? 1 : 2;
        auto gpuKernel =
            GpuKernelTable[cfg.gfx.blockSize / 64 - 1][cfg.gfx.unrollFactor - 1]
                          [wordSizeIdx][cfg.gfx.temporalMode];

#if defined(__NVCC__)
        if(cfg.gfx.useHipEvents)
            ERR_CHECK(hipEventRecord(exeInfo.startEvents[0], stream));
        gpuKernel<<<gridSize, blockSize, 0, stream>>>(
            exeInfo.subExecParamGpu, cfg.gfx.waveOrder, cfg.general.numSubIterations);
        if(cfg.gfx.useHipEvents) ERR_CHECK(hipEventRecord(exeInfo.stopEvents[0], stream));
#else
        hipExtLaunchKernelGGL(gpuKernel, gridSize, blockSize, 0, stream,
                              cfg.gfx.useHipEvents ? exeInfo.startEvents[0] : NULL,
                              cfg.gfx.useHipEvents ? exeInfo.stopEvents[0] : NULL, 0,
                              exeInfo.subExecParamGpu, cfg.gfx.waveOrder,
                              cfg.general.numSubIterations);
#endif
        ERR_CHECK(hipStreamSynchronize(stream));
    }
    auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec =
        std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() *
        1000.0 / cfg.general.numSubIterations;

    if(iteration >= 0)
    {
        if(cfg.gfx.useHipEvents && !cfg.gfx.useMultiStream)
        {
            float gpuDeltaMsec;
            ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, exeInfo.startEvents[0],
                                          exeInfo.stopEvents[0]));
            gpuDeltaMsec /= cfg.general.numSubIterations;
            exeInfo.totalDurationMsec += gpuDeltaMsec;
        }
        else
        {
            exeInfo.totalDurationMsec += cpuDeltaMsec;
        }

        // Determine timing for each of the individual transfers that were part of this
        // launch
        if(!cfg.gfx.useMultiStream)
        {
            for(size_t i = 0; i < exeInfo.resources.size(); i++)
            {
                TransferResources& rss           = exeInfo.resources[i];
                long long          minStartCycle = std::numeric_limits<long long>::max();
                long long          maxStopCycle  = std::numeric_limits<long long>::min();
                std::set<std::pair<int, int>> CUs;

                for(auto subExecIdx : rss.subExecIdx)
                {
                    minStartCycle = std::min(
                        minStartCycle, exeInfo.subExecParamGpu[subExecIdx].startCycle);
                    maxStopCycle = std::max(
                        maxStopCycle, exeInfo.subExecParamGpu[subExecIdx].stopCycle);
                    if(cfg.general.recordPerIteration)
                    {
                        CUs.insert(std::make_pair(
                            exeInfo.subExecParamGpu[subExecIdx].xccId,
                            GetId(exeInfo.subExecParamGpu[subExecIdx].hwId)));
                    }
                }
                double deltaMsec =
                    (maxStopCycle - minStartCycle) / (double) (exeInfo.wallClockRate);
                deltaMsec /= cfg.general.numSubIterations;
                rss.totalDurationMsec += deltaMsec;
                if(cfg.general.recordPerIteration)
                {
                    rss.perIterMsec.push_back(deltaMsec);
                    rss.perIterCUs.push_back(CUs);
                }
            }
        }
    }
    return ERR_NONE;
}

// DMA Executor-related functions
//========================================================================================

// Execute a single DMA Transfer
static ErrResult
ExecuteDmaTransfer(int const iteration, bool const useSubIndices,
                   hipStream_t const stream, hipEvent_t const startEvent,
                   hipEvent_t const stopEvent, ConfigOptions const& cfg,
                   TransferResources& resources)
{
    auto cpuStart = std::chrono::high_resolution_clock::now();

    int subIterations = 0;
    if(!useSubIndices && !cfg.dma.useHsaCopy)
    {
        if(cfg.dma.useHipEvents) ERR_CHECK(hipEventRecord(startEvent, stream));

        // Use hipMemcpy
        do
        {
            ERR_CHECK(hipMemcpyAsync(resources.dstMem[0], resources.srcMem[0],
                                     resources.numBytes, hipMemcpyDefault, stream));
        } while(++subIterations != cfg.general.numSubIterations);

        if(cfg.dma.useHipEvents) ERR_CHECK(hipEventRecord(stopEvent, stream));
        ERR_CHECK(hipStreamSynchronize(stream));
    }
    else
    {
#if defined(__NVCC__)
        return { ERR_FATAL, "HSA copy not supported on NVIDIA hardware" };
#else
        // Use HSA async copy
        do
        {
            hsa_signal_store_screlease(resources.signal, 1);
            if(!useSubIndices)
            {
                ERR_CHECK(hsa_amd_memory_async_copy(
                    resources.dstMem[0], resources.dstAgent, resources.srcMem[0],
                    resources.srcAgent, resources.numBytes, 0, NULL, resources.signal));
            }
            else
            {
                HSA_CALL(hsa_amd_memory_async_copy_on_engine(
                    resources.dstMem[0], resources.dstAgent, resources.srcMem[0],
                    resources.srcAgent, resources.numBytes, 0, NULL, resources.signal,
                    resources.sdmaEngineId, true));
            }
            // Wait for SDMA transfer to complete
            while(hsa_signal_wait_scacquire(resources.signal, HSA_SIGNAL_CONDITION_LT, 1,
                                            UINT64_MAX, HSA_WAIT_STATE_ACTIVE) >= 1)
                ;
        } while(++subIterations != cfg.general.numSubIterations);
#endif
    }
    auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec =
        std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() *
        1000.0 / cfg.general.numSubIterations;

    if(iteration >= 0)
    {
        double deltaMsec = cpuDeltaMsec;
        if(!useSubIndices && !cfg.dma.useHsaCopy && cfg.dma.useHipEvents)
        {
            float gpuDeltaMsec;
            ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));
            deltaMsec = gpuDeltaMsec / cfg.general.numSubIterations;
        }
        resources.totalDurationMsec += deltaMsec;
        if(cfg.general.recordPerIteration) resources.perIterMsec.push_back(deltaMsec);
    }
    return ERR_NONE;
}

// Execute a single DMA executor
static ErrResult
RunDmaExecutor(int const iteration, ConfigOptions const& cfg, int const exeIndex,
               ExeInfo& exeInfo)
{
    auto cpuStart = std::chrono::high_resolution_clock::now();
    ERR_CHECK(hipSetDevice(exeIndex));

    vector<std::future<ErrResult>> asyncTransfers;
    for(size_t i = 0; i < exeInfo.resources.size(); i++)
    {
        asyncTransfers.emplace_back(std::async(
            std::launch::async, ExecuteDmaTransfer, iteration, exeInfo.useSubIndices,
            exeInfo.streams[i], cfg.dma.useHipEvents ? exeInfo.startEvents[i] : NULL,
            cfg.dma.useHipEvents ? exeInfo.stopEvents[i] : NULL, std::cref(cfg),
            std::ref(exeInfo.resources[i])));
    }

    for(auto& asyncTransfer : asyncTransfers)
        ERR_CHECK(asyncTransfer.get());

    auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec =
        std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() *
        1000.0 / cfg.general.numSubIterations;
    if(iteration >= 0) exeInfo.totalDurationMsec += deltaMsec;
    return ERR_NONE;
}

// Executor-related functions
//========================================================================================
static ErrResult
RunExecutor(int const iteration, ConfigOptions const& cfg, ExeDevice const& exeDevice,
            ExeInfo& exeInfo)
{
    switch(exeDevice.exeType)
    {
        case EXE_CPU: return RunCpuExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
        case EXE_GPU_GFX:
            return RunGpuExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
        case EXE_GPU_DMA:
            return RunDmaExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
#ifdef NIC_EXEC_ENABLED
        case EXE_NIC: return RunNicExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
#endif
        default: return { ERR_FATAL, "Unsupported executor (%d)", exeDevice.exeType };
    }
}

}  // End of anonymous namespace
//========================================================================================
/// @endcond

ErrResult::ErrResult(ErrType err)
: errType(err)
, errMsg(""){};

ErrResult::ErrResult(hipError_t err)
{
    if(err == hipSuccess)
    {
        this->errType = ERR_NONE;
        this->errMsg  = "";
    }
    else
    {
        this->errType = ERR_FATAL;
        this->errMsg  = std::string("HIP Error: ") + hipGetErrorString(err);
    }
}

#if !defined(__NVCC__)
ErrResult::ErrResult(hsa_status_t err)
{
    if(err == HSA_STATUS_SUCCESS)
    {
        this->errType = ERR_NONE;
        this->errMsg  = "";
    }
    else
    {
        const char* errString = NULL;
        hsa_status_string(err, &errString);
        this->errType = ERR_FATAL;
        this->errMsg  = std::string("HSA Error: ") + errString;
    }
}
#endif

ErrResult::ErrResult(ErrType errType, const char* format, ...)
{
    this->errType = errType;
    va_list args, args_temp;
    va_start(args, format);
    va_copy(args_temp, args);

    int len = vsnprintf(nullptr, 0, format, args);
    if(len < 0)
    {
        va_end(args_temp);
        va_end(args);
    }
    else
    {
        this->errMsg.resize(len);
        vsnprintf(this->errMsg.data(), len + 1, format, args_temp);
    }
    va_end(args_temp);
    va_end(args);
}

bool
RunTransfers(ConfigOptions const& cfg, std::vector<Transfer> const& transfers,
             TestResults& results)
{
    // Clear all errors;
    auto& errResults = results.errResults;
    errResults.clear();

    // Check for valid configuration
    if(ConfigOptionsHaveErrors(cfg, errResults)) return false;

    // Check for valid transfers
    if(TransfersHaveErrors(cfg, transfers, errResults)) return false;

    // Collect up transfers by executor
    int                          minNumSrcs  = MAX_SRCS + 1;
    int                          maxNumSrcs  = 0;
    size_t                       maxNumBytes = 0;
    std::map<ExeDevice, ExeInfo> executorMap;
    for(size_t i = 0; i < transfers.size(); i++)
    {
        Transfer const& t = transfers[i];
        ExeDevice       exeDevice;
        ERR_APPEND(GetActualExecutor(cfg, t.exeDevice, exeDevice), errResults);

        TransferResources resource = {};
        resource.transferIdx       = i;

        ExeInfo& exeInfo = executorMap[exeDevice];
        exeInfo.totalBytes += t.numBytes;
        exeInfo.totalSubExecs += t.numSubExecs;
        exeInfo.useSubIndices |=
            (t.exeSubIndex != -1 ||
             (t.exeDevice.exeType == EXE_GPU_GFX && !cfg.gfx.prefXccTable.empty()));
        exeInfo.resources.push_back(resource);
        minNumSrcs  = std::min(minNumSrcs, (int) t.srcs.size());
        maxNumSrcs  = std::max(maxNumSrcs, (int) t.srcs.size());
        maxNumBytes = std::max(maxNumBytes, t.numBytes);
    }

    // Loop over each executor and prepare
    // - Allocates memory for each Transfer
    // - Set up work for subexecutors
    vector<TransferResources*> transferResources;
    for(auto& exeInfoPair : executorMap)
    {
        ExeDevice const& exeDevice = exeInfoPair.first;
        ExeInfo&         exeInfo   = exeInfoPair.second;
        ERR_APPEND(PrepareExecutor(cfg, transfers, exeDevice, exeInfo), errResults);

        for(auto& resource : exeInfo.resources)
        {
            transferResources.push_back(&resource);
        }
    }

    // Prepare reference src/dst arrays - only once for largest size
    size_t                maxN = maxNumBytes / sizeof(float);
    vector<float>         outputBuffer(maxN);
    vector<vector<float>> dstReference(maxNumSrcs + 1, vector<float>(maxN));
    {
        size_t                initOffset = cfg.data.byteOffset / sizeof(float);
        vector<vector<float>> srcReference(maxNumSrcs, vector<float>(maxN));
        memset(dstReference[0].data(), MEMSET_CHAR, maxNumBytes);

        for(int numSrcs = 0; numSrcs < maxNumSrcs; numSrcs++)
        {
            PrepareReference(cfg, srcReference[numSrcs], numSrcs);
            for(size_t i = 0; i < maxN; i++)
            {
                dstReference[numSrcs + 1][i] =
                    (numSrcs == 0 ? 0 : dstReference[numSrcs][i]) +
                    srcReference[numSrcs][i];
            }
        }
        // Release un-used partial sums
        for(int numSrcs = 0; numSrcs < minNumSrcs; numSrcs++)
            dstReference[numSrcs].clear();

        // Initialize all src memory buffers
        for(auto resource : transferResources)
        {
            for(size_t srcIdx = 0; srcIdx < resource->srcMem.size(); srcIdx++)
            {
                ERR_APPEND(hipMemcpy(resource->srcMem[srcIdx] + initOffset,
                                     srcReference[srcIdx].data(), resource->numBytes,
                                     hipMemcpyDefault),
                           errResults);
            }
        }
    }

    // Pause before starting when running in iteractive mode
    if(cfg.general.useInteractive)
    {
        printf("Memory prepared:\n");

        for(size_t i = 0; i < transfers.size(); i++)
        {
            ExeInfo const& exeInfo = executorMap[transfers[i].exeDevice];
            (void) exeInfo;  // May be unused
            printf("Transfer %03zu:\n", i);
            for(size_t iSrc = 0; iSrc < transfers[i].srcs.size(); ++iSrc)
                printf("  SRC %0zu: %p\n", iSrc, transferResources[i]->srcMem[iSrc]);
            for(size_t iDst = 0; iDst < transfers[i].dsts.size(); ++iDst)
                printf("  DST %0zu: %p\n", iDst, transferResources[i]->dstMem[iDst]);
        }
        printf("Hit <Enter> to continue: ");
        if(scanf("%*c") != 0)
        {
            printf("[ERROR] Unexpected input\n");
            exit(1);
        }
        printf("\n");
    }

    // Perform iterations
    size_t numTimedIterations = 0;
    double totalCpuTimeSec    = 0.0;
    for(int iteration = -cfg.general.numWarmups;; iteration++)
    {
        // Stop if number of iterations/seconds has reached limit
        if(cfg.general.numIterations > 0 && iteration >= cfg.general.numIterations) break;
        if(cfg.general.numIterations < 0 && totalCpuTimeSec > -cfg.general.numIterations)
            break;

        // Start CPU timing for this iteration
        auto cpuStart = std::chrono::high_resolution_clock::now();

        // Execute all Transfers in parallel
        std::vector<std::future<ErrResult>> asyncExecutors;
        for(auto& exeInfoPair : executorMap)
        {
            asyncExecutors.emplace_back(
                std::async(std::launch::async, RunExecutor, iteration, std::cref(cfg),
                           std::cref(exeInfoPair.first), std::ref(exeInfoPair.second)));
        }

        // Wait for all threads to finish
        for(auto& asyncExecutor : asyncExecutors)
        {
            ERR_APPEND(asyncExecutor.get(), errResults);
        }

        // Stop CPU timing for this iteration
        auto   cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double deltaSec =
            std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() /
            cfg.general.numSubIterations;

        if(cfg.data.alwaysValidate)
        {
            ERR_APPEND(ValidateAllTransfers(cfg, transfers, transferResources,
                                            dstReference, outputBuffer),
                       errResults);
        }

        if(iteration >= 0)
        {
            ++numTimedIterations;
            totalCpuTimeSec += deltaSec;
        }
    }

    // Pause for interactive mode
    if(cfg.general.useInteractive)
    {
        printf("Transfers complete. Hit <Enter> to continue: ");
        if(scanf("%*c") != 0)
        {
            printf("[ERROR] Unexpected input\n");
            exit(1);
        }
        printf("\n");
    }

    // Validate results
    if(!cfg.data.alwaysValidate)
    {
        ERR_APPEND(ValidateAllTransfers(cfg, transfers, transferResources, dstReference,
                                        outputBuffer),
                   errResults);
    }

    // Prepare results
    results.exeResults.clear();
    results.tfrResults.clear();
    results.tfrResults.resize(transfers.size());
    results.numTimedIterations    = numTimedIterations;
    results.totalBytesTransferred = 0;
    results.avgTotalDurationMsec  = (totalCpuTimeSec * 1000.0) / numTimedIterations;
    results.overheadMsec          = results.avgTotalDurationMsec;
    for(auto& exeInfoPair : executorMap)
    {
        ExeDevice const& exeDevice = exeInfoPair.first;
        ExeInfo&         exeInfo   = exeInfoPair.second;

        // Copy over executor results
        ExeResult& exeResult      = results.exeResults[exeDevice];
        exeResult.numBytes        = exeInfo.totalBytes;
        exeResult.avgDurationMsec = exeInfo.totalDurationMsec / numTimedIterations;
        exeResult.avgBandwidthGbPerSec =
            (exeResult.numBytes / 1.0e6) / exeResult.avgDurationMsec;
        exeResult.sumBandwidthGbPerSec = 0.0;
        exeResult.transferIdx.clear();
        results.totalBytesTransferred += exeInfo.totalBytes;
        results.overheadMsec =
            std::min(results.overheadMsec,
                     (results.avgTotalDurationMsec - exeResult.avgDurationMsec));

        // Copy over transfer results
        for(auto const& rss : exeInfo.resources)
        {
            int const transferIdx = rss.transferIdx;
            exeResult.transferIdx.push_back(transferIdx);

            TransferResult& tfrResult = results.tfrResults[transferIdx];
            tfrResult.exeDevice       = exeDevice;
#ifdef NIC_EXEC_ENABLED
            tfrResult.exeDstDevice = { exeDevice.exeType, rss.dstNicIndex };
#else
            tfrResult.exeDstDevice = exeDevice;
#endif
            tfrResult.numBytes        = rss.numBytes;
            tfrResult.avgDurationMsec = rss.totalDurationMsec / numTimedIterations;
            tfrResult.avgBandwidthGbPerSec =
                (rss.numBytes / 1.0e6) / tfrResult.avgDurationMsec;
            if(cfg.general.recordPerIteration)
            {
                tfrResult.perIterMsec = rss.perIterMsec;
                tfrResult.perIterCUs  = rss.perIterCUs;
            }
            exeResult.sumBandwidthGbPerSec += tfrResult.avgBandwidthGbPerSec;
        }
    }
    results.avgTotalBandwidthGbPerSec =
        (results.totalBytesTransferred / 1.0e6) / results.avgTotalDurationMsec;

    // Teardown executors
    for(auto& exeInfoPair : executorMap)
    {
        ExeDevice const& exeDevice = exeInfoPair.first;
        ExeInfo&         exeInfo   = exeInfoPair.second;
        ERR_APPEND(TeardownExecutor(cfg, exeDevice, transfers, exeInfo), errResults);
    }

    return true;
}

int
GetIntAttribute(IntAttribute attribute)
{
    switch(attribute)
    {
        case ATR_GFX_MAX_BLOCKSIZE: return MAX_BLOCKSIZE;
        case ATR_GFX_MAX_UNROLL: return MAX_UNROLL;
        default: return -1;
    }
}

std::string
GetStrAttribute(StrAttribute attribute)
{
    switch(attribute)
    {
        case ATR_SRC_PREP_DESCRIPTION:
            return "Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)";
        default: return "";
    }
}

ErrResult
ParseTransfers(std::string line, std::vector<Transfer>& transfers)
{
    // Replace any round brackets or '->' with spaces,
    for(int i = 1; line[i]; i++)
        if(line[i] == '(' || line[i] == ')' || line[i] == '-' || line[i] == ':' ||
           line[i] == '>')
            line[i] = ' ';

    transfers.clear();

    // Read in number of transfers
    int                numTransfers = 0;
    std::istringstream iss(line);
    iss >> numTransfers;
    if(iss.fail()) return ERR_NONE;

    // If numTransfers < 0, read 5-tuple (srcMem, exeMem, dstMem, #CUs, #Bytes)
    // otherwise read triples (srcMem, exeMem, dstMem)
    bool const advancedMode = (numTransfers < 0);
    numTransfers            = abs(numTransfers);

    int         numSubExecs;
    std::string srcStr, exeStr, dstStr, numBytesToken;

    if(!advancedMode)
    {
        iss >> numSubExecs;
        if(numSubExecs < 0 || iss.fail())
        {
            return { ERR_FATAL,
                     "Parsing error: Number of blocks to use (%d) must be non-negative",
                     numSubExecs };
        }
    }

    for(int i = 0; i < numTransfers; i++)
    {
        Transfer transfer;

        if(!advancedMode)
        {
            iss >> srcStr >> exeStr >> dstStr;
            transfer.numSubExecs = numSubExecs;
            if(iss.fail())
            {
                return { ERR_FATAL,
                         "Parsing error: Unable to read valid Transfer %d (SRC EXE DST) "
                         "triplet",
                         i + 1 };
            }
            transfer.numBytes = 0;
        }
        else
        {
            iss >> srcStr >> exeStr >> dstStr >> transfer.numSubExecs >> numBytesToken;
            if(iss.fail())
            {
                return { ERR_FATAL,
                         "Parsing error: Unable to read valid Transfer %d (SRC EXE DST "
                         "$CU #Bytes) tuple",
                         i + 1 };
            }
            if(sscanf(numBytesToken.c_str(), "%lu", &transfer.numBytes) != 1)
            {
                return { ERR_FATAL,
                         "Parsing error: Unable to read valid Transfer %d (SRC EXE DST "
                         "#CU #Bytes) tuple",
                         i + 1 };
            }

            char units = numBytesToken.back();
            switch(toupper(units))
            {
                case 'G': transfer.numBytes *= 1024;
                case 'M': transfer.numBytes *= 1024;
                case 'K': transfer.numBytes *= 1024;
            }
        }

        ERR_CHECK(ParseMemType(srcStr, transfer.srcs));
        ERR_CHECK(ParseMemType(dstStr, transfer.dsts));
        ERR_CHECK(ParseExeType(exeStr, transfer.exeDevice, transfer.exeSubIndex));
        transfers.push_back(transfer);
    }
    return ERR_NONE;
}

int
GetNumExecutors(ExeType exeType)
{
    switch(exeType)
    {
        case EXE_CPU: return 1;  // Single CPU node (NUMA not supported)
        case EXE_GPU_GFX:
        case EXE_GPU_DMA:
        {
            int        numDetectedGpus = 0;
            hipError_t status          = hipGetDeviceCount(&numDetectedGpus);
            if(status != hipSuccess) numDetectedGpus = 0;
            return numDetectedGpus;
        }
#ifdef NIC_EXEC_ENABLED
        case EXE_NIC:
        case EXE_NIC_NEAREST:
        {
            return GetIbvDeviceList().size();
        }
#endif
        default: return 0;
    }
}

int
GetNumSubExecutors(ExeDevice exeDevice)
{
    int const& exeIndex = exeDevice.exeIndex;

    switch(exeDevice.exeType)
    {
        case EXE_CPU:
        {
            return std::thread::hardware_concurrency();  // Return total CPU cores
        }
        case EXE_GPU_GFX:
        {
            int numGpus = GetNumExecutors(EXE_GPU_GFX);
            if(exeIndex < 0 || numGpus <= exeIndex) return 0;
            int        numDeviceCUs = 0;
            hipError_t status       = hipDeviceGetAttribute(
                &numDeviceCUs, hipDeviceAttributeMultiprocessorCount, exeIndex);
            if(status != hipSuccess) numDeviceCUs = 0;
            return numDeviceCUs;
        }
        case EXE_GPU_DMA:
        {
            return 1;
        }
        default: return 0;
    }
}

int
GetNumExecutorSubIndices(ExeDevice exeDevice)
{
    // Executor subindices are not supported on NVIDIA hardware
#if defined(__NVCC__)
    return 0;
#else
    switch(exeDevice.exeType)
    {
        case EXE_CPU: return 0;
        case EXE_GPU_GFX:
        {
            hsa_agent_t agent;
            ErrResult   err = GetHsaAgent(exeDevice, agent);
            if(err.errType != ERR_NONE) return 0;
            int numXccs = 1;
            if(hsa_agent_get_info(agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NUM_XCC,
                                  &numXccs) != HSA_STATUS_SUCCESS)
                return 1;
            return numXccs;
        }
        case EXE_GPU_DMA:
        {
            std::set<int> engineIds;
            ErrResult     err;

            // Get HSA agent for this GPU
            hsa_agent_t agent;
            err = GetHsaAgent(exeDevice, agent);
            if(err.errType != ERR_NONE) return 0;

            int numTotalEngines = 0, numEnginesA = 0, numEnginesB = 0;
            if(hsa_agent_get_info(agent,
                                  (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NUM_SDMA_ENG,
                                  &numEnginesA) == HSA_STATUS_SUCCESS)
                numTotalEngines += numEnginesA;
            if(hsa_agent_get_info(agent,
                                  (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG,
                                  &numEnginesB) == HSA_STATUS_SUCCESS)
                numTotalEngines += numEnginesB;

            return numTotalEngines;
        }
        default: return 0;
    }
#endif
}

int
GetClosestCpuNumaToGpu(int gpuIndex)
{
    // Closest NUMA is not supported on NVIDIA hardware at this time
#if defined(__NVCC__)
    return -1;
#else
    hsa_agent_t gpuAgent;
    ErrResult   err = GetHsaAgent({ EXE_GPU_GFX, gpuIndex }, gpuAgent);
    if(err.errType != ERR_NONE) return -1;

    hsa_agent_t closestCpuAgent;
    if(hsa_agent_get_info(gpuAgent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NEAREST_CPU,
                          &closestCpuAgent) == HSA_STATUS_SUCCESS)
    {
        int numCpus = GetNumExecutors(EXE_CPU);
        for(int i = 0; i < numCpus; i++)
        {
            hsa_agent_t cpuAgent;
            err = GetHsaAgent({ EXE_CPU, i }, cpuAgent);
            if(err.errType != ERR_NONE) return -1;
            if(cpuAgent.handle == closestCpuAgent.handle) return i;
        }
    }
    return -1;
#endif
}

int
GetClosestCpuNumaToNic(int nicIndex)
{
#ifdef NIC_EXEC_ENABLED
    int numNics = GetNumExecutors(EXE_NIC);
    if(nicIndex < 0 || nicIndex >= numNics) return -1;
    return GetIbvDeviceList()[nicIndex].numaNode;
#else
    (void) nicIndex;  // Suppress unused parameter warning
    return -1;
#endif
}

int
GetClosestNicToGpu(int gpuIndex)
{
#ifdef NIC_EXEC_ENABLED
    static bool             isInitialized = false;
    static std::vector<int> closestNicId;

    int numGpus = GetNumExecutors(EXE_GPU_GFX);
    if(gpuIndex < 0 || gpuIndex >= numGpus) return -1;

    // Build closest NICs per GPU on first use
    if(!isInitialized)
    {
        closestNicId.resize(numGpus, -1);

        // Build up list of NIC bus addresses
        std::vector<std::string> ibvAddressList;
        auto const&              ibvDeviceList = GetIbvDeviceList();
        for(auto const& ibvDevice : ibvDeviceList)
            ibvAddressList.push_back(ibvDevice.hasActivePort ? ibvDevice.busId : "");

        // Track how many times a device has been assigned as "closest"
        // This allows distributed work across devices using multiple ports (sharing the
        // same busID) NOTE: This isn't necessarily optimal, but likely to work in most
        // cases involving multi-port Counter example:
        //
        //  G0 prefers (N0,N1), picks N0
        //  G1 prefers (N1,N2), picks N1
        //  G2 prefers N0,      picks N0
        //
        //  instead of G0->N1, G1->N2, G2->N0

        std::vector<int> assignedCount(ibvDeviceList.size(), 0);

        // Loop over each GPU to find the closest NIC(s) based on PCIe address
        for(int i = 0; i < numGpus; i++)
        {
            // Collect PCIe address for the GPU
            char       hipPciBusId[64];
            hipError_t err = hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), i);
            if(err != hipSuccess)
            {
#    ifdef VERBS_DEBUG
                printf("Failed to get PCI Bus ID for HIP device %d: %s\n", i,
                       hipGetErrorString(err));
#    endif
                closestNicId[i] = -1;
                continue;
            }

            // Find closest NICs
            std::set<int> closestNicIdxs =
                GetNearestDevicesInTree(hipPciBusId, ibvAddressList);

            // Pick the least-used NIC to assign as closest
            int closestIdx = -1;
            for(auto idx : closestNicIdxs)
            {
                if(closestIdx == -1 || assignedCount[idx] < assignedCount[closestIdx])
                    closestIdx = idx;
            }

            // The following will only use distance between bus IDs
            // to determine the closest NIC to GPU if the PCIe tree approach fails
            if(closestIdx < 0)
            {
#    ifdef VERBS_DEBUG
                printf("[WARN] Falling back to PCIe bus ID distance to determine "
                       "proximity\n");
#    endif

                int minDistance = std::numeric_limits<int>::max();
                for(int j = 0; j < ibvDeviceList.size(); j++)
                {
                    if(ibvDeviceList[j].busId != "")
                    {
                        int distance =
                            GetBusIdDistance(hipPciBusId, ibvDeviceList[j].busId);
                        if(distance < minDistance && distance >= 0)
                        {
                            minDistance = distance;
                            closestIdx  = j;
                        }
                    }
                }
            }
            closestNicId[i] = closestIdx;
            if(closestIdx != -1) assignedCount[closestIdx]++;
        }
        isInitialized = true;
    }
    return closestNicId[gpuIndex];
#else
    (void) gpuIndex;  // Suppress unused parameter warning
    return -1;
#endif
}

// Undefine CUDA compatibility macros
#if defined(__NVCC__)

// ROCm specific
#    undef wall_clock64
#    undef gcnArchName

// Datatypes
#    undef hipDeviceProp_t
#    undef hipError_t
#    undef hipEvent_t
#    undef hipStream_t

// Enumerations
#    undef hipDeviceAttributeClockRate
#    undef hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#    undef hipDeviceAttributeMultiprocessorCount
#    undef hipErrorPeerAccessAlreadyEnabled
#    undef hipFuncCachePreferShared
#    undef hipMemcpyDefault
#    undef hipMemcpyDeviceToHost
#    undef hipMemcpyHostToDevice
#    undef hipSuccess

// Functions
#    undef hipDeviceCanAccessPeer
#    undef hipDeviceEnablePeerAccess
#    undef hipDeviceGetAttribute
#    undef hipDeviceGetPCIBusId
#    undef hipDeviceSetCacheConfig
#    undef hipDeviceSynchronize
#    undef hipEventCreate
#    undef hipEventDestroy
#    undef hipEventElapsedTime
#    undef hipEventRecord
#    undef hipFree
#    undef hipGetDeviceCount
#    undef hipGetDeviceProperties
#    undef hipGetErrorString
#    undef hipHostFree
#    undef hipHostMalloc
#    undef hipMalloc
#    undef hipMallocManaged
#    undef hipMemcpy
#    undef hipMemcpyAsync
#    undef hipMemset
#    undef hipMemsetAsync
#    undef hipSetDevice
#    undef hipStreamCreate
#    undef hipStreamDestroy
#    undef hipStreamSynchronize
#endif

// Kernel macros
#undef GetHwId
#undef GetXccId

// Undefine helper macros
#undef ERR_CHECK
#undef ERR_APPEND
}  // namespace TransferBench
