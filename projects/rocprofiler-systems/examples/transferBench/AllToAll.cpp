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

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Include necessary headers
#include "TransferBench.hpp"

using namespace TransferBench;

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                                    \
    do                                                                                   \
    {                                                                                    \
        hipError_t error = (cmd);                                                        \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)           \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";    \
            exit(-1);                                                                    \
        }                                                                                \
    } while(0)

// Default configuration values
// Reduced to 16KB (1 << 14) for minimal data capture during profiling
size_t const DEFAULT_BYTES_PER_TRANSFER = (1 << 14);
char const   ExeTypeName[5][4]          = { "CPU", "GPU", "DMA", "NIC", "NIC" };

// Simplified EnvVars class for standalone use
class EnvVars
{
public:
    // Environment variables (using minimal defaults for profiling)
    int                           numIterations    = 1;
    int                           numSubIterations = 1;
    int                           numWarmups       = 0;
    int                           showIterations   = 0;
    int                           useInteractive   = 0;
    int                           alwaysValidate   = 0;
    int                           blockBytes       = 256;
    int                           byteOffset       = 0;
    std::vector<float>            fillPattern;
    std::vector<int>              fillCompress;
    int                           validateDirect = 0;
    int                           validateSource = 0;
    int                           useHsaDma      = 0;
    int                           gfxBlockOrder  = 0;
    int                           gfxBlockSize   = 256;
    std::vector<uint32_t>         cuMask;
    std::vector<std::vector<int>> prefXccTable;
    int                           gfxTemporal      = 0;
    int                           gfxUnroll        = 4;
    int                           useHipEvents     = 1;
    int                           useSingleStream  = 1;
    int                           gfxSingleTeam    = 1;
    int                           gfxWaveOrder     = 0;
    int                           gfxWordSize      = 4;
    int                           hideEnv          = 0;
    int                           minNumVarSubExec = 1;
    int                           maxNumVarSubExec = 0;
    int                           outputToCsv      = 0;
    int                           samplingFactor   = 1;
    int                           ibGidIndex       = -1;
    int                           roceVersion      = 2;
    int                           ipAddressFamily  = 4;
    uint8_t                       ibPort           = 1;
    int                           nicRelaxedOrder  = 1;
    std::string                   closestNicStr    = "";
    int                           gpuMaxHwQueues   = 4;

    // Constructor that collects values from environment
    EnvVars()
    {
        int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
        (void) numDetectedGpus;  // May be unused

        // Get architecture-specific defaults
        hipDeviceProp_t prop;
        HIP_CALL(hipGetDeviceProperties(&prop, 0));
        std::string fullName = prop.gcnArchName;
        std::string archName = fullName.substr(0, fullName.find(':'));

        int defaultGfxUnroll = 4;
        if(archName == "gfx906")
            defaultGfxUnroll = 8;
        else if(archName == "gfx90a")
            defaultGfxUnroll = 8;
        else if(archName == "gfx942")
            defaultGfxUnroll = 4;
        else if(archName == "gfx950")
            defaultGfxUnroll = 4;

        // Read environment variables
        alwaysValidate   = GetEnvVar("ALWAYS_VALIDATE", 0);
        blockBytes       = GetEnvVar("BLOCK_BYTES", 256);
        byteOffset       = GetEnvVar("BYTE_OFFSET", 0);
        gfxBlockOrder    = GetEnvVar("GFX_BLOCK_ORDER", 0);
        gfxBlockSize     = GetEnvVar("GFX_BLOCK_SIZE", 256);
        gfxSingleTeam    = GetEnvVar("GFX_SINGLE_TEAM", 1);
        gfxTemporal      = GetEnvVar("GFX_TEMPORAL", 0);
        gfxUnroll        = GetEnvVar("GFX_UNROLL", defaultGfxUnroll);
        gfxWaveOrder     = GetEnvVar("GFX_WAVE_ORDER", 0);
        gfxWordSize      = GetEnvVar("GFX_WORD_SIZE", 4);
        hideEnv          = GetEnvVar("HIDE_ENV", 0);
        minNumVarSubExec = GetEnvVar("MIN_VAR_SUBEXEC", 1);
        maxNumVarSubExec = GetEnvVar("MAX_VAR_SUBEXEC", 0);
        numIterations    = GetEnvVar("NUM_ITERATIONS", 1);
        numSubIterations = GetEnvVar("NUM_SUBITERATIONS", 1);
        numWarmups       = GetEnvVar("NUM_WARMUPS", 0);
        outputToCsv      = GetEnvVar("OUTPUT_TO_CSV", 0);
        samplingFactor   = GetEnvVar("SAMPLING_FACTOR", 1);
        showIterations   = GetEnvVar("SHOW_ITERATIONS", 0);
        useHipEvents     = GetEnvVar("USE_HIP_EVENTS", 1);
        useHsaDma        = GetEnvVar("USE_HSA_DMA", 0);
        useInteractive   = GetEnvVar("USE_INTERACTIVE", 0);
        useSingleStream  = GetEnvVar("USE_SINGLE_STREAM", 1);
        validateDirect   = GetEnvVar("VALIDATE_DIRECT", 0);
        validateSource   = GetEnvVar("VALIDATE_SOURCE", 0);
        ibGidIndex       = GetEnvVar("IB_GID_INDEX", -1);
        ibPort           = GetEnvVar("IB_PORT_NUMBER", 1);
        roceVersion      = GetEnvVar("ROCE_VERSION", 2);
        ipAddressFamily  = GetEnvVar("IP_ADDRESS_FAMILY", 4);
        nicRelaxedOrder  = GetEnvVar("NIC_RELAX_ORDER", 1);
        closestNicStr    = GetEnvVar("CLOSEST_NIC", "");
        gpuMaxHwQueues   = GetEnvVar("GPU_MAX_HW_QUEUES", 4);
    }

    // Helper function that gets environment variable or sets to default value
    static int GetEnvVar(std::string const& varname, int defaultValue)
    {
        if(getenv(varname.c_str())) return atoi(getenv(varname.c_str()));
        return defaultValue;
    }

    static std::string GetEnvVar(std::string const& varname,
                                 std::string const& defaultValue)
    {
        if(getenv(varname.c_str())) return getenv(varname.c_str());
        return defaultValue;
    }

    void Print(std::string const& name, int32_t const value, const char* format,
               ...) const
    {
        printf("%-20s%s%12d%s", name.c_str(), outputToCsv ? "," : " = ", value,
               outputToCsv ? "," : " : ");
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    }

    void Print(std::string const& name, std::string const& value, const char* format,
               ...) const
    {
        printf("%-20s%s%12s%s", name.c_str(), outputToCsv ? "," : " = ", value.c_str(),
               outputToCsv ? "," : " : ");
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    }

    // Display env var settings (simplified)
    void DisplayEnvVars() const
    {
        std::string nicSupport = "";
#if NIC_EXEC_ENABLED
        nicSupport = " (with NIC support)";
#endif
        if(!outputToCsv)
        {
            printf("Standalone AllToAll v%s%s\n", TransferBench::VERSION,
                   nicSupport.c_str());
            printf("===============================================================\n");
            if(!hideEnv)
                printf("[Common]                              (Suppress by setting "
                       "HIDE_ENV=1)\n");
        }
        else if(!hideEnv)
            printf("EnvVar,Value,Description,(Standalone AllToAll v%s)\n",
                   TransferBench::VERSION);
        if(hideEnv) return;

        Print("NUM_ITERATIONS", numIterations, "Running %d timed iteration(s)",
              numIterations);
        Print("NUM_WARMUPS", numWarmups, "Running %d warmup iteration(s) per Test",
              numWarmups);
        Print("USE_SINGLE_STREAM", useSingleStream, "Using single stream per GFX %s",
              useSingleStream ? "device" : "Transfer");
        Print("GFX_UNROLL", gfxUnroll, "Using GFX unroll factor of %d", gfxUnroll);
        printf("\n");
    }

    // Display usage instructions
    static void DisplayUsage()
    {
        printf("Environment variables:\n");
        printf("======================\n");
        printf(" NUM_ITERATIONS    - # of timed iterations per test (default=1)\n");
        printf(
            " NUM_WARMUPS       - # of untimed warmup iterations per test (default=0)\n");
        printf(" USE_SINGLE_STREAM - Use a single stream per GPU GFX executor "
               "(default=1)\n");
        printf(" GFX_UNROLL        - Unroll factor for GFX kernel (default=4)\n");
        printf(
            " HIDE_ENV          - Hide environment variable value listing (default=0)\n");
        printf(" OUTPUT_TO_CSV     - Outputs to CSV format if set (default=0)\n");
        printf(" SHOW_ITERATIONS   - Show per-iteration timing info (default=0)\n");
        printf("\n");
        printf("AllToAll specific variables:\n");
        printf(" A2A_DIRECT        - Only using direct links (default=1)\n");
        printf(" A2A_LOCAL         - Include local transfers (default=0)\n");
        printf(" A2A_MODE          - Transfer mode: 0=Copy, 1=Read-Only, 2=Write-Only "
               "(default=0)\n");
        printf(" NUM_GPU_DEVICES   - Number of GPUs to use (default=4 detected)\n");
        printf(
            " NUM_SUB_EXEC      - Number of subexecutors/CUs per Transfer (default=1)\n");
        printf(" USE_DMA_EXEC      - Use DMA executor instead of GFX (default=0)\n");
        printf(" USE_FINE_GRAIN    - Use fine-grained memory (default=1)\n");
        printf(" USE_REMOTE_READ   - Use DST as executor instead of SRC (default=0)\n");
    }

    TransferBench::ConfigOptions ToConfigOptions()
    {
        TransferBench::ConfigOptions cfg;

        cfg.general.numIterations      = numIterations;
        cfg.general.numSubIterations   = numSubIterations;
        cfg.general.numWarmups         = numWarmups;
        cfg.general.recordPerIteration = showIterations;
        cfg.general.useInteractive     = useInteractive;

        cfg.data.alwaysValidate = alwaysValidate;
        cfg.data.blockBytes     = blockBytes;
        cfg.data.byteOffset     = byteOffset;
        cfg.data.fillCompress   = fillCompress;
        cfg.data.fillPattern    = fillPattern;
        cfg.data.validateDirect = validateDirect;
        cfg.data.validateSource = validateSource;

        cfg.dma.useHipEvents = useHipEvents;
        cfg.dma.useHsaCopy   = useHsaDma;

        cfg.gfx.blockOrder     = gfxBlockOrder;
        cfg.gfx.blockSize      = gfxBlockSize;
        cfg.gfx.cuMask         = cuMask;
        cfg.gfx.prefXccTable   = prefXccTable;
        cfg.gfx.unrollFactor   = gfxUnroll;
        cfg.gfx.temporalMode   = gfxTemporal;
        cfg.gfx.useHipEvents   = useHipEvents;
        cfg.gfx.useMultiStream = !useSingleStream;
        cfg.gfx.useSingleTeam  = gfxSingleTeam;
        cfg.gfx.waveOrder      = gfxWaveOrder;
        cfg.gfx.wordSize       = gfxWordSize;

        cfg.nic.ibGidIndex      = ibGidIndex;
        cfg.nic.ibPort          = ibPort;
        cfg.nic.ipAddressFamily = ipAddressFamily;
        cfg.nic.useRelaxedOrder = nicRelaxedOrder;
        cfg.nic.roceVersion     = roceVersion;

        std::vector<int> closestNics;
        if(closestNicStr != "")
        {
            std::stringstream ss(closestNicStr);
            std::string       item;
            while(std::getline(ss, item, ','))
            {
                try
                {
                    int nic = std::stoi(item);
                    closestNics.push_back(nic);
                } catch(const std::invalid_argument& e)
                {
                    printf("[ERROR] Invalid NIC index (%s) by user in %s\n", item.c_str(),
                           closestNicStr.c_str());
                    exit(1);
                }
            }
            cfg.nic.closestNics = closestNics;
        }
        return cfg;
    }
};

// Forward declarations
void
PrintResults(EnvVars const& ev, int const testNum, std::vector<Transfer> const& transfers,
             TransferBench::TestResults const& results);
void
PrintErrors(std::vector<ErrResult> const& errors);
void
CheckForError(ErrResult const& error);
std::string
MemDevicesToStr(std::vector<MemDevice> const& memDevices);

// Helper function that converts MemDevices to a string
std::string
MemDevicesToStr(std::vector<MemDevice> const& memDevices)
{
    if(memDevices.empty()) return "N";
    std::stringstream ss;
    for(auto const& m : memDevices)
        ss << TransferBench::MemTypeStr[m.memType] << m.memIndex;
    return ss.str();
}

// Helper function to print warning / exit on fatal error
void
CheckForError(ErrResult const& error)
{
    switch(error.errType)
    {
        case ERR_NONE: return;
        case ERR_WARN: printf("[WARN] %s\n", error.errMsg.c_str()); return;
        case ERR_FATAL: printf("[ERROR] %s\n", error.errMsg.c_str()); exit(1);
        default: break;
    }
}

// Helper function to print list of errors
void
PrintErrors(std::vector<ErrResult> const& errors)
{
    bool isFatal = false;
    for(auto const& err : errors)
    {
        printf("[%s] %s\n", err.errType == ERR_FATAL ? "ERROR" : "WARN",
               err.errMsg.c_str());
        isFatal |= (err.errType == ERR_FATAL);
    }
    if(isFatal) exit(1);
}

// Print TransferBench test results
void
PrintResults(EnvVars const& ev, int const testNum, std::vector<Transfer> const& transfers,
             TransferBench::TestResults const& results)
{
    char   sep                = ev.outputToCsv ? ',' : '|';
    size_t numTimedIterations = results.numTimedIterations;

    if(!ev.outputToCsv) printf("Test %d:\n", testNum);

    // Loop over each executor
    for(auto exeInfoPair : results.exeResults)
    {
        ExeDevice const& exeDevice = exeInfoPair.first;
        ExeResult const& exeResult = exeInfoPair.second;
        ExeType const    exeType   = exeDevice.exeType;
        int32_t const    exeIndex  = exeDevice.exeIndex;

        printf(" Executor: %3s %02d %c %8.3f GB/s %c %8.3f ms %c %12lu bytes %c %-7.3f "
               "GB/s (sum)\n",
               ExeTypeName[exeType], exeIndex, sep, exeResult.avgBandwidthGbPerSec, sep,
               exeResult.avgDurationMsec, sep, exeResult.numBytes, sep,
               exeResult.sumBandwidthGbPerSec);

        // Loop over each transfer
        for(int idx : exeResult.transferIdx)
        {
            Transfer const&       t = transfers[idx];
            TransferResult const& r = results.tfrResults[idx];

            char exeSubIndexStr[32] = "";
            if(t.exeSubIndex != -1) sprintf(exeSubIndexStr, ".%d", t.exeSubIndex);
            printf("     Transfer %02d  %c %8.3f GB/s %c %8.3f ms %c %12lu bytes %c %s "
                   "-> %c%03d%s:%03d -> %s\n",
                   idx, sep, r.avgBandwidthGbPerSec, sep, r.avgDurationMsec, sep,
                   r.numBytes, sep, MemDevicesToStr(t.srcs).c_str(),
                   TransferBench::ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex,
                   exeSubIndexStr, t.numSubExecs, MemDevicesToStr(t.dsts).c_str());

            // Show per-iteration timing information
            if(ev.showIterations)
            {
                // Check that per-iteration information exists
                if(r.perIterMsec.size() != numTimedIterations)
                {
                    printf("[ERROR] Per iteration timing data unavailable: Expected %lu "
                           "data points, but have %lu\n",
                           numTimedIterations, r.perIterMsec.size());
                    exit(1);
                }

                // Compute standard deviation and track iterations by speed
                std::set<std::pair<double, int>> times;
                double                           stdDevTime = 0;
                double                           stdDevBw   = 0;
                for(size_t i = 0; i < numTimedIterations; i++)
                {
                    times.insert(
                        std::make_pair(r.perIterMsec[i], static_cast<int>(i + 1)));
                    double const varTime = fabs(r.avgDurationMsec - r.perIterMsec[i]);
                    stdDevTime += varTime * varTime;

                    double iterBandwidthGbs =
                        (t.numBytes / 1.0E9) / r.perIterMsec[i] * 1000.0f;
                    double const varBw = fabs(iterBandwidthGbs - r.avgBandwidthGbPerSec);
                    stdDevBw += varBw * varBw;
                }
                stdDevTime = sqrt(stdDevTime / numTimedIterations);
                stdDevBw   = sqrt(stdDevBw / numTimedIterations);

                // Loop over iterations (fastest to slowest)
                for(auto& time : times)
                {
                    double iterDurationMsec = time.first;
                    double iterBandwidthGbs =
                        (t.numBytes / 1.0E9) / iterDurationMsec * 1000.0f;
                    printf("      Iter %03d    %c %8.3f GB/s %c %8.3f ms %c", time.second,
                           sep, iterBandwidthGbs, sep, iterDurationMsec, sep);

                    std::set<int> usedXccs;
                    if(static_cast<size_t>(time.second - 1) < r.perIterCUs.size())
                    {
                        printf(" CUs:");
                        for(auto x : r.perIterCUs[time.second - 1])
                        {
                            printf(" %02d:%02d", x.first, x.second);
                            usedXccs.insert(x.first);
                        }
                    }

                    printf(" XCCs:");
                    for(auto x : usedXccs)
                        printf(" %02d", x);
                    printf("\n");
                }
                printf("      StandardDev %c %8.3f GB/s %c %8.3f ms %c\n", sep, stdDevBw,
                       sep, stdDevTime, sep);
            }
        }
    }
    printf(" Aggregate (CPU)  %c %8.3f GB/s %c %8.3f ms %c %12lu bytes %c Overhead: %.3f "
           "ms\n",
           sep, results.avgTotalBandwidthGbPerSec, sep, results.avgTotalDurationMsec, sep,
           results.totalBytesTransferred, sep, results.overheadMsec);
}

// AllToAll Preset Implementation
void
AllToAllPreset(EnvVars& ev, size_t const numBytesPerTransfer,
               std::string const presetName)
{
    (void) presetName;  // May be unused
    enum
    {
        A2A_COPY       = 0,
        A2A_READ_ONLY  = 1,
        A2A_WRITE_ONLY = 2,
        A2A_CUSTOM     = 3,
    };
    char a2aModeStr[4][20] = { "Copy", "Read-Only", "Write-Only", "Custom" };

    // Force single-stream mode for all-to-all benchmark
    ev.useSingleStream = 1;

    // Force to gfx unroll 2 unless explicitly set
    ev.gfxUnroll = EnvVars::GetEnvVar("GFX_UNROLL", 2);

    int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

    // Collect env vars for this preset
    int a2aDirect = EnvVars::GetEnvVar("A2A_DIRECT", 1);
    int a2aLocal  = EnvVars::GetEnvVar("A2A_LOCAL", 0);
    int numGpus   = EnvVars::GetEnvVar("NUM_GPU_DEVICES", std::min(4, numDetectedGpus));
    int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 0);
    int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC", 1);
    int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC", 0);
    int useFineGrain  = EnvVars::GetEnvVar("USE_FINE_GRAIN", 1);
    int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);

    // A2A_MODE may be 0,1,2 or else custom numSrcs:numDsts
    int numSrcs, numDsts;
    int a2aMode = 0;
    if(getenv("A2A_MODE") && sscanf(getenv("A2A_MODE"), "%d:%d", &numSrcs, &numDsts) == 2)
    {
        a2aMode = A2A_CUSTOM;
    }
    else
    {
        a2aMode = EnvVars::GetEnvVar("A2A_MODE", 0);
        if(a2aMode < 0 || a2aMode > 2)
        {
            printf("[ERROR] a2aMode must be between 0 and 2, or else numSrcs:numDsts\n");
            exit(1);
        }
        numSrcs = (a2aMode == A2A_WRITE_ONLY ? 0 : 1);
        numDsts = (a2aMode == A2A_READ_ONLY ? 0 : 1);
    }

    // Print off environment variables
    ev.DisplayEnvVars();
    if(!ev.hideEnv)
    {
        if(!ev.outputToCsv) printf("[AllToAll Related]\n");
        ev.Print("A2A_DIRECT", a2aDirect,
                 a2aDirect ? "Only using direct links" : "Full all-to-all");
        ev.Print("A2A_LOCAL", a2aLocal, "%s local transfers",
                 a2aLocal ? "Include" : "Exclude");
        ev.Print("A2A_MODE",
                 (a2aMode == A2A_CUSTOM)
                     ? std::to_string(numSrcs) + ":" + std::to_string(numDsts)
                     : std::to_string(a2aMode),
                 (a2aMode == A2A_CUSTOM) ? (std::to_string(numSrcs) + " read(s) " +
                                            std::to_string(numDsts) + " write(s)")
                                               .c_str()
                                         : a2aModeStr[a2aMode]);
        ev.Print("NUM_GPU_DEVICES", numGpus, "Using %d GPUs", numGpus);
        ev.Print("NUM_QUEUE_PAIRS", numQueuePairs,
                 "Using %d queue pairs for NIC transfers", numQueuePairs);
        ev.Print("NUM_SUB_EXEC", numSubExecs, "Using %d subexecutors/CUs per Transfer",
                 numSubExecs);
        ev.Print("USE_DMA_EXEC", useDmaExec, "Using %s executor",
                 useDmaExec ? "DMA" : "GFX");
        ev.Print("USE_FINE_GRAIN", useFineGrain, "Using %s-grained memory",
                 useFineGrain ? "fine" : "coarse");
        ev.Print("USE_REMOTE_READ", useRemoteRead, "Using %s as executor",
                 useRemoteRead ? "DST" : "SRC");
        printf("\n");
    }

    // Validate env vars
    if(numGpus < 0 || numGpus > numDetectedGpus)
    {
        printf("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus,
               numDetectedGpus);
        exit(1);
    }
    if(useDmaExec && (numSrcs != 1 || numDsts != 1))
    {
        printf("[ERROR] DMA execution can only be used for copies (A2A_MODE=0)\n");
        exit(1);
    }

    // Collect the number of GPU devices to use
    MemType memType = useFineGrain ? MEM_GPU_FINE : MEM_GPU;
    ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

    std::map<std::pair<int, int>, int> reIndex;
    std::vector<Transfer>              transfers;
    for(int i = 0; i < numGpus; i++)
    {
        for(int j = 0; j < numGpus; j++)
        {
            // Check whether or not to execute this pair
            if(i == j)
            {
                if(!a2aLocal) continue;
            }
            else if(a2aDirect)
            {
#if !defined(__NVCC__)
                uint32_t linkType, hopCount;
                HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
                if(hopCount != 1) continue;
#endif
            }

            // Build Transfer and add it to list
            TransferBench::Transfer transfer;
            transfer.numBytes = numBytesPerTransfer;
            for(int x = 0; x < numSrcs; x++)
                transfer.srcs.push_back({ memType, i });

            // When using multiple destinations, the additional destinations are "local"
            if(numDsts) transfer.dsts.push_back({ memType, j });
            for(int x = 1; x < numDsts; x++)
                transfer.dsts.push_back({ memType, i });
            transfer.exeDevice   = { exeType, (useRemoteRead ? j : i) };
            transfer.exeSubIndex = -1;
            transfer.numSubExecs = numSubExecs;

            reIndex[std::make_pair(i, j)] = transfers.size();
            transfers.push_back(transfer);
        }
    }

    // Create a ring using NICs
    std::vector<int> nicTransferIdx(numGpus);
    if(numQueuePairs > 0)
    {
        int numNics = TransferBench::GetNumExecutors(EXE_NIC);
        (void) numNics;  // May be unused
        for(int i = 0; i < numGpus; i++)
        {
            TransferBench::Transfer transfer;
            transfer.numBytes = numBytesPerTransfer;
            transfer.srcs.push_back({ memType, i });
            transfer.dsts.push_back({ memType, (i + 1) % numGpus });
            transfer.exeDevice   = { TransferBench::EXE_NIC_NEAREST, i };
            transfer.exeSubIndex = (i + 1) % numGpus;
            transfer.numSubExecs = numQueuePairs;
            nicTransferIdx[i]    = transfers.size();
            transfers.push_back(transfer);
        }
    }

    printf("GPU-GFX All-To-All benchmark:\n");
    printf("==========================\n");
    printf("- Copying %lu bytes between %s pairs of GPUs using %d CUs (%lu Transfers)\n",
           numBytesPerTransfer, a2aDirect ? "directly connected" : "all", numSubExecs,
           transfers.size());
    if(transfers.size() == 0)
    {
        printf("Error: No valid transfers created. Check GPU count, a2aLocal=%d, "
               "a2aDirect=%d settings, and GPU topology/connectivity.\n",
               a2aLocal, a2aDirect);
        return;
    }

    // Execute Transfers
    TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
    TransferBench::TestResults   results;
    if(!TransferBench::RunTransfers(cfg, transfers, results))
    {
        for(auto const& err : results.errResults)
            printf("%s\n", err.errMsg.c_str());
        exit(0);
    }
    else
    {
        PrintResults(ev, 1, transfers, results);
    }

    // Print results
    char separator = (ev.outputToCsv ? ',' : ' ');
    printf("\nSummary: [%lu bytes per Transfer] [%s:%d] [%d Read(s) %d Write(s)]\n",
           numBytesPerTransfer, useDmaExec ? "DMA" : "GFX", numSubExecs, numSrcs,
           numDsts);
    printf(
        "===========================================================================\n");
    printf("SRC\\DST ");
    for(int dst = 0; dst < numGpus; dst++)
        printf("%cGPU %02d    ", separator, dst);
    if(numQueuePairs > 0) printf("%cNIC(%02d QP)", separator, numQueuePairs);
    printf("   %cSTotal     %cActual\n", separator, separator);

    double              totalBandwidthGpu  = 0.0;
    double              minActualBandwidth = std::numeric_limits<double>::max();
    double              maxActualBandwidth = 0.0;
    std::vector<double> colTotalBandwidth(numGpus + 2, 0.0);
    for(int src = 0; src < numGpus; src++)
    {
        double rowTotalBandwidth = 0;
        int    transferCount     = 0;
        double minBandwidth      = std::numeric_limits<double>::max();
        printf("GPU %02d", src);
        for(int dst = 0; dst < numGpus; dst++)
        {
            if(reIndex.count(std::make_pair(src, dst)))
            {
                int const transferIdx = reIndex[std::make_pair(src, dst)];
                TransferBench::TransferResult const& r = results.tfrResults[transferIdx];
                colTotalBandwidth[dst] += r.avgBandwidthGbPerSec;
                rowTotalBandwidth += r.avgBandwidthGbPerSec;
                totalBandwidthGpu += r.avgBandwidthGbPerSec;
                minBandwidth = std::min(minBandwidth, r.avgBandwidthGbPerSec);
                transferCount++;
                printf("%c%8.3f  ", separator, r.avgBandwidthGbPerSec);
            }
            else
            {
                printf("%c%8s  ", separator, "N/A");
            }
        }

        if(numQueuePairs > 0)
        {
            TransferBench::TransferResult const& r =
                results.tfrResults[nicTransferIdx[src]];
            colTotalBandwidth[numGpus] += r.avgBandwidthGbPerSec;
            rowTotalBandwidth += r.avgBandwidthGbPerSec;
            totalBandwidthGpu += r.avgBandwidthGbPerSec;
            minBandwidth = std::min(minBandwidth, r.avgBandwidthGbPerSec);
            transferCount++;
            printf("%c%8.3f  ", separator, r.avgBandwidthGbPerSec);
        }
        double actualBandwidth = minBandwidth * transferCount;
        printf("   %c%8.3f   %c%8.3f\n", separator, rowTotalBandwidth, separator,
               actualBandwidth);
        minActualBandwidth = std::min(minActualBandwidth, actualBandwidth);
        maxActualBandwidth = std::max(maxActualBandwidth, actualBandwidth);
        colTotalBandwidth[numGpus + 1] += rowTotalBandwidth;
    }
    printf("\nRTotal");
    for(int dst = 0; dst < numGpus; dst++)
    {
        printf("%c%8.3f  ", separator, colTotalBandwidth[dst]);
    }
    if(numQueuePairs > 0)
    {
        printf("%c%8.3f  ", separator, colTotalBandwidth[numGpus]);
    }
    printf("   %c%8.3f   %c%8.3f   %c%8.3f\n", separator, colTotalBandwidth[numGpus + 1],
           separator, minActualBandwidth, separator, maxActualBandwidth);
    printf("\n");

    printf("Average   bandwidth (GPU Timed): %8.3f GB/s\n",
           totalBandwidthGpu / transfers.size());
    printf("Aggregate bandwidth (GPU Timed): %8.3f GB/s\n", totalBandwidthGpu);
    printf("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n",
           results.avgTotalBandwidthGbPerSec);

    PrintErrors(results.errResults);
}

// Display usage instructions
void
DisplayUsage(char const* cmdName)
{
    std::string nicSupport = "";
#if NIC_EXEC_ENABLED
    nicSupport = " (with NIC support)";
#endif
    printf("Standalone AllToAll v%s%s\n", TransferBench::VERSION, nicSupport.c_str());
    printf("========================================\n");

    printf("Usage: %s [N]\n", cmdName);
    printf("  N     : (Optional) Number of bytes to copy per Transfer.\n");
    printf("          If not specified, defaults to %lu bytes. Must be a multiple of 4 "
           "bytes\n",
           DEFAULT_BYTES_PER_TRANSFER);
    printf("          May append a suffix ('K', 'M', 'G') for kilobytes / megabytes / "
           "gigabytes\n");
    printf("\n");

    EnvVars::DisplayUsage();
}

// Main function
int
main(int argc, char** argv)
{
    // Collect environment variables
    EnvVars ev;

    // Display usage instructions if requested
    if(argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))
    {
        DisplayUsage(argv[0]);
        exit(0);
    }

    // Determine number of bytes to run per Transfer
    size_t numBytesPerTransfer = argc > 1 ? atoll(argv[1]) : DEFAULT_BYTES_PER_TRANSFER;
    if(argc > 1)
    {
        // Adjust bytes if unit specified
        char units = argv[1][strlen(argv[1]) - 1];
        switch(units)
        {
            case 'G':
            case 'g': numBytesPerTransfer *= 1024;
            case 'M':
            case 'm': numBytesPerTransfer *= 1024;
            case 'K':
            case 'k': numBytesPerTransfer *= 1024;
        }
    }
    if(numBytesPerTransfer % 4)
    {
        printf("[ERROR] numBytesPerTransfer (%lu) must be a multiple of 4\n",
               numBytesPerTransfer);
        exit(1);
    }

    printf("Running AllToAll benchmark with %lu bytes per transfer\n\n",
           numBytesPerTransfer);

    // Run AllToAll preset
    AllToAllPreset(ev, numBytesPerTransfer, "AllToAll");

    return 0;
}
