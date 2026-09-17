/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "EnvVars.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CollectiveArgs.hpp"
#include "ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  int const UT_SINGLE_PROCESS = (1<<0);
  int const UT_MULTI_PROCESS  = (1<<1);

  // Upper bound on GPUs the unit tests enumerate/sweep over.
  static constexpr int kMaxDetectedGpus = 16;
  // A device reports one of these CU counts only when running in CPX (compute-partition) mode.
  static constexpr int kCpxDeviceCuCounts[] = {20, 38};

  // Forward declaration; defined below. Used by the GPU-probe child.
  ncclResult_t busIdToInt64(const char* busId, int64_t* id);

  // Entrypoint for the GPU-probe child re-exec'd by DetectGpuInfo(). When
  // RCCL_UT_GPU_PROBE_FD is set, enumerate the GPUs here (in this fresh execv()'d
  // image), write the results back over that pipe fd, and _exit(). Running the HIP
  // calls after execv() -- instead of directly in a bare fork() child -- is what
  // makes detection safe under rocprofv3 --hip-trace: rocprofiler-sdk cannot be
  // used across a fork(), so a forked child that calls HIP deadlocks, whereas a
  // fresh exec'd image re-initializes the tracer cleanly. In every non-probe
  // process this returns immediately.
  void EnvVars::RunGpuProbeChildIfRequested()
  {
    char const* fdStr = getenv("RCCL_UT_GPU_PROBE_FD");
    if (fdStr == nullptr)
    {
      return;
    }
    int const fd = atoi(fdStr);

    int numGpus = 0;
    if (hipGetDeviceCount(&numGpus) != hipSuccess)
    {
      numGpus = 0;
    }
    if (numGpus > kMaxDetectedGpus)
    {
      numGpus = kMaxDetectedGpus;  // matches the cap the caller applies to numDetectedGpus
    }

    // Arch flags: true only if EVERY visible device matches the 5-char prefix.
    int isGfx94 = (numGpus > 0);
    int isGfx95 = (numGpus > 0);
    int isGfx12 = (numGpus > 0);
    int isGfx90 = (numGpus > 0);
    for (int dev = 0; dev < numGpus; ++dev)
    {
      hipDeviceProp_t devProp;
      if (hipGetDeviceProperties(&devProp, dev) != hipSuccess)
      {
        isGfx94 = isGfx95 = isGfx12 = isGfx90 = 0;
        break;
      }
      char const* tok  = strtok(devProp.gcnArchName, ":");
      char const* arch = (tok != nullptr) ? tok : "";
      isGfx94 &= (std::strncmp("gfx94", arch, 5) == 0);
      isGfx95 &= (std::strncmp("gfx95", arch, 5) == 0);
      isGfx12 &= (std::strncmp("gfx12", arch, 5) == 0);
      isGfx90 &= (std::strncmp("gfx90", arch, 5) == 0);
    }

    // CPX mode: inferred from the reduced CU count on device 0 (matches the old
    // getDeviceMode()).
    int isCpx = 0;
    int numDeviceCUs = 0;
    if (hipDeviceGetAttribute(&numDeviceCUs, hipDeviceAttributeMultiprocessorCount, 0) == hipSuccess)
    {
      for (int cuCount : kCpxDeviceCuCounts)
      {
        if (numDeviceCUs == cuCount)
        {
          isCpx = 1;
          break;
        }
      }
    }

    // GPU priority order: group devices by physical PCI bus id, largest group
    // first (matches the old getDevicePriority()). Falls back to identity order on
    // any error.
    std::vector<int> priority;
    std::unordered_map<int64_t, std::vector<int>> uniqueIdToGpuIndexes;
    bool priorityOk = true;
    for (int dev = 0; dev < numGpus; ++dev)
    {
      char busIdStr[] = "00000000:00:00.0";
      if (hipDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), dev) != hipSuccess)
      {
        priorityOk = false;
        break;
      }
      int64_t busId = 0;
      busIdToInt64(busIdStr, &busId);
      uniqueIdToGpuIndexes[busId].push_back(dev);
    }
    if (priorityOk)
    {
      std::vector<std::pair<int64_t, std::vector<int>>> sortedIds(
          uniqueIdToGpuIndexes.begin(), uniqueIdToGpuIndexes.end());
      std::sort(sortedIds.begin(), sortedIds.end(),
                [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
      for (const auto& entry : sortedIds)
      {
        priority.insert(priority.end(), entry.second.begin(), entry.second.end());
      }
    }
    if ((int)priority.size() != numGpus)
    {
      priority.resize(numGpus);
      for (int i = 0; i < numGpus; ++i)
      {
        priority[i] = i;
      }
    }

    // Serialize: numGpus, 4 arch flags, cpx flag, then numGpus priority ints.
    auto writeAll = [fd](void const* buf, size_t len)
    {
      size_t off = 0;
      while (off < len)
      {
        ssize_t n = write(fd, static_cast<char const*>(buf) + off, len - off);
        if (n <= 0)
        {
          break;
        }
        off += (size_t)n;
      }
    };
    writeAll(&numGpus, sizeof(numGpus));
    writeAll(&isGfx94, sizeof(isGfx94));
    writeAll(&isGfx95, sizeof(isGfx95));
    writeAll(&isGfx12, sizeof(isGfx12));
    writeAll(&isGfx90, sizeof(isGfx90));
    writeAll(&isCpx,   sizeof(isCpx));
    if (numGpus > 0)
    {
      writeAll(priority.data(), (size_t)numGpus * sizeof(int));
    }

    close(fd);
    fflush(nullptr);
    _exit(EXIT_SUCCESS);
  }

  // Enumerate GPUs (count / arch / CPX / priority) by fork()+execv()'ing a fresh
  // copy of this test binary as a probe child (RunGpuProbeChildIfRequested), so all
  // HIP calls run in a fresh process image -- safe under rocprofv3 --hip-trace,
  // which deadlocks if HIP is used in a bare fork() child. Results come back over a
  // pipe; on any failure the members keep their caller-initialized defaults.
  void EnvVars::DetectGpuInfo(bool* isCpxOut, std::vector<int>* priorityOut)
  {
    int pipefd[2];
    if (pipe(pipefd) == -1)
    {
      TEST_ERROR("Unable to create pipe for GPU detection");
      return;
    }

    fflush(nullptr);
    pid_t pid = fork();
    if (0 == pid)
    {
      // Child: hand the write-end fd to the re-exec'd probe via the environment,
      // then replace this image so the HIP calls run in a fresh (fork-free)
      // process. Best-effort: drop the profiler tool hooks so the tiny probe is
      // not itself traced.
      close(pipefd[0]);
      char fdStr[16];
      snprintf(fdStr, sizeof(fdStr), "%d", pipefd[1]);
      setenv("RCCL_UT_GPU_PROBE_FD", fdStr, 1);
      unsetenv("ROCP_TOOL_LIBRARIES");
      unsetenv("HSA_TOOLS_LIB");

      char  self[] = "/proc/self/exe";
      char* argv[] = { self, nullptr };
      execv(self, argv);
      // execv() only returns on failure.
      _exit(EXIT_FAILURE);
    }
    else if (pid < 0)
    {
      close(pipefd[0]);
      close(pipefd[1]);
      TEST_ERROR("Unable to fork GPU-probe child");
      return;
    }

    // Parent: read the serialized probe results.
    close(pipefd[1]);
    auto readAll = [&](void* buf, size_t len) -> bool
    {
      size_t off = 0;
      while (off < len)
      {
        ssize_t n = read(pipefd[0], static_cast<char*>(buf) + off, len - off);
        if (n <= 0)
        {
          return false;
        }
        off += (size_t)n;
      }
      return true;
    };

    int numGpus = 0, g94 = 0, g95 = 0, g12 = 0, g90 = 0, cpx = 0;
    bool ok = readAll(&numGpus, sizeof(numGpus)) && readAll(&g94, sizeof(g94))
              && readAll(&g95, sizeof(g95)) && readAll(&g12, sizeof(g12))
              && readAll(&g90, sizeof(g90)) && readAll(&cpx, sizeof(cpx));
    std::vector<int> priority;
    if (ok && numGpus > 0)
    {
      priority.resize(numGpus);
      ok = readAll(priority.data(), (size_t)numGpus * sizeof(int));
    }

    int status = 0;
    waitpid(pid, &status, 0);
    close(pipefd[0]);

    if (!ok)
    {
      TEST_ERROR("GPU-probe child did not return valid device info");
      return;
    }

    numDetectedGpus = numGpus;
    isGfx94 = (g94 != 0);
    isGfx95 = (g95 != 0);
    isGfx12 = (g12 != 0);
    isGfx90 = (g90 != 0);
    if (isCpxOut != nullptr)
    {
      *isCpxOut = (cpx != 0);
    }
    if (priorityOut != nullptr)
    {
      *priorityOut = priority;
    }
  }

  // NOTE: getDeviceCount(), getDeviceMode() and getDevicePriority() were removed.
  // Their bare fork()+HIP probes deadlock under rocprofv3 --hip-trace (the forked
  // child inherits rocprofiler-sdk state that cannot survive a fork()). GPU
  // enumeration is now performed by the fork()+execv() probe above
  // (DetectGpuInfo / RunGpuProbeChildIfRequested).

  ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
    char hexStr[17];  // Longest possible int64 hex string + null terminator.
    int hexOffset = 0;
    for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
      char c = busId[i];
      if (c == ':') continue;
      if (c == '.') break; //ignore everything after . as they belong to same physical pci
      if ((c >= '0' && c <= '9') ||
          (c >= 'A' && c <= 'F') ||
          (c >= 'a' && c <= 'f')) {
        hexStr[hexOffset++] = busId[i];
      } else break;
    }
    hexStr[hexOffset] = '\0';
    *id = strtol(hexStr, NULL, 16);
    return ncclSuccess;
  }

  EnvVars::EnvVars()
  {
    // If this process is the GPU-probe child re-exec'd by DetectGpuInfo(), do the
    // enumeration, report it over the pipe, and _exit() -- this never returns here.
    RunGpuProbeChildIfRequested();

    // Skip fork+HIP calls in re-exec'd children: GPU enumeration is irrelevant
    // there and concurrent hipGetDeviceCount forks cause KFD contention.
    const bool isIsolatedChild = (std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr);

    // Collect GPU info (count / arch / CPX / priority). All HIP calls run inside a
    // fork()+execv()'d probe child (DetectGpuInfo), so this is safe under
    // rocprofv3 --hip-trace; a bare fork()+HIP child deadlocks the tracer.
    // NOTE: HIP must not be used in this parent before the tests launch their own
    // child processes, hence the isolated probe.
    numDetectedGpus = 0;
    isGfx94 = isGfx95 = isGfx12 = isGfx90 = false;
    bool             isCpxMode = false;
    std::vector<int> detectedPriority;
    if (!isIsolatedChild)
    {
      DetectGpuInfo(&isCpxMode, &detectedPriority);
    }
    numDetectedGpus = min(numDetectedGpus, kMaxDetectedGpus);
    // A non-isolated test process detecting zero GPUs means the probe failed (e.g. execv
    // or HIP error). Proceeding would silently run zero sweep cases and still exit 0, so
    // fail loudly instead of masking a broken run as green.
    if (!isIsolatedChild && numDetectedGpus == 0)
    {
      TEST_ERROR("GPU detection found 0 devices; aborting to avoid a silently-empty test run");
      exit(EXIT_FAILURE);
    }
    // CPX only applies on gfx94 (matches the original getDeviceMode() gating).
    isCpxMode = isCpxMode && isGfx94;

    debugPause     = GetEnvVar("UT_DEBUG_PAUSE" , 0);
    showNames      = GetEnvVar("UT_SHOW_NAMES"  , 1);
    minGpus        = GetEnvVar("UT_MIN_GPUS"    , 1);
    maxGpus        = GetEnvVar("UT_MAX_GPUS"    , numDetectedGpus);
    processMask    = GetEnvVar("UT_PROCESS_MASK", UT_SINGLE_PROCESS | UT_MULTI_PROCESS);
    verbose        = GetEnvVar("UT_VERBOSE"     , 0);
    printValues    = GetEnvVar("UT_PRINT_VALUES", 0);
    maxRanksPerGpu = GetEnvVar("UT_MAX_RANKS_PER_GPU", 1);
    showTiming     = GetEnvVar("UT_SHOW_TIMING",  1);
    useInteractive = GetEnvVar("UT_INTERACTIVE",  0);
    timeoutUs      = GetEnvVar("UT_TIMEOUT_US" ,  5000000);
    useMultithreading = GetEnvVar("UT_MULTITHREAD", false);
    commPool          = GetEnvVar("UT_COMM_POOL", true);

    // Total number of reduction ops
    int numOps = ncclNumOps;

    gpuPriorityOrder.resize(numDetectedGpus);
    for (int i = 0; i < numDetectedGpus; i++)
    {
      gpuPriorityOrder[i] = i;
    }
    // Apply the CPX priority ordering gathered by DetectGpuInfo() (gfx94 + CPX only).
    if (isCpxMode && (int)detectedPriority.size() == numDetectedGpus)
    {
      gpuPriorityOrder = detectedPriority;
    }

    // Test only pow2 number of GPUs for cpx mode to reduce the runtime for UT
    onlyPow2Gpus   = GetEnvVar("UT_POW2_GPUS"   , isCpxMode); // Default value set based on whether system is in CPX mode. UT_POW2_GPUS set by user overrides it.

    std::vector<std::string> redOpStrings = GetEnvVarsList("UT_REDOPS");
    for (auto s : redOpStrings)
    {
      for (int i = 0; i < numOps; ++i)
      {
        if (!strcmp(s.c_str(), ncclRedOpNames[i]))
        {
          redOps.push_back((ncclRedOp_t)i);
          break;
        }
      }
    }
    // Default back to all ops if no strings are found
    if (redOps.empty())
    {
      for (int i = 0; i < numOps; i++)
        redOps.push_back((ncclRedOp_t)i);
    }

    // Limit number of supported datatypes if only allReduce is built
    std::vector<std::string> dtStrings = GetEnvVarsList("UT_DATATYPES");
    for (auto s : dtStrings)
    {
      for (int i = 0; i < ncclNumTypes; ++i)
      {
        if (!strcmp(s.c_str(), ncclDataTypeNames[i]))
        {
          dataTypes.push_back((ncclDataType_t)i);
        }
      }
    }

    // Default option if no valid datatypes are found in env var
    if (dataTypes.empty())
    {
      dataTypes.push_back(ncclFloat32);
      dataTypes.push_back(ncclInt8);
      dataTypes.push_back(ncclUint8);
      dataTypes.push_back(ncclInt32);
      dataTypes.push_back(ncclUint32);
      dataTypes.push_back(ncclInt64);
      dataTypes.push_back(ncclUint64);
      dataTypes.push_back(ncclFloat16);
      dataTypes.push_back(ncclFloat32);
      dataTypes.push_back(ncclFloat64);
      dataTypes.push_back(ncclBfloat16);
      dataTypes.push_back(ncclFloat8e4m3);
      dataTypes.push_back(ncclFloat8e5m2);
    }

    // Build list of possible # GPU ranks based on env vars
    numGpusList.clear();
    for (int i = minGpus; i <= maxGpus; i++)
      if (!onlyPow2Gpus || ((i & (i-1)) == 0))
        numGpusList.push_back(i);

    // Build isMultiProcessList
    isMultiProcessList.clear();
    if (this->processMask & UT_SINGLE_PROCESS) isMultiProcessList.push_back(0);
    if (this->processMask & UT_MULTI_PROCESS)  isMultiProcessList.push_back(1);
  }

  std::vector<ncclRedOp_t> const& EnvVars::GetAllSupportedRedOps()
  {
    return redOps;
  }

  std::vector<ncclDataType_t> const& EnvVars::GetAllSupportedDataTypes()
  {
    return dataTypes;
  }

  std::vector<int> const& EnvVars::GetNumGpusList()
  {
    return numGpusList;
  }

  std::vector<int> const& EnvVars::GetGpuPriorityOrder()
  {
    return gpuPriorityOrder;
  }

  std::vector<int> const& EnvVars::GetIsMultiProcessList()
  {
    return isMultiProcessList;
  }

  int EnvVars::GetEnvVar(std::string const varname, int defaultValue)
  {
    if (getenv(varname.c_str()))
      return atoi(getenv(varname.c_str()));
    return defaultValue;
  };

  std::vector<std::string> EnvVars::GetEnvVarsList(std::string const varname)
  {
    std::vector<std::string> result;
    if (getenv(varname.c_str()))
    {
      std::string env = getenv(varname.c_str());
      std::replace(env.begin(), env.end(), ';', ',');
      std::istringstream ss(env);
      std::string token;
      while (std::getline(ss, token, ','))
      {
        result.push_back(token);
      }
    }
    return result;
  }

  void EnvVars::ShowConfig()
  {
    std::vector<std::tuple<std::string, int, std::string>> supported =
      {
        std::make_tuple("UT_DEBUG_PAUSE"      , debugPause    , "Pause for debugger attach"),
        std::make_tuple("UT_SHOW_NAMES"       , showNames     , "Show test case names"),
        std::make_tuple("UT_MIN_GPUS"         , minGpus       , "Minimum number of GPUs to use"),
        std::make_tuple("UT_MAX_GPUS"         , maxGpus       , "Maximum number of GPUs to use"),
        std::make_tuple("UT_POW2_GPUS"        , onlyPow2Gpus  , "Only allow power-of-2 # of GPUs"),
        std::make_tuple("UT_PROCESS_MASK"     , processMask   , "Whether to run single/multi process"),
        std::make_tuple("UT_VERBOSE"          , verbose       , "Show verbose unit test output"),
        std::make_tuple("UT_REDOPS"           , -1            , "List of reduction ops to test"),
        std::make_tuple("UT_DATATYPES"        , -1            , "List of datatypes to test"),
        std::make_tuple("UT_MAX_RANKS_PER_GPU", maxRanksPerGpu, "Maximum number of ranks using the same GPU"),
        std::make_tuple("UT_PRINT_VALUES"     , printValues   , "Print array values (-1 for all)"),
        std::make_tuple("UT_SHOW_TIMING"      , showTiming    , "Show timing table"),
        std::make_tuple("UT_INTERACTIVE"      , useInteractive, "Run in interactive mode"),
        std::make_tuple("UT_TIMEOUT_US"       , timeoutUs     , "Timeout limit for collective calls in us"),
        std::make_tuple("UT_MULTITHREAD"      , useMultithreading, "Multi-thread single-process ranks"),
        std::make_tuple("UT_COMM_POOL"        , commPool      , "Reuse child processes across configs"),
        std::make_tuple("UT_DEVICE_DATA"      , -1            , "Build/validate test data on GPU (0=host path; default on)"),
        std::make_tuple("UT_DEVICE_DATA_MIN_ELEMS", -1        , "Min elements for the device-data path (default 1Mi)"),
        std::make_tuple("UT_DEVICE_DATA_FAULT", -1            , "Negative control: corrupt one expected element (default off)"),
      };

    printf("================================================================================\n");
    printf(" Environment variables:\n");
    for (auto p : supported)
    {
      printf(" - %-20s %-42s (%3d) %s\n", std::get<0>(p).c_str(), std::get<2>(p).c_str(), std::get<1>(p),
             getenv(std::get<0>(p).c_str()) ? getenv(std::get<0>(p).c_str()) : "<unset>");
    }
    printf("================================================================================\n");
  }
}
