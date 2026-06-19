/*
Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "topo_expl_api.h"
#include "model_descs.h"
#include "utils.h"
#include "debug.h"
#include "topo_expl_tests.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <limits.h>
#include <climits>
#include <cassert>
#include <string>
#include <cctype>

extern const char* ncclFuncStr[];
extern const char* ncclAlgoStr[];
extern const char* ncclProtoStr[];

int main(int argc, char* argv[])
{
  if (cmdOptionExists(argv, argv + argc, {"-h", "--help"})) {
    print_help();
    return TOPO_EXPL_SUCCESS;
  }

  if (cmdOptionExists(argv, argv + argc, {"-t", "--test"})) {
    return run_test_suite_from_args(argc, argv);
  }

  if (cmdOptionExists(argv, argv + argc, {"--include-models", "--include-nodes", "--exclude-models", "--exclude-nodes"})) {
    fprintf(stderr, "WARN: --include-models, --include-nodes, --exclude-models, and --exclude-nodes "
            "require -t (test mode).\n");
    print_help();
    return TOPO_EXPL_INVALID_ARG;
  }

  std::string gpuArch = "";
  char* archStr = getCmdOption(argv, argv + argc, {"-a", "--arch"});
  if (archStr) {
    if (cmdOptionExists(argv, argv + argc, {"-m", "--model", "-n", "--nodes"})) {
      fprintf(stderr, "WARN: -a/--arch cannot be used with -m/--model or -n/--nodes.\n");
      print_help();
      return TOPO_EXPL_INVALID_ARG;
    }
    gpuArch = std::string(archStr);
    if (gpuArch.find("gfx") == std::string::npos && 
      gpuArch.find("GFX") == std::string::npos) {
      gpuArch = "gfx" + gpuArch;
    }
    std::transform(gpuArch.begin(), gpuArch.end(), gpuArch.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  }

  if (!cmdOptionExists(argv, argv + argc, {"-m", "--model"})) {
    printf("Quick start: NCCL_DEBUG=VERSION ./topo_expl [-m/--model model_id] [-n/--nodes numNodes=1]\n");
    printf("For detailed help, run: ./topo_expl -h or ./topo_expl --help\n");
    printf("\n");
    print_available_models(gpuArch);
    return TOPO_EXPL_SUCCESS;
  }

  int model_id = 0;
  char* mi = getCmdOption(argv, argv + argc, {"-m", "--model"});
  if (mi)
    model_id = atol(mi);

  if (model_id >= num_models) {
    printf("Invalid model_id %d\n", model_id);
    return TOPO_EXPL_INVALID_ARG;
  }

  NodeModelDesc *desc = &model_descs[model_id];
  int numNodes = 1;
  char* numNodesStr = getCmdOption(argv, argv + argc, {"-n", "--nodes"});
  if (numNodesStr)
    numNodes = atol(numNodesStr);

  printf("Generating topology using %d: %s\n", model_id, desc->description);

  // Extract number of GPUs per node from model description
  int gpusPerNode = 0;
  extractArchAndGpus(desc->description, &gpusPerNode);
  int nranks = gpusPerNode * numNodes;
  printf("nnodes = %d, nranks = %d\n", numNodes, nranks);

  // Create topology explorer context
  TopoExplConfig config;
  config.xmlTopoFile = desc->filename;
  config.numNodes = numNodes;

  TopoExplContext* context = nullptr;
  TopoExplResult result = topoExplCreate(&config, &context);
  
  if (result != TOPO_EXPL_SUCCESS || !context) {
    printf("Error: Failed to create topology explorer context\n");
    topoExplDestroy(context);
    return TOPO_EXPL_ERROR;
  }
  
  for (int i = 0; i < nranks; i++) {
    int nodeId, cudaDev;
    uint64_t busId;
    result = topoExplGetRankInfo(context, i, &nodeId, &cudaDev, &busId);
    if (result == TOPO_EXPL_SUCCESS) {
      printf("Rank %d: node %d cudaDev %d GPU busId %lx\n", i, nodeId, cudaDev, busId);
    }
  }

  int numAlgos = topoExplGetNumAlgos();
  int numProtos = topoExplGetNumProtos();
  
  for (uint64_t len = 8; len <= 4294967296L; len *= 2) {
    float minTime = 3600000000.0;
    int bestAlgo = -1;
    int bestProto = -1;
    
    // Try all algorithm/protocol combinations for AllReduce
    for (int a = 0; a < numAlgos; a++) {
      for (int p = 0; p < numProtos; p++) {
        float time;
        TopoExplResult timeResult = topoExplGetAlgoTime(
            context, TOPO_FUNC_ALLREDUCE, 
            static_cast<TopoExplAlgo>(a),
            static_cast<TopoExplProto>(p),
            len, &time);
        if (timeResult == TOPO_EXPL_SUCCESS && time >= 0 && time < minTime) {
          bestAlgo = a;
          bestProto = p;
          minTime = time;
        }
      }
    }
    
    if (bestAlgo == -1 || bestProto == -1) {
      printf("Error : no algorithm/protocol available\n");
      WARN("Error : no algorithm/protocol available");
      topoExplDestroy(context);
      return TOPO_EXPL_INTERNAL_ERROR;
    }
    INFO(NCCL_TUNING, "%10lu %s %s time %f", len, 
         ncclAlgoStr[bestAlgo],
         ncclProtoStr[bestProto],
         minTime);
  }

  // Arrays to store function types
  std::vector<TopoExplFunc> funcTypes = {
    TOPO_FUNC_ALLREDUCE,
    TOPO_FUNC_ALLGATHER,
    TOPO_FUNC_REDUCESCATTER,
    TOPO_FUNC_REDUCE,
    TOPO_FUNC_BROADCAST
  };

  std::cout << "\nRunning fp32 production choices for algorithm/protocol/maxChannels" << std::endl;
  // RCCL tuning results
  printf("| %-15s | %-15s | %-15s | %-10s | %-10s | %-12s |\n", 
         "Max Size(B)", "Count", "Collective", "Algorithm", "Protocol", "Max Channels");
  printf("|-----------------|-----------------|-----------------|------------|------------|--------------|\n");
  
  for (size_t i = 0; i < funcTypes.size(); ++i) {
    for (uint64_t count = 8; count <= 1073741824L; count *= 2) { // Up to 1 gigabyte
      TopoExplAlgoInfo info;
      result = topoExplGetAlgoInfo(context, funcTypes[i], count, 
                                       TOPO_DTYPE_FLOAT32, &info);
      
      if (result != TOPO_EXPL_SUCCESS) {
        printf("Error: Failed to get algorithm info for count %lu\n", count);
        continue;
      }
      
      printf("| %-15lu | %-15lu | %-15s | %-10s | %-10s | %-12d |\n",
        info.maxSizeBytes,
        count,
        ncclFuncStr[funcTypes[i]],
        ncclAlgoStr[static_cast<int>(info.algo)],
        ncclProtoStr[static_cast<int>(info.proto)],
        info.maxChannels);
    }
  }

  // Cleanup
  topoExplDestroy(context);
  printf("Done generating topology using %d: %s\n", model_id, desc->description);

  return TOPO_EXPL_SUCCESS;
}
