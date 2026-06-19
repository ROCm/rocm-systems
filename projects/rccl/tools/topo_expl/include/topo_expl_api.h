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

#ifndef TOPO_EXPL_API_H_
#define TOPO_EXPL_API_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TopoExplContext TopoExplContext;
typedef struct TopoExplComm TopoExplComm;

// Return codes
typedef enum {
  TOPO_EXPL_SUCCESS = 0,
  TOPO_EXPL_ERROR = 1,
  TOPO_EXPL_INVALID_ARG = 2,
  TOPO_EXPL_INTERNAL_ERROR = 3
} TopoExplResult;

// Collective function types
typedef enum {
  TOPO_FUNC_ALLGATHER = 0,
  TOPO_FUNC_ALLREDUCE = 1,
  TOPO_FUNC_ALLTOALL = 2,
  TOPO_FUNC_ALLTOALL_GDA = 3,
  TOPO_FUNC_ALLTOALLV_GDA = 4,
  TOPO_FUNC_BROADCAST = 5,
  TOPO_FUNC_REDUCE = 6,
  TOPO_FUNC_REDUCESCATTER = 7,
  TOPO_FUNC_SENDRECV = 8
} TopoExplFunc;

// Data types
typedef enum {
  TOPO_DTYPE_FLOAT32 = 0,
  TOPO_DTYPE_FLOAT64 = 1,
  TOPO_DTYPE_FLOAT16 = 2,
  TOPO_DTYPE_INT32 = 3,
  TOPO_DTYPE_INT64 = 4,
  TOPO_DTYPE_INT8 = 5,
  TOPO_DTYPE_UINT8 = 6,
  TOPO_DTYPE_BFLOAT16 = 7
} TopoExplDataType;

// Algorithms
typedef enum {
  TOPO_ALGO_TREE = 0,
  TOPO_ALGO_RING = 1,
  TOPO_ALGO_COLLNET_DIRECT = 2,
  TOPO_ALGO_COLLNET_CHAIN = 3,
  TOPO_ALGO_NVLS = 4,
  TOPO_ALGO_NVLS_TREE = 5,
  TOPO_ALGO_PAT = 6
} TopoExplAlgo;

// Protocols
typedef enum {
  TOPO_PROTO_LL = 0,
  TOPO_PROTO_LL128 = 1,
  TOPO_PROTO_SIMPLE = 2
} TopoExplProto;

// Algorithm/Protocol selection
typedef struct {
  TopoExplAlgo algo;
  TopoExplProto proto;
  int maxChannels;
  uint64_t maxSizeBytes;
} TopoExplAlgoInfo;

// Topo explorer configuration
typedef struct {
  const char* xmlTopoFile;  // Path to topology XML file
  int numNodes;              // Number of nodes (default: 1)
} TopoExplConfig;

// Create topo explorer context from XML file
TopoExplResult topoExplCreate(
    const TopoExplConfig* config,
    TopoExplContext** context);

// Destroy topo explorer context
void topoExplDestroy(TopoExplContext* context);

// Get algorithm/protocol selection for a given collective operation
TopoExplResult topoExplGetAlgoInfo(
    TopoExplContext* context,
    TopoExplFunc func,
    uint64_t count,
    TopoExplDataType dataType,
    TopoExplAlgoInfo* info);

// Get the number of algorithms
int topoExplGetNumAlgos(void);

// Get the number of protocols
int topoExplGetNumProtos(void);

// Get rank information (node ID, cudaDev, busId)
TopoExplResult topoExplGetRankInfo(
    TopoExplContext* context,
    int rank,
    int* nodeId,
    int* cudaDev,
    uint64_t* busId);

// Get tuning time for a specific algorithm/protocol combination
TopoExplResult topoExplGetAlgoTime(
    TopoExplContext* context,
    TopoExplFunc func,
    TopoExplAlgo algo,
    TopoExplProto proto,
    uint64_t count,
    float* time);

#ifdef __cplusplus
}
#endif

#endif // TOPO_EXPL_API_H_
