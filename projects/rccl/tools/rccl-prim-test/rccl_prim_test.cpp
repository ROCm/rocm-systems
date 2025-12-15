/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * @file rccl_prim_test.cpp
 *
 * test performance if individual rccl primitives
 */
#include <cstdio>  //fprintf
#include <iostream> //cerr
#include <unistd.h> //usleep
#include <cstring>
#include <chrono>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include "copy_kernel.h"

#define MAX_GPU 16
#define MAX_WORKGROUPS 56
#define THREADS 256
#define NGPUS 2

#define COPY_UNROLL       4
#define REDUCE_UNROLL     2
#define DOUBLECOPY_UNROLL 2
#define DOUBLECOPYLOCAL_UNROLL 2
#define REDUCECOPY_UNROLL 2

#define PRINT_GPU0_ONLY 1

#define RST  "\x1B[0m"
#define KBLU  "\x1B[34m"
#define FBLU(x) KBLU x RST
#define BOLD(x) "\x1B[1m" x RST

struct transfer_data_t {
  float *dest0[MAX_WORKGROUPS]; //remote fine grain
  float *src0[MAX_WORKGROUPS];  //local fine grain
  float *dest1[MAX_WORKGROUPS]; //local coarse grain
  float *dest2[MAX_WORKGROUPS]; //local fine grain
  float *src1[MAX_WORKGROUPS];  //local coarse grain
  int N;
  int gpu;
  int ngpu;
  uint64_t *remOpCount;
};

struct profiling_data_t {
  uint64_t write_cycles[MAX_WORKGROUPS];
  uint64_t bytes_transferred[MAX_WORKGROUPS];
};


#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

void print_table_header(void) {
  fprintf(stderr, "%120s","=================================================================================================================================\n");
  fprintf(stderr, "%-20s %-13s %-13s %-13s %-13s %-20s %-20s %-10s\n","[Originating GPU]", "[Directions]", "[WorkGroup]", "[linktype]", "[time(ms)]" , "[bytes_transferred]",  "[throughput(GB/s)]", "[StdDev]");
  fprintf(stderr, "%120s","=================================================================================================================================\n");
}

void print_table_summary_line(void) {
  fprintf(stderr, "%120s","---------------------------------------------------------------------------------------------------------------------------------\n");
}

enum Ops {
  OP_COPY,
  OP_LOCALCOPY,
  OP_DOUBLECOPY,
  OP_DOUBLECOPYLOCAL,
  OP_REDUCE,
  OP_REDUCECOPY,
  OP_READ,
  NUM_OPS,
};

template<int op, int sync>
__global__ void flag_sync_kernel(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data, uint64_t opCount) {
  size_t tid = threadIdx.x;
  uint64_t curr_time;
  int bid = blockIdx.x;
  int n = transfer_data->N;

  const float *srcs[NGPUS];
  float *dsts[NGPUS];

  // signal self ready and wait until all GPUs are ready
  if (tid == 0) {
    __atomic_fetch_add(&transfer_data->remOpCount[transfer_data->gpu], 1, __ATOMIC_SEQ_CST);
    if (sync) {
      for (int i = 0; i < transfer_data->ngpu; i++) {
        while (LOAD(&transfer_data->remOpCount[i]) < opCount) {};
      }
    }
  }
  __syncthreads();

  if (tid == 0)
    curr_time = wall_clock64();

  if (op == OP_COPY) {
    srcs[0] = transfer_data->src0[bid];
    dsts[0] = transfer_data->dest0[bid];
    ReduceOrCopyMulti<COPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 1>(threadIdx.x, THREADS,
      1, srcs, 1, dsts, n);
  }
  if (op == OP_LOCALCOPY) {
    srcs[0] = transfer_data->src0[bid];
    dsts[0] = transfer_data->dest1[bid];
    ReduceOrCopyMulti<COPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 1>(threadIdx.x, THREADS,
      1, srcs, 1, dsts, n);
  }
  if (op == OP_DOUBLECOPY) {
    srcs[0] = transfer_data->src0[bid];
    dsts[0] = transfer_data->dest0[bid];
    dsts[1] = transfer_data->dest1[bid];
    ReduceOrCopyMulti<DOUBLECOPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 2>(threadIdx.x, THREADS,
      1, srcs, 2, dsts, n);
  }
  if (op == OP_DOUBLECOPYLOCAL) {
    srcs[0] = transfer_data->src0[bid];
    dsts[0] = transfer_data->dest1[bid];
    dsts[1] = transfer_data->dest2[bid];
    ReduceOrCopyMulti<DOUBLECOPYLOCAL_UNROLL, FuncPassA<float>, float, 1, 1, 1, 2>(threadIdx.x, THREADS,
      1, srcs, 2, dsts, n);
  }
  if (op == OP_REDUCE) {
    srcs[0] = transfer_data->src0[bid];
    srcs[1] = transfer_data->src1[bid];
    dsts[0] = transfer_data->dest0[bid];
    ReduceOrCopyMulti<REDUCE_UNROLL, FuncSum<float>, float, 1, 2, 1, 1>(threadIdx.x, THREADS,
      2, srcs, 1, dsts, n);
  }
  if (op == OP_REDUCECOPY) {
    srcs[0] = transfer_data->src0[bid];
    srcs[1] = transfer_data->src1[bid];
    dsts[0] = transfer_data->dest0[bid];
    dsts[1] = transfer_data->dest1[bid];
    ReduceOrCopyMulti<REDUCECOPY_UNROLL, FuncSum<float>, float, 1, 2, 1, 2>(threadIdx.x, THREADS,
      2, srcs, 2, dsts, n);
  }
  if (op == OP_READ) {
    // Swapped the dest0 and src0 in passed parameter of copy kernel so that it can utilized for as a read kernel.
    // fetch op will happen on transfer_data->dest0[bid] and store op will happen on transfer_data->src0[bid]
    srcs[0] = transfer_data->dest0[bid];
    dsts[0] = transfer_data->src0[bid];
    ReduceOrCopyMulti<COPY_UNROLL, FuncPassA<float>, float, 1, 1, 1, 1>(threadIdx.x, THREADS,
      1, srcs, 1, dsts, n);
  }

  __syncthreads();

  if (tid == 0) {
    __atomic_fetch_add(&(profiling_data->write_cycles[bid]), __builtin_amdgcn_s_memrealtime() - curr_time, __ATOMIC_SEQ_CST);
    __atomic_fetch_add(&(profiling_data->bytes_transferred[bid]), n * sizeof(float), __ATOMIC_SEQ_CST);
  }
}

typedef void(*flag_sync_kernel_t)(struct transfer_data_t* transfer_data, struct profiling_data_t* profiling_data, uint64_t opCount);

static flag_sync_kernel_t const flagSyncKerns[NUM_OPS*2] = {
  flag_sync_kernel<OP_COPY, 0>,
  flag_sync_kernel<OP_COPY, 1>,
  flag_sync_kernel<OP_LOCALCOPY, 0>,
  flag_sync_kernel<OP_LOCALCOPY, 1>,
  flag_sync_kernel<OP_DOUBLECOPY, 0>,
  flag_sync_kernel<OP_DOUBLECOPY, 1>,
  flag_sync_kernel<OP_DOUBLECOPYLOCAL, 0>,
  flag_sync_kernel<OP_DOUBLECOPYLOCAL, 1>,
  flag_sync_kernel<OP_REDUCE, 0>,
  flag_sync_kernel<OP_REDUCE, 1>,
  flag_sync_kernel<OP_REDUCECOPY, 0>,
  flag_sync_kernel<OP_REDUCECOPY, 1>,
  flag_sync_kernel<OP_READ, 0>,
  flag_sync_kernel<OP_READ, 1>,
};

__global__ void initTestDataKernel(float* data, const size_t N, const int gpu) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N) {
    data[tid] = 1.0/(float)(gpu*17 + tid%77);
    tid += blockDim.x * gridDim.x;
  }
}

#define HIPCHECK(cmd)                                                          \
do {                                                                           \
  hipError_t error = (cmd);                                                    \
  if (error != hipSuccess)                                                     \
  {                                                                            \
    std::cerr << "Encountered HIP error (" << error << ") at line "            \
              << __LINE__ << " in file " << __FILE__ << "\n";                  \
    exit(-1);                                                                  \
  }                                                                            \
} while (0)

static void setupPeers(uint32_t *info, bool* is_xgmi) {
  int deviceCnt, dev;

  // is_xgmi indicates all link are one hop XGMI
  *is_xgmi = 1;
  HIPCHECK(hipGetDeviceCount(&deviceCnt));
  HIPCHECK(hipGetDevice(&dev));
  //! If gpus are not peer enabled, enable them
  for (int i = 0; i < deviceCnt; i++) {
    HIPCHECK(hipSetDevice(i));
    for (int j = 0; j < deviceCnt; j++) {
      if (i != j) {
	int p2p;
        HIPCHECK(hipDeviceCanAccessPeer(&p2p, i, j));
        if (!p2p) {
          printf("Cannot enable peer access between device %d and %d. You may use HIP_VISIBLE_DEVICES to limit GPUs.\n",
           i, j);
          exit(-1);
        }
        HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
        uint32_t linktype;
        hipError_t error = hipExtGetLinkTypeAndHopCount(i, j, &linktype, &info[i*deviceCnt+j]);
        if (error != hipSuccess)
          *is_xgmi = 0;
        if (linktype != 4 || info[i*deviceCnt+j] != 1) *is_xgmi = 0;
      }
      else
        info[i*deviceCnt+j] = 0;
    }
  }
  HIPCHECK(hipSetDevice(dev));
}

static void parseChordalRing(char **str) {
  static const char *ringBase = "0 6 7 4 5 3 2 1|0 5 6 3 7 1 4 2|0 4 6 2 7 5 1 3|0 1 2 3 5 4 7 6|0 2 4 1 7 3 6 5|0 3 1 5 7 2 6 4";
  static char ringRemap[256];
  int id[8], dist[8];
  int i;

  int ngpus;
  HIPCHECK(hipGetDeviceCount(&ngpus));
  // single node CR8G only
  if (ngpus != 8)
    return;
  // validate chordal ring and calculate distance
  for (i=0; i<ngpus; i++) {
    int sum = ngpus*(ngpus-1)/2 - i;
    int count = 0;
    for (int n = 0; n<ngpus; n++) {
      uint32_t linktype, hop;
      hipError_t error = hipExtGetLinkTypeAndHopCount(i, n, &linktype, &hop);
      if (error != hipSuccess)
        return;
      if (linktype != 4 || hop != 1) continue;
      sum -= n;
      count ++;
    }
    if(count != ngpus-2 || sum < 0 || sum > ngpus-1) {
      return;
    }
    dist[i] = sum;
  }
  // remap GPU ids
  for (i = 0; i<ngpus; i++) id[i] = i;
  for (i = 0; i<ngpus; i++) {
    if (dist[i] == ngpus-1-i) continue;
    int j, m, n, temp;
    for (j=i+1; j < ngpus; j++)
      if(dist[j] == ngpus-1-i) break;
    m = dist[i]; n = dist[j]; dist[i] = n; dist[j] = m;
    temp = id[m]; id[m] = id[n]; id[n] = temp; temp =dist[m];
    dist[m] = dist[n]; dist[n] = temp;
  }
  // create chordal ring based on reference and remapped ids
  for (i = 0; i <strlen(ringBase); i++) {
    if (ringBase[i] >= '0' && ringBase[i] <= '9')
      ringRemap[i] = id[ringBase[i]-'0']+'0';
    else
      ringRemap[i] = ringBase[i];
  }
  ringRemap[i] = 0;
  *str = ringRemap;
  return;
}

static void printRing(int id, int *ring, int deviceCnt) {
  printf("Ring %d: ", id);
  for (int i = 0; i < deviceCnt; i++)
    printf("%1d ", ring[i]);
  printf("\n");
}

static void findConnect(uint32_t *info, int *ring, int deviceCnt) {
  int n = 0, curr = 0, best;
  uint32_t temp[MAX_GPU*MAX_GPU];
  for (int i = 0; i < deviceCnt*deviceCnt; i++) temp[i] = 0;
  for (int i = 0; i < deviceCnt; i++) {
    for (int j = 0; j < deviceCnt; j++) temp[j*deviceCnt+curr] = 1;
    ring[n] = curr;
    n++;
    int hops = 99;
    for (int j = 0; j < deviceCnt; j++) {
      if (temp[curr*deviceCnt+j]) continue;
      if (info[curr*deviceCnt+j] < hops) {
        best = j;
        hops = info[curr*deviceCnt+j];
      }
    }
    curr = best;
  }
}

static int findNextGpu(int *ring, int gpu, int deviceCnt) {
  int i;
  for (i = 0; i < deviceCnt; i ++)
    if (ring[i] == gpu) break;
  return ring[(i+1)%deviceCnt];
}

static void setupRings(uint32_t *info, int *ring_0, int *ring_1) {
  int deviceCnt, dev;
  HIPCHECK(hipGetDeviceCount(&deviceCnt));
  printf("Connection matrix:\n");
  for (int i = 0; i < deviceCnt; i++) {
    for (int j = 0; j < deviceCnt; j++)
      printf("%2d ", info[i*deviceCnt+j]);
    printf("\n");
  }
  findConnect(info, ring_0, deviceCnt);
  ring_1[0] =0;
  for (int i = 1; i < deviceCnt; i++)
    ring_1[i] = ring_0[deviceCnt-i];
}

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}


static const char* link_type_name[] = {"HT", "QPI", "PCIE", "IB", "XGMI"};


int main(int argc,char* argv[])
{
  if (cmdOptionExists(argv, argv + argc, "-h")) {
    printf("./rccl_prim_test -w num_workgroups -p copy|localcopy|doublecopy|doublecopylocal|reduce|reducecopy|all -i iterations -n bytes -r \"0 1 2 3|3 2 1 0\"\n");
    exit(0);
  }

  int workgroups = 0;
  char *wg = getCmdOption(argv, argv + argc, "-w");
  if (wg)
    workgroups = atol(wg);
  printf("Benchmarking using %d workgroups\n", workgroups);

  int iters = 1000;
  char *it = getCmdOption(argv, argv + argc, "-i");
  if (it)
    iters = atol(it);
  printf("Benchmarking using %d iterations\n", iters);

  uint64_t nBytes = 2097152;
  char *nb = getCmdOption(argv, argv + argc, "-n");
  if (nb)
    nBytes = atol(nb);
  printf("Benchmarking using %ld bytes\n", nBytes);
  uint64_t N = nBytes/sizeof(float);

  int sync = 0;
  char *s = getCmdOption(argv, argv + argc, "-s");
  if (s)
    sync = atol(s);
  if (sync) printf("Sync all GPUs before operation\n");

  char *r = getCmdOption(argv, argv + argc, "-r");
  if (r) printf("User specified ring topology: %s\n", r);

  const char *ops[] = {"copy", "localcopy", "doublecopy", "doublecopylocal", "reduce", "reducecopy", "read", "all"};
  char *prim = getCmdOption(argv, argv + argc, "-p");
  int op = NUM_OPS, begin_op, end_op;
  if (prim) {
    for (op = 0; op < sizeof(ops); op++)
      if (!strcmp((const char *)prim, ops[op]))
        break;
  }
  if (op < NUM_OPS ) {
    begin_op = op;
    end_op = op + 1;
  } else {
    begin_op = 0;
    end_op = NUM_OPS;
    printf("Benchmarking all ops\n");
  }

  int nGpu = 1;
  HIPCHECK(hipGetDeviceCount(&nGpu));

  uint32_t connection_info[MAX_GPU*MAX_GPU];
  // Enable peer access
  bool is_xgmi;
  char *cr8g = 0;
  static const char *ring_4p3l = "0 1 2 3|0 1 3 2|0 2 1 3|0 2 3 1|0 3 1 2|0 3 2 1";
  static const char *ring_8p1h = "0 1 3 2 4 5 7 6|6 7 5 4 2 3 1 0|0 1 5 4 6 7 3 2|2 3 7 6 4 5 1 0";
  static const char *ring_16p1h = "0 1 3 2 6 7 15 14 10 11 9 8 12 13 5 4|0 1 2 3 7 6 13 12 8 9 10 11 15 14 5 4|0 2 3 7 6 14 15 11 10 8 9 13 12 4 5 1|4 5 13 12 8 9 11 10 14 15 7 6 2 3 1 0|4 5 14 15 11 10 9 8 12 13 6 7 3 2 1 0|1 5 4 12 13 9 8 10 11 15 14 6 7 3 2 0";
  static const char *ring_gfx942_8p = "0 1 2 3 4 5 6 7|0 1 2 3 4 5 7 6|0 2 4 1 3 6 5 7|0 2 4 6 1 7 3 5|0 3 1 5 2 7 4 6|0 3 5 1 6 2 7 4|0 4 1 7 3 6 2 5|7 6 5 4 3 2 1 0|6 7 5 4 3 2 1 0|7 5 6 3 1 4 2 0|5 3 7 1 6 4 2 0|6 4 7 2 5 1 3 0|4 7 2 6 1 5 3 0|5 2 6 3 7 1 4 0";

  setupPeers(connection_info, &is_xgmi);
  if (!r) {
    parseChordalRing(&cr8g);
    if (nGpu == 4 && is_xgmi) r = (char *)ring_4p3l;
    if (nGpu == 8 && cr8g) r = (char *)cr8g;
    if (nGpu == 8 && !cr8g) {
      hipDeviceProp_t prop;
      HIPCHECK(hipGetDeviceProperties(&prop, 0));
      if (strncmp(prop.gcnArchName, "gfx942", 6) == 0) {
        r = (char *)ring_gfx942_8p;
        if(!workgroups) workgroups = 28;
      } else {
        r = (char *)ring_8p1h;
        if(!workgroups) workgroups = 16;
      }
    }
    if (nGpu == 16) {
      r = (char *)ring_16p1h;
      if(!workgroups) workgroups = 24;
    }
  }

  if(!workgroups) workgroups = 1;
  // clockwise and counter clockwise rings
  int ring[MAX_WORKGROUPS][MAX_GPU];
  for (int i = 0; i < MAX_WORKGROUPS; i++)
    for (int j = 0; j <MAX_GPU; j++)
      ring[i][j] =  -1;

  int num_rings = 0;
  if (r) {
    int j = 0, n = 0;
    int state = 0;
    do {
      int digit = r[n] - '0';
      if (digit >= 0 && digit <= 9) {
        if (state)
          ring[num_rings][j] = ring[num_rings][j]*10 + digit;
        else {
          ring[num_rings][j] = digit;
          state = 1;
        }
      }
      else {
        state = 0;
        j++;
        if (r[n] == ' ') continue;
        if (r[n] == '|') {
          num_rings ++;
          j = 0;
          continue;
        }
      }
    } while (r[n++] != 0x0);
    num_rings ++;
  } else {
    setupRings(connection_info, ring[0], ring[1]);
    num_rings = 2;
  }

  // duplicate rings
  for (int i = num_rings; i < MAX_WORKGROUPS; i++) {
    for (int j = 0; j <MAX_GPU; j++)
      ring[i][j] =  ring[i%num_rings][j];
  }

  // data buffers
  float *buff[MAX_GPU*MAX_WORKGROUPS], *buff_coarse[MAX_GPU*MAX_WORKGROUPS];
  float *buff_fine[MAX_GPU*MAX_WORKGROUPS]; // additional fine grain buffer for local double copy
  struct transfer_data_t h_transfer_data[MAX_GPU], *transfer_data[MAX_GPU];
  struct profiling_data_t *profiling_data[MAX_GPU], *d_profiling_data[MAX_GPU];
  hipStream_t stream[MAX_GPU];

  uint64_t *remOpCount, *d_remOpCount;
  HIPCHECK(hipHostMalloc((void**)&remOpCount, sizeof(uint64_t)*MAX_GPU, hipHostMallocMapped));
  HIPCHECK(hipHostGetDevicePointer((void**)&d_remOpCount, (void*)remOpCount, 0));

  // print rings
  for (int i = 0; i < workgroups; i++) {
    printRing(i, ring[i], nGpu);
  }

  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipSetDevice(i));
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, i));
    printf("#   device %d [0x%02x] %s\n",
                    i, prop.pciBusID, prop.name);
    //create stream
    HIPCHECK(hipStreamCreate(&stream[i]));
    profiling_data[i] = (struct profiling_data_t *)malloc(sizeof(struct profiling_data_t)*iters);
    HIPCHECK(hipMalloc((void**) &d_profiling_data[i], sizeof(struct profiling_data_t)*iters));

    HIPCHECK(hipExtMallocWithFlags((void**) &transfer_data[i], sizeof(struct transfer_data_t), strncmp(prop.gcnArchName, "gfx942", 6) == 0 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
    for (int j = 0; j < workgroups; j++) {
      HIPCHECK(hipExtMallocWithFlags((void**) &buff[i*MAX_WORKGROUPS+j], 2*N*sizeof(float), strncmp(prop.gcnArchName, "gfx942", 6) == 0 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
      // additional fine grained buffer for local doublecopy, only need 1 buffer (not used by remote)
      HIPCHECK(hipExtMallocWithFlags((void**) &buff_fine[i*MAX_WORKGROUPS+j], N*sizeof(float), strncmp(prop.gcnArchName, "gfx942", 6) == 0 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
      HIPCHECK(hipMalloc((void**) &buff_coarse[i*MAX_WORKGROUPS+j], 2*N*sizeof(float)));
      //randomize test data
      hipLaunchKernelGGL(initTestDataKernel,
          /*grid dim x,y,z*/        dim3(32, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream[i],
          /*kernel args*/           buff[i*MAX_WORKGROUPS+j], 2*N, 0);
      hipLaunchKernelGGL(initTestDataKernel,
          /*grid dim x,y,z*/        dim3(32, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream[i],
          /*kernel args*/           buff_fine[i*MAX_WORKGROUPS+j], N, 0);
      hipLaunchKernelGGL(initTestDataKernel,
          /*grid dim x,y,z*/        dim3(32, 1, 1),
          /*block dim x,y,z*/       dim3(THREADS, 1, 1),
          /*dynamic shared mem*/    0,
          /*stream*/                stream[i],
          /*kernel args*/           buff_coarse[i*MAX_WORKGROUPS+j], 2*N, 0);
    }
  }

  for (int i = 0; i < nGpu; i ++) {
    for (int j = 0; j < workgroups; j++) {
      int next_gpu;
      next_gpu = findNextGpu(ring[j], i, nGpu);
      //printf("GPU %d Ring %d -> Next GPU %d\n", i, j, next_gpu);
      h_transfer_data[i].dest0[j] = buff[next_gpu*MAX_WORKGROUPS+j] + N;
      h_transfer_data[i].dest1[j] = buff_coarse[i*MAX_WORKGROUPS+j] + N;
      h_transfer_data[i].dest2[j] = buff_fine[i*MAX_WORKGROUPS+j]; // additional local fine grain
      h_transfer_data[i].src0[j] = buff[i*MAX_WORKGROUPS+j];
      h_transfer_data[i].src1[j] = buff_coarse[i*MAX_WORKGROUPS+j];
    }
    h_transfer_data[i].N = N;
    h_transfer_data[i].gpu = i;
    h_transfer_data[i].ngpu = nGpu;
    h_transfer_data[i].remOpCount = d_remOpCount;
  }

  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMemcpyAsync(transfer_data[i], &h_transfer_data[i],
                            sizeof(struct transfer_data_t), hipMemcpyHostToDevice,
                            stream[i]));
    HIPCHECK(hipStreamSynchronize(stream[i]));
  }

  void *args[MAX_GPU*3];
  hipLaunchParams *launchParamsList= reinterpret_cast<hipLaunchParams *>(
            malloc(sizeof(hipLaunchParams)*MAX_GPU));

  uint64_t opCount = workgroups;
  for (int op = begin_op; op < end_op; op ++) {
    const char *OpsName[] = {"Copy", "Local Copy", "Double Copy", "doublecopylocal", "Reduce", "ReduceCopy", "Read"};
    printf("\n[Testing %s]: \n", OpsName[op]);
    // 20 warm up cycles
    for (int j = 0; j < 20; j ++) {
      for (int i = 0; i < nGpu; i ++) {
#if 0
        args[i*3] = &transfer_data[i];
        args[i*3+1] = &d_profiling_data[i];
        args[i*3+2] = &opCount;
        launchParamsList[i].func =
                reinterpret_cast<void *>(flagSyncKerns[op*2 + sync]);
        launchParamsList[i].gridDim   = dim3(workgroups, 1, 1),
        launchParamsList[i].blockDim  = dim3(THREADS, 1, 1),
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream    = stream[i];
        launchParamsList[i].args      = args + i*3;
      }
      hipExtLaunchMultiKernelMultiDevice(launchParamsList, nGpu,
        hipCooperativeLaunchMultiDeviceNoPreSync|hipCooperativeLaunchMultiDeviceNoPostSync);
#else
        HIPCHECK(hipSetDevice(i));
        //launch the kernel
        hipLaunchKernelGGL(flagSyncKerns[op*2 + sync],
            /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
            /*block dim x,y,z*/       dim3(THREADS, 1, 1),
            /*dynamic shared mem*/    0,
            /*stream*/                stream[i],
            /*kernel args*/           transfer_data[i], d_profiling_data[i]+j, opCount);
      }
#endif
      opCount+=workgroups;
    }

    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipSetDevice(i));
      HIPCHECK(hipMemsetAsync(d_profiling_data[i], 0, sizeof(struct profiling_data_t)*iters, stream[i]));
      HIPCHECK(hipStreamSynchronize(stream[i]));
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < iters; j ++) {
      for (int i = 0; i < nGpu; i ++) {
#if 0
        args[i*3] = &transfer_data[i];
        args[i*3+1] = &d_profiling_data[i];
        args[i*3+2] = &opCount;
        launchParamsList[i].func =
                reinterpret_cast<void *>(flagSyncKerns[op*2 + sync]);
        launchParamsList[i].gridDim   = dim3(workgroups, 1, 1),
        launchParamsList[i].blockDim  = dim3(THREADS, 1, 1),
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream    = stream[i];
        launchParamsList[i].args      = args + i*3;
      }
      hipExtLaunchMultiKernelMultiDevice(launchParamsList, nGpu,
        hipCooperativeLaunchMultiDeviceNoPreSync|hipCooperativeLaunchMultiDeviceNoPostSync);
#else
        HIPCHECK(hipSetDevice(i));
        //launch the kernel
        hipLaunchKernelGGL(flagSyncKerns[op*2 + sync],
            /*grid dim x,y,z*/        dim3(workgroups, 1, 1),
            /*block dim x,y,z*/       dim3(THREADS, 1, 1),
            /*dynamic shared mem*/    0,
            /*stream*/                stream[i],
            /*kernel args*/           transfer_data[i], d_profiling_data[i]+j, opCount);
      }
#endif
      opCount+=workgroups;
    }

    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipSetDevice(i));
      HIPCHECK(hipStreamSynchronize(stream[i]));
    }

    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
    std::cout << BOLD(FBLU("[GPU to GPU Transfer Profiling Data]"))<<std::endl;
    print_table_header();
    for (int i = 0; i < nGpu; i ++) {
      HIPCHECK(hipMemcpyAsync(profiling_data[i], d_profiling_data[i],
                              sizeof(struct profiling_data_t)*iters, hipMemcpyDeviceToHost,
                              stream[i]));
      HIPCHECK(hipStreamSynchronize(stream[i]));

      uint64_t max_write_cycle = 0;
      uint64_t bytes_transferred = 0;

      hipDeviceProp_t prop;
      HIPCHECK(hipGetDeviceProperties(&prop, i));
      double vega_gpu_rtc_freq, bw_std_dev = 0, mean_write_cycle = 0;
      if (strncmp(prop.gcnArchName, "gfx942", 6) == 0)
        vega_gpu_rtc_freq = 1.0E8;
      else
        vega_gpu_rtc_freq = 2.5E7;
      for (int j = 0; j < workgroups; j++) {
        int next_gpu;
        next_gpu = findNextGpu(ring[j], i, nGpu);

        uint32_t linktype;
        uint32_t hopcount;
        HIPCHECK(hipExtGetLinkTypeAndHopCount(i, next_gpu , &linktype, &hopcount));

        //find mean/max/stddev of iterations
        uint64_t iter_max_write_cycle = 0, iter_bytes_transferred = 0;
        double iter_bw_std_dev = 0, iter_total_write_cycle = 0;
        for (int k = 0; k < iters; k++) {
          iter_max_write_cycle = std::max(iter_max_write_cycle, (profiling_data[i]+k)->write_cycles[j]);
          iter_total_write_cycle = iter_total_write_cycle + (profiling_data[i]+k)->write_cycles[j];
          iter_bytes_transferred = iter_bytes_transferred + (profiling_data[i]+k)->bytes_transferred[j];
        }
        bytes_transferred += iter_bytes_transferred;
        double t1 = iter_total_write_cycle/vega_gpu_rtc_freq;
        max_write_cycle = std::max(max_write_cycle, (uint64_t)iter_total_write_cycle);
        mean_write_cycle = mean_write_cycle + iter_total_write_cycle;
        for (int k = 0; k < iters; k++) {
          double t0 = (double)(profiling_data[i]+k)->write_cycles[j]/vega_gpu_rtc_freq;
          iter_bw_std_dev += std::pow((double)(profiling_data[i]+k)->bytes_transferred[j]/(t0*1.0E9) - (double)(profiling_data[i]+k)->bytes_transferred[j]*iters/(iter_total_write_cycle*1.0E9/vega_gpu_rtc_freq), 2);
        }
        iter_bw_std_dev = std::sqrt(iter_bw_std_dev/iters);

        //store bytes_transferred and write_cycle from all itres into in first iter entry
        profiling_data[i]->write_cycles[j] = (uint64_t)iter_total_write_cycle;
        profiling_data[i]->bytes_transferred[j] = iter_bytes_transferred;
        fprintf(stderr, "%-20d %-d->%-10d %-13d %-13s %-13.3f  %-20lu  %-8.2f         %.3f\n",
          i,i, next_gpu,j,link_type_name[linktype], t1*1000, iter_bytes_transferred, (double)(iter_bytes_transferred)/(t1*1.0E9), iter_bw_std_dev);
      }
      //calculate stddev for rings
      mean_write_cycle /= workgroups;
      for (int j = 0; j < workgroups; j++) {
        double t0 = (double)profiling_data[i]->write_cycles[j]/vega_gpu_rtc_freq;
        bw_std_dev += std::pow((double)profiling_data[i]->bytes_transferred[j]/(t0*1.0E9) - (double)profiling_data[i]->bytes_transferred[j]/(mean_write_cycle*1.0E9/vega_gpu_rtc_freq), 2);
      }
      bw_std_dev = std::sqrt(bw_std_dev/workgroups);
      print_table_summary_line();
      double total = 0;
      total = (double)max_write_cycle/vega_gpu_rtc_freq;
      fprintf(stderr, " Workgroups throughput standard deviation %-20.3f %-13.3f  %-20lu  %-.2f\n",
        bw_std_dev, total*1000, bytes_transferred, (double)bytes_transferred/(total*1.0E9));
      print_table_summary_line();
#ifdef PRINT_GPU0_ONLY
      break;
#endif
    }
    std::cout << BOLD(FBLU("[Application Level Transfer Profiling Data]"))<<std::endl;
    uint64_t total_bytes_transferred = profiling_data[0]->bytes_transferred[0] * workgroups ;
    print_table_summary_line();
    fprintf(stderr, " %-61s %-13.3f  %-20lu  %-.2f\n",
      "Total" , deltaSec*1000, total_bytes_transferred, (double)total_bytes_transferred/(deltaSec*1.0E9));
    print_table_summary_line();
  }

  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipStreamDestroy(stream[i]));
    HIPCHECK(hipFree((void*) transfer_data[i]));
    for (int j = 0; j < workgroups; j++) {
      HIPCHECK(hipFree((void*) buff[i*MAX_WORKGROUPS+j]));
      HIPCHECK(hipFree((void*) buff_coarse[i*MAX_WORKGROUPS+j]));
      HIPCHECK(hipFree((void*) buff_fine[i*MAX_WORKGROUPS+j]));
    }
    HIPCHECK(hipFree((void*) d_profiling_data[i]));
    free(profiling_data[i]);
  }

  printf("opCount: ");
  for (int i = 0; i < nGpu; i++)
    printf("%ld ", remOpCount[i]);
  printf("\n");
  HIPCHECK(hipHostFree((void*)remOpCount));
}
