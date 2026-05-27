/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef _TESTER_HPP_
#define _TESTER_HPP_

#include <algorithm>
#include <rocshmem/rocshmem.hpp>
#include <vector>
#include <climits>

#include "tester_arguments.hpp"
#include "../src/util.hpp"
#include "verify_results_kernels.hpp"

/******************************************************************************
 * TESTER CLASS TYPES
 *****************************************************************************/

// X-macro listing all test types as X(CamelName, NumericValue).
// The enum value for each entry is CamelName##TestType.
// The canonical string name used by -a is the lowercased CamelName.
#define ROCSHMEM_FOREACH_TEST_TYPE(X) \
  X(Get,                        0)  \
  X(GetNBI,                     1)  \
  X(Put,                        2)  \
  X(PutNBI,                     3)  \
  X(AMO_FAdd,                   4)  \
  X(AMO_FInc,                   5)  \
  X(AMO_Fetch,                  6)  \
  X(AMO_FCswap,                 7)  \
  X(AMO_Add,                    8)  \
  X(AMO_Inc,                    9)  \
  X(AMO_Cswap,                 10)  \
  X(Init,                      11)  \
  X(PingPong,                  12)  \
  X(RandomAccess,              13)  \
  X(BarrierAll,                14)  \
  X(SyncAll,                   15)  \
  X(TeamSync,                  16)  \
  X(Collect,                   17)  \
  X(TeamFCollect,              18)  \
  X(TeamAllToAll,              19)  \
  X(TeamAllToAllv,             20)  \
  X(ShmemPtr,                  21)  \
  X(P,                         22)  \
  X(G,                         23)  \
  X(WGGet,                     24)  \
  X(WGGetNBI,                  25)  \
  X(WGPut,                     26)  \
  X(WGPutNBI,                  27)  \
  X(WAVEGet,                   28)  \
  X(WAVEGetNBI,                29)  \
  X(WAVEPut,                   30)  \
  X(WAVEPutNBI,                31)  \
  X(TeamBroadcast,             32)  \
  X(TeamReduction,             33)  \
  X(TeamCtxGet,                34)  \
  X(TeamCtxGetNBI,             35)  \
  X(TeamCtxPut,                36)  \
  X(TeamCtxPutNBI,             37)  \
  X(TeamCtxInfra,              38)  \
  X(PutNBIMR,                  39)  \
  X(AMO_Set,                   40)  \
  X(AMO_Swap,                  41)  \
  X(AMO_FetchAnd,              42)  \
  X(AMO_FetchOr,               43)  \
  X(AMO_FetchXor,              44)  \
  X(AMO_And,                   45)  \
  X(AMO_Or,                    46)  \
  X(AMO_Xor,                   47)  \
  X(PingAll,                   48)  \
  X(PutSignal,                 49)  \
  X(WGPutSignal,               50)  \
  X(WAVEPutSignal,             51)  \
  X(PutSignalNBI,              52)  \
  X(WGPutSignalNBI,            53)  \
  X(WAVEPutSignalNBI,          54)  \
  X(SignalFetch,               55)  \
  X(WGSignalFetch,             56)  \
  X(WAVESignalFetch,           57)  \
  X(TeamWGBarrier,             58)  \
  X(DefaultCTXGet,             59)  \
  X(DefaultCTXGetNBI,          60)  \
  X(DefaultCTXPut,             61)  \
  X(DefaultCTXPutNBI,          62)  \
  X(DefaultCTXP,               63)  \
  X(DefaultCTXG,               64)  \
  X(WAVEBarrierAll,            65)  \
  X(WGBarrierAll,              66)  \
  X(WAVESyncAll,               67)  \
  X(WGSyncAll,                 68)  \
  X(TeamBarrier,               69)  \
  X(TeamWAVEBarrier,           70)  \
  X(TeamWAVESync,              71)  \
  X(TeamWGSync,                72)  \
  X(TeamCtxInfraSingle,        73)  \
  X(TeamCtxInfraBlock,         74)  \
  X(TeamCtxInfraOddEven,       75)  \
  X(TeamAlltoallmemOnStream,   76)  \
  X(BarrierAllOnStream,        77)  \
  X(TeamBroadcastmemOnStream,  78)  \
  X(GetmemOnStream,            79)  \
  X(PutmemOnStream,            80)  \
  X(PutmemSignalOnStream,      81)  \
  X(SignalWaitUntilOnStream,   82)  \
  X(FloodPut,                  83)  \
  X(FloodPutNBI,               84)  \
  X(FloodP,                    85)  \
  X(FloodGet,                  86)  \
  X(FloodGetNBI,               87)  \
  X(FloodG,                    88)  \
  X(HipModuleInit,             89)  \
  X(FloodAdd,                  90)  \
  X(FloodFAdd,                 91)  \
  X(FloodWaitAmo,              92)  \
  X(DeviceBitcode,             93)  \
  X(LibraryInfo,               94)  \
  X(TeamCtxSharedInfra,        95)  \
  X(QuietOnStream,             96)  \
  X(SyncAllOnStream,           97)  \
  X(TeamCtxSubsetParentInfra,  98)  \
  X(FenceOrderPutWaveSignal,   99)  \
  X(FenceOrderPutLargeSmall,   100) \
  X(FenceOrderFanout,          101) \
  X(FenceOrderPutWaveNbiChunks, 102)

#define _ROCSHMEM_ENUM_ENTRY(name, val) name##TestType = val,
enum TestType {
  ROCSHMEM_FOREACH_TEST_TYPE(_ROCSHMEM_ENUM_ENTRY)
};
#undef _ROCSHMEM_ENUM_ENTRY

enum OpType { PutType = 0, GetType = 1 };

typedef int ShmemContextType;

/******************************************************************************
 * TESTER INTERFACE
 *****************************************************************************/
class Tester {
 public:
  explicit Tester(TesterArguments args);
  virtual ~Tester();

  virtual void execute();

  static std::vector<Tester *> create(TesterArguments args);

  void *alloc_test_buffer(size_t size, enum UserBufType user_buf_type = USER_BUF_TYPE_HEAP);
  void free_test_buffer(void *buffer, enum UserBufType user_buf_type = USER_BUF_TYPE_HEAP);

 protected:
  virtual void resetBuffers(uint64_t size) = 0;

  virtual void preLaunchKernel() {}

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) = 0;

  virtual void postLaunchKernel() {}

  virtual void verifyResults(uint64_t size) = 0;

  size_t max_msg_size = 0;
  size_t num_msgs = 0;
  size_t num_timed_msgs = 0;
  int num_loops = 0;
  int size_factor = 1;
  int bw_factor = 1;
  int rtt_factor = 1;
  int num_warps = 0;
  int wf_size = 0;
  int device_id = 0;
  int wall_clk_rate = 0; //in kilohertz

  TesterArguments args;

  TestType _type;
  ShmemContextType _shmem_context = 8;  // SHMEM_CTX_WP_PRIVATE

  hipStream_t stream;
  hipDeviceProp_t deviceProps;

  long long int *timer = nullptr;
  long long int *start_time = nullptr;
  long long int *end_time = nullptr;
  long long int min_start_time = 0;
  long long int max_end_time = 0;
  uint32_t num_timers = 0;

  bool *verification_error;

 protected:
  bool _print_results = true;

 private:
  bool _print_header = true;
  void print(uint64_t size);

  void barrier();

  double gpuCyclesToMicroseconds(long long int cycles);

  double timerAvgInMicroseconds();

  bool peLaunchesKernel();

  hipEvent_t start_event;
  hipEvent_t stop_event;
};

//TODO remove altogether? THere is a small difference in print format
#undef CHECK_HIP
#define CHECK_HIP(instr) do {                                               \
  hipError_t error = (instr);                                               \
  if (error != hipSuccess) {                                                \
    fprintf(stderr, "error: " #instr ": %s (%d) at %s:%d\n",                \
      hipGetErrorString(error), error, __FILE__, __LINE__);                 \
    abort();                                                                \
  }                                                                         \
} while(0)

#endif /* _TESTER_HPP */
