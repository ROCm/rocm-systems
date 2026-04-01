/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #include "cuda_runtime.h"
 #include "common.h"
 #include "rccl_compat.h"
 #include <cstdlib>
 
 // ====================================================================
 // Per-channel initialization support
 //
 // When NCCL_TESTS_PERCHANNEL_INIT=1, each channel's contiguous data
 // segment is filled with a unique value determined by the fill mode.
 // Channel count is read from NCCL_MIN_NCHANNELS (default 16).
 //
 // Three fill modes (set NCCL_TESTS_FILL_MODE=1|2|3, default 1):
 //
 //   Mode 1 (ascending):
 //     value = rank * nChannels + channel
 //     Decode: rank = value / nChannels, channel = value % nChannels
 //
 //   Mode 2 (descending):
 //     value = (nRanks * nChannels - 1) - (rank * nChannels + channel)
 //     Decode: combo = (nRanks*nCh-1) - value, rank = combo/nCh, ch = combo%nCh
 //
 //   Mode 3 (transposed):
 //     value = channel * nRanks + rank
 //     Decode: rank = value % nRanks, channel = value / nRanks
 //
 // All modes produce values in [0 .. nRanks*nChannels-1].
 // Different bit patterns help distinguish encoding-dependent artifacts
 // from true transport errors.
 // ====================================================================
 
 __host__ __device__ static float encode_value(int rank, int ch, int nChannels,
                                      int nRanks, int fillMode) {
   switch (fillMode) {
   case 2:  return (float)((nRanks * nChannels - 1) - (rank * nChannels + ch));
   case 3:  return (float)(ch * nRanks + rank);
   default: return (float)(rank * nChannels + ch);
   }
 }
 
 __global__ void fill_per_channel_kernel(float* buf, size_t count_per_rank,
                                         size_t floats_per_channel, int nChannels,
                                         int rank, int nRanks, int fillMode) {
   size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
   if (idx < count_per_rank) {
     int ch = (int)(idx / floats_per_channel);
     if (ch >= nChannels) ch = nChannels - 1;
     buf[idx] = encode_value(rank, ch, nChannels, nRanks, fillMode);
   }
 }
 
 __global__ void fill_expected_allgather_kernel(float* buf, size_t count_per_rank,
                                                size_t floats_per_channel, int nChannels,
                                                int nRanks, int fillMode) {
   size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
   size_t total = count_per_rank * (size_t)nRanks;
   if (idx < total) {
     int src_rank = (int)(idx / count_per_rank);
     size_t off = idx % count_per_rank;
     int ch = (int)(off / floats_per_channel);
     if (ch >= nChannels) ch = nChannels - 1;
     buf[idx] = encode_value(src_rank, ch, nChannels, nRanks, fillMode);
   }
 }
 
 // Read NCCL_TESTS_PERCHANNEL_INIT env var (0 or 1)
 static int getPerChannelInit() {
   static int val = -1;
   if (val == -1) {
     const char* env = getenv("NCCL_TESTS_PERCHANNEL_INIT");
     val = (env && atoi(env) > 0) ? 1 : 0;
   }
   return val;
 }
 
 // Read NCCL_MIN_NCHANNELS env var for channel count (default 16)
 static int getNumChannels() {
   static int val = -1;
   if (val == -1) {
     const char* env = getenv("NCCL_MIN_NCHANNELS");
     val = (env && atoi(env) > 0) ? atoi(env) : 16;
   }
   return val;
 }
 
 // Read NCCL_TESTS_FILL_MODE env var: 1=ascending, 2=descending, 3=transposed (default 1)
 static int getFillMode() {
   static int val = -1;
   if (val == -1) {
     const char* env = getenv("NCCL_TESTS_FILL_MODE");
     val = (env) ? atoi(env) : 1;
     if (val < 1 || val > 3) val = 1;
   }
   return val;
 }
 
 // ====================================================================
 
 void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
   size_t base = (count/nranks) & -(16/eltSize);
   *sendcount = base;
   *recvcount = base*nranks;
   *sendInplaceOffset = base;
   *recvInplaceOffset = 0;
   *paramcount = base;
 }
 
 // ---- Fault injection for testing debug features ----
// Corrupts the SEND buffer during initData so the AllGather naturally
// propagates wrong data.  No NCCL / stream interaction at all.
//
// NCCL_TESTS_INJECT_ERROR=<mode> (0=off, 1=NaN, 2=wrong_rank, 3=wrong_channel, 4=corrupt)
// NCCL_TESTS_INJECT_RANK=<rank>  which rank's send buffer to corrupt (default 1)
// NCCL_TESTS_INJECT_CH=<ch>      which channel to corrupt (default 0)
static int getInjectError() {
  static int val = -1;
  if (val == -1) {
    const char* env = getenv("NCCL_TESTS_INJECT_ERROR");
    val = (env) ? atoi(env) : 0;
  }
  return val;
}

__global__ void inject_fill_kernel(float* buf, size_t offset, size_t len, float value) {
  size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (idx < len) {
    buf[offset + idx] = value;
  }
}

__global__ void inject_nan_fill_kernel(float* buf, size_t offset, size_t len) {
  size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (idx < len) {
    uint32_t nan_bits = 0xFFFFFFFF;
    memcpy(&buf[offset + idx], &nan_bits, sizeof(float));
  }
}

static void injectSendError(float* sendBuf, size_t sendcount, int rank,
                             int nranks, int nChannels, int fillMode) {
  int mode = getInjectError();
  if (mode == 0) return;

  const char* irEnv = getenv("NCCL_TESTS_INJECT_RANK");
  int targetRank = (irEnv) ? atoi(irEnv) : 1;
  if (rank != targetRank) return;

  const char* icEnv = getenv("NCCL_TESTS_INJECT_CH");
  int targetCh = (icEnv) ? atoi(icEnv) : 0;
  if (targetCh >= nChannels) targetCh = nChannels - 1;

  size_t floatsPerCh = sendcount / nChannels;
  size_t chOffset = targetCh * floatsPerCh;
  int threads = 256;
  size_t blocks = (floatsPerCh + threads - 1) / threads;

  static int printed = 0;
  if (!printed) {
    printed = 1;
    printf("[INJECT] mode=%d targetRank=%d targetCh=%d nranks=%d sendcount=%zu floatsPerCh=%zu\n",
           mode, targetRank, targetCh, nranks, sendcount, floatsPerCh);
  }

  switch (mode) {
  case 1:
    inject_nan_fill_kernel<<<blocks, threads>>>(sendBuf, chOffset, floatsPerCh);
    break;
  case 2: {
    int fakeRank = (targetRank + 1) % nranks;
    float fakeVal = encode_value(fakeRank, targetCh, nChannels, nranks, fillMode);
    inject_fill_kernel<<<blocks, threads>>>(sendBuf, chOffset, floatsPerCh, fakeVal);
    break;
  }
  case 3: {
    int fakeCh = (targetCh + 1) % nChannels;
    float fakeVal = encode_value(rank, fakeCh, nChannels, nranks, fillMode);
    inject_fill_kernel<<<blocks, threads>>>(sendBuf, chOffset, floatsPerCh, fakeVal);
    break;
  }
  case 4:
    inject_fill_kernel<<<blocks, threads>>>(sendBuf, chOffset, floatsPerCh, -999.0f);
    break;
  }
}

// ====================================================================

testResult_t AllGatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
   size_t sendcount = args->sendBytes / wordSize(type);
   size_t recvcount = args->expectedBytes / wordSize(type);
   int nranks = args->nProcs*args->nThreads*args->nGpus;
 
   // ---- Per-channel initialization path (float only) ----
   if (getPerChannelInit() && type == ncclFloat) {
     int nChannels = getNumChannels();
     if (sendcount > 0 && sendcount % (size_t)nChannels == 0) {
       size_t floats_per_channel = sendcount / nChannels;
       int fillMode = getFillMode();
       int maxVal = nranks * nChannels - 1;
 
       // Print fill-mode summary once from rank 0
       static int fillSummaryPrinted = 0;
       int rank0 = ((args->proc*args->nThreads + args->thread)*args->nGpus + 0);
       if (!fillSummaryPrinted && rank0 == 0) {
         fillSummaryPrinted = 1;
         printf("\n");
         printf("============================================================\n");
         printf("  Per-Channel AllGather Initialization — Fill Mode %d\n", fillMode);
         printf("============================================================\n");
         printf("  nranks=%d  nChannels=%d  sendcount=%zu  floats_per_ch=%zu\n",
                nranks, nChannels, sendcount, floats_per_channel);
         printf("  recvbuf pre-fill: 0xFF (NaN) — detects unwritten regions\n");
         printf("  value range: [0 .. %d]\n\n", maxVal);
 
         if (fillMode == 1) {
           printf("  Encoding: value = rank * %d + channel  (ascending)\n", nChannels);
           printf("  Decoding: rank = value / %d,  channel = value %% %d\n\n", nChannels, nChannels);
         } else if (fillMode == 2) {
           printf("  Encoding: value = %d - (rank * %d + channel)  (descending)\n", maxVal, nChannels);
           printf("  Decoding: combo = %d - value,  rank = combo / %d,  channel = combo %% %d\n\n",
                  maxVal, nChannels, nChannels);
         } else {
           printf("  Encoding: value = channel * %d + rank  (transposed)\n", nranks);
           printf("  Decoding: rank = value %% %d,  channel = value / %d\n\n", nranks, nranks);
         }
 
         printf("  Sample entries:\n");
         int sampleRanks[] = {0, 0, 1, nranks/2, nranks - 1, nranks - 1};
         int sampleChs[]   = {0, nChannels - 1, 0, nChannels/2, 0, nChannels - 1};
         int nSamples = 6;
         for (int s = 0; s < nSamples; s++) {
           int sr = sampleRanks[s];
           int sc = sampleChs[s];
           float val;
           switch (fillMode) {
           case 2:  val = (float)(maxVal - (sr * nChannels + sc)); break;
           case 3:  val = (float)(sc * nranks + sr); break;
           default: val = (float)(sr * nChannels + sc); break;
           }
           printf("    rank=%5d  ch=%2d  ->  value=%.0f\n", sr, sc, val);
         }
         printf("============================================================\n\n");
         fflush(stdout);
       }
 
       for (int i=0; i<args->nGpus; i++) {
         CUDACHECK(cudaSetDevice(args->gpus[i]));
         int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
         // Fill recv buffer with 0xFF bytes → every float becomes NaN.
         // NaN != any valid value, so stale regions are always detected.
         // Valid per-channel values are non-negative integers [0 .. nranks*nCh-1];
         // zero (cudaMemset 0) would be confused with rank=0 channel=0.
         CUDACHECK(cudaMemset(args->recvbuffs[i], 0xFF, args->expectedBytes));
 
         float* data = in_place
             ? (float*)((char*)args->recvbuffs[i] + rank * args->sendBytes)
             : (float*)args->sendbuffs[i];
 
         int threads = 256;
         size_t blocks = (sendcount + threads - 1) / threads;
        fill_per_channel_kernel<<<blocks, threads>>>(
            data, sendcount, floats_per_channel, nChannels, rank, nranks, fillMode);
        CUDACHECK(cudaGetLastError());

        // Corrupt the send buffer AFTER filling — the AllGather will
        // naturally propagate the wrong data, triggering debug features.
        injectSendError(data, sendcount, rank, nranks, nChannels, fillMode);

        size_t total_floats = recvcount;
        size_t blocks_exp = (total_floats + threads - 1) / threads;
        fill_expected_allgather_kernel<<<blocks_exp, threads>>>(
            (float*)args->expected[i], sendcount, floats_per_channel, nChannels, nranks, fillMode);
        CUDACHECK(cudaGetLastError());

        CUDACHECK(cudaDeviceSynchronize());
       }
       return testSuccess;
     }
     // sendcount not divisible by nChannels — fall through to default
   }
 
   // ---- Original initialization path ----
   for (int i=0; i<args->nGpus; i++) {
     CUDACHECK(cudaSetDevice(args->gpus[i]));
     int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
     CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
     void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
     TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
     for (int j=0; j<nranks; j++) {
       TESTCHECK(InitData((char*)args->expected[i] + args->sendBytes*j, sendcount, 0, type, ncclSum, 33*rep + j, 1, 0));
     }
     CUDACHECK(cudaDeviceSynchronize());
   }
   return testSuccess;
 }
 
 testResult_t  AllGatherGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
   if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
   NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFunc_t::ncclFuncAllGather , count, type , 0, 0, 1, algo, proto, nchannels));
   return testSuccess;
 }
 
 
 void AllGatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
   double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;
 
   *algBw = baseBw;
   double factor = ((double)(nranks - 1))/((double)nranks);
   *busBw = baseBw * factor;
 }
 
testResult_t AllGatherRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
   if (deviceImpl == 0) {
     char* sptr = (char*)sendbuff + sendoffset;
     char* rptr = (char*)recvbuff + recvoffset;
     NCCLCHECK(ncclAllGather(sptr, rptr, count, type, comm, stream));
   } else {
     return testNotImplemented;
   }
   return testSuccess;
 }
 
 struct testColl allGatherTest = {
   "AllGather",
   AllGatherGetCollByteCount,
   AllGatherInitData,
   AllGatherGetBw,
   AllGatherRunColl,
   AllGatherGetAlgoProtoChannels
 };
 
 void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
   size_t paramcount, sendInplaceOffset, recvInplaceOffset;
   AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
 }
 
 testResult_t AllGatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
   args->collTest = &allGatherTest;
   ncclDataType_t *run_types;
   const char **run_typenames;
   int type_count;
 
   if ((int)type != -1) {
     type_count = 1;
     run_types = &type;
     run_typenames = &typeName;
   } else {
     type_count = test_typenum;
     run_types = test_types;
     run_typenames = test_typenames;
   }
 
   for (int i=0; i<type_count; i++) {
     TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
   }
   return testSuccess;
 }
 
 struct testEngine ncclTestEngine = {
   AllGatherGetBuffSize,
   AllGatherRunTest
 };
 