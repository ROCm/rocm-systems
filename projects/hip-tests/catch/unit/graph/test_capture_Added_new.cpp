#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(error)                                   \
{                                                          \
  hipError_t localError = error;                           \
  if ((localError != hipSuccess) &&                        \
      (localError != hipErrorPeerAccessAlreadyEnabled)) {  \
        printf("error: '%s'(%d) from %s at %s:%d\n",       \
               hipGetErrorString(localError),              \
               localError, #error, __FUNCTION__, __LINE__);\
  exit(0);                                                 \
  }                                                        \
}

#define REQUIRE(x)                            \
{                                             \
  if(x) {                                     \
  } else {                                    \
    printf("At line %d : FALSE\n", __LINE__); \
    exit(0);                                  \
  }                                           \
}

__global__ void addOneKernel(int* a, size_t size) {
  size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < size; i += stride) {
    a[i] += 1;
  }
}

// main_stream
void checkRelaxed(hipStreamCaptureMode captureMode)
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream;
  hipGraph_t graph;

  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");
  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipStreamSynchronize(main_stream);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamSynchronize error on main stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
   
  mode = hipStreamCaptureModeRelaxed;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipDeviceSynchronize();
  if (err != hipSuccess)
    printf("mode relaxed hipDeviceSynchronize error on main stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

   
  mode = hipStreamCaptureModeRelaxed;
  err = hipThreadExchangeStreamCaptureMode(&mode);


  hipEvent_t e;
  err = hipEventCreate(&e);
  err = hipEventRecord(e, main_stream);
  err = hipEventSynchronize(e);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipEventSynchronize error on main stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  mode = hipStreamCaptureModeRelaxed;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipEventCreate(&e);
  err = hipEventRecord(e, main_stream);
  err = hipStreamWaitEvent(main_stream, e, 0);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipStreamWaitEvent error on main stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
   
  mode = hipStreamCaptureModeRelaxed;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipStreamQuery(main_stream);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipStreamQuery error on main stream %d\n", err);
  }
  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
}

void checkGlobal(hipStreamCaptureMode captureMode)
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream;
  hipGraph_t graph;

  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");
  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  hipStreamCaptureMode mode = hipStreamCaptureModeGlobal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipStreamSynchronize(main_stream);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamSynchronize error on main stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
   
  mode = hipStreamCaptureModeGlobal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipDeviceSynchronize();
  if (err != hipSuccess)
    printf("mode relaxed hipDeviceSynchronize error on main stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

   
  mode = hipStreamCaptureModeGlobal;
  err = hipThreadExchangeStreamCaptureMode(&mode);


  hipEvent_t e;
  err = hipEventCreate(&e);
  err = hipEventRecord(e, main_stream);
  err = hipEventSynchronize(e);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipEventSynchronize error on main stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  mode = hipStreamCaptureModeGlobal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipEventCreate(&e);
  err = hipEventRecord(e, main_stream);
  err = hipStreamWaitEvent(main_stream, e, 0);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipStreamWaitEvent error on main stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
   
  mode = hipStreamCaptureModeGlobal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipStreamQuery(main_stream);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipStreamQuery error on main stream %d\n", err);
  }
  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
}

void checkThreadLocal(hipStreamCaptureMode captureMode)
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream;
  hipGraph_t graph;

  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");
  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  hipStreamCaptureMode mode = hipStreamCaptureModeThreadLocal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipStreamSynchronize(main_stream);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamSynchronize error on main stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
   
  mode = hipStreamCaptureModeThreadLocal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipDeviceSynchronize();
  if (err != hipSuccess)
    printf("mode relaxed hipDeviceSynchronize error on main stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

   
  mode = hipStreamCaptureModeThreadLocal;
  err = hipThreadExchangeStreamCaptureMode(&mode);


  hipEvent_t e;
  err = hipEventCreate(&e);
  err = hipEventRecord(e, main_stream);
  err = hipEventSynchronize(e);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipEventSynchronize error on main stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  mode = hipStreamCaptureModeThreadLocal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipEventCreate(&e);
  err = hipEventRecord(e, main_stream);
  err = hipStreamWaitEvent(main_stream, e, 0);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipStreamWaitEvent error on main stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, captureMode);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
   
  mode = hipStreamCaptureModeThreadLocal;
  err = hipThreadExchangeStreamCaptureMode(&mode);

  err = hipStreamQuery(main_stream);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipStreamQuery error on main stream %d\n", err);
  }
  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
}



void checkRelaxedSideStream()
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream, side_stream;
  hipGraph_t graph;

  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");

  err = hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 2 error\n");

  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeRelaxed);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipStreamSynchronize(side_stream);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamSynchronize error on side_stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeRelaxed);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  hipEvent_t e;
  err = hipEventCreate(&e);
  err = hipEventRecord(e, side_stream);
  err = hipEventSynchronize(e);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipEventSynchronize error on side_stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeRelaxed);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
  err = hipEventCreate(&e);
  err = hipEventRecord(e, side_stream);
  err = hipStreamWaitEvent(side_stream, e, 0);
  if (err != hipSuccess)
  {
    printf("mode relaxed hipStreamWaitEvent error on side_stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);

  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeRelaxed);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipStreamQuery(side_stream);
  if (err != hipSuccess)
  {
    printf("mode Relaxed hipStreamQuery error on side stream %d\n", err);
  }
  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode Relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
}

void checkGlobalSideStream()
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream, side_stream;
  hipGraph_t graph;

  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");

  err = hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 2 error\n");

  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeGlobal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipStreamSynchronize(side_stream);
  if (err != hipSuccess)
    printf("mode Global hipStreamSynchronize error on side_stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode Global hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeGlobal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  hipEvent_t e;
  err = hipEventCreate(&e);
  err = hipEventRecord(e, side_stream);
  err = hipEventSynchronize(e);
  if (err != hipSuccess)
  {
    printf("mode Global hipEventSynchronize error on side_stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode Global hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeGlobal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipEventCreate(&e);
  err = hipEventRecord(e, side_stream);
  err = hipStreamWaitEvent(side_stream, e, 0);
  if (err != hipSuccess)
  {
    printf("mode Global hipStreamWaitEvent error on side_stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode Global hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeGlobal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipStreamQuery(side_stream);
  if (err != hipSuccess)
  {
    printf("mode Global hipStreamQuery error on side stream %d\n", err);
  }
  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode Global hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
}

void checkThreadLocalSideStream()
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream, side_stream;
  hipGraph_t graph;

  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");

  err = hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 2 error\n");

  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeThreadLocal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipStreamSynchronize(side_stream);
  if (err != hipSuccess)
    printf("mode ThreadLocal hipStreamSynchronize error on side_stream %d\n", err);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode ThreadLocal hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeThreadLocal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  hipEvent_t e;
  err = hipEventCreate(&e);
  err = hipEventRecord(e, side_stream);
  err = hipEventSynchronize(e);
  if (err != hipSuccess)
  {
    printf("mode ThreadLocal hipEventSynchronize error on side_stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode ThreadLocal hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeThreadLocal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");
  err = hipEventCreate(&e);
  err = hipEventRecord(e, side_stream);
  err = hipStreamWaitEvent(side_stream, e, 0);
  if (err != hipSuccess)
  {
    printf("mode ThreadLocal hipStreamWaitEvent error on side_stream %d\n", err);
  }
  hipEventDestroy(e);

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode ThreadLocal hipStreamEndCapture error on main stream %d\n", err);

  //---------------------------------------------------------------------------

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeThreadLocal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipStreamQuery(side_stream);
  if (err != hipSuccess)
  {
    printf("mode ThreadLocal hipStreamQuery error on side stream %d\n", err);
  }
  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode ThreadLocal hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
}


void checkRelaxedSideStreamWithOps()
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream, side_stream1, side_stream2;
  hipGraph_t graph;

  constexpr int N = 40;
  constexpr int Nbytes = N * sizeof(int);

  std::vector<int> hostMem(N);
  std::fill(hostMem.begin(), hostMem.end(), 5);

  int* devMem1 = nullptr;
  HIP_CHECK(hipMalloc(&devMem1, Nbytes));
  REQUIRE(devMem1 != nullptr);

  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");

  err = hipStreamCreateWithFlags(&side_stream1, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 2 error\n");

  err = hipStreamCreateWithFlags(&side_stream2, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 3 error\n");

  //---------------------------------------------------------------------------
  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeRelaxed);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  hipEvent_t H2D_event;
  err = hipEventCreate(&H2D_event);

  err = hipMemcpyAsync(devMem1, hostMem.data(), Nbytes,
                           hipMemcpyHostToDevice, side_stream1);

  err = hipEventRecord(H2D_event, side_stream1);

  err = hipStreamWaitEvent(side_stream2, H2D_event, 0);

  addOneKernel<<< 1, 1, 0, side_stream2 >>>(devMem1 , N);

  err = hipMemcpyAsync(hostMem.data(), devMem1, Nbytes,
                       hipMemcpyDeviceToHost, side_stream2);

  err = hipStreamSynchronize(side_stream2);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamSynchronize error on side_stream %d\n", err);

  err = hipStreamQuery(side_stream1);
  if (err != hipSuccess)
  {
    printf("mode Relaxed hipStreamQuery error on side stream 1 -> %d\n", err);
  }
  err = hipStreamQuery(side_stream2);
  if (err != hipSuccess)
  {
    printf("mode Relaxed hipStreamQuery error on side stream 2 -> %d\n", err);
  }

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamEndCapture error on main stream %d\n", err);
  //---------------------------------------------------------------------------
  
  for (int i = 0; i < N; i++) {
    if (hostMem[i] != 6) {
      std::cout << "At index : " << i << " Got value : " << hostMem[i]
                << " Expected value : 6 \n"
                << std::endl;
      REQUIRE(false);
    }
  }

}




int main()
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream, side_stream;
  hipGraph_t graph;
  /*
  std::cout << "==========================================================================" << std::endl;
  checkRelaxed(hipStreamCaptureModeRelaxed);
  std::cout << "==========================================================================" << std::endl;
  checkRelaxed(hipStreamCaptureModeThreadLocal);
  std::cout << "==========================================================================" << std::endl;
  checkRelaxed(hipStreamCaptureModeGlobal);
  std::cout << "==========================================================================" << std::endl;
  */
  /*
  std::cout << "==========================================================================" << std::endl;
  checkGlobal(hipStreamCaptureModeRelaxed);
  std::cout << "==========================================================================" << std::endl;
  checkGlobal(hipStreamCaptureModeThreadLocal);
  std::cout << "==========================================================================" << std::endl;
  checkGlobal(hipStreamCaptureModeGlobal);
  */
  
  /*
  std::cout << "==========================================================================" << std::endl;
  checkThreadLocal(hipStreamCaptureModeRelaxed);
  std::cout << "==========================================================================" << std::endl;
  checkThreadLocal(hipStreamCaptureModeThreadLocal);
  std::cout << "==========================================================================" << std::endl;
  checkThreadLocal(hipStreamCaptureModeGlobal);
  */

  /*
  checkRelaxedSideStream();
  std::cout << "==========================================================================" << std::endl;
  checkGlobalSideStream();
  std::cout << "==========================================================================" << std::endl;
  checkThreadLocalSideStream();
  std::cout << "==========================================================================" << std::endl;
  */
  /*
  err = hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 1 error\n");

  err = hipStreamBeginCapture(main_stream, hipStreamCaptureModeThreadLocal);
  if (err != hipSuccess)
    printf("hipStreamBeginCapture error\n");

  err = hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking);
  if (err != hipSuccess)
    printf("hipStreamCreateWithFlags 2 error\n");

  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
  err = hipThreadExchangeStreamCaptureMode(&mode);
  if (err != hipSuccess)
    printf("hipThreadExchangeStreamCaptureMode 1 error\n");

  err = hipStreamSynchronize(side_stream);
  if (err != hipSuccess)
    printf("mode relaxed hipStreamSynchronize error\n");

  err = hipDeviceSynchronize();
  if (err != hipSuccess)
    printf("%d mode relaxed hipDeviceSynchronize error\n", __LINE__);

  err = hipStreamSynchronize(main_stream);
  if (err != hipSuccess)
    printf("mode relaxed main stream hipStreamSynchronize error\n");

  std::cout << "At line : " << __LINE__ << " mode = " << mode << std::endl;
  err = hipThreadExchangeStreamCaptureMode(&mode);
  if (err != hipSuccess)
    printf("hipThreadExchangeStreamCaptureMode 2 error\n");

  err = hipStreamSynchronize(side_stream);
  if (err != hipSuccess)
    printf("mode hipStreamCaptureModeThreadLocal hipStreamSynchronize error\n");

  mode = hipStreamCaptureModeGlobal;
  err = hipThreadExchangeStreamCaptureMode(&mode);
  if (err != hipSuccess)
    printf("hipThreadExchangeStreamCaptureMode 3 error\n");

  err = hipStreamSynchronize(side_stream);
  if (err != hipSuccess)
    printf("mode hipStreamCaptureModeGlobal hipStreamSynchronize error\n");

  err = hipStreamEndCapture(main_stream, &graph);
  if (err != hipSuccess)
    printf("hipStreamEndCapture error\n");

  */
  
  checkRelaxedSideStreamWithOps();
  
  return 0;
}
