#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>

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
   
  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
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

   
  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
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

  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
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
   
  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
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

int main()
{
  hipError_t err = hipSuccess;
  hipStream_t main_stream, side_stream;
  hipGraph_t graph;
  checkRelaxed(hipStreamCaptureModeRelaxed);
  checkRelaxed(hipStreamCaptureModeThreadLocal);
  checkRelaxed(hipStreamCaptureModeGlobal);
  checkRelaxedSideStream();
  checkGlobalSideStream();
  checkThreadLocalSideStream();
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
    printf("mode relaxed hipDeviceSynchronize error\n");

  err = hipStreamSynchronize(main_stream);
  if (err != hipSuccess)
    printf("mode relaxed main stream hipStreamSynchronize error\n");

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

  return 0;
}
