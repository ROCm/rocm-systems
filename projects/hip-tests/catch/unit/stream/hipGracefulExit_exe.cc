#include <hip/hip_runtime.h>

#include <windows.h>
#include <tlhelp32.h>

#include <cstdio>
#include <set>
#include <vector>

// Lightweight kernel — just enough to keep the worker thread busy.
__global__ void increment(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += 1.0f;
}

static std::set<DWORD> snapshotThreads() {
  std::set<DWORD> ids;
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snap == INVALID_HANDLE_VALUE) return ids;
  THREADENTRY32 te{};
  te.dwSize = sizeof(te);
  DWORD pid = GetCurrentProcessId();
  if (Thread32First(snap, &te)) {
    do {
      if (te.th32OwnerProcessID == pid) ids.insert(te.th32ThreadID);
    } while (Thread32Next(snap, &te));
  }
  CloseHandle(snap);
  return ids;
}

// Spawn a HIP stream (which internally creates a HostQueue worker thread in
// clr/rocclr/platform/commandqueue.cpp), queue async GPU work on it, then
// kill the worker thread ungracefully via TerminateThread and wait for it to
// actually exit, so that the atexit cleanup path
// (__hipUnregisterFatBinary -> SyncAllStreams -> HostQueue::finish) is forced
// to call finish() on a dead worker thread.
//
// Without the fix for rocm-systems PR#3790: finish() blocks waiting for a
// signal from the dead thread -> hangs.
// With the fix: exits cleanly with code 0.
int main() {
  float* d_data = nullptr;
  if (hipMalloc(&d_data, 1024 * sizeof(float)) != hipSuccess) return 1;

  // Stabilize all HIP-internal background threads before snapshotting.
  (void)hipDeviceSynchronize();
  Sleep(200);

  auto before = snapshotThreads();

  hipStream_t stream;
  if (hipStreamCreate(&stream) != hipSuccess) return 1;
  Sleep(100);  // let the HostQueue worker thread start

  auto after = snapshotThreads();

  // Queue async work while the worker thread is still alive.
  increment<<<4, 256, 0, stream>>>(d_data, 1024);

  // Kill all threads that appeared after hipStreamCreate. We expect exactly one:
  // the HostQueue worker thread whose handle Os::isThreadAlive checks. Any
  // additional threads (e.g. PAL/ROCr threads from createVirtualDevice()) are
  // killed defensively and reported as a warning.
  std::vector<DWORD> killedTids;
  for (DWORD tid : after) {
    if (!before.count(tid)) {
      HANDLE h = OpenThread(THREAD_TERMINATE | SYNCHRONIZE, FALSE, tid);
      if (h) {
        TerminateThread(h, 0);
        // TerminateThread is asynchronous — wait for the thread to actually exit
        // so that GetExitCodeThread returns 0 (not STILL_ACTIVE=259) during atexit.
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
        killedTids.push_back(tid);
      }
    }
  }

  if (killedTids.empty()) {
    // AMD_DIRECT_DISPATCH mode: no worker OS thread is created, nothing to test.
    return 0;
  } else if (killedTids.size() > 1) {
    // Unexpected: more threads appeared than just the HostQueue worker.
    printf("[hipGracefulExit_exe] warning: expected 1 new thread, got %zu\n", killedTids.size());
    for (DWORD tid : killedTids)
      printf("[hipGracefulExit_exe] killed tid=%lu\n", tid);
    fflush(stdout);
  }

  // Do NOT call hipStreamDestroy — the stream stays in the active list.
  // atexit: __hipUnregisterFatBinary -> SyncAllStreams -> finish() is called
  // on this stream whose worker thread is now dead.
  // Without the fix: finish() blocks on the dead thread -> hangs.
  // With the fix: exits cleanly with code 0.
  return 0;
}
