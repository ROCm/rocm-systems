/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

namespace {

__global__ void NoopKernel() {}

}  // namespace

#if !defined(__HIP_DEVICE_COMPILE__)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

#define HIP_CHECK(error)                                                                           \
  do {                                                                                             \
    hipError_t local_error = error;                                                                \
    if ((local_error != hipSuccess) && (local_error != hipErrorPeerAccessAlreadyEnabled)) {        \
      std::cerr << #error << " failed with " << hipGetErrorString(local_error) << " ("             \
                << local_error << ")" << std::endl;                                                \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

constexpr DWORD kTerminateWorkerDelayMs = 1000;

std::vector<DWORD> GetCurrentProcessThreadIds() {
  std::vector<DWORD> thread_ids;
  const DWORD current_process_id = GetCurrentProcessId();
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snapshot == INVALID_HANDLE_VALUE) {
    return thread_ids;
  }

  THREADENTRY32 entry = {};
  entry.dwSize = sizeof(entry);
  if (Thread32First(snapshot, &entry)) {
    do {
      if (entry.th32OwnerProcessID == current_process_id &&
          entry.th32ThreadID != GetCurrentThreadId()) {
        thread_ids.push_back(entry.th32ThreadID);
      }
      entry.dwSize = sizeof(entry);
    } while (Thread32Next(snapshot, &entry));
  }
  CloseHandle(snapshot);
  return thread_ids;
}

std::vector<DWORD> GetNewThreadIds(const std::vector<DWORD>& before,
                                   const std::vector<DWORD>& after) {
  std::vector<DWORD> new_thread_ids;
  for (DWORD thread_id : after) {
    if (std::find(before.begin(), before.end(), thread_id) == before.end()) {
      new_thread_ids.push_back(thread_id);
    }
  }
  return new_thread_ids;
}

DWORD WINAPI TerminateWorkerThreadAfterDelay(LPVOID data) {
  HANDLE worker_thread = static_cast<HANDLE>(data);
  Sleep(kTerminateWorkerDelayMs);
  const BOOL terminated = TerminateThread(worker_thread, 0);
  if (terminated) {
    WaitForSingleObject(worker_thread, 1000);
  } else {
    std::cerr << "Failed to terminate HIP worker thread after delay" << std::endl;
  }
  CloseHandle(worker_thread);
  return terminated ? 0 : 1;
}

}  // namespace

int main() {
  HIP_CHECK(hipSetDevice(0));

  const std::vector<DWORD> before_stream_create = GetCurrentProcessThreadIds();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  const std::vector<DWORD> after_stream_create = GetCurrentProcessThreadIds();
  const std::vector<DWORD> worker_thread_ids =
      GetNewThreadIds(before_stream_create, after_stream_create);
  if (worker_thread_ids.size() != 1) {
    std::cerr << "Expected one HIP worker thread, found " << worker_thread_ids.size() << std::endl;
    return 1;
  }

  HANDLE worker_thread = OpenThread(THREAD_TERMINATE | THREAD_SUSPEND_RESUME | SYNCHRONIZE, FALSE,
                                    worker_thread_ids.front());
  if (worker_thread == NULL) {
    std::cerr << "Failed to open HIP worker thread " << worker_thread_ids.front() << std::endl;
    return 1;
  }

  NoopKernel<<<1, 1, 0, stream>>>();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

  // Suspend worker so it cannot dequeue/process anything
  if (SuspendThread(worker_thread) == static_cast<DWORD>(-1)) {
    std::cerr << "Failed to suspend HIP worker thread " << worker_thread_ids.front() << std::endl;
    CloseHandle(worker_thread);
    return 1;
  }

  HIP_CHECK(hipEventRecord(event, stream));

  HANDLE killer_thread =
      CreateThread(nullptr, 0, TerminateWorkerThreadAfterDelay, worker_thread, 0, nullptr);
  if (killer_thread == NULL) {
    std::cerr << "Failed to create HIP worker killer thread" << std::endl;
    CloseHandle(worker_thread);
    return 1;
  }
  CloseHandle(killer_thread);

  return 0;
}

#endif
