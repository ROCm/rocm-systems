/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>

#include "hip_event.hpp"
#if !defined(_MSC_VER)
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <io.h>
#endif

// ================================================================================================
namespace hip {

hipError_t ihipEventCreateWithFlags(hipEvent_t* event, unsigned flags);

#if !defined(_MSC_VER)
namespace {
// Name of the POSIX shm object backing an IPC event: "/hip_<pid>_<nonce>_<counter>", all hex.
// The pid alone is not unique on a host: processes in different PID namespaces that share
// /dev/shm (e.g. containers started with --ipc=host) get identical pids, and "/hip_<pid>_<n>"
// then names the same object in both processes -- one silently re-initializes the other's live
// event. The random per-process nonce keeps names unique; it is regenerated after fork().
// At most 5 + 8 + 1 + 8 + 1 + 8 = 31 characters, so the name always fits the 32-byte handle.
// Lock-free on purpose: a mutex held by another thread during fork() would stay locked forever
// in the child.
std::atomic<uint64_t> ipc_name_id{0};  // (pid << 32) | nonce of the current process
std::atomic<uint32_t> ipc_name_counter{0};

std::string ipcNamePrefix() {
  const uint64_t pid = static_cast<uint32_t>(getpid());
  uint64_t id = ipc_name_id.load(std::memory_order_acquire);
  while ((id >> 32) != pid) {  // first use in this process, e.g. after fork()
    const uint64_t fresh = (pid << 32) | std::random_device{}();
    if (ipc_name_id.compare_exchange_weak(id, fresh, std::memory_order_acq_rel)) {
      id = fresh;
    }
  }
  char buf[24];
  snprintf(buf, sizeof(buf), "/hip_%x_%08x_", static_cast<unsigned>(id >> 32),
           static_cast<unsigned>(id));
  return buf;
}

std::string newIpcName() {
  char buf[9];
  snprintf(buf, sizeof(buf), "%x",
           static_cast<unsigned>(ipc_name_counter.fetch_add(1, std::memory_order_relaxed)));
  return ipcNamePrefix() + buf;
}

// True if this process created `name`. Replaces the pid comparison against owners_process_id,
// which reports "same process" for any process with the same pid in another PID namespace.
bool isOwnIpcName(const std::string& name) {
  const std::string prefix = ipcNamePrefix();
  return name.compare(0, prefix.size(), prefix) == 0;
}

// Creates (exclusively) or opens (must exist) the shm object of an IPC event and maps it.
// Returns 0 or an errno value. Opening never creates: a missing object used to be re-created
// zero-filled by the importer, which turned its stream wait into a silent no-op.
int mapIpcShmem(const std::string& name, bool create, ihipIpcEventShmem_t** shmem) {
  constexpr size_t kSize = sizeof(ihipIpcEventShmem_t);
  const int fd = create
      ? shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRWXU | S_IRWXG | S_IRWXO)
      : shm_open(name.c_str(), O_RDWR, 0);
  if (fd < 0) {
    return errno;
  }
  int err = 0;
  struct stat st;
  if (create) {
    if (ftruncate(fd, kSize) != 0) {
      err = errno;
    }
  } else if (fstat(fd, &st) != 0) {
    err = errno;
  } else if (static_cast<size_t>(st.st_size) < kSize) {
    err = EINVAL;
  }
  void* ptr = MAP_FAILED;
  if (err == 0) {
    ptr = mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      err = errno;
    }
  }
  close(fd);
  if (err != 0) {
    if (create) {
      shm_unlink(name.c_str());
    }
    return err;
  }
  *shmem = static_cast<ihipIpcEventShmem_t*>(ptr);
  return 0;
}
}  // namespace
#endif

bool IPCEvent::createIpcEventShmemIfNeeded() {
  if (ipc_evt_.ipc_shmem_) {
    // ipc_shmem_ already created, no need to create it again
    return true;
  }

#if !defined(_MSC_VER)
  int err = EEXIST;
  for (int attempt = 0; err == EEXIST && attempt < 4; ++attempt) {
    ipc_evt_.ipc_name_ = newIpcName();
    err = mapIpcShmem(ipc_evt_.ipc_name_, true, &ipc_evt_.ipc_shmem_);
  }
  if (err != 0) {
    ipc_evt_.ipc_shmem_ = nullptr;
    return false;
  }
#else
  char name_template[] = "/hip_XXXXXX";
  _mktemp_s(name_template, sizeof(name_template));
  ipc_evt_.ipc_name_ = name_template;
  ipc_evt_.ipc_name_.replace(0, 5, "/hip_");

  if (!amd::Os::MemoryMapFileTruncated(
          ipc_evt_.ipc_name_.c_str(),
          const_cast<const void**>(reinterpret_cast<void**>(&(ipc_evt_.ipc_shmem_))),
          sizeof(hip::ihipIpcEventShmem_t))) {
    return false;
  }
#endif
  ipc_evt_.ipc_creator_ = true;

  ipc_evt_.ipc_shmem_->owners = 1;
  ipc_evt_.ipc_shmem_->read_index = -1;
  ipc_evt_.ipc_shmem_->write_index = 0;
  for (uint32_t sig_idx = 0; sig_idx < IPC_SIGNALS_PER_EVENT; ++sig_idx) {
    ipc_evt_.ipc_shmem_->signal[sig_idx] = 0;
  }

  // device sets 0 to this ptr when the ipc event is completed
  hipError_t status =
      ihipHostRegister(&ipc_evt_.ipc_shmem_->signal, sizeof(uint32_t) * IPC_SIGNALS_PER_EVENT, 0);
  if (status != hipSuccess) {
    return false;
  }
  return true;
}

// ================================================================================================
hipError_t IPCEvent::query() {
  if (ipc_evt_.ipc_shmem_) {
    int prev_read_idx = ipc_evt_.ipc_shmem_->read_index;
    if (prev_read_idx < 0) {
      // Never recorded: -1 % IPC_SIGNALS_PER_EVENT would index signal[-1].
      return hipSuccess;
    }
    int offset = (prev_read_idx % IPC_SIGNALS_PER_EVENT);
    if (ipc_evt_.ipc_shmem_->read_index < prev_read_idx + IPC_SIGNALS_PER_EVENT &&
        ipc_evt_.ipc_shmem_->signal[offset] != 0) {
      return hipErrorNotReady;
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::synchronize() {
  if (ipc_evt_.ipc_shmem_) {
    int prev_read_idx = ipc_evt_.ipc_shmem_->read_index;
    if (prev_read_idx >= 0) {
      int offset = (prev_read_idx % IPC_SIGNALS_PER_EVENT);
      while ((ipc_evt_.ipc_shmem_->read_index < prev_read_idx + IPC_SIGNALS_PER_EVENT) &&
             (ipc_evt_.ipc_shmem_->signal[offset] != 0)) {
        amd::Os::sleep(1);
      }
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::streamWait(hip::Stream* stream, uint flags) {
  // Waiting on an event that was never recorded is a no-op (as for cudaStreamWaitEvent);
  // before, a local interprocess event without shm dereferenced a null pointer here.
  if (ipc_evt_.ipc_shmem_ == nullptr) {
    return hipSuccess;
  }
  const int read_index = ipc_evt_.ipc_shmem_->read_index;
  if (read_index < 0) {
    return hipSuccess;
  }
  // read_index counts every record of the event, the signal ring has IPC_SIGNALS_PER_EVENT
  // slots (see enqueueRecordCommand/query/synchronize). Without the modulo, from the 33rd
  // record on every wait addressed memory past the registered signal array:
  // hipErrorInvalidValue, or a wait on unrelated memory.
  const int offset = read_index % IPC_SIGNALS_PER_EVENT;
  hipError_t status =
      ihipStreamOperation(reinterpret_cast<hipStream_t>(stream), ROCCLR_COMMAND_STREAM_WAIT_VALUE,
                          &(ipc_evt_.ipc_shmem_->signal[offset]), 0, 1, 1, sizeof(uint32_t));
  return status;
}

// ================================================================================================
hipError_t IPCEvent::recordCommand(amd::Command*& command, amd::HostQueue* stream, uint32_t flags,
                                   bool batch_flush) {
  command = new amd::Marker(*stream, kMarkerDisableFlush);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::enqueueRecordCommand(hip::Stream* stream, amd::Command* command) {
  amd::Event& tEvent = command->event();
  if (!createIpcEventShmemIfNeeded()) {
    command->release();  // ownership passed to us; it was never enqueued
    return hipErrorInvalidValue;
  }
  int write_index = ipc_evt_.ipc_shmem_->write_index++;
  int offset = write_index % IPC_SIGNALS_PER_EVENT;
  while (ipc_evt_.ipc_shmem_->signal[offset] != 0) {
    amd::Os::sleep(1);
  }
  // Lock signal.
  ipc_evt_.ipc_shmem_->signal[offset] = 1;
  ipc_evt_.ipc_shmem_->owners_device_id = deviceId();
  command->enqueue();

  // Set event_ in order to release marked command when event is destroyed
  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &command->event();

  // device writes 0 to signal after the hipEventRecord command is completed
  // the signal value is checked by WaitThenDecrementSignal cb
  hipError_t status =
      ihipStreamOperation(reinterpret_cast<hipStream_t>(stream), ROCCLR_COMMAND_STREAM_WRITE_VALUE,
                          &(ipc_evt_.ipc_shmem_->signal[offset]), 0, 0, 0, sizeof(uint32_t));

  if (status != hipSuccess) {
    return status;
  }

  // Update read index to indicate new signal.
  int expected = write_index - 1;
  while (!ipc_evt_.ipc_shmem_->read_index.compare_exchange_weak(expected, write_index)) {
    amd::Os::sleep(1);
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::GetHandle(ihipIpcEventHandle_t* handle) {
  if (!createIpcEventShmemIfNeeded()) {
    return hipErrorInvalidValue;
  }
  if (ipc_evt_.ipc_name_.size() >= sizeof(handle->shmem_name)) {
    return hipErrorInvalidValue;  // the name plus NUL must fit into shmem_name
  }
  ipc_evt_.ipc_shmem_->owners_device_id = deviceId();
  ipc_evt_.ipc_shmem_->owners_process_id = amd::Os::getProcessId();
  memset(handle->shmem_name, 0, HIP_IPC_HANDLE_SIZE);
  ipc_evt_.ipc_name_.copy(handle->shmem_name, std::string::npos);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::OpenHandle(ihipIpcEventHandle_t* handle) {
#if !defined(_MSC_VER)
  ipc_evt_.ipc_name_ =
      std::string(handle->shmem_name, strnlen(handle->shmem_name, sizeof(handle->shmem_name)));
  if (isOwnIpcName(ipc_evt_.ipc_name_)) {
    // If this is in the same process, return error.
    return hipErrorInvalidContext;
  }
  if (mapIpcShmem(ipc_evt_.ipc_name_, false, &ipc_evt_.ipc_shmem_) != 0) {
    ipc_evt_.ipc_shmem_ = nullptr;
    return hipErrorInvalidValue;  // the exporter's object is gone (or never existed)
  }
#else
  ipc_evt_.ipc_name_ = handle->shmem_name;
  if (!amd::Os::MemoryMapFileTruncated(ipc_evt_.ipc_name_.c_str(),
                                       (const void**)&(ipc_evt_.ipc_shmem_),
                                       sizeof(ihipIpcEventShmem_t))) {
    return hipErrorInvalidValue;
  }

  if (amd::Os::getProcessId() == ipc_evt_.ipc_shmem_->owners_process_id.load()) {
    // If this is in the same process, return error.
    return hipErrorInvalidContext;
  }
#endif

  ipc_evt_.ipc_shmem_->owners += 1;
  // device sets 0 to this ptr when the ipc event is completed
  hipError_t status = hipSuccess;
  status =
      ihipHostRegister(&ipc_evt_.ipc_shmem_->signal, sizeof(uint32_t) * IPC_SIGNALS_PER_EVENT, 0);
  return status;
}

// ================================================================================================
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  HIP_INIT_API(hipIpcGetEventHandle, handle, event);

  if (handle == nullptr || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  HIP_RETURN(e->GetHandle(reinterpret_cast<ihipIpcEventHandle_t*>(handle)));
}

hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  HIP_INIT_API(hipIpcOpenEventHandle, event, handle);

  hipError_t status = hipSuccess;
  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  status = ihipEventCreateWithFlags(event, hipEventDisableTiming | hipEventInterprocess);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(*event);
  ihipIpcEventHandle_t* iHandle = reinterpret_cast<ihipIpcEventHandle_t*>(&handle);

  status = e->OpenHandle(iHandle);
  // Free the event in case of failure
  if (status != hipSuccess) {
    delete e;
  }
  HIP_RETURN(status);
}
}  // namespace hip
