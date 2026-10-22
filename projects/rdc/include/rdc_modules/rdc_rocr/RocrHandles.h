/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RDC_MODULES_RDC_ROCR_ROCRHANDLES_H_
#define RDC_MODULES_RDC_ROCR_ROCRHANDLES_H_

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <string>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_modules/rdc_rocr/GpuOutcome.h"
#include "rdc_modules/rdc_rocr/common.h"

// One holder per kind of object a check makes, so that every path out of a check releases what it
// made before the test object's hsa_shut_down(), which stays last. Each holder is move-only, so
// no handle has two deleters, and starts empty. What abandons the objects a kernel or a copy
// can still touch is the flag of the release context, which every reset() reads: a holder whose
// context says the work could not be stopped keeps its handle and runs no deleter.

namespace amd {
namespace rdc {

// A release that failed, logged at error level with the call and its status. A destructor must
// not throw, so the line is put together inside a catch-all and dropped rather than ending the
// process when it cannot be.
inline void log_release_failure(const char* call, hsa_status_t status) {
  try {
    RDC_LOG(RDC_ERROR, hsa_error_text(std::string("Fail to call ") + call + "().", status));
  } catch (...) {
  }
}

// Where a test object's holders report a release that failed, and the flag they read before they
// release anything: a queue that cannot be destroyed abandons the rest, as a kernel or a copy
// that could not be stopped does at its deadline, and then nothing that work can touch is freed.
// The GPUs whose work a check could not stop, which the module records for the life of the
// process: an abandoned kernel or copy, and a queue that could not be destroyed, leave memory the
// GPU may still write into and a queue nobody can wait on, so no later run checks them. A run
// reads the record before each of its GPUs, before any HSA call.
void abandon_gpu(uint32_t entity);
std::vector<uint32_t> abandoned_gpus(void);

class ReleaseContext {
 public:
  ReleaseContext(uint32_t entity, GpuOutcome* outcome) : entity_(entity), outcome_(outcome) {}

  // The GPU's failure, named by the call that failed and its status.
  void release_failed(const char* call, hsa_status_t status) {
    log_release_failure(call, status);
    try {
      if (outcome_ != nullptr) {
        outcome_->add(RDC_DIAG_RESULT_FAIL,
                      hsa_error_text(std::string("Fail to call ") + call + "().", status));
      }
    } catch (...) {
    }
  }

  // Work on the GPU that could not be stopped: nothing that work can touch is released from
  // here on, the module pins itself so that its queue error callback and its record stay mapped,
  // and the GPU is recorded, which leaves the checks disabled until the process restarts.
  void abandon() {
    if (abandoned_) {
      return;
    }
    abandoned_ = true;
    abandon_gpu(entity_);
  }
  bool abandoned() const { return abandoned_; }

  // Where a sub-test's own sentences go, which is null only where no outcome was given.
  GpuOutcome* outcome() const { return outcome_; }

 private:
  uint32_t entity_ = 0;  // the GPU's entity id, which names it in the record and in the log
  GpuOutcome* outcome_ = nullptr;
  bool abandoned_ = false;
};

// The kinds of object the checks make. Each says what an empty holder holds, how the object is
// released and which call that is, and whether a failed release abandons the rest.
struct QueueKind {
  using handle_type = hsa_queue_t*;
  static handle_type empty() { return nullptr; }
  static bool is_empty(handle_type handle) { return handle == nullptr; }
  static hsa_status_t destroy(handle_type handle) { return hsa_queue_destroy(handle); }
  static const char* call() { return "hsa_queue_destroy"; }
  // A queue that cannot be destroyed may still run the work that was put on it.
  static constexpr bool kAbandonsTheRest = true;
};

struct SignalKind {
  using handle_type = hsa_signal_t;
  static handle_type empty() { return hsa_signal_t{0}; }
  static bool is_empty(handle_type handle) { return handle.handle == 0; }
  static hsa_status_t destroy(handle_type handle) { return hsa_signal_destroy(handle); }
  static const char* call() { return "hsa_signal_destroy"; }
  static constexpr bool kAbandonsTheRest = false;
};

struct PoolBlockKind {
  using handle_type = void*;
  static handle_type empty() { return nullptr; }
  static bool is_empty(handle_type handle) { return handle == nullptr; }
  // A pool allocation is returned to its pool, not with hsa_memory_free.
  static hsa_status_t destroy(handle_type handle) { return hsa_amd_memory_pool_free(handle); }
  static const char* call() { return "hsa_amd_memory_pool_free"; }
  static constexpr bool kAbandonsTheRest = false;
};

struct ExecutableKind {
  using handle_type = hsa_executable_t;
  static handle_type empty() { return hsa_executable_t{0}; }
  static bool is_empty(handle_type handle) { return handle.handle == 0; }
  static hsa_status_t destroy(handle_type handle) { return hsa_executable_destroy(handle); }
  static const char* call() { return "hsa_executable_destroy"; }
  static constexpr bool kAbandonsTheRest = false;
};

struct CodeObjectReaderKind {
  using handle_type = hsa_code_object_reader_t;
  static handle_type empty() { return hsa_code_object_reader_t{0}; }
  static bool is_empty(handle_type handle) { return handle.handle == 0; }
  static hsa_status_t destroy(handle_type handle) { return hsa_code_object_reader_destroy(handle); }
  static const char* call() { return "hsa_code_object_reader_destroy"; }
  static constexpr bool kAbandonsTheRest = false;
};

// The error status of a queue, which ROCr's error callback writes from a thread of its own and
// which each wait and the check before each doorbell read.
using QueueStatus = std::atomic<hsa_status_t>;

// The status block of one queue, made just before it and released just after it is destroyed, so
// that no block is shared or reused: ROCr may call a queue's error callback from its own threads
// until hsa_queue_destroy returns. The block of a queue that was abandoned is given up with it
// and never freed, so that an error reported later writes into memory that is still allocated.
struct QueueStatusKind {
  using handle_type = QueueStatus*;
  static handle_type empty() { return nullptr; }
  static bool is_empty(handle_type handle) { return handle == nullptr; }
  static hsa_status_t destroy(handle_type handle) {
    delete handle;
    return HSA_STATUS_SUCCESS;
  }
  static const char* call() { return "delete"; }
  static constexpr bool kAbandonsTheRest = false;
};

// One object of that kind, released when the holder goes out of scope.
template <typename Kind>
class HsaHolder {
 public:
  using handle_type = typename Kind::handle_type;

  HsaHolder() = default;
  explicit HsaHolder(ReleaseContext* context) : context_(context) {}
  HsaHolder(const HsaHolder&) = delete;
  HsaHolder& operator=(const HsaHolder&) = delete;
  HsaHolder(HsaHolder&& other) : handle_(other.handle_), context_(other.context_) {
    other.handle_ = Kind::empty();  // a move leaves the source empty, so no handle has two
                                    // deleters
  }
  HsaHolder& operator=(HsaHolder&& other) {
    if (this != &other) {
      reset();
      handle_ = other.handle_;
      context_ = other.context_;
      other.handle_ = Kind::empty();
    }
    return *this;
  }
  ~HsaHolder() { reset(); }

  // Takes the object a call has just made, releasing the one this holder held, if any.
  void reset(handle_type handle = Kind::empty()) {
    if (!Kind::is_empty(handle_)) {
      if (context_ != nullptr && context_->abandoned()) {
        // The work this object may be part of could not be stopped: nothing it can touch is
        // released, and the handle is given up as it is.
        release_failed_ = true;
      } else {
        const hsa_status_t err = Kind::destroy(handle_);
        if (err != HSA_STATUS_SUCCESS) {
          release_failed_ = true;
          if (context_ != nullptr) {
            if (Kind::kAbandonsTheRest) {
              context_->abandon();
            }
            context_->release_failed(Kind::call(), err);
          } else {
            log_release_failure(Kind::call(), err);
          }
        }
      }
    }
    handle_ = handle;
  }

  handle_type get() const { return handle_; }
  bool empty() const { return Kind::is_empty(handle_); }

  // Whether an object this holder held was not released: a release that failed, or one that was
  // not made because the work on the GPU could not be stopped. A check whose sentence says what
  // it did with an object reads this, so that it reports a free that failed as what it was.
  bool release_failed() const { return release_failed_; }

  // The block of a pool allocation, read as the type the check keeps in it.
  template <typename T>
  T* as() const {
    return static_cast<T*>(handle_);
  }

 private:
  handle_type handle_ = Kind::empty();
  ReleaseContext* context_ = nullptr;
  bool release_failed_ = false;
};

using QueueHolder = HsaHolder<QueueKind>;
using QueueStatusHolder = HsaHolder<QueueStatusKind>;
using SignalHolder = HsaHolder<SignalKind>;
using PoolBlockHolder = HsaHolder<PoolBlockKind>;
using ExecutableHolder = HsaHolder<ExecutableKind>;
using CodeObjectReaderHolder = HsaHolder<CodeObjectReaderKind>;

// The descriptor of a kernel file, which the search opens and hands over in it. It is closed as
// soon as the code object reader has been made from it, whatever that call returned, and on any
// path out before that.
class FileDescriptorHolder {
 public:
  FileDescriptorHolder() = default;
  explicit FileDescriptorHolder(int descriptor) : descriptor_(descriptor) {}
  FileDescriptorHolder(const FileDescriptorHolder&) = delete;
  FileDescriptorHolder& operator=(const FileDescriptorHolder&) = delete;
  FileDescriptorHolder(FileDescriptorHolder&& other) : descriptor_(other.descriptor_) {
    other.descriptor_ = -1;
  }
  FileDescriptorHolder& operator=(FileDescriptorHolder&& other) {
    if (this != &other) {
      reset();
      descriptor_ = other.descriptor_;
      other.descriptor_ = -1;
    }
    return *this;
  }
  ~FileDescriptorHolder() { reset(); }

  void reset(int descriptor = -1) {
    if (descriptor_ >= 0) {
      if (::close(descriptor_) != 0) {
        try {
          RDC_LOG(RDC_ERROR, "Fail to close the kernel file, errno: " << errno);
        } catch (...) {
        }
      }
    }
    descriptor_ = descriptor;
  }

  int get() const { return descriptor_; }
  bool empty() const { return descriptor_ < 0; }

 private:
  int descriptor_ = -1;
};

// The reference to ROCr that a check holds while it looks at its GPUs, so that ROCr is loaded
// once for the check rather than once per test object, and so that nothing it leaves behind can
// take ROCr's count to zero in the middle of a run. ROCr counts these references, so another
// module of the process that holds one, as the profiler module does from its first profiler
// field, keeps ROCr loaded across checks.
class RocrReference {
 public:
  RocrReference() : status_(hsa_init()) {}
  RocrReference(const RocrReference&) = delete;
  RocrReference& operator=(const RocrReference&) = delete;
  ~RocrReference() {
    if (status_ != HSA_STATUS_SUCCESS) {
      // No reference was taken: an unpaired release would end the references other modules hold.
      return;
    }
    const hsa_status_t err = hsa_shut_down();
    if (err != HSA_STATUS_SUCCESS) {
      log_release_failure("hsa_shut_down", err);
    }
  }

  // What hsa_init() answered: the rest of the run is this GPU's failure when it failed.
  hsa_status_t status() const { return status_; }

 private:
  hsa_status_t status_ = HSA_STATUS_ERROR;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_ROCRHANDLES_H_
