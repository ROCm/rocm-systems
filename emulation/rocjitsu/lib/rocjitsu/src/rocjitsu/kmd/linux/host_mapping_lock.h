// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_HOST_MAPPING_LOCK_H_
#define ROCJITSU_KMD_LINUX_HOST_MAPPING_LOCK_H_

/// @file host_mapping_lock.h
/// @brief Serializes host mapping changes against accesses that hold a pointer.

#include "util/observable_shared_mutex.h"
#include <cassert>
#include <cstdint>
#include <new>
#include <optional>

namespace rocjitsu {

/// @brief Guards the process's mapping layout for the length of an access.
///
/// @details Almost every emulated access asks the kernel to move the bytes, so
/// the mapping cannot change underneath it: the syscall either works or
/// reports. An atomic cannot be expressed that way without ceasing to be
/// atomic, so it establishes that a page is writable and then modifies it in
/// place -- and between those two steps the application could unmap the page or
/// drop its write permission, which is a host fault or a write into whatever
/// replaced it.
///
/// Holding this shared for the whole check-and-modify, and exclusive around
/// every mapping call the interposer sees or the driver makes for itself,
/// removes that window. It is deliberately taken around the bare syscall and
/// nothing else: the driver's own mapping work re-enters the memory model and
/// takes its VMID lock, so widening this to cover that would invert the two
/// orders and deadlock.
///
/// Exactly one of these exists per process, because what it guards -- this
/// process's mapping layout -- is per-process. Hanging it off a GpuMemory or a
/// driver would give each instance its own lock and silently serialize nothing,
/// and the interposer's mmap/munmap/mprotect hooks are free functions with no
/// object to reach for in the first place.
///
/// Inline rather than defined in a translation unit of its own: gpu_memory.h
/// takes this lock and is pulled in by the ISA execution libraries, so an
/// out-of-line definition would make every target that merely decodes
/// instructions link the KMD layer. A static local in an inline function is
/// still one object across the program, so the single instance holds.
///
/// Mappings changed by a raw syscall rather than through libc are not seen by
/// the interposer and so are not covered. Nothing in the emulated stack does
/// that, and an application that does is mutating memory its own GPU is using.
inline util::ObservableSharedMutex &host_mapping_lock() {
  static util::ObservableSharedMutex lock;
  return lock;
}

/// @brief Whether this thread already holds the mapping layout stable for a
/// batch of emulated GPU accesses.
inline thread_local uint32_t host_mapping_batch_depth = 0;

/// @brief Hold the host mapping layout stable across a functional CU quantum.
///
/// @details The interposer takes the exclusive side around mmap, munmap, and
/// mprotect. Retaining the shared side across a bounded quantum therefore
/// preserves the existing pointer-lifetime guarantee while amortizing reader
/// admission over the many small GPU accesses executed in that quantum.
class HostMappingBatchGuard {
public:
  HostMappingBatchGuard() {
    if (host_mapping_batch_depth == 0)
      lock_.emplace(host_mapping_lock().lock_shared());
    ++host_mapping_batch_depth;
  }

  ~HostMappingBatchGuard() {
    assert(host_mapping_batch_depth != 0);
    --host_mapping_batch_depth;
  }

  HostMappingBatchGuard(const HostMappingBatchGuard &) = delete;
  HostMappingBatchGuard &operator=(const HostMappingBatchGuard &) = delete;

private:
  // The destructor body lowers the depth before member destruction releases
  // the shared lock, so nested per-access guards cannot observe a false gap.
  std::optional<std::shared_lock<util::DistributedSharedMutex>> lock_;
};

/// @brief Shared mapping guard for one access, unless its thread is in a batch.
class HostMappingAccessGuard {
public:
  HostMappingAccessGuard() {
    if (host_mapping_batch_depth == 0)
      lock_.emplace(host_mapping_lock().lock_shared());
  }

  HostMappingAccessGuard(const HostMappingAccessGuard &) = delete;
  HostMappingAccessGuard &operator=(const HostMappingAccessGuard &) = delete;

  void unlock() {
    if (lock_)
      lock_->unlock();
  }

private:
  std::optional<std::shared_lock<util::DistributedSharedMutex>> lock_;
};

[[nodiscard]] inline HostMappingAccessGuard lock_host_mapping_for_access() {
  return HostMappingAccessGuard();
}

/// @brief Reset the mapping lock in a fork child that inherited no GPU backend.
/// @pre Called by the child atfork handler, before publishing its fresh context.
/// No other thread survives fork, and the inherited lock must not be destroyed
/// or acquired: a vanished parent thread may have held it.
inline void reset_host_mapping_lock_after_fork() {
  host_mapping_batch_depth = 0;
  new (&host_mapping_lock()) util::ObservableSharedMutex();
}

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_HOST_MAPPING_LOCK_H_
