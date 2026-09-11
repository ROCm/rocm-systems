// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_HOST_MAPPING_LOCK_H_
#define ROCJITSU_KMD_LINUX_HOST_MAPPING_LOCK_H_

/// @file host_mapping_lock.h
/// @brief Serializes host mapping changes against accesses that hold a pointer.

#include "util/observable_shared_mutex.h"

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

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_HOST_MAPPING_LOCK_H_
