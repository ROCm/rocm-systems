// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The KFD_EVENT_FMT_* payload layouts from the Linux kernel's include/uapi/linux/kfd_ioctl.h
// (Copyright 2014 Advanced Micro Devices, Inc., MIT license). Compared against upstream master by
// GpuUnit.KFD_EVENT_FMT__MatchesMainlineKernelHeader; refresh the block from there when it fails.

#ifndef TESTS_AMD_SMI_TEST_UNIT_GPU_KFD_EVENT_FORMATS_H_
#define TESTS_AMD_SMI_TEST_UNIT_GPU_KFD_EVENT_FORMATS_H_

#define KFD_EVENT_FMT_UPDATE_GPU_RESET(reset_seq_num, reset_cause) \
  "%x %s\n", (reset_seq_num), (reset_cause)

#define KFD_EVENT_FMT_THERMAL_THROTTLING(bitmask, counter) "%llx:%llx\n", (bitmask), (counter)

#define KFD_EVENT_FMT_VMFAULT(pid, task_name) "%x:%s\n", (pid), (task_name)

#define KFD_EVENT_FMT_PAGEFAULT_START(ns, pid, addr, node, rw) \
  "%lld -%d @%lx(%x) %c\n", (ns), (pid), (addr), (node), (rw)

#define KFD_EVENT_FMT_PAGEFAULT_END(ns, pid, addr, node, migrate_update) \
  "%lld -%d @%lx(%x) %c\n", (ns), (pid), (addr), (node), (migrate_update)

#define KFD_EVENT_FMT_MIGRATE_START(ns, pid, start, size, from, to, prefetch_loc, preferred_loc, \
                                    migrate_trigger)                                             \
  "%lld -%d @%lx(%lx) %x->%x %x:%x %d\n", (ns), (pid), (start), (size), (from), (to),            \
      (prefetch_loc), (preferred_loc), (migrate_trigger)

#define KFD_EVENT_FMT_MIGRATE_END(ns, pid, start, size, from, to, migrate_trigger, error_code) \
  "%lld -%d @%lx(%lx) %x->%x %d %d\n", (ns), (pid), (start), (size), (from), (to),             \
      (migrate_trigger), (error_code)

#define KFD_EVENT_FMT_QUEUE_EVICTION(ns, pid, node, evict_trigger) \
  "%lld -%d %x %d\n", (ns), (pid), (node), (evict_trigger)

#define KFD_EVENT_FMT_QUEUE_RESTORE(ns, pid, node, rescheduled) \
  "%lld -%d %x %c\n", (ns), (pid), (node), (rescheduled)

#define KFD_EVENT_FMT_UNMAP_FROM_GPU(ns, pid, addr, size, node, unmap_trigger) \
  "%lld -%d @%lx(%lx) %x %d\n", (ns), (pid), (addr), (size), (node), (unmap_trigger)

#define KFD_EVENT_FMT_PROCESS(pid, task_name) "%x %s\n", (pid), (task_name)

#endif  // TESTS_AMD_SMI_TEST_UNIT_GPU_KFD_EVENT_FORMATS_H_
