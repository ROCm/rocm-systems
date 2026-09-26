/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hip_test_context.hh"
#include "hip_test_filesystem.hh"

#include <string>

#if !HT_WIN
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>

// Naming for the AF_UNIX rendezvous used by the mempool and VMM IPC tests.
// Shared by both so the two implementations cannot drift apart.
//
// Both ends of the rendezvous derive the name independently instead of
// exchanging it, the same way the shared-memory segment in
// hipMemPoolExportToShareableHandle.cc derives the parent identity with
// getParentProcessId(). That property is what makes this survive
// SpawnProc, which fork()s and then execvp()s: the child re-runs this code
// in a fresh process image and computes the same answer. A randomly
// generated per-run token would not - the child would compute a different
// one and bind a name the parent never addresses.
namespace hip_ipc {

// Identifies the mount namespace, i.e. "which temp directory is this".
// Peers that share a temp directory compute the same tag; peers that do not
// (separate containers) compute different ones, so identical pids in
// separate pid namespaces cannot collide on a shared temp mount.
//
// The kernel reuses namespace inode numbers once a namespace is destroyed, so
// this is unique among *concurrent* peers rather than over all time. That is
// the property needed here: a socket belonging to an exited peer is already
// gone, and createSocket() unlinks any leftover before binding.
inline std::string NamespaceTag() {
  unsigned long long id = 0;
  struct stat st;
  if (stat("/proc/self/ns/mnt", &st) == 0) {
    id = static_cast<unsigned long long>(st.st_ino);
  } else {
    // No procfs (for example WSL1, where these tests also run). Fall back to
    // the UTS name, which containers default to their own id.
    char host[65] = {};
    if (gethostname(host, sizeof(host) - 1) == 0) {
      for (const char* p = host; *p != '\0'; ++p) {
        id = id * 131 + static_cast<unsigned char>(*p);
      }
    }
  }
  char buf[9];
  snprintf(buf, sizeof(buf), "%08llx", id & 0xffffffffULL);
  return std::string(buf);
}

// Socket path for the peer owning `pid`. Lives in the temp directory because
// installed test directories may be read-only. The sender names its peer from
// the peer's pid, which it already knows from fork()/spawn(), so no handshake
// is needed.
inline std::string SocketPath(pid_t pid) {
  static const std::string tag = NamespaceTag();
  return (fs::temp_directory_path() / ("hip_test_ipc_" + tag + "_" + std::to_string(pid)))
      .string();
}

}  // namespace hip_ipc
#endif  // !HT_WIN
