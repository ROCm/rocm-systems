// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <gtest/gtest.h>
#include <iterator>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>

namespace rocjitsu::test {

// Apply one errno to the calling thread and children it subsequently creates.
inline void set_syscall_error(unsigned number, unsigned error) {
  sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, number, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | error),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program{static_cast<unsigned short>(std::size(filter)), filter};
  ASSERT_EQ(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
  ASSERT_EQ(prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program), 0);
}

} // namespace rocjitsu::test
