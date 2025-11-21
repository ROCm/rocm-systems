/*
 * Copyright (C) Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#ifdef __linux__
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <dlfcn.h>

bool testMallocFromChild() {
  int fd[2];
  pid_t childpid;
  bool testResult = false;

  // create pipe descriptors
  pipe(fd);

  childpid = fork();
  if (childpid > 0) {  // Parent
    close(fd[1]);
    // parent will wait to read the device cnt
    read(fd[0], &testResult, sizeof(testResult));

    // close the read-descriptor
    close(fd[0]);

    // wait for child exit
    wait(NULL);

    return testResult;

  } else if (!childpid) {  // Child
    // writing only, no need for read-descriptor
    close(fd[0]);

    char* A_d = nullptr;
    hipError_t ret = hipMalloc(&A_d, 1024);

    printf("hipMalloc returned : %s\n", hipGetErrorString(ret));
    if (ret == hipSuccess)
      testResult = true;
    else
      testResult = false;

    // send the value on the write-descriptor:
    write(fd[1], &testResult, sizeof(testResult));

    // close the write descriptor:
    close(fd[1]);
    exit(0);
  }
  return false;
}


TEST_CASE("ChildMalloc") {
  auto res = testMallocFromChild();
  REQUIRE(res == true);
}

#endif
