/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <sys/wait.h>
#include<proxy.h> 
#include <comm.h>
#include <ipcsocket.h>

#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(Ipcsocket, SocketInitNullHandle){
    int rank = 0;
    uint64_t hash = 0x1234;
    volatile uint32_t abortFlag = 0;
    // Call function with NULL handle
    ncclResult_t result = ncclIpcSocketInit(NULL, rank, hash, &abortFlag);
    // Check that it returns the correct error code
    ASSERT_EQ(result, ncclInternalError);
  }

  TEST(Ipcsocket, SocketGetFdNullHandle){
    int fd = -1;
    ncclResult_t result = ncclIpcSocketGetFd(nullptr, &fd);
    EXPECT_EQ(result, ncclInvalidArgument);
  }

  TEST(Ipcsocket, SocketCloseNullHandle){
    ncclResult_t result = ncclIpcSocketClose(nullptr);
    EXPECT_EQ(result, ncclInternalError);
  }

  TEST(Ipcsocket, SocketCloseNegativeHandle){
    ncclIpcSocket handle = {};
    handle.fd = -1;
    ncclResult_t result = ncclIpcSocketClose(&handle);
    EXPECT_EQ(result, ncclSuccess);
  }

  TEST(Ipcsocket, SendAndReceiveFd) {
    int pipeFd[2]; // for sync from child -> parent
    ASSERT_EQ(pipe(pipeFd), 0);
  
    pid_t pid = fork();
    ASSERT_NE(pid, -1);
  
    const int rank = 1;
    const uint64_t hash = 0x12345678;
    volatile uint32_t abortFlag = 0;
  
    if (pid == 0) {
      // === Child: Receiver ===
      close(pipeFd[0]);
  
      char sockPath[108];
      snprintf(sockPath, sizeof(sockPath), "/tmp/ipc_sock_%lx", hash);
      unlink(sockPath);
      
      int listenFd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
      ASSERT_GT(listenFd, 0);
      
      struct sockaddr_un addr = {};
      addr.sun_family = AF_UNIX;
      strncpy(addr.sun_path, sockPath, sizeof(addr.sun_path) - 1);
      
      ASSERT_EQ(bind(listenFd, (struct sockaddr*)&addr, sizeof(addr)), 0);
      
      ASSERT_EQ(listen(listenFd, 1), 0);
      // Signal parent we're ready to accept
      ASSERT_EQ(write(pipeFd[1], "r", 1), 1);
    
      close(pipeFd[1]);
  
      int connFd = accept(listenFd, NULL, NULL);
      if (connFd < 0) { perror("accept"); exit(4); }
      ASSERT_GT(connFd, 0);

      ncclIpcSocket handle = {
        .fd = connFd,
        .abortFlag = &abortFlag,
      };
      strncpy(handle.socketName, sockPath, sizeof(handle.socketName));
  
      int recvFd = -1;
      ASSERT_EQ(ncclIpcSocketRecvFd(&handle, &recvFd), ncclSuccess);
      ASSERT_GE(recvFd, 0);
  
      // Optionally verify FD
      struct stat st;
      ASSERT_EQ(fstat(recvFd, &st), 0);
      
      close(recvFd);
      
      // Send a new FD back to parent
      int fdToSend = open("/dev/null", O_RDONLY);
      ASSERT_GE(fdToSend, 0);
      ASSERT_EQ(ncclIpcSocketSendFd(&handle, fdToSend, rank, hash), ncclSuccess);
      close(fdToSend);

      close(connFd);
      close(listenFd);
      unlink(sockPath);
      
      _exit(0);
    } else {
      // === Parent: Sender ===
      close(pipeFd[1]);
  
      char tmp;
      ASSERT_EQ(read(pipeFd[0], &tmp, 1), 1); // wait for child to listen
      close(pipeFd[0]);
  
      char sockPath[108];
      snprintf(sockPath, sizeof(sockPath), "/tmp/ipc_sock_%lx", hash);
  
      int sockFd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
      ASSERT_GT(sockFd, 0);
  
      struct sockaddr_un addr = {};
      addr.sun_family = AF_UNIX;
      strncpy(addr.sun_path, sockPath, sizeof(addr.sun_path) - 1);
  
      ASSERT_EQ(connect(sockFd, (struct sockaddr*)&addr, sizeof(addr)), 0);
  
      ncclIpcSocket handle = {
        .fd = sockFd,
        .abortFlag = &abortFlag,
      };
      strncpy(handle.socketName, sockPath, sizeof(handle.socketName));
  
      int fdToSend = open("/dev/null", O_RDONLY);
      ASSERT_GE(fdToSend, 0);
      ASSERT_EQ(ncclIpcSocketSendFd(&handle, fdToSend, rank, hash), ncclSuccess);
      close(fdToSend);
      
      // Receive FD from child
      int recvBackFd = -1;
      ASSERT_EQ(ncclIpcSocketRecvFd(&handle, &recvBackFd), ncclSuccess);
      ASSERT_GE(recvBackFd, 0);

      struct stat st;
      ASSERT_EQ(fstat(recvBackFd, &st), 0);
      close(recvBackFd);

      close(sockFd);
  
      int status = 0;
      waitpid(pid, &status, 0);
      EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
  }
}
