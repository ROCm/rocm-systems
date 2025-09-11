/*
Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#endif
#include "hip_test_context.hh"

#define checkVMMSupported(device)                                                                  \
  {                                                                                                \
    int value = 0;                                                                                 \
    hipDeviceAttribute_t attr = hipDeviceAttributeVirtualMemoryManagementSupported;                \
    HIP_CHECK(hipDeviceGetAttribute(&value, attr, device));                                        \
    if (value == 0) {                                                                              \
      HipTest::HIP_SKIP_TEST("Machine does not support VMM. Skipping Test..");                     \
      return;                                                                                      \
    }                                                                                              \
  }

#ifdef __linux__
#define checkSysCallErrors(result)                                                                 \
  if (result == -1) {                                                                              \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(EXIT_FAILURE);                 \
  }

typedef pid_t Process;
typedef int hipShareableHdl;
struct ipcHdl {
    int socket;
    char *name;
};

class ipcSocketCom {
  ipcHdl *handle;
  // method to create socket from server
  int createSocket() {
    int server_fd;
    struct sockaddr_un servaddr;

    char name[16];
    // Create a unique socket name based on current pid
    sprintf(name, "%u", getpid());

    // Create the socket handle
    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }

    memset(handle, 0, sizeof(*handle));
    handle->socket = -1;
    handle->name = NULL;

    // Creating socket
    if ((server_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == 0) {
      perror("Socket failure: Socket creation failed");
      return -1;
    }

    unlink(name);
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_UNIX;

    size_t len = strlen(name);
    if (len > (sizeof(servaddr.sun_path) - 1)) {
      perror("Socket failure: Cannot bind provided name to socket. Name too large");
      return -1;
    }

    strncpy(servaddr.sun_path, name, len);

    if (bind(server_fd, (struct sockaddr *)&servaddr, SUN_LEN(&servaddr)) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->name = new char[strlen(name) + 1];
    strcpy(handle->name, name);
    handle->socket = server_fd;
    return 0;
  }
  // method to create socket from client
  int openSocket() {
    int sock = 0;
    struct sockaddr_un cliaddr;

    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }
    memset(handle, 0, sizeof(*handle));

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      perror("IPC failure:Socket creation error");
      return -1;
    }

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char name[16];

    // Create a unique socket name based on current process id.
    sprintf(name, "%u", getpid());

    strcpy(cliaddr.sun_path, name);
    if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->socket = sock;
    handle->name = new char[strlen(name) + 1];
    strcpy(handle->name, name);

    return 0;
  }
  // method to close socket
  int closeSocket() {
    if (!handle) {
      return -1;
    }

    if (handle->name) {
      unlink(handle->name);
      delete[] handle->name;
    }
    close(handle->socket);
    delete handle;
    return 0;
  }
public:
  ipcSocketCom() = default;
  ipcSocketCom(bool isServer) {
    if (isServer) {
      checkSysCallErrors(createSocket());
    } else {
      checkSysCallErrors(openSocket());
    }
  }
  ~ipcSocketCom() {
  }
  int closeThisSock() {
    return closeSocket();
  }
  // method to receive shareable handle via socket
  int recvShareableHdl(hipShareableHdl *shHandle) {
    int dummy_data;
    struct msghdr msg;
    struct iovec iov[1];

    // Union to guarantee alignment requirements for control array
    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);
    iov[0].iov_base = &dummy_data;
    iov[0].iov_len = sizeof(dummy_data);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    if ((n = recvmsg(handle->socket, &msg, 0)) <= 0) {
      perror("Socket failure: Receiving data over socket failed");
      return -1;
    }
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
       (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        return -1;
      }
      memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
      *(int *)shHandle = receivedfd;
    } else {
      return -1;
    }

    return 0;
  }
  // method to send shareable handle via sockets
  int sendShareableHdl(hipShareableHdl shareableHdl, Process process) {
    struct msghdr msg;
    struct iovec iov[1];
    int dummy_data = 0;
    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    struct sockaddr_un cliaddr;

    // Construct client address to send this SHareable handle to
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    strcpy(cliaddr.sun_path, std::to_string(process).c_str());

    // Send corresponding shareable handle to the client
    int sendfd = (int)shareableHdl;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = &dummy_data;
    iov[0].iov_len = sizeof(dummy_data);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
    if (sendResult <= 0) {
      perror("Socket failure: Sending data over socket failed");
      return -1;
    }

    return 0;
  }
};
#endif
constexpr int threadsPerBlk = 64;
