
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include "hip_test_context.hh"
#include "hip_test_process.hh"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#if HT_WIN
typedef HANDLE hipShareableHdl;
#else
#ifdef HT_AMD
typedef int64_t hipShareableHdl;
#else
typedef int hipShareableHdl;
#endif
#endif

struct ipcHdl {
#if HT_WIN
    HANDLE mailslot;
#else
    int socket = 0;
#endif
    std::unique_ptr<char[]> name;
};

class ipcSocketCom {
  std::unique_ptr<ipcHdl> handle;

  int createSocket() {
#if HT_WIN
    handle.reset(new (std::nothrow) ipcHdl());
    if (!handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }
    handle->mailslot = INVALID_HANDLE_VALUE;
    return 0;
#else
    int server_fd;
    struct sockaddr_un servaddr;

    char name[16];
    socklen_t addr_len;

    sprintf(name, "%u", getpid());

    handle.reset(new (std::nothrow) ipcHdl());
    if (!handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }

    handle->socket = -1;

    if ((server_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      perror("Socket failure: Socket creation failed");
      return -1;
    }

    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_UNIX;

    size_t len = strlen(name);
    if (len > (sizeof(servaddr.sun_path) - 2)) {
      perror("Socket failure: Cannot bind provided name to socket. Name too large");
      return -1;
    }

    servaddr.sun_path[0] = '\0';
    strncpy(&servaddr.sun_path[1], name, len);
    addr_len = offsetof(struct sockaddr_un, sun_path) + 1 + len;

    if (bind(server_fd, (struct sockaddr *)&servaddr, addr_len) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->name.reset(new char[strlen(name) + 1]);
    strcpy(handle->name.get(), name);
    handle->socket = server_fd;
    return 0;
#endif
  }

  int openSocket() {
#if HT_WIN
    handle.reset(new (std::nothrow) ipcHdl());

    if (!handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }

    handle->mailslot = INVALID_HANDLE_VALUE;
    char name[128];
    sprintf(name, "\\\\.\\mailslot\\hipMemPoolIPC_%lu",
            (unsigned long)GetCurrentProcessId());
    handle->mailslot = CreateMailslot(name, 0, MAILSLOT_WAIT_FOREVER, NULL);
    if (handle->mailslot == INVALID_HANDLE_VALUE) {
      fprintf(stderr, "CreateMailslot failed (%lu)\n", GetLastError());
      handle.reset();
      return -1;
    }
    handle->name.reset(new char[strlen(name) + 1]);
    strcpy(handle->name.get(), name);
    return 0;
#else
    int sock = 0;
    struct sockaddr_un cliaddr;

    handle.reset(new (std::nothrow) ipcHdl());

    if (!handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      perror("IPC failure:Socket creation error");
      return -1;
    }

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char name[16];

    sprintf(name, "%u", getpid());

    strcpy(cliaddr.sun_path, name);
    if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->socket = sock;
    handle->name.reset(new char[strlen(name) + 1]);
    strcpy(handle->name.get(), name);

    return 0;
#endif
  }

  int closeSocket() {
#if HT_WIN
    if (!handle) return -1;
    if (handle->mailslot != INVALID_HANDLE_VALUE) {
      CloseHandle(handle->mailslot);
    }

    handle.reset();
    return 0;
#else
    if (!handle) {
      return -1;
    }

    if (handle->name) {

      if (handle->name[0] != '\0') {
        //abstract names do not need unlink()
        unlink(handle->name.get());
      }
    }
    close(handle->socket);
    handle.reset();
    return 0;
#endif
  }

public:
  ipcSocketCom() = default;
  ipcSocketCom(bool isServer) {
    int result;

    if (isServer) {
      result = createSocket();
    } else {
      result = openSocket();
    }

    if (result == -1) {
      fprintf(stderr,
              "Failure at %s socket: %s\n", isServer? "creating" : "opening",
              strerror(errno));
      exit(EXIT_FAILURE);
    }
  }
  ~ipcSocketCom() {
    closeThisSock();
  }
  int closeThisSock() {
    return closeSocket();
  }

  int recvShareableHdl(hipShareableHdl *shHandle) {
#if HT_WIN
    DWORD cbRead = 0;
    if (!ReadFile(handle->mailslot, shHandle, sizeof(*shHandle), &cbRead, NULL)) {
      fprintf(stderr, "ReadFile failed (%lu)\n", GetLastError());
      return -1;
    }
    if (cbRead != sizeof(*shHandle)) {
      fprintf(stderr, "ReadFile returned unexpected size (%lu, expected %zu)\n",
              cbRead, sizeof(*shHandle));
      return -1;
    }
    return 0;
#else
    struct msghdr msg = {};
    struct iovec iov[1];

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un = {};

    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    int dummy_data;

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
#endif
  }

  int sendShareableHdl(hipShareableHdl shareableHdl, Process process) {
#if HT_WIN
    HANDLE hProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, process.dwProcessId);
    if (hProcess == NULL) {
      fprintf(stderr, "OpenProcess failed (%lu)\n", GetLastError());
      return -1;
    }
    HANDLE hDup = INVALID_HANDLE_VALUE;
    if (!DuplicateHandle(GetCurrentProcess(), shareableHdl, hProcess,
                         &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      CloseHandle(hProcess);
      fprintf(stderr, "DuplicateHandle failed (%lu)\n", GetLastError());
      return -1;
    }
    CloseHandle(hProcess);

    char slotName[128];
    sprintf(slotName, "\\\\.\\mailslot\\hipMemPoolIPC_%lu",
            (unsigned long)process.dwProcessId);
    HANDLE hFile = CreateFile(slotName, GENERIC_WRITE, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
      fprintf(stderr, "CreateFile for mailslot failed (%lu)\n", GetLastError());
      return -1;
    }
    DWORD cbWritten;
    if (!WriteFile(hFile, &hDup, sizeof(hDup), &cbWritten, NULL)) {
      CloseHandle(hFile);
      fprintf(stderr, "WriteFile failed (%lu)\n", GetLastError());
      return -1;
    }
    CloseHandle(hFile);
    return 0;
#else
    struct msghdr msg = {};
    struct iovec iov[1];
    int dummy_data = 0;

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un = {};

    struct cmsghdr *cmptr;
    struct sockaddr_un cliaddr;

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    strcpy(cliaddr.sun_path, std::to_string(process).c_str());

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
#endif
  }
};
