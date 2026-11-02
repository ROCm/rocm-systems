
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
    int socket = -1;
#endif
    std::string name;
};

//users of this 
class ipcSocketCom {
  // the maximum size of the user provided part of the name for the UNIX domain socket/Windows
  // mailslot; basically a result of the size of sockaddr_un::sun_path
  static constexpr int kSocketPrefixMaxSize = sizeof(sockaddr_un::sun_path) - 32 - 1;
  std::unique_ptr<ipcHdl> handle;

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

  // @isServer   Only makes a difference on Windows; if false, a mailslot is created
  // @userPrefix an extra string that helps with avoiding clashes between ipc unix domain socket
  //             names. This could be a test name + section for instance. This will be part of the
  //             IPC object name in addition to the PID and a random nonce. Note: on Linux names
  //             that are too long will be truncated so the whole identifier fits into the
  //             sockaddr_un::sun_path array
  ipcSocketCom(bool isServer, const std::string& userPrefix) :
    isServer_(isServer),
    userPrefix_(userPrefix)
  }

  // @return true if the initialization failed (e.g. the IPC objects could not be created)
  [[nodiscard]] bool initialize()
  {
    handle.reset(new ipcHdl());
#if HT_WIN
    handle.reset(new ipcHdl());
    handle->mailslot = INVALID_HANDLE_VALUE;
    return;
#else
    int result;

    if (isServer) {
      result = createSocket(userPrefix);
    } else {
      result = openSocket();
    }

    if (result == -1) {
      UNSCOPED_INFO("Failure at"s 
                    << (isServer? "creating"s : "opening"s)
                    << " socket: "
                    << strerror(errno));
      // TODO g-h-c how to return the error? this is a constructor
    }
#endif
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
      //fprintf(stderr, "ReadFile failed (%lu)\n", GetLastError());
      return -1;
    }
    if (cbRead != sizeof(*shHandle)) {
      //fprintf(stderr, "ReadFile returned unexpected size (%lu, expected %zu)\n",
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
      //fprintf(stderr, "OpenProcess failed (%lu)\n", GetLastError());
      return -1;
    }
    HANDLE hDup = INVALID_HANDLE_VALUE;
    if (!DuplicateHandle(GetCurrentProcess(), shareableHdl, hProcess,
                         &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      CloseHandle(hProcess);
      //fprintf(stderr, "DuplicateHandle failed (%lu)\n", GetLastError());
      return -1;
    }
    CloseHandle(hProcess);

    char slotName[128];
    sprintf(slotName, "\\\\.\\mailslot\\hipMemPoolIPC_%lu",
            (unsigned long)process.dwProcessId);
    HANDLE hFile = CreateFile(slotName, GENERIC_WRITE, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
      //fprintf(stderr, "CreateFile for mailslot failed (%lu)\n", GetLastError());
      return -1;
    }
    DWORD cbWritten;
    if (!WriteFile(hFile, &hDup, sizeof(hDup), &cbWritten, NULL)) {
      CloseHandle(hFile);
      //fprintf(stderr, "WriteFile failed (%lu)\n", GetLastError());
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
  }
#endif

private:

  static std::string generateRandomHex(size_t length = 6)
  {
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    std::stringstream ss;

    while (ss.str().length() < length) {
      ss << std::hex << std::setfill('0') << std::setw(16) << dist(rng);
    }

    return ss.str().substr(0, length);
  }

  // @userPrefix a user-provided string that will be prepended to the name; must be less than 
  //             kSocketPrefixMaxSize characters; otherwise it will be truncated before used
  // @return     an abstract name containing \0, the user prefix and a 6 hex-digit random name
  static std::string generateIPCObjName(const std::string& userPrefix)
  {
    using namespace std::string_literals;

    return "\0"s +
           userPrefix.substr(0, kSocketPrefixMaxSize) +
           getpid() +
           generateRandomHex();
  }

  // @return the file descriptor of the newly created socket or -1 if there is a failure
  int createAbstractSocket()
  {
    // Address length is sizeof(sun_family) + 1 (null byte) + string length
    int serverFd;

    if ((serverFd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      perror("Socket failure: Socket creation failed");
      return -1;
    } else {
      handle->socket = serverFd;
    }

    return serverFd;
  }

  // @return bind() return code. Note that if errno points to EADDRINUSE, this means there was a
  //         name collision, but the bind can be retried, as names contain a random part and
  //         subsequent binds might succeed
  int bindSocket(int serverFd, const std::string& userPrefix) const
  {
    sockaddr_un servaddr = {};
    socklen_t addr_len;
    std::string name = generateIPCObjName();
    int code;

    strncpy(servaddr.sun_path, name.data(), name.size());
    addr_len = offsetof(struct sockaddr_un, sun_path) + 1 + len;
    servaddr.sun_family = AF_UNIX;

    if ((code = bind(serverFd, (struct sockaddr *)&servaddr, addr_len)) < 0) {
      UNSCOPED_INFO("bind failed: " << strerror(errno));
      return -1;
    } else {
      handle->name = name;
    }

    return code;
  }
  
  bool isServer_;
  std::string userPrefix_;
};
