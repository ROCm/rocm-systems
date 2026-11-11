
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

struct ipcHdl {
#if HT_WIN
    HANDLE mailslot;
#else
    int socket = -1;
#endif
    std::string name;
};

// Creates a UNIX domain socket (Linux) or a mailslot (Windows) and allows to receive or send
// handles through it. Users of this class must call initialize() and check its result
// The static method generateIPCPrefixName() generates a unique name and can optionally be used to
// name the IPC objects, to be called from the parent before spawning children, to make sure all
// participants use the same prefix. The PID will be appended to the prefix so parent and children
// will actually have different IPC object names
// Note that direction of communication is always parent -> child
template <class ShareableHandleT = int>
class ipcSocketCom {
  static constexpr int kPidBufferSize = 16;
  static constexpr int kNonceDigitsSize = 6;
  // the maximum size of the user provided part of the name for the UNIX domain socket/Windows
  // mailslot; basically a result of the size of sockaddr_un::sun_path, minus 16 bytes for a PID,
  // 6 bytes for a random nonce, the initial '\0' and the dot '.' symbol
  static constexpr int kSocketUserPrefixMaxSize = 108 -
                                              kNonceDigitsSize -
                                              kPidBufferSize -
                                              2;
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

    close(handle->socket);
    handle.reset();
    return 0;
#endif
  }

public:
  ipcSocketCom() = default;

  // @isServer   Only makes a difference on Windows; if false, a mailslot is created. If true,
  //             the mailslot is not created
  ipcSocketCom(bool isServer, const std::string& prefix) :
    isServer_(isServer),
    serverFd_(-1),
    prefix_(prefix)
  {
  }

  // @return true if the initialization succeeded or false otherwise (e.g. the IPC objects could
  //         not be created)
  [[nodiscard]] bool initialize()
  {
#if !HT_WIN
    // maximum name size without the PID (which is added at bind() time)
    int maxNameSize = kSocketUserPrefixMaxSize + kNonceDigitsSize + 2;
    if (prefix_.size() > maxNameSize) {
      UNSCOPED_INFO("IPC object name prefix is too long: " <<
                    prefix_.size() <<
                    " maximum is " <<
                    maxNameSize);
      return false;
    }
#endif

    handle.reset(new ipcHdl());

#if HT_WIN
    handle->mailslot = INVALID_HANDLE_VALUE;

    if (!isServer_) {
      std::string name = prefix_ +
                         std::to_string((unsigned long)GetCurrentProcessId());

      handle->mailslot = CreateMailslot(name.c_str(), 0, MAILSLOT_WAIT_FOREVER, NULL);
      if (handle->mailslot == INVALID_HANDLE_VALUE) {
        UNSCOPED_INFO("CreateMailslot failed: " << GetLastError());
        handle.reset();
        return false;
      }
      handle->name = name;
    }
    return true;
#else
    serverFd_ = createAbstractSocket();

    if (serverFd_ == -1) {
      UNSCOPED_INFO("Failure at "
                    << (isServer_? "creating" : "opening")
                    << " socket: "
                    << strerror(errno));
      return false;
    }

    if (bindSocket() == -1) {
      return false;
    }
    return true;
#endif
  }
  ~ipcSocketCom() {
    closeThisSock();
  }
  int closeThisSock() {
    return closeSocket();
  }

  int recvShareableHdl(ShareableHandleT* shHandle) {
#if HT_WIN
    DWORD cbRead = 0;
    if (!ReadFile(handle->mailslot, shHandle, sizeof(*shHandle), &cbRead, NULL)) {
      UNSCOPED_INFO("ReadFile failed (" << GetLastError() << ")\n");
      return -1;
    }
    if (cbRead != sizeof(*shHandle)) {
      UNSCOPED_INFO("ReadFile returned unexpected size (" << cbRead << " expected " <<
                    sizeof(*shHandle) << ")");
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
      UNSCOPED_INFO("Socket failure: Receiving data over socket failed: " << strerror(errno));
      return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
       (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        return -1;
      }

      std::memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
      *reinterpret_cast<int*>(shHandle) = receivedfd;
    } else {
      return -1;
    }

    return 0;
#endif
  }

  int sendShareableHdl(ShareableHandleT shareableHdl, Process process) {
#if HT_WIN
    HANDLE hProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, process.dwProcessId);
    if (hProcess == NULL) {
      UNSCOPED_INFO("OpenProcess failed (" << GetLastError() << ")");
      return -1;
    }
    HANDLE hDup = INVALID_HANDLE_VALUE;
    if (!DuplicateHandle(GetCurrentProcess(), shareableHdl, hProcess,
                         &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      CloseHandle(hProcess);
      UNSCOPED_INFO("DuplicateHandle failed (" << GetLastError() << ")");
      return -1;
    }
    CloseHandle(hProcess);

    std::string slotName = prefix_ +
                           std::to_string(static_cast<unsigned long>(process.dwProcessId));
    HANDLE hFile = CreateFile(slotName.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
      UNSCOPED_INFO("CreateFile for mailslot failed (" << GetLastError() << ")");
      return -1;
    }
    DWORD cbWritten;
    if (!WriteFile(hFile, &hDup, sizeof(hDup), &cbWritten, NULL)) {
      CloseHandle(hFile);
      UNSCOPED_INFO("WriteFile failed (" << GetLastError() << ")");
      return -1;
    }
    CloseHandle(hFile);
    return 0;
#else
    struct msghdr msg = {};
    struct iovec iov[1];
    int dummy_data = 0;
    std::string name = prefix_ + std::to_string(process);

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un = {};

    struct cmsghdr *cmptr;
    struct sockaddr_un cliaddr = {};

    cliaddr.sun_family = AF_UNIX;
    std::memcpy(cliaddr.sun_path, name.data(), name.size());

    int sendfd = (int)shareableHdl;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));
    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = offsetof(struct sockaddr_un, sun_path) + name.size();
    iov[0].iov_base = &dummy_data;
    iov[0].iov_len = sizeof(dummy_data);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
    if (sendResult <= 0) {
      UNSCOPED_INFO("Socket failure: Sending data over socket failed: " << strerror(errno));
      return -1;
    }
    return 0;
#endif
  }

  // @userPrefix a user-provided string that will be prepended to the name; must be less than
  //             kSocketUserPrefixMaxSize characters; otherwise it will be truncated before used.
  //             e.g. a test name
  // @return     UNIX: an abstract name containing \0, the user prefix, a 6 hex-digit random
  //             number and '.'
  //             Windows: a mailsot name the user prefix and a 6 hex-digit random number and '.'
  static std::string generateIPCObjPrefix(const std::string& userPrefix)
  {
    using namespace std::string_literals;
    std::string commonSuffix = userPrefix.substr(0, kSocketUserPrefixMaxSize) +
                               generateRandomHex() +
                               ".";
#if HT_WIN
    return "\\\\.\\mailslot\\hipMemPoolIPC_"s + commonSuffix;
#else
    return "\0"s + commonSuffix;
#endif
  }

private:

  static std::string generateRandomHex(size_t length = kNonceDigitsSize)
  {
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    std::stringstream ss;

    while (ss.str().length() < length) {
      ss << std::hex << std::setfill('0') << std::setw(16) << dist(rng);
    }

    return ss.str().substr(0, length);
  }

#if !HT_WIN
  // @return the file descriptor of the newly created socket or -1 if there is a failure
  int createAbstractSocket()
  {
    if ((serverFd_ = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      UNSCOPED_INFO("Socket failure: Socket creation failed: " << strerror(errno));
      return -1;
    }

    return handle->socket = serverFd_;
  }

  // @return bind() return code. Note that if errno points to EADDRINUSE, this means there was a
  //         name collision, but the bind can be retried, as names contain a random part and
  //         subsequent binds might succeed
  int bindSocket() const
  {
    sockaddr_un servaddr = {};
    socklen_t addr_len;
    int code;
    std::string name = prefix_ + std::to_string(getpid());

    std::memcpy(servaddr.sun_path, name.data(), name.size());
    addr_len = offsetof(struct sockaddr_un, sun_path) + name.size();
    servaddr.sun_family = AF_UNIX;

    UNSCOPED_INFO("Binding IPC socket with name: " << name.substr(0, name.size()));
    if ((code = bind(serverFd_, reinterpret_cast<sockaddr *>(&servaddr), addr_len)) < 0) {
      UNSCOPED_INFO("bind failed: " << strerror(errno) << " when binding with name: " << name);
      return -1;
    } else {
      handle->name = name;
    }

    return code;
  }
#endif

  bool isServer_;
  int serverFd_;
  std::string prefix_;
};
