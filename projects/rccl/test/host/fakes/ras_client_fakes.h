/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_RAS_CLIENT_FAKES_H_
#define RCCL_TEST_HOST_FAKES_RAS_CLIENT_FAKES_H_

// Controllable seams for the libc surface that src/ras/client.cc calls.
//
// client.cc is a standalone executable whose entire dependency surface is libc
// (sockets, getopt, stdio) plus four macros from ras_internal.h -- it needs no
// HIP runtime and no librccl. ras-client-test.cc #includes the hipified
// client.cc and macro-renames each libc call below to its micro_* trampoline,
// which dispatches through the std::function slot declared here. Install
// per-test behaviour with ScopedHook; ResetRasClientFakes() restores defaults.

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <getopt.h>

#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// Thrown by the default exit seam so a test can assert both the status and the
// state client.cc left behind, which a death test cannot observe. Death tests
// remain available: install a hook that calls ::exit / ::_exit instead.
struct RasClientExit {
  int status;
};

// One scripted result for the read seam. `ret` < 0 makes the read fail with
// `err` in errno; `ret` == 0 is EOF; otherwise `data` is copied to the caller
// (truncated to the caller's buffer) and its length is returned.
struct RasReadStep {
  ssize_t ret;
  int err;
  std::string data;
};

// ---------------------------------------------------------------------------
// Seams. Each defaults to the success behaviour described in the .cc.
// ---------------------------------------------------------------------------
extern std::function<ssize_t(int, const void*, size_t)> g_write;
extern std::function<ssize_t(int, void*, size_t)> g_read;
extern std::function<int(int)> g_close;
extern std::function<int(int, int, int)> g_socket;
extern std::function<int(int, const struct sockaddr*, socklen_t)> g_connect;
extern std::function<int(int, int, int, const void*, socklen_t)> g_setsockopt;
extern std::function<int(const char*, const char*, const struct addrinfo*, struct addrinfo**)> g_getaddrinfo;
extern std::function<void(struct addrinfo*)> g_freeaddrinfo;
extern std::function<int(const struct sockaddr*, socklen_t, char*, socklen_t, char*, socklen_t, int)> g_getnameinfo;
extern std::function<const char*(int)> g_gaiStrerror;
extern std::function<size_t(const void*, size_t, size_t, FILE*)> g_fwrite;
extern std::function<int(FILE*)> g_fflush;
extern std::function<void(int)> g_exit;

// ---------------------------------------------------------------------------
// Observation points fed by the default seams. A test that installs its own
// hook over a seam stops feeding the corresponding record.
// ---------------------------------------------------------------------------
extern std::string g_writtenData;      // every byte the client wrote to the socket
extern std::string g_stdoutData;       // every byte the client fwrite()'d
extern std::vector<int> g_closedFds;   // fds passed to close(), in order
extern std::vector<RasReadStep> g_readScript;  // consumed front-to-back by the default read
extern size_t g_readScriptPos;
extern int g_nextSocketFd;             // what the default socket() hands back (-1 to fail it)
extern int g_socketFailErrno;          // errno the default socket() sets when g_nextSocketFd is -1
extern int g_lastSetsockoptOptname;    // SO_SNDTIMEO / SO_RCVTIMEO of the last setsockopt
extern struct timeval g_lastSetsockoptTimeval;
extern int g_getaddrinfoResult;        // non-zero makes the default getaddrinfo fail with that code
extern int g_addrinfoCount;            // how many entries the default getaddrinfo returns
extern int g_freeaddrinfoCalls;
extern int g_connectResult;            // 0 succeeds; non-zero fails and sets errno to g_connectErrno
extern int g_connectErrno;

// Queues one scripted read result. Reads past the end of the script return 0 (EOF).
void ScriptRead(ssize_t ret, int err, std::string data);

// Convenience: script one successful read that delivers `data`.
void ScriptReadData(std::string data);

// Restores every seam to its default and clears every record above.
void ResetRasClientFakes();

// Resets the file-scope state of client.cc (hostName/port/timeout/verbose/
// monitorMode/format/events/sock) plus getopt's global parse position.
// Defined in ras-client-test.cc, which is the TU that owns those statics.
void ResetRasClientGlobals();

#endif  // RCCL_TEST_HOST_FAKES_RAS_CLIENT_FAKES_H_
