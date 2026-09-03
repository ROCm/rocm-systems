/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_LIBC_FAKES_H_
#define RCCL_TEST_HOST_FAKES_LIBC_FAKES_H_

// Controllable seams for the libc socket / stdio / process surface.
//
// For units whose external dependencies are libc rather than HIP or nccl --
// src/ras/client.cc is the first, and the socket-facing halves of
// ras/client_support.cc, misc/socket.cc and bootstrap.cc are the obvious next
// ones. Such a unit needs no HIP runtime and no nccl fakes at all.
//
// fakes/libc_seam.h macro-renames each call in the unit under test to the
// matching micro_* trampoline, which dispatches through the std::function slot
// declared here. Install per-test behaviour with ScopedHook; ResetLibcFakes()
// restores every default and clears every record.
//
// Add a symbol here when a unit under test reaches it -- with a working
// default and a reset, never as a hardcoded always-succeed. Symbols not yet
// needed by any unit (bind, listen, accept, send, recv, poll) are deliberately
// absent: a seam written without its caller gets the recording surface wrong.

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// Thrown by the default exit seam so a test can assert both the status and the
// state the unit left behind, which a death test cannot observe. Death tests
// remain available: install a hook that calls ::exit / ::_exit instead.
struct MicroExit {
  int status;
};

// One scripted result for the read seam. `ret` < 0 makes the read fail with
// `err` in errno; `ret` == 0 is EOF; a positive `ret` is the byte count the
// step promises and must equal data.size(), which ScriptReadData derives for
// you. The delivery is truncated to the caller's buffer, so a read asking for
// fewer bytes than the step offers gets a short read, not an overrun.
struct MicroReadStep {
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
extern std::string g_writtenData;       // every byte the unit wrote to a descriptor
extern std::string g_stdoutData;        // every byte the unit fwrite()'d, whichever stream it chose
extern FILE* g_lastFwriteStream;        // stream of the last fwrite; distinguishes stdout from stderr
extern std::vector<int> g_closedFds;    // fds passed to close(), in order
extern std::vector<MicroReadStep> g_readScript;  // consumed front-to-back by the default read
extern size_t g_readScriptPos;
extern int g_nextSocketFd;              // what the default socket() hands back (-1 to fail it)
extern int g_socketFailErrno;           // errno the default socket() sets when g_nextSocketFd is -1; no test drives it
                                        // yet, since the one socket-failure test needs per-call behaviour and hooks
extern int g_lastSetsockoptOptname;     // SO_SNDTIMEO / SO_RCVTIMEO of the last setsockopt
extern struct timeval g_lastSetsockoptTimeval;
extern int g_getaddrinfoResult;         // non-zero makes the default getaddrinfo fail with that code
extern int g_addrinfoCount;             // how many entries the default getaddrinfo returns
extern int g_addrinfoBasePort;          // entry i gets port g_addrinfoBasePort + i
extern int g_freeaddrinfoCalls;
extern int g_connectResult;             // 0 succeeds; non-zero fails and sets errno to g_connectErrno
extern int g_connectErrno;

// Queues one scripted read result. Reads past the end of the script return 0 (EOF).
void ScriptRead(ssize_t ret, int err, std::string data);

// Convenience: script one successful read that delivers `data`.
void ScriptReadData(std::string data);

// Restores every seam to its default and clears every record above.
void ResetLibcFakes();

#endif  // RCCL_TEST_HOST_FAKES_LIBC_FAKES_H_
