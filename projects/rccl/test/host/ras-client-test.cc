/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/client.cc.
//
// client.cc is a standalone executable: every helper is `static` and its whole
// dependency surface is libc plus four macros from ras_internal.h. This TU
// #includes the hipified client.cc directly, so the statics become callable,
// and macro-renames each libc entry point to a micro_* trampoline that
// dispatches through a swappable slot in fakes/ras_client_fakes.h.
//
// getopt_long is deliberately NOT seamed: its parse behaviour is part of what
// parseArgs is being tested for. ResetRasClientGlobals() resets optind instead.

#include <gtest/gtest.h>

// Every libc header client.cc reaches through os.h, pulled in BEFORE the macro
// renames below so the renames never rewrite a declaration.
#include <arpa/inet.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/ras_client_fakes.h"

extern "C" {
ssize_t micro_write(int, const void*, size_t);
ssize_t micro_read(int, void*, size_t);
int micro_close(int);
int micro_socket(int, int, int);
int micro_connect(int, const struct sockaddr*, socklen_t);
int micro_setsockopt(int, int, int, const void*, socklen_t);
int micro_getaddrinfo(const char*, const char*, const struct addrinfo*, struct addrinfo**);
void micro_freeaddrinfo(struct addrinfo*);
int micro_getnameinfo(const struct sockaddr*, socklen_t, char*, socklen_t, char*, socklen_t, int);
const char* micro_gai_strerror(int);
size_t micro_fwrite(const void*, size_t, size_t, FILE*);
int micro_fflush(FILE*);
void micro_exit(int) __attribute__((noreturn));
}

#define write micro_write
#define read micro_read
#define close micro_close
#define socket micro_socket
#define connect micro_connect
#define setsockopt micro_setsockopt
#define getaddrinfo micro_getaddrinfo
#define freeaddrinfo micro_freeaddrinfo
#define getnameinfo micro_getnameinfo
#define gai_strerror micro_gai_strerror
#define fwrite micro_fwrite
#define fflush micro_fflush
#define exit micro_exit
// client.cc's main() would collide with the gtest main in main_altrsmi.cpp.
#define main rasClientMain

#include RAS_CLIENT_CC_PATH

#undef main
#undef exit
#undef fflush
#undef fwrite
#undef gai_strerror
#undef getnameinfo
#undef freeaddrinfo
#undef getaddrinfo
#undef setsockopt
#undef connect
#undef socket
#undef close
#undef read
#undef write

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

void ResetRasClientGlobals() {
  hostName = "localhost";
  port = STR(NCCL_RAS_CLIENT_PORT);
  timeout = -1;
  verbose = false;
  monitorMode = false;
  format = nullptr;
  events = nullptr;
  sock = -1;
  // glibc treats optind == 0 (not 1) as "reinitialize everything", which is what
  // repeated getopt_long calls in one process need.
  optind = 0;
}

// ===========================================================================
// Default fixture for every test in this file. Tests that need extra per-test
// state derive from it; report any derived suite name so it can be registered
// in test/test_categories_micro_ras_client.yaml -- gtest's '*' does not match
// across the literal '.', so an unlisted suite never runs under CTest.
// ===========================================================================
class RasClientMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetRasClientFakes();
    ResetRasClientGlobals();
  }
  void TearDown() override {
    ResetRasClientFakes();
    ResetRasClientGlobals();
  }
};

// Scaffolding smoke test: proves the macro seams reach the fakes and that the
// file-scope statics of client.cc are reachable and resettable from this TU.
TEST_F(RasClientMicrotest, Scaffolding_DefaultSeams_AreReachableAndReset) {
  EXPECT_STREQ("localhost", hostName);
  EXPECT_EQ(-1, timeout);
  EXPECT_EQ(-1, sock);

  char buf[8] = {0};
  EXPECT_EQ(4, socketWrite(7, "abcd", 4));
  EXPECT_EQ("abcd", g_writtenData);

  ScriptReadData("hi\n");
  EXPECT_EQ(3, rasRead(7, buf, sizeof(buf)));
  EXPECT_STREQ("hi\n", buf);

  EXPECT_THROW(micro_exit(3), RasClientExit);
}
