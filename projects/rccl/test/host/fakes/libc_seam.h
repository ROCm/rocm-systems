/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Installs the libc seam over a unit under test.
//
// Include this AFTER every header that declares one of the names renamed below
// (so a rename never rewrites a declaration) and immediately BEFORE the
// #include <UNIT>_CC_PATH. Include fakes/libc_seam_undef.h straight after the
// unit to take the renames back off, so the test body itself still sees real
// libc. There is no include guard on purpose: the push/pop pair is positional.
//
//   #include <netdb.h>            // ... and the rest of the unit's headers
//   #include "fakes/libc_fakes.h"
//   #include "fakes/libc_seam.h"
//   #include MY_UNIT_CC_PATH
//   #include "fakes/libc_seam_undef.h"
//
// Each name below becomes a micro_* trampoline that dispatches through the
// matching std::function slot in libc_fakes.h, so a test can swap the
// behaviour per case with ScopedHook.
//
// Not seamed here: getopt_long, and the str*/snprintf family. Their behaviour
// is usually part of what the unit is being tested for, and faking them would
// assert the test's model of libc rather than the unit's use of it.

// The headers that declare the names renamed below, so this file satisfies its own ordering rule and a unit following
// the recipe above cannot get it wrong by forgetting one. Include guards make the includer's own copies free.
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

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
void micro_perror(const char*);
void micro_exit(int) __attribute__((noreturn));
}  // extern "C"

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
#define perror micro_perror
#define exit micro_exit
