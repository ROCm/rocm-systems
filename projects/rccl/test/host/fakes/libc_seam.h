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
//
// malloc/calloc/free are seamed so a unit's out-of-memory arm is reachable on
// demand; their defaults call real libc, so installing the seam without driving
// it changes nothing. free() is seamed too so a test can assert the failing
// path released what it had already taken.
//
// fprintf is seamed but not swappable per test: it always forwards to the real vfprintf so stderr output still
// happens, and only records the call count and the FILE* argument, never the formatted text.

// The headers that declare the names renamed below, so this file satisfies its own ordering rule and a unit following
// the recipe above cannot get it wrong by forgetting one. Include guards make the includer's own copies free.
#include <dirent.h>
#include <dlfcn.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
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
int micro_bind(int, const struct sockaddr*, socklen_t);
int micro_listen(int, int);
int micro_accept(int, struct sockaddr*, socklen_t*);
int micro_fcntl(int, int, ...);
ssize_t micro_recv(int, void*, size_t, int);
ssize_t micro_send(int, const void*, size_t, int);
int micro_setsockopt(int, int, int, const void*, socklen_t);
int micro_getaddrinfo(const char*, const char*, const struct addrinfo*, struct addrinfo**);
void micro_freeaddrinfo(struct addrinfo*);
int micro_getnameinfo(const struct sockaddr*, socklen_t, char*, socklen_t, char*, socklen_t, int);
const char* micro_gai_strerror(int);
size_t micro_fwrite(const void*, size_t, size_t, FILE*);
int micro_fflush(FILE*);
void micro_perror(const char*);
void micro_exit(int) __attribute__((noreturn));
void* micro_malloc(size_t);
void* micro_calloc(size_t, size_t);
void micro_free(void*);
int micro_fprintf(FILE*, const char*, ...) __attribute__((format(printf, 2, 3)));
int micro_gethostname(char*, size_t);
int micro_access(const char*, int);
int micro_poll(struct pollfd*, nfds_t, int);
int micro_getsockopt(int, int, int, void*, socklen_t*);
int micro_getsockname(int, struct sockaddr*, socklen_t*);
int micro_shutdown(int, int);
ssize_t micro_recvmsg(int, struct msghdr*, int);
ssize_t micro_sendmsg(int, const struct msghdr*, int);
int micro_pipe(int*);
int micro_getifaddrs(struct ifaddrs**);
void micro_freeifaddrs(struct ifaddrs*);
int micro_getrlimit(int, struct rlimit*);
int micro_setrlimit(int, const struct rlimit*);
int micro_sched_getaffinity(pid_t, size_t, cpu_set_t*);
void* micro_dlopen(const char*, int);
char* micro_dlerror(void);
char* micro_realpath(const char*, char*);
FILE* micro_fopen(const char*, const char*);
DIR* micro_opendir(const char*);
int micro_mkstemp(char*);
int micro_fallocate(int, int, off_t, off_t);
void* micro_mmap(void*, size_t, int, int, int, off_t);
int micro_munmap(void*, size_t);
int micro_unlink(const char*);
sighandler_t micro_signal(int, sighandler_t);
}  // extern "C"

#define write micro_write
#define read micro_read
#define close micro_close
#define socket micro_socket
#define connect micro_connect
#define bind micro_bind
#define listen micro_listen
#define accept micro_accept
#define fcntl micro_fcntl
#define recv micro_recv
#define send micro_send
#define setsockopt micro_setsockopt
#define getaddrinfo micro_getaddrinfo
#define freeaddrinfo micro_freeaddrinfo
#define getnameinfo micro_getnameinfo
#define gai_strerror micro_gai_strerror
#define fwrite micro_fwrite
#define fflush micro_fflush
#define perror micro_perror
#define exit micro_exit
#define malloc micro_malloc
#define calloc micro_calloc
#define free micro_free
#define fprintf micro_fprintf
#define gethostname micro_gethostname
#define access micro_access
#define poll micro_poll
#define getsockopt micro_getsockopt
#define getsockname micro_getsockname
#define shutdown micro_shutdown
#define recvmsg micro_recvmsg
#define sendmsg micro_sendmsg
#define pipe micro_pipe
#define getifaddrs micro_getifaddrs
#define freeifaddrs micro_freeifaddrs
#define getrlimit micro_getrlimit
#define setrlimit micro_setrlimit
#define sched_getaffinity micro_sched_getaffinity
#define dlopen micro_dlopen
#define dlerror micro_dlerror
#define realpath micro_realpath
#define fopen micro_fopen
#define opendir micro_opendir
#define mkstemp micro_mkstemp
#define fallocate micro_fallocate
#define mmap micro_mmap
#define munmap micro_munmap
#define unlink micro_unlink
#define signal micro_signal
