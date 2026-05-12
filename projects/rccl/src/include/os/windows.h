#ifndef NCCL_WINDOWS_H_
#define NCCL_WINDOWS_H_

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#ifndef strncasecmp
#define strncasecmp _strnicmp
#endif
#include <windows.h>
#pragma warning(push)
#pragma warning(disable: 4005)
#include <winsock2.h>
#pragma warning(pop)
#include <ws2ipdef.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#include <processthreadsapi.h>
#include <io.h>
#include <process.h>
#include <time.h>
#include <profileapi.h>
#include <libloaderapi.h>
#include <string.h>

#ifndef SHUT_WR
#define SHUT_WR SD_SEND
#endif

#define NCCL_INVALID_SOCKET INVALID_SOCKET
typedef SOCKET ncclSocketDescriptor;

typedef DWORD_PTR ncclAffinity;

typedef unsigned long ncclPid_t;

int gettimeofday(struct timeval* tv, void* tz);

#define NCCL_POLLIN POLLRDNORM
#define NCCL_POLLERR (POLLHUP | POLLERR | POLLNVAL)

#endif
