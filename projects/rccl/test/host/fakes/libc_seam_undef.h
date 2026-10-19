/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Removes the libc seam installed by fakes/libc_seam.h.
//
// Include immediately after the #include <UNIT>_CC_PATH so the test body sees
// real libc again; a test that wants a seamed call can name micro_* directly.
// No include guard on purpose: the push/pop pair is positional.

#undef signal
#undef unlink
#undef munmap
#undef mmap
#undef fallocate
#undef mkstemp
#undef opendir
#undef fopen
#undef realpath
#undef dlerror
#undef dlopen
#undef sched_getaffinity
#undef setrlimit
#undef getrlimit
#undef freeifaddrs
#undef getifaddrs
#undef pipe
#undef sendmsg
#undef recvmsg
#undef shutdown
#undef getsockname
#undef getsockopt
#undef poll
#undef access
#undef gethostname
#undef free
#undef calloc
#undef malloc
#undef fprintf
#undef exit
#undef perror
#undef fflush
#undef fwrite
#undef gai_strerror
#undef getnameinfo
#undef freeaddrinfo
#undef getaddrinfo
#undef setsockopt
#undef send
#undef recv
#undef fcntl
#undef accept
#undef listen
#undef bind
#undef connect
#undef socket
#undef close
#undef read
#undef write
