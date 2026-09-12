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

#undef exit
#undef perror
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
