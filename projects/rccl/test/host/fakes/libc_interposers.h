/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// dlsym(RTLD_NEXT) interposers for the libc entry points the units under test
// call directly (init.cc:1025/1029, misc/utils.cc:75). Named for what they stand
// in for -- libc -- because no RCCL production TU owns these symbols.

#ifndef RCCL_TEST_HOST_LIBC_INTERPOSERS_H_
#define RCCL_TEST_HOST_LIBC_INTERPOSERS_H_

#include <cstddef>

void SetGethostnameFail(bool fail);
void SetDladdrFail(bool fail);
size_t LastGethostnameLen();

void ResetLibcInterposers();

#endif  // RCCL_TEST_HOST_LIBC_INTERPOSERS_H_
