// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <pthread.h>

static void*
worker(void*)
{
    return nullptr;
}

int
main()
{
    pthread_t tid{};
    if(pthread_create(&tid, nullptr, &worker, nullptr) != 0) return 1;
    if(pthread_join(tid, nullptr) != 0) return 2;
    return 0;
}
