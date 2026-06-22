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
    pthread_t tid;
    pthread_create(&tid, nullptr, &worker, nullptr);
    pthread_join(tid, nullptr);
    return 0;
}
