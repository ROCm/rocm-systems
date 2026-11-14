// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern char** environ;
inline void
require(bool ok)
{
    if(!ok)
    {
        std::perror("tree fixture");
        std::exit(2);
    }
}
inline void
put(int fd, char value)
{
    require(write(fd, &value, 1) == 1);
}
inline char
get(int fd)
{
    char    value = 0;
    ssize_t count;
    do
    {
        count = read(fd, &value, 1);
    } while(count < 0 && errno == EINTR);
    require(count == 1);
    return value;
}
inline pid_t
launch(char* const* args, bool spawn)
{
    pid_t child;
    if(spawn)
        require(posix_spawn(&child, args[0], nullptr, nullptr, args, environ) == 0);
    else
    {
        child = fork();
        require(child >= 0);
        if(child == 0)
        {
            execv(args[0], args);
            _exit(127);
        }
    }
    return child;
}
inline int
wait_child(pid_t child)
{
    int status = 0;
    while(waitpid(child, &status, 0) < 0)
        require(errno == EINTR);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 2;
}
