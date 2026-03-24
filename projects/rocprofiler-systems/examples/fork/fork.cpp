// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocprofiler-systems/user.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <set>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// async-signal-safe integer to decimal string
int
itoa_safe(int val, char* out)
{
    if(val == 0)
    {
        out[0] = '0';
        return 1;
    }
    char tmp[12];
    int  n = 0;
    while(val > 0)
    {
        tmp[n++] = '0' + (val % 10);
        val /= 10;
    }
    for(int i = 0; i < n; i++)
        out[i] = tmp[n - 1 - i];
    return n;
}

// async-signal-safe print: "[name] pid = X, ppid = Y\n" to stderr
void
print_info(const char* name)
{
    char buf[256];
    int  pos = 0;

    buf[pos++] = '[';
    int nlen   = strlen(name);
    memcpy(buf + pos, name, nlen);
    pos += nlen;
    const char s1[] = "] pid = ";
    memcpy(buf + pos, s1, sizeof(s1) - 1);
    pos += sizeof(s1) - 1;
    pos += itoa_safe(getpid(), buf + pos);
    const char s2[] = ", ppid = ";
    memcpy(buf + pos, s2, sizeof(s2) - 1);
    pos += sizeof(s2) - 1;
    pos += itoa_safe(getppid(), buf + pos);
    buf[pos++] = '\n';
    auto _rc   = write(STDERR_FILENO, buf, pos);
    (void) _rc;
}

// async-signal-safe print: "[pid X] msg\n" to stderr
void
safe_write(const char* msg)
{
    char       buf[64];
    int        pos  = 0;
    const char s1[] = "[pid ";
    memcpy(buf + pos, s1, sizeof(s1) - 1);
    pos += sizeof(s1) - 1;
    pos += itoa_safe(getpid(), buf + pos);
    const char s2[] = "] ";
    memcpy(buf + pos, s2, sizeof(s2) - 1);
    pos += sizeof(s2) - 1;
    int mlen = strlen(msg);
    memcpy(buf + pos, msg, mlen);
    pos += mlen;
    buf[pos++] = '\n';
    auto _rc   = write(STDERR_FILENO, buf, pos);
    (void) _rc;
}

int
run(const char* _name, int nchildren)
{
    auto _barrier  = pthread_barrier_t{};
    auto _threads  = std::vector<std::thread>{};
    auto _children = std::vector<pid_t>{};
    _children.resize(nchildren, 0);
    pthread_barrier_init(&_barrier, nullptr, nchildren + 1);
    for(int i = 0; i < nchildren; ++i)
    {
        rocprofsys_user_push_region("launch_child");
        auto _run = [&_barrier, &_children, i, _name](uint64_t _nsec) {
            pthread_barrier_wait(&_barrier);
            _children.at(i) = fork();
            if(_children.at(i) == 0)
            {
                // child code
                // must be async-signal-safe (see signal-safety(7))
                print_info(_name);
                safe_write("child job starting");
                sleep(_nsec);
                safe_write("child job complete");
                _exit(EXIT_SUCCESS);
            }
            else
            {
                pthread_barrier_wait(&_barrier);
            }
        };
        _threads.emplace_back(_run, i + 1);
        rocprofsys_user_pop_region("launch_child");
    }

    // all child threads should start executing their fork once this returns
    pthread_barrier_wait(&_barrier);
    // wait for the threads to successfully fork
    pthread_barrier_wait(&_barrier);

    rocprofsys_user_push_region("wait_for_children");

    int   _status   = 0;
    pid_t _wait_pid = 0;
    // parent waits for all the child processes
    for(auto& itr : _children)
    {
        while(itr == 0)
        {
        }
        printf("[%s][%i] performing waitpid(%i, ...)\n", _name, getpid(), itr);
        while((_wait_pid = waitpid(itr, &_status, WUNTRACED | WNOHANG)) <= 0)
        {
            if(_wait_pid == 0) continue;

            printf("[%s][%i] returned from waitpid(%i) with pid = %i (status = %i) :: ",
                   _name, getpid(), itr, _wait_pid, _status);
            if(WIFEXITED(_status))
            {
                printf("exited, status=%d\n", WEXITSTATUS(_status));
            }
            else if(WIFSIGNALED(_status))
            {
                printf("killed by signal %d\n", WTERMSIG(_status));
            }
            else if(WIFSTOPPED(_status))
            {
                printf("stopped by signal %d\n", WSTOPSIG(_status));
            }
            else if(WIFCONTINUED(_status))
            {
                printf("continued\n");
            }
            else
            {
                printf("unknown\n");
            }
        }
    }

    printf("[%s][%i] joining threads ...\n", _name, getpid());
    for(auto& itr : _threads)
        itr.join();

    rocprofsys_user_pop_region("wait_for_children");

    printf("[%s][%i] returning (error code: %i) ...\n", _name, getpid(), _status);
    return _status;
}

int
main(int argc, char** argv)
{
    int _nfork = 4;
    int _nrep  = 1;
    if(argc > 1) _nfork = std::stoi(argv[1]);
    if(argc > 2) _nrep = std::stoi(argv[2]);

    print_info(argv[0]);
    for(int i = 0; i < _nrep; ++i)
    {
        auto _ec = run(argv[0], _nfork);
        if(_ec != 0) return _ec;
    }

    printf("[%s][%i] job complete\n", argv[0], getpid());
    return EXIT_SUCCESS;
}
