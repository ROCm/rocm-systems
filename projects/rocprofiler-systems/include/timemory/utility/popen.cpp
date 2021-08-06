// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "timemory/macros/os.hpp"

#if !defined(TIMEMORY_WINDOWS)

#    include "timemory/utility/popen.hpp"
#    include <limits>

#    if !defined(OPEN_MAX)
#        define OPEN_MAX 1024
#    endif

#    if !defined(NGROUPS_MAX)
#        define NGROUPS_MAX 16
#    endif

extern "C"
{
    extern char** environ;
}

namespace tim
{
namespace popen
{
//
//--------------------------------------------------------------------------------------//
//
struct group_info
{
    int   ngroups = -1;
    gid_t group_id;
    uid_t user_id;
    gid_t groups[NGROUPS_MAX];
};
//
//--------------------------------------------------------------------------------------//
//
inline group_info&
get_group_info()
{
    static group_info _instance;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
void
drop_privileges(int permanent)
{
    gid_t newgid = getgid();
    gid_t oldgid = getegid();
    uid_t newuid = getuid();
    uid_t olduid = geteuid();

    if(permanent == 0)
    {
        // Save information about the privileges that are being dropped so that they
        // can be restored later.
        //
        get_group_info().group_id = oldgid;
        get_group_info().user_id  = olduid;
        get_group_info().ngroups  = getgroups(NGROUPS_MAX, get_group_info().groups);
    }

    // If root privileges are to be dropped, be sure to pare down the ancillary
    // groups for the process before doing anything else because the setgroups(  )
    // system call requires root privileges.  Drop ancillary groups regardless of
    // whether privileges are being dropped temporarily or permanently.
    //
    if(olduid == 0)
        setgroups(1, &newgid);

    if(newgid != oldgid)
    {
#    if !defined(TIMEMORY_LINUX)
        auto ret = setegid(newgid);
        if(ret != 0)
            abort();
        if(permanent != 0 && setgid(newgid) == -1)
            abort();
#    else
        if(setregid((permanent ? newgid : oldgid), newgid) == -1)
            abort();
#    endif
    }

    if(newuid != olduid)
    {
#    if !defined(TIMEMORY_LINUX)
        auto ret = seteuid(newuid);
        if(ret != 0)
            abort();
        if(permanent != 0 && setuid(newuid) == -1)
            abort();
#    else
        if(setreuid(((permanent != 0) ? newuid : olduid), newuid) == -1)
            abort();
#    endif
    }

    // verify that the changes were successful
    if(permanent != 0)
    {
        if(newgid != oldgid && (setegid(oldgid) != -1 || getegid() != newgid))
            abort();
        if(newuid != olduid && (seteuid(olduid) != -1 || geteuid() != newuid))
            abort();
    }
    else
    {
        if(newgid != oldgid && getegid() != newgid)
            abort();
        if(newuid != olduid && geteuid() != newuid)
            abort();
    }
}
//
//--------------------------------------------------------------------------------------//
//
void
restore_privileges()
{
    if(geteuid() != get_group_info().user_id)
        if(seteuid(get_group_info().user_id) == -1 ||
           geteuid() != get_group_info().user_id)
            abort();
    if(getegid() != get_group_info().group_id)
        if(setegid(get_group_info().group_id) == -1 ||
           getegid() != get_group_info().group_id)
            abort();
    if(get_group_info().user_id == 0U)
        setgroups(get_group_info().ngroups, get_group_info().groups);
}
//
//--------------------------------------------------------------------------------------//
//
int
open_devnull(int fd)
{
    FILE* f = nullptr;
    switch(fd)
    {
        case 0: f = freopen("/dev/null", "rb", stdin); break;
        case 1: f = freopen("/dev/null", "wb", stdout); break;
        case 2: f = freopen("/dev/null", "wb", stderr); break;
        default: break;
    }
    return (f != nullptr && fileno(f) == fd) ? 1 : 0;
}
//
//--------------------------------------------------------------------------------------//
//
void
sanitize_files()
{
    // int         fds;
    struct stat st;

    // Make sure all open descriptors other than the standard ones are closed
    // if((fds = getdtablesize()) == -1)
    //    fds = OPEN_MAX;

    // closing these files results in the inability to read the pipe from the parent
    // for(int fd = 3; fd < fds; ++fd)
    //    close(fd);

    // Verify that the standard descriptors are open.  If they're not, attempt to
    // open them using /dev/null.  If any are unsuccessful, abort.
    for(int fd = 0; fd < 3; ++fd)
    {
        if(fstat(fd, &st) == -1 && (errno != EBADF || open_devnull(fd) == 0))
        {
            abort();
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
pid_t
fork()
{
    pid_t childpid;

    if((childpid = ::fork()) == -1)
        return -1;

    // If this is the parent process, there's nothing more to do
    if(childpid != 0)
        return childpid;

    // This is the child process
    tim::popen::sanitize_files();    // Close all open files.
    tim::popen::drop_privileges(1);  // Permanently drop privileges.

    return 0;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PIPE*
popen(const char* path, char** argv, char** envp)
{
    int            stdin_pipe[2]  = { 0, 0 };
    int            stdout_pipe[2] = { 0, 0 };
    TIMEMORY_PIPE* p              = nullptr;

    static char** _argv = []() {
        static auto _tmp = new char*[1];
        _tmp[0]          = nullptr;
        return _tmp;
    }();

    if(envp == nullptr)
        envp = environ;
    if(argv == nullptr)
        argv = _argv;

    p = new TIMEMORY_PIPE;

    if(!p)
        return nullptr;

    p->read_fd   = nullptr;
    p->write_fd  = nullptr;
    p->child_pid = -1;

    if(pipe(stdin_pipe) == -1)
    {
        delete p;
        return nullptr;
    }

    if(pipe(stdout_pipe) == -1)
    {
        close(stdin_pipe[1]);
        close(stdin_pipe[0]);
        delete p;
        return nullptr;
    }

    if(!(p->read_fd = fdopen(stdout_pipe[0], "r")))
    {
        close(stdout_pipe[1]);
        close(stdout_pipe[0]);
        close(stdin_pipe[1]);
        close(stdin_pipe[0]);
        delete p;
        return nullptr;
    }

    if(!(p->write_fd = fdopen(stdin_pipe[1], "w")))
    {
        fclose(p->read_fd);
        close(stdout_pipe[1]);
        close(stdin_pipe[1]);
        close(stdin_pipe[0]);
        delete p;
        return nullptr;
    }

    if((p->child_pid = tim::popen::fork()) == -1)
    {
        fclose(p->write_fd);
        fclose(p->read_fd);
        close(stdout_pipe[1]);
        close(stdin_pipe[0]);
        delete p;
        return nullptr;
    }

    if(p->child_pid == 0)
    {
        // this is the child process
        close(stdout_pipe[0]);
        close(stdin_pipe[1]);
        if(stdin_pipe[0] != 0)
        {
            dup2(stdin_pipe[0], 0);
            close(stdin_pipe[0]);
        }
        if(stdout_pipe[1] != 1)
        {
            dup2(stdout_pipe[1], 1);
            close(stdout_pipe[1]);
        }
        execve(path, argv, envp);
        exit(127);
    }

    close(stdout_pipe[1]);
    close(stdin_pipe[0]);

    return p;
}
//
//--------------------------------------------------------------------------------------//
//
int
pclose(TIMEMORY_PIPE* p)
{
    int   status = p->child_status;
    pid_t pid    = -1;

    // clean up memory
    auto _clean = [&]() {
        if(p->read_fd)
            fclose(p->read_fd);
        if(p->write_fd)
            fclose(p->write_fd);
        delete p;
    };

    if(status != std::numeric_limits<int>::max())
    {
        _clean();
        if(WIFEXITED(status))
        {
            // printf("process %i exited, status=%d\n", p->child_pid,
            // WEXITSTATUS(status));
            return EXIT_SUCCESS;
        }
        else if(WIFSIGNALED(status))
        {
            printf("process %i killed by signal %d\n", p->child_pid, WTERMSIG(status));
            return EXIT_FAILURE;
        }
        else if(WIFSTOPPED(status))
        {
            printf("process %i stopped by signal %d\n", p->child_pid, WSTOPSIG(status));
            // return EXIT_FAILURE;
        }
        else if(WIFCONTINUED(status))
        {
            printf("process %i continued\n", p->child_pid);
            // return EXIT_FAILURE;
        }
    }
    else
    {
        if(p->child_pid != -1)
        {
            do
            {
                pid = waitpid(p->child_pid, &status, 0);
            } while(pid == -1 && errno == EINTR);
        }
    }
    _clean();
    if(pid != -1 && WIFEXITED(status))
        return WEXITSTATUS(status);
    return (pid == -1 ? -1 : 0);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace popen
}  // namespace tim

#else

namespace
{
int windows_popen = 0;
}
#endif
