// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "mproc.hpp"
#include "common.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "logger/debug.hpp"

namespace rocprofsys
{
namespace mproc
{
std::set<int>
get_concurrent_processes(int _ppid)
{
    std::set<int> _children = {};
    if(_ppid > 0)
    {
        auto          _inp = fmt::format("/proc/{}/task/{}/children", _ppid, _ppid);
        std::ifstream _ifs{ _inp };
        if(!_ifs)
        {
            LOG_WARNING("File '{}' cannot be read. Returning empty set.", _inp);
            return _children;
        }

        while(_ifs)
        {
            int _v = -1;
            _ifs >> _v;
            if(!_ifs.good() || _ifs.eof()) break;
            if(_v < 0) continue;
            _children.emplace(_v);
        }
    }
    return _children;
}

int
get_process_index(int _pid, int _ppid)
{
    auto _children = get_concurrent_processes(_ppid);
    for(auto itr = _children.begin(); itr != _children.end(); ++itr)
    {
        if(*itr == _pid) return std::distance(_children.begin(), itr);
    }
    return -1;
}

int
wait_pid(pid_t _pid, int _opts)
{
    int   _status = 0;
    pid_t _pid_v  = -1;
    _opts |= WUNTRACED;
    do
    {
        if((_opts & WNOHANG) > 0)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
        }
        _pid_v = waitpid(_pid, &_status, _opts);
    } while(_pid_v <= 0);
    return _status;
}

int
diagnose_status(pid_t _pid, int _status, [[maybe_unused]] int _verbose)
{
    bool _normal_exit      = (WIFEXITED(_status) > 0);
    bool _unhandled_signal = (WIFSIGNALED(_status) > 0);
    bool _core_dump        = (WCOREDUMP(_status) > 0);
    bool _stopped          = (WIFSTOPPED(_status) > 0);
    int  _exit_status      = WEXITSTATUS(_status);
    int  _stop_signal      = (_stopped) ? WSTOPSIG(_status) : 0;
    int  _ec               = (_unhandled_signal) ? WTERMSIG(_status) : 0;

    LOG_TRACE("diagnosing status for process {} :: status: {}... normal exit: {}, "
              "unhandled signal: {}, core dump: {}, stopped: {}, exit status: {}, stop "
              "signal: {}, exit code: {}",
              _pid, _status, std::to_string(_normal_exit),
              std::to_string(_unhandled_signal), std::to_string(_core_dump),
              std::to_string(_stopped), _exit_status, _stop_signal, _ec);

    if(!_normal_exit)
    {
        if(_ec == 0) _ec = EXIT_FAILURE;
        LOG_ERROR("process {} terminated abnormally. exit code: {}", _pid, _ec);
    }

    if(_stopped)
    {
        LOG_ERROR("process {} stopped with signal {}. exit code: {}", _pid, _stop_signal,
                  _ec);
    }

    if(_core_dump)
    {
        LOG_CRITICAL("process {} terminated and produced a core dump. exit code: {}",
                     _pid, _ec);
    }

    if(_unhandled_signal)
    {
        LOG_ERROR("process {} terminated because it received a signal "
                  "({}) that was not handled. exit code: {}",
                  _pid, _ec, _ec);
    }

    if(!_normal_exit && _exit_status > 0)
    {
        if(_exit_status == 127)
        {
            LOG_CRITICAL("execv in process {} failed. exit code: {}", _pid, _ec);
        }
        else
        {
            LOG_CRITICAL("process {} terminated with a non-zero status. exit code: {}",
                         _pid, _ec);
        }
    }

    return _ec;
}
}  // namespace mproc
}  // namespace rocprofsys
