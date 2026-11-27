// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "KFDRemoteTestInfra.hpp"

#include <algorithm>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <map>
#include <iostream>
#include <stdio.h>

#include "GoogleTestExtension.hpp"
#include "KFDTestFlags.hpp"

namespace KFDRemoteTestInfra {

namespace {

enum class RecvStatus { Line, Timeout, Closed };

int64_t NowMs();
RecvStatus ReceiveLine(int fd, int errFd, const std::string& errPrefix,
                       char* buffer, size_t bufSize, int timeoutMs);
bool SendRemoteMessage(int fd, const std::string& message);
void DrainRemoteFd(int fd, const std::string& prefix);
void ResetPendingForFd(int fd);
void TerminateChild(pid_t pid);
bool SpawnRemoteProcess(const std::string& remoteHost,
                             const std::string& remoteBinaryPath,
                             const std::string& testFilter,
                             pid_t& childPid,
                             int stdinPipe[2],
                             int stdoutPipe[2],
                             int stderrPipe[2]);

/* Where a remote child writes protocol messages, see SetupRemoteChildStdio(). */
int g_ProtocolOutFd = STDOUT_FILENO;

}  // namespace

bool SetupRemoteChildStdio() {
    /* A parent that went away must fail the test, not kill it with SIGPIPE. */
    signal(SIGPIPE, SIG_IGN);

    std::cout.flush();
    fflush(stdout);

    int fd = dup(STDOUT_FILENO);
    if (fd < 0)
        return false;
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        close(fd);
        return false;
    }
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    g_ProtocolOutFd = fd;
    return true;
}

void InitSession(RemoteSession& session) {
    session.childPid = -1;
    session.stdinFd = -1;
    session.stdoutFd = -1;
    session.stderrFd = -1;
}

bool StartRemoteTest(const std::string& remoteHost,
                     const std::string& remoteBinaryPath,
                     const std::string& testFilter,
                     RemoteSession& session) {
    int stdinPipe[2] = {-1, -1};
    int stdoutPipe[2] = {-1, -1};
    int stderrPipe[2] = {-1, -1};

    /* A remote that exited early must fail the test, not kill kfdtest with
     * SIGPIPE on the next write to it.
     */
    signal(SIGPIPE, SIG_IGN);

    if (!SpawnRemoteProcess(remoteHost, remoteBinaryPath, testFilter,
                                 session.childPid, stdinPipe, stdoutPipe, stderrPipe)) {
        return false;
    }

    session.stdinFd = stdinPipe[1];
    session.stdoutFd = stdoutPipe[0];
    session.stderrFd = stderrPipe[0];
    return true;
}

bool SendToRemote(RemoteSession& session, const std::string& message) {
    if (session.stdinFd < 0)
        return false;
    return SendRemoteMessage(session.stdinFd, message);
}

bool ReceiveFromRemote(RemoteSession& session, char* buffer, size_t bufSize, int timeoutMs) {
    if (session.stdoutFd < 0)
        return false;
    return ReceiveLine(session.stdoutFd, session.stderrFd, session.stderrPrefix,
                       buffer, bufSize, timeoutMs) == RecvStatus::Line;
}

void DrainRemoteStderr(RemoteSession& session, const std::string& prefix) {
    if (session.stderrFd >= 0)
        DrainRemoteFd(session.stderrFd, prefix);
}

void CloseRemoteInput(RemoteSession& session) {
    if (session.stdinFd >= 0) {
        ResetPendingForFd(session.stdinFd);
        close(session.stdinFd);
        session.stdinFd = -1;
    }
}

bool WaitExpectedMessage(RemoteSession& session, const char* expectedMessage,
                            int timeoutMs, const std::string& stderrPrefix) {
    if (!expectedMessage || session.stdoutFd < 0)
        return false;

    const int64_t deadline = NowMs() + timeoutMs;
    char line[256];

    for (;;) {
        int remainingMs = static_cast<int>(std::max<int64_t>(deadline - NowMs(), 0));
        RecvStatus status = ReceiveLine(session.stdoutFd, session.stderrFd, stderrPrefix,
                                        line, sizeof(line), remainingMs);
        if (status == RecvStatus::Line) {
            if (strcmp(line, expectedMessage) == 0)
                return true;
            LOG() << "Parent: Ignoring remote stdout line: " << line << std::endl;
            continue;
        }

        DrainRemoteStderr(session, stderrPrefix);
        if (status == RecvStatus::Closed)
            LOG() << "Parent: Remote closed the connection while waiting for "
                  << expectedMessage << std::endl;
        else
            LOG() << "Parent: Timed out after " << timeoutMs << " ms waiting for "
                  << expectedMessage << std::endl;
        return false;
    }
}

bool WaitRemoteExit(RemoteSession& session, int expectedStatus) {
    if (session.childPid <= 0)
        return true;

    const int64_t deadline = NowMs() + std::max<int64_t>(g_TestTimeOut, 1000);
    int wstatus = 0;
    pid_t ret;

    /* Keep draining stderr so the remote cannot block on a full pipe. */
    while ((ret = waitpid(session.childPid, &wstatus, WNOHANG)) == 0 ||
           (ret < 0 && errno == EINTR)) {
        DrainRemoteStderr(session, session.stderrPrefix);
        if (NowMs() >= deadline) {
            LOG() << "Parent: Remote did not exit, terminating it" << std::endl;
            TerminateChild(session.childPid);
            session.childPid = -1;
            return false;
        }
        usleep(10000);
    }
    DrainRemoteStderr(session, session.stderrPrefix);
    session.childPid = -1;

    if (ret < 0 || !WIFEXITED(wstatus)) {
        LOG() << "Parent: Remote did not exit normally" << std::endl;
        return false;
    }
    if (WEXITSTATUS(wstatus) != expectedStatus) {
        LOG() << "Parent: Remote exited with status " << WEXITSTATUS(wstatus)
              << ", expected " << expectedStatus << std::endl;
        return false;
    }
    return true;
}

void CleanupSession(RemoteSession& session) {
    if (session.stdinFd >= 0) {
        ResetPendingForFd(session.stdinFd);
        close(session.stdinFd);
        session.stdinFd = -1;
    }
    if (session.stdoutFd >= 0) {
        ResetPendingForFd(session.stdoutFd);
        close(session.stdoutFd);
        session.stdoutFd = -1;
    }
    if (session.stderrFd >= 0) {
        ResetPendingForFd(session.stderrFd);
        close(session.stderrFd);
        session.stderrFd = -1;
    }

    if (session.childPid > 0) {
        TerminateChild(session.childPid);
        session.childPid = -1;
    }
}

void InitStdIoRemoteSession(RemoteSession& session) {
    InitSession(session);
    session.stdinFd = g_ProtocolOutFd;
    session.stdoutFd = STDIN_FILENO;
}

bool SendRemoteMsg(RemoteSession& session, const char* msg) {
    if (!msg)
        return false;
    if (!SendToRemote(session, msg)) {
        LOG() << "Failed to send " << msg << " to remote: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool ExpectRemoteMsg(RemoteSession& session, const char* expected, int timeoutMs) {
    if (!expected || session.stdoutFd < 0)
        return false;

    char msg[256] = {};
    RecvStatus status = ReceiveLine(session.stdoutFd, session.stderrFd, session.stderrPrefix,
                                    msg, sizeof(msg), timeoutMs);
    if (status == RecvStatus::Closed) {
        LOG() << "Remote closed the connection while waiting for " << expected << std::endl;
        return false;
    }
    if (status == RecvStatus::Timeout) {
        LOG() << "Timed out after " << timeoutMs << " ms waiting for " << expected << std::endl;
        return false;
    }
    if (strcmp(msg, expected) != 0) {
        LOG() << "Expected " << expected << " from remote, got: " << msg << std::endl;
        return false;
    }
    return true;
}

void PrintProtocolMsg(const char* role, const char* msg) {
    std::cout << (role ? role : "") << " protocol: " << (msg ? msg : "") << std::endl;
}

bool ReceiveUalinkHandleMsg(RemoteSession& session,
                            uint64_t (&ualink_handle)[2],
                            int timeoutMs,
                            const char* prefix) {
    char msg[256] = {};
    if (!ReceiveFromRemote(session, msg, sizeof(msg), timeoutMs))
        return false;

    unsigned long long h0 = 0;
    unsigned long long h1 = 0;
    const char* expectedPrefix = (prefix && prefix[0]) ? prefix : "UALINK_HANDLE";
    char format[64] = {};
    snprintf(format, sizeof(format), "%s %%llu %%llu", expectedPrefix);
    if (sscanf(msg, format, &h0, &h1) != 2) {
        LOG() << "Expected " << expectedPrefix << " from remote, got: " << msg << std::endl;
        return false;
    }

    ualink_handle[0] = static_cast<uint64_t>(h0);
    ualink_handle[1] = static_cast<uint64_t>(h1);
    return true;
}

bool SendResultStringMsg(RemoteSession& session,
                         const std::string& result,
                         const char* prefix) {
    const char* pfx = (prefix && prefix[0]) ? prefix : "RESULT";
    std::string msg = std::string(pfx) + " " + result;
    return SendToRemote(session, msg);
}

bool ReceiveResultStringMsg(RemoteSession& session,
                            std::string& result,
                            int timeoutMs,
                            const char* prefix) {
    char msg[256] = {};
    if (!ReceiveFromRemote(session, msg, sizeof(msg), timeoutMs))
        return false;

    const char* expectedPrefix = (prefix && prefix[0]) ? prefix : "RESULT";
    size_t prefixLen = strlen(expectedPrefix);

    if (strncmp(msg, expectedPrefix, prefixLen) != 0)
        return false;

    if (msg[prefixLen] != ' ')
        return false;

    const char* value = msg + prefixLen + 1;
    if (*value == '\0')
        return false;

    result = value;
    return true;
}

namespace {

enum class ReadStatus { Data, NoData, Closed };

int64_t NowMs() {
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

/* Per-fd line-buffer for partial reads. Keyed by fd, so it MUST be cleared
 * when an fd is closed (see ResetPendingForFd); otherwise a later session that
 * reuses the same fd number could inherit stale partial-line data.
 */
std::map<int, std::string>& PendingBufferMap() {
    static std::map<int, std::string> pendingByFd;
    return pendingByFd;
}

void ResetPendingForFd(int fd) {
    PendingBufferMap().erase(fd);
}

bool ExtractLine(std::string& pending, char* out, size_t outSize) {
    size_t newlinePos = pending.find('\n');
    if (newlinePos == std::string::npos)
        return false;

    size_t copyLen = std::min(newlinePos, outSize - 1);
    memcpy(out, pending.data(), copyLen);
    out[copyLen] = '\0';
    pending.erase(0, newlinePos + 1);
    return true;
}

/* Append whatever can be read from fd right now to its pending buffer. */
ReadStatus ReadPending(int fd) {
    char chunk[512];
    ssize_t n;

    do {
        n = read(fd, chunk, sizeof(chunk));
    } while (n < 0 && errno == EINTR);

    if (n > 0) {
        PendingBufferMap()[fd].append(chunk, static_cast<size_t>(n));
        return ReadStatus::Data;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return ReadStatus::NoData;
    return ReadStatus::Closed;
}

/* Log complete lines buffered for fd; at EOF also log a trailing partial line. */
void LogPendingLines(int fd, const std::string& prefix, bool atEof) {
    std::string& pending = PendingBufferMap()[fd];
    char line[512];

    while (ExtractLine(pending, line, sizeof(line)))
        LOG() << prefix << line << std::endl;
    if (atEof && !pending.empty()) {
        LOG() << prefix << pending << std::endl;
        pending.clear();
    }
}

/* Wait up to timeoutMs (forever if negative) for a complete line on fd. While
 * waiting, lines arriving on errFd (if >= 0) are logged with errPrefix, so a
 * remote that writes a lot to stderr cannot block on a full pipe.
 */
RecvStatus ReceiveLine(int fd, int errFd, const std::string& errPrefix,
                       char* buffer, size_t bufSize, int timeoutMs) {
    if (!buffer || bufSize == 0)
        return RecvStatus::Closed;

    const int64_t deadline = NowMs() + std::max(timeoutMs, 0);

    for (;;) {
        if (ExtractLine(PendingBufferMap()[fd], buffer, bufSize))
            return RecvStatus::Line;

        struct timeval tv;
        struct timeval *tvPtr = nullptr;
        if (timeoutMs >= 0) {
            int64_t remainingMs = std::max<int64_t>(deadline - NowMs(), 0);
            tv.tv_sec = remainingMs / 1000;
            tv.tv_usec = (remainingMs % 1000) * 1000;
            tvPtr = &tv;
        }

        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(fd, &readSet);
        if (errFd >= 0)
            FD_SET(errFd, &readSet);

        int ready = select(std::max(fd, errFd) + 1, &readSet, nullptr, nullptr, tvPtr);
        if (ready < 0) {
            if (errno == EINTR)
                continue;
            return RecvStatus::Closed;
        }
        if (ready == 0)
            return RecvStatus::Timeout;

        if (errFd >= 0 && FD_ISSET(errFd, &readSet)) {
            ReadStatus status = ReadPending(errFd);
            LogPendingLines(errFd, errPrefix, status == ReadStatus::Closed);
            /* Stop watching a closed stderr so select() does not spin on it. */
            if (status == ReadStatus::Closed)
                errFd = -1;
        }

        if (FD_ISSET(fd, &readSet) && ReadPending(fd) == ReadStatus::Closed)
            return ExtractLine(PendingBufferMap()[fd], buffer, bufSize) ?
                   RecvStatus::Line : RecvStatus::Closed;
    }
}

bool SendRemoteMessage(int fd, const std::string& message) {
    std::string msg = message + "\n";
    const char* data = msg.data();
    size_t remaining = msg.size();

    while (remaining > 0) {
        ssize_t n = write(fd, data, remaining);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        data += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}

void DrainRemoteFd(int fd, const std::string& prefix) {
    ReadStatus status;

    do {
        status = ReadPending(fd);
        LogPendingLines(fd, prefix, status == ReadStatus::Closed);
    } while (status == ReadStatus::Data);
}

/* SIGTERM the local ssh, give it a moment to exit, then SIGKILL so we never
 * block forever reaping it.
 */
void TerminateChild(pid_t pid) {
    const int64_t deadline = NowMs() + 2000;
    pid_t ret;

    kill(pid, SIGTERM);
    while ((ret = waitpid(pid, nullptr, WNOHANG)) == 0 ||
           (ret < 0 && errno == EINTR)) {
        if (NowMs() >= deadline) {
            LOG() << "Parent: Remote ignored SIGTERM, killing it" << std::endl;
            kill(pid, SIGKILL);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR)
                ;
            return;
        }
        usleep(10000);
    }
}

/* Quote s as a single word for the remote shell that runs the ssh command. */
std::string ShellQuote(const std::string& s) {
    std::string quoted = "'";

    for (char c : s) {
        if (c == '\'')
            quoted += "'\\''";
        else
            quoted += c;
    }
    return quoted + "'";
}

bool SpawnRemoteProcess(const std::string& remoteHost,
                             const std::string& remoteBinaryPath,
                             const std::string& testFilter,
                             pid_t& childPid,
                             int stdinPipe[2],
                             int stdoutPipe[2],
                             int stderrPipe[2]) {
    auto closePipe = [](int p[2]) {
        if (p[0] >= 0) { close(p[0]); p[0] = -1; }
        if (p[1] >= 0) { close(p[1]); p[1] = -1; }
    };

    /* Build everything before fork(): the child may only exec. */
    std::string remoteCmd = ShellQuote(remoteBinaryPath) + " " +
                            ShellQuote("--gtest_filter=" + testFilter) +
                            " --remotechild 1 --timeout " +
                            std::to_string(g_TestTimeOut);
    if (g_RemoteNodeId >= 0)
        remoteCmd += " --node " + std::to_string(g_RemoteNodeId);
    /* BatchMode makes ssh fail instead of prompting for a password or host
     * key, ConnectTimeout bounds the connect, and the keepalives detect a
     * remote host that has gone away.
     */
    const char* sshArgv[] = {
        "ssh", "-T",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "--", remoteHost.c_str(), remoteCmd.c_str(),
        nullptr
    };

    if (pipe(stdinPipe))
        return false;
    if (pipe(stdoutPipe)) {
        closePipe(stdinPipe);
        return false;
    }
    if (pipe(stderrPipe)) {
        closePipe(stdinPipe);
        closePipe(stdoutPipe);
        return false;
    }

    childPid = fork();
    if (childPid == -1) {
        closePipe(stdinPipe);
        closePipe(stdoutPipe);
        closePipe(stderrPipe);
        return false;
    }

    if (childPid == 0) {
        dup2(stdinPipe[0], STDIN_FILENO);
        dup2(stdoutPipe[1], STDOUT_FILENO);
        dup2(stderrPipe[1], STDERR_FILENO);
        close(stdinPipe[0]);
        close(stdinPipe[1]);
        close(stdoutPipe[0]);
        close(stdoutPipe[1]);
        close(stderrPipe[0]);
        close(stderrPipe[1]);

        signal(SIGPIPE, SIG_DFL);
        execvp("ssh", const_cast<char* const*>(sshArgv));
        _exit(127);
    }

    close(stdinPipe[0]);
    close(stdoutPipe[1]);
    close(stderrPipe[1]);
    fcntl(stderrPipe[0], F_SETFL, O_NONBLOCK);
    return true;
}

}  // namespace

}  // namespace KFDRemoteTestInfra
