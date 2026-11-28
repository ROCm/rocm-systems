// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef __KFD_REMOTE_TEST_INFRA__H__
#define __KFD_REMOTE_TEST_INFRA__H__

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <string>

namespace KFDRemoteTestInfra {

struct RemoteSession {
    pid_t childPid = -1;
    int stdinFd = -1;
    int stdoutFd = -1;
    int stderrFd = -1;
    /* Prefix for remote stderr lines printed while waiting on the remote. */
    std::string stderrPrefix = "Child[stderr]: ";
};

/* In a remote child, stdout is the control channel to the parent. Move it to
 * a private fd and send everything else written to stdout (gtest output) to
 * stderr, so it cannot be mistaken for a protocol message. Must be called
 * before any test runs.
 */
bool SetupRemoteChildStdio();

void InitSession(RemoteSession& session);
bool StartRemoteTest(const std::string& remoteHost,
                     const std::string& remoteBinaryPath,
                     const std::string& testFilter,
                     RemoteSession& session);
bool SendToRemote(RemoteSession& session, const std::string& message);
bool ReceiveFromRemote(RemoteSession& session, char* buffer, size_t bufSize, int timeoutMs);
void DrainRemoteStderr(RemoteSession& session, const std::string& prefix);
void CloseRemoteInput(RemoteSession& session);
bool WaitExpectedMessage(RemoteSession& session, const char* expectedMessage, int timeoutMs,
                            const std::string& stderrPrefix);
bool WaitRemoteExit(RemoteSession& session, int expectedStatus = 0);
void CleanupSession(RemoteSession& session);

void InitStdIoRemoteSession(RemoteSession& session);
bool SendRemoteMsg(RemoteSession& session, const char* msg);
bool ExpectRemoteMsg(RemoteSession& session, const char* expected, int timeoutMs);
void PrintProtocolMsg(const char* role, const char* msg);
bool ReceiveUalinkHandleMsg(RemoteSession& session,
                            uint64_t (&ualink_handle)[2],
                            int timeoutMs,
                            const char* prefix = "UALINK_HANDLE");
bool SendResultStringMsg(RemoteSession& session,
                         const std::string& result,
                         const char* prefix = "RESULT");
bool ReceiveResultStringMsg(RemoteSession& session,
                            std::string& result,
                            int timeoutMs,
                            const char* prefix = "RESULT");

}  // namespace KFDRemoteTestInfra

#endif  // __KFD_REMOTE_TEST_INFRA__H__
