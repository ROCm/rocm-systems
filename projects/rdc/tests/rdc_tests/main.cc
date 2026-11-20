/*
Copyright (c) 2019 - Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <gtest/gtest.h>
#include <netdb.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "functional/rdci_discovery.h"
#include "functional/rdci_dmon.h"
#include "functional/rdci_fieldgroup.h"
#include "functional/rdci_group.h"
#include "functional/rdci_stats.h"
#include "rdc/rdc.h"
#include "rdc_tests/test_base.h"
#include "rdc_tests/test_common.h"

static RDCTstGlobals* sRDCGlvalues = nullptr;

static void SetFlags(TestBase* test) {
  assert(sRDCGlvalues != nullptr);

  test->set_verbosity(sRDCGlvalues->verbosity);
  test->set_dont_fail(sRDCGlvalues->dont_fail);
  test->set_monitor_server_ip(sRDCGlvalues->monitor_server_ip);
  test->set_monitor_server_port(sRDCGlvalues->monitor_server_port);
  test->set_secure(sRDCGlvalues->secure);
  test->set_mode(sRDCGlvalues->standalone);
}

static void RunCustomTestProlog(TestBase* test) {
  SetFlags(test);

  test->DisplayTestInfo();
  test->SetUp();
  test->Run();
  return;
}
static void RunCustomTestEpilog(TestBase* test) {
  test->DisplayResults();
  test->Close();
  return;
}

// If the test case one big test, you should use RunGenericTest()
// to run the test case. OTOH, if the test case consists of multiple
// functions to be run as separate tests, follow this pattern:
//   * RunCustomTestProlog(test)  // Run() should contain minimal code
//   * <insert call to actual test function within test case>
//   * RunCustomTestEpilog(test)
static void RunGenericTest(TestBase* test) {
  RunCustomTestProlog(test);
  RunCustomTestEpilog(test);
  return;
}

TEST(rdctstReadOnly, TestRdciDiscovery) {
  TestRdciDiscovery tst;
  RunGenericTest(&tst);
}

TEST(rdctstReadOnly, TestRdciGroup) {
  TestRdciGroup tst;
  RunGenericTest(&tst);
}

TEST(rdctstReadOnly, TestRdciDmon) {
  TestRdciDmon tst;
  RunGenericTest(&tst);
}

TEST(rdctstReadOnly, TestRdciFieldgroup) {
  TestRdciFieldgroup tst;
  RunGenericTest(&tst);
}

TEST(rdctstReadOnly, TestRdciStats) {
  TestRdciStats tst;
  RunGenericTest(&tst);
}

// rdctst's standalone tests connect to localhost:50051 (functional/rdci_*.cc).
static const char kRdcdPort[] = "50051";

// Whether something accepts connections on localhost:50051.
static bool rdcdPortInUse() {
  struct addrinfo hints = {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* addresses = nullptr;
  if (getaddrinfo("localhost", kRdcdPort, &hints, &addresses) != 0) {
    return false;
  }
  bool in_use = false;
  for (struct addrinfo* a = addresses; a != nullptr && !in_use; a = a->ai_next) {
    int fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) {
      continue;
    }
    in_use = connect(fd, a->ai_addr, a->ai_addrlen) == 0;
    close(fd);
  }
  freeaddrinfo(addresses);
  return in_use;
}

// Stops the rdcd that startRDCD() started, and only that one: SIGTERM, then SIGKILL if it
// has not exited within 15 seconds. Returns 0 once it is reaped.
static int stopRDCD(pid_t pid) {
  kill(pid, SIGTERM);
  for (int i = 0; i < 150; ++i) {
    pid_t ret = waitpid(pid, nullptr, WNOHANG);
    if (ret == pid) {
      return 0;
    }
    if (ret < 0) {
      perror("waitpid() failed for rdcd");
      return -1;
    }
    usleep(100 * 1000);
  }
  std::cout << "rdcd did not exit within 15 s of SIGTERM; killing it" << std::endl;
  kill(pid, SIGKILL);
  if (waitpid(pid, nullptr, 0) != pid) {
    perror("waitpid() failed for rdcd");
    return -1;
  }
  return 0;
}

// Starts rdcd and waits until it accepts connections on localhost:50051. Returns its pid, or
// -1. Another server on that port is left alone: rdctst fails instead.
static pid_t startRDCD(const std::string& rdcd_path, char* envp[]) {
  if (rdcdPortInUse()) {
    std::cout << "localhost:" << kRdcdPort << " is in use. Stop the server there, or test it"
              << " with --start_rdcd= (no path)." << std::endl;
    return -1;
  }
  const char* rdcd_cl[] = {rdcd_path.c_str(), "-u", nullptr};
  const pid_t parent = getpid();
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork() failed");
    return -1;
  }
  if (pid == 0) {
    // rdcd gets SIGTERM if rdctst dies first.
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() != parent) {
      _exit(127);
    }
    execve(rdcd_cl[0], const_cast<char**>(rdcd_cl), envp);
    std::string err_msg = "ERROR: Child process failed to start " + rdcd_path;
    perror(err_msg.c_str());
    _exit(127);
  }

  // Give rdcd up to 30 seconds to listen.
  for (int i = 0; i < 300; ++i) {
    int status = 0;
    if (waitpid(pid, &status, WNOHANG) == pid) {
      std::cout << "rdcd exited at start with " << (WIFEXITED(status) ? "status " : "signal ")
                << (WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status)) << std::endl;
      return -1;
    }
    if (rdcdPortInUse()) {
      return pid;
    }
    usleep(100 * 1000);
  }
  std::cout << "rdcd did not listen on localhost:" << kRdcdPort << " within 30 s" << std::endl;
  stopRDCD(pid);
  return -1;
}

int main(int argc, char** argv, char* envp[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // disable ROCProfiler because its RDC plugin implementation does not work
  // with GTest
  setenv("RDC_DISABLE_ROCP", "yes", 0);

  RDCTstGlobals settings;
  int ret;
  int rdcd_pid = -1;

  // Set some default values
  settings.verbosity = 1;
  settings.monitor_verbosity = 1;
  settings.num_iterations = 1;
  settings.dont_fail = false;
  settings.rdcd_path = "/usr/sbin/rdcd";
  settings.monitor_server_ip = "";
  settings.monitor_server_port = "";
  settings.secure = false;
  settings.standalone = false;
  settings.batch_mode = false;

  if (ProcessCmdline(&settings, argc, argv)) {
    return 1;
  }

  if (!settings.batch_mode) {
    // Select the embedded mode and standalone mode dynamically.
    std::cout << "Start rdci in: \n";
    std::cout << "0 - Embedded mode \n";
    std::cout << "1 - Standalone mode \n";
    while (!(std::cin >> settings.standalone)) {
      std::cout << "Invalid input.\n";
      std::cin.clear();
      std::cin.ignore();
    }
    std::cout << std::endl;
    std::cout << (settings.standalone ? "Standalone mode selected.\n"
                                      : "Embedded mode selected.\n");
  }
  sRDCGlvalues = &settings;

  if (!settings.standalone || settings.batch_mode) {
    if (settings.batch_mode) {
      // The test needs to know to run Embedded if we are in batch mode
      sRDCGlvalues->standalone = false;
    }
    std::cout << "****************************************" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "Running tests in Embedded mode (no rdcd)" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "****************************************" << std::endl;
    ret = RUN_ALL_TESTS();
  }

  if (settings.standalone || settings.batch_mode) {
    if (settings.batch_mode) {
      settings.standalone = true;
    }
    std::cout << "*********************************************" << std::endl;
    std::cout << "*********************************************" << std::endl;
    std::cout << "*********************************************" << std::endl;
    std::cout << "Running tests in Stand-alone mode (with rdcd)" << std::endl;
    std::cout << "*********************************************" << std::endl;
    std::cout << "*********************************************" << std::endl;
    std::cout << "*********************************************" << std::endl;

    if (settings.monitor_server_ip == "") {
      if (settings.rdcd_path != "") {
        rdcd_pid = startRDCD(settings.rdcd_path, envp);
        if (rdcd_pid == -1) {
          std::cout << "Failed to start rdcd. Exiting" << std::endl;
          return -1;
        }
      } else if (!rdcdPortInUse()) {
        std::cout << "rdcd is not running on localhost:" << kRdcdPort
                  << ". Use -d (--start_rdcd) to have rdcd started. Exiting test." << std::endl;
        return 1;
      }
    }
    ret = RUN_ALL_TESTS();
    if (rdcd_pid != -1) {
      if (stopRDCD(rdcd_pid)) {
        return -1;
      }
    }
    if (ret) {
      return ret;
    }
  }

  return ret;
}
