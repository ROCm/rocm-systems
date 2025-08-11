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

#include <dirent.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
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

static int getPIDFromName(std::string name) {
  int pid = -1;

  DIR* dir_ptr = opendir("/proc");
  if (dir_ptr != NULL) {
    struct dirent* dentry;
    while (pid < 0 && (dentry = readdir(dir_ptr))) {
      int id = atoi(dentry->d_name);
      if (id > 0) {
        std::string commPath = std::string("/proc/") + dentry->d_name + "/comm";
        std::ifstream cmdFile(commPath.c_str());
        std::string cmdLine;
        getline(cmdFile, cmdLine);
        if (!cmdLine.empty()) {
          if (name == cmdLine) {
            pid = id;
            break;
          }
        }
      }
    }
  }
  closedir(dir_ptr);
  return pid;
}

static int killRDCD(int pid = 0) {
  if (pid == 0) {
    pid = getPIDFromName("rdcd");

    if (pid == -1) {
      // rdcd is apparently not running
      return 0;
    }
  }

  int ret = kill(pid, SIGTERM);

  if (ret != 0) {
    perror("Failed to kill existing rdcd instance. Error: ");
    return errno;
  }

  int status;
  int err = waitpid(pid, &status, WNOHANG);
  if (err < 0) {
    perror("waitpid() failed for rdcd.");
    return errno;
  } else if (err > 0) {
    std::cout << "Killed rdcd process " << pid << std::endl;
  }
  // Try several times; it may take some time for rdcd to clean up.
  for (int i = 0; i < 20; ++i) {
    sleep(0.01);

    pid = getPIDFromName("rdcd");

    if (pid == -1) {
      return 0;
    }
    ret = kill(pid, SIGTERM);
  }

  return 0;
}

static int startRDCD(std::string* rdcd_path, char* envp[]) {
  assert(rdcd_path != nullptr);
  const char* rdcd_cl[128] = {rdcd_path->c_str(), "-u", NULL};
  int pid = fork();

  if (pid == 0) {
    if (-1 == execve(rdcd_cl[0], (char**)rdcd_cl, envp)) {  // NOLINT
      std::string err_msg = "ERROR: Child process failed to start ";
      err_msg += *rdcd_path;
      perror(err_msg.c_str());
      return -1;
    }
  }

  // Give rdcd a second to get to steady state.
  sleep(1);

  return pid;
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
        if (killRDCD()) {
          return -1;
        }

        rdcd_pid = startRDCD(&settings.rdcd_path, envp);
        assert(rdcd_pid == getPIDFromName("rdcd"));
        if (rdcd_pid != getPIDFromName("rdcd")) {
          std::cout << "Failed to start rdcd. Exiting" << std::endl;
          return -1;
        }
      } else {
        if (getPIDFromName("rdcd") == -1) {
          std::cout << "rdcd is not running. Use -d (--start_rdcd) to have "
                       "rdcd started."
                       " Exiting test."
                    << std::endl;
          return 1;
        }
      }
    }
    ret = RUN_ALL_TESTS();
    if (rdcd_pid != -1) {
      if (killRDCD(rdcd_pid)) {
        return -1;
      }
    }
    if (ret) {
      return ret;
    }
  }

  return ret;
}
