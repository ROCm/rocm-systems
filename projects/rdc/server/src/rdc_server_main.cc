
/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include <assert.h>
#include <fcntl.h>
#include <grpcpp/grpcpp.h>
#include <sys/resource.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/capability.h>

#include <iostream>
#include <memory>
#include <string>
#include <csignal>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rocm_smi/rocm_smi.h"
#include "rdc/rdc_server_main.h"
#include "rdc/rdc_rsmi_service.h"
#include "rdc/rdc_server_utils.h"

static bool sShutDownServer = false;
static bool sRestartServer = false;
static const char *kDaemonName = "rdcd";
static const char *kRDCDHomeDir = "/";
static const char *kDaemonLockFile = "/var/run/rdcd.lock";

RDCServer::RDCServer() : server_address_("0.0.0.0:50051"),
                         rsmi_service_(nullptr), rdc_admin_service_(nullptr) {
}

RDCServer::~RDCServer() {
}

void
RDCServer::Initialize() {
}

// TODO(cfreehil): read server config from YAML file. Config can include things
// like server address, Secure/Insecure creds, rsmi_init flags, etc.
void
RDCServer::Run() {
  ::grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());

  // Register services as the instances through which we'll communicate with
  // clients. These are synchronous services.
  if (start_rdc_admin_service()) {
    rdc_admin_service_ = new amd::rdc::RDCAdminServiceImpl();
    builder.RegisterService(rdc_admin_service_);
  }

  if (start_rsmi_service()) {
    rsmi_service_ = new amd::rdc::RsmiServiceImpl();
    builder.RegisterService(rsmi_service_);

    // TODO(cfreehil): pass flags from cnfg file
    rsmi_status_t ret = rsmi_service_->Initialize(0);

    if (ret != RSMI_STATUS_SUCCESS) {
      std::cerr << "Failed to start RSMI service" << std::endl;
      return;
    }
  }

  // Finally assemble the server.
  // std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  server_ = builder.BuildAndStart();

  std::cout <<  "Server listening on " << server_address_.c_str() << std::endl;

  server_->Wait();
}

static void HandleSignal(int sig) {
  std::cout <<  "Caught signal " << sig << std::endl;

  // For most signals, we will want to exit, so make that the default case
  // Handle the other signals specifically.
  switch (sig) {
    case SIGINT:
    case SIGTERM:
      sShutDownServer = true;
      break;

      // Grpc doesn't seem to handle stopping and restarting well, so
      // user must manually do these steps
//    case SIGHUP:
//      sRestartServer = true;
//      break;

    default:
      std::cerr << "Unexpected signal caught" << std::endl;
  }
}

static void InitializeSignalHandling(void) {
  // signal(SIGHUP, HandleSignal);
  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);
}

void
RDCServer::ShutDown(void) {
  server_->Shutdown();

  if (rsmi_service_) {
    delete rsmi_service_;
    rsmi_service_ = nullptr;
  }

  if (rdc_admin_service_) {
    delete rdc_admin_service_;
    rdc_admin_service_ = nullptr;
  }
}

static void * ProcessSignalLoop(void *server_ptr) {
  assert(server_ptr != nullptr);
  RDCServer *server = reinterpret_cast<RDCServer *>(server_ptr);

  while (1) {
    if (sShutDownServer) {
      std::cout <<  "Shutting down RDC Server." << std::endl;
      server->ShutDown();
      // We will need to add shutdown of any completion queues
      // here, when/if we add them
      break;
    } else if (sRestartServer) {
      std::cout <<  "Re-starting RDC Server." << std::endl;
      // We will need to add shutdown of any completion queues
      // here, when/if we add them
      server->ShutDown();
      server->Run();

      sRestartServer = false;
    }
    sleep(1);
  }

  pthread_exit(0);
}

static void ExitIfAlreadyRunning(void) {
  int single_proc_fh;
  ssize_t fsz;

  single_proc_fh = open(kDaemonLockFile, O_RDWR|O_CREAT, 0640);
  if (single_proc_fh < 0) {
    std::cerr << "Failed to open file lock:" << kDaemonLockFile << std::endl;
    exit(1);
  }

  if (lockf(single_proc_fh, F_TLOCK, 0) < 0) {
    std::cerr << "Daemon already running. Exiting this instance." << std::endl;
    exit(0);
  }

  std::string pid_str = std::to_string(getpid());
  fsz = write(single_proc_fh, pid_str.c_str(), pid_str.size());
  assert(static_cast<unsigned int>(fsz) == pid_str.size());
}

static void
MakeDaemon() {
  int fd0;
  struct rlimit max_files;

  // RSMI, for one thing, will need to be able to read/write files
  // Note that umask turns *off* permission for a given bit, so you we want
  // the complement of the permissions we want files to have.
  umask(027);

  // To Do; Make this optional based on CL option. By default, don't do this.
  // Instead rely on serviced to make it a daemon.
#if 0
  pid_t pid;
  // We want to dissassociate with calling process, so fork, and let
  // daemon live in child process. Parent will exit.
  if ((pid = fork()) < 0) {
    std::cerr << "Failed to fork rdcd daemon." << std::endl;
  } else if (pid != 0) {  // parent
    exit(0);
  }

  setsid();

  // Insulate from pgrp leader death
  signal(SIGHUP, SIG_IGN);
  if ((pid = fork()) < 0) {
      std::cerr << "Failed to fork after signal(SIGHUP, SIG_IGN)" << std::endl;
  } else if (pid != 0) {  // parent
      exit(0);
  }
#endif

  // chdir to dir that will always be available
  if (chdir(kRDCDHomeDir) < 0) {
      std::cerr << "Failed to change directory to " <<kRDCDHomeDir << std::endl;
  }

  // Determine max. number of open files possible. We need to close all
  // open descriptors.
  if (getrlimit(RLIMIT_NOFILE, &max_files) < 0) {
      std::cerr << kDaemonName << ": can't get file limit" << std::endl;
  }

  // Close files
  if (max_files.rlim_max > 1024) {
    max_files.rlim_max = 1024;
  }
  close(0);  // close stdin; leave stdout and stderr open
  for (uint32_t i = 3; i < max_files.rlim_max; i++) {
    close(i);
  }

  // Direct stdin to /dev/null.
  fd0 = open("/dev/null", O_RDWR);
  if (fd0 != 0) {
      std::cerr << "unexpected fildes: " << fd0 << std::endl;
      exit(1);
  }

  ExitIfAlreadyRunning();

  InitializeSignalHandling();
}

int main(int argc, char** argv) {
  RDCServer rdc_server;
  int err;

  (void)argc;   // Ignore for now
  (void)argv;

  MakeDaemon();

  rdc_server.Initialize();

  bool cap_enabled;

  err = GetCapability(CAP_DAC_OVERRIDE, CAP_EFFECTIVE, &cap_enabled);
  if (err) {
    std::cerr << "Failed to get capability" << std::endl;
    return 1;
  }
  if (!cap_enabled) {
    std::cerr <<
      "Expected CAP_DAC_OVERRIDE CAP_EFFECTIVE to be enabled, but it not." <<
                                                                    std::endl;
    return 1;
  }

  err = GetCapability(CAP_DAC_OVERRIDE, CAP_PERMITTED, &cap_enabled);
  if (err) {
    std::cerr << "Failed to get capability" << std::endl;
    return 1;
  }
  if (!cap_enabled) {
    std::cerr <<
      "Expected CAP_DAC_OVERRIDE CAP_PERMITTED to be enabled, but it not." <<
                                                                     std::endl;
    return 1;
  }

  // Don't allow rwx access to all files to ever be inheritable. We may need
  // relax this restriction if some new feature requires it.
  err = ModifyCapability(CAP_DAC_OVERRIDE, CAP_INHERITABLE, false);
  if (err) {
    std::cerr << "Failed to disable CAP_DAC_OVERRIDE, CAP_INHERITABLE" <<
                                                                    std::endl;
    return 1;
  }

  // By default, disable CAP_DAC_OVERRIDE. Turn on, when needed.
  err = ModifyCapability(CAP_DAC_OVERRIDE, CAP_EFFECTIVE, false);
  if (err) {
    std::cerr << "Failed to disable CAP_DAC_OVERRIDE, CAP_EFFECTIVE" <<
                                                                    std::endl;
    return 1;
  }

  // Create a thread to handle signals to shutdown gracefully
  pthread_t sig_listen_thread;
  int thr_ret = pthread_create(&sig_listen_thread, NULL,
                                              ProcessSignalLoop, &rdc_server);

  if (thr_ret) {
    std::cerr <<
    "Failed to create ProcessSignalLoop. pthread_create() returned " <<
                                                                      thr_ret;
    return 1;
  }

  // TODO(cfreehil): Eventually, set these by reading a config file
  rdc_server.set_start_rsmi_service(true);
  rdc_server.set_start_rdc_admin_service(true);

  rdc_server.Run();

  if (sShutDownServer) {
    std::cout <<  "RDC server successfully shut down." << std::endl;
    return 0;
  } else {
    std::cerr << "RDC server failed to start." << std::endl;
    return 1;
  }
}
