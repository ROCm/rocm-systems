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
#include "rdc/rdc_server_main.h"

#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <grpcpp/grpcpp.h>
#include <pthread.h>
#include <pwd.h>
#include <sys/capability.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "common/rdc_capabilities.h"
#include "common/rdc_utils.h"
#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc_api_service.h"

// TODO(cfreehil):
// The following need to be made configurable (e.g., from YAML):
//  * location of ssl keys and certificates
//  * which method of authentication to use (with root CA or not)
//  * whether server accepts only unauthenticated connections or only
//    authenticated
//  * rsmi_init flags
//  * which RDC grpc services to start
//  * port on which to listen

static std::atomic<bool> sShutDownServer{false};
static const char* kDaemonName = "rdcd";
static const char* kRDCDHomeDir = "/";
static const char* kDaemonLockFileRoot = "/var/run/rdcd.lock";
static const char* kDaemonLockFile = "/tmp/rdcd.lock";

// Pinned certificates
static const char* kDefaultRDCServerCertPinPath = "/etc/rdc/server/rdc_server.crt";
static const char* kDefaultRDCServerKeyPinPath = "/etc/rdc/server/private/rdc_server.key";
static const char* kDefaultRDCClientCertPinPath = "/etc/rdc/client/rdc_client.crt";

// PKI certificates
static const char* kDefaultRDCServerCertKeyPkiPath = "/etc/rdc/server/private/rdc_server_cert.key";
static const char* kDefaultRDCServerCertPemPkiPath = "/etc/rdc/server/certs/rdc_server_cert.pem";
static const char* kDefaultRDCClientCACertPemPkiPath = "/etc/rdc/client/certs/rdc_cacert.pem";

static const char* kDefaultListenAddress = "0.0.0.0";
static const char* kDefaultListenPort = "50051";
static const uint32_t kRSMIUMask = 027;
// How long a stop waits for RPCs in progress before it cancels them.
static constexpr auto kShutdownGracePeriod = std::chrono::seconds(5);

RDCServer::RDCServer()
    : secure_creds_(false),
      rdc_admin_service_(nullptr),
      api_service_(nullptr),
      stop_requested_(false) {}

RDCServer::~RDCServer() {
  // gRPC services must outlive the server that dispatches to them.
  server_.reset();
  delete rdc_admin_service_;
  delete api_service_;
}

// TODO(cfreehil): resolve here command line options with
// (future) config file options
void RDCServer::Initialize(RdcdCmdLineOpts* cl) {
  cmd_line_ = cl;
  server_address_ = cmd_line_->listen_address;
  server_address_ += ":";
  server_address_ += cmd_line_->listen_port;
  secure_creds_ = !cmd_line_->no_authentication;
  use_pinned_certs_ = cmd_line_->use_pinned_certs;
  log_debug_ = cmd_line_->log_dbg;
}

static int ConstructSSLOptsPin(grpc::SslServerCredentialsOptions* ssl_opts) {
  assert(ssl_opts != nullptr);
  if (ssl_opts == nullptr) {
    return -EINVAL;
  }

  // Ensure the required paths exists before going forward
  if (!amd::rdc::FileExists(kDefaultRDCServerKeyPinPath) ||
      !amd::rdc::FileExists(kDefaultRDCServerCertPinPath) ||
      !amd::rdc::FileExists(kDefaultRDCClientCertPinPath)) {
    return -ENOENT;
  }

  std::string ser_key;
  std::string ser_crt;
  std::string cli_crt;
  int ret;
  ret = amd::rdc::ReadFile(kDefaultRDCServerKeyPinPath, &ser_key);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCServerCertPinPath, &ser_crt);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCClientCertPinPath, &cli_crt);
  if (ret) {
    return ret;
  }

  grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp = {ser_key, ser_crt};
  ssl_opts->client_certificate_request = GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
  ssl_opts->pem_root_certs = cli_crt;
  ssl_opts->pem_key_cert_pairs.push_back(pkcp);
  return 0;
}

static int ConstructSSLOptsPKI(grpc::SslServerCredentialsOptions* ssl_opts) {
  assert(ssl_opts != nullptr);
  if (ssl_opts == nullptr) {
    return -EINVAL;
  }

  // Ensure the required paths exists before going forward
  if (!amd::rdc::FileExists(kDefaultRDCServerCertKeyPkiPath) ||
      !amd::rdc::FileExists(kDefaultRDCServerCertPemPkiPath) ||
      !amd::rdc::FileExists(kDefaultRDCClientCACertPemPkiPath)) {
    return -ENOENT;
  }

  std::string ser_key;
  std::string ser_crt;
  std::string cli_crt;
  int ret;
  ret = amd::rdc::ReadFile(kDefaultRDCServerCertKeyPkiPath, &ser_key);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCServerCertPemPkiPath, &ser_crt);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCClientCACertPemPkiPath, &cli_crt);
  if (ret) {
    return ret;
  }

  grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp = {ser_key, ser_crt};
  ssl_opts->client_certificate_request = GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
  ssl_opts->pem_root_certs = cli_crt;
  ssl_opts->pem_key_cert_pairs.push_back(pkcp);
  return 0;
}

bool RDCServer::Run() {
  ::grpc::ServerBuilder builder;
  int ret;
  if (secure_creds_) {
    grpc::SslServerCredentialsOptions ssl_opts{};

    if (use_pinned_certs_) {
      ret = ConstructSSLOptsPin(&ssl_opts);
    } else {
      ret = ConstructSSLOptsPKI(&ssl_opts);
    }

    if (ret) {
      std::cerr << "Failed to process OpenSSL keys and certificates. Errno: " << -ret << std::endl;
      return false;
    }
    builder.AddListeningPort(server_address_, grpc::SslServerCredentials(ssl_opts));
  } else {
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
  }

  // Register services as the instances through which we'll communicate with
  // clients. These are synchronous services.
  if (start_rdc_admin_service()) {
    rdc_admin_service_ = new amd::rdc::RDCAdminServiceImpl();
    builder.RegisterService(rdc_admin_service_);
  }

  if (start_api_service()) {
    auto* api_service = new amd::rdc::RdcAPIServiceImpl();
    builder.RegisterService(api_service);

    // TODO(bill_liu): pass flags from cnfg file
    rdc_status_t ret = api_service->Initialize(0);
    {
      // Published after Initialize() returns, for ShutDown(); the destructor frees it.
      std::lock_guard<std::mutex> lock(mutex_);
      api_service_ = api_service;
    }

    if (ret != RDC_ST_OK) {
      std::cerr << "Failed to start API service" << std::endl;
      return false;
    }
  }

  // Finally assemble the server. It is null if, for example, the port is in use.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    server_ = builder.BuildAndStart();
    if (!server_) {
      std::cerr << "Failed to listen on " << server_address_ << std::endl;
      return false;
    }
    if (stop_requested_) {
      // Asked to stop while starting.
      server_->Shutdown(std::chrono::system_clock::now() + kShutdownGracePeriod);
    }
  }

  std::cout << "Server listening on " << server_address_.c_str() << std::endl;
  std::cout << "Accepting " << (secure_creds_ ? "Authenticated" : "Unauthenticated")
            << " connections only." << std::endl;
  server_->Wait();
  return true;
}

// Writes "<text><sig>\n" with write(2): a signal handler cannot use iostreams.
static void WriteFromSignalHandler(int fd, const char* text, size_t text_len, int sig) {
  char buf[64];
  char digits[12];  // more than the 10 digits of a 32-bit unsigned int
  size_t len = 0;
  while (len < text_len && len < sizeof(buf) - sizeof(digits) - 1) {
    buf[len] = text[len];
    len++;
  }
  size_t n = 0;
  unsigned int value = static_cast<unsigned int>(sig);
  do {
    digits[n++] = static_cast<char>('0' + value % 10);
    value /= 10;
  } while (value != 0 && n < sizeof(digits));
  while (n > 0) {
    buf[len++] = digits[--n];
  }
  buf[len++] = '\n';
  ssize_t written = write(fd, buf, len);
  (void)written;  // nothing to do about a failed write here
}

static void HandleSignal(int sig) {
  static const char kCaught[] = "Caught signal ";
  WriteFromSignalHandler(STDOUT_FILENO, kCaught, sizeof(kCaught) - 1, sig);

  // For most signals, we will want to exit, so make that the default case
  // Handle the other signals specifically.
  switch (sig) {
    case SIGINT:
    case SIGTERM:
      if (sShutDownServer) {
        // A second request, or a stop that is stuck: exit now.
        _exit(1);
      }
      sShutDownServer = true;
      break;

      // SIGHUP (restart) is not handled: gRPC does not restart a server well.

    default:
      static const char kUnexpected[] = "Unexpected signal caught: ";
      WriteFromSignalHandler(STDERR_FILENO, kUnexpected, sizeof(kUnexpected) - 1, sig);
  }
}

static void InitializeSignalHandling(void) {
  // signal(SIGHUP, HandleSignal);
  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);
}

// The name of a user; the number if it has no (usable) passwd entry.
static std::string UserName(uid_t uid) {
  struct passwd pw;
  struct passwd* result = nullptr;
  long buf_size = sysconf(_SC_GETPW_R_SIZE_MAX);  // a hint; entries can be larger
  std::vector<char> buf(buf_size > 0 ? buf_size : 1024);
  int ret;

  while ((ret = getpwuid_r(uid, &pw, buf.data(), buf.size(), &result)) == ERANGE &&
         buf.size() < (1 << 20)) {
    buf.resize(buf.size() * 2);
  }
  return (ret == 0 && result) ? pw.pw_name : std::to_string(uid);
}

static int FileOwner(const char* fn, std::string* owner) {
  struct stat info;
  int ret;

  assert(owner);
  if (owner == nullptr) {
    return EINVAL;
  }
  ret = stat(fn, &info);
  if (ret) {
    perror("Failed to stat lock file");
    return errno;
  }
  *owner = UserName(info.st_uid);
  return 0;
}

// Stops serving; the destructor frees the server and its services.
void RDCServer::ShutDown(void) {
  std::lock_guard<std::mutex> lock(mutex_);
  stop_requested_ = true;
  // Either may be missing if the server is still starting or failed to start.
  if (api_service_) {
    api_service_->Shutdown();
  }
  if (server_) {
    server_->Shutdown(std::chrono::system_clock::now() + kShutdownGracePeriod);
  }
}

static void* ProcessSignalLoop(void* server_ptr) {
  assert(server_ptr != nullptr);
  RDCServer* server = reinterpret_cast<RDCServer*>(server_ptr);

  while (!sShutDownServer) {
    usleep(100 * 1000);
  }
  std::cout << "Shutting down RDC Server." << std::endl;
  server->ShutDown();
  // We will need to add shutdown of any completion queues
  // here, when/if we add them
  return nullptr;
}

static bool FileIsLocked(std::string fn) {
  struct flock fl;
  int fh;
  int ret;

  auto close_fh = [&]() {
    ret = close(fh);
    if (ret) {
      perror(fn.c_str());
    }
  };

  (void)memset(&fl, 0, sizeof(struct flock));
  errno = 0;
  fh = open(fn.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);  // O_NONBLOCK: a FIFO does not block
  if (fh == -1) {
    if (errno == ENOENT || errno == EACCES) {
      // No previous lock, or one this user cannot see: it does not stop this rdcd.
      return false;
    }
    perror(("Failed to open lock file " + fn).c_str());
    exit(1);
  }

  ret = fcntl(fh, F_GETLK, &fl);
  if (ret) {
    perror("Failed to get file-lock status. Is rdcd already running?");
    exit(1);
  }

  if (fl.l_type == F_UNLCK) {
    close_fh();
    return false;
  }

  close_fh();
  return true;
}

[[noreturn]] static void ExitAlreadyRunning(const char* lock_file) {
  std::cerr << "File " << lock_file << " is locked. Is rdcd already running?" << std::endl;
  exit(1);
}

// A root rdcd takes kDaemonLockFileRoot. Any other rdcd takes kDaemonLockFile, which any local
// user can create, so a root rdcd does not look at it; another user's rdcd stops for a running
// root rdcd whose lock file it can read.
static void ExitIfAlreadyRunning(bool is_root) {
  const char* lock_fn = is_root ? kDaemonLockFileRoot : kDaemonLockFile;
  int lock_fh;
  ssize_t fsz;

  if (!is_root && FileIsLocked(kDaemonLockFileRoot)) {
    ExitAlreadyRunning(kDaemonLockFileRoot);
  }

  // Temporarily adjust file-mask to create file with right permissions
  umask(023);
  // Never through a symbolic link (ELOOP), never a directory (EISDIR) or socket (ENXIO), and only
  // a regular file of rdcd's own user with no other name: rdcd writes into it, and a hard link
  // could name another file of that user.
  lock_fh = open(lock_fn, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0644);
  bool refused = lock_fh < 0 && (errno == ELOOP || errno == EISDIR || errno == ENXIO);
  struct stat info;
  if (lock_fh >= 0 && (fstat(lock_fh, &info) != 0 || !S_ISREG(info.st_mode) ||
                       info.st_uid != geteuid() || info.st_nlink != 1)) {
    refused = true;
  }
  if (refused) {
    std::cerr << "Lock file " << lock_fn << " must be a regular file owned by "
              << UserName(geteuid())
              << " with no other link; if no rdcd uses it, remove it (as its owner or root), "
                 "then start rdcd again."
              << std::endl;
    exit(1);
  }

  if (lock_fh < 0) {
    std::string user;
    int ret = FileOwner(lock_fn, &user);
    if (ret) {
      std::cerr << "Failed to determine owner of lock file " << lock_fn << ": " << strerror(ret)
                << std::endl;
      exit(1);
    }
    std::cerr << "Failed to open file lock:" << lock_fn << " owned by user: " << user
              << ". If starting rdcd as a different user, delete this "
                 "lock-file first."
              << std::endl;
    exit(1);
  }

  if (lockf(lock_fh, F_TLOCK, 0) < 0) {
    ExitAlreadyRunning(lock_fn);
  }
  umask(kRSMIUMask);

  std::string pid_str = std::to_string(getpid());
  // Best effort: if this fails, digits of an earlier, longer pid may follow this one.
  if (ftruncate(lock_fh, 0) != 0) {
    perror(("Failed to truncate lock file " + std::string(lock_fn)).c_str());
  }
  fsz = write(lock_fh, pid_str.c_str(), pid_str.size());
  assert(static_cast<unsigned int>(fsz) == pid_str.size());
}

static void MakeDaemon(bool is_root) {
  int fd0;
  struct rlimit max_files;

  // RSMI, for one thing, will need to be able to read/write files
  // Note that umask turns *off* permission for a given bit, so you we want
  // the complement of the permissions we want files to have.
  umask(kRSMIUMask);

  // To Do; Make this optional based on CL option. By default, don't do this.
  // Instead rely on serviced to make it a daemon.
#if 0
  pid_t pid;
  // We want to disassociate with calling process, so fork, and let
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
    std::cerr << "Failed to change directory to " << kRDCDHomeDir << std::endl;
  }

  // Determine max. number of open files possible. We need to close all
  // open descriptors.
  if (getrlimit(RLIMIT_NOFILE, &max_files) < 0) {
    std::cerr << kDaemonName << ": can't get file limit" << std::endl;
    max_files.rlim_max = 1024;
  }

  // Close files; leave stdin, stdout and stderr open
  if (max_files.rlim_max > 1024) {
    max_files.rlim_max = 1024;
  }
  for (uint32_t i = 3; i < max_files.rlim_max; i++) {
    close(i);
  }

  // Direct stdin to /dev/null, whatever descriptor 0 was (closed, or in use).
  fd0 = open("/dev/null", O_RDWR);
  if (fd0 < 0 || (fd0 != 0 && dup2(fd0, 0) < 0)) {
    perror("Failed to open /dev/null as stdin");
    exit(1);
  }
  if (fd0 != 0) {
    close(fd0);
  }

  // After the close loop: closing a descriptor of the lock file would drop the lock.
  ExitIfAlreadyRunning(is_root);

  InitializeSignalHandling();
}

// Group arguments by type, in this order:
//  * required_argument
//  * optional_argument
//  * no_argument
static const struct option long_options[] = {{"address", required_argument, nullptr, 'a'},
                                             {"port", required_argument, nullptr, 'p'},
                                             // Any options with optionals args would go here; e.g.,
                                             // {"start_rdcd", optional_argument, nullptr, 'd'},
                                             {"unauth_comm", no_argument, nullptr, 'u'},
                                             {"pinned_cert", no_argument, nullptr, 'i'},
                                             {"debug", no_argument, nullptr, 'd'},
                                             {"version", no_argument, nullptr, 'v'},
                                             {"help", no_argument, nullptr, 'h'},

                                             {nullptr, 0, nullptr, 0}};
static const char* short_options = "a:p:uidvh";

static void PrintHelp(void) {
  std::cout << "Optional rdcd Arguments:\n"
               "--address, -a <IPv4 address> specify address on which to listen; "
               "default is 0.0.0.0\n"
               "--port, -p <port> specify port on which to listen; "
               "default is to listen on port 50051\n"
               "--unauth_comm, -u don't do authentication with communications"
               " with client. When this flag is not specified, by default, "
               "PKI authentication is used\n"
               "--pinned_cert, -i used \"pinned\" certificates instead of PKI "
               "authentication. This is for test purposes.\n"
               "--debug, -d output debug messages\n"
               "--version, -v Output version information\n"
               "--help, -h print this message\n";
}

uint32_t ProcessCmdline(RdcdCmdLineOpts* cmdl_opts, int arg_cnt, char** arg_list) {
  int a;
  int ind = -1;

  assert(cmdl_opts != nullptr);

  while (true) {
    a = getopt_long(arg_cnt, arg_list, short_options, long_options, &ind);

    if (a == -1) {
      break;
    }

    switch (a) {
      case 'a':
        if (!amd::rdc::IsIP(optarg)) {
          std::cerr << "\"" << optarg << "\" is not a valid IP address." << std::endl;
          return -1;
        }
        cmdl_opts->listen_address = optarg;
        break;

      case 'p':
        if (!amd::rdc::IsNumber(optarg)) {
          std::cerr << "\"" << optarg << "\" is not a valid port number." << std::endl;
          return -1;
        }
        cmdl_opts->listen_port = optarg;
        break;

      case 'u':
        cmdl_opts->no_authentication = true;
        break;

      case 'i':
        cmdl_opts->use_pinned_certs = true;
        break;

      case 'd':
        cmdl_opts->log_dbg = true;
        break;

      case 'v':
#ifdef CURRENT_GIT_HASH
        std::cout << "RDCD : " << RDC_SERVER_VERSION_STRING << "+" << QUOTE(CURRENT_GIT_HASH)
                  << std::endl;
#else
        std::cout << "RDCD : " << RDC_SERVER_VERSION_STRING << std::endl;
#endif
        exit(0);

      case 'h':
        PrintHelp();
        exit(0);

      default:  // getopt_long() has named the unknown option or the missing argument
        PrintHelp();
        return -1;
    }
  }

  // Check for incompatibilities
  if (cmdl_opts->use_pinned_certs && cmdl_opts->no_authentication) {
    std::cerr << "--pinned_cert and --unauth_comm are incompatible options." << std::endl;
    return -1;
  }
  return 0;
}

static void init_cmd_line_opts(RdcdCmdLineOpts* opts) {
  assert(opts != nullptr);
  opts->listen_address = kDefaultListenAddress;
  opts->listen_port = kDefaultListenPort;
  opts->no_authentication = false;
  opts->use_pinned_certs = false;
  opts->log_dbg = false;
}

int main(int argc, char** argv) {
  RDCServer rdc_server;
  RdcdCmdLineOpts cmd_line_opts;
  int err;
  uid_t caller_id = geteuid();

  bool is_root = (caller_id == 0);

  init_cmd_line_opts(&cmd_line_opts);
  auto result = ProcessCmdline(&cmd_line_opts, argc, argv);

  if (result != 0) {
    std::cerr << "Error occurred during ProcessCmdline." << std::endl;
    return 1;
  }

  // Can read the certificates and private key when authentication.
  if (!cmd_line_opts.no_authentication) {
    if (cmd_line_opts.use_pinned_certs && (access(kDefaultRDCServerCertPinPath, R_OK) != 0 ||
                                           access(kDefaultRDCServerKeyPinPath, R_OK) != 0 ||
                                           access(kDefaultRDCClientCertPinPath, R_OK) != 0)) {
      std::cerr << "The user needs read access to the pinned "
                << "certificates and private key." << std::endl;
      return 1;
    }

    if (!cmd_line_opts.use_pinned_certs && (access(kDefaultRDCServerCertKeyPkiPath, R_OK) != 0 ||
                                            access(kDefaultRDCServerCertPemPkiPath, R_OK) != 0 ||
                                            access(kDefaultRDCClientCACertPemPkiPath, R_OK) != 0)) {
      std::cerr << "The user needs read access to the PKI "
                << "certificates and private key." << std::endl;
      return 1;
    }
  }

  MakeDaemon(is_root);

  rdc_server.Initialize(&cmd_line_opts);

  bool cap_enabled;

  err = ::amd::rdc::GetCapability(CAP_DAC_OVERRIDE, CAP_EFFECTIVE, &cap_enabled);
  if (err) {
    std::cerr << "Failed to get capability" << std::endl;
    return 1;
  }

  if (cap_enabled) {
    err = amd::rdc::GetCapability(CAP_DAC_OVERRIDE, CAP_PERMITTED, &cap_enabled);
    if (err) {
      std::cerr << "Failed to get capability" << std::endl;
      return 1;
    }
    if (!cap_enabled) {
      std::cerr << "CAP_DAC_OVERRIDE CAP_PERMITTED is not enabled" << std::endl;
    }
  } else {
    std::cerr << "CAP_DAC_OVERRIDE CAP_EFFECTIVE is not enabled." << std::endl;
  }
  if (!cap_enabled) {
    std::cerr << "rdcd functionality is limited to read-only" << std::endl;
  }

  // Don't allow rwx access to all files to ever be inheritable. We may need
  // relax this restriction if some new feature requires it.
  err = amd::rdc::ModifyCapability(CAP_DAC_OVERRIDE, CAP_INHERITABLE, false);
  if (err) {
    std::cerr << "Failed to disable CAP_DAC_OVERRIDE, CAP_INHERITABLE" << std::endl;
    return 1;
  }

  // By default, disable CAP_DAC_OVERRIDE. Turn on, when needed.
  err = amd::rdc::ModifyCapability(CAP_DAC_OVERRIDE, CAP_EFFECTIVE, false);
  if (err) {
    std::cerr << "Failed to disable CAP_DAC_OVERRIDE, CAP_EFFECTIVE" << std::endl;
    return 1;
  }

  // Create a thread to handle signals to shutdown gracefully
  pthread_t sig_listen_thread;
  int thr_ret = pthread_create(&sig_listen_thread, NULL, ProcessSignalLoop, &rdc_server);

  if (thr_ret) {
    std::cerr << "Failed to create ProcessSignalLoop. pthread_create() returned " << thr_ret;
    return 1;
  }

  // TODO(cfreehil): Eventually, set these by reading a config file
  rdc_server.set_start_rdc_admin_service(true);
  rdc_server.set_start_api_service(true);

  bool started = rdc_server.Run();
  if (!started) {
    std::cerr << "RDC server failed to start." << std::endl;
  }

  // The signal thread stops the server, also after a failed start; wait for it.
  // rdc_server's destructor then frees the server and its services.
  sShutDownServer = true;
  pthread_join(sig_listen_thread, nullptr);

  if (!started) {
    return 1;
  }
  std::cout << "RDC server successfully shut down." << std::endl;
  return 0;
}
