/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "include/amd_cuid.h"
#include "src/hmac.h"
#include "src/ipc_protocol.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

// static hmac instance for daemon
static cuid_hmac daemon_hmac = cuid_hmac();

// Global log file stream
static std::unique_ptr<std::ofstream> g_log_file;
static bool g_logging_to_file = false;

static std::ostream &log_out() {
  if (g_logging_to_file && g_log_file && g_log_file->is_open()) {
    *g_log_file << "timestamp: " << time(nullptr) << ": ";
    return *g_log_file;
  }
  std::cout << "timestamp: " << time(nullptr) << ": ";
  return std::cout;
}

static std::ostream &log_err() {
  if (g_logging_to_file && g_log_file && g_log_file->is_open()) {
    *g_log_file << "timestamp: " << time(nullptr) << ": ";
    return *g_log_file;
  }
  std::cerr << "timestamp: " << time(nullptr) << ": ";
  return std::cerr;
}

static void init_logging(bool enabled) {
  if (enabled) {
    g_log_file =
        std::make_unique<std::ofstream>("/var/log/amdcuid.log", std::ios::app);
    if (g_log_file->is_open()) {
      g_logging_to_file = true;
      // Add timestamp to log entry
      time_t now = time(nullptr);
      *g_log_file << "\n=== Log started at " << ctime(&now);
    }
  }
}

// Daemon Server
class CuidDaemonServer {
public:
  CuidDaemonServer() : is_running_(false), server_fd_(-1) {}
  ~CuidDaemonServer() { stop(); }

  amdcuid_status_t start() {
    if (is_running_) {
      return AMDCUID_STATUS_SUCCESS; // Already running
    }

    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
      return AMDCUID_STATUS_IPC_ERROR;
    }

    unlink(AMDCUID_SOCKET_PATH); // Remove existing socket file

    // bind to socket path
    struct sockaddr_un server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    strncpy(server_addr.sun_path, AMDCUID_SOCKET_PATH,
            sizeof(server_addr.sun_path) - 1);

    if (bind(server_fd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) <
        0) {
      close(server_fd_);
      return AMDCUID_STATUS_IPC_ERROR;
    }

    // Set permissions; only root permissions can access
    chmod(AMDCUID_SOCKET_PATH, 0600);

    if (listen(server_fd_, 5) < 0) {
      close(server_fd_);
      return AMDCUID_STATUS_IPC_ERROR;
    }

    is_running_ = true;
    server_thread_ = std::thread(&CuidDaemonServer::accept_loop, this);

    return AMDCUID_STATUS_SUCCESS;
  }

  void stop() {
    if (!is_running_) {
      return;
    }
    is_running_ = false;
    if (server_fd_ >= 0) {
      shutdown(server_fd_, SHUT_RDWR);
      close(server_fd_);
      server_fd_ = -1;
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    unlink(AMDCUID_SOCKET_PATH);
  }

private:
  std::atomic<bool> is_running_;
  int server_fd_;
  std::thread server_thread_;

  void accept_loop() {
    while (is_running_) {
      int client_fd = accept(server_fd_, nullptr, nullptr);
      if (client_fd < 0) {
        continue;
      }

      handle_client(client_fd);
      close(client_fd);
    }
  }

  void handle_client(int client_fd) {
    IpcRequest request;
    if (recv(client_fd, &request, sizeof(request), 0) != sizeof(request)) {
      return;
    }

    IpcResponse response;
    memset(&response, 0, sizeof(response));

    // Handle different request types
    switch (request.type) {
    case IpcMessageType::ADD_DEVICE:
      response.status = handle_add_device(request, response.device_handle);
      break;
    case IpcMessageType::REFRESH_DEVICES:
      response.status = amdcuid_refresh();
      break;
    default:
      response.status = AMDCUID_STATUS_INVALID_ARGUMENT;
      break;
    }

    send(client_fd, &response, sizeof(response), 0);
  }

  amdcuid_status_t handle_add_device(const IpcRequest &request,
                                     amdcuid_id_t &device_handle) {
    std::string dev_path(request.device_path);
    amdcuid_device_type_t device_type = request.device_type;
    amdcuid_status_t status = amdcuid_get_handle_by_dev_path(
        dev_path.c_str(), device_type, &device_handle);

    return status;
  }
};

int main() {
  // Note: We can't log to file yet until we read the config
  std::cout << "AMD CUID Daemon Starting..." << std::endl;

  if (geteuid() != 0) {
    std::cerr << "Root privileges required to detect relevant devices and "
                 "generate CUID. Exiting"
              << std::endl;
    return 1;
  }

  // if no HMAC key exists, generate and store it
  int fd = open(daemon_hmac.get_key_file_path().c_str(), O_RDONLY);
  if (fd < 0) {
    uint8_t key[32];
    amdcuid_status_t key_status = amdcuid_generate_hash_key(key);
    if (key_status != AMDCUID_STATUS_SUCCESS) {
      log_err() << "Error generating/loading HMAC key (status: "
                << amdcuid_status_to_string(key_status) << ")" << std::endl;
      return 1;
    }
    key_status = amdcuid_set_hash_key(key);
    if (key_status != AMDCUID_STATUS_SUCCESS) {
      log_err() << "Error setting HMAC key for daemon (status: "
                << amdcuid_status_to_string(key_status) << ")" << std::endl;
      return 1;
    }
  } else {
    // The key file exists. Verify size matches the expected key length before
    // proceeding, to avoid using a corrupt key file.
    struct stat key_stat;
    bool stat_ok = (fstat(fd, &key_stat) == 0);
    close(fd);
    if (!stat_ok || key_stat.st_size != key_length) {
      log_err() << "Error: HMAC key file has unexpected size ("
                << (stat_ok ? key_stat.st_size : -1) << " bytes, expected "
                << key_length << "). Key file may be corrupt; "
                << "remove it to allow regeneration." << std::endl;
      return 1;
    }
  }

  // read config file first get logging options and whether to run as a daemon
  // or only on boot
  std::ifstream config_file("/opt/amdcuid/etc/amdcuid_daemon.conf");
  std::vector<std::string> config_lines;

  if (config_file.is_open()) {
    std::string line;
    while (std::getline(config_file, line)) {
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') {
        continue;
      }
      size_t eq = line.find('=');
      if (eq != std::string::npos) {
        config_lines.push_back(line.substr(eq + 1));
      }
    }
    config_file.close();
  } else {
    std::cerr << "Failed to open config file. Exiting." << std::endl;
    return 1;
  }

  if (config_lines.size() < 2) {
    std::cerr << "Insufficient config parameters. Exiting." << std::endl;
    return 1;
  }

  bool logging_enabled = (config_lines[1] == "true");

  // Initialize file logging if enabled
  init_logging(logging_enabled);
  log_out() << "AMD CUID Daemon initialized with logging "
            << (logging_enabled ? "enabled" : "disabled") << std::endl;

  if (config_lines[0] == "true") {
    // in daemon mode, we expect to receive device events via IPC from clients
    // and from udev
    CuidDaemonServer server;
    amdcuid_status_t status = server.start();
    if (status != AMDCUID_STATUS_SUCCESS) {
      log_err() << "Error: Failed to start daemon server (status: "
                << amdcuid_status_to_string(status) << ")" << std::endl;
      return 1;
    }
    log_out() << "Daemon server started, listening for device events..."
              << std::endl;

    // Keep the main thread alive while the server is running
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    // On shutdown (not reachable in current code)
    log_out()
        << "Daemon server stopping, no longer listening for device events."
        << std::endl;
    server.stop();
  } else {
    log_out()
        << "Running in non-daemon mode, generating/updating CUID files once..."
        << std::endl;
    // non-daemon mode discovers devices on bootup and updates their CUIDs once
    // discover devices by refreshing
    amdcuid_status_t status = amdcuid_refresh();
    if (status != AMDCUID_STATUS_SUCCESS) {
      log_err() << "Error: Failed to generate CUID files (status: "
                << amdcuid_status_to_string(status) << ")" << std::endl;
      return 1;
    }

    log_out() << "CUID files generated/updated successfully." << std::endl;

    // get handle count for logging
    uint32_t count = 0;
    amdcuid_id_t dummy[1] = {};
    status = amdcuid_get_all_handles(dummy, &count);

    log_out() << "Total devices with CUIDs: " << count << std::endl;
  }

  log_out() << "AMD CUID Daemon Exiting..." << std::endl;
  return 0;
}
