// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_vfu_vector_add_test.cpp
/// @brief CTest-integrated smoke test for the rocjitsu vfio-user GPU.
///
/// Orchestrates the full end-to-end test: starts rocjitsu-vfu, launches QEMU,
/// waits for the guest to boot, runs a HIP vector_add program inside the VM,
/// and verifies the result. Registered as a CTest test with LABELS "vfu".
///
/// Prerequisites (set as environment variables or CTest properties):
///   RJ_VFU_BIN        Path to the rocjitsu-vfu binary
///   RJ_VFU_CONFIG     Path to the rocjitsu JSON topology config
///   RJ_VFU_SOCKET     UNIX socket path (default: /tmp/rocjitsu-vfu-0.sock)
///   RJ_GUEST_IMAGE    Path to QCOW2 guest image with ROCm installed
///   RJ_SSH_PORT       SSH forwarding port (default: 2222)
///   RJ_SSH_KEY        SSH private key path (optional)

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <thread>
#include <chrono>

namespace {

std::string env_or(const char *key, const char *fallback) {
  const char *v = std::getenv(key);
  return v ? v : fallback;
}

bool wait_for_port(int port, int timeout_s = 120) {
  using namespace std::chrono_literals;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    // Try connecting via shell command (avoids needing socket headers here).
    std::string cmd = std::format("nc -z 127.0.0.1 {} 2>/dev/null", port);
    if (std::system(cmd.c_str()) == 0)
      return true;
    std::this_thread::sleep_for(2s);
  }
  return false;
}

int ssh_run(int port, const std::string &key, const std::string &cmd) {
  std::string ssh = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
                    " -o ConnectTimeout=10";
  if (!key.empty())
    ssh += " -i " + key;
  ssh += std::format(" -p {} root@localhost ", port);
  ssh += "'" + cmd + "'";
  return std::system(ssh.c_str());
}

} // namespace

class VfuSmokeTest : public ::testing::Test {
protected:
  std::string vfu_bin    = env_or("RJ_VFU_BIN", "rocjitsu-vfu");
  std::string config     = env_or("RJ_VFU_CONFIG", "configs/gfx950_mi350p_kmd.json");
  std::string socket_path = env_or("RJ_VFU_SOCKET", "/tmp/rocjitsu-vfu-0.sock");
  std::string guest_image = env_or("RJ_GUEST_IMAGE", "");
  int         ssh_port   = std::stoi(env_or("RJ_SSH_PORT", "2222"));
  std::string ssh_key    = env_or("RJ_SSH_KEY", "");

  void SetUp() override {
    if (guest_image.empty())
      GTEST_SKIP() << "RJ_GUEST_IMAGE not set; skipping vfio-user integration test";
  }
};

TEST_F(VfuSmokeTest, GuestSeesAmdGpu) {
  // Remove stale socket.
  std::string rm_cmd = "rm -f " + socket_path;
  std::system(rm_cmd.c_str());

  // Start rocjitsu-vfu in the background.
  std::string vfu_cmd = vfu_bin + " --socket " + socket_path + " --config " + config + " &";
  ASSERT_EQ(std::system(vfu_cmd.c_str()), 0) << "Failed to start rocjitsu-vfu";

  // Give vfu server a moment to bind the socket.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Construct vfio-user QEMU device JSON.
  std::string vfio_dev = std::format(
      "{{\"driver\":\"vfio-user-pci\",\"socket\":{{\"path\":\"{}\",\"type\":\"unix\"}}}}",
      socket_path);

  // Launch QEMU with the vfio-user device.
  std::string qemu_cmd = std::format(
      "qemu-system-x86_64 -accel kvm -m 16G -smp 4 "
      "-drive file={},format=qcow2,if=virtio "
      "-netdev user,id=net0,hostfwd=tcp::{}-:22 "
      "-device virtio-net-pci,netdev=net0 "
      "-device '{}' -display none -serial null &",
      guest_image, ssh_port, vfio_dev);
  ASSERT_EQ(std::system(qemu_cmd.c_str()), 0) << "Failed to start QEMU";

  // Wait for SSH port to open.
  ASSERT_TRUE(wait_for_port(ssh_port, 120)) << "Guest SSH did not become ready in 120s";

  // Check 1: amdgpu device visible in lspci.
  EXPECT_EQ(ssh_run(ssh_port, ssh_key, "lspci | grep -q '1002:75c8'"), 0)
      << "MI350P device (1002:75c8) not found in guest lspci";

  // Check 2: amdgpu driver loaded.
  EXPECT_EQ(ssh_run(ssh_port, ssh_key, "dmesg | grep -q 'amdgpu'"), 0)
      << "amdgpu driver not loaded in guest";

  // Check 3: rocminfo sees GPU.
  EXPECT_EQ(ssh_run(ssh_port, ssh_key, "rocminfo | grep -q gfx950"), 0)
      << "rocminfo did not report gfx950 GPU";

  // Check 4: HIP vector_add produces correct results.
  std::string hip_src = R"(
#include <hip/hip_runtime.h>
#include <stdio.h>
__global__ void vadd(float *a, float *b, float *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  const int N = 1024;
  float *da, *db, *dc;
  hipMalloc(&da, N*sizeof(float));
  hipMalloc(&db, N*sizeof(float));
  hipMalloc(&dc, N*sizeof(float));
  float ha[N], hb[N], hc[N];
  for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = N-i; }
  hipMemcpy(da, ha, N*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(db, hb, N*sizeof(float), hipMemcpyHostToDevice);
  vadd<<<N/64,64>>>(da, db, dc, N);
  hipMemcpy(hc, dc, N*sizeof(float), hipMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    if ((int)hc[i] != N) { printf("MISMATCH at %d: %f\n",i,hc[i]); return 1; }
  }
  puts("PASS"); return 0;
}
)";

  // Write the HIP source to a temp file via SSH heredoc.
  std::string write_src = "cat > /tmp/vadd.hip << 'HIPSRC'\n" + hip_src + "\nHIPSRC";
  ssh_run(ssh_port, ssh_key, write_src);

  std::string compile_and_run = "hipcc /tmp/vadd.hip -o /tmp/vadd && /tmp/vadd | grep -q PASS";
  EXPECT_EQ(ssh_run(ssh_port, ssh_key, compile_and_run), 0)
      << "HIP vector_add test failed inside VM";

  // Cleanup: kill background processes.
  std::system("pkill -f rocjitsu-vfu 2>/dev/null; pkill -f qemu-system-x86_64 2>/dev/null");
  std::system(("rm -f " + socket_path).c_str());
}
