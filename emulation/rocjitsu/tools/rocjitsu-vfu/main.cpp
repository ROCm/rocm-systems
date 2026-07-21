// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief rocjitsu-vfu: expose rocjitsu as a PCIe AMD GPU via libvfio-user.
///
/// Starts a vfio-user server that QEMU connects to, presenting the rocjitsu
/// GPU emulator as an AMD Instinct MI350P (GFX950) PCIe device inside the VM.
///
/// Usage:
///   rocjitsu-vfu [options]
///
/// Options:
///   --socket <path>   UNIX socket path (default: /tmp/rocjitsu-vfu-0.sock)
///   --config <path>   rocjitsu JSON config (default: gfx950_mi350p_kmd.json)
///   --rebar           Advertise full 144 GB VRAM BAR (requires guest ReBAR support)
///
/// QEMU invocation:
///   qemu-system-x86_64 -accel kvm -m 16G \
///     -device '{"driver":"vfio-user-pci","socket":{"path":"/tmp/rocjitsu-vfu-0.sock","type":"unix"}}'

#include "rocjitsu/vfu/gpu_pci_device_provider.h"
#include "rocjitsu/vfu/pci_config.h"
#include "rocjitsu/kmd/linux/vfio_device_host.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
  std::string socket_path = "/tmp/rocjitsu-vfu-0.sock";
  std::string config_path = "configs/gfx950_mi350p_kmd.json";
  uint64_t vram_bar_size  = rocjitsu::vfu::BarSizes::kBar0VramDefault;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
      socket_path = argv[++i];
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--rebar") == 0) {
      vram_bar_size = rocjitsu::vfu::BarSizes::kBar0VramFull;
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf(
          "Usage: rocjitsu-vfu [--socket PATH] [--config PATH] [--rebar]\n"
          "\n"
          "  --socket PATH   UNIX socket for QEMU vfio-user connection\n"
          "                  (default: /tmp/rocjitsu-vfu-0.sock)\n"
          "  --config PATH   rocjitsu topology JSON config\n"
          "                  (default: configs/gfx950_mi350p_kmd.json)\n"
          "  --rebar         Advertise full 144 GB VRAM BAR instead of 256 MB\n"
          "\n"
          "QEMU invocation:\n"
          "  qemu-system-x86_64 -accel kvm -m 16G \\\n"
          "    -device '{\"driver\":\"vfio-user-pci\","
          "\"socket\":{\"path\":\"/tmp/rocjitsu-vfu-0.sock\",\"type\":\"unix\"}}'\n");
      return 0;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return 1;
    }
  }

  std::fprintf(stderr, "[rocjitsu-vfu] config: %s\n", config_path.c_str());
  std::fprintf(stderr, "[rocjitsu-vfu] socket: %s\n", socket_path.c_str());
  std::fprintf(stderr, "[rocjitsu-vfu] VRAM BAR size: %llu MB\n",
               static_cast<unsigned long long>(vram_bar_size / (1024 * 1024)));

  // Create the rocjitsu VM.
  rj_vm_t *vm_handle = nullptr;
  rj_status_t st = rj_vm_create(config_path.c_str(), RJ_VM_MODE_DAEMON, &vm_handle);
  if (st != ROCJITSU_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-vfu] rj_vm_create failed: %d\n",
                 static_cast<int>(st));
    return 1;
  }

  rocjitsu::SimulatedKfd *kfd =
      vm_handle->vm ? vm_handle->vm->driver() : nullptr;
  if (!kfd) {
    std::fprintf(stderr, "[rocjitsu-vfu] SimulatedKfd not initialized\n");
    rj_vm_destroy(vm_handle);
    return 1;
  }

  uint32_t guest_pid = 0;
  st = rj_vm_device_open(vm_handle, 0, &guest_pid);
  if (st != ROCJITSU_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-vfu] rj_vm_device_open failed: %d\n",
                 static_cast<int>(st));
    rj_vm_destroy(vm_handle);
    return 1;
  }

  // Build the device graph: one GpuPciDevice per GPU.
  rocjitsu::vfu::GpuPciDeviceProvider provider(kfd, guest_pid, vram_bar_size);

  // Stand up one VfioDeviceHost per device, each on its own socket.
  // Today this is a single GPU; adding a NIC means registering another provider.
  int result = 0;
  auto devices = provider.pci_devices();
  for (size_t i = 0; i < devices.size(); ++i) {
    std::string sock = socket_path;
    if (devices.size() > 1)
      sock += "." + std::to_string(i);

    rocjitsu::VfioDeviceHost host(sock, devices[i]);
    result = host.run();
    if (result != 0)
      break;
  }

  rj_vm_device_close(vm_handle, guest_pid);
  rj_vm_destroy(vm_handle);
  return result;
}
