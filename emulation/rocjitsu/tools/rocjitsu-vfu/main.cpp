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

#include "rocjitsu/vfu/vfu_server.h"
#include "rocjitsu/vfu/pci_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
  rocjitsu::vfu::VfuServerOptions opts;
  opts.socket_path = "/tmp/rocjitsu-vfu-0.sock";
  opts.config_path = "configs/gfx950_mi350p_kmd.json";
  opts.vram_bar_size = rocjitsu::vfu::BarSizes::kBar0VramDefault;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
      opts.socket_path = argv[++i];
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      opts.config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--rebar") == 0) {
      opts.vram_bar_size = rocjitsu::vfu::BarSizes::kBar0VramFull;
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

  std::fprintf(stderr, "[rocjitsu-vfu] config: %s\n", opts.config_path.c_str());
  std::fprintf(stderr, "[rocjitsu-vfu] socket: %s\n", opts.socket_path.c_str());
  std::fprintf(stderr, "[rocjitsu-vfu] VRAM BAR size: %llu MB\n",
               static_cast<unsigned long long>(opts.vram_bar_size / (1024 * 1024)));

  rocjitsu::vfu::VfuServer server(std::move(opts));
  return server.run();
}
