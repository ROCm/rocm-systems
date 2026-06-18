// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/sysfs.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unistd.h>

namespace rocjitsu {

namespace fs = std::filesystem;

Sysfs::~Sysfs() { cleanup(); }

Sysfs::Sysfs(Sysfs &&other) noexcept : topology_dir_(std::move(other.topology_dir_)) {
  other.topology_dir_.clear();
}

Sysfs &Sysfs::operator=(Sysfs &&other) noexcept {
  if (this != &other) {
    cleanup();
    topology_dir_ = std::move(other.topology_dir_);
    other.topology_dir_.clear();
  }
  return *this;
}

void Sysfs::write_file(const std::string &path, const std::string &content) {
  std::ofstream f(path);
  f << content;
}

void Sysfs::make_dir(const std::string &path) { fs::create_directories(path); }

void Sysfs::cleanup() {
  if (!topology_dir_.empty()) {
    fs::remove_all(topology_dir_);
    topology_dir_.clear();
  }
  if (!drm_dir_.empty()) {
    fs::remove_all(drm_dir_);
    drm_dir_.clear();
  }
}

void Sysfs::setup_environment() {}

void Sysfs::write_generation_id() { write_file(topology_dir_ + "/generation_id", "1\n"); }

void Sysfs::write_system_properties(uint32_t num_devices) {
  std::ostringstream ss;
  ss << "platform_oem 0\n"
        "platform_id 0\n"
        "platform_rev 0\n"
     << "num_devices " << num_devices << "\n";
  write_file(topology_dir_ + "/system_properties", ss.str());
}

void Sysfs::write_cpu_node(const std::string &nodes_dir, uint32_t num_gpu_links) {
  std::string node_dir = nodes_dir + "/0";
  make_dir(node_dir);
  make_dir(node_dir + "/mem_banks/0");

  for (uint32_t i = 0; i < num_gpu_links; ++i)
    make_dir(node_dir + "/io_links/" + std::to_string(i));

  write_file(node_dir + "/gpu_id", "0\n");

  long nproc = sysconf(_SC_NPROCESSORS_ONLN);
  if (nproc < 1)
    nproc = 1;

  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  uint64_t total_ram = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);

  std::ostringstream props;
  props << "cpu_cores_count " << nproc << "\n"
        << "simd_count 0\n"
        << "mem_banks_count 1\n"
        << "caches_count 0\n"
        << "io_links_count " << num_gpu_links << "\n"
        << "cpu_core_id_base 0\n"
        << "simd_id_base 0\n"
        << "max_waves_per_simd 0\n"
        << "lds_size_in_kb 0\n"
        << "gds_size_in_kb 0\n"
        << "num_gws 0\n"
        << "wave_front_size 0\n"
        << "array_count 0\n"
        << "simd_arrays_per_engine 0\n"
        << "cu_per_simd_array 0\n"
        << "simd_per_cu 0\n"
        << "max_slots_scratch_cu 0\n"
        << "gfx_target_version 0\n"
        << "vendor_id 2\n"
        << "device_id 0\n"
        << "location_id 0\n"
        << "domain 0\n"
        << "drm_render_minor 0\n"
        << "hive_id 0\n"
        << "num_sdma_engines 0\n"
        << "num_sdma_xgmi_engines 0\n"
        << "num_sdma_queues_per_engine 0\n"
        << "num_cp_queues 0\n"
        << "max_engine_clk_fcompute 0\n"
        << "max_engine_clk_ccompute 3000\n"
        << "local_mem_size 0\n"
        << "fw_version 0\n"
        << "capability 0\n"
        << "sdma_fw_version 0\n"
        << "vram_public 0\n"
        << "vram_size 0\n";
  write_file(node_dir + "/properties", props.str());

  std::ostringstream mem;
  mem << "heap_type 0\n"
      << "size_in_bytes " << total_ram << "\n"
      << "flags 0\n"
      << "width 0\n"
      << "mem_clk_max 0\n";
  write_file(node_dir + "/mem_banks/0/properties", mem.str());

  for (uint32_t i = 0; i < num_gpu_links; ++i) {
    std::ostringstream link;
    link << "type 2\n"
         << "version_major 0\n"
         << "version_minor 0\n"
         << "node_from 0\n"
         << "node_to " << (i + 1) << "\n"
         << "weight 20\n"
         << "min_latency 0\n"
         << "max_latency 0\n"
         << "min_bandwidth 0\n"
         << "max_bandwidth 0\n"
         << "recommended_transfer_size 0\n"
         << "num_hops 1\n"
         << "flags 1\n";
    write_file(node_dir + "/io_links/" + std::to_string(i) + "/properties", link.str());
  }
}

void Sysfs::write_gpu_node(const std::string &nodes_dir, uint32_t node_idx, const GpuInfo &gpu,
                           uint32_t total_gpus) {
  std::string node_dir = nodes_dir + "/" + std::to_string(node_idx);
  make_dir(node_dir);
  make_dir(node_dir + "/mem_banks/0");
  make_dir(node_dir + "/caches/0");
  make_dir(node_dir + "/caches/1");

  // IO links: link 0 = to CPU, links 1..N-1 = XGMI to peer GPUs
  uint32_t num_io_links = 1 + (total_gpus > 1 ? total_gpus - 1 : 0);
  for (uint32_t i = 0; i < num_io_links; ++i)
    make_dir(node_dir + "/io_links/" + std::to_string(i));

  std::ostringstream gpu_id;
  gpu_id << gpu.gpu_id << "\n";
  write_file(node_dir + "/gpu_id", gpu_id.str());
  write_file(node_dir + "/name", gpu.marketing_name + "\n");

  uint32_t cap = gpu.capability;
  if (cap == 0) {
    cap = (1u << 1) | (1u << 5) | (1u << 7) | (4u << 8) | (2u << 12) | (1u << 14) | (1u << 15) |
          (1u << 16) | (1u << 17) | (1u << 18) | (1u << 20) | (1u << 21) | (1u << 26) | (1u << 27) |
          (1u << 28) | (1u << 29) | (1u << 30) | (1u << 31);
  }
  const uint32_t asic_revision = gpu.revision_id;
  cap = (cap & ~(0xFu << 22)) | ((asic_revision & 0xFu) << 22);

  uint32_t p2p_links = total_gpus > 1 ? total_gpus - 1 : 0;

  std::ostringstream props;
  props << "cpu_cores_count 0\n"
        << "simd_count " << gpu.simd_count << "\n"
        << "mem_banks_count 1\n"
        << "caches_count 2\n"
        << "io_links_count " << num_io_links << "\n"
        << "p2p_links_count " << p2p_links << "\n"
        << "cpu_core_id_base 0\n"
        << "simd_id_base 2147487744\n"
        << "max_waves_per_simd " << gpu.max_waves_per_simd << "\n"
        << "lds_size_in_kb " << gpu.lds_size_kb << "\n"
        << "gds_size_in_kb 0\n"
        << "num_gws 64\n"
        << "wave_front_size " << gpu.wave_front_size << "\n"
        << "array_count " << gpu.num_shader_engines << "\n"
        << "simd_arrays_per_engine " << gpu.num_shader_arrays_per_engine << "\n"
        << "cu_per_simd_array " << gpu.num_cu_per_sh << "\n"
        << "simd_per_cu " << gpu.simd_per_cu << "\n"
        << "max_slots_scratch_cu " << gpu.max_slots_scratch_cu << "\n"
        << "gfx_target_version " << gpu.gfx_target_version << "\n"
        << "vendor_id " << gpu.vendor_id << "\n"
        << "device_id " << gpu.device_id << "\n"
        << "location_id " << gpu.location_id << "\n"
        << "domain " << gpu.domain << "\n"
        << "drm_render_minor " << gpu.drm_render_minor << "\n"
        << "hive_id " << gpu.hive_id << "\n"
        << "num_sdma_engines " << gpu.num_sdma_engines << "\n"
        << "num_sdma_xgmi_engines " << gpu.num_sdma_xgmi_engines << "\n"
        << "num_sdma_queues_per_engine 2\n"
        << "num_cp_queues " << gpu.num_cp_queues << "\n"
        << "max_engine_clk_fcompute " << gpu.max_engine_clk_fcompute << "\n"
        << "max_engine_clk_ccompute 0\n"
        << "local_mem_size " << gpu.local_mem_size << "\n"
        << "fw_version " << gpu.fw_version << "\n"
        << "capability " << cap << "\n"
        << "capability2 " << gpu.capability2 << "\n"
        << "debug_prop " << gpu.debug_prop << "\n"
        << "sdma_fw_version " << gpu.sdma_fw_version << "\n"
        << "unique_id " << gpu.unique_id << "\n"
        << "num_xcc " << gpu.num_xcc << "\n"
        << "vram_public 1\n"
        << "vram_size " << gpu.local_mem_size << "\n";

  if (gpu.family_id > 0)
    props << "family_id " << gpu.family_id << "\n";

  write_file(node_dir + "/properties", props.str());

  std::ostringstream mem;
  mem << "heap_type 1\n"
      << "size_in_bytes " << gpu.local_mem_size << "\n"
      << "flags 0\n"
      << "width " << gpu.mem_width << "\n"
      << "mem_clk_max " << gpu.mem_clk_max << "\n";
  write_file(node_dir + "/mem_banks/0/properties", mem.str());
  write_file(node_dir + "/mem_banks/0/used_memory", "0\n");

  std::ostringstream l1;
  l1 << "processor_id_low 0\n"
     << "level 1\n"
     << "size " << gpu.l1_size_kb << "\n"
     << "cache_line_size " << gpu.l1_line_size << "\n"
     << "cache_lines_per_tag 1\n"
     << "association " << gpu.l1_assoc << "\n"
     << "latency 0\n"
     << "type 9\n"
     << "sibling_map 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
  write_file(node_dir + "/caches/0/properties", l1.str());

  std::ostringstream l2;
  l2 << "processor_id_low 0\n"
     << "level 2\n"
     << "size " << gpu.l2_size_kb << "\n"
     << "cache_line_size " << gpu.l2_line_size << "\n"
     << "cache_lines_per_tag 1\n"
     << "association " << gpu.l2_assoc << "\n"
     << "latency 0\n"
     << "type 9\n"
     << "sibling_map 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
  write_file(node_dir + "/caches/1/properties", l2.str());

  // IO link 0: GPU → CPU (PCIe, type 2)
  {
    std::ostringstream link;
    link << "type 2\n"
         << "version_major 0\n"
         << "version_minor 0\n"
         << "node_from " << node_idx << "\n"
         << "node_to 0\n"
         << "weight 20\n"
         << "min_latency 0\n"
         << "max_latency 0\n"
         << "min_bandwidth 0\n"
         << "max_bandwidth 0\n"
         << "recommended_transfer_size 0\n"
         << "num_hops 1\n"
         << "flags 1\n";
    write_file(node_dir + "/io_links/0/properties", link.str());
  }

  // IO links 1..N-1: XGMI to peer GPUs (type 11)
  if (total_gpus > 1) {
    uint32_t link_idx = 1;
    for (uint32_t peer = 1; peer <= total_gpus; ++peer) {
      if (peer == node_idx)
        continue;
      std::ostringstream link;
      link << "type 11\n"
           << "version_major 0\n"
           << "version_minor 0\n"
           << "node_from " << node_idx << "\n"
           << "node_to " << peer << "\n"
           << "weight 15\n"
           << "min_latency 0\n"
           << "max_latency 0\n"
           << "min_bandwidth 50000\n"
           << "max_bandwidth 50000\n"
           << "recommended_transfer_size 0\n"
           << "num_hops 1\n"
           << "flags 1\n";
      write_file(node_dir + "/io_links/" + std::to_string(link_idx++) + "/properties", link.str());
    }

    uint32_t p2p_idx = 0;
    for (uint32_t peer = 1; peer <= total_gpus; ++peer) {
      if (peer == node_idx)
        continue;
      make_dir(node_dir + "/p2p_links/" + std::to_string(p2p_idx));
      std::ostringstream plink;
      plink << "type 11\n"
            << "version_major 0\n"
            << "version_minor 0\n"
            << "node_from " << node_idx << "\n"
            << "node_to " << peer << "\n"
            << "weight 15\n"
            << "min_latency 0\n"
            << "max_latency 0\n"
            << "min_bandwidth 50000\n"
            << "max_bandwidth 50000\n"
            << "recommended_transfer_size 0\n"
            << "num_hops 1\n"
            << "flags 1\n";
      write_file(node_dir + "/p2p_links/" + std::to_string(p2p_idx) + "/properties", plink.str());
      ++p2p_idx;
    }
  }
}

void Sysfs::write_drm_tree(const std::vector<GpuInfo> &gpus) {
  char tmpl[] = "/tmp/rocjitsu_drm_XXXXXX";
  char *dir = mkdtemp(tmpl);
  if (!dir)
    return;
  drm_dir_ = dir;

  for (size_t i = 0; i < gpus.size(); ++i) {
    auto &gpu = gpus[i];
    uint32_t render_minor = gpu.drm_render_minor;
    std::string render_name = "renderD" + std::to_string(render_minor);
    std::string card_name = "card" + std::to_string(i);

    std::ostringstream vendor_hex, device_hex, revision_hex;
    vendor_hex << "0x" << std::hex << gpu.vendor_id << "\n";
    device_hex << "0x" << std::hex << gpu.device_id << "\n";
    revision_hex << "0x" << std::hex << std::setw(2) << std::setfill('0') << gpu.pci_revision_id
                 << "\n";

    uint32_t bus = (gpu.location_id >> 8) & 0xFF;
    uint32_t dev = (gpu.location_id >> 3) & 0x1F;
    uint32_t func = gpu.location_id & 0x7;
    std::ostringstream uevent;
    uevent << "DRIVER=amdgpu\n"
           << std::hex << std::uppercase << "PCI_ID=" << std::setw(4) << std::setfill('0')
           << gpu.vendor_id << ":" << std::setw(4) << std::setfill('0') << gpu.device_id << "\n"
           << std::dec << "PCI_SLOT_NAME=" << std::setw(4) << std::setfill('0') << std::hex
           << gpu.domain << ":" << std::setw(2) << std::setfill('0') << bus << ":" << std::setw(2)
           << std::setfill('0') << dev << "." << func << "\n";

    for (const std::string &entry_name : {card_name, render_name}) {
      std::string device_dir = drm_dir_ + "/" + entry_name + "/device";
      make_dir(device_dir + "/drm/" + card_name);
      make_dir(device_dir + "/drm/" + render_name);
      write_file(device_dir + "/vendor", vendor_hex.str());
      write_file(device_dir + "/device", device_hex.str());
      write_file(device_dir + "/uevent", uevent.str());
      // drmParseSubsystemType does readlink("subsystem") then strncmp for "/pci"
      std::filesystem::create_symlink("../../../bus/pci", device_dir + "/subsystem");
      // drmParsePciDeviceInfo reads all five files; missing any causes -ENODEV
      write_file(device_dir + "/revision", revision_hex.str());
      write_file(device_dir + "/subsystem_vendor", vendor_hex.str());
      write_file(device_dir + "/subsystem_device", device_hex.str());
    }
  }

  make_dir(drm_dir_ + "/dev_dri");
  for (size_t i = 0; i < gpus.size(); ++i) {
    auto &gpu = gpus[i];
    write_file(drm_dir_ + "/dev_dri/card" + std::to_string(i), "");
    write_file(drm_dir_ + "/dev_dri/renderD" + std::to_string(gpu.drm_render_minor), "");
  }

  write_file(drm_dir_ + "/version", "drm 1.1.0\n");
}

std::string Sysfs::generate(const GpuInfo &gpu) { return generate(std::vector<GpuInfo>{gpu}); }

std::string Sysfs::generate(const std::vector<GpuInfo> &gpus) {
  cleanup();

  char tmpl[] = "/tmp/rocjitsu_topology_XXXXXX";
  char *dir = mkdtemp(tmpl);
  if (!dir)
    return {};

  topology_dir_ = dir;
  if (!gpus.empty())
    gpu_info_ = gpus[0];

  auto num_gpus = static_cast<uint32_t>(gpus.size());

  write_generation_id();
  write_system_properties(1 + num_gpus);

  std::string nodes_dir = topology_dir_ + "/nodes";
  make_dir(nodes_dir);

  write_cpu_node(nodes_dir, num_gpus);
  for (uint32_t i = 0; i < num_gpus; ++i)
    write_gpu_node(nodes_dir, i + 1, gpus[i], num_gpus);

  write_drm_tree(gpus);

  return topology_dir_;
}

} // namespace rocjitsu
