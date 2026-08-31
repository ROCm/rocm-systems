// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/sysfs.h"

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "rocjitsu/kmd/linux/rpc.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_sysfs.h"
RJ_DIAGNOSTIC_POP

#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace fs = std::filesystem;

namespace {

/// @brief Create a process-tagged scratch directory under the XDG runtime root.
/// @details Replaces raw mkdtemp("/tmp/...") so synthetic sysfs trees live under
/// rpc_default_runtime_dir() (tmpfs, reaped at logout) and carry the owning PID
/// in their name (rocjitsu_<kind>_<pid>_XXXXXX) so reap_stale_sysfs_dirs() can
/// remove the trees an earlier SIGKILL/crash orphaned. Returns empty on failure.
/// @note The PID tag makes reaping best-effort, not exact: PID reuse means a
/// recycled-but-live PID can keep a genuinely-orphaned tree from being reaped (a
/// bounded leak until logout clears the tmpfs), and the mkdtemp XXXXXX suffix keeps
/// two same-PID generations from colliding. This matches the launcher reaper
/// (main.cpp reap_stale_runtime_dirs); the tmpfs backstop bounds any leak.
std::string make_tagged_dir(const char *kind) {
  std::error_code ec;
  std::string root = rpc_default_runtime_dir();
  // Fail closed on an empty root: an empty base would make tmpl "/rocjitsu_..._XXXXXX"
  // and try to mkdtemp scratch dirs in the filesystem root. rpc_default_runtime_dir()
  // already treats a set-but-empty $ROCJITSU_RUNTIME_DIR as unset; guard defensively.
  if (root.empty())
    return {};
  // Errors from create_directories are intentionally not checked here: if the root
  // could not be created, mkdtemp below fails and returns empty, which both callers
  // already treat as "no synthetic tree" via an empty-string early return.
  fs::create_directories(root, ec);
  std::string tmpl = root + "/rocjitsu_" + kind + "_" + std::to_string(getpid()) + "_XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char *dir = mkdtemp(buf.data());
  return dir ? std::string(dir) : std::string{};
}

/// @brief Remove orphaned rocjitsu_{drm,topology}_<pid>_* dirs of dead processes.
/// @details RAII cleanup (Sysfs::cleanup) only fires on graceful teardown; a
/// SIGKILL/OOM/crash leaks the scratch tree forever. This sweeps the runtime
/// root once per process, deleting any tagged dir whose embedded PID is no
/// longer alive (kill(pid, 0) == ESRCH). The live process's own dirs are kept.
void reap_stale_sysfs_dirs() {
  // Advance the iterator with an error_code (not the throwing operator++): another
  // rocjitsu process sharing this runtime root may remove_all an entry concurrently,
  // and a throw here would escape std::call_once and abort topology generation on a
  // best-effort cleanup path. Best-effort — any filesystem error just ends the scan.
  std::error_code ec;
  std::string root = rpc_default_runtime_dir();
  // Never scan an empty root: directory_iterator("") walks the CWD, and this loop
  // remove_all's matching entries. rpc_default_runtime_dir() already maps a set-but-
  // empty $ROCJITSU_RUNTIME_DIR to a real path; guard here too since we delete.
  if (root.empty())
    return;
  fs::directory_iterator it(root, ec);
  const fs::directory_iterator end;
  for (; !ec && it != end; it.increment(ec)) {
    const auto &entry = *it;
    // Only reap a real directory, never chase a symlink out of the runtime root:
    // symlink_status() does NOT follow the link, whereas is_directory() would let a
    // symlink-to-directory pass and have remove_all() delete its target. Matches the
    // hardened launcher reaper (main.cpp reap_stale_runtime_dirs).
    std::error_code st_ec;
    auto st = fs::symlink_status(entry.path(), st_ec);
    if (st_ec || st.type() != fs::file_type::directory)
      continue;
    std::string name = entry.path().filename().string();
    if (name.rfind("rocjitsu_drm_", 0) != 0 && name.rfind("rocjitsu_topology_", 0) != 0)
      continue;
    // Extract the PID between the last two underscores: <prefix>_<pid>_<suffix>.
    auto suffix_us = name.rfind('_');
    if (suffix_us == std::string::npos || suffix_us == 0)
      continue;
    auto pid_us = name.rfind('_', suffix_us - 1);
    if (pid_us == std::string::npos)
      continue;
    std::string pid_str = name.substr(pid_us + 1, suffix_us - pid_us - 1);
    pid_t pid = 0;
    auto [ptr, perr] = std::from_chars(pid_str.data(), pid_str.data() + pid_str.size(), pid);
    if (perr != std::errc{} || ptr != pid_str.data() + pid_str.size() || pid <= 0)
      continue;
    if (kill(pid, 0) == -1 && errno == ESRCH) {
      // Use a dedicated error_code for the removal: reusing the loop-control `ec`
      // would let one un-removable orphan (e.g. EACCES on another user's tree)
      // set `ec` and terminate the whole sweep, skipping every remaining stale
      // dir for the life of the process (this runs under std::call_once).
      std::error_code rm_ec;
      fs::remove_all(entry.path(), rm_ec);
    }
  }
}

// The debug-topology derivation (trap-debug capability/capability2/debug_prop
// per GFXIP) lives in kfd_topology.h so the DBG_TRAP GET_DEVICE_SNAPSHOT path
// and this sysfs topology generator share one source of truth.
using kmd::DebugTopology;

} // namespace

Sysfs::~Sysfs() { cleanup(); }

Sysfs::Sysfs(Sysfs &&other) noexcept
    : topology_dir_(std::move(other.topology_dir_)), drm_dir_(std::move(other.drm_dir_)) {
  // Clear both owned-tree paths in the moved-from object so its cleanup() cannot
  // remove_all a tree this object now owns.
  other.topology_dir_.clear();
  other.drm_dir_.clear();
}

Sysfs &Sysfs::operator=(Sysfs &&other) noexcept {
  if (this != &other) {
    cleanup();
    topology_dir_ = std::move(other.topology_dir_);
    drm_dir_ = std::move(other.drm_dir_);
    other.topology_dir_.clear();
    other.drm_dir_.clear();
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
    std::error_code ec;
    fs::remove_all(topology_dir_, ec);
    topology_dir_.clear();
  }
  if (!drm_dir_.empty()) {
    std::error_code ec;
    fs::remove_all(drm_dir_, ec);
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
  // KFD publishes a name file for every topology node.
  write_file(node_dir + "/name", "\n");

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

  const DebugTopology topology = kmd::effective_topology_for(
      gpu.gfx_target_version, gpu.capability, gpu.capability2, gpu.debug_prop, gpu.revision_id);

  uint32_t p2p_links = total_gpus > 1 ? total_gpus - 1 : 0;

  // array_count is the one geometry property KFD reports per node rather than
  // per XCC: node_show() emits node_props.array_count * NUM_XCC, while
  // simd_arrays_per_engine and cu_per_simd_array stay per-XCC. The DBG_TRAP
  // device snapshot passes the unscaled value through instead (kfd_debug.c),
  // which is why the two paths report different numbers for the same property.
  // Normalize once (GpuInfo::effective_num_xcc) so array_count and the num_xcc
  // property cannot disagree.
  const uint32_t num_xcc = gpu.effective_num_xcc();
  const uint32_t node_array_count = gpu.array_count_per_xcc() * num_xcc;

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
        << "array_count " << node_array_count << "\n"
        << "simd_arrays_per_engine " << gpu.effective_arrays_per_engine() << "\n"
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
        << "num_sdma_queues_per_engine " << gpu.num_sdma_queues_per_engine << "\n"
        << "num_cp_queues " << gpu.num_cp_queues << "\n"
        << "max_engine_clk_fcompute " << gpu.max_engine_clk_fcompute << "\n"
        << "max_engine_clk_ccompute 0\n"
        << "local_mem_size " << gpu.local_mem_size << "\n"
        << "fw_version " << gpu.fw_version << "\n"
        << "capability " << topology.capability << "\n"
        << "capability2 " << topology.capability2 << "\n"
        << "debug_prop " << topology.debug_prop << "\n"
        << "sdma_fw_version " << gpu.sdma_fw_version << "\n"
        << "unique_id " << gpu.unique_id << "\n"
        << "num_xcc " << num_xcc << "\n"
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
  drm_dir_ = make_tagged_dir("drm");
  if (drm_dir_.empty())
    return;

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

  static std::once_flag reap_once;
  std::call_once(reap_once, reap_stale_sysfs_dirs);

  topology_dir_ = make_tagged_dir("topology");
  if (topology_dir_.empty())
    return {};

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

Sysfs::GpuInfo gpu_info_from_config(const config::KfdDeviceConfig &dev, uint32_t num_xcc) {
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = dev.gpu_id;
  gpu.gfx_target_version = dev.gfx_target_version;
  gpu.vendor_id = dev.vendor_id;
  gpu.device_id = dev.device_id;
  gpu.family_id = dev.family_id;
  gpu.unique_id = dev.unique_id;
  gpu.location_id = dev.location_id;
  gpu.domain = dev.domain;
  gpu.hive_id = dev.hive_id;
  gpu.capability = dev.capability;
  gpu.capability2 = dev.capability2;
  gpu.debug_prop = dev.debug_prop;
  gpu.drm_render_minor = dev.drm_render_minor;
  gpu.marketing_name = dev.marketing_name;
  gpu.revision_id = dev.revision_id;
  gpu.pci_revision_id = dev.pci_revision_id;
  gpu.simd_count = dev.simd_count;
  gpu.max_waves_per_simd = dev.max_waves_per_simd;
  gpu.num_shader_engines = dev.num_shader_engines;
  gpu.num_shader_arrays_per_engine = dev.num_shader_arrays_per_engine;
  gpu.num_cu_per_sh = dev.num_cu_per_sh;
  gpu.simd_per_cu = dev.simd_per_cu;
  gpu.wave_front_size = dev.wave_front_size;
  gpu.max_slots_scratch_cu = dev.max_slots_scratch_cu;
  gpu.local_mem_size = dev.local_mem_size;
  gpu.vram_type = dev.vram_type;
  gpu.lds_size_kb = dev.lds_size_kb;
  gpu.mem_width = dev.mem_width;
  gpu.mem_clk_max = dev.mem_clk_max;
  gpu.l1_size_kb = dev.l1_size_kb;
  gpu.l1_line_size = dev.l1_line_size;
  gpu.l1_assoc = dev.l1_assoc;
  gpu.l2_size_kb = dev.l2_size_kb;
  gpu.l2_line_size = dev.l2_line_size;
  gpu.l2_assoc = dev.l2_assoc;
  gpu.num_sdma_engines = dev.num_sdma_engines;
  gpu.num_sdma_xgmi_engines = dev.num_sdma_xgmi_engines;
  gpu.num_sdma_queues_per_engine = dev.num_sdma_queues_per_engine;
  gpu.num_cp_queues = dev.num_cp_queues;
  gpu.max_engine_clk_fcompute = dev.max_engine_clk_fcompute;
  gpu.num_xcc = num_xcc;
  return gpu;
}

} // namespace rocjitsu
