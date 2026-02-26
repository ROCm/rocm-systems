#include "detect_virtualization.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
namespace fs = std::filesystem;

// ---------- Utility ----------
static bool contains(const std::string& line, const std::string& token) {
  return line.find(token) != std::string::npos;
}

static bool file_contains(const std::string& path, const std::string& token) {
  std::ifstream file(path);
  if (!file) return false;
  std::string line;
  while (std::getline(file, line)) {
    if (contains(line, token)) return true;
  }
  return false;
}

// ---------- Detect SR-IOV Virtual Function (PCI-level) ----------
bool is_sriov_virtual_function() {
  try {
    if (!fs::exists("/sys/bus/pci/devices")) return false;
    for (const auto& device : fs::directory_iterator("/sys/bus/pci/devices")) {
      if (fs::exists(device.path() / "physfn")) {
        return true;  // This device is a VF
      }
    }
  } catch (...) {
    // Fail open 
    return false;
  }
  return false;
}

// ---------- Detect Hypervisor (VM) ----------
bool is_running_in_vm() {
  try {
    // CPU hypervisor flag
    if (file_contains("/proc/cpuinfo", "hypervisor")) return true;
    // sysfs hypervisor interface
    // if (fs::exists("/sys/hypervisor")) return true;
    // DMI product name
    std::ifstream dmi("/sys/class/dmi/id/product_name");
    if (dmi) {
      std::string line;
      std::getline(dmi, line);
      std::transform(line.begin(), line.end(), line.begin(), ::tolower);
      if (contains(line, "kvm") || contains(line, "vmware") || contains(line, "virtualbox") ||
          contains(line, "hyper-v") || contains(line, "qemu") || contains(line, "xen"))
        return true;
    }
  } catch (...) {
    return false;
  }
  return false;
}

// ---------- Detect Container ----------
bool is_running_in_container() {
  try {
    if (fs::exists("/.dockerenv")) return true;
    if (file_contains("/proc/1/cgroup", "docker") || file_contains("/proc/1/cgroup", "kubepods") ||
        file_contains("/proc/1/cgroup", "containerd"))
      return true;
  } catch (...) {
    return false;
  }
  return false;
}

// ---------- Master Check  ----------
bool is_virtualization_enabled() {
  return is_sriov_virtual_function() || is_running_in_vm();
}
