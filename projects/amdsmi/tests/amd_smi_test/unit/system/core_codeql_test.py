# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[4]
SOURCE = ROOT / "src/amd_smi/amd_smi.cc"


def source_slice(start: str, end: str) -> str:
    source = SOURCE.read_text(encoding="utf-8")
    return source[source.index(start) : source.index(end, source.index(start))]


def compile_and_run(program: str) -> None:
    with tempfile.TemporaryDirectory(prefix="amdsmi-agent-core-codeql-") as temporary:
        executable = pathlib.Path(temporary) / "regression"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(ROOT / "include"),
                "-x",
                "c++",
                "-",
                "-o",
                str(executable),
            ],
            input=program,
            text=True,
            check=True,
        )
        result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


class CoreCodeqlTests(unittest.TestCase):
    def test_initramfs_arguments_preserve_bounds_without_execution(self) -> None:
        arguments = source_slice("    int dev_null = open(", "    _exit(1);").replace(
            "strncpy(", "banned_copy("
        )
        compile_and_run(
            r"""
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

static unsigned banned_calls;
static std::string executed_path;
static std::vector<std::string> executed_arguments;
[[maybe_unused]] static char* banned_copy(char* destination, const char* source, size_t count) {
  ++banned_calls;
  return std::strncpy(destination, source, count);
}
static int disabled_open(const char*, int) { return -1; }
static int capture_exec(const char* path, char* const arguments[]) {
  executed_path = path;
  executed_arguments.clear();
  for (size_t index = 0; arguments[index] != nullptr; ++index) {
    assert(index < 3);
    executed_arguments.emplace_back(arguments[index]);
  }
  return -1;
}
struct Tool {
  const char* path;
  const char* arg1;
  const char* arg2;
};
#define open disabled_open
#define execv capture_exec
static void construct_arguments(const Tool* selected) {
"""
            + arguments
            + r"""
}
#undef open
#undef execv

int main() {
  const std::string long_path(300, 'p');
  const std::string long_argument(30, 'a');
  const Tool cases[] = {{"/usr/bin/dracut", "-f", nullptr},
                        {"/usr/sbin/update-initramfs", "-u", nullptr},
                        {"/usr/bin/mkinitcpio", "-P", nullptr},
                        {"/usr/bin/dracut", "-f", "--force"},
                        {long_path.c_str(), long_argument.c_str(), long_argument.c_str()}};
  for (const auto& tool : cases) {
    construct_arguments(&tool);
    assert(executed_path == tool.path);
    assert(executed_arguments.size() == (tool.arg2 == nullptr ? 2 : 3));
    assert(executed_arguments[0] == std::string(tool.path).substr(0, 255));
    assert(executed_arguments[1] == std::string(tool.arg1).substr(0, 15));
    if (tool.arg2 != nullptr)
      assert(executed_arguments[2] == std::string(tool.arg2).substr(0, 15));
  }
  if (banned_calls != 0) {
    std::cerr << "legacy initramfs argument copy executed " << banned_calls << " times\n";
    return 1;
  }
}
"""
        )

    def test_carveout_descriptions_preserve_truncation_and_padding(self) -> None:
        helper = (
            source_slice(
                "static amdsmi_status_t get_gpu_uma_carveout_info_internal(",
                "amdsmi_status_t amdsmi_get_gpu_uma_carveout_info(",
            )
            .replace("std::ifstream", "FixtureInputStream")
            .replace("strncpy(", "banned_copy(")
        )
        compile_and_run(
            r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include "amd_smi/amdsmi.h"

static std::string options;
static bool missing_current;
static bool missing_options;
static unsigned banned_calls;
[[maybe_unused]] static char* banned_copy(char* destination, const char* source, size_t count) {
  ++banned_calls;
  return std::strncpy(destination, source, count);
}
struct FixtureInputStream : std::istringstream {
  explicit FixtureInputStream(const std::string& path)
      : std::istringstream(path.find("carveout_options") != std::string::npos ? options : "2\n") {
    if (path.find("carveout_options") != std::string::npos ? missing_options : missing_current)
      setstate(std::ios::failbit);
  }
  void close() {}
};
namespace amd { namespace smi {
struct AMDSmiGPUDevice {
  std::string get_gpu_path() { return "card0"; }
};
}}
"""
            + helper
            + r"""
int main() {
  amd::smi::AMDSmiGPUDevice device;
  for (const std::string& description : {std::string(), std::string("Minimum (512 MB)"),
       std::string("name\0ignored", 12), std::string(255, 'p'), std::string(256, 'q'),
       std::string(1024, 'r')}) {
    options = "bad line\ninvalid: ignored\n0:" + description + "\n";
    amdsmi_uma_carveout_info_t info;
    std::memset(&info, 0xa5, sizeof(info));
    assert(get_gpu_uma_carveout_info_internal(&device, &info) == AMDSMI_STATUS_SUCCESS);
    assert(info.current_index == 2 && info.num_options == 1);
    std::array<char, AMDSMI_MAX_STRING_LENGTH> expected{};
    const auto copied = std::min(description.find('\0'),
                                std::min(description.size(), expected.size() - 1));
    std::copy_n(description.data(), copied, expected.data());
    assert(std::memcmp(info.options[0].description, expected.data(), expected.size()) == 0);
    assert(info.options[0].index == 0);
  }
  amdsmi_uma_carveout_info_t info{};
  assert(get_gpu_uma_carveout_info_internal(nullptr, &info) == AMDSMI_STATUS_INVAL);
  assert(get_gpu_uma_carveout_info_internal(&device, nullptr) == AMDSMI_STATUS_INVAL);
  missing_current = true;
  assert(get_gpu_uma_carveout_info_internal(&device, &info) == AMDSMI_STATUS_NOT_SUPPORTED);
  missing_current = false;
  missing_options = true;
  assert(get_gpu_uma_carveout_info_internal(&device, &info) == AMDSMI_STATUS_FILE_ERROR);
  if (banned_calls != 0) {
    std::cerr << "legacy carveout string copy executed " << banned_calls << " times\n";
    return 1;
  }
}
"""
        )

    def test_pcie_speed_does_not_scan_unused_unbounded_unit(self) -> None:
        parsing = source_slice(
            "  std::string path_max_link_speed =", "  // pcie speed in sysfs returns in GT/s"
        ).replace("fscanf(", "tracked_scan(")
        compile_and_run(
            r"""
#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include "amd_smi/amdsmi.h"

static std::string fixture;
static unsigned unbounded_scans;
static FILE* open_fixture(const char*, const char*) {
  return fmemopen(fixture.data(), fixture.size(), "r");
}
static int tracked_scan(FILE* stream, const char* format, double* speed) {
  if (std::string(format).find("%*s") != std::string::npos) ++unbounded_scans;
  return std::fscanf(stream, format, speed);
}
struct Device {
  std::string get_gpu_path() { return "card0"; }
};
#define fopen open_fixture
#define LOG_ERROR(message) do {} while (0)
static amdsmi_status_t parse_speed(double& pcie_speed) {
  FILE* fp;
  Device device;
  auto* gpu_device = &device;
  std::ostringstream ss;
"""
            + parsing
            + r"""
  return AMDSMI_STATUS_SUCCESS;
}
#undef fopen
#undef LOG_ERROR

int main() {
  for (const char* valid : {"16.0 GT/s PCIe", "16", " 16.0 ", "16.0ignored"}) {
    fixture = valid;
    double speed = 0;
    assert(parse_speed(speed) == AMDSMI_STATUS_SUCCESS);
    assert(speed == 16.0);
  }
  for (const char* invalid : {" ", "not-a-speed GT/s"}) {
    fixture = invalid;
    double speed = 0;
    assert(parse_speed(speed) == AMDSMI_STATUS_API_FAILED);
  }
  if (unbounded_scans != 0) {
    std::cerr << "unused unbounded unit scan executed " << unbounded_scans << " times\n";
    return 1;
  }
}
"""
        )

    def test_process_names_preserve_bounded_string_semantics(self) -> None:
        helper = source_slice(
            "static void merge_proc_into_pid_map(",
            "amdsmi_status_t amdsmi_get_gpu_process_list_by_pid(",
        ).replace("std::strncpy", "banned_copy")
        compile_and_run(
            r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include "amd_smi/amdsmi.h"

static unsigned banned_calls;
[[maybe_unused]] static char* banned_copy(char* destination, const char* source, size_t count) {
  ++banned_calls;
  return std::strncpy(destination, source, count);
}
"""
            + helper
            + r"""
int main() {
  for (const std::string& input : {std::string(), std::string("worker"),
       std::string("proc\0junk", 9), std::string(255, 'p'), std::string(256, 'q')}) {
    amdsmi_proc_info_t process{};
    std::memset(process.name, 0x5a, sizeof(process.name));
    std::memset(process.container_name, 0x5a, sizeof(process.container_name));
    std::memcpy(process.name, input.data(), input.size());
    std::memcpy(process.container_name, input.data(), input.size());
    if (input.size() < sizeof(process.name)) {
      process.name[input.size()] = '\0';
      process.container_name[input.size()] = '\0';
    }
    process.pid = 42;
    process.mem = 1234;
    std::map<uint32_t, amdsmi_proc_info_by_pid_t> processes;
    merge_proc_into_pid_map(processes, 3, process);
    std::array<char, AMDSMI_MAX_STRING_LENGTH> expected{};
    const auto copied = std::min(input.find('\0'), std::min(input.size(), expected.size() - 1));
    std::copy_n(input.data(), copied, expected.data());
    auto& result = processes.at(process.pid);
    assert(std::memcmp(result.name, expected.data(), expected.size()) == 0);
    assert(std::memcmp(result.container_name, expected.data(), expected.size()) == 0);
    assert(result.pid == 42 && result.num_gpus == 1);
    assert(result.gpus[0].gpu_index == 3 && result.gpus[0].mem == 1234);
    process.name[0] = 'z';
    merge_proc_into_pid_map(processes, 4, process);
    assert(std::memcmp(result.name, expected.data(), expected.size()) == 0);
    assert(result.num_gpus == 2 && result.gpus[1].gpu_index == 4);
  }
  if (banned_calls != 0) {
    std::cerr << "legacy bounded string copy executed " << banned_calls << " times\n";
    return 1;
  }
}
"""
        )

    def test_ttm_override_contract_and_trusted_provenance(self) -> None:
        helpers = source_slice(
            "static std::string ttm_module_name()", "amdsmi_status_t amdsmi_get_ttm_info("
        )
        tagged_helpers = (
            helpers.replace("std::string", "TaggedString")
            .replace("ttm_module_name", "tagged_ttm_module_name")
            .replace("ttm_modprobe_name", "tagged_ttm_modprobe_name")
        )
        compile_and_run(
            r"""
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

static std::string loaded_module;
static int probe_count;
static int probe_access(const char* path, int) {
  ++probe_count;
  return std::string(path) == "/sys/module/" + loaded_module ? 0 : -1;
}

struct TaggedString : std::string {
  bool tainted;
  explicit TaggedString(const char* environment) : std::string(environment), tainted(true) {}
  template <size_t length>
  TaggedString(const char (&literal)[length]) : std::string(literal), tainted(false) {}
  bool operator==(const char* other) const { return compare(other) == 0; }
};

#define access probe_access
"""
            + helpers
            + tagged_helpers
            + r"""
#undef access

int main() {
  for (const char* module : {"ttm", "amdttm", "amd_ttm"}) {
    assert(setenv("AMDSMI_TTM_SYSFS_NAME", module, 1) == 0);
    probe_count = 0;
    assert(ttm_module_name() == module);
    const auto tagged = tagged_ttm_module_name();
    if (tagged.tainted) {
      std::cerr << "environment provenance escaped for " << module << '\n';
      return 1;
    }
    assert(probe_count == 0);
    const std::string expected = std::string(module) == "amd_ttm" ? "amd-ttm" : module;
    assert(ttm_modprobe_name(ttm_module_name()) == expected);
    assert(tagged_ttm_modprobe_name(tagged) == expected.c_str());
    assert(!tagged_ttm_modprobe_name(tagged).tainted);
  }
  for (const char* invalid : {"", "../ttm", "amd-ttm", "TTM", "ttm/../amdttm", "ttm\n"}) {
    assert(setenv("AMDSMI_TTM_SYSFS_NAME", invalid, 1) == 0);
    for (const char* detected : {"amdttm", "amd_ttm", "ttm"}) {
      loaded_module = detected;
      assert(ttm_module_name() == detected);
      const auto tagged = tagged_ttm_module_name();
      assert(tagged == detected);
      assert(!tagged.tainted);
    }
  }
  assert(unsetenv("AMDSMI_TTM_SYSFS_NAME") == 0);
  loaded_module.clear();
  assert(ttm_module_name() == "ttm");
  assert(!tagged_ttm_module_name().tainted);
}
"""
        )


if __name__ == "__main__":
    unittest.main()
