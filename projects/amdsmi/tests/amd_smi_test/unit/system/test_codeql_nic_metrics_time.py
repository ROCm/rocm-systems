import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]


def extract_block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    position = opening + 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return source[start:position]


PREFIX = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>
std::atomic<unsigned> shared_time_calls{0};
extern "C" tm* __real_localtime(const time_t*);
extern "C" char* __real_asctime(const tm*);
extern "C" char* __real_ctime(const time_t*);
extern "C" tm* __wrap_localtime(const time_t* value) {
  ++shared_time_calls;
  return __real_localtime(value);
}
extern "C" char* __wrap_asctime(const tm* value) {
  ++shared_time_calls;
  return __real_asctime(value);
}
extern "C" char* __wrap_ctime(const time_t* value) {
  ++shared_time_calls;
  return __real_ctime(value);
}
std::string removeNewLines(std::string value) {
  value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
  return value;
}
"""


class CodeqlNicMetricsTimeRegression(unittest.TestCase):
    def run_cpp(self, body: str) -> None:
        with tempfile.TemporaryDirectory(prefix="amdsmi-codeql-nic-metrics-time-") as directory:
            source = Path(directory) / "regression.cc"
            binary = Path(directory) / "regression"
            source.write_text(PREFIX + body, encoding="ascii")
            subprocess.run(
                ["c++", "-std=c++17", "-pthread", str(source), "-o", str(binary),
                 "-Wl,--wrap=localtime,--wrap=asctime,--wrap=ctime"],
                check=True,
            )
            environment = dict(os.environ, TZ="UTC", LC_ALL="C")
            result = subprocess.run(
                [str(binary), str(source)], env=environment, capture_output=True, text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nic_loop_indices_cover_large_containers(self) -> None:
        source = (ROOT / "src/nic/ai-nic/amdsmi_unified/src/smi_nic_interface.cpp").read_text(
            encoding="utf-8"
        )
        function = extract_block(source, "smi_nic_status_t smi_get_nic_rdma_dev_info(")
        loops = []
        for condition in ("j < ibs.size()", "k < ib_ports.size()"):
            position = function.index(condition)
            start = function.rindex("for (", 0, position)
            loops.append(function[start:function.index("{", position)])
        self.run_cpp(r"""
int main() {
  for (const std::size_t count : {0, 1, 255, 256, 300}) {
    for (const std::size_t limit : {4, 16, 512}) {
      const auto SMI_NIC_MAX_RDMA_DEV = limit;
      const auto SMI_NIC_MAX_PORTS = limit;
      std::vector<unsigned> ibs(count), ib_ports(count);
      std::size_t rdma_count = 0;
      std::size_t visits = 0;
""" + loops[0] + r"""{
        assert(j == visits);
        ++visits;
        ++rdma_count;
      }
      assert(visits == std::min(count, limit));
      visits = 0;
""" + loops[1] + r"""{
        assert(k == visits);
        ++visits;
      }
      assert(visits == std::min(count, limit));
    }
  }
}
""")

    def test_file_creation_time_uses_caller_owned_storage(self) -> None:
        source = (ROOT / "rocm_smi/src/rocm_smi_utils.cc").read_text(encoding="utf-8")
        function = extract_block(source, "std::string getFileCreationDate(")
        self.run_cpp(function + r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  struct stat metadata{};
  assert(stat(argv[1], &metadata) == 0);
  tm broken_down{};
  assert(localtime_r(&metadata.st_ctime, &broken_down));
  char expected[26]{};
  assert(asctime_r(&broken_down, expected));
  assert(getFileCreationDate(argv[1]) == removeNewLines(expected));
  assert(getFileCreationDate(std::string(argv[1]) + "/missing").empty());
  if (shared_time_calls != 0) {
    std::cerr << "file creation time uses shared libc time storage\n";
    return 1;
  }
}
""")

    def test_logger_time_uses_caller_owned_storage(self) -> None:
        source = (ROOT / "rocm_smi/src/rocm_smi_logger.cc").read_text(encoding="utf-8")
        function = extract_block(source, "std::string ROCmLogging::Logger::getCurrentTime(")
        self.run_cpp(r"""
namespace ROCmLogging {
class Logger {
 public:
  static std::string getCurrentTime();
};
}
""" + function + r"""
int main() {
  auto check = []() {
    for (unsigned iteration = 0; iteration < 200; ++iteration) {
      const auto text = ROCmLogging::Logger::getCurrentTime();
      assert(text.size() >= 24 && text.size() <= 26);
      assert(text[4] == '-' && text[7] == '-' && text[10] == ' ');
      assert(text[13] == ':' && text[16] == ':' && text[19] == '.');
      assert(text.find('\n') == std::string::npos);
      for (std::size_t position = 20; position < text.size(); ++position) {
        assert(text[position] >= '0' && text[position] <= '9');
      }
    }
  };
  std::vector<std::thread> threads;
  for (unsigned index = 0; index < 8; ++index) threads.emplace_back(check);
  for (auto& thread : threads) thread.join();
  if (shared_time_calls != 0) {
    std::cerr << "logger time uses shared libc time storage\n";
    return 1;
  }
}
""")

    def test_metrics_timestamp_preserves_ctime_newline(self) -> None:
        source = (ROOT / "rocm_smi/src/rocm_smi_gpu_metrics.cc").read_text(encoding="utf-8")
        start = source.index("    const auto timestamp_time =", source.index("auto header_output ="))
        end = source.index('    tmp_outstream_metrics << "Version:', start)
        self.run_cpp(r"""
std::string render_timestamp(std::time_t timestamp) {
  std::ostringstream tmp_outstream_metrics;
  const std::string kDoubleLine = "+==============================+";
  auto timestamp_to_time_point = [](std::time_t value) { return value; };
  const auto m_gpu_metrics_updated_timestamp = timestamp;
""" + source[start:end] + r"""
  return tmp_outstream_metrics.str();
}
int main() {
  auto check = []() {
    for (unsigned iteration = 0; iteration < 200; ++iteration) {
      assert(render_timestamp(0) ==
          "\n+==============================+\n*** GPU Metrics Header: ***\n"
          "Timestamp:  [0] Thu Jan  1 00:00:00 1970\n");
      assert(render_timestamp(86400) ==
          "\n+==============================+\n*** GPU Metrics Header: ***\n"
          "Timestamp:  [86400] Fri Jan  2 00:00:00 1970\n");
    }
  };
  std::vector<std::thread> threads;
  for (unsigned index = 0; index < 8; ++index) threads.emplace_back(check);
  for (auto& thread : threads) thread.join();
  if (shared_time_calls != 0) {
    std::cerr << "metrics timestamp uses shared libc time storage\n";
    return 1;
  }
}
""")


if __name__ == "__main__":
    unittest.main()