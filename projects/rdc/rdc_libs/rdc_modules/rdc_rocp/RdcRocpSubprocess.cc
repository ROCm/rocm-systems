/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocp/RdcRocpSubprocess.h"

#include <dlfcn.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

static const std::map<rdc_field_t, const char*> temp_field_map_k = {
    {RDC_FI_PROF_OCCUPANCY_PERCENT, "OccupancyPercent"},
    {RDC_FI_PROF_ACTIVE_CYCLES, "GRBM_GUI_ACTIVE"},
    {RDC_FI_PROF_ACTIVE_WAVES, "SQ_WAVES"},
    {RDC_FI_PROF_ELAPSED_CYCLES, "GRBM_COUNT"},
    {RDC_FI_PROF_TENSOR_ACTIVE_PERCENT, "MfmaUtil"},
    {RDC_FI_PROF_GPU_UTIL_PERCENT, "GPU_UTIL"},
    {RDC_FI_PROF_EVAL_MEM_R_BW, "FETCH_SIZE"},
    {RDC_FI_PROF_EVAL_MEM_W_BW, "WRITE_SIZE"},
    {RDC_FI_PROF_EVAL_FLOPS_16, "TOTAL_16_OPS"},
    {RDC_FI_PROF_EVAL_FLOPS_32, "TOTAL_32_OPS"},
    {RDC_FI_PROF_EVAL_FLOPS_64, "TOTAL_64_OPS"},
    {RDC_FI_PROF_EVAL_FLOPS_16_PERCENT, "RDC_OPS_16_PER_SIMDCYCLE"},
    {RDC_FI_PROF_EVAL_FLOPS_32_PERCENT, "RDC_OPS_32_PER_SIMDCYCLE"},
    {RDC_FI_PROF_EVAL_FLOPS_64_PERCENT, "RDC_OPS_64_PER_SIMDCYCLE"},
    {RDC_FI_PROF_VALU_PIPE_ISSUE_UTIL, "ValuPipeIssueUtil"},
    {RDC_FI_PROF_SM_ACTIVE, "VALUBusy"},
    {RDC_FI_PROF_OCC_PER_ACTIVE_CU, "MeanOccupancyPerActiveCU"},
    {RDC_FI_PROF_OCC_ELAPSED, "GRBM_GUI_ACTIVE"},
    {RDC_FI_PROF_CPC_CPC_STAT_BUSY, "CPC_CPC_STAT_BUSY"},
    {RDC_FI_PROF_CPC_CPC_STAT_IDLE, "CPC_CPC_STAT_IDLE"},
    {RDC_FI_PROF_CPC_CPC_STAT_STALL, "CPC_CPC_STAT_STALL"},
    {RDC_FI_PROF_CPC_CPC_TCIU_BUSY, "CPC_CPC_TCIU_BUSY"},
    {RDC_FI_PROF_CPC_CPC_TCIU_IDLE, "CPC_CPC_TCIU_IDLE"},
    {RDC_FI_PROF_CPC_CPC_UTCL2IU_BUSY, "CPC_CPC_UTCL2IU_BUSY"},
    {RDC_FI_PROF_CPC_CPC_UTCL2IU_IDLE, "CPC_CPC_UTCL2IU_IDLE"},
    {RDC_FI_PROF_CPC_CPC_UTCL2IU_STALL, "CPC_CPC_UTCL2IU_STALL"},
    {RDC_FI_PROF_CPC_ME1_BUSY_FOR_PACKET_DECODE, "CPC_ME1_BUSY_FOR_PACKET_DECODE"},
    {RDC_FI_PROF_CPC_ME1_DC0_SPI_BUSY, "CPC_ME1_DC0_SPI_BUSY"},
    {RDC_FI_PROF_CPC_UTCL1_STALL_ON_TRANSLATION, "CPC_UTCL1_STALL_ON_TRANSLATION"},
    {RDC_FI_PROF_CPC_ALWAYS_COUNT, "CPC_ALWAYS_COUNT"},
    {RDC_FI_PROF_CPC_ADC_VALID_CHUNK_NOT_AVAIL, "CPC_ADC_VALID_CHUNK_NOT_AVAIL"},
    {RDC_FI_PROF_CPC_ADC_DISPATCH_ALLOC_DONE, "CPC_ADC_DISPATCH_ALLOC_DONE"},
    {RDC_FI_PROF_CPC_ADC_VALID_CHUNK_END, "CPC_ADC_VALID_CHUNK_END"},
    {RDC_FI_PROF_CPC_SYNC_FIFO_FULL_LEVEL, "CPC_SYNC_FIFO_FULL_LEVEL"},
    {RDC_FI_PROF_CPC_SYNC_FIFO_FULL, "CPC_SYNC_FIFO_FULL"},
    {RDC_FI_PROF_CPC_GD_BUSY, "CPC_GD_BUSY"},
    {RDC_FI_PROF_CPC_TG_SEND, "CPC_TG_SEND"},
    {RDC_FI_PROF_CPC_WALK_NEXT_CHUNK, "CPC_WALK_NEXT_CHUNK"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE0_SPI, "CPC_STALLED_BY_SE0_SPI"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE1_SPI, "CPC_STALLED_BY_SE1_SPI"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE2_SPI, "CPC_STALLED_BY_SE2_SPI"},
    {RDC_FI_PROF_CPC_STALLED_BY_SE3_SPI, "CPC_STALLED_BY_SE3_SPI"},
    {RDC_FI_PROF_CPC_LTE_ALL, "CPC_LTE_ALL"},
    {RDC_FI_PROF_CPC_SYNC_WRREQ_FIFO_BUSY, "CPC_SYNC_WRREQ_FIFO_BUSY"},
    {RDC_FI_PROF_CPC_CANE_BUSY, "CPC_CANE_BUSY"},
    {RDC_FI_PROF_CPC_CANE_STALL, "CPC_CANE_STALL"},
    {RDC_FI_PROF_CPF_CMP_UTCL1_STALL_ON_TRANSLATION, "CPF_CMP_UTCL1_STALL_ON_TRANSLATION"},
    {RDC_FI_PROF_CPF_CPF_STAT_BUSY, "CPF_CPF_STAT_BUSY"},
    {RDC_FI_PROF_CPF_CPF_STAT_IDLE, "CPF_CPF_STAT_IDLE"},
    {RDC_FI_PROF_CPF_CPF_STAT_STALL, "CPF_CPF_STAT_STALL"},
    {RDC_FI_PROF_CPF_CPF_TCIU_BUSY, "CPF_CPF_TCIU_BUSY"},
    {RDC_FI_PROF_CPF_CPF_TCIU_IDLE, "CPF_CPF_TCIU_IDLE"},
    {RDC_FI_PROF_CPF_CPF_TCIU_STALL, "CPF_CPF_TCIU_STALL"},
    {RDC_FI_PROF_SIMD_UTILIZATION, "SIMD_UTILIZATION"},
};

RdcRocpSubprocess::RdcRocpSubprocess() {
  for (const auto& [k, v] : temp_field_map_k) {
    field_to_metric_.insert({k, v});
  }

  eval_fields_ = {
      RDC_FI_PROF_EVAL_MEM_R_BW,         RDC_FI_PROF_EVAL_MEM_W_BW,
      RDC_FI_PROF_EVAL_FLOPS_16,         RDC_FI_PROF_EVAL_FLOPS_32,
      RDC_FI_PROF_EVAL_FLOPS_64,         RDC_FI_PROF_EVAL_FLOPS_16_PERCENT,
      RDC_FI_PROF_EVAL_FLOPS_32_PERCENT, RDC_FI_PROF_EVAL_FLOPS_64_PERCENT,
  };

  rocpctl_path_ = find_rocpctl_binary();
  RDC_LOG(RDC_DEBUG, "RdcRocpSubprocess initialized, rocpctl at: " << rocpctl_path_);
  RDC_LOG(RDC_DEBUG, "RdcRocpSubprocess supports " << field_to_metric_.size() << " fields");
}

std::string RdcRocpSubprocess::find_rocpctl_binary() const {
  namespace fs = std::filesystem;

  const char* env_path = std::getenv("RDC_ROCPCTL_PATH");
  if (env_path != nullptr && fs::exists(env_path)) {
    return env_path;
  }

  Dl_info dl_info = {};
  if (dladdr(reinterpret_cast<const void*>(&temp_field_map_k), &dl_info) != 0) {
    try {
      fs::path lib_path = fs::canonical(dl_info.dli_fname);
      fs::path candidate = lib_path.parent_path() / "rocpctl";
      if (fs::exists(candidate)) {
        return candidate.string();
      }
      candidate = lib_path.parent_path().parent_path() / "bin" / "rocpctl";
      if (fs::exists(candidate)) {
        return candidate.string();
      }
    } catch (const fs::filesystem_error& e) {
      RDC_LOG(RDC_ERROR, "Failed to resolve rocpctl path: " << e.what());
    }
  }

  const std::vector<std::string> common_paths = {
      "/opt/rocm/bin/rocpctl",
      "/opt/rocm/rdc/bin/rocpctl",
      "/usr/bin/rocpctl",
      "/usr/local/bin/rocpctl",
  };
  for (const auto& p : common_paths) {
    if (fs::exists(p)) {
      return p;
    }
  }

  return "rocpctl";
}

bool RdcRocpSubprocess::is_disabled_on_failure() const {
  return fatal_failure_.load() || consecutive_failures_.load() >= static_cast<int>(kFailThreshold);
}

void RdcRocpSubprocess::inc_failure_count() {
  int count = ++consecutive_failures_;
  if (count == static_cast<int>(kFailThreshold)) {
    RDC_LOG(RDC_ERROR, "rocpctl has failed " << kFailThreshold
                                             << " times consecutively, disabling profiler");
  }
}

void RdcRocpSubprocess::reset_failure_count() {
  consecutive_failures_.store(0);
  fatal_failure_.store(false);
}

void RdcRocpSubprocess::set_fatal_failure() {
  fatal_failure_.store(true);
  RDC_LOG(RDC_ERROR, "rocpctl encountered a fatal failure (core dump/abort), profiler disabled");
}

rdc_status_t RdcRocpSubprocess::parse_json_output(const std::string& json_str,
                                                   RocpctlResult& result) {
  result.gpu_metrics.clear();

  auto find_quoted_value = [](const std::string& s, const std::string& key,
                              size_t start) -> std::pair<std::string, size_t> {
    std::string search_key = "\"" + key + "\"";
    size_t pos = s.find(search_key, start);
    if (pos == std::string::npos) return {"", std::string::npos};
    pos = s.find(':', pos + search_key.size());
    if (pos == std::string::npos) return {"", std::string::npos};
    pos++;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n')) pos++;
    if (pos >= s.size()) return {"", std::string::npos};
    if (s[pos] == '"') {
      size_t end = s.find('"', pos + 1);
      if (end == std::string::npos) return {"", std::string::npos};
      return {s.substr(pos + 1, end - pos - 1), end + 1};
    }
    size_t end = pos;
    while (end < s.size() && s[end] != ',' && s[end] != '}' && s[end] != ']') end++;
    std::string val = s.substr(pos, end - pos);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\n' || val.back() == '\t'))
      val.pop_back();
    return {val, end};
  };

  size_t gm_pos = json_str.find("\"GpuMetrics\"");
  if (gm_pos == std::string::npos) {
    RDC_LOG(RDC_ERROR, "JSON parse error: GpuMetrics key not found");
    return RDC_ST_BAD_PARAMETER;
  }

  size_t arr_start = json_str.find('[', gm_pos);
  if (arr_start == std::string::npos) return RDC_ST_BAD_PARAMETER;

  size_t pos = arr_start + 1;
  while (pos < json_str.size()) {
    size_t obj_start = json_str.find('{', pos);
    if (obj_start == std::string::npos) break;

    int depth = 1;
    size_t obj_end = obj_start + 1;
    while (obj_end < json_str.size() && depth > 0) {
      if (json_str[obj_end] == '{') depth++;
      else if (json_str[obj_end] == '}') depth--;
      if (depth > 0) obj_end++;
    }
    if (depth != 0) break;

    std::string gpu_obj = json_str.substr(obj_start, obj_end - obj_start + 1);
    RocpctlGpuResult gpu_result;

    auto [gpu_id, p1] = find_quoted_value(gpu_obj, "GpuId", 0);
    auto [drm_id, p2] = find_quoted_value(gpu_obj, "DrmRenderId", 0);
    auto [node_id, p3] = find_quoted_value(gpu_obj, "LogicalNodeId", 0);
    gpu_result.gpu_id = gpu_id;
    gpu_result.drm_render_id = drm_id;
    gpu_result.logical_node_id = node_id;

    size_t metrics_pos = gpu_obj.find("\"Metrics\"");
    if (metrics_pos != std::string::npos) {
      size_t m_arr_start = gpu_obj.find('[', metrics_pos);
      if (m_arr_start != std::string::npos) {
        size_t m_pos = m_arr_start + 1;
        while (m_pos < gpu_obj.size()) {
          size_t m_obj_start = gpu_obj.find('{', m_pos);
          if (m_obj_start == std::string::npos) break;
          size_t m_obj_end = gpu_obj.find('}', m_obj_start);
          if (m_obj_end == std::string::npos) break;

          std::string metric_obj = gpu_obj.substr(m_obj_start, m_obj_end - m_obj_start + 1);
          auto [field_name, p4] = find_quoted_value(metric_obj, "Field", 0);
          auto [field_val, p5] = find_quoted_value(metric_obj, "Value", 0);

          if (!field_name.empty() && !field_val.empty()) {
            RocpctlGpuMetric m;
            m.field = field_name;
            try {
              m.value = std::stod(field_val);
            } catch (...) {
              m.value = 0.0;
            }
            gpu_result.metrics.push_back(m);
          }
          m_pos = m_obj_end + 1;
        }
      }
    }

    result.gpu_metrics.push_back(std::move(gpu_result));
    pos = obj_end + 1;
  }

  return result.gpu_metrics.empty() ? RDC_ST_BAD_PARAMETER : RDC_ST_OK;
}

rdc_status_t RdcRocpSubprocess::exec_rocpctl(const std::vector<std::string>& metric_names,
                                              RocpctlResult& result) {
  if (is_disabled_on_failure()) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  std::string cmd = rocpctl_path_;
  cmd += " -d " + std::to_string(kCollectionDurationUs);
  cmd += " -p " + std::to_string(kDefaultPtlDelayMs);
  for (const auto& m : metric_names) {
    cmd += " " + m;
  }

  RDC_LOG(RDC_DEBUG, "Executing rocpctl: " << cmd);

  int pipe_fd[2];
  if (pipe(pipe_fd) == -1) {
    RDC_LOG(RDC_ERROR, "Failed to create pipe: " << strerror(errno));
    inc_failure_count();
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  int stderr_pipe[2];
  if (pipe(stderr_pipe) == -1) {
    close(pipe_fd[0]);
    close(pipe_fd[1]);
    RDC_LOG(RDC_ERROR, "Failed to create stderr pipe: " << strerror(errno));
    inc_failure_count();
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  pid_t pid = fork();
  if (pid == -1) {
    close(pipe_fd[0]);
    close(pipe_fd[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    RDC_LOG(RDC_ERROR, "Failed to fork: " << strerror(errno));
    inc_failure_count();
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  if (pid == 0) {
    // Child process
    close(pipe_fd[0]);
    close(stderr_pipe[0]);
    dup2(pipe_fd[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(pipe_fd[1]);
    close(stderr_pipe[1]);

    setenv("ROCPROFILER_DEVICE_LOCK_AT_START", "1", 1);

    execl("/bin/bash", "bash", "-c", cmd.c_str(), nullptr);
    _exit(127);
  }

  // Parent process
  close(pipe_fd[1]);
  close(stderr_pipe[1]);

  std::string stdout_data;
  std::string stderr_data;
  std::array<char, 4096> buf;
  bool timed_out = false;

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(kSubprocessTimeoutSec);

  // Read stdout with timeout via select()
  while (true) {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(pipe_fd[0], &read_fds);

    auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      timed_out = true;
      break;
    }

    struct timeval tv;
    tv.tv_sec = remaining.count() / 1000000;
    tv.tv_usec = remaining.count() % 1000000;

    int ret = select(pipe_fd[0] + 1, &read_fds, nullptr, nullptr, &tv);
    if (ret <= 0) {
      if (ret == 0) timed_out = true;
      break;
    }

    ssize_t n = read(pipe_fd[0], buf.data(), buf.size());
    if (n <= 0) break;
    stdout_data.append(buf.data(), n);
  }

  // Quick non-blocking read of stderr
  {
    struct timeval tv = {0, 100000};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(stderr_pipe[0], &fds);
    while (select(stderr_pipe[0] + 1, &fds, nullptr, nullptr, &tv) > 0) {
      ssize_t n = read(stderr_pipe[0], buf.data(), buf.size());
      if (n <= 0) break;
      stderr_data.append(buf.data(), n);
      FD_ZERO(&fds);
      FD_SET(stderr_pipe[0], &fds);
      tv = {0, 100000};
    }
  }

  close(pipe_fd[0]);
  close(stderr_pipe[0]);

  if (timed_out) {
    RDC_LOG(RDC_ERROR, "rocpctl timed out after " << kSubprocessTimeoutSec << "s, killing pid "
                                                   << pid);
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    inc_failure_count();
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  int status = 0;
  waitpid(pid, &status, 0);

  if (WIFSIGNALED(status)) {
    int sig = WTERMSIG(status);
    RDC_LOG(RDC_ERROR, "rocpctl killed by signal " << sig);
    if (sig == SIGABRT || sig == SIGSEGV) {
      set_fatal_failure();
    } else {
      inc_failure_count();
    }
    if (!stderr_data.empty()) {
      RDC_LOG(RDC_ERROR, "rocpctl stderr: " << stderr_data);
    }
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    RDC_LOG(RDC_ERROR, "rocpctl exited with code " << exit_code);
    if (!stderr_data.empty()) {
      RDC_LOG(RDC_ERROR, "rocpctl stderr: " << stderr_data);
      if (stderr_data.find("dumped") != std::string::npos ||
          stderr_data.find("aborted") != std::string::npos) {
        set_fatal_failure();
        return RDC_ST_FAIL_LOAD_MODULE;
      }
    }
    inc_failure_count();
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  rdc_status_t parse_status = parse_json_output(stdout_data, result);
  if (parse_status != RDC_ST_OK) {
    RDC_LOG(RDC_ERROR, "Failed to parse rocpctl JSON output");
    inc_failure_count();
    return parse_status;
  }

  reset_failure_count();
  return RDC_ST_OK;
}

rdc_status_t RdcRocpSubprocess::get_or_refresh_cache(const std::vector<std::string>& metrics) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto now = std::chrono::steady_clock::now();
  if (!cached_result_.gpu_metrics.empty() &&
      (now - cached_result_.timestamp) < kCacheTTL) {
    return RDC_ST_OK;
  }

  RocpctlResult result;
  auto start_time = std::chrono::high_resolution_clock::now();
  rdc_status_t exec_st = exec_rocpctl(metrics, result);
  auto stop_time = std::chrono::high_resolution_clock::now();

  if (exec_st != RDC_ST_OK) {
    cached_result_.gpu_metrics.clear();
    return exec_st;
  }

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(stop_time - start_time).count();

  cached_result_.gpu_metrics.clear();
  for (const auto& gpu : result.gpu_metrics) {
    std::map<std::string, double> metric_map;
    for (const auto& m : gpu.metrics) {
      metric_map[m.field] = m.value;
    }
    cached_result_.gpu_metrics[gpu.gpu_id] = metric_map;
  }
  cached_result_.timestamp = now;
  cached_result_.elapsed_time_ms = elapsed_ms;

  return RDC_ST_OK;
}

const char* RdcRocpSubprocess::get_field_id_from_name(rdc_field_t field) {
  auto it = field_to_metric_.find(field);
  if (it == field_to_metric_.end()) {
    return "";
  }
  return it->second;
}

const std::vector<rdc_field_t> RdcRocpSubprocess::get_field_ids() {
  std::vector<rdc_field_t> ids;
  ids.reserve(field_to_metric_.size());
  for (const auto& [k, v] : field_to_metric_) {
    ids.push_back(k);
  }
  return ids;
}

rdc_status_t RdcRocpSubprocess::apply_field_transformation(
    rdc_field_t field, double raw_value, double elapsed_time_ms,
    const std::map<std::string, double>& sampled_values, rdc_field_value_data* output,
    rdc_field_type_t* type) {
  *type = DOUBLE;

  const bool is_eval_field = (eval_fields_.find(field) != eval_fields_.end());

  double divided_dbl = NAN;
  if (is_eval_field) {
    if (elapsed_time_ms != 0.0) {
      divided_dbl = raw_value / elapsed_time_ms;
    } else {
      RDC_LOG(RDC_ERROR, "Elapsed time is zero, cannot divide.");
      return RDC_ST_BAD_PARAMETER;
    }
  }

  switch (field) {
    case RDC_FI_PROF_GPU_UTIL_PERCENT:
      output->dbl = raw_value / 100.0;
      break;

    case RDC_FI_PROF_OCC_ELAPSED: {
      const double active_cycles_val = raw_value;
      if (active_cycles_val != 0.0) {
        auto occ_field_it = field_to_metric_.find(RDC_FI_PROF_OCC_PER_ACTIVE_CU);
        if (occ_field_it != field_to_metric_.end()) {
          auto occ_sampled_it = sampled_values.find(occ_field_it->second);
          if (occ_sampled_it != sampled_values.end()) {
            output->dbl = occ_sampled_it->second / active_cycles_val;
          } else {
            return RDC_ST_BAD_PARAMETER;
          }
        } else {
          return RDC_ST_BAD_PARAMETER;
        }
      } else {
        return RDC_ST_BAD_PARAMETER;
      }
    } break;

    case RDC_FI_PROF_EVAL_FLOPS_16_PERCENT:
      if (!is_eval_field) return RDC_ST_BAD_PARAMETER;
      output->dbl = divided_dbl / 2048.0;
      break;

    case RDC_FI_PROF_EVAL_FLOPS_32_PERCENT:
    case RDC_FI_PROF_EVAL_FLOPS_64_PERCENT:
      if (!is_eval_field) return RDC_ST_BAD_PARAMETER;
      output->dbl = divided_dbl / 256.0;
      break;

    default:
      if (is_eval_field) {
        output->dbl = divided_dbl;
      } else {
        output->dbl = raw_value;
      }
      break;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcRocpSubprocess::rocp_lookup(rdc_gpu_field_t gpu_field,
                                             rdc_field_value_data* data,
                                             rdc_field_type_t* type) {
  if (data == nullptr) return RDC_ST_BAD_PARAMETER;
  *type = DOUBLE;

  auto field_it = field_to_metric_.find(gpu_field.field_id);
  if (field_it == field_to_metric_.end()) {
    return RDC_ST_BAD_PARAMETER;
  }

  std::vector<std::string> metrics = {field_it->second};
  if (gpu_field.field_id == RDC_FI_PROF_OCC_ELAPSED) {
    auto occ_it = field_to_metric_.find(RDC_FI_PROF_OCC_PER_ACTIVE_CU);
    if (occ_it != field_to_metric_.end()) {
      metrics.push_back(occ_it->second);
    }
  }

  rdc_status_t cache_st = get_or_refresh_cache(metrics);
  if (cache_st != RDC_ST_OK) return cache_st;

  std::lock_guard<std::mutex> lock(cache_mutex_);

  std::string gpu_key = std::to_string(gpu_field.gpu_index);
  std::map<std::string, double>* gpu_metrics = nullptr;
  auto it = cached_result_.gpu_metrics.find(gpu_key);
  if (it != cached_result_.gpu_metrics.end()) {
    gpu_metrics = &it->second;
  } else if (gpu_field.gpu_index < cached_result_.gpu_metrics.size()) {
    auto map_it = cached_result_.gpu_metrics.begin();
    std::advance(map_it, gpu_field.gpu_index);
    gpu_metrics = &map_it->second;
  }

  if (gpu_metrics == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  const std::string& metric_name = field_it->second;
  auto val_it = gpu_metrics->find(metric_name);
  if (val_it == gpu_metrics->end()) {
    return RDC_ST_BAD_PARAMETER;
  }

  return apply_field_transformation(gpu_field.field_id, val_it->second,
                                    cached_result_.elapsed_time_ms, *gpu_metrics, data, type);
}

rdc_status_t RdcRocpSubprocess::rocp_lookup_bulk(const std::vector<rdc_gpu_field_t>& fields,
                                                  std::vector<rdc_field_value_data>& values,
                                                  std::vector<rdc_field_type_t>& types,
                                                  std::vector<rdc_status_t>& statuses) {
  if (fields.empty()) return RDC_ST_OK;

  values.resize(fields.size());
  types.resize(fields.size());
  statuses.resize(fields.size());

  std::vector<std::string> metrics_to_sample;
  for (size_t i = 0; i < fields.size(); i++) {
    types[i] = DOUBLE;
    statuses[i] = RDC_ST_OK;

    auto field_it = field_to_metric_.find(fields[i].field_id);
    if (field_it == field_to_metric_.end()) {
      statuses[i] = RDC_ST_BAD_PARAMETER;
      continue;
    }

    const std::string metric_name = field_it->second;
    if (std::find(metrics_to_sample.begin(), metrics_to_sample.end(), metric_name) ==
        metrics_to_sample.end()) {
      metrics_to_sample.push_back(metric_name);
    }

    if (fields[i].field_id == RDC_FI_PROF_OCC_ELAPSED) {
      auto occ_it = field_to_metric_.find(RDC_FI_PROF_OCC_PER_ACTIVE_CU);
      if (occ_it != field_to_metric_.end()) {
        std::string occ_name = occ_it->second;
        if (std::find(metrics_to_sample.begin(), metrics_to_sample.end(), occ_name) ==
            metrics_to_sample.end()) {
          metrics_to_sample.push_back(occ_name);
        }
      }
    }
  }

  rdc_status_t exec_status = get_or_refresh_cache(metrics_to_sample);
  if (exec_status != RDC_ST_OK) {
    for (size_t i = 0; i < fields.size(); i++) {
      statuses[i] = exec_status;
    }
    return exec_status;
  }

  std::lock_guard<std::mutex> lock(cache_mutex_);

  for (size_t i = 0; i < fields.size(); i++) {
    if (statuses[i] != RDC_ST_OK) continue;

    auto field_it = field_to_metric_.find(fields[i].field_id);
    if (field_it == field_to_metric_.end()) continue;

    std::string gpu_key = std::to_string(fields[i].gpu_index);
    std::map<std::string, double>* gpu_metrics = nullptr;
    auto it = cached_result_.gpu_metrics.find(gpu_key);
    if (it != cached_result_.gpu_metrics.end()) {
      gpu_metrics = &it->second;
    } else if (fields[i].gpu_index < cached_result_.gpu_metrics.size()) {
      auto map_it = cached_result_.gpu_metrics.begin();
      std::advance(map_it, fields[i].gpu_index);
      gpu_metrics = &map_it->second;
    }

    if (gpu_metrics == nullptr) {
      statuses[i] = RDC_ST_BAD_PARAMETER;
      continue;
    }

    const std::string& metric_name = field_it->second;
    auto val_it = gpu_metrics->find(metric_name);
    if (val_it == gpu_metrics->end()) {
      statuses[i] = RDC_ST_BAD_PARAMETER;
      continue;
    }

    statuses[i] = apply_field_transformation(fields[i].field_id, val_it->second,
                                             cached_result_.elapsed_time_ms, *gpu_metrics,
                                             &values[i], &types[i]);
  }

  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
