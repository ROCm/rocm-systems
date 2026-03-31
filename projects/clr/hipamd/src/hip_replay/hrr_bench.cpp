/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

// hrr-bench: Benchmark and reproduce tool for HRR trace archives.
//
// Subcommands:
//   list      - List all kernels in a capture
//   kernel    - Benchmark a single kernel (N iterations)
//   app       - Replay full trace with timing
//   repro     - Reproduce crashes/NaN with diagnostics
//   compare   - Compare two captures
//   stress    - Stress test a single kernel
//   export    - Export kernel as standalone .hip + CMakeLists.txt
//
// Usage: hrr-bench <subcommand> <capture.hrr> [options]

#include "hrr_reader.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t err = (call);                                                  \
    if (err != hipSuccess) {                                                  \
      fprintf(stderr, "[HRR] HIP error %d (%s) at %s:%d\n", err,             \
              hipGetErrorString(err), __FILE__, __LINE__);                    \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// --- Shared replay infrastructure ---
struct KernelSetup {
  hipFunction_t func = nullptr;
  hipModule_t module = nullptr;
  const hrr::KernelLaunchEvent* kl = nullptr;
  size_t event_index = 0;

  // Live GPU buffers for this kernel's args
  std::unordered_map<uint64_t, void*> alloc_map;
  std::unordered_map<uint64_t, size_t> alloc_sizes;
  std::vector<void*> arg_ptrs;
  std::vector<std::vector<uint8_t>> arg_storage;
};

static void* translate_ptr(KernelSetup& ks, uint64_t handle) {
  auto it = ks.alloc_map.find(handle);
  if (it != ks.alloc_map.end()) return it->second;
  for (auto& [h, ptr] : ks.alloc_map) {
    auto sz_it = ks.alloc_sizes.find(h);
    if (sz_it != ks.alloc_sizes.end() &&
        handle >= h && handle < h + sz_it->second) {
      return static_cast<char*>(ptr) + (handle - h);
    }
  }
  return nullptr;
}

// Set up a kernel for isolated execution: allocate buffers, restore inputs
static int setup_kernel(const hrr::Archive& archive, size_t kernel_idx,
                        KernelSetup& ks) {
  // Find the kernel_idx-th KERNEL_LAUNCH event
  size_t count = 0;
  for (size_t i = 0; i < archive.events.size(); i++) {
    if (archive.events[i].header.event_type == hrr::EVENT_KERNEL_LAUNCH &&
        archive.events[i].kernel_launch) {
      if (count == kernel_idx) {
        ks.kl = archive.events[i].kernel_launch;
        ks.event_index = i;
        break;
      }
      count++;
    }
  }

  if (!ks.kl) {
    fprintf(stderr, "[HRR] Kernel index %zu not found\n", kernel_idx);
    return 1;
  }

  // Replay all MALLOC events up to this kernel to build alloc map
  for (size_t i = 0; i <= ks.event_index; i++) {
    const auto& ev = archive.events[i];
    if (ev.header.event_type == hrr::EVENT_MALLOC) {
      void* ptr = nullptr;
      HIP_CHECK(hipMalloc(&ptr, ev.malloc_ev.size));
      ks.alloc_map[ev.malloc_ev.ptr_handle] = ptr;
      ks.alloc_sizes[ev.malloc_ev.ptr_handle] = ev.malloc_ev.size;
    } else if (ev.header.event_type == hrr::EVENT_FREE) {
      auto it = ks.alloc_map.find(ev.malloc_ev.ptr_handle);
      if (it != ks.alloc_map.end()) {
        hipFree(it->second);
        ks.alloc_map.erase(it);
        ks.alloc_sizes.erase(ev.malloc_ev.ptr_handle);
      }
    }
  }

  // Load code objects and find the kernel function
  for (auto& [hex, path] : archive.code_objects) {
    std::vector<uint8_t> co_data;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) continue;
    fseek(f, 0, SEEK_END);
    co_data.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    fread(co_data.data(), 1, co_data.size(), f);
    fclose(f);

    hipModule_t mod = nullptr;
    if (hipModuleLoadData(&mod, co_data.data()) == hipSuccess) {
      hipFunction_t func = nullptr;
      if (hipModuleGetFunction(&func, mod, ks.kl->kernel_name.c_str()) ==
          hipSuccess && func) {
        ks.func = func;
        ks.module = mod;
        break;
      }
      hipModuleUnload(mod);
    }
  }

  if (!ks.func) {
    fprintf(stderr, "[HRR] Kernel '%s' not found in code objects\n",
            ks.kl->kernel_name.c_str());
    return 1;
  }

  // Build arg pointers
  for (const auto& arg : ks.kl->args) {
    if (arg.value_kind == 2) continue;  // hidden
    ks.arg_storage.emplace_back();
    auto& storage = ks.arg_storage.back();
    if (arg.value_kind == 1 && arg.data.size() >= 8) {
      uint64_t handle;
      memcpy(&handle, arg.data.data(), 8);
      void* live_ptr = translate_ptr(ks, handle);
      storage.resize(sizeof(void*));
      memcpy(storage.data(), &live_ptr, sizeof(void*));
    } else {
      storage = arg.data;
    }
    ks.arg_ptrs.push_back(storage.data());
  }

  return 0;
}

// Restore input buffer snapshots from blob store
static void restore_inputs(const hrr::Archive& archive, KernelSetup& ks) {
  if (!ks.kl) return;
  for (const auto& snap : ks.kl->snapshots) {
    if (snap.direction == 0) {
      void* dst = translate_ptr(ks, snap.ptr_handle);
      if (dst) {
        std::vector<uint8_t> blob;
        if (hrr::read_blob(archive, snap.hash_lo, snap.hash_hi, blob)) {
          hipMemcpy(dst, blob.data(), snap.length, hipMemcpyHostToDevice);
        }
      }
    }
  }
}

static void cleanup_kernel(KernelSetup& ks) {
  for (auto& [h, ptr] : ks.alloc_map) hipFree(ptr);
  if (ks.module) hipModuleUnload(ks.module);
}

// --- Subcommands ---

static int cmd_list(const hrr::Archive& archive) {
  printf("%-4s %-50s %-15s %-15s %-6s %-8s\n",
         "ID", "Kernel", "Grid", "Block", "ShMem", "Args");
  printf("%-4s %-50s %-15s %-15s %-6s %-8s\n",
         "--", "------", "----", "-----", "-----", "----");

  size_t kid = 0;
  for (const auto& ev : archive.events) {
    if (ev.header.event_type != hrr::EVENT_KERNEL_LAUNCH || !ev.kernel_launch)
      continue;
    const auto& kl = *ev.kernel_launch;
    char grid[32], block[32];
    snprintf(grid, sizeof(grid), "[%u,%u,%u]",
             kl.grid[0], kl.grid[1], kl.grid[2]);
    snprintf(block, sizeof(block), "[%u,%u,%u]",
             kl.block[0], kl.block[1], kl.block[2]);
    std::string name = kl.kernel_name;
    if (name.size() > 50) name = name.substr(0, 47) + "...";
    printf("%-4zu %-50s %-15s %-15s %-6u %-8zu\n",
           kid, name.c_str(), grid, block, kl.shared_mem, kl.args.size());
    kid++;
  }
  return 0;
}

static int cmd_kernel(const hrr::Archive& archive, size_t kernel_id,
                      int iterations, int warmup) {
  KernelSetup ks;
  int ret = setup_kernel(archive, kernel_id, ks);
  if (ret != 0) return ret;

  const auto& kl = *ks.kl;
  printf("Kernel: %s\n", kl.kernel_name.c_str());
  printf("Grid: [%u,%u,%u]  Block: [%u,%u,%u]  SharedMem: %u\n",
         kl.grid[0], kl.grid[1], kl.grid[2],
         kl.block[0], kl.block[1], kl.block[2], kl.shared_mem);
  printf("Iterations: %d  Warmup: %d\n\n", iterations, warmup);

  // Warmup
  for (int i = 0; i < warmup; i++) {
    restore_inputs(archive, ks);
    hipModuleLaunchKernel(ks.func,
                          kl.grid[0], kl.grid[1], kl.grid[2],
                          kl.block[0], kl.block[1], kl.block[2],
                          kl.shared_mem, nullptr,
                          ks.arg_ptrs.data(), nullptr);
    hipDeviceSynchronize();
  }

  // Timed iterations
  std::vector<float> times(iterations);
  for (int i = 0; i < iterations; i++) {
    restore_inputs(archive, ks);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    hipModuleLaunchKernel(ks.func,
                          kl.grid[0], kl.grid[1], kl.grid[2],
                          kl.block[0], kl.block[1], kl.block[2],
                          kl.shared_mem, nullptr,
                          ks.arg_ptrs.data(), nullptr);

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&times[i], start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
  }

  // Statistics
  std::sort(times.begin(), times.end());
  float sum = std::accumulate(times.begin(), times.end(), 0.0f);
  float mean = sum / iterations;
  float median = times[iterations / 2];
  float p95 = times[static_cast<size_t>(iterations * 0.95)];
  float p99 = times[static_cast<size_t>(iterations * 0.99)];

  printf("Min:     %.3f ms\n", times.front());
  printf("Median:  %.3f ms\n", median);
  printf("Mean:    %.3f ms\n", mean);
  printf("P95:     %.3f ms\n", p95);
  printf("P99:     %.3f ms\n", p99);
  printf("Max:     %.3f ms\n", times.back());
  printf("Throughput: %.1f kernel/s\n", 1000.0f / mean);

  cleanup_kernel(ks);
  return 0;
}

static int cmd_repro(const hrr::Archive& archive, bool check_nan) {
  // Full replay looking for errors
  printf("Replaying %zu events...\n", archive.event_count);

  // Simplified replay (reuse replay_event from hrr_replay would be ideal,
  // but we inline a minimal version here)
  std::unordered_map<uint64_t, void*> allocs;
  std::unordered_map<uint64_t, size_t> sizes;
  std::unordered_map<std::string, hipModule_t> modules;

  for (size_t i = 0; i < archive.events.size(); i++) {
    const auto& ev = archive.events[i];

    switch (ev.header.event_type) {
      case hrr::EVENT_MALLOC: {
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, ev.malloc_ev.size);
        if (err != hipSuccess) {
          printf("Event %zu: MALLOC size=%llu -> %s\n",
                 i, (unsigned long long)ev.malloc_ev.size,
                 hipGetErrorString(err));
          return 1;
        }
        allocs[ev.malloc_ev.ptr_handle] = ptr;
        sizes[ev.malloc_ev.ptr_handle] = ev.malloc_ev.size;
        break;
      }

      case hrr::EVENT_FREE: {
        auto it = allocs.find(ev.malloc_ev.ptr_handle);
        if (it != allocs.end()) {
          hipFree(it->second);
          allocs.erase(it);
        }
        break;
      }

      case hrr::EVENT_MEMCPY: {
        if (ev.memcpy_ev.kind == 1 && ev.memcpy_ev.hash_lo != 0) {
          std::vector<uint8_t> blob;
          if (hrr::read_blob(archive, ev.memcpy_ev.hash_lo,
                             ev.memcpy_ev.hash_hi, blob)) {
            auto it = allocs.find(ev.memcpy_ev.dst_addr);
            // Try direct lookup or range
            void* dst = nullptr;
            if (it != allocs.end()) dst = it->second;
            if (dst) {
              hipMemcpy(dst, blob.data(), ev.memcpy_ev.size,
                        hipMemcpyHostToDevice);
            }
          }
        }
        break;
      }

      case hrr::EVENT_KERNEL_LAUNCH: {
        if (!ev.kernel_launch) break;
        const auto& kl = *ev.kernel_launch;

        // Find kernel
        hipFunction_t func = nullptr;
        for (auto& [hex, mod] : modules) {
          if (hipModuleGetFunction(&func, mod, kl.kernel_name.c_str()) ==
              hipSuccess && func) break;
          func = nullptr;
        }

        if (!func) {
          // Try loading code objects
          for (auto& [hex, path] : archive.code_objects) {
            if (modules.count(hex)) continue;
            std::vector<uint8_t> co;
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            co.resize(ftell(f));
            fseek(f, 0, SEEK_SET);
            fread(co.data(), 1, co.size(), f);
            fclose(f);
            hipModule_t mod = nullptr;
            if (hipModuleLoadData(&mod, co.data()) == hipSuccess) {
              modules[hex] = mod;
              if (hipModuleGetFunction(&func, mod, kl.kernel_name.c_str()) ==
                  hipSuccess && func) break;
              func = nullptr;
            }
          }
        }

        if (!func) {
          printf("Event %zu: KERNEL_LAUNCH '%s' -> function not found\n",
                 i, kl.kernel_name.c_str());
          break;
        }

        // Restore snapshots
        for (const auto& snap : kl.snapshots) {
          if (snap.direction == 0) {
            auto it = allocs.find(snap.ptr_handle);
            if (it != allocs.end()) {
              std::vector<uint8_t> blob;
              if (hrr::read_blob(archive, snap.hash_lo, snap.hash_hi, blob)) {
                hipMemcpy(it->second, blob.data(), snap.length,
                          hipMemcpyHostToDevice);
              }
            }
          }
        }

        // Build args
        std::vector<void*> arg_ptrs;
        std::vector<std::vector<uint8_t>> arg_store;
        for (const auto& arg : kl.args) {
          if (arg.value_kind == 2) continue;
          arg_store.emplace_back();
          auto& s = arg_store.back();
          if (arg.value_kind == 1 && arg.data.size() >= 8) {
            uint64_t handle;
            memcpy(&handle, arg.data.data(), 8);
            auto it = allocs.find(handle);
            void* p = it != allocs.end() ? it->second : nullptr;
            s.resize(sizeof(void*));
            memcpy(s.data(), &p, sizeof(void*));
          } else {
            s = arg.data;
          }
          arg_ptrs.push_back(s.data());
        }

        hipError_t err = hipModuleLaunchKernel(
            func, kl.grid[0], kl.grid[1], kl.grid[2],
            kl.block[0], kl.block[1], kl.block[2],
            kl.shared_mem, nullptr, arg_ptrs.data(), nullptr);

        if (err != hipSuccess) {
          printf("Event %zu: KERNEL_LAUNCH '%s' -> %s\n",
                 i, kl.kernel_name.c_str(), hipGetErrorString(err));
          return 1;
        }

        // NaN/Inf check
        if (check_nan) {
          hipDeviceSynchronize();
          for (const auto& arg : kl.args) {
            if (arg.value_kind != 1 || arg.data.size() < 8) continue;
            uint64_t handle;
            memcpy(&handle, arg.data.data(), 8);
            auto it = allocs.find(handle);
            if (it == allocs.end()) continue;
            auto sz_it = sizes.find(handle);
            if (sz_it == sizes.end()) continue;

            size_t buf_sz = sz_it->second;
            std::vector<uint8_t> host(buf_sz);
            hipMemcpy(host.data(), it->second, buf_sz,
                      hipMemcpyDeviceToHost);

            // Scan for NaN/Inf (treat as float32)
            size_t num_f32 = buf_sz / 4;
            const float* fp = reinterpret_cast<const float*>(host.data());
            for (size_t j = 0; j < num_f32; j++) {
              if (std::isnan(fp[j]) || std::isinf(fp[j])) {
                printf("Event %zu: KERNEL_LAUNCH '%s' output buffer "
                       "(handle=0x%llx) contains %s at offset %zu\n",
                       i, kl.kernel_name.c_str(),
                       (unsigned long long)handle,
                       std::isnan(fp[j]) ? "NaN" : "Inf", j * 4);
                break;
              }
            }
          }
        }
        break;
      }

      case hrr::EVENT_DEVICE_SYNC:
        hipDeviceSynchronize();
        break;

      default:
        break;
    }
  }

  printf("Replay complete. No errors detected.\n");
  for (auto& [h, p] : allocs) hipFree(p);
  for (auto& [h, m] : modules) hipModuleUnload(m);
  return 0;
}

static int cmd_export(const hrr::Archive& archive, size_t kernel_id,
                      const std::string& output_dir, bool safe_mode) {
  // Find the kernel
  size_t count = 0;
  const hrr::KernelLaunchEvent* kl = nullptr;
  for (const auto& ev : archive.events) {
    if (ev.header.event_type == hrr::EVENT_KERNEL_LAUNCH && ev.kernel_launch) {
      if (count == kernel_id) {
        kl = ev.kernel_launch;
        break;
      }
      count++;
    }
  }

  if (!kl) {
    fprintf(stderr, "Kernel %zu not found\n", kernel_id);
    return 1;
  }

  fs::create_directories(output_dir);

  // Copy code object
  for (auto& [hex, path] : archive.code_objects) {
    fs::copy_file(path, output_dir + "/kernel.hsaco",
                  fs::copy_options::overwrite_existing);
    break;  // Use first code object (could be smarter)
  }

  // Write input buffers
  std::mt19937 rng(42);
  size_t buf_idx = 0;
  for (const auto& snap : kl->snapshots) {
    std::vector<uint8_t> blob;
    if (!hrr::read_blob(archive, snap.hash_lo, snap.hash_hi, blob)) continue;

    if (safe_mode) {
      // Randomize non-zero bytes, preserve zeros
      for (auto& b : blob) {
        if (b != 0) b = static_cast<uint8_t>(rng() & 0xFF);
      }
    }

    std::string fname = output_dir + "/" +
        (snap.direction == 0 ? "input_" : "expected_output_") +
        std::to_string(buf_idx) + ".bin";
    FILE* f = fopen(fname.c_str(), "wb");
    if (f) {
      fwrite(blob.data(), 1, blob.size(), f);
      fclose(f);
    }
    buf_idx++;
  }

  // Generate repro.hip
  {
    std::ofstream hip(output_dir + "/repro.hip");
    hip << "// Auto-generated kernel reproducer from HRR trace\n"
        << "// Kernel: " << kl->kernel_name << "\n"
        << "// Grid: [" << kl->grid[0] << "," << kl->grid[1] << ","
        << kl->grid[2] << "]\n"
        << "// Block: [" << kl->block[0] << "," << kl->block[1] << ","
        << kl->block[2] << "]\n\n"
        << "#include <hip/hip_runtime.h>\n"
        << "#include <cstdio>\n"
        << "#include <cstdlib>\n"
        << "#include <cstring>\n"
        << "#include <fstream>\n"
        << "#include <vector>\n\n"
        << "#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess)"
        << " { fprintf(stderr, \"HIP error %d: %s\\n\", e, hipGetErrorString(e"
        << ")); exit(1); } } while(0)\n\n"
        << "std::vector<char> read_file(const char* path) {\n"
        << "  std::ifstream f(path, std::ios::binary);\n"
        << "  return {std::istreambuf_iterator<char>(f), {}};\n"
        << "}\n\n"
        << "int main() {\n"
        << "  HIP_CHECK(hipInit(0));\n\n"
        << "  // Load code object\n"
        << "  auto co = read_file(\"kernel.hsaco\");\n"
        << "  hipModule_t mod;\n"
        << "  HIP_CHECK(hipModuleLoadData(&mod, co.data()));\n\n"
        << "  hipFunction_t func;\n"
        << "  HIP_CHECK(hipModuleGetFunction(&func, mod, \""
        << kl->kernel_name << "\"));\n\n";

    // Allocate buffers and load input data
    buf_idx = 0;
    size_t arg_idx = 0;
    for (const auto& arg : kl->args) {
      if (arg.value_kind == 2) continue;
      if (arg.value_kind == 1 && arg.data.size() >= 8) {
        uint64_t handle;
        memcpy(&handle, arg.data.data(), 8);
        // Find size from snapshots
        size_t buf_size = 0;
        for (const auto& snap : kl->snapshots) {
          if (snap.ptr_handle == handle) {
            buf_size = snap.length;
            break;
          }
        }
        if (buf_size == 0) buf_size = 4096;  // fallback

        hip << "  // Arg " << arg_idx << ": pointer (buffer " << buf_idx
            << ", " << buf_size << " bytes)\n"
            << "  void* buf_" << buf_idx << ";\n"
            << "  HIP_CHECK(hipMalloc(&buf_" << buf_idx << ", "
            << buf_size << "));\n";

        // Check if there's an input snapshot for this buffer
        bool has_input = false;
        for (const auto& snap : kl->snapshots) {
          if (snap.ptr_handle == handle && snap.direction == 0) {
            has_input = true;
            break;
          }
        }
        if (has_input) {
          hip << "  {\n"
              << "    auto data = read_file(\"input_" << buf_idx << ".bin\");\n"
              << "    HIP_CHECK(hipMemcpy(buf_" << buf_idx
              << ", data.data(), data.size(), hipMemcpyHostToDevice));\n"
              << "  }\n";
        }
        hip << "\n";
        buf_idx++;
      } else {
        hip << "  // Arg " << arg_idx << ": scalar (" << arg.size
            << " bytes)\n";
        hip << "  uint8_t scalar_" << arg_idx << "[" << arg.size << "] = {";
        for (size_t i = 0; i < arg.data.size(); i++) {
          if (i > 0) hip << ",";
          hip << "0x" << std::hex << (int)arg.data[i] << std::dec;
        }
        hip << "};\n\n";
      }
      arg_idx++;
    }

    // Build arg pointers
    hip << "  // Build kernel args\n"
        << "  void* args[] = {";
    arg_idx = 0;
    buf_idx = 0;
    bool first = true;
    for (const auto& arg : kl->args) {
      if (arg.value_kind == 2) continue;
      if (!first) hip << ", ";
      first = false;
      if (arg.value_kind == 1) {
        hip << "&buf_" << buf_idx;
        buf_idx++;
      } else {
        hip << "scalar_" << arg_idx;
      }
      arg_idx++;
    }
    hip << "};\n\n";

    // Launch
    hip << "  // Launch kernel\n"
        << "  HIP_CHECK(hipModuleLaunchKernel(func,\n"
        << "    " << kl->grid[0] << ", " << kl->grid[1] << ", "
        << kl->grid[2] << ",\n"
        << "    " << kl->block[0] << ", " << kl->block[1] << ", "
        << kl->block[2] << ",\n"
        << "    " << kl->shared_mem << ", nullptr, args, nullptr));\n"
        << "  HIP_CHECK(hipDeviceSynchronize());\n\n"
        << "  printf(\"Kernel executed successfully.\\n\");\n\n";

    // Cleanup
    buf_idx = 0;
    for (const auto& arg : kl->args) {
      if (arg.value_kind == 2) continue;
      if (arg.value_kind == 1) {
        hip << "  hipFree(buf_" << buf_idx << ");\n";
        buf_idx++;
      }
    }

    hip << "  hipModuleUnload(mod);\n"
        << "  return 0;\n"
        << "}\n";
  }

  // Generate CMakeLists.txt
  {
    std::ofstream cmake(output_dir + "/CMakeLists.txt");
    cmake << "cmake_minimum_required(VERSION 3.21)\n"
          << "project(hrr_repro LANGUAGES CXX)\n\n"
          << "find_package(hip REQUIRED)\n\n"
          << "add_executable(repro repro.hip)\n"
          << "target_link_libraries(repro PRIVATE hip::host hip::device)\n"
          << "set_source_files_properties(repro.hip PROPERTIES LANGUAGE HIP)\n";
  }

  printf("Exported to %s/\n", output_dir.c_str());
  printf("  CMakeLists.txt  - cmake -B build && cmake --build build\n");
  printf("  repro.hip       - standalone reproducer\n");
  printf("  kernel.hsaco    - code object\n");
  printf("  input_*.bin     - input buffer data%s\n",
         safe_mode ? " (SANITIZED)" : "");

  return 0;
}

static void print_usage(const char* argv0) {
  fprintf(stderr,
    "Usage: %s <subcommand> <capture.hrr> [options]\n\n"
    "Subcommands:\n"
    "  list                       List all kernels\n"
    "  kernel --id N              Benchmark kernel N\n"
    "    --iterations N           Number of iterations (default: 100)\n"
    "    --warmup N               Warmup iterations (default: 10)\n"
    "  app                        Replay full trace with timing\n"
    "  repro                      Reproduce crashes\n"
    "    --check-nan              Check for NaN/Inf in outputs\n"
    "  export --id N --output DIR Export kernel as standalone .hip\n"
    "    --safe                   Sanitize buffer data\n"
    "  stress --id N              Stress test a kernel\n"
    "    --iterations N           Number of iterations (default: 1000)\n",
    argv0);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  std::string subcmd = argv[1];
  std::string archive_path = argv[2];

  // Parse common options
  size_t kernel_id = 0;
  int iterations = 100;
  int warmup = 10;
  bool check_nan = false;
  bool safe_mode = false;
  std::string output_dir;

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--id") == 0 && i + 1 < argc) {
      kernel_id = atol(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      iterations = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--check-nan") == 0) {
      check_nan = true;
    } else if (strcmp(argv[i], "--safe") == 0) {
      safe_mode = true;
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      output_dir = argv[++i];
    }
  }

  // Load archive
  hrr::Archive archive;
  if (!hrr::load_archive(archive_path, archive)) {
    return 1;
  }

  printf("[HRR] Archive: %zu events, %zu kernels, %zu blobs, "
         "%zu code objects\n",
         archive.event_count, archive.kernel_count,
         archive.blob_count, archive.code_object_count);

  HIP_CHECK(hipInit(0));

  if (subcmd == "list") {
    return cmd_list(archive);
  } else if (subcmd == "kernel") {
    return cmd_kernel(archive, kernel_id, iterations, warmup);
  } else if (subcmd == "repro") {
    return cmd_repro(archive, check_nan);
  } else if (subcmd == "export") {
    if (output_dir.empty()) output_dir = "repro_" + std::to_string(kernel_id);
    return cmd_export(archive, kernel_id, output_dir, safe_mode);
  } else if (subcmd == "stress") {
    return cmd_kernel(archive, kernel_id, iterations > 1000 ? iterations : 1000,
                      warmup);
  } else if (subcmd == "app") {
    // Full replay with timing - delegate to repro without nan check
    return cmd_repro(archive, false);
  } else {
    fprintf(stderr, "Unknown subcommand: %s\n", subcmd.c_str());
    print_usage(argv[0]);
    return 1;
  }
}
