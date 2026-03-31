/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

// hrr-replay: Replay a .hrr trace archive on the current GPU.
//
// Usage: hrr-replay <capture.hrr> [--verify] [--timing] [--kernel-filter NAME]

#include "hrr_reader.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <chrono>

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t err = (call);                                                  \
    if (err != hipSuccess) {                                                  \
      fprintf(stderr, "[HRR] HIP error %d (%s) at %s:%d\n", err,             \
              hipGetErrorString(err), __FILE__, __LINE__);                    \
      return 1;                                                               \
    }                                                                         \
  } while (0)

struct ReplayState {
  // Handle -> live GPU pointer
  std::unordered_map<uint64_t, void*> alloc_map;
  // Handle -> allocation size
  std::unordered_map<uint64_t, size_t> alloc_sizes;
  // Module handle -> hipModule_t
  std::unordered_map<uint64_t, hipModule_t> module_map;
  // Code object hash -> loaded module
  std::unordered_map<std::string, hipModule_t> co_modules;
  // Stream handle -> hipStream_t
  std::unordered_map<uint32_t, hipStream_t> stream_map;

  bool verify = false;
  bool timing = false;
  std::string kernel_filter;

  size_t kernels_launched = 0;
  size_t verify_pass = 0;
  size_t verify_fail = 0;
  double total_kernel_ms = 0.0;
};

static void* translate_ptr(ReplayState& state, uint64_t handle) {
  auto it = state.alloc_map.find(handle);
  if (it != state.alloc_map.end()) return it->second;
  // Might be a sub-allocation (handle is base + offset)
  // Try range lookup
  for (auto& [h, ptr] : state.alloc_map) {
    auto sz_it = state.alloc_sizes.find(h);
    if (sz_it != state.alloc_sizes.end()) {
      if (handle >= h && handle < h + sz_it->second) {
        return static_cast<char*>(ptr) + (handle - h);
      }
    }
  }
  return nullptr;
}

static hipModule_t get_module_for_co(ReplayState& state,
                                     const hrr::Archive& archive,
                                     uint64_t hash_lo, uint64_t hash_hi) {
  std::string hex = hrr::hash_hex(hash_lo, hash_hi);
  auto it = state.co_modules.find(hex);
  if (it != state.co_modules.end()) return it->second;

  std::vector<uint8_t> co_data;
  if (!hrr::read_code_object(archive, hash_lo, hash_hi, co_data)) {
    fprintf(stderr, "[HRR] Code object %s not found\n", hex.c_str());
    return nullptr;
  }

  hipModule_t mod = nullptr;
  hipError_t err = hipModuleLoadData(&mod, co_data.data());
  if (err != hipSuccess) {
    fprintf(stderr, "[HRR] Failed to load code object %s: %d\n",
            hex.c_str(), err);
    return nullptr;
  }

  state.co_modules[hex] = mod;
  return mod;
}

static int replay_event(ReplayState& state, const hrr::Archive& archive,
                        const hrr::Event& ev) {
  switch (ev.header.event_type) {
    case hrr::EVENT_MALLOC: {
      void* ptr = nullptr;
      HIP_CHECK(hipMalloc(&ptr, ev.malloc_ev.size));
      state.alloc_map[ev.malloc_ev.ptr_handle] = ptr;
      state.alloc_sizes[ev.malloc_ev.ptr_handle] = ev.malloc_ev.size;
      break;
    }

    case hrr::EVENT_FREE: {
      auto it = state.alloc_map.find(ev.malloc_ev.ptr_handle);
      if (it != state.alloc_map.end()) {
        hipFree(it->second);
        state.alloc_map.erase(it);
        state.alloc_sizes.erase(ev.malloc_ev.ptr_handle);
      }
      break;
    }

    case hrr::EVENT_MEMCPY: {
      const auto& mc = ev.memcpy_ev;
      if (mc.kind == 1 && mc.hash_lo != 0) {  // H2D with blob data
        std::vector<uint8_t> blob;
        if (hrr::read_blob(archive, mc.hash_lo, mc.hash_hi, blob)) {
          void* dst = translate_ptr(state, mc.dst_addr);
          if (dst) {
            HIP_CHECK(hipMemcpy(dst, blob.data(), mc.size,
                                hipMemcpyHostToDevice));
          }
        }
      } else if (mc.kind == 3) {  // D2D
        void* dst = translate_ptr(state, mc.dst_addr);
        void* src = translate_ptr(state, mc.src_addr);
        if (dst && src) {
          HIP_CHECK(hipMemcpy(dst, src, mc.size, hipMemcpyDeviceToDevice));
        }
      }
      break;
    }

    case hrr::EVENT_MEMSET: {
      if (ev.raw_payload.size() >= 20) {
        uint64_t dst_addr;
        uint32_t value;
        uint64_t size;
        memcpy(&dst_addr, ev.raw_payload.data(), 8);
        memcpy(&value, ev.raw_payload.data() + 8, 4);
        memcpy(&size, ev.raw_payload.data() + 12, 8);
        void* dst = translate_ptr(state, dst_addr);
        if (dst) {
          HIP_CHECK(hipMemset(dst, static_cast<int>(value), size));
        }
      }
      break;
    }

    case hrr::EVENT_MODULE_LOAD: {
      hipModule_t mod = get_module_for_co(state, archive,
                                          ev.module_load_ev.hash_lo,
                                          ev.module_load_ev.hash_hi);
      if (mod) {
        state.module_map[ev.module_load_ev.module_handle] = mod;
      }
      break;
    }

    case hrr::EVENT_KERNEL_LAUNCH: {
      if (!ev.kernel_launch) break;
      const auto& kl = *ev.kernel_launch;

      if (!state.kernel_filter.empty()) {
        if (kl.kernel_name.find(state.kernel_filter) == std::string::npos) {
          break;
        }
      }

      // Restore input buffer snapshots
      for (const auto& snap : kl.snapshots) {
        if (snap.direction == 0) {  // input
          void* dst = translate_ptr(state, snap.ptr_handle);
          if (dst) {
            std::vector<uint8_t> blob;
            if (hrr::read_blob(archive, snap.hash_lo, snap.hash_hi, blob)) {
              hipMemcpy(dst, blob.data(), snap.length, hipMemcpyHostToDevice);
            }
          }
        }
      }

      // Find kernel function in loaded modules
      hipFunction_t func = nullptr;
      for (auto& [handle, mod] : state.module_map) {
        hipError_t err = hipModuleGetFunction(&func, mod,
                                              kl.kernel_name.c_str());
        if (err == hipSuccess && func) break;
        func = nullptr;
      }
      // Also try all code object modules
      if (!func) {
        for (auto& [hex, mod] : state.co_modules) {
          hipError_t err = hipModuleGetFunction(&func, mod,
                                                kl.kernel_name.c_str());
          if (err == hipSuccess && func) break;
          func = nullptr;
        }
      }

      if (!func) {
        fprintf(stderr, "[HRR] Kernel '%s' not found in any loaded module\n",
                kl.kernel_name.c_str());
        break;
      }

      // Build kernarg buffer from captured args
      std::vector<void*> arg_ptrs;
      std::vector<std::vector<uint8_t>> arg_storage;

      for (const auto& arg : kl.args) {
        if (arg.value_kind == 2) continue;  // skip hidden

        arg_storage.emplace_back();
        auto& storage = arg_storage.back();

        if (arg.value_kind == 1 && arg.data.size() >= 8) {
          // Pointer arg: translate handle to live pointer
          uint64_t handle;
          memcpy(&handle, arg.data.data(), 8);
          void* live_ptr = translate_ptr(state, handle);
          storage.resize(sizeof(void*));
          memcpy(storage.data(), &live_ptr, sizeof(void*));
        } else {
          // Scalar arg: use raw bytes
          storage = arg.data;
        }
        arg_ptrs.push_back(storage.data());
      }

      // Launch with optional timing
      hipEvent_t start = nullptr, stop = nullptr;
      if (state.timing) {
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start);
      }

      HIP_CHECK(hipModuleLaunchKernel(
          func,
          kl.grid[0], kl.grid[1], kl.grid[2],
          kl.block[0], kl.block[1], kl.block[2],
          kl.shared_mem, nullptr,
          arg_ptrs.data(), nullptr));

      if (state.timing) {
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        state.total_kernel_ms += ms;
        hipEventDestroy(start);
        hipEventDestroy(stop);
      }

      state.kernels_launched++;

      // Verify output buffers if requested
      if (state.verify) {
        hipDeviceSynchronize();
        for (const auto& snap : kl.snapshots) {
          if (snap.direction == 1) {  // output
            void* src = translate_ptr(state, snap.ptr_handle);
            if (!src) continue;

            std::vector<uint8_t> expected;
            if (!hrr::read_blob(archive, snap.hash_lo, snap.hash_hi,
                                expected)) {
              continue;
            }

            std::vector<uint8_t> actual(snap.length);
            hipMemcpy(actual.data(), src, snap.length,
                      hipMemcpyDeviceToHost);

            if (memcmp(actual.data(), expected.data(), snap.length) == 0) {
              state.verify_pass++;
            } else {
              state.verify_fail++;
              // Find max absolute diff (treating as float32)
              size_t num_f32 = snap.length / 4;
              float max_diff = 0.0f;
              const float* a = reinterpret_cast<const float*>(actual.data());
              const float* e = reinterpret_cast<const float*>(expected.data());
              for (size_t i = 0; i < num_f32; i++) {
                float d = std::fabs(a[i] - e[i]);
                if (d > max_diff) max_diff = d;
              }
              fprintf(stderr,
                      "[HRR] MISMATCH kernel '%s' output buffer "
                      "(handle=0x%llx, max_diff=%.6g)\n",
                      kl.kernel_name.c_str(),
                      (unsigned long long)snap.ptr_handle, max_diff);
            }
          }
        }
      }
      break;
    }

    case hrr::EVENT_DEVICE_SYNC:
      HIP_CHECK(hipDeviceSynchronize());
      break;

    default:
      break;
  }
  return 0;
}

static void print_usage(const char* argv0) {
  fprintf(stderr,
    "Usage: %s <capture.hrr> [options]\n"
    "\n"
    "Options:\n"
    "  --verify            Compare output buffers with recorded snapshots\n"
    "  --timing            Report per-kernel GPU timing\n"
    "  --kernel-filter STR Only replay kernels containing STR in name\n"
    "  --help              Show this help\n",
    argv0);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string archive_path;
  ReplayState state;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--verify") == 0) {
      state.verify = true;
    } else if (strcmp(argv[i], "--timing") == 0) {
      state.timing = true;
    } else if (strcmp(argv[i], "--kernel-filter") == 0 && i + 1 < argc) {
      state.kernel_filter = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else if (argv[i][0] != '-') {
      archive_path = argv[i];
    }
  }

  if (archive_path.empty()) {
    fprintf(stderr, "[HRR] No archive path specified\n");
    return 1;
  }

  // Load archive
  hrr::Archive archive;
  if (!hrr::load_archive(archive_path, archive)) {
    return 1;
  }

  printf("[HRR] Loaded archive: %zu events, %zu kernels, %zu blobs, "
         "%zu code objects\n",
         archive.event_count, archive.kernel_count,
         archive.blob_count, archive.code_object_count);

  // Init HIP
  HIP_CHECK(hipInit(0));

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "[HRR] No GPU devices found\n");
    return 1;
  }

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("[HRR] Replaying on: %s (%s)\n", props.name, props.gcnArchName);

  // Replay events
  auto wall_start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < archive.events.size(); i++) {
    int ret = replay_event(state, archive, archive.events[i]);
    if (ret != 0) {
      fprintf(stderr, "[HRR] Replay failed at event %zu (%s)\n",
              i, hrr::event_type_name(archive.events[i].header.event_type));
      return ret;
    }
  }

  hipDeviceSynchronize();
  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(
                       wall_end - wall_start).count();

  // Report
  printf("[HRR] Replay complete: %zu kernels launched\n",
         state.kernels_launched);
  printf("[HRR] Wall time: %.1f ms\n", wall_ms);

  if (state.timing) {
    printf("[HRR] Total GPU kernel time: %.1f ms\n", state.total_kernel_ms);
  }

  if (state.verify) {
    printf("[HRR] Verification: %zu passed, %zu failed\n",
           state.verify_pass, state.verify_fail);
  }

  // Cleanup
  for (auto& [handle, ptr] : state.alloc_map) {
    hipFree(ptr);
  }
  for (auto& [hex, mod] : state.co_modules) {
    hipModuleUnload(mod);
  }

  return state.verify_fail > 0 ? 1 : 0;
}
