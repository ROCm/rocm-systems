/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * =============================================================================
 * KFD Data Manager - Isolated Process Architecture for KFD IOCTL Operations
 * =============================================================================
 *
 * PURPOSE:
 * --------
 * The AMD KFD driver creates persistent process entries at
 * /sys/class/kfd/kfd/proc/<pid> when opening /dev/kfd. These persist until
 * process termination, blocking driver reloads and partition changes.
 *
 * This module uses fork-and-exit pattern: KFD operations run in short-lived
 * child processes that exit immediately, ensuring prompt cleanup.
 *
 * CACHING:
 * --------
 * Batch queries support time-based caching with O(1) lookup by gpu_id.
 * On cache miss, a single fork refreshes data for ALL GPUs.
 * Cache is invalidated if any GPU query fails (full invalidation).
 *
 * TYPICAL TIMING (per query):
 * ---------------------------
 *   - Cache hit:             < 1 us (O(1) hash lookup)
 *   - Cache miss (fork):     200-800 us typical
 *     - Fork overhead:       100-300 us
 *     - IOCTL execution:     50-200 us
 *     - Cleanup verification: 10-50 us (stat-based polling)
 *
 * CONFIGURATION (environment variables):
 * --------------------------------------
 *   AMDSMI_KFD_USE_ORIG_VRAM:          1=use original ioctl (default: 0)
 *   AMDSMI_KFD_DISABLE_INOTIFY_POLLING: 1=use stat polling (default: 1)
 *   AMDSMI_KFD_INOTIFY_POLL_MS:        poll timeout in ms (default: 2)
 *   AMDSMI_KFD_CLEANUP_POLL_US:        stat poll interval in us (default: 250)
 *   AMDSMI_KFD_CACHE_TTL_MS:           cache TTL in ms (default: 250, 0=disabled)
 *   AMDSMI_KFD_MAX_CLEANUP_WAIT_MS:    max wait for cleanup in ms (default: 100)
 *
 * THREAD SAFETY: All public functions are thread-safe.
 * =============================================================================
 */

#ifndef ROCM_SMI_KFD_DATA_MANAGER_H_
#define ROCM_SMI_KFD_DATA_MANAGER_H_

#include <cstdint>
#include <vector>

namespace amd::smi::kfd {

/// Configuration for KFD Data Manager timing and behavior
struct KFDManagerConfig {
  /// Use legacy direct ioctl instead of fork pattern (default: false)
  /// Set AMDSMI_KFD_USE_ORIG_VRAM=1 to enable
  bool use_original_vram_fcn = false;

  /// Use stat-based polling instead of inotify (default: true, faster)
  /// Set AMDSMI_KFD_DISABLE_INOTIFY_POLLING=0 to re-enable
  bool disable_inotify_polling = true;

  /// Poll timeout for inotify in milliseconds (default: 2ms)
  /// Set AMDSMI_KFD_INOTIFY_POLL_MS to adjust
  /// Ignored if disable_inotify_polling=true
  int64_t inotify_poll_ms = 2;

  /// Stat polling interval in microseconds (default: 250us)
  /// Set AMDSMI_KFD_CLEANUP_POLL_US to adjust
  int64_t cleanup_poll_us = 250;

  /// Cache time-to-live in milliseconds (default: 250ms, 0=disabled)
  /// When enabled, batch queries use O(1) cache lookup
  /// Set AMDSMI_KFD_CACHE_TTL_MS to adjust
  int64_t cache_ttl_ms = 250;

  /// Maximum wait time for child process cleanup in milliseconds (default: 100ms)
  /// This is a safety bound - normal cleanup takes <50ms. If this timeout fires,
  /// there's likely a kernel/driver issue that won't resolve with more waiting.
  /// Set AMDSMI_KFD_MAX_CLEANUP_WAIT_MS to adjust
  int64_t max_cleanup_wait_ms = 100;
};

/// Supported KFD operation types
enum class OpType : uint32_t { kQueryAvailableVram = 0, kOpTypeCount };

/// Result container for KFD operations
struct QueryResult {
  int err_code = 0;    ///< 0 on success, errno on failure
  uint64_t value = 0;  ///< Operation result (interpretation depends on OpType)
};

/// Load configuration from environment variables into cfg
void LoadConfigFromEnvironment(KFDManagerConfig& cfg);  // NOLINT(runtime/references)

/// Get current configuration (thread-safe)
KFDManagerConfig GetCurrentConfig();

/// Initialize manager with specified config (first call wins, thread-safe)
[[nodiscard]] int InitializeManager(const KFDManagerConfig& cfg);

/// Execute KFD operation in isolated child process (thread-safe, no caching)
[[nodiscard]] QueryResult ExecuteIsolatedQuery(OpType op, uint32_t gpu_id);

/// Batch query with caching - O(1) cache lookup by gpu_id
/// If target's cache is valid, returns immediately
/// Otherwise, refreshes cache for ALL gpu_ids in one fork
[[nodiscard]] QueryResult ExecuteBatchQueryCached(OpType op, const std::vector<uint32_t>& gpu_ids,
                                                  uint32_t target_gpu_id);

/// Purge cache entries (gpu_id < 0 means all entries for the operation)
void PurgeCacheEntries(OpType op, int32_t gpu_id = -1);

/// Purge all cache entries
void PurgeAllCacheEntries();

/// Convenience wrapper to query available VRAM (no caching)
[[nodiscard]] int QueryAvailableVram(uint32_t gpu_id, uint64_t* out_available);

/// Convenience wrapper for batched VRAM query (with caching)
[[nodiscard]] int QueryAvailableVramBatch(const std::vector<uint32_t>& gpu_ids,
                                          uint32_t target_gpu_id, uint64_t* out_available);

/// Forcefully terminate any active helper processes
void TerminateActiveHelpers();

}  // namespace amd::smi::kfd

#endif  // ROCM_SMI_KFD_DATA_MANAGER_H_
