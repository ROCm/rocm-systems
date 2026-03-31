/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

// hrr-info: Print summary of a .hrr trace archive.
//
// Usage: hrr-info <capture.hrr> [--events]

#include "hrr_reader.h"

#include <cstdio>
#include <cstring>
#include <map>

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <capture.hrr> [--events]\n", argv[0]);
    return 1;
  }

  std::string path = argv[1];
  bool show_events = false;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--events") == 0) show_events = true;
  }

  hrr::Archive archive;
  if (!hrr::load_archive(path, archive)) {
    return 1;
  }

  printf("HRR Archive: %s\n", path.c_str());
  printf("========================================\n");
  printf("Events:       %zu\n", archive.event_count);
  printf("Kernels:      %zu\n", archive.kernel_count);
  printf("Blobs:        %zu\n", archive.blob_count);
  printf("Code Objects: %zu\n", archive.code_object_count);
  printf("\n");

  // Event type breakdown
  std::map<uint16_t, size_t> type_counts;
  for (const auto& ev : archive.events) {
    type_counts[ev.header.event_type]++;
  }

  printf("Event Type Breakdown:\n");
  printf("  %-20s %s\n", "Type", "Count");
  printf("  %-20s %s\n", "----", "-----");
  for (auto& [type, count] : type_counts) {
    printf("  %-20s %zu\n", hrr::event_type_name(type), count);
  }
  printf("\n");

  // Kernel summary
  if (archive.kernel_count > 0) {
    printf("Kernel Summary:\n");
    printf("  %-4s %-50s %-15s %-15s %s\n",
           "ID", "Kernel", "Grid", "Block", "SharedMem");
    printf("  %-4s %-50s %-15s %-15s %s\n",
           "--", "------", "----", "-----", "---------");

    size_t kid = 0;
    std::map<std::string, size_t> kernel_calls;
    for (const auto& ev : archive.events) {
      if (ev.header.event_type == hrr::EVENT_KERNEL_LAUNCH && ev.kernel_launch) {
        const auto& kl = *ev.kernel_launch;
        kernel_calls[kl.kernel_name]++;

        if (kid < 20) {  // Show first 20
          char grid_str[32], block_str[32];
          snprintf(grid_str, sizeof(grid_str), "[%u,%u,%u]",
                   kl.grid[0], kl.grid[1], kl.grid[2]);
          snprintf(block_str, sizeof(block_str), "[%u,%u,%u]",
                   kl.block[0], kl.block[1], kl.block[2]);

          // Truncate long names
          std::string name = kl.kernel_name;
          if (name.size() > 50) {
            name = name.substr(0, 47) + "...";
          }

          printf("  %-4zu %-50s %-15s %-15s %u\n",
                 kid, name.c_str(), grid_str, block_str, kl.shared_mem);
        }
        kid++;
      }
    }

    if (kid > 20) {
      printf("  ... and %zu more\n", kid - 20);
    }

    printf("\nKernel Call Counts:\n");
    printf("  %-60s %s\n", "Kernel", "Calls");
    printf("  %-60s %s\n", "------", "-----");
    for (auto& [name, count] : kernel_calls) {
      std::string display = name;
      if (display.size() > 60) display = display.substr(0, 57) + "...";
      printf("  %-60s %zu\n", display.c_str(), count);
    }
  }

  // Print all events if requested
  if (show_events) {
    printf("\nEvent Log:\n");
    printf("  %-6s %-16s %-12s %-6s %-6s %s\n",
           "Seq", "Timestamp(ns)", "Type", "Stream", "Dev", "Details");
    for (const auto& ev : archive.events) {
      printf("  %-6llu %-16llu %-12s %-6u %-6u",
             (unsigned long long)ev.header.sequence_id,
             (unsigned long long)ev.header.timestamp_ns,
             hrr::event_type_name(ev.header.event_type),
             ev.header.stream_id,
             ev.header.device_id);

      // Print event-specific details
      switch (ev.header.event_type) {
        case hrr::EVENT_MALLOC:
          printf(" handle=0x%llx size=%llu",
                 (unsigned long long)ev.malloc_ev.ptr_handle,
                 (unsigned long long)ev.malloc_ev.size);
          break;
        case hrr::EVENT_KERNEL_LAUNCH:
          if (ev.kernel_launch) {
            printf(" %s [%u,%u,%u] args=%zu snaps=%zu",
                   ev.kernel_launch->kernel_name.c_str(),
                   ev.kernel_launch->grid[0],
                   ev.kernel_launch->grid[1],
                   ev.kernel_launch->grid[2],
                   ev.kernel_launch->args.size(),
                   ev.kernel_launch->snapshots.size());
          }
          break;
        default:
          break;
      }
      printf("\n");
    }
  }

  return 0;
}
