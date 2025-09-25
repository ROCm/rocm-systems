/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#ifdef _WIN32

#include "sys/mman.h"  // Include our own header for declarations and constants
#include <windows.h>
#include <unordered_map>
#include <mutex>
#include <sys/types.h>  // For off_t definition (same as header uses)
#include <cstdio>        // For snprintf
#include <cstdlib>       // For atol

struct MappingInfo {
    HANDLE handle;
    size_t length;
};

static std::unordered_map<void*, MappingInfo> g_mappings;
static std::mutex g_mappings_mutex;
static bool g_shared_mapping_used = false;  // Only allow one shared mapping per process

extern "C" {

void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    (void)addr; (void)prot; (void)fd; (void)offset;

    if (flags & MAP_ANONYMOUS) {
        if (flags & MAP_SHARED) {
            // Shared anonymous memory for IPC - only allow one per process
            // Use parent PID for uniqueness across several runs of same test
            static char shared_name[64];

            // Check if shared mapping already used
            if (g_shared_mapping_used) {
                // Only fail if there are actually active mappings
                std::lock_guard<std::mutex> lock(g_mappings_mutex);
                if (!g_mappings.empty()) {
                    return MAP_FAILED;  // Only one shared mapping allowed
                }
            }
            g_shared_mapping_used = true;

            DWORD parent_pid;
            const char* child_marker = getenv("ROCR_IPC_CHILD_PROCESS");
            if (child_marker) {
                // Child process - get parent PID from environment
                const char* parent_pid_str = getenv("ROCR_IPC_PARENT_PID");
                if (parent_pid_str) {
                    parent_pid = (DWORD)atol(parent_pid_str);
                } else {
                    return MAP_FAILED;
                }
            } else {
                // Parent process - use our own PID
                parent_pid = GetCurrentProcessId();
            }

            // Use Local\ namespace instead of Global\ - no admin privileges required
            snprintf(shared_name, sizeof(shared_name), "Local\\rocrtst_ipc_%lu", parent_pid);

            // CreateFileMappingA takes size as high/low DWORD pair for 64-bit size
            DWORD size_high = (DWORD)(length >> 32);
            DWORD size_low = (DWORD)(length & 0xFFFFFFFF);

            HANDLE handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                               size_high, size_low, shared_name);
            if (handle == NULL) {
                return MAP_FAILED;
            }

            void* mapped_addr = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, length);
            if (mapped_addr == NULL) {
                CloseHandle(handle);
                return MAP_FAILED;
            }

            {
                std::lock_guard<std::mutex> lock(g_mappings_mutex);
                g_mappings[mapped_addr] = {handle, length};
            }
            return mapped_addr;

        } else {
            // Private anonymous memory (like gtest) - just use VirtualAlloc
            void* mapped_addr = VirtualAlloc(NULL, length, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (mapped_addr == NULL) {
                return MAP_FAILED;
            }

            {
                std::lock_guard<std::mutex> lock(g_mappings_mutex);
                g_mappings[mapped_addr] = {NULL, length}; // NULL handle for VirtualAlloc
            }
            return mapped_addr;
        }
    }
    return MAP_FAILED;
}

int munmap(void* addr, size_t length) {
    if (!addr) {
        return -1;
    }

    HANDLE handle = NULL;
    {
        std::lock_guard<std::mutex> lock(g_mappings_mutex);
        auto it = g_mappings.find(addr);
        if (it != g_mappings.end()) {
            if (length != 0 && length != it->second.length) {
                return -1;
            }
            handle = it->second.handle;
            g_mappings.erase(it);
        } else {
            return -1;
        }
    }

    if (handle) {
        // Shared memory mapping - reset the flag so next test can create shared mapping
        UnmapViewOfFile(addr);
        CloseHandle(handle);
        g_shared_mapping_used = false;
    } else {
        // Private memory mapping (VirtualAlloc)
        VirtualFree(addr, 0, MEM_RELEASE);
    }
    return 0;
}
} // extern "C"

#endif // _WIN32