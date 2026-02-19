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

#ifndef CUID_FILE_H
#define CUID_FILE_H

#include "include/amd_cuid.h"
#include "src/cuid_util.h"
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <cstring>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

/**
 * @brief Lock types for file operations
 */
enum class CuidLockType {
    SHARED,     ///< Shared lock for reading (multiple readers allowed)
    EXCLUSIVE   ///< Exclusive lock for writing (single writer, no readers)
};

/**
 * @brief RAII-style file lock using fcntl() advisory locking
 * 
 * This class provides process-safe file locking for CUID files using POSIX
 * fcntl() advisory locks. It creates a separate lock file (e.g., /tmp/cuid.lock)
 * to synchronize access to the actual CUID data files.
 * 
 * Features:
 * - POSIX compliant (works on NFS)
 * - Supports shared (read) and exclusive (write) locks
 * - Automatic lock release on destruction (RAII)
 * - Blocking acquisition with timeout support
 * 
 * Usage:
 *   // For reading
 *   CuidFileLock lock("/tmp/cuid", CuidLockType::SHARED);
 *   if (lock.acquire()) { ... read operations ... }
 * 
 *   // For writing
 *   CuidFileLock lock("/tmp/cuid", CuidLockType::EXCLUSIVE);
 *   if (lock.acquire()) { ... write operations ... }
 */
class CuidFileLock {
public:
    /**
     * @brief Construct a file lock for the given CUID file path
     * @param file_path Path to the CUID file (lock file will be file_path + ".lock")
     * @param lock_type Type of lock to acquire (SHARED for read, EXCLUSIVE for write)
     */
    CuidFileLock(const std::string& file_path, CuidLockType lock_type);
    
    /**
     * @brief Destructor - automatically releases the lock
     */
    ~CuidFileLock();
    
    // Non-copyable, non-movable (RAII resource)
    CuidFileLock(const CuidFileLock&) = delete;
    CuidFileLock& operator=(const CuidFileLock&) = delete;
    CuidFileLock(CuidFileLock&&) = delete;
    CuidFileLock& operator=(CuidFileLock&&) = delete;
    
    /**
     * @brief Acquire the lock (blocking)
     * @return true if lock was acquired successfully, false on error
     */
    bool acquire();
    
    /**
     * @brief Acquire the lock with a timeout
     * @param timeout_seconds Maximum time to wait for lock (0 = non-blocking, -1 = infinite)
     * @return true if lock was acquired, false if timeout or error
     */
    bool acquire_with_timeout(int timeout_seconds);
    
    /**
     * @brief Try to acquire the lock (non-blocking)
     * @return true if lock was acquired, false if lock is held by another process
     */
    bool try_acquire();
    
    /**
     * @brief Release the lock explicitly (also called by destructor)
     */
    void release();
    
    /**
     * @brief Check if lock is currently held
     * @return true if lock is held
     */
    bool is_locked() const { return is_locked_; }
    
    /**
     * @brief Get the lock file path
     * @return Path to the lock file
     */
    const std::string& get_lock_file_path() const { return lock_file_path_; }

private:
    std::string lock_file_path_;
    CuidLockType lock_type_;
    int lock_fd_;
    bool is_locked_;
};

struct CuidFileEntry {
    amdcuid_device_type_t device_type;
    uint32_t device_index;  // e.g., 0 for GPU:0, 1 for GPU:1

    amdcuid_id_t primary_cuid;     // Only available in privileged file
    amdcuid_id_t derived_cuid;

    uint64_t hardware_fingerprint;  // Only available in privileged file
    uint16_t vendor_id = 0;
    uint16_t device_id = 0;
    uint8_t revision_id = 0;

    // Device-specific information
    uint16_t family = 0;                // For CPU
    uint16_t model = 0;                 // For CPU
    uint16_t pci_class = 0;             // For PCIe devices (GPU, NIC)
    uint16_t unit_id = 0;               // For CPU and GPU
    std::string device_node;       // For GPU: /sys/class/drm/renderD128, NIC: /sys/class/net/eth0
    std::string package_core_id;   // For CPU: "0:0" (package:core)
    std::string bdf;               // PCIe Bus:Device.Function
    std::string mac_address;       // For NIC

    time_t last_update;            // Unix timestamp

    CuidFileEntry() 
        : device_type(AMDCUID_DEVICE_TYPE_NONE)
        , device_index(0)
        , last_update(0)
    {
        memset(primary_cuid.bytes, 0, sizeof(primary_cuid.bytes));
        memset(derived_cuid.bytes, 0, sizeof(derived_cuid.bytes));
    }
};

/**
 * @brief CUID File handler for reading and writing device CUID information
 */
class CuidFile {
public:
    /**
     * @brief Constructor
     * @param file_path Path to the CUID file (e.g., /tmp/cuid or /tmp/priv_cuid)
     * @param is_privileged Whether this is a privileged file (includes primary CUIDs)
     */
    CuidFile(const std::string& file_path, bool is_privileged = false);
    
    amdcuid_status_t load();
    amdcuid_status_t save();
    amdcuid_status_t add_entry(const CuidFileEntry& entry);
    amdcuid_status_t remove_entry(const amdcuid_id_t& handle);
    
    const std::vector<CuidFileEntry>& get_entries() const { return entries_; }

    amdcuid_status_t find_by_device_node(const std::string& device_node, CuidFileEntry& entry) const;
    amdcuid_status_t find_by_bdf(const std::string& bdf, CuidFileEntry& entry) const;
    amdcuid_status_t find_by_package_core_id(const std::string& package_core_id, CuidFileEntry& entry) const;
    amdcuid_status_t find_by_device_type(amdcuid_device_type_t device_type, CuidFileEntry& entry) const;
    amdcuid_status_t find_by_derived_cuid(const amdcuid_id_t& derived_cuid, CuidFileEntry& entry) const;
    
    void clear() { entries_.clear(); }
    bool exists() const;
    const std::string& get_file_path() const { return file_path_; }
    
    /**
     * @brief Check if this is a privileged file
     */
    bool is_privileged() const { return is_privileged_; }

    // static utility to group entries by device type
    void get_grouped_entries(std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>>& grouped) const;

private:
    std::string file_path_;
    bool is_privileged_;
    std::vector<CuidFileEntry> entries_;
    
    // Helper functions
    amdcuid_device_type_t string_to_device_type(const std::string& str) const;
    amdcuid_id_t string_to_cuid(const std::string& str) const;
    std::string trim(const std::string& str) const;
    bool parse_section_header(const std::string& line, amdcuid_device_type_t& type, uint32_t& index) const;
};

/**
 * @brief Utility class for generating CUID files from discovered devices
 */
class CuidFileGenerator {
public:
    /**
     * @brief Generate CUID files from device manager
     * @param devices Vector of discovered devices
     * @param key_file_path Path to the HMAC key file for derived CUID generation
     * @param unprivileged_file Path to write unprivileged CUID file (default: /tmp/cuid)
     * @param privileged_file Path to write privileged CUID file (default: /tmp/priv_cuid)
     * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
     */
    static amdcuid_status_t generate_from_devices(
        const std::vector<std::shared_ptr<class CuidDevice>>& devices,
        const std::string& key_file_path,
        const std::string& unprivileged_file = CuidUtilities::cuid_file(),
        const std::string& privileged_file = CuidUtilities::priv_cuid_file()
    );
};

#endif // CUID_FILE_H
