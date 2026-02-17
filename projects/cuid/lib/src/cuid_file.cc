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

#include "src/cuid_file.h"
#include "src/cuid_device.h"
#include "src/cuid_gpu.h"
#include "src/cuid_cpu.h"
#include "src/cuid_nic.h"
#include "src/cuid_platform.h"
#include "src/hmac.h"
#include "src/cuid_internal.h"
#include "src/cuid_util.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fcntl.h>
#include <cerrno>

// ============================================================================
// CuidFileLock Implementation
// ============================================================================

CuidFileLock::CuidFileLock(const std::string& file_path, CuidLockType lock_type)
    : lock_file_path_(file_path + ".lock")
    , lock_type_(lock_type)
    , lock_fd_(-1)
    , is_locked_(false)
{
}

CuidFileLock::~CuidFileLock() {
    release();
}

bool CuidFileLock::acquire() {
    if (is_locked_) {
        return true;  // Already locked
    }

    // For shared (read) locks, we only need O_RDONLY access
    // For exclusive (write) locks, we need O_RDWR access
    // This allows unprivileged users to acquire shared locks on files created by root
    int open_flags = (lock_type_ == CuidLockType::EXCLUSIVE) ? (O_RDWR | O_CREAT) : O_RDONLY;
    
    // For exclusive locks (creating/writing), ensure proper permissions by clearing umask
    if (lock_type_ == CuidLockType::EXCLUSIVE) {
        mode_t old_umask = umask(0);
        lock_fd_ = open(lock_file_path_.c_str(), open_flags, 0666);
        umask(old_umask);
        // Ensure permissions are correct even if file already existed
        if (lock_fd_ >= 0) {
            fchmod(lock_fd_, 0666);
        }
    } else {
        // Try to open existing file for shared (read) lock
        lock_fd_ = open(lock_file_path_.c_str(), open_flags, 0666);
    }
    
    // If file doesn't exist and we need a shared lock, try to create it
    if (lock_fd_ < 0 && lock_type_ == CuidLockType::SHARED && errno == ENOENT) {
        // Try to create the lock file - may fail if not privileged, which is OK
        // The file should be created by root when generating CUIDs
        mode_t old_umask = umask(0);  // Temporarily clear umask for proper permissions
        lock_fd_ = open(lock_file_path_.c_str(), O_RDWR | O_CREAT, 0666);
        umask(old_umask);  // Restore umask
        
        // If created successfully, also chmod to ensure permissions are correct
        if (lock_fd_ >= 0) {
            fchmod(lock_fd_, 0666);
        }
    }
    
    if (lock_fd_ < 0) {
        LOG(ERROR, "CuidFileLock: Failed to open lock file " << lock_file_path_ 
            << ": " << strerror(errno));
        return false;
    }

    // Set up the lock structure
    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;  // Lock entire file
    fl.l_type = (lock_type_ == CuidLockType::EXCLUSIVE) ? F_WRLCK : F_RDLCK;

    // F_SETLKW: blocking call - wait until lock is available
    if (fcntl(lock_fd_, F_SETLKW, &fl) < 0) {
        LOG(ERROR, "CuidFileLock: Failed to acquire lock on " << lock_file_path_ 
            << ": " << strerror(errno));
        close(lock_fd_);
        lock_fd_ = -1;
        return false;
    }

    is_locked_ = true;
    LOG(DEBUG, "CuidFileLock: Acquired " 
        << (lock_type_ == CuidLockType::EXCLUSIVE ? "exclusive" : "shared")
        << " lock on " << lock_file_path_);
    return true;
}

bool CuidFileLock::acquire_with_timeout(int timeout_seconds) {
    // Special cases
    if (timeout_seconds == 0) {
        return try_acquire();  // Non-blocking
    }
    if (timeout_seconds < 0) {
        return acquire();  // Infinite wait
    }

    if (is_locked_) {
        return true;  // Already locked
    }

    // For shared (read) locks, we only need O_RDONLY access
    // For exclusive (write) locks, we need O_RDWR access
    int open_flags = (lock_type_ == CuidLockType::EXCLUSIVE) ? (O_RDWR | O_CREAT) : O_RDONLY;
    
    // For exclusive locks (creating/writing), ensure proper permissions by clearing umask
    if (lock_type_ == CuidLockType::EXCLUSIVE) {
        mode_t old_umask = umask(0);
        lock_fd_ = open(lock_file_path_.c_str(), open_flags, 0666);
        umask(old_umask);
        if (lock_fd_ >= 0) {
            fchmod(lock_fd_, 0666);
        }
    } else {
        lock_fd_ = open(lock_file_path_.c_str(), open_flags, 0666);
    }
    
    // If file doesn't exist and we need a shared lock, try to create it
    if (lock_fd_ < 0 && lock_type_ == CuidLockType::SHARED && errno == ENOENT) {
        mode_t old_umask = umask(0);
        lock_fd_ = open(lock_file_path_.c_str(), O_RDWR | O_CREAT, 0666);
        umask(old_umask);
        if (lock_fd_ >= 0) {
            fchmod(lock_fd_, 0666);
        }
    }
    
    if (lock_fd_ < 0) {
        LOG(ERROR, "CuidFileLock: Failed to open lock file " << lock_file_path_ 
            << ": " << strerror(errno));
        return false;
    }

    // Set up the lock structure
    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;  // Lock entire file
    fl.l_type = (lock_type_ == CuidLockType::EXCLUSIVE) ? F_WRLCK : F_RDLCK;

    // Retry loop with timeout
    const int retry_interval_ms = 50;  // 50ms between retries
    const int max_retries = (timeout_seconds * 1000) / retry_interval_ms;
    
    for (int retry = 0; retry <= max_retries; ++retry) {
        // Try non-blocking acquire
        if (fcntl(lock_fd_, F_SETLK, &fl) == 0) {
            is_locked_ = true;
            LOG(DEBUG, "CuidFileLock: Acquired " 
                << (lock_type_ == CuidLockType::EXCLUSIVE ? "exclusive" : "shared")
                << " lock on " << lock_file_path_ << " after " << retry << " retries");
            return true;
        }

        // Check if it's a "lock held" error vs real error
        if (errno != EACCES && errno != EAGAIN) {
            LOG(ERROR, "CuidFileLock: Failed to acquire lock on " << lock_file_path_ 
                << ": " << strerror(errno));
            close(lock_fd_);
            lock_fd_ = -1;
            return false;
        }

        // Wait before retry (unless this is the last iteration)
        if (retry < max_retries) {
            usleep(retry_interval_ms * 1000);  // Convert ms to microseconds
        }
    }

    // Timeout reached
    LOG(WARN, "CuidFileLock: Timeout after " << timeout_seconds 
        << " seconds waiting for lock on " << lock_file_path_);
    close(lock_fd_);
    lock_fd_ = -1;
    return false;
}

bool CuidFileLock::try_acquire() {
    if (is_locked_) {
        return true;  // Already locked
    }

    // For shared (read) locks, we only need O_RDONLY access
    // For exclusive (write) locks, we need O_RDWR access
    int open_flags = (lock_type_ == CuidLockType::EXCLUSIVE) ? (O_RDWR | O_CREAT) : O_RDONLY;
    
    // For exclusive locks (creating/writing), ensure proper permissions by clearing umask
    if (lock_type_ == CuidLockType::EXCLUSIVE) {
        mode_t old_umask = umask(0);
        lock_fd_ = open(lock_file_path_.c_str(), open_flags, 0666);
        umask(old_umask);
        if (lock_fd_ >= 0) {
            fchmod(lock_fd_, 0666);
        }
    } else {
        lock_fd_ = open(lock_file_path_.c_str(), open_flags, 0666);
    }
    
    // If file doesn't exist and we need a shared lock, try to create it
    if (lock_fd_ < 0 && lock_type_ == CuidLockType::SHARED && errno == ENOENT) {
        mode_t old_umask = umask(0);
        lock_fd_ = open(lock_file_path_.c_str(), O_RDWR | O_CREAT, 0666);
        umask(old_umask);
        if (lock_fd_ >= 0) {
            fchmod(lock_fd_, 0666);
        }
    }
    
    if (lock_fd_ < 0) {
        LOG(ERROR, "CuidFileLock: Failed to open lock file " << lock_file_path_ 
            << ": " << strerror(errno));
        return false;
    }

    // Set up the lock structure
    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;  // Lock entire file
    fl.l_type = (lock_type_ == CuidLockType::EXCLUSIVE) ? F_WRLCK : F_RDLCK;

    // F_SETLK: non-blocking - return immediately if can't acquire
    if (fcntl(lock_fd_, F_SETLK, &fl) < 0) {
        if (errno == EACCES || errno == EAGAIN) {
            // Lock is held by another process
            LOG(DEBUG, "CuidFileLock: Lock on " << lock_file_path_ 
                << " held by another process");
        } else {
            LOG(ERROR, "CuidFileLock: Failed to try_acquire lock on " << lock_file_path_ 
                << ": " << strerror(errno));
        }
        close(lock_fd_);
        lock_fd_ = -1;
        return false;
    }

    is_locked_ = true;
    LOG(DEBUG, "CuidFileLock: Acquired " 
        << (lock_type_ == CuidLockType::EXCLUSIVE ? "exclusive" : "shared")
        << " lock on " << lock_file_path_);
    return true;
}

void CuidFileLock::release() {
    if (!is_locked_ || lock_fd_ < 0) {
        return;
    }

    // Unlock the file
    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;  // Unlock entire file
    fl.l_type = F_UNLCK;

    if (fcntl(lock_fd_, F_SETLK, &fl) < 0) {
        LOG(ERROR, "CuidFileLock: Failed to release lock on " << lock_file_path_ 
            << ": " << strerror(errno));
    }

    close(lock_fd_);
    lock_fd_ = -1;
    is_locked_ = false;
    LOG(DEBUG, "CuidFileLock: Released lock on " << lock_file_path_);
}

// ============================================================================
// CuidFile Implementation
// ============================================================================

CuidFile::CuidFile(const std::string& file_path, bool is_privileged)
    : file_path_(file_path)
    , is_privileged_(is_privileged)
{
}

bool CuidFile::exists() const {
    struct stat buffer;
    return (stat(file_path_.c_str(), &buffer) == 0);
}

std::string CuidFile::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

amdcuid_device_type_t CuidFile::string_to_device_type(const std::string& str) const {
    if (str == "PLATFORM") return AMDCUID_DEVICE_TYPE_PLATFORM;
    if (str == "CPU") return AMDCUID_DEVICE_TYPE_CPU;
    if (str == "GPU") return AMDCUID_DEVICE_TYPE_GPU;
    if (str == "NIC") return AMDCUID_DEVICE_TYPE_NIC;
    return AMDCUID_DEVICE_TYPE_NONE;
}

amdcuid_id_t CuidFile::string_to_cuid(const std::string& str) const {
    amdcuid_id_t id;
    memset(id.bytes, 0, sizeof(id.bytes));
    
    // Parse UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    // Remove dashes and parse hex bytes
    std::string hex_only;
    for (char c : str) {
        if (c != '-') hex_only += c;
    }
    
    if (hex_only.length() != 32) {
        return id; // Return zero-filled ID on parse error
    }
    
    for (int i = 0; i < 16; ++i) {
        std::string byte_str = hex_only.substr(i * 2, 2);
        id.bytes[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    }
    
    return id;
}

bool CuidFile::parse_section_header(const std::string& line, 
                                     amdcuid_device_type_t& type, 
                                     uint32_t& index) const {
    // Parse lines like [GPU:0] or [PLATFORM]
    if (line.empty() || line[0] != '[' || line.back() != ']') {
        return false;
    }
    
    std::string content = line.substr(1, line.length() - 2);
    size_t colon_pos = content.find(':');
    
    if (colon_pos != std::string::npos) {
        // Format: TYPE:INDEX (index is in hex)
        std::string type_str = content.substr(0, colon_pos);
        std::string index_str = content.substr(colon_pos + 1);
        type = string_to_device_type(type_str);
        index = std::stoul(index_str, nullptr, 16);
    } else {
        // Format: PLATFORM (no index)
        type = string_to_device_type(content);
        index = 0;
    }
    
    return type != AMDCUID_DEVICE_TYPE_NONE;
}

amdcuid_status_t CuidFile::load() {
    entries_.clear();

    // Acquire shared lock for reading
    CuidFileLock lock(file_path_, CuidLockType::SHARED);
    if (!lock.acquire()) {
        LOG(ERROR, "CuidFile::load: Failed to acquire shared lock for " << file_path_);
        return AMDCUID_STATUS_FILE_ERROR;
    }

    std::ifstream file(file_path_);
    if (!file.is_open()) {
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }

    std::string line;
    CuidFileEntry current_entry;
    bool in_section = false;

    while (std::getline(file, line)) {
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }

        // Check for section header
        if (line[0] == '[') {
            // Save previous entry if valid
            if (in_section) {
                entries_.push_back(current_entry);
            }

            // Start new section
            amdcuid_device_type_t type;
            uint32_t index;
            if (parse_section_header(line, type, index)) {
                current_entry = CuidFileEntry();
                current_entry.device_type = type;
                current_entry.device_index = index;
                in_section = true;
            } else {
                in_section = false;
            }
            continue;
        }

        // Parse key=value pairs
        if (in_section) {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string value = trim(line.substr(eq_pos + 1));
                
                if (key == "primary_cuid") {
                    current_entry.primary_cuid = string_to_cuid(value);
                } else if (key == "derived_cuid") {
                    current_entry.derived_cuid = string_to_cuid(value);
                } else if (key == "device_node") {
                    current_entry.device_node = value;
                } else if (key == "package_core_id") {
                    current_entry.package_core_id = value;
                } else if (key == "bdf") {
                    current_entry.bdf = value;
                } else if (key == "mac_address") {
                    current_entry.mac_address = value;
                } else if (key == "hardware_fingerprint") {
                    current_entry.hardware_fingerprint = std::stoull(value);
                } else if (key == "vendor_id") {
                    current_entry.vendor_id = static_cast<uint16_t>(std::stoul(value, nullptr, 16));
                } else if (key == "device_id") {
                    current_entry.device_id = static_cast<uint16_t>(std::stoul(value, nullptr, 16));
                } else if (key == "revision_id") {
                    current_entry.revision_id = static_cast<uint8_t>(std::stoul(value, nullptr, 16));
                } else if (key == "family") {
                    current_entry.family = static_cast<uint16_t>(std::stoul(value, nullptr, 16));
                } else if (key == "model") {
                    current_entry.model = static_cast<uint16_t>(std::stoul(value, nullptr, 16));
                } else if (key == "pci_class") {
                    current_entry.pci_class = static_cast<uint16_t>(std::stoul(value, nullptr, 16));
                } else if (key == "unit_id") {
                    current_entry.unit_id = static_cast<uint16_t>(std::stoul(value, nullptr, 16));
                } else if (key == "last_update") {
                    current_entry.last_update = std::stol(value);
                }
            }
        }
    }

    // Save last entry
    if (in_section) {
        entries_.push_back(current_entry);
    }

    file.close();
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::save() {
    // Acquire exclusive lock for writing
    CuidFileLock lock(file_path_, CuidLockType::EXCLUSIVE);
    if (!lock.acquire()) {
        LOG(ERROR, "CuidFile::save: Failed to acquire exclusive lock for " << file_path_);
        return AMDCUID_STATUS_FILE_ERROR;
    }

    // Create temporary file first for atomic write
    std::string temp_path = file_path_ + ".tmp";
    std::ofstream file(temp_path);
    
    if (!file.is_open()) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    
    // Write header comment
    file << "# AMD CUID Device Information File\n";
    file << "# Auto-generated by AMD CUID library\n";
    file << "# DO NOT EDIT MANUALLY\n";
    file << "# File: " << file_path_ << "\n";
    if (is_privileged_) {
        file << "# Type: Privileged (contains primary CUIDs)\n";
        file << "# Permissions: Root access only\n";
    } else {
        file << "# Type: Unprivileged (derived CUIDs only)\n";
        file << "# Permissions: Readable by all users\n";
    }
    file << "\n";
    
    // Group entries by type for better organization
    std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>> grouped;
    get_grouped_entries(grouped);
    
    // Define output order
    std::vector<amdcuid_device_type_t> order = {
        AMDCUID_DEVICE_TYPE_GPU,
        AMDCUID_DEVICE_TYPE_CPU,
        AMDCUID_DEVICE_TYPE_NIC,
        AMDCUID_DEVICE_TYPE_PLATFORM
    };
    
    for (auto type : order) {
        if (grouped.find(type) == grouped.end()) continue;
        
        for (const auto& entry : grouped[type]) {
            // Write section header
            if (entry.device_type == AMDCUID_DEVICE_TYPE_PLATFORM) {
                file << "[" << CuidUtilities::device_type_to_string(entry.device_type) << "]\n";
            } else {
                file << "[" << CuidUtilities::device_type_to_string(entry.device_type) 
                     << ":" << entry.device_index << "]\n";
            }
            
            // Write primary CUID (privileged file only)
            if (is_privileged_) {
                file << "primary_cuid=" << CuidUtilities::get_cuid_as_string(&entry.primary_cuid) << "\n";
            }
            
            // Write derived CUID
            file << "derived_cuid=" << CuidUtilities::get_cuid_as_string(&entry.derived_cuid) << "\n";

            // Write hardware fingerprint (privileged file only)
            if (is_privileged_)
            {
                file << "hardware_fingerprint=" << entry.hardware_fingerprint << "\n";
            }

            // Write device-specific fields
            if (entry.vendor_id != 0) {
                file << "vendor_id=" << std::hex << std::setw(4) << std::setfill('0') << entry.vendor_id << "\n";
            }
            if (entry.device_id != 0) {
                file << "device_id=" << std::hex << std::setw(4) << std::setfill('0') << entry.device_id << "\n";
            }
            if (entry.revision_id != 0) {
                file << "revision_id=" << std::hex << std::setw(2) << std::setfill('0') << static_cast<uint16_t>(entry.revision_id) << "\n";
            }
            if (entry.family != 0) {
                file << "family=" << std::hex << std::setw(4) << std::setfill('0') << entry.family << "\n";
            }
            if (entry.model != 0) {
                file << "model=" << std::hex << std::setw(4) << std::setfill('0') << entry.model << "\n";
            }
            if (entry.pci_class != 0) {
                file << "pci_class=" << std::hex << std::setw(4) << std::setfill('0') << entry.pci_class << "\n";
            }
            if (entry.unit_id != 0) {
                file << "unit_id=" << std::hex << std::setw(4) << std::setfill('0') << entry.unit_id << "\n";
            }
            if (!entry.device_node.empty()) {
                file << "device_node=" << entry.device_node << "\n";
            }
            if (!entry.package_core_id.empty()) {
                file << "package_core_id=" << entry.package_core_id << "\n";
            }
            if (!entry.bdf.empty()) {
                file << "bdf=" << entry.bdf << "\n";
            }
            if (!entry.mac_address.empty()) {
                file << "mac_address=" << entry.mac_address << "\n";
            }
            
            // Write timestamp
            file << "last_update=" << entry.last_update << "\n";
            file << "\n";
        }
    }
    
    file.close();
    
    // Atomically move temp file to actual file
    if (rename(temp_path.c_str(), file_path_.c_str()) != 0) {
        unlink(temp_path.c_str());
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    
    // Set permissions
    if (!is_privileged_) {
        // Unprivileged file: readable by all (644)
        chmod(file_path_.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    } else {
        // Privileged file: readable by root only (600)
        chmod(file_path_.c_str(), S_IRUSR | S_IWUSR);
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::add_entry(const CuidFileEntry& entry) {
    // Check if entry with same derived CUID exists
    for (auto& existing : entries_) {
        if (memcmp(existing.derived_cuid.bytes, entry.derived_cuid.bytes, sizeof(amdcuid_id_t::bytes)) == 0) {
            // Update existing entry
            existing = entry;
            return AMDCUID_STATUS_SUCCESS;
        }
    }

    // Add new entry
    entries_.push_back(entry);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::remove_entry(const amdcuid_id_t& handle) {
    // search for entry by derived CUID and move that entry to the back, then erase it
    auto it = std::remove_if(entries_.begin(), entries_.end(),
                             [&handle](const CuidFileEntry& e) {
                                 return (memcmp(e.derived_cuid.bytes, handle.bytes, sizeof(amdcuid_id_t::bytes)) == 0);
                             });
    if (it != entries_.end()) {
        entries_.erase(it, entries_.end());
        return AMDCUID_STATUS_SUCCESS;
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_device_node(const std::string& device_node, 
                                                 CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (e.device_node == device_node) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_bdf(const std::string& bdf, 
                                         CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (e.bdf == bdf) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_package_core_id(const std::string& package_core_id, 
                                                     CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (e.package_core_id == package_core_id) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_device_type(amdcuid_device_type_t device_type, 
                                                 CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (e.device_type == device_type) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_derived_cuid(const amdcuid_id_t& derived_cuid, 
                                                    CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (memcmp(e.derived_cuid.bytes, derived_cuid.bytes, sizeof(amdcuid_id_t::bytes)) == 0) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

void CuidFile::get_grouped_entries(std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>>& grouped) const{
    grouped.clear();
    for (const auto& entry : entries_) {
        grouped[entry.device_type].push_back(entry);
    }
}


// ============================================================================
// CuidFileGenerator Implementation
// ============================================================================

amdcuid_status_t CuidFileGenerator::generate_from_devices(
    const std::vector<std::shared_ptr<CuidDevice>>& devices,
    const std::string& key_file_path,
    const std::string& unprivileged_file,
    const std::string& privileged_file)
{
    // Check if we have root privileges
    if (geteuid() != 0) {
        std::cerr << "Error: Root privileges are required to generate CUID files." << std::endl;
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    // Initialize HMAC for derived CUID generation
    cuid_hmac hmac = cuid_hmac();
    if (!hmac.is_valid()) {
        std::cerr << "Error: Failed to initialize HMAC with key" << std::endl;
        return AMDCUID_STATUS_KEY_ERROR;
    }

    // Create file handlers
    CuidFile unpriv_cuid_file(unprivileged_file, false);
    CuidFile priv_cuid_file(privileged_file, true);

    // Clear existing entries
    unpriv_cuid_file.clear();
    priv_cuid_file.clear();

    // Get current timestamp
    time_t now = time(nullptr);

    // Track device indices per type
    std::map<amdcuid_device_type_t, uint32_t> device_counters;

    // Process each device
    for (const auto& device : devices) {
        if (!device) continue;

        CuidFileEntry entry;
        entry.device_type = device->type();
        entry.device_index = device_counters[entry.device_type]++;
        entry.last_update = now;

        // Get primary CUID
        amdcuid_primary_id primary_id = {};
        amdcuid_status_t status = device->get_primary_cuid(primary_id);
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Warning: Failed to get primary CUID for device type " 
                      << entry.device_type << " status: " << status << std::endl;
            continue;
        }
        entry.primary_cuid = primary_id.UUIDv8_representation;

        // Generate derived CUID using HMAC
        amdcuid_derived_id derived_id = {};
        status = device->get_derived_cuid(derived_id, &hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Warning: Failed to generate derived CUID for device type " 
                      << entry.device_type << " status: " << status << std::endl;
            continue;
        }
        entry.derived_cuid = derived_id.UUIDv8_representation;

        // Fill in device-specific information
        switch (entry.device_type) {
            case AMDCUID_DEVICE_TYPE_GPU: {
                auto gpu = std::dynamic_pointer_cast<CuidGpu>(device);
                if (gpu) {
                    const auto& info = gpu->get_info();
                    entry.vendor_id = info.header.fields.gpu.vendor_id;
                    entry.device_id = info.header.fields.gpu.device_id;
                    entry.revision_id = info.header.fields.gpu.revision_id;
                    entry.pci_class = info.header.fields.gpu.pci_class;
                    entry.unit_id = info.header.fields.gpu.unit_id;
                    entry.device_node = info.render_node;
                    entry.bdf = info.bdf;
                }
                break;
            }
            case AMDCUID_DEVICE_TYPE_CPU: {
                auto cpu = std::dynamic_pointer_cast<CuidCpu>(device);
                if (cpu) {
                    const auto& info = cpu->get_info();
                    entry.vendor_id = info.header.fields.cpu.vendor_id;
                    entry.device_id = info.header.fields.cpu.device_id;
                    entry.revision_id = info.header.fields.cpu.revision_id;
                    entry.family = info.header.fields.cpu.family;
                    entry.model = info.header.fields.cpu.model;
                    entry.unit_id = info.header.fields.cpu.unit_id;
                    // Format: package:core
                    entry.package_core_id = std::to_string(info.header.fields.cpu.physical_id) + 
                                          ":" + std::to_string(info.header.fields.cpu.core);
                }
                break;
            }
            case AMDCUID_DEVICE_TYPE_NIC: {
                auto nic = std::dynamic_pointer_cast<CuidNic>(device);
                if (nic) {
                    const auto& info = nic->get_info();
                    entry.vendor_id = info.header.fields.nic.vendor_id;
                    entry.device_id = info.header.fields.nic.device_id;
                    entry.revision_id = info.header.fields.nic.revision_id;
                    entry.pci_class = info.header.fields.nic.pci_class;
                    entry.device_node = info.network_interface;
                    std::string mac_address;
                    if (nic->get_mac_address(mac_address) == AMDCUID_STATUS_SUCCESS) {
                        entry.mac_address = mac_address;
                    }
                    entry.bdf = info.bdf;
                }
                break;
            }
            case AMDCUID_DEVICE_TYPE_PLATFORM: {
                auto platform = std::dynamic_pointer_cast<CuidPlatform>(device);
                // Platform only has vendor_id
                if (platform) {
                    const auto& info = platform->get_info();
                    entry.vendor_id = info.header.fields.platform.vendor_id;
                }
                break;
            }
            default:
                break;
        }

        // Add to both files
        unpriv_cuid_file.add_entry(entry);
        priv_cuid_file.add_entry(entry);
    }
    
    // Save unprivileged file
    amdcuid_status_t status = unpriv_cuid_file.save();
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to save unprivileged CUID file: " 
                  << unprivileged_file << std::endl;
        return status;
    }
    std::cout << "Successfully generated: " << unprivileged_file << std::endl;

    // Save privileged file
    status = priv_cuid_file.save();
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to save privileged CUID file: " 
                    << privileged_file << std::endl;
        return status;
    }
    std::cout << "Successfully generated: " << privileged_file << std::endl;

    return AMDCUID_STATUS_SUCCESS;
}
