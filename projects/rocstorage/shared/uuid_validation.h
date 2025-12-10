// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <string>
#include <algorithm>
#include <cctype>

namespace rocstorage {
namespace validation {

// UUID Validation for rocstorage
//
// UUIDs are used as suffixes in table names (e.g., rocpd_event_<uuid>).
// The rocstorage-reader extracts GUIDs by parsing table names after a known
// prefix (e.g., "rocpd_info_node_"). If UUIDs contain underscores or special
// characters, this extraction can fail.
//
// Valid UUID characters: alphanumeric only (a-z, A-Z, 0-9)
// This ensures the reader can reliably extract GUIDs from table names.

/**
 * @brief Check if a UUID string is valid for use in rocstorage table names.
 *
 * Valid UUIDs contain only alphanumeric characters (a-z, A-Z, 0-9).
 * This restriction ensures compatibility with rocstorage-reader's GUID
 * extraction logic which parses table name suffixes.
 *
 * @param uuid The UUID string to validate
 * @return true if the UUID is valid, false otherwise
 */
inline bool is_valid_uuid(const std::string& uuid) {
    if (uuid.empty()) {
        return false;
    }

    return std::all_of(uuid.begin(), uuid.end(), [](unsigned char c) {
        return std::isalnum(c);
    });
}

/**
 * @brief Get an error message for an invalid UUID.
 *
 * @param uuid The invalid UUID string
 * @return A descriptive error message
 */
inline std::string get_uuid_validation_error(const std::string& uuid) {
    if (uuid.empty()) {
        return "UUID cannot be empty";
    }

    for (size_t i = 0; i < uuid.size(); ++i) {
        unsigned char c = uuid[i];
        if (!std::isalnum(c)) {
            return "UUID contains invalid character '" + std::string(1, c) +
                   "' at position " + std::to_string(i) +
                   ". Only alphanumeric characters (a-z, A-Z, 0-9) are allowed.";
        }
    }

    return "UUID is valid";
}

} // namespace validation
} // namespace rocstorage
