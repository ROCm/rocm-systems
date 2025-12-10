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

// Shared schema version between rocstorage (writer) and rocstorage-reader
// This version must be updated in sync when schema changes are made.
//
// Version History:
//   v3: Current version - rocpd_* table format with UUID suffixes
//   v4: (Future) Track-based indexing

#define ROCSTORAGE_SCHEMA_VERSION_MAJOR 3
#define ROCSTORAGE_SCHEMA_VERSION_MINOR 0
#define ROCSTORAGE_SCHEMA_VERSION_PATCH 0

#define ROCSTORAGE_SCHEMA_VERSION_STRING "3"

// For C++ code
#ifdef __cplusplus
namespace rocstorage {
namespace schema {

constexpr int kSchemaVersionMajor = ROCSTORAGE_SCHEMA_VERSION_MAJOR;
constexpr int kSchemaVersionMinor = ROCSTORAGE_SCHEMA_VERSION_MINOR;
constexpr int kSchemaVersionPatch = ROCSTORAGE_SCHEMA_VERSION_PATCH;
constexpr const char* kSchemaVersionString = ROCSTORAGE_SCHEMA_VERSION_STRING;

} // namespace schema
} // namespace rocstorage
#endif
