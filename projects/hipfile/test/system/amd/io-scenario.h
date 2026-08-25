/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "configuration.h"
#include "context.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-test.h"
#include "io-verify.h"
#include "test-common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <optional>
#include <string>
#include <vector>

namespace hipFileTest {

// ---------------------------------------------------------------------------
// Kernel launch shape.
// ---------------------------------------------------------------------------
enum class GridMode { Default, OneWorkgroup, ManyWorkgroups };

constexpr unsigned kManyWorkgroups = 300;

inline dim3
gridFor(GridMode mode, size_t n)
{
    switch (mode) {
        case GridMode::OneWorkgroup:
            return dim3(1);
        case GridMode::ManyWorkgroups:
            return dim3(kManyWorkgroups);
        case GridMode::Default:
        default:
            return defaultGrid(n);
    }
}

// ---------------------------------------------------------------------------
// One parameter type for every data-modification suite.
// ---------------------------------------------------------------------------
struct IoTestScenario {
    std::string               name{};                             // the full gtest case name
    IoTestBackend             backend  = IoTestBackend::Fallback; // backend fulfilling the I/O request
    size_t                    io_bytes = 4_KiB;                   // bytes transferred through the hipFile API
    hoff_t                    file_off = 0;                       // where the data sits in the file
    hoff_t                    buf_off  = 0;                       // where the data sits in the device buffer
    size_t                    stride   = 1;                       // 1 == modify every element
    GridMode                  grid     = GridMode::Default;       // workgroup count the kernel launches with
    std::optional<ExtendCase> ext      = std::nullopt;            // set only for extending writes
};

// A value of one of the parameters of an `IoTestScenario` and the string to reflect
// it with in the test name.
template <class T> struct Axis {
    T           value;
    std::string name;
};

// ---------------------------------------------------------------------------
// This class is responsible for building a series of IoTestScenarios. Allowing you
// to use a single class/object to represent all of the parameters of a test instance,
// as opposed to many small classes.
//
// This class contains functions that mimic functionality of GTest's `param_generator`s
// (e.g., `over` for `testing::Combine`, `add` for `testing::Values`).
// ---------------------------------------------------------------------------
class IoTestScenarioSet {
public:
    explicit IoTestScenarioSet(const IoTestScenario &base) : rows_{base}
    {
    }

    // This function is like GTest's `testing::Combine`. You specify a `field` of `IoTestScenario`
    // that you would like to vary over a range of values defined by `axis`, and `over`
    // multiplies each existing `IoTestScenario` in `rows_` to have an instance that takes on
    // each of the values in `axis`.
    template <class F, class Range> IoTestScenarioSet &over(F IoTestScenario::*field, const Range &axis)
    {
        std::vector<IoTestScenario> expanded;
        for (const IoTestScenario &row : rows_) {
            for (const auto &a : axis) {
                IoTestScenario next = row;
                next.*field         = a.value;
                next.name           = next.name.empty() ? a.name : next.name + "_" + a.name;
                expanded.push_back(next);
            }
        }
        rows_ = expanded;
        return *this;
    }

    // If you were to repeatedly use this function, this function is like GTest's `testing::Values`,
    // except you can append to an existing set of `IoTestScenario`.
    // This function should we used for adding bespoke test parameter scenarios that cannot be
    // expressed by a matrix of individual parameters that are combined by `over`.
    IoTestScenarioSet &add(const IoTestScenario &one)
    {
        rows_.push_back(one);
        return *this;
    }

    // This function combines two sets of `IoTestScenario`, which typically isn't possible in GTest,
    // unless you used multiple `INSTANTIATE_TEST_SUITE_P` calls.
    IoTestScenarioSet &add(const IoTestScenarioSet &other)
    {
        rows_.insert(rows_.end(), other.rows_.begin(), other.rows_.end());
        return *this;
    }

    std::vector<IoTestScenario> build() const
    {
        return rows_;
    }

private:
    std::vector<IoTestScenario> rows_;
};

inline std::string
ioTestScenarioName(const testing::TestParamInfo<IoTestScenario> &info)
{
    return info.param.name;
}

// ---------------------------------------------------------------------------
// Axis values that are common between many test suites.
// ---------------------------------------------------------------------------
HIPFILE_WARN_NO_EXIT_DTOR_OFF
inline const std::array<Axis<IoTestBackend>, 2> kBackends{{
    {IoTestBackend::Fastpath, "Fastpath"},
    {IoTestBackend::Fallback, "Fallback"},
}};

inline const std::array<Axis<size_t>, 3> kCombinedSizes{{
    {4_KiB, "sub_chunk"},
    {kChunkBytes + 4_KiB, "cross_chunk"},
    {2 * kChunkBytes, "multi_chunk"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

// ---------------------------------------------------------------------------
// Element policies that determine the datatype that the GPU kernels will operate on.
// ---------------------------------------------------------------------------
struct IntElementPolicy {
    using Elem = int32_t;

    static constexpr uint8_t kDevFill = kSentinelByte;

    static constexpr size_t elems(size_t bytes)
    {
        return bytes / sizeof(Elem);
    }

    static void seedFileData(int fd, hoff_t off, size_t n)
    {
        seedFilePattern(fd, off, n);
    }

    static void seedFileSlack(int fd, hoff_t off, size_t n)
    {
        seedFileConstant(fd, off, n, kSentinel);
    }

    static void seedDeviceData(void *buf, hoff_t off, size_t n)
    {
        seedDevicePattern(buf, off, n);
    }

    static std::vector<Elem> readFile(int fd, hoff_t off, size_t n)
    {
        return readFileInts(fd, off, n);
    }

    static void assertFileSlack(const Elem *arr, size_t from, size_t to)
    {
        assertConstant(arr, from, to, kSentinel);
    }

    static void assertModified(const Elem *arr, size_t n, size_t stride)
    {
        assertModifiedPattern(arr, n, stride);
    }

    static void verifyAndModify(Elem *start, size_t alloc_n, size_t data_start, size_t n, dim3 grid,
                                dim3 workgroup, size_t stride)
    {
        assertVerifyAndModify(start, alloc_n, data_start, n, grid, workgroup, stride);
    }
};

struct ByteElementPolicy {
    using Elem = uint8_t;

    static constexpr uint8_t kDevFill = kByteDevSlack;

    static constexpr size_t elems(size_t bytes)
    {
        return bytes;
    }

    static void seedFileData(int fd, hoff_t off, size_t n)
    {
        seedFileBytesConstant(fd, off, n, kByteEntry);
    }

    static void seedFileSlack(int fd, hoff_t off, size_t n)
    {
        seedFileBytesConstant(fd, off, n, kByteFileSlack);
    }

    static void seedDeviceData(void *buf, hoff_t off, size_t n)
    {
        ASSERT_EQ(hipSuccess,
                  hipMemset(static_cast<uint8_t *>(buf) + static_cast<size_t>(off), kByteEntry, n));
    }

    static std::vector<Elem> readFile(int fd, hoff_t off, size_t n)
    {
        return readFileBytes(fd, off, n);
    }

    static void assertFileSlack(const Elem *arr, size_t from, size_t to)
    {
        assertBytesConstant(arr, from, to, kByteFileSlack);
    }

    static void assertModified(const Elem *arr, size_t n, size_t stride)
    {
        assertBytesModified(arr, n, stride);
    }

    static void verifyAndModify(Elem *start, size_t alloc_bytes, size_t data_start, size_t n, dim3 grid,
                                dim3 workgroup, size_t stride)
    {
        assertVerifyAndModifyBytes(start, alloc_bytes, data_start, n, grid, workgroup, stride);
    }
};

// ---------------------------------------------------------------------------
// The base fixture for all of the data modification suites to reuse.
//
// Individual suites should use ftruncate to size the file's initial length before
// calling this SetUp.
// ---------------------------------------------------------------------------
template <class Policy> struct DataModificationBase : public testing::TestWithParam<IoTestScenario> {

    const IoTestBackend backend{GetParam().backend};   // backend fulfilling the I/O request
    const size_t        io_bytes{GetParam().io_bytes}; // bytes transferred using the hipFile API
    // Over-allocate a device sentinel region on each side of the data.
    // Device layout (each sentinel region 4_KiB, data io_bytes):
    // [head device sentinel region][data][tail device sentinel region].
    const size_t buffer_bytes{io_bytes + 2 * 4_KiB};        // total device buffer size
    const size_t io_elems{Policy::elems(io_bytes)};         // transferred elements
    const size_t buffer_elems{Policy::elems(buffer_bytes)}; // elements spanning the whole buffer

    Tmpfile                tmpfile;
    hipFileHandle_t        tmpfile_handle{nullptr};
    void                  *device_buffer{nullptr}; // allocated by SetUp
    typename Policy::Elem *buffer_start{nullptr};  // typed view of device_buffer

    DataModificationBase() : tmpfile{test_env.ais_capable_dir}
    {
    }

    void SetUp() override
    {
        hipFile::Context<hipFile::Configuration>::get()->fastpath(false);
        hipFile::Context<hipFile::Configuration>::get()->fallback(false);

        switch (backend) {
            case IoTestBackend::Fastpath:
                hipFile::Context<hipFile::Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                hipFile::Context<hipFile::Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        buffer_start = static_cast<typename Policy::Elem *>(device_buffer);
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, Policy::kDevFill, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        if (backend == IoTestBackend::Fastpath) {
            enforceFastpathGate(tmpfile_handle, device_buffer);
        }
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }
};

// ---------------------------------------------------------------------------
// Test bodies that are shared between multiple data modification suites.
//
// These bodies need to be called with ASSERT_NO_FATAL_FAILURE from a TEST_P in order for
// failures to propagate.
// ---------------------------------------------------------------------------

// Round trip with both the device sentinel regions and the file sentinel regions guarded
// (checked to make sure they are not erroneously modified).
template <class Policy>
void
runAllRegionsTest(DataModificationBase<Policy> &f)
{
    using Elem = typename Policy::Elem;

    const IoTestScenario &s = f.GetParam();

    const size_t n        = f.io_elems;
    const size_t slack_n  = Policy::elems(4_KiB);
    const hoff_t buf_off  = s.buf_off;
    const hoff_t file_off = s.file_off;
    const hoff_t head_off = file_off - static_cast<hoff_t>(4_KiB); // head file sentinel region
    const hoff_t tail_off = file_off + static_cast<hoff_t>(f.io_bytes);

    // File layout (each sentinel region slack_n elements, data n elements; data begins at
    // file_off past the chunk boundary):
    // [unwritten hole = 0][head file sentinel region][data][tail file sentinel region].
    Policy::seedFileSlack(f.tmpfile.fd, head_off, slack_n);
    Policy::seedFileData(f.tmpfile.fd, file_off, n);
    Policy::seedFileSlack(f.tmpfile.fd, tail_off, slack_n);

    ASSERT_EQ(static_cast<ssize_t>(f.io_bytes),
              hipFileRead(f.tmpfile_handle, f.device_buffer, f.io_bytes, file_off, buf_off));

    // Device layout (each sentinel region slack_n elements, data n elements):
    // [head device sentinel region][data][tail device sentinel region].
    Policy::verifyAndModify(f.buffer_start, f.buffer_elems, Policy::elems(static_cast<size_t>(buf_off)), n,
                            gridFor(s.grid, n), dim3(kDefaultWorkgroupSize), s.stride);

    ASSERT_EQ(static_cast<ssize_t>(f.io_bytes),
              hipFileWrite(f.tmpfile_handle, f.device_buffer, f.io_bytes, file_off, buf_off));

    // Start of file that was supposed to be untouched remains untouched.
    assertHoleZero(f.tmpfile.fd, 0, head_off);
    // Existing data before write is fully intact.
    std::vector<Elem> head = Policy::readFile(f.tmpfile.fd, head_off, slack_n);
    Policy::assertFileSlack(head.data(), 0, slack_n);
    // Modified data is correctly modified.
    std::vector<Elem> body = Policy::readFile(f.tmpfile.fd, file_off, n);
    Policy::assertModified(body.data(), n, s.stride);
    // Existing data after write is fully intact.
    std::vector<Elem> tail = Policy::readFile(f.tmpfile.fd, tail_off, slack_n);
    Policy::assertFileSlack(tail.data(), 0, slack_n);
}

} // namespace hipFileTest
