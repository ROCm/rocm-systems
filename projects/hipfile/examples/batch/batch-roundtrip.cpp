/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* batch-roundtrip - Seed an input file, read it into a registered GPU buffer
 * using a continuously refilled batch, write it in a second batch phase, and
 * verify the output matches the input.
 *
 * Usage: ./batch-roundtrip READ_FILE WRITE_FILE FILE_SIZE GPUID
 *
 *   READ_FILE   Input path to create or truncate and seed with test data.
 *   WRITE_FILE  Distinct output path to create or truncate.
 *   FILE_SIZE   Positive base-10 payload size in bytes.
 *   GPUID       GPU device index to use.
 *
 * The GPU buffer holds the whole file rounded up to a 4 KiB boundary. Each
 * phase starts with up to BR_BATCH_SIZE requests (default 128), waits for at
 * least one completion, then submits one replacement for each returned event
 * until all 4 KiB requests have completed. All reads finish before any writes
 * are submitted. The final padded write is truncated to FILE_SIZE.
 */

#include "examples_common.h"

#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

/// @brief Maximum number of operations kept in flight at once.
#ifndef BR_BATCH_SIZE
#define BR_BATCH_SIZE 128U
#endif

static_assert(BR_BATCH_SIZE > 0, "BR_BATCH_SIZE must be greater than zero");
static_assert(BR_BATCH_SIZE <= 128, "BR_BATCH_SIZE exceeds hipFile's maximum batch capacity");

namespace {

constexpr size_t BATCH_IO_SIZE = BLOCK_ALIGN;

struct BatchCookie {
    size_t chunk_index;
    size_t expected_bytes;
};

struct BatchRequest {
    hipFileIOParams_t            operation{};
    std::unique_ptr<BatchCookie> cookie;
    bool                         submitted{};
};

template <typename Integral>
    requires std::is_integral_v<Integral>
Integral
parse_integral(std::string_view text)
{
    Integral    value{};
    const char *end    = text.data() + text.size();
    const auto  result = std::from_chars(text.data(), end, value, 10);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument{"Invalid integral"};
    }
    return value;
}

const char *
operation_name(hipFileOpcode_t opcode)
{
    switch (opcode) {
        case hipFileBatchRead:
            return "read";
        case hipFileBatchWrite:
            return "write";
        default:
            return "unknown";
    }
}

void
configure_request(BatchRequest &request, size_t chunk_index, hipFileHandle_t file_handle,
                  hipFileOpcode_t opcode, void *device_buffer, size_t payload_size)
{
    const size_t operation_offset = chunk_index * BATCH_IO_SIZE;
    const size_t bytes_remaining  = payload_size - std::min(payload_size, operation_offset);
    const size_t expected_bytes =
        opcode == hipFileBatchRead ? std::min(bytes_remaining, BATCH_IO_SIZE) : BATCH_IO_SIZE;

    if (request.cookie == nullptr)
        request.cookie = std::make_unique<BatchCookie>();
    request.cookie->chunk_index    = chunk_index;
    request.cookie->expected_bytes = expected_bytes;

    request.operation                       = {};
    request.operation.mode                  = hipFileBatch;
    request.operation.u.batch.devPtr_base   = device_buffer;
    request.operation.u.batch.file_offset   = static_cast<int64_t>(operation_offset);
    request.operation.u.batch.devPtr_offset = static_cast<int64_t>(operation_offset);
    request.operation.u.batch.size          = BATCH_IO_SIZE;
    request.operation.fh                    = file_handle;
    request.operation.opcode                = opcode;
    request.operation.cookie                = request.cookie.get();
    request.submitted                       = false;
}

int
run_batch_phase(hipFileBatchHandle_t batch_handle, hipFileHandle_t file_handle, hipFileOpcode_t opcode,
                void *device_buffer, size_t payload_size, unsigned batch_capacity)
{
    const char  *op_name         = operation_name(opcode);
    const size_t operation_count = align_up(payload_size, BATCH_IO_SIZE) / BATCH_IO_SIZE;
    const size_t request_count   = std::min(operation_count, static_cast<size_t>(batch_capacity));

    std::vector<BatchRequest> requests(request_count);
    size_t                    next_chunk = 0;
    for (auto &request : requests) {
        configure_request(request, next_chunk, file_handle, opcode, device_buffer, payload_size);
        ++next_chunk;
    }

    std::vector<hipFileIOEvents_t> events(batch_capacity);
    std::vector<hipFileIOParams_t> submissions;
    submissions.reserve(batch_capacity);
    size_t in_flight = 0;

    while (!requests.empty()) {
        submissions.clear();
        const size_t available = static_cast<size_t>(batch_capacity) - in_flight;
        for (auto &request : requests) {
            if (submissions.size() == available)
                break;
            if (request.submitted)
                continue;
            submissions.push_back(request.operation);
            request.submitted = true;
        }

        if (!submissions.empty()) {
            const hipFileError_t hipfile_err = hipFileBatchIOSubmit(
                batch_handle, static_cast<unsigned>(submissions.size()), submissions.data(), /*flags=*/0);
            if (hipFileSuccess != hipfile_err.err) {
                fprintf(stderr, "Could not submit %s batch (%s)\n", op_name,
                        hipFileGetOpErrorString(hipfile_err.err));
                return 1;
            }
            in_flight += submissions.size();
        }

        unsigned             nr = batch_capacity;
        const hipFileError_t hipfile_err =
            hipFileBatchIOGetStatus(batch_handle, /*min_nr=*/1, &nr, events.data(), /*timeout=*/nullptr);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "Could not get %s batch status (%s)\n", op_name,
                    hipFileGetOpErrorString(hipfile_err.err));
            return 1;
        }
        if (nr == 0 || nr > in_flight) {
            fprintf(stderr, "Invalid number of %s batch events: %u\n", op_name, nr);
            return 1;
        }

        for (unsigned i = 0; i < nr; ++i) {
            const hipFileIOEvents_t &event = events[i];
            const auto request = std::find_if(requests.begin(), requests.end(), [&event](const auto &entry) {
                return event.cookie == entry.cookie.get();
            });
            if (request == requests.end()) {
                fprintf(stderr, "%s batch returned an unknown cookie\n", op_name);
                return 1;
            }
            const BatchCookie *cookie = request->cookie.get();
            if (event.status != hipFileComplete) {
                fprintf(stderr, "%s batch request %zu failed: status=%d ret=%zd\n", op_name,
                        cookie->chunk_index, static_cast<int>(event.status), static_cast<ssize_t>(event.ret));
                return 1;
            }
            if (event.ret != cookie->expected_bytes) {
                fprintf(stderr, "Short %s for batch request %zu: %zu of %zu bytes\n", op_name,
                        cookie->chunk_index, event.ret, cookie->expected_bytes);
                return 1;
            }

            if (next_chunk < operation_count) {
                configure_request(*request, next_chunk, file_handle, opcode, device_buffer, payload_size);
                ++next_chunk;
            }
            else {
                if (request != requests.end() - 1)
                    std::iter_swap(request, requests.end() - 1);
                requests.pop_back();
            }
        }
        in_flight -= nr;
    }

    return 0;
}

} // namespace

int
main(int argc, char *argv[])
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s READ_FILE WRITE_FILE FILE_SIZE GPUID\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *read_path  = argv[1];
    const char *write_path = argv[2];

    size_t payload_size;
    try {
        payload_size = parse_integral<size_t>(argv[3]);
    }
    catch (const std::invalid_argument &) {
        fprintf(stderr, "Invalid FILE_SIZE: %s\n", argv[3]);
        return EXIT_FAILURE;
    }
    if (payload_size == 0) {
        fprintf(stderr, "FILE_SIZE must be greater than zero\n");
        return EXIT_FAILURE;
    }

    unsigned gpu_id;
    try {
        gpu_id = parse_integral<unsigned>(argv[4]);
    }
    catch (const std::invalid_argument &) {
        fprintf(stderr, "Invalid GPU ID: %s\n", argv[4]);
        return EXIT_FAILURE;
    }

    if (seed_read_file(read_path, payload_size))
        return EXIT_FAILURE;

    struct stat read_stat;
    if (stat(read_path, &read_stat) || read_stat.st_size != static_cast<off_t>(payload_size)) {
        fprintf(stderr, "Could not create %s with size %zu\n", read_path, payload_size);
        return EXIT_FAILURE;
    }

    const size_t   buffer_size    = align_up(payload_size, BATCH_IO_SIZE);
    const unsigned batch_capacity = BR_BATCH_SIZE;

    int                  read_fd = -1, write_fd = -1;
    hipFileHandle_t      read_handle = nullptr, write_handle = nullptr;
    hipFileBatchHandle_t batch_handle     = nullptr;
    bool                 read_handle_open = false, write_handle_open = false;
    void                *device_buffer     = nullptr;
    bool                 buffer_registered = false;
    int                  exit_status       = EXIT_FAILURE;
    hipFileError_t       hipfile_err{};

    hipError_t hip_err = hipSetDevice(static_cast<int>(gpu_id));
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not select GPU %u (%d)\n", gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    hip_err = hipMalloc(&device_buffer, buffer_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not allocate %zu bytes on GPU %u (%d)\n", buffer_size, gpu_id, hip_err);
        return EXIT_FAILURE;
    }

    hip_err = hipMemset(device_buffer, 0, buffer_size);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not zero device buffer (%d)\n", hip_err);
        goto cleanup;
    }
    hip_err = hipDeviceSynchronize();
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not synchronize after memset (%d)\n", hip_err);
        goto cleanup;
    }

    hipfile_err = hipFileBufRegister(device_buffer, buffer_size, 0);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Buffer register failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
        goto cleanup;
    }
    buffer_registered = true;

    if (open_file(read_path, O_RDONLY | O_DIRECT, 0, &read_fd, &read_handle))
        goto cleanup;
    read_handle_open = true;

    if (open_file(write_path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                  &write_fd, &write_handle)) {
        goto cleanup;
    }
    write_handle_open = true;

    hipfile_err = hipFileBatchIOSetUp(&batch_handle, batch_capacity);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Could not set up batch handle (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
        goto cleanup;
    }

    if (run_batch_phase(batch_handle, read_handle, hipFileBatchRead, device_buffer, payload_size,
                        batch_capacity)) {
        goto cleanup;
    }
    if (run_batch_phase(batch_handle, write_handle, hipFileBatchWrite, device_buffer, payload_size,
                        batch_capacity)) {
        goto cleanup;
    }

    if (ftruncate(write_fd, static_cast<off_t>(payload_size))) {
        fprintf(stderr, "Could not truncate %s (%s)\n", write_path, strerror(errno));
        goto cleanup;
    }

    {
        const int close_status = close_file(write_path, write_fd, write_handle);
        write_fd               = -1;
        write_handle           = nullptr;
        write_handle_open      = false;
        if (close_status)
            goto cleanup;
    }
    {
        const int close_status = close_file(read_path, read_fd, read_handle);
        read_fd                = -1;
        read_handle            = nullptr;
        read_handle_open       = false;
        if (close_status)
            goto cleanup;
    }

    {
        uint64_t hash;
        if (verify_files_match(read_path, write_path, payload_size, &hash))
            goto cleanup;
        printf("OK  %s == %s  (%zu bytes, hash 0x%016" PRIx64 ")\n", read_path, write_path, payload_size,
               hash);
    }

    exit_status = EXIT_SUCCESS;

cleanup:
    if (batch_handle != nullptr)
        hipFileBatchIODestroy(batch_handle);

    if (write_handle_open) {
        if (close_file(write_path, write_fd, write_handle))
            exit_status = EXIT_FAILURE;
    }
    if (read_handle_open) {
        if (close_file(read_path, read_fd, read_handle))
            exit_status = EXIT_FAILURE;
    }

    if (buffer_registered) {
        hipfile_err = hipFileBufDeregister(device_buffer);
        if (hipFileSuccess != hipfile_err.err) {
            fprintf(stderr, "Buffer deregister failed (%s)\n", hipFileGetOpErrorString(hipfile_err.err));
            exit_status = EXIT_FAILURE;
        }
    }

    hip_err = hipFree(device_buffer);
    if (hipSuccess != hip_err) {
        fprintf(stderr, "Could not free device buffer (%d)\n", hip_err);
        exit_status = EXIT_FAILURE;
    }

    return exit_status;
}
