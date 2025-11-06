// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <unistd.h>

// Random number generator
std::random_device rd;
std::mt19937_64 gen(rd());

uint64_t random_uint64() {
    std::uniform_int_distribution<uint64_t> dist;
    return dist(gen);
}

uint32_t random_uint32() {
    std::uniform_int_distribution<uint32_t> dist;
    return dist(gen);
}

double random_double() {
    std::uniform_real_distribution<double> dist(0.0, 1000000.0);
    return dist(gen);
}

// Structure to hold a record and its data
struct TestRecord {
    rocprofiler_record_header_t header;
    std::vector<char> data;
};

// Generate a random HIP API record
TestRecord generate_hip_api_record() {
    TestRecord rec;

    rocprofiler_buffer_tracing_hip_api_record_t hip_record;
    hip_record.size = sizeof(hip_record);
    hip_record.kind = ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API;
    hip_record.operation = random_uint32() % 100;
    hip_record.correlation_id.internal = random_uint64();
    hip_record.start_timestamp = random_uint64();
    hip_record.end_timestamp = hip_record.start_timestamp + random_uint64() % 1000000;
    hip_record.thread_id = random_uint64();

    rec.data.resize(sizeof(hip_record));
    std::memcpy(rec.data.data(), &hip_record, sizeof(hip_record));

    rec.header.category = ROCPROFILER_BUFFER_CATEGORY_TRACING;
    rec.header.kind = ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API;
    rec.header.payload = nullptr; // Will be set later

    return rec;
}

// Generate a random HSA API record
TestRecord generate_hsa_api_record() {
    TestRecord rec;

    rocprofiler_buffer_tracing_hsa_api_record_t hsa_record;
    hsa_record.size = sizeof(hsa_record);
    hsa_record.kind = ROCPROFILER_BUFFER_TRACING_HSA_CORE_API;
    hsa_record.operation = random_uint32() % 50;
    hsa_record.correlation_id.internal = random_uint64();
    hsa_record.start_timestamp = random_uint64();
    hsa_record.end_timestamp = hsa_record.start_timestamp + random_uint64() % 500000;
    hsa_record.thread_id = random_uint64();

    rec.data.resize(sizeof(hsa_record));
    std::memcpy(rec.data.data(), &hsa_record, sizeof(hsa_record));

    rec.header.category = ROCPROFILER_BUFFER_CATEGORY_TRACING;
    rec.header.kind = ROCPROFILER_BUFFER_TRACING_HSA_CORE_API;
    rec.header.payload = nullptr;

    return rec;
}

// Generate a random kernel dispatch record
TestRecord generate_kernel_dispatch_record() {
    TestRecord rec;

    rocprofiler_buffer_tracing_kernel_dispatch_record_t kernel_record;
    kernel_record.size = sizeof(kernel_record);
    kernel_record.kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
    kernel_record.operation = ROCPROFILER_KERNEL_DISPATCH_ENQUEUE;
    kernel_record.correlation_id.internal = random_uint64();
    kernel_record.start_timestamp = random_uint64();
    kernel_record.end_timestamp = kernel_record.start_timestamp + random_uint64() % 10000000;
    kernel_record.thread_id = random_uint64();

    kernel_record.dispatch_info.size = sizeof(rocprofiler_kernel_dispatch_info_t);
    kernel_record.dispatch_info.agent_id.handle = random_uint64();
    kernel_record.dispatch_info.queue_id.handle = random_uint64();
    kernel_record.dispatch_info.kernel_id = random_uint64();
    kernel_record.dispatch_info.dispatch_id = random_uint64();
    kernel_record.dispatch_info.private_segment_size = random_uint32() % 65536;
    kernel_record.dispatch_info.group_segment_size = random_uint32() % 65536;
    kernel_record.dispatch_info.workgroup_size.x = 256;
    kernel_record.dispatch_info.workgroup_size.y = 1;
    kernel_record.dispatch_info.workgroup_size.z = 1;
    kernel_record.dispatch_info.grid_size.x = 1024 * 256;
    kernel_record.dispatch_info.grid_size.y = 1;
    kernel_record.dispatch_info.grid_size.z = 1;

    rec.data.resize(sizeof(kernel_record));
    std::memcpy(rec.data.data(), &kernel_record, sizeof(kernel_record));

    rec.header.category = ROCPROFILER_BUFFER_CATEGORY_TRACING;
    rec.header.kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
    rec.header.payload = nullptr;

    return rec;
}

// Generate a random memory copy record
TestRecord generate_memory_copy_record() {
    TestRecord rec;

    rocprofiler_buffer_tracing_memory_copy_record_t mem_record;
    mem_record.size = sizeof(mem_record);
    mem_record.kind = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
    mem_record.operation = ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE;
    mem_record.correlation_id.internal = random_uint64();
    mem_record.thread_id = random_uint64();
    mem_record.start_timestamp = random_uint64();
    mem_record.end_timestamp = mem_record.start_timestamp + random_uint64() % 2000000;
    mem_record.dst_agent_id.handle = random_uint64();
    mem_record.src_agent_id.handle = random_uint64();
    mem_record.bytes = random_uint64() % (1024 * 1024 * 1024); // Up to 1GB
    mem_record.dst_address.value = random_uint64();
    mem_record.src_address.value = random_uint64();

    rec.data.resize(sizeof(mem_record));
    std::memcpy(rec.data.data(), &mem_record, sizeof(mem_record));

    rec.header.category = ROCPROFILER_BUFFER_CATEGORY_TRACING;
    rec.header.kind = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
    rec.header.payload = nullptr;

    return rec;
}

// Generate a random counter dispatch header
TestRecord generate_counter_dispatch_header() {
    TestRecord rec;

    rocprofiler_dispatch_counting_service_record_t counter_header;
    counter_header.num_records = 5; // Will generate 5 counter values
    counter_header.correlation_id.internal = random_uint64();
    counter_header.start_timestamp = random_uint64();
    counter_header.end_timestamp = counter_header.start_timestamp + random_uint64() % 5000000;

    counter_header.dispatch_info.size = sizeof(rocprofiler_kernel_dispatch_info_t);
    counter_header.dispatch_info.agent_id.handle = random_uint64();
    counter_header.dispatch_info.queue_id.handle = random_uint64();
    counter_header.dispatch_info.kernel_id = random_uint64();
    counter_header.dispatch_info.dispatch_id = random_uint64();
    counter_header.dispatch_info.private_segment_size = random_uint32() % 32768;
    counter_header.dispatch_info.group_segment_size = random_uint32() % 32768;
    counter_header.dispatch_info.workgroup_size.x = 64;
    counter_header.dispatch_info.workgroup_size.y = 1;
    counter_header.dispatch_info.workgroup_size.z = 1;
    counter_header.dispatch_info.grid_size.x = 512 * 64;
    counter_header.dispatch_info.grid_size.y = 1;
    counter_header.dispatch_info.grid_size.z = 1;

    rec.data.resize(sizeof(counter_header));
    std::memcpy(rec.data.data(), &counter_header, sizeof(counter_header));

    rec.header.category = ROCPROFILER_BUFFER_CATEGORY_COUNTERS;
    rec.header.kind = ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER;
    rec.header.payload = nullptr;

    return rec;
}

// Generate a random counter value record
TestRecord generate_counter_value_record() {
    TestRecord rec;

    rocprofiler_counter_record_t counter_record;
    counter_record.id = random_uint64(); // Counter instance ID
    counter_record.counter_value = random_double();
    counter_record.dispatch_id = random_uint64();
    counter_record.user_data.value = random_uint64();
    counter_record.agent_id.handle = random_uint64();

    rec.data.resize(sizeof(counter_record));
    std::memcpy(rec.data.data(), &counter_record, sizeof(counter_record));

    rec.header.category = ROCPROFILER_BUFFER_CATEGORY_COUNTERS;
    rec.header.kind = ROCPROFILER_COUNTER_RECORD_VALUE;
    rec.header.payload = nullptr;

    return rec;
}

int main() {
    std::vector<TestRecord> records;

    // Generate various record types
    // Tracing records
    records.push_back(generate_hip_api_record());
    records.push_back(generate_hip_api_record());
    records.push_back(generate_hsa_api_record());
    records.push_back(generate_kernel_dispatch_record());
    records.push_back(generate_kernel_dispatch_record());
    records.push_back(generate_memory_copy_record());

    // Counter records
    records.push_back(generate_counter_dispatch_header());
    for (int i = 0; i < 5; ++i) {
        records.push_back(generate_counter_value_record());
    }

    // Build the buffer data and calculate offsets
    std::vector<char> buffer_data;
    std::vector<rocprofiler_record_header_t> headers;

    for (auto& rec : records) {
        // Set the payload pointer to the offset in the buffer
        rec.header.payload = reinterpret_cast<void*>(buffer_data.size());
        headers.push_back(rec.header);

        // Append the record data to the buffer
        buffer_data.insert(buffer_data.end(), rec.data.begin(), rec.data.end());
    }

    // Write the record_header_buffer format to stdout
    // Format: size_t index, size_t headers_size, headers[], ring_buffer

    size_t num_records = records.size();
    size_t headers_size = headers.size();

    // Write header metadata
    fwrite(&num_records, sizeof(size_t), 1, stdout);
    fwrite(&headers_size, sizeof(size_t), 1, stdout);
    fwrite(headers.data(), sizeof(rocprofiler_record_header_t), headers_size, stdout);

    // Write ring_buffer format
    // Format: size_t buffer_size, size_t read_count, size_t write_count, char[] data
    size_t buffer_size = buffer_data.size();
    size_t read_count = 0;
    size_t write_count = num_records;

    fwrite(&buffer_size, sizeof(size_t), 1, stdout);
    fwrite(&read_count, sizeof(size_t), 1, stdout);
    fwrite(&write_count, sizeof(size_t), 1, stdout);
    fwrite(buffer_data.data(), 1, buffer_size, stdout);

    return 0;
}
