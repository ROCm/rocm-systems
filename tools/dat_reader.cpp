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
#include <rocprofiler-sdk/counters.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

// Helper function to get category name
const char* get_category_name(uint32_t category) {
    switch(category) {
        case ROCPROFILER_BUFFER_CATEGORY_NONE: return "NONE";
        case ROCPROFILER_BUFFER_CATEGORY_TRACING: return "TRACING";
        case ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING: return "PC_SAMPLING";
        case ROCPROFILER_BUFFER_CATEGORY_COUNTERS: return "COUNTERS";
        default: return "UNKNOWN";
    }
}

// Helper function to get tracing kind name
const char* get_tracing_kind_name(uint32_t kind) {
    switch(kind) {
        case ROCPROFILER_BUFFER_TRACING_NONE: return "NONE";
        case ROCPROFILER_BUFFER_TRACING_HSA_CORE_API: return "HSA_CORE_API";
        case ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API: return "HSA_AMD_EXT_API";
        case ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API: return "HSA_IMAGE_EXT_API";
        case ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API: return "HSA_FINALIZE_EXT_API";
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API: return "HIP_RUNTIME_API";
        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API: return "HIP_COMPILER_API";
        case ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API: return "MARKER_CORE_API";
        case ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API: return "MARKER_CONTROL_API";
        case ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API: return "MARKER_NAME_API";
        case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY: return "MEMORY_COPY";
        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: return "KERNEL_DISPATCH";
        case ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY: return "SCRATCH_MEMORY";
        case ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT: return "CORRELATION_ID_RETIREMENT";
        case ROCPROFILER_BUFFER_TRACING_RCCL_API: return "RCCL_API";
        case ROCPROFILER_BUFFER_TRACING_OMPT: return "OMPT";
        case ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION: return "MEMORY_ALLOCATION";
        case ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION: return "RUNTIME_INITIALIZATION";
        case ROCPROFILER_BUFFER_TRACING_ROCDECODE_API: return "ROCDECODE_API";
        case ROCPROFILER_BUFFER_TRACING_ROCJPEG_API: return "ROCJPEG_API";
        case ROCPROFILER_BUFFER_TRACING_HIP_STREAM: return "HIP_STREAM";
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT: return "HIP_RUNTIME_API_EXT";
        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT: return "HIP_COMPILER_API_EXT";
        case ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT: return "ROCDECODE_API_EXT";
        default: return "UNKNOWN";
    }
}

// Helper function to get counter record kind name
const char* get_counter_kind_name(uint32_t kind) {
    switch(kind) {
        case ROCPROFILER_COUNTER_RECORD_NONE: return "NONE";
        case ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER: return "DISPATCH_HEADER";
        case ROCPROFILER_COUNTER_RECORD_VALUE: return "COUNTER_VALUE";
        default: return "UNKNOWN";
    }
}

// Helper function to print a generic API record
void print_api_record(size_t record_num, const char* kind_name,
                      uint64_t size, uint32_t operation,
                      uint64_t correlation_id, uint64_t start_ts,
                      uint64_t end_ts, uint64_t thread_id) {
    std::cout << "Record #" << record_num << ":\n";
    std::cout << "  Category: TRACING\n";
    std::cout << "  Kind: " << kind_name << "\n";
    std::cout << "  Operation: " << operation << "\n";
    std::cout << "  Correlation ID: " << correlation_id << "\n";
    std::cout << "  Thread ID: " << thread_id << "\n";
    std::cout << "  Start Timestamp: " << start_ts << " ns\n";
    std::cout << "  End Timestamp: " << end_ts << " ns\n";
    std::cout << "  Duration: " << (end_ts - start_ts) << " ns\n";
    std::cout << "\n";
}

// Helper function to print HIP API record
void print_hip_api_record(size_t record_num, const char* kind_name, void* payload) {
    auto* record = static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(payload);
    print_api_record(record_num, kind_name, record->size, record->operation,
                     record->correlation_id.internal, record->start_timestamp,
                     record->end_timestamp, record->thread_id);
}

// Helper function to print HSA API record
void print_hsa_api_record(size_t record_num, const char* kind_name, void* payload) {
    auto* record = static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(payload);
    print_api_record(record_num, kind_name, record->size, record->operation,
                     record->correlation_id.internal, record->start_timestamp,
                     record->end_timestamp, record->thread_id);
}

// Helper function to print kernel dispatch record
void print_kernel_dispatch_record(size_t record_num, void* payload) {
    auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(payload);
    std::cout << "Record #" << record_num << ":\n";
    std::cout << "  Category: TRACING\n";
    std::cout << "  Kind: KERNEL_DISPATCH\n";
    std::cout << "  Operation: " << record->operation << "\n";
    std::cout << "  Correlation ID: " << record->correlation_id.internal << "\n";
    std::cout << "  Thread ID: " << record->thread_id << "\n";
    std::cout << "  Start Timestamp: " << record->start_timestamp << " ns\n";
    std::cout << "  End Timestamp: " << record->end_timestamp << " ns\n";
    std::cout << "  Duration: " << (record->end_timestamp - record->start_timestamp) << " ns\n";
    std::cout << "  Dispatch ID: " << record->dispatch_info.dispatch_id << "\n";
    std::cout << "  Agent ID: " << record->dispatch_info.agent_id.handle << "\n";
    std::cout << "  Queue ID: " << record->dispatch_info.queue_id.handle << "\n";
    std::cout << "  Kernel ID: " << record->dispatch_info.kernel_id << "\n";
    std::cout << "  Grid Size: [" << record->dispatch_info.grid_size.x
              << ", " << record->dispatch_info.grid_size.y
              << ", " << record->dispatch_info.grid_size.z << "]\n";
    std::cout << "  Workgroup Size: [" << record->dispatch_info.workgroup_size.x
              << ", " << record->dispatch_info.workgroup_size.y
              << ", " << record->dispatch_info.workgroup_size.z << "]\n";
    std::cout << "\n";
}

// Helper function to print memory copy record
void print_memory_copy_record(size_t record_num, void* payload) {
    auto* record = static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(payload);
    std::cout << "Record #" << record_num << ":\n";
    std::cout << "  Category: TRACING\n";
    std::cout << "  Kind: MEMORY_COPY\n";
    std::cout << "  Operation: " << record->operation << "\n";
    std::cout << "  Correlation ID: " << record->correlation_id.internal << "\n";
    std::cout << "  Thread ID: " << record->thread_id << "\n";
    std::cout << "  Start Timestamp: " << record->start_timestamp << " ns\n";
    std::cout << "  End Timestamp: " << record->end_timestamp << " ns\n";
    std::cout << "  Duration: " << (record->end_timestamp - record->start_timestamp) << " ns\n";
    std::cout << "  Dst Agent ID: " << record->dst_agent_id.handle << "\n";
    std::cout << "  Src Agent ID: " << record->src_agent_id.handle << "\n";
    std::cout << "  Bytes: " << record->bytes << "\n";
    std::cout << "\n";
}

// Helper function to print counter dispatch header record
void print_counter_dispatch_header(size_t record_num, void* payload) {
    auto* record = static_cast<rocprofiler_dispatch_counting_service_record_t*>(payload);
    std::cout << "Record #" << record_num << ":\n";
    std::cout << "  Category: COUNTERS\n";
    std::cout << "  Kind: DISPATCH_HEADER\n";
    std::cout << "  Num Counter Records: " << record->num_records << "\n";
    std::cout << "  Correlation ID: " << record->correlation_id.internal << "\n";
    std::cout << "  Start Timestamp: " << record->start_timestamp << " ns\n";
    std::cout << "  End Timestamp: " << record->end_timestamp << " ns\n";
    std::cout << "  Duration: " << (record->end_timestamp - record->start_timestamp) << " ns\n";
    std::cout << "  Dispatch ID: " << record->dispatch_info.dispatch_id << "\n";
    std::cout << "  Agent ID: " << record->dispatch_info.agent_id.handle << "\n";
    std::cout << "  Queue ID: " << record->dispatch_info.queue_id.handle << "\n";
    std::cout << "  Kernel ID: " << record->dispatch_info.kernel_id << "\n";
    std::cout << "  Grid Size: [" << record->dispatch_info.grid_size.x
              << ", " << record->dispatch_info.grid_size.y
              << ", " << record->dispatch_info.grid_size.z << "]\n";
    std::cout << "  Workgroup Size: [" << record->dispatch_info.workgroup_size.x
              << ", " << record->dispatch_info.workgroup_size.y
              << ", " << record->dispatch_info.workgroup_size.z << "]\n";
    std::cout << "\n";
}

// Helper function to print counter value record
void print_counter_value_record(size_t record_num, void* payload) {
    auto* record = static_cast<rocprofiler_counter_record_t*>(payload);
    std::cout << "Record #" << record_num << ":\n";
    std::cout << "  Category: COUNTERS\n";
    std::cout << "  Kind: COUNTER_VALUE\n";

    // Try to get counter name
    rocprofiler_counter_id_t counter_id;
    if (rocprofiler_query_record_counter_id(record->id, &counter_id) == ROCPROFILER_STATUS_SUCCESS) {
        rocprofiler_counter_info_v0_t counter_info{};
        if (rocprofiler_query_counter_info(counter_id,
                                          ROCPROFILER_COUNTER_INFO_VERSION_0,
                                          &counter_info) == ROCPROFILER_STATUS_SUCCESS) {
            std::cout << "  Counter Name: " << counter_info.name << "\n";
            std::cout << "  Counter Description: " << counter_info.description << "\n";
        }
    }

    std::cout << "  Counter Instance ID: " << record->id << "\n";
    std::cout << "  Counter Value: " << record->counter_value << "\n";
    std::cout << "  Dispatch ID: " << record->dispatch_id << "\n";
    std::cout << "  Agent ID: " << record->agent_id.handle << "\n";
    std::cout << "  User Data: " << record->user_data.value << "\n";
    std::cout << "\n";
}

// Helper function to print generic record
void print_generic_record(size_t record_num, uint32_t category, uint32_t kind, uint64_t hash) {
    std::cout << "Record #" << record_num << ":\n";
    std::cout << "  Category: " << get_category_name(category) << " (" << category << ")\n";
    std::cout << "  Kind: " << kind << "\n";
    std::cout << "  Hash: 0x" << std::hex << hash << std::dec << "\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    // Read stdin into a temporary file since we need to seek
    char temp_filename[] = "/tmp/dat_reader_XXXXXX";
    int temp_fd = mkstemp(temp_filename);
    if (temp_fd == -1) {
        std::cerr << "Error: Failed to create temporary file\n";
        return 1;
    }

    // Read from stdin and write to temp file
    char buffer[4096];
    ssize_t bytes_read;
    while ((bytes_read = read(STDIN_FILENO, buffer, sizeof(buffer))) > 0) {
        ssize_t bytes_written = write(temp_fd, buffer, bytes_read);
        if (bytes_written != bytes_read) {
            std::cerr << "Error: Failed to write to temporary file\n";
            close(temp_fd);
            unlink(temp_filename);
            return 1;
        }
    }
    close(temp_fd);

    // Open the temp file for reading in binary mode
    std::fstream fs;
    fs.open(temp_filename, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        std::cerr << "Error: Failed to open temporary file for reading\n";
        unlink(temp_filename);
        return 1;
    }

    try {
        // Read record_header_buffer format
        // Format: size_t _idx, size_t _sz, rocprofiler_record_header_t[_sz]
        size_t num_records = 0;
        size_t headers_size = 0;

        fs.read(reinterpret_cast<char*>(&num_records), sizeof(num_records));
        fs.read(reinterpret_cast<char*>(&headers_size), sizeof(headers_size));

        if (!fs.good()) {
            std::cerr << "Error: Failed to read header information\n";
            fs.close();
            unlink(temp_filename);
            return 1;
        }

        std::cout << "Total record headers: " << num_records << "\n";
        std::cout << "Headers vector size: " << headers_size << "\n";
        std::cout << "----------------------------------------\n\n";

        // Read all record headers
        std::vector<rocprofiler_record_header_t> headers(headers_size);
        fs.read(reinterpret_cast<char*>(headers.data()),
                sizeof(rocprofiler_record_header_t) * headers_size);

        if (!fs.good()) {
            std::cerr << "Error: Failed to read record headers\n";
            fs.close();
            unlink(temp_filename);
            return 1;
        }

        // Read ring_buffer format
        // Format: size_t size, size_t read_count, size_t write_count, char[size]
        size_t buffer_size = 0;
        size_t read_count = 0;
        size_t write_count = 0;

        fs.read(reinterpret_cast<char*>(&buffer_size), sizeof(buffer_size));
        fs.read(reinterpret_cast<char*>(&read_count), sizeof(read_count));
        fs.read(reinterpret_cast<char*>(&write_count), sizeof(write_count));

        if (!fs.good()) {
            std::cerr << "Error: Failed to read buffer information\n";
            fs.close();
            unlink(temp_filename);
            return 1;
        }

        // Read buffer data
        std::vector<char> buffer_data(buffer_size);
        fs.read(buffer_data.data(), buffer_size);

        if (!fs.good()) {
            std::cerr << "Error: Failed to read buffer data\n";
            fs.close();
            unlink(temp_filename);
            return 1;
        }

        fs.close();

        // The payload pointers in the headers are relative to the original buffer
        // We need to adjust them to point into our buffer_data
        // The payload pointer value is actually an offset from the start of the original buffer

        // Process each record
        size_t valid_records = 0;
        for (size_t i = 0; i < num_records; ++i) {
            auto& header = headers[i];

            // Skip invalid records
            if (header.hash == 0 || header.payload == nullptr) {
                continue;
            }

            valid_records++;

            // Calculate the offset of the payload in the original buffer
            // The payload pointer is stored as an absolute address from the original save
            // We need to treat it as an offset
            uintptr_t payload_offset = reinterpret_cast<uintptr_t>(header.payload);

            // Adjust payload to point into our buffer_data
            // For safety, we'll check if the offset is within bounds
            if (payload_offset < buffer_size) {
                void* adjusted_payload = &buffer_data[payload_offset];

                // Decode based on category and kind
                if (header.category == ROCPROFILER_BUFFER_CATEGORY_TRACING) {
                    const char* kind_name = get_tracing_kind_name(header.kind);

                    switch(header.kind) {
                        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API:
                        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API:
                        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT:
                        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT:
                            print_hip_api_record(valid_records, kind_name, adjusted_payload);
                            break;

                        case ROCPROFILER_BUFFER_TRACING_HSA_CORE_API:
                        case ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API:
                        case ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API:
                        case ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API:
                            print_hsa_api_record(valid_records, kind_name, adjusted_payload);
                            break;

                        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
                            print_kernel_dispatch_record(valid_records, adjusted_payload);
                            break;

                        case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
                            print_memory_copy_record(valid_records, adjusted_payload);
                            break;

                        default:
                            print_generic_record(valid_records, header.category,
                                               header.kind, header.hash);
                            break;
                    }
                } else if (header.category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS) {
                    switch(header.kind) {
                        case ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER:
                            print_counter_dispatch_header(valid_records, adjusted_payload);
                            break;

                        case ROCPROFILER_COUNTER_RECORD_VALUE:
                            print_counter_value_record(valid_records, adjusted_payload);
                            break;

                        default:
                            print_generic_record(valid_records, header.category,
                                               header.kind, header.hash);
                            break;
                    }
                } else {
                    print_generic_record(valid_records, header.category,
                                       header.kind, header.hash);
                }
            } else {
                std::cerr << "Warning: Record #" << (i+1)
                         << " has payload offset (" << payload_offset
                         << ") beyond buffer size (" << buffer_size << ")\n";
            }
        }

        std::cout << "----------------------------------------\n";
        std::cout << "Total valid records: " << valid_records << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: Exception occurred: " << e.what() << "\n";
        unlink(temp_filename);
        return 1;
    }

    // Clean up temp file
    unlink(temp_filename);

    return 0;
}
