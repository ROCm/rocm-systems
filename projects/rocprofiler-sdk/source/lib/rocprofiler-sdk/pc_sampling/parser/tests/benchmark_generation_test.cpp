// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <gtest/gtest.h>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <type_traits>

#include "lib/common/container/record_header_buffer.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/mocks.hpp"

#define GFXIP_MAJOR 9
#define GFXIP_MINOR 4

/**
 * Benchmarks generation and parsing overhead for PC sampling
 *
 * This test measures:
 * 1. Record generation time - time to create raw hardware samples
 * 2. Parsing throughput - time to parse samples into output records
 * 3. Memory footprint - total buffer size and bytes per sample
 *
 * USAGE:
 * ------
 * This benchmark works with BOTH single-record (v0) and multi-record approaches:
 *
 * Single-record mode (old v0 approach):
 *   export ROCPROFILER_PC_SAMPLING_USE_MULTI_RECORD=0
 *   ./pcs_gen_bench_test
 *
 * Multi-record mode (new approach - DEFAULT):
 *   export ROCPROFILER_PC_SAMPLING_USE_MULTI_RECORD=1  # or unset (default)
 *   ./pcs_gen_bench_test
 *
 * MULTI-RECORD FORMAT:
 * --------------------
 * - Host trap: 3 records per sample (base + hw_id + workgroup)
 * - Stochastic: 4 records per sample (base + hw_id + workgroup + snapshot_state)
 *
 * COMPARISON:
 * -----------
 * Run both modes and compare:
 * - Generation time should be SAME (creates raw samples)
 * - Parsing time will be DIFFERENT (multi-record does more work)
 * - Memory footprint will be DIFFERENT (multi-record uses 3-4x more records)
 */
template <typename PcSamplingRecordT>
static bool
BenchmarkGenerationAndParsing(bool bWarmup)
{
    constexpr size_t SAMPLE_PER_DISPATCH = 8192;
    constexpr size_t DISP_PER_QUEUE      = 8;
    constexpr size_t NUM_QUEUES          = 4;

    // Determine expected record multiplier for multi-record mode
    // Host trap: 3 records (base + hw_id + workgroup)
    // Stochastic: 4 records (base + hw_id + workgroup + snapshot_state)
    constexpr bool is_host_trap =
        std::is_same<PcSamplingRecordT, rocprofiler_pc_sampling_record_host_trap_v0_t>::value;
    constexpr size_t expected_multi_record_multiplier = is_host_trap ? 3 : 4;

    // Detect which mode we're running in
    const char* env = std::getenv("ROCPROFILER_PC_SAMPLING_USE_MULTI_RECORD");
    bool        is_multi_record =
        (env == nullptr) ||
        !(std::string(env) == "0" || std::string(env) == "false" || std::string(env) == "FALSE");

    auto buffer = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
    std::array<std::vector<std::shared_ptr<MockDispatch<PcSamplingRecordT>>>, NUM_QUEUES>
        active_dispatches;

    for(size_t q = 0; q < NUM_QUEUES; q++)
    {
        auto queue = std::make_shared<MockQueue<PcSamplingRecordT>>(DISP_PER_QUEUE * 2, buffer);
        for(size_t d = 0; d < DISP_PER_QUEUE; d++)
            active_dispatches[q].push_back(
                std::make_shared<MockDispatch<PcSamplingRecordT>>(queue));
    }

    constexpr size_t TOTAL_NUM_SAMPLES = NUM_QUEUES * DISP_PER_QUEUE * SAMPLE_PER_DISPATCH;
    buffer->genUpcomingSamples(TOTAL_NUM_SAMPLES);

    // ===================================================================
    // GENERATION PHASE - Measure time to create raw hardware samples
    // ===================================================================
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    for(auto& queue : active_dispatches)
        for(auto& dispatch : queue)
            for(size_t i = 0; i < SAMPLE_PER_DISPATCH; i++)
                MockWave(dispatch).genPCSample();

    auto t_gen_end = std::chrono::high_resolution_clock::now();

    // Record buffer state after generation
    size_t buffer_packets_generated = buffer->packets.size();
    size_t buffer_bytes_generated   = buffer_packets_generated * sizeof(packet_union_t);

    // ===================================================================
    // PARSING PHASE - Measure time to parse and emit output records
    // ===================================================================

    size_t records_parsed      = 0;
    size_t buffer_bytes_parsed = 0;

    auto t_parse_start = std::chrono::high_resolution_clock::now();

    if(is_multi_record)
    {
        // Multi-record mode: use parse_buffer_multi_record() with real buffer
        rocprofiler::buffer::instance real_buff;
        std::atomic<uint64_t>         instance_id_gen{1};

        // Initialize both internal buffers (double-buffering)
        // Size: 262,144 samples × 4 records/sample × 80 bytes/record ≈ 80 MB
        // Use 100 MB per buffer to be safe
        constexpr size_t BUFFER_SIZE = 100 * 1024 * 1024;  // 100 MB

        real_buff.buffers[0].allocate(BUFFER_SIZE);
        real_buff.buffers[1].allocate(BUFFER_SIZE);
        real_buff.watermark = BUFFER_SIZE;

        CHECK_PARSER(parse_buffer_multi_record((generic_sample_t*) buffer->packets.data(),
                                               buffer->packets.size(),
                                               GFXIP_MAJOR,
                                               GFXIP_MINOR,
                                               &real_buff,
                                               instance_id_gen));

        // Count records and calculate total bytes from both buffers
        records_parsed      = 0;
        buffer_bytes_parsed = 0;

        for(size_t buf_idx = 0; buf_idx < 2; ++buf_idx)
        {
            real_buff.buffers[buf_idx].process_record_headers(
                std::false_type{},  // Don't clear records
                [&](rocprofiler::common::container::record_header_buffer::record_ptr_vec_t&&
                        records) {
                    records_parsed += records.size();

                    for(auto* hdr_ptr : records)
                    {
                        // All PC sampling records have uint64_t size as first field.
                        // Cast payload to uint64_t* to read the size field.
                        uint64_t record_size = *static_cast<uint64_t*>(hdr_ptr->payload);
                        buffer_bytes_parsed += record_size;
                    }
                });
        }
    }
    else
    {
        // Single-record v0 mode: use parse_buffer<T>() + buffer->emplace()
        // This mimics the production path: parse → generate_upcoming_pc_record → emplace

        // Step 1: Parse to intermediate v0 records (like production _parse does)
        constexpr size_t                      max_records = TOTAL_NUM_SAMPLES;
        std::pair<PcSamplingRecordT*, size_t> userdata;
        userdata.first  = new PcSamplingRecordT[max_records];
        userdata.second = max_records;

        PcSamplingRecordT* output_start = userdata.first;  // Save original pointer

        user_callback_t<PcSamplingRecordT> user_cb =
            [](PcSamplingRecordT** sample, uint64_t size, void* userdata_) {
                auto* pair = reinterpret_cast<std::pair<PcSamplingRecordT*, size_t>*>(userdata_);
                if(pair->second < size) size = pair->second;  // Safety check
                *sample = pair->first;
                pair->first += size;   // Advance pointer for next allocation
                pair->second -= size;  // Reduce remaining space
                return size;
            };

        // user_callback_t<PcSamplingRecordT> user_cb =
        // [](PcSamplingRecordT** sample, uint64_t size, void* userdata_) {
        //     auto* pair = reinterpret_cast<std::pair<PcSamplingRecordT*, size_t>*>(userdata_);
        //     assert(TOTAL_NUM_SAMPLES == pair->second);
        //     *sample = pair->first;
        //     return size;
        // };

        CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                                  buffer->packets.size(),
                                  GFXIP_MAJOR,
                                  GFXIP_MINOR,
                                  user_cb,
                                  &userdata));

        size_t num_v0_records = max_records - userdata.second;  // Actual number filled by parser

        // Step 2: Insert v0 records into buffer (like generate_upcoming_pc_record does)
        // This mimics emplace_records_in_buffer() called by production code
        rocprofiler::buffer::instance real_buff;
        constexpr size_t              BUFFER_SIZE = 100 * 1024 * 1024;  // 100 MB
        real_buff.buffers[0].allocate(BUFFER_SIZE);
        real_buff.buffers[1].allocate(BUFFER_SIZE);
        real_buff.watermark = BUFFER_SIZE;

        // Determine record kind based on template parameter
        constexpr rocprofiler_pc_sampling_record_kind_t record_kind =
            std::is_same<PcSamplingRecordT, rocprofiler_pc_sampling_record_host_trap_v0_t>::value
                ? ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE
                : ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE;

        // Emplace each v0 record into buffer (1x per sample)
        for(size_t i = 0; i < num_v0_records; i++)
        {
            // Skip invalid stochastic samples (size == 0), matching production behavior
            if constexpr(std::is_same<PcSamplingRecordT,
                                      rocprofiler_pc_sampling_record_stochastic_v0_t>::value)
            {
                if(output_start[i].size == 0) continue;
            }

            real_buff.emplace(
                ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING, record_kind, output_start[i]);
        }

        // Step 3: Count records and bytes from buffer (for fair comparison with multi-record)
        records_parsed      = 0;
        buffer_bytes_parsed = 0;

        for(size_t buf_idx = 0; buf_idx < 2; ++buf_idx)
        {
            real_buff.buffers[buf_idx].process_record_headers(
                std::false_type{},  // Don't clear records
                [&](rocprofiler::common::container::record_header_buffer::record_ptr_vec_t&&
                        records) {
                    records_parsed += records.size();

                    for(auto* hdr_ptr : records)
                    {
                        // All PC sampling records have uint64_t size as first field
                        uint64_t record_size = *static_cast<uint64_t*>(hdr_ptr->payload);
                        buffer_bytes_parsed += record_size;
                    }
                });
        }

        delete[] output_start;
    }

    auto t_parse_end = std::chrono::high_resolution_clock::now();

    double record_amplification = static_cast<double>(records_parsed) / TOTAL_NUM_SAMPLES;

    // ===================================================================
    // METRICS CALCULATION
    // ===================================================================
    double gen_time_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    double parse_time_ms =
        std::chrono::duration<double, std::milli>(t_parse_end - t_parse_start).count();
    double total_time_ms = gen_time_ms + parse_time_ms;

    double gen_msamples_per_sec   = (TOTAL_NUM_SAMPLES / gen_time_ms) / 1000.0;
    double parse_msamples_per_sec = (TOTAL_NUM_SAMPLES / parse_time_ms) / 1000.0;
    double total_msamples_per_sec = (TOTAL_NUM_SAMPLES / total_time_ms) / 1000.0;

    double gen_mb_per_sec   = (buffer_bytes_generated / gen_time_ms) / 1024.0;
    double parse_mb_per_sec = (buffer_bytes_parsed / parse_time_ms) / 1024.0;

    double bytes_per_sample_generated =
        static_cast<double>(buffer_bytes_generated) / TOTAL_NUM_SAMPLES;
    double bytes_per_sample_parsed = static_cast<double>(buffer_bytes_parsed) / TOTAL_NUM_SAMPLES;

    // ===================================================================
    // REPORTING
    // ===================================================================
    if(!bWarmup)
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Mode: " << (is_multi_record ? "MULTI-RECORD (new)" : "SINGLE-RECORD v0 (old)")
                  << std::endl;
        if(is_multi_record)
        {
            std::cout << "Expected records/sample: " << expected_multi_record_multiplier << "x"
                      << std::endl;
        }
        std::cout << "========================================" << std::endl;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n=== Generation Phase ===" << std::endl;
        std::cout << "  Time:       " << std::setw(8) << gen_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setw(8) << gen_msamples_per_sec
                  << " Msamples/s (logical)" << std::endl;
        std::cout << "  Bandwidth:  " << std::setw(8) << gen_mb_per_sec << " MB/s" << std::endl;

        std::cout << "\n=== Parsing Phase ===" << std::endl;
        std::cout << "  Time:       " << std::setw(8) << parse_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setw(8) << parse_msamples_per_sec
                  << " Msamples/s (logical)" << std::endl;
        std::cout << "  Bandwidth:  " << std::setw(8) << parse_mb_per_sec << " MB/s" << std::endl;

        std::cout << "\n=== Total (Generation + Parsing) ===" << std::endl;
        std::cout << "  Time:       " << std::setw(8) << total_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setw(8) << total_msamples_per_sec
                  << " Msamples/s (logical)" << std::endl;

        std::cout << "\n=== Memory Footprint ===" << std::endl;
        std::cout << "  Logical samples:    " << TOTAL_NUM_SAMPLES << std::endl;
        std::cout << "  Generated packets:  " << buffer_packets_generated << std::endl;
        std::cout << "  Parsed records:     " << records_parsed << std::endl;
        std::cout << "  Record amplification: " << std::setprecision(2) << record_amplification
                  << "x";
        if(is_multi_record)
        {
            std::cout << " (expected " << expected_multi_record_multiplier << "x)";
        }
        std::cout << std::endl;
        std::cout << "  Generated buffer:   " << buffer_bytes_generated << " bytes ("
                  << (buffer_bytes_generated / 1024.0 / 1024.0) << " MB)" << std::endl;
        std::cout << "  Parsed output:      " << buffer_bytes_parsed << " bytes ("
                  << (buffer_bytes_parsed / 1024.0 / 1024.0) << " MB)" << std::endl;
        std::cout << "  Bytes/sample (gen): " << bytes_per_sample_generated << std::endl;
        std::cout << "  Bytes/sample (out): " << bytes_per_sample_parsed << std::endl;
        std::cout << std::endl;
    }

    return true;
}

TEST(pcs_parser, benchmark_generation_test)
{
    // Detect which mode we're running in
    const char* env = std::getenv("ROCPROFILER_PC_SAMPLING_USE_MULTI_RECORD");
    bool        is_multi_record =
        (env == nullptr) ||
        !(std::string(env) == "0" || std::string(env) == "false" || std::string(env) == "FALSE");

    std::cout << "\n========================================" << std::endl;
    std::cout << "PC Sampling Generation + Parsing Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Current mode: " << (is_multi_record ? "MULTI-RECORD" : "SINGLE-RECORD v0")
              << std::endl;
    std::cout << "To switch modes, set ROCPROFILER_PC_SAMPLING_USE_MULTI_RECORD=0 or 1"
              << std::endl;

    // Tests for host trap v0 records (3x records in multi-record mode)
    std::cout << "\n>>> Testing rocprofiler_pc_sampling_record_host_trap_v0_t <<<" << std::endl;
    std::cout << "    Multi-record mode: 3 records per sample (base + hw_id + workgroup)\n"
              << std::endl;
    EXPECT_EQ(BenchmarkGenerationAndParsing<rocprofiler_pc_sampling_record_host_trap_v0_t>(true),
              true);  // Warmup
    EXPECT_EQ(BenchmarkGenerationAndParsing<rocprofiler_pc_sampling_record_host_trap_v0_t>(false),
              true);  // Run 1
    EXPECT_EQ(BenchmarkGenerationAndParsing<rocprofiler_pc_sampling_record_host_trap_v0_t>(false),
              true);  // Run 2

    // Tests for stochastic v0 records (4x records in multi-record mode)
    std::cout << "\n>>> Testing rocprofiler_pc_sampling_record_stochastic_v0_t <<<" << std::endl;
    std::cout << "    Multi-record mode: 4 records per sample (base + hw_id + workgroup + "
                 "snapshot_state)\n"
              << std::endl;
    EXPECT_EQ(BenchmarkGenerationAndParsing<rocprofiler_pc_sampling_record_stochastic_v0_t>(true),
              true);  // Warmup
    EXPECT_EQ(BenchmarkGenerationAndParsing<rocprofiler_pc_sampling_record_stochastic_v0_t>(false),
              true);  // Run 1
    EXPECT_EQ(BenchmarkGenerationAndParsing<rocprofiler_pc_sampling_record_stochastic_v0_t>(false),
              true);  // Run 2

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Complete" << std::endl;
    std::cout << "========================================\n" << std::endl;
}
