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

#include "mocked_types.hpp"

#include "core/trace_cache/buffer_storage.hpp"
#include "core/trace_cache/storage_parser.hpp"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{

struct sample_1_hash
{
    size_t operator()(const test_sample_1& s) const
    {
        size_t h = std::hash<int>{}(s.value);

        for(size_t i = 0; i < s.text.size(); ++i)
        {
            h ^= std::hash<char>{}(s.text[i]) + 0x9e3779b9 + (h << 6) + (h >> 2) + i;
        }

        return h;
    }
};

struct sample_2_hash
{
    size_t operator()(const test_sample_2& s) const
    {
        size_t h1 = std::hash<double>{}(s.data);
        size_t h2 = std::hash<uint32_t>{}(s.sample_id);
        return h1 ^ (h2 << 1);
    }
};

struct sample_3_hash
{
    size_t operator()(const test_sample_3& s) const
    {
        size_t h = 0;
        for(auto byte : s.payload)
        {
            h ^= std::hash<uint8_t>{}(byte) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct sample_4_hash
{
    size_t operator()(const test_sample_4& s) const
    {
        size_t h = 0;
        for(auto val : s.data)
        {
            h ^= std::hash<uint32_t>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

}  // namespace

class integration_sample_processor_t
{
public:
    integration_sample_processor_t() = default;

    void set_expected_samples_1(const std::vector<test_sample_1>& samples)
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        m_expected_samples_1.clear();
        for(const auto& s : samples)
        {
            m_expected_samples_1[s]++;
        }
    }

    void set_expected_samples_2(const std::vector<test_sample_2>& samples)
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        m_expected_samples_2.clear();
        for(const auto& s : samples)
        {
            m_expected_samples_2[s]++;
        }
    }

    void set_expected_samples_3(const std::vector<test_sample_3>& samples)
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        m_expected_samples_3.clear();
        for(const auto& s : samples)
        {
            m_expected_samples_3[s]++;
        }
    }

    void set_expected_samples_4(const std::vector<test_sample_4>& samples)
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        m_expected_samples_4.clear();
        for(const auto& s : samples)
        {
            m_expected_samples_4[s]++;
        }
    }

    void execute_sample_processing(test_type_identifier_t type_identifier,
                                   const rocprofsys::trace_cache::cacheable_t& value)
    {
        switch(type_identifier)
        {
            case test_type_identifier_t::sample_type_1:
            {
                const auto& sample = static_cast<const test_sample_1&>(value);
                std::lock_guard<std::mutex> lock(m_data_mutex);
                m_sample_1_count++;
                check_sample_1(sample);
                break;
            }
            case test_type_identifier_t::sample_type_2:
            {
                const auto& sample = static_cast<const test_sample_2&>(value);
                std::lock_guard<std::mutex> lock(m_data_mutex);
                m_sample_2_count++;
                check_sample_2(sample);
                break;
            }
            case test_type_identifier_t::sample_type_3:
            {
                const auto& sample = static_cast<const test_sample_3&>(value);
                std::lock_guard<std::mutex> lock(m_data_mutex);
                m_sample_3_count++;
                check_sample_3(sample);
                break;
            }
            case test_type_identifier_t::sample_type_4:
            {
                const auto& sample = static_cast<const test_sample_4&>(value);
                std::lock_guard<std::mutex> lock(m_data_mutex);
                m_sample_4_count++;
                check_sample_4(sample);
                break;
            }
            default: break;
        }
    }

    int  get_sample_1_count() const { return m_sample_1_count.load(); }
    int  get_sample_2_count() const { return m_sample_2_count.load(); }
    int  get_sample_3_count() const { return m_sample_3_count.load(); }
    int  get_sample_4_count() const { return m_sample_4_count.load(); }
    bool all_expected_samples_found() const
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        return m_expected_samples_1.empty() && m_expected_samples_2.empty() &&
               m_expected_samples_3.empty() && m_expected_samples_4.empty();
    }

private:
    void check_sample_1(const test_sample_1& sample)
    {
        auto it = m_expected_samples_1.find(sample);
        EXPECT_NE(it, m_expected_samples_1.end());
        if(it != m_expected_samples_1.end())
        {
            it->second--;
            if(it->second == 0)
            {
                m_expected_samples_1.erase(it);
            }
        }
    }

    void check_sample_2(const test_sample_2& sample)
    {
        auto it = m_expected_samples_2.find(sample);
        EXPECT_NE(it, m_expected_samples_2.end());
        if(it != m_expected_samples_2.end())
        {
            it->second--;
            if(it->second == 0)
            {
                m_expected_samples_2.erase(it);
            }
        }
    }

    void check_sample_3(const test_sample_3& sample)
    {
        auto it = m_expected_samples_3.find(sample);
        EXPECT_NE(it, m_expected_samples_3.end());
        if(it != m_expected_samples_3.end())
        {
            it->second--;
            if(it->second == 0)
            {
                m_expected_samples_3.erase(it);
            }
        }
    }

    void check_sample_4(const test_sample_4& sample)
    {
        auto it = m_expected_samples_4.find(sample);
        EXPECT_NE(it, m_expected_samples_4.end());
        if(it != m_expected_samples_4.end())
        {
            it->second--;
            if(it->second == 0)
            {
                m_expected_samples_4.erase(it);
            }
        }
    }

    std::atomic<int>                                      m_sample_1_count{ 0 };
    std::atomic<int>                                      m_sample_2_count{ 0 };
    std::atomic<int>                                      m_sample_3_count{ 0 };
    std::atomic<int>                                      m_sample_4_count{ 0 };
    std::unordered_map<test_sample_1, int, sample_1_hash> m_expected_samples_1;
    std::unordered_map<test_sample_2, int, sample_2_hash> m_expected_samples_2;
    std::unordered_map<test_sample_3, int, sample_3_hash> m_expected_samples_3;
    std::unordered_map<test_sample_4, int, sample_4_hash> m_expected_samples_4;
    mutable std::mutex                                    m_data_mutex;
};

class trace_cache_module_integration_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_file_path =
            "integration_test_cache_" + std::to_string(test_counter++) + ".bin";
        std::remove(test_file_path.c_str());
    }

    void TearDown() override { std::remove(test_file_path.c_str()); }

    std::string             test_file_path;
    static std::atomic<int> test_counter;
};

std::atomic<int> trace_cache_module_integration_test::test_counter{ 0 };

TEST_F(trace_cache_module_integration_test, buffer_fragmentation_handling)
{
    std::vector<std::string> large_texts;
    large_texts.reserve(100);
    std::vector<test_sample_1> large_samples;
    large_samples.reserve(100);
    std::vector<test_sample_3> small_samples;
    small_samples.reserve(100);

    for(int i = 0; i < 100; ++i)
    {
        large_texts.push_back(std::string(1000, 'A' + (i % 26)));
        large_samples.push_back({ i, large_texts[i] });

        std::vector<uint8_t> small_payload(10, static_cast<uint8_t>(i));
        small_samples.emplace_back(small_payload);
    }

    std::vector<test_sample_1> expected_1;
    std::vector<test_sample_3> expected_3;
    expected_1.reserve(100);
    expected_3.reserve(50);

    {
        rocprofsys::trace_cache::buffer_storage<
            rocprofsys::trace_cache::flush_worker_factory_t, test_type_identifier_t>
            storage(test_file_path);
        storage.start();

        for(size_t i = 0; i < large_samples.size(); ++i)
        {
            storage.store(large_samples[i]);
            expected_1.push_back(large_samples[i]);
            if(i % 2 == 0 && i < small_samples.size())
            {
                storage.store(small_samples[i]);
                expected_3.push_back(small_samples[i]);
            }
        }

        storage.shutdown();
    }

    auto processor = std::make_shared<integration_sample_processor_t>();
    processor->set_expected_samples_1(expected_1);
    processor->set_expected_samples_3(expected_3);

    rocprofsys::trace_cache::storage_parser<test_type_identifier_t, test_sample_1,
                                            test_sample_2, test_sample_3, test_sample_4>
        parser(test_file_path);

    parser.load(processor);

    EXPECT_EQ(processor->get_sample_1_count(), 100);
    EXPECT_EQ(processor->get_sample_3_count(), 50);
    EXPECT_EQ(processor->get_sample_1_count() + processor->get_sample_3_count(), 150);
}

TEST_F(trace_cache_module_integration_test, content_validation_edge_cases)
{
    std::vector<std::string> strings;
    strings.reserve(4);
    strings.emplace_back("max_value");
    strings.emplace_back("min_value");
    strings.emplace_back("");
    strings.emplace_back("Special\n\t\r\0chars");

    test_sample_1 max_int(std::numeric_limits<int>::max(), strings[0]);
    test_sample_1 min_int(std::numeric_limits<int>::min(), strings[1]);
    test_sample_1 zero_int(0, strings[2]);
    test_sample_1 special_chars(123, strings[3]);

    test_sample_2 max_double(std::numeric_limits<double>::max(),
                             std::numeric_limits<uint32_t>::max());
    test_sample_2 min_double(std::numeric_limits<double>::lowest(), 0);
    test_sample_2 infinity(std::numeric_limits<double>::infinity(), 42);
    test_sample_2 neg_infinity(-std::numeric_limits<double>::infinity(), 43);

    std::vector<uint8_t> max_vector(10000, 0xFF);
    test_sample_3        large_payload(max_vector);
    test_sample_3        empty_payload;
    std::vector<uint8_t> single_zero = { 0x00 };
    test_sample_3        zero_payload(single_zero);

    std::vector<test_sample_1> expected_1;
    std::vector<test_sample_2> expected_2;
    std::vector<test_sample_3> expected_3;
    expected_1.reserve(4);
    expected_2.reserve(4);
    expected_3.reserve(3);

    {
        rocprofsys::trace_cache::buffer_storage<
            rocprofsys::trace_cache::flush_worker_factory_t, test_type_identifier_t>
            storage(test_file_path);
        storage.start();

        storage.store(max_int);
        expected_1.push_back(max_int);
        storage.store(min_int);
        expected_1.push_back(min_int);
        storage.store(zero_int);
        expected_1.push_back(zero_int);
        storage.store(special_chars);
        expected_1.push_back(special_chars);

        storage.store(max_double);
        expected_2.push_back(max_double);
        storage.store(min_double);
        expected_2.push_back(min_double);
        storage.store(infinity);
        expected_2.push_back(infinity);
        storage.store(neg_infinity);
        expected_2.push_back(neg_infinity);

        storage.store(large_payload);
        expected_3.push_back(large_payload);
        storage.store(empty_payload);
        expected_3.push_back(empty_payload);
        storage.store(zero_payload);
        expected_3.push_back(zero_payload);

        storage.shutdown();
    }

    auto processor = std::make_shared<integration_sample_processor_t>();
    processor->set_expected_samples_1(expected_1);
    processor->set_expected_samples_2(expected_2);
    processor->set_expected_samples_3(expected_3);

    rocprofsys::trace_cache::storage_parser<test_type_identifier_t, test_sample_1,
                                            test_sample_2, test_sample_3, test_sample_4>
        parser(test_file_path);
    parser.load(processor);

    EXPECT_EQ(processor->get_sample_1_count(), 4);
    EXPECT_EQ(processor->get_sample_2_count(), 4);
    EXPECT_EQ(processor->get_sample_3_count(), 3);
}

TEST_F(trace_cache_module_integration_test, stress_test_multiple_fragmentations)
{
    const int iterations            = 10;
    const int samples_per_iteration = 10000;

    std::mt19937                          rng(42);
    std::uniform_int_distribution<int>    value_dist(1, 1000);
    std::uniform_int_distribution<size_t> size_dist(1, 500);

    std::vector<std::string> texts;
    texts.reserve(iterations * samples_per_iteration);
    std::vector<test_sample_1> expected_1;
    expected_1.reserve(iterations * samples_per_iteration);

    {
        rocprofsys::trace_cache::buffer_storage<
            rocprofsys::trace_cache::flush_worker_factory_t, test_type_identifier_t>
            storage(test_file_path);
        storage.start();

        for(int iter = 0; iter < iterations; ++iter)
        {
            for(int i = 0; i < samples_per_iteration; ++i)
            {
                int    value     = value_dist(rng);
                size_t text_size = size_dist(rng);
                texts.push_back(std::string(text_size, 'X'));

                test_sample_1 sample(value, texts.back());
                expected_1.push_back(sample);
                storage.store(sample);
            }
        }

        storage.shutdown();
    }

    auto processor = std::make_shared<integration_sample_processor_t>();
    processor->set_expected_samples_1(expected_1);

    rocprofsys::trace_cache::storage_parser<test_type_identifier_t, test_sample_1,
                                            test_sample_2, test_sample_3, test_sample_4>
        parser(test_file_path);
    parser.load(processor);

    EXPECT_EQ(processor->get_sample_1_count(), iterations * samples_per_iteration);
}

TEST_F(trace_cache_module_integration_test, performance_write_test)
{
    const int    sample_count = 50000;
    const size_t payload_size = 1024 * 2;

    std::vector<std::string> payloads;
    payloads.reserve(sample_count);
    std::vector<test_sample_1> samples;
    samples.reserve(sample_count);
    for(int i = 0; i < sample_count; ++i)
    {
        payloads.push_back(std::string(payload_size, static_cast<char>(i % 255)));
        samples.push_back({ i, payloads[i] });
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    {
        rocprofsys::trace_cache::buffer_storage<
            rocprofsys::trace_cache::flush_worker_factory_t, test_type_identifier_t>
            storage(test_file_path);
        storage.start();

        for(const auto& sample : samples)
        {
            storage.store(sample);
        }

        storage.shutdown();
    }

    using unit = std::chrono::microseconds;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_in_microseconds =
        std::chrono::duration_cast<unit>(end_time - start_time);
    auto period = static_cast<double>(unit::period().den);

    double avg_write_time =
        static_cast<double>(duration_in_microseconds.count()) / sample_count;
    double throughput =
        (sample_count * payload_size) / (duration_in_microseconds.count() / period);

    EXPECT_LT(avg_write_time, 50.0);
    EXPECT_GT(throughput, 10 * 1024.0);

    auto processor = std::make_shared<integration_sample_processor_t>();
    processor->set_expected_samples_1(samples);

    rocprofsys::trace_cache::storage_parser<test_type_identifier_t, test_sample_1,
                                            test_sample_2, test_sample_3, test_sample_4>
        parser(test_file_path);
    parser.load(processor);

    EXPECT_EQ(processor->get_sample_1_count(), sample_count);
}

TEST_F(trace_cache_module_integration_test, concurrent_write_read_validation)
{
    const int                thread_count       = 4;
    const int                samples_per_thread = 250;
    const int                total_samples      = thread_count * samples_per_thread;
    std::vector<std::thread> writers;
    std::vector<int>         thread_counters(thread_count, 0);

    std::vector<std::vector<std::string>> thread_strings(thread_count);
    for(int t = 0; t < thread_count; ++t)
    {
        thread_strings[t].reserve(samples_per_thread);
        for(int i = 0; i < samples_per_thread; ++i)
        {
            thread_strings[t].push_back("thread_" + std::to_string(t) + "_sample_" +
                                        std::to_string(i));
        }
    }

    std::vector<test_sample_1> expected_1;
    expected_1.reserve(total_samples);

    for(int t = 0; t < thread_count; ++t)
    {
        for(int i = 0; i < samples_per_thread; ++i)
        {
            expected_1.emplace_back(t, thread_strings[t][i]);
        }
    }

    {
        rocprofsys::trace_cache::buffer_storage<
            rocprofsys::trace_cache::flush_worker_factory_t, test_type_identifier_t>
            storage(test_file_path);
        storage.start();

        for(int t = 0; t < thread_count; ++t)
        {
            writers.emplace_back([&, thread_id = t]() {
                for(int i = 0; i < samples_per_thread; ++i)
                {
                    test_sample_1 sample(thread_id, thread_strings[thread_id][i]);

                    storage.store(sample);
                    thread_counters[thread_id]++;

                    if(i % 10 == 0)
                    {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                }
            });
        }

        for(auto& writer : writers)
        {
            writer.join();
        }

        storage.shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    int total_written = 0;
    for(int counter : thread_counters)
    {
        EXPECT_EQ(counter, samples_per_thread);
        total_written += counter;
    }
    EXPECT_EQ(total_written, total_samples);

    auto processor = std::make_shared<integration_sample_processor_t>();
    processor->set_expected_samples_1(expected_1);

    rocprofsys::trace_cache::storage_parser<test_type_identifier_t, test_sample_1,
                                            test_sample_2, test_sample_3, test_sample_4>
        parser(test_file_path);
    parser.load(processor);

    EXPECT_EQ(processor->get_sample_1_count(), total_samples);
    EXPECT_TRUE(processor->all_expected_samples_found());
}

TEST_F(trace_cache_module_integration_test, uint32_vector_element_size_handling)
{
    std::vector<test_sample_4> expected_4;
    expected_4.reserve(100);

    {
        rocprofsys::trace_cache::buffer_storage<
            rocprofsys::trace_cache::flush_worker_factory_t, test_type_identifier_t>
            storage(test_file_path);
        storage.start();

        for(int i = 0; i < 100; ++i)
        {
            std::vector<uint32_t> data;
            data.reserve(10);
            for(int j = 0; j < 10; ++j)
            {
                data.push_back(static_cast<uint32_t>(i * 1000 + j));
            }
            test_sample_4 sample(data);
            expected_4.push_back(sample);
            storage.store(sample);
        }

        storage.shutdown();
    }

    auto processor = std::make_shared<integration_sample_processor_t>();
    processor->set_expected_samples_4(expected_4);

    rocprofsys::trace_cache::storage_parser<test_type_identifier_t, test_sample_1,
                                            test_sample_2, test_sample_3, test_sample_4>
        parser(test_file_path);
    parser.load(processor);

    EXPECT_EQ(processor->get_sample_4_count(), 100);
    EXPECT_TRUE(processor->all_expected_samples_found());
}

TEST_F(trace_cache_module_integration_test, mixed_vector_element_sizes)
{
    std::vector<test_sample_3> expected_3;
    std::vector<test_sample_4> expected_4;
    expected_3.reserve(50);
    expected_4.reserve(50);

    {
        rocprofsys::trace_cache::buffer_storage<
            rocprofsys::trace_cache::flush_worker_factory_t, test_type_identifier_t>
            storage(test_file_path);
        storage.start();

        for(int i = 0; i < 100; ++i)
        {
            if(i % 2 == 0)
            {
                std::vector<uint8_t> payload(20, static_cast<uint8_t>(i));
                test_sample_3        sample(payload);
                expected_3.push_back(sample);
                storage.store(sample);
            }
            else
            {
                std::vector<uint32_t> data;
                data.reserve(5);
                for(int j = 0; j < 5; ++j)
                {
                    data.push_back(static_cast<uint32_t>(i * 100 + j));
                }
                test_sample_4 sample(data);
                expected_4.push_back(sample);
                storage.store(sample);
            }
        }

        storage.shutdown();
    }

    auto processor = std::make_shared<integration_sample_processor_t>();
    processor->set_expected_samples_3(expected_3);
    processor->set_expected_samples_4(expected_4);

    rocprofsys::trace_cache::storage_parser<test_type_identifier_t, test_sample_1,
                                            test_sample_2, test_sample_3, test_sample_4>
        parser(test_file_path);
    parser.load(processor);

    EXPECT_EQ(processor->get_sample_3_count(), 50);
    EXPECT_EQ(processor->get_sample_4_count(), 50);
    EXPECT_TRUE(processor->all_expected_samples_found());
}
