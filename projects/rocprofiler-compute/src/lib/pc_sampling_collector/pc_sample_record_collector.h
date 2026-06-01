// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "pc_sampling_record.h"
#include "ps_file_writer.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rocprofiler_compute_tool
{

class pc_sample_record_collector_t
{
public:
    using ptr = std::shared_ptr<pc_sample_record_collector_t>;
    static ptr create(std::shared_ptr<code_object_translator_t> translator);

    virtual ~pc_sample_record_collector_t() = default;
    virtual void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) = 0;
    virtual void add_record(const pc_sampling_record_t& rec) = 0;
    // Append a whole buffer batch under a single lock acquisition.
    virtual void add_records(const std::vector<pc_sampling_record_t>& recs) = 0;
    virtual void write(ps_file_writer_t& writer)                            = 0;
};

class pc_sample_record_collector_impl_t : public pc_sample_record_collector_t
{
public:
    pc_sample_record_collector_impl_t(std::shared_ptr<code_object_translator_t> translator);
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    void add_record(const pc_sampling_record_t& rec) override;
    void add_records(const std::vector<pc_sampling_record_t>& recs) override;
    void write(ps_file_writer_t& writer) override;

private:
    std::shared_ptr<code_object_translator_t>    m_translator;
    std::mutex                                   m_mutex;
    std::vector<pc_sampling_record_t>            m_records;
    std::map<std::pair<uint64_t, uint64_t>, int> m_inst_index;
    std::vector<std::string>                     m_instructions;
    std::vector<std::string>                     m_comments;
};

}  // namespace rocprofiler_compute_tool
