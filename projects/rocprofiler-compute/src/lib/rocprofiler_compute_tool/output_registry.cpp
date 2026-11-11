// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "output_registry.h"

#include "sdk_callbacks.h"

#include <exception>
#include <iostream>
#include <utility>

namespace rocprofiler_compute_tool
{

void OutputRegistry::register_writer(std::shared_ptr<OutputWriter> writer)
{
    if (writer)
        m_writers.push_back(std::move(writer));
}

bool OutputRegistry::replace_writer(std::string_view name, std::shared_ptr<OutputWriter> writer)
{
    if (!writer)
        return false;

    for (auto& registered : m_writers)
    {
        if (registered->name() == name)
        {
            registered = std::move(writer);
            return true;
        }
    }
    return false;
}

void OutputRegistry::generate_all(tool_data_t& tool_data)
{
    for (const auto& writer : m_writers)
    {
        try
        {
            writer->write(tool_data);
        }
        catch (const std::exception& e)
        {
            std::cerr << "[rocprofiler-compute] [" << __FUNCTION__ << "] ERROR: writer "
                      << writer->name() << " failed: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[rocprofiler-compute] [" << __FUNCTION__ << "] ERROR: writer "
                      << writer->name() << " failed" << std::endl;
        }
    }
}

void OutputRegistry::reset()
{
    m_writers.clear();
}

}  // namespace rocprofiler_compute_tool
