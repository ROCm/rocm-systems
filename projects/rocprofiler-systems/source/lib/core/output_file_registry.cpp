// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "output_file_registry.hpp"

#include <utility>

namespace rocprofsys
{

output_file_registry&
output_file_registry::get_instance()
{
    static output_file_registry instance;
    return instance;
}

void
output_file_registry::register_file(std::string label, std::string path,
                                    std::string viewer)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back({ std::move(label), std::move(path), std::move(viewer) });
}

void
output_file_registry::register_file(config::output_file entry)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back(std::move(entry));
}

std::vector<config::output_file>
output_file_registry::get_files() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_files;
}

void
output_file_registry::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.clear();
}

}  // namespace rocprofsys
