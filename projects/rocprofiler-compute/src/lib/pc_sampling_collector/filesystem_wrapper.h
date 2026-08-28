// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <filesystem>
#include <memory>
#include <system_error>

namespace rocprofiler_compute_tool
{
class filesystem_wrapper_t
{
public:
    using ptr = std::shared_ptr<filesystem_wrapper_t>;
    static ptr create();

    virtual ~filesystem_wrapper_t() = default;

    virtual std::filesystem::path absolute(const std::filesystem::path& path, std::error_code& error) = 0;
    virtual std::filesystem::file_status status(const std::filesystem::path& path,
                                                std::error_code&             error)            = 0;
    virtual bool create_directories(const std::filesystem::path& path, std::error_code& error) = 0;
    virtual bool copy_file(const std::filesystem::path&  source,
                           const std::filesystem::path&  destination,
                           std::filesystem::copy_options options,
                           std::error_code&              error)                                = 0;
    virtual bool exists(const std::filesystem::file_status& status)                            = 0;
    virtual bool is_regular_file(const std::filesystem::file_status& status)                   = 0;
    virtual bool has_parent_path(const std::filesystem::path& path)                            = 0;
    virtual std::filesystem::path parent_path(const std::filesystem::path& path)               = 0;
    virtual std::filesystem::path weakly_canonical(const std::filesystem::path& path,
                                                   std::error_code&             error)         = 0;
    virtual std::filesystem::path relative_path(const std::filesystem::path& path)             = 0;
};

class filesystem_wrapper_impl_t : public filesystem_wrapper_t
{
public:
    std::filesystem::path absolute(const std::filesystem::path& path, std::error_code& error) override;
    std::filesystem::file_status status(const std::filesystem::path& path, std::error_code& error) override;
    bool create_directories(const std::filesystem::path& path, std::error_code& error) override;
    bool copy_file(const std::filesystem::path&  source,
                   const std::filesystem::path&  destination,
                   std::filesystem::copy_options options,
                   std::error_code&              error) override;
    bool exists(const std::filesystem::file_status& status) override;
    bool is_regular_file(const std::filesystem::file_status& status) override;
    bool has_parent_path(const std::filesystem::path& path) override;
    std::filesystem::path parent_path(const std::filesystem::path& path) override;
    std::filesystem::path weakly_canonical(const std::filesystem::path& path,
                                           std::error_code&             error) override;
    std::filesystem::path relative_path(const std::filesystem::path& path) override;
};
}  // namespace rocprofiler_compute_tool
