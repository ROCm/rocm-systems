// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <nlohmann/json.hpp>

#include <span>
#include <string>

namespace rocprofsys
{
class process_environment_info
{
public:
    /** Snapshots the calling process's environment, redacting $HOME. */
    process_environment_info();

    ~process_environment_info()                                              = default;
    process_environment_info(const process_environment_info&)                = default;
    process_environment_info(process_environment_info&&) noexcept            = default;
    process_environment_info& operator=(const process_environment_info&)     = default;
    process_environment_info& operator=(process_environment_info&&) noexcept = default;

    /**
     * Records which MPI process this snapshot belongs to.
     * These keys are synthesized at MPI_Init, not read from the environment, and are
     * stored as JSON numbers.
     * @param rank Rank in MPI_COMM_WORLD.
     * @param size Size of MPI_COMM_WORLD.
     */
    void add_mpi_identity(int rank, int size);

    [[nodiscard]] std::string to_json() const;

private:
    nlohmann::json m_entries = nlohmann::json::object();
};
}  // namespace rocprofsys
