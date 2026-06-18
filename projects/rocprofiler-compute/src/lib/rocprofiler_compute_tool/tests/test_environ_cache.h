// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

class TestEnvironCache : public ::testing::Test
{
protected:
    // Owns the storage backing a NULL-terminated char** envp.
    // Non-copyable and non-movable: tests must declare it as a named local
    // so the validity scope of data() is explicit.
    class Envp
    {
    public:
        explicit Envp(const std::vector<std::string>& entries)
        {
            m_owned.reserve(entries.size());
            m_pointers.reserve(entries.size() + 1);

            for (const auto& entry : entries)
            {
                m_owned.emplace_back(entry.begin(), entry.end());
                m_owned.back().push_back('\0');
                m_pointers.push_back(m_owned.back().data());
            }

            m_pointers.push_back(nullptr);
        }

        Envp(const Envp&)            = delete;
        Envp& operator=(const Envp&) = delete;
        Envp(Envp&&)                 = delete;
        Envp& operator=(Envp&&)      = delete;

        char** data() { return m_pointers.data(); }

    private:
        std::vector<std::vector<char>> m_owned;
        std::vector<char*>             m_pointers;
    };
};
