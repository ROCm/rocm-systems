// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Downstream consumer smoke test for the installed profiler-hub package.
// Verifies that:
//   - public headers are installed under <prefix>/include/profiler-hub/
//   - find_package(profiler-hub) resolves the profiler-hub::profiler-hub target
//   - the library links and a public API symbol is reachable at runtime

#include <profiler-hub/storage.hpp>

#include <iostream>

int
main()
{
    try
    {
        profiler_hub::storage_t storage{ ":memory:", "profiler_hub-find-package-smoke" };
        const auto              version = storage.get_storage_version();
        std::cout << "profiler_hub storage opened. schema version: " << version.major
                  << "." << version.minor << "." << version.patch << '\n';
    } catch(const std::exception& e)
    {
        std::cerr << "profiler_hub consumer smoke FAILED: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
