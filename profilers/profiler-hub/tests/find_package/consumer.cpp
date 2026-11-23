// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Downstream consumer smoke test for the installed profiler-hub package.
// Verifies that:
//   - public headers are installed under <prefix>/include/profiler-hub/
//   - find_package(profiler-hub) resolves the profiler-hub::profiler-hub target
//   - the library links and a public API symbol is reachable at runtime

#include <profiler-hub/cpp/storage.hpp>

#include <iostream>

int
main()
{
    try
    {
        profiler_hub::storage_t storage{ ":memory:", "profiler_hub-find-package-smoke" };
        const auto              library = storage.get_storage_version();
        const auto              schema  = storage.get_writer_schema_version();
        std::cout << "profiler_hub storage opened. library version: " << library.major
                  << "." << library.minor << "." << library.patch
                  << ", schema version: " << schema.major << "." << schema.minor << "."
                  << schema.patch << '\n';
    } catch(const std::exception& e)
    {
        std::cerr << "profiler_hub consumer smoke FAILED: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
