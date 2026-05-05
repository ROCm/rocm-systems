// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Downstream consumer smoke test for the installed rocpdsna package.
// Verifies that:
//   - public headers are installed under <prefix>/include/rocpdsna/
//   - find_package(rocpdsna) resolves the rocpdsna::rocpdsna target
//   - the library links and a public API symbol is reachable at runtime

#include <rocpdsna/storage.hpp>

#include <iostream>

int
main()
{
    try
    {
        rocpdsna::storage_t storage{ ":memory:", "rocpdsna-find-package-smoke" };
        const auto          version = storage.get_storage_version();
        std::cout << "rocpdsna storage opened. schema version: " << version.major << "."
                  << version.minor << "." << version.patch << '\n';
    } catch(const std::exception& e)
    {
        std::cerr << "rocpdsna consumer smoke FAILED: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
