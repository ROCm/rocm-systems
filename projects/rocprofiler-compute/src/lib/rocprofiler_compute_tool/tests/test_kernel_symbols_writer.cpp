// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_kernel_symbols_writer.h"

#include <unistd.h>

#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>

using namespace rocprofiler_compute_tool;

namespace
{
constexpr const char* kHeader =
    "kernel_id,kernel_name,kernel_short_name,arch_vgpr,accum_vgpr,sgpr\n";

std::filesystem::path test_directory()
{
    return std::filesystem::temp_directory_path() /
           ("kernel_symbols_writer_test_" + std::to_string(::getpid()));
}
}  // namespace

std::string TestKernelSymbolsWriter::format()
{
    std::string csv;

    EXPECT_TRUE(format_kernel_symbols_csv(m_tool_data,
                                          [&csv](std::string_view batch)
                                          {
                                              csv.append(batch);
                                              return true;
                                          }));

    return csv;
}

void TestKernelSymbolsWriter::add_symbol(uint64_t kernel_id, const std::string& kernel_name)
{
    kernel_symbol_record_t symbol{};
    symbol.kernel_name                    = kernel_name;
    symbol.kernel_short_name              = "vecCopy";
    symbol.arch_vgpr_count                = 8;
    symbol.accum_vgpr_count               = 0;
    symbol.sgpr_count                     = 16;
    m_tool_data.kernel_symbols[kernel_id] = std::move(symbol);
}

TEST_F(TestKernelSymbolsWriter, NoSymbols_WritesOnlyTheHeader)
{
    EXPECT_EQ(format(), kHeader);
}

TEST_F(TestKernelSymbolsWriter, Symbols_AreWrittenInKernelIdOrder)
{
    // Insertion order is not id order, and the map does not preserve either.
    add_symbol(9, "vecCopy(double*, double*, double*, int)");
    add_symbol(3, "vecCopy(double*, double*, double*, int)");

    EXPECT_EQ(format(),
              std::string{kHeader} + "3,\"vecCopy(double*, double*, double*, int)\",\"vecCopy\",8,0,16\n" +
                  "9,\"vecCopy(double*, double*, double*, int)\",\"vecCopy\",8,0,16\n");
}

TEST_F(TestKernelSymbolsWriter, SinkFailure_IsReported)
{
    add_symbol(1, "vecCopy(double*, double*, double*, int)");

    EXPECT_FALSE(format_kernel_symbols_csv(m_tool_data, [](std::string_view) { return false; }));
}

TEST_F(TestKernelSymbolsWriter, NoSymbols_WritesNoFile)
{
    const auto directory = test_directory();
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    m_tool_data.kernel_symbols_filename = (directory / "1234_kernel_symbols.csv.gz").string();

    KernelSymbolsWriter writer;
    writer.write(m_tool_data);

    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{directory},
                            std::filesystem::directory_iterator{}),
              0);

    std::filesystem::remove_all(directory);
}

TEST_F(TestKernelSymbolsWriter, KernelSymbolsFilename_IsTheOnlyFileWritten)
{
    const auto directory = test_directory();
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    const auto path = directory / "1234_kernel_symbols.csv.gz";

    m_tool_data.kernel_symbols_filename = path.string();
    add_symbol(1, "vecCopy(double*, double*, double*, int)");

    KernelSymbolsWriter writer;
    writer.write(m_tool_data);

    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{directory},
                            std::filesystem::directory_iterator{}),
              1);

    std::filesystem::remove_all(directory);
}
