// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_source_snapshotter.h"

#include <algorithm>
#include <system_error>

using namespace rocprofiler_compute_tool;

namespace
{
std::filesystem::file_status regular_file_status()
{
    return std::filesystem::file_status{std::filesystem::file_type::regular};
}

std::filesystem::file_status directory_status()
{
    return std::filesystem::file_status{std::filesystem::file_type::directory};
}

std::error_code permission_denied_error()
{
    return std::make_error_code(std::errc::permission_denied);
}

std::error_code missing_file_error()
{
    return std::make_error_code(std::errc::no_such_file_or_directory);
}

bool contains_destination(const std::vector<mock_filesystem_wrapper_t::copy_file_call_t>& copy_file_calls,
                          const std::filesystem::path& destination_path)
{
    return std::any_of(copy_file_calls.begin(),
                       copy_file_calls.end(),
                       [&destination_path](const auto& call)
                       { return call.destination == destination_path; });
}
}  // namespace

void test_source_snapshotter_t::SetUp()
{
    m_filesystem  = std::make_shared<mock_filesystem_wrapper_t>();
    m_snapshotter = std::make_shared<source_snapshotter_impl_t>(m_filesystem);
}

std::filesystem::path test_source_snapshotter_t::destination_path(const std::filesystem::path& source_path) const
{
    return m_destination_root / source_path.relative_path();
}

void test_source_snapshotter_t::set_regular_source(const std::filesystem::path& source_path)
{
    m_filesystem->set_status(source_path, regular_file_status());
}

void test_source_snapshotter_t::expect_no_copy() const
{
    EXPECT_TRUE(m_filesystem->get_create_directories_calls().empty());
    EXPECT_TRUE(m_filesystem->get_copy_file_calls().empty());
}

TEST_F(test_source_snapshotter_t, ProvidedEmptyInput_DoesNothing)
{
    EXPECT_NO_THROW(m_snapshotter->snapshot({}, m_destination_root));

    EXPECT_TRUE(m_filesystem->get_absolute_calls().empty());
    EXPECT_TRUE(m_filesystem->get_status_calls().empty());
    expect_no_copy();
}

TEST_F(test_source_snapshotter_t, Create_ReturnsSnapshotter)
{
    EXPECT_NE(source_snapshotter_t::create(), nullptr);
}

TEST_F(test_source_snapshotter_t, ProvidedEmptyPath_SkipsIt)
{
    EXPECT_NO_THROW(m_snapshotter->snapshot({std::filesystem::path{}}, m_destination_root));

    EXPECT_TRUE(m_filesystem->get_absolute_calls().empty());
    EXPECT_TRUE(m_filesystem->get_status_calls().empty());
    expect_no_copy();
}

TEST_F(test_source_snapshotter_t, ProvidedMissingSource_SkipsIt)
{
    const std::filesystem::path source_path = "/sources/missing.cpp";

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, m_destination_root));

    EXPECT_EQ(m_filesystem->get_status_calls(), std::vector<std::filesystem::path>{source_path});
    expect_no_copy();
}

TEST_F(test_source_snapshotter_t, ProvidedDirectorySource_SkipsIt)
{
    const std::filesystem::path source_path = "/sources/include";
    m_filesystem->set_status(source_path, directory_status());

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, m_destination_root));

    EXPECT_EQ(m_filesystem->get_status_calls(), std::vector<std::filesystem::path>{source_path});
    expect_no_copy();
}

TEST_F(test_source_snapshotter_t, ProvidedRegularSource_CreatesParentAndCopiesToMirroredPath)
{
    const std::filesystem::path source_path = "/sources/app/kernel.cpp";
    const auto                  destination = destination_path(source_path);
    set_regular_source(source_path);

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, m_destination_root));

    EXPECT_EQ(m_filesystem->get_create_directories_calls(),
              std::vector<std::filesystem::path>{destination.parent_path()});
    ASSERT_EQ(m_filesystem->get_copy_file_calls().size(), 1);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].source, source_path);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].destination, destination);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].options,
              std::filesystem::copy_options::overwrite_existing);
}

TEST_F(test_source_snapshotter_t, ProvidedSingleComponentMirror_CreatesDestinationRoot)
{
    const std::filesystem::path source_path      = "/kernel.cpp";
    const std::filesystem::path destination_root = "snapshot";
    const auto                  destination      = destination_root / source_path.relative_path();
    set_regular_source(source_path);

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, destination_root));

    EXPECT_EQ(m_filesystem->get_has_parent_path_calls(), std::vector<std::filesystem::path>{destination});
    EXPECT_EQ(m_filesystem->get_create_directories_calls(),
              std::vector<std::filesystem::path>{destination.parent_path()});
    ASSERT_EQ(m_filesystem->get_copy_file_calls().size(), 1);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].destination, destination);
}

TEST_F(test_source_snapshotter_t, ProvidedRelativeSource_ChecksAbsolutePathAndCopiesSuppliedPath)
{
    const std::filesystem::path relative_source_path = "sources/app/kernel.cpp";
    const std::filesystem::path absolute_source_path = "/working/sources/app/kernel.cpp";
    const auto                  destination          = destination_path(absolute_source_path);
    m_filesystem->set_absolute(relative_source_path, absolute_source_path);
    set_regular_source(absolute_source_path);

    EXPECT_NO_THROW(m_snapshotter->snapshot({relative_source_path}, m_destination_root));

    EXPECT_EQ(m_filesystem->get_absolute_calls(),
              std::vector<std::filesystem::path>{relative_source_path});
    EXPECT_EQ(m_filesystem->get_status_calls(), std::vector<std::filesystem::path>{absolute_source_path});
    EXPECT_EQ(m_filesystem->get_weakly_canonical_calls(),
              std::vector<std::filesystem::path>{absolute_source_path});
    EXPECT_EQ(m_filesystem->get_relative_path_calls(),
              std::vector<std::filesystem::path>{absolute_source_path});
    ASSERT_EQ(m_filesystem->get_copy_file_calls().size(), 1);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].source, relative_source_path);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].destination, destination);
}

TEST_F(test_source_snapshotter_t, ProvidedAbsoluteError_SkipsIt)
{
    const std::filesystem::path source_path = "sources/missing.cpp";
    m_filesystem->set_absolute_error(source_path, missing_file_error());

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, m_destination_root));

    EXPECT_EQ(m_filesystem->get_absolute_calls(), std::vector<std::filesystem::path>{source_path});
    EXPECT_TRUE(m_filesystem->get_status_calls().empty());
    expect_no_copy();
}

TEST_F(test_source_snapshotter_t, ProvidedWeaklyCanonicalError_SkipsItAndContinues)
{
    const std::filesystem::path failed_source_path  = "/sources/failed.cpp";
    const std::filesystem::path present_source_path = "/sources/present.cpp";
    set_regular_source(failed_source_path);
    set_regular_source(present_source_path);
    m_filesystem->set_weakly_canonical_error(failed_source_path, permission_denied_error());
    m_filesystem->set_weakly_canonical(present_source_path, present_source_path);

    EXPECT_NO_THROW(
        m_snapshotter->snapshot({failed_source_path, present_source_path}, m_destination_root));

    const std::vector<std::filesystem::path> expected_canonical_calls = {failed_source_path,
                                                                         present_source_path};
    EXPECT_EQ(m_filesystem->get_weakly_canonical_calls(), expected_canonical_calls);
    EXPECT_EQ(m_filesystem->get_relative_path_calls(),
              std::vector<std::filesystem::path>{present_source_path});
    ASSERT_EQ(m_filesystem->get_copy_file_calls().size(), 1);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].source, present_source_path);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].destination, destination_path(present_source_path));
}

TEST_F(test_source_snapshotter_t, ProvidedMixedPresentAndMissingSources_CopiesOnlyPresent)
{
    const std::filesystem::path present_source_path = "/sources/present.cpp";
    const std::filesystem::path missing_source_path = "/sources/missing.cpp";
    set_regular_source(present_source_path);

    EXPECT_NO_THROW(
        m_snapshotter->snapshot({present_source_path, missing_source_path}, m_destination_root));

    ASSERT_EQ(m_filesystem->get_copy_file_calls().size(), 1);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].source, present_source_path);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].destination, destination_path(present_source_path));
}

TEST_F(test_source_snapshotter_t, ProvidedCreateDirectoryError_SkipsCopy)
{
    const std::filesystem::path source_path = "/sources/app/kernel.cpp";
    set_regular_source(source_path);
    m_filesystem->set_create_directories_error(permission_denied_error());

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, m_destination_root));

    EXPECT_EQ(m_filesystem->get_create_directories_calls(),
              std::vector<std::filesystem::path>{destination_path(source_path).parent_path()});
    EXPECT_TRUE(m_filesystem->get_copy_file_calls().empty());
}

TEST_F(test_source_snapshotter_t, ProvidedCopyError_DoesNotThrow)
{
    const std::filesystem::path source_path = "/sources/app/kernel.cpp";
    set_regular_source(source_path);
    m_filesystem->set_copy_file_error(permission_denied_error());

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, m_destination_root));

    ASSERT_EQ(m_filesystem->get_copy_file_calls().size(), 1);
    EXPECT_EQ(m_filesystem->get_copy_file_calls()[0].source, source_path);
}

TEST_F(test_source_snapshotter_t, ProvidedSameBasenameDifferentDirectories_CopiesBoth)
{
    const std::filesystem::path first_source_path  = "/sources/a/kernel.cpp";
    const std::filesystem::path second_source_path = "/sources/b/kernel.cpp";
    set_regular_source(first_source_path);
    set_regular_source(second_source_path);

    EXPECT_NO_THROW(m_snapshotter->snapshot({first_source_path, second_source_path}, m_destination_root));

    ASSERT_EQ(m_filesystem->get_copy_file_calls().size(), 2);
    EXPECT_TRUE(contains_destination(m_filesystem->get_copy_file_calls(),
                                     destination_path(first_source_path)));
    EXPECT_TRUE(contains_destination(m_filesystem->get_copy_file_calls(),
                                     destination_path(second_source_path)));
}

TEST_F(test_source_snapshotter_t, ProvidedStatusError_SkipsIt)
{
    const std::filesystem::path source_path = "/sources/missing.cpp";
    m_filesystem->set_status_error(source_path, missing_file_error());

    EXPECT_NO_THROW(m_snapshotter->snapshot({source_path}, m_destination_root));

    EXPECT_EQ(m_filesystem->get_status_calls(), std::vector<std::filesystem::path>{source_path});
    expect_no_copy();
}
