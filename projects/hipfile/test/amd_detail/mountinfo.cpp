/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "mmountinfo.h"
#include "mountinfo.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <libmount/libmount.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <tuple>
#include <utility>

struct libmnt_context;
struct libmnt_fs;
struct libmnt_table;

using namespace hipFile;
using namespace testing;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(LibMountHelper, GetMountInfoThrowsOnContextCreationFailure)
{
    StrictMock<MLibMount> mlibmount;

    auto dev{makedev(123, 456)};

    EXPECT_CALL(mlibmount, mnt_new_context).Times(1).WillOnce(Return(nullptr));

    ASSERT_THAT(([=] { LibMountHelper().getMountInfo(dev); }),
                ThrowsMessage<std::runtime_error>(StrEq("libmount: Could not create context")));
}

TEST(LibMountHelper, GetMountInfoThrowsOnGetMountTableFailure)
{
    StrictMock<MLibMount> mlibmount;

    auto dev{makedev(123, 456)};
    auto cxt = reinterpret_cast<libmnt_context *>(0xFACEFEED);

    EXPECT_CALL(mlibmount, mnt_new_context).WillOnce(Return(cxt));
    EXPECT_CALL(mlibmount, mnt_context_get_mtab(cxt, NotNull())).WillOnce(Return(-1));
    EXPECT_CALL(mlibmount, mnt_free_context(cxt));

    ASSERT_THAT(([=] { LibMountHelper().getMountInfo(dev); }),
                ThrowsMessage<std::runtime_error>(StrEq("libmount: Could not get mount table")));
}

TEST(LibMountHelper, GetMountInfoThowsOnGetFilesystemOptionFailure)
{
    StrictMock<MLibMount> mlibmount;

    auto dev{makedev(123, 456)};
    auto cxt = reinterpret_cast<libmnt_context *>(0xFACEFEED);
    auto tbl = reinterpret_cast<libmnt_table *>(0x0BADF00D);
    auto fs  = reinterpret_cast<libmnt_fs *>(0xCAFEBABE);

    EXPECT_CALL(mlibmount, mnt_new_context).WillOnce(Return(cxt));
    EXPECT_CALL(mlibmount, mnt_context_get_mtab(cxt, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(tbl), Return(0)));
    EXPECT_CALL(mlibmount, mnt_table_find_devno(tbl, dev, MNT_ITER_FORWARD)).WillOnce(Return(fs));
    EXPECT_CALL(mlibmount, mnt_fs_get_fstype(fs)).WillOnce(Return("ext4"));
    EXPECT_CALL(mlibmount, mnt_fs_get_option(fs, StrEq("data"), NotNull(), _)).WillOnce(Return(-1));
    EXPECT_CALL(mlibmount, mnt_free_context(cxt));

    ASSERT_THAT(([=] { LibMountHelper().getMountInfo(dev); }),
                ThrowsMessage<std::runtime_error>(StrEq("libmount: Could not get mount option: data")));
}

TEST(LibMountHelper, GetMountInfoReturnsEmptyOptionIfNoInfoForDevFound)
{
    StrictMock<MLibMount> mlibmount;

    auto dev{makedev(123, 456)};
    auto cxt = reinterpret_cast<libmnt_context *>(0xFACEFEED);
    auto tbl = reinterpret_cast<libmnt_table *>(0x0BADF00D);

    EXPECT_CALL(mlibmount, mnt_new_context).WillOnce(Return(cxt));
    EXPECT_CALL(mlibmount, mnt_context_get_mtab(cxt, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(tbl), Return(0)));
    EXPECT_CALL(mlibmount, mnt_table_find_devno(tbl, dev, MNT_ITER_FORWARD)).WillOnce(Return(nullptr));
    EXPECT_CALL(mlibmount, mnt_free_context(cxt));

    auto optional_mount_info = LibMountHelper().getMountInfo(dev);

    ASSERT_FALSE(optional_mount_info.has_value());
}

class LibMountHelperFilesystemTypeParam : public TestWithParam<std::tuple<const char *, FilesystemType>> {};

TEST_P(LibMountHelperFilesystemTypeParam, GetMountInfoHandlesFilesystemType)
{
    StrictMock<MLibMount> mlibmount;

    auto mnt_fs_get_type_return{std::get<0>(GetParam())};
    auto expected_filesystem_type{std::get<1>(GetParam())};

    auto dev{makedev(123, 456)};
    auto cxt = reinterpret_cast<libmnt_context *>(0xFACEFEED);
    auto tbl = reinterpret_cast<libmnt_table *>(0x0BADF00D);
    auto fs  = reinterpret_cast<libmnt_fs *>(0xCAFEBABE);

    EXPECT_CALL(mlibmount, mnt_new_context).WillOnce(Return(cxt));
    EXPECT_CALL(mlibmount, mnt_context_get_mtab(cxt, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(tbl), Return(0)));
    EXPECT_CALL(mlibmount, mnt_table_find_devno(tbl, dev, MNT_ITER_FORWARD)).WillOnce(Return(fs));
    EXPECT_CALL(mlibmount, mnt_fs_get_fstype(fs)).WillOnce(Return(mnt_fs_get_type_return));
    EXPECT_CALL(mlibmount, mnt_free_context(cxt));

    auto optional_mount_info = LibMountHelper().getMountInfo(dev);
    ASSERT_TRUE(optional_mount_info.has_value());

    auto mount_info{std::move(*optional_mount_info)};
    ASSERT_EQ(mount_info.type, expected_filesystem_type);
}

INSTANTIATE_TEST_SUITE_P(
    LibMountHelperFilesystemType, LibMountHelperFilesystemTypeParam,
    Values(
        std::make_tuple("autofs", FilesystemType::other), std::make_tuple("beegfs", FilesystemType::other),
        std::make_tuple("binfmt_misc", FilesystemType::other), std::make_tuple("bpf", FilesystemType::other),
        std::make_tuple("btrfs", FilesystemType::other), std::make_tuple("cgroup2", FilesystemType::other),
        std::make_tuple("configfs", FilesystemType::other), std::make_tuple("debugfs", FilesystemType::other),
        std::make_tuple("devpts", FilesystemType::other), std::make_tuple("devtmpfs", FilesystemType::other),
        std::make_tuple("efivarfs", FilesystemType::other), std::make_tuple("ext2", FilesystemType::other),
        std::make_tuple("ext3", FilesystemType::other), std::make_tuple("f2fs", FilesystemType::other),
        std::make_tuple("fusectl", FilesystemType::other),
        std::make_tuple("hugetlbfs", FilesystemType::other), std::make_tuple("jfs", FilesystemType::other),
        std::make_tuple("mqueue", FilesystemType::other), std::make_tuple("nfs4", FilesystemType::other),
        std::make_tuple("proc", FilesystemType::other), std::make_tuple("pstore", FilesystemType::other),
        std::make_tuple("rpc_pipefs", FilesystemType::other),
        std::make_tuple("securityfs", FilesystemType::other),
        std::make_tuple("squashfs", FilesystemType::other), std::make_tuple("sysfs", FilesystemType::other),
        std::make_tuple("tmpfs", FilesystemType::other), std::make_tuple("tracefs", FilesystemType::other),
        std::make_tuple("vfat", FilesystemType::other), std::make_tuple("xfs", FilesystemType::xfs),
        std::make_tuple("zfs", FilesystemType::other)));

class LibMountHelperExt4FilesystemTypeParam
    : public TestWithParam<std::tuple<int, const char *, ExtJournalingMode>> {};

TEST_P(LibMountHelperExt4FilesystemTypeParam, GetMountInfoHandlesExt4FilesystemType)
{
    StrictMock<MLibMount> mlibmount;

    auto mnt_fs_get_option_return{std::get<0>(GetParam())};
    auto mnt_fs_get_option_value{std::get<1>(GetParam())};
    auto expected_journaling_mode{std::get<2>(GetParam())};

    auto dev{makedev(123, 456)};
    auto cxt = reinterpret_cast<libmnt_context *>(0xFACEFEED);
    auto tbl = reinterpret_cast<libmnt_table *>(0x0BADF00D);
    auto fs  = reinterpret_cast<libmnt_fs *>(0xCAFEBABE);

    EXPECT_CALL(mlibmount, mnt_new_context).WillOnce(Return(cxt));
    EXPECT_CALL(mlibmount, mnt_context_get_mtab(cxt, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(tbl), Return(0)));
    EXPECT_CALL(mlibmount, mnt_table_find_devno(tbl, dev, MNT_ITER_FORWARD)).WillOnce(Return(fs));
    EXPECT_CALL(mlibmount, mnt_fs_get_fstype(fs)).WillOnce(Return("ext4"));
    EXPECT_CALL(mlibmount, mnt_fs_get_option(fs, StrEq("data"), NotNull(), _))
        .WillOnce(DoAll(SetArgPointee<2>(const_cast<char *>(mnt_fs_get_option_value)),
                        Return(mnt_fs_get_option_return)));
    EXPECT_CALL(mlibmount, mnt_free_context(cxt));

    auto optional_mountinfo = LibMountHelper().getMountInfo(dev);
    ASSERT_TRUE(optional_mountinfo.has_value());

    auto mountinfo{std::move(*optional_mountinfo)};
    ASSERT_EQ(mountinfo.type, FilesystemType::ext4);
    ASSERT_EQ(mountinfo.options.ext4.journaling_mode, expected_journaling_mode);
}

INSTANTIATE_TEST_SUITE_P(LibMountHelperExt4FilesystemType, LibMountHelperExt4FilesystemTypeParam,
                         Values(std::make_tuple(1, nullptr, ExtJournalingMode::ordered),
                                std::make_tuple(0, "ordered", ExtJournalingMode::ordered),
                                std::make_tuple(0, "journal", ExtJournalingMode::journal),
                                std::make_tuple(0, "writeback", ExtJournalingMode::writeback),
                                std::make_tuple(0, "foobar", ExtJournalingMode::unknown),
                                std::make_tuple(0, "order", ExtJournalingMode::unknown)));

TEST(LibMountHelper, GetMountInfoHandlesXfsFileSystem)
{
    StrictMock<MLibMount> mlibmount;

    auto dev{makedev(123, 456)};
    auto cxt = reinterpret_cast<libmnt_context *>(0xFACEFEED);
    auto tbl = reinterpret_cast<libmnt_table *>(0x0BADF00D);
    auto fs  = reinterpret_cast<libmnt_fs *>(0xCAFEBABE);

    EXPECT_CALL(mlibmount, mnt_new_context).WillOnce(Return(cxt));
    EXPECT_CALL(mlibmount, mnt_context_get_mtab(cxt, NotNull()))
        .WillOnce(DoAll(SetArgPointee<1>(tbl), Return(0)));
    EXPECT_CALL(mlibmount, mnt_table_find_devno(tbl, dev, MNT_ITER_FORWARD)).WillOnce(Return(fs));
    EXPECT_CALL(mlibmount, mnt_fs_get_fstype(fs)).WillOnce(Return("xfs"));
    EXPECT_CALL(mlibmount, mnt_free_context(cxt));

    auto optional_mountinfo = LibMountHelper().getMountInfo(dev);
    ASSERT_TRUE(optional_mountinfo.has_value());

    auto mountinfo{std::move(*optional_mountinfo)};
    ASSERT_EQ(mountinfo.type, FilesystemType::xfs);
}
HIPFILE_WARN_NO_GLOBAL_CTOR_ON
