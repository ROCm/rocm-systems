/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "mountinfo.h"

#include <cstring>
#include <libmount/libmount.h>
#include <stdexcept>

struct libmnt_context;
struct libmnt_fs;
struct libmnt_table;

namespace hipFile {

LibMount::~LibMount()
{
}

libmnt_context *
LibMount::mnt_new_context()
{
    return ::mnt_new_context();
}

void
LibMount::mnt_free_context(struct libmnt_context *cxt)
{
    ::mnt_free_context(cxt);
}

int
LibMount::mnt_context_get_mtab(libmnt_context *cxt, libmnt_table **tb)
{
    return ::mnt_context_get_mtab(cxt, tb);
}

libmnt_fs *
LibMount::mnt_table_find_devno(libmnt_table *tb, dev_t devno, int direction)
{
    return ::mnt_table_find_devno(tb, devno, direction);
}

const char *
LibMount::mnt_fs_get_fstype(libmnt_fs *fs)
{
    return ::mnt_fs_get_fstype(fs);
}

int
LibMount::mnt_fs_get_option(libmnt_fs *fs, const char *name, char **value, size_t *valsz)
{
    return ::mnt_fs_get_option(fs, name, value, valsz);
}

LibMountHelper::~LibMountHelper()
{
}

std::optional<MountInfo>
LibMountHelper::getMountInfo(dev_t dev) const
{
    auto libmount{Context<LibMount>::get()};

    auto mnt_ctx{libmount->mnt_new_context()};
    if (!mnt_ctx) {
        throw std::runtime_error("libmount: Could not create context");
    }

    struct libmnt_table *mnt_tbl{};
    if (libmount->mnt_context_get_mtab(mnt_ctx, &mnt_tbl) != 0) {
        libmount->mnt_free_context(mnt_ctx);
        throw std::runtime_error("libmount: Could not get mount table");
    }

    struct libmnt_fs *mnt_fs{libmount->mnt_table_find_devno(mnt_tbl, dev, MNT_ITER_FORWARD)};
    if (!mnt_fs) {
        libmount->mnt_free_context(mnt_ctx);
        return {};
    }

    MountInfo mountinfo{};

    const char *fstype{libmount->mnt_fs_get_fstype(mnt_fs)};
    if (fstype && !strcmp(fstype, "ext4")) {
        mountinfo.type = FilesystemType::ext4;

        char *value;
        switch (libmount->mnt_fs_get_option(mnt_fs, "data", &value, nullptr)) {
            case 0: // option found
                if (value && !strcmp(value, "ordered")) {
                    mountinfo.options.ext4.journaling_mode = ExtJournalingMode::ordered;
                }
                else if (value && !strcmp(value, "journal")) {
                    mountinfo.options.ext4.journaling_mode = ExtJournalingMode::journal;
                }
                else if (value && !strcmp(value, "writeback")) {
                    mountinfo.options.ext4.journaling_mode = ExtJournalingMode::writeback;
                }
                else {
                    mountinfo.options.ext4.journaling_mode = ExtJournalingMode::unknown;
                }
                break;
            case 1: // option not found
                mountinfo.options.ext4.journaling_mode = ExtJournalingMode::ordered;
                break;
            default: // error
                libmount->mnt_free_context(mnt_ctx);
                throw std::runtime_error("libmount: Could not get mount option: data");
        }
    }
    else if (fstype && !strcmp(fstype, "xfs")) {
        mountinfo.type = FilesystemType::xfs;
    }
    else {
        mountinfo.type = FilesystemType::other;
    }

    libmount->mnt_free_context(mnt_ctx);
    return std::make_optional(mountinfo);
}

}
