/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "mountinfo.h"

#include <gmock/gmock.h>

namespace hipFile {

struct MLibMount : LibMount {
    ContextOverride<LibMount> co;

    MLibMount() : co{this}
    {
    }

    MOCK_METHOD(libmnt_context *, mnt_new_context, (), (override));
    MOCK_METHOD(void, mnt_free_context, (struct libmnt_context * cxt), (override));
    MOCK_METHOD(int, mnt_context_get_mtab, (libmnt_context * cxt, libmnt_table **tb), (override));
    MOCK_METHOD(libmnt_fs *, mnt_table_find_devno, (libmnt_table * tb, dev_t devno, int direction),
                (override));
    MOCK_METHOD(const char *, mnt_fs_get_fstype, (libmnt_fs * fs), (override));
    MOCK_METHOD(int, mnt_fs_get_option, (libmnt_fs * fs, const char *name, char **value, size_t *valsz),
                (override));
};

struct MLibMountHelper : LibMountHelper {
    ContextOverride<LibMountHelper> co;

    MLibMountHelper() : co{this}
    {
    }

    MOCK_METHOD(std::optional<MountInfo>, getMountInfo, (dev_t dev), (const override));
};
}
