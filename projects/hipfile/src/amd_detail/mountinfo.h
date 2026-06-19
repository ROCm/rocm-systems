/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <optional>
#include <sys/types.h>

struct libmnt_context;
struct libmnt_fs;
struct libmnt_table;

namespace hipFile {

/// @brief The filesystem type
enum class FilesystemType {
    ext4,
    xfs,
    other,
};

/// @brief ext (3/4) journaling mode
enum class ExtJournalingMode {
    journal,
    ordered,
    writeback,
    unknown,
};

/// @brief ext4 mount options
struct Ext4MountOptions {
    ExtJournalingMode journaling_mode;
};

/// @brief The options specified when the filesystem was mounted
union MountOptions {
    Ext4MountOptions ext4;
};

/// @brief File system mount information
struct MountInfo {
    /// @brief Filesystem type
    FilesystemType type;

    /// @brief Filesystem mount options
    MountOptions options;
};

/// @brief (incomplete) libmount interface wrapper.
///
/// Consider extending/using LibMountHelper instead of directly interfacing with
/// LibMount.
class LibMount {
public:
    virtual ~LibMount();

    virtual libmnt_context *mnt_new_context();
    virtual void            mnt_free_context(struct libmnt_context *cxt);
    virtual int             mnt_context_get_mtab(libmnt_context *cxt, libmnt_table **tb);
    virtual libmnt_fs      *mnt_table_find_devno(libmnt_table *tb, dev_t devno, int direction);
    virtual const char     *mnt_fs_get_fstype(libmnt_fs *fs);
    virtual int             mnt_fs_get_option(libmnt_fs *fs, const char *name, char **value, size_t *valsz);
};

/// @brief Simplify libmount interactions.
class LibMountHelper {
public:
    virtual ~LibMountHelper();

    virtual std::optional<MountInfo> getMountInfo(dev_t dev) const;
};

}
