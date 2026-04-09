/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "file.h"
#include "file-descriptor.h"
#include "mountinfo.h"
#include "passkey.h"
#include "sys.h"

#include <fcntl.h>
#include <utility>
#include <string>
#include <sys/sysmacros.h>
#include <syslog.h>
#include <system_error>
#include <unistd.h>

using namespace std;

namespace hipFile {

UnregisteredFile::UnregisteredFile(int fd)
    : client_fd(FileDescriptor::make_unmanaged(fd)), buffered_fd{}, unbuffered_fd{},
      stx{Context<Sys>::get()->statx(fd, "", AT_EMPTY_PATH,
#if defined(STATX_DIOALIGN)
                                     STATX_TYPE | STATX_MODE | STATX_DIOALIGN
#else
                                     STATX_TYPE | STATX_MODE
#endif
                                     )},
      flags{Context<Sys>::get()->fcntl(fd, F_GETFL, 0)},
      mountinfo{Context<LibMountHelper>::get()->getMountInfo(makedev(stx.stx_dev_major, stx.stx_dev_minor))},
#if defined(STATX_DIOALIGN)
      m_dio_mem_align{stx.stx_mask & STATX_DIOALIGN ? stx.stx_dio_mem_align : 4096},
      m_dio_offset_align{stx.stx_mask & STATX_DIOALIGN ? stx.stx_dio_offset_align : 4096}
#else
      m_dio_mem_align{4096}, m_dio_offset_align{4096}
#endif
{
    std::string path = "/proc/self/fd/" + std::to_string(fd);

    if (flags & O_DIRECT) {
        unbuffered_fd = FileDescriptor::make_unmanaged(fd);
        buffered_fd   = FileDescriptor::make_managed(
            Context<Sys>::get()->open(path.c_str(), (flags | O_CLOEXEC) & ~O_DIRECT));
    }
    else {
        buffered_fd = FileDescriptor::make_unmanaged(fd);
        try {
            unbuffered_fd = FileDescriptor::make_managed(
                Context<Sys>::get()->open(path.c_str(), (flags | O_CLOEXEC) | O_DIRECT));
        }
        catch (const std::system_error &e) {
            if (e.code().value() != EINVAL) {
                throw;
            }
            unbuffered_fd      = nullopt;
            m_dio_mem_align    = 0;
            m_dio_offset_align = 0;
        }
    }
}

hipFileHandle_t
IFile::handle() const noexcept
{
    return reinterpret_cast<hipFileHandle_t>(const_cast<IFile *>(this));
}

File::File(UnregisteredFile &&uf, const PassKey<FileMap> &)
    : m_client_fd{std::move(uf.client_fd)}, m_buffered_fd{std::move(uf.buffered_fd)},
      m_unbuffered_fd{std::move(uf.unbuffered_fd)}, m_dio_mem_align{uf.m_dio_mem_align},
      m_dio_offset_align{uf.m_dio_offset_align},
      m_is_block_device{(uf.stx.stx_mask & STATX_TYPE) && S_ISBLK(uf.stx.stx_mode)},
      m_is_regular_file{(uf.stx.stx_mask & STATX_TYPE) && S_ISREG(uf.stx.stx_mode)},
      m_on_ext4_ordered{uf.mountinfo && uf.mountinfo->type == FilesystemType::ext4 &&
                        uf.mountinfo->options.ext4.journaling_mode == ExtJournalingMode::ordered},
      m_on_xfs{uf.mountinfo && uf.mountinfo->type == FilesystemType::xfs}
{
}

int
File::clientFd() const noexcept
{
    return m_client_fd.get();
}

int
File::bufferedFd() const noexcept
{
    return m_buffered_fd.get();
}

optional<int>
File::unbufferedFd() const noexcept
{
    if (m_unbuffered_fd) {
        return m_unbuffered_fd.value().get();
    }
    return nullopt;
}

uint32_t
File::dioMemAlign() const noexcept
{
    return m_dio_mem_align;
}

uint32_t
File::dioOffsetAlign() const noexcept
{
    return m_dio_offset_align;
}

bool
File::isBlockDevice() const noexcept
{
    return m_is_block_device;
}

bool
File::isRegularFile() const noexcept
{
    return m_is_regular_file;
}

bool
File::onExt4Ordered() const noexcept
{
    return m_on_ext4_ordered;
}

bool
File::onXfs() const noexcept
{
    return m_on_xfs;
}

shared_ptr<IFile>
FileMap::getFile(hipFileHandle_t fh)
{
    auto itr = from_fh.find(fh);
    if (from_fh.end() == itr) {
        throw FileNotRegistered();
    }

    return itr->second;
}

hipFileHandle_t
FileMap::registerFile(UnregisteredFile &&uf)
{
    if (from_fd.end() != from_fd.find(uf.client_fd.get())) {
        throw FileAlreadyRegistered();
    }

    auto file                 = std::shared_ptr<IFile>(new File(std::move(uf), PassKey<FileMap>{}));
    from_fd[file->clientFd()] = file;
    from_fh[file->handle()]   = file;

    return file->handle();
}

void
FileMap::deregisterFile(hipFileHandle_t fh)
{
    auto itr = from_fh.find(fh);

    if (from_fh.end() == itr) {
        throw FileNotRegistered();
    }

    if (2 < itr->second.use_count()) {
        throw FileOperationsOutstanding();
    }

    from_fd.erase(itr->second->clientFd());
    from_fh.erase(fh);
}

void
FileMap::clear()
{
    from_fd.clear();
    from_fh.clear();
}

FileMap::~FileMap()
{
    // Complain in the logs if we're shutting down a FileMap
    // with files that are still in use.
    int count = 0;
    for (const auto &p : from_fh) {
        if (p.second.use_count() > 2) {
            count++;
        }
    }

    if (count > 0) {
        Context<Sys>::get()->syslog(LOG_CRIT, "FileMap state is being destructed with in-use files");
    }
}

}
