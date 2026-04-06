// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Custom SQLite VFS: mmap-backed WAL files
//
// Design overview
// ---------------
// SQLite passes SQLITE_OPEN_WAL in the flags argument to xOpen when it opens
// the WAL file.  This shim VFS intercepts those opens, pre-allocates the WAL
// file to a user-supplied size, and mmap()s the region.  xWrite/xRead then
// operate directly on the mapped memory instead of going through system calls.
//
// For every other file type (main database, shm) the shim delegates straight
// to the underlying unix VFS without modification.
//
// File object layout
// ------------------
// SQLite allocates szOsFile bytes for each file object.  We set
//   szOsFile = sizeof(WalMmapFile) + parent->szOsFile
// so the parent file object lives immediately after our header in the same
// allocation:
//
//   [ WalMmapFile | <parent sqlite3_file> ]
//    ^              ^
//    pFile          pFile->pUnderlying
//
// For non-WAL files we still call the parent xOpen with pFile as the buffer
// (safe because szOsFile >= parent->szOsFile).  The parent sets pFile->pMethods
// to its own io_methods, so subsequent calls on those files go directly to the
// parent without touching WalMmapFile fields.

#include "wal_mmap_vfs.hpp"

#include <sqlite3.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace rocpdsna::data_storage
{

namespace
{

// ---------------------------------------------------------------------------
// Per-WAL-file object
// ---------------------------------------------------------------------------
struct WalMmapFile
{
    sqlite3_file  base;         // must be first; holds pMethods
    sqlite3_file* pUnderlying;  // parent file object (for locking / sync)
    int           fd;           // our own fd for ftruncate + mmap
    void*         pMmap;        // mmap'd region (MAP_FAILED if not mmap'd)
    sqlite3_int64 mmapSize;     // total pre-allocated bytes
    sqlite3_int64 logicalSize;  // bytes of valid WAL data written so far
};

// ---------------------------------------------------------------------------
// io_methods: forwarding helpers
// ---------------------------------------------------------------------------
static WalMmapFile*
wm(sqlite3_file* p)
{
    return reinterpret_cast<WalMmapFile*>(p);
}

static int
walMmapClose(sqlite3_file* pFile)
{
    WalMmapFile* p = wm(pFile);
    if(p->pMmap != MAP_FAILED)
    {
        ::msync(p->pMmap, static_cast<size_t>(p->logicalSize), MS_SYNC);
        ::munmap(p->pMmap, static_cast<size_t>(p->mmapSize));
        p->pMmap = MAP_FAILED;
    }
    if(p->fd >= 0)
    {
        ::close(p->fd);
        p->fd = -1;
    }
    return p->pUnderlying->pMethods->xClose(p->pUnderlying);
}

static int
walMmapRead(sqlite3_file* pFile, void* zBuf, int iAmt, sqlite3_int64 iOfst)
{
    WalMmapFile* p       = wm(pFile);
    sqlite3_int64 end    = iOfst + iAmt;
    sqlite3_int64 avail  = p->logicalSize - iOfst;

    if(avail <= 0)
    {
        ::memset(zBuf, 0, static_cast<size_t>(iAmt));
        return SQLITE_IOERR_SHORT_READ;
    }

    if(avail >= iAmt)
    {
        ::memcpy(zBuf,
                 static_cast<char*>(p->pMmap) + iOfst,
                 static_cast<size_t>(iAmt));
        return SQLITE_OK;
    }

    // Partial read: copy what we have, zero-fill the rest.
    ::memcpy(zBuf, static_cast<char*>(p->pMmap) + iOfst, static_cast<size_t>(avail));
    ::memset(static_cast<char*>(zBuf) + avail, 0, static_cast<size_t>(iAmt - avail));
    (void) end;
    return SQLITE_IOERR_SHORT_READ;
}

static int
walMmapWrite(sqlite3_file* pFile, const void* zBuf, int iAmt, sqlite3_int64 iOfst)
{
    WalMmapFile*  p   = wm(pFile);
    sqlite3_int64 end = iOfst + iAmt;

    if(end > p->mmapSize)
        return SQLITE_IOERR_WRITE;  // exceeded pre-allocated region

    ::memcpy(static_cast<char*>(p->pMmap) + iOfst, zBuf, static_cast<size_t>(iAmt));

    if(end > p->logicalSize)
        p->logicalSize = end;

    return SQLITE_OK;
}

static int
walMmapTruncate(sqlite3_file* pFile, sqlite3_int64 size)
{
    WalMmapFile* p = wm(pFile);
    if(size < p->logicalSize)
        p->logicalSize = size;
    return SQLITE_OK;
}

static int
walMmapSync(sqlite3_file* pFile, int flags)
{
    WalMmapFile* p = wm(pFile);
    if(p->pMmap != MAP_FAILED && p->logicalSize > 0)
    {
        int ms_flags = (flags & SQLITE_SYNC_FULL) ? MS_SYNC : MS_ASYNC;
        ::msync(p->pMmap, static_cast<size_t>(p->logicalSize), ms_flags);
    }
    return p->pUnderlying->pMethods->xSync(p->pUnderlying, flags);
}

static int
walMmapFileSize(sqlite3_file* pFile, sqlite3_int64* pSize)
{
    *pSize = wm(pFile)->logicalSize;
    return SQLITE_OK;
}

static int
walMmapLock(sqlite3_file* pFile, int eLock)
{
    return wm(pFile)->pUnderlying->pMethods->xLock(wm(pFile)->pUnderlying, eLock);
}

static int
walMmapUnlock(sqlite3_file* pFile, int eLock)
{
    return wm(pFile)->pUnderlying->pMethods->xUnlock(wm(pFile)->pUnderlying, eLock);
}

static int
walMmapCheckReservedLock(sqlite3_file* pFile, int* pResOut)
{
    return wm(pFile)->pUnderlying->pMethods->xCheckReservedLock(wm(pFile)->pUnderlying,
                                                                pResOut);
}

static int
walMmapFileControl(sqlite3_file* pFile, int op, void* pArg)
{
    return wm(pFile)->pUnderlying->pMethods->xFileControl(wm(pFile)->pUnderlying, op, pArg);
}

static int
walMmapSectorSize(sqlite3_file* pFile)
{
    return wm(pFile)->pUnderlying->pMethods->xSectorSize(wm(pFile)->pUnderlying);
}

static int
walMmapDeviceCharacteristics(sqlite3_file* pFile)
{
    return wm(pFile)->pUnderlying->pMethods->xDeviceCharacteristics(
        wm(pFile)->pUnderlying);
}

static const sqlite3_io_methods walMmapMethods = {
    1,                           // iVersion (v1: no shm/fetch methods needed for WAL)
    walMmapClose,
    walMmapRead,
    walMmapWrite,
    walMmapTruncate,
    walMmapSync,
    walMmapFileSize,
    walMmapLock,
    walMmapUnlock,
    walMmapCheckReservedLock,
    walMmapFileControl,
    walMmapSectorSize,
    walMmapDeviceCharacteristics,
};

// ---------------------------------------------------------------------------
// VFS app data: carries parent VFS pointer + mmap size to xOpen
// ---------------------------------------------------------------------------
struct VfsAppData
{
    sqlite3_vfs* pParent;
    size_t       walMmapSize;
};

// ---------------------------------------------------------------------------
// xOpen: intercept WAL files; pass everything else through
// ---------------------------------------------------------------------------
static int
walMmapOpen(sqlite3_vfs* pVfs,
            const char*  zPath,
            sqlite3_file* pFile,
            int           flags,
            int*          pOutFlags)
{
    auto*         appData    = static_cast<VfsAppData*>(pVfs->pAppData);
    sqlite3_vfs*  pParent    = appData->pParent;
    size_t        mmapSize   = appData->walMmapSize;

    // Underlying file object lives immediately after our WalMmapFile header.
    sqlite3_file* pUnderlying =
        reinterpret_cast<sqlite3_file*>(reinterpret_cast<char*>(pFile) +
                                        sizeof(WalMmapFile));

    // For non-WAL files (database, shm, journal, temp) delegate directly.
    // pFile has enough space because szOsFile >= parent->szOsFile.
    if(!(flags & SQLITE_OPEN_WAL))
    {
        return pParent->xOpen(pParent, zPath, pFile, flags, pOutFlags);
    }

    // --- WAL file: open via parent to get locking/sync, then set up mmap ---

    int rc = pParent->xOpen(pParent, zPath, pUnderlying, flags, pOutFlags);
    if(rc != SQLITE_OK)
        return rc;

    WalMmapFile* p    = reinterpret_cast<WalMmapFile*>(pFile);
    p->pUnderlying    = pUnderlying;
    p->fd             = -1;
    p->pMmap          = MAP_FAILED;
    p->mmapSize       = static_cast<sqlite3_int64>(mmapSize);
    p->logicalSize    = 0;

    // Open a separate fd for ftruncate + mmap.
    p->fd = ::open(zPath, O_RDWR | O_CREAT, 0644);
    if(p->fd < 0)
    {
        pUnderlying->pMethods->xClose(pUnderlying);
        return SQLITE_CANTOPEN;
    }

    // Pre-allocate the file to mmapSize so the mapping covers the full region
    // without needing mremap() as the WAL grows.
    if(::ftruncate(p->fd, static_cast<off_t>(mmapSize)) != 0)
    {
        ::close(p->fd);
        pUnderlying->pMethods->xClose(pUnderlying);
        return SQLITE_IOERR_WRITE;
    }

    p->pMmap = ::mmap(nullptr,
                      mmapSize,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED,
                      p->fd,
                      0);
    if(p->pMmap == MAP_FAILED)
    {
        ::close(p->fd);
        pUnderlying->pMethods->xClose(pUnderlying);
        return SQLITE_IOERR_MMAP;
    }

    pFile->pMethods = &walMmapMethods;
    return SQLITE_OK;
}

// ---------------------------------------------------------------------------
// All other VFS methods delegate to the parent VFS
// ---------------------------------------------------------------------------
static int
walMmapDelete(sqlite3_vfs* pVfs, const char* zPath, int dirSync)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xDelete(d->pParent, zPath, dirSync);
}

static int
walMmapAccess(sqlite3_vfs* pVfs, const char* zPath, int flags, int* pResOut)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xAccess(d->pParent, zPath, flags, pResOut);
}

static int
walMmapFullPathname(sqlite3_vfs* pVfs, const char* zPath, int nOut, char* zOut)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xFullPathname(d->pParent, zPath, nOut, zOut);
}

static void*
walMmapDlOpen(sqlite3_vfs* pVfs, const char* zFilename)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xDlOpen(d->pParent, zFilename);
}

static void
walMmapDlError(sqlite3_vfs* pVfs, int nByte, char* zErrMsg)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    d->pParent->xDlError(d->pParent, nByte, zErrMsg);
}

static void (*walMmapDlSym(sqlite3_vfs* pVfs, void* p, const char* zSym))(void)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xDlSym(d->pParent, p, zSym);
}

static void
walMmapDlClose(sqlite3_vfs* pVfs, void* pHandle)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    d->pParent->xDlClose(d->pParent, pHandle);
}

static int
walMmapRandomness(sqlite3_vfs* pVfs, int nByte, char* zOut)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xRandomness(d->pParent, nByte, zOut);
}

static int
walMmapSleep(sqlite3_vfs* pVfs, int microseconds)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xSleep(d->pParent, microseconds);
}

static int
walMmapCurrentTime(sqlite3_vfs* pVfs, double* pTimeOut)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xCurrentTime(d->pParent, pTimeOut);
}

static int
walMmapGetLastError(sqlite3_vfs* pVfs, int nBuf, char* zBuf)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    return d->pParent->xGetLastError(d->pParent, nBuf, zBuf);
}

static int
walMmapCurrentTimeInt64(sqlite3_vfs* pVfs, sqlite3_int64* pTimeOut)
{
    auto* d = static_cast<VfsAppData*>(pVfs->pAppData);
    // xCurrentTimeInt64 is a v2 method; fall back if parent is v1.
    if(d->pParent->iVersion >= 2 && d->pParent->xCurrentTimeInt64)
        return d->pParent->xCurrentTimeInt64(d->pParent, pTimeOut);
    double t = 0.0;
    int    rc = d->pParent->xCurrentTime(d->pParent, &t);
    *pTimeOut  = static_cast<sqlite3_int64>((t - 2440587.5) * 86400.0 * 1000.0);
    return rc;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
const char*
register_wal_mmap_vfs(size_t mmap_size)
{
    // One registration per distinct mmap_size value.
    static std::mutex              s_mu;
    static std::map<size_t, std::string> s_names;

    std::lock_guard<std::mutex> lock(s_mu);

    auto it = s_names.find(mmap_size);
    if(it != s_names.end())
        return it->second.c_str();

    sqlite3_vfs* parent = sqlite3_vfs_find(nullptr);  // default VFS

    // VfsAppData and sqlite3_vfs must outlive the process; allocate with new.
    auto* appData       = new VfsAppData{ parent, mmap_size };
    auto* vfs           = new sqlite3_vfs{};

    vfs->iVersion          = 2;
    vfs->szOsFile          = static_cast<int>(sizeof(WalMmapFile)) + parent->szOsFile;
    vfs->mxPathname        = parent->mxPathname;
    vfs->pNext             = nullptr;
    vfs->pAppData          = appData;
    vfs->xOpen             = walMmapOpen;
    vfs->xDelete           = walMmapDelete;
    vfs->xAccess           = walMmapAccess;
    vfs->xFullPathname     = walMmapFullPathname;
    vfs->xDlOpen           = walMmapDlOpen;
    vfs->xDlError          = walMmapDlError;
    vfs->xDlSym            = walMmapDlSym;
    vfs->xDlClose          = walMmapDlClose;
    vfs->xRandomness       = walMmapRandomness;
    vfs->xSleep            = walMmapSleep;
    vfs->xCurrentTime      = walMmapCurrentTime;
    vfs->xGetLastError     = walMmapGetLastError;
    vfs->xCurrentTimeInt64 = walMmapCurrentTimeInt64;

    std::string name = "rocpdsna-wal-mmap-" + std::to_string(mmap_size);
    auto [ins, ok]   = s_names.emplace(mmap_size, std::move(name));

    vfs->zName = ins->second.c_str();
    sqlite3_vfs_register(vfs, /* makeDflt= */ 0);

    return ins->second.c_str();
}

}  // namespace rocpdsna::data_storage
