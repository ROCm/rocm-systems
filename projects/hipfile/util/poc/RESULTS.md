# hipFile synchronous I/O metadata POC

This POC compares `pread`/`pwrite` with `hipFileRead`/`hipFileWrite` for
zero-sized and aligned, normal-sized I/O. Each API operates on a separately
seeded file with identical contents and metadata. The program captures
`fstat` and `statx` immediately before and after the operation, then captures
them again after the file has been closed and reopened.

The shell driver builds hipFile, the POC, and `ais-stats`, then runs the matrix
twice:

- fastpath only: `HIPFILE_ALLOW_COMPAT_MODE=false`
- forced fallback: `HIPFILE_ALLOW_COMPAT_MODE=true` and
  `HIPFILE_FORCE_COMPAT_MODE=true`

Run it on an AIS-capable filesystem with:

```console
./util/poc/compare-sync-io-file-stats.sh AIS_DIRECTORY [GPU_ID]
```

## Test matrix

The POC uses a 1 MiB seeded file and the direct-I/O alignment reported by
`statx` (4 KiB I/O in the run below):

| Scenario | Size | Offset |
| --- | ---: | ---: |
| zero-sized read in range | 0 | 0 |
| zero-sized read beyond EOF | 0 | 2 MiB |
| normal read in range | 4 KiB | 0 |
| zero-sized write in range | 0 | 0 |
| zero-sized write beyond EOF | 0 | 2 MiB |
| normal overwrite | 4 KiB | 0 |
| normal extending write | 4 KiB | 1 MiB |

The comparisons cover return values, file-content hashes, read-buffer hashes,
and the following metadata:

- `fstat`: device, inode, mode, link count, UID, GID, special-device ID,
  size, preferred block size, allocated blocks, and access/modification/change
  timestamps.
- `statx`: result mask, preferred block size, attributes and attribute mask,
  link count, UID, GID, mode, inode, size, allocated blocks,
  access/birth/change/modification timestamps, device IDs, and direct-I/O
  alignment when supported by the build headers.

Structural fields use exact before/after comparisons. Inode and timestamp
values differ naturally between the two independently created files, so those
are compared by whether each API changed the field.

## Results from cgy-rowlet

Rerun on 2026-08-19 as user `johtyler`, GPU 0:

```console
./util/poc/compare-sync-io-file-stats.sh /mnt/ais/ext4/johtyler
./util/poc/compare-sync-io-file-stats.sh /mnt/ais/xfs/johtyler
```

The filesystems were `/dev/nvme1n1p1` (ext4, `relatime`) and
`/dev/nvme1n1p2` (XFS, `relatime`). `ais-check` reported both as AIS capable.

| Filesystem | hipFile backend | Result |
| --- | --- | --- |
| ext4 | fastpath only | 7/7 passed |
| ext4 | forced fallback | 7/7 passed |
| XFS | fastpath only | 6/7 passed |
| XFS | forced fallback | 7/7 passed |

`ais-stats` confirmed that the backend selection was effective:

| Run mode | Fastpath rejection count | Fastpath bytes (read/write) | Fallback bytes (read/write) |
| --- | ---: | ---: | ---: |
| fastpath only | 0 | 4096 / 8192 | 0 / 0 |
| forced fallback | 7 | 0 / 0 | 4096 / 8192 |

Zero-sized operations are represented by the rejection count but do not add
to the byte totals.

The metadata effects were consistent across ext4 and XFS and across fastpath
and fallback, apart from the transient XFS allocation difference described
below:

- Both zero-sized reads and writes returned zero and changed neither file
  contents nor any captured metadata, including at an aligned offset beyond
  EOF.
- A normal read changed only `atime`; file contents and all other captured
  metadata remained unchanged. The POSIX and hipFile read-buffer hashes
  matched.
- A normal overwrite changed `mtime` and `ctime`; size and allocated blocks
  remained unchanged and the resulting file-content hashes matched.
- An extending write changed size from 1,048,576 to 1,052,672 bytes, allocated
  blocks, `mtime`, and `ctime`; the resulting file-content hashes matched.
- Mode (`0100640`), link count, UID/GID, device IDs, preferred block size,
  attributes, birth time, and direct-I/O alignment remained unchanged in
  every case.

### XFS fastpath observation

Only the immediate allocated-block count for the fastpath extending write
differed:

```text
                         before  immediate after  after close/reopen
POSIX pwrite st_blocks      2048             6144                2056
hipFileWrite st_blocks      2048             2056                2056
```

Both `fstat.st_blocks` and `statx.stx_blocks` reported this difference. XFS's
buffered POSIX write exposed a transient speculative preallocation, whereas
the hipFile direct fastpath exposed the final allocation immediately. After
close/reopen, both APIs reported 2056 blocks and all other metadata and content
comparisons matched. Forced fallback showed the same immediate 6144-block
allocation for POSIX and hipFile, followed by 2056 blocks after close/reopen.

This makes the XFS fastpath result an observation about when allocation state
is visible rather than a persistent file-property difference.

## POC limitation

This is a POSIX-parity POC, not an independent metadata allowlist. For most
structural properties it proves that POSIX and hipFile have identical
before/after transitions. If both APIs unexpectedly changed a property such
as mode in exactly the same way, the comparison would still pass. Production
integration tests should additionally define the permitted changes for each
operation and assert that every other captured property remains unchanged.
