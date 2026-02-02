# Changelog for rocSHMEM
## Unreleased - rocSHMEM 3.x.x for ROCm 7.x.x
### Added
* Added new APIs:
   * `rocshmem_TYPENAME_alltoall_wg`

## Unreleased -- rocSHMEM 3.2.1 for ROCm x.x.x
### Added
### Changed
### Removed
### Resolved issues
### Known issues

## rocSHMEM 3.2.0 for ROCm 7.2.0
### Added
* Added the GDA conduit for AMD Pensando IONIC
### Changed
* Dependency libraries are loaded dynamically
* The following APIs now have an implementation for the GDA conduit
   * `rocshmem_p`
   * fetching atomics `rochsmem_<TYPE>_fetch_<op>`
   * collective APIs
* The following APIs now have an implementation for the IPC conduit
   * `rocshmem_<TYPE>_atomic_{and,or,xor,swap}`
   * `rocshmem_<TYPE>_atomic_fetch_{and,or,xor,swap}`
### Known issues
* Only 64bit rocSHMEM atomic APIs are implemented for the GDA conduit

## rocSHMEM 3.1.0 for ROCm 7.1.1
### Added
* Allow for IPC, RO, GDA backends to be selected at runtime
* Added the GDA conduit for different NIC vendors
   * Broadcom BNXT\_RE (Thor 2)
   * Mellanox MLX5 (IB and RoCE ConnectX-7)
* Added new APIs:
   * `rocshmem_get_device_ctx`
   * `rocshmem_ctx_pe_quiet`
   * `rocshmem_pe_quiet`

### Changed
* The following APIs have been deprecated:
  * `rocshmem_wg_init`
  * `rocshmem_wg_finalize`
  * `rocshmem_wg_init_thread`
* `rocshmem_ptr`  can now return non-null pointer to
   a shared memory region when the IPC transport is available to reach that region.
   Previously, it would return a null pointer.
* `ROCSHMEM_RO_DISABLE_IPC` was renamed to `ROCSHMEM_DISABLE_MIXED_IPC`.
  This enviroment variable was not documented for prior releases.
  It is now documented to inform users who were using this undocumented feature.

### Removed
* rocSHMEM no-longer requires rocPRIM and rocThrust as dependencies
* Removed MPI compile-time dependency

### Known issues
* Only a subset of rocSHMEM APIs are implemented for the GDA conduit

## rocSHMEM 3.0.0 for ROCm 7.0.0
### Added

* Added the Reverse Offload conduit
* Added new APIs:
   * `rocshmem_ctx_barrier`
   * `rocshmem_ctx_barrier_wave`
   * `rocshmem_ctx_barrier_wg`
   * `rocshmem_barrier_all`
   * `rocshmem_barrier_all_wave`
   * `rocshmem_barrier_all_wg`
   * `rocshmem_ctx_sync`
   * `rocshmem_ctx_sync_wave`
   * `rocshmem_ctx_sync_wg`
   * `rocshmem_sync_all`
   * `rocshmem_sync_all_wave`
   * `rocshmem_sync_all_wg`
   * `rocshmem_init_attr`
   * `rocshmem_get_uniqueid`
   * `rocshmem_set_attr_uniqueid_args`
* Added dlmalloc based allocator
* Added XNACK support
* Added support for initialization with MPI communicators other than `MPI_COMM_WORLD`

### Changed

* Changed collective APIs to use `_wg` suffix rather than `_wg_` infix

### Resolved Issues
* Resolved segfault in `rocshmem_wg_ctx_create`, now provides nullptr if ctx cannot be created

## rocSHMEM 2.0.1 for ROCm 6.4.2

### Resolved Issues

* Resolved incorrect output for `rocshmem_ctx_my_pe` and `rocshmem_ctx_n_pes`
* Resolved multi-team errors by providing team specific buffers in `rocshmem_ctx_wg_team_sync`
* Resolved missing implementation of `rocshmem_g` for IPC conduit
