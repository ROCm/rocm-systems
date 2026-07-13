# Changelog for rocSHMEM
## Unreleased - rocSHMEM 3.6.0 for ROCm x.x.x
## rocSHMEM 3.5.0 for ROCm 7.14

### Added
* Added new APIs:
   * `rocshmem_align`
   * `rocshmem_calloc`
   * `rocshmem_buffer_unregister_all`
   * `rocshmem_buffer_register/unregister` for GDA backend
   * `rocshmem_buffer_register_symmetric` for IPC backend
   * `rocshmem_buffer_unregister_symmetric` for IPC backend
   * `rocshmem_reduce_on_stream`
   * `rocshmem_team_split_2D`
* Added tile-granular RMA operations for the IPC backend
* Added host-initiated RMA operations in the IPC backend for the non-MPI
   bootstrapping path: put, get, fence, quiet, arithmetic AMOs, and P2P sync ops
* Added team creation using non-contiguous parent teams in the IPC backend
* Added Python bindings of memory-management APIs
* Added Python bindings coverage for team APIs
* Added support for GPU initiated operations using the SDMA engines
* Added ASAN build support

### Changed
* Changed default `ROCSHMEM_DEBUG_LEVEL` from `WARN` to `ERROR`
* Performance optimizations:
   * Separated put/get memcpy primitives to apply correct cache coherence semantics and fences
   * Use constmem for backend variables and provider muxing
   * Updated O(1) IPC availability check using pattern detection

## rocSHMEM 3.4.0 for ROCm 7.13
### Added
* Added new APIs:
   * `rocshmem_quiet_on_stream`
   * `rocshmem_sync_all_on_stream`
   * `rocshmem_TYPENAME_alltoall_wg`
   * `rocshmem_TYPENAME_alltoallv_wg`
   * `rocshmem_team_my_pe`
   * `rocshmem_team_n_pes`
   * `rocshmem_barrier`
   * `rocshmem_barrier_wave`
   * `rocshmem_barrier_wg`
   * `rocshmem_buffer_register`
   * `rocshmem_buffer_unregister`
   * `rocshmem_info_get_version`
   * `rocshmem_info_get_name`
   * `rocshmem_vendor_get_version_info`
* Added library constants: `ROCSHMEM_MAJOR_VERSION`, `ROCSHMEM_MINOR_VERSION`,
  `ROCSHMEM_MAX_NAME_LEN`, `ROCSHMEM_VENDOR_STRING`, `ROCSHMEM_VERSION`,
  `ROCSHMEM_VENDOR_MAJOR_VERSION`, `ROCSHMEM_VENDOR_MINOR_VERSION`,
  `ROCSHMEM_VENDOR_PATCH_VERSION`
* Added vendor string and backend metadata to the `rocshmem_info` output
* Added `ROCSHMEM_TEAM_WORLD` for device code
* Added `ROCSHMEM_TEAM_SHARED` predefined team for PEs sharing a common memory domain (same node)
* Added new environment variables:
  * `OVERRIDE_NIC_FIRMWARE_CHECK`
  * `ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX`
  * `ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX`
  * `ROCSHMEM_MAX_SYMM_REGIONS`
* Added VMM POSIX memory allocator (`USE_HEAP_DEVICE_VMM_POSIX`)
   * Uses HIP Virtual Memory Management (VMM) APIs for fine-grained memory control
   * Requires ROCm 7.0+ and Linux kernel 5.6+
   * Not compatible with MPI-based initialization (use `ROCSHMEM_INIT_WITH_UNIQUEID` instead)
### Changed
* Use CQ collapsing for the Mellanox MLX5 GDA conduit

## rocSHMEM 3.2.1 for ROCm 7.2.1
### Added
* Warn if large BAR is not available
### Resolved Issues
* GDA Backend will disable itself when no GDA compatible NICs are available rather than crashing
* Fix memory coherency issues on gfx1201
### Known issues
* Only 64bit rocSHMEM atomic APIs are implemented for the GDA conduit

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
  This environment variable was not documented for prior releases.
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
