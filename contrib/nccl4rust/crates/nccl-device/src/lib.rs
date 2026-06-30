// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-style, `no_std` wrappers around [`nccl_device_sys`].
//!
//! The wrapper does not own NCCL host state or CUDA memory. A [`DevComm`] must
//! point at a device-resident copy of the public communicator produced by
//! `ncclDevCommCreate`, and a [`Window`] remains owned by its host communicator.
//! These functions are intended to be inlined into CUDA-Oxide device kernels;
//! they are not CPU-callable host wrappers.

#![no_std]

use core::ffi::c_void;
use core::marker::PhantomData;

pub use nccl_device_sys as sys;

/// Borrowed device pointer to a public NCCL device communicator.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct DevComm<'a> {
    raw: *const sys::ncclDevComm_t,
    _lifetime: PhantomData<&'a sys::ncclDevComm_t>,
}

impl<'a> DevComm<'a> {
    /// Borrow a device-resident `ncclDevComm_t`.
    ///
    /// # Safety
    ///
    /// `raw` must point to a live, correctly aligned device copy produced by a
    /// compatible NCCL runtime, and that allocation must outlive `'a`.
    pub const unsafe fn from_raw(raw: *const sys::ncclDevComm_t) -> Self {
        Self {
            raw,
            _lifetime: PhantomData,
        }
    }

    pub const fn as_raw(self) -> *const sys::ncclDevComm_t {
        self.raw
    }
}

/// A non-owning NCCL registered-window handle.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Window<'a> {
    raw: sys::ncclWindow_t,
    _lifetime: PhantomData<&'a c_void>,
}

impl<'a> Window<'a> {
    /// Borrow a window registered on the host.
    ///
    /// # Safety
    ///
    /// `raw` must remain registered for `'a`, and all requested byte ranges
    /// must be inside that registration.
    pub const unsafe fn from_raw(raw: sys::ncclWindow_t) -> Self {
        Self {
            raw,
            _lifetime: PhantomData,
        }
    }

    pub const fn as_raw(self) -> sys::ncclWindow_t {
        self.raw
    }
}

/// A non-owning multimem mapping associated with an NCCL device communicator.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultimemHandle<'a> {
    raw: sys::ncclMultimemHandle_t,
    _lifetime: PhantomData<&'a sys::ncclDevComm_t>,
}

impl<'a> MultimemHandle<'a> {
    /// Borrow a multimem handle populated by `ncclDevCommCreate`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid handle produced by a compatible NCCL runtime,
    /// and its parent device communicator and resources must outlive `'a`.
    pub const unsafe fn from_raw(raw: sys::ncclMultimemHandle_t) -> Self {
        Self {
            raw,
            _lifetime: PhantomData,
        }
    }

    pub const fn as_raw(self) -> sys::ncclMultimemHandle_t {
        self.raw
    }
}

/// Value-type NCCL team descriptor.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Team(sys::ncclTeam_t);

impl Team {
    pub const fn size(self) -> i32 {
        self.0.n_ranks
    }

    pub const fn rank(self) -> i32 {
        self.0.rank
    }

    pub const fn stride(self) -> i32 {
        self.0.stride
    }

    pub const fn as_raw(self) -> sys::ncclTeam_t {
        self.0
    }
}

#[inline(always)]
pub fn rank(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_rank(comm.as_raw()) }
}

#[inline(always)]
pub fn size(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_n_ranks(comm.as_raw()) }
}

#[inline(always)]
pub fn lsa_rank(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_lsa_rank(comm.as_raw()) }
}

#[inline(always)]
pub fn lsa_size(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_lsa_size(comm.as_raw()) }
}

#[inline(always)]
pub fn world(comm: DevComm<'_>) -> Team {
    let mut n_ranks = 0;
    let mut rank = 0;
    let mut stride = 0;
    team_world_device(comm, &mut n_ranks, &mut rank, &mut stride);
    Team(sys::ncclTeam_t {
        n_ranks,
        rank,
        stride,
    })
}

#[inline(always)]
fn team_world_device(comm: DevComm<'_>, n_ranks: *mut i32, rank: *mut i32, stride: *mut i32) {
    unsafe { sys::nccl4rust_team_world(comm.as_raw(), n_ranks, rank, stride) }
}

#[inline(always)]
pub fn lsa(comm: DevComm<'_>) -> Team {
    let mut n_ranks = 0;
    let mut rank = 0;
    let mut stride = 0;
    team_lsa_device(comm, &mut n_ranks, &mut rank, &mut stride);
    Team(sys::ncclTeam_t {
        n_ranks,
        rank,
        stride,
    })
}

#[inline(always)]
fn team_lsa_device(comm: DevComm<'_>, n_ranks: *mut i32, rank: *mut i32, stride: *mut i32) {
    unsafe { sys::nccl4rust_team_lsa(comm.as_raw(), n_ranks, rank, stride) }
}

#[inline(always)]
pub fn rail(comm: DevComm<'_>) -> Team {
    let mut n_ranks = 0;
    let mut rank = 0;
    let mut stride = 0;
    team_rail_device(comm, &mut n_ranks, &mut rank, &mut stride);
    Team(sys::ncclTeam_t {
        n_ranks,
        rank,
        stride,
    })
}

#[inline(always)]
fn team_rail_device(comm: DevComm<'_>, n_ranks: *mut i32, rank: *mut i32, stride: *mut i32) {
    unsafe { sys::nccl4rust_team_rail(comm.as_raw(), n_ranks, rank, stride) }
}

/// Translate a rank in `team` to its world rank.
///
/// Returns `None` when `rank` is outside the team.
#[inline(always)]
pub fn rank_to_world(comm: DevComm<'_>, team: Team, rank: i32) -> Option<i32> {
    if rank < 0 || rank >= team.size() {
        None
    } else {
        Some(rank_to_world_device(
            comm,
            team.size(),
            team.rank(),
            team.stride(),
            rank,
        ))
    }
}

#[inline(always)]
fn rank_to_world_device(
    comm: DevComm<'_>,
    team_n_ranks: i32,
    team_rank: i32,
    team_stride: i32,
    rank: i32,
) -> i32 {
    unsafe {
        sys::nccl4rust_team_rank_to_world(comm.as_raw(), team_n_ranks, team_rank, team_stride, rank)
    }
}

/// Translate a rank in `team` to its LSA rank.
///
/// Returns `None` when `rank` is outside the team.
#[inline(always)]
pub fn rank_to_lsa(comm: DevComm<'_>, team: Team, rank: i32) -> Option<i32> {
    if rank < 0 || rank >= team.size() {
        None
    } else {
        Some(rank_to_lsa_device(
            comm,
            team.size(),
            team.rank(),
            team.stride(),
            rank,
        ))
    }
}

#[inline(always)]
fn rank_to_lsa_device(
    comm: DevComm<'_>,
    team_n_ranks: i32,
    team_rank: i32,
    team_stride: i32,
    rank: i32,
) -> i32 {
    unsafe {
        sys::nccl4rust_team_rank_to_lsa(comm.as_raw(), team_n_ranks, team_rank, team_stride, rank)
    }
}

#[inline(always)]
fn local_ptr_device(window: Window<'_>, byte_offset: usize) -> *mut c_void {
    unsafe { sys::nccl4rust_get_local_pointer(window.as_raw(), byte_offset) }
}

/// Translate a registered window offset to this rank's local pointer.
///
/// # Safety
///
/// `byte_offset` and the resulting `T` access must be in bounds and correctly
/// aligned for the registered window. The caller must also uphold all aliasing
/// and synchronization requirements for accesses through the returned pointer.
#[inline(always)]
pub unsafe fn local_ptr<T>(window: Window<'_>, byte_offset: usize) -> *mut T {
    local_ptr_device(window, byte_offset).cast()
}

#[inline(always)]
fn lsa_ptr_device(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut c_void {
    unsafe { sys::nccl4rust_get_lsa_pointer(window.as_raw(), byte_offset, peer) }
}

/// Translate a registered window offset for an LSA-local peer.
///
/// # Safety
///
/// `peer` must name a member of the window's LSA team. `byte_offset` and the
/// resulting `T` access must be in bounds and correctly aligned, and the
/// caller must uphold all aliasing and synchronization requirements.
#[inline(always)]
pub unsafe fn lsa_ptr<T>(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut T {
    lsa_ptr_device(window, byte_offset, peer).cast()
}

#[inline(always)]
fn peer_ptr_device(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut c_void {
    unsafe { sys::nccl4rust_get_peer_pointer(window.as_raw(), byte_offset, peer) }
}

/// Translate a registered window offset for a world peer.
///
/// # Safety
///
/// `peer` must be LSA-accessible from this rank. `byte_offset` and the
/// resulting `T` access must be in bounds and correctly aligned, and the
/// caller must uphold all aliasing and synchronization requirements.
#[inline(always)]
pub unsafe fn peer_ptr<T>(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut T {
    peer_ptr_device(window, byte_offset, peer).cast()
}

#[inline(always)]
fn team_peer_ptr_device(
    window: Window<'_>,
    byte_offset: usize,
    team_n_ranks: i32,
    team_rank: i32,
    team_stride: i32,
    peer: i32,
) -> *mut c_void {
    unsafe {
        sys::nccl4rust_get_peer_pointer_team(
            window.as_raw(),
            byte_offset,
            team_n_ranks,
            team_rank,
            team_stride,
            peer,
        )
    }
}

/// Translate a registered window offset for a peer in `team`.
///
/// # Safety
///
/// `team` and `peer` must describe an LSA-accessible peer for this window.
/// `byte_offset` and the resulting `T` access must be in bounds and correctly
/// aligned, and the caller must uphold all aliasing and synchronization
/// requirements.
#[inline(always)]
pub unsafe fn team_peer_ptr<T>(
    window: Window<'_>,
    byte_offset: usize,
    team: Team,
    peer: i32,
) -> *mut T {
    team_peer_ptr_device(
        window,
        byte_offset,
        team.size(),
        team.rank(),
        team.stride(),
        peer,
    )
    .cast()
}

#[inline(always)]
fn multimem_ptr_device(
    window: Window<'_>,
    byte_offset: usize,
    handle: MultimemHandle<'_>,
) -> *mut c_void {
    unsafe {
        sys::nccl4rust_get_multimem_pointer(
            window.as_raw(),
            byte_offset,
            handle.as_raw().mc_base_ptr,
        )
    }
}

/// Translate a registered window offset through a multimem team mapping.
///
/// # Safety
///
/// `handle` must be compatible with `window`. `byte_offset` and the resulting
/// `T` access must be in bounds and correctly aligned, the selected system
/// must support multimem, and the caller must uphold all aliasing and
/// synchronization requirements.
#[inline(always)]
pub unsafe fn multimem_ptr<T>(
    window: Window<'_>,
    byte_offset: usize,
    handle: MultimemHandle<'_>,
) -> *mut T {
    multimem_ptr_device(window, byte_offset, handle).cast()
}

#[inline(always)]
fn lsa_multimem_ptr_device(
    window: Window<'_>,
    byte_offset: usize,
    comm: DevComm<'_>,
) -> *mut c_void {
    unsafe { sys::nccl4rust_get_lsa_multimem_pointer(window.as_raw(), byte_offset, comm.as_raw()) }
}

/// Translate a registered window offset through the communicator's LSA
/// multimem mapping.
///
/// # Safety
///
/// LSA multimem must have been enabled when `comm` was created, and `window`
/// must be registered with its parent host communicator. `byte_offset` and the
/// resulting `T` access must be in bounds and correctly aligned, and the
/// caller must uphold all aliasing and synchronization requirements.
#[inline(always)]
pub unsafe fn lsa_multimem_ptr<T>(
    window: Window<'_>,
    byte_offset: usize,
    comm: DevComm<'_>,
) -> *mut T {
    lsa_multimem_ptr_device(window, byte_offset, comm).cast()
}

/// Synchronize one participating thread per rank in the LSA team.
///
/// # Safety
///
/// Every rank in the LSA team must execute the same barrier index, and the
/// communicator must have reserved that index in `lsaBarrierCount`.
#[inline(always)]
pub unsafe fn lsa_barrier_thread(comm: DevComm<'_>, index: u32, multimem: bool) {
    lsa_barrier_thread_device(comm, index, multimem)
}

#[inline(always)]
fn lsa_barrier_thread_device(comm: DevComm<'_>, index: u32, multimem: bool) {
    unsafe { sys::nccl4rust_lsa_barrier_thread(comm.as_raw(), index, multimem) }
}

/// Synchronize a full warp on every rank in the LSA team.
///
/// # Safety
///
/// All lanes in each participating warp and every LSA rank must execute this
/// call convergently with the same reserved barrier index.
#[inline(always)]
pub unsafe fn lsa_barrier_warp(comm: DevComm<'_>, index: u32, multimem: bool) {
    lsa_barrier_warp_device(comm, index, multimem)
}

#[inline(always)]
fn lsa_barrier_warp_device(comm: DevComm<'_>, index: u32, multimem: bool) {
    unsafe { sys::nccl4rust_lsa_barrier_warp(comm.as_raw(), index, multimem) }
}

/// Synchronize a full CTA on every rank in the LSA team.
///
/// # Safety
///
/// All threads in each CTA and every LSA rank must execute this call
/// convergently with the same reserved barrier index.
#[inline(always)]
pub unsafe fn lsa_barrier_cta(comm: DevComm<'_>, index: u32, multimem: bool) {
    lsa_barrier_cta_device(comm, index, multimem)
}

#[inline(always)]
fn lsa_barrier_cta_device(comm: DevComm<'_>, index: u32, multimem: bool) {
    unsafe { sys::nccl4rust_lsa_barrier_cta(comm.as_raw(), index, multimem) }
}

/// Reduce f32 values from every LSA peer and copy the sum to every peer.
///
/// # Safety
///
/// All CTA threads must participate convergently. Source and destination
/// ranges must be valid `count * size_of::<f32>()` ranges in their windows.
/// Callers must provide the entry/exit ordering required by NCCL, normally
/// with [`lsa_barrier_cta`] around this primitive.
#[inline(always)]
pub unsafe fn lsa_reduce_sum_copy_f32_cta(
    comm: DevComm<'_>,
    src: Window<'_>,
    src_byte_offset: usize,
    dst: Window<'_>,
    dst_byte_offset: usize,
    count: usize,
) {
    lsa_reduce_sum_copy_f32_cta_device(comm, src, src_byte_offset, dst, dst_byte_offset, count)
}

#[inline(always)]
fn lsa_reduce_sum_copy_f32_cta_device(
    comm: DevComm<'_>,
    src: Window<'_>,
    src_byte_offset: usize,
    dst: Window<'_>,
    dst_byte_offset: usize,
    count: usize,
) {
    unsafe {
        sys::nccl4rust_lsa_reduce_sum_copy_f32_cta(
            comm.as_raw(),
            src.as_raw(),
            src_byte_offset,
            dst.as_raw(),
            dst_byte_offset,
            count,
        )
    }
}
