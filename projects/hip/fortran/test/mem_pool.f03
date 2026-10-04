!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!
! HIP runtime stream-ordered memory pools (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipDeviceGetDefaultMemPool, hipMemPoolGetAccess and the
! stream-ordered allocator (hipMallocAsync / hipFreeAsync) that draws from the
! pool.
!
! hipMemPoolGetAccess returns its hipMemAccessFlags through the FIRST argument,
! by reference. The device that owns a pool always has read/write access to it,
! so unlike the cache-config getters this one has a single exact expected value
! and the test asserts it: a binding that passed the enum by value (or wrote
! through a stale address) would leave the -1 sentinel in place and fail here.
!
! The allocation round-trip that follows proves the queried pool is the one the
! allocator really uses: 1 MiB is taken from it on a stream, filled with a known
! pattern, copied back and verified before being returned with hipFreeAsync.
!!!!!!!!!!!!!!
!
program mem_pool
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 262144      ! 1 MiB of int32
  integer(c_int), target :: hbuf(n)
  type(c_ptr) :: pool = c_null_ptr
  type(c_ptr) :: stream = c_null_ptr
  type(c_ptr) :: dptr = c_null_ptr
  type(hipMemLocation) :: location
  integer(kind(hipMemAccessFlagsProtNone)) :: flags
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") &
    "-- Running test 'hip mem_pool' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 4

  call hipCheck(hipSetDevice(0))

  ! 1. Every device exposes a default pool; it must be a real handle.
  call hipCheck(hipDeviceGetDefaultMemPool(pool, 0))
  if (.not. c_associated(pool)) then
     write(*,*) "FAILED! hipDeviceGetDefaultMemPool returned a null pool"
     call exit(1)
  end if

  ! 2. Device 0 owns that pool, so it must report read/write access to it.
  location%type = hipMemLocationTypeDevice
  location%id   = 0
  flags = -1
  call hipCheck(hipMemPoolGetAccess(flags, pool, location))
  if (flags /= hipMemAccessFlagsProtReadWrite) then
     write(*,*) "FAILED! hipMemPoolGetAccess reported ", flags, &
                " for the owning device (expected hipMemAccessFlagsProtReadWrite = ", &
                hipMemAccessFlagsProtReadWrite, ")"
     call exit(1)
  end if

  ! 3. Allocate out of the pool on a stream and round-trip a known pattern.
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipMallocAsync(dptr, nbytes, stream))
  if (.not. c_associated(dptr)) then
     write(*,*) "FAILED! hipMallocAsync returned a null pointer"
     call exit(1)
  end if

  do i = 1, n
     hbuf(i) = i
  end do
  call hipCheck(hipMemcpyAsync(dptr, c_loc(hbuf(1)), nbytes, hipMemcpyHostToDevice, stream))

  hbuf = 0
  call hipCheck(hipMemcpyAsync(c_loc(hbuf(1)), dptr, nbytes, hipMemcpyDeviceToHost, stream))
  call hipCheck(hipStreamSynchronize(stream))

  do i = 1, n
     if (hbuf(i) /= i) then
        write(*,*) "FAILED! hbuf(", i, ") = ", hbuf(i), " (expected ", i, ")"
        call exit(1)
     end if
  end do

  call hipCheck(hipFreeAsync(dptr, stream))
  call hipCheck(hipStreamSynchronize(stream))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED! default pool access flags = ", flags

end program mem_pool
