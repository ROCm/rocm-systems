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
! HIP copies between two GPUs (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Copies a buffer from GPU 0 to GPU 1 with hipMemcpyPeer, and back to a second
! buffer on GPU 0 with hipMemcpyPeerAsync on a stream, checking the data each
! time. Both copies work with or without peer access, which only changes the
! path the data takes. Also checks that hipDeviceGetP2PAttribute agrees with
! hipDeviceCanAccessPeer. Needs two GPUs: with fewer it exits 77, which
! CTest reports as skipped.
!!!!!!!!!!!!!!
!
program peer_copy
  use iso_c_binding
  use hip

  implicit none

  integer, parameter :: N = 1024
  integer(c_size_t), parameter :: Nbytes = int(N, c_size_t) * 8

  real(8), target :: hx(N), hy(N)
  integer(c_int), target :: supported
  integer(c_int) :: ndev, canAccess
  type(c_ptr) :: d0 = c_null_ptr, d1 = c_null_ptr, d0_back = c_null_ptr
  type(c_ptr) :: stream = c_null_ptr
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip peer_copy' (Fortran 2003 interfaces) - "

  call hipCheck(hipGetDeviceCount(ndev))
  if (ndev < 2) then
     write(*,*) "SKIPPED! (needs two GPUs, found ", ndev, ")"
     call exit(77)
  end if

  ! The P2P attribute and the peer query answer the same question.
  call hipCheck(hipDeviceCanAccessPeer(canAccess, 0, 1))
  call hipCheck(hipDeviceGetP2PAttribute(c_loc(supported), hipDevP2PAttrAccessSupported, 0, 1))
  if (supported /= canAccess) then
     write(*,*) "FAILED! hipDevP2PAttrAccessSupported = ", supported, &
                " but hipDeviceCanAccessPeer = ", canAccess
     call exit(1)
  end if

  do i = 1, N
     hx(i) = dble(i)
  end do

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(d0, Nbytes))
  call hipCheck(hipMalloc(d0_back, Nbytes))
  call hipCheck(hipMemcpy(d0, c_loc(hx), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipStreamCreate(stream))

  call hipCheck(hipSetDevice(1))
  call hipCheck(hipMalloc(d1, Nbytes))

  ! GPU 0 -> GPU 1, then read GPU 1.
  call hipCheck(hipMemcpyPeer(d1, 1, d0, 0, Nbytes))
  hy = 0.0d0
  call hipCheck(hipMemcpy(c_loc(hy), d1, Nbytes, hipMemcpyDeviceToHost))
  do i = 1, N
     if (hy(i) /= hx(i)) then
        write(*,*) "FAILED! hipMemcpyPeer: y(", i, ") = ", hy(i), " expected ", hx(i)
        call exit(1)
     end if
  end do

  ! GPU 1 -> a second buffer on GPU 0, asynchronously on a stream of GPU 0.
  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMemcpyPeerAsync(d0_back, 0, d1, 1, Nbytes, stream))
  call hipCheck(hipStreamSynchronize(stream))
  hy = 0.0d0
  call hipCheck(hipMemcpy(c_loc(hy), d0_back, Nbytes, hipMemcpyDeviceToHost))
  do i = 1, N
     if (hy(i) /= hx(i)) then
        write(*,*) "FAILED! hipMemcpyPeerAsync: y(", i, ") = ", hy(i), " expected ", hx(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipStreamDestroy(stream))
  call hipCheck(hipFree(d0))
  call hipCheck(hipFree(d0_back))
  call hipCheck(hipSetDevice(1))
  call hipCheck(hipFree(d1))

  write(*,*) "PASSED!"

end program peer_copy
