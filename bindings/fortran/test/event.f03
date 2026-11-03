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
! HIP runtime event timing
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipEventCreate, hipEventRecord, hipEventSynchronize,
! hipEventElapsedTime and hipEventDestroy by timing a device memset on the
! default stream.
!!!!!!!!!!!!!!
!
program event
  use iso_c_binding
  use hip

  implicit none

  type(c_ptr) :: estart = c_null_ptr, estop = c_null_ptr, dptr = c_null_ptr
  real(c_float) :: ms
  integer(c_size_t), parameter :: nbytes = 4 * 1024 * 1024

  write(*,"(a)",advance="no") "-- Running test 'hip event' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dptr, nbytes))

  call hipCheck(hipEventCreate(estart))
  call hipCheck(hipEventCreate(estop))

  ! Record around some device work on the default stream (stream = c_null_ptr).
  call hipCheck(hipEventRecord(estart, c_null_ptr))
  call hipCheck(hipMemset(dptr, 0, nbytes))
  call hipCheck(hipEventRecord(estop, c_null_ptr))

  call hipCheck(hipEventSynchronize(estop))
  call hipCheck(hipEventElapsedTime(ms, estart, estop))

  if (.not. (ms >= 0.0)) then
     write(*,*) "FAILED! elapsed time = ", ms, " ms (expected >= 0)"
     call exit(1)
  end if

  call hipCheck(hipEventDestroy(estart))
  call hipCheck(hipEventDestroy(estop))
  call hipCheck(hipFree(dptr))

  write(*,*) "PASSED!"

end program event
