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
! HIP runtime device-management queries
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipGetDeviceCount, hipSetDevice/hipGetDevice, hipDeviceGetAttribute,
! hipDeviceGetLimit, hipDeviceTotalMem and hipMemGetInfo, checking basic
! invariants that hold on any working device.
!!!!!!!!!!!!!!
!
program device_management
  use iso_c_binding
  use hip

  implicit none

  integer(c_int)    :: ndev, dev, warp, nmp
  integer(c_size_t) :: totmem, freemem, total2, stacklimit

  write(*,"(a)",advance="no") "-- Running test 'hip device_management' (Fortran 2003 interfaces) - "

  call hipCheck(hipGetDeviceCount(ndev))
  if (ndev < 1) then
     write(*,*) "FAILED! device count = ", ndev
     call exit(1)
  end if

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipGetDevice(dev))
  if (dev /= 0) then
     write(*,*) "FAILED! current device = ", dev, " (expected 0)"
     call exit(1)
  end if

  call hipCheck(hipDeviceGetAttribute(warp, hipDeviceAttributeWarpSize, 0))
  if (warp <= 0) then
     write(*,*) "FAILED! warp size = ", warp
     call exit(1)
  end if

  call hipCheck(hipDeviceGetAttribute(nmp, hipDeviceAttributeMultiprocessorCount, 0))
  if (nmp <= 0) then
     write(*,*) "FAILED! multiprocessor count = ", nmp
     call exit(1)
  end if

  call hipCheck(hipDeviceGetLimit(stacklimit, hipLimitStackSize))
  if (stacklimit <= 0) then
     write(*,*) "FAILED! stack size limit = ", stacklimit
     call exit(1)
  end if

  call hipCheck(hipDeviceTotalMem(totmem, 0))
  if (totmem <= 0) then
     write(*,*) "FAILED! total device memory = ", totmem
     call exit(1)
  end if

  call hipCheck(hipMemGetInfo(freemem, total2))
  if (total2 <= 0 .or. freemem > total2) then
     write(*,*) "FAILED! memGetInfo free = ", freemem, " total = ", total2
     call exit(1)
  end if

  write(*,*) "PASSED!"

end program device_management
