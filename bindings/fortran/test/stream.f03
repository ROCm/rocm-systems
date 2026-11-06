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
! HIP runtime stream management
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipStreamCreate, hipStreamGetDevice, hipStreamSynchronize and
! hipStreamDestroy.
!!!!!!!!!!!!!!
!
program test_stream
  use iso_c_binding
  use hip

  implicit none

  type(c_ptr)    :: stream = c_null_ptr
  integer(c_int) :: dev, sdev

  write(*,"(a)",advance="no") "-- Running test 'hip stream' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipGetDevice(dev))

  call hipCheck(hipStreamCreate(stream))
  if (.not. c_associated(stream)) then
     write(*,*) "FAILED! stream is null after hipStreamCreate"
     call exit(1)
  end if

  call hipCheck(hipStreamGetDevice(stream, sdev))
  if (sdev /= dev) then
     write(*,*) "FAILED! stream device = ", sdev, " (expected ", dev, ")"
     call exit(1)
  end if

  call hipCheck(hipStreamSynchronize(stream))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program test_stream
