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
! HIP runtime hipMemset
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Sets every byte of a device buffer to a known value and copies it back to
! verify.
!!!!!!!!!!!!!!
!
program memset
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  integer(c_int8_t), target :: hbuf(n)
  type(c_ptr) :: dptr = c_null_ptr
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip memset' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t)   ! one byte per element

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dptr, nbytes))

  ! Set every byte to 7.
  call hipCheck(hipMemset(dptr, 7, nbytes))

  hbuf = 0
  call hipCheck(hipMemcpy(c_loc(hbuf(1)), dptr, nbytes, hipMemcpyDeviceToHost))

  do i = 1, n
     if (hbuf(i) /= 7_c_int8_t) then
        write(*,*) "FAILED! hbuf(", i, ") = ", hbuf(i), " (expected 7)"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dptr))

  write(*,*) "PASSED!"

end program memset
