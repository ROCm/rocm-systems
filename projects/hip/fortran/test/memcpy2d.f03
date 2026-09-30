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
! HIP runtime hipMemcpy2D (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Copies a contiguous MxN column-major matrix host -> device -> host with the
! raw type(c_ptr) hipMemcpy2D (pitch/width in BYTES) and verifies the round trip.
!!!!!!!!!!!!!!
!
program memcpy2d
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: M = 4, N = 3
  real(c_double), target :: hsrc(M,N), hdst(M,N)
  type(c_ptr) :: dptr = c_null_ptr
  integer(c_size_t) :: pitch, width, height
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'hip memcpy2d' (Fortran 2003 interfaces) - "

  do j = 1, N
     do i = 1, M
        hsrc(i,j) = real((j-1)*M + i, c_double)
     end do
  end do
  hdst = 0.0d0

  ! Contiguous device buffer: row pitch equals the column length in bytes.
  pitch  = int(M, c_size_t) * 8
  width  = int(M, c_size_t) * 8   ! bytes copied per "row" (a full column here)
  height = int(N, c_size_t)       ! number of rows (columns here)

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dptr, int(M,c_size_t) * int(N,c_size_t) * 8))

  call hipCheck(hipMemcpy2D(dptr, pitch, c_loc(hsrc(1,1)), pitch, width, height, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy2D(c_loc(hdst(1,1)), pitch, dptr, pitch, width, height, hipMemcpyDeviceToHost))

  do j = 1, N
     do i = 1, M
        if (hdst(i,j) /= hsrc(i,j)) then
           write(*,*) "FAILED! hdst(", i, ",", j, ") = ", hdst(i,j), " expected ", hsrc(i,j)
           call exit(1)
        end if
     end do
  end do

  call hipCheck(hipFree(dptr))

  write(*,*) "PASSED!"

end program memcpy2d
