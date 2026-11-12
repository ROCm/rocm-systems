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
! HIP runtime hipMemcpy2D (Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Copies a contiguous MxN matrix host -> device -> host with the native-array
! hipMemcpy2D overload (pitch/width in ELEMENTS) and verifies the round trip.
!!!!!!!!!!!!!!
!
program memcpy2d
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: M = 4, N = 3
  real(c_double), target :: hsrc(M,N), hdst(M,N)
  real(c_double), pointer :: dDev(:,:)
  integer(c_size_t) :: pitch, width, height
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'hip memcpy2d' (Fortran 2008 interfaces) - "

  do j = 1, N
     do i = 1, M
        hsrc(i,j) = real((j-1)*M + i, c_double)
     end do
  end do
  hdst = 0.0d0

  ! With the native overload, pitch/width are counts of elements, height rows.
  pitch  = int(M, c_size_t)
  width  = int(M, c_size_t)
  height = int(N, c_size_t)

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dDev, int(M,c_size_t), int(N,c_size_t)))

  call hipCheck(hipMemcpy2D(dDev, pitch, hsrc, pitch, width, height, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy2D(hdst, pitch, dDev, pitch, width, height, hipMemcpyDeviceToHost))

  do j = 1, N
     do i = 1, M
        if (hdst(i,j) /= hsrc(i,j)) then
           write(*,*) "FAILED! hdst(", i, ",", j, ") = ", hdst(i,j), " expected ", hsrc(i,j)
           call exit(1)
        end if
     end do
  end do

  call hipCheck(hipFree(dDev))

  write(*,*) "PASSED!"

end program memcpy2d
