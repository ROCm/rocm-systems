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
! HIP runtime hipMallocManaged (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Allocates unified (managed) memory, writes it from the host, overwrites it
! from the device (hipMemset), and checks the host sees the device's write.
!!!!!!!!!!!!!!
!
program malloc_managed
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  type(c_ptr) :: mptr = c_null_ptr
  real(c_double), pointer :: p(:)
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip malloc_managed' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMallocManaged(mptr, nbytes, hipMemAttachGlobal))
  call c_f_pointer(mptr, p, [n])

  ! Host writes the managed buffer.
  do i = 1, n
     p(i) = real(i, c_double)
  end do

  ! Device overwrites it with zero bytes.
  call hipCheck(hipMemset(mptr, 0, nbytes))
  call hipCheck(hipDeviceSynchronize())

  ! Host must observe the device's write.
  do i = 1, n
     if (p(i) /= 0.0d0) then
        write(*,*) "FAILED! p(", i, ") = ", p(i), " (expected 0)"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(mptr))

  write(*,*) "PASSED!"

end program malloc_managed
