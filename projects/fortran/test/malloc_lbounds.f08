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
! HIP runtime hipFree/hipHostFree on arrays with non-default lower bounds
! (Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Allocates device memory with hipMalloc(dims=, lbounds=) and pinned host memory
! with hipHostMalloc, remaps the pinned array to lower bounds of zero, then
! frees both. hipFree/hipHostFree must pass the base address of the allocation,
! not the address of element (1,1). ROCm happens to accept an address inside
! the allocation as well, so on AMD this test cannot catch a wrong free; it
! guards the bounds, the round trip, and that both frees succeed.
!!!!!!!!!!!!!!
!
program malloc_lbounds
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: m = 16, n = 8
  real(c_double), pointer :: dp(:,:)      ! device memory, lbounds (0,0)
  real(c_double), pointer :: hp(:,:)      ! pinned host memory
  real(c_double), pointer :: hp0(:,:)     ! hp remapped to lbounds (0,0)
  real(c_double), target  :: hcheck(0:m-1,0:n-1)
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'hip malloc_lbounds' (Fortran 2008 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dp, dims=[m,n], lbounds=[0,0]))
  if (any(lbound(dp) /= [0,0]) .or. any(ubound(dp) /= [m-1,n-1])) then
     write(*,*) "FAILED! hipMalloc bounds ", lbound(dp), ubound(dp)
     call exit(1)
  end if

  call hipCheck(hipHostMalloc(hp, m, n, hipHostMallocDefault))
  hp0(0:,0:) => hp
  do j = 0, n-1
     do i = 0, m-1
        hp0(i,j) = real(i + m*j, c_double)
     end do
  end do

  call hipCheck(hipMemcpy(c_loc(dp), c_loc(hp0), int(m*n,c_size_t)*8_c_size_t, hipMemcpyHostToDevice))
  hcheck = 0.0d0
  call hipCheck(hipMemcpy(c_loc(hcheck), c_loc(dp), int(m*n,c_size_t)*8_c_size_t, hipMemcpyDeviceToHost))
  if (any(hcheck /= hp0)) then
     write(*,*) "FAILED! round trip mismatch"
     call exit(1)
  end if

  call hipCheck(hipFree(dp))
  call hipCheck(hipHostFree(hp0))

  write(*,*) "PASSED!"

end program malloc_lbounds
