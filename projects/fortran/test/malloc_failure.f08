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
! HIP runtime hipMalloc/hipHostMalloc failure (Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Asks for more memory than any device or host has. The allocation must fail,
! and leave the pointer unassociated (it was associated before the call), so
! that hipFree(ptr, only_if_allocated=.true.) does nothing, and hipFree(ptr)
! frees NULL, which HIP accepts.
!!!!!!!!!!!!!!
!
program malloc_failure
  use iso_c_binding
  use hip

  implicit none

  ! 2**58 doubles: 2 EiB, more than any device or host memory.
  integer(c_size_t), parameter :: huge_len = 2_c_size_t**58
  real(c_double), pointer :: dp(:), hp(:)
  real(c_double), target  :: dummy(4)
  integer(c_int) :: res

  write(*,"(a)",advance="no") "-- Running test 'hip malloc_failure' (Fortran 2008 interfaces) - "

  call hipCheck(hipSetDevice(0))

  dp => dummy
  res = hipMalloc(dp, huge_len)
  if (res == hipSuccess) then
     write(*,*) "FAILED! hipMalloc of 2 EiB succeeded"
     call exit(1)
  end if
  if (associated(dp)) then
     write(*,*) "FAILED! hipMalloc failed but left the pointer associated"
     call exit(1)
  end if
  call hipCheck(hipFree(dp, only_if_allocated=.true.))
  call hipCheck(hipFree(dp))

  hp => dummy
  res = hipHostMalloc(hp, huge_len, hipHostMallocDefault)
  if (res == hipSuccess) then
     write(*,*) "FAILED! hipHostMalloc of 2 EiB succeeded"
     call exit(1)
  end if
  if (associated(hp)) then
     write(*,*) "FAILED! hipHostMalloc failed but left the pointer associated"
     call exit(1)
  end if
  call hipCheck(hipHostFree(hp, only_if_allocated=.true.))

  ! Clear the sticky error of the failed allocations.
  res = hipGetLastError()

  write(*,*) "PASSED!"

end program malloc_failure
