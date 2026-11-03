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
! hipDeviceFlushGPUDirectRDMAWrites (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! The flush is supported when hipDeviceAttributeGPUDirectRDMAFlushWritesOptions
! has the hipFlushGPUDirectRDMAWritesOptionHost bit. Flushes the current device
! to both scopes and expects hipSuccess if it is supported, hipErrorNotSupported
! otherwise; an unknown scope must give hipErrorInvalidValue either way.
!!!!!!!!!!!!!!
!
program rdma_flush
  use iso_c_binding
  use hip

  implicit none

  integer(c_int) :: options, expected

  write(*,"(a)",advance="no") "-- Running test 'hip rdma_flush' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipDeviceGetAttribute(options, hipDeviceAttributeGPUDirectRDMAFlushWritesOptions, 0))
  if (iand(options, hipFlushGPUDirectRDMAWritesOptionHost) /= 0) then
     expected = hipSuccess
  else
     expected = hipErrorNotSupported
  end if

  call check("ToOwner", hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice, &
                                                          hipFlushGPUDirectRDMAWritesToOwner), expected)
  call check("ToAllDevices", hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice, &
                                                               hipFlushGPUDirectRDMAWritesToAllDevices), expected)
  call check("invalid scope", hipDeviceFlushGPUDirectRDMAWrites(hipFlushGPUDirectRDMAWritesTargetCurrentDevice, &
                                                                0), hipErrorInvalidValue)

  write(*,*) "PASSED!"

contains

  subroutine check(what, got, want)
    character(len=*), intent(in) :: what
    integer(c_int), intent(in) :: got, want
    if (got /= want) then
       write(*,*) "FAILED! flush ", what, " returned ", got, " expected ", want, &
            " (flush options = ", options, ")"
       call exit(1)
    end if
  end subroutine check

end program rdma_flush
