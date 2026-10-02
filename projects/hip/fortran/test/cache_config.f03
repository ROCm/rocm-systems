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
! HIP runtime cache / shared-memory configuration (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipDeviceSetCacheConfig / hipDeviceGetCacheConfig and
! hipDeviceSetSharedMemConfig / hipDeviceGetSharedMemConfig. Both getters take
! their enum output by reference, so the variable is passed directly and HIP
! writes into it -- the point of this test is that the write actually lands.
!
! Each getter is primed with the sentinel -1 and must come back holding a legal
! member of its enum. AMD hardware has no configurable L1/shared split and no
! configurable shared-memory bank width, so the setters are accepted but may be
! ignored; asserting a hard round-trip would be a false failure. Checking that
! the sentinel was overwritten with a value inside the enum is what actually
! distinguishes a working by-reference binding from a broken one.
!
! The hipCtx* twins of these four routines (hipCtxGetCacheConfig,
! hipCtxGetSharedMemConfig) are deliberately not called here: they are ROCm-only
! (#ifndef USE_CUDA_NAMES in hipfort.F90) and these hand-written tests are not
! preprocessed, so referencing them would break the NVIDIA backend build. They
! stay covered by the exhaustive symbol test test_hip.F03.
!!!!!!!!!!!!!!
!
program cache_config
  use iso_c_binding
  use hip

  implicit none

  integer(kind(hipFuncCachePreferNone)) :: cconf
  integer(kind(hipSharedMemBankSizeDefault)) :: sconf

  write(*,"(a)",advance="no") &
    "-- Running test 'hip cache_config' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))

  ! 1. The default cache config of a fresh device must be a legal enum member.
  cconf = -1
  call hipCheck(hipDeviceGetCacheConfig(cconf))
  call check_cache_config(cconf, "default")

  ! 2. Every member of hipFuncCache_t must be accepted by the setter, and the
  !    getter must keep reporting a legal member afterwards.
  call set_and_get_cache_config(hipFuncCachePreferShared, "PreferShared")
  call set_and_get_cache_config(hipFuncCachePreferL1,     "PreferL1")
  call set_and_get_cache_config(hipFuncCachePreferEqual,  "PreferEqual")
  call set_and_get_cache_config(hipFuncCachePreferNone,   "PreferNone")

  ! 3. Same contract for the shared-memory bank width.
  sconf = -1
  call hipCheck(hipDeviceGetSharedMemConfig(sconf))
  call check_shared_config(sconf, "default")

  call set_and_get_shared_config(hipSharedMemBankSizeFourByte,  "FourByte")
  call set_and_get_shared_config(hipSharedMemBankSizeEightByte, "EightByte")
  call set_and_get_shared_config(hipSharedMemBankSizeDefault,   "Default")

  write(*,*) "PASSED! cache config = ", cconf, " shared mem config = ", sconf

contains

  subroutine check_cache_config(got, what)
    integer(kind(hipFuncCachePreferNone)), intent(in) :: got
    character(len=*), intent(in) :: what
    if (got /= hipFuncCachePreferNone   .and. got /= hipFuncCachePreferShared .and. &
        got /= hipFuncCachePreferL1     .and. got /= hipFuncCachePreferEqual) then
       write(*,*) "FAILED! hipDeviceGetCacheConfig (", what, ") returned ", got, &
                  " which is not a member of hipFuncCache_t"
       call exit(1)
    end if
  end subroutine check_cache_config

  subroutine set_and_get_cache_config(want, what)
    integer(kind(hipFuncCachePreferNone)), intent(in) :: want
    character(len=*), intent(in) :: what
    call hipCheck(hipDeviceSetCacheConfig(want))
    cconf = -1
    call hipCheck(hipDeviceGetCacheConfig(cconf))
    call check_cache_config(cconf, what)
  end subroutine set_and_get_cache_config

  subroutine check_shared_config(got, what)
    integer(kind(hipSharedMemBankSizeDefault)), intent(in) :: got
    character(len=*), intent(in) :: what
    if (got /= hipSharedMemBankSizeDefault  .and. &
        got /= hipSharedMemBankSizeFourByte .and. &
        got /= hipSharedMemBankSizeEightByte) then
       write(*,*) "FAILED! hipDeviceGetSharedMemConfig (", what, ") returned ", got, &
                  " which is not a member of hipSharedMemConfig"
       call exit(1)
    end if
  end subroutine check_shared_config

  subroutine set_and_get_shared_config(want, what)
    integer(kind(hipSharedMemBankSizeDefault)), intent(in) :: want
    character(len=*), intent(in) :: what
    call hipCheck(hipDeviceSetSharedMemConfig(want))
    sconf = -1
    call hipCheck(hipDeviceGetSharedMemConfig(sconf))
    call check_shared_config(sconf, what)
  end subroutine set_and_get_shared_config

end program cache_config
