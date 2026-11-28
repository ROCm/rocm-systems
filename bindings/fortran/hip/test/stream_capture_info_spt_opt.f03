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
! HIP per-thread capture-info queries with optional outputs (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Binding behaviour checked: hipStreamGetCaptureInfo_v2_spt gets the optional
! arguments of its base routine hipStreamGetCaptureInfo_v2:
!
!   * _typed (status as an enum variable): called with only the status, the
!     id_out, graph_out, dependencies_out and numDependencies_out it omits are
!     passed as NULL; called again with id_out present;
!   * _opt (outputs as type(c_ptr) addresses, graph_out and dependencies_out
!     optional): called with graph_out present and dependencies_out absent.
!
! hipStreamGetCaptureInfo_spt (_typed, pId optional) is called both ways too.
! Inside a capture every query must report Active and the same capture id as
! the base hipStreamGetCaptureInfo_v2; outputs are primed with sentinels.
!!!!!!!!!!!!!!
!
program stream_capture_info_spt_opt
  use iso_c_binding
  use hip
  implicit none
  integer(c_size_t), parameter :: nbytes = 256_c_size_t
  type(c_ptr) :: stream = c_null_ptr, dptr = c_null_ptr, graph = c_null_ptr
  type(c_ptr) :: graph_out
  integer(kind(hipStreamCaptureStatusNone)), target :: capstat
  integer(c_int64_t), target :: id_base, id
  integer(c_size_t), target :: ndeps
  write(*,"(a)",advance="no") &
    "-- Running test 'hip stream_capture_info_spt_opt' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipMalloc(dptr, nbytes))

  ! Outside a capture, the status-only form reports None.
  capstat = -1
  call hipCheck(hipStreamGetCaptureInfo_v2_spt(stream, capstat))
  call expect_status(capstat, hipStreamCaptureStatusNone, "v2_spt (status only) before capture")

  call hipCheck(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal))
  call hipCheck(hipMemsetAsync(dptr, 5, nbytes, stream))

  ! Reference id from the base routine (_typed, other outputs omitted).
  capstat = -1; id_base = -1
  call hipCheck(hipStreamGetCaptureInfo_v2(stream, capstat, id_base))
  call expect_status(capstat, hipStreamCaptureStatusActive, "v2 during capture")
  if (id_base <= 0) then
     write(*,*) "FAILED! hipStreamGetCaptureInfo_v2 id = ", id_base; call exit(1)
  end if

  ! _typed, status only: every optional output absent.
  capstat = -1
  call hipCheck(hipStreamGetCaptureInfo_v2_spt(stream, capstat))
  call expect_status(capstat, hipStreamCaptureStatusActive, "v2_spt (status only) during capture")

  ! _typed with id_out present.
  capstat = -1; id = -1
  call hipCheck(hipStreamGetCaptureInfo_v2_spt(stream, capstat, id_out=id))
  call expect_status(capstat, hipStreamCaptureStatusActive, "v2_spt (status, id) during capture")
  call expect_id(id, "v2_spt _typed")

  ! _opt: addresses as type(c_ptr), graph_out present, dependencies_out absent.
  capstat = -1; id = -1; ndeps = 99; graph_out = c_null_ptr
  call hipCheck(hipStreamGetCaptureInfo_v2_spt(stream, c_loc(capstat), c_loc(id), &
       graph_out=graph_out, numDependencies_out=c_loc(ndeps)))
  call expect_status(capstat, hipStreamCaptureStatusActive, "v2_spt _opt during capture")
  call expect_id(id, "v2_spt _opt")
  if (.not. c_associated(graph_out)) then
     write(*,*) "FAILED! v2_spt _opt returned a null graph while capturing"; call exit(1)
  end if
  if (ndeps /= 1) then
     write(*,*) "FAILED! v2_spt _opt numDependencies = ", ndeps, " (expected 1)"; call exit(1)
  end if

  ! hipStreamGetCaptureInfo_spt (_typed): pId absent, then present.
  capstat = -1
  call hipCheck(hipStreamGetCaptureInfo_spt(stream, capstat))
  call expect_status(capstat, hipStreamCaptureStatusActive, "spt (status only) during capture")
  capstat = -1; id = -1
  call hipCheck(hipStreamGetCaptureInfo_spt(stream, capstat, pId=id))
  call expect_status(capstat, hipStreamCaptureStatusActive, "spt (status, pId) during capture")
  call expect_id(id, "spt _typed")

  call hipCheck(hipStreamEndCapture(stream, graph))
  if (.not. c_associated(graph)) then
     write(*,*) "FAILED! hipStreamEndCapture produced a null graph"; call exit(1)
  end if
  capstat = -1
  call hipCheck(hipStreamGetCaptureInfo_v2_spt(stream, capstat))
  call expect_status(capstat, hipStreamCaptureStatusNone, "v2_spt (status only) after capture")

  call hipCheck(hipGraphDestroy(graph))
  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))
  call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"

contains

  subroutine expect_status(got, want, what)
    integer(kind(hipStreamCaptureStatusNone)), intent(in) :: got, want
    character(len=*), intent(in) :: what
    if (got /= want) then
       write(*,*) "FAILED! ", what, ": status = ", got, " (expected ", want, ")"; call exit(1)
    end if
  end subroutine expect_status

  subroutine expect_id(got, what)
    integer(c_int64_t), intent(in) :: got
    character(len=*), intent(in) :: what
    if (got /= id_base) then
       write(*,*) "FAILED! ", what, ": capture id = ", got, " (base v2 reports ", id_base, ")"
       call exit(1)
    end if
  end subroutine expect_id

end program stream_capture_info_spt_opt
