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
! HIP runtime graphs via stream capture (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Captures a device memset into a HIP graph, instantiates it, launches the
! executable graph and verifies the buffer was written. Exercises
! hipStreamBeginCapture / hipStreamIsCapturing / hipStreamEndCapture /
! hipGraphInstantiate / hipGraphLaunch / hipGraphExecDestroy / hipGraphDestroy.
!!!!!!!!!!!!!!
!
program test_graph
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  integer(c_int8_t), target :: hbuf(n)
  type(c_ptr) :: stream = c_null_ptr
  type(c_ptr) :: graph  = c_null_ptr
  type(c_ptr) :: gexec  = c_null_ptr
  type(c_ptr) :: errnode = c_null_ptr
  type(c_ptr) :: dptr = c_null_ptr
  integer(c_size_t) :: nbytes
  integer(kind(hipStreamCaptureStatusNone)), target :: capstat
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip graph' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t)   ! one byte per element

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipMalloc(dptr, nbytes))

  ! Capture a device memset (value 5) into a graph.
  call check_capture(hipStreamCaptureStatusNone, "before capture")
  call hipCheck(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal))
  call check_capture(hipStreamCaptureStatusActive, "during capture")
  call hipCheck(hipMemsetAsync(dptr, 5, nbytes, stream))
  call hipCheck(hipStreamEndCapture(stream, graph))
  call check_capture(hipStreamCaptureStatusNone, "after capture")
  if (.not. c_associated(graph)) then
     write(*,*) "FAILED! captured graph is null"
     call exit(1)
  end if

  ! Instantiate and launch the executable graph.
  call hipCheck(hipGraphInstantiate(gexec, graph, errnode, c_null_ptr, 0_c_size_t))
  if (.not. c_associated(gexec)) then
     write(*,*) "FAILED! instantiated graph is null"
     call exit(1)
  end if
  call hipCheck(hipGraphLaunch(gexec, stream))
  call hipCheck(hipStreamSynchronize(stream))

  hbuf = 0
  call hipCheck(hipMemcpy(c_loc(hbuf(1)), dptr, nbytes, hipMemcpyDeviceToHost))
  do i = 1, n
     if (hbuf(i) /= 5_c_int8_t) then
        write(*,*) "FAILED! hbuf(", i, ") = ", hbuf(i), " (expected 5)"
        call exit(1)
     end if
  end do

  call hipCheck(hipGraphExecDestroy(gexec))
  call hipCheck(hipGraphDestroy(graph))
  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

contains

  subroutine check_capture(want, what)
    integer(kind(hipStreamCaptureStatusNone)), intent(in) :: want
    character(len=*), intent(in) :: what
    capstat = -1
    call hipCheck(hipStreamIsCapturing(stream, c_loc(capstat)))
    if (capstat /= want) then
       write(*,*) "FAILED! capture status ", what, " = ", capstat, " (expected ", want, ")"
       call exit(1)
    end if
  end subroutine check_capture

end program test_graph
