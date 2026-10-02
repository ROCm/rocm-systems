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

program test_roctx
  use iso_c_binding
  use roctx

  implicit none

  character(kind=c_char), dimension(6), target :: msg = &
      [c_char_"r", c_char_"o", c_char_"c", c_char_"t", c_char_"x", c_null_char]
  integer(c_int64_t), target :: tid
  integer(c_int64_t) :: id1, id2

  write(*,"(a)",advance="no") "-- Running test 'roctx' (Fortran 2003 interfaces) - "

  call roctxMark(c_loc(msg))

  call check(roctxRangePush(c_loc(msg)), 0, "first roctxRangePush")
  call check(roctxRangePush(c_loc(msg)), 1, "nested roctxRangePush")
  call check(roctxRangePop(), 1, "nested roctxRangePop")
  call check(roctxRangePop(), 0, "first roctxRangePop")
  call check(roctxRangePop(), -1, "unbalanced roctxRangePop")

  id1 = roctxRangeStart(c_loc(msg))
  id2 = roctxRangeStart(c_loc(msg))
  if (id1 <= 0 .or. id2 <= id1) then
     write(*,*) "FAILED! roctxRangeStart returned ", id1, id2
     call exit(1)
  end if
  call roctxRangeStop(id2)
  call roctxRangeStop(id1)

  tid = 0
  call check(roctxGetThreadId(c_loc(tid)), 0, "roctxGetThreadId")
  if (tid == 0) then
     write(*,*) "FAILED! roctxGetThreadId left the thread id at 0"
     call exit(1)
  end if

  call check(roctxProfilerPause(tid), 0, "roctxProfilerPause")
  call check(roctxProfilerResume(tid), 0, "roctxProfilerResume")
  call check(roctxNameOsThread(c_loc(msg)), 0, "roctxNameOsThread")
  call check(roctxNameHsaAgent(c_loc(msg), c_null_ptr), 0, "roctxNameHsaAgent")
  call check(roctxNameHipDevice(c_loc(msg), 0_c_int), 0, "roctxNameHipDevice")
  call check(roctxNameHipStream(c_loc(msg), c_null_ptr), 0, "roctxNameHipStream")

  write(*,*) "PASSED!"

contains

  subroutine check(got, expected, what)
    integer(c_int), intent(in) :: got
    integer, intent(in) :: expected
    character(*), intent(in) :: what

    if (got /= expected) then
       write(*,*) "FAILED! ", what, " returned ", got, " (expected ", expected, ")"
       call exit(1)
    end if
  end subroutine check

end program test_roctx
