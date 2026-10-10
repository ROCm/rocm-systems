!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

module roctx
  use, intrinsic :: iso_c_binding
  implicit none

  interface

    !---------------------------------------------
    ! roctxMark
    !---------------------------------------------
    subroutine roctxMark(message) &
       bind(C, name="roctxMarkA")
       import :: c_ptr
       type(c_ptr), value :: message
    end subroutine roctxMark

    !---------------------------------------------
    ! roctxRangePush
    !---------------------------------------------
    function roctxRangePush(message) &
       result(RangePush) &
       bind(C, name="roctxRangePushA")
       import :: c_ptr, c_int
       type(c_ptr), value :: message
       integer(c_int) :: RangePush
    end function roctxRangePush

    !---------------------------------------------
    ! roctxRangePop
    !---------------------------------------------
    function roctxRangePop() &
       result(RangePop) &
       bind(C, name="roctxRangePop")
       import :: c_int
       integer(c_int) :: RangePop
    end function roctxRangePop

    !---------------------------------------------
    ! roctxRangeStart
    !---------------------------------------------
    function roctxRangeStart(message) &
       result(RangeStart) &
       bind(C, name="roctxRangeStartA")
       import :: c_ptr, c_int64_t
       type(c_ptr), value :: message
       integer(c_int64_t) :: RangeStart
    end function roctxRangeStart

    !---------------------------------------------
    ! roctxRangeStop
    !---------------------------------------------
    subroutine roctxRangeStop(id) &
       bind(C, name="roctxRangeStop")
       import :: c_int64_t
       integer(c_int64_t), value :: id
    end subroutine roctxRangeStop

    !---------------------------------------------
    ! roctxProfilerPause
    !---------------------------------------------
    function roctxProfilerPause(tid) &
       result(ProfilerPause) &
       bind(C, name="roctxProfilerPause")
       import :: c_int64_t, c_int
       integer(c_int64_t), value :: tid
       integer(c_int) :: ProfilerPause
    end function roctxProfilerPause

    !---------------------------------------------
    ! roctxProfilerResume
    !---------------------------------------------
    function roctxProfilerResume(tid) &
       result(ProfilerResume) &
       bind(C, name="roctxProfilerResume")
       import :: c_int64_t, c_int
       integer(c_int64_t), value :: tid
       integer(c_int) :: ProfilerResume
    end function roctxProfilerResume

    !---------------------------------------------
    ! roctxNameOsThread
    !---------------------------------------------
    function roctxNameOsThread(name) &
       result(NameOsThread) &
       bind(C, name="roctxNameOsThread")
       import :: c_ptr, c_int
       type(c_ptr), value :: name
       integer(c_int) :: NameOsThread
    end function roctxNameOsThread

    !---------------------------------------------
    ! roctxNameHsaAgent
    !---------------------------------------------
    function roctxNameHsaAgent(name, agent) &
       result(NameHsaAgent) &
       bind(C, name="roctxNameHsaAgent")
       import :: c_ptr, c_int
       type(c_ptr), value :: name
       type(c_ptr), value :: agent
       integer(c_int) :: NameHsaAgent
    end function roctxNameHsaAgent

    !---------------------------------------------
    ! roctxNameHipDevice
    !---------------------------------------------
    function roctxNameHipDevice(name, device_id) &
       result(NameHipDevice) &
       bind(C, name="roctxNameHipDevice")
       import :: c_ptr, c_int
       type(c_ptr), value :: name
       integer(c_int), value :: device_id
       integer(c_int) :: NameHipDevice
    end function roctxNameHipDevice

    !---------------------------------------------
    ! roctxNameHipStream
    !---------------------------------------------
    function roctxNameHipStream(name, stream) &
       result(NameHipStream) &
       bind(C, name="roctxNameHipStream")
       import :: c_ptr, c_int
       type(c_ptr), value :: name
       type(c_ptr), value :: stream
       integer(c_int) :: NameHipStream
    end function roctxNameHipStream

    !---------------------------------------------
    ! roctxGetThreadId
    !---------------------------------------------
    function roctxGetThreadId(tid) &
       result(GetThreadId) &
       bind(C, name="roctxGetThreadId")
       import :: c_int64_t, c_int
       integer(c_int64_t) :: tid
       integer(c_int) :: GetThreadId
    end function roctxGetThreadId

  end interface

end module roctx
