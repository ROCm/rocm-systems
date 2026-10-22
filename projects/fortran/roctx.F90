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

module roctx
  use, intrinsic :: iso_c_binding
  implicit none

  interface roctxMark
    subroutine roctxMark_(message) bind(c, name="roctxMarkA")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: message
    end subroutine

    module procedure roctxMark_typed
  end interface

  interface roctxRangePush
    function roctxRangePush_(message) bind(c, name="roctxRangePushA")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxRangePush_
      type(c_ptr),value :: message
    end function

    module procedure roctxRangePush_typed
  end interface

  interface roctxRangePop
    function roctxRangePop_() bind(c, name="roctxRangePop")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxRangePop_
    end function
  end interface

  interface roctxRangeStart
    function roctxRangeStart_(message) bind(c, name="roctxRangeStartA")
      use iso_c_binding
      import
      implicit none
      integer(c_int64_t) :: roctxRangeStart_
      type(c_ptr),value :: message
    end function

    module procedure roctxRangeStart_typed
  end interface

  interface roctxRangeStop
    subroutine roctxRangeStop_(id) bind(c, name="roctxRangeStop")
      use iso_c_binding
      import
      implicit none
      integer(c_int64_t),value :: id
    end subroutine
  end interface

  interface roctxProfilerPause
    function roctxProfilerPause_(tid) bind(c, name="roctxProfilerPause")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxProfilerPause_
      integer(c_int64_t),value :: tid
    end function
  end interface

  interface roctxProfilerResume
    function roctxProfilerResume_(tid) bind(c, name="roctxProfilerResume")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxProfilerResume_
      integer(c_int64_t),value :: tid
    end function
  end interface

  interface roctxNameOsThread
    function roctxNameOsThread_(name) bind(c, name="roctxNameOsThread")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxNameOsThread_
      type(c_ptr),value :: name
    end function

    module procedure roctxNameOsThread_typed
  end interface

  interface roctxNameHsaAgent
    function roctxNameHsaAgent_(name,agent) bind(c, name="roctxNameHsaAgent")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxNameHsaAgent_
      type(c_ptr),value :: name
      type(c_ptr),value :: agent
    end function

    module procedure roctxNameHsaAgent_typed
  end interface

  interface roctxNameHipDevice
    function roctxNameHipDevice_(name,device_id) bind(c, name="roctxNameHipDevice")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxNameHipDevice_
      type(c_ptr),value :: name
      integer(c_int),value :: device_id
    end function

    module procedure roctxNameHipDevice_typed
  end interface

  interface roctxNameHipStream
    function roctxNameHipStream_(name,stream) bind(c, name="roctxNameHipStream")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxNameHipStream_
      type(c_ptr),value :: name
      type(c_ptr),value :: stream
    end function

    module procedure roctxNameHipStream_typed
  end interface

  interface roctxGetThreadId
    function roctxGetThreadId_(tid) bind(c, name="roctxGetThreadId")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: roctxGetThreadId_
      type(c_ptr),value :: tid
    end function
  end interface


  contains

    subroutine roctxMark_typed(message)
      use iso_c_binding
      implicit none
      character(kind=c_char,len=*),intent(in),target :: message
      !
      call roctxMark_(c_loc(message))
    end subroutine

    function roctxRangePush_typed(message)
      use iso_c_binding
      implicit none
      integer(c_int) :: roctxRangePush_typed
      character(kind=c_char,len=*),intent(in),target :: message
      !
      roctxRangePush_typed = roctxRangePush_(c_loc(message))
    end function

    function roctxRangeStart_typed(message)
      use iso_c_binding
      implicit none
      integer(c_int64_t) :: roctxRangeStart_typed
      character(kind=c_char,len=*),intent(in),target :: message
      !
      roctxRangeStart_typed = roctxRangeStart_(c_loc(message))
    end function

    function roctxNameOsThread_typed(name)
      use iso_c_binding
      implicit none
      integer(c_int) :: roctxNameOsThread_typed
      character(kind=c_char,len=*),intent(in),target :: name
      !
      roctxNameOsThread_typed = roctxNameOsThread_(c_loc(name))
    end function

    function roctxNameHsaAgent_typed(name,agent)
      use iso_c_binding
      implicit none
      integer(c_int) :: roctxNameHsaAgent_typed
      character(kind=c_char,len=*),intent(in),target :: name
      type(c_ptr), value :: agent
      !
      roctxNameHsaAgent_typed = roctxNameHsaAgent_(c_loc(name),agent)
    end function

    function roctxNameHipDevice_typed(name,device_id)
      use iso_c_binding
      implicit none
      integer(c_int) :: roctxNameHipDevice_typed
      character(kind=c_char,len=*),intent(in),target :: name
      integer(c_int) :: device_id
      !
      roctxNameHipDevice_typed = roctxNameHipDevice_(c_loc(name),device_id)
    end function

    function roctxNameHipStream_typed(name,stream)
      use iso_c_binding
      implicit none
      integer(c_int) :: roctxNameHipStream_typed
      character(kind=c_char,len=*),intent(in),target :: name
      type(c_ptr), value :: stream
      !
      roctxNameHipStream_typed = roctxNameHipStream_(c_loc(name),stream)
    end function

end module roctx
