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
! HIP runtime host-side stream callbacks
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipStreamAddCallback and hipLaunchHostFunc. Queues an async memset
! on a stream, then registers one callback of each kind via c_funloc. After
! synchronize: asserts both callbacks fired (counter == 2) and device data from
! the preceding memset is intact (ordering: callbacks trail enqueued device work).
!!!!!!!!!!!!!!
!
program stream_callback
  use iso_c_binding
  use hip

  implicit none

  ! Explicit interfaces for the external bind(c) procedures so c_funloc can see them.
  interface
    subroutine stream_cb(stream, status, userData) bind(c)
      use iso_c_binding
      use hip
      implicit none
      type(c_ptr),                      value :: stream
      integer(kind(hipSuccess)),        value :: status
      type(c_ptr),                      value :: userData
    end subroutine
    subroutine host_fn(userData) bind(c)
      use iso_c_binding
      implicit none
      type(c_ptr), value :: userData
    end subroutine
  end interface

  integer(c_int), parameter   :: n = 256
  integer(c_int8_t), target   :: hbuf(n)
  integer(c_int),   target    :: counter = 0   ! incremented by each callback
  type(c_ptr)                 :: stream = c_null_ptr, dptr = c_null_ptr
  integer(c_size_t)           :: nbytes

  write(*,"(a)",advance="no") "-- Running test 'hip stream_callback' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t)   ! one byte per element

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dptr, nbytes))
  call hipCheck(hipStreamCreate(stream))

  ! Queue device work; the callbacks below must not fire until this completes.
  call hipCheck(hipMemsetAsync(dptr, 7, nbytes, stream))

  call hipCheck(hipStreamAddCallback(stream, c_funloc(stream_cb), c_loc(counter), 0_c_int))
  call hipCheck(hipLaunchHostFunc(stream,    c_funloc(host_fn),   c_loc(counter)))

  call hipCheck(hipStreamSynchronize(stream))

  if (counter /= 2) then
    write(*,*) "FAILED! callback counter =", counter, "(expected 2)"
    call exit(1)
  end if

  ! Copy device data back; proves memset completed before either callback ran.
  hbuf = 0_c_int8_t
  call hipCheck(hipMemcpy(c_loc(hbuf(1)), dptr, nbytes, hipMemcpyDeviceToHost))
  if (any(hbuf /= 7_c_int8_t)) then
    write(*,*) "FAILED! device buffer not filled with 7 after memset"
    call exit(1)
  end if

  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program stream_callback

! hipStreamCallback_t: void (*)(hipStream_t stream, hipError_t status, void* userData)
subroutine stream_cb(stream, status, userData) bind(c)
  use iso_c_binding
  use hip
  implicit none
  type(c_ptr),               value :: stream
  integer(kind(hipSuccess)), value :: status
  type(c_ptr),               value :: userData
  integer(c_int), pointer :: counter
  call c_f_pointer(userData, counter)
  counter = counter + 1
end subroutine stream_cb

! hipHostFn_t: void (*)(void* userData)
subroutine host_fn(userData) bind(c)
  use iso_c_binding
  implicit none
  type(c_ptr), value :: userData
  integer(c_int), pointer :: counter
  call c_f_pointer(userData, counter)
  counter = counter + 1
end subroutine host_fn
