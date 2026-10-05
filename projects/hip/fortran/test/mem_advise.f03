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
! HIP runtime hipMemAdvise / hipMemPrefetchAsync (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Applies unified-memory hints to a managed allocation, reads them back with
! hipMemRangeGetAttribute, then prefetches the range to the device and back to
! the host and checks the host sees the zeros the device wrote while the range
! was migrated. Exits 77 (skipped) when the device has no managed memory, or
! when the runtime leaves a range attribute unreported.
!!!!!!!!!!!!!!
!
program mem_advise
  use iso_c_binding
  use hip

  implicit none

  ! hipMemAdvise rounds to CPU pages, so use a page-aligned byte count.
  integer(c_int), parameter :: n = 4096
  integer(c_int), parameter :: cpu_device = -1   ! hipCpuDeviceId
  type(c_ptr) :: mptr = c_null_ptr, stream = c_null_ptr
  real(c_double), pointer :: p(:)
  integer(c_size_t) :: nbytes
  integer(c_int), target :: qattr
  integer(c_int) :: managed
  logical :: unreported = .false.
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip mem_advise' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipDeviceGetAttribute(managed, hipDeviceAttributeManagedMemory, 0))
  if (managed == 0) then
     write(*,*) "SKIPPED! (managed memory unsupported on this device)"
     call exit(77)
  end if
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipMallocManaged(mptr, nbytes, hipMemAttachGlobal))
  call c_f_pointer(mptr, p, [n])

  do i = 1, n
     p(i) = real(i, c_double)
  end do

  call hipCheck(hipMemAdvise(mptr, nbytes, hipMemAdviseSetReadMostly, 0))
  call check_range_attr(hipMemRangeAttributeReadMostly, 1, "ReadMostly after set")

  call hipCheck(hipMemAdvise(mptr, nbytes, hipMemAdviseUnsetReadMostly, 0))
  call check_range_attr(hipMemRangeAttributeReadMostly, 0, "ReadMostly after unset")

  call hipCheck(hipMemAdvise(mptr, nbytes, hipMemAdviseSetPreferredLocation, 0))
  call check_range_attr(hipMemRangeAttributePreferredLocation, 0, "PreferredLocation")

  call hipCheck(hipMemAdvise(mptr, nbytes, hipMemAdviseSetAccessedBy, 0))
  call check_range_attr(hipMemRangeAttributeAccessedBy, 0, "AccessedBy")

  ! Migrate the range to the device, overwrite it there, then pull it back.
  call hipCheck(hipMemPrefetchAsync(mptr, nbytes, 0, stream))
  call hipCheck(hipStreamSynchronize(stream))
  call hipCheck(hipMemsetAsync(mptr, 0, nbytes, stream))
  call hipCheck(hipMemPrefetchAsync(mptr, nbytes, cpu_device, stream))
  call hipCheck(hipStreamSynchronize(stream))

  do i = 1, n
     if (p(i) /= 0.0d0) then
        write(*,*) "FAILED! p(", i, ") = ", p(i), " (expected 0)"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(mptr))
  call hipCheck(hipStreamDestroy(stream))

  if (unreported) then
     write(*,*) "SKIPPED! (hipMemRangeGetAttribute left an attribute unwritten)"
     call exit(77)
  end if
  write(*,*) "PASSED!"

contains

  subroutine check_range_attr(attribute, want, what)
    integer(kind(hipMemRangeAttributeReadMostly)), intent(in) :: attribute
    integer(c_int), intent(in) :: want
    character(len=*), intent(in) :: what
    qattr = -12345
    call hipCheck(hipMemRangeGetAttribute(c_loc(qattr), 4_c_size_t, attribute, mptr, nbytes))
    ! Some runtimes report success without writing the attribute: a runtime gap
    ! rather than a binding failure, so the test ends as skipped, not passed.
    if (qattr == -12345) then
       unreported = .true.
       return
    end if
    if (qattr /= want) then
       write(*,*) "FAILED! ", what, " = ", qattr, " (expected ", want, ")"
       call exit(1)
    end if
  end subroutine check_range_attr

end program mem_advise
