!!!!!!!!!!!!!!
! HIP runtime virtual memory management (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Reserves a virtual address range, backs it with a physical allocation, makes
! it accessible to the device, uses it as ordinary device memory, and unwinds
! the mapping again.
!!!!!!!!!!!!!!
!
program virtual_memory
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 1024

  type(hipMemAllocationProp) :: prop
  type(hipMemAccessDesc)     :: desc
  type(c_ptr)                :: vptr = c_null_ptr
  type(c_ptr)                :: handle = c_null_ptr
  integer(c_size_t), target  :: granularity
  integer(c_size_t)          :: nbytes, padded
  real(c_double), target     :: hx(n), hy(n)
  integer                    :: i

  write(*,"(a)",advance="no") "-- Running test 'hip virtual_memory' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))

  prop%type = hipMemAllocationTypePinned
  prop%requestedHandleType = 0
  prop%location%type = hipMemLocationTypeDevice
  prop%location%id = 0
  prop%win32HandleMetaData = c_null_ptr
  prop%allocFlags = 0

  granularity = 0
  call hipCheck(hipMemGetAllocationGranularity(c_loc(granularity), prop, &
                                               hipMemAllocationGranularityMinimum))
  if (granularity <= 0) then
     write(*,*) "FAILED! allocation granularity =", granularity
     call exit(1)
  end if

  ! A physical allocation has to be a whole number of granularity units.
  nbytes = int(n, c_size_t) * 8
  padded = ((nbytes + granularity - 1) / granularity) * granularity

  call hipCheck(hipMemAddressReserve(vptr, padded, 0_c_size_t, c_null_ptr, 0_c_int64_t))
  call hipCheck(hipMemCreate(handle, padded, prop, 0_c_int64_t))
  call hipCheck(hipMemMap(vptr, padded, 0_c_size_t, handle, 0_c_int64_t))

  ! Mapped memory starts out inaccessible; the device needs read/write access.
  desc%location%type = hipMemLocationTypeDevice
  desc%location%id = 0
  desc%flags = hipMemAccessFlagsProtReadWrite
  call hipCheck(hipMemSetAccess(vptr, padded, desc, 1_c_size_t))

  do i = 1, n
     hx(i) = real(i, c_double)
     hy(i) = 0.0_c_double
  end do

  call hipCheck(hipMemcpy(vptr, c_loc(hx(1)), nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(c_loc(hy(1)), vptr, nbytes, hipMemcpyDeviceToHost))
  do i = 1, n
     if (hy(i) /= hx(i)) then
        write(*,*) "FAILED! hy(", i, ") = ", hy(i), " expected ", hx(i)
        call exit(1)
     end if
  end do

  ! The device must be able to write the range as well.
  call hipCheck(hipMemset(vptr, 0, nbytes))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hy(1)), vptr, nbytes, hipMemcpyDeviceToHost))
  do i = 1, n
     if (hy(i) /= 0.0_c_double) then
        write(*,*) "FAILED! hy(", i, ") = ", hy(i), " after memset (expected 0)"
        call exit(1)
     end if
  end do

  call hipCheck(hipMemUnmap(vptr, padded))
  call hipCheck(hipMemRelease(handle))
  call hipCheck(hipMemAddressFree(vptr, padded))

  write(*,*) "PASSED!"

end program virtual_memory
